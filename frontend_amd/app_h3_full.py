#!/usr/bin/env python3
"""MiniMax-H3 管理前端(重写版) —— 文生视频 / 图生视频 / 编图(Reference-to-Image)。

后端: MinimaxH3-ONNX FastAPI WebUI(默认 http://127.0.0.1:7860), 通过 onnx_adapter 调用。
特性:
  - 文生视频(text)  / 图生视频(first 首帧) / 编图(R2I: 生成视频→抽帧当图, 参考图可多张)
  - 时长 Slider 0.5~15s;  分辨率 宽高比 + 档位最高 2.0MP
  - 显存自适应钳制: 根据后端 GPU 总量自动限制提交分辨率, 避免 T4 16GB OOM
  - 取码/凭码取回 + 实时 block 进度 + 进行中任务表 + 一键取消(软)
运行:
  python app_h3_full.py --backend http://127.0.0.1:7860 --host 127.0.0.1 --port 7863
"""
import os, sys, math, random, time, threading, json, re as _re, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_adapter as B
import gradio as gr

# ---------------- 分辨率 ----------------
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3), "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4), "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16), "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
RES_MULTIPLE = 32
# 分辨率档位(label -> 目标 MP; 后端每边硬上限 1024 ≈ 1.05MP, 更高会由显存钳制或后端拒绝)
MP_CHOICES = ["0.05 MP", "0.1 MP（极快）", "0.2 MP（快速）", "0.35 MP（标准）",
              "0.5 MP", "0.8 MP", "1.0 MP（后端上限）", "1.5 MP", "2.0 MP"]

BACKEND_MAX_SIDE = 1024          # 后端 Pydantic width/height 上限

def _parse_mp(mp):
    m = _re.search(r"(\d+(?:\.\d+)?)\s*MP", str(mp))
    return float(m.group(1)) if m else 0.35

def _gpu_total_gb():
    """从后端 /api/profiles 取 GPU 总量(GB), 失败返回 None。"""
    try:
        d = B._http_json("GET", "/api/profiles")
        p = d[0] if isinstance(d, list) and d else d
        m = (p or {}).get("memory") or {}
        return (m.get("total_bytes") or 0) / (1024 ** 3)
    except Exception:
        return None

def _safe_max_pixels_for_gpu():
    """根据 GPU 显存给出安全像素上限。T4/16GB 用实测安全值; 显存越大越宽松。
    实测: 0.35MP 在 T4 main_block 峰值 ~7GB(剩余~8GB) 稳定; 1024²(1.05MP) 会 OOM(~14.7GB)。"""
    gb = _gpu_total_gb()
    if gb is None:
        gb = 15.0  # 默认按小显存保守
    if gb <= 0:
        gb = 15.0
    # 16GB 保守给 0.45MP; 随显存线性放宽, 封顶到后端 1024 边长对应的 ~1.05MP
    base = 0.45 * (1024 * 1024)
    cap_pix = BACKEND_MAX_SIDE * BACKEND_MAX_SIDE
    if gb >= 30:
        return cap_pix          # 大显存(如 MI300X 级)放开到后端上限
    if gb >= 20:
        return int(cap_pix * 0.8)
    # 16GB 及以下
    return int(base)

def clamp_safe(w, h):
    """把 (w,h) 钳制到: 后端边长上限 + 当前 GPU 安全像素内, 尽量保持宽高比并向下 32 对齐。"""
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return 640, 360
    safe_pix = _safe_max_pixels_for_gpu()
    w2, h2 = w, h
    for _ in range(6):
        k = 1.0
        if max(w, h) > BACKEND_MAX_SIDE:
            k = min(k, BACKEND_MAX_SIDE / float(max(w, h)))
        if w * h > safe_pix:
            k = min(k, math.sqrt(safe_pix / float(w * h)))
        w, h = max(1, int(w * k)), max(1, int(h * k))
        w2 = max(RES_MULTIPLE, int(w // RES_MULTIPLE) * RES_MULTIPLE)
        h2 = max(RES_MULTIPLE, int(h // RES_MULTIPLE) * RES_MULTIPLE)
        if w2 * h2 <= safe_pix and max(w2, h2) <= BACKEND_MAX_SIDE:
            return w2, h2
        w, h = w2, h2
    return w2, h2

def resolve_from_mp(aspect_label, mp_label):
    """由宽高比 + MP 档算出 (w,h), 已经过显存/边长钳制。"""
    if str(mp_label).strip().endswith("2.0 MP"):
        pass
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_label]
    total = _parse_mp(mp_label) * 1024 * 1024
    scale = math.sqrt(total / (w_ratio * h_ratio))
    w = max(RES_MULTIPLE, int(w_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE)
    h = max(RES_MULTIPLE, int(h_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE)
    return clamp_safe(w, h)

def follow_ref_size(ref_path):
    """读参考图原始尺寸, 返回保持宽高比但经钳制的 (w,h)。"""
    try:
        from PIL import Image
        with Image.open(ref_path) as im:
            w, h = im.size
        return clamp_safe(w, h)
    except Exception:
        return 640, 360

# ---------------- 任务注册表 / 轮询 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKUP = os.path.join(BASE_DIR, "pickup_full")
os.makedirs(PICKUP, exist_ok=True)
_JOBS = {}          # code -> {kind,jid,status,created,out_mp4,audio,images[],mode,resolution_note}
_JOBS_LOCK = threading.Lock()
_R2I_JOBS = {}
_ACTIVE = {}        # code -> {kind,status,created,start}
_CANCEL = set()
_LOCK = threading.Lock()
_RUNNING = 0
_GEN_LOCK = threading.Lock()
LOG = []
LOG_LOCK = threading.Lock()
LOG_MAX = 200

def _new_code():
    while True:
        c = f"{random.randint(0, 999999):06d}"
        if c not in _JOBS:
            return c

def add_log(msg):
    with LOG_LOCK:
        LOG.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        del LOG[: max(0, len(LOG) - LOG_MAX)]
    return "\n".join(LOG)

def log_html():
    with LOG_LOCK:
        return "\n".join(LOG[-60:]) or "（暂无日志）"

def _is_cancelled(code):
    with _LOCK:
        return code in _CANCEL

def _set_job(code, **kw):
    with _JOBS_LOCK:
        _JOBS.setdefault(code, {}).update(kw)

def _set_active(code, **kw):
    with _LOCK:
        _ACTIVE.setdefault(code, {}).update(kw)

def active_list():
    with _LOCK:
        now = time.time()
        out = []
        for c, v in _ACTIVE.items():
            if v.get("status") in ("completed", "failed", "cancelled"):
                continue
            el = int(now - (v.get("start") or now))
            out.append([c, v.get("kind", ""), v.get("status", ""), f"{el}s"])
    return out

def cancel_code(code):
    c = str(code or "").strip()
    if len(c.split()) > 1:
        c = c.split()[1]
    if not c.isdigit():
        return f"无法取消: 无效码 {code}"
    with _LOCK:
        _CANCEL.add(c)
        if c in _ACTIVE:
            _ACTIVE[c]["status"] = "cancelled"
    with _JOBS_LOCK:
        if c in _JOBS:
            _JOBS[c]["status"] = "cancelled"
    add_log(f"✕ 已取消任务 {c}(前端停止轮询; 后端该条可能继续跑完)")
    return f"已取消 {c}"

def cancel_latest_task():
    with _LOCK:
        running = [c for c, v in _ACTIVE.items()
                   if v.get("status") not in ("completed", "failed", "cancelled")]
    if not running:
        return "当前无进行中任务"
    latest = max(running, key=lambda c: _ACTIVE[c].get("start") or 0)
    return cancel_code(latest)

# ---------------- 生成执行 ----------------
def _submit_job(kind, prompt, refs, start_path, w, h, duration, steps, use_lora,
                resolution_note, edit_start_img=None):
    code = _new_code()
    _set_job(code, kind=kind, status="queued", created=time.time(), out_mp4=None, audio=None,
             images=None, mode=kind, resolution_note=resolution_note)
    _set_active(code, kind=kind, status="queued", created=time.time(), start=time.time())
    threading.Thread(target=_exec, args=(code, kind, prompt, refs, start_path, edit_start_img,
                                         w, h, duration, steps, use_lora), daemon=True).start()
    return f"🔑 {code}"

def _exec(code, kind, prompt, refs, start_path, edit_start_img, w, h, duration, steps, use_lora):
    global _RUNNING
    cond = "text"
    sp = None
    if kind == "edit" and edit_start_img:
        # 编图: 上传首帧作条件, 保持主体; 由抽帧输出"编辑后图"
        try:
            sp = B.upload_image(edit_start_img)
        except Exception as e:
            _set_job(code, status=f"failed: 上传首帧失败 {e}")
            _set_active(code, status=f"failed: {e}")
            add_log(f"任务 {code} 上传首帧失败: {e}")
            return
        cond = "first"
    elif kind == "i2v" and start_path:
        cond = "first"
        sp = start_path
    try:
        add_log(f"[{kind}] 任务 {code} 入队 (W={w} H={h} 时长={duration}s 步数={steps} cond={cond})")
        # 并行度=1
        while _RUNNING >= 1:
            if _is_cancelled(code):
                return
            time.sleep(2)
        _GEN_LOCK.acquire(); _RUNNING = 1
        try:
            if _is_cancelled(code):
                return
            _set_active(code, start=time.time(), status="running")
            _set_job(code, status="running")
            jid = B.submit(prompt=prompt, start_image_path=sp,
                           steps=int(steps), seed=random.randint(0, 2**31 - 1),
                           width=w, height=h, duration_seconds=float(duration),
                           use_acceleration_lora=bool(use_lora),
                           references=None, conditioning=cond)
            _set_job(code, jid=jid)
            add_log(f"提交后端 ok jid={jid[:8]}… mode={cond}")
            _poll(code, jid, kind)
        finally:
            _RUNNING = 0; _GEN_LOCK.release()
    except Exception as e:
        _set_job(code, status=f"failed: {e}")
        _set_active(code, status=f"failed: {e}")
        add_log(f"任务 {code} 失败: {e}")
    finally:
        st = _JOBS.get(code, {}).get("status", "done")
        _set_active(code, status="completed" if str(st) == "completed" else st)

def _poll(code, jid, kind):
    last = ""
    while True:
        if _is_cancelled(code):
            return
        done, st = B.poll(jid)
        if done:
            if st.get("status") == "completed":
                mp4 = os.path.join(PICKUP, f"{code}.mp4")
                B.download_output(jid, mp4)
                audio = os.path.join(PICKUP, f"{code}.m4a")
                try:
                    B.extract_audio(mp4, audio)
                except Exception:
                    audio = None
                images = None
                if kind in ("edit",):
                    img_dir = os.path.join(PICKUP, f"{code}_imgs")
                    try:
                        images = B.extract_all_frames(mp4, img_dir)
                    except Exception:
                        images = None
                _set_job(code, status="completed", out_mp4=mp4, audio=audio, images=images)
                add_log(f"任务 {code} 完成 -> {mp4}" + (f" / {len(images)} 图" if images else ""))
            else:
                err = st.get("message") or st.get("error") or st.get("status") or "failed"
                _set_job(code, status=f"failed: {err}")
                _set_active(code, status=f"failed: {err}")
                add_log(f"任务 {code} 失败: {err}")
            return
        msg = st.get("message") or ""
        if msg != last:
            last = msg
            _set_job(code, status=msg)
        time.sleep(5)

# ---------------- 供 UI 回调 ----------------
def generate_t2v(prompt, aspect, mp, duration, steps, lora):
    if not (prompt or "").strip():
        return "⚠️ 请填 prompt"
    w, h = resolve_from_mp(aspect, mp)
    note = f"{w}×{h}"
    code = _submit_job("t2v", prompt, None, None, w, h, duration, steps, lora, note)
    add_log(f"文生视频提交: code={code} res={note}")
    return f"🔑 {code} · 分辨率 {note}"

def generate_i2v(prompt, ref_images, aspect, mp, duration, steps, lora, follow_ref):
    imgs = _as_list(ref_images)
    if not imgs:
        return "⚠️ 图生视频请上传 1 张首帧图"
    first = imgs[0]
    if follow_ref:
        w, h = follow_ref_size(first)
    else:
        w, h = resolve_from_mp(aspect, mp)
    note = f"{w}×{h}"
    sp = B.upload_image(first)
    code = _submit_job("i2v", prompt, None, sp, w, h, duration, steps, lora, note)
    add_log(f"图生视频提交: code={code} res={note}")
    return f"🔑 {code} · 分辨率 {note}"

def generate_edit(prompt, edit_images, aspect, mp, duration, steps, lora, follow_ref):
    imgs = _as_list(edit_images)
    if not imgs:
        return "⚠️ 编图请上传参考图"
    if follow_ref:
        w, h = follow_ref_size(imgs[0])
    else:
        w, h = resolve_from_mp(aspect, mp)
    note = f"{w}×{h}"
    # 编图实现: 与现有可跑通的 R2I 一致 —— 第一张参考图作为首帧条件(图生视频,保持主体),
    # 其余参考图通过 prompt 里的 <Picture N> 语义参与。onnx_adapter 走 segmented,
    # 避免 references 要求的 native(5-15s)路径(该路径更慢更易OOM)。
    code = _submit_job("edit", prompt, None, None, w, h, duration, steps, lora, note,
                       edit_start_img=imgs[0])
    add_log(f"编图提交: code={code} 首帧={os.path.basename(imgs[0])} res={note}")
    return f"🔑 {code} · 分辨率 {note}"

def _as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]

def _extract_code(text):
    c = str(text or "").strip()
    if len(c.split()) > 1:
        c = c.split()[1]
    return c if c.isdigit() and len(c) == 6 else None

def nice_progress(code_text):
    c = _extract_code(code_text)
    if not c:
        return "🟢 就绪: 填写参数后点生成(此区实时显示 block 进度)"
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v:
        return "🟡 等待任务初始化…"
    s = v.get("status")
    if isinstance(s, str) and s.startswith("failed"):
        return f"🔴 {s}"
    m = _re.search(r"main_block_(\d+)_([a-z_]+)", str(s))
    if m:
        return f"🎬 主块 {m.group(1)}/50 · {m.group(2)}"
    return f"🟡 {s}…"

def _retrieve(code_text, kind):
    c = _extract_code(code_text)
    if not c:
        return [], None, None, ""
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v or v.get("status") != "completed":
        return [], None, None, "（任务未完成或码无效）"
    out_mp4 = v.get("out_mp4")
    mp4 = out_mp4 if out_mp4 and os.path.isfile(out_mp4) else None
    aud = v.get("audio") if v.get("audio") and os.path.isfile(v.get("audio")) else None
    imgs = [x for x in (v.get("images") or []) if os.path.isfile(x)]
    note = v.get("resolution_note", "")
    if kind == "edit":
        return imgs, mp4, aud, f"取图码 {c} · 分辨率 {note}"
    return [], mp4, aud, f"取视频码 {c} · 分辨率 {note}"

def retrieve_video(code_text):
    imgs, mp4, aud, note = _retrieve(code_text, "video")
    return mp4, aud

def retrieve_edit(code_text):
    imgs, mp4, aud, note = _retrieve(code_text, "edit")
    return imgs, mp4, aud, note

def check_latest(code_text, kind):
    c = _extract_code(code_text)
    if not c:
        return [], None, None
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v:
        return [], None, None
    return _retrieve(code_text, kind)[:3]

def sysinfo():
    lines = []
    try:
        with open("/proc/loadavg") as f:
            lines.append("CPU: " + " ".join(f.read().split()[:3]))
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            d = {}
            for ln in f:
                k, v = ln.split(":", 1); d[k.strip()] = int(v.split()[0]) / 1048576
        used = d.get("MemTotal", 0) - d.get("MemAvailable", 0)
        lines.append(f"RAM {used:.1f}/{d.get('MemTotal', 0):.1f}GiB")
    except Exception:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        row = [x.strip() for x in r.stdout.strip().splitlines()[:1]]
        if row:
            nm, mu, mt, gu = [x.strip() for x in row[0].split(",")]
            lines.append(f"GPU {nm} {mu}/{mt}MiB util{gu}%")
    except Exception:
        pass
    try:
        lines.append("后端 " + ("✅就绪" if B.onnx_health() else "⚠️离线"))
        gb = _gpu_total_gb()
        if gb:
            lines.append(f"显存{gb:.0f}GB→安全上限≈{_safe_max_pixels_for_gpu()/1e6:.2f}MP")
    except Exception:
        lines.append("后端 ⚠️")
    return "  |  ".join(lines)

def _build_tabs(tag, kind):
    pass

def build():
    with gr.Blocks(title="MiniMax-H3 管理前端") as demo:
        gr.Markdown("## MiniMax-H3 · ONNX · 文生/图生/编图\n"
                    "并行度=1, 自动排队; 分辨率已按后端边长(≤1024)+当前 GPU 显存自动校准, 避免 OOM。\n"
                    "⚠️ 分辨率/时长越大越慢(逐块流式), 15s 会分成多段, 很慢, 请耐心或先用短时长。")
        top_status = gr.Markdown("🟢 就绪")
        with gr.Tabs():
            # ===== 文生视频 =====
            with gr.Tab("文生视频"):
                t_prompt = gr.Textbox(label="Prompt", lines=5, value="A red apple on a wooden table, soft light, cinematic")
                with gr.Row():
                    t_aspect = gr.Dropdown(label="宽高比", choices=list(ASPECT_RATIOS.keys()), value="16:9 (Widescreen)")
                    t_mp = gr.Dropdown(label="分辨率档位(≤后端上限, 显存自动校准)", choices=MP_CHOICES, value="0.35 MP（标准）")
                with gr.Row():
                    t_dur = gr.Slider(minimum=0.5, maximum=15.0, value=3.0, step=0.5, label="时长(秒)")
                    t_steps = gr.Slider(minimum=1, maximum=50, value=4, step=1, label="采样步数(4=turbo)")
                    t_lora = gr.Checkbox(label="LoRA加速(turbo,建议4步)", value=True)
                t_btn = gr.Button("生成文生视频", variant="primary")
                t_code = gr.Textbox(label="🔑 取视频码", interactive=False, lines=2)
                t_prog = gr.Markdown("🟢 就绪")
                t_pick_code = gr.Textbox(label="凭码取回(输入6位码)", placeholder="如 123456", lines=1)
                t_get = gr.Button("取回视频")
                t_video = gr.Video(label="结果视频", format="mp4")
                t_audio = gr.Audio(label="音频", type="filepath")
                t_btn.click(generate_t2v, [t_prompt, t_aspect, t_mp, t_dur, t_steps, t_lora], [t_code])
                t_timer = gr.Timer(value=3)
                t_timer.tick(nice_progress, [t_code], [t_prog])
                t_get.click(retrieve_video, [t_pick_code], [t_video, t_audio])
            # ===== 图生视频 =====
            with gr.Tab("图生视频"):
                i_prompt = gr.Textbox(label="Prompt(描述运动/编辑)", lines=4, value="The subject comes alive, subtle natural motion, same character.")
                i_ref = gr.File(label="首帧参考图(必传 1 张)", file_types=["image"], type="filepath")
                with gr.Row():
                    i_aspect = gr.Dropdown(label="宽高比(关闭跟随时用)", choices=list(ASPECT_RATIOS.keys()), value="16:9 (Widescreen)")
                    i_mp = gr.Dropdown(label="分辨率档位", choices=MP_CHOICES, value="0.35 MP（标准）")
                i_follow = gr.Checkbox(label="跟随参考图分辨率(长边≤1024且显存校准)", value=True)
                with gr.Row():
                    i_dur = gr.Slider(minimum=0.5, maximum=15.0, value=3.0, step=0.5, label="时长(秒)")
                    i_steps = gr.Slider(minimum=1, maximum=50, value=4, step=1, label="采样步数")
                    i_lora = gr.Checkbox(label="LoRA加速", value=True)
                i_btn = gr.Button("生成图生视频", variant="primary")
                i_code = gr.Textbox(label="🔑 取视频码", interactive=False, lines=2)
                i_prog = gr.Markdown("🟢 就绪")
                i_pick = gr.Textbox(label="凭码取回", placeholder="6位码", lines=1)
                i_get = gr.Button("取回视频")
                i_video = gr.Video(label="结果视频", format="mp4")
                i_audio = gr.Audio(label="音频", type="filepath")
                i_btn.click(generate_i2v, [i_prompt, i_ref, i_aspect, i_mp, i_dur, i_steps, i_lora, i_follow],
                            [i_code])
                i_timer = gr.Timer(value=3)
                i_timer.tick(nice_progress, [i_code], [i_prog])
                i_get.click(retrieve_video, [i_pick], [i_video, i_audio])
            # ===== 编图(R2I) =====
            with gr.Tab("编图 / 参考图编辑"):
                gr.Markdown("上传参考图(多张) + 描述要做的编辑; 用 `<Picture 1>` 等引用顺序。\n"
                            "后端=生成短视频后抽帧当图(挑最佳帧)。")
                e_prompt = gr.Textbox(label="编辑 Prompt", lines=5,
                                      value="<Picture 1> is the subject. Edit: keep the subject identical, "
                                            "change the background to a beautiful sunset beach, cinematic.")
                e_ref = gr.File(label="参考图(必传, 可多张)", file_count="multiple",
                                file_types=["image"], type="filepath")
                with gr.Row():
                    e_aspect = gr.Dropdown(label="宽高比(关闭跟随时用)", choices=list(ASPECT_RATIOS.keys()), value="16:9 (Widescreen)")
                    e_mp = gr.Dropdown(label="分辨率档位", choices=MP_CHOICES, value="0.35 MP（标准）")
                e_follow = gr.Checkbox(label="跟随参考图分辨率", value=True)
                with gr.Row():
                    e_dur = gr.Slider(minimum=0.5, maximum=15.0, value=1.5, step=0.5, label="时长(秒, 编图建议0.5~3)")
                    e_steps = gr.Slider(minimum=1, maximum=50, value=4, step=1, label="采样步数")
                    e_lora = gr.Checkbox(label="LoRA加速", value=True)
                e_btn = gr.Button("开始编图", variant="primary")
                e_code = gr.Textbox(label="🔑 取图码", interactive=False, lines=2)
                e_prog = gr.Markdown("🟢 就绪")
                e_pick = gr.Textbox(label="凭码取回", placeholder="6位码", lines=1)
                e_get = gr.Button("取回图片/视频")
                with gr.Tabs():
                    with gr.Tab("输出图片"):
                        e_gallery = gr.Gallery(label="帧图(挑最佳)", columns=3, height=360)
                    with gr.Tab("输出视频"):
                        e_video = gr.Video(label="QC 视频", format="mp4")
                    with gr.Tab("音频"):
                        e_audio = gr.Audio(label="音频", type="filepath")
                e_note = gr.Markdown("")
                e_btn.click(generate_edit, [e_prompt, e_ref, e_aspect, e_mp, e_dur, e_steps, e_lora, e_follow],
                            [e_code])
                e_timer = gr.Timer(value=3)
                e_timer.tick(nice_progress, [e_code], [e_prog])
                e_get.click(retrieve_edit, [e_pick], [e_gallery, e_video, e_audio, e_note])
            # ===== 任务与监控 =====
            with gr.Tab("任务 & 监控"):
                sys_md = gr.Markdown("读取中…")
                act_df = gr.Dataframe(headers=["取码", "类型", "状态", "已运行"], interactive=False)
                with gr.Row():
                    cancel_dd = gr.Dropdown(label="选择取消", choices=[], scale=3)
                    cancel_sel = gr.Button("✕ 取消选中", variant="stop", scale=1)
                    btn_cancel_latest = gr.Button("✕ 取消最新", variant="stop", scale=1)
                log_tb = gr.Textbox(label="📋 任务日志", lines=14, interactive=False)
                def _refresh():
                    acts = active_list()
                    return sysinfo(), acts, [a[0] for a in acts], log_html()
                m_timer = gr.Timer(value=3)
                m_timer.tick(_refresh, [], [sys_md, act_df, cancel_dd, log_tb])
                cancel_sel.click(cancel_code, [cancel_dd], [log_tb])
                btn_cancel_latest.click(cancel_latest_task, [], [log_tb])
        log_timer = gr.Timer(value=5)
        def _top():
            with _JOBS_LOCK:
                n = sum(1 for v in _JOBS.values() if v.get("status") not in ("completed",) and not str(v.get("status")).startswith("failed"))
            if _RUNNING:
                return "🔵 正在生成…(并行度=1, 其余排队)"
            if n:
                return "🟠 有任务排队/处理中"
            return "🟢 就绪: 无任务"
        log_timer.tick(_top, [], [top_status])
    return demo

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=B.ONNX_BASE)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7863)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    B.ONNX_BASE = a.backend
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port,
                                                   share=a.share, inbrowser=False)
