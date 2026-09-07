#!/usr/bin/env python3
"""🎬 视频生成 Tab —— 与 AMD MiniMax-H3 前端页面/控件一致, 后端接 ONNX WebUI。

复刻 AMD 前端"视频生成"页全部控件与交互:
  - Prompt(支持 <Picture N> 标签)
  - 多图参考(1-5张) / 参考视频(<=5s) / 多段参考音频(<=3段)
  - 宽高比 Dropdown + 分辨率档位 Radio(含 ⚡超低32x32 仅音频)
  - 时长 Slider(1-5s) + ⚡快速路线步数(4-8, turbo)
  - Generate -> 🔑 取视频码 -> 后台生成 -> 凭码取回 + 自动标签同步

后端: 通过 onnx_adapter 调 MinimaxH3-ONNX WebUI(默认 http://127.0.0.1:7860)。
由于 ONNX 后端逐块流式、较慢, 输出用"码 -> job_id"注册表 + 轮询。
"""
import os, sys, math, random, time, threading, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnx_adapter as B  # 后端适配层

import gradio as gr

# ---------- 分辨率选择器(与 AMD 一致) ----------
ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3), "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4), "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16), "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}
RES_MULTIPLE = 32
ULTRA_LOW_PRESETS = {"⚡ 超低 32×32（最低像素，仅音频）": (32, 32)}
ULTRA_LOW_CHOICES = list(ULTRA_LOW_PRESETS.keys())
MP_CHOICES = ULTRA_LOW_CHOICES + ["0.2 MP（快速）", "0.4 MP（标准）", "0.6 MP", "0.8 MP", "1.0 MP（高清）"]

def is_ultra_low(mp): return str(mp).strip() in ULTRA_LOW_PRESETS

import re as _re

def _parse_mp(megapixels):
    """把 '0.2 MP（快速）' 这类 label 解析成数值 0.2。"""
    s = str(megapixels).strip()
    m = _re.search(r"(\d+(?:\.\d+)?)\s*MP", s)
    if m:
        return float(m.group(1))
    # 兜底: 取字符串里第一个数字
    m = _re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.2

def resolve_resolution(aspect_label, megapixels):
    mp_str = str(megapixels).strip()
    if mp_str in ULTRA_LOW_PRESETS:
        return ULTRA_LOW_PRESETS[mp_str]
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_label]
    total = _parse_mp(megapixels) * 1024 * 1024
    scale = math.sqrt(total / (w_ratio * h_ratio))
    w = round(w_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE
    h = round(h_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE
    return max(RES_MULTIPLE, int(w)), max(RES_MULTIPLE, int(h))

# ---------- 取码机制 ----------
PICKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickup")
os.makedirs(PICKUP_DIR, exist_ok=True)
_JOBS = {}          # code -> {jid, status, out_mp4, audio, created}
_JOBS_LOCK = threading.Lock()
_RUNNING = 0        # 并行度=1 门控
_GEN_LOCK = threading.Lock()

def _new_code():
    while True:
        c = f"{random.randint(0, 999999):06d}"
        if c not in _JOBS:
            return c

def _record(code, jid):
    with _JOBS_LOCK:
        _JOBS[code] = {"jid": jid, "status": "queued", "out_mp4": None, "audio": None,
                        "created": time.time()}

def _cleanup_pickup(max_age_h=24, max_codes=200):
    now = time.time()
    with _JOBS_LOCK:
        for c in [k for k, v in _JOBS.items() if now - v["created"] > max_age_h * 3600][:200]:
            _JOBS.pop(c, None)

# ============ 任务监控: 进行中任务 / 日志 / 系统 / 一键取消(个人使用, 点即取消) ============
_ACTIVE = {}          # code -> {mode, created, start, status}
_ACTIVE_LOCK = threading.Lock()
_CANCEL = set()       # 被点 ✕ 取消的 code
_CANCEL_LOCK = threading.Lock()
LOG = []
LOG_LOCK = threading.Lock()
LOG_MAX = 200

def add_log(msg):
    ts = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        LOG.append(f"[{ts}] {msg}")
        del LOG[: max(0, len(LOG) - LOG_MAX)]
    return "\n".join(LOG)

def active_list():
    with _ACTIVE_LOCK:
        now = time.time()
        out = []
        for c, v in list(_ACTIVE.items()):
            if v["status"] in ("completed", "failed", "cancelled"):
                continue
            el = int(now - (v.get("start") or now))
            out.append([c, v.get("mode", ""), v.get("status", ""), f"{el}s"])
        return out

def cancel_code(code):
    """个人使用: 点 ✕ 直接取消指定任务(前端停止轮询并标记取消)。"""
    c = str(code or "").strip()
    if len(str(c).split()) > 1:
        c = c.split()[1]
    if not c.isdigit():
        return f"无法取消: 无效码 {code}"
    with _CANCEL_LOCK:
        _CANCEL.add(c)
    with _JOBS_LOCK:
        if c in _JOBS:
            _JOBS[c]["status"] = "cancelled"
    with _R2I_JOBS_LOCK:
        if c in _R2I_JOBS:
            _R2I_JOBS[c]["status"] = "cancelled"
    add_log(f"✕ 已取消任务 {c}（后端该条可能继续跑完，界面立即释放）")
    return f"已取消 {c}"

def cancel_latest():
    """取消最近提交的一个进行中任务。"""
    with _ACTIVE_LOCK:
        running = [c for c, v in _ACTIVE.items()
                   if v["status"] not in ("completed", "failed", "cancelled")]
    if not running:
        return "当前无进行中任务"
    latest = max(running, key=lambda c: _ACTIVE[c].get("start") or 0)
    return cancel_code(latest)

def sysinfo():
    """系统监控: CPU/内存/GPU/后端。返回 markdown 字符串。"""
    lines = []
    # CPU
    try:
        with open("/proc/loadavg") as f:
            load = f.read().split()[:3]
        lines.append(f"CPU 负载: {' '.join(load)}")
    except Exception:
        pass
    # mem
    try:
        with open("/proc/meminfo") as f:
            d = {}
            for ln in f:
                k, v = ln.split(":", 1)
                d[k.strip()] = int(v.split()[0]) / 1048576
        used = d.get("MemTotal", 0) - d.get("MemAvailable", 0)
        lines.append(f"内存: {used:.1f} / {d.get('MemTotal', 0):.1f} GiB")
    except Exception:
        pass
    # gpu
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.stdout.strip():
            rows = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
            if rows:
                name, mu, mt, gu = [x.strip() for x in rows[0].split(",")]
                gpu_n = f"GPU0: {name}" if len(rows) > 1 else f"GPU: {name}"
                lines.append(f"{gpu_n}  {mu}/{mt}MiB  util {gu}%")
    except Exception:
        pass
    # backend
    try:
        lines.append("后端: " + ("✅ 就绪" if B.onnx_health() else "⚠️ 离线"))
    except Exception:
        pass
    return "  |  ".join(lines)

def task_log_html():
    with LOG_LOCK:
        return "\n".join(LOG[-60:]) or "（暂无日志）"

# ---------- 生成(后台线程, 并行度1) ----------
def _is_cancelled(code):
    with _CANCEL_LOCK:
        return code in _CANCEL

def _upload_refs(ref_images, ref_video, ref_audios):
    """上传引用资源, 返回 references 列表(供 submit references= 参数)。"""
    refs = []
    if ref_images:
        files = ref_images if isinstance(ref_images, list) else [ref_images]
        for i, fp in enumerate(files, 1):
            if fp and os.path.isfile(fp):
                try:
                    path = B.upload_image(fp)
                    refs.append({"type": "image", "path": path, "index": i})
                except Exception as e:
                    add_log(f"⚠️ 上传参考图失败: {e}")
    if ref_video:
        # ref_video 来自 gr.Video, 可能是路径或 dict
        vp = ref_video if isinstance(ref_video, str) else (ref_video.get("path") if isinstance(ref_video, dict) else None)
        if vp and os.path.isfile(str(vp)):
            refs.append({"type": "video", "path": str(vp)})
    if ref_audios:
        files = ref_audios if isinstance(ref_audios, list) else [ref_audios]
        for fp in files:
            if fp and os.path.isfile(str(fp)):
                refs.append({"type": "audio", "path": str(fp)})
    return refs

def _exec_gen(code, prompt, ref_images, ref_video, ref_audios, seconds, aspect, mp, turbo_steps):
    global _RUNNING
    try:
        with _ACTIVE_LOCK:
            _ACTIVE[code] = {"mode": "video", "created": time.time(), "start": time.time(),
                             "status": "queued"}
        add_log(f"🎬 视频任务 {code} 入队")
        w, h = resolve_resolution(aspect, mp) if aspect and mp else (256, 256)
        # 并行度=1 排队门控
        while _RUNNING >= 1:
            if _is_cancelled(code):
                return
            time.sleep(2)
        _GEN_LOCK.acquire(); _RUNNING = 1
        try:
            if _is_cancelled(code):
                return
            with _ACTIVE_LOCK:
                _ACTIVE[code]["start"] = time.time(); _ACTIVE[code]["status"] = "running"
            # 检查后端 turbo LoRA 就绪状态
            lora = bool(turbo_steps and int(turbo_steps) <= 8)
            if lora:
                try:
                    lora = B.onnx_profile_generation_ready()
                except Exception:
                    pass
            # 上传引用资源(ref2va)
            references = _upload_refs(ref_images, ref_video, ref_audios)
            # 首帧图上传
            start_path = None
            if ref_images:
                first = ref_images[0] if isinstance(ref_images, list) else ref_images
                if first and os.path.isfile(first):
                    try:
                        start_path = B.upload_image(first)
                    except Exception as e:
                        add_log(f"⚠️ 上传首帧图失败: {e}")
            jid = B.submit(prompt=prompt, start_image_path=start_path, steps=int(turbo_steps),
                           seed=random.randint(0, 2**31 - 1),
                           width=w, height=h, duration_seconds=float(seconds),
                           use_acceleration_lora=lora,
                           references=references if references else None)
            with _JOBS_LOCK:
                _JOBS[code]["jid"] = jid
                _JOBS[code]["status"] = "running"
            add_log(f"视频任务 {code} 已提交后端(jid={jid[:8]}…)，开始生成")
            _poll_and_store(code, jid)
        finally:
            _RUNNING = 0; _GEN_LOCK.release()
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[code]["status"] = f"failed: {e}"
        add_log(f"视频任务 {code} 失败: {e}")
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE[code]["status"] = _JOBS.get(code, {}).get("status", "done")

def _poll_and_store(code, jid):
    last = ""
    while True:
        if _is_cancelled(code):
            with _JOBS_LOCK:
                _JOBS[code]["status"] = "cancelled"
            return
        done, st = B.poll(jid)
        if done:
            if st.get("status") == "completed":
                local = os.path.join(PICKUP_DIR, f"{code}.mp4")
                B.download_output(jid, local)
                audio = os.path.join(PICKUP_DIR, f"{code}.m4a")
                try:
                    B.extract_audio(local, audio)
                except Exception:
                    audio = None
                with _JOBS_LOCK:
                    _JOBS[code].update(status="completed", out_mp4=local, audio=audio)
                add_log(f"🎬 视频任务 {code} 完成 -> {local}")
            else:
                with _JOBS_LOCK:
                    _JOBS[code]["status"] = f"failed: {st.get('status')}"
                add_log(f"🎬 视频任务 {code} 失败: {st.get('status')}")
            return
        msg = st.get("message") or ""
        if msg != last:
            last = msg
            with _JOBS_LOCK:
                _JOBS[code]["status"] = msg
        time.sleep(5)

def generate(prompt, ref_images, ref_video, ref_audios, seconds, aspect, mp, turbo_steps):
    """Generate 回调: 立刻发码, 后台生成。返回给界面的取码字符串。"""
    code = _new_code()
    _record(code, None)
    threading.Thread(target=_exec_gen, args=(code, prompt, ref_images, ref_video,
                     ref_audios, seconds, aspect, mp, turbo_steps), daemon=True).start()
    return f"🔑 {code}\n（你的取视频码，请复制保存；生成完成后凭码在下方取回，24 小时有效）"

def _tag_img(ref_images):
    tags = []
    if ref_images:
        files = ref_images if isinstance(ref_images, list) else [ref_images]
        for i, _ in enumerate(files, 1):
            tags.append(f"<Picture {i}>")
    return " ".join(tags)

def _sync_picture_tags_video(prompt, ref_images):
    base = prompt or ""
    tags = _tag_img(ref_images)
    if tags:
        # 在提示词顶部插入引用标签行
        return f"{tags}\n{base}"
    return base

def check_latest_video(code):
    if not code or "取视频码" in str(code):
        return None, None
    # 解析 6 位数字
    c = str(code).strip().split()[1] if len(str(code).split()) > 1 else str(code).strip()
    if not c.isdigit() or len(c) != 6:
        return None, None
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v:
        return None, None
    if v["out_mp4"] and os.path.isfile(v["out_mp4"]):
        return v["out_mp4"], (v["audio"] if v["audio"] and os.path.isfile(v["audio"]) else None)
    return None, None

def retrieve_video(inp):
    c = str(inp or "").strip()
    if not c.isdigit() or len(c) != 6:
        return None, None
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v or not v["out_mp4"] or not os.path.isfile(v["out_mp4"]):
        return None, None
    return v["out_mp4"], (v["audio"] if v["audio"] and os.path.isfile(v["audio"]) else None)

def refresh_status():
    with _JOBS_LOCK:
        running = sum(1 for v in _JOBS.values() if v["status"] not in ("completed",) and not str(v["status"]).startswith("failed"))
    if _RUNNING:
        return "🔵 正在生成…（并行度=1，其余任务自动排队）"
    if running:
        return "🟠 有任务排队/处理中"
    return "🟢 服务就绪：暂无任务，点击 Generate 开始生成"

# ============ R2I 图片编辑(按说明: 生成视频后抽帧当图) ============
R2I_PICKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickup_r2i")
os.makedirs(R2I_PICKUP_DIR, exist_ok=True)
_R2I_JOBS = {}   # code -> status dict
_R2I_JOBS_LOCK = threading.Lock()

R2I_DEFAULT_PROMPT = """subject_definitions:
<Subject 1> is the person in <Picture 1>: a person with their original facial identity, hairstyle, facial features, skin tone, clothing, body proportions and overall appearance.

summary:
[reference generation] Edit <Subject 1> by changing only the background to a beautiful sunset beach while preserving identity, appearance and composition.

retention_analysis:
<Subject 1> (appears in [Shot 1]): partially_preserved - the person's identity is retained exactly: face, hairstyle, eyes, facial features, skin tone, clothing, body proportions and overall appearance remain unchanged. The original background is replaced with a new background: a beautiful sunset beach with golden sand, ocean waves and warm orange sky.

detailed_description:
The target image uses a realistic photography style with natural skin texture, realistic lighting, detailed hair strands and high-quality portrait rendering.

[Shot 1] A static shot frames <Subject 1> while preserving the original camera angle and composition. The person keeps the same face, hairstyle, clothing and identity. The background is changed to a beautiful sunset beach scene.

overall_soundscape:
N/A

non_diegetic_music:
N/A

# ---- 多图引用说明 ----
# 上传多张图时，Prompt 中用 <Picture 1> <Picture 2> <Picture 3> ... 引用对应顺序的参考图。
# 例如：将 <Picture 1> 的人放到 <Picture 2> 的场景中：
#   <Subject 1> is the person in <Picture 1>.
#   <Subject 2> is the scene/location in <Picture 2>.
#   [reference generation] Place <Subject 1> in the environment of <Subject 2>."""

def _pick_aspect_mp(follow_ref, aspect_label, megapixels, ref_images):
    """按 AMD 语义决定分辨率：跟随参考图分辨率(读图尺寸) 或 手动 aspect×MP。"""
    if follow_ref and ref_images:
        try:
            from PIL import Image
            p0 = ref_images[0]
            if os.path.isfile(p0):
                with Image.open(p0) as im:
                    w, h = im.size
                w = max(32, int(round(w / 32) * 32))
                h = max(32, int(round(h / 32) * 32))
                return min(w, 1024), min(h, 1024)
        except Exception:
            pass
        return 256, 256
    return resolve_resolution(aspect_label, megapixels) if aspect_label and megapixels else (256, 256)

def _exec_r2i(code, prompt, ref_images, duration_seconds, use_lora, follow_ref, aspect_label, megapixels):
    global _RUNNING
    try:
        with _ACTIVE_LOCK:
            _ACTIVE[code] = {"mode": "r2i", "created": time.time(), "start": time.time(),
                             "status": "queued"}
        add_log(f"🖼️ R2I 任务 {code} 入队")
        while _RUNNING >= 1:
            if _is_cancelled(code):
                return
            time.sleep(2)
        _GEN_LOCK.acquire(); _RUNNING = 1
        try:
            if _is_cancelled(code):
                return
            with _ACTIVE_LOCK:
                _ACTIVE[code]["start"] = time.time(); _ACTIVE[code]["status"] = "running"
            w, h = _pick_aspect_mp(follow_ref, aspect_label, megapixels, ref_images)
            save_dir = os.path.join(R2I_PICKUP_DIR, code)
            os.makedirs(save_dir, exist_ok=True)
            with _R2I_JOBS_LOCK:
                _R2I_JOBS[code] = {"status": "running", "code": code}
            # 用第一张参考图作为首帧做"图生视频"(最贴近"保持参考主体"), 其余参考图并入 prompt 语义
            first = ref_images[0] if ref_images else None
            start_path = None
            if first and os.path.isfile(first):
                start_path = B.upload_image(first)
            steps = 4 if use_lora else 20   # AMD: LoRA开=4步(turbo)/关=20步
            jid = B.submit(prompt=prompt, start_image_path=start_path, steps=int(steps),
                           seed=random.randint(0, 2**31 - 1), width=w, height=h,
                           duration_seconds=float(duration_seconds), use_acceleration_lora=bool(use_lora),
                           conditioning="first" if start_path else "text")
            add_log(f"R2I 任务 {code} 已提交后端(jid={jid[:8]}…)")
            done = False
            while not done:
                if _is_cancelled(code):
                    with _R2I_JOBS_LOCK:
                        _R2I_JOBS[code]["status"] = "cancelled"
                    return
                done, st = B.poll(jid)
                with _R2I_JOBS_LOCK:
                    _R2I_JOBS[code]["status"] = st.get("message") or st.get("status") or "running"
                if not done:
                    time.sleep(5)
            if st.get("status") == "completed":
                mp4 = os.path.join(save_dir, "video.mp4")
                B.download_output(jid, mp4)
                img_dir = os.path.join(save_dir, "images")
                os.makedirs(img_dir, exist_ok=True)
                frames = B.extract_all_frames(mp4, img_dir)   # 视频截图当图片
                audio = os.path.join(save_dir, "audio.m4a")
                try:
                    B.extract_audio(mp4, audio)
                except Exception:
                    audio = None
                with _R2I_JOBS_LOCK:
                    _R2I_JOBS[code] = {"status": "completed", "code": code,
                                       "images": frames, "video": mp4, "audio": audio}
                add_log(f"R2I 任务 {code} 完成: {len(frames)} 张图片")
            else:
                with _R2I_JOBS_LOCK:
                    _R2I_JOBS[code]["status"] = f"failed: {st.get('status')}"
        finally:
            _RUNNING = 0; _GEN_LOCK.release()
    except Exception as e:
        with _R2I_JOBS_LOCK:
            _R2I_JOBS[code] = {"status": f"failed: {e}", "code": code}
        add_log(f"R2I 任务 {code} 失败: {e}")
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE[code]["status"] = _R2I_JOBS.get(code, {}).get("status", "done")

def generate_r2i(prompt, r2i_images, duration_seconds, use_lora, follow_ref, aspect, megapixels):
    code = _new_code()
    _R2I_JOBS[code] = {"status": "queued", "code": code}
    imgs = []
    if r2i_images:
        imgs = [r2i_images] if isinstance(r2i_images, str) else list(r2i_images)
    threading.Thread(target=_exec_r2i,
                     args=(code, prompt, imgs, duration_seconds, use_lora, follow_ref,
                           aspect, megapixels), daemon=True).start()
    return f"🔑 {code}\n（你的取图码，请复制保存；生成完成后凭码取回图片/视频，24 小时有效）"

def _r2i_result(code):
    """返回 (images, video, download, audio) 四元组(与 AMD 前端一致)。"""
    c = str(code or "").strip()
    if len(str(c).split()) > 1:
        c = c.split()[1]
    if not c.isdigit() or len(c) != 6:
        return [], None, None, None
    j = _R2I_JOBS.get(c)
    if not j or j.get("status") != "completed":
        return [], None, None, None
    imgs = [x for x in j.get("images", []) if os.path.isfile(x)]
    vid = j.get("video") if os.path.isfile(j.get("video", "")) else None
    aud = j.get("audio") if j.get("audio") and os.path.isfile(j.get("audio")) else None
    return imgs, vid, vid, aud   # download = 视频文件路径

def check_latest_r2i(pickup_code_text):
    return _r2i_result(pickup_code_text)

def retrieve_r2i(code):
    return _r2i_result(code)

def _sync_picture_tags_r2i(prompt, files):
    tags = _tag_img(files)
    if tags:
        return f"{tags}\n{prompt or ''}"
    return prompt

def _extract_code(code_text):
    c = str(code_text or "").strip()
    if len(str(c).split()) > 1:
        c = c.split()[1]
    c = c.strip()
    return c if c.isdigit() and len(c) == 6 else None

def _nice_progress(status):
    """把后端 job 状态/message 加工成前端可读进度行。"""
    s = str(status or "")
    if s in ("queued", "running", "completed", "cancelled"):
        return f"🟡 {s}…"
    # 后端 message 形如: 'Ref2VA: main_block_37_attention_qkv' / 'Qwen: MLP' 等
    # 提取 block 编号高亮
    import re as _re
    m = _re.search(r"main_block_(\d+)_([a-z_]+)", s)
    if m:
        block, op = int(m.group(1)), m.group(2)
        return f"🎬 {s}  (主块 {block}/50 · {op})"
    return f"🟡 {s}"

def progress_for_video(code_text):
    c = _extract_code(code_text)
    if not c:
        return "🟢 就绪：点 Generate 开始生成（此区会实时显示 block 进度）"
    with _JOBS_LOCK:
        v = _JOBS.get(c)
    if not v:
        return "🟡 等待任务初始化…"
    return _nice_progress(v.get("status"))

def progress_for_r2i(code_text):
    c = _extract_code(code_text)
    if not c:
        return "🟢 就绪：上传参考图后点生成"
    with _R2I_JOBS_LOCK:
        v = _R2I_JOBS.get(c)
    if not v:
        return "🟡 等待任务初始化…"
    return _nice_progress(v.get("status"))

# ============ 内存上限设置(适配不同设备) ============
# 后端由 run-h3-backend-24gb.sh 启动时会放进 /sys/fs/cgroup/h3backend
# 这里提供在页面上动态调整该 cgroup 内存上限的能力(需 root / 已用 wrapper 启动)。
H3_CGROUP = "/sys/fs/cgroup/h3backend"
MEM_PRESETS = {"自动(不限制)": 0, "8 GB（轻量/老设备）": 8, "12 GB": 12,
               "16 GB": 16, "20 GB": 20, "24 GB（推荐,T4/云）": 24,
               "28 GB": 28, "32 GB": 32}

def _cg(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None

def _cg_write(path, val):
    with open(path, "w") as f:
        f.write(str(val))

def _cgroup_available():
    return os.path.isdir(H3_CGROUP) and os.access(H3_CGROUP + "/memory.max", os.W_OK)

def get_mem_status():
    """返回当前 cgroup 内存限制/用量的可读文本。"""
    if not _cgroup_available():
        return ("⚠️ 未检测到 h3backend cgroup。\n"
                "请用 run-h3-backend-24gb.sh 启动后端，才能在此页调整内存上限。\n"
                "（直接 ./launch-webui-linux.sh 启动则无 cgroup 内存控制）")
    max_b = int(_cg(H3_CGROUP + "/memory.max") or 0)
    high_b = int(_cg(H3_CGROUP + "/memory.high") or 0)
    cur_b = int(_cg(H3_CGROUP + "/memory.current") or 0)
    # anon
    anon_b = 0
    for ln in open(H3_CGROUP + "/memory.stat"):
        k, v = ln.split()
        if k == "anon":
            anon_b = int(v)
    g = 1024 ** 3
    cur_g = cur_b / g
    return (f"**后端 cgroup 内存限制**：\n"
            f"- 硬上限 memory.max = {max_b/g:.1f} GB\n"
            f"- 软限 memory.high = {high_b/g:.1f} GB\n"
            f"- 当前用量 = {cur_g:.1f} GB（进程 anon = {anon_b/g:.1f} GB）\n"
            f"- 提示：把上限调到**高于当前用量**再应用，否则可能触发后端 OOM。")

def set_mem_limit(gb):
    """把 h3backend cgroup 的 memory.max 设为 gb GB(0=不限制)。返回状态文本。"""
    if not _cgroup_available():
        return "⚠️ 后端未运行在 h3backend cgroup 中，无法设置。请用 run-h3-backend-24gb.sh 启动。"
    g = 1024 ** 3
    if gb and gb > 0:
        max_b = gb * g
        high_b = max(2 * g, max_b - 2 * g)   # soft limit 略低于硬限
        try:
            _cg_write(H3_CGROUP + "/memory.max", max_b)
            if os.path.exists(H3_CGROUP + "/memory.high"):
                _cg_write(H3_CGROUP + "/memory.high", high_b)
            return f"✅ 已设置后端内存上限 = {gb} GB（high={high_b/g:.1f}G）"
        except Exception as e:
            return f"❌ 设置失败: {e}"
    else:
        # 0 = 不限制
        try:
            _cg_write(H3_CGROUP + "/memory.max", "max")
            return "✅ 已设置为不限制内存"
        except Exception as e:
            return f"❌ 设置失败: {e}"

def apply_mem_number(gb):
    # gb 为 None 或 <=0 视为"不限制"
    try:
        gb = int(float(gb or 0))
    except Exception:
        gb = 0
    if gb <= 0:
        return set_mem_limit(0)
    return set_mem_limit(gb)

def build():
    with gr.Blocks(title="MiniMax-H3 · ONNX 低显存") as demo:
        gr.Markdown("## MiniMax-H3 · ONNX 低显存 · ref2va 版\n"
                    "标签页切换功能：**🎬 视频生成**（文生/图生视频）| 🖼️ R2I 图片编辑（后续接入）\n"
                    "⚠️ 首次生成较慢（模型逐块流式载入），**不要关闭**；并行度=1，同时使用自动排队。")
        status = gr.Markdown("🟢 服务就绪：暂无任务，点击 Generate 开始生成")
        with gr.Tabs():
            # ===== Tab 1: 视频生成(与 AMD 页面一致) =====
            with gr.Tab("🎬 视频生成"):
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=6,
                                        value='A red apple on a wooden table, soft light, cinematic')
                with gr.Row():
                    ref_images = gr.File(label="参考图片（多图上传，1-5 张，可选）", file_count='multiple',
                                         file_types=['image'], type="filepath")
                with gr.Row():
                    ref_video = gr.Video(label="参考视频（可选，≤5秒）", format="mp4")
                    ref_audios = gr.Files(label="参考音频（可选，最多 3 段，每段≤5秒）", file_count='multiple',
                                          file_types=['audio'], type="filepath")
                with gr.Row():
                    aspect_ratio = gr.Dropdown(label="宽高比", choices=list(ASPECT_RATIOS.keys()),
                                               value="16:9 (Widescreen)")
                    megapixels = gr.Radio(label="分辨率档位（⚡超低档=快速生成音频，可当 TTS 用）",
                                          choices=MP_CHOICES, value="0.2 MP（快速）")
                    seconds = gr.Slider(label="时长（秒）", minimum=1, maximum=5, step=1, value=3)
                with gr.Row():
                    turbo_steps = gr.Slider(label="⚡ 快速路线步数（4=最快，6~8=质量更好）", minimum=4,
                                            maximum=8, step=1, value=4)
                run_btn = gr.Button("Generate", variant="primary")
                pickup_code = gr.Textbox(label="🔑 你的取视频码（请复制保存；生成完成后凭码取回，24 小时有效）",
                                         value="（点 Generate 后，这里立即显示你的取视频码）",
                                         interactive=False, lines=2)
                run_btn.click(fn=generate, inputs=[prompt, ref_images, ref_video, ref_audios,
                             seconds, aspect_ratio, megapixels, turbo_steps],
                             outputs=[pickup_code])
                ref_images.change(fn=_sync_picture_tags_video, inputs=[prompt, ref_images], outputs=[prompt])
                prog_md = gr.Markdown("🟢 就绪：点 Generate 开始生成（此区实时显示 block 进度）")
                prog_timer = gr.Timer(value=3)
                prog_timer.tick(fn=progress_for_video, inputs=[pickup_code], outputs=[prog_md])
                latest_output = gr.Video(label="▶ 最新生成的视频（仅当前页面；刷新后请凭码取回）", format="mp4")
                latest_audio = gr.Audio(label="🔊 单独音频（仅当前页面，可单独下载）", type="filepath")
                latest_timer = gr.Timer(value=3)
                latest_timer.tick(fn=check_latest_video, inputs=[pickup_code],
                                  outputs=[latest_output, latest_audio])
                with gr.Row():
                    retrieve_input = gr.Textbox(label="输入取视频码（6 位数字）", placeholder="如 123456",
                                                lines=1, scale=3)
                    retrieve_btn = gr.Button("取回视频", scale=1)
                retrieve_output = gr.Video(label="取回的视频")
                retrieve_audio = gr.Audio(label="🔊 单独音频（可单独下载）", type="filepath")
                retrieve_btn.click(fn=retrieve_video, inputs=[retrieve_input],
                                   outputs=[retrieve_output, retrieve_audio])
            # ===== Tab 2: R2I 图片编辑(布局与 AMD/魔搭原版一致; 后端=生成视频后抽帧当图) =====
            with gr.Tab("🖼️ R2I 图片编辑"):
                gr.Markdown("### 参考图编辑 (Reference-to-Image)\n"
                            "上传参考图（**支持多图**）+ 六段式 Prompt，将视频模型当图像编辑器用。\n"
                            "Prompt 中用 `<Picture 1>` `<Picture 2>` `<Picture 3>` ... 引用对应顺序的参考图。\n"
                            "输出所有帧图片（挑最好的帧）+ QC 视频。\n"
                            "⚠️ 生成需排队等待，**点生成后可离开页面**，凭取图码 24 小时内随时取回。")
                with gr.Row():
                    with gr.Column(scale=1):
                        r2i_images = gr.File(label="参考图片（必传，可多选上传）",
                                             file_count="multiple", file_types=["image"], type="filepath")
                        r2i_prompt = gr.Textbox(label="提示词（六段式语法，用 <Picture 1> <Picture 2> ... 引用多图）",
                                                lines=15, value=R2I_DEFAULT_PROMPT)
                        with gr.Accordion("高级设置", open=False):
                            r2i_follow_ref = gr.Checkbox(value=True, label="跟随参考图分辨率（关闭则手动选择宽高比+分辨率）")
                            r2i_aspect = gr.Dropdown(label="宽高比", choices=list(ASPECT_RATIOS.keys()),
                                                     value="16:9 (Widescreen)", visible=False)
                            r2i_megapixels = gr.Radio(label="分辨率档位（⚡超低档=快速生成音频，可当 TTS 用）",
                                                      choices=MP_CHOICES, value="1.0 MP（高清）", visible=False)
                            r2i_duration = gr.Slider(0.2, 5.0, value=0.9, step=0.1,
                                                     label="时长（秒 · 最高 5 秒 · 0.9s=22帧候选）")
                            r2i_lora = gr.Checkbox(value=True, label="LoRA加速（turbo·开=4步 / 关=20步）")
                        r2i_btn = gr.Button("生成", variant="primary")
                        r2i_pickup = gr.Textbox(
                            label="🔑 你的取图码（请复制保存；生成完成后可凭码取回图片和视频，24 小时有效）",
                            value="（点生成后，这里立即显示你的取图码）", interactive=False, lines=2)
                        r2i_prog_md = gr.Markdown("🟢 就绪：上传参考图后点生成（此区实时显示 block 进度）")
                        r2i_prog_timer = gr.Timer(value=3)
                        r2i_prog_timer.tick(fn=progress_for_r2i, inputs=[r2i_pickup], outputs=[r2i_prog_md])
                    with gr.Column(scale=1):
                        # ---- 最新输出（当前页面轮询，刷新后凭码取回） ----
                        gr.Markdown("#### ▶ 最新生成结果（仅当前页面；刷新后请凭码取回）")
                        with gr.Tabs():
                            with gr.Tab("输出图片"):
                                r2i_latest_gallery = gr.Gallery(label="所有帧（挑选最佳）", columns=3, height=400)
                            with gr.Tab("输出视频"):
                                r2i_latest_video = gr.Video(label="QC 视频", format="mp4")
                                r2i_latest_download = gr.File(label="📥 下载视频", interactive=False)
                            with gr.Tab("输出音频"):
                                r2i_latest_audio = gr.Audio(label="🔊 单独音频（可单独下载）", type="filepath")
                        r2i_timer = gr.Timer(value=3)
                        r2i_timer.tick(fn=check_latest_r2i, inputs=[r2i_pickup],
                                       outputs=[r2i_latest_gallery, r2i_latest_video,
                                                r2i_latest_download, r2i_latest_audio])
                        # ---- 凭码取回区 ----
                        gr.Markdown("---\n#### 🔑 凭取图码取回（离开页面后重新加载图片和视频）")
                        with gr.Row():
                            r2i_retrieve_in = gr.Textbox(label="输入取图码（6 位数字）",
                                                         placeholder="如 123456", lines=1, scale=3)
                            r2i_retrieve_btn = gr.Button("取回图片和视频", scale=1)
                        with gr.Tabs():
                            with gr.Tab("取回图片"):
                                r2i_retrieve_out_g = gr.Gallery(label="取回的图片", columns=3, height=400)
                            with gr.Tab("取回视频"):
                                r2i_retrieve_out_v = gr.Video(label="取回的视频", format="mp4")
                                r2i_retrieve_out_dl = gr.File(label="📥 下载视频", interactive=False)
                            with gr.Tab("取回音频"):
                                r2i_retrieve_out_a = gr.Audio(label="🔊 单独音频（可单独下载）", type="filepath")
                        r2i_retrieve_btn.click(fn=retrieve_r2i, inputs=[r2i_retrieve_in],
                                               outputs=[r2i_retrieve_out_g, r2i_retrieve_out_v,
                                                        r2i_retrieve_out_dl, r2i_retrieve_out_a])
                # 跟随参考图分辨率 时隐藏手动宽高比/分辨率档位(与 AMD 一致)
                r2i_follow_ref.change(
                    fn=lambda checked: [gr.update(visible=not checked), gr.update(visible=not checked)],
                    inputs=[r2i_follow_ref], outputs=[r2i_aspect, r2i_megapixels])
                r2i_btn.click(fn=generate_r2i,
                              inputs=[r2i_prompt, r2i_images, r2i_duration, r2i_lora,
                                      r2i_follow_ref, r2i_aspect, r2i_megapixels],
                              outputs=[r2i_pickup])
                r2i_images.change(fn=_sync_picture_tags_r2i, inputs=[r2i_prompt, r2i_images],
                                  outputs=[r2i_prompt])
            # ===== Tab 3: 任务与监控 =====
            with gr.Tab("📋 任务 & 监控"):
                gr.Markdown("个人使用：**在下方选进行中任务 → 点 ✕ 取消**；或点「✕ 取消最新」。取消后界面立即释放。")
                sysmon_md = gr.Markdown("读取系统信息…")
                with gr.Accordion("💾 后端内存上限设置（适配不同设备）", open=False):
                    mem_status_md = gr.Markdown(get_mem_status())
                    gr.Markdown("**手动输入内存上限（GB）**，输入后点“应用”。参考：T4/云=24，16GB 机=12~16，内存紧张机越小越稳；**填 0 = 不限制**。")
                    with gr.Row():
                        mem_input = gr.Number(label="后端内存上限 (GB)，0=不限制", value=24, minimum=0,
                                              precision=0, scale=2)
                        mem_apply = gr.Button("应用内存上限", variant="primary", scale=1)
                    mem_apply.click(fn=apply_mem_number, inputs=[mem_input], outputs=[mem_status_md])
                    mem_status_timer = gr.Timer(value=5)
                    mem_status_timer.tick(fn=get_mem_status, outputs=[mem_status_md])
                with gr.Row():
                    active_df = gr.Dataframe(headers=["取码", "类型", "状态", "已运行"],
                                             label="🗂️ 进行中任务", interactive=False,
                                             wrap=True)
                with gr.Row():
                    cancel_dd = gr.Dropdown(label="选择要取消的任务（选码）", choices=[], scale=3)
                    cancel_sel = gr.Button("✕ 取消选中任务", variant="stop", scale=1)
                    cancel_latest_btn = gr.Button("✕ 取消最新任务", variant="stop", scale=1)
                cancel_msg = gr.Markdown("")
                task_log = gr.Textbox(label="📋 任务日志", lines=14, interactive=False)
                # 轮询刷新
                mon_timer = gr.Timer(value=3)
                def _refresh_all():
                    acts = active_list()
                    return (sysinfo(), acts, [a[0] for a in acts], task_log_html())
                mon_timer.tick(fn=_refresh_all,
                               outputs=[sysmon_md, active_df, cancel_dd, task_log])
                cancel_sel.click(fn=cancel_code, inputs=[cancel_dd], outputs=[cancel_msg])
                cancel_latest_btn.click(fn=cancel_latest, outputs=[cancel_msg])
                # 日志也在顶部刷新
                log_timer2 = gr.Timer(value=5)
                log_timer2.tick(fn=task_log_html, outputs=[task_log])
        log_timer = gr.Timer(value=5)
        log_timer.tick(fn=refresh_status, outputs=[status])
        _cleanup_pickup()
    return demo

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=B.ONNX_BASE)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    B.ONNX_BASE = a.backend
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(server_name=a.host, server_port=a.port,
                                                   share=a.share, inbrowser=False)
