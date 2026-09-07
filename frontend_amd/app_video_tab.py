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
import os, sys, math, random, time, threading, json, shutil
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

def resolve_resolution(aspect_label, megapixels):
    mp_str = str(megapixels).strip()
    if mp_str in ULTRA_LOW_PRESETS:
        return ULTRA_LOW_PRESETS[mp_str]
    w_ratio, h_ratio = ASPECT_RATIOS[aspect_label]
    total = float(megapixels) * 1024 * 1024
    scale = math.sqrt(total / (w_ratio * h_ratio))
    w = round(w_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE
    h = round(h_ratio * scale / RES_MULTIPLE) * RES_MULTIPLE
    return int(w), int(h)

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

# ---------- 生成(后台线程, 并行度1) ----------
def _exec_gen(code, prompt, ref_images, ref_video, ref_audios, seconds, aspect, mp, turbo_steps):
    global _RUNNING
    try:
        w, h = resolve_resolution(aspect, mp) if aspect and mp else (256, 256)
        # 并行度=1 排队门控
        while _RUNNING >= 1:
            time.sleep(2)
        _GEN_LOCK.acquire(); _RUNNING = 1
        try:
            # Ref2VA Turbo 4-step LoRA adapter 已导出(acceleration_ready=True)。
            # 用 turbo 快速路线时开启 use_acceleration_lora(4步最佳; 步数由 turbo_steps 控制)。
            lora = True   # adapter 已就绪, turbo 快速路线开启
            jid = B.submit(prompt=prompt, start_image_path=None, steps=int(turbo_steps),
                           seed=random.randint(0, 2**31 - 1),
                           width=w, height=h, duration_seconds=float(seconds),
                           use_acceleration_lora=lora)
            with _JOBS_LOCK:
                _JOBS[code]["jid"] = jid
                _JOBS[code]["status"] = "running"
            _poll_and_store(code, jid)
        finally:
            _RUNNING = 0; _GEN_LOCK.release()
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[code]["status"] = f"failed: {e}"

def _poll_and_store(code, jid):
    last = ""
    while True:
        done, st = B.poll(jid)
        if done:
            if st.get("status") == "completed":
                mp4 = B.get_video_path(jid)
                local = os.path.join(PICKUP_DIR, f"{code}.mp4")
                B.download_output(jid, local)
                audio = os.path.join(PICKUP_DIR, f"{code}.m4a")
                try:
                    B.extract_audio(local, audio)
                except Exception:
                    audio = None
                with _JOBS_LOCK:
                    _JOBS[code].update(status="completed", out_mp4=local, audio=audio)
            else:
                with _JOBS_LOCK:
                    _JOBS[code]["status"] = f"failed: {st.get('status')}"
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

R2I_DEFAULT_PROMPT = (
    "A cinematic portrait of <Subject 1> from <Picture 1>. "
    "Realistic, natural skin, soft lighting, high detail, shallow depth of field."
)

def _exec_r2i(code, prompt, ref_images, seconds, aspect, mp, turbo_steps):
    global _RUNNING
    try:
        # 并行度=1 排队
        while _RUNNING >= 1:
            time.sleep(2)
        _GEN_LOCK.acquire(); _RUNNING = 1
        try:
            w, h = resolve_resolution(aspect, mp) if aspect and mp else (256, 256)
            save_dir = os.path.join(R2I_PICKUP_DIR, code)
            os.makedirs(save_dir, exist_ok=True)
            _R2I_JOBS[code] = {"status": "running", "code": code}
            # 用第一张参考图作为首帧做图生视频(最贴近"保持参考主体")
            first = ref_images[0] if ref_images else None
            start_path = None
            if first and os.path.isfile(first):
                with open(first, "rb") as f:
                    start_path = B.upload_image(f.read(), os.path.basename(first))
            jid = B.submit(prompt=prompt, start_image_path=start_path, steps=int(turbo_steps),
                           seed=random.randint(0, 2**31 - 1), width=w, height=h,
                           duration_seconds=float(seconds), use_acceleration_lora=True,
                           conditioning="first" if start_path else "text")
            # 轮询直到完成
            done = False
            while not done:
                done, st = B.poll(jid)
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
                _R2I_JOBS[code] = {"status": "completed", "code": code,
                                   "images": frames, "video": mp4, "audio": audio}
            else:
                _R2I_JOBS[code]["status"] = f"failed: {st.get('status')}"
        finally:
            _RUNNING = 0; _GEN_LOCK.release()
    except Exception as e:
        _R2I_JOBS[code] = {"status": f"failed: {e}", "code": code}

def generate_r2i(prompt, r2i_images, seconds, aspect, mp, turbo_steps):
    code = _new_code()
    _R2I_JOBS[code] = {"status": "queued", "code": code}
    imgs = []
    if r2i_images:
        imgs = [r2i_images] if isinstance(r2i_images, str) else list(r2i_images)
    threading.Thread(target=_exec_r2i, args=(code, prompt, imgs, seconds, aspect, mp,
                     turbo_steps), daemon=True).start()
    return f"🔑 {code}\n（你的取图码，请复制保存；生成完成后凭码取回图片/视频，24 小时有效）"

def _r2i_result(code):
    c = str(code or "").strip()
    if len(str(c).split()) > 1:
        c = c.split()[1]
    if not c.isdigit() or len(c) != 6:
        return [], None, None
    j = _R2I_JOBS.get(c)
    if not j or j.get("status") != "completed":
        return [], None, None
    imgs = [x for x in j.get("images", []) if os.path.isfile(x)]
    vid = j.get("video") if os.path.isfile(j.get("video", "")) else None
    aud = j.get("audio") if j.get("audio") and os.path.isfile(j.get("audio")) else None
    return imgs, vid, aud

def check_latest_r2i(pickup_code_text):
    return _r2i_result(pickup_code_text)

def retrieve_r2i(code):
    return _r2i_result(code)

def _sync_picture_tags_r2i(prompt, files):
    tags = _tag_img(files)
    if tags:
        return f"# 参考图: {tags}\n{prompt or ''}"
    return prompt

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
            # ===== Tab 2: R2I 图片编辑(与 AMD 页面一致; 后端=生成视频后抽帧当图) =====
            with gr.Tab("🖼️ R2I 图片编辑"):
                gr.Markdown("上传参考图 → 生成视频 → **自动抽帧为图片输出**（按说明：编图=视频截图）。可另取 QC 视频与单独音频。")
                with gr.Row():
                    r2i_prompt = gr.Textbox(label="提示词（用 <Picture 1> 引用参考图；可六段式描述主体/运镜/音景）",
                                            lines=8, value=R2I_DEFAULT_PROMPT)
                with gr.Row():
                    r2i_images = gr.File(label="参考图片（多图，可选）", file_count="multiple",
                                         file_types=["image"], type="filepath")
                with gr.Row():
                    r2i_aspect = gr.Dropdown(label="宽高比", choices=list(ASPECT_RATIOS.keys()),
                                             value="1:1 (Square)")
                    r2i_mp = gr.Radio(label="分辨率档位", choices=MP_CHOICES, value="0.2 MP（快速）")
                    r2i_seconds = gr.Slider(label="生成时长（秒）", minimum=1, maximum=5, step=1, value=2)
                with gr.Row():
                    r2i_steps = gr.Slider(label="⚡ 快速路线步数（4~8）", minimum=4, maximum=8, value=4, step=1)
                r2i_run = gr.Button("生成（编辑）图片", variant="primary")
                r2i_pickup = gr.Textbox(label="🔑 你的取图码（请复制保存；完成后凭码取回图片/视频）",
                                        value="（点生成后显示取图码）", lines=2, interactive=False)
                r2i_run.click(fn=generate_r2i, inputs=[r2i_prompt, r2i_images, r2i_seconds,
                             r2i_aspect, r2i_mp, r2i_steps], outputs=[r2i_pickup])
                r2i_images.change(fn=_sync_picture_tags_r2i, inputs=[r2i_prompt, r2i_images],
                                  outputs=[r2i_prompt])
                with gr.Row():
                    r2i_latest_gallery = gr.Gallery(label="🖼️ 最新生成图片（视频截图）", columns=3,
                                                    height="auto", object_fit="contain")
                    r2i_latest_video = gr.Video(label="▶ QC 视频", format="mp4")
                    r2i_latest_audio = gr.Audio(label="🔊 单独音频", type="filepath")
                r2i_timer = gr.Timer(value=3)
                r2i_timer.tick(fn=check_latest_r2i, inputs=[r2i_pickup],
                               outputs=[r2i_latest_gallery, r2i_latest_video, r2i_latest_audio])
                with gr.Row():
                    r2i_retrieve_in = gr.Textbox(label="输入取图码（6 位数字）", placeholder="如 654321", scale=3)
                    r2i_retrieve_btn = gr.Button("取回图片/视频", scale=1)
                r2i_retrieve_out_g = gr.Gallery(label="取回的图片", columns=3, height="auto")
                r2i_retrieve_out_v = gr.Video(label="取回的视频", format="mp4")
                r2i_retrieve_out_a = gr.Audio(label="🔊 单独音频", type="filepath")
                r2i_retrieve_btn.click(fn=retrieve_r2i, inputs=[r2i_retrieve_in],
                                       outputs=[r2i_retrieve_out_g, r2i_retrieve_out_v,
                                                r2i_retrieve_out_a])
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
