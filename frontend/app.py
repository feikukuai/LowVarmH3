#!/usr/bin/env python3
"""H3 视频生成 Gradio 前端 —— 驱动 MinimaxH3-ONNX WebUI(7860) 出片。

用法:  python app.py [--backend http://127.0.0.1:7860] [--port 7861]
依赖:  gradio>=4, requests (已装到 .venv)
后端须已用 launch-webui-linux.sh 启动并完成模型导出。
"""
import argparse, json, time, threading, urllib.request, urllib.error, urllib.parse, os, uuid

import gradio as gr
import requests

BACKEND = "http://127.0.0.1:7860"

def api_get(path):
    r = requests.get(BACKEND + path, timeout=30)
    r.raise_for_status()
    return r.json()

def api_post(path, payload):
    r = requests.post(BACKEND + path, json=payload, timeout=60)
    if not (200 <= r.status_code < 300):
        try:
            raise RuntimeError(f"HTTP {r.status_code}: {r.json()}")
        except Exception:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    # 后端提交返回 202 Accepted(含 job 对象)
    return r.json()

def upload_image(raw_bytes: bytes, filename: str) -> str:
    """上传首帧参考图, 返回后端 path"""
    r = requests.post(BACKEND + "/api/images/upload",
                      data=raw_bytes,
                      headers={"x-filename": urllib.parse.quote(filename)},
                      timeout=60)
    r.raise_for_status()
    return r.json()["path"]

def submit_job(prompt, start_img_path, steps, seed, width, height, dur):
    payload = dict(prompt=prompt, steps=int(steps), seed=int(seed),
                   width=int(width), height=int(height), duration_seconds=float(dur),
                   temporal_mode="segmented",
                   conditioning_mode=("first" if start_img_path else "text"),
                   start_image_path=start_img_path)
    job = api_post("/api/jobs/inference", payload)
    return job["id"]

def _run(prompt, start_image, steps, seed, width, height, dur):
    # start_image: 若为 None 则纯文本; 若为 dict(gradio Image 输出)取 path
    start_img_path = None
    if start_image is not None:
        imgp = start_image if isinstance(start_image, str) else start_image.get("path") or start_image.get("orig_name")
        if imgp and os.path.isfile(imgp):
            with open(imgp, "rb") as f:
                data = f.read()
            start_img_path = upload_image(data, os.path.basename(imgp))
            yield "首帧已上传", None
        elif imgp:
            start_img_path = imgp
    try:
        jid = submit_job(prompt, start_img_path, steps, seed, width, height, dur)
    except Exception as e:
        yield f"提交失败: {e}", None
        return
    yield f"已提交任务 {jid}（正在排队/生成，请稍候…）", None
    last = ""
    while True:
        time.sleep(4)
        try:
            st = api_get(f"/api/jobs/{jid}")
        except Exception as e:
            yield f"后端异常: {e}", None
            return
        msg = st.get("message") or st.get("status") or ""
        prog = st.get("progress")
        if msg != last:
            last = msg
            yield f"[{prog:.1%}] {msg}", None
        s = st.get("status")
        if s == "completed":
            out = st["result"]["output"]
            yield "✅ 生成完成", out
            return
        if s in ("failed", "cancelled"):
            yield f"❌ {s}: {json.dumps(st, ensure_ascii=False)[:500]}", None
            return

def build():
    with gr.Blocks(title="MiniMax-H3 视频生成 (ONNX 低显存)") as demo:
        gr.Markdown("# 🎬 MiniMax-H3 视频生成\n低显存 ONNX 后端出片。分辨率/时长越小越快。")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="提示词", lines=4,
                                    value="a red apple on a wooden table, soft light")
                start_image = gr.Image(label="首帧参考图(可选)", type="filepath", height=200)
            with gr.Column(scale=2):
                width = gr.Dropdown(label="宽度", value=128, choices=[128, 256, 384, 512, 640, 768])
                height = gr.Dropdown(label="高度", value=128, choices=[128, 256, 384, 512, 640, 768])
                steps = gr.Slider(label="采样步数", minimum=1, maximum=12, value=1, step=1)
                dur = gr.Slider(label="时长(秒)", minimum=0.8, maximum=10, value=0.8, step=0.1)
                seed = gr.Number(label="随机种子", value=3, precision=0)
                btn = gr.Button("生成视频", variant="primary")
        with gr.Row():
            status = gr.Label(label="进度")
        with gr.Row():
            video = gr.Video(label="结果视频")
            audio = gr.Audio(label="提取音频(可选)")
        btn.click(_run, [prompt, start_image, steps, seed, width, height, dur],
                  [status, video])
        gr.Markdown("生成采用逐块流式(显存极省)，偏慢属正常。完成后可自行用 ffmpeg 提取音轨：`ffmpeg -i 结果.mp4 -vn 音频.m4a`。")
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=BACKEND)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    BACKEND = a.backend
    demo = build()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=a.host, server_port=a.port, share=a.share, inbrowser=False)
