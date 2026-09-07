#!/usr/bin/env python3
"""ONNX 后端适配层 —— 把 AMD 前端(ComfyUI耦合)的请求翻译成 MinimaxH3-ONNX WebUI 调用。

目标: 在不改 AMD 前端 UI 的前提下, 后端从 ComfyUI 换成我们跑通的 ONNX WebUI(7860)。
AMD 前端原本调 ComfyUI 的地方(实例管理/工作流JSON/prompt/submit/get results)
在此收敛为 5 个能力:
  1) text_to_video(prompt, width, height, seconds, steps, seed)
  2) image_to_video(prompt, start_img_file, width, height, seconds, steps, seed)   # 首帧条件
  3) ref_to_video(prompt, images[], video, audios[], ...)  # 多参考(ref2va), 尽力映射
  4) extract_frame(video_mp4, time_s) -> png   # R2I = 从生成视频取帧(用户说明"编图=视频截图")
  5) speed_route: use_acceleration_lora (turbo 4步)  由 ONNX adapter 支持
"""
import os, json, time, subprocess, urllib.request, urllib.error, urllib.parse

# 指向已跑通的 ONNX WebUI
ONNX_BASE = os.environ.get("H3_ONNX_BASE", "http://127.0.0.1:7860")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

def _http_json(method, path, payload=None, timeout=120, raw=None):
    url = ONNX_BASE + path
    if raw is not None:
        req = urllib.request.Request(url, data=raw, method=method)
        return req
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            raise RuntimeError(f"HTTP {e.code}: {json.loads(e.read())}")
        except Exception:
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:400]}")

def upload_image(path):
    with open(path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(ONNX_BASE + "/api/images/upload", data=data, method="POST")
    req.add_header("x-filename", urllib.parse.quote(os.path.basename(path)))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["path"]

def _duration_to_frames_ignored(seconds):  # ONNX 后端内部处理 duration
    return None

def submit(prompt, start_image_path, steps, seed, width, height, duration_seconds,
           use_acceleration_lora=False, references=None, conditioning="text"):
    payload = dict(prompt=prompt, steps=int(steps), seed=int(seed),
                   width=int(width), height=int(height),
                   duration_seconds=float(duration_seconds),
                   temporal_mode="segmented",
                   conditioning_mode=conditioning,
                   start_image_path=start_image_path,
                   use_acceleration_lora=bool(use_acceleration_lora))
    if references:
        payload["references"] = references
    return _http_json("POST", "/api/jobs/inference", payload)["id"]

def poll(job_id):
    """返回 (done, status_dict)。"""
    st = _http_json("GET", f"/api/jobs/{job_id}")
    s = st.get("status")
    if s in ("completed", "failed", "cancelled"):
        return True, st
    return False, st

def get_video_path(job_id):
    st = _http_json("GET", f"/api/jobs/{job_id}")
    if st.get("status") == "completed":
        return st["result"]["output"]
    return None

def extract_frame(mp4, out_png, at_s=0.0):
    """从视频抽一帧当图片(R2I 语义=视频截图)。"""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-ss", f"{at_s:.2f}", "-i", mp4,
                    "-frames:v", "1", out_png], check=True, capture_output=True)
    return out_png

def extract_audio(mp4, out_m4a):
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-vn", "-ac", "2", out_m4a],
                   check=True, capture_output=True)
    return out_m4a

def extract_all_frames(mp4, out_dir, fps=None):
    """从视频抽出全部(或按 fps)帧为 PNG(R2I 语义=生成视频截图)。返回帧文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%04d.png")
    cmd = [FFMPEG, "-y", "-i", mp4]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [pattern]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("frame_"))

def download_output(job_id, dst):
    url = ONNX_BASE + f"/api/jobs/{job_id}/output"
    urllib.request.urlretrieve(url, dst)
    return dst

def onnx_profile_generation_ready():
    try:
        d = _http_json("GET", "/api/profiles")
        p = d[0] if isinstance(d, list) and d else d
        return bool(p.get("generation_ready"))
    except Exception:
        return False

def onnx_health():
    try:
        return _http_json("GET", "/api/health") or True
    except Exception:
        return False
