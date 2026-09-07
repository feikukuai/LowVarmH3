#!/usr/bin/env python3
"""端到端出片验证脚本：向 h3-workbench WebUI 提交一个最小文生视频任务并轮询到出 mp4。

前提：WebUI 已用 launch-webui-linux.sh 启动在本机 7860，模型/导出/tokenizer 均已就绪，
     且系统装了 ffmpeg。

用法:
    python verify_and_generate.py                      # 用默认参数
    python verify_and_generate.py --prompt "..." --out out.mp4
"""
import argparse, json, time, urllib.request, sys

def http(method, url, data=None, timeout=120):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:7860")
    p.add_argument("--prompt", default="a red apple on a wooden table, soft light")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--duration", type=float, default=0.8)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="generated_video.mp4")
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    B = a.base

    # 1) 健康检查
    for _ in range(30):
        try:
            http("GET", B + "/api/health"); break
        except Exception:
            time.sleep(2)
    else:
        print("后端未就绪"); sys.exit(1)

    # 2) 提交任务（文生视频）
    payload = dict(prompt=a.prompt, steps=a.steps, seed=a.seed,
                   width=a.width, height=a.height, duration_seconds=a.duration,
                   temporal_mode="segmented", conditioning_mode="text")
    job = http("POST", B + "/api/jobs/inference", payload)
    jid = job["id"]
    print(f"job={jid}", flush=True)

    # 3) 轮询
    last = ""
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            st = http("GET", B + f"/api/jobs/{jid}")
        except Exception as e:
            print("后端进程异常退出:", e); sys.exit(1)
        msg = st.get("message") or st.get("status") or ""
        if msg != last:
            print(f"[{st.get('progress'):.3f}] {msg}", flush=True); last = msg
        s = st.get("status")
        if s == "completed":
            print("完成。服务器产物:", st["result"]["output"], flush=True)
            break
        if s in ("failed", "cancelled"):
            print("失败:", s, json.dumps(st)[:800]); sys.exit(1)
        time.sleep(5)
    else:
        print("超时"); sys.exit(1)

    # 4) 下载 mp4
    urllib.request.urlretrieve(B + f"/api/jobs/{jid}/output", a.out)
    print("已保存:", a.out)

if __name__ == "__main__":
    main()
