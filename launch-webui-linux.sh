#!/usr/bin/env bash
# ============================================================
# MiniMax H3 Edge Workbench — Linux/T4 (低显存) 跑通启动脚本
# 在 NVIDIA T4 16GB / 30GB RAM 上已验证端到端出片(mp4,含音频)
#
# 三个关键修复(缺一不可):
#  1) LD_LIBRARY_PATH 加入 venv 内 nvidia CUDA 库(cuDNN/cuBLAS/...)
#     -> 让 ORT CUDA EP 与 torch 解析到一致 cuDNN/cuBLAS
#  2) H3_DEVICE_RESIDENT_HIDDEN=0
#     -> 绕开 ORT CUDA EP 在 T4 上"设备常驻 hidden-state(CUDA OrtValue)"
#        的原生段错误(Fatal Python error: Segmentation fault)
#  3) H3_WEIGHT_PREFETCH_DEPTH=1 / H3_WEIGHT_PREFETCH_WORKERS=1
#     -> 压低 host RAM 峰值,避免流式读取 40GB bf16 时被 cgroup OOM killer 杀掉
# 另: 需系统安装 ffmpeg(最后 mp4 封装用), 缺则报 [Errno 2] ffmpeg
#
# 用法: ./launch-webui-linux.sh [host] [port]
# ============================================================
set -e
cd "$(dirname "$0")"

# 若有 pyenv python 3.11.1(本项目要求)则加入 PATH
if [ -d /root/.pyenv/versions/3.11.1/bin ]; then
    export PATH="/root/.pyenv/versions/3.11.1/bin:$PATH"
fi
HOST="${1:-127.0.0.1}"
PORT="${2:-7860}"

# 修复1: venv 内 nvidia CUDA 组件 lib 目录
NVLIB=".venv/lib/python3.11/site-packages/nvidia"
for d in cudnn cublas cufft curand cusolver cusparse nccl; do
    if [ -d "$PWD/$NVLIB/$d/lib" ]; then
        export LD_LIBRARY_PATH="$PWD/$NVLIB/$d/lib:${LD_LIBRARY_PATH:-}"
    fi
done
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda-12.2/lib64"

# 修复2: 绕开 T4 上设备常驻 hidden-state 的段错误
export H3_DEVICE_RESIDENT_HIDDEN=0
# 修复3: 压低 host RAM, 避免流式读取大模型时 OOM
export H3_WEIGHT_PREFETCH_DEPTH=1
export H3_WEIGHT_PREFETCH_WORKERS=1

# 确保 ffmpeg 可用(最后 mp4 封装)
export PATH="/usr/bin:${PATH}"

mkdir -p .h3-workbench onnx_models exported qwen_tokenizer
echo "[launch] LD_LIBRARY_PATH set with nvidia CUDA libs"
echo "[launch] H3_DEVICE_RESIDENT_HIDDEN=0 (绕过段错误)"
echo "[launch] H3_WEIGHT_PREFETCH_DEPTH=1 (压低RAM防OOM)"
echo "[launch] starting h3-workbench on $HOST:$PORT (workspace: $(pwd))"
exec .venv/bin/python -m h3_workbench --host "$HOST" --port "$PORT" --workspace "$(pwd)"
