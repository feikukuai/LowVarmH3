# LowVarmH3 — MiniMax-H3 低显存跑通方案（Linux / NVIDIA T4 实测）

在 **NVIDIA T4 16GB + 30GB RAM（Linux）** 上，把 [MinimaxH3-ONNX](https://github.com/MARK42IRPC/MinimaxH3-ONNX)（MiniMax-H3 Edge Workbench）**端到端跑通并出片**（MP4，含音频）。

> 实测效果：文生视频 128×128 全程 **显存峰值约 3GB、宿主 anon 约 4.5~9GB**，稳定出片不 OOM、不段错误。
> 说明：推理是**逐块流式**（加载→算完→卸载→下一块），所以 GPU/内存占用低，代价是偏慢。

---

## 一、为什么需要这个仓库

原版 MinimaxH3-ONNX 面向 **Windows / RTX 3050+**。在 **Linux + T4** 上直接跑会踩三个坑，全部修好才能出片：

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 1 | ORT 找不到 cuDNN / 与 torch 混用 | CUDA EP 加载失败或原生崩溃 | `LD_LIBRARY_PATH` 指向 venv 内 `nvidia/{cudnn,cublas,...}/lib` |
| 2 | 设备常驻 hidden-state 段错误 | `Fatal Python error: Segmentation fault`（跑 ref2va 图时） | `H3_DEVICE_RESIDENT_HIDDEN=0`（hidden 走 host） |
| 3 | 流式读 40GB 模型 OOM | 进程被 cgroup OOM killer 静默杀掉 | `H3_WEIGHT_PREFETCH_DEPTH=1`、`H3_WEIGHT_PREFETCH_WORKERS=1` |
| + | 缺 ffmpeg | 最后 `[Errno 2] ffmpeg`，无法封装 mp4 | `apt-get install -y ffmpeg` |

**三个 env 修复 + LD_LIBRARY_PATH 已固化在 [`launch-webui-linux.sh`](launch-webui-linux.sh)。**

---

## 二、快速开始

### 1) 准备代码与 Python 环境
```bash
# 需要 Python 3.11 + uv
git clone --depth 1 https://github.com/MARK42IRPC/MinimaxH3-ONNX.git h3
cd h3
# 应用 Linux 支持补丁(pyproject.toml 原版 [tool.uv].environments 仅允许 win32)
git apply ../LowVarmH3/pyproject.linux.patch   # 或手动把 environments 改成含 'linux'
uv lock --python 3.11
uv sync --locked --extra dev --no-editable --python 3.11
```

### 2) 准备模型与 tokenizer
见 [docs/MODELS_AND_PATHS.md](docs/MODELS_AND_PATHS.md)（含下载源、放置路径、导出命令）。

关键点：`minimax_h3_ref2va_pruned_bf16.safetensors`（40GB）放工作区根目录；VAE / Qwen int8 可复用已有或下载后放到指定路径；tokenizer 放 `qwen_tokenizer/`。

### 3) 安装 ffmpeg
```bash
apt-get update && apt-get install -y ffmpeg
```

### 4) 启动（关键：用本仓库的启动脚本）
```bash
cd h3
cp /path/to/LowVarmH3/launch-webui-linux.sh .   # 覆盖为修复版
./launch-webui-linux.sh 127.0.0.1 7860
```
启动后打开 http://127.0.0.1:7860

### 5) 验证出片
```bash
python scripts/verify_and_generate.py            # 提交一个最小任务并轮询到出 mp4
```
或调用任意推理 HTTP 接口（schema 见脚本内注释）。

---

## 三、目录
```
LowVarmH3/
├── launch-webui-linux.sh        # 关键修复版 Linux 启动脚本(3 个修复固化)
├── pyproject.linux.patch        # MinimaxH3-ONNX → Linux 支持补丁
├── scripts/
│   └── verify_and_generate.py   # 端到端出片验证脚本(HTTP)
└── docs/
    └── MODELS_AND_PATHS.md      # 模型下载源 + 放置路径 + 导出产物清单
```

> 本仓库只包含「让 MinimaxH3-ONNX 在 Linux/低显存跑通」的补丁与脚本，**不包含模型文件**(几十 GB，请按 MODELS_AND_PATHS 下载)。
