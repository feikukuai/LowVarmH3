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
bash /path/to/LowVarmH3/patch-linux.sh          # 把 environments 改成含 'linux'
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
├── run-h3-backend-24gb.sh       # 24GB cgroup 内存上限后端启动(解决宿主OOM)
├── patch-linux.sh               # MinimaxH3-ONNX → Linux 平台支持(pyproject)
├── scripts/
│   └── verify_and_generate.py   # 端到端出片验证脚本(HTTP)
├── frontend/
│   ├── gradio_h3_simple.py      # 简单版 Gradio 前端(文生/首帧→视频, 7861, 已验证)
│   └── onnx_adapter.py          # ONNX 后端适配层(把前端调用翻译成 7860)
├── frontend_amd/
│   └── app_video_tab.py         # AMD 风格完整前端(3 Tab + block进度 + 内存上限设置)
└── docs/
    ├── MODELS_AND_PATHS.md      # 模型下载源 + 放置路径 + 导出产物清单
    └── RUNNING_NOTES.md         # 🆕 实测跑通记录(时间/分辨率/耗时, 供社区参考)
```
> 📄 想看**真实跑通耗时/分辨率**请直接看 [`docs/RUNNING_NOTES.md`](docs/RUNNING_NOTES.md)。

> 本仓库只包含「让 MinimaxH3-ONNX 在 Linux/低显存跑通」的补丁与脚本，**不包含模型文件**(几十 GB，请按 MODELS_AND_PATHS 下载)。

## 四、Gradio 前端

### A. AMD 风格完整前端（推荐，三 Tab，7862）
```bash
# 先启动后端(见上节 4)
cd <MinimaxH3-ONNX 目录>
.venv/bin/python gradio_frontend_amd/app_video_tab.py \
    --backend http://127.0.0.1:7860 --host 127.0.0.1 --port 7862
# 打开 http://127.0.0.1:7862
```
功能（页面/控件向 AMD MiniMax-H3 前端对齐）：
| Tab | 功能 |
|---|---|
| 🎬 视频生成 | prompt、多图参考、参考视频(≤5s)、多段参考音频、宽高比(8种)、分辨率档位(含⚡32×32超低)、时长、⚡快速步数4-8、🔑取码、凭码取回、最新视频+单独音频、`<Picture N>` 自动标签同步 |
| 🖼️ R2I 图片编辑 | 多图参考、六段式提示词、🔑取图码、图片 Gallery、QC 视频、单独音频、凭码取回（后端=生成视频→ffmpeg 抽帧当图） |
| 📋 任务 & 监控 | 系统监控(CPU/内存/GPU/后端)、进行中任务表、✕ 一键取消(个人使用无需输码)、任务日志 |

- **turbo LoRA**：需后端已导出 Ref2VA Turbo 4-step adapter（`/api/jobs/download-export` preset `ref2va_turbo_v0_1`，后端 `acceleration_ready=True`），前端自动开启 `use_acceleration_lora`。
- 取消为**前端软取消**（停止轮询+界面释放）；后端已提交那条可能继续跑完（后端无 cancel API）。

### B. 简单版前端（最小，7861，已验证出片）
```bash
cd <MinimaxH3-ONNX 目录>
.venv/bin/python gradio_frontend/gradio_h3_simple.py \
    --backend http://127.0.0.1:7860 --host 127.0.0.1 --port 7861
```
`onnx_adapter.py` 是把前端从 ComfyUI 换成 ONNX 的翻译层。

---

## 五、开机自动启动（仅限腾讯 Cloud Studio 等支持 `preview.yml` 的空间）

把下面追加到你的 `.vscode/preview.yml`（工作空间打开即自动拉起后端+前端，前端 `autoOpen:true` 默认打开）：

```yaml
  # ---- MiniMax-H3 ONNX 后端 (FastAPI, 7860) ----
  - port: 7860
    run: ./launch-webui-linux.sh 127.0.0.1 7860
    root: ./MinimaxH3-ONNX          # 按你实际 checkout 路径改
    name: H3-ONNX-Backend
    autoOpen: false
  # ---- MiniMax-H3 AMD 风格前端 (Gradio, 7862) ----
  - port: 7862
    run: /abs/path/to/MinimaxH3-ONNX/.venv/bin/python app_video_tab.py \
         --backend http://127.0.0.1:7860 --host 127.0.0.1 --port 7862
    root: ./MinimaxH3-ONNX/gradio_frontend_amd
    name: H3-Frontend
    autoOpen: true
```

> ⚠️ **注意**：`preview.yml` 是 **腾讯 Cloud Studio 的自动启动机制**，仅适用于这类空间。**其它环境（本地机、Colab、普通 Docker 等）不适用**，请按各自情况改用 systemd / supervisor / `nohup` / 启动脚本自行拉起这两个服务。路径（`./MinimaxH3-ONNX` 等）也需按你实际目录调整。
