# 模型下载源 + 放置路径 + 导出产物清单

> 本文档记录跑通 MinimaxH3-ONNX 所需的**全部模型/依赖文件**：下载源（HuggingFace 优先，ModelScope 回落）、应放置的**工作区路径**、以及 ONNX 导出产物。
> 模型不入 git（几十 GB），按此文档放到工作区后即可被 exporter/WebUI 识别。

工作区路径统一记为 `$WS`（即 MinimaxH3-ONNX 的 checkout 目录，WebUI 的 `--workspace`）。

---

## 1) 源模型 safetensors（放 `$WS/` 工作区根目录，导出用）

| 文件（放到 `$WS/`） | 大小 | HuggingFace 下载 | ModelScope 下载 |
|---|---|---|---|
| `minimax_h3_ref2va_pruned_bf16.safetensors` | 40GB | `https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` | `https://www.modelscope.cn/api/v1/models/Comfy-Org/MiniMax-H3/repo?Revision=master&FilePath=diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` |
| `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27GB | `.../text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 同上路径 `text_encoders/...` |
| `minimax_h3_video_vae_fp16.safetensors` | 5.2GB | `.../vae/minimax_h3_video_vae_fp16.safetensors` | 同上路径 `vae/...` |
| `minimax_h3_audio_vae_fp32.safetensors` | 605MB | `.../vae/minimax_h3_audio_vae_fp32.safetensors` | 同上路径 `vae/...` |

> HuggingFace 仓库 `Comfy-Org/MiniMax-H3` 与 ModelScope 同名仓库的文件**字节数一致**，可互换。下载命令（支持断点续传）：
> ```bash
> curl -sSL -C - -o "$WS/minimax_h3_ref2va_pruned_bf16.safetensors" \
>   "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors"
> ```

### 本机已实测的存放方式
- **复用已存在**：若 `video_vae`、`audio_vae`、`qwen int8` 已存在于别处（如 ComfyUI 的 `models/vae`、`models/text_encoders`），可**符号链接**到 `$WS/`（WebUI 按文件名在 `$WS` 找源文件，并校验字节数）：
  ```bash
  ln -s /path/to/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors "$WS/"
  ```
- **必须真实下载**：`ref2va_pruned_bf16`（40GB）无其它来源，需下载到 `$WS/`。

---

## 2) H3 tokenizer（放 `$WS/qwen_tokenizer/`，仅 ModelScope 有）

源仓库 `Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS`，4 个文件，放 `$WS/qwen_tokenizer/`：
```
qwen_tokenizer/{tokenizer.json, tokenizer_config.json, vocab.json, merges.txt}
```
下载示例：
```bash
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  curl -sSL -C - -o "$WS/qwen_tokenizer/$f" \
    "https://www.modelscope.cn/api/v1/models/Mark42IRPC/Minimax-H3-int8-fl2va-onnx-50CLIPS/repo?Revision=master&FilePath=qwen_tokenizer/$f"
done
```

> 注意：WebUI 就绪条件 `generation_ready` 需要 tokenizer 在 `qwen_tokenizer/` 内且非空。

---

## 3) ONNX 导出产物（放 `$WS/onnx_models/`，由 exporter 生成）

源文件就位后，用 exporter 生成 ONNX（分片/虚拟切片，不复制大权重）：
```bash
cd $WS
PY=.venv/bin/python
$PY -m h3_workbench.exporter export minimax_h3_audio_vae_fp32.safetensors        --output onnx_models/audio_vae
$PY -m h3_workbench.exporter export qwen3vl_32b_minimax_h3_int8_convrot.safetensors --output onnx_models/qwen3vl_32b_minimax_h3_int8_virtual
$PY -m h3_workbench.exporter export minimax_h3_video_vae_fp16.safetensors        --output onnx_models/video_vae
$PY -m h3_workbench.exporter export minimax_h3_ref2va_pruned_bf16.safetensors    --output onnx_models/minimax_h3_ref2va_pruned_bf16_virtual
```
导出后会写 `manifest.json`，字段含 `validation_passed: true` 才算成功。本机实测产物大小：
```
onnx_models/audio_vae/                                    580MB   (audio_encoder/decoder.onnx)
onnx_models/qwen3vl_32b_minimax_h3_int8_virtual/          3.1MB   (共享拓扑小图)
onnx_models/video_vae/                                    5.0GB   (36 块 video_decoder_block 分片)
onnx_models/minimax_h3_ref2va_pruned_bf16_virtual/        3.2GB   (316 张 main_block 小图)
```

---

## 4) 输出目录

- 出片 mp4 默认写到 `$WS/.h3-workbench/outputs/`（例：`h3-xxxxxxxxxxxx.mp4`，h264+acc）。

---

## 5) 运行时依赖/补丁（不放模型，见仓库根）
- Python 3.11 + `uv`；安装 MinimaxH3-ONNX 依赖需先应用 [`../patch-linux.sh`](../patch-linux.sh)
- 系统 `ffmpeg`（`apt-get install -y ffmpeg`）
- 用 `../launch-webui-linux.sh` 启动（内置 3 个关键 env 修复）
