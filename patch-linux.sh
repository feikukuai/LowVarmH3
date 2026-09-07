#!/usr/bin/env bash
# 让 MinimaxH3-ONNX 支持 Linux：修改 pyproject.toml 的 [tool.uv] environments
# 在 MinimaxH3-ONNX 目录里执行:  bash /path/to/LowVarmH3/patch-linux.sh
set -e
cd "$(dirname "$0")"
PP="pyproject.toml"
if [ ! -f "$PP" ]; then
  echo "未找到 pyproject.toml，请在 MinimaxH3-ONNX 目录运行" >&2
  exit 1
fi
cp "$PP" "$PP.bak"
python3 - <<'PY'
import io
p="pyproject.toml"
t=open(p).read()
if "sys_platform == 'linux'" in t:
    print("已包含 linux，无需修改")
else:
    t=t.replace("environments = [\"sys_platform == 'win32'\"]",
                "environments = [\"sys_platform == 'win32'\", \"sys_platform == 'linux'\"]")
    open(p,"w").write(t)
    print("pyproject.toml 已加入 linux 平台支持")
PY
