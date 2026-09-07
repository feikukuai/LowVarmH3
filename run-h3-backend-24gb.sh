#!/usr/bin/env bash
# ============================================================
# 在 24GB cgroup 内存上限内运行 Minimax-H3 ONNX 后端。
#
# 目的: 本机宿主实际可用内存约 25GB(实测 commit>22GB 会撞宿主 OOM)。
# 给后端单独建一个 memory.max=24GB 的子 cgroup, 让后端在这个保守预算内
# 流式运行, 不会顶到宿主实墙去杀别的进程/被宿主误杀 -> 更稳地跑通。
#
# 用法: bash run-h3-backend-24gb.sh [limit_gb]   (默认 24)
# ============================================================
set -e
cd "$(dirname "$0")"

LIMIT_GB="${1:-24}"
LIMIT_BYTES=$((LIMIT_GB * 1024 * 1024 * 1024))
CGROUP="/sys/fs/cgroup/h3backend"

# 1) 建 cgroup 并设内存上限(尽力而为, 非特权环境会失败则降级为直接启动)
if mkdir -p "$CGROUP" 2>/dev/null && [ -w "$CGROUP/memory.max" ]; then
    echo "$LIMIT_BYTES" > "$CGROUP/memory.max" 2>/dev/null || true
    # 软限低于硬限: 让内核更早回收页缓存, 降低峰值
    if [ -w "$CGROUP/memory.high" ]; then
        echo $((LIMIT_BYTES - 2 * 1024 * 1024 * 1024)) > "$CGROUP/memory.high" 2>/dev/null || true
    fi
    # 把当前 shell 移入该 cgroup, exec 后端后子进程全部继承
    if echo $$ > "$CGROUP/cgroup.procs" 2>/dev/null; then
        echo "[launch] 已启用 24GB cgroup 限制: $CGROUP  memory.max=$(cat $CGROUP/memory.max)"
    else
        echo "[launch] 警告: 无法移入 cgroup, 将无内存限制运行"
    fi
else
    echo "[launch] 警告: 无法创建 cgroup(非特权), 无内存硬限运行"
fi

# 2) 启动后端(继承上面的 cgroup/环境)
echo "[launch] 启动 Minimax-H3 ONNX 后端(上限 ${LIMIT_GB}GB)..."
exec ./launch-webui-linux.sh 127.0.0.1 7860
