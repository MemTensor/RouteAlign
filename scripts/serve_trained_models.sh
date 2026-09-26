#!/bin/bash
# 微调任务画像模型的 sglang 服务管理（DemoRouter3_train）
#
# 按 config/trained_models.yaml 注册表，为每个模型在指定 GPU 上启动一个
# sglang OpenAI 兼容服务（每 GPU 一个，互不干扰）。
#
# 用法（任意目录执行）：
#   serve_trained_models.sh start [模型名...]   # 启动（默认全部 6 个）
#   serve_trained_models.sh stop  [模型名...]   # 停止
#   serve_trained_models.sh status              # 健康检查
set -u

PY="${SGLANG_PYTHON:-python}"          # sglang 环境的 python，可用 SGLANG_PYTHON 覆盖
PY_YAML="${PYTHON:-python}"            # 仅需能 import yaml 的 python
export PYTHONNOUSERSITE=1  # 避免 ~/.local 下 transformers 遮蔽 sglang 环境依赖

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REG="$ROOT/config/trained_models.yaml"
LOG_DIR="$ROOT/logs/sglang"
mkdir -p "$LOG_DIR"

# 输出 "name path gpu port mem max_running" 每行一个模型
list_models() {
    "$PY_YAML" - "$REG" "$@" <<'EOF'
import sys, yaml
reg = yaml.safe_load(open(sys.argv[1]))
models = reg["models"]
serve = reg.get("serve") or {}
names = sys.argv[2:] or list(models)
for n in names:
    if n not in models:
        print(f"unknown model: {n}", file=sys.stderr)
        sys.exit(1)
    m = models[n]
    print(n, m["model_path"], m["gpu"], m["port"],
          serve.get("mem_fraction_static", 0.85),
          serve.get("max_running_requests", 256))
EOF
}

cmd="${1:-status}"
shift || true

case "$cmd" in
start)
    list_models "$@" | while read -r name path gpu port mem maxrun; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" -o /dev/null 2>&1; then
            echo "[skip] $name already serving on :$port"
            continue
        fi
        echo "[start] $name on GPU $gpu, port $port (log: $LOG_DIR/$name.log)"
        CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -m sglang.launch_server \
            --model-path "$path" \
            --served-model-name "$name" \
            --host 127.0.0.1 --port "$port" \
            --mem-fraction-static "$mem" \
            --max-running-requests "$maxrun" \
            > "$LOG_DIR/$name.log" 2>&1 &
        echo $! > "$LOG_DIR/$name.pid"
    done
    echo "等待服务就绪：curl http://127.0.0.1:<port>/v1/models 或用 status 子命令查看"
    ;;
stop)
    list_models "$@" | while read -r name path gpu port mem maxrun; do
        pidfile="$LOG_DIR/$name.pid"
        if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            kill "$(cat "$pidfile")" && echo "[stop] $name (pid $(cat "$pidfile"))"
            rm -f "$pidfile"
        else
            # 兜底：按端口匹配 sglang 进程
            pids=$(pgrep -f "sglang.launch_server.*--port $port" || true)
            if [ -n "$pids" ]; then
                kill $pids && echo "[stop] $name (pids: $pids)"
            else
                echo "[skip] $name not running"
            fi
        fi
    done
    ;;
status)
    list_models | while read -r name path gpu port mem maxrun; do
        if curl -sf "http://127.0.0.1:${port}/v1/models" -o /dev/null 2>&1; then
            echo "[up]   $name  :$port"
        else
            echo "[down] $name  :$port"
        fi
    done
    ;;
*)
    echo "用法: $0 [start|stop|status] [模型名...]" >&2
    exit 1
    ;;
esac
