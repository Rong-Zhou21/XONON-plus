#!/usr/bin/env bash
# ============================================================================
# Pillar-up 能力针对性验证
#
# 流程：
#   1. 在 *宿主机* 启动 monitor_server（如果还没起）：
#        nohup python3 monitor_server.py --port 8080 \
#            > /tmp/xenon_monitor.log 2>&1 &
#      然后浏览器打开：http://<host_ip>:8080/
#
#   2. 进容器：
#        docker exec -it xenon_plus_case bash
#        cd /app/repo
#
#   3. 跑本脚本（默认 stone benchmark task 0 / world_seed=10 / 10 分钟）：
#        bash scripts/run_pillar_verification.sh
#
#      或换种子：
#        WORLD_SEED=42 bash scripts/run_pillar_verification.sh
#
# 行为：
#   - 出生后 *不调用 STEVE-1*，直接用 raw_step 让智能体低头 + 持续 attack 挖
#     向下，自然采集 cobblestone / dirt。
#   - 当 hotbar 中 placeable 方块累计 >= 8 块时停止下挖。
#   - 调用 env.pillar_up(target_dy=20) 抬升智能体。
#   - 全程 wrapper 自动推 POV 帧到 monitor_server，浏览器实时观察。
#   - 落地后输出：dig_dy / pillar_dy / blocks_used / reason / success。
#   - 完整结果 JSON 落到 /tmp/pillar_verify_seed*_<时间戳>.json。
# ============================================================================

set -u

# ===== 可调参数 =====
BENCHMARK="${BENCHMARK:-stone}"
WORLD_SEED="${WORLD_SEED:-10}"
SEED="${SEED:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
GPU="${GPU:-0}"
EXP_NUM="${EXP_NUM:-70001}"
# instant: pillar at surface with preloaded 32 cobblestone (fast, mechanism-only test)
# natural: dig down with raw actions to harvest blocks first, then pillar
VERIFY_MODE="${VERIFY_MODE:-instant}"
export VERIFY_MODE
# ====================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/minerl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/app/LLM}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN_BACKEND="${QWEN_BACKEND:-vllm}"
export QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}"
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export XENON_DISABLE_STUCK_KILL="${XENON_DISABLE_STUCK_KILL:-1}"

# 关键：开启 monitor 推送
export MONITOR_URL="${MONITOR_URL:-http://172.17.0.1:8080/push}"
export MONITOR_FPS="${MONITOR_FPS:-15}"

LOG_FILE="/tmp/pillar_verify_${BENCHMARK}_seed${WORLD_SEED}_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
  pkill -9 -f "launchClient" 2>/dev/null || true
}

cleanup
sleep 3

# monitor server 探活（不阻塞执行）
HEALTH_URL="${MONITOR_URL%/push}/healthz"
if curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q "200"; then
  echo "[ok] monitor_server alive at ${MONITOR_URL%/push}"
else
  echo "[!!] monitor_server NOT reachable at $HEALTH_URL"
  echo "    在宿主机执行: nohup python3 monitor_server.py --port 8080 > /tmp/xenon_monitor.log 2>&1 &"
  echo "    浏览器打开:  http://<host>:8080/"
fi

cat <<INFO
==========================================
 Pillar-up verification
==========================================
 benchmark    : $BENCHMARK
 world_seed   : $WORLD_SEED
 seed         : $SEED
 server_port  : $SERVER_PORT
 GPU          : $GPU
 exp_num      : $EXP_NUM
 monitor_url  : $MONITOR_URL
 log          : $LOG_FILE
 start_time   : $(date)
==========================================
INFO

xvfb-run -a python -u scripts/verify_pillar_up.py \
  server.port="$SERVER_PORT" \
  env.times=1 \
  env.max_minutes=10 \
  benchmark="$BENCHMARK" \
  evaluate=[0] \
  prefix=ours_planning \
  exp_num="$EXP_NUM" \
  seed="$SEED" \
  world_seed="$WORLD_SEED" \
  2>&1 | tee "$LOG_FILE"

RC=${PIPESTATUS[0]}

cat <<INFO

==========================================
 end_time   : $(date)
 exit_code  : $RC
 log        : $LOG_FILE
==========================================
INFO

LATEST=$(ls -t /tmp/pillar_verify_seed*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo
  echo "=== latest pillar verification JSON: $LATEST ==="
  python3 -c "
import json
d = json.load(open('$LATEST'))
print(json.dumps(d, indent=2, default=str))
"
fi

cleanup
exit "$RC"
