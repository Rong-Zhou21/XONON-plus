#!/usr/bin/env bash
# ============================================================================
# XENON-plus 单任务运行脚本（视频/结果落到 v3 目录）
#
# 从全新终端到跑起来的完整流程：
#   1. ssh / 进入有 docker 的宿主机
#   2. 进入容器：
#        docker exec -it xenon_plus_case bash
#   3. 进入仓库：
#        cd /app/repo
#   4. (按需) 修改下面 USER PARAMS 块；或者直接在命令行覆盖：
#        bash scripts/run_task_v3.sh
#      或：
#        BENCHMARK=iron TASK_ID=2 MAX_MINUTES=20 bash scripts/run_task_v3.sh
#
# 输出位置（自动创建）：
#   videos/v3/<Category_Task>/<biome>/<status>/*.mp4
#   exp_results/v3/ours_planning_<task>_<exp_num>_*.json
#   /tmp/xenon_v3_<...>.log   （本次完整 stdout / stderr）
#
# 注意：本轮决策器仅完成离线训练（artifacts/decisioner/rads_v2.pt），
# 尚未接入 runtime。本脚本跑的是当前 retrieval-only 流程，结果可作为
# 接入决策器后的 v3 baseline。
# ============================================================================

set -u

# ===== USER PARAMS（按需修改） =====================================
BENCHMARK="${BENCHMARK:-iron}"       # wood / stone / iron / gold / diamond / redstone / armor
TASK_ID="${TASK_ID:-10}"             # 任务编号，参考 src/optimus1/conf/benchmark/*.yaml
WORLD_SEED="${WORLD_SEED:-10}"       # Minecraft world seed（建议与 TASK_ID 一致以便复现）
SEED="${SEED:-0}"                    # 算法 seed
MAX_MINUTES="${MAX_MINUTES:-10}"     # 单任务最长分钟数
EXP_NUM="${EXP_NUM:-99000}"          # 结果文件编号（避免重名）
SERVER_PORT="${SERVER_PORT:-9100}"   # STEVE-1 action server 端口
GPU="${GPU:-0}"                      # CUDA_VISIBLE_DEVICES

# 决策器开关：1=启用 RADS 决策器，0=baseline retrieval-only
DECISIONER_ENABLED="${DECISIONER_ENABLED:-0}"
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_v2.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"
# ====================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

# ===== 运行时环境（通常无需改，可用外部 env var 覆盖） =====
export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/minerl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/app/LLM}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN_BACKEND="${QWEN_BACKEND:-vllm}"
export QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}"
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export XENON_DISABLE_STUCK_KILL="${XENON_DISABLE_STUCK_KILL:-1}"

VIDEO_DIR="videos/v3"
RESULTS_DIR="exp_results/v3"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

LOG_FILE="/tmp/xenon_v3_${BENCHMARK}_t${TASK_ID}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
}

# 跑前清理上一次可能残留的 MineRL/Malmo/Xvfb 进程
cleanup
sleep 3

if [ "$DECISIONER_ENABLED" = "1" ]; then
  DECISIONER_LABEL="ENABLED  (ckpt=$DECISIONER_CKPT, min_p=$DECISIONER_MIN_P)"
else
  DECISIONER_LABEL="disabled (baseline retrieval-only)"
fi

cat <<INFO
==========================================
 XENON-plus v3 single-task run
==========================================
 benchmark    : $BENCHMARK
 task_id      : $TASK_ID
 world_seed   : $WORLD_SEED
 seed         : $SEED
 max_minutes  : $MAX_MINUTES
 exp_num      : $EXP_NUM
 server_port  : $SERVER_PORT
 GPU          : $GPU
 video_dir    : $VIDEO_DIR
 results_dir  : $RESULTS_DIR
 log          : $LOG_FILE
 decisioner   : $DECISIONER_LABEL
 start_time   : $(date)
==========================================
INFO

if [ "$DECISIONER_ENABLED" = "1" ]; then
  DECISIONER_OVERRIDES=(
    "memory.case_memory.decisioner.enabled=true"
    "memory.case_memory.decisioner.checkpoint=$DECISIONER_CKPT"
    "memory.case_memory.decisioner.min_p_success=$DECISIONER_MIN_P"
  )
else
  DECISIONER_OVERRIDES=("memory.case_memory.decisioner.enabled=false")
fi

xvfb-run -a python -u src/optimus1/main_planning.py \
  server.port="$SERVER_PORT" \
  env.times=1 \
  env.max_minutes="$MAX_MINUTES" \
  benchmark="$BENCHMARK" \
  evaluate="[$TASK_ID]" \
  prefix=ours_planning \
  exp_num="$EXP_NUM" \
  seed="$SEED" \
  world_seed="$WORLD_SEED" \
  record.video.path="$VIDEO_DIR" \
  results.path="$RESULTS_DIR" \
  "${DECISIONER_OVERRIDES[@]}" \
  2>&1 | tee "$LOG_FILE"

RC=${PIPESTATUS[0]}

echo
cat <<INFO
==========================================
 end_time   : $(date)
 exit_code  : $RC
INFO

RESULT_FILE=$(ls -t "$RESULTS_DIR"/*_"$EXP_NUM"_*.json 2>/dev/null | head -1)
if [ -n "$RESULT_FILE" ]; then
  echo " result_file: $RESULT_FILE"
  python3 - "$RESULT_FILE" <<'PY'
import json
import sys
p = sys.argv[1]
try:
    d = json.load(open(p))
    print(f" task          : {d.get('task')}")
    print(f" success       : {d.get('success')}")
    print(f" status_detail : {d.get('status_detailed')}")
    print(f" steps         : {d.get('steps')}")
    print(f" minutes       : {d.get('minutes')}")
    print(f" video_file    : {d.get('video_file')}")
    print(f" failed_wp     : {d.get('failed_waypoints')}")
except Exception as e:
    print(f" parse error: {e}", file=sys.stderr)
PY
else
  echo " no result JSON produced (task crashed early; see log)"
fi

cat <<INFO
==========================================
INFO

# 跑后再清一次，避免影响下一任务
cleanup

exit "$RC"
