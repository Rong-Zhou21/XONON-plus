#!/usr/bin/env bash
# ============================================================================
# XENON-plus V7 — targeted Armor run after pillar-up lateral-offset change
#
# Runs 10 trials each, with gold armor first:
#   Armor task 12 Craft golden chestplate
#   Armor task 10 Craft golden leggings
#   Armor task 6  Craft diamond chestplate
#
# Results:
#   exp_results/v7/ours_planning_<task>_<exp_num>_*.json
#   videos/v7/<Category_Task>/<biome>/<status>/*.mp4
# ============================================================================

set -u

PYTHON_BIN="${PYTHON_BIN:-/home/yzb/.conda/envs/vllm_qwen2_5_vl/bin/python}"
DECISIONER_ENABLED="${DECISIONER_ENABLED:-1}"
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_v2.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"

EXP_NUM_BASE="${EXP_NUM_BASE:-370000}"
RUN_LABEL="${RUN_LABEL:-v7_pillar_lateral}"
GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
SEED_BASE="${SEED_BASE:-0}"
TRIALS="${TRIALS:-10}"
TASK_COOLDOWN_SEC="${TASK_COOLDOWN_SEC:-3}"
SKIP_DONE="${SKIP_DONE:-1}"
MAX_RETRIES_ON_CRASH="${MAX_RETRIES_ON_CRASH:-3}"
CRASH_RETRY_COOLDOWN_SEC="${CRASH_RETRY_COOLDOWN_SEC:-15}"
PERCEPTION_ACTION_SUITE="${PERCEPTION_ACTION_SUITE:-1}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/minerl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/yzb/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN_BACKEND="${QWEN_BACKEND:-vllm}"
export QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}"
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export XENON_DISABLE_STUCK_KILL="${XENON_DISABLE_STUCK_KILL:-1}"
export MONITOR_URL="${MONITOR_URL:-}"
export XENON_PERCEPTION_ACTION_SUITE="$PERCEPTION_ACTION_SUITE"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,172.17.0.1}"
export no_proxy="${no_proxy:-$NO_PROXY}"

# V7 behavior under test: after overshoot relevel, move while attacking until
# x/z changes, then continue the original dig-down prompt at the new position.
export XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT="${XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT:-1}"
export XENON_OVERSHOOT_ENABLE_Y_TRIGGER="${XENON_OVERSHOOT_ENABLE_Y_TRIGGER:-0}"
export XENON_OVERSHOOT_LATERAL_BLOCKS="${XENON_OVERSHOOT_LATERAL_BLOCKS:-1}"
export XENON_OVERSHOOT_LATERAL_MAX_STEPS="${XENON_OVERSHOOT_LATERAL_MAX_STEPS:-240}"
export XENON_OVERSHOOT_LATERAL_RETRIES="${XENON_OVERSHOOT_LATERAL_RETRIES:-3}"
export XENON_CORRIDOR_FEET_PITCH="${XENON_CORRIDOR_FEET_PITCH:-12.0}"
export XENON_CORRIDOR_AXIS_LOCK="${XENON_CORRIDOR_AXIS_LOCK:-1}"
export XENON_CORRIDOR_YAW_MODE="${XENON_CORRIDOR_YAW_MODE:-hold}"
export XENON_CORRIDOR_BLOCKED_FRONT_PITCH="${XENON_CORRIDOR_BLOCKED_FRONT_PITCH:-0.0}"
export XENON_CORRIDOR_UP_PITCH="${XENON_CORRIDOR_UP_PITCH:--65.0}"
export XENON_CORRIDOR_BLOCKED_FEET_PITCH="${XENON_CORRIDOR_BLOCKED_FEET_PITCH:-55.0}"
export XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH="${XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH:-65.0}"
export XENON_CORRIDOR_MIN_MOVE_DELTA="${XENON_CORRIDOR_MIN_MOVE_DELTA:-0.20}"
export XENON_CORRIDOR_UNSTUCK_PASSES="${XENON_CORRIDOR_UNSTUCK_PASSES:-2}"
export XENON_CORRIDOR_MOVE_ATTACK_TICKS="${XENON_CORRIDOR_MOVE_ATTACK_TICKS:-8}"
export XENON_CORRIDOR_STOP_AFTER_MOVE="${XENON_CORRIDOR_STOP_AFTER_MOVE:-1}"
export XENON_CORRIDOR_BLOCKED_HEAD_BUDGET="${XENON_CORRIDOR_BLOCKED_HEAD_BUDGET:-24}"
export XENON_CORRIDOR_BLOCKED_UP_BUDGET="${XENON_CORRIDOR_BLOCKED_UP_BUDGET:-10}"
export XENON_CORRIDOR_BLOCKED_FEET_BUDGET="${XENON_CORRIDOR_BLOCKED_FEET_BUDGET:-36}"
export XENON_CORRIDOR_ATTACK_FORWARD="${XENON_CORRIDOR_ATTACK_FORWARD:-1}"
export XENON_LATERAL_MAX_Y_DROP="${XENON_LATERAL_MAX_Y_DROP:-0.75}"

VIDEO_DIR="${VIDEO_DIR:-videos/v7}"
RESULTS_DIR="${RESULTS_DIR:-exp_results/v7}"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

SUMMARY_FILE="${SUMMARY_FILE:-/tmp/xenon_v7_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_summary.log}"

TASK_IDS=(12 10 6)
declare -A TASK_NAMES=(
  [6]="diamond_chestplate"
  [10]="golden_leggings"
  [12]="golden_chestplate"
)

cleanup() {
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
  pkill -9 -f "launchClient" 2>/dev/null || true
}

is_abnormal_status() {
  case "$1" in
    env_step_timeout|crash_*) return 0 ;;
    *) return 1 ;;
  esac
}

is_timeout_runtime_result() {
  local result_file="$1"
  [ -n "$result_file" ] && [ -f "$result_file" ] || return 1
  "$PYTHON_BIN" - "$result_file" <<'PY' 2>/dev/null
import json, sys

try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)

payload = json.dumps(data, ensure_ascii=False, default=str)
if data.get("status_detailed") == "crash_RuntimeError" and "Timeout" in payload:
    sys.exit(0)
sys.exit(1)
PY
}

delete_abnormal_artifacts() {
  local result_file="$1"
  local video_file=""
  if [ -n "$result_file" ] && [ -f "$result_file" ]; then
    video_file=$("$PYTHON_BIN" - "$result_file" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
print(d.get("video_file") or "")
PY
)
    rm -f "$result_file"
  fi
  if [ -n "$video_file" ] && [ -f "$video_file" ]; then
    rm -f "$video_file"
  fi
}

if [ "$DECISIONER_ENABLED" = "1" ]; then
  DECISIONER_OVERRIDES=(
    "memory.case_memory.decisioner.enabled=true"
    "memory.case_memory.decisioner.checkpoint=$DECISIONER_CKPT"
    "memory.case_memory.decisioner.min_p_success=$DECISIONER_MIN_P"
  )
else
  DECISIONER_OVERRIDES=("memory.case_memory.decisioner.enabled=false")
fi

TOTAL_TASKS=$(( ${#TASK_IDS[@]} * TRIALS ))
DONE=0
SKIPPED=0
SUCCESS=0
FAIL=0
NORESULT=0

cat <<INFO | tee "$SUMMARY_FILE"
============================================================
 XENON-plus V7 — targeted Armor run
============================================================
 run_label              : $RUN_LABEL
 python                 : $PYTHON_BIN
 decisioner             : $([ "$DECISIONER_ENABLED" = "1" ] && echo "ENABLED  ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P" || echo "disabled (baseline)")
 perception_action_suite: $([ "$PERCEPTION_ACTION_SUITE" = "1" ] && echo "ON" || echo "OFF")
 exp_num_base           : $EXP_NUM_BASE
 trials_per_task        : $TRIALS
 armor_task_ids         : ${TASK_IDS[*]}
 overshoot_y_trigger    : $XENON_OVERSHOOT_ENABLE_Y_TRIGGER
 pillar_lateral_blocks  : $XENON_OVERSHOOT_LATERAL_BLOCKS
 lateral_retries        : $XENON_OVERSHOOT_LATERAL_RETRIES
 lateral_body_pitch     : $XENON_CORRIDOR_FEET_PITCH
 axis_lock              : $XENON_CORRIDOR_AXIS_LOCK
 yaw_mode               : $XENON_CORRIDOR_YAW_MODE
 blocked_front_pitch    : $XENON_CORRIDOR_BLOCKED_FRONT_PITCH
 blocked_up_pitch       : $XENON_CORRIDOR_UP_PITCH
 blocked_up_budget      : $XENON_CORRIDOR_BLOCKED_UP_BUDGET
 blocked_feet_pitch     : $XENON_CORRIDOR_BLOCKED_FEET_PITCH
 blocked_feet_max_pitch : $XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH
 min_forward_delta      : $XENON_CORRIDOR_MIN_MOVE_DELTA
 unstuck_passes         : $XENON_CORRIDOR_UNSTUCK_PASSES
 move_attack_ticks      : $XENON_CORRIDOR_MOVE_ATTACK_TICKS
 stop_after_move        : $XENON_CORRIDOR_STOP_AFTER_MOVE
 lateral_attack_forward : $XENON_CORRIDOR_ATTACK_FORWARD
 lateral_max_y_drop     : $XENON_LATERAL_MAX_Y_DROP
 GPU                    : $GPU
 server_port            : $SERVER_PORT
 video_dir              : $VIDEO_DIR
 results_dir            : $RESULTS_DIR
 summary                : $SUMMARY_FILE
 max_abnormal_retries   : $MAX_RETRIES_ON_CRASH
 skip_done              : $SKIP_DONE
 start_time             : $(date)
============================================================
INFO

cleanup
sleep 3

for TID in "${TASK_IDS[@]}"; do
  for ((REP=0; REP<TRIALS; REP++)); do
    EXP_NUM=$((EXP_NUM_BASE + TID * 100 + REP))
    SEED=$((SEED_BASE + REP))
    WORLD_SEED=$((SEED_BASE + REP))
    MAXMIN=30
    DONE=$((DONE + 1))

    EXISTING=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
    if [ "$SKIP_DONE" = "1" ] && [ -n "$EXISTING" ]; then
      SKIPPED=$((SKIPPED + 1))
      printf "[%2d/%d] armor task=%-2s %-20s rep=%-2s exp=%-6s SKIP (%s)\n" \
        "$DONE" "$TOTAL_TASKS" "$TID" "${TASK_NAMES[$TID]}" "$REP" "$EXP_NUM" \
        "$(basename "$EXISTING")" | tee -a "$SUMMARY_FILE"
      continue
    fi

    attempt=0
    STATUS=""
    T_START=$(date +%s)

    while : ; do
      if [ "$attempt" -gt 0 ]; then
        printf "       retry %d/%d (abnormal exit, last status=%s)\n" \
          "$attempt" "$MAX_RETRIES_ON_CRASH" "$STATUS" | tee -a "$SUMMARY_FILE"
        cleanup
        sleep "$CRASH_RETRY_COOLDOWN_SEC"
      fi

      LOG_FILE="/tmp/xenon_v7_${RUN_LABEL}_armor_t${TID}_rep${REP}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"

      if [ "$attempt" -eq 0 ]; then
        printf "[%2d/%d] armor task=%-2s %-20s rep=%-2s exp=%-6s seed=%-2s start=%s log=%s\n" \
          "$DONE" "$TOTAL_TASKS" "$TID" "${TASK_NAMES[$TID]}" "$REP" "$EXP_NUM" "$SEED" \
          "$(date +%H:%M:%S)" "$LOG_FILE" | tee -a "$SUMMARY_FILE"
      else
        printf "       retry log=%s\n" "$LOG_FILE" | tee -a "$SUMMARY_FILE"
      fi

      xvfb-run -a "$PYTHON_BIN" -u src/optimus1/main_planning.py \
        server.port="$SERVER_PORT" \
        env.times=1 \
        env.max_minutes="$MAXMIN" \
        benchmark=armor \
        evaluate="[$TID]" \
        prefix=ours_planning \
        exp_num="$EXP_NUM" \
        seed="$SEED" \
        world_seed="$WORLD_SEED" \
        record.video.path="$VIDEO_DIR" \
        results.path="$RESULTS_DIR" \
        "${DECISIONER_OVERRIDES[@]}" \
        > "$LOG_FILE" 2>&1
      RC=$?

      RESULT_FILE=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
      if [ -n "$RESULT_FILE" ]; then
        STATUS=$("$PYTHON_BIN" - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{'SUCCESS' if d.get('success') else 'FAIL'} status={d.get('status_detailed')} steps={d.get('steps')} minutes={d.get('minutes')}")
PY
)
      else
        STATUS="NO_RESULT (rc=$RC)"
      fi

      should_retry=0
      abnormal_exit=0
      if [ -z "$RESULT_FILE" ]; then
        abnormal_exit=1
      else
        STATUS_DETAILED=$("$PYTHON_BIN" - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("status_detailed", ""))
PY
)
        IS_SUCCESS=$(echo "$STATUS" | grep -c "^SUCCESS")
        if [ "$IS_SUCCESS" -eq 0 ] && is_abnormal_status "$STATUS_DETAILED" && ! is_timeout_runtime_result "$RESULT_FILE"; then
          abnormal_exit=1
        fi
      fi

      if [ "$abnormal_exit" -eq 1 ]; then
        delete_abnormal_artifacts "$RESULT_FILE"
        RESULT_FILE=""
        if [ "$attempt" -lt "$MAX_RETRIES_ON_CRASH" ]; then
          should_retry=1
        else
          STATUS="NO_RESULT_ABNORMAL_RETRIES_EXHAUSTED last=$STATUS"
        fi
      fi

      if [ "$should_retry" -eq 1 ]; then
        attempt=$((attempt + 1))
        continue
      fi
      break
    done

    T_ELAPSED=$(( $(date +%s) - T_START ))

    if [ -n "$RESULT_FILE" ]; then
      if echo "$STATUS" | grep -q "^SUCCESS"; then
        SUCCESS=$((SUCCESS + 1))
      else
        FAIL=$((FAIL + 1))
      fi
    else
      NORESULT=$((NORESULT + 1))
    fi

    if [ "$attempt" -gt 0 ]; then
      printf "       result=%s elapsed=%ds (after %d retries)\n" "$STATUS" "$T_ELAPSED" "$attempt" | tee -a "$SUMMARY_FILE"
    else
      printf "       result=%s elapsed=%ds\n" "$STATUS" "$T_ELAPSED" | tee -a "$SUMMARY_FILE"
    fi

    cleanup
    sleep "$TASK_COOLDOWN_SEC"
  done
done

cat <<INFO | tee -a "$SUMMARY_FILE"

============================================================
 V7 targeted Armor — end summary
 end_time   : $(date)
 total      : $TOTAL_TASKS
 success    : $SUCCESS
 fail       : $FAIL
 no_result  : $NORESULT
 skipped    : $SKIPPED
 results    : $RESULTS_DIR
 videos     : $VIDEO_DIR
============================================================
INFO
