#!/usr/bin/env bash
# ============================================================================
# XENON-plus V5 — Armor-only verification run (13 tasks)
#
# 用途：在三处 bug 修复（pillar-up logger format / _topo_sort KeyError /
#      respawn-on-lava is_alive 检测 / bedrock-stuck 触发 pillar-up）之后，
#      用 Armor 的 13 个任务做端到端验证。
#
# 视频/结果默认写入 v5 目录（避免覆盖 v4 历史数据）：
#     videos/v5/<Category_Task>/<biome>/<status>/*.mp4
#     exp_results/v5/ours_planning_<task>_<exp_num>_*.json
#
# exp_num_base 默认 33000（armor 段为 33600..33612）。
# ============================================================================

set -u

# ===== USER PARAMS =====================================================
DECISIONER_ENABLED="${DECISIONER_ENABLED:-1}"      # V5 默认启用
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_v2.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"

if [ "$DECISIONER_ENABLED" = "1" ]; then
  EXP_NUM_BASE="${EXP_NUM_BASE:-33000}"
  RUN_LABEL="${RUN_LABEL:-v5_decisioner}"
else
  EXP_NUM_BASE="${EXP_NUM_BASE:-33500}"
  RUN_LABEL="${RUN_LABEL:-v5_baseline}"
fi

GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
SEED="${SEED:-0}"
TASK_COOLDOWN_SEC="${TASK_COOLDOWN_SEC:-3}"
SKIP_DONE="${SKIP_DONE:-1}"

PERCEPTION_ACTION_SUITE="${PERCEPTION_ACTION_SUITE:-1}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

# ===== 运行时环境 =====
export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/minerl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/app/LLM}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN_BACKEND="${QWEN_BACKEND:-vllm}"
export QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}"
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export XENON_DISABLE_STUCK_KILL="${XENON_DISABLE_STUCK_KILL:-1}"
export MONITOR_URL="${MONITOR_URL:-}"
export XENON_PERCEPTION_ACTION_SUITE="$PERCEPTION_ACTION_SUITE"

VIDEO_DIR="${VIDEO_DIR:-videos/v5}"
RESULTS_DIR="${RESULTS_DIR:-exp_results/v5}"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

SUMMARY_FILE="${SUMMARY_FILE:-/tmp/xenon_v5_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_summary.log}"

# Armor 13 tasks. exp_offset_base 600 -> exp_num 33600..33612
JOBS=(
  "armor   0  30 600"
  "armor   1  30 600"
  "armor   2  30 600"
  "armor   3  30 600"
  "armor   4  30 600"
  "armor   5  30 600"
  "armor   6  30 600"
  "armor   7  30 600"
  "armor   8  30 600"
  "armor   9  30 600"
  "armor   10 30 600"
  "armor   11 30 600"
  "armor   12 30 600"
)

cleanup() {
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
  pkill -9 -f "launchClient" 2>/dev/null || true
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

cat <<INFO | tee "$SUMMARY_FILE"
============================================================
 XENON-plus V5 — Armor verification run (13 tasks)
============================================================
 run_label              : $RUN_LABEL
 decisioner             : $([ "$DECISIONER_ENABLED" = "1" ] && echo "ENABLED  ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P" || echo "disabled (baseline)")
 perception_action_suite: $([ "$PERCEPTION_ACTION_SUITE" = "1" ] && echo "ON" || echo "OFF")
 exp_num_base           : $EXP_NUM_BASE
 GPU                    : $GPU
 server_port            : $SERVER_PORT
 video_dir              : $VIDEO_DIR
 results_dir            : $RESULTS_DIR
 summary                : $SUMMARY_FILE
 skip_done              : $SKIP_DONE
 start_time             : $(date)
============================================================
 Bug fixes verified in this run:
   1) pillar-up logger format-string TypeError (was crashing every armor task)
   2) relative_graph _topo_sort KeyError on raw-material leaves
   3) lava-respawn missed by health-only detector (now uses is_alive transition)
   4) bedrock-stuck pillar-up trigger (escapes when y<8 and no activity)
============================================================
INFO

cleanup
sleep 3

DONE=0
SKIPPED=0
SUCCESS=0
FAIL=0
NORESULT=0
TOTAL_TASKS="${#JOBS[@]}"

for JOB in "${JOBS[@]}"; do
  read -r BENCH TID MAXMIN OFF <<< "$JOB"
  EXP_NUM=$((EXP_NUM_BASE + OFF + TID))
  WORLD_SEED="$TID"

  DONE=$((DONE + 1))

  EXISTING=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
  if [ "$SKIP_DONE" = "1" ] && [ -n "$EXISTING" ]; then
    SKIPPED=$((SKIPPED + 1))
    printf "[%2d/%d] %-9s task=%-2s exp=%-6s  SKIP (already exists: %s)\n" \
      "$DONE" "$TOTAL_TASKS" "$BENCH" "$TID" "$EXP_NUM" \
      "$(basename "$EXISTING")" | tee -a "$SUMMARY_FILE"
    continue
  fi

  MAX_RETRIES_ON_CRASH="${MAX_RETRIES_ON_CRASH:-2}"
  CRASH_RETRY_STEP_THRESHOLD="${CRASH_RETRY_STEP_THRESHOLD:-1000}"
  CRASH_RETRY_COOLDOWN_SEC="${CRASH_RETRY_COOLDOWN_SEC:-15}"

  attempt=0
  STATUS=""
  T_START=$(date +%s)

  while : ; do
    if [ "$attempt" -gt 0 ]; then
      printf "       retry %d/%d (env crash detected, last status=%s)\n" \
        "$attempt" "$MAX_RETRIES_ON_CRASH" "$STATUS" | tee -a "$SUMMARY_FILE"
      cleanup
      sleep "$CRASH_RETRY_COOLDOWN_SEC"
    fi

    LOG_FILE="/tmp/xenon_v5_${RUN_LABEL}_${BENCH}_t${TID}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"

    if [ "$attempt" -eq 0 ]; then
      printf "[%2d/%d] %-9s task=%-2s exp=%-6s max_min=%-2s start=%s log=%s\n" \
        "$DONE" "$TOTAL_TASKS" "$BENCH" "$TID" "$EXP_NUM" "$MAXMIN" \
        "$(date +%H:%M:%S)" "$LOG_FILE" | tee -a "$SUMMARY_FILE"
    else
      printf "       retry log=%s\n" "$LOG_FILE" | tee -a "$SUMMARY_FILE"
    fi

    xvfb-run -a python -u src/optimus1/main_planning.py \
      server.port="$SERVER_PORT" \
      env.times=1 \
      env.max_minutes="$MAXMIN" \
      benchmark="$BENCH" \
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
      STATUS=$(python3 - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{'SUCCESS' if d.get('success') else 'FAIL'} status={d.get('status_detailed')} steps={d.get('steps')} minutes={d.get('minutes')}")
PY
)
    else
      STATUS="NO_RESULT (rc=$RC)"
    fi

    should_retry=0
    if [ "$attempt" -lt "$MAX_RETRIES_ON_CRASH" ]; then
      if [ -z "$RESULT_FILE" ]; then
        should_retry=1
      else
        STATUS_DETAILED=$(python3 - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("status_detailed", ""))
PY
)
        STATUS_STEPS=$(python3 - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(int(d.get("steps", 0) or 0))
PY
)
        IS_SUCCESS=$(echo "$STATUS" | grep -c "^SUCCESS")
        if [ "$IS_SUCCESS" -eq 0 ]; then
          if [ "$STATUS_DETAILED" = "env_step_timeout" ] && [ "$STATUS_STEPS" -lt "$CRASH_RETRY_STEP_THRESHOLD" ]; then
            should_retry=1
          elif echo "$STATUS_DETAILED" | grep -qE "^crash_" && [ "$STATUS_STEPS" -lt "$CRASH_RETRY_STEP_THRESHOLD" ]; then
            should_retry=1
          fi
        fi
      fi
    fi

    if [ "$should_retry" -eq 1 ]; then
      if [ -n "$RESULT_FILE" ]; then
        RETRY_DIR="$RESULTS_DIR/retry_failures"
        mkdir -p "$RETRY_DIR"
        RETRY_KEEP="$RETRY_DIR/retry${attempt}_$(basename "$RESULT_FILE")"
        mv "$RESULT_FILE" "$RETRY_KEEP"
        printf "       preserved retry failure result=%s\n" "$RETRY_KEEP" | tee -a "$SUMMARY_FILE"
      fi
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

cat <<INFO | tee -a "$SUMMARY_FILE"

============================================================
 V5 Armor verification — end summary
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
