#!/usr/bin/env bash
# ============================================================================
# XENON-plus 全 67 任务批跑脚本（输出到 v4，启用感知-动作 suite）
#
# 任务分布（与 v2 一致）：
#     wooden  : 0..9    (10) max_minutes=3
#     stone   : 0..8    (9)  max_minutes=6
#     iron    : 0..15   (16) max_minutes=10
#     golden  : 0..5    (6)  max_minutes=30
#     diamond : 0..6    (7)  max_minutes=30
#     redstone: 0..5    (6)  max_minutes=30
#     armor   : 0..12   (13) max_minutes=30
#     total   : 67
#
# 视频/结果默认写入 v4 目录（可被 VIDEO_DIR / RESULTS_DIR 环境变量覆盖）：
#     videos/v4/<Category_Task>/<biome>/<status>/*.mp4
#     exp_results/v4/ours_planning_<task>_<exp_num>_*.json
#
# 默认启用本轮新加的 "环境感知 + 行动" 统一开关
#     PERCEPTION_ACTION_SUITE=1
# 它在 main_planning 启动时把以下 9 项 feature 一起切到推荐默认：
#     underground:   pillar-up overshoot relevel + 3-block tunnel
#                    pre-mining height check (per-feature 默认 OFF)
#     surface:       low-air escape, surface search, tree-explore prompt
#     inventory:     hotbar-pressure cleanup, dropped-resource pickup
#     stagnation:    movement-stagnant escape, tunnel-stagnant recovery
# 想关任意单项做消融：直接 export 对应 XENON_ENABLE_* 即可（suite 不覆盖）。
#
# 启动：
#     # 1) 默认：suite=ON，baseline planner
#     bash scripts/run_v3_full_benchmark.sh
#
#     # 2) 决策器（RADS）+ suite=ON
#     DECISIONER_ENABLED=1 bash scripts/run_v3_full_benchmark.sh
#
#     # 3) 完整基线对照：suite=OFF（重现"加 perception-action 之前"行为）
#     PERCEPTION_ACTION_SUITE=0 bash scripts/run_v3_full_benchmark.sh
#
#     # 4) 单项消融：suite=ON 但关 pillar-up
#     XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT=0 bash scripts/run_v3_full_benchmark.sh
#
# 断点续跑：脚本会跳过 exp_results/v4/ 中已存在的同 exp_num 结果文件。
# 中途 Ctrl-C 后再次运行即可从下一个未完成任务继续。
#
# 总耗时（worst case）：
#     10*3 + 9*6 + 16*10 + 32*30 = 30 + 54 + 160 + 960 = 1204 分钟 ≈ 20 小时
# 实际通常比 max_minutes 短。
# ============================================================================

set -u

# ===== USER PARAMS（按需修改） =====================================
DECISIONER_ENABLED="${DECISIONER_ENABLED:-0}"   # 0=baseline, 1=RADS
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_v2.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"

# exp_num 编号起点：baseline=31xxx，决策器=32xxx，避免对比时撞名
if [ "$DECISIONER_ENABLED" = "1" ]; then
  EXP_NUM_BASE="${EXP_NUM_BASE:-32000}"
  RUN_LABEL="${RUN_LABEL:-decisioner}"
else
  EXP_NUM_BASE="${EXP_NUM_BASE:-31000}"
  RUN_LABEL="${RUN_LABEL:-baseline}"
fi

GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
SEED="${SEED:-0}"
TASK_COOLDOWN_SEC="${TASK_COOLDOWN_SEC:-3}"    # 每任务结束后清进程的等待秒数（10→3，节省 67×7s≈8min）
SKIP_DONE="${SKIP_DONE:-1}"                    # 1=跳过 exp_results/v4 已有的 exp_num

# ===== Perception-Action Suite（这一轮新增的"环境感知 + 行动"统一开关）=====
# 1 = 全量启用（推荐，作为本轮创新点的实验默认）
# 0 = 全部回退到旧基线（用于消融对照）
# 单 feature 想反向覆盖时，直接 export 对应 XENON_ENABLE_* 即可，suite 不会改写。
PERCEPTION_ACTION_SUITE="${PERCEPTION_ACTION_SUITE:-1}"
# ====================================================================

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
# Propagate the suite toggle into the python process. main_planning.main()
# calls PerceptionActionSuite.apply_from_env() right after env registration,
# which cascades this into every per-feature env var the wrapper / planner
# reads. User-set per-feature env vars (XENON_ENABLE_*) are not overwritten,
# so single-feature ablations still work.
export XENON_PERCEPTION_ACTION_SUITE="$PERCEPTION_ACTION_SUITE"

VIDEO_DIR="${VIDEO_DIR:-videos/v4}"
RESULTS_DIR="${RESULTS_DIR:-exp_results/v4}"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

SUMMARY_FILE="${SUMMARY_FILE:-/tmp/xenon_v3_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_summary.log}"

# 67 个任务的 (benchmark, task_id, max_minutes, exp_offset_base)
# exp_offset_base 用于让每个 benchmark 的 exp_num 段不冲突
JOBS=(
  # bench task max_min offset_base
  "wooden  0  3  0"
  "wooden  1  3  0"
  "wooden  2  3  0"
  "wooden  3  3  0"
  "wooden  4  3  0"
  "wooden  5  3  0"
  "wooden  6  3  0"
  "wooden  7  3  0"
  "wooden  8  3  0"
  "wooden  9  3  0"
  "stone   0  6  100"
  "stone   1  6  100"
  "stone   2  6  100"
  "stone   3  6  100"
  "stone   4  6  100"
  "stone   5  6  100"
  "stone   6  6  100"
  "stone   7  6  100"
  "stone   8  6  100"
  "iron    0  10 200"
  "iron    1  10 200"
  "iron    2  10 200"
  "iron    3  10 200"
  "iron    4  10 200"
  "iron    5  10 200"
  "iron    6  10 200"
  "iron    7  10 200"
  "iron    8  10 200"
  "iron    9  10 200"
  "iron    10 10 200"
  "iron    11 10 200"
  "iron    12 10 200"
  "iron    13 10 200"
  "iron    14 10 200"
  "iron    15 10 200"
  "golden  0  30 300"
  "golden  1  30 300"
  "golden  2  30 300"
  "golden  3  30 300"
  "golden  4  30 300"
  "golden  5  30 300"
  "diamond 0  30 400"
  "diamond 1  30 400"
  "diamond 2  30 400"
  "diamond 3  30 400"
  "diamond 4  30 400"
  "diamond 5  30 400"
  "diamond 6  30 400"
  "redstone 0 30 500"
  "redstone 1 30 500"
  "redstone 2 30 500"
  "redstone 3 30 500"
  "redstone 4 30 500"
  "redstone 5 30 500"
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
 XENON-plus full benchmark (67 tasks, v4 outputs)
============================================================
 run_label              : $RUN_LABEL
 decisioner             : $([ "$DECISIONER_ENABLED" = "1" ] && echo "ENABLED  ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P" || echo "disabled (baseline)")
 perception_action_suite: $([ "$PERCEPTION_ACTION_SUITE" = "1" ] && echo "ON  (suite-managed perception+action features active)" || echo "OFF (baseline; suite-managed features off)")
 exp_num_base           : $EXP_NUM_BASE
 GPU                    : $GPU
 server_port            : $SERVER_PORT
 video_dir              : $VIDEO_DIR
 results_dir            : $RESULTS_DIR
 summary                : $SUMMARY_FILE
 skip_done              : $SKIP_DONE
 start_time             : $(date)
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

  # 已有结果则跳过
  EXISTING=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
  if [ "$SKIP_DONE" = "1" ] && [ -n "$EXISTING" ]; then
    SKIPPED=$((SKIPPED + 1))
    printf "[%2d/%d] %-9s task=%-2s exp=%-6s  SKIP (already exists: %s)\n" \
      "$DONE" "$TOTAL_TASKS" "$BENCH" "$TID" "$EXP_NUM" \
      "$(basename "$EXISTING")" | tee -a "$SUMMARY_FILE"
    continue
  fi

  # 任务级重试机制 ---------------------------------------------
  # MineRL/Malmo 偶尔在 Java 客户端启动后 ~8 秒内超时（"TimeoutError:
  # timed out" / "Connection timed out!"），导致 status_detailed=
  # env_step_timeout 且 steps 极少（~100-200）。这是环境 race，重试
  # 一次通常就能成功。同样 crash_TypeError / crash_RuntimeError 在前
  # 1000 步内的早期崩溃也常因 Java 启动状态没跑稳。
  #
  # 重试条件（任一即可）：
  #   * status_detailed == env_step_timeout 且 steps < CRASH_RETRY_STEP_THRESHOLD
  #   * status_detailed 以 "crash_" 开头 且 steps < CRASH_RETRY_STEP_THRESHOLD
  #
  # 上限由 MAX_RETRIES_ON_CRASH 控制（默认 2），关闭重试设为 0。
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

    LOG_FILE="/tmp/xenon_v3_${RUN_LABEL}_${BENCH}_t${TID}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"

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

    # 是否需要重试？
    should_retry=0
    if [ "$attempt" -lt "$MAX_RETRIES_ON_CRASH" ]; then
      if [ -z "$RESULT_FILE" ]; then
        # 完全没结果文件 — 进程崩溃太早，连 results.json 都没写出来
        should_retry=1
      else
        # 有结果文件，看 status_detailed + steps
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
        # 已经成功 → 永远不重试
        IS_SUCCESS=$(echo "$STATUS" | grep -c "^SUCCESS")
        if [ "$IS_SUCCESS" -eq 0 ]; then
          # env_step_timeout + 早期 → 重试
          if [ "$STATUS_DETAILED" = "env_step_timeout" ] && [ "$STATUS_STEPS" -lt "$CRASH_RETRY_STEP_THRESHOLD" ]; then
            should_retry=1
          # crash_*Error + 早期 → 重试
          elif echo "$STATUS_DETAILED" | grep -qE "^crash_" && [ "$STATUS_STEPS" -lt "$CRASH_RETRY_STEP_THRESHOLD" ]; then
            should_retry=1
          fi
        fi
      fi
    fi

    if [ "$should_retry" -eq 1 ]; then
      # 删除当次失败的结果，避免下次尝试时被 SKIP 逻辑误判已完成
      [ -n "$RESULT_FILE" ] && rm -f "$RESULT_FILE"
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
 end_time   : $(date)
 run_label  : $RUN_LABEL
 total      : $TOTAL_TASKS
 success    : $SUCCESS
 fail       : $FAIL
 no_result  : $NORESULT
 skipped    : $SKIPPED
 results    : $RESULTS_DIR
 videos     : $VIDEO_DIR
 summary    : $SUMMARY_FILE
============================================================

# 汇总各 benchmark 成功数
INFO

python3 - "$RESULTS_DIR" "$EXP_NUM_BASE" <<'PY' 2>/dev/null | tee -a "$SUMMARY_FILE"
import collections
import glob
import json
import os
import sys

results_dir = sys.argv[1]
exp_base = int(sys.argv[2])

# Only count files in this batch's exp_num range [base, base+1000)
by_bench = collections.defaultdict(lambda: {"total": 0, "success": 0})
for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if "summary" in os.path.basename(p):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    e = int(d.get("exp_num", -1))
    if e < exp_base or e >= exp_base + 1000:
        continue
    b = d.get("benchmark", "?")
    by_bench[b]["total"] += 1
    if d.get("success"):
        by_bench[b]["success"] += 1

print("Per-benchmark within this batch:")
total_t, total_s = 0, 0
for b in sorted(by_bench):
    s = by_bench[b]
    print(f"  {b:<10}  success {s['success']:>2}/{s['total']:>2}")
    total_t += s["total"]
    total_s += s["success"]
print(f"  {'OVERALL':<10}  success {total_s}/{total_t}")
PY
