#!/usr/bin/env bash
# ============================================================================
# XENON-plus iron+gold benchmark (with pillar-up upgrade flow)
#
# 仿照 run_v3_full_benchmark.sh 的结构，只跑 iron + gold 两个 benchmark：
#     iron    : 0..15  (16) max_minutes=10
#     golden  : 0..5   (6)  max_minutes=30
#     total   : 22 任务
#
# 默认开启本轮新加的"环境感知 + 抬升"特性：
#     XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT=1
#       触发条件：挖矿子目标进行中，已经收集/看见更高级的矿物，但目标矿物
#       数量未达 -> wrapper.raise_to_height(目标矿带中位 Y) 抬回正确层 ->
#       STEVE-1 prompt 切到 "dig forward and mine X" 横向挖。
#
# 视频/结果输出固定到 v4 目录：
#     videos/v4/<Category_Task>/<biome>/<status>/*.mp4
#     exp_results/v4/ours_planning_<task>_<exp_num>_*.json
#
# ----------------------------------------------------------------------------
# 启动示例（在 docker 容器内执行；ours_planning 共享 case_memory）：
#
#   # 1) 默认：开 pillar-up 特性，不开决策器（pure baseline + 抬升）
#   bash scripts/run_iron_gold_pillar.sh
#
#   # 2) 把决策器一起开（RADS 检索 + 抬升）
#   DECISIONER_ENABLED=1 bash scripts/run_iron_gold_pillar.sh
#
#   # 3) 关掉抬升做对照实验
#   PILLAR_UP_ENABLED=0 bash scripts/run_iron_gold_pillar.sh
#
#   # 4) 只跑 iron / 只跑 golden
#   ONLY_BENCH=iron   bash scripts/run_iron_gold_pillar.sh
#   ONLY_BENCH=golden bash scripts/run_iron_gold_pillar.sh
#
#   # 5) 只跑某些 task ID (空格分隔)
#   ONLY_BENCH=iron ONLY_TASKS="1 5 7 8 14" bash scripts/run_iron_gold_pillar.sh
#
# 断点续跑：
#   * 默认 SKIP_DONE=1，已写入 exp_results/v4/ 中同 exp_num 的结果会跳过。
#   * Ctrl-C 后再次执行同一命令即可从下一个未完成任务继续。
#
# 总耗时 (worst case)：
#   16*10 + 6*30 = 160 + 180 = 340 分钟 ≈ 5.7 小时
#   实际通常更短（任务在 max_minutes 之前可能因 success/replan-loop 提前结束）。
# ============================================================================

set -u

# ===== USER PARAMS（按需 export 后再启动） ===========================

# 抬升特性开关（默认 ON——这是这一轮新加的功能）
PILLAR_UP_ENABLED="${PILLAR_UP_ENABLED:-1}"

# 决策器开关（默认 OFF）
DECISIONER_ENABLED="${DECISIONER_ENABLED:-0}"
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_v2.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"

# exp_num 编号区段：iron 0..15 -> 80200..80215, gold 0..5 -> 80300..80305
# 决策器开启时偏到 81000 段，避免和无决策器对比时撞名
if [ "$DECISIONER_ENABLED" = "1" ]; then
  EXP_NUM_BASE="${EXP_NUM_BASE:-81000}"
  RUN_LABEL_DEFAULT="iron_gold_pillar_decisioner"
else
  EXP_NUM_BASE="${EXP_NUM_BASE:-80000}"
  RUN_LABEL_DEFAULT="iron_gold_pillar_baseline"
fi
RUN_LABEL="${RUN_LABEL:-$RUN_LABEL_DEFAULT}"

# 限定只跑某个 benchmark / 某些 task
ONLY_BENCH="${ONLY_BENCH:-}"          # "" / "iron" / "golden"
ONLY_TASKS="${ONLY_TASKS:-}"          # 空字符 = 不限制；否则 "1 5 7 8 14" 这样

GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
SEED="${SEED:-0}"
TASK_COOLDOWN_SEC="${TASK_COOLDOWN_SEC:-10}"
SKIP_DONE="${SKIP_DONE:-1}"

# Pillar-up 细调（默认与 main_planning.py 内部默认值一致）
XENON_OVERSHOOT_RELEVEL_MIN_DY="${XENON_OVERSHOOT_RELEVEL_MIN_DY:-2}"
XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS="${XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS:-64}"
XENON_OVERSHOOT_RELEVEL_MAX_STEPS="${XENON_OVERSHOOT_RELEVEL_MAX_STEPS:-600}"

# Monitor 推送（开浏览器实时观看 agent）
MONITOR_URL="${MONITOR_URL:-http://172.17.0.1:8080/push}"
MONITOR_FPS="${MONITOR_FPS:-15}"
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

# 抬升相关 env
export XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT="$PILLAR_UP_ENABLED"
export XENON_OVERSHOOT_RELEVEL_MIN_DY
export XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS
export XENON_OVERSHOOT_RELEVEL_MAX_STEPS

# Monitor 推送
export MONITOR_URL
export MONITOR_FPS

VIDEO_DIR="videos/v4"
RESULTS_DIR="exp_results/v4"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

SUMMARY_FILE="${SUMMARY_FILE:-/tmp/xenon_iron_gold_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_summary.log}"

# ===== 任务表 ==========
# (benchmark, task_id, max_minutes, exp_offset_base)
JOBS=(
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
 XENON-plus iron+gold benchmark (with pillar-up upgrade flow)
============================================================
 run_label    : $RUN_LABEL
 pillar_up    : $([ "$PILLAR_UP_ENABLED" = "1" ] && echo "ENABLED" || echo "DISABLED")
 decisioner   : $([ "$DECISIONER_ENABLED" = "1" ] && echo "ENABLED  ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P" || echo "disabled")
 only_bench   : ${ONLY_BENCH:-<all>}
 only_tasks   : ${ONLY_TASKS:-<all>}
 exp_num_base : $EXP_NUM_BASE
 GPU          : $GPU
 server_port  : $SERVER_PORT
 video_dir    : $VIDEO_DIR
 results_dir  : $RESULTS_DIR
 summary      : $SUMMARY_FILE
 monitor_url  : $MONITOR_URL
 skip_done    : $SKIP_DONE
 start_time   : $(date)
============================================================
INFO

cleanup
sleep 3

DONE=0
SKIPPED=0
SUCCESS=0
FAIL=0
NORESULT=0

# 在过滤后再算 TOTAL_TASKS
FILTERED_JOBS=()
for JOB in "${JOBS[@]}"; do
  read -r BENCH TID MAXMIN OFF <<< "$JOB"
  # benchmark 过滤
  if [ -n "$ONLY_BENCH" ] && [ "$BENCH" != "$ONLY_BENCH" ]; then
    continue
  fi
  # task id 过滤
  if [ -n "$ONLY_TASKS" ]; then
    found=0
    for t in $ONLY_TASKS; do
      if [ "$TID" = "$t" ]; then found=1; break; fi
    done
    if [ "$found" = "0" ]; then continue; fi
  fi
  FILTERED_JOBS+=("$JOB")
done
TOTAL_TASKS="${#FILTERED_JOBS[@]}"

if [ "$TOTAL_TASKS" -eq 0 ]; then
  echo "[!!] no jobs after filtering (ONLY_BENCH=$ONLY_BENCH, ONLY_TASKS=$ONLY_TASKS)" | tee -a "$SUMMARY_FILE"
  exit 1
fi

for JOB in "${FILTERED_JOBS[@]}"; do
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

  LOG_FILE="/tmp/xenon_iron_gold_${RUN_LABEL}_${BENCH}_t${TID}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"
  T_START=$(date +%s)

  printf "[%2d/%d] %-9s task=%-2s exp=%-6s max_min=%-2s start=%s log=%s\n" \
    "$DONE" "$TOTAL_TASKS" "$BENCH" "$TID" "$EXP_NUM" "$MAXMIN" \
    "$(date +%H:%M:%S)" "$LOG_FILE" | tee -a "$SUMMARY_FILE"

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

  T_ELAPSED=$(( $(date +%s) - T_START ))

  RESULT_FILE=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
  if [ -n "$RESULT_FILE" ]; then
    STATUS=$(python3 - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
recovery = d.get("recovery_events") or {}
# Count pillar-up triggers
pup = 0
for k in ("pillar_up", "pillar_up_smart", "raise_to_height", "raise_to_ore_band"):
    v = recovery.get(k)
    if isinstance(v, list):
        pup += len(v)
    elif isinstance(v, int):
        pup += v
print(
    f"{'SUCCESS' if d.get('success') else 'FAIL'} "
    f"status={d.get('status_detailed')} steps={d.get('steps')} "
    f"minutes={d.get('minutes')} pillar_up_calls={pup}"
)
PY
)
    if echo "$STATUS" | grep -q "^SUCCESS"; then
      SUCCESS=$((SUCCESS + 1))
    else
      FAIL=$((FAIL + 1))
    fi
  else
    STATUS="NO_RESULT (rc=$RC)"
    NORESULT=$((NORESULT + 1))
  fi

  printf "       result=%s elapsed=%ds\n" "$STATUS" "$T_ELAPSED" | tee -a "$SUMMARY_FILE"

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

# 汇总各 benchmark 成功数 (本批 exp_num 区段内)
INFO

python3 - "$RESULTS_DIR" "$EXP_NUM_BASE" <<'PY' 2>/dev/null | tee -a "$SUMMARY_FILE"
import collections, glob, json, os, sys

results_dir = sys.argv[1]
exp_base = int(sys.argv[2])
by_bench = collections.defaultdict(
    lambda: {"total": 0, "success": 0, "pillar_up_calls": 0, "pillar_up_active_runs": 0}
)
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
    rec = d.get("recovery_events") or {}
    pup = 0
    for k in ("pillar_up", "pillar_up_smart", "raise_to_height", "raise_to_ore_band"):
        v = rec.get(k)
        if isinstance(v, list):
            pup += len(v)
        elif isinstance(v, int):
            pup += v
    if pup > 0:
        by_bench[b]["pillar_up_active_runs"] += 1
    by_bench[b]["pillar_up_calls"] += pup

print("Per-benchmark within this batch:")
total_t, total_s = 0, 0
for b in sorted(by_bench):
    s = by_bench[b]
    print(
        f"  {b:<10}  success {s['success']:>2}/{s['total']:<2}  "
        f"pillar_up_runs={s['pillar_up_active_runs']:>2}  "
        f"pillar_up_calls={s['pillar_up_calls']:>3}"
    )
    total_t += s["total"]
    total_s += s["success"]
print(f"  {'OVERALL':<10}  success {total_s}/{total_t}")
PY
