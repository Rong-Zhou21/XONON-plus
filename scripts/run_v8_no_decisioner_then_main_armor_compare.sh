#!/usr/bin/env bash
# Run the method-comparison batch requested for V8:
#   1. XENON-plus V8 with RADS decisioner disabled.
#   2. XENON-main under the same task/seed/resource setup.
#
# Completion is based on parseable JSON result files, not attempt count.

set -u
set -o pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/yzb/.conda/envs/vllm_qwen2_5_vl/bin/python}"
PLUS_REPO="${PLUS_REPO:-/home/yzb/zhourong/XENON-plus}"
MAIN_REPO="${MAIN_REPO:-/home/yzb/zhourong/XENON-main}"

TRIALS="${TRIALS:-10}"
SEED_BASE="${SEED_BASE:-0}"
MAX_VALIDATION_ROUNDS="${MAX_VALIDATION_ROUNDS:-10}"

PLUS_EXP_NUM_BASE="${PLUS_EXP_NUM_BASE:-880000}"
MAIN_EXP_NUM_BASE="${MAIN_EXP_NUM_BASE:-890000}"
PLUS_SERVER_PORT="${PLUS_SERVER_PORT:-9100}"
MAIN_SERVER_PORT="${MAIN_SERVER_PORT:-9101}"
QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PLUS_GPU="${PLUS_GPU:-1}"
MAIN_GPU="${MAIN_GPU:-1}"

PLUS_RESULTS_DIR="${PLUS_RESULTS_DIR:-exp_results/v8}"
PLUS_VIDEO_DIR="${PLUS_VIDEO_DIR:-videos/v8}"
MAIN_RESULTS_DIR="${MAIN_RESULTS_DIR:-exp_results/v8_xenonmain}"
MAIN_VIDEO_DIR="${MAIN_VIDEO_DIR:-videos/v8_xenonmain}"

SUPERVISOR_LOG="${SUPERVISOR_LOG:-/tmp/xenon_v8_no_decisioner_compare_$(date +%Y%m%d_%H%M%S).log}"

TASK_IDS=(12 10 6)
declare -A TASK_NAMES=(
  [6]="diamond_chestplate"
  [10]="golden_leggings"
  [12]="golden_chestplate"
)

log() {
  printf "%s %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

cleanup_minecraft() {
  pkill -f "[j]ava.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "[x]vfb-run|[X]vfb" 2>/dev/null || true
  pkill -9 -f "[l]aunchClient" 2>/dev/null || true
}

reset_output_dir() {
  local repo="$1"
  local rel="$2"
  if [ -z "$repo" ] || [ -z "$rel" ] || [ "$rel" = "/" ] || [ "$rel" = "." ]; then
    log "refusing to reset unsafe path repo=$repo rel=$rel"
    return 1
  fi
  rm -rf "$repo/$rel"
  mkdir -p "$repo/$rel"
}

validate_results() {
  local repo="$1"
  local results_rel="$2"
  local exp_base="$3"
  local label="$4"

  "$PYTHON_BIN" - "$repo/$results_rel" "$exp_base" "$TRIALS" "$label" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
exp_base = int(sys.argv[2])
trials = int(sys.argv[3])
label = sys.argv[4]
tasks = [(12, "golden_chestplate"), (10, "golden_leggings"), (6, "diamond_chestplate")]

invalid = 0
if results_dir.exists():
    for path in results_dir.glob("*.json"):
        try:
            with path.open() as fh:
                json.load(fh)
        except Exception:
            invalid += 1
            try:
                path.unlink()
            except OSError:
                pass

all_ok = True
print(f"[{label}] result_dir={results_dir}")
for tid, name in tasks:
    valid = 0
    missing = []
    for rep in range(trials):
        exp_num = exp_base + tid * 100 + rep
        files = sorted(results_dir.glob(f"*_{exp_num}_*.json")) if results_dir.exists() else []
        ok = False
        for path in files:
            try:
                with path.open() as fh:
                    json.load(fh)
                ok = True
                break
            except Exception:
                continue
        if ok:
            valid += 1
        else:
            missing.append(exp_num)
    print(f"[{label}] valid_json_{name}={valid}/{trials} missing={missing}")
    if valid < trials:
        all_ok = False
print(f"[{label}] invalid_json_removed={invalid}")
sys.exit(0 if all_ok else 1)
PY
}

run_plus_until_valid() {
  local round
  for round in $(seq 1 "$MAX_VALIDATION_ROUNDS"); do
    log "plus_v8_no_decisioner round=$round start"
    (
      cd "$PLUS_REPO" || exit 1
      RUN_LABEL="v8_no_decisioner_round${round}" \
      DECISIONER_ENABLED=0 \
      EXP_NUM_BASE="$PLUS_EXP_NUM_BASE" \
      TRIALS="$TRIALS" \
      SEED_BASE="$SEED_BASE" \
      SERVER_PORT="$PLUS_SERVER_PORT" \
      GPU="$PLUS_GPU" \
      RESULTS_DIR="$PLUS_RESULTS_DIR" \
      VIDEO_DIR="$PLUS_VIDEO_DIR" \
      SKIP_DONE=1 \
      XENON_RANDOM_ORE_RARE_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_GOLD_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_REDSTONE_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_DIAMOND_MULTIPLIER=1.0 \
      QWEN_VLLM_BASE_URL="$QWEN_VLLM_BASE_URL" \
      QWEN_VLLM_MODEL="$QWEN_VLLM_MODEL" \
      bash scripts/run_v7_armor_targeted.sh
    ) 2>&1 | tee -a "$SUPERVISOR_LOG"

    if validate_results "$PLUS_REPO" "$PLUS_RESULTS_DIR" "$PLUS_EXP_NUM_BASE" "plus_v8_no_decisioner" 2>&1 | tee -a "$SUPERVISOR_LOG"; then
      log "plus_v8_no_decisioner valid-result check PASS"
      return 0
    fi
    log "plus_v8_no_decisioner still missing valid files; rerunning only missing reps"
    cleanup_minecraft
    sleep 10
  done
  log "plus_v8_no_decisioner failed to reach valid-result target"
  return 1
}

run_main_until_valid() {
  local round
  for round in $(seq 1 "$MAX_VALIDATION_ROUNDS"); do
    log "xenon_main_v8_compare round=$round start"
    (
      cd "$MAIN_REPO" || exit 1
      RUN_LABEL="v8_xenonmain_round${round}" \
      EXP_NUM_BASE="$MAIN_EXP_NUM_BASE" \
      TRIALS="$TRIALS" \
      SEED_BASE="$SEED_BASE" \
      SERVER_PORT="$MAIN_SERVER_PORT" \
      GPU="$MAIN_GPU" \
      RESULTS_DIR="$MAIN_RESULTS_DIR" \
      VIDEO_DIR="$MAIN_VIDEO_DIR" \
      CLEAR_RESULTS=0 \
      SKIP_DONE=1 \
      XENON_RANDOM_ORE_RARE_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_GOLD_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_REDSTONE_MULTIPLIER=1.0 \
      XENON_RANDOM_ORE_DIAMOND_MULTIPLIER=1.0 \
      QWEN_VLLM_BASE_URL="$QWEN_VLLM_BASE_URL" \
      QWEN_VLLM_MODEL="$QWEN_VLLM_MODEL" \
      bash scripts/run_v7_dynamic_gold15x_armor.sh
    ) 2>&1 | tee -a "$SUPERVISOR_LOG"

    if validate_results "$MAIN_REPO" "$MAIN_RESULTS_DIR" "$MAIN_EXP_NUM_BASE" "xenon_main_v8_compare" 2>&1 | tee -a "$SUPERVISOR_LOG"; then
      log "xenon_main_v8_compare valid-result check PASS"
      return 0
    fi
    log "xenon_main_v8_compare still missing valid files; rerunning only missing reps"
    cleanup_minecraft
    sleep 10
  done
  log "xenon_main_v8_compare failed to reach valid-result target"
  return 1
}

log "V8 comparison supervisor start"
log "tasks=${TASK_IDS[*]} trials=$TRIALS seed_base=$SEED_BASE"
log "plus: decisioner=disabled exp_base=$PLUS_EXP_NUM_BASE gpu=$PLUS_GPU results=$PLUS_REPO/$PLUS_RESULTS_DIR videos=$PLUS_REPO/$PLUS_VIDEO_DIR"
log "main: exp_base=$MAIN_EXP_NUM_BASE gpu=$MAIN_GPU results=$MAIN_REPO/$MAIN_RESULTS_DIR videos=$MAIN_REPO/$MAIN_VIDEO_DIR"
log "shared: DefaultWorldGenerator, random_ore_multiplier=1.0 (all ores ~=10% per eligible call), mobs disabled/peaceful via evaluate.yaml, planner=$QWEN_VLLM_BASE_URL model=$QWEN_VLLM_MODEL"

cleanup_minecraft
reset_output_dir "$PLUS_REPO" "$PLUS_RESULTS_DIR" || exit 1
reset_output_dir "$PLUS_REPO" "$PLUS_VIDEO_DIR" || exit 1

run_plus_until_valid || exit 1

log "cleaning XENON-main V8 comparison outputs before main run"
cleanup_minecraft
reset_output_dir "$MAIN_REPO" "$MAIN_RESULTS_DIR" || exit 1
reset_output_dir "$MAIN_REPO" "$MAIN_VIDEO_DIR" || exit 1

run_main_until_valid || exit 1

cleanup_minecraft
log "V8 comparison supervisor completed"
