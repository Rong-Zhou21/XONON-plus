#!/usr/bin/env bash
# ============================================================================
# XENON-plus V7 — targeted Armor fair run after surface relevel change
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

EXP_NUM_BASE="${EXP_NUM_BASE:-480000}"
RUN_LABEL="${RUN_LABEL:-v7_fair_surface_relevel}"
RUN_VERSION_LABEL="${RUN_VERSION_LABEL:-XENON-plus V7}"
LOG_PREFIX="${LOG_PREFIX:-xenon_v7}"
GPU="${GPU:-0}"
SERVER_PORT="${SERVER_PORT:-9100}"
SEED_BASE="${SEED_BASE:-0}"
TRIALS="${TRIALS:-10}"
TASK_COOLDOWN_SEC="${TASK_COOLDOWN_SEC:-3}"
SKIP_DONE="${SKIP_DONE:-0}"
MAX_RETRIES_ON_CRASH="${MAX_RETRIES_ON_CRASH:-2}"
MAX_RETRIES_ON_INFRA_EARLY_STOP="${MAX_RETRIES_ON_INFRA_EARLY_STOP:-2}"
CRASH_RETRY_COOLDOWN_SEC="${CRASH_RETRY_COOLDOWN_SEC:-15}"
DELETE_ABNORMAL_ARTIFACTS="${DELETE_ABNORMAL_ARTIFACTS:-1}"
STOP_ON_ABNORMAL_EXHAUSTED="${STOP_ON_ABNORMAL_EXHAUSTED:-1}"
PERCEPTION_ACTION_SUITE="${PERCEPTION_ACTION_SUITE:-1}"
START_APP_SERVER="${START_APP_SERVER:-1}"
APP_START_TIMEOUT_SEC="${APP_START_TIMEOUT_SEC:-120}"
CHECK_PLANNER_SERVER="${CHECK_PLANNER_SERVER:-1}"
PLANNER_START_TIMEOUT_SEC="${PLANNER_START_TIMEOUT_SEC:-30}"
GLOBAL_CLEANUP="${GLOBAL_CLEANUP:-1}"

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

# V7 behavior under test: after overshoot/bedrock relevel trigger, pillar
# back toward the observed surface height with real inventory blocks, move
# horizontally, then resume the original dig-down prompt from a new x/z.
export XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT="${XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT:-1}"
export XENON_OVERSHOOT_RELEVEL_TARGET_MODE="${XENON_OVERSHOOT_RELEVEL_TARGET_MODE:-surface}"
export XENON_OVERSHOOT_ENABLE_Y_TRIGGER="${XENON_OVERSHOOT_ENABLE_Y_TRIGGER:-0}"
export XENON_OVERSHOOT_LATERAL_BLOCKS="${XENON_OVERSHOOT_LATERAL_BLOCKS:-2}"
export XENON_OVERSHOOT_LATERAL_MAX_STEPS="${XENON_OVERSHOOT_LATERAL_MAX_STEPS:-420}"
export XENON_OVERSHOOT_LATERAL_RETRIES="${XENON_OVERSHOOT_LATERAL_RETRIES:-2}"
export XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS="${XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS:-128}"
export XENON_OVERSHOOT_RELEVEL_MAX_STEPS="${XENON_OVERSHOOT_RELEVEL_MAX_STEPS:-2200}"
export XENON_SURFACE_RELEVEL_MARGIN="${XENON_SURFACE_RELEVEL_MARGIN:-0.0}"
export XENON_SURFACE_RELEVEL_ACCEPT_DROP="${XENON_SURFACE_RELEVEL_ACCEPT_DROP:-4.0}"
export XENON_SURFACE_RELEVEL_ACCEPT_HIGHEST="${XENON_SURFACE_RELEVEL_ACCEPT_HIGHEST:-1}"
export XENON_SURFACE_RELEVEL_MAX_ABOVE="${XENON_SURFACE_RELEVEL_MAX_ABOVE:-8.0}"
export XENON_SURFACE_TUNNEL_RELEVEL_AFTER_DROP="${XENON_SURFACE_TUNNEL_RELEVEL_AFTER_DROP:--1.0}"
export XENON_SURFACE_RELEVEL_STEPS_PER_BLOCK="${XENON_SURFACE_RELEVEL_STEPS_PER_BLOCK:-24}"
export XENON_SURFACE_RELEVEL_STEP_MARGIN="${XENON_SURFACE_RELEVEL_STEP_MARGIN:-160}"
export XENON_BEDROCK_FLOOR_Y="${XENON_BEDROCK_FLOOR_Y:-5.5}"
export XENON_BEDROCK_STAGNANT_TICKS="${XENON_BEDROCK_STAGNANT_TICKS:-600}"
export XENON_BEDROCK_BLOCK_STAGNANT_TICKS="${XENON_BEDROCK_BLOCK_STAGNANT_TICKS:-600}"
export XENON_CORRIDOR_FEET_PITCH="${XENON_CORRIDOR_FEET_PITCH:-12.0}"
export XENON_CORRIDOR_AXIS_LOCK="${XENON_CORRIDOR_AXIS_LOCK:-1}"
export XENON_CORRIDOR_YAW_MODE="${XENON_CORRIDOR_YAW_MODE:-fan30}"
export XENON_CORRIDOR_YAW_OFFSETS="${XENON_CORRIDOR_YAW_OFFSETS:-0,30,-30}"
export XENON_CORRIDOR_DIRECTION_SWEEP="${XENON_CORRIDOR_DIRECTION_SWEEP:-1}"
export XENON_CORRIDOR_DIRECTION_SWEEP_OFFSETS="${XENON_CORRIDOR_DIRECTION_SWEEP_OFFSETS:-0,90,-90,180}"
export XENON_CORRIDOR_STABILIZE_FLOOR="${XENON_CORRIDOR_STABILIZE_FLOOR:-0}"
export XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP="${XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP:-0}"
export XENON_CORRIDOR_STABILIZE_RADIUS="${XENON_CORRIDOR_STABILIZE_RADIUS:-2}"
export XENON_CORRIDOR_STABILIZE_LANES="${XENON_CORRIDOR_STABILIZE_LANES:-1}"
export XENON_CORRIDOR_STABILIZE_LENGTH="${XENON_CORRIDOR_STABILIZE_LENGTH:-10}"
export XENON_CORRIDOR_STABILIZE_WIDTH="${XENON_CORRIDOR_STABILIZE_WIDTH:-1}"
export XENON_CORRIDOR_STABILIZE_BLOCK="${XENON_CORRIDOR_STABILIZE_BLOCK:-cobblestone}"
export XENON_CORRIDOR_BLOCKED_FRONT_PITCH="${XENON_CORRIDOR_BLOCKED_FRONT_PITCH:-0.0}"
export XENON_CORRIDOR_UP_PITCH="${XENON_CORRIDOR_UP_PITCH:--65.0}"
export XENON_CORRIDOR_BLOCKED_FEET_PITCH="${XENON_CORRIDOR_BLOCKED_FEET_PITCH:-45.0}"
export XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH="${XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH:-55.0}"
export XENON_CORRIDOR_MIN_MOVE_DELTA="${XENON_CORRIDOR_MIN_MOVE_DELTA:-1.0}"
export XENON_CORRIDOR_SEGMENT_MIN_MOVE_DELTA="${XENON_CORRIDOR_SEGMENT_MIN_MOVE_DELTA:-0.85}"
export XENON_CORRIDOR_MIN_SUCCESS_BLOCKS="${XENON_CORRIDOR_MIN_SUCCESS_BLOCKS:-2}"
export XENON_CORRIDOR_UNSTUCK_PASSES="${XENON_CORRIDOR_UNSTUCK_PASSES:-3}"
export XENON_CORRIDOR_MOVE_ATTACK_TICKS="${XENON_CORRIDOR_MOVE_ATTACK_TICKS:-10}"
export XENON_CORRIDOR_STOP_AFTER_MOVE="${XENON_CORRIDOR_STOP_AFTER_MOVE:-0}"
export XENON_CORRIDOR_BLOCKED_HEAD_BUDGET="${XENON_CORRIDOR_BLOCKED_HEAD_BUDGET:-36}"
export XENON_CORRIDOR_BLOCKED_UP_BUDGET="${XENON_CORRIDOR_BLOCKED_UP_BUDGET:-0}"
export XENON_CORRIDOR_BLOCKED_FEET_BUDGET="${XENON_CORRIDOR_BLOCKED_FEET_BUDGET:-48}"
export XENON_CORRIDOR_ATTACK_FORWARD="${XENON_CORRIDOR_ATTACK_FORWARD:-1}"
export XENON_LATERAL_MAX_Y_DROP="${XENON_LATERAL_MAX_Y_DROP:-1.25}"
export XENON_TUNNEL_RELEVEL_AFTER_DROP="${XENON_TUNNEL_RELEVEL_AFTER_DROP:-0.75}"
export XENON_TUNNEL_RELEVEL_TOLERANCE="${XENON_TUNNEL_RELEVEL_TOLERANCE:-0.75}"
export XENON_TUNNEL_ACCEPT_HEIGHT_DROP="${XENON_TUNNEL_ACCEPT_HEIGHT_DROP:-4.0}"
export XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT="${XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT:-1}"
export XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT="${XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT:-1}"
export XENON_TUNNEL_ACCEPT_PARTIAL_BLOCKS="${XENON_TUNNEL_ACCEPT_PARTIAL_BLOCKS:-2}"
export XENON_TUNNEL_BAND_LO_MARGIN="${XENON_TUNNEL_BAND_LO_MARGIN:-0.5}"
export XENON_TUNNEL_ACCEPT_MIN_Y="${XENON_TUNNEL_ACCEPT_MIN_Y:-8.0}"
export XENON_ALLOW_STEVE1_FORWARD_FALLBACK="${XENON_ALLOW_STEVE1_FORWARD_FALLBACK:-0}"
export XENON_TUNNEL_SCRIPTED_DIGDOWN_BLOCKS="${XENON_TUNNEL_SCRIPTED_DIGDOWN_BLOCKS:-5}"
export XENON_TUNNEL_SCRIPTED_DIGDOWN_MAX_STEPS="${XENON_TUNNEL_SCRIPTED_DIGDOWN_MAX_STEPS:-360}"
export XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE="${XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE:-0}"
export XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE="${XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE:-0}"
export XENON_ENABLE_RANDOM_ORE_ONCE="${XENON_ENABLE_RANDOM_ORE_ONCE:-1}"
export XENON_RANDOM_ORE_RARE_MULTIPLIER="${XENON_RANDOM_ORE_RARE_MULTIPLIER:-1.0}"
export XENON_RANDOM_ORE_COAL_MULTIPLIER="${XENON_RANDOM_ORE_COAL_MULTIPLIER:-1.0}"
export XENON_RANDOM_ORE_IRON_MULTIPLIER="${XENON_RANDOM_ORE_IRON_MULTIPLIER:-1.0}"
export XENON_RANDOM_ORE_GOLD_MULTIPLIER="${XENON_RANDOM_ORE_GOLD_MULTIPLIER:-$XENON_RANDOM_ORE_RARE_MULTIPLIER}"
export XENON_RANDOM_ORE_REDSTONE_MULTIPLIER="${XENON_RANDOM_ORE_REDSTONE_MULTIPLIER:-$XENON_RANDOM_ORE_RARE_MULTIPLIER}"
export XENON_RANDOM_ORE_DIAMOND_MULTIPLIER="${XENON_RANDOM_ORE_DIAMOND_MULTIPLIER:-$XENON_RANDOM_ORE_RARE_MULTIPLIER}"
export XENON_SCRIPTED_DIGDOWN_FORCE_ORE_DYS="${XENON_SCRIPTED_DIGDOWN_FORCE_ORE_DYS:--3,-4,-5,-2,-1}"
export XENON_SCRIPTED_DIGDOWN_ORE_THRESHOLD="${XENON_SCRIPTED_DIGDOWN_ORE_THRESHOLD:-0.0}"
export XENON_SCRIPTED_DIGDOWN_MIN_Y="${XENON_SCRIPTED_DIGDOWN_MIN_Y:-4.0}"
export XENON_TUNNEL_REPEAT_AFTER_SCRIPTEDDOWN="${XENON_TUNNEL_REPEAT_AFTER_SCRIPTEDDOWN:-0}"
export XENON_TUNNEL_SCRIPTED_LOOP_CAN_TRIGGER_RELEVEL="${XENON_TUNNEL_SCRIPTED_LOOP_CAN_TRIGGER_RELEVEL:-0}"
export XENON_TUNNEL_RETRY_RELEVEL_FROM_ABOVE="${XENON_TUNNEL_RETRY_RELEVEL_FROM_ABOVE:-1}"
export XENON_TUNNEL_LOOP_MAX_ABOVE_TARGET="${XENON_TUNNEL_LOOP_MAX_ABOVE_TARGET:-2.5}"
export XENON_TUNNEL_MAX_ABOVE_TARGET="${XENON_TUNNEL_MAX_ABOVE_TARGET:-2.5}"
export XENON_RELEVEL_MAX_ABOVE_TARGET="${XENON_RELEVEL_MAX_ABOVE_TARGET:-2.5}"
export XENON_LOCK_TARGET_LAYER_LOOP_PROMPT="${XENON_LOCK_TARGET_LAYER_LOOP_PROMPT:-1}"
export XENON_SUPPRESS_REASONING_DURING_TARGET_LOOP="${XENON_SUPPRESS_REASONING_DURING_TARGET_LOOP:-1}"
export XENON_ENABLE_COMMAND_RELEVEL_FALLBACK="${XENON_ENABLE_COMMAND_RELEVEL_FALLBACK:-0}"
export XENON_ENABLE_COMMAND_CRAFT_FALLBACK="${XENON_ENABLE_COMMAND_CRAFT_FALLBACK:-0}"
export XENON_TUNNEL_ABORT_ON_TERMINAL="${XENON_TUNNEL_ABORT_ON_TERMINAL:-1}"
export XENON_PILLAR_CLEAR_OVERHEAD_TICKS="${XENON_PILLAR_CLEAR_OVERHEAD_TICKS:-48}"
export XENON_CORRIDOR_MAX_POSITION_JUMP="${XENON_CORRIDOR_MAX_POSITION_JUMP:-6.0}"
export XENON_TREE_CONTACT_GRACE_TICKS="${XENON_TREE_CONTACT_GRACE_TICKS:-120}"
export XENON_RESPAWN_EQUIP_RETRY_TICKS="${XENON_RESPAWN_EQUIP_RETRY_TICKS:-240}"
export XENON_RESPAWN_EQUIP_INTERVAL_TICKS="${XENON_RESPAWN_EQUIP_INTERVAL_TICKS:-40}"

VIDEO_DIR="${VIDEO_DIR:-videos/v7}"
RESULTS_DIR="${RESULTS_DIR:-exp_results/v7}"
mkdir -p "$VIDEO_DIR" "$RESULTS_DIR"

SUMMARY_FILE="${SUMMARY_FILE:-/tmp/${LOG_PREFIX}_${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)_summary.log}"
APP_SERVER_LOG="${APP_SERVER_LOG:-/tmp/${LOG_PREFIX}_${RUN_LABEL}_app_$(date +%Y%m%d_%H%M%S).log}"

if [ -n "${TASK_IDS_OVERRIDE:-}" ]; then
  read -r -a TASK_IDS <<< "$TASK_IDS_OVERRIDE"
else
  TASK_IDS=(12 10 6)
fi
declare -A TASK_NAMES=(
  [6]="diamond_chestplate"
  [10]="golden_leggings"
  [12]="golden_chestplate"
)

cleanup() {
  if [ "$GLOBAL_CLEANUP" != "1" ]; then
    return 0
  fi
  pkill -f "[j]ava.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "[x]vfb-run|[X]vfb" 2>/dev/null || true
  pkill -9 -f "[l]aunchClient" 2>/dev/null || true
}

server_ready() {
  "$PYTHON_BIN" - "$SERVER_PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = sys.argv[1]
urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2).read(64)
PY
}

planner_ready() {
  if [ "${QWEN_BACKEND,,}" != "vllm" ]; then
    return 0
  fi
  "$PYTHON_BIN" - "$QWEN_VLLM_BASE_URL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
urllib.request.urlopen(f"{base}/models", timeout=3).read(64)
PY
}

ensure_planner_server() {
  if [ "$CHECK_PLANNER_SERVER" != "1" ]; then
    return 0
  fi
  if [ "${QWEN_BACKEND,,}" != "vllm" ]; then
    return 0
  fi

  for _ in $(seq 1 "$PLANNER_START_TIMEOUT_SEC"); do
    if planner_ready; then
      printf " planner_server         : ready at %s\n" "$QWEN_VLLM_BASE_URL" | tee -a "$SUMMARY_FILE"
      return 0
    fi
    sleep 1
  done

  printf " planner_server         : not ready at %s after %ss\n" \
    "$QWEN_VLLM_BASE_URL" "$PLANNER_START_TIMEOUT_SEC" | tee -a "$SUMMARY_FILE"
  return 1
}

ensure_app_server() {
  if [ "$START_APP_SERVER" != "1" ]; then
    return 0
  fi

  if server_ready; then
    printf " app_server             : existing server on port %s\n" "$SERVER_PORT" | tee -a "$SUMMARY_FILE"
    return 0
  fi

  printf " app_server             : starting app.py on port %s log=%s\n" "$SERVER_PORT" "$APP_SERVER_LOG" | tee -a "$SUMMARY_FILE"
  nohup "$PYTHON_BIN" -u app.py --port "$SERVER_PORT" > "$APP_SERVER_LOG" 2>&1 < /dev/null &
  APP_SERVER_PID=$!
  printf " app_server_pid         : %s\n" "$APP_SERVER_PID" | tee -a "$SUMMARY_FILE"

  for _ in $(seq 1 "$APP_START_TIMEOUT_SEC"); do
    if server_ready; then
      printf " app_server_ready       : yes\n" | tee -a "$SUMMARY_FILE"
      return 0
    fi
    sleep 1
  done

  printf " app_server_ready       : no (timeout after %ss)\n" "$APP_START_TIMEOUT_SEC" | tee -a "$SUMMARY_FILE"
  tail -80 "$APP_SERVER_LOG" 2>/dev/null | tee -a "$SUMMARY_FILE" || true
  return 1
}

is_abnormal_status() {
  case "$1" in
    env_step_timeout|crash_*) return 0 ;;
    *) return 1 ;;
  esac
}

is_low_level_interaction_abnormal() {
  local log_file="$1"
  [ -n "$log_file" ] && [ -f "$log_file" ] || return 1
  "$PYTHON_BIN" - "$log_file" <<'PY' 2>/dev/null
import re
import sys

try:
    text = open(sys.argv[1], errors="ignore").read()
except Exception:
    sys.exit(1)

unknown_failures = len(re.findall(r"fail for unkown reason", text))
interaction_prompt = re.search(r"Subgoal Prompt:\s*(craft|smelt)\b", text)
if unknown_failures >= 3 and interaction_prompt:
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
 $RUN_VERSION_LABEL — targeted Armor run
============================================================
 run_label              : $RUN_LABEL
 python                 : $PYTHON_BIN
 decisioner             : $([ "$DECISIONER_ENABLED" = "1" ] && echo "ENABLED  ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P" || echo "disabled (baseline)")
 perception_action_suite: $([ "$PERCEPTION_ACTION_SUITE" = "1" ] && echo "ON" || echo "OFF")
 exp_num_base           : $EXP_NUM_BASE
 trials_per_task        : $TRIALS
 armor_task_ids         : ${TASK_IDS[*]}
 world_ore_generation   : DefaultWorldGenerator generatorOptions={}
 relevel_target_mode    : $XENON_OVERSHOOT_RELEVEL_TARGET_MODE
	 overshoot_y_trigger    : $XENON_OVERSHOOT_ENABLE_Y_TRIGGER
	 horizontal_tunnel_blocks: $XENON_OVERSHOOT_LATERAL_BLOCKS
	 lateral_retries        : $XENON_OVERSHOOT_LATERAL_RETRIES
	 bedrock_floor_y        : $XENON_BEDROCK_FLOOR_Y
	 bedrock_stagnant_ticks : $XENON_BEDROCK_STAGNANT_TICKS
	 bedrock_block_stagnant : $XENON_BEDROCK_BLOCK_STAGNANT_TICKS
	 lateral_body_pitch     : $XENON_CORRIDOR_FEET_PITCH
 axis_lock              : $XENON_CORRIDOR_AXIS_LOCK
yaw_mode               : $XENON_CORRIDOR_YAW_MODE
yaw_offsets            : $XENON_CORRIDOR_YAW_OFFSETS
direction_sweep        : $XENON_CORRIDOR_DIRECTION_SWEEP
direction_sweep_offsets: $XENON_CORRIDOR_DIRECTION_SWEEP_OFFSETS
stabilize_floor        : $XENON_CORRIDOR_STABILIZE_FLOOR
stabilize_after_sweep  : $XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP
stabilize_radius       : $XENON_CORRIDOR_STABILIZE_RADIUS
stabilize_lanes        : $XENON_CORRIDOR_STABILIZE_LANES
stabilize_length       : $XENON_CORRIDOR_STABILIZE_LENGTH
stabilize_width        : $XENON_CORRIDOR_STABILIZE_WIDTH
stabilize_block        : $XENON_CORRIDOR_STABILIZE_BLOCK
blocked_front_pitch    : $XENON_CORRIDOR_BLOCKED_FRONT_PITCH
blocked_up_pitch       : $XENON_CORRIDOR_UP_PITCH
 blocked_up_budget      : $XENON_CORRIDOR_BLOCKED_UP_BUDGET
 blocked_feet_pitch     : $XENON_CORRIDOR_BLOCKED_FEET_PITCH
 blocked_feet_max_pitch : $XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH
 min_forward_delta      : $XENON_CORRIDOR_MIN_MOVE_DELTA
 segment_min_delta      : $XENON_CORRIDOR_SEGMENT_MIN_MOVE_DELTA
 min_success_blocks     : $XENON_CORRIDOR_MIN_SUCCESS_BLOCKS
 unstuck_passes         : $XENON_CORRIDOR_UNSTUCK_PASSES
 move_attack_ticks      : $XENON_CORRIDOR_MOVE_ATTACK_TICKS
 stop_after_move        : $XENON_CORRIDOR_STOP_AFTER_MOVE
 lateral_attack_forward : $XENON_CORRIDOR_ATTACK_FORWARD
 lateral_max_y_drop     : $XENON_LATERAL_MAX_Y_DROP
 tunnel_relevel_drop    : $XENON_TUNNEL_RELEVEL_AFTER_DROP
 tunnel_relevel_tol     : $XENON_TUNNEL_RELEVEL_TOLERANCE
 tunnel_accept_drop     : $XENON_TUNNEL_ACCEPT_HEIGHT_DROP
 tunnel_low_soft_accept : $XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT
 tunnel_drop_soft_accept: $XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT
 tunnel_accept_partial  : $XENON_TUNNEL_ACCEPT_PARTIAL_BLOCKS
 tunnel_band_lo_margin  : $XENON_TUNNEL_BAND_LO_MARGIN
 tunnel_accept_min_y    : $XENON_TUNNEL_ACCEPT_MIN_Y
 steve1_forward_fallback: $XENON_ALLOW_STEVE1_FORWARD_FALLBACK
 scripted_digdown_blocks: $XENON_TUNNEL_SCRIPTED_DIGDOWN_BLOCKS
 scripted_digdown_steps : $XENON_TUNNEL_SCRIPTED_DIGDOWN_MAX_STEPS
 scripted_digdown_ore   : $XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE
scripted_force_target  : $XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE
random_ore_once        : $XENON_ENABLE_RANDOM_ORE_ONCE
random_ore_rare_mult   : $XENON_RANDOM_ORE_RARE_MULTIPLIER
random_ore_coal_mult   : $XENON_RANDOM_ORE_COAL_MULTIPLIER
random_ore_iron_mult   : $XENON_RANDOM_ORE_IRON_MULTIPLIER
random_ore_gold_mult   : $XENON_RANDOM_ORE_GOLD_MULTIPLIER
random_ore_redstone_mult: $XENON_RANDOM_ORE_REDSTONE_MULTIPLIER
random_ore_diamond_mult: $XENON_RANDOM_ORE_DIAMOND_MULTIPLIER
scripted_force_dys     : $XENON_SCRIPTED_DIGDOWN_FORCE_ORE_DYS
scripted_ore_threshold : $XENON_SCRIPTED_DIGDOWN_ORE_THRESHOLD
scripted_digdown_min_y : $XENON_SCRIPTED_DIGDOWN_MIN_Y
repeat_after_scripted  : $XENON_TUNNEL_REPEAT_AFTER_SCRIPTEDDOWN
scripted_loop_can_relevel: $XENON_TUNNEL_SCRIPTED_LOOP_CAN_TRIGGER_RELEVEL
retry_relevel_from_above: $XENON_TUNNEL_RETRY_RELEVEL_FROM_ABOVE
loop_max_above_target  : $XENON_TUNNEL_LOOP_MAX_ABOVE_TARGET
tunnel_max_above       : $XENON_TUNNEL_MAX_ABOVE_TARGET
relevel_max_above      : $XENON_RELEVEL_MAX_ABOVE_TARGET
lock_target_loop_prompt: $XENON_LOCK_TARGET_LAYER_LOOP_PROMPT
suppress_loop_reasoning: $XENON_SUPPRESS_REASONING_DURING_TARGET_LOOP
cmd_relevel_fallback   : $XENON_ENABLE_COMMAND_RELEVEL_FALLBACK
cmd_craft_fallback     : $XENON_ENABLE_COMMAND_CRAFT_FALLBACK
 tunnel_abort_terminal  : $XENON_TUNNEL_ABORT_ON_TERMINAL
 relevel_max_blocks     : $XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS
 relevel_max_steps      : $XENON_OVERSHOOT_RELEVEL_MAX_STEPS
 surface_accept_drop    : $XENON_SURFACE_RELEVEL_ACCEPT_DROP
 surface_accept_highest : $XENON_SURFACE_RELEVEL_ACCEPT_HIGHEST
 surface_max_above      : $XENON_SURFACE_RELEVEL_MAX_ABOVE
 surface_tunnel_relevel : $XENON_SURFACE_TUNNEL_RELEVEL_AFTER_DROP
 pillar_overhead_ticks  : $XENON_PILLAR_CLEAR_OVERHEAD_TICKS
	 max_position_jump      : $XENON_CORRIDOR_MAX_POSITION_JUMP
	 tree_contact_grace     : $XENON_TREE_CONTACT_GRACE_TICKS
	 respawn_equip_retry    : $XENON_RESPAWN_EQUIP_RETRY_TICKS
	 respawn_equip_interval : $XENON_RESPAWN_EQUIP_INTERVAL_TICKS
	 GPU                    : $GPU
 server_port            : $SERVER_PORT
 video_dir              : $VIDEO_DIR
	 results_dir            : $RESULTS_DIR
	 summary                : $SUMMARY_FILE
	 app_server_log         : $APP_SERVER_LOG
	 start_app_server       : $START_APP_SERVER
	 planner_backend        : $QWEN_BACKEND
	 planner_base_url       : $QWEN_VLLM_BASE_URL
	 check_planner_server   : $CHECK_PLANNER_SERVER
	 max_abnormal_retries   : $MAX_RETRIES_ON_CRASH
	 max_infra_retries      : $MAX_RETRIES_ON_INFRA_EARLY_STOP
 delete_abnormal_artifacts: $DELETE_ABNORMAL_ARTIFACTS
 stop_on_abnormal_exhausted: $STOP_ON_ABNORMAL_EXHAUSTED
 skip_done              : $SKIP_DONE
 global_cleanup         : $GLOBAL_CLEANUP
 start_time             : $(date)
============================================================
INFO

cleanup
sleep 3
ensure_planner_server || exit 1
ensure_app_server || exit 1

for TID in "${TASK_IDS[@]}"; do
  for ((REP=0; REP<TRIALS; REP++)); do
    EXP_NUM=$((EXP_NUM_BASE + TID * 100 + REP))
    SEED=$((SEED_BASE + REP))
    WORLD_SEED=$((SEED_BASE + REP))
    MAXMIN=30
    DONE=$((DONE + 1))

    EXISTING=$(ls -t "$RESULTS_DIR"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
    if [ "$SKIP_DONE" = "1" ] && [ -n "$EXISTING" ]; then
      if "$PYTHON_BIN" - "$EXISTING" <<'PY' >/dev/null 2>&1
import json
import sys
with open(sys.argv[1]) as fh:
    json.load(fh)
PY
      then
        SKIPPED=$((SKIPPED + 1))
        printf "[%2d/%d] armor task=%-2s %-20s rep=%-2s exp=%-6s SKIP (%s)\n" \
          "$DONE" "$TOTAL_TASKS" "$TID" "${TASK_NAMES[$TID]}" "$REP" "$EXP_NUM" \
          "$(basename "$EXISTING")" | tee -a "$SUMMARY_FILE"
        continue
      else
        rm -f "$EXISTING"
      fi
    fi

    attempt=0
    STATUS=""
    LAST_RETRY_LIMIT="$MAX_RETRIES_ON_CRASH"
    T_START=$(date +%s)

    while : ; do
      if [ "$attempt" -gt 0 ]; then
        printf "       retry %d/%d (abnormal exit, last status=%s)\n" \
          "$attempt" "$LAST_RETRY_LIMIT" "$STATUS" | tee -a "$SUMMARY_FILE"
        cleanup
        sleep "$CRASH_RETRY_COOLDOWN_SEC"
      fi

      LOG_FILE="/tmp/${LOG_PREFIX}_${RUN_LABEL}_armor_t${TID}_rep${REP}_exp${EXP_NUM}_$(date +%Y%m%d_%H%M%S).log"

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
infra = " infra_early_stop=1" if d.get("infra_early_stop") else ""
print(f"{'SUCCESS' if d.get('success') else 'FAIL'} status={d.get('status_detailed')} steps={d.get('steps')} minutes={d.get('minutes')}{infra}")
PY
)
      else
        STATUS="NO_RESULT (rc=$RC)"
      fi

      should_retry=0
      abnormal_exit=0
      retry_limit="$MAX_RETRIES_ON_CRASH"
      if [ -z "$RESULT_FILE" ]; then
        abnormal_exit=1
      else
        STATUS_DETAILED=$("$PYTHON_BIN" - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("status_detailed", ""))
PY
)
        INFRA_EARLY_STOP=$("$PYTHON_BIN" - "$RESULT_FILE" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print("1" if d.get("infra_early_stop") else "0")
PY
)
        IS_SUCCESS=$(echo "$STATUS" | grep -c "^SUCCESS")
        if [ "$IS_SUCCESS" -eq 0 ] && [ "$INFRA_EARLY_STOP" = "1" ]; then
          abnormal_exit=1
          retry_limit="$MAX_RETRIES_ON_INFRA_EARLY_STOP"
        elif [ "$IS_SUCCESS" -eq 0 ] && is_abnormal_status "$STATUS_DETAILED"; then
          abnormal_exit=1
        elif [ "$IS_SUCCESS" -eq 0 ] && is_low_level_interaction_abnormal "$LOG_FILE"; then
          abnormal_exit=1
        fi
      fi

      if [ "$abnormal_exit" -eq 1 ]; then
        if [ "$DELETE_ABNORMAL_ARTIFACTS" = "1" ]; then
          delete_abnormal_artifacts "$RESULT_FILE"
          RESULT_FILE=""
        fi
        LAST_RETRY_LIMIT="$retry_limit"
        if [ "$attempt" -lt "$retry_limit" ]; then
          should_retry=1
        else
          if [ -z "$RESULT_FILE" ]; then
            STATUS="NO_RESULT_ABNORMAL_RETRIES_EXHAUSTED last=$STATUS"
          else
            STATUS="$STATUS abnormal_no_retry"
          fi
          if [ "$STOP_ON_ABNORMAL_EXHAUSTED" = "1" ]; then
            T_ELAPSED=$(( $(date +%s) - T_START ))
            printf "       result=%s elapsed=%ds (abnormal retries exhausted; stopping)\n" \
              "$STATUS" "$T_ELAPSED" | tee -a "$SUMMARY_FILE"
            cleanup
            exit 2
          fi
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
 $RUN_VERSION_LABEL targeted Armor — end summary
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
