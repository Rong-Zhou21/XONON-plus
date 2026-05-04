#!/usr/bin/env bash
# scripts/rerun_iron_gold_with_pillar_up.sh
#
# Re-run the iron + gold tasks that have historically not succeeded,
# with the new perception+pillar-up feature ON (default), so we can
# observe whether the overshoot-relevel + horizontal-mine flow helps.
#
# Per the user spec: "允许失败，但要保证正常跑完一遍". So:
#   * one attempt per task
#   * retry ONLY when the run aborted abnormally (crash, no result JSON,
#     env_step_timeout with too-few steps, "logs" failed_waypoints)
#   * up to XENON_MAX_VALID_ATTEMPTS attempts (default 3)
#   * a failure verdict counts as "completed" - we move on
#
# Usage (inside the container):
#     bash scripts/rerun_iron_gold_with_pillar_up.sh [SUMMARY_LOG_PATH]
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir" || exit 1

export PYTHONPATH="$repo_dir:$repo_dir/src:$repo_dir/minerl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/app/LLM}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export QWEN_BACKEND=${QWEN_BACKEND:-vllm}
export QWEN_VLLM_BASE_URL=${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}
export QWEN_VLLM_MODEL=${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
export XENON_DISABLE_STUCK_KILL=${XENON_DISABLE_STUCK_KILL:-1}

# Switch the new pillar-up features ON. The overshoot-relevel default
# is already 1 in main_planning, but spell it out so the run is
# self-documenting in the log.
export XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT=${XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT:-1}
export XENON_OVERSHOOT_RELEVEL_MIN_DY=${XENON_OVERSHOOT_RELEVEL_MIN_DY:-2}
export XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS=${XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS:-64}
export XENON_OVERSHOOT_RELEVEL_MAX_STEPS=${XENON_OVERSHOOT_RELEVEL_MAX_STEPS:-600}

# Stream POV frames to the host monitor for live observation.
export MONITOR_URL=${MONITOR_URL:-http://172.17.0.1:8080/push}
export MONITOR_FPS=${MONITOR_FPS:-15}

PYTHON_BIN=${PYTHON_BIN:-python}

summary=${1:-/tmp/xenon_plus_iron_gold_pillar_summary.log}
max_attempts=${XENON_MAX_VALID_ATTEMPTS:-3}
task_cooldown_seconds=${XENON_TASK_COOLDOWN_SECONDS:-10}

printf "=== XENON-plus iron+gold rerun (with pillar-up upgrade flow) ===\n" > "$summary"
printf "start: %s\nrepo: %s\nmax_attempts: %s\ncooldown: %ss\n" \
  "$(date)" "$repo_dir" "$max_attempts" "$task_cooldown_seconds" >> "$summary"
printf "XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT=%s\n\n" \
  "$XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT" >> "$summary"

cleanup_env_between_tasks() {
  if [ "${XENON_CLEAN_ENV_BETWEEN_TASKS:-1}" != "1" ]; then
    sleep "$task_cooldown_seconds"
    return
  fi
  printf "cleanup: stopping stale Minecraft/Malmo/Xvfb, sleeping %ss\n" \
    "$task_cooldown_seconds" | tee -a "$summary"
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
  sleep "$task_cooldown_seconds"
}

run_task() {
  category="$1"
  bench="$2"
  task_id="$3"
  exp="$4"
  max_minutes="$5"
  attempt=1

  while [ "$attempt" -le "$max_attempts" ]; do
    run_exp=$((exp + (attempt - 1) * 10000))
    log="/tmp/xenon_plus_pillar_${bench}_t${task_id}_exp${run_exp}_a${attempt}.log"

    printf -- "---- %s bench=%s task=%s exp=%s attempt=%s start=%s ----\n" \
      "$category" "$bench" "$task_id" "$run_exp" "$attempt" "$(date +%F_%H:%M:%S)" \
      | tee -a "$summary"

    xvfb-run -a "$PYTHON_BIN" -u src/optimus1/main_planning.py \
      server.port=9100 env.times=1 env.max_minutes="$max_minutes" benchmark="$bench" \
      evaluate=["$task_id"] prefix=ours_planning exp_num="$run_exp" \
      seed=0 world_seed="$task_id" \
      > "$log" 2>&1
    rc=$?

    decision=$(python - "$bench" "$task_id" "$run_exp" "$rc" "$log" <<'PY'
import glob
import json
import os
import sys

_, bench, task_id, exp, rc, log = sys.argv
pattern = f"exp_results/v1/ours_planning_*_{int(exp):03}_*_*.json"
files = sorted(glob.glob(pattern), key=os.path.getmtime)
print(f"rc={rc}")
if not files:
    print(f"result=NO_RESULT benchmark={bench} task_id={task_id} exp={exp}")
    print(f"log={log}")
    print("retry=yes")
    raise SystemExit

p = files[-1]
try:
    with open(p) as f:
        data = json.load(f)
except Exception as exc:
    print(f"result=BAD_JSON file={p} error={exc}")
    print(f"log={log}")
    print("retry=yes")
    raise SystemExit

video = data.get("video_file") or ""
status = data.get("status_detailed")
steps = int(data.get("steps") or 0)
failed_waypoints = data.get("failed_waypoints") or []
recovery = data.get("recovery_events") or {}
ru = {}
for k in ("pillar_up", "pillar_up_smart", "raise_to_height", "raise_to_ore_band"):
    if k in recovery:
        ru[k] = len(recovery[k]) if isinstance(recovery[k], list) else recovery[k]
print(f"result_file={p}")
print(
    "task={task} success={success} status={status} steps={steps} minutes={minutes}".format(
        task=data.get("task"),
        success=data.get("success"),
        status=status,
        steps=steps,
        minutes=data.get("minutes"),
    )
)
print(f"failed_waypoints={failed_waypoints}")
print(f"pillar_recovery={ru}")
print(f"log={log}")
infra_early_stop = (status == "env_step_timeout" and steps < 300) or (steps < 300 and failed_waypoints == ["logs"])
print("retry=yes" if infra_early_stop else "retry=no")
PY
    )
    printf "%s\n\n" "$decision" | tee -a "$summary"

    if ! printf "%s\n" "$decision" | grep -q "retry=yes"; then
      break
    fi
    attempt=$((attempt + 1))
    cleanup_env_between_tasks
  done
  cleanup_env_between_tasks
}

# ---- Iron: never-succeeded task IDs ---------------------------------
run_task Iron   iron   1  80601 10
run_task Iron   iron   5  80605 10
run_task Iron   iron   7  80607 10
run_task Iron   iron   8  80608 10
run_task Iron   iron   14 80614 10

# ---- Gold: tasks with prior failures --------------------------------
run_task Gold   golden 1  80201 30
run_task Gold   golden 2  80202 30

printf "end: %s\n" "$(date)" | tee -a "$summary"
