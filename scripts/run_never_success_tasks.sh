#!/usr/bin/env bash
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

# Use the Python from the already prepared experiment environment. Do not
# silently switch to another conda env, because MineRL/Malmo compatibility is
# sensitive to the exact runtime used for previous successful runs.
PYTHON_BIN=${PYTHON_BIN:-python}

summary=${1:-/tmp/xenon_plus_never_success_summary.log}
max_attempts=${XENON_MAX_VALID_ATTEMPTS:-3}
task_cooldown_seconds=${XENON_TASK_COOLDOWN_SECONDS:-10}

printf "=== XENON-plus never-success task rerun ===\nstart: %s\nrepo: %s\nmax_attempts: %s\ncooldown_seconds: %s\n" \
  "$(date)" "$repo_dir" "$max_attempts" "$task_cooldown_seconds" > "$summary"

cleanup_env_between_tasks() {
  if [ "${XENON_CLEAN_ENV_BETWEEN_TASKS:-1}" != "1" ]; then
    sleep "$task_cooldown_seconds"
    return
  fi
  printf "cleanup: stopping possible stale MineRL/Malmo/Xvfb processes, then sleeping %ss\n" \
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
    log="/tmp/xenon_plus_never_success_${bench}_t${task_id}_exp${run_exp}_a${attempt}.log"

    printf -- "---- category=%s benchmark=%s task=%s exp=%s attempt=%s start %s ----\n" \
      "$category" "$bench" "$task_id" "$run_exp" "$attempt" "$(date +%F_%H:%M:%S)" | tee -a "$summary"

    xvfb-run -a "$PYTHON_BIN" -u src/optimus1/main_planning.py \
      server.port=9100 env.times=1 env.max_minutes="$max_minutes" benchmark="$bench" \
      evaluate=["$task_id"] prefix=ours_planning exp_num="$run_exp" seed=0 world_seed="$task_id" \
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
print(f"recovery_events={data.get('recovery_events')}")
print(f"video={video}")
print(f"video_exists={bool(video and os.path.exists(video))}")
print(f"log={log}")
infra_early_stop = (status == "env_step_timeout" and steps < 300) or (steps < 300 and failed_waypoints == ["logs"])
print("retry=yes" if infra_early_stop else "retry=no")
PY
    )
    printf "%s\n\n" "$decision" | tee -a "$summary"

    printf -- "---- end category=%s benchmark=%s task=%s exp=%s attempt=%s end %s ----\n\n" \
      "$category" "$bench" "$task_id" "$run_exp" "$attempt" "$(date +%F_%H:%M:%S)" >> "$summary"

    if ! printf "%s\n" "$decision" | grep -q "retry=yes"; then
      break
    fi
    attempt=$((attempt + 1))
    cleanup_env_between_tasks
  done
  cleanup_env_between_tasks
}

run_task Iron iron 1 10601 10
run_task Iron iron 5 10605 10
run_task Iron iron 7 10607 10
run_task Iron iron 8 10608 10
run_task Iron iron 14 10614 10

run_task Diamond diamond 6 10706 20

run_task Redstone redstone 0 10800 20
run_task Redstone redstone 5 10805 20

run_task Armor armor 5 10905 20
run_task Armor armor 6 10906 20
run_task Armor armor 7 10907 20
run_task Armor armor 8 10908 20
run_task Armor armor 10 10910 20
run_task Armor armor 11 10911 20
run_task Armor armor 12 10912 20

printf "end: %s\n" "$(date)" | tee -a "$summary"
