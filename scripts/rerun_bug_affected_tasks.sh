#!/usr/bin/env bash
set -u

cd /app/repo || exit 1

export PYTHONPATH=/app/repo:/app/repo/src:${PYTHONPATH:-}
export HF_HOME=/app/LLM
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export QWEN_BACKEND=vllm
export QWEN_VLLM_BASE_URL=http://172.17.0.1:8000/v1
export QWEN_VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export XENON_DISABLE_STUCK_KILL=1

summary=${1:-/tmp/xenon_plus_bugfix_rerun_summary.log}
printf "=== XENON-plus bug affected task rerun ===\nstart: %s\n" "$(date)" > "$summary"

run_task_with_retry() {
  category="$1"
  bench="$2"
  task_id="$3"
  exp="$4"
  max_attempts="${5:-3}"
  attempt=1

  while [ "$attempt" -le "$max_attempts" ]; do
    run_exp=$((exp + attempt - 1))
    log="/tmp/xenon_plus_bugfix_${bench}_t${task_id}_exp${run_exp}_a${attempt}.log"
    printf -- "---- category=%s benchmark=%s task=%s exp=%s attempt=%s start %s ----\n" \
      "$category" "$bench" "$task_id" "$run_exp" "$attempt" "$(date +%F_%H:%M:%S)" | tee -a "$summary"

    xvfb-run -a python -u src/optimus1/main_planning.py \
      server.port=9100 env.times=1 benchmark="$bench" \
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
with open(p) as f:
    data = json.load(f)

video = data.get("video_file") or ""
video_exists = bool(video and os.path.exists(video))
status = data.get("status_detailed")
steps = data.get("steps") or 0
failed_waypoints = data.get("failed_waypoints") or []

print(f"result_file=/app/repo/{p}")
print(
    "task={task} success={success} status={status} steps={steps} minutes={minutes}".format(
        task=data.get("task"),
        success=data.get("success"),
        status=status,
        steps=steps,
        minutes=data.get("minutes"),
    )
)
print(f"video={video}")
print(f"video_exists={video_exists}")
print(f"failed_waypoints={failed_waypoints}")
print(f"log={log}")

infra_early_stop = status == "env_step_timeout" or (steps < 300 and failed_waypoints == ["logs"])
print("retry=yes" if infra_early_stop else "retry=no")
PY
)
    printf "%s\n\n" "$decision" | tee -a "$summary"
    if ! printf "%s\n" "$decision" | grep -q "retry=yes"; then
      break
    fi
    attempt=$((attempt + 1))
    sleep 5
  done
}

# Stone failures affected by craft helper / early environment timeout bugs.
run_task_with_retry Stone stone 4 1104 3
run_task_with_retry Stone stone 5 1105 3
run_task_with_retry Stone stone 7 1107 3

# Iron failures affected by early environment timeout classification or failure video naming.
run_task_with_retry Iron iron 1 2101 3
run_task_with_retry Iron iron 2 2102 3
run_task_with_retry Iron iron 4 2104 3
run_task_with_retry Iron iron 5 2105 3
run_task_with_retry Iron iron 7 2107 3
run_task_with_retry Iron iron 8 2108 3
run_task_with_retry Iron iron 10 2110 3
run_task_with_retry Iron iron 11 2111 3
run_task_with_retry Iron iron 12 2112 3
run_task_with_retry Iron iron 13 2113 3
run_task_with_retry Iron iron 14 2114 3
run_task_with_retry Iron iron 15 2115 3

printf "end: %s\n" "$(date)" | tee -a "$summary"
