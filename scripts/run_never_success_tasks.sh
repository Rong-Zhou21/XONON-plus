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
PYTHON_BIN=${PYTHON_BIN:-/home/yzb/.conda/envs/vllm_qwen2_5_vl/bin/python}

summary=${1:-/tmp/xenon_plus_never_success_summary.log}
max_attempts=${XENON_MAX_VALID_ATTEMPTS:-3}

printf "=== XENON-plus never-success task rerun ===\nstart: %s\nrepo: %s\n" "$(date)" "$repo_dir" > "$summary"

run_task() {
  category="$1"
  bench="$2"
  task_id="$3"
  exp="$4"
  attempt=1

  while [ "$attempt" -le "$max_attempts" ]; do
    run_exp=$((exp + (attempt - 1) * 10000))
    log="/tmp/xenon_plus_never_success_${bench}_t${task_id}_exp${run_exp}_a${attempt}.log"

    printf -- "---- category=%s benchmark=%s task=%s exp=%s attempt=%s start %s ----\n" \
      "$category" "$bench" "$task_id" "$run_exp" "$attempt" "$(date +%F_%H:%M:%S)" | tee -a "$summary"

    xvfb-run -a "$PYTHON_BIN" -u src/optimus1/main_planning.py \
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
infra_early_stop = status == "env_step_timeout" or (steps < 300 and failed_waypoints == ["logs"])
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
    sleep 3
  done
}

run_task Gold golden 1 9101
run_task Gold golden 2 9102

run_task Diamond diamond 4 9204
run_task Diamond diamond 5 9205
run_task Diamond diamond 6 9206

run_task Iron iron 1 9301
run_task Iron iron 5 9305
run_task Iron iron 7 9307
run_task Iron iron 8 9308
run_task Iron iron 10 9310
run_task Iron iron 14 9314

printf "end: %s\n" "$(date)" | tee -a "$summary"
