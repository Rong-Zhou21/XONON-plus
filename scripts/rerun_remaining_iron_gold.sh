#!/usr/bin/env bash
# scripts/rerun_remaining_iron_gold.sh
# Continue from where rerun_iron_gold_with_pillar_up.sh stopped:
# iron task 14 + gold tasks 1, 2 (iron task 8 was a planner deadlock
# at game-step 7041 — terminated by hand; we're not retrying it).
set -u
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir" || exit 1

export PYTHONPATH="$repo_dir:$repo_dir/src:$repo_dir/minerl:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/app/LLM}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export QWEN_BACKEND=${QWEN_BACKEND:-vllm}
export QWEN_VLLM_BASE_URL=${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}
export QWEN_VLLM_MODEL=${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
export XENON_DISABLE_STUCK_KILL=${XENON_DISABLE_STUCK_KILL:-1}
export XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT=1
export MONITOR_URL=${MONITOR_URL:-http://172.17.0.1:8080/push}
export MONITOR_FPS=${MONITOR_FPS:-15}

PYTHON_BIN=${PYTHON_BIN:-python}
summary=${1:-/tmp/xenon_plus_iron_gold_remaining.log}
max_attempts=${XENON_MAX_VALID_ATTEMPTS:-3}
task_cooldown_seconds=${XENON_TASK_COOLDOWN_SECONDS:-10}

printf "=== iron+gold remaining rerun ===\nstart=%s\n\n" "$(date)" > "$summary"

cleanup_env() {
  pkill -f "java.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
  pkill -f "xvfb-run|Xvfb" 2>/dev/null || true
  sleep "$task_cooldown_seconds"
}

run_task() {
  category="$1"; bench="$2"; task_id="$3"; exp="$4"; max_minutes="$5"
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
      seed=0 world_seed="$task_id" > "$log" 2>&1
    rc=$?

    decision=$(python - "$bench" "$task_id" "$run_exp" "$rc" "$log" <<'PY'
import glob, json, os, sys
_, bench, task_id, exp, rc, log = sys.argv
pattern = f"exp_results/v1/ours_planning_*_{int(exp):03}_*_*.json"
files = sorted(glob.glob(pattern), key=os.path.getmtime)
print(f"rc={rc}")
if not files:
    print(f"result=NO_RESULT bench={bench} task_id={task_id} exp={exp}")
    print("retry=yes"); raise SystemExit
p = files[-1]
try: data = json.load(open(p))
except Exception as e:
    print(f"result=BAD_JSON file={p} error={e}"); print("retry=yes"); raise SystemExit
recovery = data.get("recovery_events") or {}
ru = {k: (len(v) if isinstance(v, list) else v) for k, v in recovery.items()
      if k in ("pillar_up","pillar_up_smart","raise_to_height","raise_to_ore_band")}
print(f"result_file={p}")
print("task={t} success={s} status={st} steps={steps} minutes={m}".format(
    t=data.get("task"), s=data.get("success"), st=data.get("status_detailed"),
    steps=data.get("steps"), m=data.get("minutes")))
print(f"failed_waypoints={data.get('failed_waypoints')}")
print(f"pillar_recovery={ru}")
status = data.get("status_detailed"); steps = int(data.get("steps") or 0)
infra = (status == "env_step_timeout" and steps < 300) or (steps < 300 and (data.get("failed_waypoints") or []) == ["logs"])
print("retry=yes" if infra else "retry=no")
PY
    )
    printf "%s\n\n" "$decision" | tee -a "$summary"
    printf "%s" "$decision" | grep -q "retry=yes" || break
    attempt=$((attempt + 1)); cleanup_env
  done
  cleanup_env
}

run_task Iron iron 14 80614 10
run_task Gold golden 1 80201 30
run_task Gold golden 2 80202 30

printf "end=%s\n" "$(date)" | tee -a "$summary"
