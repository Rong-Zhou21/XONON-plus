#!/usr/bin/env bash
# XENON-plus FORMAL main experiment — full-power plus (满血).
#
#   - 67 goals x TRIALS reps (default 10), on NEW world seeds (rep 10-19),
#     disjoint from the cold-start seeds (rep 0-9) -> zero leakage for RADS.
#   - Full power: RADS decisioner ON (trained on all 10 cold-start reps) +
#     perception-action skill suite ON.
#   - Retrieval library = FULL 7035 cold-start cases, FROZEN during eval
#     (memory.is_fixed=true) so all episodes see the same fixed library.
#   - Abnormal exits (no result JSON, crash_*, env_step_timeout,
#     env_malmo_logger_error) are deleted and re-run; SKIP_DONE resumes.
#
# Environment fixes (from bring-up): conda python, host HF cache, vllm backend
# so app.py doesn't load the planner locally, CUDA_VISIBLE_DEVICES=1 (GPU0 is
# full with vllm), proxies unset (httpx chokes on socks://).
#
# Outputs: exp_results/mainexp_plus_full/  videos/mainexp_plus_full/

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

PYBIN="${PYBIN:-/home/yzb/.conda/envs/vllm_qwen2_5_vl/bin/python}"
SERVER_PORT="${SERVER_PORT:-9100}"
GPU="${GPU:-1}"
TRIALS="${TRIALS:-10}"
REP_START="${REP_START:-10}"          # NEW seeds: rep 10..19 (cold-start used 0..9)
EXP_BASE_START="${EXP_BASE_START:-3000000}"
EXP_STRIDE="${EXP_STRIDE:-100000}"
WORLD_SEED_STRIDE="${WORLD_SEED_STRIDE:-1000}"
RESULTS_ROOT="${RESULTS_ROOT:-exp_results/mainexp_plus_full}"
VIDEO_ROOT="${VIDEO_ROOT:-videos/mainexp_plus_full}"
SKIP_DONE="${SKIP_DONE:-1}"
MAX_RETRIES="${MAX_RETRIES:-10}"
RETRY_COOLDOWN="${RETRY_COOLDOWN:-15}"
APP_START_TIMEOUT="${APP_START_TIMEOUT:-180}"
DECISIONER_CKPT="${DECISIONER_CKPT:-artifacts/decisioner/rads_coldstart.pt}"
DECISIONER_MIN_P="${DECISIONER_MIN_P:-0.20}"
MASTER="${MASTER:-/tmp/mainexp_plus_$(date +%Y%m%d_%H%M%S)_master.log}"

# --- environment ---
export CUDA_VISIBLE_DEVICES="$GPU"
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/minerl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/yzb/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export QWEN_BACKEND="${QWEN_BACKEND:-vllm}"
export QWEN_VLLM_BASE_URL="${QWEN_VLLM_BASE_URL:-http://172.17.0.1:8000/v1}"
export QWEN_VLLM_MODEL="${QWEN_VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export XENON_DISABLE_STUCK_KILL="${XENON_DISABLE_STUCK_KILL:-1}"
# full power: skill suite ON
export XENON_PERCEPTION_ACTION_SUITE="${XENON_PERCEPTION_ACTION_SUITE:-1}"

mkdir -p "$RESULTS_ROOT" "$VIDEO_ROOT"
log() { echo "$@" | tee -a "$MASTER"; }

ensure_app() {
  if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${SERVER_PORT}\b"; then
    log " app_server: existing server on port $SERVER_PORT"; return 0; fi
  local applog="/tmp/mainexp_plus_app_${SERVER_PORT}_$(date +%H%M%S).log"
  log " app_server: starting app.py on $SERVER_PORT (GPU $GPU, vllm, proxy-free) log=$applog"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYBIN" -u app.py --port "$SERVER_PORT" > "$applog" 2>&1 < /dev/null &
  local t=0
  while ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${SERVER_PORT}\b"; do
    sleep 3; t=$((t+3))
    if [ "$t" -ge "$APP_START_TIMEOUT" ]; then log " app_server FAILED"; tail -8 "$applog"|tee -a "$MASTER"; return 1; fi
  done
  log " app_server: ready"
}

# benchmark TID OFF MAXMIN  (67 goals)
JOBS=(
  "wooden 0 0 3" "wooden 1 0 3" "wooden 2 0 3" "wooden 3 0 3" "wooden 4 0 3"
  "wooden 5 0 3" "wooden 6 0 3" "wooden 7 0 3" "wooden 8 0 3" "wooden 9 0 3"
  "stone 0 100 6" "stone 1 100 6" "stone 2 100 6" "stone 3 100 6" "stone 4 100 6"
  "stone 5 100 6" "stone 6 100 6" "stone 7 100 6" "stone 8 100 6"
  "iron 0 200 10" "iron 1 200 10" "iron 2 200 10" "iron 3 200 10" "iron 4 200 10"
  "iron 5 200 10" "iron 6 200 10" "iron 7 200 10" "iron 8 200 10" "iron 9 200 10"
  "iron 10 200 10" "iron 11 200 10" "iron 12 200 10" "iron 13 200 10" "iron 14 200 10" "iron 15 200 10"
  "golden 0 300 30" "golden 1 300 30" "golden 2 300 30" "golden 3 300 30" "golden 4 300 30" "golden 5 300 30"
  "diamond 0 400 30" "diamond 1 400 30" "diamond 2 400 30" "diamond 3 400 30" "diamond 4 400 30" "diamond 5 400 30" "diamond 6 400 30"
  "redstone 0 500 30" "redstone 1 500 30" "redstone 2 500 30" "redstone 3 500 30" "redstone 4 500 30" "redstone 5 500 30"
  "armor 0 600 30" "armor 1 600 30" "armor 2 600 30" "armor 3 600 30" "armor 4 600 30" "armor 5 600 30" "armor 6 600 30"
  "armor 7 600 30" "armor 8 600 30" "armor 9 600 30" "armor 10 600 30" "armor 11 600 30" "armor 12 600 30"
)

log "============================================================"
log " XENON-plus FORMAL main experiment — FULL POWER"
log " reps $REP_START..$((REP_START+TRIALS-1)) (NEW seeds) · 67 goals -> $((67*TRIALS)) runs"
log " decisioner=ON ckpt=$DECISIONER_CKPT min_p=$DECISIONER_MIN_P · suite=ON · library=FROZEN(7035)"
log " gpu=$GPU port=$SERVER_PORT results=$RESULTS_ROOT"
log " start: $(date)"
log "============================================================"
ensure_app || exit 1

for REP in $(seq "$REP_START" $((REP_START+TRIALS-1))); do
  log ""
  log "######## FORMAL PLUS REP $REP ($(date +%T)) ########"
  for JOB in "${JOBS[@]}"; do
    read -r BENCH TID OFF MAXMIN <<< "$JOB"
    EXP_NUM=$(( EXP_BASE_START + REP*EXP_STRIDE + OFF + TID ))
    WSEED=$(( REP*WORLD_SEED_STRIDE + TID ))
    EXISTING=$(ls -t "$RESULTS_ROOT"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
    if [ "$SKIP_DONE" = "1" ] && [ -n "$EXISTING" ]; then
      log "[rep$REP] $BENCH t$TID exp=$EXP_NUM SKIP"; continue; fi
    attempt=0
    while :; do
      attempt=$((attempt+1))
      pkill -9 -f launchClient 2>/dev/null || true
      LOG_FILE="/tmp/mainexp_plus_${BENCH}_t${TID}_rep${REP}_exp${EXP_NUM}_$(date +%H%M%S).log"
      log "[rep$REP] $BENCH t$TID exp=$EXP_NUM wseed=$WSEED attempt=$attempt $(date +%T)"
      xvfb-run -a "$PYBIN" -u src/optimus1/main_planning.py \
        server.port="$SERVER_PORT" env.times=1 env.max_minutes="$MAXMIN" \
        benchmark="$BENCH" evaluate="[$TID]" prefix=ours_planning \
        exp_num="$EXP_NUM" seed="$REP" world_seed="$WSEED" \
        results.path="$RESULTS_ROOT" record.video.path="$VIDEO_ROOT" \
        memory.is_fixed=true \
        memory.case_memory.decisioner.enabled=true \
        memory.case_memory.decisioner.checkpoint="$DECISIONER_CKPT" \
        memory.case_memory.decisioner.min_p_success="$DECISIONER_MIN_P" \
        > "$LOG_FILE" 2>&1
      RC=$?
      RESULT=$(ls -t "$RESULTS_ROOT"/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
      if [ -n "$RESULT" ]; then
        STDET=$("$PYBIN" -c "import json,sys;d=json.load(open(sys.argv[1]));print(str(d.get('status_detailed','')),bool(d.get('infra_early_stop')))" "$RESULT" 2>/dev/null)
        if echo "$STDET" | grep -qiE "crash_|env_step_timeout|env_malmo_logger_error|True"; then
          log "       -> ABNORMAL ($STDET) delete+retry $attempt/$MAX_RETRIES"; rm -f "$RESULT"
        else
          ST=$(grep -oE '_(success|failed)_' <<< "$(basename "$RESULT")"|tr -d _)
          log "       -> normal end ($ST, $STDET) file=$(basename "$RESULT")"; break
        fi
      else
        log "       -> ABNORMAL (no result rc=$RC) retry $attempt/$MAX_RETRIES"
      fi
      ensure_app || true
      [ "$attempt" -ge "$MAX_RETRIES" ] && { log "       -> give up after $MAX_RETRIES"; break; }
      sleep "$RETRY_COOLDOWN"
    done
  done
  log "######## REP $REP done ($(date +%T)) ########"
done

log ""
log "============================================================"
log " FORMAL PLUS complete: $(date)"
T=$(ls "$RESULTS_ROOT"/*.json 2>/dev/null|wc -l); S=$(ls "$RESULTS_ROOT"/*_success_*.json 2>/dev/null|wc -l)
log " results: total=$T success=$S"
log "============================================================"
