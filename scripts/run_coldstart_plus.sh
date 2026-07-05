#!/usr/bin/env bash
# XENON-plus cold-start case-collection.
#
# Goal: accumulate the initial case library from scratch (empty cases.json).
#   - 67 goals (the 7-group benchmark subset enumerated in run_v3_full_benchmark.sh)
#   - TRIALS repetitions per goal (default 10), each with a distinct world seed
#   - Decisioner OFF (no RADS ckpt yet; planner-only behaviour policy). Cases are
#     still recorded by the case memory, which is what we want for Path A.
#   - Skill library (perception-action suite) ON  -> training distribution matches
#     plus deployment.
#   - Ore environment = plus current mechanism (code defaults: (x,y,z) cell de-dup,
#     gold x1.0, dig-down spawn halved after the pillar-up skill triggers).
#   - Abnormal exits are deleted and re-run (handled inside run_v3_full_benchmark.sh
#     via MAX_RETRIES_ON_CRASH); SKIP_DONE makes the whole sweep resumable.
#
# Outputs:
#   exp_results/coldstart_plus/
#   videos/coldstart_plus/

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

TRIALS="${TRIALS:-10}"
SERVER_PORT="${SERVER_PORT:-9100}"
GPU="${GPU:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-exp_results/coldstart_plus}"
VIDEO_ROOT="${VIDEO_ROOT:-videos/coldstart_plus}"
# Per-rep exp-number stride; max OFF+TID is 600+12=612 < 100000, so reps never collide.
EXP_STRIDE="${EXP_STRIDE:-100000}"
EXP_BASE_START="${EXP_BASE_START:-2000000}"
# Per-rep world-seed stride; max TID is 15, so 1000 keeps worlds distinct per rep.
WORLD_SEED_STRIDE="${WORLD_SEED_STRIDE:-1000}"

MASTER_SUMMARY="${MASTER_SUMMARY:-/tmp/coldstart_plus_$(date +%Y%m%d_%H%M%S)_master.log}"

echo "============================================================" | tee -a "$MASTER_SUMMARY"
echo " XENON-plus COLD-START case collection" | tee -a "$MASTER_SUMMARY"
echo " trials/goal   : $TRIALS   (67 goals -> $((67 * TRIALS)) runs)" | tee -a "$MASTER_SUMMARY"
echo " decisioner    : OFF (cold start, planner-only; cases still recorded)" | tee -a "$MASTER_SUMMARY"
echo " skill suite   : ON" | tee -a "$MASTER_SUMMARY"
echo " ore env       : plus current (cell de-dup + gold 1.0 + halve-after-pillar)" | tee -a "$MASTER_SUMMARY"
echo " results root  : $RESULTS_ROOT" | tee -a "$MASTER_SUMMARY"
echo " video root    : $VIDEO_ROOT" | tee -a "$MASTER_SUMMARY"
echo " master log    : $MASTER_SUMMARY" | tee -a "$MASTER_SUMMARY"
echo " start         : $(date)" | tee -a "$MASTER_SUMMARY"
echo "============================================================" | tee -a "$MASTER_SUMMARY"

for REP in $(seq 0 $((TRIALS - 1))); do
  echo "" | tee -a "$MASTER_SUMMARY"
  echo "######## COLD-START REP $REP / $((TRIALS - 1)) ($(date +%T)) ########" | tee -a "$MASTER_SUMMARY"

  DECISIONER_ENABLED=0 \
  PERCEPTION_ACTION_SUITE=1 \
  SKIP_DONE=1 \
  START_APP_SERVER=1 \
  SERVER_PORT="$SERVER_PORT" \
  GPU="$GPU" \
  SEED="$REP" \
  WORLD_SEED_BASE="$((REP * WORLD_SEED_STRIDE))" \
  EXP_NUM_BASE="$((EXP_BASE_START + REP * EXP_STRIDE))" \
  RUN_LABEL="coldstart_plus_rep${REP}" \
  RESULTS_DIR="$RESULTS_ROOT" \
  VIDEO_DIR="$VIDEO_ROOT" \
    bash scripts/run_v3_full_benchmark.sh 2>&1 | tee -a "$MASTER_SUMMARY"

  echo "######## REP $REP done ($(date +%T)) ########" | tee -a "$MASTER_SUMMARY"
done

echo "" | tee -a "$MASTER_SUMMARY"
echo "============================================================" | tee -a "$MASTER_SUMMARY"
echo " COLD-START complete: $(date)" | tee -a "$MASTER_SUMMARY"
CASES=$(python3 -c "import json;print(len(json.load(open('src/optimus1/memories/ours_planning/v1/case_memory/cases.json')).get('cases',[])))" 2>/dev/null || echo "?")
echo " case library size: $CASES cases" | tee -a "$MASTER_SUMMARY"
echo " results: $RESULTS_ROOT  videos: $VIDEO_ROOT" | tee -a "$MASTER_SUMMARY"
echo "============================================================" | tee -a "$MASTER_SUMMARY"
