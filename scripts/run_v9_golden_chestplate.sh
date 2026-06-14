#!/usr/bin/env bash
# XENON-plus V9 — minimal validation run for the option-skill scheduler.
#
# Runs Armor task 12 (Craft golden chestplate) 10 times by default.
# Outputs:
#   exp_results/v9/
#   videos/v9/

set -u
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

mkdir -p exp_results/v9 videos/v9

RUN_VERSION_LABEL="${RUN_VERSION_LABEL:-XENON-plus V9}"
LOG_PREFIX="${LOG_PREFIX:-xenon_v9}"
RUN_LABEL="${RUN_LABEL:-v9_option_skill_golden_chestplate}"
TASK_IDS_OVERRIDE="${TASK_IDS_OVERRIDE:-12}"
TRIALS="${TRIALS:-10}"
EXP_NUM_BASE="${EXP_NUM_BASE:-990000}"
RESULTS_DIR="${RESULTS_DIR:-exp_results/v9}"
VIDEO_DIR="${VIDEO_DIR:-videos/v9}"
SERVER_PORT="${SERVER_PORT:-9100}"
DECISIONER_ENABLED="${DECISIONER_ENABLED:-1}"
PERCEPTION_ACTION_SUITE="${PERCEPTION_ACTION_SUITE:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
GLOBAL_CLEANUP="${GLOBAL_CLEANUP:-1}"
STOP_ON_ABNORMAL_EXHAUSTED="${STOP_ON_ABNORMAL_EXHAUSTED:-1}"
SUMMARY_FILE="${SUMMARY_FILE:-/tmp/xenon_v9_golden_chestplate_$(date +%Y%m%d_%H%M%S)_summary.log}"

export RUN_VERSION_LABEL
export LOG_PREFIX
export RUN_LABEL
export TASK_IDS_OVERRIDE
export TRIALS
export EXP_NUM_BASE
export RESULTS_DIR
export VIDEO_DIR
export SERVER_PORT
export DECISIONER_ENABLED
export PERCEPTION_ACTION_SUITE
export SKIP_DONE
export GLOBAL_CLEANUP
export STOP_ON_ABNORMAL_EXHAUSTED
export SUMMARY_FILE

bash scripts/run_v7_armor_targeted.sh
