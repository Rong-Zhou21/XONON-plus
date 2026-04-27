#!/bin/bash
# File: scripts/run_all_benchmarks.sh
# 按顺序跑全部 7 个 benchmark（wooden/stone/iron/golden/diamond/redstone/armor）。
# 配置保持与原论文相同（各 benchmark yaml 的 max_minutes）。
# 允许任务失败；每个任务仅跑 1 次 (times=1)。

set -u

cd /app/repo
export HF_HOME=/app/LLM
export HF_ENDPOINT=https://hf-mirror.com
export MONITOR_URL="http://172.17.0.1:8080/push"
export MONITOR_FPS=15

SUMMARY="/tmp/all_benchmarks_summary.log"
echo "=== XENON 7 Benchmarks 批量运行 ===" > "$SUMMARY"
echo "开始时间: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

BATCH_START=$(date +%s)

# 任务列表 (benchmark:task_ids:max_minutes:exp_base)
# exp_base 用于避免结果文件重名
# wooden 之前跑过 0-4 (100-104)，补 5-6
# golden 之前跑过 2,3,5 (202,203,205)，补 0,1,4
# 其他 benchmark 从 0 开始跑
declare -a JOBS=(
  "wooden:5 6:3:105"
  "stone:0 1 2 3 4 5 6:6:300"
  "iron:0 1 2 3 4 5 6:10:400"
  "golden:0 1 4:30:200"
  "diamond:0 1 2 3 4 5 6:30:500"
  "redstone:0 1 2 3 4 5:30:600"
  "armor:5:30:700"
)

for JOB in "${JOBS[@]}"; do
    IFS=':' read -r BENCH TASK_IDS MAX_MIN EXP_BASE <<< "$JOB"
    echo "" | tee -a "$SUMMARY"
    echo "======== Benchmark: $BENCH (max_min=$MAX_MIN) ========" | tee -a "$SUMMARY"

    for TASK_ID in $TASK_IDS; do
        EXP_NUM=$((EXP_BASE + TASK_ID))
        LOG="/tmp/exp_${BENCH}_t${TASK_ID}.log"

        echo "---- $BENCH task=$TASK_ID exp=$EXP_NUM 开始 $(date +%T) ----" | tee -a "$SUMMARY"

        # 清理上一次残留的 Minecraft 进程
        pkill -9 -f launchClient 2>/dev/null || true
        pkill -9 java 2>/dev/null || true
        sleep 3

        T_START=$(date +%s)
        xvfb-run -a python -m optimus1.main_planning \
            server.port=9000 \
            env.times=1 \
            env.max_minutes=$MAX_MIN \
            benchmark=$BENCH \
            evaluate="[${TASK_ID}]" \
            prefix="ours_planning" \
            exp_num=${EXP_NUM} \
            seed=0 \
            world_seed=${TASK_ID} \
            > "$LOG" 2>&1

        RC=$?
        T_ELAPSED=$(( $(date +%s) - T_START ))

        # 匹配结果文件
        RESULT_FILE=$(ls -t /app/repo/exp_results/v1/*_${EXP_NUM}_*.json 2>/dev/null | head -1)

        if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
            STATUS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print('SUCCESS' if d.get('success') else 'FAIL', 'steps=%s'%d.get('steps','?'), 'min=%s'%d.get('minutes','?'), d.get('task','?'))" 2>/dev/null || echo "PARSE_ERROR")
        else
            # 结果没落盘 → 从日志里看原因
            if grep -q "TypeError" "$LOG" 2>/dev/null; then
                STATUS="CRASH_TypeError (rc=$RC, elapsed=${T_ELAPSED}s)"
            elif grep -q "CUDA out of memory" "$LOG" 2>/dev/null; then
                STATUS="CRASH_OOM (rc=$RC)"
            elif grep -q "Connection refused" "$LOG" 2>/dev/null; then
                STATUS="SERVER_DOWN (rc=$RC)"
            else
                STATUS="NO_RESULT (rc=$RC, elapsed=${T_ELAPSED}s)"
            fi
        fi

        echo "  结果: $STATUS" | tee -a "$SUMMARY"
        echo "  文件: $RESULT_FILE" | tee -a "$SUMMARY"
        echo "  日志: $LOG" | tee -a "$SUMMARY"

        sleep 3
    done
done

BATCH_END=$(date +%s)
BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))

echo "" | tee -a "$SUMMARY"
echo "=== 全部结束 $(date) ===" | tee -a "$SUMMARY"
echo "总用时: ${BATCH_ELAPSED} 秒 ($((BATCH_ELAPSED/60)) 分)" | tee -a "$SUMMARY"

# 最终结果聚合
echo "" | tee -a "$SUMMARY"
echo "=== 全部结果 JSON 聚合 ===" | tee -a "$SUMMARY"
cd /app/repo && python3 analyze_results.py 2>/dev/null | tee -a "$SUMMARY"
