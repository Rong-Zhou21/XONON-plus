#!/bin/bash
# File: scripts/run_golden_all.sh
# 批量跑 Gold benchmark 全部 6 个任务（Planning 模式 + Oracle 依赖图）。
# 在容器内直接运行此脚本（需要 app.py 已在 :9000 运行）。

set -u

cd /app/repo
export HF_HOME=/app/LLM
export HF_ENDPOINT=https://hf-mirror.com
export MONITOR_URL="http://172.17.0.1:8080/push"
export MONITOR_FPS=15

SUMMARY="/tmp/golden_summary.log"
echo "=== Gold benchmark 批量运行 ===" > "$SUMMARY"
echo "开始时间: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

START_TIME=$(date +%s)

# Gold 共 6 个任务: 0..5
for TASK_ID in 0 1 2 3 4 5; do
    EXP_NUM=$((200 + TASK_ID))
    LOG="/tmp/exp_golden_t${TASK_ID}.log"

    echo "---- task=$TASK_ID exp_num=$EXP_NUM 开始 $(date +%T) ----" | tee -a "$SUMMARY"

    pkill -9 -f launchClient 2>/dev/null || true
    pkill -9 java 2>/dev/null || true
    sleep 3

    xvfb-run -a python -m optimus1.main_planning \
        server.port=9000 \
        env.times=1 \
        env.max_minutes=30 \
        benchmark=golden \
        evaluate="[${TASK_ID}]" \
        prefix="ours_planning" \
        exp_num=${EXP_NUM} \
        seed=0 \
        world_seed=${TASK_ID} \
        > "$LOG" 2>&1

    RC=$?

    RESULT_FILE=$(ls -t /app/repo/exp_results/v1/*_${EXP_NUM}_*.json 2>/dev/null | head -1)
    if [ -z "$RESULT_FILE" ]; then
        RESULT_FILE=$(ls -t /app/repo/exp_results/v1/*.json 2>/dev/null | head -1)
    fi

    if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
        STATUS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print('SUCCESS' if d.get('success') else 'FAIL', d.get('steps','?'), d.get('status_detailed','?'), d.get('task','?'))" 2>/dev/null)
    else
        STATUS="NO_RESULT_FILE rc=$RC"
    fi

    echo "  结果: $STATUS" | tee -a "$SUMMARY"
    echo "  文件: $RESULT_FILE" | tee -a "$SUMMARY"
    echo "  日志: $LOG" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"

    sleep 3
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo "=== 批量运行结束 $(date) ===" | tee -a "$SUMMARY"
echo "总用时: ${ELAPSED} 秒 ($((ELAPSED/60)) 分)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "=== Gold 结果汇总 ===" | tee -a "$SUMMARY"
ls /app/repo/exp_results/v1/*.json 2>/dev/null | while read f; do
    python3 -c "
import json
d = json.load(open('$f'))
n = d.get('exp_num', -1)
if 200 <= n <= 205:
    print(f'  exp{n} task={d.get(\"task\")} success={d.get(\"success\")} steps={d.get(\"steps\")} min={d.get(\"minutes\")}')
" 2>/dev/null
done | tee -a "$SUMMARY"
