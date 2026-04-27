#!/bin/bash
# File: scripts/run_wooden_all.sh
# 批量跑 wooden benchmark 全部 7 个任务（Planning 模式 + Oracle 依赖图）。
# 在容器内直接运行此脚本（需要 app.py 已在 :9000 运行）。

set -u

cd /app/repo
export HF_HOME=/app/LLM
export HF_ENDPOINT=https://hf-mirror.com
export MONITOR_URL="http://172.17.0.1:8080/push"
export MONITOR_FPS=15

SUMMARY="/tmp/wooden_summary.log"
echo "=== wooden benchmark 批量运行 ===" > "$SUMMARY"
echo "开始时间: $(date)" >> "$SUMMARY"
echo "" >> "$SUMMARY"

START_TIME=$(date +%s)

# wooden 共 7 个任务: 0..6
for TASK_ID in 0 1 2 3 4 5 6; do
    EXP_NUM=$((100 + TASK_ID))
    LOG="/tmp/exp_wooden_t${TASK_ID}.log"

    echo "---- task=$TASK_ID exp_num=$EXP_NUM 开始 $(date +%T) ----" | tee -a "$SUMMARY"

    # 清理上一次残留的 Minecraft 进程
    pkill -9 -f launchClient 2>/dev/null || true
    pkill -9 java 2>/dev/null || true
    sleep 3

    xvfb-run -a python -m optimus1.main_planning \
        server.port=9000 \
        env.times=1 \
        env.max_minutes=10 \
        benchmark=wooden \
        evaluate="[${TASK_ID}]" \
        prefix="ours_planning" \
        exp_num=${EXP_NUM} \
        seed=0 \
        world_seed=${TASK_ID} \
        > "$LOG" 2>&1

    RC=$?

    # 从 exp_results 下找最新匹配的结果
    RESULT_FILE=$(ls -t /app/repo/exp_results/v1/*exp_num=${EXP_NUM}*.json 2>/dev/null | head -1)
    if [ -z "$RESULT_FILE" ]; then
        # 退而求其次：按时间找最新的 wooden 结果
        RESULT_FILE=$(ls -t /app/repo/exp_results/v1/*.json 2>/dev/null | head -1)
    fi

    if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
        STATUS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print('SUCCESS' if d.get('success') else 'FAIL', d.get('steps','?'), d.get('status_detailed','?'))" 2>/dev/null)
    else
        STATUS="NO_RESULT_FILE rc=$RC"
    fi

    echo "  结果: $STATUS" | tee -a "$SUMMARY"
    echo "  日志: $LOG" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"

    sleep 3
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo "=== 批量运行结束 $(date) ===" | tee -a "$SUMMARY"
echo "总用时: ${ELAPSED} 秒" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "汇总所有 exp_num>=100 的结果:" | tee -a "$SUMMARY"
for i in 100 101 102 103 104 105 106; do
    f=$(ls /app/repo/exp_results/v1/*exp_num=${i}*.json 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        # 可能文件名格式是 _${i:0:3}_
        :
    fi
done
# 直接用文件名里的 exp_num 字段匹配
ls /app/repo/exp_results/v1/*.json 2>/dev/null | while read f; do
    python3 -c "import json,sys; d=json.load(open('$f')); n=d.get('exp_num',-1);
print(f'  exp{n:03d} task={d.get(\"task\")} success={d.get(\"success\")} steps={d.get(\"steps\")}' if n>=100 else '', end='')" 2>/dev/null
done | grep -v '^$' | tee -a "$SUMMARY"
