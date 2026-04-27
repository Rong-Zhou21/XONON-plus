#!/bin/bash
# File: scripts/watchdog.sh
# 监控 main_planning 进程，超过 MAX_SECONDS 自动 kill。
# 默认 20 分钟 = 1200s。

set -u
MAX_SECONDS=${1:-1200}

while true; do
    # 找 main_planning 进程，按启动时间排序
    PID=$(pgrep -f 'optimus1.main_planning' -o 2>/dev/null | head -1)
    if [ -z "$PID" ]; then
        sleep 30
        continue
    fi
    # 已经运行秒数
    ELAPSED=$(ps -o etimes= -p $PID 2>/dev/null | tr -d ' ')
    if [ -z "$ELAPSED" ]; then
        sleep 30
        continue
    fi
    if [ $ELAPSED -gt $MAX_SECONDS ]; then
        # 读命令行确认任务标识
        CMD=$(ps -o cmd= -p $PID 2>/dev/null | head -c 200)
        echo "[watchdog $(date +%T)] kill PID=$PID elapsed=${ELAPSED}s (max=${MAX_SECONDS}s) cmd=$CMD" >> /tmp/watchdog.log
        kill -9 $PID 2>/dev/null
        pkill -9 -f launchClient 2>/dev/null
        pkill -9 java 2>/dev/null
    fi
    sleep 30
done
