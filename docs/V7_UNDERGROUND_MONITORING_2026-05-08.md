# V7 地下探索机制监控记录

日期：2026-05-08  
实验脚本：`scripts/run_v7_armor_targeted.sh`  
实验目录：`exp_results/v7`、`videos/v7`  
汇总日志：`/tmp/xenon_v7_v7_pillar_lateral_20260508_221710_summary.log`

## 23:17 初始保存

已把当前代码快照推送到 GitHub：

- commit: `8bc4825 Save v7 underground exploration snapshot`
- remote: `origin/main`

本次提交只包含代码、脚本和分析文档，没有提交正在运行的 case memory / save_decomposed_plan 运行产物。

## 23:30 监控发现

当前运行：

- task: Armor task 12 `golden_chestplate`
- rep: 3
- exp: `371203`
- log: `/tmp/xenon_v7_v7_pillar_lateral_armor_t12_rep3_exp371203_20260508_230920.log`

前三次结果：

| exp | result | status | failed_waypoints |
|---:|---|---|---|
| 371200 | fail | `timeout_non_programmatic` | `gold_ingot`, `gold_ore` |
| 371201 | fail | `timeout_non_programmatic` | `gold_ore` |
| 371202 | fail | `timeout_non_programmatic` | `gold_ore` |

日志中地下机制的关键信号：

- rep0/rep1/rep2 都有多次 `lateral_shift` / `Mining shaft relocation`，但仍超时卡在 `gold_ore`。
- rep3 出现多次 `Detected death/respawn transition`，随后继续在 `gold_ore` 阶段向下挖。
- rep3 的一次 `lateral_shift` 被判定成功，但实际位移为 `horizontal_delta=0.2495`：
  - start `(175.069, 17.0, 284.531)`
  - end `(175.162, 17.0, 284.762)`
  - floor block cell 仍是 `(175, 284)`，没有真正进入新的水平方块格。

判断：当前 lateral 成功条件太宽松。`XENON_CORRIDOR_MIN_MOVE_DELTA=0.20` 会把 0.25 格的微小滑动当成成功，导致智能体可能仍在原竖井格内继续 `dig down`，没有真正实现“换水平位置后重新下挖”。

## 修正

修改点：

- `src/optimus1/env/wrapper.py`
  - 默认 `XENON_CORRIDOR_MIN_MOVE_DELTA` 从 `0.20` 提高到 `0.75`。
  - 新增 `block_cell_changed` 判定，记录 `start_block_cell` / `end_block_cell`。
  - lateral 成功必须满足：
    - 进入新的水平 block cell，或
    - 水平位移达到 `min_move_delta`
  - 若没有达到方块级换位，则返回 `success=False`，`reason=no_block_cell_change`。

- `src/optimus1/main_planning.py`
  - `_lateral_shift_succeeded()` 同步改为默认 `0.75`。
  - 优先读取 `block_cell_changed`，旧日志缺字段时从 start/end 坐标回算。

- `scripts/run_v7_armor_targeted.sh`
  - 默认 `XENON_CORRIDOR_MIN_MOVE_DELTA=0.75`。

预期效果：智能体不能再因为 0.2 格左右的同格滑动就继续向下挖；必须真正离开当前竖井水平格，地下采矿才会进入“换位置再 dig down”的循环。

## 23:36 重启实验

修复已提交并推送：

- commit: `068be48 Tighten v7 lateral relocation success`
- remote: `origin/main`

旧的 `v7_pillar_lateral` 进程已停止，保留已有 `3712xx` 结果作为旧逻辑对照。

新实验使用 `setsid` 后台启动：

```text
RUN_LABEL=v7_blockcell_lateral
EXP_NUM_BASE=372000
SKIP_DONE=0
SUMMARY_FILE=/tmp/xenon_v7_v7_blockcell_lateral_20260508_233643_summary.log
```

运行状态：

- runner pid: `811352`
- current first task: Armor task 12 `golden_chestplate`
- current first exp: `373200`
- current first log: `/tmp/xenon_v7_v7_blockcell_lateral_armor_t12_rep0_exp373200_20260508_233646.log`

注意：脚本里的实际 exp 号计算方式是：

```text
EXP_NUM = EXP_NUM_BASE + TASK_ID * 100 + REP
```

所以 `EXP_NUM_BASE=372000` 对 Armor task 12 rep0 生成 `373200`。
