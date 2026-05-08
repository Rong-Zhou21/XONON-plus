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

## 23:42 二次监控发现

新实验进入地下后，`XENON_CORRIDOR_MIN_MOVE_DELTA=0.75` 已生效，但日志暴露出另一个判定问题：

- `dig_forward_blocks()` 的清障阶段实际把智能体水平推开了约 2 格：
  - 示例：`start=(-686.7,22.0,-186.3)` -> `end=(-686.7,22.0,-184.3)`
  - `horizontal_delta=2.00`
- 但函数仍返回：
  - `blocks_dug=0/1`
  - `reason=stuck_no_forward_displacement`
  - `success=False`
- 上层因此打印：
  - `horizontal displacement failed; NOT restoring dig-down prompt yet`

判断：这次已经不是“移动不足”，而是 success 语义仍绑在 `blocks_dug` 上。对 v7 目标来说，只要最终 x/z 已经换到新的水平方块格，就应视为 relocation 成功，允许恢复原始 `dig down`。

修正：

- `src/optimus1/env/wrapper.py`
  - `success = provisional_success or (not height_drop and meaningful_final_move)`
  - 当最终坐标已换格或达到阈值时，即使 `blocks_dug=0`，也返回 `success=True`。
  - 将这种情况的 `reason` 统一为 `moved_continue_dig_down`。

修复已提交并推送：

- commit: `b44a463 Treat physical lateral relocation as success`

旧的 `v7_blockcell_lateral` 进程已停止，保留 `373200` 日志作为二次修正依据。

## 23:45 再次重启实验

新实验使用最终 success 语义后台启动：

```text
RUN_LABEL=v7_physical_lateral
EXP_NUM_BASE=374000
SKIP_DONE=0
SUMMARY_FILE=/tmp/xenon_v7_v7_physical_lateral_20260508_234531_summary.log
```

运行状态：

- runner pid: `812318`
- current first task: Armor task 12 `golden_chestplate`
- current first exp: `375200`
- current first log: `/tmp/xenon_v7_v7_physical_lateral_armor_t12_rep0_exp375200_20260508_234534.log`

## 23:48 验证第一轮地下 relocation

`v7_physical_lateral` 的第一条 run 已触发地下 relocation，关键日志：

```text
done: blocks_dug=0/1 steps_used=144 reason=moved_continue_dig_down
lateral_shift: attempt=1/3 ... horizontal_delta=2.00
Mining shaft relocation: horizontal displacement succeeded; restoring STEVE-1 prompt as dig down and mine gold_ore
start_block_cell=(-669, -189)
end_block_cell=(-669, -187)
block_cell_changed=True
```

判断：

- 修正后的 success 语义生效。
- 即使 `blocks_dug=0`，只要清障阶段已经把智能体推入新的水平 block cell，上层就会恢复 `dig down`。
- 这符合当前地下探索目标：先实现水平位置变化，再继续向下挖。

## 2026-05-09 用户反馈后的第三次修正

用户观察到：智能体并没有真正正对方块挖掘，当前单一 yaw 方向约束仍不够好。新的要求：

- 到达目标高度后，先挖一个前向两格高通道。
- 视角先水平向前挖，再斜下方挖。
- 如果按住前进不能移动，朝斜下方持续攻击。
- 同时尝试当前方向、左 30 度、右 30 度，避免单一朝向卡死。
- 恢复 `dig down` 的硬前置条件改为：水平移动至少 1 个方块距离。

代码修改：

- `src/optimus1/env/wrapper.py`
  - 默认 `XENON_CORRIDOR_YAW_MODE=fan30`。
  - 新增 `XENON_CORRIDOR_YAW_OFFSETS=0,30,-30`，按当前方向、左/右 30 度依次尝试。
  - 默认 `XENON_CORRIDOR_MIN_MOVE_DELTA=1.0`。
  - `meaningful_lateral_move` 不再用“换到相邻 block cell”兜底，必须达到水平距离阈值。
  - 默认 `XENON_CORRIDOR_BLOCKED_UP_BUDGET=0`，清障顺序变为水平前方 -> 斜下方。

- `src/optimus1/main_planning.py`
  - `_lateral_shift_succeeded()` 同步要求 `horizontal_delta >= 1.0`。

- `scripts/run_v7_armor_targeted.sh`
  - 同步默认值，并在 summary 中打印 `yaw_offsets`。

修复已提交并推送：

- commit: `db50c5d Add fan yaw corridor relocation`
- remote: `origin/main`

## 2026-05-09 01:05 fan30 重启与在线验证

旧的 v7 运行进程已停止，随后清理并重启当前 fan30 版本：

```text
RUN_LABEL=v7_fan30_corridor
EXP_NUM_BASE=376000
SKIP_DONE=0
SUMMARY_FILE=/tmp/xenon_v7_v7_fan30_corridor_20260509_010545_summary.log
```

运行状态：

- runner pid: `942968`
- current first task: Armor task 12 `golden_chestplate`
- current first exp: `377200`
- current first log: `/tmp/xenon_v7_v7_fan30_corridor_armor_t12_rep0_exp377200_20260509_010548.log`

summary 确认当前配置：

```text
yaw_mode               : fan30
yaw_offsets            : 0,30,-30
blocked_front_pitch    : 0.0
blocked_up_budget      : 0
blocked_feet_pitch     : 55.0
min_forward_delta      : 1.0
```

已验证的关键事件：

- 01:09-01:11，连续多次地下 relocation 先触发 `yaw_offset_+0 has not moved one block`，随后执行 `clearing horizontal front then diagonal-down blockers`，最后 `horizontal_delta >= 1.00` 后才恢复 `dig down`。
- 01:13，出现 0 度方向无法打通的场景；第 3 次重试使用 `yaw_offset_+30` 成功，`horizontal_delta=1.72`，随后上层打印 `Mining shaft relocation: horizontal displacement succeeded`。
- 01:15，出现一次失败横移 `horizontal_delta=0.07`，没有恢复 `dig down`；下一次达到 `horizontal_delta=1.05` 后才恢复。这验证了“至少移动 1 个方块距离”是硬前置条件。
- 01:14 起已采到 `iron_ore: 1`，说明换位后能够继续在新竖井采样；当前瓶颈是 iron 资源覆盖率，还不是“未换位就继续下挖”的逻辑错误。

当前判断：

- fan30 分支有效，`+30` 分支已经在真实地下堵塞中被触发并成功。
- 水平前方 -> 斜下方的两格高通道清理顺序有效。
- 恢复 `dig down` 前的 1 格水平位移门槛有效。
- 需要继续观察第一轮是否能凑够 iron 并进入 gold；如果长时间只在 `iron_ore=1` 附近循环，后续要评估“首次目标矿高度过低时是否需要回退到矿层中点”的策略，但这会改变当前用户要求保留的 first-target-ore-height 逻辑。

## 2026-05-09 01:41 第四次修正：提前底层卡住判定

在线监控前两轮结果：

| exp | result | status | 关键现象 |
|---:|---|---|---|
| 377200 | fail | `timeout_non_programmatic` | iron 阶段通过多次 fan30 relocation 推进到 `iron_ore=3`，但进入 gold 太晚；gold relevel 因 `no_placement_succeeded` 失败后没有恢复 `dig down`，这是安全行为。 |
| 377201 | fail | `timeout_non_programmatic` | 早期表现更好，`iron_ore` 在 y=30 附近快速完成，gold 阶段记录 `first-target-ore Y=20` 并获得 `gold_ore=3`；随后多次向下挖到 redstone/diamond 层和 bedrock，铁镐耗尽，被迫回到 `stone_pickaxe -> iron_ore -> iron_pickaxe` 链条。 |

判断：

- fan30 横移逻辑本身稳定，日志中多次 `horizontal_delta >= 1.0` 后才恢复 `dig down`。
- 当前主要损耗来自 `bedrock_stuck` 判定太慢：默认 `XENON_BEDROCK_STAGNANT_TICKS=1200`，每个竖井会在低层等待很久才抬升，增加时间和工具耐久消耗。
- 这次不恢复 `XENON_OVERSHOOT_ENABLE_Y_TRIGGER`，也不加入“低于目标高度就抬升”的纯高度触发，避免回到用户指出的旧问题。

修正：

- `src/optimus1/main_planning.py`
  - `XENON_BEDROCK_STAGNANT_TICKS` 默认值从 `1200` 降到 `600`。
- `scripts/run_v7_armor_targeted.sh`
  - 显式导出 `XENON_BEDROCK_STAGNANT_TICKS=600`。
  - summary 打印 `bedrock_stagnant_ticks`，便于后续日志核对。

预期效果：

- 仍然只在 deeper ore / bedrock-stuck 这类允许触发条件下抬升。
- 但底层卡住会更快被识别，减少在 y=2-5 附近的空耗和铁镐磨损。

## 2026-05-09 02:03 seed1 重启监控

`v7_bedrock600_fan30` 第一轮 seed0 在地表阶段卡在 `oak_log=7/8`，未进入地下机制验证。为避免把时间耗在与本次目标无关的地表砍树随机失败上，停止该运行、清空 `exp_results/v7` 和 `videos/v7`，并用 `SEED_BASE=1` 重启：

```text
RUN_LABEL=v7_bedrock600_fan30_seed1
EXP_NUM_BASE=380000
SEED_BASE=1
SKIP_DONE=0
SUMMARY_FILE=/tmp/xenon_v7_v7_bedrock600_fan30_seed1_20260509_020331_summary.log
```

运行状态：

- runner pid: `993915`
- current first task: Armor task 12 `golden_chestplate`
- current first exp: `381200`
- current first seed: `1`
- current first log: `/tmp/xenon_v7_v7_bedrock600_fan30_seed1_armor_t12_rep0_exp381200_20260509_020334.log`

已观察到的地下行为：

- 02:07 左右已进入 `gold_ore` 阶段，并持有 `iron_pickaxe`。
- 已获得 `gold_ore=5`、`diamond=2`、`iron_ore=8`，说明地下竖井采样正在推进。
- 多次 `Mining shaft relocation` 满足 `horizontal_delta >= 1.0` 后恢复 `dig down`。
- 一次横移虽然 `horizontal_delta=1.0005`，但 `height_drop=True`，上层没有恢复 `dig down`，随后继续重试；这说明防止“掉回下方竖井”的保护仍有效，但后续如果它导致明显低效，可以考虑把 `XENON_LATERAL_MAX_Y_DROP` 从 `0.75` 放宽到 `1.25`。
- `bedrock_stuck` 的日志仍可能出现 `no_activity_ticks≈1000`，因为实际触发还受采到 deeper ore、switch cooldown 和最近活动时间影响；但默认阈值已经在 summary 中确认是 `600`。

当前判断：第四次修正没有破坏 fan30 机制；地下循环仍按“目标高度/底层卡住 -> 横移一格 -> 再向下挖”执行。当前瓶颈主要是 gold 资源稀疏和长时间采矿导致的工具耐久压力，暂时不再增加新硬逻辑。

## 2026-05-09 02:20 seed1 第一轮结果

`v7_bedrock600_fan30_seed1` 第一轮：

- exp: `381200`
- task: Armor task 12 `golden_chestplate`
- seed/world_seed: `1`
- result: `FAIL`
- status: `timeout_non_programmatic`
- steps: `24492`
- minutes: `20.41`
- 最终主要卡点：`gold_ore=6/8`

观察：

- 相比前一版 seed1 的 `gold_ore=3/8` 后铁镐耗尽，这次推进到 `gold_ore=6/8`，说明更早的 bedrock-stuck 触发有一定收益。
- 日志中出现 `no_activity_ticks=668` 的 bedrock-stuck relocation，比之前常见的 1200+ 更早。
- 仍然有大量竖井采样没有命中 gold，说明当前机制虽然可行，但采样覆盖效率仍不足。
- 当前没有发现“未完成 1 格水平位移就恢复 dig down”的回归；失败主要是资源覆盖率/时间预算问题。

下一步观察重点：第二轮 seed2 是否能在 gold 阶段更稳定地超过 6/8；如果仍反复卡在 5-6 个 gold，需要考虑提高水平换位间隔或引入更轻量的目标高度局部扫描，而不是继续只靠相邻竖井抽样。
