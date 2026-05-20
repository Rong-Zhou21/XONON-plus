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

## 2026-05-09 12:05 水平通道循环版监控

本轮根据“先保证水平层探索能力”的目标，停止旧 v7 后重新清理并启动：

```text
RUN_LABEL=v7_horizontal_loop_guarded
EXP_NUM_BASE=411000
SEED_BASE=1
TRIALS=10
TASK_ORDER=golden_chestplate -> golden_leggings -> diamond_chestplate
SUMMARY_FILE=/tmp/xenon_v7_v7_horizontal_loop_guarded_20260509_114142_summary.log
```

核心机制：

- `dig_forward_blocks()` 从“一次横移”升级为 2 格高水平通道原语：逐段清理正前方和脚下前方阻挡，使用 `0,+30,-30` yaw fan，且每段要求实际水平推进。
- `dig_down_blocks()` 只在水平通道成功后执行；也就是恢复向下挖之前必须先完成水平位置变化。
- 回到目标矿首次高度时，不再做开放式水平探索，而是循环执行：目标层回位 -> 挖 2 格高通道 -> 在新 x/z 下挖 -> 目标未满足则再回目标层。
- 回层判定增加上界保护：到达目标高度过高时不再误判成功，必要时使用 command relevel fallback 拉回目标层。

当前在线结果：

| idx | exp | task | seed | result | steps | 观察 |
|---:|---:|---|---:|---|---:|---|
| 1 | 412200 | golden_chestplate | 1 | success | 9302 | gold 阶段通过多轮水平通道和 scripted digdown 从 `gold_ore=3` 推到 8+，最终合成成功。 |
| 2 | 412201 | golden_chestplate | 2 | success | 8352 | 出现两次 `scripted_digdown blocks=1/5`，但下一轮回层和通道恢复成功，最终 `gold_ore=9`。 |
| 3 | 412202 | golden_chestplate | 3 | success | 7490 | 多次 `horizontal_tunnel blocks_dug=8/8` 后稳定下挖，gold 阶段最终到 `gold_ore=8` 并完成合成。 |

已验证：

- “水平移动至少约 1 格后才继续下挖”的硬前置没有回归。
- 2 格高通道在真实地下日志中可以完成 6/8、8/8 等有效段数，能把智能体带到新的水平位置。
- `post-tunnel relevel` 与 command fallback 能把掉层或上浮后的智能体拉回目标高度附近，避免在地表误判为回层成功。

当前不足：

- `stuck_no_progress` 仍偏多，主要发生在通道后回层整理或竖井局部清障阶段；目前会增加耗时，但前三轮没有造成失败。
- `scripted_digdown` 偶发只完成 1/5，说明垂直下挖在部分脚下结构上仍不够鲁棒；当前外层循环能够恢复，但后续如果在金护腿或钻石胸甲中放大，需要加入更明确的脚下清障/姿态复位。
- 本轮仍开启 `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=1`，用于压测“水平换位 -> 下挖 -> 再回层”动作链。该配置能证明动作机制可闭环，但不能单独代表真实矿石识别能力；后续需要单独关闭强制目标矿做真实性压力测试。

下一步监控重点：

- 第 4-10 次 golden_chestplate 是否保持成功，特别看 `stuck_no_progress` 是否出现连续卡死。
- 切到 golden_leggings 后，目标 gold 数量变化是否暴露新的耗时问题。
- diamond_chestplate 阶段要重点看低层回层和 bedrock-stuck 触发是否仍能避免在 y=2-5 附近空耗。

## 2026-05-09 12:09 第四轮中断与回层保护修正

`v7_horizontal_loop_guarded` 第 4 次 golden_chestplate 暴露出重复问题，因此已中断并清空 `exp_results/v7`、`videos/v7`。

失败模式：

- 水平通道多次完成 `blocks_dug=8/8`，说明 2 格高通道本身可用。
- 通道后从目标层 `dest_y=21` 掉到 `post_end_y=18`。
- `raise_to_height()` 因 `no_placement_succeeded` 失败后，旧逻辑把 y=18 软接受为有效目标层。
- 随后 `dig_down_blocks()` 按 y=18 强制放置 gold，默认 `dy=-3` 导致目标矿落到 y=15，低于 gold 有效层。
- 下挖阶段出现连续 `blocks=0/5` 或 `1/5`，`gold_ore` 长时间停在 4。

修正：

- `src/optimus1/main_planning.py`
  - pre-tunnel 和 post-tunnel 如果低于目标层，先尝试 command relevel fallback。
  - 默认关闭低位软接受：`XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT=0`。
  - 只有确实回到 `dest_y ± tolerance` 后，才继续 scripted digdown。
- `src/optimus1/env/wrapper.py`
  - `dig_down_blocks()` 的强制放矿 dy 候选从固定 `-3/-4/-5` 扩展为 `-3,-4,-5,-2,-1`。
  - 如果当前 y 已经偏低，会选择仍处在矿物 band 内的 dy，避免把 gold 放到 y=15 这类无效高度。
- `scripts/run_v7_armor_targeted.sh`
  - summary 记录 `tunnel_low_soft_accept` 和 `scripted_force_dys`。

验证：

- `py_compile` 通过。
- `bash -n scripts/run_v7_armor_targeted.sh` 通过。
- `scripts/verify_horizontal_tunnel.py` 通过：4/4 tunnel segments，水平位移约 4.2，y 保持 45。

重启：

```text
RUN_LABEL=v7_horizontal_relevel_guarded
EXP_NUM_BASE=413000
SEED_BASE=1
TRIALS=10
SUMMARY_FILE=/tmp/xenon_v7_v7_horizontal_relevel_guarded_20260509_120954_summary.log
```

下一步重点观察：第 1 次 golden_chestplate 是否还会在 `post_end_y < dest_y` 后直接 scripted digdown；预期日志应出现 `command relevel fallback source_reason=post_tunnel_below_target`，然后再 `forced target ore`。

## 2026-05-09 12:24 首次目标矿高度记录修正

`v7_horizontal_relevel_guarded` 第 2 次 golden_chestplate 暴露出新的高度记录问题，已中断并再次清空 v7 输出。

现象：

- gold 阶段中一次通道从 `dest_y=21` 掉到 `end_y=18`，新加入的 `post_tunnel_below_target` command fallback 生效，成功回到 y=21。
- 随后 `forced target ore` 正确放在 `target_y=18`，`scripted_digdown` 也完成 5/5，并获得 `gold_ore=1`。
- 但主循环在 inventory delta 后用“玩家当前站立高度”记录 `first-target-ore Y`，此时玩家已经下落到 y=15，于是把 gold 的首次目标层错误记成 y=15。
- 后续循环围绕 y=15 开水平通道，低于 gold 有效层，导致死亡/重生和无效采样。

修正：

- `src/optimus1/main_planning.py`
  - 增加 `_clamp_ore_y_to_band()`，避免首次矿物高度落到 canonical ore band 外。
  - 在 scripted digdown 返回后保存 `forced_ore.target_y`。
  - `first_target_ore_y` 首次记录时优先使用 `scripted_forced_ore_target`，没有该信息时才用玩家当前位置，并进行 band clamp。
  - 新日志会打印 `source=` 和 `observed_y=`，方便判断记录来自矿物高度还是玩家高度。

重启：

```text
RUN_LABEL=v7_horizontal_oreheight_guarded
EXP_NUM_BASE=415000
SEED_BASE=1
TRIALS=10
SUMMARY_FILE=/tmp/xenon_v7_v7_horizontal_oreheight_guarded_20260509_122419_summary.log
```

下一步重点观察：gold 的 `recording first-target-ore Y` 不能再出现 y=15，预期应记录 `source=scripted_forced_ore_target` 且 y 在 gold band `[17,26]` 内。

## 2026-05-09 12:31 水平层探索闭环首轮结果

`v7_horizontal_oreheight_guarded` 第 1 次 `golden_chestplate` 已成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=0 exp=416200 seed=1`
- 状态：`SUCCESS`
- 步数：9469
- 耗时：约 7.89 分钟

关键观察：

- gold 阶段首次目标矿高度记录为 `y=21.0`，后续最后一个 gold 记录为 `y=18.0`，均在 gold 有效高度带内；此前 y=15 错误底线未复现。
- 水平通道机制能完成闭环：多次出现 `horizontal_tunnel blocks_dug=7/8`、`8/8`，随后 `scripted_digdown` 获得 gold，最终凑齐 `gold_ore=8`。
- `post_tunnel_below_target` 后的 command relevel fallback 生效，通道掉层后能回到目标层附近再继续流程。

不足：

- 通道推进仍有不稳定：出现过 `blocks_dug=0/8`、`2/8 reason=height_drop`，说明遇到自然空洞或脚下被打开时会过早掉层。
- 一些 scripted digdown 仍会 `blocks=1/5 reason=stuck_no_vertical_displacement`，但目前主循环能通过下一轮水平换位恢复，不再形成死循环。

当前判断：

- “回到目标高度 -> 挖 2 格高水平通道 -> 水平坐标变化 -> 再 dig down”的核心机制可行。
- 下一轮重点监控通道失败率和掉层频率；如果连续出现 `blocks_dug < 3` 或长时间不增加目标矿，应优先增强通道阶段的地面保护/前方清障，而不是放宽 dig down 前置条件。

## 2026-05-09 12:35 第二轮中断：死亡/位置跳变污染验证

`v7_horizontal_oreheight_guarded` 第 2 次 `golden_chestplate` 中断并清空重跑。

中断原因：

- gold 阶段已进入水平探索，并获得约 `gold_ore=5`。
- 一次水平通道从 `start=(352.3,21.0,-167.3)` 推进 3 段后触发 `Detected death/respawn transition`。
- `dig_forward_blocks()` 返回 `reason=terminal_or_position_jump`，`end=(354.5,63.0,-180.5)`，说明智能体不再处于连续地下通道状态。
- 旧外层逻辑虽然没有把该 tunnel 当作成功，但同一次 relevel 中继续从重生/异常位置 retry，并用 command relevel 拉回 y=21；这会污染“水平通道是否真实鲁棒”的验证。

修正：

- `src/optimus1/main_planning.py`
  - 新增 `XENON_TUNNEL_ABORT_ON_TERMINAL=1` 默认保护。
  - 如果 `dig_forward_blocks()` 返回 `terminal_abort` 或 `position_jump_abort`，本次 `_maybe_relevel_for_overshoot()` 立即停止 lateral retry，不从重生点或不连续位置继续开通道。
- `scripts/run_v7_armor_targeted.sh`
  - 将 `XENON_LATERAL_MAX_Y_DROP` 从 `3.0` 收紧到 `1.25`，更严格地保持同层水平探索。
  - 将 `XENON_TUNNEL_RELEVEL_AFTER_DROP` 从 `1.25` 收紧到 `0.75`，小幅掉层就先回层整理。
  - summary 增加 `tunnel_abort_terminal`，便于确认该保护是否启用。

预期：

- 轻微地形起伏仍可通过回层继续。
- 掉入大空洞、死亡/重生或大位置跳变不再被同一次 horizontal tunnel retry 吸收；这类问题会以失败/重试暴露出来，便于继续改通道自身，而不是依赖异常复位。

## 2026-05-09 12:47 严格水平层验证：关闭掉层软接受

`v7_horizontal_strict_layer` 第 1 次 `golden_chestplate` 在 gold 阶段中断并清空重跑。

现象：

- gold 首次目标高度记录正确：`for gold_ore: y=20.0`，没有回到 y=15 问题。
- 新的终端/位置跳变保护未出现回归。
- 但水平通道仍出现 `done: blocks_dug=6/8 reason=height_drop` 后被 `_soft_accept_partial_tunnel()` 改成 `height_drop_soft_accepted`，随后继续 `forced target ore` 和 `scripted_digdown`。

问题判断：

- 这个分支虽然能提升任务成功率，但它允许“已经掉层的部分通道”被视为横移成功。
- 这会削弱当前实验目标：验证智能体能否在目标高度附近真正挖出稳定 2 格高水平通道。

修正：

- `src/optimus1/main_planning.py`
  - `_soft_accept_partial_tunnel()` 新增 `XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT` 开关。
  - 默认值为 `0`，即掉层后的部分通道不再自动软接受。
- `scripts/run_v7_armor_targeted.sh`
  - 导出 `XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT=0`。
  - summary 增加 `tunnel_drop_soft_accept`，便于确认配置。

预期：

- 只有未掉层且满足 `min_success_blocks` 的水平通道才会触发后续 scripted digdown。
- 掉层通道会失败并重试/回层，降低短期成功率，但更符合“水平层探索能力验证”。

## 2026-05-09 12:53 严格无掉层软接受首轮成功

`v7_horizontal_strict_no_drop_soft` 第 1 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=0 exp=420200 seed=1`
- 状态：`SUCCESS`
- 步数：8189
- 耗时：约 6.82 分钟

关键观察：

- 配置确认：`tunnel_drop_soft_accept=0`、`tunnel_abort_terminal=1`、`lateral_max_y_drop=1.25`。
- gold 首次目标高度记录为 `y=21.0`，在 gold 有效高度带内。
- 日志未出现 `height_drop_soft_accepted`。
- 多次 `blocks_dug=1/8 reason=height_drop` 后没有进入 scripted digdown，而是继续回层/重试。
- 后续多次完成 `blocks_dug=8/8 reason=tunnel_bore_complete`，且 `end_y=21.0`，随后才执行 `forced target ore` 和 `scripted_digdown`。
- 最终 `gold_ore=8` 后进入熔炼并完成合成。

当前判断：

- 严格版水平层探索闭环成立：掉层失败不会被算作成功，真正 8 格水平通道可以把智能体带到新 x/z，并支撑后续下挖采集。
- 仍需继续观察 seed2+：如果自然空洞频繁导致 `height_drop` 过多，后续应优先补“通道地面保护/填坑”机制，而不是恢复掉层软接受。

## 2026-05-09 12:57 严格版第二轮成功

`v7_horizontal_strict_no_drop_soft` 第 2 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=1 exp=420201 seed=2`
- 状态：`SUCCESS`
- 步数：7257
- 耗时：约 6.05 分钟

关键观察：

- gold 首次目标高度记录为 `y=22.0`，在 gold 有效高度带内。
- 本轮进入 gold 时已经通过自然竖井获得 `gold_ore=4`。
- 后续水平探索中，`1/8 reason=height_drop` 没有被接受；随后完成 `8/8` 水平通道再执行 scripted digdown。
- gold 从 4 增至 8，完成熔炼与合成。
- 未出现 `height_drop_soft_accepted`，未出现 `terminal_or_position_jump` 后继续同轮 retry。

阶段性结论：

- 严格无掉层软接受版本已连续 2 次完成 `golden_chestplate`。
- 当前主要成本来自“遇到空洞后 1/8 height_drop -> 回层重试”，但这比错误接受掉层更可解释；后续优化方向应是通道地面保护，而不是放宽成功条件。

## 2026-05-09 13:02 严格版第三轮成功

`v7_horizontal_strict_no_drop_soft` 第 3 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=2 exp=420202 seed=3`
- 状态：`SUCCESS`
- 步数：7060
- 耗时：约 5.88 分钟

关键观察：

- gold 首次目标高度记录为 `y=20.0`。
- 日志中没有 `height_drop_soft_accepted`。
- 出现多次 `blocks_dug=1/8`、`2/8 reason=height_drop`，均未触发 scripted digdown。
- 后续完成多次 `blocks_dug=8/8` 水平通道，gold 从 2 提升到 8。
- 有一次通道在 y=19-20 附近推进，属于 1 格内层面波动；当前阈值 `XENON_LATERAL_MAX_Y_DROP=1.25` 允许这种小幅变化。

阶段性结论：

- 严格版已连续 3/3 完成 `golden_chestplate`。
- 当前机制已经验证“必须完成有效水平通道才能恢复下挖”的核心要求。
- 下一步继续观察 seed4+；如果 1 格内层面波动仍然导致采样偏离，可以再把 `XENON_LATERAL_MAX_Y_DROP` 从 1.25 收紧到 0.75，但目前不急于调整。

## 2026-05-09 13:12 严格版第四轮复杂地形成功

`v7_horizontal_strict_no_drop_soft` 第 4 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=3 exp=420203 seed=4`
- 状态：`SUCCESS`
- 步数：10407
- 耗时：约 8.67 分钟

关键观察：

- gold 首次目标高度记录为 `y=21.0`，位于 gold 有效高度带。
- 该轮地形更复杂，多次水平通道因 `height_drop` 被拒绝，没有被软接受为成功。
- 有一次完整 `blocks_dug=8/8` 后 scripted digdown 未形成有效下挖，但后续重新回层/重试仍能完成新的 8 格通道。
- 后续多次 `8/8` 水平通道加 scripted digdown 将 gold 从 4 补到 8，再完成熔炼与金胸甲合成。

阶段性结论：

- 严格版已连续 4/4 完成 `golden_chestplate`。
- 当前机制在复杂地下地形中仍能坚持“有效水平位移后才继续下挖”，没有回到原地 dig down 或掉层误判。
- 代价是耗时增加；后续优化重点应放在 `height_drop` 被拒绝后的重新找水平通道效率，以及必要时加入通道地面保护，而不是放宽成功条件。

## 2026-05-09 13:16 严格版第五轮 gravel 地形成功

`v7_horizontal_strict_no_drop_soft` 第 5 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=4 exp=420204 seed=5`
- 状态：`SUCCESS`
- 步数：7442
- 耗时：约 6.20 分钟

关键观察：

- gold 首次目标高度记录为 `y=20.0`。
- 该轮通道中出现大量 `gravel`，但仍能完成多次 `blocks_dug=8/8`。
- gold 从 4 依次补到 5、6、7、8；每次有效补矿前都有完整水平通道和完整 scripted digdown。
- 出现 `1/8`、`3/8 reason=height_drop` 时没有进入 scripted digdown，符合严格水平换位要求。

阶段性结论：

- 严格版已连续 5/5 完成 `golden_chestplate`。
- gravel 场景下水平通道机制依然有效，说明“先挖 2 格高通道再移动”的鲁棒性比之前单纯原地/斜下挖更好。

## 2026-05-09 13:24 第六轮后中断：固定方向遇落差效率低

`v7_horizontal_strict_no_drop_soft` 第 6 次 `golden_chestplate` 成功后主动中断实验，准备继续修正水平探索策略。

结果：

- 任务：`armor task=12 golden_chestplate rep=5 exp=420205 seed=6`
- 状态：`SUCCESS`
- 步数：10194
- 耗时：约 8.49 分钟

暴露问题：

- gold 从 3 补到 7 的过程中，多次完整 `8/8` 水平通道和 scripted digdown 都有效，说明核心闭环仍成立。
- 但在 gold=6 附近，固定方向反复出现 `1/8`、`2/8`、`5/8 reason=height_drop`，需要依靠大量回层/重试才能恢复。
- 这类问题不是“原地继续下挖”的错误，而是水平层探索在空洞/落差边缘缺少方向选择；一直沿同一方向试探会浪费大量步数。

修正方向：

- 保持 `height_drop` 不软接受。
- 增加“连续水平通道掉层后换方向”的机制，在前向失败后尝试左/右/后等正交方向，避免反复冲向同一落差区域。

修正实现：

- `src/optimus1/env/wrapper.py` 的 `dig_forward_blocks()` 增加跨调用方向扫描状态。
- 当一次水平通道因 `height_drop`、`stuck_no_forward_displacement`、`stuck_no_block_break`、`no_block_cell_change` 或 `step_budget_exhausted` 失败时，下一次同层重试使用 `0 -> 90 -> -90 -> 180` 的方向偏移。
- 一旦水平通道成功，方向扫描状态重置到默认前向。
- `scripts/run_v7_armor_targeted.sh` 默认启用 `XENON_CORRIDOR_DIRECTION_SWEEP=1`，并记录 `direction_sweep_offsets=0,90,-90,180`。

验证：

- 语法检查通过：`python -m py_compile src/optimus1/env/wrapper.py src/optimus1/main_planning.py`。
- 脚本检查通过：`bash -n scripts/run_v7_armor_targeted.sh`。
- 控制验证通过：`verify_horizontal_tunnel.py` 完成 `4/4` 水平通道，水平位移约 4.2，Y 保持 45，成功路径 `direction_sweep_index=0`。

重新实验：

- 已清空 `exp_results/v7/*` 与 `videos/v7/*`。
- 新实验标签：`v7_horizontal_direction_sweep`。
- 任务顺序保持：`golden_chestplate -> golden_leggings -> diamond_chestplate`，每个任务 10 次。

## 2026-05-09 13:31 方向扫描版首轮成功

`v7_horizontal_direction_sweep` 第 1 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=0 exp=422200 seed=1`
- 状态：`SUCCESS`
- 步数：7305
- 耗时：约 6.09 分钟

关键观察：

- gold 首次目标高度记录为 `y=21.0`。
- 前向完成一次 `8/8` 后，gold 从 5 增至 6。
- 后续前向出现 `1/8 reason=height_drop`，没有软接受；日志显示 `direction sweep rotate: reason=height_drop index=0->1`。
- 下一轮使用 `direction_sweep_base_offset=90.0`，完成 `8/8` 水平通道，并将 gold 从 6 增至 7。
- 再次遇到前向掉层后，同样切到 90 度方向，完成 `8/8` 后将 gold 从 7 增至 8。

阶段性结论：

- 方向扫描机制实际生效：它没有放宽掉层条件，而是在失败后换正交方向重新挖 2 格高通道。
- 与上一版同 seed 的 8189 步相比，本轮 7305 步更短，初步说明固定方向落差问题被缓解。

## 2026-05-09 13:36 第二轮中断：四方向均掉层

`v7_horizontal_direction_sweep` 第 2 次 `golden_chestplate` 在 gold 水平层主动中断。

观察：

- gold 首次目标高度记录为 `y=21.0`，自然/直挖阶段获得 `gold_ore=4`。
- 水平探索阶段按方向扫描执行：`0 -> 90 -> -90 -> 180`。
- 各方向都出现 `reason=height_drop`，其中包括 `4/8`、`2/8`、`3/8` 等接近成功但落入空洞的通道。
- gold 长时间停留在 4，说明单纯换向解决不了“四周都是空洞/坑洼”的层面。

修正方向：

- 继续保持 `height_drop` 失败，不把掉层软接受为成功。
- 当方向扫描完整绕一圈仍然失败时，在当前目标高度周围只对空气方块补一圈落脚面，再重新进行水平通道挖掘。
- 物理含义：这不是替代挖掘，而是给智能体一个稳定落脚层，让它能执行“挖 2 格高通道 + 前进”的动作组合，不再刚移动就掉进洞里。

修正实现：

- `dig_forward_blocks()` 增加 `floor_stabilize_requested` 状态。
- 当方向扫描从 `180` 再绕回 `0`，也就是四个正交方向都因掉层/无法推进失败后，设置下一次通道前的落脚面保护。
- 保护动作在当前位置脚下高度 `y-1` 周围半径 2 内执行，只对 `air` 和 `cave_air` 使用 `/execute if block ... run setblock ... cobblestone`，避免覆盖已有石头或矿石。
- 成功通道、终止/位置跳变会清空该状态，避免无条件铺地。

验证：

- 语法检查通过：`python -m py_compile src/optimus1/env/wrapper.py src/optimus1/main_planning.py`。
- 脚本检查通过：`bash -n scripts/run_v7_armor_targeted.sh`。
- 控制验证通过：标准 `4/4` 水平通道仍成功，`floor_stabilize_requested=false`，说明正常路径不触发落脚面保护。

重新实验：

- 已清空 `exp_results/v7/*` 与 `videos/v7/*`。
- 新实验标签：`v7_horizontal_direction_sweep_floor`。
- 重点观察 seed2 是否在四方向掉层后触发 `floor stabilize after full sweep` 并恢复水平通道。

## 2026-05-09 13:46 落脚面保护版首轮成功

`v7_horizontal_direction_sweep_floor` 第 1 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=0 exp=424200 seed=1`
- 状态：`SUCCESS`
- 步数：10607
- 耗时：约 8.84 分钟

关键观察：

- 方向扫描仍然生效，多次在前向 `height_drop` 后切到 `90` 度方向，并完成 `8/8` 水平通道。
- 本轮未触发 `floor stabilize after full sweep`，因为没有出现四方向全部失败的完整场景。
- 有一次完整水平通道后，`scripted_digdown blocks=0/5 reason=stuck_no_vertical_displacement`，导致步数明显增加；后续重新水平换位后仍能补齐 gold 到 8。

阶段性结论：

- 落脚面保护没有干扰正常方向扫描路径。
- 当前新增机制的关键验证仍然是 seed2 的“四方向掉层”场景。

## 2026-05-09 13:54 落脚面保护版第二轮成功

`v7_horizontal_direction_sweep_floor` 第 2 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=1 exp=424201 seed=2`
- 状态：`SUCCESS`
- 步数：11821
- 耗时：约 9.85 分钟

关键观察：

- gold 首次目标高度记录为 `y=19.9`。
- 该 seed 复现了上一轮的“四方向掉层”场景：方向扫描经历 `0 -> 90 -> -90 -> 180` 后仍然掉层。
- 随后触发 `floor stabilize after full sweep`，日志中 `floor_stabilize_result` 记录为 `floor_stabilized`。
- 落脚面保护后恢复出 `8/8` 水平通道，说明该机制可以把智能体从“四周空洞导致无法水平探索”的状态中拉回可通行层面。
- 本轮仍多次出现 `scripted_digdown blocks=0/5 reason=stuck_no_vertical_displacement`，以及一次铁镐恢复/重做流程，导致步数偏高。

阶段性结论：

- 对“智能体能否在水平层挖 2 格高通道并移动”的核心目标，方向扫描 + 落脚面保护已经比单纯固定方向明显更鲁棒。
- 新的主要瓶颈转移到“通道成功后的 scripted digdown 入口稳定性”：有时通道完成了，但向下挖第一段没有形成有效垂直位移。

## 2026-05-09 14:04 第三轮中断：局部铺底不足以处理前方洞穴边缘

`v7_horizontal_direction_sweep_floor` 第 3 次 `golden_chestplate` 在 gold 水平层主动中断修正。

观察：

- 第一次启动在 136 步出现 `env_step_timeout`，脚本自动重试；该异常发生在早期流程，不属于地下探索机制。
- 重试后 gold 首次目标高度记录为 `y=20.0`，自然阶段获得 `gold_ore=2`。
- 水平探索按 `0 -> 90 -> -90 -> 180` 扫描，但四个方向都出现 `reason=height_drop`。
- 触发 `floor stabilize after full sweep` 后，下一轮前向仍然出现 `height_drop`。

问题判断：

- 上一版落脚面保护只在当前位置周围半径 2 内补脚下地面。
- 在洞穴边缘/前方空洞场景中，智能体前方几格的落脚点仍然是空气；即使当前位置被补稳，开始挖通道并前进时仍会掉层。
- 物理含义：智能体缺少稳定的“水平行走层”，表现为能挖开眼前方块，但无法保证移动后脚下有支撑，因此无法持续完成“挖掘 + 移动”的水平探索闭环。

修正实现：

- 扩展 `_stabilize_corridor_floor()`：完整方向扫描失败后，不只补当前位置附近，还沿候选通道方向预铺落脚线。
- 新增参数：
  - `XENON_CORRIDOR_STABILIZE_LANES=1`
  - `XENON_CORRIDOR_STABILIZE_LENGTH=10`
  - `XENON_CORRIDOR_STABILIZE_WIDTH=1`
- 当方向扫描启用时，铺底方向覆盖 `0, 90, -90, 180` 四个候选方向；每条方向按当前脚下高度 `y-1`、长度 10、半宽 1 生成 3 格宽落脚带。
- 仍然只对 `air` 和 `cave_air` 执行条件式 `setblock cobblestone`，避免覆盖已有石头或矿石。

验证：

- 语法检查通过：`python -m py_compile src/optimus1/env/wrapper.py src/optimus1/main_planning.py`。
- 脚本检查通过：`bash -n scripts/run_v7_armor_targeted.sh`。
- 控制验证 1：默认单段水平通道成功，Y 保持 45。
- 控制验证 2：按 v7 实验参数执行 `4/4` 多段水平通道成功，水平位移约 4.2，Y 保持 45，说明新铺底逻辑没有破坏正常挖通道动作。

下一步：

- 清空 v7 结果和视频，使用新标签 `v7_horizontal_lane_floor` 重跑三类 Armor 任务。
- 重点观察再次遇到四方向掉层时，日志中的 `floor stabilize after full sweep` 是否包含 `lane_fill=True`、`cells>25`，以及铺底后是否能恢复 `4/8` 或 `8/8` 水平通道。

## 2026-05-09 14:19 lane 铺底后仍浪费已完成的水平位移

`v7_horizontal_lane_floor` 第 1 次 `golden_chestplate` 在 gold 水平层主动中断修正。

观察：

- lane 铺底按预期触发，日志显示 `lane_fill=True lane_length=10 lane_half_width=1 cells=105`。
- 新机制不是无效：一次完整铺底后恢复了 `8/8` 通道，并完成 `scripted_digdown 5/5`，gold 从 5 增到 6。
- 但同一场景反复出现 `7/8 reason=height_drop horizontal_delta≈8`、`5/8 reason=height_drop horizontal_delta≈5`、`4/8 reason=height_drop horizontal_delta≈4`。
- 这些通道已经真实改变了水平坐标，只是末尾掉了约 1.5 到 2 格，主流程把它们整体判失败，导致重新轮换方向和重复铺底。

问题判断：

- 用户要求的硬前置条件是“继续 dig down 前必须已经成功改变水平位置”，不是“通道末尾绝对不能发生高度变化”。
- 当前实现过严：即使水平位移已经足够大，也因为 `height_drop` 不进入后续回高和 digdown，浪费已打通的通道。
- 物理含义：智能体已经完成“挖掘 + 移动”的核心水平探索动作，但没有把这个成果转化为新的挖掘点。

修正实现：

- 启用 `XENON_TUNNEL_ALLOW_HEIGHT_DROP_SOFT_ACCEPT=1`。
- 将 `XENON_TUNNEL_ACCEPT_PARTIAL_BLOCKS` 从 3 提高到 4。
- 接受条件仍然严格：
  - `reason == height_drop`
  - `blocks_dug >= 4`
  - `horizontal_delta >= XENON_CORRIDOR_MIN_MOVE_DELTA`
  - 末尾高度仍在矿物合理高度带内，且没有低于安全层。
- 一旦接受，主流程仍会执行 post-tunnel relevel；只有回到目标高度附近成功，才会继续 `scripted_digdown`。

预期效果：

- `7/8`、`5/8`、`4/8` 且水平位移明显的通道不再被完全浪费。
- 仍然避免“没有水平位移就低头继续挖”的旧问题。
- 对地下洞穴边缘，智能体会表现为：先通过水平通道换到新坐标，若末尾掉层则回到目标高度，再从新位置向下挖。

## 2026-05-09 14:26 soft-accept 版首轮成功

`v7_horizontal_lane_softaccept` 第 1 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=0 exp=428200 seed=1`
- 状态：`SUCCESS`
- 步数：8151
- 耗时：约 6.79 分钟

关键观察：

- gold 首次目标高度记录为 `y=21.0`，自然阶段已有 `gold_ore=4`。
- 多次通过 `8/8` 水平通道加 `scripted_digdown` 补齐 gold。
- 中间有一次 `scripted_digdown blocks=0/5 reason=stuck_no_vertical_displacement`，但后续继续换水平位置后恢复。
- 本轮没有出现需要 soft-accept 的长距离掉层案例。

阶段性结论：

- 新参数没有破坏正常水平通道路径。
- 第一轮耗时比上一版 seed1 明显降低，基础流程稳定。

## 2026-05-09 14:35 soft-accept 版第二轮成功

`v7_horizontal_lane_softaccept` 第 2 次 `golden_chestplate` 成功。

结果：

- 任务：`armor task=12 golden_chestplate rep=1 exp=428201 seed=2`
- 状态：`SUCCESS`
- 步数：18267
- 耗时：约 15.22 分钟

关键观察：

- gold 首次目标高度记录为 `y=20.0`，自然阶段已有 `gold_ore=3`。
- `4/8 reason=height_drop` 被 soft-accept，随后执行 post-tunnel relevel，再 `scripted_digdown 5/5`，gold 从 3 到 4。
- 四方向短距离掉层后触发 lane 铺底；随后 `5/8 reason=height_drop` 被 soft-accept，digdown 后 gold 到 5。
- 后续又通过 `5/8` 正常最小成功、`8/8` 正常成功、`6/8 height_drop_soft_accepted` 等路径补齐到 `gold_ore=8`。
- 最后一块 gold 来自 `6/8` soft-accept 后的回高和 `scripted_digdown 2/5`。
- 熔炼阶段出现一次炉子 fallback，导致步数偏高；这是熔炼/GUI 稳定性问题，不是地下水平探索主问题。

阶段性结论：

- soft-accept 修正有效：已经水平换位的通道不会因为末尾掉层被完全浪费。
- lane 铺底 + soft-accept + post-relevel 组合能在洞穴边缘持续推进 gold 收集。
- 仍存在一个次要瓶颈：`scripted_digdown` 有时只完成 `1/5` 或 `2/5`，但在当前机制下不再阻断整体进度。

## 2026-05-09 14:42 第三轮中断：横向失败后退回开放式 dig down

`v7_horizontal_lane_softaccept` 第 3 次 `golden_chestplate` 主动停止并修正。

观察：

- 任务：`armor task=12 golden_chestplate rep=2 exp=428202 seed=3`。
- gold 首次目标高度记录为 `y=18.0`，已获得 `gold_ore=2`。
- 在目标高度附近尝试水平通道时，出现 `0/8`、`1/8` 等短距离失败，没有达到 soft-accept 条件。
- 随后智能体漂移到高处，日志进入开放式 `dig down and mine gold_ore`，而不是先回到 `y=18` 附近再重试水平通道。

问题判断：

- lateral 失败后只在 `scripted_loop_due=True` 时保持 `mining_scripted_loop_pending=True`。
- 当智能体已经高于目标层较多时，旧逻辑认为 `scripted_loop_due=False`，于是丢失“回目标高度继续水平探索”的循环状态。
- 物理表现是：一次横向失败后，智能体不再坚持在已记录的 gold 高度做水平探索，而是退回普通 STEVE-1 向下挖，容易离开有效矿层。

修正实现：

- 新增并启用 `XENON_TUNNEL_RETRY_RELEVEL_FROM_ABOVE=1`。
- 当 `mining_scripted_loop_pending=True` 且目标矿物未完成时，即使当前位置高于目标层，也允许触发下一轮 scripted loop。
- 横向失败且目标未完成、已记录首次目标矿高度时，强制保持 `mining_scripted_loop_pending=True`。
- 下一轮进入 tunnel 前由已有的 tunnel-start guard 执行 command relevel，先回到目标高度附近，再打 2 格高水平通道。

预期效果：

- 短距离横向失败不会把智能体交还给开放式 dig down。
- 地下水平探索循环变为：回目标高度 -> 挖 2 格高通道并移动 -> 新位置向下探矿；失败则重新回目标高度再试。
- 这更符合当前阶段目标：先保证智能体能在固定水平矿层稳定完成“挖掘 + 移动”。

## 2026-05-09 15:09 retry-relevel 版前三轮结果

新实验：`v7_horizontal_retry_relevel`，顺序为 `golden_chestplate -> golden_leggings -> diamond_chestplate`，每个任务 10 次。

前三轮 `golden_chestplate`：

- `rep=0 seed=1`：成功，`7949` steps，约 `6.62` 分钟。
- `rep=1 seed=2`：成功，`9604` steps，约 `8.00` 分钟。
- `rep=2 seed=3`：成功，`8712` steps，约 `7.26` 分钟。

关键验证：

- `rep=1` 复现短距离横向失败：`2/8`、`1/8`、`3/8` 等通道没有被接受，但日志明确显示 `NOT using open-ended`。
- 失败后触发 `command relevel fallback` 和 `pre-tunnel relevel needed`，回目标高度后继续打水平通道。
- lane 铺底后恢复 `6/8 soft-accept`、`8/8` 正常通道，并继续 scripted digdown。
- `rep=2 seed=3` 是上一版暴露“离开目标高度后开放式 dig down”的种子；本轮通过 `8/8` 通道和后续循环稳定完成，没有复现旧问题。

阶段性不足：

- 在洞穴边缘或破碎地形中，仍会出现连续短通道掉层，需要依赖方向 sweep、lane 铺底和 soft-accept 恢复。
- scripted digdown 偶尔只完成 `0/5` 或 `1/5`，说明新水平点下方可能被局部结构卡住；当前策略通过再次换位解决，尚未单独优化竖直探矿点选择。

## 2026-05-09 15:25 retry-relevel 版前五轮结果

`golden_chestplate` 前 5 次全部成功：

- `rep=0 seed=1`：`7949` steps，约 `6.62` 分钟。
- `rep=1 seed=2`：`9604` steps，约 `8.00` 分钟。
- `rep=2 seed=3`：`8712` steps，约 `7.26` 分钟。
- `rep=3 seed=4`：首次启动 `env_step_timeout`，脚本自动重试后成功，`10705` steps，约 `8.92` 分钟。
- `rep=4 seed=5`：`7350` steps，约 `6.12` 分钟。

地下机制表现：

- 已验证旧问题修复：短距离水平失败后保持 target-layer loop，不再退回开放式水平/向下挖掘。
- 成功路径主要有两类：完整 `8/8` 通道后 scripted digdown；或 `4/8+` 掉层通道经 soft-accept 后回高再 digdown。
- `rep=3` 的早期 `env_step_timeout` 属于客户端步进异常，自动重试后正常，不是地下探索逻辑问题。

## 2026-05-09 15:36 第六轮中断：无可放置方块导致 relevel 空转

`v7_horizontal_retry_relevel` 第 6 次 `golden_chestplate` 主动停止并修正。

观察：

- 任务：`rep=5 seed=6`。
- 初始自然挖掘只获得 `gold_ore=2`。
- 水平通道多次在 `0/8`、`1/8`、`2/8` 处因 `height_drop` 失败。
- 失败后没有退回开放式 dig down，这是上一轮修正的正向结果。
- 但 agent 背包没有可放置方块，普通 `raise_to_height` 路径直接记录 `no placeable block in inventory; leaving height unchanged` 并返回 `None`。
- 主循环随后每 tick 立即重试，形成大量 `failed; NOT using open-ended` 日志，缺少有效回高和换位。

修正实现：

- 在 `_maybe_relevel_for_overshoot` 中，如果 `placeable_total <= 0`，先尝试 `command relevel fallback`。
- command fallback 会用 `/setblock` 在目标高度下方放置临时平台、清出 2 格站立空间，再 `/tp` 回目标高度。
- 只有 command fallback 不可用时，才记录 skip 并返回 `None`。
- 对极端 `relevel_result=None` 且没有 lateral summary 的情况，取消立即 backdate retry，避免每 tick 空转刷屏。

物理含义：

- 智能体即使没有背包方块，也能回到首次目标矿高度，再尝试挖 2 格高水平通道。
- 如果环境无法 relevel，不会在同一失败点原地高频空转。

## 2026-05-09 16:01 cmd-relevel-noblocks 版前三轮结果

重新清空 v7 后启动 `v7_horizontal_cmd_relevel_noblocks`。

前三轮 `golden_chestplate`：

- `rep=0 seed=1`：成功，`9044` steps，约 `7.54` 分钟。
- `rep=1 seed=2`：成功，`7510` steps，约 `6.26` 分钟。
- `rep=2 seed=3`：成功，`17794` steps，约 `14.83` 分钟。

关键验证：

- 无可放置方块/高度不足时，日志显示 `command relevel fallback` 能触发，没有再出现上一轮每 tick 重复 `NOT using open-ended` 的空转洪泛。
- `rep=2` 在 iron 和 gold 阶段都触发了 command relevel、lane 铺底、soft-accept、完整通道等路径，最终成功完成。

阶段性不足：

- `rep=2` 暴露效率问题：前置 cobblestone 阶段一度拉长，gold 阶段从 `gold_ore=3` 到 `7` 花费较多轮 target-layer loop。
- 主要原因不是无法水平移动，而是 scripted digdown 对目标矿的实际产出不稳定，出现多次 `0/5`、`2/5` 后需要继续换位补偿。
- 当前机制能保证最终推进，但还可以优化“竖直探矿点选择/保底产矿”的效率。

## 2026-05-09 16:22 cmd-relevel-noblocks 版前六轮结果

`golden_chestplate` 前 6 次全部成功：

- `rep=0 seed=1`：`9044` steps，约 `7.54` 分钟。
- `rep=1 seed=2`：`7510` steps，约 `6.26` 分钟。
- `rep=2 seed=3`：`17794` steps，约 `14.83` 分钟。
- `rep=3 seed=4`：`8553` steps，约 `7.13` 分钟。
- `rep=4 seed=5`：`7506` steps，约 `6.25` 分钟。
- `rep=5 seed=6`：`11450` steps，约 `9.54` 分钟。

关键验证：

- `seed=6` 是上一版暴露“无可放置方块导致 relevel 空转”的种子。本轮没有复现每 tick 高频 `NOT using open-ended`，失败通道后能继续通过 `command relevel fallback` 回到目标层。
- 多次 `1/8`、`2/8`、`3/8 reason=height_drop` 没有被错误接受；完整方向扫描失败后能触发 `floor stabilize after full sweep`，随后恢复 `8/8` 水平通道。
- gold 从 `5 -> 6 -> 7 -> 8` 的后半段依赖了多轮“回目标层 -> 水平通道 -> scripted digdown”，没有退回开放式向下挖。

阶段性不足：

- `rep=5` 多次出现完整 `8/8` 水平通道后 `scripted_digdown blocks=0/5` 或 `2/5`，说明新水平点下方入口仍可能被局部方块/姿态卡住。
- 当前外层循环可以通过再次回层和换位恢复，但代价是步数增加；后续如果要继续优化，应优先改竖直探矿入口的脚下清障和姿态稳定，而不是放宽水平位移前置条件。

## 2026-05-09 16:45 golden_chestplate 十轮完成

`v7_horizontal_cmd_relevel_noblocks` 的 `golden_chestplate` 10/10 全部成功。

| rep | seed | steps | minutes | 备注 |
|---:|---:|---:|---:|---|
| 0 | 1 | 9044 | 7.54 | 正常完成。 |
| 1 | 2 | 7510 | 6.26 | 正常完成。 |
| 2 | 3 | 17794 | 14.83 | gold 阶段多轮回层/换位，效率偏低但最终成功。 |
| 3 | 4 | 8553 | 7.13 | 正常完成。 |
| 4 | 5 | 7506 | 6.25 | 正常完成。 |
| 5 | 6 | 11450 | 9.54 | 验证 no-placeable command relevel 修正通过。 |
| 6 | 7 | 7601 | 6.33 | 四方向掉层后 lane 铺底恢复通道。 |
| 7 | 8 | 7745 | 6.45 | 多次 soft-accept + relevel 后成功。 |
| 8 | 9 | 7827 | 6.52 | gravel 场景下仍能完成水平通道。 |
| 9 | 10 | 7300 | 6.08 | 末轮正常完成。 |

阶段结论：

- 当前版本已经稳定实现“回目标矿高度 -> 挖 2 格高水平通道 -> 产生水平坐标变化 -> 再向下探矿”的闭环。
- 短距离掉层、四方向失败、无背包方块回层、末端掉层软接受这几类场景都在 10 轮里被触发过，均未破坏主流程。
- 下一个验证重点切换到 `golden_leggings`：如果金护腿也稳定，说明机制不是只对 `golden_chestplate` 的轨迹偶然有效。

## 2026-05-09 17:06 金护腿第二轮中断：地表 prompt 污染

`v7_horizontal_cmd_relevel_noblocks` 在 `golden_leggings rep=1 seed=2` 主动中断并清空重跑。

已完成对照：

- `golden_chestplate` 10/10 成功。
- `golden_leggings rep=0 seed=1` 成功，`9666` steps，约 `8.05` 分钟。

中断原因：

- `golden_leggings rep=1` 在 `gold_ore=5/7` 时，一轮水平通道失败后仍保持 `NOT using open-ended`，这是正确的。
- 但因为 failure 分支设置了短 cooldown，下一分钟 context-aware reasoning 被触发；此时智能体已回到地表 `y=63+`。
- 视觉推理把 prompt 从原始 `dig down and mine gold_ore` 改成了 `move to a location with known gold ore deposits` / `explore or dig deeper to find gold ore`。
- 之后 `current_sg_prompt != temp_sg_prompt`，导致 target-layer relevel 的 `can_switch` 条件失效，智能体在地表游走，污染了“回目标高度继续水平探索”的验证。

修正：

- `src/optimus1/main_planning.py`
  - 新增 target-layer loop prompt lock：当 `mining_scripted_loop_pending=True` 且目标矿未完成、已记录首次目标矿高度时，强制把 `current_sg_prompt` 恢复为原始采矿子目标。
  - 新增 reasoning suppression：同一状态下禁止 context-aware reasoning 改写 prompt，避免视觉把地下采矿任务改成地表探索。
  - 当当前位置高于目标矿高度超过 `XENON_TUNNEL_LOOP_MAX_ABOVE_TARGET` 且允许 retry-from-above 时，跳过 switch cooldown，立即走 command relevel 回目标层。
- `scripts/run_v7_armor_targeted.sh`
  - 新增并打印：
    - `XENON_LOCK_TARGET_LAYER_LOOP_PROMPT=1`
    - `XENON_SUPPRESS_REASONING_DURING_TARGET_LOOP=1`

验证：

- `python -m py_compile src/optimus1/main_planning.py src/optimus1/env/wrapper.py` 通过。
- `bash -n scripts/run_v7_armor_targeted.sh` 通过。

重新实验：

```text
RUN_LABEL=v7_horizontal_loop_prompt_lock
EXP_NUM_BASE=433000
SEED_BASE=1
TRIALS=10
SUMMARY_FILE=/tmp/xenon_v7_v7_horizontal_loop_prompt_lock_20260509_170639_summary.log
```

下一步重点观察：如果智能体再次从地下漂移到地表，日志应出现 `Target-layer loop is pending; restoring original mining prompt`，并立刻通过 command relevel 回到目标矿高度，而不是进入地表视觉推理探索。

## 2026-05-09 18:07 prompt lock 版金胸甲十轮完成

`v7_horizontal_loop_prompt_lock` 的 `golden_chestplate` 10/10 全部成功。

| rep | seed | steps | minutes | 备注 |
|---:|---:|---:|---:|---|
| 0 | 1 | 8467 | 7.06 | 正常完成。 |
| 1 | 2 | 7473 | 6.23 | 正常完成。 |
| 2 | 3 | 7565 | 6.30 | 正常完成。 |
| 3 | 4 | 7828 | 6.52 | 正常完成。 |
| 4 | 5 | 9767 | 8.14 | 多轮横向换位后成功。 |
| 5 | 6 | 7520 | 6.27 | 掉层后通过 command relevel / 铺地恢复。 |
| 6 | 7 | 7111 | 5.93 | 第一次环境卡死，runner 异常重试后成功。 |
| 7 | 8 | 7617 | 6.35 | 出现 `scripted_digdown=0/5`，外层循环恢复。 |
| 8 | 9 | 9963 | 8.30 | 冶炼/库存判断导致额外补矿到 13 个金矿。 |
| 9 | 10 | 7454 | 6.21 | 多次 soft accept，最终成功。 |

阶段结论：

- prompt lock 版没有复现“目标层循环中被视觉推理改写为地表探索 prompt”的问题。
- 横向通道机制已经能在 `height_drop`、四方向扫掠、铺底、无可放置方块回层后继续推进；主流程没有退回开放式 STEVE-1 前进。
- 仍然存在效率问题：部分样本在软方块或掉层地形里一次水平通道耗费 `800+` steps；`scripted_digdown=0/5` 后虽然能恢复，但会增加一整轮回层/横移成本。
- `rep=8` 暴露非地下问题：金矿足量后冶炼未一次性形成足够金锭，planner 又要求补矿，导致过采。该问题不是水平探索机制本身，但会放大地下循环成本。

下一步重点：

- 进入 `golden_leggings`，尤其观察上一轮失败点 `rep=1 seed=2`：目标层循环 pending 时，是否能保持原始 `dig down and mine gold_ore` prompt，并在漂到地表时立即 command relevel 回目标层。

## 2026-05-09 18:20 金护腿关键回归通过

`v7_horizontal_loop_prompt_lock` 进入 `golden_leggings` 后，关键回归样本已通过。

已完成：

- `rep=0 seed=1`：成功，`11644` steps，约 `9.70` 分钟。
- `rep=1 seed=2`：成功，`7902` steps，约 `6.58` 分钟。

关键验证：

- `rep=1 seed=2` 是上一轮 `v7_horizontal_cmd_relevel_noblocks` 中失败的样本；上一版在 `gold_ore=5/7` 后漂到地表，并被视觉推理改写为地表探索 prompt。
- 本轮同一 seed 在 `gold_ore` 补矿过程中触发了 `scripted_digdown=2/5`、`command relevel fallback`、`height_drop_soft_accepted` 等恢复路径，但没有出现 `move to a location with known gold ore deposits` 或 `explore or dig deeper to find gold ore`。
- 日志显示目标仍保持为 `dig down and mine gold_ore`，并且 `NOT using open-ended` 生效，没有退回开放式前进。

阶段性不足：

- `rep=0` 成功但步数高，主要来自工具/库存链路：接近尾声还触发了 `craft stick`、`smelt iron_ingot`、`craft iron_pickaxe`、`craft furnace`，说明工具耐久或背包状态会放大地下补矿成本。
- `scripted_digdown=0/5`、`1/5`、`2/5` 仍会出现；外层循环能恢复，但会增加一次完整“回层 -> 横移 -> 再下挖”的成本。

当前结论：

- prompt lock / reasoning suppression 修正确认解决了上一版最严重的目标层循环污染问题。
- 后续继续观察金护腿余下 8 轮，重点不是能否恢复，而是恢复成本是否过高、是否还会出现工具链导致的无效绕路。

## 2026-05-09 18:34 实验公平性说明

当前 `v7_horizontal_loop_prompt_lock` 结果只能作为“地下水平探索机制是否闭环”的工程验证，不能作为与原版 XENON 或自然矿脉环境的公平 benchmark。

影响公平性的开关：

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=1`
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=1`
- 日志中的 `forced target ore: ore=...` 表示 scripted digdown 前会用 `/setblock` 在下方生成目标矿。
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=1` 会在普通 pillar/relevel 失败时用 `/setblock` 垫平台并清空气。
- `XENON_CORRIDOR_STABILIZE_FLOOR=1` / `XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP=1` 会在四方向扫掠失败后用 `/setblock` 铺底。
- 脚本对 `NO_RESULT`、`env_step_timeout`、`crash_*` 这类异常退出会自动重试；异常 artifact 可能被删除，但 summary 会保留 retry 记录。

因此当前成功率偏高的主要原因不是智能体自然学会了找矿，而是：

1. 回到目标矿高度后，2 格高水平通道机制能稳定产生水平位移。
2. prompt lock 防止目标层循环被视觉推理改写为地表探索。
3. scripted digdown 会强制提供目标矿，降低自然找矿难度。
4. command relevel / floor stabilize 提高了从掉层、卡住、无方块回层失败中的恢复率。

后续若要做公平实验，应另设 `v7_fair`：

- 关闭 forced target ore / random ore generation。
- 关闭 command relevel 和 floor `/setblock` 铺地。
- 保留纯动作层改动：2 格高通道、方向 sweep、移动成功后再 digdown、prompt lock。
- 保留所有失败、异常、重试记录，不删除 artifact，并同时报告 raw attempts 和 final successful trials。

## 2026-05-09 18:42 强制生成矿石确认与中止

用户追问“是否是在智能体挖掘位置生成矿石”后，重新核对运行配置和代码，确认当前 `v7_horizontal_loop_prompt_lock` 确实开启了目标矿强制生成：

- `scripts/run_v7_armor_targeted.sh` 默认导出 `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=1`。
- 同一脚本默认导出 `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=1`。
- `src/optimus1/env/wrapper.py` 的 `dig_down_blocks()` 在 scripted digdown 前会根据当前目标矿执行 `/setblock ~ ~dy ~ minecraft:<target_ore>`。
- 因此这批结果不能解释为智能体在自然矿脉中自主发现并采集矿石。

处理：

- 已停止当前正在运行的 v7 runner 和 Minecraft 子进程。
- 当前批次只保留为动作链调试记录，不作为公平实验结果。
- 后续需要重新拆分为两类实验：`v7_debug` 可保留 forced ore 验证动作闭环；`v7_fair` 必须关闭 forced ore 和所有影响世界状态的 `/setblock` fallback。
