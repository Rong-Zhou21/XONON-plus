# V7 Armor 实验环境记录：原始 XENON 动态黄金放矿 1.5x

日期：2026-05-10

## 目的

本轮不再增加全局世界生成矿石量，而是在原始 XENON 的动态放矿行为基础上，只把黄金动态放矿的期望量提升到当前的 1.5 倍。

## 原始 XENON 基线

XENON-main 的当前任务环境使用：

```python
handlers.DefaultWorldGenerator(force_reset=True)
```

也就是 MineRL handler 默认的 `generatorOptions="{}"`。本轮已经恢复这个世界生成基线，不再设置 `ironCount`、`goldCount` 或 `diamondSize`。

XENON-main 的 `random_ore()` 动态放矿逻辑会在 `_only_once=True` 后按高度尝试 `/setblock`：

- `45 <= y <= 50`：煤矿。
- `26 <= y <= 43`：铁矿。
- `14 < y <= 26`：黄金或红石。
- `y <= 14`：钻石。

原始默认参数 `thresold=0.9` 表示每次触发约 10% 概率放置 1 个矿石方块。

## 当前修改

实现位置：`src/optimus1/env/wrapper.py`

- 保留 `XENON_ENABLE_RANDOM_ORE_ONCE=1`。
- 新增/使用 `XENON_RANDOM_ORE_GOLD_MULTIPLIER=1.5`。
- 对黄金动态放矿单独调整：原始 10% 放置概率提升到 15%。
- 煤矿、铁矿、红石、钻石仍保持原始 10% 动态放矿概率。

这等价于黄金动态放矿期望量 `1.0x -> 1.5x`，但不会改变全局世界生成。

## 关闭项

以下 Plus 后来新增的命令式补偿仍保持关闭：

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`

## V7 实验设置

脚本：`scripts/run_v7_armor_targeted.sh`

任务顺序：

1. `golden_chestplate`，Armor task id 12，10 次。
2. `golden_leggings`，Armor task id 10，10 次。
3. `diamond_chestplate`，Armor task id 6，10 次。

计划启动命令：

```bash
RUN_LABEL=v7_dynamic_gold15x_xenonbase EXP_NUM_BASE=550000 TRIALS=10 bash scripts/run_v7_armor_targeted.sh
```

输出路径：

- 结果 JSON：`exp_results/v7/`
- 视频：`videos/v7/`
- summary：`/tmp/xenon_v7_v7_dynamic_gold15x_xenonbase_*.log`

## 实际启动记录

- 启动时间：2026-05-10 22:29:34 CST
- 后台 PID：`2050460`
- launcher log：`/tmp/xenon_v7_dynamic_gold15x_xenonbase_launcher_20260510_222934.log`
- summary：`/tmp/xenon_v7_v7_dynamic_gold15x_xenonbase_20260510_222934_summary.log`
- 第一条任务日志：`/tmp/xenon_v7_v7_dynamic_gold15x_xenonbase_armor_t12_rep0_exp551200_20260510_222937.log`
- 已验证世界生成：`DefaultWorldGenerator generatorOptions={}`
- 已验证动态放矿：`random_ore_once=1`
- 已验证黄金动态放矿倍率：`random_ore_gold_mult=1.5`
- 已验证 Plus 额外补偿关闭：`scripted_digdown_ore=0`、`scripted_force_target=0`、`cmd_relevel_fallback=0`、`cmd_craft_fallback=0`。
