# V7 Armor 实验环境记录：基于 XENON-main 的铁/金/钻石 1.5x 生成设置

日期：2026-05-10

> 后续方案已替代本文设置：当前实验不再增加全局世界生成矿石量，而是使用 `docs/V7_DYNAMIC_GOLD15X_ENVIRONMENT_2026-05-10.md` 中记录的“原始 XENON 世界生成 + 黄金动态放矿 1.5x”配置。

## 目的

本轮实验先阅读 `/home/yzb/zhourong/XENON-main` 的原始矿物资源环境配置，再把用户提出的“钻石、白银、黄金数量增加到 1.5 倍”落实到 Minecraft customized world generator 的全局生成参数中。Minecraft 原版没有 silver ore，因此本文中的“白银”按任务资源链映射为 `iron_ore`。

原版 XENON-main 当前任务环境的关键配置：

- `/home/yzb/zhourong/XENON-main/src/optimus1/env/custom_env.py` 中 `create_server_world_generators()` 返回 `handlers.DefaultWorldGenerator(force_reset=True)`。
- `/home/yzb/zhourong/XENON-main/minerl/minerl/herobraine/hero/handlers/server/world.py` 中 `DefaultWorldGenerator` 的默认值是 `generator_options="{}"`。
- XENON-main 还存在 `random_ore()` 的 `/setblock` 动态放矿逻辑；只要主流程设置 `_only_once=True`，wrapper 就会按当前高度以原始概率尝试在智能体下方放置对应矿物。本轮保留这个原始行为，不再因为公平性关闭它。

## 世界生成设置

实现位置：`src/optimus1/env/custom_env.py`

```python
ORE15X_WORLD_GENERATOR_OPTIONS = '{"ironCount":30,"goldCount":3,"diamondSize":12}'
```

含义：

- `ironCount=30`：铁矿每 chunk 生成尝试次数从原版默认 20 提升到 30，约 1.5x。
- `goldCount=3`：黄金矿每 chunk 生成尝试次数从原版默认 2 提升到 3，约 1.5x。
- `diamondSize=12`：钻石默认 `diamondCount=1`，不能精确设置 1.5 次生成尝试；因此保持 `diamondCount` 默认不变，把单矿脉最大尺寸从 8 提升到 12，使期望钻石块数量约 1.5x。

未改动项：

- 未显式改动 `coalCount`、`redstoneCount`、`lapisCount`。
- 未启用脚下补矿、路径补矿或命令式补偿。
- 仍使用原版任务环境同类的 `DefaultWorldGenerator(force_reset=True)`，只通过 `generatorOptions` 传入上述三个参数。

## 原版 XENON 基线保留项

保留原版动态放矿：

- `XENON_ENABLE_RANDOM_ORE_ONCE=1`

以下 Plus 后来新增的命令式补偿逻辑保持关闭，因为它们不是 XENON-main 原始基线的一部分：

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`

保留清除生物干扰设置：

- `/gamerule doMobSpawning false`
- `/difficulty peaceful`

## V7 实验设置

脚本：`scripts/run_v7_armor_targeted.sh`

任务顺序：

1. `golden_chestplate`，Armor task id 12，10 次。
2. `golden_leggings`，Armor task id 10，10 次。
3. `diamond_chestplate`，Armor task id 6，10 次。

计划启动命令：

```bash
RUN_LABEL=v7_ore15x_xenonbase EXP_NUM_BASE=540000 TRIALS=10 bash scripts/run_v7_armor_targeted.sh
```

输出路径：

- 结果 JSON：`exp_results/v7/`
- 视频：`videos/v7/`
- summary：`/tmp/xenon_v7_v7_ore15x_xenonbase_*.log`

## 实际启动记录

- 启动时间：2026-05-10 21:47:15 CST
- 后台 PID：`2044357`
- launcher log：`/tmp/xenon_v7_ore15x_xenonbase_launcher_20260510_214715.log`
- summary：`/tmp/xenon_v7_v7_ore15x_xenonbase_20260510_214715_summary.log`
- 第一条任务日志：`/tmp/xenon_v7_v7_ore15x_xenonbase_armor_t12_rep0_exp541200_20260510_214718.log`
- 已验证第一条任务日志打印：`DefaultWorldGenerator generatorOptions={"ironCount":30,"goldCount":3,"diamondSize":12}`
- 已验证原版动态放矿保留：`random_ore_once=1`
- 已验证 Plus 额外补偿关闭：`scripted_digdown_ore=0`、`scripted_force_target=0`、`cmd_relevel_fallback=0`、`cmd_craft_fallback=0`。
