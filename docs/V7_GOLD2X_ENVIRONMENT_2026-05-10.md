# V7 Armor 实验环境记录：黄金矿 2 倍生成设置

日期：2026-05-10

## 目的

本轮实验在 MineRL/Minecraft 世界生成规则层面提高地下黄金矿可获得性，用于测试 Armor 任务中的黄金装备链路。修改目标是“整个世界中每个新生成 chunk 的黄金矿脉生成尝试次数翻倍”，不是“在智能体附近、脚下或挖掘路径上生成目标矿”。

## 世界生成基线

- 世界生成器：`handlers.DefaultWorldGenerator(force_reset=True, generator_options='{"goldCount":4}')`，位置在 `src/optimus1/env/custom_env.py`。
- 每个实验仍使用脚本传入的 `seed` 和 `world_seed`；本轮 v7 脚本默认 `SEED_BASE=0`，10 次重复对应 `world_seed=0..9`。
- 未清除世界中的原生矿物，未改变钻石、铁、煤、红石等矿物生成。

## 黄金矿 2 倍机制

实现位置：`src/optimus1/env/custom_env.py` 的 `create_server_world_generators()`。

Malmo 的 `DefaultWorldGenerator` 会把 `generator_options` 传给 Minecraft 的 customized world generator。这里仅设置：

- `goldCount=4`

Minecraft customized generator 中 `goldCount` 表示每个 chunk 的黄金矿脉生成尝试次数。默认值是 `2`，本轮改为 `4`。其他黄金参数保持默认：

- `goldSize=9`
- `goldMinHeight=0`
- `goldMaxHeight=32`

物理含义：世界生成阶段对所有新生成 chunk 使用同一规则，黄金矿脉尝试次数从每 chunk 2 次提升为 4 次，因此全世界黄金矿物的期望生成量约翻倍。智能体仍需要正常找树、制作工具、移动、下挖、采集、熔炼与合成；不会因为当前位置、目标物品或当前挖掘轨迹而获得定点补矿。

## 公平性约束

以下会影响公平性的补偿逻辑保持关闭：

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`：scripted dig down 不生成路径矿。
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`：不在智能体脚下或下方强制放置当前目标矿。
- `XENON_ENABLE_RANDOM_ORE_ONCE=0`：关闭旧的一次性随机脚下补矿。
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`：不使用命令方块式回层。
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`：不使用命令式合成兜底。

代码默认也已调整为：即使从非 v7 脚本入口调用 `dig_down_blocks()`，路径/脚下补矿也默认关闭。

## 生物与基础环境

保留之前确认过的“清除生物影响”设置：

- `/gamerule doMobSpawning false`
- `/difficulty peaceful`

其他 reset 命令：

- `/gamerule sendCommandFeedback false`
- `/gamerule commandBlockOutput false`
- `/gamerule keepInventory true`
- `/effect give @a night_vision 99999 250 true`
- `/gamerule doDaylightCycle false`
- `/time set 0`
- `/gamerule doImmediateRespawn true`
- `/spawnpoint`

说明：peaceful 与禁用刷怪用于排除僵尸等敌对生物导致的死亡干扰，不改变矿物分布。

## V7 Armor 实验设置

脚本：`scripts/run_v7_armor_targeted.sh`

任务顺序：

1. `golden_chestplate`，Armor task id 12，10 次。
2. `golden_leggings`，Armor task id 10，10 次。
3. `diamond_chestplate`，Armor task id 6，10 次。

关键地下机制：

- 抬升目标：`XENON_OVERSHOOT_RELEVEL_TARGET_MODE=surface`，触发后尽量回到地表或可达最高位置。
- 抬升触发：关闭普通 Y 触发，主要依赖触底坚硬矿石或更高级矿物触发。
- 换位距离：`XENON_OVERSHOOT_LATERAL_BLOCKS=2`。
- 横向通道：智能体挖出两格高通道后，至少完成水平位置变化再继续 dig down。
- 决策器：`memory.case_memory.decisioner.enabled=true`，checkpoint `artifacts/decisioner/rads_v2.pt`，`min_p_success=0.20`。

## 本轮运行记录

计划运行命令：

```bash
RUN_LABEL=v7_gold2x_default EXP_NUM_BASE=490000 TRIALS=10 bash scripts/run_v7_armor_targeted.sh
```

输出路径：

- 结果 JSON：`exp_results/v7/`
- 视频：`videos/v7/`
- summary：启动后由脚本输出到 `/tmp/xenon_v7_v7_gold2x_default_*.log`

实际有效启动：

```bash
RUN_LABEL=v7_gold2x_global_clean EXP_NUM_BASE=510000 TRIALS=10 bash scripts/run_v7_armor_targeted.sh
```

- 启动时间：2026-05-10 19:18:31 CST。
- 后台 launcher：`/tmp/xenon_v7_gold2x_global_clean_launcher_20260510_191831.log`
- summary：`/tmp/xenon_v7_v7_gold2x_global_clean_20260510_191831_summary.log`
- 第一条任务：`golden_chestplate rep=0 exp=511200 seed=0`。
- 验证：第一条任务日志 `/tmp/xenon_v7_v7_gold2x_global_clean_armor_t12_rep0_exp511200_20260510_191837.log` 已打印 `DefaultWorldGenerator generatorOptions={"goldCount":4}`。
- 状态更新：用户反馈“矿物感觉变少”后，已于 2026-05-10 20:39 CST 左右暂停该实验，避免继续产生有疑问的数据。该轮已产生的 v7 partial results 只用于配置排查，不作为最终对比实验数据。

## 无效启动记录

18:44 左右第一次启动时误用了 spawn 附近局部增密方案，并且 reset 阶段发送命令过多，导致 Malmo `done=True` 超时，结果无效。18:54 左右第二次启动仍属于局部增密方案。19:00 左右曾用全局方案前台验证，确认配置有效，但随后为避免交互式终端占用，已停止并清空输出。19:18:31 后台启动的 `v7_gold2x_global_clean` 已在用户反馈矿物疑问后暂停；该轮 partial results 不计入最终对比。
