# V7 Fair Armor Experiment Log

## 2026-05-09 18:50 公平配置重启

目的：在保留当前地下水平探索动作逻辑的前提下，恢复正常环境配置，重新评估 Armor 任务。

本轮保留的地下探索逻辑：

- 记住首次挖到当前阶段所需矿物的高度。
- overshoot / bedrock 卡住后回到目标矿层附近。
- 在目标矿层挖 2 格高水平通道。
- 水平通道支持 `0,+30,-30` yaw fan 和 `0,90,-90,180` 方向 sweep。
- 只有水平位置变化达到阈值后，才继续 scripted dig down。
- target-layer loop 中锁定目标 prompt，避免被视觉推理改写成地表探索。

本轮关闭的影响公平性的配置：

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`：不再通过 `/setblock` 生成随机或目标矿。
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`：不再在智能体下方强制放置当前目标矿。
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`：不再通过 `/setblock` + `/tp` 命令回到目标高度。
- `XENON_CORRIDOR_STABILIZE_FLOOR=0`：不再通过 `/setblock` 铺设地下通道地板。
- `XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP=0`：四方向失败后不再命令铺底。
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`：不再通过 `/clear` + `/give` 修复合成结果。
- `MAX_RETRIES_ON_CRASH=0`：公平记录 raw attempt，不自动用重试掩盖异常退出。
- `DELETE_ABNORMAL_ARTIFACTS=0`：保留失败和异常 artifact。

清理状态：

- 已删除旧的 `exp_results/v7` 调试结果。
- 已删除旧的 `videos/v7` 调试视频。
- 已删除 `/tmp/xenon_v7_*.log` 旧运行日志。

计划任务顺序：

1. `Armor task=12`：`golden_chestplate`，10 次。
2. `Armor task=10`：`golden_leggings`，10 次。
3. `Armor task=6`：`diamond_chestplate`，10 次。

本轮结果目录：

- `exp_results/v7`
- `videos/v7`

本轮 summary 以 `/tmp/xenon_v7_v7_fair_horizontal_tunnel_*_summary.log` 记录。

## 2026-05-09 18:55 第一次公平启动中止与修正

第一次公平启动后，`golden_chestplate rep=0 seed=1` 在金矿阶段暴露出一个动作逻辑问题：

- 智能体在 `y=19` 已处于 gold_ore 有效高度带内。
- 之前记录的 first target ore height 是 `y=21`。
- command relevel 已关闭后，普通 pillar-up 多次 `reason=no_placement_succeeded`。
- 旧逻辑仍要求先回到 `y=21`，导致水平通道被跳过，并反复进入 `scripted_relevel_retry`。

修正：

- 保留公平环境配置，不恢复任何 `/setblock`、`/tp`、`/give`。
- 将 `XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT` 默认设为 `1`。
- 当 `raise_to_height()` 失败但当前位置仍满足目标矿物高度带、最低安全高度、最大可接受 drop 时，将当前位置软接受为水平探索层。
- 这样智能体可以在同一矿层内继续挖 2 格高水平通道，而不是无限执着于首次矿物高度。

验证：

- `src/optimus1/main_planning.py` 已通过 `py_compile`。
- `scripts/run_v7_armor_targeted.sh` 已通过 `bash -n`。
- 中止后的 partial v7 产物会再次清空后重跑。

## 2026-05-09 19:05 NameError 修复后重启

第二次公平启动中，`golden_chestplate rep=0 seed=1` 自然采到 `gold_ore=4`，但在回层失败后的 soft-accept 分支触发代码异常：

- 异常：`NameError: name 'tunnel_allow_low_soft_accept' is not defined`。
- 原因：新加的回层失败 soft-accept 判断位于 `_apply_lateral_shift()` 外层，但复用了该内层函数中的局部变量。
- 这不是智能体行为失败，不能计入公平结果。

修正：

- 将 soft-accept 所需的矿层高度带、最低安全高度、最大可接受 drop 等参数提升到 `_maybe_relevel_for_overshoot()` 外层作用域。
- 新增外层 `_outer_height_is_acceptable_tunnel_layer()` 给回层失败分支使用。
- `src/optimus1/main_planning.py` 已再次通过 `py_compile`。

处理：

- 当前由代码异常产生的 v7 失败 artifact 会清空。
- 从空的 `exp_results/v7` 和 `videos/v7` 重新开始公平实验。

## 2026-05-09 19:20 第三次公平启动 rep0 结果

配置确认：

- summary 中 `scripted_digdown_ore=0`、`scripted_force_target=0`。
- `cmd_relevel_fallback=0`、`cmd_craft_fallback=0`。
- `stabilize_floor=0`、`stabilize_after_sweep=0`。
- 日志未出现 `forced target ore`、command relevel fallback、command craft fallback。

`golden_chestplate rep=0 seed=1`：

- 结果：失败，`status=crash_RuntimeError`，原因是 30 分钟 `Timeout!`。
- artifact 已保留：`exp_results/v7/ours_planning_craft_golden_chestplate_471200_failed_forest_nb2k.json`。
- 失败不是代码异常，而是公平环境下未在时限内补齐金矿。
- 过程观察：智能体自然采到 `gold_ore=6/8`，中途还自然挖到 `diamond=3`。
- 地下水平探索能执行，且多次完成 `horizontal_tunnel` + `scripted_digdown`；但自然矿脉下补齐剩余 `2` 个金矿效率不足。
- 主要瓶颈：水平层探索仍偏“盲挖路径”，不能主动沿可见矿脉/洞穴扩展，且高度落差软接受后会让循环消耗大量步数。

当前处理：

- 该失败作为公平结果保留，不删除、不重试。
- 继续运行 `golden_chestplate rep=1 seed=2`。

## 2026-05-09 19:31 rep1 代码异常与地表回落逻辑修正

`golden_chestplate rep=1 seed=2` 触发新的代码异常：

- 异常：`NameError: name '_current_env_y' is not defined`。
- 原因：外层 soft-accept 分支仍调用了 `_apply_lateral_shift()` 内部的 `_current_env_y()`。
- 处理：新增外层 `_outer_current_env_y()`，并将外层 soft-accept 分支改为调用该函数。
- 验证：`src/optimus1/main_planning.py` 已通过 `py_compile`。

同时观察到一个公平环境下的真实动作逻辑问题：

- 智能体有时回到地表或高处，例如 `cur_y=61`，但目标金矿高度是 `dest_y=19`。
- 旧逻辑将 `needed_dy <= 0` 统一视为“已经在目标层或高于目标层”，然后尝试目标层水平通道。
- 在 `cur_y=61` 这种场景下，水平通道必然被 `pre_tunnel_above_target` 拒绝，并反复循环。

修正：

- 当 `cur_y > dest_y + XENON_TUNNEL_MAX_ABOVE_TARGET` 时，不进入目标层水平通道。
- 该分支返回 `None`，让原始 `dig down and mine <ore>` 继续执行，从地表/高处重新下挖。
- 这仍是动作逻辑修正，不使用 `/setblock`、`/tp`、`/give`。

由于 rep1 是代码异常，且 rep0 使用了修正前逻辑，本轮结果清空后重新从 rep0 开始。

## 2026-05-09 19:39 关闭 wrapper 一次性随机矿石

重新启动后，日志出现：

- `diamond ore at 3`
- `diamond ore at 4`
- `diamond ore at 2`

排查确认：除了 scripted digdown 的生成矿石外，`CustomEnvWrapper.step()` 中还存在一次性 `random_ore(self.env, self.ORE_MAP, ypos)` 调用，会通过 `/setblock` 改写世界。这同样影响公平性。

修正：

- 在 `src/optimus1/env/wrapper.py` 中为该一次性随机矿石逻辑增加 `XENON_ENABLE_RANDOM_ORE_ONCE` 开关。
- 默认值设为 `0`。
- `scripts/run_v7_armor_targeted.sh` 显式导出 `XENON_ENABLE_RANDOM_ORE_ONCE=0`，summary 中新增 `random_ore_once`。
- `src/optimus1/main_planning.py`、`src/optimus1/env/wrapper.py` 已通过 `py_compile`。
- `scripts/run_v7_armor_targeted.sh` 已通过 `bash -n`。

处理：

- 当前含随机矿石生成的 partial v7 结果无效。
- 再次清空 `exp_results/v7`、`videos/v7` 和 `/tmp/xenon_v7_*.log` 后，从 rep0 重跑。

## 2026-05-09 19:47 infra early stop 单独处理

关闭 `random_ore_once` 后重启，`golden_chestplate rep=0` 在 `steps=6` 失败。结果 JSON 中有 `infra_early_stop=true`，日志显示 Minecraft ready 后 Malmo 立即 quit：

- 这不是智能体行为失败。
- 如果直接计入 `FAIL`，会污染公平成功率。

修正 runner：

- 新增 `MAX_RETRIES_ON_INFRA_EARLY_STOP`，默认 `2`。
- 识别结果 JSON 中的 `infra_early_stop=true`。
- 对 infrastructure early stop 进行有限重试。
- 不删除异常 artifact；summary 保留 retry 记录。
- `MAX_RETRIES_ON_CRASH` 仍保持 `0`，普通 crash/timeout 不用重试掩盖。

处理：

- 含 infra early stop 的 partial v7 结果再次清空。
- 从空目录重新启动公平实验。

## 2026-05-09 19:55 logs waypoint 误选 `craft logs` 修正

连续 `infra_early_stop` 的直接原因不是 Minecraft 启动失败，而是决策器在 `logs` waypoint 选择了不可执行动作 `craft logs`：

- `new_craft_helper` 对 `logs` 没有 recipe，立即失败。
- 旧的 `_subgoal_action_is_feasible()` 只要 action 文本含有 `log` 就认为 logs waypoint 可行，因此没有拒绝 `craft logs`。
- 这会导致开局 6 步失败，被 `infra_early_stop` 捕获。

修正：

- 对 `logs` / `*_log` waypoint 显式拒绝 `craft` 和 `smelt` 动作。
- 只接受 `chop`、`punch`、`tree`、`collect` 类动作。
- 被拒绝后走既有 semantic fallback：`chop a tree`。
- `src/optimus1/main_planning.py` 和 `src/optimus1/env/wrapper.py` 已通过 `py_compile`。
- `scripts/run_v7_armor_targeted.sh` 已通过 `bash -n`。

处理：

- 当前由 `craft logs` 引起的 partial v7 结果无效。
- 清空后重新从 rep0 开始。

## 2026-05-09 20:00 craft-only waypoint 可行性过滤

修复 `craft logs` 后，开局 `logs` 已正确回退到 `chop a tree`，但下一步又出现同类问题：

- `planks` waypoint 被规划为 `dig down and mine planks`。
- 旧可行性过滤只限制了 logs、mine-only ores、ingot smelting，没有限制普通 craft-only 物品。

修正：

- 新增 `CRAFT_ONLY_WAYPOINTS`，包含 `planks`、`stick`、`crafting_table`、工具、furnace、Armor 目标物等。
- craft-only waypoint 显式拒绝 `mine`、`dig`、`smelt`。
- craft-only waypoint 只接受 `craft`、`make`、`create`。
- 被拒绝后仍走既有 semantic fallback，例如 `planks -> craft planks`。
- `src/optimus1/main_planning.py` 和 `src/optimus1/env/wrapper.py` 已通过 `py_compile`。
- `scripts/run_v7_armor_targeted.sh` 已通过 `bash -n`。

处理：

- 当前由 `dig down and mine planks` 引起的 partial v7 结果无效。
- 清空后重新从 rep0 开始。

## 2026-05-09 20:04 重启验证

最新重启 summary：`/tmp/xenon_v7_v7_fair_horizontal_tunnel_20260509_200136_summary.log`。

已确认：

- `scripted_digdown_ore=0`
- `scripted_force_target=0`
- `random_ore_once=0`
- `cmd_relevel_fallback=0`
- `cmd_craft_fallback=0`
- `stabilize_floor=0`
- `stabilize_after_sweep=0`

开局验证：

- `logs` waypoint 中，`craft logs` 被可行性过滤拒绝，回退为 `chop a tree`。
- `planks` waypoint 中，`dig down and mine planks` 被可行性过滤拒绝，回退为 `craft planks`。
- 当前 `golden_chestplate rep=0` 已推进到 `iron_ore` 阶段。
- 暂未发现 `forced target ore`、`ore at ...` 随机矿石生成、command relevel fallback 或 command craft fallback 日志。
