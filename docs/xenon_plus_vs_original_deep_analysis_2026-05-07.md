# XENON-plus 相比原版 XENON 的深度优化分析

生成日期：2026-05-07  
对比对象：

- 原版：`/home/yzb/zhourong/XENON-main`
- plus：`/home/yzb/zhourong/XENON-plus`

## 分析范围

本报告聚焦“plus 相比原版的真实优化”。对比时主要阅读了 `src/optimus1`、`scripts`、`docs`、`reports` 与顶层 `app.py`。以下内容没有作为核心代码优化展开：

- `__pycache__`、`.pyc`、日志、视频、运行结果等生成物。
- 大量 `memories/ours_planning/.../*.json` 运行案例文件：它们是 plus 运行后积累的数据资产，本身重要，但不是底层实现逻辑。
- 原版自带 MineRL/minerl 子树差异：plus 仓库没有把它作为主要创新点修改。

总体结论：XENON-plus 不是重写原版 XENON，而是在保留原版 OracleGraph、planner、STEVE-1、helper、MineRL 环境骨架的前提下，把系统重心从“waypoint 上有成功动作就直接复用，否则调用 VLM planner”推进为“状态化案例库 + 可训练决策器 + 环境感知动作稳定器 + 更可靠实验记录”。

## 优化点 1：动作记忆升级为状态化案例库

相关文件：

- 原版：`XENON-main/src/optimus1/memories/decomposed_memory.py`
- plus：`src/optimus1/memories/case_memory.py`
- 调用点：`src/optimus1/main_planning.py`

**代码逻辑实现**

原版 `DecomposedMemory` 的核心单位是 `waypoint -> action -> success/failure count`。`main_planning.py` 中 `make_plan()` 直接读取第一个 waypoint，然后调用 `is_succeeded_waypoint(wp)`；有成功动作就复用，没有就把相似成功 waypoint 和失败 subgoal 传给 planner。

plus 新增 `CaseBasedMemory`，核心存储变为“决策案例”。每个 case 不只包含 waypoint 和动作，还记录：

- `original_final_goal`
- `waypoint` / `waypoint_num`
- `state_snapshot`
- `candidate_actions`
- `selected_action`
- `selected_subgoal`
- `decision_trace`
- `outcome`

`state_snapshot` 由 `create_state_snapshot()` 生成，包括 inventory、equipment、location_stats、plain_inventory、biome、obs_summary。`make_plan()` 现在传入完整 `env_status`，先构造状态快照，再调用 `select_case_decision()`；只有没有可信案例时才 fallback 到原 planner。

**物理/行为含义**

原版智能体的“经验”只知道“为了 logs 以前成功过 chop a tree”，不知道当时有没有工具、在哪里、处于什么高度、背包里有什么、这条经验后来是否反复失败。plus 的案例库把经验从“静态配方动作表”变成“当前世界状态下的行动决策样本”。

在任务执行中，智能体不再只问“这个 waypoint 以前成功过吗”，而是开始具备“在类似状态下，这种行动是否值得复用”的接口。它仍然不是完整强化学习，但已经把后续训练决策器所需的 `(state, waypoint, action, outcome)` 闭环建起来了。

## 优化点 2：保留原版能力，同时把旧记忆迁移进新案例库

相关文件：`src/optimus1/memories/case_memory.py`

**代码逻辑实现**

plus 的 `CaseBasedMemory._bootstrap_from_legacy_waypoint_memory()` 会把原来的 `waypoint_to_sg/*.json` 动作计数迁移成 case，并写入 `legacy_bootstrap.done`，避免每次重复迁移。迁移时保留 action、success_count、failure_count，并构造 `source=legacy_waypoint_to_sg` 的状态快照。

**物理/行为含义**

这使 plus 不会因为换成案例库而丢掉原版已经积累的行动经验。智能体启动时仍能复用过去“某个 waypoint 该怎么做”的能力，但这批能力被包装进更丰富的案例结构里，后续可以与新运行产生的真实状态案例放在同一套检索与训练接口中。

## 优化点 3：案例复用从单条成功记录变成成功/失败聚合

相关文件：`src/optimus1/memories/case_memory.py`

**代码逻辑实现**

原版 `is_succeeded_waypoint()` 对某动作的判断是 `success > 0` 且 `success - failure > -plan_failure_threshold`。plus 的 `_best_exact_success_case()` 会聚合同 waypoint 下相同 action 的成功和失败历史：成功案例增加 score，失败案例降低 score；只有净分没有跌破失败阈值的动作才会被复用。

另外，plus 的 `select_case_decision()` 支持两级策略：

- exact waypoint 成功案例：高优先级复用。
- embedding 相似案例：必须同 waypoint 且相似度超过 `reuse_threshold`。

**物理/行为含义**

原版容易被“一条旧成功”长期绑架：即使后续同动作反复失败，只要没超过阈值，仍可能继续复用。plus 更像在做经验投票：如果某个动作历史上已经呈现失败趋势，智能体会减少直接复用，转而调用 planner。

这会让智能体在重复失败的任务上更愿意重新思考，而不是机械地执行过去一次成功过的动作。

## 优化点 4：RADS 决策器接入，案例库开始服务于可训练决策

相关文件：

- `src/optimus1/decisioner/feature.py`
- `src/optimus1/decisioner/encoder.py`
- `src/optimus1/decisioner/rads.py`
- `src/optimus1/decisioner/runtime.py`
- `scripts/export_decisioner_dataset.py`
- `scripts/train_rads.py`
- `scripts/evaluate_rads_offline.py`

**代码逻辑实现**

plus 新增 RADS，即 Retrieval-Augmented Decision Scorer。它把每个候选动作组织成 query case，然后由模型输出 `p_success`。特征来自 case 自身，包括 waypoint、final_goal、action、equipment、biome、waypoint_num、run 内位置、y 高度、关键 inventory 物品、工具拥有情况等。

RADS 模型结构：

- QueryEncoder 编码当前 query。
- CaseEncoder 编码案例库中的历史案例。
- query 与案例向量做 attention。
- attention 上下文与 action embedding 拼接后预测成功概率。
- 支持同 waypoint hard mask，避免 attention 漂到无关 waypoint。
- 支持 `(waypoint, action)` 经验成功率作为 residual prior。

运行时，`CaseBasedMemory` 若检测到 `memory.case_memory.decisioner.enabled=true`，就通过 `RADSRuntime` 对候选 action 排序；如果最高成功概率低于 `min_p_success`，直接 fallback 到 planner。

**物理/行为含义**

这一步把智能体从“检索到相似案例就照做”推进到“对每个候选动作估计成功概率”。它不只是选最像的案例，而是试图学习“当前状态下哪个动作更可能成功”。

在线比较报告显示，RADS 决策器单次运行 67 任务成功率为 49/67，而 retrieval-only baseline 的首轮成功率为 39/67。这个结果说明 plus 的决策层已经能在一些任务中避免原版机械复用的错误，尤其是在多动作 waypoint 或低置信 fallback 的场景里更明显。

## 优化点 5：规划入口从只看 inventory 扩展到完整 env_status

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

原版 `make_plan(original_final_goal, inventory, ...)` 只接收 inventory。plus 改为 `make_plan(original_final_goal, env_status, ...)`，内部再取 `env_status["inventory"]`，并把完整状态交给 `CaseBasedMemory.create_state_snapshot()`。

这使 planning 层可以看到：

- inventory 与 plain_inventory
- equipment
- location_stats
- resource_ledger
- recovery/control state 相关摘要

**物理/行为含义**

原版 planner 的世界观基本是“背包里有什么”。plus 的 planner/decisioner 能把“我现在拿着什么工具、在哪个高度、处于哪个 biome、是否刚挖到矿、背包槽位压力如何”等物理状态纳入决策。

例如同样是“mine iron_ore”，在 y=34 且有 stone_pickaxe，与 y=8 且没有合适镐，是完全不同的物理处境。plus 的状态接口为这种差异留出了表达空间。

## 优化点 6：waypoint 选择开始跳过已满足目标

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

原版直接取 `wp_list_str.splitlines()[1]` 作为第一个 waypoint。plus 新增 `_parse_waypoint_summary()`、`_inventory_count_for_waypoint()`、`_can_skip_satisfied_waypoint()`、`_select_next_planning_waypoint()`。

当 waypoint 已经在 inventory 中满足，并且属于可跳过类别时，plus 会选择下一个未满足 waypoint。执行前还会再次检查已满足 waypoint，若成立则直接记录成功，不再执行无意义动作。

**物理/行为含义**

原版可能出现“已经有 furnace 还继续 craft furnace”“刚做出 pickaxe 仍重复做 pickaxe”的行为。plus 会减少这类循环，让智能体更像一个会检查物资清单的玩家：已经拥有的关键工具或设备不重复制作，把时间用在下一步。

这对长链任务尤其重要，因为原版一次冗余 craft 可能消耗材料，反而破坏后续任务。

## 优化点 7：显式前置条件修正，防止 planner 过早选择不可执行 waypoint

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

plus 新增 `_planning_prereq_for_waypoint()`，下分两类：

- `_crafting_prereq_for_waypoint()`：针对 stone tools / furnace 等，需要 stick、crafting_table、cobblestone/blackstone/cobbled_deepslate。
- `_pickaxe_prereq_for_mining()`：针对 cobblestone、iron_ore、gold_ore、redstone、diamond 等，检查是否有足够 tier 的 pickaxe，必要时先补 stick、crafting_table、furnace、iron_ingot、stone_pickaxe 等。

`make_plan()` 对 planner 给出的 waypoint 最多展开 4 跳 prerequisite，避免无限递归。

**物理/行为含义**

原版可能在没有石镐时就去挖铁，或在没有木镐时就试图拿 cobblestone。plus 会先把物理世界中的工具链补齐：木头 -> 木镐 -> 圆石 -> 石镐 -> 铁矿 -> 铁锭 -> 铁镐 -> 钻石/红石。

这让智能体在 Minecraft 的物理规则上更接近真实玩家，不会对“没有合适工具的矿”做无效攻击。

## 优化点 8：动作语义可行性检查与 fallback subgoal

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

plus 新增 `_subgoal_action_is_feasible()` 和 `_fallback_subgoal_for_waypoint()`。无论动作来自 case 复用还是 planner，如果出现语义不匹配，比如 mining waypoint 却返回 craft/smelt 动作，plus 会拒绝该动作并生成保守 fallback。

**物理/行为含义**

原版更信任 planner 或记忆库文本，一旦文本动作与物理目标错配，智能体会执行错误类型动作。plus 增加了一层 Minecraft 语义过滤：矿物 waypoint 应该挖，木头应该砍，锭应该冶炼，工具/设备应该合成。

这不是让智能体更“聪明”地规划，而是防止低级动作类型错配造成明显失败。

## 优化点 9：craft helper 支持 tag 材料跨 stack 统计与放置

相关文件：`src/optimus1/helper/new_craft_helper.py`

**代码逻辑实现**

原版 helper 对 recipe tag 或材料检查时常通过 `find_in_inventory()` 找单个 slot，然后比较该 slot 数量。plus 新增：

- `_matching_inventory_slots()`
- `_available_inventory_quantity()`
- `_place_items_from_inventory_slots()`

可对同一个 recipe tag 的多个匹配 slot 求和，并在 shaped / shapeless crafting 时跨 slot 放入 resource slots。

典型例子：recipe 需要 `minecraft:logs` 4 个，背包里是 `oak_log: 1`、`birch_log: 3`。原版可能因为没有单 stack 满 4 而误判材料不足；plus 会把不同木头都视为 tag 匹配材料并合计。

**物理/行为含义**

Minecraft 合成系统允许 tag 材料互换，例如不同原木都能合成木板或作为某些配方材料。plus 修复后，智能体能更符合 Minecraft 配方物理规则，不会因为材料分散或木材种类不同而错误放弃合成。

这直接减少了“已经有材料但 helper 报 missing material”的假失败。

## 优化点 10：STEVE-1 prompt 切换时重置 recurrent hidden state

相关文件：`src/optimus1/models/steve_action_model.py`

**代码逻辑实现**

原版 `SteveActionModel.action()` 每次只换 prompt embedding，不重置底层 agent recurrent state。plus 新增 `_last_prompt`，当 prompt 变化时调用 `self.agent.reset(self.text_cond_scale)`。

**物理/行为含义**

STEVE-1 的动作策略不是纯前馈，它会带着前几帧的动作惯性。原版可能出现刚砍完树，切到挖矿 prompt 后仍延续跳跃、挥手、转向等旧动作模式。

plus 在每个新语言动作 prompt 开始时清空低层动作记忆，使“挖矿”“砍树”“找树”“向前挖”等 prompt 的开头动作更干净。智能体仍可能跳或转向，但这些动作更多来自当前 prompt 与视觉输入，而不是上一个 subgoal 的残留状态。

## 优化点 11：wrapper 获取当前 prompt，动作稳定从无语义变成语义相关

相关文件：

- `src/optimus1/main_planning.py`
- `src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 在执行 STEVE-1 action 时调用：

```python
env.step(action, current_sg_target, prompt=current_sg_prompt)
```

wrapper 的 `_stabilize_action()` 现在能通过 prompt 判断当前动作属于：

- surface resource acquisition：chop / punch / tree / logs / wood
- tunnel resource acquisition：mine / dig / ore / cobblestone / stone / diamond / redstone
- 普通移动或其他动作

基于语义，wrapper 触发不同的 attack hold、jump lock、movement lock、surface recovery、tunnel recovery 等。

**物理/行为含义**

原版 wrapper 看到的只是底层按钮，不知道“这次 attack 是砍树还是挖矿”。plus 让底层控制器知道高层意图：砍树可以保留一部分移动寻找目标，挖矿时则更严格锁住跳跃和横移，减少破坏方块过程中的视角/位置漂移。

这使智能体的身体动作更像“按任务类型稳定控制”，而不是所有动作一套规则。

## 优化点 12：连续 attack-hold，改善挖掘/砍伐稳定性

相关文件：`src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 在 `_stabilize_action()` 中检测到资源获取动作的 attack 后，把一次 attack 意图扩展为多个 tick 的 `attack=1`。木头默认 hold 更久，挖矿 hold 稍短。后续 hold 帧还会根据任务类型锁定 jump、sprint、sneak、横移、camera 等。

**物理/行为含义**

Minecraft 破坏方块需要持续按住攻击，短点击经常只产生挥手但不破坏方块。原版完全交给 STEVE-1，容易出现“一下一下点但没有持续破坏”的情况。

plus 的 attack-hold 让智能体在接触树干或矿石后更像玩家长按鼠标，实际破坏方块的概率更高。对 logs、cobblestone、ore 这类资源获取 waypoint 都有直接帮助。

## 优化点 13：资源账本 resource_ledger 引入非视觉事实记忆

相关文件：

- `src/optimus1/env/custom_env.py`
- `src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 在环境 observables 中新增 `pickup` 和 `mine_block` full stats。wrapper 初始化 `resource_ledger`，并在每步 `_record_resource_ledger()` 中记录：

- 当前 inventory 的正向 delta
- 历史最大 inventory
- pickup stat delta
- mine_block stat delta

`get_status()` 会把 `resource_ledger` 暴露给 planning/case memory/result JSON。

**物理/行为含义**

原版主要依赖当前 inventory 和视觉/任务 checker。plus 多了“我曾经挖断过什么”“我曾经捡到过什么”“某物最多曾到过多少”的事实账本。

这在 Minecraft 里很关键：矿石或木头可能已经被打掉但还没捡起，物品可能短暂进入 inventory 后被合成消耗，视觉也可能没看到掉落物。资源账本让智能体对物理进展有更强的非视觉感知。

## 优化点 14：账本 fallback 被限制在非消耗型目标，避免假完成

相关文件：`src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 的 `_ledger_satisfies_goal()` 不是对所有物品都用历史最大值判断完成，而是通过 `_LEDGER_FALLBACK_GOAL_ITEMS` 限制在工具、设备、成品等非消耗型目标。logs、planks、stick、cobblestone、ores、ingots 等消耗品必须看 live inventory。

**物理/行为含义**

这修复了资源账本的一个潜在副作用：如果 logs 曾经达到 4 个，之后被合成木板，历史最大值仍是 4。若直接用历史最大值判断，就会让智能体误以为 logs waypoint 已完成，进入空转或错误 replan。

plus 的设计更符合 Minecraft 物资流：原料会被消耗，工具和成品才更适合“曾经获得过即可视为拥有/完成”的 fallback。

## 优化点 15：掉落物收集与背包压力清理

相关文件：`src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 新增两类动作原语：

- `_should_collect_drops()` / `_collect_drop_action()`：当账本显示目标方块被挖掉但 inventory/pickup 没跟上，触发短暂 forward 收集。
- `_maybe_cleanup_inventory()`：当 inventory slot 接近满时，丢弃低优先级 hotbar 物品，同时保护工具、关键材料、当前目标物品等。

**物理/行为含义**

原版可能把矿石打掉后站得太远，掉落物没有进入背包；也可能因为种子、花、树叶、泥土等杂物占满槽位，导致关键资源捡不起来。

plus 让智能体具备两种玩家式反应：挖掉东西后向前走去捡，背包快满时扔掉低价值杂物。这对长任务中矿物、红石、钻石、铁锭等关键物资尤其重要。

## 优化点 16：surface 行为从纯 STEVE-1 导航变成可恢复的找树/砍树流程

相关文件：

- `src/optimus1/main_planning.py`
- `src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 在树木任务中维护 `tree_mode`：

- 初始使用原始 chop prompt。
- 若长时间没有 log 相关进展，切到 `find a tree`。
- 若检测到 log inventory/pickup 进展或持续 attack 接触，切回原 chop prompt。

wrapper 还新增 surface turn-around 与 ground pitch clamp：

- surface 卡住时做 180 度转向。
- 非攻击地面探索时限制 pitch，减少看天或看脚导致导航失效。
- 拿到 log 后短时间锁 jump，减少刚捡到木头后跳走。

**物理/行为含义**

原版执行 `chop a tree` 时，如果附近没有树或视角飘掉，STEVE-1 可能长期无效游走。plus 的行为更像玩家：砍不到树就先找树；接触到树或拿到木头后恢复砍伐；卡在坡/墙上时转身；探索时尽量保持视线接近水平。

这主要改善 logs 相关早期失败。logs 是大量 crafting 任务的第一步，稳定性收益会传递到后续铁、金、钻石、红石任务。

## 优化点 17：地下挖矿加入高度感知、overshoot 检测与 pillar-up

相关文件：

- `src/optimus1/main_planning.py`
- `src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 增加矿层相关逻辑：

- `ORE_LAYER_ORDER` 和 ore alias：区分 coal、iron、gold、redstone、diamond 的层级。
- `_effective_mining_target_ore()`：将 cobblestone 的 pillar-up 参考映射到 iron_ore。
- `_maybe_pillar_up_for_ore()`：可选的开局高度检查。
- `_maybe_relevel_for_overshoot()`：检测挖过头后 pillar up 回目标高度。
- `perceive_height_context()`：读取当前 y、目标矿层、可放置方块数量。
- `pillar_up_smart()` / `raise_to_height()` / `raise_to_ore_band()`：用可放置方块把智能体垫回目标高度。
- `dig_forward_blocks()`：pillar-up 后脚本化挖出 2 格高前进通道，再交还给 STEVE-1 的 `dig forward and mine X` prompt。

触发条件包括：

- 已看到比当前目标更深的矿。
- 当前 y 低于目标矿层下界一定 margin。
- 接近 bedrock 且长期没有矿物进展。

**物理/行为含义**

原版 `dig down and mine X` 容易一路向下挖，错过目标矿层后继续挖到更深区域甚至基岩附近。plus 加入了“矿层物理常识”：挖铁时如果已经拿到红石/钻石，说明挖太深；挖到 y 太低仍没有目标矿，也说明需要回到更合适高度。

行为上，智能体会从“只会向下挖”变成“发现过深后垫高，再水平探索”。这对 iron、gold、redstone、diamond 这类层级资源尤其关键。

## 优化点 18：挖矿前自动装备最高 tier pickaxe

相关文件：

- `src/optimus1/main_planning.py`
- `src/optimus1/env/wrapper.py`

**代码逻辑实现**

plus 新增 `_best_pickaxe_from_status()`、`_has_capable_pickaxe_for_target()`、`_ensure_best_pickaxe_equipped()`。在进入 pickaxe-mining subgoal 前，系统检查背包中最佳镐，并通过 helper 执行 `equip <best_pickaxe>`。

wrapper 的 `find_best_pickaxe()` 也在 hotbar 操作受限时尽量选择最佳 pickaxe。

**物理/行为含义**

Minecraft 中不同矿物有最低镐等级要求：铁矿需要石镐，钻石/红石/金矿需要铁镐或更高。原版即使背包里有更好的镐，也可能没有装备到手上，导致挖矿无效或效率极低。

plus 会在挖矿前主动换工具，让智能体更像玩家：挖矿前先确认手里拿的是能挖动目标的镐。

## 优化点 19：死亡/重生检测与策略重置

相关文件：

- `src/optimus1/env/wrapper.py`
- `src/optimus1/main_planning.py`

**代码逻辑实现**

plus 在 `_record_step_state()` 中记录 health、is_alive、位置窗口。通过两类信号检测重生：

- health 从低值恢复到满血且位置发生大跳变。
- `is_alive` 从 False 变 True。

触发后清空 attack_hold、escape、tunnel_recovery、surface_search、surface_turn_around、collect_drop 等低层控制状态，并设置 `policy_reset_requested`。`new_agent_do()` 读取该请求后 reset action server，并恢复当前 subgoal prompt。

**物理/行为含义**

原版死亡/重生后可能继续带着旧动作惯性：仍在 attack hold、仍认为自己在上一个 tunnel 或 surface recovery 中。plus 会把重生看成物理状态断点：身体回到出生点，低层动作状态必须清空，高层当前目标要重新下发。

这减少了死亡后“无意义延续旧动作”的情况，尤其对落岩浆、窒息、摔落等突然死亡场景有帮助。

## 优化点 20：任务 checker 异常不再默认为完成

相关文件：`src/optimus1/env/wrapper.py`

**代码逻辑实现**

原版 `wrapper.step()` 中 task checker 抛异常时会 `self._current_task_finish = True`。plus 改为记录 `checker_error` 并保持 `self._current_task_finish = False`。

**物理/行为含义**

原版存在危险的假阳性：检查器出错时，智能体会以为 subgoal 完成，继续推进后续任务。plus 更保守：检查不出来就不算完成。

这会减少“其实没拿到物品但流程继续走”的连锁错误。代价是某些 checker bug 可能导致真实完成被延后确认，但这比误判成功更安全。

## 优化点 21：Malmo/MineRL 环境异常从能力失败中分离

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

原版只要 Malmo log 中出现 `Exception` 就返回 `env_malmo_logger_error`，很多非致命 Minecraft 日志也会导致直接退出。plus 新增 `_malmo_log_has_fatal_error()`，只把 OutOfMemory、Minecraft crashed 等明确 fatal pattern 作为致命；普通 `Exception` 只记录 warning。

执行过程中如果 `env.step()` 返回 `info["error"]` 并 game_over，plus 把状态标记为 `env_step_timeout`，而不是普通 `timeout_non_programmatic`。

**物理/行为含义**

MineRL/Malmo 本身不稳定，早停和 step timeout 并不代表智能体不会做任务。plus 把“环境基础设施失败”与“智能体能力失败”拆开。

这让案例库训练更干净：环境几秒钟早停不应该作为“chop a tree 失败”的负样本污染决策器。智能体的物理能力统计也因此更可信。

## 优化点 22：pending case 的成功/失败闭环更完整

相关文件：`src/optimus1/memories/case_memory.py`、`src/optimus1/main_planning.py`

**代码逻辑实现**

plus 在每次 planner 或 case decision 被采纳时先 `record_decision()` 为 pending case。subgoal/waypoint 结束后调用 `save_success_failure()` 更新 outcome。

如果 run 失败但不是 infra early stop，`mark_pending_cases_failed()` 会把本 run 未收尾的 pending case 标记失败。若是 `env_step_timeout` 或几秒钟 logs 早停，`discard_pending_cases()` 删除这些 pending cases，避免基础设施失败污染训练。

**物理/行为含义**

原版只累加动作成功/失败计数，无法追踪一次“当时为什么这么选、后来发生什么”。plus 把一次决策从“选择动作”到“执行后结果”闭合起来。

对智能体行为来说，这意味着每次真实行动都会变成后续可学习样本；但明显非智能体原因导致的异常会被剔除。这样案例库更像实验日志与训练数据的统一来源。

## 优化点 23：视频记录更可靠，失败视频命名更接近真实失败点

相关文件：

- `src/optimus1/env/mods/recorder.py`
- `src/optimus1/main_planning.py`

**代码逻辑实现**

plus 的 recorder 改为：

- 记录 raw POV frame，不再把 prompt 文本 overlay 进视频。
- 保存前 snapshot frames，避免异步保存时帧列表继续被修改。
- 写出后检查 decoded frame count、file size、frame shape。
- 没有帧时跳过导出并 warning。

`main_planning.py` 新增 `_video_task_name()` 和 `_video_action_name()`。失败时优先用 failed_subgoals 命名视频，而不是用最后一个 completed_subgoal。

**物理/行为含义**

原版失败视频容易命名成上一个成功子任务，例如任务失败在 `dig_down_and_mine_iron_ore`，视频却叫 `craft_stone_pickaxe`。plus 的视频更容易对应真实失败动作。

这对后续调参很重要：你能更快定位智能体到底是在找树失败、挖矿卡住、craft helper 报错，还是环境早停。

## 优化点 24：结果 JSON 增加 recovery_events、resource_ledger、inventory_slots_used

相关文件：`src/optimus1/main_planning.py`

**代码逻辑实现**

plus 在 result JSON 中新增：

- `status_detailed`
- `infra_early_stop`
- `recovery_events`
- `resource_ledger`
- `inventory_slots_used`

**物理/行为含义**

原版结果只告诉你成功/失败和 subgoal 列表，plus 能告诉你失败过程中身体控制器做了什么：是否触发过 movement_escape、collect_drops、pillar_up、surface_turn_around，是否挖断过目标方块但没捡到，背包是否接近满。

这让任务失败从一个黑盒结果变成可诊断的物理行为轨迹摘要。

## 优化点 25：统一 PerceptionActionSuite，支持一键启用/消融

相关文件：

- `src/optimus1/env/perception_action.py`
- `src/optimus1/main_planning.py`
- `src/optimus1/env/__init__.py`

**代码逻辑实现**

plus 新增 `PerceptionActionSuite`，通过 `XENON_PERCEPTION_ACTION_SUITE` 统一控制一组环境感知与动作修复功能。它使用 `os.environ.setdefault()`，所以用户显式设置的单项 env var 不会被覆盖。

覆盖的功能包括：

- overshoot pillar-up
- tree explore
- surface turn-around
- ground pitch clamp
- inventory cleanup
- collect drops
- movement escape
- tunnel recovery

**物理/行为含义**

这不是直接改变单个动作，而是让整个实验体系可以清楚地区分“原始执行层”和“增强执行层”。你可以一键打开 plus 的环境感知动作系统，也可以逐项关闭做消融。

对研究来说，这保证每个行为优化都能被实验证明，而不是混杂在不可控的环境变量里。

## 优化点 26：FastAPI agent server 的重试从无限裸循环变成有界错误

相关文件：`app.py`

**代码逻辑实现**

原版对 decomposed_plan、context_aware_reasoning、retrieval、plan、fixjson、reflection、replan 等请求使用无界 `while True` 或手写 retry。plus 新增 `_agent_call_with_retries()`，统一：

- 最大重试次数
- retry sleep
- traceback 打印
- 最终抛出 HTTP 502

**物理/行为含义**

这不是 Minecraft 内的物理动作优化，但会影响智能体任务执行的稳定性。原版如果 VLM/agent server 某个调用持续失败，可能无限卡死；plus 会在重试上限后明确失败，让主流程可以记录 crash/failed，而不是无声挂起。

对长批量实验尤其重要，因为一个坏请求不会无限占住整轮 benchmark。

## 优化点 27：批跑脚本加入异常重跑、server 管理和单次公平对比

相关文件：

- `scripts/run_task_v3.sh`
- `scripts/run_v3_full_benchmark.sh`
- `scripts/rerun_*.sh`
- `scripts/verify_pillar_up*.py`

**代码逻辑实现**

plus 增加一批运行脚本，核心能力包括：

- 自动设置 PYTHONPATH、HF 离线路径、Qwen backend、CUDA。
- 可切换 `DECISIONER_ENABLED`。
- 自动启动/检查 app server。
- 批跑 67 个任务。
- 对 `env_step_timeout`、`crash_*`、无 result JSON 的异常运行删除产物并重试。
- 支持 `PERCEPTION_ACTION_SUITE` 一键消融。
- 断点续跑跳过已有结果。

**物理/行为含义**

这些脚本不改变单个智能体动作，但改变实验可信度。原版更容易把环境异常和能力失败混在一起，或者由于 app server/Malmo 残留进程导致批量结果不稳定。

plus 的实验流程更接近可复现实验：基础设施异常重跑，正常任务失败保留，单次 vs 单次对比明确区分，不拿 best-of-N 去和 single-shot 混比。

## 优化点 28：线上与离线评估闭环已经建立

相关文件：

- `reports/decisioner/offline_eval_v2.md`
- `reports/decisioner/v2_vs_v3_comparison.md`
- `docs/DECISIONER_*`

**代码逻辑实现**

plus 增加从 cases.json 导出数据、训练 RADS、离线评估、在线 67 任务对比的完整路径。离线报告包含 AUC、AP、F1、ECE、attention 诊断、per-waypoint AUC、多动作 ranking 等。在线报告明确比较 retrieval-only first attempt 与 RADS decisioner single-shot。

**物理/行为含义**

这让 plus 的优化不只是“代码上加了模块”，而是有任务行为结果反馈。当前结果显示：

- v2 retrieval-only 首次 39/67。
- v3 RADS decisioner 首次 49/67。
- 红石、钻石、铁、木制任务有明显净增。
- 仍有 logs、iron_ore、gold_ore 等执行层随机失败，说明决策器并没有解决所有低层控制问题。

也就是说，plus 已经形成了一个“运行 -> 记录案例 -> 训练决策器 -> 在线验证 -> 继续诊断执行层失败”的闭环。

## 仍然保留的原版骨架与边界

plus 没有替换所有核心模块。以下部分基本仍继承原版思路：

- OracleGraph 仍是配方/waypoint 的主要来源。
- VLM planner 仍作为 fallback，而不是被完全替换。
- STEVE-1 仍是视觉动作策略主体；plus 主要在 prompt、reset、wrapper 原语层做调度与稳定。
- helper 仍执行 craft/smelt/equip 的 GUI 操作；plus 修复材料统计与跨 slot 放置，但没有重写 GUI 自动化。

当前主要边界：

- RADS 对多动作 waypoint 最有价值，但当前大多数 waypoint 仍只有一个候选动作，决策器发挥空间有限。
- 执行层仍有随机性，logs、iron_ore、gold_ore 等资源获取失败并未完全消除。
- pillar-up、dig-forward、surface-turnaround 等脚本原语引入了更强物理先验，后续需要消融实验确认每项收益。
- resource_ledger 是强信号，但必须继续谨慎限制在非消耗目标上，否则容易造成假完成。
- 大量案例数据质量直接影响决策器；环境早停、旧 bug 样本、级联失败样本需要持续治理。

## 总结

XENON-plus 的优化可以概括为四条主线：

1. **经验表示升级**：从 `waypoint -> action count` 变成可训练的状态化 case。
2. **决策层升级**：从 exact/cosine 复用变成 RADS 成功概率评分，并保留 planner fallback。
3. **执行层物理化**：加入 prompt-aware action stabilization、资源账本、掉落收集、背包清理、找树恢复、矿层感知、pillar-up、死亡重置。
4. **实验闭环升级**：区分基础设施异常与能力失败，记录更丰富 telemetry，支持批量公平对比与消融。

从任务表现看，plus 的智能体会比原版更像一个“会检查状态、会复用经验、会在低置信时重新规划、会用 Minecraft 物理规则修正动作”的 agent。它不是让底层 STEVE-1 本身变强，而是在 STEVE-1 外围加上了状态感知、动作稳定、错误分类和训练闭环，使同一个低层策略在复杂长链任务中更少被无效动作、错误记忆和环境异常拖垮。
