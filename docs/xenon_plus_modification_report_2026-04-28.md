# XENON-plus 相比原版 XENON 的修改报告

生成日期：2026-04-28  
项目位置：`/home/yzb/zhourong/XENON-plus`  
对比基线：`/home/yzb/zhourong/XENON-main`

## 1. 当前结论

XENON-plus 当前不是对原版 XENON 的整体重写，而是在保留原版 planner、benchmark、MineRL/STEVE-1 执行骨架的基础上，做了三类核心修改：

1. 将原版简单动作记忆库替换为面向决策器训练的案例库。
2. 加入初版检索式决策器：先检索案例库，有可复用案例就直接决策，没有再回退到原 planner。
3. 修正执行层稳定性问题：动作状态重置、连续挖掘/砍伐、脱困、非视觉资源账本、背包压力处理、环境早停分类。

因此，XENON-plus 目前的研究重点已经从“planner 每次规划”转向“案例库辅助决策 + planner fallback + 执行层稳定性增强”。

## 2. 项目层面的修改

### 2.1 物理隔离

原版项目保留在：

```text
/home/yzb/zhourong/XENON-main
```

新项目位于：

```text
/home/yzb/zhourong/XENON-plus
```

这样后续可以做原版 XENON 与 XENON-plus 的对比实验，不会互相污染代码、记忆库和实验结果。

### 2.2 GitHub 维护策略

当前 GitHub 维护对象是 XENON-plus，而不是原始 XENON-main。

当前主要提交包括：

```text
6306381 feat: add case memory decisioner and stone iron results
a757748 fix: classify infra timeouts and rerun bug-affected tasks
93bcb7f feat: stabilize execution logic and add gold diamond results
```

视频文件仍保留本地，不上传 GitHub。GitHub 主要保存：

- 代码
- 案例库
- 有效实验 JSON
- 实验总结文档

## 3. 记忆库与决策流程修改

### 3.1 原版 XENON 的记忆库

原版使用 `DecomposedMemory`，核心逻辑是：

```text
waypoint 有历史成功动作 -> 直接复用
waypoint 没有历史成功动作 -> 调用 planner 生成动作
```

原版动作库主要记录：

- waypoint
- action
- success/failure 计数

问题是上下文非常少，不能很好支持后续训练一个“在特定环境下选择动作”的决策器。

### 3.2 XENON-plus 的案例库

XENON-plus 新增：

```text
src/optimus1/memories/case_memory.py
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

案例库记录的信息更接近“决策训练样本”：

- 原始最终目标
- 当前 waypoint
- 当前环境与智能体状态
- inventory
- plain_inventory
- equipment
- location_stats
- biome
- resource_ledger
- 候选动作
- 选中的动作
- 选中原因/决策 trace
- 执行结果
- success/failure/status

这已经不只是“动作库”，而是为后续训练决策器准备的数据结构。

### 3.3 初版检索式决策器

当前 XENON-plus 的动作选择流程变为：

```text
当前任务 + 当前环境状态 + 当前 waypoint
        |
        v
构造 state_snapshot
        |
        v
检索案例库
        |
        +-- 有可信案例 -> 复用案例动作
        |
        +-- 没有可信案例 -> 调用原 planner
```

原 planner 没有被替换，只作为 fallback 使用。这样可以保证 XENON-plus 理论上仍覆盖原版 XENON 的能力边界，同时为后续训练更强决策器积累案例。

## 4. 案例库质量控制修改

### 4.1 环境早停与真实失败分离

之前 MineRL/Malmo 会出现几秒钟内 `done=True` 的情况，例如：

```text
Failed to take a step (error timed out)
```

旧逻辑会把这种情况当成普通任务失败。现在 XENON-plus 将其标记为：

```text
env_step_timeout
```

这类记录不进入正式能力统计，也不作为决策器训练的失败样本。

### 4.2 已识别 bug 的机制修复，而不是只屏蔽旧样本

这里需要强调：`superseded_bug` 或移出有效训练样本只是旧数据治理，不是主要解决方案。真正要做的是让同类问题在后续实验流程中不再被误判、误记或重复污染案例库。

目前已经做的机制修复包括：

1. smoker/furnace 相关 helper 问题
   - 原问题：craft/smelt helper 在部分 recipe tag 和材料 stack 分散时会误判材料不足，导致已经具备前置材料但后续合成失败。
   - 修复位置：`src/optimus1/helper/new_craft_helper.py`
   - 修复内容：tag 材料不再只看单个 stack，而是统计多个匹配 inventory slot 的总量，并支持跨 stack 放置 shaped/shapeless recipe 材料。

2. furnace/shears 等几秒钟无执行失败
   - 原问题：MineRL/Malmo 偶发在很早阶段返回 step error 或 done，旧流程会把它当成普通失败。
   - 修复位置：`src/optimus1/env/wrapper.py`、`src/optimus1/main_planning.py`、批量实验脚本。
   - 修复内容：环境 step error 被分类为 `env_step_timeout`，不作为智能体能力失败；批量脚本遇到 `env_step_timeout` 或几秒钟 `logs` 早停会自动重试，直到拿到有效运行或达到重试上限。

3. 失败视频命名成上一个成功子任务
   - 原问题：失败时视频名取 `completed_subgoals[-1]`，导致 Iron 失败视频常被命名为 `craft_stone_pickaxe`。
   - 修复位置：`src/optimus1/main_planning.py`
   - 修复内容：失败时优先使用 `failed_subgoals[0]` 命名视频；只有成功时才使用最后完成的 subgoal。

旧案例被标注或移除的目的，是防止历史错误继续进入后续决策器训练；但这不是替代修复，后续正式实验应以修复后的机制重新产生有效案例。

### 4.3 精确案例复用改进

案例库在 exact waypoint 复用时，不再只看单个成功案例，而是聚合同一动作的成功/失败历史，避免一条旧成功案例压过后续大量失败案例。

## 5. Planner 与 helper 修改

### 5.1 Planner 保持原样

当前没有重写原始 planner。XENON-plus 的创新点不是让 planner 直接变强，而是让“是否使用某个行动方案”的决策过程逐步从案例库中学习。

### 5.2 Craft helper 的 tag 材料修复

修复位置：

```text
src/optimus1/helper/new_craft_helper.py
```

典型问题：

```text
birch_log: 3
oak_log: 1
```

Minecraft recipe tag `minecraft:logs` 需要 4 个 logs。旧 helper 只看单个 stack，导致没有单个 stack 数量达到 4 时误判材料不足。

现在 shaped/shapeless recipe 都支持跨多个匹配 stack 统计和放置 tag 材料。

## 6. 执行层修改

执行层主要修改在：

```text
src/optimus1/env/wrapper.py
src/optimus1/main_planning.py
src/optimus1/models/steve_action_model.py
```

### 6.1 prompt 切换时重置 STEVE-1 hidden state

STEVE-1 不是每一帧都独立决策的纯前馈模型。它的策略网络内部保留 recurrent hidden state，用来承接前几帧的动作上下文。例如前面几帧在靠近树、转视角、挥手，后面几帧会受到这个短期状态影响。

原流程的问题是：

```text
subgoal A: chop a tree
        |
        | STEVE-1 hidden state 持续积累“靠近树/跳跃/挥手/调整视角”的动作上下文
        v
subgoal B: dig down and mine iron_ore
        |
        | 只换文本 prompt，但 hidden state 没清空
        v
模型仍可能带着上一个 subgoal 的动作惯性输出短跳、短点击或无意义转向
```

这不是简单的“模型偶尔乱动”，而是跨 subgoal 的状态污染。文本 prompt 已经换了，但 recurrent state 还带着上一个任务的动作历史，导致新任务开头的动作分布不干净。

XENON-plus 的修改是：

```text
if prompt != last_prompt:
    reset STEVE-1 recurrent state
    last_prompt = prompt
```

修复位置：

```text
src/optimus1/models/steve_action_model.py
```

这样每个新的 language action prompt 都从干净的策略状态开始。它解决的是“跨任务动作惯性”，不是人为禁止某个动作。智能体仍然可以跳、点击、转向，但这些动作必须来自当前 prompt 和当前视觉输入，而不是上一个 subgoal 残留的 hidden state。

这个修改对三类现象有帮助：

1. subgoal 切换后仍延续上一任务动作，例如砍树后进入挖矿仍短跳/挥手。
2. context-aware reasoning 改写 prompt 后，新 prompt 初期动作受旧 prompt 干扰。
3. 同一 episode 内多个 waypoint 串行执行时，后一个 waypoint 开头动作不稳定。

它不能单独保证“永远不空跳”，因为模型本身可能在当前视觉输入下选择跳跃；所以后面又配合了 wrapper 层的 attack-hold、movement recovery 和 inventory ledger。hidden state reset 解决的是状态污染这一层问题。

### 6.2 当前 prompt 传入 wrapper

原版执行层只看到 action，不知道当前动作语义。现在 `main_planning.py` 将当前 `current_sg_prompt` 传入 `env.step(...)`。

这样 wrapper 可以区分：

- `chop`
- `mine`
- `dig`
- `break`
- 普通移动

从而做语义相关的底层动作稳定。

### 6.3 连续 attack-hold

问题：

Minecraft 中破坏方块需要持续按住 attack，一帧短点击经常无法完成挖掘/砍伐。

修改：

对于资源获取动作，执行层会把一次 attack 意图扩展为多个 tick 的持续 attack。

关键点：

- 第一帧保留模型自己的移动意图，避免阻止靠近目标。
- 后续补帧稳定 attack，减少移动/camera 漂移。

### 6.4 通用脱困原语

问题：

智能体被困在水里或卡在局部环境时，没有自救动作。

修改：

加入基于非视觉状态的 recovery primitive：

- 使用 `life_stats.air` 判断是否处于明显不利状态。
- 使用短窗口位置变化判断是否有移动意图但没有位移进展。
- 触发后执行 bounded `forward + jump + sprint + small turn`。

这不是针对某个场景写死规则，而是基于“状态恶化/位移停滞”的通用恢复。

### 6.5 非视觉资源账本

问题：

智能体可能因为背包满或视觉误判，不知道物资已经进入 inventory，从而继续重复挖掘。

修改：

新增 `resource_ledger`：

- 从 `plain_inventory` 记录物品事实。
- 记录 max observed inventory。
- 记录 positive deltas。
- waypoint 判断可以参考 ledger，而不只依赖视觉。

这样后续案例库也能记录更完整的非视觉状态。

### 6.6 背包压力处理

问题：

树叶、草、花、种子等杂物会占背包格子，影响后续矿物拾取。

修改：

当背包接近满时，执行层会尝试丢弃 hotbar 中低价值杂物：

- seeds
- flowers
- grass
- leaves
- saplings
- dirt
- sand
- gravel

保护物品包括：

- 工具
- logs/planks/stick
- crafting_table/furnace/chest
- ores/ingots/diamond/redstone/coal
- 当前目标物品

实现方式是正常 hotbar/drop 动作，不是 `/clear` 命令。

## 7. 视频与实验记录修改

### 7.1 视频目录命名

视频目录从直接任务名改为：

```text
任务种类_任务
```

例如：

```text
Wood_Craft_a_bowl
Stone_Craft_a_smoker
Iron_Craft_a_bucket
Gold_Craft_a_golden_shovel
Diamond_Craft_a_diamond_pickaxe
```

### 7.2 失败视频命名修复

旧逻辑失败时用最后成功的 subgoal 命名视频，因此 Iron 失败视频经常叫：

```text
craft_stone_pickaxe
```

现在失败时优先用失败 subgoal 命名，例如：

```text
dig_down_and_mine_iron_ore
```

## 8. Gold 和 Diamond 是否跑完

结论：Gold 6 个任务和 Diamond 7 个任务都已经执行过一轮。

之前说“Gold 4 成功 / 1 失败、Diamond 4 成功 / 0 失败”是“有效结果统计”，不是“任务没有跑完”。我当时把 `env_step_timeout` 环境早停从有效成功/失败中排除了，导致表达上容易误解。

### 8.1 Gold 6 个任务实际执行结果

| Exp | Task | Status | Result | 是否有效能力样本 |
|---:|---|---|---|---|
| 7300 | `craft_a_golden_shovel` | `success` | 成功 | 是 |
| 7301 | `craft_a_golden_pickaxe` | `timeout_non_programmatic` | 失败 | 是 |
| 7302 | `craft_a_golden_axe` | `env_step_timeout` | 环境早停 | 否 |
| 7303 | `craft_a_golden_hoe` | `success` | 成功 | 是 |
| 7304 | `craft_a_golden_sword` | `success` | 成功 | 是 |
| 7305 | `smelt_and_craft_a_gold_ingot` | `success` | 成功 | 是 |

所以 Gold 是：

```text
已执行：6/6
有效结果：5/6
有效成功：4
有效失败：1
环境早停：1
```

### 8.2 Diamond 7 个任务实际执行结果

| Exp | Task | Status | Result | 是否有效能力样本 |
|---:|---|---|---|---|
| 7400 | `craft_a_diamond_shovel` | `success` | 成功 | 是 |
| 7401 | `craft_a_diamond_pickaxe` | `success` | 成功 | 是 |
| 7402 | `craft_a_diamond_axe` | `success` | 成功 | 是 |
| 7403 | `craft_a_diamond_hoe` | `success` | 成功 | 是 |
| 7404 | `craft_a_diamond_sword` | `env_step_timeout` | 环境早停 | 否 |
| 7405 | `dig_down_and_mine_a_diamond` | `env_step_timeout` | 环境早停 | 否 |
| 7406 | `craft_a_jukebox` | `env_step_timeout` | 环境早停 | 否 |

所以 Diamond 是：

```text
已执行：7/7
有效结果：4/7
有效成功：4
有效失败：0
环境早停：3
```

### 8.3 为什么 GitHub 上看起来没做完

原因是当前上传策略是：

```text
上传有效实验 JSON
不上传 env_step_timeout 作为正式实验结果
视频不上传
```

`env_step_timeout` 对应的 JSON 和视频在本地保留，用于排查环境问题，但不作为 canonical 成功/失败记录上传。  
因此 GitHub 上只看到 Gold 的 5 个有效 JSON、Diamond 的 4 个有效 JSON，看起来像没有跑完；实际上本地已经跑完，缺失的是环境早停任务的正式有效结果。

## 9. 当前仍存在的问题

1. MineRL/Malmo 后端仍会随机早停。
   - 表现为 `env_step_timeout`。
   - 这不是智能体能力失败。
   - 后续正式实验建议对这类任务自动重试，直到得到有效运行。

2. 部分有效失败仍卡在 `logs`。
   - 例如 Iron 的 `tripwire_hook`、`iron_nugget`、`blast_furnace`。
   - 这需要结合本地视频判断是低层导航、树木定位、还是 recovery 原语干扰。

3. recovery primitive 已触发，但还需要通过视频继续调参。
   - 初始低空气阈值过敏，已从 `air < 300` 调整为 `air < 280`。

4. 资源账本已经加入，但还需要更多实验观察它对“背包满导致物品进入库存但智能体不知情”的改善程度。

## 10. 下一步建议

1. 对 Gold `golden_axe`、Diamond `diamond_sword`、`dig_down_and_mine_a_diamond`、`jukebox` 做 env_step_timeout 自动重跑，直到获得有效结果。
2. 单独复查 Iron 中失败在 `logs` 的视频，判断是否是动作模型导航问题。
3. 在 result JSON 中显式增加 `valid_for_analysis` 字段，避免以后“已执行”和“有效结果”混淆。
4. 将 `resource_ledger` 的关键摘要写入实验结果 JSON，方便后续不打开完整 case memory 也能分析资源获取状态。

补充更新：

- `scripts/run_execution_logic_benchmarks.sh`、`scripts/run_remaining_benchmarks.sh` 和 `scripts/rerun_bug_affected_tasks.sh` 已加入或修正有效运行重试逻辑。
- 默认最多重试 3 次，可通过 `XENON_MAX_VALID_ATTEMPTS` 调整。
- 重试触发条件：`status_detailed == env_step_timeout`，或步数小于 300 且失败 waypoint 只有 `logs`。
