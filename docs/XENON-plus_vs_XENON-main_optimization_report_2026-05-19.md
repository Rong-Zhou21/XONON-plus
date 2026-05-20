# XENON-plus 相比 XENON-main 的三类核心优化对比

生成日期：2026-05-19
对比目录：

- Plus：`/home/yzb/zhourong/XENON-plus`
- Main：`/home/yzb/zhourong/XENON-main`

## 0. 对比口径

本报告只围绕 XENON-plus 的三类核心创新展开：

1. 案例库
2. 决策器
3. 环境感知与行动

写法按“原 XENON-main 是什么样、XENON-plus 优化成什么样、解决了什么问题、举例说明”的结构组织。原报告中零散列出的脚本、视频、重试、日志、结果文件等内容不再作为独立创新点展开；只有当它们直接支撑这三类创新时，才作为辅助证据出现。

XENON-main 的核心闭环是：

```text
OracleGraph 生成 waypoint
  -> DecomposedMemory 查询 waypoint 的历史 action
  -> 如果该 waypoint 有成功 action，直接复用
  -> 否则调用 VLM planner 生成 action
  -> helper / STEVE-1 执行
  -> 将 waypoint-action 的 success/failure count 写回记忆
```

XENON-plus 保留 OracleGraph、VLM planner、helper、STEVE-1 和 MineRL/Malmo 的基础骨架，但把外围闭环升级为：

```text
环境状态感知
  -> 状态化案例库
  -> RADS 决策器打分
  -> 语义/前置条件/执行层修正
  -> 真实 outcome 写回案例库
  -> 离线训练再接入在线实验
```

一句话概括：原 XENON-main 主要是“有经验就复用，没有经验就问 planner”；XENON-plus 变成“记录状态化经验，用决策器判断是否适合复用，并用环境感知层让动作更符合 Minecraft 物理状态”。

## 1. 优化方向一：案例库

相关文件：

- Main：`/home/yzb/zhourong/XENON-main/src/optimus1/memories/decomposed_memory.py`
- Plus：`/home/yzb/zhourong/XENON-plus/src/optimus1/memories/case_memory.py`
- Plus 调用点：`/home/yzb/zhourong/XENON-plus/src/optimus1/main_planning.py`
- Plus 存储：`/home/yzb/zhourong/XENON-plus/src/optimus1/memories/ours_planning/v1/case_memory/cases.json`

### 1.1 原 XENON-main：动作记忆是 waypoint 级计数表

XENON-main 的 `DecomposedMemory` 存储单位很小，核心结构是：

```text
waypoint_to_sg/<waypoint>.json
  action:
    <action string>:
      success: N
      failure: M
```

它做三件事：

1. `is_succeeded_waypoint(waypoint)`：如果某个 waypoint 曾有成功动作，就取净成功分最高的 action。
2. `retrieve_similar_succeeded_waypoints(waypoint, topK)`：如果没有 exact waypoint 成功经验，就用 BERT embedding 找相似 waypoint，作为 planner few-shot 示例。
3. `save_success_failure(waypoint, action_str, is_success)`：执行后给 action 的 success 或 failure 加一。

这套机制的优点是简单，能快速积累“某个物品通常应该 craft、mine 还是 smelt”。例如：

```text
logs -> chop a tree
planks -> craft planks
cobblestone -> dig down and mine cobblestone
iron_ingot -> smelt iron_ingot
```

但它的局限也很明显：同一个 waypoint 在所有状态下都被当成同一个问题。`cobblestone` 只知道历史上 `dig down and mine cobblestone` 成功过，不知道成功时是否有木镐、是否在地表、背包是否快满、是否刚死亡、当前最终任务是什么。

### 1.2 XENON-plus：记忆单位升级为 state-waypoint-action-outcome case

XENON-plus 新增 `CaseBasedMemory`，把一次决策记录成完整 case，而不是只给 action 计数。每条 case 包含：

- `original_final_goal`：最终任务，例如 `golden_chestplate`
- `waypoint` / `waypoint_num`：当前子目标及数量
- `state_snapshot`：执行前状态
- `candidate_actions`：候选动作及来源
- `selected_action` / `selected_subgoal`：最终执行动作
- `decision_trace`：动作来自案例库、RADS、planner 还是语义 fallback
- `outcome`：pending、success、failed、timeout、crash 等
- `run_uuid`：所属实验运行

当前 live `cases.json` 共有 2080 条 case，按 outcome 粗略统计：

| outcome.status | 数量 |
|---|---:|
| success | 1670 |
| failed | 183 |
| timeout_non_programmatic | 147 |
| crash_RuntimeError | 57 |
| pending | 22 |
| crash_ConnectionError | 1 |

按决策来源统计：

| decision_trace.source | 数量 |
|---|---:|
| rads_decisioner | 1536 |
| case_memory_exact_waypoint | 407 |
| case_memory | 71 |
| semantic_fallback | 38 |
| planner | 28 |

这说明 Plus 的案例库不只是替换了文件格式，而是在真实实验中记录了大量“RADS 选择、案例复用、planner fallback、语义修正”的决策轨迹。

### 1.3 对比一：原版只知道“动作成败”，Plus 知道“状态下的动作成败”

**原 XENON-main：**

如果 `cobblestone.json` 里有：

```json
{
  "action": {
    "dig down and mine cobblestone": {
      "success": 100,
      "failure": 1
    }
  }
}
```

那么只要遇到 `cobblestone` waypoint，系统基本就会复用 `dig down and mine cobblestone`。它不会区分：

- 当前有没有 wooden_pickaxe
- 当前是不是已经在地下很深处
- 背包是否已经接近满
- 当前是做 stone_pickaxe，还是做 armor 长链任务
- 最近是否连续挖矿失败

**XENON-plus：**

Plus 会先从 `env.get_status()` 构造 `state_snapshot`：

```text
inventory + equipment + location_stats + plain_inventory + biome + obs_summary
```

因此两个 `cobblestone` case 可以被区分：

```text
case A:
  waypoint = cobblestone
  inventory = {wooden_pickaxe: 1}
  equipment = wooden_pickaxe
  ypos ~= 60
  selected_action = dig down and mine cobblestone
  outcome = success

case B:
  waypoint = cobblestone
  inventory = {wooden_pickaxe: 0}
  equipment = none
  ypos ~= 20
  selected_action = dig down and mine cobblestone
  outcome = failed
```

**解决的问题：**

原版会把这两个场景混成同一个“cobblestone 动作统计”。Plus 能记录“同一个 waypoint 在不同物理状态下成功率不同”，后续 RADS 才能学习到状态条件成功率。

### 1.4 对比二：原版很难追踪“为什么选了这个动作”，Plus 保留 decision_trace

**原 XENON-main：**

`DecomposedMemory` 只记录 action 的成败次数。后续复盘时通常只能知道：

```text
gold_ore -> dig down and mine gold_ore
success/failure count = ...
```

但不知道当时：

- 这个动作是 planner 生成的，还是记忆库复用的
- planner 参考了哪些相似 waypoint
- 是否曾有失败动作被提供给 planner
- 是否发生过语义修正
- 是否因为低置信触发了 fallback

**XENON-plus：**

每条 case 都带 `decision_trace`。例如 RADS 选择时会记录：

```text
source = rads_decisioner
p_success = ...
confidence = ...
attention_concentration = ...
selected_case_id = ...
candidates = [...]
evidence = [...]
```

semantic fallback 时会记录：

```text
source = semantic_fallback
rejected_action = ...
reason = infeasible action verb for waypoint
```

**解决的问题：**

原版失败后只能看最终结果，很难判断是 planner 选错、记忆复用错、执行失败，还是环境异常。Plus 可以把“选择依据”和“执行结果”放在同一条 case 里，后续能直接用于诊断和训练。

### 1.5 对比三：原版成功/失败计数直接写入，Plus 有 pending 到 outcome 的闭环

**原 XENON-main：**

执行结束后直接调用：

```text
save_success_failure(waypoint, action_str, is_success)
```

这会更新对应 action 的 success/failure count。记录很轻量，但没有“先记录决策，再等待真实 outcome”的中间态。

**XENON-plus：**

Plus 在采纳动作时先写 pending case：

```text
record_decision(...)
outcome = {status: pending, success: null}
```

执行完成后再结算：

```text
save_success_failure(...)
outcome = {status: success/failed/timeout..., success: true/false}
```

如果整轮任务异常结束，Plus 会区分：

- 普通能力失败：`mark_pending_cases_failed(...)`
- 基础设施 early stop：`discard_pending_cases(...)` 或排除，不污染训练数据

**解决的问题：**

原版很难把“环境崩溃、早停、未结算动作”和“智能体真的失败”分开。Plus 的 pending/outcome 机制使案例库更适合训练决策器，因为训练样本可以过滤掉 pending、infra、crash 等不可靠标签。

### 1.6 对比四：原版旧经验只能作为计数表，Plus 能迁移成训练样本

**原 XENON-main：**

历史经验保存在 `waypoint_to_sg/*.json` 里，只能被 `DecomposedMemory` 用于 exact reuse 或 similar waypoint retrieval。

**XENON-plus：**

`_bootstrap_from_legacy_waypoint_memory()` 会把原版 `waypoint_to_sg/*.json` 迁移成 case：

```text
legacy waypoint-action count
  -> case_memory/cases.json
  -> decision_trace.source = legacy_bootstrap
  -> outcome.success = success_count > 0
```

**解决的问题：**

Plus 没有丢掉原版已经积累的动作经验，而是把它纳入统一 case schema。这样旧经验可以继续参与检索，同时新实验产生的状态化 case 又能逐步替代粗粒度计数。

### 1.7 案例库方向的小结

| 维度 | XENON-main | XENON-plus | 解决的问题 |
|---|---|---|---|
| 记忆单位 | waypoint-action 计数 | state-waypoint-action-outcome case | 同一 waypoint 的不同状态无法区分 |
| 状态信息 | 基本没有 | inventory、equipment、location、biome、obs summary | 无法学习状态条件成功率 |
| 决策来源 | 不记录 | decision_trace 记录来源、候选、证据 | 失败原因不可复盘 |
| 结果闭环 | success/failure 直接计数 | pending -> resolved outcome | 可过滤 infra/pending，训练数据更干净 |
| 旧经验 | 只能原格式复用 | bootstrap 成 case | 兼容旧经验并支持后续训练 |

## 2. 优化方向二：决策器

相关文件：

- `src/optimus1/decisioner/feature.py`
- `src/optimus1/decisioner/encoder.py`
- `src/optimus1/decisioner/rads.py`
- `src/optimus1/decisioner/runtime.py`
- `scripts/export_decisioner_dataset.py`
- `scripts/train_rads.py`
- `scripts/evaluate_rads_offline.py`
- `artifacts/decisioner/rads_v2.pt`

### 2.1 原 XENON-main：没有可训练决策器，只有硬规则复用和 planner fallback

XENON-main 的决策逻辑可以概括为：

```text
if waypoint 有成功动作:
    取净成功分最高的动作
else:
    找相似 waypoint 成功例子 + 当前 waypoint 失败例子
    调 planner 生成动作
```

它不是一个训练出来的 decision model，而是一个基于历史计数和文本相似度的规则系统。它的问题是：

1. 有成功动作时过于相信历史动作。
2. 不估计当前状态下的成功概率。
3. 多候选动作时只能看成功/失败计数，不会结合当前背包、装备、高度和最终任务。
4. 没有低置信 gate。只要历史计数看起来可用，就容易直接复用。

### 2.2 XENON-plus：RADS 学习 P(success | state, waypoint, action)

Plus 新增 RADS（Retrieval-Augmented Decision Scorer）。它不直接生成新动作，而是对候选动作打分：

```text
P(success | 当前状态, waypoint, final_goal, candidate_action, 历史案例库)
```

在线流程是：

1. 对当前 waypoint 收集历史出现过的 distinct actions。
2. 每个 action 构造一个 query case。
3. RADS 输出每个 action 的 `p_success`。
4. 选最高分动作。
5. 如果最高分低于 `min_p_success=0.20`，返回 `None`，上层调用 planner。
6. 把候选动作排名和 attention evidence 写回 `decision_trace`。

RADS 的核心结构是：

```text
QueryEncoder(query) -> q
CaseEncoder(history case + outcome) -> C
attention = softmax(q @ C.T / tau)
context = attention @ C
decision_head([q, context, action_embedding]) -> success logit
P(success) = sigmoid(logit)
```

### 2.3 对比一：原版“有成功经验就复用”，Plus“先估计当前状态是否值得复用”

**原 XENON-main：**

假设 `iron_ore` 历史上 `dig down and mine iron_ore` 成功很多次，那么遇到 `iron_ore` waypoint 时，原版会倾向于直接复用该动作。

但当前状态可能是：

```text
inventory 没有 stone_pickaxe
equipment = wooden_pickaxe 或 none
```

Minecraft 规则上，木镐不能有效获得铁矿。原版的 waypoint 计数并不表达这个前置条件。

**XENON-plus：**

Plus 的决策器输入包含：

- 当前 inventory 关键物品数量
- 当前 equipment
- 是否拥有 wooden/stone/iron pickaxe
- 当前 ypos 和高度桶
- 当前 waypoint、final_goal、candidate_action
- `(waypoint, action)` 历史成功率先验

因此 RADS 能学习“同一个 `dig down and mine iron_ore`，在有石镐和无石镐时成功概率不同”。如果分数低于阈值，Plus 不复用案例，而是回退 planner 或由环境感知层先补前置条件。

**解决的问题：**

原版经验复用是静态的；Plus 让复用变成状态条件判断，避免把“历史上成功过”误当成“当前也可行”。

### 2.4 对比二：原版多候选动作容易靠计数或文本相似度漂移，Plus 用 same-waypoint attention 和动作先验约束

**原 XENON-main：**

如果某个 waypoint 有多个历史动作，原版主要看净成功计数。例如 `gold_ore` 可能出现：

```text
dig down and mine gold_ore
craft gold_ore
smelt gold_ore
```

其中 `craft gold_ore`、`smelt gold_ore` 对 Minecraft 规则并不合理，但一旦历史里存在错误动作或失败动作，原版只能把它们作为失败计数或 planner 的 failed_subgoals 使用，不能学习“为什么这种动作在什么状态下不应该选”。

**XENON-plus：**

RADS v2 加入了两个约束：

1. same-waypoint hard mask：同 waypoint 历史案例足够多时，attention 只看同 waypoint 案例。
2. `(waypoint, action)` 成功率先验：把 train split 中同一 waypoint-action 的成功率作为输出 logit 的 residual。

离线评估显示：

| 指标 | RADS v2 |
|---|---:|
| test AUC | 0.9110 |
| test AP | 0.9739 |
| top-1 attention same-waypoint rate | 0.9441 |
| cobblestone successful eval top-1 action match | 1.0000 |

**解决的问题：**

原版没有可解释的 action ranker。Plus 的 RADS 不仅输出成功率，还输出 attention evidence，并通过 same-waypoint 约束减少跨 waypoint 捷径。

### 2.5 对比三：原版 fallback 是“没有经验才问 planner”，Plus fallback 是“低置信就问 planner”

**原 XENON-main：**

planner fallback 的主要触发条件是：

```text
当前 waypoint 没有可用成功动作
```

如果已有成功动作，即使当前状态很差，系统也容易先复用。

**XENON-plus：**

RADS 的 fallback 触发条件是：

```text
最高 p_success < min_p_success
```

也就是说，即使历史上有动作，只要模型认为当前状态下成功概率低，Plus 仍会交还 planner。

线上 67 任务对比中，RADS 版本记录到 planner fallback 31 次，且这些 fallback 都出现在成功任务中：

| source | count | of which success |
|---|---:|---:|
| rads_decisioner | 628 | 606 |
| planner fallback | 31 | 31 |

报告中有两个典型 fallback 场景：

- `logs/chop a tree` 在 `craft_a_note_block` 任务中多次低置信，fallback planner 后任务成功。
- `cobblestone/dig down and mine cobblestone` 在 `craft_a_diamond_axe` 任务中多次低置信，fallback planner 后任务成功。

**解决的问题：**

原版只在“没有经验”时调用 planner；Plus 在“经验不可信”时也能调用 planner。这降低了盲目复用旧动作的风险。

### 2.6 对比四：原版实验结果不能直接训练决策器，Plus 建立离线训练闭环

**原 XENON-main：**

每次运行只更新 waypoint-action 计数。它能改进未来的 exact reuse，但不能训练一个状态条件模型。

**XENON-plus：**

Plus 的训练管线是：

```text
case_memory/cases.json
  -> scripts/export_decisioner_dataset.py
  -> data/decisioner/rads_v1.jsonl
  -> scripts/train_rads.py
  -> artifacts/decisioner/rads_v2.pt
  -> scripts/evaluate_rads_offline.py
  -> 在线开启 memory.case_memory.decisioner.enabled=true
```

导出数据按 `run_uuid` group split，避免同一局游戏的相邻 case 同时出现在 train/test。当前 `rads_v2.pt` 使用的冻结数据为：

| split | samples | runs | positive | negative |
|---|---:|---:|---:|---:|
| train | 1847 | 170 | 1542 | 305 |
| val | 351 | 36 | 330 | 21 |
| test | 376 | 38 | 307 | 69 |

训练目标是：

```text
L = BCE(success) + 0.1 * triplet_loss + 0.05 * waypoint_reconstruction
```

**解决的问题：**

原版经验只能越积越多，不能被抽象成模型。Plus 把实验结果转为监督学习样本，使系统能从大量 case 中学习“什么状态下什么动作更可能成功”。

### 2.7 决策器方向的实验证据

67 任务 single-shot 在线对比：

| 方法 | 成功/任务 | 成功率 |
|---|---:|---:|
| retrieval-only first attempt | 39/67 | 58.2% |
| RADS decisioner single-shot | 49/67 | 73.1% |
| retrieval-only best-of-N | 57/67 | 85.1% |

按 benchmark：

| benchmark | tasks | retrieval-only first | RADS | delta |
|---|---:|---:|---:|---:|
| wooden | 10 | 8 | 10 | +2 |
| stone | 9 | 7 | 6 | -1 |
| iron | 16 | 7 | 10 | +3 |
| golden | 6 | 3 | 2 | -1 |
| diamond | 7 | 5 | 7 | +2 |
| redstone | 6 | 2 | 6 | +4 |
| armor | 13 | 7 | 8 | +1 |
| total | 67 | 39 | 49 | +10 |

RADS 的收益主要体现在：不用多次重试，就能把 retrieval-only 首次尝试失败的一批任务追回来。

### 2.8 决策器方向的小结

| 维度 | XENON-main | XENON-plus | 解决的问题 |
|---|---|---|---|
| 决策方式 | 成功动作硬复用 + planner fallback | RADS 打分 + 低置信 fallback | 盲目复用旧动作 |
| 输入 | waypoint 和历史计数 | state、waypoint、final_goal、action、历史 case | 无法表达当前状态 |
| 多候选动作 | 按计数或相似例子处理 | 每个 action 估计 `P(success)` | 无法做 action rank |
| 解释性 | 只知道成功/失败次数 | p_success、attention evidence、candidate ranking | 无法复盘为什么选 |
| 学习闭环 | 只能累计计数 | case 导出、离线训练、在线接入 | 不能训练状态条件模型 |

## 3. 优化方向三：环境感知与行动

相关文件：

- Plus：`src/optimus1/main_planning.py`
- Plus：`src/optimus1/env/perception_action.py`
- Plus：`src/optimus1/env/wrapper.py`
- Plus：`src/optimus1/env/custom_env.py`
- Main 对照：`/home/yzb/zhourong/XENON-main/src/optimus1/main_planning.py`

### 3.1 原 XENON-main：高层规划和低层执行之间缺少物理状态闭环

XENON-main 的规划主要依赖：

- OracleGraph 给出的 waypoint
- 当前 inventory
- `DecomposedMemory` 的 waypoint-action 经验
- VLM planner 生成的 action 文本

执行层主要依赖：

- helper 处理 craft/smelt/equip
- STEVE-1 根据 prompt 输出低层动作
- MineRL wrapper 判断子目标是否完成

这会产生一个问题：高层 action 文本可能看起来合理，但当前物理状态并不支持执行。例如：

- 没有木镐却开始挖 cobblestone
- 没有石镐却开始挖 iron_ore
- 有石镐但没拿在手上
- 方块已挖掉但掉落物没捡到
- 背包被杂物塞满导致矿物捡不进来
- 一路向下挖过目标矿层后还继续向 bedrock 挖
- 死亡重生后仍延续死亡前的控制状态

XENON-plus 的第三类优化就是把这些环境状态和动作恢复机制纳入主循环。

### 3.2 对比一：原版 make_plan 主要看 inventory，Plus 看完整 env_status

**原 XENON-main：**

`make_plan()` 的核心输入是 `inventory`。它用 inventory 调 OracleGraph 生成 waypoint，然后问 `DecomposedMemory` 是否有成功动作。

简化流程：

```text
wp_list = OracleGraph.compile(goal, inventory)
wp = 第一项 waypoint
if action_memory.is_succeeded_waypoint(wp):
    复用成功动作
else:
    planner 生成动作
```

**XENON-plus：**

Plus 的 `make_plan()` 接收完整 `env_status`：

```text
inventory
equipment
location_stats
plain_inventory
resource_ledger
inventory_slots_used
recovery_events
```

这些信息同时服务三件事：

1. 生成案例库的 `state_snapshot`
2. 给 RADS 提供状态特征
3. 在规划阶段做环境约束修正

**解决的问题：**

原版只能做物品层面的规划；Plus 能把“手上拿什么、身处什么高度、是否有资源进展、背包是否满”等物理状态纳入决策。

### 3.3 对比二：原版可能重复执行已满足 waypoint，Plus 跳过非消耗目标

**原 XENON-main：**

OracleGraph 返回 waypoint 后，原版通常取第一个待完成项。对于长链任务，如果某些非消耗物品已经存在，原版仍可能再次进入相同 waypoint。

例子：

```text
当前已经有 furnace
后续任务仍返回 furnace waypoint
原版可能继续执行 craft furnace
```

这会消耗 8 个 cobblestone，影响后续工具或熔炼链。

**XENON-plus：**

Plus 新增已满足 waypoint 跳过逻辑：

```text
_select_next_planning_waypoint()
_can_skip_satisfied_waypoint()
_inventory_count_for_waypoint()
```

对于工具、装备、crafting_table、furnace、shield 等非消耗目标，如果 inventory 已经满足，就跳到下一个未满足 waypoint。

**解决的问题：**

避免重复制作关键设备，减少长链任务中因材料被无意义消耗导致的后续失败。

### 3.4 对比三：原版缺少显式前置条件修正，Plus 会补工具和材料前置条件

**原 XENON-main：**

如果当前 waypoint 是 `iron_ore`，历史记忆或 planner 可能直接给：

```text
dig down and mine iron_ore
```

但如果当前没有 stone_pickaxe，Minecraft 规则上这个动作不可行。原版主要依赖 planner 或历史经验自己隐式学到顺序，缺少硬约束检查。

**XENON-plus：**

Plus 新增前置条件函数：

```text
_planning_prereq_for_waypoint()
_crafting_prereq_for_waypoint()
_pickaxe_prereq_for_mining()
```

典型修正链：

```text
想挖 cobblestone
  -> 没有 wooden_pickaxe
  -> 先做 wooden_pickaxe

想挖 iron_ore
  -> 没有 stone_pickaxe
  -> 先做 stone_pickaxe

想挖 gold_ore / redstone / diamond
  -> 没有 iron_pickaxe
  -> 先补 iron_ingot / furnace / iron_pickaxe
```

**解决的问题：**

把 Minecraft 工具等级约束显式加入高层规划，减少“动作跑了很久，但规则上根本不会成功”的失败。

### 3.5 对比四：原版信任 planner/case 文本，Plus 做动作语义过滤

**原 XENON-main：**

如果 planner 或记忆系统输出了语义错误动作，原版较容易把动作交给 helper 或 STEVE-1 执行。例如调试中曾出现过类似错误：

```text
logs -> craft logs
planks -> dig down and mine planks
```

这类动作对人很明显不合理，但对文本 planner 或历史记忆来说可能只是一个字符串。

**XENON-plus：**

Plus 新增：

```text
_subgoal_action_is_feasible()
_fallback_subgoal_for_waypoint()
```

规则示例：

- `logs` 必须是 chop/punch/tree/collect 类动作。
- 矿物类 waypoint 必须是 mine/dig。
- `iron_ingot`、`gold_ingot`、`charcoal` 必须 smelt。
- craft-only 物品必须 craft/make/create。

如果动作不匹配，Plus 会拒绝原动作并生成保守 fallback：

```text
logs -> chop a tree
planks -> craft planks
iron_ingot -> smelt iron_ingot
```

**解决的问题：**

减少 planner 或记忆文本的低级语义错配，避免开局几步就因动作类型错误失败。

### 3.6 对比五：原版主要依赖 STEVE-1 原始动作，Plus 根据 prompt 稳定低层行动

**原 XENON-main：**

低层采集动作主要依赖 STEVE-1 自己输出。比如 `chop a tree` 或 `dig down and mine cobblestone`，如果 STEVE-1 只短暂 attack、视角飘走或乱跳，方块可能不会真正破坏。

**XENON-plus：**

Plus 让 wrapper 接收当前 prompt：

```text
env.step(action, current_sg_target, prompt=current_sg_prompt)
```

wrapper 可以根据 prompt 区分：

- 地表砍树
- 地表找树
- 地下挖矿
- 掉落物收集
- 停滞恢复

然后进行语义相关的动作稳定：

- 资源采集时延长 attack-hold
- 地下挖矿时限制 jump/sprint/横移/视角漂移
- 地表探索时 clamp pitch，减少看天或看脚
- 卡住时做 turn-around 或 movement escape
- prompt 切换和重生后请求 policy reset

**解决的问题：**

把“高层动作文本”和“低层动作稳定策略”连接起来，使砍树、挖石头、挖矿等动作更接近玩家的持续操作，而不是完全依赖 STEVE-1 的随机输出稳定性。

### 3.7 对比六：原版只看当前 inventory，Plus 引入 resource_ledger

**原 XENON-main：**

原版主要通过当前 inventory 和 checker 判断是否完成。Minecraft 里常见问题是：

- 方块已经被挖断，但掉落物还没捡到。
- 物品曾经进入背包，后来被合成消耗。
- 当前 inventory 没变化，但 `mine_block` 或 `pickup` 统计已经变化。

原版对这些中间状态缺少统一账本。

**XENON-plus：**

Plus 在 wrapper 中维护 `resource_ledger`，记录：

- inventory 正向 delta
- max_inventory
- pickup stat delta
- mine_block stat delta
- collected 统计

`get_status()` 会把 `resource_ledger` 暴露给规划层和结果记录。

**例子：**

```text
目标：gold_ore
智能体挖掉 gold_ore 方块
但掉落物在前方没有进入 inventory

原版：可能继续向下挖，无法明确知道“已经挖掉但没捡”
Plus：ledger 看到 mined/pickup/inventory 差异，触发 collect drops
```

**解决的问题：**

把“资源发生过变化但未体现在当前 inventory”这种状态显式化，支持掉落物收集、训练样本解释和失败诊断。

### 3.8 对比七：原版没有背包压力管理，Plus 会清理低价值物品

**原 XENON-main：**

长链任务中，智能体可能捡到 seeds、flowers、leaves、dirt、sand、gravel 等杂物。背包槽位被占满后，即使挖到了关键矿物，也可能捡不进来。

原版没有系统性的背包清理策略。

**XENON-plus：**

Plus 新增 `_maybe_cleanup_inventory()`，在槽位压力高时丢弃低价值物品，同时保护：

- 工具
- 原木、木板、stick
- crafting_table、furnace
- 矿石、锭
- diamond、redstone
- 当前目标物

**解决的问题：**

避免长链实验中被杂物占满背包，从而错失关键矿物或合成材料。

### 3.9 对比八：原版地下挖矿容易一路向下，Plus 加入矿层感知、overshoot 和水平通道

**原 XENON-main：**

地下资源任务常见策略是：

```text
dig down and mine X
```

它容易出现：

- 一直向下挖
- 错过目标矿层
- 到 bedrock 附近仍继续挖
- 发现更深层矿物后不回层
- 当前竖井资源耗尽后缺少横向探索

**XENON-plus：**

Plus 加入矿层相关机制：

- `ORE_LAYER_ORDER`
- `perceive_height_context()`
- overshoot 检测
- `pillar_up()` / `pillar_up_smart()` / `raise_to_ore_band()`
- 回到目标层后 `dig_forward_blocks()` 开水平通道
- bedrock-stuck 检测
- 死亡/重生后重置 prompt 和控制状态

**例子：**

```text
目标：gold_ore
智能体一路向下挖，先看到了 redstone/diamond 或接近 bedrock

原版：可能继续向下，越来越偏离 gold_ore 合理层
Plus：认为发生 overshoot，尝试 pillar-up 回到目标层，再横向挖通道
```

**解决的问题：**

把原版“只会向下挖”的失败模式，改造成“能感知过深、回层、换水平位置继续探索”的可恢复流程。

### 3.10 对比九：原版死亡/重生后可能延续旧控制状态，Plus 重置策略

**原 XENON-main：**

挖矿时掉落、窒息或其他原因死亡后，重生是一个物理状态断点。但如果动作模型或 wrapper 仍保留死亡前的 attack、escape、tunnel recovery 状态，重生后可能继续执行不合适动作。

**XENON-plus：**

Plus 监控 health、is_alive 和位置跳变。检测到重生后：

- 清空 attack-hold、escape、collect drops、tunnel recovery 等控制状态
- 设置 `policy_reset_requested`
- 主循环消费该请求后重置 action server
- 恢复当前 subgoal prompt
- 挖矿任务中尝试重新装备最佳 pickaxe

**解决的问题：**

把死亡/重生视为环境状态断点，避免重生后延续地下旧动作或旧 prompt 状态。

### 3.11 PerceptionActionSuite：把环境感知与行动能力统一成可消融模块

**原 XENON-main：**

没有统一的环境感知行动套件。要比较执行层机制的影响，通常只能手工改代码或改多个环境变量。

**XENON-plus：**

Plus 新增 `PerceptionActionSuite`，用一个总开关控制一组能力：

```text
XENON_PERCEPTION_ACTION_SUITE=1/0
```

该 suite 管理：

- tree explore
- surface turn-around
- ground pitch clamp
- inventory cleanup
- collect drops
- movement escape
- tunnel recovery
- overshoot pillar-up

并且使用 `os.environ.setdefault()`，不会覆盖用户已经显式设置的单项开关。

**解决的问题：**

环境感知与行动不再是一堆散落规则，而是可以整体打开、整体关闭、单项消融的执行层模块。

### 3.12 环境感知与行动方向的边界

这些机制显著改善了“动作能否在 Minecraft 物理环境中被真实执行”的问题，但它不是万能的：

- 它不能保证自然矿物搜索一定成功。
- 它不能替代视觉识别能力。
- 它不能让 STEVE-1 变成稳定的人类级低层控制器。
- armor 级长链地下任务仍是最大瓶颈。

因此这一类创新更准确的表述是：Plus 把环境状态、物理规则和恢复动作纳入闭环，使失败模式从“黑箱执行失败”变成“可感知、可恢复、可消融的执行过程”。

### 3.13 环境感知与行动方向的小结

| 维度 | XENON-main | XENON-plus | 解决的问题 |
|---|---|---|---|
| 规划输入 | 主要 inventory | 完整 env_status | 无法看装备、高度、ledger、恢复状态 |
| waypoint 处理 | 取首个待做 waypoint | 跳过已满足非消耗 waypoint | 重复制作、浪费材料 |
| 前置条件 | 隐式依赖 planner/经验 | 显式检查工具和材料 | 无工具挖矿等规则失败 |
| 动作文本 | 基本信任 planner/case | 语义过滤 + fallback | `craft logs`、`dig planks` 等错动作 |
| 低层动作 | 依赖 STEVE-1 原始输出 | prompt-aware 稳定与恢复 | attack 太短、视角漂移、卡住 |
| 资源感知 | 当前 inventory 为主 | resource_ledger | 挖掉但没捡、曾经获得但被消耗 |
| 背包压力 | 无系统清理 | 低价值物品清理 | 背包满导致关键资源捡不进来 |
| 地下探索 | 主要一路向下 | overshoot、pillar-up、水平通道 | 错过矿层后无恢复 |
| 重生处理 | 可能保留旧状态 | policy reset + 重装备 | 重生后继续旧动作 |

## 4. 三类优化如何组成闭环

三类创新不是互相独立的功能堆叠，而是一个闭环：

```text
环境感知与行动
  -> 提供更真实的 env_status、resource_ledger、recovery_events
  -> 让执行结果更可靠，也让失败更可解释

案例库
  -> 把 state、waypoint、action、decision_trace、outcome 写成 case
  -> 为检索、planner 示例和 RADS 训练提供数据

决策器
  -> 从案例库学习 P(success | state, waypoint, action)
  -> 在线选择高置信动作，低置信时回退 planner
  -> 新决策继续写回案例库，形成下一轮训练材料
```

可以用一个具体长链任务理解这三者的关系：

```text
任务：Craft golden chestplate

原 XENON-main：
  logs -> planks -> crafting_table -> stick -> wooden_pickaxe
  -> cobblestone -> furnace -> stone_pickaxe -> iron_ore
  -> iron_ingot -> iron_pickaxe -> gold_ore -> gold_ingot
  -> golden_chestplate

每个 waypoint 主要按历史动作计数复用。
如果 gold_ore 挖矿失败，记忆里只多一个 failure count。

XENON-plus：
  1. 环境感知层记录 inventory、equipment、ypos、ledger、背包压力。
  2. 案例库记录每个 waypoint 的 state_snapshot、selected_action、decision_trace。
  3. RADS 对 gold_ore、iron_ore、cobblestone 等动作估计当前成功率。
  4. 低置信时回退 planner；语义错误时 fallback；缺工具时先补工具。
  5. 执行层遇到掉落物、背包满、过深、重生等情况尝试恢复。
  6. 成功/失败 outcome 回写 case memory，后续可重新训练。
```

## 5. 总体对比表

| 层级 | XENON-main | XENON-plus | 主要解决的问题 |
|---|---|---|---|
| 经验存储 | waypoint-action count | state-waypoint-action-outcome case | 经验缺少状态和可训练标签 |
| 经验复用 | 有成功动作就复用 | 案例检索 + RADS 成功率打分 | 盲目复用历史动作 |
| planner fallback | 没成功经验时触发 | 低置信或无候选时触发 | 有经验但当前不适合时无法回退 |
| 状态表达 | inventory 为主 | inventory、equipment、location、biome、ledger、recovery | 无法表达物理状态 |
| 动作合法性 | 依赖 planner/case 文本 | 语义过滤与保守 fallback | 文本动作类型错配 |
| 前置条件 | 隐式 | 显式工具/材料检查 | 无工具挖矿、材料不足 |
| 执行稳定 | STEVE-1 原始输出为主 | prompt-aware attack/移动/视角/恢复 | 挖不掉、砍不到、卡住 |
| 地下资源 | 主要 dig down | 矿层感知、overshoot、pillar-up、水平通道 | 一路挖到 bedrock 无恢复 |
| 训练闭环 | 计数增长 | case 导出、RADS 训练、在线接入 | 不能训练状态条件决策器 |
| 可诊断性 | 结果层面为主 | decision_trace + evidence + outcome | 不知道错在决策还是执行 |

## 6. 最终结论

XENON-main 的核心优势是结构简单：OracleGraph 给 waypoint，动作记忆负责复用已有成功动作，planner 负责处理没有经验的 waypoint。它的问题是经验粒度太粗、没有可训练决策器、执行层缺少环境状态闭环。

XENON-plus 的三类优化分别补上这三个缺口：

1. **案例库**把经验从 `waypoint-action count` 升级为 `state-waypoint-action-outcome case`，让每次实验都变成可复盘、可训练的数据。
2. **决策器**用 RADS 学习 `P(success | state, waypoint, action)`，把硬复用升级为状态条件成功率判断，并在低置信时回退 planner。
3. **环境感知与行动**把装备、高度、资源账本、背包、掉落物、矿层、重生等 Minecraft 物理状态纳入规划和执行，减少“文本计划正确但物理执行失败”的情况。

因此，XENON-plus 不是简单修改 prompt，也不是完全替换原系统，而是在原 XENON 的规划和执行骨架上加入“状态化记忆、可训练决策、环境感知行动”三层闭环。
