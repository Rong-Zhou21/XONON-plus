# XENON-plus 智能体任务执行流程汇报

生成时间：2026-04-27  
项目位置：`/home/yzb/zhourong/XENON-plus`  
对比基线：`/home/yzb/zhourong/XENON-main`

## 1. 总体结论

当前 XENON-plus 的核心流程仍然继承原版 XENON 的任务执行骨架：

1. 从 benchmark 读取目标任务。
2. 初始化 Minecraft / MineRL 环境。
3. 使用世界知识图谱 `OracleGraph` 根据最终目标生成当前 waypoint。
4. 根据 waypoint 生成或复用一个可执行 subgoal/action。
5. 调用 STEVE-1 / helper 在环境中执行动作。
6. 检查 waypoint 和最终任务是否完成。
7. 记录实验结果、计划轨迹、视频和记忆。

本次主要修改集中在第 4、6、7 步：

- 原版 `DecomposedMemory` 已替换为 `CaseBasedMemory`。
- 原来的“waypoint -> 成功动作”动作库，被重构为“面向决策器的案例库”。
- 决策环节从“只看 waypoint 是否有成功动作”变为“基于当前环境状态 + waypoint 检索案例，优先复用可信案例，否则调用原 planner”。
- 原 planner 没有改动，仍然是 fallback 规划器。
- 失败、崩溃、环境日志异常现在会尽量写入失败记录，并保存视频。
- 视频目录命名已改为 `任务种类_任务`，例如 `Wood_Craft_a_bowl`。

因此，XENON-plus 当前不是重写整个智能体，而是在保持原 XENON 可比性的前提下，把“动作记忆库”升级成了“案例记忆库 + 初版检索式决策器”。

## 2. 原版 XENON 的任务执行流程

原版入口主要是：

- `src/optimus1/main_planning.py`
- `src/optimus1/memories/decomposed_memory.py`

### 2.1 原版主流程

原版执行一个任务时，大致流程如下：

1. `main()` 读取配置和 benchmark。
2. 根据环境名推断 benchmark 类型，例如 `wooden`、`stone`、`iron`。
3. 创建环境：
   - `env = env_make(...)`
   - `ServerAPI._reset(...)`
   - `obs = env.reset()`
4. 创建动作记忆库：
   - `action_memory = DecomposedMemory(cfg, logger)`
5. 调用 `new_agent_do(...)` 执行任务。
6. 在 `new_agent_do(...)` 中循环执行：
   - 检查最终目标是否已完成。
   - 如果当前没有 subgoal，则调用 `make_plan(...)`。
   - 执行 subgoal。
   - 检查 waypoint 是否达成。
   - 记录 waypoint 动作成功或失败。
7. 任务结束后：
   - 保存视频。
   - 保存 decomposed plan。
   - 写出 result JSON。

### 2.2 原版 make_plan 的逻辑

原版 `make_plan(...)` 的核心逻辑是：

1. 使用 `OracleGraph.compile(...)` 根据最终目标和当前 inventory 生成 waypoint 链。
2. 取当前最需要完成的第一个 waypoint。
3. 查询动作库：
   - `is_succeeded, sg_str = action_memory.is_succeeded_waypoint(wp)`
4. 如果该 waypoint 有历史成功动作：
   - 直接复用这个动作。
5. 如果没有成功动作：
   - 检索相似成功 waypoint：
     - `retrieve_similar_succeeded_waypoints(wp, topK)`
   - 检索失败 subgoal：
     - `retrieve_failed_subgoals(wp)`
   - 调用 planner：
     - `call_planner_with_retry(...)`

也就是说，原版决策规则非常简单：

```text
当前 waypoint 有成功动作 -> 直接复用
当前 waypoint 没有成功动作 -> 调用 planner
```

### 2.3 原版 DecomposedMemory 的数据结构

原版记忆库主要是 `waypoint_to_sg` 下的多个 JSON 文件，每个 waypoint 一个文件。

典型结构类似：

```json
{
  "action": {
    "chop a tree": {
      "success": 3,
      "failure": 1
    }
  }
}
```

它存储的核心信息只有：

- waypoint
- action 字符串
- success 计数
- failure 计数

原版的优点是简单、稳定、查询快；缺点也很明显：

- 不记录智能体当前状态。
- 不记录装备、位置、biome、inventory 细节。
- 不记录为什么选择该动作。
- 不记录候选动作集合。
- 不方便训练一个真正的决策器。
- 失败信息只以计数形式存在，无法表达失败发生时的具体上下文。

## 3. XENON-plus 当前任务执行流程

XENON-plus 的入口仍是：

- `src/optimus1/main_planning.py`

但记忆模块改为：

- `src/optimus1/memories/case_memory.py`

当前案例库文件为：

- `src/optimus1/memories/ours_planning/v1/case_memory/cases.json`

### 3.1 XENON-plus 主流程

XENON-plus 的主流程如下：

1. `main()` 读取配置和 benchmark。
2. 创建环境。
3. 创建案例记忆库：
   - `action_memory = CaseBasedMemory(cfg, logger)`
4. 进入任务执行循环。
5. 每次需要新 subgoal 时，调用修改后的 `make_plan(...)`。
6. `make_plan(...)` 不再只传入 inventory，而是传入完整 `env_status`。
7. `CaseBasedMemory` 从 `env_status` 和 `obs` 构造状态快照。
8. 决策器优先从案例库检索可复用案例。
9. 如果检索到可信案例，直接复用案例动作。
10. 如果没有可信案例，则调用原 planner。
11. planner 输出也会被记录成 pending case。
12. subgoal 执行完成后，将 pending case 更新为 success 或 failed。
13. 如果任务失败、崩溃或环境异常，则尽量将未完成 pending case 标记为失败。
14. 保存 result JSON、decomposed plan 和视频。

当前 plus 版的高层决策逻辑为：

```text
当前环境状态 + 当前 waypoint
        |
        v
构造 state_snapshot
        |
        v
检索 CaseBasedMemory
        |
        +-- 有高置信可复用案例 -> 直接使用案例动作
        |
        +-- 没有高置信案例 -> 调用原 planner
                              |
                              v
                         记录 planner 决策为 pending case
```

### 3.2 修改后的 make_plan 逻辑

XENON-plus 的 `make_plan(...)` 与原版相比有三处关键变化。

第一，输入从 `inventory` 扩展为 `env_status`：

```text
原版：make_plan(original_final_goal, inventory, ...)
plus：make_plan(original_final_goal, env_status, ...)
```

这样做的原因是案例库服务于决策器，不能只看到 inventory，还需要看到更完整的智能体状态。

第二，增加状态快照：

```python
state_snapshot = action_memory.create_state_snapshot(env_status, obs, cfg)
```

当前状态快照包括：

- `inventory`
- `equipment`
- `location_stats`
- `plain_inventory`
- `biome`
- `obs_summary`

第三，优先使用案例决策：

```python
case_decision = action_memory.select_case_decision(
    wp,
    wp_num,
    state_snapshot,
    topK,
    run_uuid,
    original_final_goal,
)
```

如果 `case_decision` 存在，则直接复用：

```text
return wp, case_decision["subgoal"], case_decision["language_action_str"], None
```

如果不存在，才进入原 planner fallback：

```text
retrieve_similar_succeeded_waypoints(...)
retrieve_failed_subgoals(...)
call_planner_with_retry(...)
record_decision(...)
```

## 4. 案例库 CaseBasedMemory 的设计

### 4.1 案例库定位

当前案例库不是原动作库的附属文件，而是原动作库的替代品。

原版依赖：

```text
DecomposedMemory
  -> waypoint_to_sg/*.json
```

XENON-plus 依赖：

```text
CaseBasedMemory
  -> case_memory/cases.json
```

为了保证冷启动时不丢失原版能力，XENON-plus 首次初始化会把旧的 `waypoint_to_sg` 文件迁移为案例：

```text
legacy waypoint_to_sg -> case_memory/cases.json
```

迁移完成后，会写入：

```text
case_memory/legacy_bootstrap.done
```

之后运行时主要读写 `cases.json`。

### 4.2 单条案例字段

当前案例大致包含：

```json
{
  "id": "...",
  "created_at": "...",
  "run_uuid": "...",
  "original_final_goal": "...",
  "environment": "...",
  "waypoint": "...",
  "waypoint_num": 1,
  "state_snapshot": {
    "inventory": {},
    "equipment": "...",
    "location_stats": {},
    "plain_inventory": {},
    "biome": "...",
    "obs_summary": {}
  },
  "similarity_text": "...",
  "candidate_actions": [],
  "selected_action": "...",
  "selected_subgoal": {},
  "selected_subgoal_str": "...",
  "decision_trace": {},
  "outcome": {
    "status": "success",
    "success": true
  }
}
```

相比原版，新增的关键能力是：

- 案例绑定具体环境状态。
- 案例记录最终目标。
- 案例记录候选动作集合。
- 案例记录最终选择动作。
- 案例记录决策来源。
- 案例记录 pending、success、failed、failed_incomplete_run 等状态。
- 案例可以作为后续训练决策器的数据样本。

### 4.3 检索逻辑

当前检索使用 `SentenceTransformer("all-MiniLM-L6-v2")` 对 `similarity_text` 编码。

`similarity_text` 由以下信息组成：

```text
waypoint
inventory
equipment
location
biome
```

决策时：

1. 优先检查同 waypoint 的成功案例。
2. 如果存在 exact waypoint success case，直接复用。
3. 如果没有 exact case，再做相似案例 top-k 检索。
4. 检索结果低于 `reuse_threshold` 时不复用。
5. 不复用时调用原 planner。

配置位置：

```yaml
memory:
  case_memory:
    reuse_threshold: 0.72
    retrieve_threshold: 0.45
    bootstrap_legacy: True
```

### 4.4 pending case 的作用

plus 版有一个原版没有的中间状态：`pending`。

当 planner 或 case memory 选定一个动作后，系统先记录一条 pending case：

```text
已经做出决策，但执行结果还未知
```

执行结束后：

- waypoint 成功：pending -> success
- waypoint 失败：pending -> failed
- 任务崩溃/异常：pending -> failed_incomplete_run 或具体失败状态

这样做的意义是：决策器训练时可以区分“做了什么决策”和“这个决策最后产生了什么结果”。

## 5. 与原版 XENON 的流程差异对比

| 环节 | 原版 XENON | XENON-plus |
|---|---|---|
| 项目位置 | `XENON-main` | `XENON-plus` |
| 记忆模块 | `DecomposedMemory` | `CaseBasedMemory` |
| 主存储 | `waypoint_to_sg/<waypoint>.json` | `case_memory/cases.json` |
| 存储粒度 | waypoint-action 计数 | 决策案例 |
| 状态信息 | 基本没有 | inventory、equipment、location、biome、obs_summary |
| 动作选择 | waypoint 有成功动作就复用 | 当前状态 + waypoint 检索案例 |
| planner 作用 | 无成功动作时生成动作 | 无可信案例时生成动作 |
| planner 是否修改 | 原版 planner | 未修改，仍作为 fallback |
| 候选动作记录 | 不显式记录 | `candidate_actions` |
| 决策来源记录 | 无 | `decision_trace` |
| 执行前状态 | 不记录 | `state_snapshot` |
| 执行后结果 | success/failure 计数 | outcome 对象 |
| 失败样本价值 | 主要用于计数惩罚 | 可作为训练负样本 |
| 崩溃处理 | 多数直接退出 | 尽量转成失败记录 |
| 视频保存 | 成功/正常失败路径保存 | 成功、失败、崩溃尽量保存 |
| 视频目录 | `Craft_a_bowl` | `Wood_Craft_a_bowl` |

## 6. 当前执行任务时的详细链路

以 wood 任务为例，当前执行链路如下。

### 6.1 初始化阶段

1. 读取配置：
   - benchmark：`wooden`
   - prefix：`ours_planning`
   - version：`v1`
2. 构造路径：
   - memory path：`src/optimus1/memories/ours_planning/v1`
   - case memory：`src/optimus1/memories/ours_planning/v1/case_memory/cases.json`
   - video path：`videos/v1`
   - result path：`exp_results/v1`
3. 初始化环境。
4. 初始化 `CaseBasedMemory`。
5. 如果未 bootstrap，则从旧动作库迁移案例。

### 6.2 waypoint 生成阶段

XENON-plus 当前仍然使用原版世界知识图谱：

```text
OracleGraph.compile(final_goal, quantity, current_inventory)
```

它会根据最终目标和 inventory 给出当前应优先完成的 waypoint。

这一点没有改变。

也就是说，当前创新点不是“让模型自己发现 recipe/waypoint”，而是“在已有 waypoint 指导下，改进动作选择和经验沉淀”。

### 6.3 决策阶段

拿到 waypoint 后，plus 版会做：

1. 读取当前 `env_status`。
2. 构造 `state_snapshot`。
3. 生成检索文本 `similarity_text`。
4. 在案例库中查找是否存在可复用案例。

优先级为：

```text
exact waypoint 成功案例
    >
相似状态下的成功案例
    >
planner fallback
```

如果使用案例，则 `decision_trace.source` 可能是：

```text
case_memory_exact_waypoint
case_memory
```

如果使用 planner，则 `decision_trace.source` 是：

```text
planner
```

### 6.4 执行阶段

执行阶段基本沿用原版：

- craft / smelt 类动作走 `NewHelper.step(...)`
- 非 craft/smelt 类动作通过 `ServerAPI.get_action(...)` 获取低层控制动作
- 每步通过环境 wrapper 更新状态和录像帧
- 通过 task checker 判断 subgoal 是否完成

这部分没有做结构性创新。

### 6.5 结果写回阶段

执行完成后：

1. 如果 waypoint 成功：
   - 对应 case outcome 标记为 success。
2. 如果 waypoint 失败：
   - 对应 case outcome 标记为 failed。
3. 如果任务失败或崩溃：
   - 同一 run 的 pending case 被标记为失败。
4. 保存 decomposed plan。
5. 保存 result JSON。
6. 保存视频。

视频保存目录现在使用：

```text
videos/v1/<任务种类>_<任务>/<biome>/<status>/<time>_<actual_done_task>_<run_uuid>.mp4
```

wood 任务示例：

```text
videos/v1/Wood_Craft_a_bowl/forest/success/...
```

## 7. 当前实验与数据状态

当前案例库位置：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

当前案例数量：

```text
95
```

当前 outcome 分布：

```text
success: 85
failed: 7
failed_incomplete_run: 3
pending: 0
```

wood 10 个任务最新有效结果均有 result JSON 和 mp4 视频。

当前视频目录已改为：

```text
videos/v1/Wood_Craft_a_bowl
videos/v1/Wood_Craft_a_chest
videos/v1/Wood_Craft_a_crafting_table
videos/v1/Wood_Craft_a_ladder
videos/v1/Wood_Craft_a_stick
videos/v1/Wood_Craft_a_wooden_axe
videos/v1/Wood_Craft_a_wooden_hoe
videos/v1/Wood_Craft_a_wooden_pickaxe
videos/v1/Wood_Craft_a_wooden_shovel
videos/v1/Wood_Craft_a_wooden_sword
```

## 8. 当前修改凸显的创新点

### 8.1 从动作库变成案例库

原版记忆的核心是：

```text
某 waypoint 下某 action 成功/失败了多少次
```

plus 版记忆的核心变成：

```text
在什么环境状态下，为哪个 waypoint 做出了什么决策，最后结果如何
```

这使得记忆库从“动作复用表”升级为“训练数据集雏形”。

### 8.2 从固定复用规则变成检索式决策

原版复用逻辑是硬规则：

```text
只要 waypoint 有成功动作，就复用成功次数最高的动作
```

plus 版引入了决策器雏形：

```text
根据当前环境状态检索案例，满足阈值才复用，否则调用 planner
```

虽然当前决策器还很简单，但接口上已经具备继续升级的空间：

- 可以换成向量数据库。
- 可以加入 cross-encoder reranker。
- 可以训练分类/排序模型。
- 可以用成功/失败案例做 preference learning。
- 可以让 planner 生成多个候选，再由决策器排序。

### 8.3 保留原 planner，保证对比公平

当前没有改原 planner 的生成逻辑。

这对后续实验很重要：

- 原版 XENON 和 XENON-plus 的差异主要集中在记忆/决策模块。
- 如果后续做 ablation，可以更清楚地说明性能差异来自案例库和决策器。
- planner 仍然作为 fallback，避免早期案例库不足导致系统不可用。

### 8.4 失败经验被结构化保存

原版失败主要影响 action 的 failure 计数。

plus 版失败会进入案例结构：

- 当前状态
- waypoint
- selected_action
- selected_subgoal
- decision_trace
- outcome

这为后续“从失败中学习”提供了数据基础。

## 9. 当前仍然保留的原版能力

为了保证 XENON-plus 和 XENON-main 可比，以下能力仍然保留：

1. `OracleGraph` 仍然负责 waypoint 生成。
2. 原 planner 仍然负责无可用案例时的动作规划。
3. STEVE-1 执行动作的方式没有重写。
4. benchmark 配置结构基本不变。
5. result JSON、decomposed plan、video 的整体保存方式仍兼容原版。
6. 原 `waypoint_to_sg` 能通过 bootstrap 转换为案例库冷启动经验。

## 10. 当前限制与下一步建议

### 10.1 当前限制

1. 案例检索仍然是 JSON + SentenceTransformer 全量编码/检索，案例库很大后会变慢。
2. `state_snapshot` 仍然偏工程化，缺少高质量语义环境描述。
3. 当前 exact waypoint 成功案例优先级较高，可能在复杂环境下过度复用。
4. planner 当前仍只生成一个最终动作，没有形成真正的多候选动作空间。
5. `candidate_actions` 当前主要记录被选中的 planner 动作，后续需要扩展为真实候选集合。
6. legacy bootstrap 案例缺少真实状态，后续训练时应降低权重或单独分层。

### 10.2 下一步建议

1. 给案例增加 `case_source` 字段：
   - `legacy_bootstrap`
   - `runtime_experience`
   - `manual_annotation`
2. 给案例增加质量字段：
   - `quality_score`
   - `usable_for_training`
   - `failure_type`
   - `manual_note`
3. 建立案例分层检索：
   - 先按 benchmark/task/waypoint 过滤。
   - 再按状态向量检索。
   - 最后按成功率和失败惩罚排序。
4. 引入向量索引：
   - FAISS
   - Chroma
   - Milvus
5. 把 planner 输出扩展为多个候选动作，再让决策器做排序。
6. 把失败视频和失败案例关联起来，便于人工标注。
7. 定期清理低质量 legacy case，避免训练时污染决策器。

## 11. 一句话概括

原版 XENON 是：

```text
OracleGraph 给 waypoint，动作库查成功动作，没有则 planner 生成。
```

当前 XENON-plus 是：

```text
OracleGraph 仍给 waypoint，但动作选择先进入案例库决策器；
案例库根据当前状态检索历史决策案例，有可信案例就复用，
没有可信案例才调用原 planner，并把本次决策及结果继续写回案例库。
```

这次修改的本质是：在不破坏原版 planner 和执行器的基础上，把“记忆复用”升级成“可训练的决策案例积累流程”。
