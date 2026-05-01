# XENON-plus 决策器设计与训练方案

日期：2026-05-01  
项目：`/home/yzb/zhourong/XENON-plus`  
参考代码：

- `/home/yzb/zhourong/Memento-main/memory/train_memory_retriever.py`
- `/home/yzb/zhourong/Memento-main/memory/parametric_memory.py`
- `/home/yzb/zhourong/Memento-main/memory/np_memory.py`
- `/home/yzb/zhourong/Memento-main/client/parametric_memory_cbr.py`
- `/home/yzb/zhourong/XENON-plus/train_value_head.py`
- `/home/yzb/zhourong/XENON-plus/src/optimus1/memories/case_memory.py`
- `/home/yzb/zhourong/XENON-plus/src/optimus1/main_planning.py`

## 1. 目标定位

当前 XENON-plus 已经把原版 XENON 的 waypoint 动作库替换为案例库，案例库记录的是某个环境状态下，针对某个 waypoint 选择了什么动作，以及最终成功或失败。下一步创新点不是让决策器替代 planner 生成新方案，而是让决策器具备“在当前状态下判断哪个已有行动更值得执行”的能力。

本文档的设计目标是：

1. 参考 Memento 的可训练案例选择器思路。
2. 参考当前 `train_value_head.py` 的轻量价值头训练思路。
3. 结合 Minecraft 任务的特点，给出 XENON-plus 决策器结构和训练方案。
4. 保持第一版实现不复杂，能够和现有案例库、planner fallback、实验记录自然衔接。

这里的“决策器”定义为：

> 在特定环境状态下，从可选行动或可复用案例中选择最优行动的模块。它不负责生成完整规划，不替代 planner；它决定是否复用案例、复用哪个案例、是否退回 planner。

## 2. 当前 XENON-plus 基线

### 2.1 运行流程

当前主流程大致为：

1. OracleGraph 给出当前任务所需 waypoint。
2. `CaseBasedMemory.create_state_snapshot()` 抽取当前环境状态，包括 inventory、equipment、location、biome、plain_inventory、obs_summary。
3. `CaseBasedMemory.select_case_decision()` 尝试从案例库中选择动作。
4. 如果有可复用案例，就直接把案例中的 `selected_action` 转成 STEVE-1 subgoal 执行。
5. 如果没有合格案例，就调用原始 planner。
6. planner 生成的行动也会被写入案例库，后续成功/失败会更新 outcome。
7. 任务结束后，`exp_results/v2/*.json` 记录任务级结果，包括 success、status_detailed、steps、minutes、failed_waypoints、resource_ledger 等。

### 2.2 当前案例库结构

当前案例库文件：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

每条 case 的核心字段包括：

```json
{
  "id": "...",
  "run_uuid": "...",
  "original_final_goal": "...",
  "environment": "...",
  "waypoint": "...",
  "waypoint_num": 1,
  "state_snapshot": {
    "inventory": {},
    "equipment": "stone_pickaxe",
    "location_stats": {},
    "plain_inventory": {},
    "biome": "...",
    "obs_summary": {}
  },
  "similarity_text": "...",
  "candidate_actions": [
    {
      "action": "dig down and mine iron_ore",
      "source": "planner"
    }
  ],
  "selected_action": "...",
  "selected_subgoal": {},
  "selected_subgoal_str": "...",
  "decision_trace": {},
  "outcome": {
    "status": "success|failed|pending|...",
    "success": true
  }
}
```

当前案例库已经具备训练决策器所需的最小闭环：

- 状态：`state_snapshot`
- 目标：`waypoint`、`waypoint_num`、`original_final_goal`
- 动作：`selected_action`
- 来源：`decision_trace.source`
- 标签：`outcome.success`、`outcome.status`
- 任务级代价：可通过 `run_uuid` 连接 `exp_results/v2/*.json` 的 `steps` 和 `minutes`

### 2.3 当前选择逻辑的不足

当前 `select_case_decision()` 仍然偏规则化：

1. 如果有相同 waypoint 的成功动作，优先 `_best_exact_success_case()`，可直接置信度 1.0 复用。
2. 否则用 `all-MiniLM-L6-v2` 对 `similarity_text` 做 cosine 检索。
3. 只要同 waypoint 的相似成功案例分数超过阈值，就复用。
4. 无合格案例时回退 planner。

这种设计能替代原始动作库，但不是一个真正训练出来的选择器。它主要回答“有没有相同 waypoint 的成功经验”，而不是回答“在当前状态下哪个行动最有价值”。

## 3. Memento 可训练选择器思路

### 3.1 非参数化 memory

`Memento-main/memory/np_memory.py` 的非参数化检索流程很直接：

1. 读取 jsonl 中的 memory。
2. 把 key 文本编码为向量。
3. 把当前 query 编码为向量。
4. cosine 相似度排序，返回 TopK。

它的优点是简单、无需训练；缺点是只能依赖文本相似度，无法直接学习“哪些案例真的有用”。

### 3.2 参数化 memory retriever

`Memento-main/memory/train_memory_retriever.py` 实现了一个成对判别模型：

```text
输入 1：case + plan 文本
输入 2：当前 query 文本
输出：truth_label，表示这个 case 对当前 query 是否有价值
```

模型结构：

1. 使用 `princeton-nlp/sup-simcse-roberta-base` 作为编码器。
2. 对 case 文本和 query 文本分别取 `[CLS]` 表征。
3. 拼接两个向量。
4. 通过两层 MLP 分类器输出二分类 logits。
5. 用 CrossEntropyLoss 训练。
6. 用 acc、f1、auc 验证。

`Memento-main/memory/parametric_memory.py` 在推理时把当前 query 和 memory pool 中每个 case 组成 pair，调用模型得到分数，再按分数排序取 TopK。

### 3.3 Memento 训练数据生成

`Memento-main/client/parametric_memory_cbr.py` 的在线更新逻辑是：

1. 对一个 query 检索若干 cases。
2. 用这些 cases 帮助 agent 解题。
3. 用 judge 判断最终答案是否正确。
4. 把每个 `(query, retrieved_case)` 都写入 training data。
5. `truth_label` 等于本次最终答案是否正确。
6. 当前 query 的最终 plan 也写回 memory。

这说明 Memento 的训练标签是“全局任务结果给检索案例的弱监督”。它并不精确知道某个 case 是否真正贡献了正确答案，但通过大量样本可以学习到更有价值的 case-query 匹配关系。

### 3.4 对 XENON-plus 的启发

Memento 对我们最重要的启发不是照搬模型，而是两点：

1. 案例选择可以从“相似度检索”升级为“可训练价值判断”。
2. 训练样本可以由 agent 自己的运行历史持续产生，不需要人工标注每一步。

但是 XENON-plus 不能直接复制 Memento：

- Memento 是问答任务，query-case 匹配主要发生在文本语义层。
- Minecraft 是连续交互任务，状态包括 inventory、equipment、位置、高度、物资账本、死亡/复活后的执行状态。
- Memento 的 `truth_label` 是整题正确性；XENON-plus 更需要 waypoint/action 级别的成败与效率。
- Memento 检索案例是为了增强 planner；XENON-plus 的第一版决策器是为了选择是否复用已有行动，必要时才回退 planner。

因此，我们应采用 Memento 的“成对打分”和“在线经验变训练数据”思想，但训练对象要改成 Minecraft 的状态-动作价值函数。

## 4. `train_value_head.py` 的训练思路

当前 `/home/yzb/zhourong/XENON-plus/train_value_head.py` 是一个更轻量的价值头方案：

```text
输入：一条 text
编码：all-MiniLM-L6-v2，输出 384 维向量
模型：MLPScorer，384 -> 256 -> 1
目标：二分类 label
损失：binary_cross_entropy_with_logits
类别不平衡：pos_weight = neg / pos
输出：value_head.pt
```

这个方案的优点：

1. 简单，成本低。
2. 和当前 case memory 已经使用的 `all-MiniLM-L6-v2` 一致。
3. 可以快速把每条 case 转成一条文本训练样本。
4. 适合作为第一版 decision value head。

当前不足：

1. 只输入单条文本，不能显式建模“当前状态 vs 候选案例”的 pair 关系。
2. 只有二分类 success label，没有 steps、失败类型、动作来源等辅助目标。
3. Dataset 每次 `__getitem__` 都在线 encode，训练会慢。
4. 没有 train/val/test 分组，容易把同一 run 的相似样本拆到训练和验证里导致泄漏。
5. 没有离线指标保存、阈值校准、TopK 选择评估。
6. 没有 hard negative，即“看起来相似但会失败”的样本。

结论：`train_value_head.py` 适合作为第一阶段原型，但后续需要升级为“retriever + value reranker”的结构。

## 5. 适合 XENON-plus 的决策器设计

### 5.1 第一版原则

第一版决策器应该保守，不要直接让模型控制全部行为：

1. 不让模型生成动作。
2. 不替换 planner。
3. 不用规则答案指导智能体。
4. 只让模型给候选案例或候选动作打分。
5. 低置信度时必须回退 planner。
6. 所有选择和结果继续写入案例库，形成后续训练数据。

### 5.2 两阶段结构

建议采用两阶段：

```text
当前状态 + waypoint
        |
        v
Stage 1: 快速检索
        |
        v
TopM 候选案例 / 候选动作
        |
        v
Stage 2: 可训练 value reranker
        |
        v
选择最高价值动作，或回退 planner
```

Stage 1 仍使用当前 `CaseBasedMemory._retrieve_cases()`：

- 用 `similarity_text` 做 embedding。
- 召回同 waypoint 或相似 waypoint 的 topM 成功/失败案例。
- 召回阶段要宽一些，宁愿多给 reranker 一些候选。

Stage 2 使用训练出来的 value head：

- 输入当前状态、waypoint、候选动作、候选案例摘要。
- 输出该候选行动在当前状态下的成功概率或价值分数。
- 按分数排序。

最终策略：

```text
如果 best_score >= min_success_prob
并且 best_score - second_score >= min_margin
并且动作可行性检查通过
则复用该案例动作
否则调用 planner
```

### 5.3 为什么不是一步到位训练 planner

当前任务阶段，直接训练 planner 不合适：

1. planner 的输出空间大，样本量还不够。
2. Minecraft 行为结果由 STEVE-1 执行能力、环境、资源分布共同决定，直接学规划会混入太多噪声。
3. 我们已有 planner fallback，风险可控。
4. 现在最需要提升的是“经验是否值得复用”的判断，而不是生成全新方案。

因此第一版训练目标应是 value/ranking，而不是生成式规划。

## 6. 决策器输入设计

### 6.1 当前状态特征

建议把当前状态序列化成稳定文本，不依赖模型自己从图像总结复杂信息。

必须包含：

- `final_task`：最终任务，如 `craft_a_golden_chestplate`
- `benchmark_type`：Wood / Stone / Iron / Gold / Diamond / Redstone / Armor
- `waypoint`：当前 waypoint
- `waypoint_num`：目标数量
- `inventory`：当前背包物品计数
- `plain_inventory`：槽位级背包信息，用于识别物品在背包还是仓库/栏位
- `equipment`：当前手持或装备
- `location_stats`：尤其是 ypos、pitch、yaw、biome_id
- `biome`：当前生物群系或环境
- `resource_ledger`：非视觉物资记录，如 max_inventory、pickup、mined_blocks、collected
- `previous_failed_actions`：同 waypoint 近期失败动作
- `recent_recovery_state`：是否刚 reset STEVE-1、是否刚死亡复活、是否刚切换 prompt

不建议第一版依赖：

- `trees visible nearby`
- `decision_reason`
- 复杂自然语言环境总结

原因是当前模型为 Qwen-2.5-7B，稳定生成高质量环境总结不一定可靠。训练数据应尽可能来自程序可观测事实。

### 6.2 候选动作特征

候选动作来自：

1. 同 waypoint 的成功案例动作。
2. 相似 waypoint 的成功案例动作。
3. 同 waypoint 的失败动作，用作 hard negative。
4. planner 当前生成的动作。
5. 后续如果 planner 支持多候选，可以直接扩展为 planner candidate list。

第一版不要求 planner 生成多个候选。即使 planner 每次只给一个动作，决策器仍然可以在“案例复用动作 vs planner fallback 动作”之间做判断。

### 6.3 候选案例摘要

候选案例摘要应包含：

- `case_waypoint`
- `case_selected_action`
- `case_outcome_success`
- `case_outcome_status`
- `case_source`
- `case_state_inventory`
- `case_equipment`
- `case_ypos`
- `case_final_goal`
- `case_steps`，如果能通过 `run_uuid` 连接到任务级结果
- `case_failure_type`，如 timeout、failed_incomplete_run、env_step_timeout、infra_early_stop

注意：legacy bootstrap case 没有真实环境和真实任务级 steps，不应作为第一版强监督训练样本。它可以继续作为冷启动检索材料，但训练时建议降低权重或排除。

## 7. 训练数据构造

### 7.1 数据来源

训练数据来自两类文件：

1. 案例库：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

2. 实验结果：

```text
exp_results/v1/*.json
exp_results/v2/*.json
```

通过 `run_uuid` 把 case 和任务结果连接起来，补充：

- `task_success`
- `status_detailed`
- `steps`
- `minutes`
- `failed_waypoints`
- `resource_ledger`
- `inventory_slots_used`
- `video_file`

### 7.2 样本粒度

建议第一版采用“case-decision 样本”：

```text
一条 case = 一次在某个 waypoint 下选择某个 action 的决策
```

每条样本包含：

```json
{
  "case_id": "xxx",
  "run_uuid": "xxx",
  "task": "craft_a_golden_pickaxe",
  "benchmark": "Gold",
  "waypoint": "gold_ingot",
  "waypoint_num": 3,
  "candidate_action": "smelt gold ore",
  "decision_source": "planner",
  "state_text": "...",
  "case_text": "...",
  "action_text": "...",
  "text": "...",
  "label_success": 1,
  "task_steps": 4783,
  "task_minutes": 3.99,
  "status_detailed": "success",
  "valid_for_training": true
}
```

这里的 `text` 是给当前轻量 value head 用的单文本输入。未来的 pair reranker 可以改成：

```json
{
  "query": "current_state + waypoint",
  "case": "candidate_case + candidate_action",
  "label_success": 1
}
```

### 7.3 标签设计

第一阶段标签：

```text
label_success = 1 if case.outcome.success is true else 0
```

需要过滤或单独标记：

- `pending`
- `infra_early_stop`
- 明确环境异常导致的开局退出
- 视频/环境 helper bug 造成的错误中断
- legacy case

这些样本不能直接当成失败样本，否则会污染决策器。

第二阶段可加入质量标签：

```text
quality = success_score - alpha * normalized_steps
```

建议不要一开始就强依赖 steps，因为当前 case 是 waypoint 级，`exp_results` 是任务级。任务级 steps 可以作为评估指标和弱质量信号，但不是精确的单个行动耗时。

第三阶段可加入失败类型：

```text
failure_type = timeout | invalid_action | stuck | resource_shortage | execution_bug | infra_error
```

这可以帮助决策器学习“看起来相似但风险类型不同”的案例。

### 7.4 hard negative

hard negative 对 Minecraft 很重要。不能只训练成功/失败二分类，否则模型会学到“某些 waypoint 总是容易成功”。

建议构造：

1. 同 waypoint 下失败的不同动作。
2. 同 final_task 下失败的动作。
3. 同一资源阶段中错误方向的动作，例如需要 smelt 却继续 dig down。
4. 同一状态下 planner fallback 后失败的案例。
5. 同一状态下 case reuse 成功与失败的对比。

注意不要把“没有尝试过的动作”直接标为负样本。未尝试不等于失败。

### 7.5 去重与样本权重

当前 `CaseBasedMemory._remove_exact_duplicate_case()` 会跳过完全相同的 resolved case。训练数据导出时仍建议再做一次去重：

```text
dedup_key = canonical(
  task,
  waypoint,
  waypoint_num,
  candidate_action,
  compact_inventory,
  equipment,
  ypos_bucket,
  biome,
  outcome.success,
  status_detailed
)
```

样本权重建议：

- 真实在线实验 case：权重 1.0
- legacy bootstrap case：权重 0 或 0.2
- 明确 bug/infra case：不进训练，保留审计
- 最近版本 v2 case：权重略高，例如 1.2，因为底层执行逻辑已经修正更多
- hard negative：可适当提高权重，例如 1.5

## 8. 模型方案

### 8.1 Stage A：轻量 value head

第一版从当前 `train_value_head.py` 演化即可：

```text
SentenceTransformer(all-MiniLM-L6-v2)
        |
384-d embedding
        |
MLP
        |
success logit
```

输入文本模板：

```text
[TASK] craft_a_golden_pickaxe
[WAYPOINT] gold_ingot x3
[STATE] inventory: iron_pickaxe=1, gold_ore=2, coal=3; equipment: iron_pickaxe; ypos: 18; biome: forest
[LEDGER] mined: gold_ore=2, diamond=1; pickup: gold_ore=2; max_inventory: gold_ore=2
[ACTION] smelt gold ore
[SOURCE] planner
[CASE] previous_success_count=..., previous_failure_count=...
```

训练目标：

```text
L_success = BCEWithLogits(success_logit, label_success)
```

这个阶段的目标是先超过“纯相似度检索”的选择效果。

### 8.2 Stage B：pair reranker

当数据量增加后，建议升级为 Memento 风格 pair reranker：

```text
query_text = 当前状态 + 当前 waypoint
case_text = 候选案例状态 + 候选动作 + 候选结果摘要
score = Q(query, case)
```

模型可以沿用 Memento 的结构：

```text
encoder(query) -> hq
encoder(case)  -> hc
concat(hq, hc, |hq-hc|, hq*hc) -> MLP -> score
```

相比 Memento 原版只拼接 `[h_case, h_query]`，建议增加 `|hq-hc|` 和 `hq*hc`，这样能更好表达状态差异。

训练目标：

```text
L_success = BCEWithLogits(score, label_success)
```

推理时：

1. 先用 embedding 召回 topM。
2. 对 topM 做 pair rerank。
3. 选择最高分案例。

这样可以避免对整个案例库做昂贵 pair scoring，同时让模型学会“当前状态和案例是否匹配”。

### 8.3 Stage C：多目标 value head

后续可以扩展为多头：

```text
shared encoder
  ├── success_head: P(success)
  ├── step_head: expected normalized steps
  └── failure_head: failure type
```

总损失：

```text
L = L_success + beta * L_steps + gamma * L_failure
```

建议顺序：

1. 先做 `success_head`。
2. 再加入 `step_head`。
3. 最后加入 `failure_head`。

不要一开始就做多目标，否则调试困难。

## 9. 决策策略

### 9.1 推理流程

建议 runtime policy：

```python
def decide(current_state, waypoint, planner_action=None):
    candidates = retrieve_cases(current_state, waypoint, top_m=16)
    candidates = filter_action_feasible(candidates, waypoint)

    scored = value_head.score(current_state, waypoint, candidates)
    best = max(scored, key=lambda x: x.score)

    if best.score >= min_success_prob and best.margin >= min_margin:
        return reuse_case(best)

    return call_planner()
```

第一版不必把 planner action 放入同一个排序池。更稳妥的做法是：

1. 先给 retrieved cases 打分。
2. 如果案例分数不够，调用 planner。
3. planner 结果执行并写入 case，未来成为训练样本。

等模型稳定后，再让 planner action 也进入候选池。

### 9.2 置信度阈值

初始阈值建议：

```yaml
decisioner:
  enabled: false
  model_path: checkpoints/decisioner/value_head.pt
  top_m: 16
  min_success_prob: 0.65
  min_margin: 0.10
  fallback_to_planner: true
  allow_legacy_training_cases: false
```

初期 `enabled` 默认 false，先做离线评估。在线实验时再打开。

### 9.3 与 planner 的关系

planner 仍然负责：

- 无优质案例时生成行动。
- 新 waypoint 或新环境阶段的冷启动。
- 给案例库提供新行动样本。

decisioner 负责：

- 判断案例动作是否值得复用。
- 避免重复使用已知低质量动作。
- 在相似动作中选更优动作。
- 学习不同状态下同一 waypoint 的行动差异。

## 10. 训练与评估方案

### 10.1 离线训练流程

建议新增脚本：

```text
scripts/export_decisioner_dataset.py
scripts/train_decision_value_head.py
scripts/evaluate_decisioner_offline.py
src/optimus1/decisioner/value_head.py
src/optimus1/decisioner/feature_text.py
```

流程：

1. 从 `cases.json` 和 `exp_results` 导出 jsonl。
2. 按 `run_uuid` 或 final task 分组划分 train/val/test。
3. 训练 value head。
4. 在 validation 上调阈值。
5. 在 test 上报告离线指标。
6. 仅当离线指标超过 retrieval-only baseline 时进入在线实验。

### 10.2 划分方式

不要随机按 case 划分。原因是同一个 run 中的多个 waypoint 高度相关，随机切分会泄漏。

推荐：

```text
group key = run_uuid
```

或者更严格：

```text
group key = original_final_goal + benchmark + seed/env
```

第一版可以：

- train：70%
- validation：15%
- test：15%

同时保留一个跨类别测试集，例如训练 Wood/Stone/Iron/Gold，测试 Diamond/Redstone/Armor 中部分任务，用于评估迁移能力。

### 10.3 离线指标

分类指标：

- AUC
- AP
- F1
- accuracy
- calibration error

决策指标：

- Top1 是否选择成功案例
- success@K
- 同 waypoint 内 action ranking accuracy
- planner fallback 触发率估计
- 高置信度样本的实际成功率

效率指标：

- 预测成功且真实成功的平均 task steps
- 成功任务的平均 steps
- 相对 retrieval-only 的 steps 变化

数据质量指标：

- 训练样本数
- 正负样本比例
- legacy 样本比例
- infra/bug 样本过滤数量
- 每个 benchmark 的样本覆盖

### 10.4 在线对比实验

至少做四组：

1. XENON-main 原版。
2. XENON-plus 当前 retrieval-only case memory。
3. XENON-plus value-head decisioner。
4. XENON-plus value-head decisioner + planner fallback。

在线指标：

- 67 个任务成功数。
- 各类别成功数：Wood、Stone、Iron、Gold、Diamond、Redstone、Armor。
- 平均 steps。
- 成功任务平均 steps。
- planner 调用次数。
- case reuse 次数。
- case reuse 成功率。
- 正常失败 vs infra early stop 数量。

实验记录仍使用当前 `exp_results/v2` 或后续 `v3`，视频不进入 GitHub，只保留结果 json 和 summary。

## 11. 与当前环境感知机制的关系

当前 XENON-plus 已有一些执行层修正：

- 地表找树探索：`chop a tree` stale 后临时切 `find a tree`。
- 收集到第一个 log 后短时间限制跳跃。
- 地下矿物层感知：目标矿物不足且看到更深层矿物时，把 prompt 从 dig down 切为 dig forward。
- resource ledger：通过非视觉方式记录采集、背包最大值、矿物变化。
- respawn 后 reset STEVE-1 和低层控制状态。

这些机制属于执行层和 prompt 调整层，不应混入第一版决策器训练成“硬规则”。但它们产生的运行结果会影响 case outcome，因此会自然进入训练数据。

决策器第一版只学习：

```text
当前状态 + waypoint + 候选动作 -> 这个动作是否值得执行
```

不直接学习：

```text
什么时候切 find a tree
什么时候切 dig forward
怎么逃出水
怎么挖矿
```

这样边界清晰，后续 debug 时也能区分是 decisioner 问题还是 executor/prompt 问题。

## 12. 关键风险

### 12.1 标签污染

失败样本不一定是智能体能力失败，可能是环境 bug、视频异常、helper bug、开局直接退出。必须在导出训练数据时区分：

- valid failure：可以训练
- infra failure：不训练，只统计
- superseded bug：不训练

否则决策器会把正确动作学成错误动作。

### 12.2 全局成功标签的信用分配问题

如果一个任务最终失败，不能简单认为所有 waypoint 动作都失败。例如前面木头、石稿成功了，最后金矿不足导致任务失败，那么前面的动作仍然可能是好动作。

当前 case outcome 已经是 waypoint/action 级，比 Memento 的全局 weak label 更适合训练。后续如果要用 task-level steps 做质量评估，需要谨慎，不要把任务失败平均摊到所有成功 waypoint 上。

### 12.3 legacy case 过强

legacy case 从原始动作库迁移而来，没有真实环境状态。如果训练时大量使用 legacy，会让模型回到“waypoint -> 固定动作”的旧模式。

建议：

- 训练时默认排除 legacy。
- 检索冷启动时可以保留 legacy。
- 在线结果足够多后逐步降低 legacy 的 runtime 优先级。

### 12.4 案例库变大后的效率

当前案例库增长后，纯 embedding 检索每次对全部 case 做相似度排序，规模到几千条还能接受，但到几万条会影响速度。

建议后续：

1. 保存 case embeddings 到磁盘，增量更新。
2. 用 FAISS 或 numpy 矩阵做快速 topM。
3. 先按 waypoint 建倒排索引，再做向量召回。
4. value reranker 只对 topM 运行，不对全库运行。

### 12.5 prompt 生成质量

不要把训练特征设计成依赖 Qwen-2.5-7B 总结的长 reasoning。环境特征应尽量来自程序状态和 resource ledger。自然语言只是结构化字段的稳定渲染。

## 13. 推荐落地顺序

### Step 1：导出数据集

新增：

```text
scripts/export_decisioner_dataset.py
```

输出：

```text
data/decisioner/value_data_v1.jsonl
data/decisioner/audit_filtered_cases_v1.jsonl
data/decisioner/dataset_summary_v1.json
```

要求：

- join `cases.json` 与 `exp_results`。
- 过滤 pending 和 infra early stop。
- 标记 legacy。
- 生成稳定 `text`。
- 按 run_uuid 分组划分 split。

### Step 2：整理训练脚本

把当前根目录 `train_value_head.py` 改造成：

```text
scripts/train_decision_value_head.py
src/optimus1/decisioner/value_head.py
```

优化点：

- 预先缓存 embeddings。
- 增加 validation。
- 保存 metrics。
- 保存 best checkpoint。
- 支持 pos_weight。
- 支持 threshold calibration。

### Step 3：离线评估

新增：

```text
scripts/evaluate_decisioner_offline.py
```

至少输出：

```text
reports/decisioner/offline_eval_v1.json
reports/decisioner/offline_eval_v1.md
```

需要比较：

- exact waypoint retrieval baseline
- cosine retrieval baseline
- value head reranker

### Step 4：受控接入运行时

在 `CaseBasedMemory.select_case_decision()` 的 retrieval 后加入可选 rerank：

```text
case_memory retrieval -> value head rerank -> threshold gate -> reuse or planner
```

配置默认关闭：

```yaml
memory:
  case_memory:
    trainable_decisioner:
      enabled: false
      checkpoint: checkpoints/decisioner/value_head.pt
      top_m: 16
      min_success_prob: 0.65
      min_margin: 0.10
```

这样不会影响当前实验基线。

### Step 5：在线验证

新建 `videos/v3` 和 `exp_results/v3`，跑完整 67 任务，比较：

- v2 retrieval-only
- v3 trainable decisioner

如果 v3 在成功数或 steps 上没有改善，就先回到离线数据质量分析，不继续扩大实验。

## 14. 第一版最小可行方案

最小可行版本只需要完成：

1. 导出 `value_data_v1.jsonl`。
2. 用当前 `train_value_head.py` 风格训练二分类 value head。
3. 离线评估 value head 是否能区分成功/失败 case。
4. 不接入 runtime，先产出评估报告。

这一步的成功标准：

- validation AUC 明显高于 0.5。
- 高置信度样本成功率高于整体成功率。
- 对同 waypoint 候选动作的排序优于 exact waypoint baseline。
- 过滤掉 infra/bug 后，负样本仍有足够数量。

如果第一版离线效果不好，优先检查：

1. 训练文本是否包含足够状态信息。
2. 失败标签是否污染。
3. 是否被 legacy 样本主导。
4. 是否 train/test 泄漏或分布不一致。
5. 是否同 waypoint 样本过少，无法学习状态差异。

## 15. 后续增强方向

### 15.1 从 binary success 到 value ranking

当每个 waypoint 有足够多同类样本后，把训练从二分类升级为排序：

```text
同 waypoint、相似状态下：
成功且 steps 少的动作 > 成功但 steps 多的动作 > 失败动作
```

可用 pairwise ranking loss：

```text
L_rank = max(0, margin - score_positive + score_negative)
```

### 15.2 加入 per-waypoint steps

当前已经有任务级 steps，但 case 级还没有精确 duration。建议后续在 waypoint 完成时记录：

- `decision_step_start`
- `decision_step_end`
- `decision_duration_steps`
- `waypoint_obtained_step`

这样决策器可以学习“哪个动作不仅成功，而且更快”。

### 15.3 失败类型分类

后续可从视频观察和日志中总结失败类型，再半自动标注：

- stuck_water
- stuck_underground
- no_tree_found
- wrong_tool
- inventory_full
- smelt_missing
- dig_too_deep
- action_infeasible
- infra_early_stop

第一版不要把这些类型写成硬逻辑，而是作为训练标签和分析维度。

### 15.4 planner 多候选接口

如果后续要让决策器真正“从 planner 候选方案中选择”，需要 planner 返回多个候选动作：

```json
[
  {"action": "dig down and mine gold_ore", "reason": "..."},
  {"action": "dig forward and mine gold_ore", "reason": "..."},
  {"action": "smelt gold ore", "reason": "..."}
]
```

但这不是第一版必要条件。第一版只需在案例动作和 planner fallback 之间做价值判断。

## 16. 推荐的文档与代码产物

建议后续新增：

```text
docs/decisioner_training_plan_implementation.md
data/decisioner/value_data_v1.jsonl
data/decisioner/dataset_summary_v1.json
reports/decisioner/offline_eval_v1.md
src/optimus1/decisioner/__init__.py
src/optimus1/decisioner/feature_text.py
src/optimus1/decisioner/value_head.py
scripts/export_decisioner_dataset.py
scripts/train_decision_value_head.py
scripts/evaluate_decisioner_offline.py
```

## 17. 结论

推荐路线是：

1. 保留当前 case memory 的 retrieval-only 决策作为基线。
2. 从 `train_value_head.py` 的轻量 MLP success scorer 开始，快速建立离线训练闭环。
3. 训练数据来自 XENON-plus 自己的 case memory 和实验结果，而不是人工写规则。
4. 第一版模型只做“案例/动作价值打分”，不生成动作，不替代 planner。
5. 等数据量和离线效果稳定后，再升级为 Memento 风格的 pair reranker。
6. 在线接入必须有阈值和 planner fallback，避免模型低置信度时强行复用错误案例。

这个方案能保持当前项目的核心创新方向：案例库不只是日志，而是可训练的决策经验库；决策器不直接获得正确答案，而是从真实成功和失败经验中学习什么行动在什么状态下更值得执行。
