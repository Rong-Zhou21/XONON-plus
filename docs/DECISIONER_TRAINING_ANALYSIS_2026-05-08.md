# XENON-plus 决策器训练机制分析

生成日期：2026-05-08  
项目目录：`/home/yzb/zhourong/XENON-plus`  
核心结论：当前决策器不是规则 planner，也不是端到端强化学习控制器，而是一个基于案例库训练出来的 **RADS（Retrieval-Augmented Decision Scorer）成功率打分器**。它学习的问题是：

```text
给定当前状态 state、当前 waypoint、最终任务 final_goal、候选动作 action，
预测这个 action 在当前状态下完成该 waypoint 的成功概率 P(success)。
```

它在运行时不直接生成新动作，而是在 case memory 已有候选动作之间排序；如果最高成功概率低于阈值，就回退到 planner。

## 1. 它在整个系统里的位置

相关文件：

- `scripts/export_decisioner_dataset.py`
- `scripts/train_rads.py`
- `scripts/evaluate_rads_offline.py`
- `src/optimus1/decisioner/feature.py`
- `src/optimus1/decisioner/encoder.py`
- `src/optimus1/decisioner/rads.py`
- `src/optimus1/decisioner/runtime.py`
- `src/optimus1/memories/case_memory.py`
- `data/decisioner/rads_v1.jsonl`
- `artifacts/decisioner/rads_v2.pt`
- `reports/decisioner/offline_eval_v2.md`
- `reports/decisioner/v2_vs_v3_comparison.md`

运行链路是：

```text
cases.json
  -> export_decisioner_dataset.py
  -> data/decisioner/rads_v1.jsonl
  -> train_rads.py
  -> artifacts/decisioner/rads_v2.pt
  -> RADSRuntime
  -> CaseBasedMemory._select_case_decision_rads()
  -> 复用历史动作，或低置信度 fallback planner
```

这说明 XENON-plus 的决策器训练是一个离线监督学习闭环：智能体以前运行产生的 case 成为训练数据，训练好的模型再用于后续 case 选择。

## 2. 训练数据怎么来

数据源是：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

导出脚本是 `scripts/export_decisioner_dataset.py`。它只使用 case memory 内已有字段，不额外 join `exp_results`、视频或 resource ledger。每条训练样本保留：

- `case_id`
- `run_uuid`
- `_position_in_run`
- `waypoint`
- `waypoint_num`
- `original_final_goal`
- `selected_action`
- `decision_source`
- `state_snapshot.inventory`
- `state_snapshot.equipment`
- `state_snapshot.biome`
- `state_snapshot.location_stats.ypos`
- `state_snapshot.location_stats.biome_id`
- `outcome.success`

过滤规则：

- 丢弃 `run_uuid == legacy`
- 丢弃 `outcome.status in {pending, excluded_infra, crash_RuntimeError}`
- 只保留 `outcome.success` 明确为 `true` 或 `false` 的样本

当前导出统计：

| 项目 | 数量 |
|---|---:|
| 原始 case | 2660 |
| 有效训练样本 | 2574 |
| drop legacy | 35 |
| drop excluded_infra | 44 |
| drop crash_RuntimeError | 6 |
| drop pending | 1 |

划分方式是按 `run_uuid` group split，而不是随机按 case split：

| split | samples | runs | positive | negative |
|---|---:|---:|---:|---:|
| train | 1847 | 170 | 1542 | 305 |
| val | 351 | 36 | 330 | 21 |
| test | 376 | 38 | 307 | 69 |

为什么要按 `run_uuid` 分组：同一次任务运行里的多个 waypoint case 高度相关。如果把同一个 run 的相邻 case 同时放进 train 和 test，模型会“看见同一局游戏的近邻经验”，离线指标会虚高。按 run 分组可以更接近新 episode 推理时的真实难度。

为什么要过滤 pending/crash/infra：这些样本的失败不一定来自动作本身。例如环境崩溃、基础设施超时、尚未结算的 pending case，都不能稳定表达“这个动作在这个状态下不好”。把它们当负样本会污染 `P(success)` 的语义。

## 3. 特征是怎么构造的

实际特征定义在 `src/optimus1/decisioner/feature.py`。总输入维度是 52：

```text
numeric dense features: 34
categorical embeddings: waypoint/action/final_goal 各 6 维，共 18
total: 52
```

数值和 one-hot 特征包括：

| 特征 | 物理含义 |
|---|---|
| equipment one-hot | 当前手上/装备状态，例如是否已有木镐、石镐、铁镐 |
| biome one-hot | 当前环境类型，目前主要是 forest/plains/unk |
| waypoint_num log1p | 当前 waypoint 需要的数量 |
| position_in_run log1p | 当前决策处在一次任务运行的第几个 case |
| ypos_norm | 当前高度，`ypos / 64` 后截断 |
| ypos_bucket | 高度桶：地表、高层、浅层、矿层、深层 |
| inventory key items | log、planks、stick、coal、cobblestone、iron、diamond、gold 等关键物资数量 |
| tool_owned_flags | 是否拥有 wooden/stone/iron pickaxe |
| inv_unique_count | 背包物品种类数 |

类别 embedding 包括：

- `waypoint_id`
- `final_goal_id`
- `action_id`

此外 v2 加入了一个单独的经验先验：

```text
P(success | waypoint, selected_action)
```

它由 train split 中同一 `(waypoint, action)` 的历史成功率 Laplace smoothing 得到，推理时以 log-odds residual 形式加到最终 logit 上。

为什么这样设计：Minecraft 任务的高层决策强依赖“现在有什么、工具等级到哪、在哪个高度、当前 waypoint 是什么、最终目标是什么”。例如同样是 `dig down and mine diamond`，如果没有铁镐、当前高度太高、背包关键材料不够，成功概率应该不同。只用文本相似度或 waypoint 名字无法表达这些物理约束。

但它也有边界：当前决策器没有直接输入第一人称图像，不会看画面识别矿物；它也不接收完整邻近方块拓扑。因此它学到的是 case 级状态-动作成功率，不是“视觉找矿/挖通路”的低层技能。

## 4. 模型结构：RADS

模型实现位于 `src/optimus1/decisioner/rads.py` 和 `src/optimus1/decisioner/encoder.py`。

核心结构：

```text
q   = QueryEncoder(query)                 # 当前待判断状态，64d
C   = CaseEncoder(library_cases)          # 训练集案例库，64d
att = softmax((q @ C.T) / tau, masked)    # 对历史案例做注意力检索
ctx = att @ C                             # 历史经验上下文
h   = MLP([q, ctx, action_embedding])
logit = decision_head(h) + prior_logit
P(success) = sigmoid(logit)
```

QueryEncoder 和 CaseEncoder 的区别：

- QueryEncoder 编码当前“还不知道结果”的候选动作。
- CaseEncoder 编码历史 case，并额外输入 `outcome` success/fail one-hot。
- 两个 encoder 共享 waypoint/final_goal/action embedding 表，保证 query 和 case 在同一语义空间里比较。

为什么用检索增强，而不是普通 MLP：

1. Minecraft case memory 本身就是系统的核心创新。RADS 保留“参考哪些历史案例”的可解释性。
2. 纯 MLP 只能输出分数，不知道分数来自哪些经验；RADS 可以输出 top-k evidence。
3. 历史 case 数量不大，训练集不到 2000 条，把 train cases 作为 attention library 成本可控。
4. 运行时可以把同 run case mask 掉，降低泄漏和自我引用。

## 5. 训练目标是什么

训练脚本是 `scripts/train_rads.py`。核心损失：

```text
L1 = BCEWithLogitsLoss(success_logit, outcome.success)
L2 = TripletMarginLoss(case vectors)
L3 = CrossEntropy(waypoint_head(case_vector), waypoint_id)

L = L1 + 0.1 * L2 + 0.05 * L3
```

### 5.1 BCE 主损失

BCE 直接训练 `P(success)`。标签来自 `outcome.success`：

- 成功 case：1
- 失败 case：0

训练集正负样本不平衡，train split 是 1542 正 / 305 负，所以脚本设置：

```text
pos_weight = n_neg / n_pos
```

这会避免模型因为正样本太多而过度偏向“全部预测成功”。当前 `pos_weight` 小于 1，本质是在降低多数类正样本的权重，让负样本仍然能影响边界。

为什么主目标不是模仿 `selected_action`：case 里记录的动作可能来自 planner、case memory 或 fallback；仅模仿动作会把历史错误也学进去。用最终 `outcome.success` 训练成功率，模型学的是“这个动作在这个状态下是否有效”，更符合决策器职责。

### 5.2 Triplet loss

Triplet 采样规则：

- anchor：某个历史 case
- positive：同 waypoint、同 outcome 的 case
- negative：同 waypoint、不同 outcome 的 case
- 优先选择不同 `run_uuid`

目标是让同一个 waypoint 下成功经验靠近成功经验，失败经验远离成功经验。

为什么需要它：如果只用 BCE，case embedding 可能只服务最终分类头，不一定形成可检索的案例空间。Triplet loss 强迫“检索出来的历史案例”本身具有成败区分能力，这样 evidence 才有意义。

### 5.3 Waypoint reconstruction

CaseEncoder 还有一个辅助 head，从 case vector 预测 waypoint。

为什么需要它：RADS 的 attention 如果不受约束，可能会跨 waypoint 找到一些全局相似但任务语义错误的案例。比如把 `craft stick` 和 `craft planks` 这类高成功率 crafting case 混在一起，离线 AUC 可能还不错，但运行时会选错动作。Waypoint reconstruction 让 case vector 保留“我到底在解决哪个 waypoint”的信息。

### 5.4 Same-waypoint hard mask

v2 在 attention 里加入同 waypoint 硬过滤：

```text
如果当前 query 的同 waypoint 历史案例数 >= same_wp_min(8)，
attention 只允许看同 waypoint case；
否则 fallback 到全库。
```

为什么这样做：v1 的总体 AUC 一度很高，但 attention top-1 只有约 9% 来自同 waypoint，说明模型在用跨 waypoint 的统计捷径。v2 强制同 waypoint 优先后，test top-1 same-waypoint rate 到 94.41%，更符合运行时动作选择的物理语义。

### 5.5 Waypoint-action 先验

v2 加入：

```text
logit += trainable_weight * logit(P_train(success | waypoint, action))
```

为什么这样做：很多 waypoint 只有一个常见动作，部分 rare action 样本很少。如果只靠 action embedding，模型可能把“craft 类动作整体成功率高”的偏见迁移到某些实际失败的 rare action 上。把 `(waypoint, action)` 的经验成功率直接加到输出层，可以让模型在小样本条件下不偏离历史统计。

## 6. 训练过程细节

默认训练参数：

| 参数 | 当前值 |
|---|---:|
| epochs | 20 |
| patience | 5 |
| batch_size | 64 |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| dropout | 0.2 |
| triplet_weight | 0.1 |
| wp_weight | 0.05 |
| same_wp_min | 8 |
| prior_logit_init_weight | 1.5 |
| triplets_per_step | 128 |
| seed | 20260501 |

训练时，每一步都会重新编码整个 train case library：

```text
library_vecs = model.encode_cases(train_library)
```

为什么不是预先固定 library_vecs：如果固定向量，CaseEncoder 无法通过当前 batch 的 loss 继续学习。当前 train set 小于 2000 条，重新编码全库成本可以接受，因此选择端到端更新 QueryEncoder、CaseEncoder 和 attention scorer。

训练时还会 mask：

- 当前 query 所属 `run_uuid` 的所有 library case
- query 自己对应的 case

这是为了避免模型直接从同一次运行里“抄答案”。

当前 `rads_v2.pt` 的最佳模型出现在 epoch 14：

| 指标 | val | test |
|---|---:|---:|
| samples | 351 | 376 |
| pos_rate | 0.9402 | 0.8165 |
| AUC | 0.8763 | 0.9110 |
| AP | 0.9917 | 0.9739 |
| best F1 | 0.9455 | 0.9535 |
| ECE | 0.1190 | 0.0866 |
| mean attention concentration | 0.8429 | 0.8417 |

## 7. 推理时怎么用

运行时接入在 `src/optimus1/memories/case_memory.py`。

当配置开启：

```text
memory.case_memory.decisioner.enabled=true
memory.case_memory.decisioner.checkpoint=artifacts/decisioner/rads_v2.pt
memory.case_memory.decisioner.min_p_success=0.20
```

流程变为：

1. `CaseBasedMemory` 读取 checkpoint，加载 `RADSRuntime`。
2. 对当前 waypoint，从 case memory 中收集所有历史出现过的 distinct action。
3. 每个 action 构造一个 query case：
   - 当前 `state_snapshot`
   - 当前 `waypoint`
   - 当前 `waypoint_num`
   - 当前 `original_final_goal`
   - 候选 `selected_action`
   - 当前 run 内位置 `_position_in_run`
4. RADS 分别输出每个 action 的 `p_success`。
5. 按 `p_success` 排序。
6. 如果最高分低于 `min_p_success=0.20`，返回 `None`，上层调用 planner。
7. 如果最高分达标，就复用该 action 的代表 case，并把 candidates/evidence 写入 `decision_trace`。

这意味着运行时的智能体表现不是“模型自己想出新动作”，而是：

- 对已有历史动作做成功率排序；
- 对低置信状态主动交还 planner；
- 保留 evidence，方便复盘“它为什么选这个动作”。

## 8. 训练带来的效果

### 8.1 离线效果

`reports/decisioner/offline_eval_v2.md` 显示：

| 方法 | test AUC | AP | best F1 | ECE |
|---|---:|---:|---:|---:|
| RADS v2 | 0.9110 | 0.9739 | 0.9535 | 0.0866 |
| history_count_baseline | 0.8555 | 0.9431 | 0.9119 | 0.0419 |
| majority_class_baseline | 0.5000 | 0.8165 | 0.8990 | 0.0000 |

RADS 比现有 `_best_exact_success_case` 风格的历史计数 baseline AUC 高约 0.0555，说明它确实学到了状态条件下的成功率，而不只是复述 `(waypoint, action)` 的全局成功次数。

更关键的是 attention 诊断：

| 指标 | v1 | v2 |
|---|---:|---:|
| top-1 attention same-waypoint | 0.0931 | 0.9441 |
| top-1 attention same-outcome | 0.1463 | 0.5239 |
| top-1 attention same-both | 0.0027 | 0.4894 |

这说明 v2 的检索证据从“统计上可能有用但任务语义混乱”变成了“高度集中在同 waypoint 的历史经验”。虽然 v2 的 test AUC 比 v1 低一些，但它更适合作为运行时 action ranker。

多动作 waypoint 上也能看到修正：

| waypoint | v2 表现 |
|---|---|
| cobblestone | successful eval 上 top-1 match 从 v1 的 0% 到 v2 的 100% |
| stone | successful eval 上 top-1 match 从 v1 的 0% 到 v2 的 100% |

这正是 `same_wp_min` 和 `(waypoint, action)` 先验要解决的问题：避免模型在多候选 waypoint 上选到语义不合适但全局看起来“像成功动作”的候选。

### 8.2 在线效果

`reports/decisioner/v2_vs_v3_comparison.md` 的 67 任务单次运行对比：

| 方法 | success / 67 | rate |
|---|---:|---:|
| v2 retrieval-only first attempt | 39 | 58.2% |
| v3 RADS decisioner single shot | 49 | 73.1% |
| v2 best-of-N retries | 57 | 85.1% |

线上提升约 15 个百分点。RADS 把 retrieval-only 首次失败的 18 个任务里追回了 10 个，说明它的收益不是只停留在离线指标上。

按 benchmark：

| benchmark | tasks | v2 first | v3 RADS | delta |
|---|---:|---:|---:|---:|
| wooden | 10 | 8 | 10 | +2 |
| stone | 9 | 7 | 6 | -1 |
| iron | 16 | 7 | 10 | +3 |
| golden | 6 | 3 | 2 | -1 |
| diamond | 7 | 5 | 7 | +2 |
| redstone | 6 | 2 | 6 | +4 |
| armor | 13 | 7 | 8 | +1 |
| total | 67 | 39 | 49 | +10 |

决策来源统计：

| source | count | success |
|---|---:|---:|
| rads_decisioner | 628 | 606 |
| planner fallback | 31 | 31 |

这个结果说明 `min_p_success` gate 是有用的：当 RADS 判断已有案例动作风险太高时，fallback planner 并没有破坏任务，反而在记录中全部出现在成功任务里。

## 9. 物理行为层面的含义

训练后的决策器对智能体行为的影响可以概括为三点。

第一，它让“复用历史动作”从硬规则变成状态条件判断。原来的 retrieval-only 逻辑更像：

```text
这个 waypoint 以前成功过某个动作 -> 直接复用
```

RADS 则变成：

```text
这个 waypoint 的每个候选动作，在当前 inventory/equipment/ypos/final_goal 下分别有多大成功率？
```

这会让智能体在工具不足、位置不合适、任务上下文不同的时候更容易拒绝盲目复用。

第二，它让多动作 waypoint 更稳定。例如 `cobblestone`、`stone` 这类 waypoint 可能存在多个历史动作，普通历史计数或文本相似度容易被全局成功率带偏。RADS v2 通过同 waypoint attention 和 waypoint-action 先验，让选择更贴近当前 waypoint 的真实候选动作。

第三，它不会直接提升低层操作能力。比如挖矿时卡住、看不准矿、pillar-up 后横向移动失败、砍树执行超时，这些主要发生在 `env/wrapper.py` 和 STEVE-1 执行层。RADS 只能决定“应该尝试哪个高层动作，或者是否回退 planner”，不能自己学会转头、挖通路或识别画面里的矿物。

这也解释了线上报告里的失败分布：v3 losses 多集中在 logs、iron_ore、gold_ore 等资源采集执行失败，而不是决策器明显选错动作。

## 10. 为什么不是直接训练一个更强 planner

当前训练方式保守但合理：

1. 现有数据只有 case 级状态、动作和 outcome，没有逐帧专家轨迹，无法直接训练低层控制器。
2. Minecraft 在线 rollout 成本高，离线复用历史 case 是低成本可迭代路径。
3. planner 仍然保留，RADS 只做 rerank/gate，失败时有 fallback。
4. case memory 本身会继续积累新经验，后续可以重新导出数据再训练。
5. RADS 的 evidence 可解释，便于区分“决策错了”还是“执行层没完成”。

所以当前训练目标不是让智能体“一步到位真正理解世界”，而是先让它在已有经验上学会一个更稳的高层动作价值函数。

## 11. 当前局限

1. 候选动作覆盖不足。线上报告指出 74 个 waypoint 里约 70 个只有一个历史 action，RADS 在这些位置只能做 accept/reject，不能真正多选一。
2. 决策器没有视觉输入。它不能在挖矿前直接从画面判断眼前是哪种矿物，也不能判断周围方块拓扑。
3. 标签是弱监督。`outcome.success=false` 可能来自执行失败、超时、环境随机性，并不总是动作选择错误。
4. 训练数据来自旧策略分布。后续 v7 的挖矿、pillar-up、横移机制改变了数据分布，当前 `rads_v2.pt` 没有见过这些新行为的完整结果。
5. val split 负样本很少，只有 21 个 negative，因此 val AUC 波动较大，应该更看重 test 和在线 A/B。
6. `min_p_success=0.20` 是经验阈值，不是经过系统 threshold sweep 得到的最优值。

## 12. 后续建议

如果要让决策器继续进化，优先级建议如下：

1. 等 v7 机制稳定后，把新 case memory 重新导出为 `rads_v2` 或 `rads_v3` 数据集，重新训练 RADS，让它看见新的地下挖矿分布。
2. 增加候选动作多样性，尤其是 mining waypoint：同一个 `gold_ore` 或 `diamond` 可以有不同高层策略，例如直接 dig down、relevel 后 lateral dig、planner fallback、回地表补工具等。
3. 把 failure reason 分层，不要把所有失败都压成一个 `success=false`。例如 execution_stuck、timeout、missing_tool、resource_not_found、crafting_failed 应该有不同训练意义。
4. 加入更直接的环境感知特征，例如最近采集统计、resource ledger、是否处于地下、是否刚触发 overshoot/pillar_up、是否近期水平位移失败。
5. 对 `min_p_success` 做离线和在线阈值扫描，找出更适合不同 waypoint 的 fallback threshold。
6. 如果未来要让它“真正判断环境和矿物”，需要引入视觉/方块感知输入，或者让低层执行模块提供结构化观测；仅靠当前 case 特征无法学会看图识矿。

## 13. 当前训练如何复现

从当前 case memory 重新生成数据集：

```bash
python scripts/export_decisioner_dataset.py
```

训练当前默认的 RADS v2：

```bash
python scripts/train_rads.py \
  --data data/decisioner/rads_v1.jsonl \
  --out artifacts/decisioner/rads_v2.pt \
  --report reports/decisioner/training_log_v2.json
```

离线评估：

```bash
python scripts/evaluate_rads_offline.py \
  --artifact artifacts/decisioner/rads_v2.pt \
  --data data/decisioner/rads_v1.jsonl \
  --report_md reports/decisioner/offline_eval_v2.md \
  --report_json reports/decisioner/offline_eval_v2.json
```

线上启用方式是在运行脚本里传入：

```text
memory.case_memory.decisioner.enabled=true
memory.case_memory.decisioner.checkpoint=artifacts/decisioner/rads_v2.pt
memory.case_memory.decisioner.min_p_success=0.20
```

例如 `scripts/run_v7_armor_targeted.sh` 默认会开启 `DECISIONER_ENABLED=1`，使用 `artifacts/decisioner/rads_v2.pt`。

## 14. 一句话总结

当前决策器的训练方式是：从 case memory 导出状态-动作-结果样本，用 RADS 学习 `P(success | state, waypoint, action)`，并通过检索增强、same-waypoint attention、triplet loss、waypoint reconstruction 和 waypoint-action 先验，让模型既能打分也能给出历史证据。它的实际效果是把高层案例复用从硬规则提升为可学习的成功率判断，线上单次成功率从 58.2% 提高到 73.1%；但它仍然不是视觉挖矿能力或低层移动控制器，资源采集失败仍主要需要执行层和感知层继续优化。
