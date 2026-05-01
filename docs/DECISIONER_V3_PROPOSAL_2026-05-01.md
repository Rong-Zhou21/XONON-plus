# XENON-plus 决策器 v3 实施方案

日期：2026-05-01
本轮范围：**仅离线训练决策器**，不接入 runtime，不改案例库 schema，不改 wrapper。

## 1. 边界（本轮明确不做）

- 不修改 `cases.json` schema 或 `case_memory.py` 的现有逻辑
- 不修改 `env/wrapper.py` 的执行层（包括地表找树机制）
- 不修改 OracleGraph、planner、STEVE-1
- 不预测 timeout、不做 budget abort、不主动放弃 waypoint
- 不做主动多样性数据采集
- 仅训练 + 离线评估；runtime 接入留到下一轮

## 2. 设计核心

决策器 = **Retrieval-Augmented Decision Scorer (RADS)**：
对查询 (state, waypoint, candidate_action, task_context)，从已编码的案例库中做 cross-attention，返回 P(success) + top-k attention cases 作为可解释依据。

可信度 = `P(success) × attention_concentration`。attention 越集中在少数 case → 模型对该状态有清晰历史经验支持。

## 3. 输入特征（仅用 cases.json 已有字段，~50 维）

```
Categorical embeddings (28d):
  - waypoint_id           (74 类 → 6d)
  - final_goal_id         (67 类 → 6d)
  - candidate_action_id   (~30 distinct → 6d)
  - equipment_id          (10 类 → 6d)
  - biome_id              (~10 类 → 4d)

Numeric (6d):
  - waypoint_num          log1p
  - position_in_run       log1p (从 cases.json 的 created_at/index 派生)
  - ypos                  normalized
  - ypos_bucket           5d onehot

Inventory key items log1p (13d):
  - log_total, planks_total, stick, coal, cobblestone,
    iron_ore, iron_ingot, diamond, gold_ore, gold_ingot,
    crafting_table, furnace, inv_unique_count

Tool ownership flags (3d):
  - has_wooden_pickaxe, has_stone_pickaxe, has_iron_pickaxe
```

总约 50 维。明确**不依赖** resource_ledger / exp_results / save_decomposed_plan（这些不在 cases.json 中）。`position_in_run` 从案例库自身的 run_uuid 分组派生，仍属"案例库已有信息"。

## 4. 模型

```
QueryEncoder: 50d feature → MLP(128) → 64d
CaseEncoder:  50d feature + outcome_onehot(2d) → MLP(128) → 64d
              + 辅助 head: 64d → N_waypoints (waypoint reconstruction)

RADS:
  q = QueryEncoder(query)
  C = CaseEncoder(library)              # offline pre-computed
  α = softmax(q·Cᵀ / τ)                  # τ trainable
  ctx = α @ C
  logit = MLP_head([q, ctx, action_emb]) # → P(success)
  evidence = top-k cases by α

  推理时 mask same run_uuid cases from retrieval pool
```

参数量预估 < 50K，artifact 总大小 < 1MB。

## 5. 训练目标

```
L1 = BCE(P(success), case.outcome.success)              # 主目标
L2 = TripletMargin(case_anchor, c+, c-)                  # 同 wp+同 outcome 拉近
                                                         # 同 wp+异 outcome 推远
L3 = CrossEntropy(WP_head(C), case.waypoint_id)          # 强制保留 waypoint 信息

L = L1 + 0.3·L2 + 0.2·L3
```

防泄漏：按 `run_uuid` group split (70/15/15)；attention retrieval pool 排除同 run case。

## 6. 数据

```
源:        cases.json (2660)
剔除:      legacy(35) + excluded_infra(44) + crash(6) + pending(1) → 净 ~2575
派生字段:  position_in_run (按 run_uuid 分组、created_at 排序的索引)
划分:      group split by run_uuid, 70/15/15
权重:      pos_weight = N_neg / N_pos (BCEWithLogitsLoss)
```

## 7. 评估指标（离线）

- 总体 AUC、F1、ECE
- 按 waypoint 分层 AUC（重点 cobblestone, iron_ore, gold_ore）
- Top-1 multi-action accuracy（cobblestone 等多候选 waypoint）
- Attention 诊断：top-1 attention case 与 query 同 waypoint 的比例
- baseline 对比：
  - (a) `success_count - failure_count` 全局排序（现行 `_best_exact_success_case`）
  - (b) 朴素 majority class

成功标准：
- 总体 AUC ≥ 0.75
- cobblestone 分层 AUC ≥ 0.70
- Attention top-1 同 waypoint 比例 ≥ 0.70
- Multi-action waypoint Top-1 accuracy 高于 baseline (a)

## 8. 落地文件

```
docs/DECISIONER_V3_PROPOSAL_2026-05-01.md         (本文档)

scripts/export_decisioner_dataset.py
scripts/train_rads.py
scripts/evaluate_rads_offline.py

src/optimus1/decisioner/__init__.py
src/optimus1/decisioner/feature.py
src/optimus1/decisioner/encoder.py
src/optimus1/decisioner/rads.py
src/optimus1/decisioner/runtime.py                (推理接口，本轮仅定义不接入)

data/decisioner/rads_v1.jsonl                     (导出数据集)
data/decisioner/rads_v1_summary.json              (数据集统计)
artifacts/decisioner/rads_v1.pt                   (模型 + library cache)
reports/decisioner/offline_eval_v1.md             (评估报告)
reports/decisioner/training_log_v1.json           (训练曲线)
```

## 9. 下一轮（不在本轮范围）

- runtime 接入：`case_memory.select_case_decision` 增加 `decisioner.enabled` 分支
- 在线 A/B：跑 v3 实验，对比 v2 retrieval-only
- 执行层改进（biome-aware 找树等）
- 把 explore 升格为一阶 action（让 logs waypoint 有多候选）
