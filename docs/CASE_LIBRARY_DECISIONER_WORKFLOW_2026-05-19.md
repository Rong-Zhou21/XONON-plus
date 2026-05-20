# XENON-plus 案例库与决策器工作流程说明

生成日期：2026-05-19
项目目录：`/home/yzb/zhourong/XENON-plus`

## 1. 总览

XENON-plus 的“案例库 + 决策器”不是一个独立脚本，而是嵌入主实验循环的高层决策闭环：

```text
env.get_status()
  -> CaseBasedMemory.create_state_snapshot()
  -> CaseBasedMemory.select_case_decision()
     -> RADS 决策器打分，或案例库检索
     -> 低置信时 fallback 到 VLM planner
  -> record_decision() 写 pending case
  -> helper / STEVE-1 / env wrapper 执行
  -> save_success_failure() 结算 outcome
  -> cases.json 成为下一轮检索和离线训练数据
```

它解决的核心问题是：原版 XENON 只能记住“某个 waypoint 以前哪个动作成功过”，Plus 要记住“在什么状态下、为了什么最终目标、对哪个 waypoint、选择了哪个动作、执行结果如何”。

## 2. 案例库如何参与实验

### 2.1 初始化

实验启动时，`src/optimus1/main_planning.py` 初始化：

```text
action_memory = CaseBasedMemory(cfg, logger)
```

`CaseBasedMemory` 会完成几件事：

1. 定位存储目录：`src/optimus1/memories/ours_planning/v1/case_memory/`
2. 读取 `cases.json`
3. 如配置允许，执行 legacy bootstrap，把旧 `waypoint_to_sg/*.json` 迁移成 case
4. 用 SentenceTransformer 编码每条 case 的 `similarity_text`
5. 建立 pending case 索引
6. 如果 `memory.case_memory.decisioner.enabled=true`，加载 `RADSRuntime`

配置入口位于 `src/optimus1/conf/evaluate.yaml`：

```yaml
memory:
  case_memory:
    reuse_threshold: 0.72
    retrieve_threshold: 0.45
    bootstrap_legacy: True
    decisioner:
      enabled: False
      checkpoint: artifacts/decisioner/rads_v2.pt
      min_p_success: 0.20
      device: cuda
      log_evidence_topk: 5
```

批跑脚本可以覆盖这些配置。例如 `scripts/run_v3_full_benchmark.sh` 通过 Hydra override 开启：

```text
memory.case_memory.decisioner.enabled=true
memory.case_memory.decisioner.checkpoint=artifacts/decisioner/rads_v2.pt
memory.case_memory.decisioner.min_p_success=0.20
```

### 2.2 每次需要新计划时

当当前 subgoal 为空时，主循环调用 `make_plan()`：

```text
env_status = env.get_status()
waypoint, subgoal, language_action_str, error_message = make_plan(...)
```

`make_plan()` 内部先用 OracleGraph 生成 waypoint 列表，然后做 Plus 的状态化规划修正：

1. 根据当前 inventory 跳过已满足的非消耗 waypoint
2. 检查工具和材料前置条件
3. 构造 `state_snapshot`
4. 调用 `select_case_decision()`

`state_snapshot` 由 `CaseBasedMemory.create_state_snapshot()` 生成，字段包括：

- `inventory`
- `equipment`
- `location_stats`
- `plain_inventory`
- `biome`
- `obs_summary`

这个快照会被同时用于：

- 案例库相似度检索
- RADS 特征抽取
- pending case 记录
- 后续训练数据导出

### 2.3 有决策器时的在线选择

当 `decisioner.enabled=true` 时，`select_case_decision()` 会进入 `_select_case_decision_rads()`：

1. 从当前 waypoint 的历史 case 中收集 distinct actions。
2. 每个 action 选一个代表 case，优先选成功 case。
3. 为每个 action 构造 query case：
   - 当前 `state_snapshot`
   - 当前 `waypoint`
   - 当前 `waypoint_num`
   - 当前 `original_final_goal`
   - 候选 `selected_action`
   - 当前 run 内 `_position_in_run`
4. 调用 `RADSRuntime.score()` 得到 `p_success`、`confidence`、attention evidence。
5. 按 `p_success` 排序。
6. 若最高 `p_success < min_p_success`，返回 `None`，上层调用 VLM planner。
7. 若最高分达标，复用该 action 对应的 subgoal。

写入 `decision_trace` 的信息包括：

- `source: rads_decisioner`
- `p_success`
- `confidence`
- `attention_concentration`
- `selected_case_id`
- 所有候选动作的分数
- top-k evidence cases

这使线上实验可以复盘“为什么选这个动作”。

### 2.4 没有决策器或低置信时

如果没有启用 RADS，或 RADS 判断低置信，Plus 退回到案例检索 + planner 路径：

1. `_best_exact_success_case()` 尝试找同 waypoint 的净成功动作。
2. `_retrieve_cases()` 用 `similarity_text` 做 embedding cosine retrieval。
3. `retrieve_similar_succeeded_waypoints()` 把相似成功案例转换成 planner examples。
4. `retrieve_failed_subgoals()` 把失败动作提供给 planner 避免重复。
5. 调用 `ServerAPI.get_decomposed_plan()` 生成新 subgoal。

生成后，Plus 还会做动作语义过滤：

- `logs` 不能是 craft/smelt
- `planks` 不能是 dig/mine
- ores 必须是 mine/dig
- ingot 必须 smelt
- craft-only 物品必须 craft/make/create

如果 planner 或 case 给出不可行动作，Plus 用 `_fallback_subgoal_for_waypoint()` 替换，并把被拒绝动作写进 `decision_trace`。

### 2.5 记录 pending case

无论动作来自 RADS、case memory、planner 还是 semantic fallback，只要被系统采纳，都会调用：

```text
CaseBasedMemory.record_decision(...)
```

此时写入 `cases.json` 的 outcome 是：

```json
{
  "status": "pending",
  "success": null
}
```

pending case 的意义是：先把“当时为什么这样选”完整保存下来，等真实执行结束后再填结果。

### 2.6 执行与结算

动作执行分两类：

1. `craft` / `smelt`：由 `NewHelper` 做 GUI 自动化。
2. 采集、挖矿、找树等动作：由 `ServerAPI.get_action()` 调 STEVE-1，再调用 `env.step(action, current_sg_target, prompt=current_sg_prompt)`。

执行结束后，主循环用 waypoint 检查器判断是否真的完成：

```text
waypoint_success = env.check_waypoint_finish([waypoint, 1])
action_memory.save_success_failure(...)
```

`save_success_failure()` 会找到对应 pending case，把 outcome 改成：

```json
{
  "status": "success | failed | timeout_non_programmatic | ...",
  "success": true | false,
  "recorded_at": "...",
  "state_snapshot": {...}
}
```

如果整轮任务失败，主循环还会处理未结算 case：

- 基础设施 early stop：`discard_pending_cases(run_uuid)`
- 普通失败或超时：`mark_pending_cases_failed(run_uuid, reason=status_detailed)`

因此 case memory 中的样本不是离线人工整理的，而是每次实验自然产生的状态-动作-结果记录。

## 3. 案例 schema

典型 case 结构如下：

```json
{
  "id": "run_uuid:000108:timestamp_ms",
  "created_at": "2026-05-xxTxx:xx:xxZ",
  "run_uuid": "episode id",
  "original_final_goal": "craft_a_iron_pickaxe",
  "environment": "plains",
  "waypoint": "iron_ore",
  "waypoint_num": 3,
  "state_snapshot": {
    "inventory": {"stone_pickaxe": 1, "cobblestone": 4},
    "equipment": "stone_pickaxe",
    "location_stats": {"xpos": 0, "ypos": 34, "zpos": 0, "pitch": 0, "yaw": 0},
    "plain_inventory": {},
    "biome": "plains",
    "obs_summary": {}
  },
  "similarity_text": "waypoint: iron_ore; inventory: ...",
  "candidate_actions": [
    {"action": "dig down and mine iron_ore", "source": "rads_decisioner"}
  ],
  "selected_action": "dig down and mine iron_ore",
  "selected_subgoal": {"task": "dig down and mine iron_ore", "goal": ["iron_ore", 3]},
  "selected_subgoal_str": "{\"task\": \"dig down and mine iron_ore\", ...}",
  "decision_trace": {
    "source": "rads_decisioner",
    "p_success": 0.91,
    "selected_case_id": "...",
    "evidence": []
  },
  "outcome": {
    "status": "success",
    "success": true,
    "recorded_at": "..."
  }
}
```

当前 live `cases.json` 共 2080 条。按 `decision_trace.source` 粗略统计：

| source | 数量 |
|---|---:|
| rads_decisioner | 1536 |
| case_memory_exact_waypoint | 407 |
| case_memory | 71 |
| semantic_fallback | 38 |
| planner | 28 |

这说明当前案例库既记录旧式 case memory 复用，也大量记录了 RADS 在线选择。

## 4. 决策器训练流程

### 4.1 数据导出

训练从案例库导出：

```bash
python scripts/export_decisioner_dataset.py
```

输入：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

输出：

```text
data/decisioner/rads_v1.jsonl
data/decisioner/rads_v1_summary.json
```

导出脚本只保留训练需要的字段：

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
- 只保留 `outcome.success` 明确为 true/false 的样本

当前 `rads_v2.pt` 使用的导出快照统计：

| 项目 | 数量 |
|---|---:|
| raw_case_total | 2660 |
| valid_sample_total | 2574 |
| drop legacy | 35 |
| drop excluded_infra | 44 |
| drop crash_RuntimeError | 6 |
| drop pending | 1 |

按 `run_uuid` 分组切分：

| split | samples | runs | positive | negative |
|---|---:|---:|---:|---:|
| train | 1847 | 170 | 1542 | 305 |
| val | 351 | 36 | 330 | 21 |
| test | 376 | 38 | 307 | 69 |

按 `run_uuid` 切分的原因是：同一次 Minecraft 运行里的连续 waypoint 高度相关。若随机按 case 切分，模型可能在 test 中看到同一局游戏的邻近状态，离线指标会虚高。

### 4.2 特征构造

特征定义在 `src/optimus1/decisioner/feature.py`。总输入维度是 52：

```text
numeric dense features: 34
categorical embeddings: waypoint / final_goal / action 各 6 维，共 18
total: 52
```

数值和 one-hot 特征包括：

| 特征 | 含义 |
|---|---|
| equipment one-hot | 当前手持或装备状态 |
| biome one-hot | 当前 biome |
| waypoint_num log1p | 当前 waypoint 需求数量 |
| position_in_run log1p | 当前 case 在本轮 run 中的位置 |
| ypos_norm | 当前高度归一化 |
| ypos_bucket | 高度桶：地表、高层、浅层、矿层、深层 |
| inventory key items | logs、planks、stick、coal、cobblestone、iron、gold、diamond、redstone 等 |
| tool_owned_flags | 是否拥有 wooden/stone/iron pickaxe |
| inv_unique_count | 背包物品种类数 |

类别 embedding 包括：

- `waypoint_id`
- `final_goal_id`
- `action_id`

v2 还加入 `(waypoint, action)` 的 Laplace-smoothed 成功率先验，并在最终 logit 上以 residual 方式使用。

### 4.3 模型训练

训练命令：

```bash
python scripts/train_rads.py \
  --data data/decisioner/rads_v1.jsonl \
  --out artifacts/decisioner/rads_v2.pt \
  --report reports/decisioner/training_log_v2.json
```

RADS 模型由三部分构成：

```text
QueryEncoder(query) -> 当前状态-动作查询向量
CaseEncoder(case + outcome) -> 历史案例向量
Attention(query, train-case library) -> 历史经验上下文
DecisionHead([query, context, action_embedding]) -> success logit
```

损失函数：

```text
L1 = BCEWithLogitsLoss(P_success_logit, outcome.success)
L2 = TripletMarginLoss(case vectors)
L3 = CrossEntropy(waypoint_head(case_vector), waypoint_id)

L = L1 + 0.1 * L2 + 0.05 * L3
```

训练时的关键防泄漏机制：

- 每个 batch 都重新编码 train case library，使 CaseEncoder 端到端学习。
- attention retrieval pool 会 mask 当前 query 所属 `run_uuid` 的所有 case。
- 训练时还 mask query 自己对应的 case。

默认重要参数：

| 参数 | 值 |
|---|---:|
| epochs | 20 |
| patience | 5 |
| batch_size | 64 |
| lr | 1e-3 |
| weight_decay | 1e-4 |
| dropout | 0.2 |
| same_wp_min | 8 |
| triplet_weight | 0.1 |
| wp_weight | 0.05 |
| prior_logit_init_weight | 1.5 |
| seed | 20260501 |

当前 best checkpoint 出现在 epoch 14。

### 4.4 离线评估

评估命令：

```bash
python scripts/evaluate_rads_offline.py \
  --artifact artifacts/decisioner/rads_v2.pt \
  --data data/decisioner/rads_v1.jsonl \
  --report_md reports/decisioner/offline_eval_v2.md \
  --report_json reports/decisioner/offline_eval_v2.json
```

`rads_v2.pt` 的 test 指标：

| 指标 | 数值 |
|---|---:|
| n | 376 |
| pos_rate | 0.8165 |
| AUC | 0.9110 |
| AP | 0.9739 |
| best F1 | 0.9535 |
| best threshold | 0.10 |
| ECE | 0.0866 |
| mean attention concentration | 0.8417 |
| top-1 attention same-waypoint rate | 0.9441 |

基线对比：

| 方法 | AUC | AP | F1 | ECE |
|---|---:|---:|---:|---:|
| RADS v2 | 0.9110 | 0.9739 | 0.9535 | 0.0866 |
| history_count_baseline | 0.8555 | 0.9431 | 0.9119 | 0.0419 |
| majority_class_baseline | 0.5000 | 0.8165 | 0.8990 | 0.0000 |

这些指标说明 RADS 不只是复述历史成功次数，而是学到了状态条件下的成功率差异。

### 4.5 产物内容

`artifacts/decisioner/rads_v2.pt` 保存：

- `model_state`
- `spec`
- `config`
- `library_vecs`
- `library_meta`
- `train_runs`
- `best_epoch`
- `best_val`
- `test`

运行时 `RADSRuntime.load()` 读取这个 bundle，直接使用 train case library 的编码向量作为 attention memory。

## 5. 决策器在线实验如何接入

### 5.1 单任务或全集运行

脚本中通常通过环境变量控制：

```text
DECISIONER_ENABLED=1
DECISIONER_CKPT=artifacts/decisioner/rads_v2.pt
DECISIONER_MIN_P=0.20
```

然后脚本转成 Hydra 参数：

```text
memory.case_memory.decisioner.enabled=true
memory.case_memory.decisioner.checkpoint=artifacts/decisioner/rads_v2.pt
memory.case_memory.decisioner.min_p_success=0.20
```

关闭决策器时：

```text
memory.case_memory.decisioner.enabled=false
```

此时系统仍使用 CaseBasedMemory，但退回 retrieval-only 路径，可作为消融 baseline。

### 5.2 线上结果

`reports/decisioner/v2_vs_v3_comparison.md` 给出的 67 任务 single-shot 对比：

| 方法 | success / 67 | rate |
|---|---:|---:|
| v2 retrieval-only first attempt | 39 | 58.2% |
| v3 RADS decisioner single shot | 49 | 73.1% |
| v2 best-of-N retries | 57 | 85.1% |

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

该实验说明，决策器在线参与方式不是替换整个 agent，而是在每次 waypoint 动作选择时进行 rerank/gate。收益来自：

- 对多候选 waypoint 选择更合适动作
- 在低置信状态触发 planner fallback
- 避免机械复用历史成功动作

### 5.3 与实验结果文件的关系

每次实验结束后，Plus 会写：

- `cases.json`：案例库，供后续检索和训练
- `save_decomposed_plan/<status>/*.json`：任务级 plan 记录
- Hydra 输出目录里的 result JSON：包含 success、steps、failed_waypoints、resource_ledger、recovery_events 等

其中，决策器训练只直接读取 `cases.json`。result JSON 和 recovery events 主要用于诊断与消融分析；除非后续扩展特征，否则它们不会自动进入 RADS 训练。

## 6. 决策器的边界

RADS 的作用是高层动作成功率估计，不是低层 Minecraft 控制器。

它能做：

- 在已有候选动作中排序
- 根据当前状态判断是否值得复用历史动作
- 低置信时回退 planner
- 输出 evidence 方便复盘

它不能做：

- 直接生成从未出现过的新动作
- 直接看第一人称图像识别矿物
- 替代 STEVE-1 完成移动、转向、挖掘和收集
- 单独解决自然矿物搜索、卡住、掉落物捡取、背包满等执行层问题

因此，XENON-plus 的完整创新必须把案例库、决策器、环境感知与行动三者一起看：案例库提供数据，决策器学习选择，环境感知与行动层负责让选择在 Minecraft 物理环境中尽可能真实执行并回写结果。

## 7. 一句话总结

案例库让 XENON-plus 拥有“状态-动作-结果”的可训练记忆；RADS 决策器从这些 case 中学习 `P(success | state, waypoint, action)`，在线对历史候选动作进行打分和回退控制；实验运行又把新的选择与结果写回案例库，形成持续迭代的闭环。
