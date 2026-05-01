# XENON-plus 决策器 v2 与现行方法的差异（带例子）

日期：2026-05-01
对照对象：

- 现行方法：`src/optimus1/memories/case_memory.py` 中的 `select_case_decision()` + `_best_exact_success_case()` + cosine retrieval
- 现行训练原型：根目录 `train_value_head.py`（单 text + MiniLM 384 维 + MLP + BCE）
- v2 方案：见 `docs/DECISIONER_V2_PROPOSAL_2026-05-01.md`（本文件给出与现行方法的差异和实例）

本文件不给完整训练代码，只用真实案例库中的数据，举具体例子说明：在同一个 (state, waypoint) 下，现行方法会怎么决策、v2 会怎么决策、为什么不一样。

所有例子里的状态、动作、成败统计都来自当前 `cases.json` 和 `exp_results/v2/`。

---

## 0. 一句话差异

> 现行方法回答的是「这个 waypoint 历史上谁赢得多」。
> v2 回答的是「在我现在的状态下，这个动作有多大成功概率，会跑多久，可能怎么死」。

---

## 1. 全景对比

| 维度 | 现行方法 | v2 |
|---|---|---|
| 决策对象 | 在历史成功 action 中选一个复用 | 对（候选）action 做状态条件下的风险评估 |
| 关键短路 | `_best_exact_success_case()`：同 waypoint 净胜场 → 1.0 置信度直接用 | 取消短路（白名单除外），所有决策都走打分 |
| 状态使用 | `similarity_text` 把 inventory/equipment/biome/ypos 拼成字符串，再用 MiniLM 编码 | 结构化路（inventory count/ypos bucket/equipment onehot 等 ~80 维） + 文本路（仅编码动作语义）|
| 训练目标 | 单一 BCE：success / fail | 多任务：success + log_steps + failure_mode |
| 训练数据来源 | 直接用 `cases.json`，包含 90% 由短路自我复制产生的样本 | 先做主动多样性采集，破除 feedback loop，再训练 |
| 失败定义 | `outcome.success == false` 即负样本 | 区分 normal_fail / timeout / crash / infra；后两类不进训练 |
| Runtime 流程 | exact 短路 → 否则 cosine retrieve → 否则 planner | retrieve TopM → 多任务打分 → utility = P(succ)·exp(-λ·steps) → 阈值门控；高 timeout 概率主动重规划 |

---

## 2. 例 1：状态条件成败（cobblestone, ypos 差异）

### 数据事实

`cobblestone` 是当前唯一有真实 ranking 信号的 waypoint：

```
dig down and mine cobblestone: 215 success, 231 failed
craft cobblestone:                0 success,  2 failed
smelt cobblestone:                0 success,  2 failed
```

按状态分桶后差异更明显：

```
cobblestone success cases: n=215, ypos_mean=64.7,  inventory_unique_items_mean=6.4
cobblestone failed  cases: n=235, ypos_mean=39.6,  inventory_unique_items_mean=10.9
```

也就是说，cobblestone 失败时智能体平均在地下 ~40 高度，背包平均装了 ~11 种不同物品（背包压力高）；成功时在地表 ~65，背包还干净。这是一个非常强的状态信号。

### 同一个新状态：智能体此刻在 ypos=35，背包里有 12 种物品

#### 现行方法

`_best_exact_success_case("cobblestone")` 直接看：

```
action = "dig down and mine cobblestone"
score = success_count - failure_count
      = 215 - 231
      = -16
```

`-16 > -plan_failure_threshold (=-3)` 不成立 → 这个 action 被排除。

如果 `plan_failure_threshold=-50` 之类的宽松值，则 score=-16 仍然过线，置信度 = 1.0，直接复用 `dig down and mine cobblestone`。**不看智能体当前 ypos 是 35 还是 65**。

退一步走 cosine retrieval 时，`similarity_text` 里 ypos=35 也只是被压成一个字符串里的 "ypos:35.0" 子串，对 MiniLM 来说这和 "ypos:65.0" 的语义相似度几乎一样。

> 结论：现行方法在地下高度和地表高度做出完全相同的决策。

#### v2

结构化路输入（节选）：

```
ypos_bucket = [0,0,1,0,0]   # bucket = 浅地下
inventory_pressure = 12     # 共 12 类物品
equipment = "stone_pickaxe"
waypoint = "cobblestone"
```

文本路：MiniLM 编码 `"dig down and mine cobblestone"` → 语义动作向量。

多任务输出（示意值）：

```
P(success | s, a)        = 0.32
E[steps | s, a, success] = 5500
P(failure_mode = timeout)= 0.45
P(failure_mode = stuck)  = 0.20
```

decision logic：

```
P(success)=0.32 < τ_high(0.55) → 不复用
→ 升级 planner 重新出动作（可能输出"先回地表再挖"这类候选）
→ 同时把 P(timeout)=0.45 写入 trace，供执行层决定是否调短 max_minutes
```

> 关键差别：v2 看到了「ypos=35 + 高背包压力」就把 cobblestone 这个 waypoint 标为高风险，而不是按全局多数决盲目复用。

---

## 3. 例 2：前置条件失败（iron_ore + wooden_pickaxe）

### 数据事实

`iron_ore` 失败案例的真实状态（直接从 `cases.json` 取）：

```
case A (failed, goal=bucket):
  inventory: {oak_log:2, oak_planks:7, stick:2, wooden_pickaxe:1, stone_pickaxe:1, ...}
  equipment: wooden_pickaxe   ← 关键
  ypos: 50
  action: "dig down and mine iron_ore"
```

对比 success 案例：

```
case B (success):
  inventory: {birch_log:2, birch_planks:8, stick:4, wooden_pickaxe:1, stone_pickaxe:1, ...}
  equipment: stone_pickaxe   ← 关键
  ypos: 61
  action: "dig down and mine iron_ore"
```

A 和 B 的差别核心是 equipment：A 装备了 wooden_pickaxe（不能挖铁矿），B 装备了 stone_pickaxe。Minecraft 规则上前者注定失败。

### 同一个新状态：inventory 里有 stone_pickaxe，但 equipment="wooden_pickaxe"

#### 现行方法

`_best_exact_success_case("iron_ore")` 看的是：

```
action = "dig down and mine iron_ore" 的所有 case 净分
       = 128 success - 29 failed
       = +99 → 置信度 1.0
       → 直接复用
```

不会感知到当前 equipment 错了。智能体会带着木镐冲下去，挖不动，浪费 ~3000 steps，最后 timeout 或者 stuck。

#### v2

结构化路输入：

```
equipment = "wooden_pickaxe"
inventory: {stone_pickaxe: 1, ...}   ← 有但没装
waypoint = "iron_ore"
```

训练时模型看到的负样本恰好就是 case A 这种「inv 有 stone_pickaxe 但 equipment 是 wooden_pickaxe」的失败 pattern，能学到：

```
P(success | wooden_pickaxe + iron_ore)  ≈ 0.10
P(success | stone_pickaxe + iron_ore)   ≈ 0.85
```

decision logic：

```
P(success)=0.10 < τ_low(0.30)
→ 不复用，fall back to planner
→ planner 这一次有机会输出「先装备 stone_pickaxe」之类的修正
```

> 关键差别：v2 学到了「equipment 不对 → 大概率失败」，避免一次必然失败的执行。这种 case 在数据里有现成负样本（case A 就是），现行方法不用结构化 equipment 字段，所以学不到。

---

## 4. 例 3：多候选动作排序（charcoal / stone / smooth_stone）

### 数据事实

```
charcoal:
  smelt charcoal:               2 success, 0 failed   ← 正确
  craft charcoal:               0 success, 2 failed
  dig down and mine charcoal:   0 success, 1 failed

stone:
  smelt stone:                  7 success, 0 failed   ← 正确
  dig down and mine stone:      0 success, 2 failed
  craft stone:                  0 success, 2 failed

smooth_stone:
  smelt smooth_stone:           2 success, 1 failed   ← 唯一成功过
  dig down and mine smooth_stone: 0 success, 2 failed
  craft smooth_stone:           0 success, 2 failed
```

这是真正存在动作分歧的样本，但样本量极小（每动作只有 1-7 条）。

### 同一个新状态：waypoint=charcoal，inventory 里有 oak_log:2 + furnace:1 + crafting_table:1

#### 现行方法

`_best_exact_success_case("charcoal")` 用 `success_count - failure_count`：

```
smelt charcoal:      +2 -0 = +2  → 选这个 ✓ （恰好对）
craft charcoal:      +0 -2 = -2
dig ... charcoal:    +0 -1 = -1
```

恰好给出正确答案。但请注意：这个排名**没看 inventory**。如果智能体当前 inventory 里没有 furnace（炼制必需品），现行方法仍会复用 `smelt charcoal`，然后执行层失败。

可以在数据里找到一个反例（构造）：smooth_stone 的 `smelt smooth_stone` 也有 1 次失败，可能就是 furnace 缺失或者 fuel 缺失，但现行模型把它和 2 次成功平均了，看不到差异。

#### v2

模型输入会显式带 `inventory.furnace`、`inventory.coal`、`inventory.oak_log` 这些数值。学到：

```
P(success | smelt charcoal, has_furnace=1, has_log≥1, has_coal≥0) ≈ 0.90
P(success | smelt charcoal, has_furnace=0)                        ≈ 0.10  ← 学到前置条件
P(success | craft charcoal, ...)                                  ≈ 0.05  ← 错误动作
P(success | dig charcoal, ...)                                    ≈ 0.05
```

decision logic：

```
按 utility 排序，最高的是 smelt charcoal (P=0.90)
→ τ_high=0.55 通过，reuse
→ 但若智能体真的没有 furnace，模型给出 P=0.10，自动转 planner
```

> 关键差别：v2 在「正确动作 + 前置条件不满足」时仍能保护，现行方法做不到。即使排序碰巧对了，也是因为另一动作的负样本足够多，而不是因为状态匹配。

---

## 5. 例 4：失败模式预测（blast_furnace timeout）

### 数据事实

`exp_results/v2/` 中失败任务的 `status_detailed` 分布：

```
timeout_non_programmatic: 24   ← 主导失败模式
failed (能力不足):         19
crash_RuntimeError:        6
env_step_timeout:          1
```

具体例子（来自 handoff 文档）：

```
craft_a_blast_furnace: failed=timeout, steps=11997 / max=12000  ← 跑到天荒地老才超时
craft_a_iron_axe:      failed=crash,   steps=12001
```

`blast_furnace` 失败的特征是「执行能跑、但巨慢，最后 timeout」。这种失败在 case 级别很难标，但在 task 级别很明显。

### 同一个新状态：智能体刚拿到 iron_ingot:5，waypoint=smooth_stone，max_minutes=10

#### 现行方法

只看 `outcome.success`，不区分 timeout vs normal fail。在 case 库里 smooth_stone 的 smelt 失败 1 次、成功 2 次，二分类 label 完全等价。

执行时不预知耗时，等 10 分钟跑满了才发现 timeout。这就是 v2 实测里 `craft_a_blast_furnace` 11997 steps 的真实失败模式。

#### v2

`steps_pred` 头会输出 `E[steps]`，`failure_mode` 头会输出 `P(timeout)`。

```
E[steps | s, smelt smooth_stone] = 8500
max_steps_budget                  = 10 * 60 * 20 = 12000  (10 min × 1200 step/min)
remaining_budget                  = 12000 - 已用 7000 = 5000
→ E[steps]=8500 > remaining_budget=5000
→ P(complete_in_budget) ≈ 0.20
→ early_warning: 主动报告 timeout 风险，让 planner 重排或者放弃这个 waypoint 改走 stone（更便宜的等价路径）
```

> 关键差别：v2 第一次给「步数预算 vs 预测步数」一个建模位置，避免了 `blast_furnace` 这种「跑到 11997 steps 才发现挂了」的情况。

---

## 6. 例 5：输入特征对比（同一个 case，模型看到的差异）

举一个 case 库里的真实 case，inventory 比较丰富：

```python
{
  "waypoint": "iron_ore",
  "state_snapshot": {
    "inventory": {
      "andesite": 4, "crafting_table": 1, "diorite": 4, "dirt": 3,
      "furnace": 1, "granite": 3, "oak_log": 2, "oak_planks": 7,
      "oak_sapling": 2, "stick": 2, "stone_pickaxe": 1, "wooden_pickaxe": 1
    },
    "equipment": "wooden_pickaxe",
    "location_stats": {"ypos": 50.0, "biome_id": 0, ...}
  },
  "selected_action": "dig down and mine iron_ore",
  "outcome": {"status": "failed"}
}
```

### 现行 `train_value_head.py` 看到的输入

由 `similarity_text` 拼出（接近）：

```
"waypoint: iron_ore; inventory: andesite:4, crafting_table:1, diorite:4,
 dirt:3, furnace:1, granite:3, oak_log:2, oak_planks:7, oak_sapling:2,
 stick:2, stone_pickaxe:1, wooden_pickaxe:1; equipment: wooden_pickaxe;
 location: biome_id:0, pitch:0, xpos:..., ypos:50.0, yaw:0, zpos:...; biome: forest"
```

→ MiniLM encode 整段 → 384 维 dense vector → MLP → success logit。

问题：

1. `equipment: wooden_pickaxe` 这条关键信息和 `oak_planks:7` 在 token 上是一个量级，MiniLM 没有理由特别关注它。
2. `ypos:50.0` 和 `ypos:65.0` 在 MiniLM 里基本是邻近 token，模型很难区分「地下挖掘 vs 地表挖掘」这种 14 个 ypos 的差距。
3. inventory 里的关键计数（stone_pickaxe:1）和无关物品（dirt:3、andesite:4）完全没区分权重。

### v2 看到的输入

```python
# 结构化路（约 80 维）
struct = {
  "wp_onehot[iron_ore]": 1,
  "equipment_onehot[wooden_pickaxe]": 1,   # 关键：单独占一维
  "equipment_onehot[stone_pickaxe]": 0,
  "ypos_bucket": [0, 0, 1, 0, 0],          # 浅地下 bucket
  "inv_count[stone_pickaxe]": 1,           # 有但没装备
  "inv_count[wooden_pickaxe]": 1,
  "inv_count[iron_ingot]": 0,
  "inv_count[coal]": 0,
  "inv_count[oak_log]": 2,
  "inv_pressure_total_unique_items": 12,   # 背包压力指标
  "biome_onehot[forest]": 1,
  "prior_failed_in_run[stone_pickaxe]": 0,
  ...
}

# 文本路（仅动作语义，约 384 维）
action_text = "dig down and mine iron_ore"
emb_action = MiniLM(action_text)

# 模型
h_struct = MLP_struct(struct)              # 80 → 64
h_text   = MLP_text(emb_action)            # 384 → 64
h        = concat(h_struct, h_text)        # 128
logits   = multi_head(h)                   # 三个头
```

模型立刻能看到「`equipment_onehot[wooden_pickaxe] = 1` 同时 `wp_onehot[iron_ore] = 1`」这一组特征的负相关关系，而不必从一段拼接的英文里去猜。

> 关键差别：把数值/枚举字段从文本里抠出来，给模型一个数值上能区分的输入空间。文本路只保留动作语义这种「真正需要泛化」的部分。

---

## 7. 例 6：主动多样性采集（破除 1-action-per-waypoint feedback loop）

### 现状

70/74 个 waypoint 只有 1 个 unique action 出现过。原因是：

```
第一次 planner 给出 action A，A 成功
→ exact_waypoint 短路记下「A 是 +1」
→ 下一次同 waypoint，短路直接复用 A，新 case 再 +1
→ planner 永远没有第二次机会出 B
→ 数据里永远只有 A
```

任何 ranker 在这种数据上都会退化为 `waypoint → A` 的硬映射。

### v2 的具体做法（举一个真实任务）

任务：`craft_a_iron_axe`（当前 v2 失败，crash_RuntimeError，12001 steps）

启动 `diversity_mode`：

```yaml
diversity_collection:
  enabled: true
  target_tasks: [craft_a_iron_axe, craft_a_blast_furnace, craft_a_golden_pickaxe, ...]
  target_waypoints: [iron_ingot, iron_pickaxe, gold_ore, ...]   # 高失败率
  override_exact_shortcut: true       # 关掉 _best_exact_success_case
  planner_min_candidates: 2           # 强制 planner 至少出 2 个候选
  selection: epsilon_greedy           # 50% 概率不选 historical winner
  epsilon: 0.5
  runs: 5                             # 单任务跑 5 次
```

跑 5 次后，对 `iron_ingot` 这个 waypoint，可能产生：

```
case D1: action="smelt iron_ore",        outcome=success, ypos=15
case D2: action="dig down + smelt nearby", outcome=failed, ypos=55  (planner 提出但执行不好)
case D3: action="smelt iron_ore",        outcome=success, ypos=20
case D4: action="dig forward + smelt",   outcome=success, ypos=12
case D5: action="smelt iron_ore",        outcome=failed, ypos=8     (深矿层熔炉缺 fuel)
```

这 5 条样本里：

- D1, D3, D5 同一动作不同状态 → 状态条件成败信号
- D2, D4 引入了新动作 → 真正的动作排序信号
- D5 在 ypos 极深时失败 → 深度阈值信号

整个采集过程产生约 50 runs × 8 waypoint = ~400 条新 case，覆盖 10 个失败任务和高失败 waypoint。

> 关键差别：现行 case 库已经 2660 条，但都是「同一道题做了 2660 遍」；v2 主动制造「不同的题」，让模型有学的对象。

---

## 8. 例 7：runtime 决策流程对比（决策树）

### 现行流程（伪代码）

```python
def make_plan(waypoint, state):
    case = best_exact_success_case(waypoint)         # 看 success_count - failure_count
    if case:
        return reuse(case)                            # 0 看状态
    retrieved = cosine_retrieve(waypoint, state)
    best = retrieved[0] if retrieved else None
    if best and best.score >= 0.72:
        return reuse(best)
    return planner_fallback(waypoint, state)
```

### v2 流程（伪代码）

```python
def make_plan(waypoint, state, budget_remaining):
    candidates = retrieve_top_m(waypoint, state, m=8)   # 同 waypoint + 跨 waypoint 候补
    if waypoint in WHITELIST_TRIVIAL:                   # planks/stick 等 100% 成功
        return shortcut_reuse(candidates[0])

    scored = []
    for c in candidates:
        x = build_features(state, c.action, run_context)
        p_succ, e_steps, p_modes = value_head(x)
        utility = p_succ * exp(-lambda_ * e_steps / budget_remaining)
        scored.append((c, p_succ, e_steps, p_modes, utility))

    best = max(scored, key=lambda r: r.utility)

    # 风险预警
    if best.p_modes['timeout'] > 0.7:
        return planner_with_constraint(waypoint, state, "shorter_path_required")

    # 三段阈值
    if best.utility >= tau_high:
        return reuse(best)
    if best.utility >= tau_low:
        return reuse(best, with_safety_monitor=True)    # 中置信度，加监控
    return planner_fallback(waypoint, state, hint=best.p_modes)
```

> 关键差别：v2 有三个新决策位——budget-aware utility、failure-mode 预警、中置信度的 safety_monitor 模式。

---

## 9. 训练数据治理对比

| 项 | 现行 `train_value_head.py` 隐含做法 | v2 |
|---|---|---|
| pending case (`status='pending'`) | 全量进训练 → label 不明 | 排除 |
| infra early-stop (`excluded_infra` 44 条) | 进训练 → 当成 fail | 排除（不是模型能力问题） |
| crash (`crash_RuntimeError` 6 条) | 进训练 → 当成 fail | 单独标 `failure_mode=crash`，不参与 success/fail label |
| legacy bootstrap (35 条) | 进训练 | 默认排除（无真实状态）；冷启动检索时单独保留 |
| 同 run_uuid 的多 waypoint | 随机 split | 按 `run_uuid` group split，避免泄漏 |
| 高失败 waypoint 样本权重 | 1.0 | cobblestone/iron_ore 等 1.5 |
| v2 实验结果 vs v1 | 同等 | v2 权重略高（执行层修正后更可信） |

---

## 10. 一张表收口

| 场景 | 现行方法的行为 | v2 的行为 | 谁更好 |
|---|---|---|---|
| 智能体在地下 ypos=35，要 cobblestone | 复用历史 dig down（不看 ypos） | P(success)=0.32 → planner 出地表方案 | v2 |
| equipment=wooden，要 iron_ore | exact 短路复用 dig down → 必败 | P(success)=0.10 → 提示先换装备 | v2 |
| 要 charcoal，inv 有 furnace+log+coal | smelt charcoal（碰巧对） | smelt charcoal P=0.90，确认前置条件后复用 | 平 |
| 要 charcoal，inv **没有** furnace | smelt charcoal → 执行层失败 | P(success)=0.10 → planner 重出 | v2 |
| 任务 budget 剩 5 min，要 smooth_stone（预测要 7 min） | 照跑，跑到 timeout | 提前预警，换更便宜的 stone 路径 | v2 |
| 全新 waypoint（库里没见过） | retrieve 空 → planner | retrieve 跨 waypoint 召回 + 多任务打分 → planner | 平/v2 |
| 需要在 cobblestone 的 dig/craft/smelt 中选 | 净胜场 → dig（盲选） | 状态条件下打分 → 多数情况 dig，少数情况 smelt | v2 |
| 同一动作连续成功 100 次 | 第 101 次仍盲信 | 第 101 次仍打分（状态变了就重审） | v2 |

---

## 11. 不变的部分（避免误解 v2 是大重写）

- 案例库 schema 不变。
- `record_decision` / `save_success_failure` 接口不变。
- planner、OracleGraph、STEVE-1 action server、env wrapper 全部不动。
- `exp_results/v2/*.json` 格式不动。
- 视频/JSON 路径不动。
- 只在 `select_case_decision()` 内部增加一个「打分 + 阈值门控」分支，并新增 `src/optimus1/decisioner/` 目录。
- 训练脚本独立放 `scripts/`，不影响在线运行。
- runtime 默认 `enabled: false`，先离线跑通再切。

---

## 12. 何时 v2 反而不如现行方法（诚实声明）

1. **特别简单的 waypoint**（planks、stick、crafting_table）：当前 100% 成功，多任务模型反而引入预测噪声。**对策**：白名单短路保留。
2. **多样性采集失败**：如果 50 runs 后仍然没采集到足够的动作分歧，模型还是会退化。**对策**：必须先看 Step 2 的诊断报告，再决定是否进 Step 3。
3. **失败模式标签噪声**：`status_detailed` 自动派生的 `timeout/crash/normal_fail` 不一定干净，第一次训可能 failure_mode 头学不出来。**对策**：先只训 success 头，failure_mode 头作为 stretch goal。
4. **结构化特征工程负担**：要维护 inventory 关键物品列表、ypos 桶、equipment 枚举。**对策**：从 30 个核心物品起，第一版不追求全。

---

## 13. 接下来需要从你这里确认

1. 是否同意「先采集多样性数据再训练」这条路径（Step 2 是 prerequisite，不是 nice-to-have）？
2. 多样性采集时是否同意全局关掉 `_best_exact_success_case` 短路（白名单除外）？
3. 失败模式头第一版是否暂缓（只训 success + steps 两个头）？
4. 结构化特征：inventory 取多少种物品（建议起步 30，可缩到 12 优先核心矿/工具）？

确认后我再写：

- `scripts/export_decisioner_dataset.py`
- `scripts/run_diversity_collection.py`（新）
- `src/optimus1/decisioner/feature_text.py`
- `src/optimus1/decisioner/value_head.py`
- `scripts/train_decision_value_head.py`
- `scripts/evaluate_decisioner_offline.py`
