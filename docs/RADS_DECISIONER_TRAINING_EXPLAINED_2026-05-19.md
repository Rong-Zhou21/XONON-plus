# RADS 决策器训练流程通俗说明

生成日期：2026-05-19
项目目录：`/home/yzb/zhourong/XENON-plus`

## 1. 先说结论：决策器到底在训练什么

XENON-plus 的决策器不是训练一个新的 Minecraft 控制器，也不是训练一个新的 planner。它训练的是一个成功率评分器：

```text
输入：当前状态 state + 当前 waypoint + 最终任务 final_goal + 候选动作 action
输出：这个 action 在当前状态下完成 waypoint 的成功概率 P(success)
```

例如当前任务是 `craft_a_iron_pickaxe`，当前 waypoint 是 `iron_ore`，候选动作是：

```text
dig down and mine iron_ore
```

决策器要回答的问题是：

```text
在当前背包、装备、高度、任务上下文下，
执行 dig down and mine iron_ore 成功的概率有多大？
```

所以它的职责是“判断已有候选动作是否值得复用”，不是“自己创造新动作”。如果它认为所有历史动作都不可靠，就回退给原来的 planner。

## 2. 总体训练管线

训练过程分五步：

```text
1. 在线实验产生 cases.json
2. export_decisioner_dataset.py 从 cases.json 导出训练样本
3. train_rads.py 把样本转成特征并训练 RADS
4. evaluate_rads_offline.py 离线评估
5. 在线运行时加载 rads_v2.pt，对候选动作打分
```

对应文件：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
  -> data/decisioner/rads_v1.jsonl
  -> artifacts/decisioner/rads_v2.pt
  -> reports/decisioner/offline_eval_v2.md
```

## 3. 第一步：实验如何产生训练样本

每次 XENON-plus 做一个 waypoint 决策，都会先写一条 pending case。

例如智能体准备做 `iron_ore`：

```json
{
  "waypoint": "iron_ore",
  "waypoint_num": 3,
  "original_final_goal": "craft_a_iron_pickaxe",
  "state_snapshot": {
    "inventory": {
      "stone_pickaxe": 1,
      "cobblestone": 4,
      "stick": 2
    },
    "equipment": "stone_pickaxe",
    "location_stats": {
      "ypos": 35
    },
    "biome": "plains"
  },
  "selected_action": "dig down and mine iron_ore",
  "outcome": {
    "status": "pending",
    "success": null
  }
}
```

执行结束后，系统会把 pending case 改成真实结果：

```json
{
  "outcome": {
    "status": "success",
    "success": true
  }
}
```

或者：

```json
{
  "outcome": {
    "status": "failed",
    "success": false
  }
}
```

这样，每条 case 就变成一个监督学习样本：

```text
(state, waypoint, final_goal, selected_action) -> success / failure
```

## 4. 一个最小例子：假设案例库只有 6 条

为了直观说明，假设 `cases.json` 里只有下面 6 条已结算 case：

| case | final_goal | waypoint | state 摘要 | selected_action | success |
|---|---|---|---|---|---:|
| A | craft_wooden_pickaxe | logs | 背包空，地表 | chop a tree | true |
| B | craft_stone_pickaxe | cobblestone | 有 wooden_pickaxe，y=60 | dig down and mine cobblestone | true |
| C | craft_stone_pickaxe | cobblestone | 没有 wooden_pickaxe，y=60 | dig down and mine cobblestone | false |
| D | craft_iron_pickaxe | iron_ore | 有 stone_pickaxe，y=35 | dig down and mine iron_ore | true |
| E | craft_iron_pickaxe | iron_ore | 只有 wooden_pickaxe，y=35 | dig down and mine iron_ore | false |
| F | craft_golden_pickaxe | gold_ore | 有 iron_pickaxe，y=20 | dig down and mine gold_ore | true |

训练时，模型应该学到类似规律：

```text
logs + chop a tree:
  通常成功，尤其在地表开局。

cobblestone + dig down:
  有 wooden_pickaxe 时更可能成功。
  没有 wooden_pickaxe 时更可能失败。

iron_ore + dig down:
  有 stone_pickaxe 或更好工具时更可能成功。
  只有 wooden_pickaxe 时更可能失败。

gold_ore + dig down:
  需要 iron_pickaxe，否则成功率应该低。
```

这就是决策器训练的本质：从历史 case 中学习“状态条件下的动作成功率”。

## 5. 第二步：导出训练数据

导出脚本是：

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

导出时会过滤掉不适合训练的 case：

| 被过滤的 case | 原因 |
|---|---|
| `run_uuid == legacy` | legacy case 缺少真实在线状态，主要是旧记忆迁移 |
| `outcome.status == pending` | 还不知道成功失败 |
| `outcome.status == excluded_infra` | 基础设施异常，不代表动作能力 |
| `outcome.status == crash_RuntimeError` | crash 样本标签不稳定 |
| `outcome.success` 不是 true/false | 不能作为监督标签 |

导出后，每一行 JSONL 都是一条训练样本。它保留的字段比原始 case 更精简：

```json
{
  "case_id": "...",
  "run_uuid": "...",
  "_position_in_run": 5,
  "waypoint": "iron_ore",
  "waypoint_num": 3,
  "original_final_goal": "craft_a_iron_pickaxe",
  "selected_action": "dig down and mine iron_ore",
  "state_snapshot": {
    "inventory": {
      "stone_pickaxe": 1,
      "cobblestone": 4
    },
    "equipment": "stone_pickaxe",
    "biome": "plains",
    "location_stats": {
      "ypos": 35,
      "biome_id": 0
    }
  },
  "outcome": {
    "status": "success",
    "success": true
  },
  "split": "train"
}
```

### 为什么按 run_uuid 切分

训练集、验证集、测试集不是随机按 case 切分，而是按 `run_uuid` 分组切分。

原因是同一次 Minecraft 运行中的 case 高度相关。例如一局任务里：

```text
logs -> planks -> crafting_table -> stick -> wooden_pickaxe -> cobblestone
```

这些 case 共享同一个世界、同一条轨迹、相近的 inventory 变化。如果把同一个 run 里的前半部分放进 train，后半部分放进 test，模型就等于在 test 时见过同一局游戏的上下文，指标会虚高。

按 `run_uuid` 分组后，一整局运行只会属于 train、val、test 其中一个 split。

当前 `rads_v2.pt` 使用的数据快照是：

| split | samples | runs | positive | negative |
|---|---:|---:|---:|---:|
| train | 1847 | 170 | 1542 | 305 |
| val | 351 | 36 | 330 | 21 |
| test | 376 | 38 | 307 | 69 |

## 6. 第三步：把 case 转成模型能读的特征

特征抽取代码在：

```text
src/optimus1/decisioner/feature.py
```

模型不会直接读取整个 JSON。它会把 case 转成几类特征。

### 6.1 类别特征

三类文本字段会变成 ID，再通过 embedding 变成向量：

| 字段 | 例子 |
|---|---|
| `waypoint` | `iron_ore` |
| `original_final_goal` | `craft_a_iron_pickaxe` |
| `selected_action` | `dig down and mine iron_ore` |

例如：

```text
waypoint_id: iron_ore -> 17
final_goal_id: craft_a_iron_pickaxe -> 23
action_id: dig down and mine iron_ore -> 41
```

每个 ID 会被映射成 6 维 embedding。

### 6.2 数值和 one-hot 特征

模型还会读取这些结构化状态：

| 特征 | 例子 | 作用 |
|---|---|---|
| equipment one-hot | `stone_pickaxe` | 判断当前手上工具 |
| biome one-hot | `plains` | 判断环境类别 |
| waypoint_num | `3` | 目标数量 |
| position_in_run | `5` | 当前是本轮第几个决策 |
| ypos | `35` | 当前高度 |
| ypos_bucket | `30-50` | 粗粒度高度层 |
| key inventory items | logs、planks、stick、ore、ingot 等数量 | 判断材料是否足够 |
| tool_owned_flags | 是否有 wooden/stone/iron pickaxe | 判断工具等级 |
| inv_unique_count | 背包物品种类数 | 粗略反映背包压力 |

总特征维度是：

```text
numeric dense features: 34
categorical embeddings: 18
total: 52
```

### 6.3 一个特征例子

原始 case：

```text
waypoint = iron_ore
final_goal = craft_a_iron_pickaxe
selected_action = dig down and mine iron_ore
inventory = {stone_pickaxe: 1, stick: 2, cobblestone: 4}
equipment = stone_pickaxe
ypos = 35
success = true
```

抽取后大致变成：

```text
numeric:
  equipment_onehot = stone_pickaxe
  biome_onehot = plains
  waypoint_num = log1p(3)
  position_in_run = log1p(5)
  ypos_norm = 35 / 64
  ypos_bucket = 30-50
  inventory_features:
    stick = log1p(2)
    cobblestone = log1p(4)
    iron_ore = 0
    iron_ingot = 0
    ...
  tool_owned_flags:
    wooden_pickaxe = 0
    stone_pickaxe = 1
    iron_pickaxe = 0

ids:
  waypoint_id = iron_ore
  final_goal_id = craft_a_iron_pickaxe
  action_id = dig down and mine iron_ore

label:
  success = 1
```

## 7. 第四步：RADS 模型怎么训练

训练脚本是：

```bash
python scripts/train_rads.py \
  --data data/decisioner/rads_v1.jsonl \
  --out artifacts/decisioner/rads_v2.pt \
  --report reports/decisioner/training_log_v2.json
```

RADS 的结构可以理解成三块：

```text
QueryEncoder:
  编码“当前要判断的 query”

CaseEncoder:
  编码“历史案例库里的 case”

DecisionHead:
  结合 query + 历史案例 attention context + action embedding
  输出 P(success)
```

### 7.1 query 是什么

训练时，每条样本都会轮流当 query。

例如 query 是：

```text
当前样本 D:
  waypoint = iron_ore
  state = 有 stone_pickaxe, y=35
  action = dig down and mine iron_ore
  label = success
```

模型要预测这个 query 的成功概率。

### 7.2 library 是什么

RADS 不是只看 query 本身，还会看训练集里的历史 case library。

对上面的 query D，library 里可能有：

```text
A: logs + chop a tree -> success
B: cobblestone + 有 wooden_pickaxe + dig cobblestone -> success
C: cobblestone + 无 wooden_pickaxe + dig cobblestone -> failed
E: iron_ore + 只有 wooden_pickaxe + dig iron_ore -> failed
F: gold_ore + 有 iron_pickaxe + dig gold_ore -> success
```

模型会用 attention 计算 query 和这些历史 case 的相关性。理想情况下，对于 query D：

```text
更关注：
  E: iron_ore + 只有 wooden_pickaxe + failed
  其他 iron_ore 成功样本

较少关注：
  logs、planks、gold_ore 等无关 waypoint
```

v2 里有 same-waypoint hard mask：如果同 waypoint 的历史样本足够多，attention 会优先限制在同 waypoint 内。这样做是为了防止模型用跨任务捷径，例如用“craft 类动作总体成功率高”去错误影响 `cobblestone` 或 `iron_ore`。

### 7.3 前向计算过程

一次前向计算可以写成：

```text
q = QueryEncoder(query)
C = CaseEncoder(train_library_cases)
attention = softmax(q @ C.T / tau)
context = attention @ C
logit = DecisionHead(q, context, action_embedding)
p_success = sigmoid(logit)
```

用简单话说：

1. QueryEncoder 把“当前状态 + 当前动作”编码成向量。
2. CaseEncoder 把历史案例编码成向量，且历史案例带 success/failure outcome。
3. Attention 找出哪些历史案例最像当前 query、最值得参考。
4. DecisionHead 综合当前 query 和历史经验，输出成功概率。

## 8. 第五步：模型怎么知道自己错了

训练不是凭感觉调参数，而是有明确 loss。

总损失是：

```text
L = BCE(success) + 0.1 * TripletLoss + 0.05 * WaypointReconstruction
```

### 8.1 BCE：成功率预测要对

BCE 是主损失。它要求：

```text
成功样本 -> p_success 应该高
失败样本 -> p_success 应该低
```

例子：

```text
query D:
  有 stone_pickaxe，挖 iron_ore，真实 success=true
  模型预测 p_success=0.30

这就错了，BCE 会推动 p_success 变高。
```

另一个例子：

```text
query E:
  只有 wooden_pickaxe，挖 iron_ore，真实 success=false
  模型预测 p_success=0.85

这也错了，BCE 会推动 p_success 变低。
```

BCE 解决的是“成功率数值本身要准确”。

### 8.2 Triplet loss：相似成功经验要靠近，失败经验要远离

Triplet loss 用三个样本：

```text
anchor:   iron_ore + 有 stone_pickaxe -> success
positive: iron_ore + 有 stone_pickaxe -> success
negative: iron_ore + 只有 wooden_pickaxe -> failed
```

它要求：

```text
anchor 和 positive 的 case 向量更近
anchor 和 negative 的 case 向量更远
```

这样训练出来的历史案例向量不只是为了分类服务，也真的形成了一个“案例检索空间”。

Triplet loss 解决的是：attention evidence 不能乱，它应该能检索到语义和结果都更相关的历史案例。

### 8.3 Waypoint reconstruction：case 向量要记住自己在解决哪个 waypoint

CaseEncoder 还有一个辅助任务：

```text
从 case vector 预测 waypoint_id
```

也就是说，编码后的 case 向量应该仍然能看出它是 `logs`、`cobblestone`、`iron_ore` 还是 `gold_ore`。

为什么需要这个？

如果没有约束，模型可能学到一些捷径：

```text
craft 类动作整体成功率高
mine 类动作整体更难
```

这种统计可能有用，但会导致 attention 跑到无关 waypoint 上。Waypoint reconstruction 会迫使 case vector 保留“当前到底在解决哪个 waypoint”的语义。

### 8.4 Waypoint-action 先验：保留历史基础成功率

RADS v2 还会计算：

```text
P(success | waypoint, action)
```

例如训练集中：

```text
cobblestone + dig down and mine cobblestone:
  success = 80
  failure = 20
  prior ~= 0.80

cobblestone + craft cobblestone:
  success = 0
  failure = 3
  prior 很低
```

这个 prior 会以 logit residual 的方式加到最终输出上。

作用是：在样本不多时，不让模型被 action embedding 带偏。例如 `craft` 这个词在很多合成任务里成功率很高，但 `craft cobblestone` 本身是错动作，先验可以把它压下去。

## 9. 一个完整 toy 训练轮次

假设当前训练 batch 里有两条 query：

| query | state | action | label |
|---|---|---|---:|
| Q1 | 有 stone_pickaxe，y=35 | dig down and mine iron_ore | 1 |
| Q2 | 只有 wooden_pickaxe，y=35 | dig down and mine iron_ore | 0 |

训练开始时模型可能还不会区分工具等级：

```text
Q1 p_success = 0.55
Q2 p_success = 0.60
```

这显然不好：

- Q1 是成功样本，应该更高。
- Q2 是失败样本，应该更低。

经过 BCE 更新后，模型参数会朝这个方向变化：

```text
stone_pickaxe + iron_ore + dig -> 分数上升
wooden_pickaxe + iron_ore + dig -> 分数下降
```

Triplet loss 同时推动：

```text
成功 iron_ore case 彼此靠近
失败 iron_ore case 与成功 iron_ore case 拉远
```

Waypoint reconstruction 同时推动：

```text
iron_ore case 的向量保留 iron_ore 信息
不要只记住“mine 类动作”
```

训练很多 batch 之后，模型会更接近：

```text
Q1 p_success = 0.90
Q2 p_success = 0.15
```

这时在线运行时，如果当前状态像 Q2，RADS 就会低置信并回退 planner 或触发上层前置条件修正，而不是盲目复用 `dig down and mine iron_ore`。

## 10. 防止答案泄漏：为什么要 mask same run

训练时，library 是 train set 本身。这样有一个风险：query 可能在 library 里找到同一局运行的近邻 case，甚至找到自己。

所以训练时会 mask：

```text
1. 当前 query 自己
2. 与 query 拥有相同 run_uuid 的所有 library case
```

例子：

```text
run_001:
  case 1 logs success
  case 2 planks success
  case 3 iron_ore failed

训练 case 3 时，case 1 和 case 2 也会被 mask。
```

原因是同一局运行的 case 太相关。如果不 mask，模型可能不是学会“一般规律”，而是记住“这一局的轨迹上下文”。

## 11. 训练完成后保存了什么

训练完成后输出：

```text
artifacts/decisioner/rads_v2.pt
```

这个文件不是只保存模型权重，还保存：

| 内容 | 作用 |
|---|---|
| `model_state` | RADS 模型参数 |
| `spec` | waypoint/action/final_goal vocab 和特征配置 |
| `config` | 模型超参数 |
| `library_vecs` | train cases 编码后的向量 |
| `library_meta` | 每个 library case 的 id、waypoint、action、success |
| `best_epoch` | 最佳验证轮次 |
| `best_val` / `test` | 离线指标 |

保存 `library_vecs` 很重要。在线推理时，RADS 不需要重新训练，也不需要重新编码训练集；它直接加载历史案例向量，作为 attention memory。

## 12. 训练好以后在线怎么用

在线运行时，在配置中开启：

```text
memory.case_memory.decisioner.enabled=true
memory.case_memory.decisioner.checkpoint=artifacts/decisioner/rads_v2.pt
memory.case_memory.decisioner.min_p_success=0.20
```

当系统遇到一个 waypoint，例如：

```text
waypoint = cobblestone
当前状态 = 有 wooden_pickaxe，y=62
```

CaseBasedMemory 会找出这个 waypoint 历史上出现过的候选动作，例如：

```text
candidate 1: dig down and mine cobblestone
candidate 2: craft cobblestone
candidate 3: smelt cobblestone
```

RADS 分别打分：

| candidate action | p_success |
|---|---:|
| dig down and mine cobblestone | 0.91 |
| craft cobblestone | 0.03 |
| smelt cobblestone | 0.02 |

于是系统选择：

```text
dig down and mine cobblestone
```

如果当前状态很差，例如没有 wooden_pickaxe：

| candidate action | p_success |
|---|---:|
| dig down and mine cobblestone | 0.12 |
| craft cobblestone | 0.03 |
| smelt cobblestone | 0.02 |

最高分 `0.12 < min_p_success=0.20`，RADS 不会强行选历史动作，而是返回 `None`，让上层调用 planner 或前置条件修正。

## 13. 真实训练指标怎么看

当前 `artifacts/decisioner/rads_v2.pt` 的离线 test 指标：

| 指标 | 数值 |
|---|---:|
| test AUC | 0.9110 |
| test AP | 0.9739 |
| best F1 | 0.9535 |
| ECE | 0.0866 |
| top-1 attention same-waypoint rate | 0.9441 |

这些指标大致含义：

| 指标 | 通俗解释 |
|---|---|
| AUC | 成功样本是否整体排在失败样本前面 |
| AP | 在正样本较多时，模型找成功样本的质量 |
| F1 | 某个阈值下成功/失败分类的综合效果 |
| ECE | 概率是否校准，比如预测 0.8 的样本是否真的约 80% 成功 |
| same-waypoint rate | attention 最关注的历史 case 是否来自同一个 waypoint |

线上 67 任务 single-shot 对比：

| 方法 | success / 67 | rate |
|---|---:|---:|
| retrieval-only first attempt | 39 | 58.2% |
| RADS decisioner single shot | 49 | 73.1% |

这说明训练出来的决策器不只是离线指标好，在线也能把一部分任务从“第一次尝试失败”变成“单次成功”。

## 14. 它不会学到什么

为了避免误解，需要明确 RADS 的边界。

RADS 会学习：

```text
在这个状态下，这个高层 action 完成 waypoint 的概率。
```

RADS 不会学习：

```text
如何移动鼠标看见矿物
如何精确挖方块
如何从画面识别地形
如何生成全新的 Minecraft 策略
如何替代 STEVE-1 低层控制
```

例如，如果当前 action 是：

```text
dig down and mine gold_ore
```

RADS 可以判断这个 action 在当前 state 下成功率高不高。但真正执行时，还是要靠：

- STEVE-1 输出动作
- env wrapper 稳定 attack / movement
- resource_ledger 记录资源
- PerceptionActionSuite 做掉落物、背包、矿层恢复

所以 RADS 是“高层动作价值判断器”，不是“低层挖矿智能体”。

## 15. 一句话总结

RADS 决策器的训练流程是：把 XENON-plus 在线实验产生的 case memory 转成监督学习样本，用状态、waypoint、最终目标和候选动作预测成功/失败；训练时通过历史案例 attention、BCE、triplet loss、waypoint reconstruction 和 waypoint-action 先验，学到一个可解释的 `P(success | state, waypoint, action)` 评分器；上线后它对历史候选动作排序，低置信时回退 planner，并把新决策继续写回案例库。
