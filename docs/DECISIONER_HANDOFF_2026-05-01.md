# XENON-plus 决策器创新接手文档

更新时间：2026-05-01  
项目目录：`/home/yzb/zhourong/XENON-plus`  
原版对照目录：`/home/yzb/zhourong/XENON-main`  
当前主线最新提交：`bca2131 data: record iron log-jump reruns`

## 1. 当前一句话状态

XENON-plus 已经完成了第一阶段改造：把原 XENON 的 waypoint 动作记忆库替换为面向决策器的案例库，并加入了初步的环境感知、物资账本、视频保存修正和地表/地下执行稳定逻辑。当前适合进入第二阶段：基于案例库构建可训练、可评估的决策器。

任务级 step 记录已经存在：每次实验的 `exp_results/v2/*.json` 中有 `steps` 和 `minutes` 字段，`save_decomposed_plan/*/*.json` 中每条 plan 也保存了 `steps`。当前 case 条目本身没有 per-case duration 字段。

## 2. Git 与工作区注意事项

当前关键提交：

- `bca2131`：记录 log 后禁跳逻辑下的 iron 重跑结果。
- `761cc9e`：地表跳跃只在拿到 log 后短时锁定。
- `1126936`：首次收窄 jump locking 到地下/树接触。
- `3176ac2`：记录 v2 rerun outcomes。
- `4d90de4`：修复 planner waypoint 需求保留与树接触检测。
- `6b68e12`：使用 resource ledger 辅助 waypoint 完成判断。

当前有一个未跟踪文件：

- `train_value_head.py`

它看起来是一个基于 `all-MiniLM-L6-v2` embedding 的二分类 value head 训练原型，尚未纳入 git。新窗口接手时不要误删，决定是否纳入前先确认用途和数据格式。

## 3. 目录速览

核心代码：

- `src/optimus1/main_planning.py`  
  主任务流程、waypoint 生成、案例检索决策、planner fallback、实验结果 JSON 写出。

- `src/optimus1/memories/case_memory.py`  
  新案例库与检索式决策器。当前替代原 `DecomposedMemory`。

- `src/optimus1/env/wrapper.py`  
  环境封装、物资账本、动作稳定、地表/地下执行逻辑、背包清理、respawn reset、视频帧记录入口。

- `src/optimus1/env/mods/recorder.py`  
  视频保存。当前保存连续 raw POV 帧，不再写 prompt 侧栏。

- `src/optimus1/util/video.py`  
  OpenCV mp4v 写出视频，统一按首帧尺寸写 raw POV。

配置与数据：

- `src/optimus1/conf/evaluate.yaml`  
  默认版本为 `v1`，memory 路径基于 `${prefix}/${version}`。

- `src/optimus1/conf/benchmark/*.yaml`  
  Wood/Stone/Iron/Gold/Diamond/Redstone/Armor 各类任务定义。

- `src/optimus1/memories/ours_planning/v1/case_memory/cases.json`  
  当前案例库。注意：即使视频/实验结果在 `v2`，案例库仍在 `v1`，因为运行时通常只 override `record.video.path` 与 `results.path`，没有 override `version`。

- `exp_results/v2/`  
  当前主要实验结果 JSON。每条任务级结果包含 `steps`、`minutes`、`success`、`status_detailed`、`video_file` 等。

- `videos/v2/`  
  当前主要视频目录。视频不上传 Git，只保留本地。

## 4. 当前任务执行主流程

入口：

```bash
python -u src/optimus1/main_planning.py \
  server.port=9100 env.times=1 benchmark=iron \
  "evaluate=[10]" prefix=ours_planning exp_num=53010 seed=0 world_seed=10 \
  record.video.path=videos/v2 results.path=exp_results/v2
```

高层流程：

1. `main_planning.py` 读取 benchmark 任务。
2. 初始化 MineRL/Minecraft 环境。
3. 初始化 `CaseBasedMemory`。
4. `OracleGraph` 根据最终目标和当前 inventory 生成 waypoint。
5. `make_plan()` 为当前 waypoint 选择动作：
   - 先用 `CaseBasedMemory` 检索可复用案例。
   - 有可信案例则直接复用案例中的 subgoal。
   - 没有可信案例则调用原 planner。
   - planner 输出也记录为 pending case。
6. 执行 subgoal：
   - `craft/smelt` 走 helper。
   - `chop/dig/mine` 走 STEVE-1 action server。
7. waypoint 成功或失败后，更新对应 case 的 outcome。
8. 任务结束后写：
   - `exp_results/v2/*.json`
   - `save_decomposed_plan/<success|failed>/*.json`
   - `videos/v2/<Category_Task>/<biome>/<status>/*.mp4`

## 5. 案例库当前设计

实现文件：`src/optimus1/memories/case_memory.py`

当前案例库文件：

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

当前统计：

- case 总数：2660
- resolved：2659
- success：2212
- failed：403
- pending：1
- legacy bootstrap：35

唯一 pending case：

```text
run_uuid=LGW99zExP6RQ3qVTeVeThB
waypoint=logs
selected_action=chop a tree
original_final_goal=golden_helmet
```

后续训练前建议过滤 `outcome.status == "pending"`。

### 5.1 Case schema

典型字段：

```json
{
  "id": "run_uuid:index:timestamp",
  "created_at": "UTC time",
  "run_uuid": "episode id",
  "original_final_goal": "final task goal",
  "environment": "forest/plains/legacy",
  "waypoint": "iron_ingot",
  "waypoint_num": 3,
  "state_snapshot": {
    "inventory": {},
    "equipment": "stone_pickaxe",
    "location_stats": {},
    "plain_inventory": {},
    "biome": "plains",
    "obs_summary": {}
  },
  "similarity_text": "waypoint + inventory + equipment + location + biome",
  "candidate_actions": [
    {"action": "smelt iron_ingot", "source": "planner_selected"}
  ],
  "selected_action": "smelt iron_ingot",
  "selected_subgoal": {"task": "smelt iron_ingot", "goal": ["iron_ingot", 3]},
  "selected_subgoal_str": "...",
  "decision_trace": {
    "source": "case_memory | planner | semantic_fallback",
    "confidence": 1.0,
    "selected_case_id": "..."
  },
  "outcome": {
    "status": "success | failed | timeout_non_programmatic | crash_RuntimeError | pending",
    "success": true,
    "recorded_at": "UTC time",
    "state_snapshot": {}
  }
}
```

### 5.2 检索逻辑

当前 `CaseBasedMemory` 使用：

- `SentenceTransformer("all-MiniLM-L6-v2")`
- `similarity_text`
- cosine similarity

决策顺序：

1. `_best_exact_success_case(waypoint)`  
   当前优先 exact waypoint 成功案例。这个路径的 `confidence=1.0`，不严格看状态相似度。

2. `_retrieve_cases(..., successful_only=True)`  
   检索成功案例，要求同 waypoint 且分数超过 `reuse_threshold=0.72`。

3. planner fallback  
   如果没有可信案例，原 planner 生成动作。

失败案例使用方式：

- `retrieve_failed_subgoals(waypoint)` 按 waypoint/action 聚合成功失败分数。
- 当某动作净分低于 `plan_failure_threshold` 时，传给 planner 作为 failed subgoals。

去重逻辑：

- `_remove_exact_duplicate_case()` 会移除完全相同且已 resolved 的 case。
- `id`、`created_at`、`run_uuid` 和 `outcome.recorded_at` 不参与 duplicate key。

## 6. 与原版 XENON 的差异

原版 XENON 的记忆单元：

```text
waypoint -> action -> success/failure count
```

XENON-plus 的记忆单元：

```text
environment/state + waypoint + candidate actions + selected action + outcome
```

原版流程：

```text
waypoint 有成功动作 -> 复用
没有 -> planner
```

当前 plus 流程：

```text
当前状态 + waypoint
  -> 检索案例
  -> 高置信案例复用
  -> 否则 planner fallback
  -> 决策记录成 pending case
  -> waypoint/episode 结束后更新 success/failed
```

注意：原 planner 没有被重写；当前创新点主要在 memory/decision layer 和执行感知层。

## 7. 环境感知与执行逻辑当前状态

实现文件：`src/optimus1/env/wrapper.py`

### 7.1 Resource ledger

`resource_ledger` 记录：

- `last_inventory`
- `last_pickup_stats`
- `last_mine_block_stats`
- `max_inventory`
- `collected`
- `pickup`
- `mined_blocks`

用途：

- 解决“视觉没看见但物品已经进背包/统计”的问题。
- 辅助 waypoint 完成判断。
- 辅助 pending drop 判断。
- 给结果 JSON 提供 `resource_ledger`。

### 7.2 背包清理

当背包压力过高时，`_maybe_cleanup_inventory()` 会尝试丢弃非 waypoint、非关键工具的低价值物品，例如：

- leaves
- flowers
- grass
- sapling
- dirt/gravel/sand
- granite/diorite/andesite 等

保护对象包括：

- 当前 goal 展开的物品
- 工具
- logs/planks
- crafting table/furnace/coal 等关键中间物

### 7.3 地表找树与跳跃

当前逻辑：

- 地表找树、接近树、未拿到 log 前不禁跳。
- 检测到 inventory/pickup 中出现 `_log` 增量后，短时间禁 `jump/sprint`。
- 默认 `XENON_SURFACE_LOG_JUMP_LOCK_TICKS=30`。

原因：

之前禁跳是为了解决短跳、短点击导致砍树/挖矿动作被打断的问题。但全局禁跳会让智能体无法跨越地表一格高地形。现在只在明确拿到 log 后短暂稳定，避免误伤地表搜索和接近树。

### 7.4 地表探索 prompt

在 `main_planning.py` 中，tree chopping 有一个轻量探索机制：

- 如果 `chop a tree` 长时间没有 log 相关进度，临时切到 `find a tree`。
- 如果检测到 log activity 或持续攻击接触，切回原始 `chop a tree`。
- 只用于树/木头类地表任务，不用于地下挖矿。

相关环境变量：

- `XENON_TREE_EXPLORE_PROMPT`
- `XENON_TREE_CHOP_STALE_TICKS`
- `XENON_TREE_EXPLORE_TICKS`
- `XENON_TREE_CONTACT_ATTACK_TICKS`

### 7.5 地下挖掘方向调整

地下 mining subgoal 默认 `dig_down`。

如果目标矿物还没收集够，但 resource ledger 发现更深层矿物已经出现，说明可能挖过目标层：

- `dig_down` 临时切换为 `dig_forward`
- 目标数量满足后切回原 prompt

这不是直接写具体矿物坐标规则，而是通过“是否出现更深层矿物”调整 STEVE-1 prompt。

### 7.6 Respawn reset

检测死亡/复活后：

- 清除低层控制状态
- 请求 STEVE-1 policy reset
- 恢复当前 waypoint-aware prompt

没有让 planner 重新介入。

### 7.7 视频保存

当前视频保存为连续 raw POV：

- `RecorderMod.step()` 每 step 保存 `obs["pov"]` copy。
- `RecorderMod.save()` 先 snapshot 帧列表，再异步写出。
- `write_video()` 使用 OpenCV `mp4v` 写 raw RGB 转 BGR。

视频路径示例：

```text
videos/v2/Iron_Craft_a_tripwire_hook/plains/success/2026_05_01_...mp4
```

JSON 不和视频放一起；实验 JSON 在 `exp_results/v2`。

## 8. 当前实验状态

当前 v2 按任务唯一成功统计：

| 类别 | 总数 | 成功 | 未成功 |
|---|---:|---:|---:|
| Wood | 10 | 10 | 0 |
| Stone | 9 | 7 | 2 |
| Iron | 16 | 14 | 2 |
| Gold | 6 | 5 | 1 |
| Diamond | 7 | 7 | 0 |
| Redstone | 6 | 5 | 1 |
| Armor | 13 | 9 | 4 |

当前仍未成功任务：

| 类别 | task_id | task |
|---|---:|---|
| Stone | 2 | `craft_a_stone_axe` |
| Stone | 5 | `craft_a_smoker` |
| Iron | 2 | `craft_a_iron_axe` |
| Iron | 14 | `craft_a_blast_furnace` |
| Gold | 1 | `craft_a_golden_pickaxe` |
| Redstone | 2 | `craft_an_activator_rail` |
| Armor | 6 | `craft_diamond_chestplate` |
| Armor | 7 | `craft_diamond_leggings` |
| Armor | 10 | `craft_golden_leggings` |
| Armor | 12 | `craft_golden_chestplate` |

重要汇总文件：

- `exp_results/v2/full_67_v2_20260429_summary.json`
- `exp_results/v2/rerun_unsuccessful_v2_20260430_summary.json`
- `exp_results/v2/iron_unsuccessful_after_jump_20260501_summary.json`
- `exp_results/v2/iron_never_success_log_lock_20260501_summary.json`

最近一次 iron never-success 重跑：

| 任务 | 结果 | steps |
|---|---:|---:|
| `craft_a_iron_axe` | failed, `crash_RuntimeError` | 12001 |
| `craft_a_tripwire_hook` | success | 4783 |
| `craft_an_iron_bars` | success | 6007 |
| `craft_a_blast_furnace` | failed, `timeout_non_programmatic` | 11997 |

## 9. 任务级 step 记录

任务级 step 记录已经可用。

来源一：`exp_results/v2/*.json`

```json
{
  "task": "craft_a_tripwire_hook",
  "success": true,
  "status_detailed": "success",
  "steps": 4783,
  "minutes": 3.99,
  "video_file": "videos/v2/..."
}
```

来源二：`src/optimus1/memories/ours_planning/v1/save_decomposed_plan/<status>/*.json`

每条 plan append 时保存：

```json
{
  "id": "run_uuid",
  "goal": "...",
  "video": "...",
  "planning": [],
  "status": "success",
  "steps": 4783
}
```

当前 case 条目没有记录：

- `episode_steps`
- `waypoint_start_step`
- `waypoint_end_step`
- `duration_steps`

如果后续只做任务级评估，现有 `exp_results` 足够。如果要训练决策器学习“哪个动作更快”，建议在新 case 中补充这些字段，或通过 `run_uuid` 从 result JSON join 出 `episode_steps`。

## 10. 实验运行环境

当前常用容器：

```bash
docker exec -it xenon_plus_case bash
cd /app/repo
```

常用环境变量：

```bash
export PYTHONPATH=/app/repo:/app/repo/src:/app/repo/minerl:${PYTHONPATH:-}
export HF_HOME=/app/LLM
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export QWEN_BACKEND=vllm
export QWEN_VLLM_BASE_URL=http://172.17.0.1:8000/v1
export QWEN_VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export XENON_DISABLE_STUCK_KILL=1
```

本地 vLLM 服务：

```text
http://172.17.0.1:8000/v1
Qwen/Qwen2.5-VL-7B-Instruct
```

当前 action server 使用端口：

```text
server.port=9100
```

单任务运行示例：

```bash
xvfb-run -a python -u src/optimus1/main_planning.py \
  server.port=9100 env.times=1 env.max_minutes=10 benchmark=iron \
  "evaluate=[10]" prefix=ours_planning exp_num=53010 seed=0 world_seed=10 \
  record.video.path=videos/v2 results.path=exp_results/v2
```

任务之间建议清理：

```bash
pkill -f "[j]ava.*(GradleStart|Minecraft|Malmo)" 2>/dev/null || true
pkill -x Xvfb 2>/dev/null || true
sleep 10
```

## 11. 决策器创新建议

当前系统还只是“检索式案例决策器 + planner fallback”。下一步建议不要直接替换 planner，而是做一个可评估的 decisioner layer。

### 11.1 第一件事：导出训练数据

从 `cases.json` 导出 JSONL，过滤：

- 保留 `outcome.status != "pending"`
- 过滤 `infra_early_stop` 对应 run，如果能通过 result JSON join 到。
- 成功案例作为正样本。
- 有效失败案例作为负样本。

建议样本结构：

```json
{
  "case_id": "...",
  "run_uuid": "...",
  "text": "waypoint: ... inventory: ... equipment: ... action: ...",
  "waypoint": "iron_ingot",
  "selected_action": "smelt iron_ingot",
  "label": 1,
  "outcome_status": "success",
  "episode_steps": 4783,
  "task": "craft_a_tripwire_hook"
}
```

任务级 `episode_steps` 可以通过 `run_uuid` 从 `exp_results/v2/*.json` join。

### 11.2 第二件事：离线评估

在 runtime 改动前，先做离线评估：

- retrieval baseline：当前 exact/retrieval 逻辑。
- trained value head：用 case text/action 打分。
- combined：retrieval topK 后用 value head rerank。

评估指标：

- action top-1 accuracy
- success case ranking AUC
- success@K
- 平均 predicted success 与真实 success 的校准误差
- 平均 steps / success steps 作为效率指标

### 11.3 第三件事：runtime 低风险接入

建议先不要替换 `make_plan()` 的整体结构，只加一个可插拔 reranker：

```text
CaseBasedMemory.retrieve topK successful/failed cases
  -> DecisionScorer 给候选 action 打分
  -> 高分且超过阈值则复用
  -> 否则 planner fallback
```

这样能保持与当前 baseline 可比。

### 11.4 不建议做的事

- 不要写“正确答案规则”直接告诉智能体该做什么。
- 不要把 helper/env bug 的失败当作负样本。
- 不要让 case 只按 waypoint exact success 盲目复用，否则训练决策器的收益难体现。
- 不要把视频上传 Git。
- 不要先改 planner；当前创新点应集中在 decision layer。

## 12. 当前设计中的关键限制

1. Exact waypoint success 复用太强  
   `_best_exact_success_case()` 可能绕开状态相似度，直接复用同 waypoint 成功动作。后续训练式决策器要考虑削弱或改成 scorer rerank。

2. Case 没有 per-waypoint duration  
   当前可以做任务级 step 评估，但不能直接算某个 waypoint 的平均耗时。

3. Case 文件较大  
   当前 2660 cases，每次初始化会 encode 全量 `similarity_text`。规模继续增长后，启动和检索速度可能成为问题。

4. 失败标签需要清洗  
   `crash_RuntimeError`、`env_step_timeout`、helper bug、正常能力失败要分开。训练前必须做 failure taxonomy。

5. 版本命名容易混淆  
   `results/videos` 用 `v2`，但 memory 仍在 `v1`。后续如果建立正式实验版本，建议显式决定是否把 memory 也切到 `v2` 或新目录。

## 13. 给新窗口的推荐执行顺序

1. 先读本文件。
2. 确认工作区：

```bash
cd /home/yzb/zhourong/XENON-plus
git status --short
git log --oneline -5
```

3. 看当前案例库与决策器：

```bash
sed -n '1,220p' src/optimus1/memories/case_memory.py
sed -n '220,520p' src/optimus1/memories/case_memory.py
```

4. 看决策接入点：

```bash
sed -n '220,360p' src/optimus1/main_planning.py
```

5. 看任务级 steps：

```bash
python - <<'PY'
import json, glob, os
for p in sorted(glob.glob('exp_results/v2/*.json'))[:5]:
    if 'summary' in os.path.basename(p):
        continue
    d=json.load(open(p))
    print(d['task'], d['success'], d['steps'], d['minutes'])
PY
```

6. 先写离线数据导出和评估脚本，再考虑 runtime 接入。

## 14. 最小下一步任务建议

建议新窗口从这三个文件开始：

1. 新增 `scripts/export_decision_cases.py`  
   读取 `cases.json` 和 `exp_results/v2/*.json`，按 `run_uuid` join 出 task-level steps，导出 `decision_cases.jsonl`。

2. 新增 `scripts/evaluate_case_decisioner.py`  
   做 retrieval baseline、训练 scorer baseline 的离线评估。

3. 可选整理 `train_value_head.py`  
   当前未跟踪，若继续使用，应改成读取 `decision_cases.jsonl`，并保存模型、配置、评估结果。

完成这些后，再考虑在 `CaseBasedMemory.select_case_decision()` 里接入训练式 scorer。
