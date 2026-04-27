# XENON 端到端任务追踪：Craft a diamond pickaxe / diamond axe

> **素材来源**：本文档以 `run_uuid = gPYVQFXVidRsvAmMg4MvK8`（"Craft a diamond axe"，forest biome，2026‑04‑18 18:15 启动，10281 步 / 8.57 分钟成功）的真实日志为主素材，逐步还原输入输出。
> "钻石镐 (diamond_pickaxe)" 与 "钻石斧 (diamond_axe)" 的 **依赖链完全一致（前 12 个 waypoint 相同，最后一个合成目标的物品 id 不同）**，因此流程、数据结构、I/O 形态 100% 通用，差异点单独列在 §11。
> 关键日志文件：`logs/eval/2026-04-18/18-15-21-365247/main_planning.log`
> 对应结果：`exp_results/v1/ours_planning_craft_a_diamond_axe_502_success_forest_gPYV.json`

---

## 目录

0. [整体架构速览](#0-整体架构速览)
1. [阶段 A — 启动 `app.py`（Planner + Controller HTTP 服务）](#1-阶段-a--启动-apppyplanner--controller-http-服务)
2. [阶段 B — 启动 `main_planning`（MineRL 实验进程）](#2-阶段-b--启动-main_planningminerl-实验进程)
3. [阶段 C — 启动 MCP‑Reborn（Minecraft Java 服务端）](#3-阶段-c--启动-mcp-rebornminecraft-java-服务端)
4. [阶段 D — Server reset + Env reset（每 trial 1 次）](#4-阶段-d--server-reset--env-resetreset每-trial-1-次)
5. [阶段 E — 顶层 waypoint 分解（OracleGraph 决定骨架）](#5-阶段-e--顶层-waypoint-分解oraclegraph-决定骨架)
6. [阶段 F — `make_plan` 一步：由 waypoint 变成 subgoal](#6-阶段-f--make_plan-一步由-waypoint-变成-subgoal)
7. [阶段 G — 底层执行：craft/smelt 走 Helper，mine/chop 走 STEVE‑1](#7-阶段-g--底层执行craftsmelt-走-helpermine-chop-走-steve-1)
8. [阶段 H — 运行中的视觉 reasoning 干预](#8-阶段-h--运行中的视觉-reasoning-干预可选)
9. [阶段 I — 13 个 waypoint 的真实 I/O 逐步还原](#9-阶段-i--13-个-waypoint-的真实-io-逐步还原)
10. [阶段 J — 成功判定、记忆写回、结果落盘](#10-阶段-j--成功判定记忆写回结果落盘)
11. [diamond_pickaxe 与 diamond_axe 的唯一差异](#11-diamond_pickaxe-与-diamond_axe-的唯一差异)
12. [一张图的"数据流总览"](#12-一张图的数据流总览)

---

## 0. 整体架构速览

三个进程、两张 GPU、一个容器：

```
┌─── 宿主机 (/home/yzb/zhourong/XENON-main) ───────────────────────────────┐
│  docker container: xenon_run  (image sjlee1218/xenon:latest)             │
│  ┌───────────────────────────┐        ┌────────────────────────────────┐│
│  │ [Process A] app.py        │ 9000   │ [Process B] main_planning.py   ││
│  │ FastAPI HTTP server       │◀──────▶│ 控制主循环、记忆读写、env.step ││
│  │ - Qwen2.5-VL-7B  (GPU0)   │ REST   │ (DecomposedMemory, OracleGraph)││
│  │ - STEVE-1 + MineCLIP(GPU0)│        └─────────────┬──────────────────┘│
│  │ - NewHelper (脚本化合成)  │                      │ gym API            │
│  └───────────────────────────┘                      ▼                    │
│                                        ┌────────────────────────────────┐│
│                                        │ [Process C] MCP-Reborn (Java)  ││
│                                        │ 定制版 Minecraft 服务端 (12352)││
│                                        │ 真实世界、物理、inventory、GUI ││
│                                        └────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

**核心数据通路**：

1. **观测 (obs)**：Minecraft → MineRL → `env.step()` 返回 dict，其中 `obs["pov"]` 是 **360×640×3 的 RGB 图像**（智能体第一人称视角），`obs["inventory"]` 是一个 dict `{item_name: count}`。
2. **决策 (action)**：`main_planning` 把 `obs["pov"]` base64 编码后 POST 给 `app.py:/chat`，由里面的 STEVE‑1 吐回一个 **MineRL action dict**（键盘鼠标动作）。
3. **规划 (subgoal)**：只在"当前 waypoint 没有成功记忆"时才调 VLM，请求体带 `waypoint` 字符串 + 当前 POV 图像，VLM 返回 JSON `{"task": "...", "goal": ["<item>", N]}`。

**物理意义**：app.py 是"大脑 + 小脑"（Planner 是大脑负责分解任务、Controller 是小脑负责把自然语言 prompt 转为低级动作）；main_planning 是"躯干"（它才是最外层的控制主循环、它决定此时该问 Planner 还是该问 Controller）；MCP‑Reborn 是"身体 + 世界"。三者通过 REST + gym 消息解耦。

---

## 1. 阶段 A — 启动 `app.py`（Planner + Controller HTTP 服务）

### 1.1 宿主机侧命令

```bash
docker exec -d xenon_run bash -c '
  cd /app/repo && 
  export HF_HOME=/app/LLM && 
  export HF_ENDPOINT=https://hf-mirror.com && 
  export QWEN_DEVICE_ID=0 && 
  xvfb-run -a python app.py --port 9000 \
    --plan_model Qwen/Qwen2.5-VL-7B-Instruct \
    > /tmp/app_server.log 2>&1'
```

### 1.2 容器内发生了什么（输入→输出）

| # | 动作（代码位置） | 输入 | 输出 |
|---|---|---|---|
| 1 | 解析 argparse（`app.py:219`） | `--plan_model`, `--in_model`, `--port`, … | 配置对象 `args` |
| 2 | `AgentFactory(...)` 构造（`src/optimus1/server/agent.py`） | `args` | 一个 `agent` 实例，持有三个模型句柄 |
| 3 | `PlanningModel(...)`（`src/optimus1/models/qwen_vl_planning.py:79`） | `model_path=Qwen/Qwen2.5-VL-7B-Instruct`, `device_id=0` | `Qwen2_5_VLForConditionalGeneration` 权重（bf16, ~15.6 GB）加载到 `cuda:0`，以及 `AutoProcessor` |
| 4 | `load_mineclip_wconfig()` | `checkpoints/mineclip/attn.pth` | **MineCLIP**：把语言 prompt 编码成 goal embedding（供 STEVE‑1 条件化） |
| 5 | `load_steve1_policy(...)` | `checkpoints/vpt/2x.model`, `checkpoints/steve1/steve1.weights`, `checkpoints/steve1/steve1_prior.pt` | **STEVE‑1**：VPT 策略 + 先验，条件于 MineCLIP goal embedding，输出 MineRL action |
| 6 | `NewHelper` 初始化 | 无 | 脚本化 GUI 合成/冶炼器（完全不走神经网络，直接操控 inventory slot） |
| 7 | `uvicorn.run(app, port=9000)` | FastAPI app | 监听 `127.0.0.1:9000` 提供 `/chat`, `/reset`, `/shutdown` |

**物理意义**：这一步把 3 个"能力"灌进一个常驻进程，再用 HTTP 把它们"打包"成黑盒。以后不管是 planning 还是 action 请求，`main_planning` 只要 POST 到 9000 端口，服务内部自行路由到 Qwen-VL / STEVE-1 / Helper。

### 1.3 `/chat` 的 type 字段（路由表）

见 `app.py:93-228`：

| `req.type` | 处理者 | 语义 |
|---|---|---|
| `"decomposed_plan"` | `agent.decomposed_plan` → Qwen-VL | **顶层规划**：给一个 waypoint，产出 `{task, goal}` JSON |
| `"context_aware_reasoning"` | `agent.context_aware_reasoning` → Qwen-VL ×2 | **运行中介入**：看当前画面决定要不要换 prompt |
| `"action"` | `agent.action` → MineCLIP→STEVE‑1 | **单步动作**：文本 prompt + 图像 → 鼠标键盘 dict |
| `"plan"`, `"replan"`, `"fixjson"`, `"retrieval"`, `"reflection"` | 旧 Optimus‑1 兼容接口 | XENON planning 主路径**不触发**这些 |

---

## 2. 阶段 B — 启动 `main_planning`（MineRL 实验进程）

### 2.1 宿主机侧命令

```bash
docker exec xenon_run bash -c '
  cd /app/repo && 
  export HF_HOME=/app/LLM && 
  xvfb-run -a python -m optimus1.main_planning \
    server.port=9000 env.times=1 benchmark=diamond \
    evaluate="[1]" prefix="ours_planning" \
    exp_num=502 seed=0 world_seed=2'
```

- `benchmark=diamond` → 加载 `src/optimus1/conf/benchmark/diamond.yaml`
- `evaluate=[1]` → 从该 yaml 的 `all_task` 里取 **id=1** 的 `{instruction: "Craft a diamond pickaxe", goal: "diamond_pickaxe"}`
  - 真实 trace 用的是 `evaluate=[2]`（diamond_axe）
- 环境变量 seed 与 world_seed 让 Minecraft 生成可复现的地图

### 2.2 Hydra 配置拼装（`main_planning.py:559-607`）

输出示例（真实日志摘抄）：

```
main_ours_planning.py is executed.
benchmark: diamond
is_fixed_memory: False
Growing memory
prefix: ours_planning
Load pretrained SentenceTransformer: all-MiniLM-L6-v2
Running Tasks: ['Craft a diamond axe']
```

**物理意义**：这里并**没有**加载 Qwen 或 STEVE-1（那些在 app.py 侧）。本进程只加载一个轻量级的 sentence-transformers（`all-MiniLM-L6-v2`, 90 MB）用来给 waypoint 做 embedding 相似度检索——这就是为什么实验跑起来之后你看到的 GPU 峰值是在 app.py 侧、而 main_planning 侧 GPU 占用很小。

### 2.3 DecomposedMemory 初始化（`memories/decomposed_memory.py:46-82`）

输入：`cfg["memory"]` 字段 + 磁盘上已有的 `src/optimus1/memories/ours_planning/v1/waypoint_to_sg/*.json`。

过程：
1. 为每个 `waypoint.json`（如 `logs.json`、`cobblestone.json`）用 MiniLM 编码 waypoint **字符串本身**，得到一个 384 维向量，存成 `<waypoint>.pt`（以便之后跨任务做相似度 top‑K 检索）。
2. 扫一遍所有 `*.json`，按 `is_succeeded_waypoint()` 规则（`success>0` 且 `success - failure > -plan_failure_threshold(2)`）挑出**已成功的 waypoint 列表** `self.succeeded_waypoints`。

输出（状态）：
```python
self.succeeded_waypoints = ['logs', 'planks', 'stick', 'crafting_table',
    'wooden_pickaxe', 'cobblestone', 'stone_pickaxe', 'furnace',
    'iron_ore', 'iron_ingot', 'iron_pickaxe', 'diamond', …]
```

**物理意义**：这是 XENON 的"经验库"加载。本次实验之后任何一次 `make_plan` 都先问它一句"这个 waypoint 我有没有靠谱的成功动作？"，有就直接复用，没有才打扰 VLM。

---

## 3. 阶段 C — 启动 MCP‑Reborn（Minecraft Java 服务端）

在 `main_planning.py:609` 的 `env = env_make(...)` 里，MineRL 会 fork 出一个 Java 子进程。真实日志（第 120 行）：

```
[minerl.env.malmo.instance.d00c3d][INFO] - Starting Minecraft process:
  ['/app/repo/minerl/minerl/env/../MCP-Reborn/launchClient.sh',
   '-port', '12352', '-env', '-runDir', '.../MCP-Reborn/run']
[minerl.env.malmo.instance.d00c3d][INFO] - Minecraft process ready
[minerl.env.malmo][INFO] - Logging output of Minecraft to ./logs/mc_3352.log
```

- Java 子进程监听 `localhost:12352`，等 MineRL 通过 Malmo 协议发指令。
- Python 侧（`main_planning`）用 gym 风格 API (`env.reset()`, `env.step(action)`) 和它通话。
- 进程启动完成后会发一个 "Minecraft process ready"。日志中那些 OpenAL / Narrator 报错可忽略（没有声卡的 xvfb 环境下是常态）。

**物理意义**：这是真正的"世界"。所有物品、方块、物理、inventory 都在这个 Java 进程里演算。Python 侧看到的 obs 完全来源于它。

---

## 4. 阶段 D — Server reset + Env reset（每 trial 1 次）

`main_planning.py:630-644`：

```python
ServerAPI._reset(cfg["server"])   # GET http://127.0.0.1:9000/reset
obs = env.reset()                 # Minecraft respawn 到新世界
```

### 4.1 Server reset 的作用

`app.py:/reset` → `AgentFactory.reset()`：不卸载模型权重，只是把 `agent` 实例内部的短期状态（STEVE‑1 recurrent state、上一次动作缓存、GUI 状态机）清零。

### 4.2 Env reset 的输出结构

返回的 `obs` 是一个 dict：

```python
{
  "pov":          np.ndarray[360, 640, 3] uint8,   # 第一人称 RGB
  "inventory":    {},                              # 开局空
  "equipped_items":{"mainhand":{"type":"air", "damage":0, "maxDamage":0}},
  "location_stats":{"xpos":..., "ypos":64.0, "zpos":..., "biome_id":..., ...},
  "life_stats":   {"life":20.0, "food":20.0, "saturation":5.0},
  "use_item":     {...},
  ...
}
```

随后 `wrapper.py` 会把 YAML 里的若干 `/gamerule` 命令灌入 Minecraft（关闭日夜循环、永久夜视、保留 inventory、即时重生）——这是实验性 trick，目的是让环境更稳定、避免"天黑才能挖矿"这种干扰。

真实日志：

```
[yellow]In agent_do(), prefix: ours_planning[/yellow]
[yellow]original_final_goal: diamond_axe[/yellow]
normal! env_malmo_logger_path: /app/repo/logs/mc_3352.log exists.
```

---

## 5. 阶段 E — 顶层 waypoint 分解（OracleGraph 决定骨架）

`main_planning.py:320-342` 的主循环第一行：当当前没有 `subgoal` 时，调用 `make_plan(original_final_goal=diamond_axe, ...)`。

### 5.1 OracleGraph.compile 的输入输出

代码 `main_planning.py:106-121` → `relative_graph.py`。

- **输入**：`item="diamond_axe"`, `number=1`, `cur_inventory={}` (或已有物品 dict)。
- **做的事**：按硬编码配方图（"oracle"：配方是先验正确的）反向展开，把 `diamond_axe` 依赖链折平成**有序 waypoint 表**，并把 inventory 中已有物品扣掉，只列出**仍需产出**的物品数量。
- **输出**：一个多行字符串 `pretty_result`，形如：

真实日志第 155-166 行（trial 开始，inventory 为空）：

```
wp_list_str: craft 1 diamond_axe summary:
1. logs: need 9
2. planks: need 27
3. stick: need 8
4. crafting_table: need 1
5. wooden_pickaxe: need 1
6. cobblestone: need 11
7. stone_pickaxe: need 1
8. furnace: need 1
9. iron_ore: need 3
10. iron_ingot: need 3
11. iron_pickaxe: need 1
12. diamond: need 3
13. diamond_axe: need 1
```

**物理意义**：它不是一次性"全部规划好"然后机械执行，而是每次只取**第一行的 waypoint**（当前最紧迫的前置）。这意味着当 inventory 变化时，下一次进入 `make_plan`，第一行就自动变成了新的目标，**相当于隐式的 re-plan**。

### 5.2 每完成一步会触发重新 compile

看日志第 193-211 行（已拿到 9 个 logs 之后再进 make_plan）：

```
In make_plan
wp_list_str: craft 1 diamond_axe summary:
1. planks: need 27
2. stick: need 8
3. crafting_table: need 1
...
```

`logs` 已从表里消失，第一行自动升级为 `planks` —— OracleGraph 把当前 inventory 当输入，永远只列"还缺的"。

---

## 6. 阶段 F — `make_plan` 一步：由 waypoint 变成 subgoal

`main_planning.py:124-170`。

### 6.1 算法流程（伪代码）

```python
def make_plan(goal, inventory, memory, oracle, topK, cfg, logger, obs, ...):
    # 1) 取下一个 waypoint
    wp_list_str = oracle.compile(goal, 1, inventory)
    first_line  = wp_list_str.splitlines()[1]
    wp, wp_num  = parse(first_line)                 # ("planks", 27) 等

    # 2) 问记忆：这个 wp 有没有"成功过的动作短语"？
    is_succeeded, sg_str = memory.is_succeeded_waypoint(wp)
    if is_succeeded:
        subgoal = render_subgoal(sg_str, wp_num)    # 命中：直接复用
        return wp, subgoal, ...

    # 3) 未命中：从记忆里挑 topK 相似成功 wp 做 few-shot、
    #    挑已知失败 subgoal 做 "别再用这条" 的负面提示，
    #    再让 VLM 生成一个 subgoal
    similar = memory.retrieve_similar_succeeded_waypoints(wp, topK)
    failed  = memory.retrieve_failed_subgoals(wp)
    subgoal, lang_act, err = call_planner_with_retry(
        cfg, obs, wp, wp_num, similar, failed, ...
    )
    return wp, subgoal, lang_act, err
```

### 6.2 记忆命中示例（第 210-213 行）

```
waypoint: planks, waypoint_num: 27
is_succeeded: True
After make_plan()
[yellow]Waypoint: planks, Subgoal: {'task': 'craft planks', 'goal': ['planks', 27]}[/yellow]
```

`planks.json` 里有 `{"craft planks": {"success": N, "failure": …}}`，于是直接构造 `{"task": "craft planks", "goal": ["planks", 27]}` —— **零 VLM 调用**。

### 6.3 记忆未命中 → VLM 规划（第 880-926 行，真实 trace）

当走到最后一步 `diamond_axe` 时，因为之前从未成功合成过 `diamond_axe`（它是首次出现的新 waypoint），记忆未命中：

```
In make_plan
waypoint: diamond_axe, waypoint_num: 1
is_succeeded: False
No success experience for waypoint: diamond_axe, so, call planner to generate a plan.
```

`similar_wp_sg_dict`（MiniLM 相似度 top‑3）：
```json
{
  "stone_axe":   "{\"task\": \"craft stone_axe\",   \"goal\": [\"stone_axe\",   1]}",
  "stone_sword": "{\"task\": \"craft stone_sword\", \"goal\": [\"stone_sword\", 1]}",
  "golden_axe":  "{\"task\": \"craft golden_axe\",  \"goal\": [\"golden_axe\",  1]}"
}
```

`failed_sg_list = []`（它之前没有失败记录，全新 waypoint）。

#### 发给 Qwen‑VL 的 **完整 prompt**（日志原样）：

```
For an item name, you need to make a plan using examples.
I will give you examples of which plans are needed to achieve an item.
[Example]
<item name>
stone_axe
<task planning>
{"task": "craft stone_axe", "goal": ["stone_axe", 1]}

[Example]
<item name>
stone_sword
<task planning>
{"task": "craft stone_sword", "goal": ["stone_sword", 1]}

[Example]
<item name>
golden_axe
<task planning>
{"task": "craft golden_axe", "goal": ["golden_axe", 1]}


[Your turn]
Here is <item name>, you MUST output <task planning> in JSON format.
You can make <task planning> by selecting an option from below:
1. {"task": "dig down and mine diamond_axe", "goal": ["diamond_axe", 1]}
2. {"task": "craft diamond_axe", "goal": ["diamond_axe", 1]}
3. {"task": "smelt diamond_axe", "goal": ["diamond_axe", 1]}

<item name>
diamond_axe
<task planning>
```

#### Qwen‑VL 的 **完整输出**：

```
{"task": "craft diamond_axe", "goal": ["diamond_axe", 1]}
```

注意：这个 prompt **不传图像**（`images=None`, `qwen_vl_planning.py:133`），它纯粹是 few‑shot 文本多项选择题——把"craft/mine/smelt"三个动词选一个。这是 XENON 相对 Optimus‑1 的一个关键裁剪：规划不依赖视觉。

**物理意义**：记忆里 3 个相似 waypoint 都走了 `craft ...` 这条路，VLM 理所当然选了"craft diamond_axe"。即便 VLM 在别处会犯糊涂，这里的选项集只有 3 条（mine/craft/smelt），搭配 few‑shot，错误空间极小——这是 XENON 稳定性的主要来源之一。

### 6.4 数据流 Python → app.py → 模型

`ServerAPI.get_decomposed_plan()` 组装一个 HTTP POST：

**Request body** (字段见 `server/api/request.py`):
```python
{
  "type": "decomposed_plan",
  "waypoint": "diamond_axe",
  "similar_wp_sg_dict": {...},
  "failed_sg_list_for_wp": [],
  "rgb_images": [<base64 of obs['pov']>],    # base64 JPEG, 约 50-120 KB
  "hydra_path": "/app/repo/logs/eval/2026-04-18/18-15-21-365247",
  "run_uuid":   "gPYVQFXVidRsvAmMg4MvK8",
  "current_step": 10262
}
```

**Server 侧** (`app.py:93-110`)：
1. `base64_to_image` 把图 base64 解码成 JPEG 文件落盘到 `hydra_path/<uuid>/imgs/<task>_<step>.jpg`（后续 reflection/debug 需要，保存在同一次实验的 Hydra 目录下）。
2. `agent.decomposed_plan(waypoint, image_path_latest, similar_dict, failed_list)`
   → `PlanningModel.decomposed_plan`（`qwen_vl_planning.py:126-198`）
   → 前面那段 few‑shot prompt 交给 `self._inference(prompt, None)`（image 传 `None`，纯文本推理）
   → Qwen‑VL 以 `max_new_tokens=256` 贪婪解码
   → 拿到裸字符串 `{"task": "craft diamond_axe", "goal": ["diamond_axe", 1]}`
3. `MCResponse(response=plans, message=prompt)` 回 JSON。

**Client 侧**后处理（`main_planning.py:85, util/prompt.py:20`）：
`render_subgoal(sg_str, wp_num=1)` 把 `"1"` 覆盖成真实需要量（合成 axe 需要 1 个），然后把 task/goal 字段取出来，变成：

```python
subgoal = {"task": "craft diamond_axe", "goal": ["diamond_axe", 1]}
language_action_str = "craft diamond_axe"
```

---

## 7. 阶段 G — 底层执行：craft/smelt 走 Helper，mine/chop 走 STEVE‑1

`main_planning.py:361-516` 根据 `subgoal["task"]` 的首动词分流。

### 7.1 分流规则（`op = current_sg_prompt.split(" ")[0]`）

| 首动词 | 执行者 | 物理意义 |
|---|---|---|
| `craft`, `smelt`, `equip`, 含 "smelt" 或 "create" | `NewHelper.step()` —— **脚本化 GUI** | 打开背包→把材料拖到 crafting grid→拿出产物。不走神经网络。 |
| `chop`, `dig`, `mine`, `find`, `explore`, … | `STEVE‑1` —— **策略网络** | 每步把当前 POV 图像和 prompt 喂模型，吐鼠标键盘动作。 |

### 7.2 craft/smelt 分支（Helper）

`main_planning.py:368-440`。典型流程（以 `craft planks` 为例，日志第 215-232 行）：

```
Start crafting
target: planks
target_num: 27
is_tag: True

self._craft_plank is called
In crafting_once()
target: oak_planks
iter_num: 7                # 合成次数；一次出 4 个，27 个要 7 次
target_num: 28
recipe_info: {"type":"minecraft:crafting_shapeless", "group":"planks",
              "ingredients":[{"tag":"minecraft:oak_logs"}],
              "result":{"item":"minecraft:oak_planks", "count":4}}

[Current Inventory: {}, ...]
[Current Inventory: {'oak_log': 2}, ...]
[Current Inventory: {'oak_log': 2, 'oak_planks': 28}, ...]
[green]craft planks Success[/green]!
```

Helper 干的事（物理意义）：
1. 从 `cur_inventory` 推出可用的 tag 匹配（比如 `minecraft:oak_logs` 有 `oak_log`）。
2. 发送 MineRL GUI 动作序列：按 E 打开背包 → 鼠标拖动 `oak_log` 到 `crafting grid` → 鼠标点击 result slot → 关闭背包。
3. 每执行完一次就读 `env.inventory`，看看 `oak_planks` 有没有加 4，验证结果。

**所有 craft/smelt 都是这种"读配方 + 脚本化点鼠标"，完全绕开视觉和策略网络**——这是 XENON 在 crafting 子目标上成功率几乎 100% 的原因。

### 7.3 chop/mine 分支（STEVE‑1）

`main_planning.py:441-517`。流程：

```python
while not subgoal_done and not game_over:
    action = ServerAPI.get_action(
        cfg.server, obs, current_sg_prompt, step=env.num_steps,
        hydra_path=..., run_uuid=...
    )    # → HTTP POST type="action"
    obs, reward, game_over, info = env.step(action, current_sg_target)

    if env.inventory_change():
        new_item = env.inventory_change_what()
        if check_waypoint_item_obtained(new_item, waypoint):
            step_waypoint_obtained = env.num_steps

    if env.num_steps - step_waypoint_obtained >= 1200:
        reasoning_dict, ... = call_reasoning_with_retry(...)   # 见 §8
        if reasoning_dict["need_intervention"]:
            current_sg_prompt = reasoning_dict["task"]         # 改 prompt

    if env.current_subgoal_finish:
        break
```

#### 7.3.1 HTTP `type="action"` 的数据流

**Request**:
```python
{
  "type": "action",
  "task_or_instruction": "chop a tree",           # 或 "dig down and mine cobblestone"
  "rgb_images": [<base64 obs['pov']>],
  "current_step": 0,
  "hydra_path": "/app/repo/logs/eval/.../",
  "run_uuid": "gPYVQFXVidRsvAmMg4MvK8"
}
```

**Server 侧** (`app.py:170-178`)：`agent.action(prompt, rgb_obs)` 做两件事：
1. 如果是**新 prompt**（和上一步不同），用 MineCLIP encode 一下文本，拿到 512 维 goal embedding，缓存起来。
2. 把 `obs['pov']` 张量 + MineCLIP embedding 喂给 STEVE‑1 （VPT backbone + prior），输出 `MineRL action dict`。

**Response** (`MCResponse.response`)：形如：
```python
{
  "ESC":0, "attack":1, "back":0, "camera":[-3.2, 5.7], "drop":0,
  "forward":1, "hotbar.1":0, "hotbar.2":0, ..., "hotbar.9":0,
  "inventory":0, "jump":0, "left":0, "pickItem":0, "right":0,
  "sneak":0, "sprint":1, "swapHands":0, "use":0
}
```
每个键对应键盘/鼠标一个通道，数值 0/1 或 `camera=[pitch_delta, yaw_delta]`。

**Env 侧** (`env.step(action)`)：把 action 通过 Malmo 发给 Java 进程，Minecraft tick 一次（~50 ms 游戏时间），返回新的 obs。

#### 7.3.2 "inventory 变化即成功"的判定

`env.inventory_change_what()` 比较两步前后的 inventory dict，返回新增物品。`check_waypoint_item_obtained`（`main_planning.py:223-238`）用宽松匹配（"log" in waypoint and "log" in new_item）判断是否拿到目标。

**物理意义**：STEVE-1 本身不"知道"自己该结束，是 Python 侧在每一 env step 之后查 inventory，发现 "oak_log" 从 8 变到 9 → 计数 +1；到达 waypoint_num=9 时 `env.current_subgoal_finish` 变 True，主循环跳出 inner loop。

---

## 8. 阶段 H — 运行中的视觉 reasoning 干预（可选）

当 STEVE‑1 在一个 waypoint 上超过 `MINUTE=1200` 步都没新增目标物品时，触发 `call_reasoning_with_retry`。

### 8.1 两阶段 VLM 调用

**第一次调用 — 看图说话**（`description_prompt`）：
```
Given a Minecraft game image, describe nearby Minecraft objects, like tree, grass, cobblestone, etc.
[Example]
"There is a large tree with dark green leaves surrounding the area."
"The image shows a dark, cave-like environment in Minecraft. The player is digging downwards. ..."
[Your turn]
Describe the given image, simply and clearly like the examples.
```

输入：当前 `obs['pov']`；输出：一句自然语言描述，如 `"The image shows a dark, narrow tunnel of stone blocks."`

**第二次调用 — 要不要干预**（`context_aware_reasoning_prompt`）：
```
<task>: dig down and mine cobblestone
<visual_description>: The image shows a dark, narrow tunnel of stone blocks. The player is digging downwards.
<goal>: cobblestone
<reasoning>:
```

输出（JSON）：
```json
{
  "need_intervention": false,
  "thoughts": "The player is already digging down and is likely to find cobblestone.",
  "task": ""
}
```

如果 `need_intervention=true`，把 STEVE‑1 的 prompt 换成新的 `task`（如 `"explore to find trees"`）。

### 8.2 真实 trace 里发生的情况

第 457-477 行显示 cobblestone 中段触发了 reasoning，但 Qwen‑VL 推理超过 60 s 超时（`Read timed out`）×3 重试均失败 → 但**并未终止**，因为这条路径只是"optional intervention"；同时在这 3 次 60 s 空转里 STEVE‑1 实际上仍在每步继续工作，随后 inventory 自然增加，任务照常完成：

```
[green]dig down and mine cobblestone Success :smile: [/green]!
```

**物理意义**：reasoning 是"锦上添花"不是"刚需"。它能把"离树太远 → 改为 explore to find trees"这种情况救回来，但 STEVE‑1 在大多数常见场景（dig cobblestone 的 underground 场景、森林里 chop tree）都能靠自己完成。

---

## 9. 阶段 I — 13 个 waypoint 的真实 I/O 逐步还原

下面逐步贴出 trace 中每个 waypoint 的关键输入/输出（inventory 用缩写），来源全部是 `main_planning.log` 实际行。数据用 **"inventory 变化 → 使用的子系统"** 的格式。

> 列标说明：
> - 步(step)：MineRL env 的 tick 数（`env.num_steps`）；1 min ≈ 1200 步
> - 执行者：Helper=脚本化 GUI；STEVE‑1=策略网络
> - 记忆：命中=直接用历史成功动作；VLM=调用 Qwen‑VL 规划

### 9.1 Waypoint 01: `logs` (need 9)

| 字段 | 值 |
|---|---|
| OracleGraph 第一行 | `1. logs: need 9` |
| 记忆 | **命中**（`logs.json` 中 `"chop a tree": {"success": 54, "failure": 3}`） |
| Subgoal | `{"task": "chop a tree", "goal": ["logs", 9]}` |
| 执行者 | **STEVE‑1** |
| Prompt（重复发给 STEVE‑1） | `"chop a tree"` |
| 初始 inventory | `{}` |
| 过程（trace 片段） | `oak_log: 1 @ step 81 → 2 @ 157 → 3 @ 233 → ... → 9 @ step 797` |
| 结束 inventory | `{"oak_log": 9}` |
| 耗时 | 793 steps (~ 40 s game-time) |

### 9.2 Waypoint 02: `planks` (need 27)

| 字段 | 值 |
|---|---|
| OracleGraph 第一行 | `1. planks: need 27`（logs 已消失） |
| 记忆 | **命中** (`craft planks`, success=N) |
| Subgoal | `{"task": "craft planks", "goal": ["planks", 27]}` |
| 执行者 | **NewHelper** |
| Recipe 读回 | `crafting_shapeless, ingredients=[tag:oak_logs], result.count=4` |
| 迭代次数 | iter_num=7（7×4=28 ≥ 27） |
| 过程（trace） | `{} → {oak_log:2} → {oak_log:2, oak_planks:28}`（一次性出 28） |
| 耗时 | 172 steps |

### 9.3 Waypoint 03: `stick` (need 8)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "craft stick", "goal": ["stick", 8]}` |
| Recipe | `pattern=['#','#'], key:{#: tag:planks}, result.count=4` |
| 执行者 | NewHelper |
| 过程 | 第 1 次尝试出错（`error_msg: fail for unkown reason`）→ 重试 1 次 → 成功，inventory 变为 `{stick: 8}` |
| 耗时 | 219 steps |

**注意**：即使 Helper 脚本化了，MineRL 偶尔会因 timing 原因失败。主循环检测到失败会重新调用一次同样的 subgoal，第二次成功。

### 9.4 Waypoint 04: `crafting_table` (need 1)

| 字段 | 值 |
|---|---|
| 记忆 | 命中（`crafting_table.json`: `craft crafting_table, success=49, failure=2`） |
| Subgoal | `{"task": "craft crafting_table", "goal": ["crafting_table", 1]}` |
| Recipe | `pattern=['##','##'], key:{#: tag:planks}`，消耗 4 个 planks |
| 过程 | `oak_planks: 24→20`（消耗 4）, `crafting_table: 0→1` |
| 耗时 | 112 steps |

### 9.5 Waypoint 05: `wooden_pickaxe` (need 1)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "craft wooden_pickaxe", "goal": ["wooden_pickaxe", 1]}` |
| Recipe | `pattern=['XXX',' # ',' # '], key:{#:stick, X:tag:planks}`, 消耗 3 planks + 2 sticks |
| 过程 | craft 完成后 Helper 自动 equip（`main_planning.py:387-395`）：`Current Equipment: wooden_pickaxe` |
| 耗时 | 321 steps |

**物理意义**：`wooden_pickaxe` waypoint 有个特殊后处理：`if "pickaxe" in waypoint` 时 Helper 会自动调用 `equip wooden_pickaxe`，保证下一步 mine 时主手里握着镐子（否则挖 stone 会变成 dirt 飞舞，挖不下来）。

### 9.6 Waypoint 06: `cobblestone` (need 11)

| 字段 | 值 |
|---|---|
| 记忆 | 命中（`cobblestone.json`: `dig down and mine cobblestone, success=40, failure=1`） |
| Subgoal | `{"task": "dig down and mine cobblestone", "goal": ["cobblestone", 11]}` |
| 执行者 | **STEVE‑1** |
| 过程 | 从 ypos=64 向下挖，依次拿到 `dirt×4 → cobblestone×N → coal×2 → andesite×5 → granite×3 → ...`；到 cobblestone=11 时 env 判定完成 |
| 副产品 | 顺带捡到 `coal:4, andesite:5, granite:3`（都是自然矿物掉落） |
| 中途 | 在第 4732 步触发过一次 context_aware_reasoning（超时失败），但不影响继续进行 |
| 耗时 | **3190 steps**（最重的采集任务） |

**物理意义**：这是 STEVE‑1 的主场之一。prompt 固定为 `"dig down and mine cobblestone"`，MineCLIP encode 出一个 goal embedding，STEVE‑1 就一路向下挖。主循环只在每个 tick 读 inventory 看 cobblestone 够没够。

### 9.7 Waypoint 07: `stone_pickaxe` (need 1)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "craft stone_pickaxe", "goal": ["stone_pickaxe", 1]}` |
| Recipe | `key:{#:stick, X:cobblestone}`，3 cobblestone + 2 sticks |
| 过程 | 自动 equip → `Current Equipment: stone_pickaxe` |
| 耗时 | 319 steps |

### 9.8 Waypoint 08: `furnace` (need 1)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "craft furnace", "goal": ["furnace", 1]}` |
| Recipe | `['###','# #','###']`, 需 8 cobblestone |
| 执行者 | Helper |
| 耗时 | 352 steps |

### 9.9 Waypoint 09: `iron_ore` (need 3)

| 字段 | 值 |
|---|---|
| 记忆 | 命中（`iron_ore.json`: `dig down and mine iron_ore, success=22, failure=6`） |
| Subgoal | `{"task": "dig down and mine iron_ore", "goal": ["iron_ore", 3]}` |
| 执行者 | STEVE‑1 |
| 过程 | 继续下挖到 y<25；代码中有个保护逻辑：如果目标还是 iron_ore 且 ypos 已经低于 25 则 "Return to ground"（防止卡在基岩附近）。trace 里确实触发过一次 `Return to ground...` |
| 耗时 | 1949 steps |

### 9.10 Waypoint 10: `iron_ingot` (need 3)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "smelt iron_ingot", "goal": ["iron_ingot", 3]}` |
| 执行者 | **Helper**（smelt 也走脚本化）：把 iron_ore 放 furnace 的上槽、coal 放下槽、取 ingot |
| 过程 | inventory: `iron_ore:3, coal:4 → iron_ingot:3, coal:1`（烧 3 块耗 3 煤，留 1） |
| 耗时 | 986 steps |

### 9.11 Waypoint 11: `iron_pickaxe` (need 1)

| 字段 | 值 |
|---|---|
| 记忆 | 命中 |
| Subgoal | `{"task": "craft iron_pickaxe", "goal": ["iron_pickaxe", 1]}` |
| Recipe | `X=iron_ingot, #=stick`，3 ingot + 2 sticks |
| 过程 | 自动 equip → `Current Equipment: iron_pickaxe` |
| 耗时 | 347 steps |

**物理意义**：只有 iron_pickaxe 才能挖 diamond，所以这一步的 auto-equip 决定了下一步能不能成功。

### 9.12 Waypoint 12: `diamond` (need 3)

| 字段 | 值 |
|---|---|
| 记忆 | 命中（`diamond.json`: `dig down and mine diamond, success=14, failure=1`） |
| Subgoal | `{"task": "dig down and mine diamond", "goal": ["diamond", 3]}` |
| 执行者 | STEVE‑1 |
| 过程（trace 第 860-863 行） | 在 ypos=10 附近发现矿脉；`diamond:0 → 1 @ step 10203 → 2 @ 10225 → 3 @ 10246`（三颗钻石几乎连续冒出） |
| 副产品 | 拿到 `redstone:25, gold_ore:2`（这些在钻石层顺带挖到） |
| 耗时 | 1225 steps |

### 9.13 Waypoint 13: `diamond_axe` (need 1)    ← **首次未命中**

（流程见 §6.3-6.4，此处补充执行过程）

| 字段 | 值 |
|---|---|
| 记忆 | **未命中** → VLM 规划 |
| Qwen‑VL 输出 | `{"task": "craft diamond_axe", "goal": ["diamond_axe", 1]}` |
| Recipe | `pattern=['XX','X#',' #'], key:{#:stick, X:diamond}`, 消耗 3 diamond + 2 sticks |
| 执行者 | Helper |
| 过程（trace 第 945-948 行） | `diamond:3→0, stick:2→0, diamond_axe:0→1` |
| 耗时 | 296 steps |

至此 `env.check_original_goal_finish(['diamond_axe', 1])` 返回 True，主循环跳出：

```
[green]Original Goal: diamond_axe is achieved![/green]
```

---

## 10. 阶段 J — 成功判定、记忆写回、结果落盘

### 10.1 成功判定（`env_wrapper.check_original_goal_finish`）

```python
def check_original_goal_finish(self, goal):
    item, num = goal                      # ["diamond_axe", 1]
    inv = self.env.get_inventory()
    return inv.get(item, 0) >= num
```

**物理意义**：只认 inventory 数量，不看过程。子目标顺序对不对、挖了多少 dirt、走了多少路全部不计。

### 10.2 记忆写回（每成功/失败一个 waypoint）

`main_planning.py:525-531`：

```python
waypoint_success = env.check_waypoint_finish([waypoint, 1])
action_memory.save_success_failure(waypoint, language_action_str, is_success=waypoint_success)
```

进到 `decomposed_memory._save_success_failure`：

- 以 flock 排他锁打开 `src/optimus1/memories/ours_planning/v1/waypoint_to_sg/<wp>.json`
- 对应 `action_str` 的 `success += 1` 或 `failure += 1`
- 如果是该 waypoint 第一次成功，顺带重新计算 MiniLM embedding 存到 `<wp>.pt`，并把 `wp` 加入 `self.succeeded_waypoints`

**本次 trace 写回后**，`diamond_axe.json` 从不存在变为：
```json
{"action": {"craft diamond_axe": {"success": 1, "failure": 0}}}
```
下一次再遇到 craft diamond_axe 任务就**不会再调用 VLM**了。

### 10.3 结果落盘（`main_planning.py:697-762`）

三个地方各写一份：

1. `<hydra_path>/ours_planning_craft_a_diamond_axe_502_success_forest_gPYV.json`
2. `exp_results/v1/ours_planning_craft_a_diamond_axe_502_success_forest_gPYV.json`
3. `videos/v1/Craft_a_diamond_axe/forest/success/2026_04_18_18_35_12_craft_diamond_axe_gPYVQFXVidRsvAmMg4MvK8.mp4`

JSON 关键字段（真实）：
```json
{
  "run_uuid": "gPYVQFXVidRsvAmMg4MvK8",
  "success":  true,
  "steps":    10281,
  "minutes":  8.57,
  "completed_subgoals": [
    {"task":"chop a tree",               "goal":["logs",9]},
    {"task":"craft planks",              "goal":["planks",27]},
    {"task":"craft stick",               "goal":["stick",8]},
    {"task":"craft crafting_table",      "goal":["crafting_table",1]},
    {"task":"craft wooden_pickaxe",      "goal":["wooden_pickaxe",1]},
    {"task":"dig down and mine cobblestone","goal":["cobblestone",11]},
    {"task":"craft stone_pickaxe",       "goal":["stone_pickaxe",1]},
    {"task":"craft furnace",             "goal":["furnace",1]},
    {"task":"dig down and mine iron_ore","goal":["iron_ore",3]},
    {"task":"smelt iron_ingot",          "goal":["iron_ingot",3]},
    {"task":"craft iron_pickaxe",        "goal":["iron_pickaxe",1]},
    {"task":"dig down and mine diamond", "goal":["diamond",3]},
    {"task":"craft diamond_axe",         "goal":["diamond_axe",1]}
  ],
  "failed_subgoals": [],
  "failed_waypoints": [],
  "metrics": { "chop a tree_0":{"StepMonitor":793}, ... "craft diamond_axe_12":{"StepMonitor":296} }
}
```

各子目标步数累加 = 793+172+219+112+321+3190+319+352+1949+986+347+1225+296 = **10281**，与外层 `steps` 相符。

---

## 11. diamond_pickaxe 与 diamond_axe 的唯一差异

从 `src/optimus1/conf/benchmark/diamond.yaml`：

```yaml
- {id: 1, type: craft, instruction: Craft a diamond pickaxe, goal: diamond_pickaxe}
- {id: 2, type: craft, instruction: Craft a diamond axe,     goal: diamond_axe}
```

1. **前 12 个 waypoint 一模一样**（logs → planks → stick → crafting_table → wooden_pickaxe → cobblestone → stone_pickaxe → furnace → iron_ore → iron_ingot → iron_pickaxe → diamond），因为 OracleGraph 的反向展开只看最终物品的合成配方树，而 diamond_axe/diamond_pickaxe 的前置条件相同（都需要 3 diamond + 2 stick）。
2. **第 13 步 subgoal 变为** `{"task":"craft diamond_pickaxe","goal":["diamond_pickaxe",1]}`。
3. **Recipe 不同**（只是 pattern 不同，原材料相同）：
   - diamond_axe: `['XX','X#',' #']` (L 形)
   - diamond_pickaxe: `['XXX',' # ',' # ']` (十字形)
   - 都是 3 diamond + 2 stick。
4. VLM 对 diamond_pickaxe 的 few‑shot 示例会换成 `stone_pickaxe / iron_pickaxe / golden_pickaxe`（MiniLM 相似度 top-3），但最终输出同样是 `{"task": "craft diamond_pickaxe", …}`。
5. 成功落盘的文件名将变为 `ours_planning_craft_a_diamond_pickaxe_501_success_forest_<uuid>.json`。

换言之：把 §9.13 里所有 `diamond_axe` 字样替换为 `diamond_pickaxe` 即为 diamond_pickaxe 任务的 trace；其他 12 步完全相同。

---

## 12. 一张图的"数据流总览"

```
                      ┌──────────────────────────────────────────────────────────┐
                      │  main_planning.py — 控制主循环 (Process B)                │
                      │                                                          │
  task="Craft a       │   while True:                                            │
  diamond pickaxe"    │     if subgoal is None:                                  │
  ────────────────▶   │       wp_list = OracleGraph.compile(                     │
                      │          goal, 1, inventory)    # inventory→wp list      │
                      │       wp, wp_num = parse(wp_list[0])                     │
                      │       # ── make_plan ─────────────────────────────       │
                      │       if DecomposedMemory.is_succeeded(wp):              │
                      │          subgoal = cached_action(wp)                     │
                      │       else:                                              │
                      │          sim = memory.retrieve_similar(wp, topK=3)       │
                      │          fail = memory.retrieve_failed(wp)               │
                      │          ┌──────────────────────────────────────┐        │
                      │          │POST /chat type=decomposed_plan        ├─────▶ │───┐
                      │          │  waypoint, sim, fail, pov_base64      │        │   │
                      │          └──────────────────────────────────────┘        │   │
                      │                          ▲                               │   │
                      │                          │ JSON {task, goal}             │   │
                      │                          │                               │   │
                      │     op = subgoal["task"].split()[0]                      │   │
                      │     if op in {"craft","smelt","equip"}:                  │   │
                      │        NewHelper.step(subgoal)  # scripted GUI ←┐        │   │
                      │     else:                                      │        │   │
                      │        while not subgoal_done:                 │        │   │
                      │          ┌──────────────────────────────┐      │        │   │
                      │          │POST /chat type=action        ├─────▶│───────▶│───┤
                      │          │  prompt, pov_base64           │      │        │   │
                      │          └──────────────────────────────┘      │        │   │
                      │                          ▲                     │        │   │
                      │                          │ MineRL action dict  │        │   │
                      │                          │                     │        │   │
                      │          obs = env.step(action)  ◀──────────── │────────┼───┤
                      │          if env.current_subgoal_finish: break  │        │   │
                      │                                                │        │   │
                      │     memory.save_success_failure(wp, lang_act,  │        │   │
                      │                                  is_success)   │        │   │
                      │     subgoal = None                             │        │   │
                      │   if env.check_original_goal_finish(goal):     │        │   │
                      │     save_json + save_video + exit              │        │   │
                      └────────────────────────────────────────────────│────────┘   │
                                                                       │            │
     ┌────────────────────────────────────────────────────────────────┘            │
     │                                                                             │
     │  app.py — FastAPI server (Process A)                                        │
     │  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
     │  │ PlanningModel        │  │ STEVE-1 Controller    │  │ NewHelper (script) │ │
     │  │ Qwen2.5-VL-7B bf16   │  │ VPT + MineCLIP prior  │  │ GUI click sequence │ │
     │  │ → JSON {task,goal}   │  │ → action dict         │  │ → action sequence  │ │
     │  │ (few-shot text only) │  │ (text+image cond.)    │  │ (uses recipe_info) │ │
     │  └──────────────────────┘  └──────────────────────┘  └────────────────────┘ │
     │       GPU 0: ~15.6 GB          GPU 0: ~200 MB            CPU, negligible    │
     └─────────────────────────────────────────────────────────────────────────────┘
                            │
                            │ MineRL gym API
                            ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │  MCP-Reborn — Minecraft Java server (Process C, port 12352)                 │
     │  真实世界：block tick、物理、mob、inventory、GUI、世界生成                  │
     │  obs = {pov: np.uint8[360,640,3], inventory: dict, location: ..., ...}     │
     └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录 A — 本 trace 的关键数字

| 指标 | 值 |
|---|---|
| 任务 | Craft a diamond axe（与 diamond_pickaxe 流程同构） |
| 总步数 | 10281 (≈ 8.57 min 游戏内) |
| Waypoint 数 | 13 |
| 记忆命中 | 12 / 13 (仅 diamond_axe 未命中需 VLM) |
| VLM `decomposed_plan` 调用次数 | **1** 次 |
| VLM `context_aware_reasoning` 调用 | 3 次尝试（全超时，但不影响成功） |
| STEVE‑1 action 请求 | ≈ 7200+ 次 (mining 类 waypoint 累计) |
| Helper 调用 | 7 次 (planks / stick / crafting_table / wooden_pickaxe / stone_pickaxe / furnace / iron_pickaxe / iron_ingot / diamond_axe) |
| 视频保存 | `videos/v1/Craft_a_diamond_axe/forest/success/2026_04_18_18_35_12_craft_diamond_axe_gPYVQFXVidRsvAmMg4MvK8.mp4` |

## 附录 B — 若要亲自复现 diamond_pickaxe 的命令

```bash
# 1) 保证 app.py 在跑（已跑则略过）
docker exec -d xenon_run bash -c '
  cd /app/repo && export HF_HOME=/app/LLM &&
  export QWEN_DEVICE_ID=0 &&
  xvfb-run -a python app.py --port 9000 \
    --plan_model Qwen/Qwen2.5-VL-7B-Instruct \
    > /tmp/app_server.log 2>&1'
sleep 30   # 等 Qwen 加载完毕，nvidia-smi 应看到 GPU0 显存上升到 ~16 GB

# 2) 跑 diamond_pickaxe（evaluate=[1]）
docker exec xenon_run bash -c '
  cd /app/repo && export HF_HOME=/app/LLM &&
  xvfb-run -a python -m optimus1.main_planning \
    server.port=9000 env.times=1 benchmark=diamond \
    evaluate="[1]" prefix="ours_planning" \
    exp_num=999 seed=0 world_seed=0'

# 3) 结果
ls -la exp_results/v1/ours_planning_craft_a_diamond_pickaxe_999_*.json
python analyze_results.py --detailed
```

---

**文档结束。** 如果需要更细粒度（比如某个 STEVE-1 单步 action 的完整 JSON，或 Helper 脚本里"打开背包→拖格子"的具体 GUI 事件序列），告诉我要哪个片段，我可以再从 `mc_3352.log`/`logs/eval/.../*.log` 里摘抄对应时间段。
