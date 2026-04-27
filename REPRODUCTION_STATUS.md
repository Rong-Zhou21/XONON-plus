# XENON 项目复现进度报告

## 项目概述

**XENON** (Experience-based Knowledge Correction for Robust Planning in Minecraft) 是 ICLR 2026 的论文工作，通过外部经验记忆库 (DecomposedMemory + HypothesizedRecipeGraph) 指导 Minecraft 开放世界智能体的行动规划。

- 规划器：Qwen2.5-VL-7B-Instruct（视觉语言模型）
- 动作控制器：STEVE-1（基于 VPT 的 Minecraft 策略模型）
- 环境：MineRL 1.0.2 + MCP-Reborn (自定义 Minecraft)

---

## 已完成的复现准备

### 1. 环境搭建 ✅

- Docker 镜像：`sjlee1218/xenon:latest`（已本地存在）
- 容器配置：挂载源码、HF 缓存、GPU 设备
- 环境变量：
  - `HF_HOME=/app/LLM`
  - `HF_ENDPOINT=https://hf-mirror.com` (HF 镜像源)

### 2. 关键资源就位 ✅

| 资源 | 大小 | 来源 | 状态 |
|---|---|---|---|
| STEVE-1 checkpoint (VPT + MineCLIP) | ~1.4GB | 从同级目录 `XENON-main（失败）/checkpoints/` 复用 | ✅ |
| MCP-Reborn 已构建 jar | - | 从同级目录复用 (`mcprec-6.13.jar`) | ✅ |
| Qwen2.5-VL-7B-Instruct | ~16GB | HF 缓存 `~/.cache/huggingface` | ✅ |
| sentence-transformers/all-MiniLM-L6-v2 | ~90MB | HF 缓存 | ✅ |

### 3. Python 包安装 ✅

- `minerl==1.0.2`（editable 安装）
- `optimus1==0.1.0`（editable 安装）
- 依赖：torch 2.8.0+cu126, transformers 4.57.6, sentence-transformers 3.4.1 等

---

## 代码修改清单（可复现性 & 正确性）

### 修改 1：跳过已构建的 MCP-Reborn（避免 pip install 时重复构建/清理）

**文件**：`minerl/setup.py`（prep_mcp 函数开头）

```python
def prep_mcp():
    mydir = os.path.abspath(os.path.dirname(__file__))
    # Skip prep_mcp if MCP-Reborn is already built (has shadowJar output)
    prebuilt_jar_dir = os.path.join(mydir, 'minerl', 'MCP-Reborn', 'build', 'libs')
    if os.path.isdir(prebuilt_jar_dir) and any(
        f.endswith('.jar') for f in os.listdir(prebuilt_jar_dir)
    ):
        print(f'[minerl setup] MCP-Reborn already built at {prebuilt_jar_dir}, skipping prep_mcp()')
        try:
            unpack_assets()
        except Exception as e:
            print(f'[minerl setup] unpack_assets failed (may be ok): {e}')
        return
    ...
```

**原因**：原始 `prep_mcp()` 会重新执行 `setup_mcp.sh`，这会删除已有 MCP-Reborn 目录并重新下载/构建，浪费时间且无网络无法成功。

### 修改 2：文件权限修复（Docker root 写入问题）

**文件**：`app.py`, `src/optimus1/main_planning.py`, `src/optimus1/main_exploration.py`, `src/optimus1/memories/decomposed_memory.py`, `src/optimus1/memories/hypothesized_recipe_graph.py`

在每个文件开头加入：

```python
import os
os.umask(0o000)   # 确保 Docker 中 root 创建的文件对宿主机用户 yzb 可读可写
```

**原因**：Docker 容器内以 root 运行（uid=0），创建的文件默认权限 644，宿主机用户 yzb (uid=1000) 无法编辑。设置 umask(0) 使创建的文件默认权限 666 (文件) / 777 (目录)，允许宿主机用户读写。

### 修改 3：Qwen 模型 device 选择更健壮

**文件**：`src/optimus1/models/qwen_vl_planning.py`

```python
def __init__(self, model_path="Qwen/Qwen2.5-VL-7B-Instruct", device_id=0, ...):
    import os as _os
    if _os.environ.get("QWEN_DEVICE_ID") is not None:
        device_id = int(_os.environ["QWEN_DEVICE_ID"])
    if torch.cuda.is_available() and device_id >= torch.cuda.device_count():
        device_id = 0
    self.device = f"cuda:{device_id}"
```

**原因**：原代码硬编码 `device_id=1`，当只有 1 张 GPU 可见时会报 "invalid device ordinal"。现在支持环境变量覆盖和自动回退。

### 修改 4：commands 执行失败容错（reset 鲁棒性）

**文件**：`src/optimus1/env/wrapper.py`

```python
obs = self.env.reset()
if commands:
    for cmd in commands:
        try:
            self.env.execute_cmd(cmd)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"execute_cmd({cmd!r}) failed: {e}; continuing."
            )
return obs
```

**原因**：`/gamerule sendCommandFeedback false` 等命令在某些 MineRL/MCP 版本下会触发 `java.lang.NumberFormatException`，之前会导致整个 reset 失败。现在只记录警告，继续执行（这些命令只是便利性设置）。

### 修改 5：结果分析脚本（便于人工验证）

**新增文件**：`analyze_results.py`

提供 `exp_results/` 目录下所有 JSON 结果的聚合统计：
- 按 benchmark (wooden/stone/iron/golden/diamond/redstone/armor) 分组
- 每个任务的成功率、平均步数、完成子目标数、失败路径点
- 日志扫描（错误/成功/超时统计）

用法：
```bash
python analyze_results.py                         # 默认统计
python analyze_results.py --detailed              # 详细每次运行
python analyze_results.py --dir exp_results/v1    # 指定结果目录
```

---

## 当前运行阻塞：GPU 资源不足 ⚠️

### 现象

运行 `main_planning` 时，`app.py` 服务器加载 Qwen2.5-VL-7B 到 GPU 触发 OOM：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 0 has a total capacity of 23.64 GiB of which 69.56 MiB is free.
  Process 4184951 has 2.18 GiB memory in use.
  Process 433522 has 19.98 GiB memory in use.   <-- 外部进程（非本次任务）
```

### 当前 GPU 实时状态

| GPU | 总容量 | 已用 | 空闲 | 主要占用 |
|---|---|---|---|---|
| 0 (RTX 4090D) | 24 GB | 23.6 GB | 0.6 GB | 外部 PID 4184951 + PID 433522 |
| 1 (RTX 4090D) | 24 GB | 19.7 GB | 4 GB | 外部 PID 4184951 |

Qwen2.5-VL-7B bf16 需约 16GB，4-bit 量化需约 5-6GB。当前两张 GPU 均无法独立承载。

## 2026-04-26 更新：已接入本地 vLLM 并完成最小端到端复现 ✅

### vLLM 模型对齐

用户原本的本地 vLLM 启动命令使用：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct ...
```

该模型是 text-only Qwen2.5，不能处理 XENON 中 `context_aware_reasoning` 的 Minecraft 画面输入。本实验需要视觉语言模型，因此已对齐为：

```bash
Qwen/Qwen2.5-VL-7B-Instruct
```

宿主机成功启动命令：

```bash
conda activate vllm_qwen2_5_vl
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

说明：
- `--host 0.0.0.0`：使 Docker 容器可通过 `http://172.17.0.1:8000/v1` 访问宿主机 vLLM。
- `--max-model-len 16384` 在 RTX 4090D 24GB + `gpu-memory-utilization=0.8` 下 KV cache 不足；已降为 `8192` 并将利用率调到 `0.9`。
- Qwen2.5-VL-7B 本地 HF 缓存约 16GB，已存在于 `~/.cache/huggingface`。

### 代码新增能力

已修改：

- `src/optimus1/models/qwen_vl_planning.py`
- `src/optimus1/server/agent.py`

新增环境变量：

```bash
QWEN_BACKEND=vllm
QWEN_VLLM_BASE_URL=http://172.17.0.1:8000/v1
QWEN_VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

也可以通过 `--plan_model vllm:Qwen/Qwen2.5-VL-7B-Instruct` 显式启用 vLLM 后端。未设置这些变量时，仍保持原来的本地 `transformers.from_pretrained()` 加载路径。

### 成功复现实验

运行命令：

```bash
docker exec xenon_run bash -lc '
  cd /app/repo &&
  export HF_HOME=/app/LLM &&
  export HF_ENDPOINT=https://hf-mirror.com &&
  export HF_HUB_OFFLINE=1 &&
  export QWEN_BACKEND=vllm &&
  export QWEN_VLLM_BASE_URL=http://172.17.0.1:8000/v1 &&
  export QWEN_VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct &&
  xvfb-run -a python -m optimus1.main_planning \
    server.port=9200 \
    env.times=1 \
    env.max_minutes=3 \
    benchmark=wooden \
    evaluate="[5]" \
    prefix="ours_planning" \
    exp_num=902 \
    seed=0 \
    world_seed=5 \
    plan_model="vllm:Qwen/Qwen2.5-VL-7B-Instruct"'
```

结果：

```text
Task: craft_a_crafting_table
Benchmark: wooden
exp_num: 902
success: true
steps: 370
minutes: 0.31
completed_subgoals:
  - chop a tree -> logs
  - craft planks -> planks
  - craft crafting_table -> crafting_table
failed_subgoals: []
```

结果文件：

```text
exp_results/v1/ours_planning_craft_a_crafting_table_902_success_forest_AJpf.json
```

### 建议的解决方案（按优先级）

1. **等待外部进程释放**（推荐）：外部进程（4184951, 433522）不是本次任务的，待其结束后重新运行脚本即可。可用 `watch -n 10 nvidia-smi` 监控。

2. **使用 AWQ 量化模型**（需下载 ~5GB）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-AWQ
   # 然后 --plan_model Qwen/Qwen2.5-VL-7B-Instruct-AWQ
   ```
   AWQ 版本仅需约 6GB 显存。

3. **切换到更小的规划器**：Qwen2-VL-2B-Instruct（~4GB），但需在代码里适配并且可能影响 benchmark 结果的可比性。

---

## 运行实验的完整命令（GPU 充足后）

### 启动 app.py 服务器

```bash
docker exec -d xenon_run bash -c '
  cd /app/repo && 
  export HF_HOME=/app/LLM && 
  export HF_ENDPOINT=https://hf-mirror.com && 
  export CUDA_VISIBLE_DEVICES=0 && 
  xvfb-run -a python app.py --port 9000 \
    --plan_model Qwen/Qwen2.5-VL-7B-Instruct \
    > /tmp/app_server.log 2>&1'
```

### 运行 planning 实验（wooden 最简单任务）

```bash
docker exec xenon_run bash -c '
  cd /app/repo && 
  export HF_HOME=/app/LLM && 
  export HF_ENDPOINT=https://hf-mirror.com && 
  xvfb-run -a python -m optimus1.main_planning \
    server.port=9000 env.times=1 benchmark=wooden \
    evaluate="[0]" prefix="ours_planning" \
    exp_num=1 seed=0 world_seed=0'
```

### 批量运行（参考 `scripts/run_planning_diamond.sh`）

```bash
# 运行所有 wooden 任务
for i in 0 1 2 3 4 5 6; do
  docker exec xenon_run bash -c "... evaluate=[$i] ..."
done
```

### 查看结果

```bash
# 从宿主机（yzb 用户）直接读取
cat exp_results/v1/*.json

# 或用分析脚本
python analyze_results.py --detailed
```

### 关停 app.py 服务器

```bash
docker exec xenon_run python -m optimus1.util.server_api --port 9000
```

---

## 关键目录与文件说明

```
XENON-main/
├── app.py                          # FastAPI 服务：封装 Qwen + STEVE-1 agent
├── analyze_results.py              # 【新增】结果分析脚本
├── REPRODUCTION_STATUS.md          # 【新增】本报告
├── checkpoints/                    # STEVE-1 / VPT / MineCLIP 权重
│   ├── vpt/2x.model
│   ├── steve1/steve1.weights
│   ├── steve1/steve1_prior.pt
│   └── mineclip/attn.pth
├── minerl/                         # MineRL 1.0.2 源码（修补了 setup.py）
│   └── minerl/MCP-Reborn/          # 预构建的 Minecraft Java mod
├── src/optimus1/
│   ├── main_planning.py            # Planning 主入口（修补了 umask）
│   ├── main_exploration.py         # Exploration 主入口
│   ├── conf/
│   │   ├── evaluate.yaml
│   │   └── benchmark/{wooden,stone,iron,golden,diamond,redstone,armor}.yaml
│   ├── memories/
│   │   ├── decomposed_memory.py    # 轨迹/动作经验记忆
│   │   ├── hypothesized_recipe_graph.py  # 学到的物品依赖图（Exploration 模式）
│   │   └── relative_graph.py       # Oracle 依赖图（Planning 模式）
│   ├── models/
│   │   ├── qwen_vl_planning.py     # 【修补 device】
│   │   └── steve1/                 # STEVE-1 动作模型
│   ├── env/
│   │   └── wrapper.py              # 【修补 commands 容错】
│   └── server/agent.py             # Agent 工厂，被 app.py 调用
├── scripts/
│   ├── run_planning_diamond.sh
│   └── run_exploration.sh
├── exp_results/v1/                 # 实验结果 JSON（每次运行一份）
├── logs/                           # Hydra 日志 + Minecraft 进程日志
└── videos/v1/                      # 录制的视频回放
```

---

## 结果的可验证性说明

每次实验产生的 JSON 结果文件（位于 `exp_results/v1/` 和 `logs/eval/{date}/{time}/`）包含：

- `run_uuid`：唯一运行 ID
- `success`: bool，任务是否成功完成
- `steps` / `minutes`：用时
- `completed_subgoals`：成功达成的子目标列表
- `failed_subgoals`：失败的子目标
- `failed_waypoints`：失败的路径点（waypoint）
- `metrics`：SuccessMonitor / StepMonitor 收集的指标
- `video_file`：对应的视频回放路径（可人工查看验证）

**权限修复后**：宿主机 yzb 用户可直接 `cat`/`vim`/`jq` 这些文件，也可以用 `python analyze_results.py` 批量分析。

---

## 下一步

1. 等待 GPU 释放（或切换到 AWQ/2B 版本）
2. 先跑 wooden benchmark (7 个任务 × 1~3 次) 验证整个 pipeline
3. 按 `scripts/run_planning_diamond.sh` 的方式批量跑 diamond benchmark（论文主实验）
4. 用 `analyze_results.py --detailed` 汇总结果，与论文 Table 1 的指标对比
