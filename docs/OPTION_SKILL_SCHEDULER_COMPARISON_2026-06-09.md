# 环境感知与行动技能化调度对比

日期：2026-06-09

## 结论

当前优化后的方法已经把环境感知与恢复行动封装为 `option:*` 技能，并由独立的 option 决策器进行调度。原本硬规则逻辑不再直接决定“马上执行哪个恢复动作”，而是退化为候选技能的合法性与安全门；最终是否执行、执行哪个技能、以及后续如何评价该技能，由 `OptionDecisioner` 和 option 事件记录负责。

这带来的核心优势是：恢复动作从隐式硬规则变成可调度、可记录、可评估、可迭代的数据单元，同时保留原硬规则对危险动作的边界约束。

## 代码位置

- 技能候选与激活：`src/optimus1/env/options.py`
- option 决策器：`src/optimus1/decisioner/option_selector.py`
- wrapper 接入、事件记录、结果判定：`src/optimus1/env/wrapper.py`
- 配置入口：`src/optimus1/conf/evaluate.yaml`
- 轻量验证：`scripts/verify_option_decisioner.py`

## 原本机制

原本的环境感知与行动主要是硬规则内联在环境 wrapper 中：

1. wrapper 从环境状态中读取位置、血量、空气、背包、目标进展、停滞计数等信号。
2. `_should_*` 类判断函数直接判断是否需要恢复动作。
3. 一旦规则触发，wrapper 直接修改低层控制状态或 action，例如设置 `escape_ticks`、`tunnel_recovery_ticks`、`collect_drop_ticks`、强制 attack/jump/turn 等。
4. 这些恢复动作的优先级由代码顺序、阈值和互斥条件隐式决定。
5. 恢复动作是否有效主要从最终任务是否继续推进间接观察，没有独立的技能调用记录和技能级成败标签。

这种方式的优点是直接、稳定、易于添加单个补丁；问题是策略分散在 wrapper 中，恢复动作之间没有统一调度层，失败经验也很难沉淀为下一次决策依据。

## 当前机制

当前机制把原来的环境动作封装为 option 技能，执行流程如下：

1. `build_option_context(...)` 汇总当前环境上下文：
   - 当前 waypoint 和 prompt
   - 背包占用和相关掉落物
   - 目标进展停滞时间
   - 移动停滞、资源停滞、地表卡住计数
   - 当前位置高度和位移
   - 当前是否属于地表资源、地下资源或一般资源获取

2. `build_env_option_candidates(...)` 根据硬规则生成候选技能：
   - `option:surface_escape`
   - `option:movement_escape`
   - `option:collect_drops`
   - `option:surface_search`
   - `option:tunnel_recovery`
   - `option:surface_turn_around`

3. `OptionDecisioner.select(...)` 对候选技能打分并选择一个执行：
   - 优先使用相同上下文签名下的历史成功率。
   - 没有足够上下文历史时，退回到该 option 的全局历史。
   - 冷启动时使用规则候选的默认分。
   - 低历史分不会硬禁用规则候选，只会体现在 trace reason 中。

4. `activate_env_option(...)` 只激活被选中的技能：
   - 设置对应 tick 控制字段。
   - 重置必要的低层控制状态。
   - 更新 recovery event counter。

5. wrapper 记录 option invocation 和 outcome：
   - 调用前保存环境快照。
   - 技能结束后保存环境快照。
   - 结果写入 `case_memory/option_events.jsonl`。

## 成败判定差异

原本机制基本没有独立的“技能成功/失败”概念，容易把任务最终失败误认为某个恢复动作失败，或者把任务最终成功误认为中途所有恢复动作都有效。

当前机制按 option 使用前后的环境状态变化判定技能结果。以 `option:tunnel_recovery` 为例，成功不等于最终任务成功，而是满足以下任一环境变化：

- 目标矿物 observed/mine delta 增加。
- 水平方向位移达到恢复阈值。
- 垂直方向变化达到恢复阈值。

这样可以把“技能是否帮 agent 脱离当前局部困境”和“整个任务是否最终完成”分开统计。

## 优势

1. 决策边界更清晰

   原本硬规则直接执行动作；现在硬规则只产生候选，调度器决定是否执行以及执行哪个 option。这样恢复动作从散落的 if/else 变成统一的技能调度问题。

2. 可观测性更强

   每次 option 调用都会记录上下文、选择原因、调用前状态、调用后状态和 outcome。后续可以直接分析某个 option 在某类状态下是否有效，而不必只看最终任务成败。

3. 更容易迭代

   新增环境恢复动作时，只需要封装为新的 `EnvOptionCandidate` 和 activation 逻辑，再让决策器纳入调度。无需继续把优先级写死在 wrapper 的动作修改链条里。

4. 保留安全边界

   硬规则没有被完全删除，而是作为候选生成的安全门保留。决策器不会凭空执行一个当前环境不合法的 option，这比完全开放动作空间更稳。

5. 可以累积技能级经验

   `option_events.jsonl` 为后续学习式调度提供数据基础。即使当前 `OptionDecisioner` 还是轻量统计式选择器，也已经具备从历史 outcome 调整 ranking 的能力。

6. 避免任务级标签污染技能评价

   任务失败可能是因为后续合成、熔炼、导航或超时；任务成功也可能掩盖中途无效恢复。当前 state-delta outcome 能更准确地评价单个技能。

7. 更适合做消融实验

   可以分别比较：
   - 只有硬规则候选但不启用决策器。
   - 启用 option 决策器。
   - 不同 option 成功阈值。
   - 不同上下文签名或历史窗口。

## 当前保留的边界

当前优化不是把所有动作都交给一个开放式策略模型。以下部分仍保留原机制：

- 高层任务分解和普通动作选择仍由现有 planner、case memory 和 STEVE-1 路径负责。
- option 候选仍由硬规则生成。
- option 激活后仍通过 wrapper 里已有的低层控制字段执行，例如 `tunnel_recovery_ticks`、`surface_search_ticks`、`collect_drop_ticks`。
- option 决策器目前是轻量统计调度器，不是端到端神经网络策略。

这是一种保守迁移：先把硬规则动作封装成技能并纳入调度，而不是一次性替换整个控制系统。

## 与当前实验环境的关系

当前 V9 重跑使用的是：

- 所有矿物运行时注入倍率为 `1.0`，即 eligible call 下约 `10%` 单次触发概率。
- 生矿账本保留 `(x, y, z)` block cell 去重，因此横向换矿柱后，同一 Y 层可以重新获得一次生矿机会。
- `DELETE_ABNORMAL_ARTIFACTS=1`，异常 retry 前会清理异常结果/视频，避免污染 V9 记录。
- launcher 默认对 `env_step_timeout/crash_*` 普通异常重试 2 次，对 `infra_early_stop` 也重试 2 次；如果失败日志反复出现低层 `craft/smelt` 的 `fail for unkown reason`，也按交互异常处理。
- 正常任务失败不是异常：只要主程序正常写出终态结果，`failed`、`timeout_programmatic`、`timeout_non_programmatic` 这类失败应保留并计入成功率分母。
- `STOP_ON_ABNORMAL_EXHAUSTED=1` 为默认值。异常重试耗尽时停止整组实验，而不是继续生成缺口、换 seed 补齐，或把补齐后的结果解释成真实性能成功率。

这部分属于环境一致性和实验统计控制，不改变 option 技能调度本身。

## 验证方式

已使用轻量验证脚本覆盖以下行为：

- 冷启动 rule-gated option 可以被调度。
- option invocation/outcome 会写入事件文件。
- 历史失败会改变 ranking/trace，但不会硬禁用规则允许的技能。
- option 激活会设置对应控制 tick。
- `(x, y, z)` 生矿账本允许不同矿柱独立去重。

命令：

```bash
python scripts/verify_option_decisioner.py
python -m compileall -q scripts/verify_option_decisioner.py src/optimus1/env/wrapper.py src/optimus1/env/options.py src/optimus1/decisioner/option_selector.py
```
