# V7 平均步长分析快照

日期：2026-05-10

## 统计口径

数据来自本地结果 JSON 的 `steps` 字段。当前 `v7_dynamic_gold15x_xenonbase` 仍在运行，因此本文只统计已经落盘的结果。

注意：

- 失败样本也会有 `steps`，通常接近超时步数，会显著抬高“全部样本平均步长”。
- 因此本文同时给出 `全部平均`、`成功平均`、`失败平均`。
- 早期 v1-v6 的实验配置与当前 v7 不完全一致，只能作为历史参考，不能直接当作严格对照。

## 当前 V7：动态黄金放矿 1.5x

当前目录：`exp_results/v7/`

已落盘 3 条，均为 `golden_chestplate`：

| exp_num | task | result | steps | status |
| --- | --- | --- | ---: | --- |
| 551200 | golden_chestplate | success | 10296 | success |
| 551201 | golden_chestplate | failed | 21167 | timeout_non_programmatic |
| 551202 | golden_chestplate | failed | 20241 | timeout_non_programmatic |

汇总：

| scope | n | success | fail | avg_all | avg_success | avg_fail | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plus v7 current | 3 | 1 | 2 | 17234.7 | 10296.0 | 20704.0 | 10296 | 21167 |

## Plus 历史版本整体步长

本地可读取且包含 `steps` 字段的结果共 506 条。

| version | n | success | fail | avg_all | avg_success | avg_fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 170 | 70 | 100 | 8944.9 | 5895.6 | 11079.4 |
| v2 | 107 | 57 | 50 | 8294.2 | 5310.5 | 11695.7 |
| v3 | 67 | 49 | 18 | 9735.9 | 5543.9 | 21147.4 |
| v4 | 67 | 55 | 12 | 6993.1 | 4730.5 | 17363.0 |
| v5 | 46 | 34 | 12 | 8256.6 | 6032.1 | 14559.5 |
| v6 | 46 | 41 | 5 | 5295.5 | 4795.0 | 9399.2 |
| v7 | 3 | 1 | 2 | 17234.7 | 10296.0 | 20704.0 |

## 三个 Armor 目标任务历史参考

只筛选当前关注的三个任务：`golden_chestplate`、`golden_leggings`、`diamond_chestplate`。

| task | n | success | fail | avg_all | avg_success | avg_fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| golden_chestplate | 16 | 2 | 14 | 17020.1 | 15212.5 | 17278.4 |
| golden_leggings | 12 | 0 | 12 | 19984.8 | - | 19984.8 |
| diamond_chestplate | 14 | 2 | 12 | 22173.4 | 7124.0 | 24681.6 |

## 当前结论

1. 当前 v7 刚开始，只完成了 `golden_chestplate` 的前 3 次，样本太少，暂时不能判断该环境配置的真实性能。
2. 当前 v7 已完成样本中，成功样本步长为 10296；两个失败样本都接近 20k+ 步，说明失败主要来自长时间未完成而不是早期崩溃。
3. 从历史 Armor 目标任务看，`golden_leggings` 是最弱项：本地已有记录里 12 次全失败；`diamond_chestplate` 成功样本平均步长较低，但失败样本平均非常高，说明一旦资源链没跑通就会长时间卡住。
4. 严格的 Plus-vs-Main 对照需要等当前 Plus 30 次完成，并等待已挂起的 XENON-main 接力实验跑完后，再按相同口径比较。
