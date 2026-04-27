#!/usr/bin/env python3
# File: final_report.py
"""生成 XENON 7 benchmark 最终评估报告。"""
import json, glob, os
from collections import defaultdict

RESULTS_DIR = "exp_results/v1"
files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))

# 按 benchmark 聚合
by_bench = defaultdict(list)
all_data = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        continue
    bench = d.get("benchmark", "unknown")
    by_bench[bench].append(d)
    all_data.append(d)

print("=" * 80)
print("XENON 7 Benchmarks 复现最终评估报告")
print("=" * 80)
print(f"Planning 模式 | 规划器: Qwen2.5-VL-7B-Instruct | 动作控制器: STEVE-1")
print(f"配置: Oracle 依赖图 + env.times=1 + 各 benchmark yaml 默认 max_minutes")
print(f"结果文件总数: {len(files)}")
print()

bench_order = ["wooden", "stone", "iron", "golden", "diamond", "redstone", "armor"]

# ================== Per-benchmark 汇总 ==================
print("## 各 Benchmark 汇总")
print()
print(f"{'Benchmark':<10} {'运行次数':<8} {'成功':<6} {'失败':<6} {'成功率':<8} {'平均步数(成功)':<14} {'平均时长(成功)':<14}")
print("-" * 80)

total_runs = total_success = 0
for bench in bench_order:
    runs = by_bench.get(bench, [])
    if not runs:
        print(f"{bench:<10} {'0':<8} (未记录)")
        continue
    n = len(runs)
    succ = sum(1 for r in runs if r.get("success"))
    fail = n - succ
    succ_runs = [r for r in runs if r.get("success")]
    avg_steps = sum(r.get("steps", 0) for r in succ_runs) / len(succ_runs) if succ_runs else 0
    avg_mins = sum(r.get("minutes", 0) for r in succ_runs) / len(succ_runs) if succ_runs else 0
    print(f"{bench:<10} {n:<8} {succ:<6} {fail:<6} {succ/n*100:>5.1f}%   {avg_steps:<14.0f} {avg_mins:<14.2f}")
    total_runs += n
    total_success += succ

print("-" * 80)
print(f"{'TOTAL':<10} {total_runs:<8} {total_success:<6} {total_runs-total_success:<6} "
      f"{total_success/max(total_runs,1)*100:>5.1f}%")
print()

# ================== 每任务详情 ==================
print("## 各任务详情（本次新跑 + 历史累计）")
print()
for bench in bench_order:
    runs = by_bench.get(bench, [])
    if not runs:
        continue
    print(f"### {bench}")
    runs_sorted = sorted(runs, key=lambda r: (r.get("task", ""), r.get("exp_num", 0)))
    for r in runs_sorted:
        task = r.get("task", "?")
        succ = "✅" if r.get("success") else "❌"
        steps = r.get("steps", "?")
        mins = r.get("minutes", "?")
        exp = r.get("exp_num", "?")
        biome = r.get("biome", "?")
        print(f"  {succ} exp={exp:<4} task={task:<32} steps={steps:<6} min={mins:<6} biome={biome}")
    print()
