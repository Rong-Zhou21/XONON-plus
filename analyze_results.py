#!/usr/bin/env python3
"""Analyze XENON experiment results.

Usage:
    python analyze_results.py [--dir exp_results/v1] [--log-dir logs/eval]
"""

import argparse
import json
import os
import glob
from collections import defaultdict


def load_results(result_dir):
    """Load all result JSON files."""
    results = []
    for f in sorted(glob.glob(os.path.join(result_dir, "**/*.json"), recursive=True)):
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                data["_file"] = f
                results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read {f}: {e}")
    return results


def print_summary(results):
    """Print a human-readable summary of all experiment results."""
    if not results:
        print("No results found.")
        return

    print("=" * 80)
    print("XENON Experiment Results Summary")
    print("=" * 80)

    # Group by benchmark
    by_benchmark = defaultdict(list)
    for r in results:
        by_benchmark[r.get("benchmark", "unknown")].append(r)

    for benchmark, runs in sorted(by_benchmark.items()):
        print(f"\n{'─' * 60}")
        print(f"Benchmark: {benchmark.upper()}")
        print(f"{'─' * 60}")

        # Group by task
        by_task = defaultdict(list)
        for r in runs:
            by_task[r.get("task", "unknown")].append(r)

        total_success = 0
        total_runs = 0

        for task, task_runs in sorted(by_task.items()):
            successes = sum(1 for r in task_runs if r.get("success"))
            n = len(task_runs)
            total_success += successes
            total_runs += n
            sr = successes / n if n > 0 else 0

            steps_list = [r.get("steps", 0) for r in task_runs if r.get("steps")]
            avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0

            print(f"\n  Task: {task}")
            print(f"    Success Rate: {successes}/{n} ({sr:.2%})")
            if avg_steps > 0:
                print(f"    Avg Steps: {avg_steps:.0f} ({avg_steps / 1200:.1f} min)")

            for i, r in enumerate(task_runs):
                status_icon = "OK" if r.get("success") else "FAIL"
                detail = r.get("status_detailed", "")
                steps = r.get("steps", 0) or 0
                exp_num = r.get("exp_num", "?")
                wp_failed = r.get("failed_waypoints", [])
                completed = r.get("completed_subgoals", [])

                print(f"    Run #{exp_num}: [{status_icon}] {detail}, "
                      f"steps={steps}, "
                      f"completed_sg={len(completed)}, "
                      f"failed_wp={wp_failed}")

        if total_runs > 0:
            overall_sr = total_success / total_runs
            print(f"\n  Overall: {total_success}/{total_runs} ({overall_sr:.2%})")

    print(f"\n{'=' * 80}")
    print(f"Total experiments: {len(results)}")
    print(f"Total successes: {sum(1 for r in results if r.get('success'))}")
    print(f"Overall SR: {sum(1 for r in results if r.get('success')) / len(results):.2%}")
    print(f"{'=' * 80}")


def print_detailed(results):
    """Print detailed per-run information."""
    print("\n\nDetailed Results:")
    print("=" * 80)
    for r in results:
        print(f"\nFile: {r.get('_file', 'unknown')}")
        print(f"  Task: {r.get('task')}")
        print(f"  Goal: {r.get('goal')}")
        print(f"  Success: {r.get('success')}")
        print(f"  Status: {r.get('status_detailed')}")
        print(f"  Steps: {r.get('steps')} ({(r.get('steps') or 0) / 1200:.1f} min)")
        print(f"  Seed: {r.get('seed')}, Exp#: {r.get('exp_num')}")
        print(f"  Biome: {r.get('biome')}")

        completed = r.get("completed_subgoals", [])
        if completed:
            print(f"  Completed subgoals ({len(completed)}):")
            for sg in completed:
                if isinstance(sg, dict):
                    print(f"    - {sg.get('task', sg)}")
                else:
                    print(f"    - {sg}")

        failed = r.get("failed_subgoals", [])
        if failed:
            print(f"  Failed subgoals ({len(failed)}):")
            for sg in failed:
                if isinstance(sg, dict):
                    print(f"    - {sg.get('task', sg)}")
                else:
                    print(f"    - {sg}")

        wp_failed = r.get("failed_waypoints", [])
        if wp_failed:
            print(f"  Failed waypoints: {wp_failed}")

        metrics = r.get("metrics", {})
        if metrics:
            print(f"  Metrics: {json.dumps(metrics, indent=4)}")


def scan_logs(log_dir):
    """Scan log files for key events."""
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    print("\n\nLog Summary:")
    print("=" * 80)
    log_files = sorted(glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True))

    for lf in log_files:
        try:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except IOError:
            continue

        errors = content.count("ERROR")
        successes = content.count("Success")
        timeouts = content.count("Timeout")
        malmo_errors = content.count("env_malmo_logger_error")

        if errors or successes or timeouts:
            dirname = os.path.basename(os.path.dirname(lf))
            print(f"  {dirname}: {successes} successes, {errors} errors, "
                  f"{timeouts} timeouts, {malmo_errors} malmo_errors")


def main():
    parser = argparse.ArgumentParser(description="Analyze XENON experiment results")
    parser.add_argument("--dir", default="exp_results/v1", help="Result JSON directory")
    parser.add_argument("--log-dir", default="logs/eval", help="Log directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-run info")
    args = parser.parse_args()

    results = load_results(args.dir)
    print_summary(results)

    if args.detailed:
        print_detailed(results)

    scan_logs(args.log_dir)


if __name__ == "__main__":
    main()
