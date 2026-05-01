#!/usr/bin/env python3
"""Export the decisioner training dataset from cases.json.

Inputs (no joins, only the case library):
    src/optimus1/memories/ours_planning/v1/case_memory/cases.json

Outputs:
    data/decisioner/rads_v1.jsonl          - one JSON per valid case
    data/decisioner/rads_v1_summary.json   - dataset statistics

Filtering rules (must match docs/DECISIONER_V3_PROPOSAL_2026-05-01.md):
    drop run_uuid == 'legacy'
    drop outcome.status in {pending, excluded_infra, crash_RuntimeError}
    keep outcome.success in {True, False} only

Derived field:
    _position_in_run: integer index of this case among same-run cases,
                      ordered by parsing the numeric position in case 'id'.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_CASES_PATH = "src/optimus1/memories/ours_planning/v1/case_memory/cases.json"
DEFAULT_OUTPUT_DIR = "data/decisioner"

DROP_OUTCOMES = {"pending", "excluded_infra", "crash_RuntimeError"}


def _load_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as fp:
        data = json.load(fp)
    return data.get("cases", []) if isinstance(data, dict) else []


def _is_valid_case(case: Dict[str, Any]) -> bool:
    if case.get("run_uuid") == "legacy":
        return False
    outcome = case.get("outcome", {}) or {}
    if outcome.get("status") in DROP_OUTCOMES:
        return False
    if outcome.get("success") not in (True, False):
        return False
    return True


def _parse_position_in_run(case: Dict[str, Any]) -> int:
    """Case id format is 'run_uuid:000035:1777273171279'. Use middle field."""
    cid = str(case.get("id", ""))
    parts = cid.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return 0
    return 0


def _slim(case: Dict[str, Any], position_in_run: int) -> Dict[str, Any]:
    """Return a minimal case payload for training; keep schema stable."""
    state = case.get("state_snapshot", {}) or {}
    location = state.get("location_stats", {}) or {}
    outcome = case.get("outcome", {}) or {}
    return {
        "case_id": case.get("id"),
        "run_uuid": case.get("run_uuid"),
        "created_at": case.get("created_at"),
        "_position_in_run": position_in_run,
        "waypoint": case.get("waypoint"),
        "waypoint_num": case.get("waypoint_num", 1),
        "original_final_goal": case.get("original_final_goal"),
        "selected_action": case.get("selected_action"),
        "decision_source": (case.get("decision_trace") or {}).get("source", ""),
        "state_snapshot": {
            "inventory": state.get("inventory", {}) or {},
            "equipment": state.get("equipment", "none"),
            "biome": state.get("biome", ""),
            "location_stats": {
                "ypos": location.get("ypos", 64.0),
                "biome_id": location.get("biome_id", 0),
            },
        },
        "outcome": {
            "status": outcome.get("status"),
            "success": bool(outcome.get("success")),
        },
    }


def _split_by_run(samples: List[Dict[str, Any]], train: float, val: float, seed: int):
    runs = sorted({s["run_uuid"] for s in samples})
    rng = __import__("random").Random(seed)
    rng.shuffle(runs)
    n = len(runs)
    n_train = int(n * train)
    n_val = int(n * val)
    train_runs = set(runs[:n_train])
    val_runs = set(runs[n_train : n_train + n_val])
    test_runs = set(runs[n_train + n_val :])
    out = []
    for s in samples:
        if s["run_uuid"] in train_runs:
            split = "train"
        elif s["run_uuid"] in val_runs:
            split = "val"
        else:
            split = "test"
        s = dict(s)
        s["split"] = split
        out.append(s)
    return out, len(train_runs), len(val_runs), len(test_runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH)
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    raw_total = len(cases)

    # First pass: filter + derive position_in_run
    valid_cases = []
    drop_counts = collections.Counter()
    for c in cases:
        if c.get("run_uuid") == "legacy":
            drop_counts["legacy"] += 1
            continue
        outcome_status = (c.get("outcome", {}) or {}).get("status")
        if outcome_status in DROP_OUTCOMES:
            drop_counts[outcome_status] += 1
            continue
        if (c.get("outcome", {}) or {}).get("success") not in (True, False):
            drop_counts["unknown_success_flag"] += 1
            continue
        valid_cases.append(c)

    samples = [_slim(c, _parse_position_in_run(c)) for c in valid_cases]

    # Group split by run_uuid
    samples_with_split, n_train_runs, n_val_runs, n_test_runs = _split_by_run(
        samples, args.train, args.val, args.seed
    )

    # Stats
    by_split = collections.Counter(s["split"] for s in samples_with_split)
    by_split_label = collections.Counter(
        (s["split"], int(s["outcome"]["success"])) for s in samples_with_split
    )
    by_waypoint = collections.Counter(s["waypoint"] for s in samples_with_split)
    by_action = collections.Counter(s["selected_action"] for s in samples_with_split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "rads_v1.jsonl"
    out_summary = out_dir / "rads_v1_summary.json"

    with open(out_jsonl, "w") as fp:
        for s in samples_with_split:
            fp.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_cases_path": str(args.cases),
        "raw_case_total": raw_total,
        "valid_sample_total": len(samples_with_split),
        "drop_counts": dict(drop_counts),
        "splits": {
            "train": by_split.get("train", 0),
            "val": by_split.get("val", 0),
            "test": by_split.get("test", 0),
        },
        "split_runs": {
            "train": n_train_runs,
            "val": n_val_runs,
            "test": n_test_runs,
        },
        "split_label_counts": {
            f"{split}_label_{label}": cnt for (split, label), cnt in by_split_label.items()
        },
        "waypoint_count": len(by_waypoint),
        "action_count": len(by_action),
        "top_waypoints": by_waypoint.most_common(15),
        "top_actions": by_action.most_common(15),
        "args": {
            "train": args.train,
            "val": args.val,
            "seed": args.seed,
        },
    }
    with open(out_summary, "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    # Print human-readable report
    print(f"Wrote {out_jsonl} ({len(samples_with_split)} samples)")
    print(f"Wrote {out_summary}")
    print()
    print("Drop counts:", dict(drop_counts))
    print("Split sizes:", dict(by_split))
    print("Split runs:", {"train": n_train_runs, "val": n_val_runs, "test": n_test_runs})
    print("Top 5 waypoints:", by_waypoint.most_common(5))


if __name__ == "__main__":
    main()
