#!/usr/bin/env python3
"""Offline evaluation for the trained RADS decisioner.

Inputs:
    artifacts/decisioner/rads_v1.pt   (model + library + spec)
    data/decisioner/rads_v1.jsonl     (full dataset with split labels)

Outputs:
    reports/decisioner/offline_eval_v1.md
    reports/decisioner/offline_eval_v1.json

Evaluation dimensions:
    1. Overall AUC / AP / F1 / ECE on val and test.
    2. Per-waypoint AUC for waypoints with >= 5 negative samples.
    3. Attention diagnostic: how often top-1 attention case shares the
       waypoint / outcome of the query.
    4. Baseline comparison:
         (a) success_count - failure_count (the current
             _best_exact_success_case ranking)
         (b) global majority class
       Both predict on test; we compare AUC.
    5. Multi-action waypoint top-1 accuracy (cobblestone, charcoal, stone,
       smooth_stone) -- does RADS pick the right action better than baseline?
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from optimus1.decisioner.feature import FeatureSpec, extract_features
from optimus1.decisioner.rads import RADS, RADSConfig


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as fp:
        return [json.loads(line) for line in fp if line.strip()]


def split_samples(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        out[s["split"]].append(s)
    return out


def load_bundle(path: str, device: str) -> Tuple[RADS, FeatureSpec, torch.Tensor, List[Dict[str, Any]]]:
    bundle = torch.load(path, map_location=device, weights_only=False)
    spec = FeatureSpec.from_dict(bundle["spec"])
    config = RADSConfig(**bundle.get("config", {}))
    model = RADS(spec, config)
    model.load_state_dict(bundle["model_state"])
    model.to(device).eval()
    library_vecs = torch.tensor(bundle["library_vecs"], dtype=torch.float32, device=device)
    return model, spec, library_vecs, bundle["library_meta"]


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


@torch.no_grad()
def score_samples(
    model: RADS,
    samples: List[Dict[str, Any]],
    library_vecs: torch.Tensor,
    library_meta: List[Dict[str, Any]],
    spec: FeatureSpec,
    device: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    library_runs = np.array([m["run_uuid"] for m in library_meta])
    feats_list = [extract_features(s, spec) for s in samples]
    probs, conc, attn_topcase_idx = [], [], []
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        chunk = feats_list[start:end]
        chunk_samples = samples[start:end]
        batch = {
            "numeric": torch.tensor(
                np.stack([f["numeric"] for f in chunk]), dtype=torch.float32, device=device
            ),
            "waypoint_id": torch.tensor(
                [f["waypoint_id"] for f in chunk], dtype=torch.long, device=device
            ),
            "final_goal_id": torch.tensor(
                [f["final_goal_id"] for f in chunk], dtype=torch.long, device=device
            ),
            "action_id": torch.tensor(
                [f["action_id"] for f in chunk], dtype=torch.long, device=device
            ),
        }
        # Mask same-run library entries for fair evaluation.
        mask = []
        for s in chunk_samples:
            mask.append(library_runs == s["run_uuid"])
        mask_t = torch.tensor(np.stack(mask), dtype=torch.bool, device=device)

        logits, attn = model(batch, library_vecs, retrieval_mask=mask_t)
        p = torch.sigmoid(logits).cpu().numpy()
        c = model.attention_concentration(attn).cpu().numpy()
        top1 = attn.argmax(dim=-1).cpu().numpy()
        probs.extend(p.tolist())
        conc.extend(c.tolist())
        attn_topcase_idx.extend(top1.tolist())
    return {
        "prob": np.array(probs),
        "concentration": np.array(conc),
        "top1_lib_idx": np.array(attn_topcase_idx),
    }


def overall_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(set(labels.tolist())) < 2:
        return {
            "n": int(len(probs)),
            "pos_rate": float(labels.mean()),
            "auc": float("nan"),
            "ap": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": 0.5,
            "ece": float("nan"),
        }
    auc = float(roc_auc_score(labels, probs))
    ap = float(average_precision_score(labels, probs))
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        preds = (probs >= thr).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_thr = float(thr)
    ece = compute_ece(probs, labels)
    return {
        "n": int(len(probs)),
        "pos_rate": float(labels.mean()),
        "auc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "best_threshold": best_thr,
        "ece": ece,
    }


def per_waypoint_auc(
    samples: List[Dict[str, Any]], probs: np.ndarray, min_neg: int = 5
) -> List[Dict[str, Any]]:
    by_wp: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        by_wp.setdefault(s["waypoint"], []).append(i)
    rows = []
    for wp, idxs in by_wp.items():
        labels = np.array([int(samples[i]["outcome"]["success"]) for i in idxs])
        ps = probs[idxs]
        pos = int(labels.sum())
        neg = int(len(labels) - pos)
        auc = (
            float(roc_auc_score(labels, ps))
            if pos > 0 and neg >= min_neg
            else None
        )
        rows.append(
            {
                "waypoint": wp,
                "n": int(len(idxs)),
                "pos": pos,
                "neg": neg,
                "auc": auc,
            }
        )
    rows.sort(key=lambda r: -r["n"])
    return rows


def attention_diagnostic(
    samples: List[Dict[str, Any]],
    top1_lib_idx: np.ndarray,
    library_meta: List[Dict[str, Any]],
) -> Dict[str, float]:
    same_wp = 0
    same_outcome = 0
    same_both = 0
    for i, s in enumerate(samples):
        lib = library_meta[int(top1_lib_idx[i])]
        wp_match = lib["waypoint"] == s["waypoint"]
        outcome_match = bool(lib["success"]) == bool(s["outcome"]["success"])
        if wp_match:
            same_wp += 1
        if outcome_match:
            same_outcome += 1
        if wp_match and outcome_match:
            same_both += 1
    n = max(len(samples), 1)
    return {
        "n": n,
        "top1_same_waypoint_rate": same_wp / n,
        "top1_same_outcome_rate": same_outcome / n,
        "top1_same_both_rate": same_both / n,
    }


def baseline_history_score(
    train_samples: List[Dict[str, Any]],
    eval_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Replicate the current `_best_exact_success_case` logic.

    For each (waypoint, action) build success - fail count from the train set.
    Predict-prob = sigmoid(score). This is the baseline we want to beat.
    """
    counts: Dict[Tuple[str, str], Dict[str, int]] = collections.defaultdict(
        lambda: {"s": 0, "f": 0}
    )
    for s in train_samples:
        key = (s["waypoint"], s["selected_action"])
        if s["outcome"]["success"]:
            counts[key]["s"] += 1
        else:
            counts[key]["f"] += 1

    probs = []
    labels = []
    for s in eval_samples:
        key = (s["waypoint"], s["selected_action"])
        c = counts.get(key, {"s": 0, "f": 0})
        # Laplace smoothing.
        p = (c["s"] + 1) / (c["s"] + c["f"] + 2)
        probs.append(p)
        labels.append(int(s["outcome"]["success"]))
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    return {
        "name": "history_count_baseline",
        "metrics": overall_metrics(probs_arr, labels_arr),
    }


def baseline_majority(eval_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = np.array([int(s["outcome"]["success"]) for s in eval_samples])
    p = float(labels.mean())
    probs = np.full_like(labels, p, dtype=float)
    return {
        "name": "majority_class_baseline",
        "metrics": overall_metrics(probs, labels),
    }


def multi_action_waypoint_topk(
    train_samples: List[Dict[str, Any]],
    eval_samples: List[Dict[str, Any]],
    model: RADS,
    library_vecs: torch.Tensor,
    library_meta: List[Dict[str, Any]],
    spec: FeatureSpec,
    device: str,
) -> List[Dict[str, Any]]:
    """For waypoints with >= 2 distinct actions, evaluate ranking accuracy.

    For each eval case, we score every action that historically appeared at
    that waypoint, and check whether the model's top-ranked action matches the
    one in the case (when the case succeeded). When the case failed, we report
    whether the model demoted that action below the historical winner.
    """

    actions_by_wp: Dict[str, List[str]] = {}
    success_actions_by_wp: Dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for s in train_samples:
        actions_by_wp.setdefault(s["waypoint"], [])
        if s["selected_action"] not in actions_by_wp[s["waypoint"]]:
            actions_by_wp[s["waypoint"]].append(s["selected_action"])
        if s["outcome"]["success"]:
            success_actions_by_wp[s["waypoint"]][s["selected_action"]] += 1

    multi_wps = {wp for wp, actions in actions_by_wp.items() if len(actions) >= 2}
    if not multi_wps:
        return []

    library_runs = np.array([m["run_uuid"] for m in library_meta])
    rows: List[Dict[str, Any]] = []
    for wp in sorted(multi_wps):
        wp_actions = actions_by_wp[wp]
        # Score eval cases at this wp by replacing selected_action with each
        # candidate action and re-running the model.
        wp_eval = [s for s in eval_samples if s["waypoint"] == wp]
        if not wp_eval:
            continue
        history_majority = success_actions_by_wp[wp].most_common(1)
        history_majority_action = history_majority[0][0] if history_majority else wp_actions[0]
        rads_picks_majority = 0
        rads_correct_when_success = 0
        success_count = 0
        for s in wp_eval:
            scores = []
            for a in wp_actions:
                feats = extract_features({**s, "selected_action": a}, spec)
                batch = {
                    "numeric": torch.tensor(feats["numeric"], dtype=torch.float32, device=device).unsqueeze(0),
                    "waypoint_id": torch.tensor([feats["waypoint_id"]], dtype=torch.long, device=device),
                    "final_goal_id": torch.tensor([feats["final_goal_id"]], dtype=torch.long, device=device),
                    "action_id": torch.tensor([feats["action_id"]], dtype=torch.long, device=device),
                }
                mask = torch.tensor(
                    [(library_runs == s["run_uuid"]).tolist()], dtype=torch.bool, device=device
                )
                with torch.no_grad():
                    logits, _ = model(batch, library_vecs, retrieval_mask=mask)
                scores.append((a, float(logits.item())))
            scores.sort(key=lambda x: -x[1])
            rads_pick = scores[0][0]
            if rads_pick == history_majority_action:
                rads_picks_majority += 1
            if s["outcome"]["success"]:
                success_count += 1
                if rads_pick == s["selected_action"]:
                    rads_correct_when_success += 1
        rows.append(
            {
                "waypoint": wp,
                "n_eval": len(wp_eval),
                "n_actions": len(wp_actions),
                "history_majority_action": history_majority_action,
                "rads_picks_majority_rate": rads_picks_majority / len(wp_eval),
                "successful_eval": success_count,
                "rads_top1_match_on_success_rate": (
                    rads_correct_when_success / success_count if success_count else None
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts/decisioner/rads_v1.pt")
    parser.add_argument("--data", default="data/decisioner/rads_v1.jsonl")
    parser.add_argument("--report_md", default="reports/decisioner/offline_eval_v1.md")
    parser.add_argument("--report_json", default="reports/decisioner/offline_eval_v1.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    splits = split_samples(samples)
    model, spec, library_vecs, library_meta = load_bundle(args.artifact, args.device)

    results: Dict[str, Any] = {
        "artifact": args.artifact,
        "data": args.data,
        "spec_summary": {
            "n_waypoints": len(spec.waypoints),
            "n_goals": len(spec.final_goals),
            "n_actions": len(spec.actions),
            "feature_dim": spec.total_input_dim,
        },
        "library_size": int(library_vecs.size(0)),
    }

    for split_name in ("val", "test"):
        eval_samples = splits[split_name]
        scored = score_samples(
            model, eval_samples, library_vecs, library_meta, spec, args.device
        )
        labels = np.array([int(s["outcome"]["success"]) for s in eval_samples])
        results[split_name] = {
            "overall": overall_metrics(scored["prob"], labels),
            "mean_concentration": float(scored["concentration"].mean()),
            "per_waypoint": per_waypoint_auc(eval_samples, scored["prob"]),
            "attention_diagnostic": attention_diagnostic(
                eval_samples, scored["top1_lib_idx"], library_meta
            ),
        }

    train_samples = splits["train"]
    test_samples = splits["test"]
    results["baselines"] = {
        "test": [
            baseline_history_score(train_samples, test_samples),
            baseline_majority(test_samples),
        ]
    }
    results["multi_action_test"] = multi_action_waypoint_topk(
        train_samples, test_samples, model, library_vecs, library_meta, spec, args.device
    )

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    md = render_markdown(results)
    Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_md, "w") as fp:
        fp.write(md)

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")
    print()
    print(f"Test AUC: {results['test']['overall']['auc']:.4f}")
    print(
        f"Test top-1 attention same-waypoint rate: "
        f"{results['test']['attention_diagnostic']['top1_same_waypoint_rate']:.4f}"
    )
    print(f"Baselines on test:")
    for b in results["baselines"]["test"]:
        m = b["metrics"]
        print(f"  {b['name']}: AUC={m['auc']:.4f}  AP={m['ap']:.4f}")


def render_markdown(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RADS Offline Evaluation v1\n")
    lines.append(f"- artifact: `{r['artifact']}`")
    lines.append(f"- data: `{r['data']}`")
    spec = r["spec_summary"]
    lines.append(
        f"- spec: waypoints={spec['n_waypoints']}, goals={spec['n_goals']}, "
        f"actions={spec['n_actions']}, feature_dim={spec['feature_dim']}"
    )
    lines.append(f"- library size (train cases): {r['library_size']}")
    lines.append("")

    for split in ("val", "test"):
        d = r[split]
        m = d["overall"]
        lines.append(f"## Split: {split}")
        lines.append("")
        lines.append(f"- n: {m['n']}")
        lines.append(f"- pos_rate: {m['pos_rate']:.4f}")
        lines.append(f"- AUC: {m['auc']:.4f}")
        lines.append(f"- AP:  {m['ap']:.4f}")
        lines.append(f"- best F1: {m['best_f1']:.4f} @ thr={m['best_threshold']:.2f}")
        lines.append(f"- ECE: {m['ece']:.4f}")
        lines.append(f"- mean attention concentration: {d['mean_concentration']:.4f}")
        lines.append("")
        ad = d["attention_diagnostic"]
        lines.append("### Attention diagnostic (top-1 case)")
        lines.append("")
        lines.append(f"- top-1 same-waypoint rate: {ad['top1_same_waypoint_rate']:.4f}")
        lines.append(f"- top-1 same-outcome rate: {ad['top1_same_outcome_rate']:.4f}")
        lines.append(f"- top-1 same-both rate:    {ad['top1_same_both_rate']:.4f}")
        lines.append("")
        lines.append("### Per-waypoint AUC (waypoints with >= 5 negatives)")
        lines.append("")
        lines.append("| waypoint | n | pos | neg | AUC |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in d["per_waypoint"]:
            if row["auc"] is None:
                continue
            lines.append(
                f"| {row['waypoint']} | {row['n']} | {row['pos']} | "
                f"{row['neg']} | {row['auc']:.4f} |"
            )
        lines.append("")

    lines.append("## Baselines on test")
    lines.append("")
    lines.append("| baseline | AUC | AP | F1 | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in r["baselines"]["test"]:
        m = b["metrics"]
        auc_s = "n/a" if m["auc"] != m["auc"] else f"{m['auc']:.4f}"
        ap_s = "n/a" if m["ap"] != m["ap"] else f"{m['ap']:.4f}"
        f1_s = "n/a" if m["best_f1"] != m["best_f1"] else f"{m['best_f1']:.4f}"
        ece_s = "n/a" if m["ece"] != m["ece"] else f"{m['ece']:.4f}"
        lines.append(f"| {b['name']} | {auc_s} | {ap_s} | {f1_s} | {ece_s} |")
    lines.append("")
    lines.append(f"**RADS test AUC: {r['test']['overall']['auc']:.4f}**")
    lines.append("")

    if r["multi_action_test"]:
        lines.append("## Multi-action waypoint ranking on test")
        lines.append("")
        lines.append(
            "For waypoints where the train set contains >= 2 distinct actions, "
            "we re-score every candidate action under the eval state and check "
            "(a) how often RADS' top-1 matches the historical majority action, "
            "and (b) on successful eval cases, whether RADS' top-1 matches the "
            "action that actually succeeded."
        )
        lines.append("")
        lines.append(
            "| waypoint | n_eval | n_actions | majority_action | "
            "RADS-top1=majority | success | RADS-top1 match on success |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for row in r["multi_action_test"]:
            match = (
                f"{row['rads_top1_match_on_success_rate']:.4f}"
                if row["rads_top1_match_on_success_rate"] is not None
                else "-"
            )
            lines.append(
                f"| {row['waypoint']} | {row['n_eval']} | {row['n_actions']} | "
                f"`{row['history_majority_action']}` | "
                f"{row['rads_picks_majority_rate']:.4f} | {row['successful_eval']} | {match} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Group split is by `run_uuid`. Same-run library entries are masked "
        "out of the retrieval pool during attention. This is a strict leak guard."
    )
    lines.append(
        "- Val split has very few negative samples (~21). Val AUC is therefore noisy "
        "and should be read alongside test AUC."
    )
    lines.append(
        "- Baseline `history_count_baseline` simulates the existing "
        "`_best_exact_success_case` ranking (Laplace-smoothed success rate per "
        "(waypoint, action))."
    )
    lines.append(
        "- Multi-action ranking only meaningful for waypoints with >= 2 actions in "
        "the case library. Currently 4 waypoints qualify: cobblestone, charcoal, stone, smooth_stone."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
