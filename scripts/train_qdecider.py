#!/usr/bin/env python
"""Train the Digi-Q-style plain Q decisioner (comparison arm for RADS).

Uses the SAME exported dataset, splits, features and empirical prior as
scripts/train_rads.py — the only difference is the model: a plain
discriminative Q head with no case-retrieval context (and, at inference,
no abstention). See src/optimus1/decisioner/qnet.py.

Usage:
    python scripts/train_qdecider.py \
        --data data/decisioner/rads_v1.jsonl \
        --out artifacts/decisioner/qnet_coldstart.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimus1.decisioner.feature import (  # noqa: E402
    FeatureSpec,
    build_spec_from_cases,
    compute_wp_action_prior,
    extract_features,
)
from optimus1.decisioner.qnet import QNet, QNetConfig  # noqa: E402


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def batch_of(samples: List[Dict[str, Any]], spec: FeatureSpec, device: str):
    feats = [extract_features(s, spec) for s in samples]
    return {
        "numeric": torch.tensor(
            np.stack([f["numeric"] for f in feats]), dtype=torch.float32, device=device
        ),
        "waypoint_id": torch.tensor([f["waypoint_id"] for f in feats], dtype=torch.long, device=device),
        "final_goal_id": torch.tensor([f["final_goal_id"] for f in feats], dtype=torch.long, device=device),
        "action_id": torch.tensor([f["action_id"] for f in feats], dtype=torch.long, device=device),
        "wp_action_prior": torch.tensor([f["wp_action_prior"] for f in feats], dtype=torch.float32, device=device),
    }, torch.tensor([float(f["label"]) for f in feats], dtype=torch.float32, device=device)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


@torch.no_grad()
def evaluate(model: QNet, samples, spec, device, eval_batch=256) -> Dict[str, float]:
    model.eval()
    probs, labels = [], []
    for i in range(0, len(samples), eval_batch):
        chunk = samples[i : i + eval_batch]
        batch, y = batch_of(chunk, spec, device)
        p = torch.sigmoid(model(batch))
        probs.extend(p.cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())
    probs = np.array(probs)
    labels = np.array(labels)
    out = {"n": float(len(labels)), "pos_rate": float(labels.mean())}
    if len(set(labels.tolist())) >= 2:
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 19):
            f = f1_score(labels, probs >= thr, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = float(f), float(thr)
        out["f1"] = best_f1
        out["f1_thr"] = best_thr
        out["ece"] = compute_ece(probs, labels)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/decisioner/rads_v1.jsonl")
    ap.add_argument("--out", default="artifacts/decisioner/qnet_coldstart.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--use_wp_action_prior", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    samples = load_jsonl(args.data)
    splits = {"train": [], "val": [], "test": []}
    for s in samples:
        splits[s["split"]].append(s)
    train_s, val_s, test_s = splits["train"], splits["val"], splits["test"]
    print(f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    spec = build_spec_from_cases(train_s)
    if args.use_wp_action_prior:
        spec.use_wp_action_prior = True
        spec.wp_action_prior_table = compute_wp_action_prior(train_s)
        spec.wp_action_prior_default = 0.5
        print(f"using wp_action prior (table size={len(spec.wp_action_prior_table)})")

    config = QNetConfig(dropout=args.dropout)
    model = QNet(spec, config).to(args.device)

    n_pos = sum(1 for s in train_s if s["outcome"]["success"])
    n_neg = len(train_s) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=args.device)
    print(f"train pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.4f}")
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc, best_state, best_epoch, since = -1.0, None, -1, 0
    idx = list(range(len(train_s)))
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(idx)
        total = 0.0
        for i in range(0, len(idx), args.batch_size):
            chunk = [train_s[j] for j in idx[i : i + args.batch_size]]
            batch, y = batch_of(chunk, spec, args.device)
            logit = model(batch)
            loss = crit(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(chunk)
        val = evaluate(model, val_s, spec, args.device)
        print(
            f"epoch {epoch:02d} loss={total/len(train_s):.4f} "
            f"val_auc={val.get('auc', float('nan')):.4f} "
            f"val_f1={val.get('f1', float('nan')):.4f} "
            f"val_ece={val.get('ece', float('nan')):.4f}"
        )
        if val.get("auc", -1) > best_val_auc:
            best_val_auc = val["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since = 0
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at epoch {epoch} (no val_auc gain in {args.patience})")
                break

    model.load_state_dict(best_state)
    test = evaluate(model, test_s, spec, args.device)
    print(f"BEST epoch={best_epoch} val_auc={best_val_auc:.4f} test_auc={test.get('auc', float('nan')):.4f}")

    bundle = {
        "model_state": model.state_dict(),
        "spec": spec.to_dict(),
        "config": asdict(config),
        "train_runs": sorted({s["run_uuid"] for s in train_s}),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test": test,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"Saved {out} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
