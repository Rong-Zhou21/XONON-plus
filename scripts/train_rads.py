#!/usr/bin/env python3
"""Train the RADS decisioner offline.

Inputs:
    data/decisioner/rads_v1.jsonl       (from export_decisioner_dataset.py)

Outputs:
    artifacts/decisioner/rads_v1.pt     (model state, spec, library, library meta)
    reports/decisioner/training_log_v1.json

Notes:
- The case library used for cross-attention is the encoded TRAIN set itself.
  At each step we re-encode the whole library so gradients flow through both
  encoders end-to-end. This is feasible because the train set is small
  (< 2k samples) and the encoder is shallow.
- For each query case, library entries with the same run_uuid are masked out
  of the retrieval pool to prevent label leakage.
- Multi-task loss = BCE(success) + λ_t * triplet(case-vectors) + λ_wp * waypoint reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from optimus1.decisioner.feature import (
    FeatureSpec,
    build_spec_from_cases,
    compute_wp_action_prior,
    extract_features,
)
from optimus1.decisioner.rads import RADS, RADSConfig, sample_triplets


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class CaseDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], spec: FeatureSpec):
        self.samples = samples
        self.spec = spec
        # Pre-extract features so __getitem__ stays cheap.
        self._cache = [extract_features(s, spec) for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feats = self._cache[idx]
        return {
            "numeric": torch.tensor(feats["numeric"], dtype=torch.float32),
            "waypoint_id": torch.tensor(feats["waypoint_id"], dtype=torch.long),
            "final_goal_id": torch.tensor(feats["final_goal_id"], dtype=torch.long),
            "action_id": torch.tensor(feats["action_id"], dtype=torch.long),
            "wp_action_prior": torch.tensor(feats["wp_action_prior"], dtype=torch.float32),
            "label": torch.tensor(feats["label"], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def split_samples(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"train": [], "val": [], "test": []}
    for s in samples:
        out[s["split"]].append(s)
    return out


# --------------------------------------------------------------------------- #
# Library encoding                                                            #
# --------------------------------------------------------------------------- #


def build_library_batch(
    train_samples: List[Dict[str, Any]], spec: FeatureSpec, device: str
) -> Dict[str, torch.Tensor]:
    feats_list = [extract_features(s, spec) for s in train_samples]
    return {
        "numeric": torch.tensor(
            np.stack([f["numeric"] for f in feats_list]), dtype=torch.float32, device=device
        ),
        "waypoint_id": torch.tensor(
            [f["waypoint_id"] for f in feats_list], dtype=torch.long, device=device
        ),
        "final_goal_id": torch.tensor(
            [f["final_goal_id"] for f in feats_list], dtype=torch.long, device=device
        ),
        "action_id": torch.tensor(
            [f["action_id"] for f in feats_list], dtype=torch.long, device=device
        ),
        "label": torch.tensor(
            [f["label"] for f in feats_list], dtype=torch.long, device=device
        ),
    }


def build_run_uuid_mask(
    query_run_uuids: List[str], library_run_uuids: List[str], device: str
) -> torch.Tensor:
    """Return [B, N] bool tensor where True means 'same run as query, mask out'."""
    lib = np.array(library_run_uuids)
    rows = []
    for q in query_run_uuids:
        rows.append(lib == q)
    return torch.tensor(np.stack(rows), dtype=torch.bool, device=device)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(
    model: RADS,
    samples: List[Dict[str, Any]],
    library_samples: List[Dict[str, Any]],
    spec: FeatureSpec,
    device: str,
    eval_batch: int = 64,
) -> Dict[str, float]:
    model.eval()
    library_batch = build_library_batch(library_samples, spec, device)
    library_vecs = model.encode_cases(library_batch)
    library_runs = [s["run_uuid"] for s in library_samples]
    library_waypoint_ids = library_batch["waypoint_id"]

    all_logits, all_labels, all_concentration = [], [], []
    feats_list = [extract_features(s, spec) for s in samples]
    for start in range(0, len(samples), eval_batch):
        end = min(start + eval_batch, len(samples))
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
            "wp_action_prior": torch.tensor(
                [f["wp_action_prior"] for f in chunk], dtype=torch.float32, device=device
            ),
        }
        labels = [int(f["label"]) for f in chunk]
        run_uuids = [s["run_uuid"] for s in chunk_samples]
        mask = build_run_uuid_mask(run_uuids, library_runs, device)

        logits, attn = model(
            batch,
            library_vecs,
            retrieval_mask=mask,
            library_waypoint_ids=library_waypoint_ids,
        )
        conc = model.attention_concentration(attn)

        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels)
        all_concentration.extend(conc.cpu().tolist())

    probs = 1.0 / (1.0 + np.exp(-np.array(all_logits)))
    labels_arr = np.array(all_labels)
    metrics: Dict[str, float] = {
        "n_samples": float(len(samples)),
        "pos_rate": float(labels_arr.mean()),
        "mean_p": float(probs.mean()),
        "mean_concentration": float(np.mean(all_concentration)),
    }
    if len(set(labels_arr.tolist())) >= 2:
        metrics["auc"] = float(roc_auc_score(labels_arr, probs))
        metrics["ap"] = float(average_precision_score(labels_arr, probs))
        # Pick threshold that maximises F1 on the eval set.
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (probs >= thr).astype(int)
            f = f1_score(labels_arr, preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = float(thr)
        metrics["best_f1"] = float(best_f1)
        metrics["best_threshold"] = best_thr
        # Calibration error (15-bin ECE).
        ece = compute_ece(probs, labels_arr, n_bins=15)
        metrics["ece"] = float(ece)
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
        metrics["best_f1"] = float("nan")
        metrics["best_threshold"] = 0.5
        metrics["ece"] = float("nan")
    return metrics


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        cnt = mask.sum()
        if cnt == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/decisioner/rads_v1.jsonl")
    parser.add_argument("--out", default="artifacts/decisioner/rads_v2.pt")
    parser.add_argument("--report", default="reports/decisioner/training_log_v2.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stop after this many epochs of no val_auc improvement.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--triplet_weight", type=float, default=0.1)
    parser.add_argument("--wp_weight", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--same_wp_min", type=int, default=8,
                        help="Hard same-waypoint mask threshold; 0 disables.")
    parser.add_argument("--use_wp_action_prior", type=int, default=1,
                        help="Use (waypoint, action) Laplace-smoothed success rate as a residual logit prior.")
    parser.add_argument("--prior_logit_init_weight", type=float, default=1.5,
                        help="Initial scale on the prior log-odds residual (trainable).")
    parser.add_argument("--triplets_per_step", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    samples = load_jsonl(args.data)
    splits = split_samples(samples)
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    spec = build_spec_from_cases(train_samples)
    if args.use_wp_action_prior:
        spec.use_wp_action_prior = True
        spec.wp_action_prior_table = compute_wp_action_prior(train_samples)
        spec.wp_action_prior_default = 0.5
        print(
            f"using wp_action prior (table size={len(spec.wp_action_prior_table)}, "
            f"default={spec.wp_action_prior_default})"
        )
    print(
        f"vocab: waypoints={len(spec.waypoints)} goals={len(spec.final_goals)} "
        f"actions={len(spec.actions)} feature_dim={spec.total_input_dim}"
    )

    n_pos = sum(1 for s in train_samples if s["outcome"]["success"])
    n_neg = len(train_samples) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=args.device)
    print(f"train pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.4f}")

    config = RADSConfig(
        lambda_triplet=args.triplet_weight,
        lambda_wp=args.wp_weight,
        dropout=args.dropout,
        same_wp_min=args.same_wp_min,
        prior_logit_init_weight=args.prior_logit_init_weight,
    )
    model = RADS(spec, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = CaseDataset(train_samples, spec)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    train_run_uuids = [s["run_uuid"] for s in train_samples]
    train_indices_by_run: Dict[str, List[int]] = {}
    for i, s in enumerate(train_samples):
        train_indices_by_run.setdefault(s["run_uuid"], []).append(i)

    # For triplet sampling.
    triplet_meta = []
    for i, s in enumerate(train_samples):
        feats = extract_features(s, spec)
        triplet_meta.append({
            "index": i,
            "waypoint_id": feats["waypoint_id"],
            "label": int(s["outcome"]["success"]),
            "run_uuid": s["run_uuid"],
        })

    log: Dict[str, Any] = {
        "args": vars(args),
        "spec": spec.to_dict(),
        "epochs": [],
        "best": None,
    }

    rng = random.Random(args.seed)
    best_val_auc = -1.0
    best_state = None
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    epochs_since_improvement = 0

    library_batch_static = build_library_batch(train_samples, spec, args.device)
    library_waypoint_ids = library_batch_static["waypoint_id"]

    for epoch in range(args.epochs):
        model.train()
        epoch_l1, epoch_l2, epoch_l3, epoch_total, n_steps = 0.0, 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # Re-encode the entire training set as the library on every step so
            # gradients flow through both encoders.
            library_vecs = model.encode_cases(library_batch_static)

            # Build retrieval mask: same run_uuid as query -> True.
            query_indices = batch["index"].cpu().tolist()
            query_run_uuids = [train_run_uuids[i] for i in query_indices]
            mask = build_run_uuid_mask(query_run_uuids, train_run_uuids, args.device)

            # Also mask out the query itself.
            for row, qi in enumerate(query_indices):
                mask[row, qi] = True

            logits, _ = model(
                batch,
                library_vecs,
                retrieval_mask=mask,
                library_waypoint_ids=library_waypoint_ids,
            )
            l1 = model.bce_loss(logits, batch["label"], pos_weight)

            # Triplet loss on case-vectors.
            triplets = sample_triplets(triplet_meta, rng, max_triplets=args.triplets_per_step)
            if triplets:
                a_idx = torch.tensor([t[0] for t in triplets], dtype=torch.long, device=args.device)
                p_idx = torch.tensor([t[1] for t in triplets], dtype=torch.long, device=args.device)
                n_idx = torch.tensor([t[2] for t in triplets], dtype=torch.long, device=args.device)
                a_vec = library_vecs[a_idx]
                p_vec = library_vecs[p_idx]
                n_vec = library_vecs[n_idx]
                l2 = model.triplet_loss(a_vec, p_vec, n_vec)
            else:
                l2 = torch.tensor(0.0, device=args.device)

            wp_ids = library_batch_static["waypoint_id"]
            l3 = model.waypoint_recon_loss(library_vecs, wp_ids)

            loss = l1 + args.triplet_weight * l2 + args.wp_weight * l3
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_l1 += float(l1.item())
            epoch_l2 += float(l2.item())
            epoch_l3 += float(l3.item())
            epoch_total += float(loss.item())
            n_steps += 1

        n_steps = max(n_steps, 1)
        train_log = {
            "epoch": epoch,
            "train_loss": epoch_total / n_steps,
            "train_bce": epoch_l1 / n_steps,
            "train_triplet": epoch_l2 / n_steps,
            "train_wp_recon": epoch_l3 / n_steps,
        }

        val_metrics = evaluate(model, val_samples, train_samples, spec, args.device)
        train_log["val"] = val_metrics
        log["epochs"].append(train_log)

        line = (
            f"epoch {epoch:02d} loss={train_log['train_loss']:.4f} "
            f"bce={train_log['train_bce']:.4f} trip={train_log['train_triplet']:.4f} "
            f"wp={train_log['train_wp_recon']:.4f} | "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f} "
            f"val_f1={val_metrics.get('best_f1', float('nan')):.4f} "
            f"val_ece={val_metrics.get('ece', float('nan')):.4f} "
            f"conc={val_metrics.get('mean_concentration', float('nan')):.4f}"
        )
        print(line)

        # Track best by val AUC (handle NaN gracefully).
        val_auc = val_metrics.get("auc", float("nan"))
        improved = (
            isinstance(val_auc, float)
            and not (val_auc != val_auc)
            and val_auc > best_val_auc
        )
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val": val_metrics, "epoch": epoch}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if args.patience > 0 and epochs_since_improvement >= args.patience:
                print(
                    f"early stop at epoch {epoch} "
                    f"(no val_auc improvement in {args.patience} epochs)"
                )
                break

    # Final evaluation on test using best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_samples, train_samples, spec, args.device)
    log["best"] = {**best_metrics, "test": test_metrics}
    print(
        f"BEST epoch={best_epoch} val_auc={best_val_auc:.4f} "
        f"test_auc={test_metrics.get('auc', float('nan')):.4f}"
    )

    # Build library cache for runtime: encode train cases with the BEST model.
    model.eval()
    with torch.no_grad():
        library_vecs = model.encode_cases(library_batch_static).cpu().numpy()
    library_meta = [
        {
            "case_id": s["case_id"],
            "run_uuid": s["run_uuid"],
            "waypoint": s["waypoint"],
            "selected_action": s["selected_action"],
            "success": bool(s["outcome"]["success"]),
        }
        for s in train_samples
    ]

    bundle = {
        "model_state": model.state_dict(),
        "spec": spec.to_dict(),
        "config": asdict(config),
        "library_vecs": library_vecs,
        "library_meta": library_meta,
        "train_runs": sorted({s["run_uuid"] for s in train_samples}),
        "best_epoch": best_epoch,
        "best_val": best_metrics.get("val"),
        "test": test_metrics,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    print(f"Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fp:
        json.dump(log, fp, indent=2, ensure_ascii=False)
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
