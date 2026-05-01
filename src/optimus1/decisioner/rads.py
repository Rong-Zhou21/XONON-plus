"""Retrieval-Augmented Decision Scorer (RADS).

Architecture:
    q   = QueryEncoder(query)                       # [B, 64]
    C   = CaseEncoder(library_cases)                # [N, 64]
    α   = softmax((q · Cᵀ) / τ, masked)             # [B, N]
    ctx = α @ C                                     # [B, 64]
    h   = MLP([q, ctx, action_emb])                 # [B, 64]
    P   = sigmoid(linear(h))                        # [B] success probability

Training losses:
    L1 = BCEWithLogitsLoss(P_logit, label)
    L2 = TripletMarginLoss on case-vectors (anchor / pos / neg)
    L3 = CrossEntropy(WP_head(C), waypoint_id)
    L  = L1 + λ_triplet * L2 + λ_wp * L3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CaseEncoder, FeatureEmbedder, QueryEncoder
from .feature import FeatureSpec


@dataclass
class RADSConfig:
    embedding_dim: int = 6
    hidden_dim: int = 128
    output_dim: int = 64
    head_hidden_dim: int = 64
    initial_tau: float = 0.5
    triplet_margin: float = 0.3
    lambda_triplet: float = 0.3
    lambda_wp: float = 0.2


class RADS(nn.Module):
    def __init__(self, spec: FeatureSpec, config: Optional[RADSConfig] = None):
        super().__init__()
        self.spec = spec
        self.config = config or RADSConfig()
        c = self.config

        self.query_encoder = QueryEncoder(
            spec, hidden_dim=c.hidden_dim, output_dim=c.output_dim
        )
        # Share the embedding table between query and case encoders.
        self.case_encoder = CaseEncoder(
            spec,
            self.query_encoder.embedder,
            hidden_dim=c.hidden_dim,
            output_dim=c.output_dim,
        )

        action_emb_dim = spec.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(c.output_dim * 2 + action_emb_dim, c.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(c.head_hidden_dim, 1),
        )
        self.log_tau = nn.Parameter(
            torch.tensor(float(c.initial_tau), dtype=torch.float32).log()
        )

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-3, max=10.0)

    def encode_query(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.query_encoder(batch)

    def encode_cases(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.case_encoder(batch)

    def forward(
        self,
        query_batch: Dict[str, torch.Tensor],
        library_vecs: torch.Tensor,
        retrieval_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute success logit and attention.

        Args:
            query_batch: dict of query tensors (incl. action_id)
            library_vecs: [N, D] precomputed case vectors
            retrieval_mask: optional [B, N] bool, True where a library entry
                            should be excluded (e.g. same run_uuid).

        Returns:
            logits: [B]
            attn:   [B, N]
        """
        q = self.encode_query(query_batch)  # [B, D]
        action_vec = self.query_encoder.embedder.action_emb(query_batch["action_id"])

        # Scaled dot-product attention.
        scores = q @ library_vecs.T  # [B, N]
        scores = scores / self.tau
        if retrieval_mask is not None:
            scores = scores.masked_fill(retrieval_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        context = attn @ library_vecs  # [B, D]

        h = torch.cat([q, context, action_vec], dim=-1)
        logits = self.head(h).squeeze(-1)
        return logits, attn

    def attention_concentration(self, attn: torch.Tensor) -> torch.Tensor:
        """Return 1 - normalized_entropy(attn). Higher = more concentrated."""
        # Add epsilon to avoid log(0).
        eps = 1e-9
        plogp = -attn * (attn.clamp(min=eps).log())
        entropy = plogp.sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(attn.size(-1)), device=attn.device))
        normalized = entropy / (max_entropy + eps)
        return 1.0 - normalized

    # --------------------------------------------------------------------- #
    # Multi-task losses                                                     #
    # --------------------------------------------------------------------- #

    @staticmethod
    def bce_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight
        )

    def triplet_loss(
        self,
        anchor_vecs: torch.Tensor,
        positive_vecs: torch.Tensor,
        negative_vecs: torch.Tensor,
    ) -> torch.Tensor:
        # Cosine-distance variant: distance = 1 - cos_sim
        a = F.normalize(anchor_vecs, dim=-1)
        p = F.normalize(positive_vecs, dim=-1)
        n = F.normalize(negative_vecs, dim=-1)
        d_pos = 1.0 - (a * p).sum(dim=-1)
        d_neg = 1.0 - (a * n).sum(dim=-1)
        return F.relu(d_pos - d_neg + self.config.triplet_margin).mean()

    def waypoint_recon_loss(
        self, case_vecs: torch.Tensor, waypoint_ids: torch.Tensor
    ) -> torch.Tensor:
        logits = self.case_encoder.predict_waypoint(case_vecs)
        return F.cross_entropy(logits, waypoint_ids)


# --------------------------------------------------------------------------- #
# Triplet sampling helpers                                                    #
# --------------------------------------------------------------------------- #


def sample_triplets(
    cases: List[Dict[str, int]],
    rng,
    max_triplets: int = 256,
) -> List[Tuple[int, int, int]]:
    """Sample (anchor, positive, negative) index triples.

    Each item is a dict with at least 'index', 'waypoint_id', 'label',
    'run_uuid'. Positive must share waypoint_id and label with anchor; negative
    must share waypoint_id but differ in label. All three must come from
    different run_uuids.
    """
    by_wp_label: Dict[Tuple[int, int], List[int]] = {}
    for c in cases:
        key = (int(c["waypoint_id"]), int(c["label"]))
        by_wp_label.setdefault(key, []).append(int(c["index"]))

    triplets: List[Tuple[int, int, int]] = []
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    for anchor_idx in indices:
        a = cases[anchor_idx]
        pos_pool = by_wp_label.get((int(a["waypoint_id"]), int(a["label"])), [])
        neg_label = 1 - int(a["label"])
        neg_pool = by_wp_label.get((int(a["waypoint_id"]), neg_label), [])
        if not pos_pool or not neg_pool:
            continue
        # Pick a positive from a different run (best effort).
        pos_idx = _pick_different_run(pos_pool, cases, a["run_uuid"], anchor_idx, rng)
        neg_idx = _pick_different_run(neg_pool, cases, a["run_uuid"], anchor_idx, rng)
        if pos_idx is None or neg_idx is None:
            continue
        triplets.append((anchor_idx, pos_idx, neg_idx))
        if len(triplets) >= max_triplets:
            break
    return triplets


def _pick_different_run(
    pool: List[int], cases: List[Dict[str, int]], anchor_run: str, anchor_idx: int, rng
) -> Optional[int]:
    candidates = [i for i in pool if i != anchor_idx and cases[i].get("run_uuid") != anchor_run]
    if not candidates:
        candidates = [i for i in pool if i != anchor_idx]
    if not candidates:
        return None
    return rng.choice(candidates)
