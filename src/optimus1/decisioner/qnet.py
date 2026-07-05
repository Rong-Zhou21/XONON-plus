"""Digi-Q-style plain Q-value decisioner (comparison arm for RADS).

Published lineage: Digi-Q (ICLR 2025) — train a Q-value function on offline
logged experience, then select actions at inference via Best-of-N ranking by
the Q-score. Faithful essence preserved here:

* offline-trained scorer on the SAME logged cases RADS trains on;
* at decision point D it scores the SAME candidate actions and executes the
  argmax — **no retrieval augmentation, no abstention** (never defers to the
  planner unless the candidate set is empty).

Differences vs RADS are exactly the two mechanisms under study:
retrieval-augmented scoring and low-confidence abstention. Everything else
(features, data, candidate enumeration, pipeline) is identical, so the
comparison isolates the decision mechanism.

Enabled by the boolean env var ``XENON_QDECIDER=1``
(checkpoint via ``XENON_QDECIDER_CKPT``, default
``artifacts/decisioner/qnet_coldstart.pt``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from .encoder import QueryEncoder
from .feature import FeatureSpec, extract_features


@dataclass
class QNetConfig:
    hidden_dim: int = 128
    output_dim: int = 64
    head_hidden_dim: int = 64
    dropout: float = 0.2
    # Same trainable residual on the (waypoint, action) prior log-odds as RADS
    # (parity: both models see the identical empirical prior).
    prior_logit_init_weight: float = 1.5


class QNet(nn.Module):
    """Plain discriminative Q head: h = MLP([q, action_emb]) -> sigmoid.

    Structurally this is RADS minus the case-retrieval context vector — i.e.
    the "no-retrieval" scorer of the Digi-Q lineage.
    """

    def __init__(self, spec: FeatureSpec, config: QNetConfig | None = None):
        super().__init__()
        c = config or QNetConfig()
        self.config = c
        self.query_encoder = QueryEncoder(
            spec, hidden_dim=c.hidden_dim, output_dim=c.output_dim, dropout=c.dropout
        )
        action_emb_dim = spec.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(c.output_dim + action_emb_dim, c.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.head_hidden_dim, 1),
        )
        self.prior_logit_weight = nn.Parameter(
            torch.tensor(float(c.prior_logit_init_weight), dtype=torch.float32)
        )

    def forward(self, query_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        q = self.query_encoder(query_batch)                       # [B, 64]
        action_vec = self.query_encoder.embedder.action_emb(
            query_batch["action_id"]
        )                                                         # [B, emb]
        logit = self.head(torch.cat([q, action_vec], dim=-1)).squeeze(-1)
        if "wp_action_prior" in query_batch:
            prior = query_batch["wp_action_prior"].clamp(min=1e-3, max=1.0 - 1e-3)
            logit = logit + self.prior_logit_weight * torch.log(prior / (1.0 - prior))
        return logit


@dataclass
class QDecision:
    p_success: float
    candidate_action: str = ""
    waypoint: str = ""


class QRuntime:
    """Inference wrapper mirroring RADSRuntime.score()'s query contract."""

    def __init__(self, model: QNet, spec: FeatureSpec, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.spec = spec
        self.device = device

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu") -> "QRuntime":
        bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
        spec = FeatureSpec.from_dict(bundle["spec"])
        config = QNetConfig(**bundle.get("config", {}))
        model = QNet(spec, config)
        model.load_state_dict(bundle["model_state"])
        return cls(model, spec, device)

    @torch.no_grad()
    def score(self, query_case: Dict[str, Any]) -> QDecision:
        feats = extract_features(query_case, self.spec)
        batch = {
            "numeric": torch.tensor(
                feats["numeric"], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "waypoint_id": torch.tensor(
                [feats["waypoint_id"]], dtype=torch.long, device=self.device
            ),
            "final_goal_id": torch.tensor(
                [feats["final_goal_id"]], dtype=torch.long, device=self.device
            ),
            "action_id": torch.tensor(
                [feats["action_id"]], dtype=torch.long, device=self.device
            ),
            "wp_action_prior": torch.tensor(
                [feats["wp_action_prior"]], dtype=torch.float32, device=self.device
            ),
        }
        logit = self.model(batch)
        return QDecision(
            p_success=float(torch.sigmoid(logit)[0].item()),
            candidate_action=str(query_case.get("selected_action", "")),
            waypoint=str(query_case.get("waypoint", "")),
        )


__all__ = ["QNet", "QNetConfig", "QRuntime", "QDecision"]
