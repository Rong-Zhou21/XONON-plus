"""Query and case encoders for RADS.

Both encoders share the same structured input schema (~52d after embedding
lookup). The case encoder additionally consumes the case outcome (success
flag) so the case representation reflects what happened, while the query
encoder represents an as-yet-undecided situation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .feature import FeatureSpec


class FeatureEmbedder(nn.Module):
    """Maps categorical IDs to dense vectors and concatenates with numeric input."""

    def __init__(self, spec: FeatureSpec):
        super().__init__()
        self.spec = spec
        d = spec.embedding_dim
        # Index 0 is reserved for <unk>; padding_idx makes its embedding stay at 0.
        self.waypoint_emb = nn.Embedding(len(spec.waypoints), d, padding_idx=0)
        self.goal_emb = nn.Embedding(len(spec.final_goals), d, padding_idx=0)
        self.action_emb = nn.Embedding(len(spec.actions), d, padding_idx=0)

    def forward(
        self,
        numeric: torch.Tensor,
        waypoint_id: torch.Tensor,
        final_goal_id: torch.Tensor,
        action_id: torch.Tensor,
    ) -> torch.Tensor:
        wp = self.waypoint_emb(waypoint_id)
        gl = self.goal_emb(final_goal_id)
        ac = self.action_emb(action_id)
        return torch.cat([numeric, wp, gl, ac], dim=-1)


class QueryEncoder(nn.Module):
    """Encodes a query into a 64d vector."""

    def __init__(self, spec: FeatureSpec, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.embedder = FeatureEmbedder(spec)
        in_dim = spec.total_input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embedder(
            batch["numeric"],
            batch["waypoint_id"],
            batch["final_goal_id"],
            batch["action_id"],
        )
        return self.net(x)


class CaseEncoder(nn.Module):
    """Encodes a case (with known outcome) into a 64d vector.

    Shares its embedding tables with the query encoder via the parameter passed
    in `embedder` so that the same waypoint id or action id maps to the same
    embedding from both sides.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        embedder: FeatureEmbedder,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.spec = spec
        self.embedder = embedder
        # +2 for outcome onehot (success/fail)
        in_dim = spec.total_input_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        # Auxiliary head: predict waypoint id from the case representation.
        self.wp_head = nn.Linear(output_dim, len(spec.waypoints))
        self.output_dim = output_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embedder(
            batch["numeric"],
            batch["waypoint_id"],
            batch["final_goal_id"],
            batch["action_id"],
        )
        outcome = torch.zeros(x.size(0), 2, device=x.device, dtype=x.dtype)
        # batch["label"] is 0/1; -1 means unknown but this encoder is only ever
        # called on cases with known outcomes (training set).
        labels = batch["label"].clamp(min=0)
        outcome.scatter_(1, labels.unsqueeze(1), 1.0)
        x_full = torch.cat([x, outcome], dim=-1)
        return self.net(x_full)

    def predict_waypoint(self, case_vec: torch.Tensor) -> torch.Tensor:
        return self.wp_head(case_vec)
