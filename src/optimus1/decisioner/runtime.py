"""Lightweight inference wrapper for RADS.

This module is intentionally not invoked from `case_memory.py` in this round.
It defines the interface that the next round will plug in.

Typical usage (later, after enabled in YAML):
    decisioner = RADSRuntime.load(ckpt_path, library_path)
    result = decisioner.score(case_query)
    # result.p_success, result.confidence, result.evidence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .feature import FeatureSpec, extract_features
from .rads import RADS, RADSConfig


@dataclass
class DecisionEvidence:
    case_id: str
    waypoint: str
    selected_action: str
    success: Optional[bool]
    attention: float
    score_against_query: float


@dataclass
class DecisionResult:
    p_success: float
    confidence: float  # p_success * attention_concentration
    attention_concentration: float
    evidence: List[DecisionEvidence] = field(default_factory=list)
    candidate_action: str = ""
    waypoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_success": self.p_success,
            "confidence": self.confidence,
            "attention_concentration": self.attention_concentration,
            "candidate_action": self.candidate_action,
            "waypoint": self.waypoint,
            "evidence": [e.__dict__ for e in self.evidence],
        }


class RADSRuntime:
    def __init__(
        self,
        model: RADS,
        spec: FeatureSpec,
        library_vecs: torch.Tensor,
        library_meta: List[Dict[str, Any]],
        device: str = "cpu",
    ):
        self.model = model.eval()
        self.spec = spec
        self.library_vecs = library_vecs.to(device)
        self.library_meta = library_meta
        self.device = device
        # Pre-compute library_waypoint_ids for the same-waypoint mask in attention.
        wp_idx = {wp: i for i, wp in enumerate(spec.waypoints)}
        self.library_wp_ids = torch.tensor(
            [wp_idx.get(m.get("waypoint", ""), 0) for m in library_meta],
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def load(cls, artifact_path: str, device: str = "cpu") -> "RADSRuntime":
        """Load a single .pt file containing model state, spec, library vectors, library meta."""
        bundle = torch.load(artifact_path, map_location=device, weights_only=False)
        spec = FeatureSpec.from_dict(bundle["spec"])
        # Tolerate older bundles missing fields added in newer configs.
        cfg_raw = bundle.get("config") or {}
        cfg_kwargs = {
            k: v for k, v in cfg_raw.items() if k in RADSConfig.__dataclass_fields__
        }
        config = RADSConfig(**cfg_kwargs)
        model = RADS(spec, config)
        model.load_state_dict(bundle["model_state"])
        model.to(device)
        return cls(
            model=model,
            spec=spec,
            library_vecs=torch.tensor(bundle["library_vecs"], dtype=torch.float32),
            library_meta=bundle["library_meta"],
            device=device,
        )

    @torch.no_grad()
    def score(
        self,
        case_query: Dict[str, Any],
        topk_evidence: int = 5,
        exclude_run_uuid: Optional[str] = None,
    ) -> DecisionResult:
        feats = extract_features(case_query, self.spec)
        batch = {
            "numeric": torch.tensor(feats["numeric"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "waypoint_id": torch.tensor([feats["waypoint_id"]], dtype=torch.long, device=self.device),
            "final_goal_id": torch.tensor([feats["final_goal_id"]], dtype=torch.long, device=self.device),
            "action_id": torch.tensor([feats["action_id"]], dtype=torch.long, device=self.device),
            "wp_action_prior": torch.tensor([feats["wp_action_prior"]], dtype=torch.float32, device=self.device),
        }

        retrieval_mask = None
        if exclude_run_uuid is not None:
            mask = [m.get("run_uuid") == exclude_run_uuid for m in self.library_meta]
            if any(mask):
                retrieval_mask = torch.tensor([mask], dtype=torch.bool, device=self.device)

        logits, attn = self.model(
            batch,
            self.library_vecs,
            retrieval_mask=retrieval_mask,
            library_waypoint_ids=self.library_wp_ids,
        )
        p_success = torch.sigmoid(logits).item()
        concentration = float(self.model.attention_concentration(attn).item())
        confidence = p_success * concentration

        attn_row = attn.squeeze(0).detach().cpu().numpy()
        order = np.argsort(-attn_row)[:topk_evidence]
        evidence = []
        for idx in order:
            meta = self.library_meta[int(idx)]
            evidence.append(
                DecisionEvidence(
                    case_id=meta.get("case_id", ""),
                    waypoint=meta.get("waypoint", ""),
                    selected_action=meta.get("selected_action", ""),
                    success=meta.get("success"),
                    attention=float(attn_row[int(idx)]),
                    score_against_query=float(
                        (self.library_vecs[int(idx)] @ self.model.encode_query(batch).squeeze(0)).item()
                    ),
                )
            )

        return DecisionResult(
            p_success=p_success,
            confidence=confidence,
            attention_concentration=concentration,
            evidence=evidence,
            candidate_action=str(case_query.get("selected_action", "")),
            waypoint=str(case_query.get("waypoint", "")),
        )
