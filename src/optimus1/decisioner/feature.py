"""Structured feature extraction for RADS.

Only uses fields that already exist in cases.json. No external joins.

Feature layout (52 dim total):
  - waypoint_emb_id           categorical -> embedding (id only, embed in encoder)
  - final_goal_emb_id         categorical -> embedding
  - action_emb_id             categorical -> embedding
  - equipment_onehot          6d
  - biome_onehot              3d
  - waypoint_num_log1p        1d
  - position_in_run_log1p     1d
  - ypos_norm                 1d  (ypos / 64, clipped)
  - ypos_bucket               5d  onehot (0:>=80, 1:50-80, 2:30-50, 3:15-30, 4:<15)
  - inv_key_items_log1p       13d
  - tool_owned_flags          3d
  - inv_unique_count_log1p    1d

Total dense numeric feature dim = 6 + 3 + 1 + 1 + 1 + 5 + 13 + 3 + 1 = 34
Plus three categorical IDs handled separately (each becomes a 6d embedding in
the encoder, contributing 18d after lookup).
Final feature concat in encoder = 34 + 18 = 52d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


UNK = "<unk>"

INVENTORY_KEY_ITEMS: Tuple[str, ...] = (
    "log_total",
    "planks_total",
    "stick",
    "coal",
    "cobblestone",
    "iron_ore",
    "iron_ingot",
    "diamond",
    "gold_ore",
    "gold_ingot",
    "crafting_table",
    "furnace",
    "redstone",
)
LOG_FAMILY = ("oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log")
PLANK_FAMILY = (
    "oak_planks", "birch_planks", "spruce_planks", "jungle_planks", "acacia_planks", "dark_oak_planks",
)

EQUIPMENT_VOCAB: Tuple[str, ...] = (
    UNK,
    "none",
    "wooden_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "crafting_table",
)

BIOME_VOCAB: Tuple[str, ...] = (UNK, "forest", "plains")

YPOS_BUCKET_EDGES: Tuple[int, ...] = (15, 30, 50, 80)


@dataclass
class FeatureSpec:
    """Vocabulary + dimensionality bookkeeping. Built from training data."""

    waypoints: List[str]
    final_goals: List[str]
    actions: List[str]
    equipment: List[str] = field(default_factory=lambda: list(EQUIPMENT_VOCAB))
    biomes: List[str] = field(default_factory=lambda: list(BIOME_VOCAB))
    inv_key_items: List[str] = field(default_factory=lambda: list(INVENTORY_KEY_ITEMS))

    embedding_dim: int = 6
    # Optional empirical prior: P(success | waypoint, selected_action),
    # computed from the TRAIN split only. When enabled, contributes one
    # additional numeric input dim. Counters the action-embedding bias where
    # rare actions like "craft cobblestone" get pulled toward globally
    # successful "craft" actions despite their own train cases all failing.
    use_wp_action_prior: bool = False
    wp_action_prior_table: Dict[str, float] = field(default_factory=dict)
    wp_action_prior_default: float = 0.5

    @property
    def numeric_dim(self) -> int:
        # equipment_onehot + biome_onehot + scalar fields + ypos_bucket + inv + tool_flags + unique_count
        # Note: wp_action_prior is delivered as a separate batch field and added
        # as a residual to the logit, NOT part of numeric_dim.
        return (
            len(self.equipment)
            + len(self.biomes)
            + 1  # waypoint_num
            + 1  # position_in_run
            + 1  # ypos_norm
            + 5  # ypos_bucket
            + len(self.inv_key_items)
            + 3  # tool_owned_flags
            + 1  # inv_unique_count
        )

    @property
    def total_input_dim(self) -> int:
        return self.numeric_dim + 3 * self.embedding_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waypoints": self.waypoints,
            "final_goals": self.final_goals,
            "actions": self.actions,
            "equipment": self.equipment,
            "biomes": self.biomes,
            "inv_key_items": self.inv_key_items,
            "embedding_dim": self.embedding_dim,
            "numeric_dim": self.numeric_dim,
            "total_input_dim": self.total_input_dim,
            "use_wp_action_prior": self.use_wp_action_prior,
            "wp_action_prior_table": self.wp_action_prior_table,
            "wp_action_prior_default": self.wp_action_prior_default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSpec":
        return cls(
            waypoints=data["waypoints"],
            final_goals=data["final_goals"],
            actions=data["actions"],
            equipment=data.get("equipment", list(EQUIPMENT_VOCAB)),
            biomes=data.get("biomes", list(BIOME_VOCAB)),
            inv_key_items=data.get("inv_key_items", list(INVENTORY_KEY_ITEMS)),
            embedding_dim=data.get("embedding_dim", 6),
            use_wp_action_prior=bool(data.get("use_wp_action_prior", False)),
            wp_action_prior_table=dict(data.get("wp_action_prior_table", {}) or {}),
            wp_action_prior_default=float(data.get("wp_action_prior_default", 0.5)),
        )


def compute_wp_action_prior(
    train_samples: List[Dict[str, Any]], alpha: float = 2.0, prior_mean: float = 0.5
) -> Dict[str, float]:
    """Per-(waypoint, action) Laplace-smoothed success rate from train data.

    Key format: '<waypoint>|||<selected_action>'.
    """
    from collections import defaultdict
    counts: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])  # [n_succ, n_total]
    for s in train_samples:
        key = f"{_normalize_str(s.get('waypoint', ''))}|||{_normalize_str(s.get('selected_action', ''))}"
        counts[key][1] += 1.0
        if (s.get("outcome", {}) or {}).get("success") is True:
            counts[key][0] += 1.0
    return {
        k: (succ + alpha * prior_mean) / (total + alpha)
        for k, (succ, total) in counts.items()
    }


def build_spec_from_cases(cases: List[Dict[str, Any]], embedding_dim: int = 6) -> FeatureSpec:
    """Build vocabularies from a list of case dicts (typically the train split)."""

    wp_set, goal_set, action_set = set(), set(), set()
    for c in cases:
        wp_set.add(_normalize_str(c.get("waypoint", "")))
        goal_set.add(_normalize_str(c.get("original_final_goal", "")))
        action_set.add(_normalize_str(c.get("selected_action", "")))

    return FeatureSpec(
        waypoints=[UNK] + sorted(x for x in wp_set if x and x != UNK),
        final_goals=[UNK] + sorted(x for x in goal_set if x and x != UNK),
        actions=[UNK] + sorted(x for x in action_set if x and x != UNK),
        embedding_dim=embedding_dim,
    )


def _normalize_str(s: str) -> str:
    return str(s).strip().lower()


def _vocab_index(vocab: List[str], value: str) -> int:
    v = _normalize_str(value)
    if v in vocab:
        return vocab.index(v)
    return 0  # UNK is always at position 0


def _onehot(vocab: List[str], value: str) -> np.ndarray:
    arr = np.zeros(len(vocab), dtype=np.float32)
    arr[_vocab_index(vocab, value)] = 1.0
    return arr


def _ypos_bucket(ypos: float) -> np.ndarray:
    # 5 buckets: >=80 / 50-80 / 30-50 / 15-30 / <15
    arr = np.zeros(5, dtype=np.float32)
    if ypos >= 80:
        arr[0] = 1.0
    elif ypos >= 50:
        arr[1] = 1.0
    elif ypos >= 30:
        arr[2] = 1.0
    elif ypos >= 15:
        arr[3] = 1.0
    else:
        arr[4] = 1.0
    return arr


def _inventory_aggregate(inventory: Dict[str, Any]) -> Dict[str, float]:
    """Roll up per-item counts into the key-item dimensions defined above."""

    def _count(name: str) -> float:
        v = inventory.get(name)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    log_total = sum(_count(n) for n in LOG_FAMILY)
    planks_total = sum(_count(n) for n in PLANK_FAMILY)

    return {
        "log_total": log_total,
        "planks_total": planks_total,
        "stick": _count("stick"),
        "coal": _count("coal") + _count("charcoal"),
        "cobblestone": _count("cobblestone"),
        "iron_ore": _count("iron_ore"),
        "iron_ingot": _count("iron_ingot"),
        "diamond": _count("diamond"),
        "gold_ore": _count("gold_ore"),
        "gold_ingot": _count("gold_ingot"),
        "crafting_table": _count("crafting_table"),
        "furnace": _count("furnace"),
        "redstone": _count("redstone"),
    }


def _tool_owned_flags(inventory: Dict[str, Any]) -> np.ndarray:
    arr = np.zeros(3, dtype=np.float32)
    if inventory.get("wooden_pickaxe", 0) and float(inventory.get("wooden_pickaxe", 0)) > 0:
        arr[0] = 1.0
    if inventory.get("stone_pickaxe", 0) and float(inventory.get("stone_pickaxe", 0)) > 0:
        arr[1] = 1.0
    if inventory.get("iron_pickaxe", 0) and float(inventory.get("iron_pickaxe", 0)) > 0:
        arr[2] = 1.0
    return arr


def extract_features(case: Dict[str, Any], spec: FeatureSpec) -> Dict[str, Any]:
    """Convert a case dict into model inputs.

    Returns a dict with:
      - 'numeric':         np.float32 vector of dim spec.numeric_dim
      - 'waypoint_id':     int
      - 'final_goal_id':   int
      - 'action_id':       int
      - 'label':           int  (0 / 1)
      - 'outcome_known':   bool (False if label is missing)

    For query-time (no outcome), 'label' will be -1 and 'outcome_known' False.
    """

    state = case.get("state_snapshot", {}) or {}
    inventory = state.get("inventory", {}) or {}
    location = state.get("location_stats", {}) or {}

    equipment = state.get("equipment", "none")
    biome = state.get("biome", "")

    waypoint_num = float(case.get("waypoint_num", 1) or 1)
    position_in_run = float(case.get("_position_in_run", 0))
    ypos_raw = float(location.get("ypos", 64.0) or 64.0)
    ypos_norm = max(min(ypos_raw / 64.0, 4.0), 0.0)

    inv_agg = _inventory_aggregate(inventory)
    inv_vec = np.array(
        [math.log1p(inv_agg[k]) for k in spec.inv_key_items], dtype=np.float32
    )
    tool_flags = _tool_owned_flags(inventory)
    inv_unique_count = float(len(inventory))

    parts: List[np.ndarray] = [
        _onehot(spec.equipment, equipment),
        _onehot(spec.biomes, biome),
        np.array([math.log1p(max(waypoint_num, 0))], dtype=np.float32),
        np.array([math.log1p(max(position_in_run, 0))], dtype=np.float32),
        np.array([ypos_norm], dtype=np.float32),
        _ypos_bucket(ypos_raw),
        inv_vec,
        tool_flags,
        np.array([math.log1p(inv_unique_count)], dtype=np.float32),
    ]
    numeric = np.concatenate(parts).astype(np.float32)
    assert numeric.shape[0] == spec.numeric_dim, (
        f"numeric dim mismatch: got {numeric.shape[0]}, expected {spec.numeric_dim}"
    )

    # (waypoint, action) Laplace-smoothed prior rate, delivered as a separate
    # field. Used as a residual logit at the output of the decision head, so
    # the prior is always in play even if the model would otherwise ignore it.
    if spec.use_wp_action_prior:
        key = (
            f"{_normalize_str(case.get('waypoint', ''))}|||"
            f"{_normalize_str(case.get('selected_action', ''))}"
        )
        prior_rate = float(spec.wp_action_prior_table.get(key, spec.wp_action_prior_default))
    else:
        prior_rate = float(spec.wp_action_prior_default)

    label = -1
    outcome_known = False
    outcome = case.get("outcome", {}) or {}
    if outcome.get("success") is True:
        label = 1
        outcome_known = True
    elif outcome.get("success") is False:
        label = 0
        outcome_known = True

    return {
        "numeric": numeric,
        "waypoint_id": _vocab_index(spec.waypoints, case.get("waypoint", "")),
        "final_goal_id": _vocab_index(spec.final_goals, case.get("original_final_goal", "")),
        "action_id": _vocab_index(spec.actions, case.get("selected_action", "")),
        "wp_action_prior": prior_rate,
        "label": label,
        "outcome_known": outcome_known,
    }
