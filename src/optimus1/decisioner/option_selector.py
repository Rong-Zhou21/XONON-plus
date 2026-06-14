"""Lightweight option selector for environment-aware skills.

This module is intentionally separate from RADS. The current RADS checkpoint was
trained on high-level language actions, so newly introduced ``option:*`` actions
would be out-of-vocabulary. The selector below gives the environment skill path a
minimal decisioner: rules generate valid option candidates, and this class decides
whether to execute them using option-event history stored beside case memory.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@dataclass
class OptionScore:
    option: str
    score: float
    success: int
    total: int
    source: str
    context_key: str = ""


@dataclass
class OptionSelection:
    option: Optional[str]
    execute: bool
    rankings: List[OptionScore]
    reason: str

    def to_trace(self) -> Dict[str, Any]:
        return {
            "source": "option_decisioner",
            "selected_option": self.option,
            "execute": self.execute,
            "reason": self.reason,
            "rankings": [
                {
                    "option": r.option,
                    "score": r.score,
                    "success": r.success,
                    "total": r.total,
                    "source": r.source,
                    "context_key": r.context_key,
                }
                for r in self.rankings
            ],
        }


class OptionDecisioner:
    """Rule-gated option scheduler backed by option event statistics."""

    def __init__(
        self,
        memory_root: str,
        logger: Any = None,
        *,
        enabled: bool = True,
        min_score: float = 0.35,
        default_score: float = 0.65,
        prior_success: float = 1.0,
        prior_total: float = 2.0,
        min_context_total: int = 3,
        min_option_total: int = 3,
    ) -> None:
        self.memory_root = memory_root
        self.logger = logger
        self.enabled = bool(enabled)
        self.min_score = float(min_score)
        self.default_score = float(default_score)
        self.prior_success = float(prior_success)
        self.prior_total = float(prior_total)
        self.min_context_total = int(min_context_total)
        self.min_option_total = int(min_option_total)
        self.option_dir = os.path.join(memory_root, "case_memory")
        self.events_path = os.path.join(self.option_dir, "option_events.jsonl")
        os.makedirs(self.option_dir, exist_ok=True)
        self._stats, self._context_stats = self._load_stats()

    def _iter_events(self) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return
        with open(self.events_path, "r") as fp:
            fcntl.flock(fp, fcntl.LOCK_SH)
            try:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        yield event
            finally:
                fcntl.flock(fp, fcntl.LOCK_UN)

    def _load_stats(self) -> tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        stats: Dict[str, Dict[str, int]] = {}
        context_stats: Dict[str, Dict[str, int]] = {}
        for event in self._iter_events() or []:
            if event.get("event_type") != "option_outcome":
                continue
            option = str(event.get("option") or "")
            if not option:
                continue
            success = bool((event.get("outcome") or {}).get("success"))
            bucket = stats.setdefault(option, {"success": 0, "total": 0})
            bucket["total"] += 1
            if success:
                bucket["success"] += 1
            context_key = self._context_signature(option, event.get("context") or {})
            context_bucket = context_stats.setdefault(context_key, {"success": 0, "total": 0})
            context_bucket["total"] += 1
            if success:
                context_bucket["success"] += 1
        return stats, context_stats

    def _score_from_bucket(self, option: str, bucket: Dict[str, int], source: str, context_key: str = "") -> OptionScore:
        success = int(bucket.get("success", 0))
        total = int(bucket.get("total", 0))
        score = (success + self.prior_success) / (total + self.prior_total)
        return OptionScore(option, float(score), success, total, source, context_key)

    def _bucket_number(self, value: Any, edges: tuple[float, ...]) -> str:
        try:
            number = float(value)
        except Exception:
            return "unknown"
        for edge in edges:
            if number < edge:
                return f"lt{int(edge)}"
        return f"ge{int(edges[-1])}" if edges else "value"

    def _context_signature(self, option: str, context: Dict[str, Any] | None) -> str:
        context = context or {}
        if context.get("is_surface_resource"):
            mode = "surface_resource"
        elif context.get("is_tunnel_resource"):
            mode = "tunnel_resource"
        elif context.get("is_resource_acquisition"):
            mode = "resource"
        else:
            mode = "general"
        y_band = self._bucket_number(context.get("ypos"), (16, 32, 50, 65))
        pending = self._bucket_number(context.get("pending_relevant_drops"), (1, 2, 5))
        movement = self._bucket_number(context.get("movement_stagnant_ticks"), (20, 80, 160))
        resource = self._bucket_number(context.get("resource_stagnant_ticks"), (20, 100, 200))
        surface = self._bucket_number(context.get("surface_stuck_ticks"), (20, 60, 120))
        stale_goal = self._bucket_number(context.get("stale_goal_progress_ticks"), (80, 180, 320))
        return (
            f"{option}|mode={mode}|y={y_band}|pending={pending}|"
            f"move={movement}|resource={resource}|surface={surface}|stale_goal={stale_goal}"
        )

    def _score_option(self, option: str, context: Dict[str, Any] | None = None) -> OptionScore:
        context_key = self._context_signature(option, context)
        context_bucket = self._context_stats.get(context_key)
        if context_bucket and int(context_bucket.get("total", 0)) >= self.min_context_total:
            return self._score_from_bucket(option, context_bucket, "option_context_history", context_key)
        stat = self._stats.get(option)
        if not stat or int(stat.get("total", 0)) < self.min_option_total:
            return OptionScore(option, self.default_score, 0, 0, "cold_start_rule_gate", context_key)
        return self._score_from_bucket(option, stat, "option_event_history", context_key)

    def select(
        self,
        candidates: List[Any],
        context: Dict[str, Any] | None = None,
    ) -> OptionSelection:
        if not candidates:
            return OptionSelection(None, False, [], "no_option_candidates")

        # Mandatory option: the hard rule gate is the sole trigger, so the
        # scheduler always executes it when present. The scheduler still
        # records context + Δ outcome, so it can learn to arbitrate once
        # multiple recovery options co-occur, but it never suppresses or
        # alters this one. This keeps trigger timing + execution effect
        # identical to the pre-encapsulation mechanism.
        for c in candidates:
            if getattr(c, "mandatory", False):
                name = str(getattr(c, "name", c))
                return OptionSelection(
                    name,
                    True,
                    [OptionScore(name, 1.0, 0, 0, "mandatory_rule_gate")],
                    "mandatory_option",
                )

        names = [str(getattr(c, "name", c)) for c in candidates]
        if not self.enabled:
            return OptionSelection(
                names[0],
                True,
                [OptionScore(names[0], 1.0, 0, 0, "disabled_passthrough")],
                "option_decisioner_disabled",
            )

        rankings = sorted(
            (self._score_option(name, context) for name in names),
            key=lambda score: score.score,
            reverse=True,
        )
        best = rankings[0]
        if best.score < self.min_score:
            return OptionSelection(
                best.option,
                True,
                rankings,
                (
                    "selected_by_rule_gate_low_history:"
                    f"{best.score:.3f}<{self.min_score:.3f}"
                ),
            )
        return OptionSelection(best.option, True, rankings, "selected_by_option_decisioner")

    def _append_event(self, event: Dict[str, Any]) -> None:
        event = _jsonable(event)
        event.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        os.makedirs(self.option_dir, exist_ok=True)
        with open(self.events_path, "a") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
            fcntl.flock(fp, fcntl.LOCK_UN)

    def record_invocation(
        self,
        option: str,
        context: Dict[str, Any],
        reason: Dict[str, Any],
        decision_trace: Dict[str, Any],
        before: Dict[str, Any],
        planned_ticks: int,
    ) -> str:
        event_id = str(uuid.uuid4())
        self._append_event(
            {
                "event_type": "option_invocation",
                "event_id": event_id,
                "option": option,
                "context": context,
                "rule_reason": reason,
                "decision_trace": decision_trace,
                "before": before,
                "planned_ticks": int(planned_ticks),
            }
        )
        return event_id

    def record_skip(
        self,
        context: Dict[str, Any],
        candidates: List[Any],
        selection: OptionSelection,
    ) -> None:
        self._append_event(
            {
                "event_type": "option_skip",
                "context": context,
                "candidates": [
                    {
                        "option": str(getattr(c, "name", c)),
                        "reason": getattr(c, "reason", {}),
                    }
                    for c in candidates
                ],
                "decision_trace": selection.to_trace(),
            }
        )

    def record_outcome(
        self,
        event_id: str,
        option: str,
        context: Dict[str, Any],
        before: Dict[str, Any],
        after: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        self._append_event(
            {
                "event_type": "option_outcome",
                "event_id": event_id,
                "option": option,
                "context": context,
                "before": before,
                "after": after,
                "outcome": outcome,
            }
        )
        bucket = self._stats.setdefault(option, {"success": 0, "total": 0})
        bucket["total"] += 1
        if bool(outcome.get("success")):
            bucket["success"] += 1
        context_key = self._context_signature(option, context)
        context_bucket = self._context_stats.setdefault(context_key, {"success": 0, "total": 0})
        context_bucket["total"] += 1
        if bool(outcome.get("success")):
            context_bucket["success"] += 1

    def stats(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self._stats.items()}

    def context_stats(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self._context_stats.items()}
