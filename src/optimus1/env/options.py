"""Environment option wrappers for XENON-plus recovery primitives.

The existing recovery code in ``wrapper.py`` already contains useful
environment-aware primitives. This module exposes them as option candidates so a
decisioner can explicitly schedule them and log their outcomes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class EnvOptionCandidate:
    name: str
    control_key: str
    recovery_event_key: str
    planned_ticks: int
    reason: Dict[str, Any]
    # Safety-critical / behaviour-preserving options. When True the
    # decisioner must execute the candidate whenever the hard rule gate
    # has made it legal (the rule gate is the *only* trigger condition).
    # This is what lets us route an existing mechanism through the option
    # scheduler for observability/credit-assignment without changing when
    # it fires or what it does.
    mandatory: bool = False


def build_option_context(env: Any, action: Dict[str, Any], goal: tuple[str, int] | None, prompt: str | None) -> Dict[str, Any]:
    horizontal, vertical = env._position_delta()
    status = env.status_mod.get_status() if getattr(env, "status_mod", None) else {}
    loc = status.get("location_stats", {}) or {}
    return {
        "step": int(getattr(env, "num_steps", 0)),
        "waypoint": str(goal[0]) if goal else "",
        "goal": list(goal) if goal else None,
        "prompt": prompt or "",
        "inventory_slots_used": int(env._used_inventory_slots()),
        "pending_relevant_drops": int(env._pending_relevant_drops(goal)),
        "stale_progress_ticks": int(env._stale_progress_ticks()),
        "stale_goal_progress_ticks": int(env._stale_goal_progress_ticks()),
        "movement_stagnant_ticks": int(env._control_state.get("movement_stagnant_ticks", 0)),
        "resource_stagnant_ticks": int(env._control_state.get("resource_stagnant_ticks", 0)),
        "surface_stuck_ticks": int(env._control_state.get("surface_stuck_ticks", 0)),
        "horizontal_delta": float(horizontal),
        "vertical_delta": float(vertical),
        "ypos": float(loc.get("ypos", 0) or 0),
        "is_surface_resource": bool(env._is_surface_resource_acquisition(goal, prompt)),
        "is_tunnel_resource": bool(env._is_tunnel_resource_acquisition(goal, prompt)),
        "is_resource_acquisition": bool(env._is_resource_acquisition(goal, prompt)),
    }


def build_env_option_candidates(
    env: Any,
    action: Dict[str, Any],
    goal: tuple[str, int] | None,
    prompt: str | None,
) -> List[EnvOptionCandidate]:
    """Return currently rule-triggered option candidates.

    The rules remain the hard safety gate. The option decisioner only sees
    candidates that are already considered legal and relevant by the existing
    hand-written checks.
    """

    candidates: List[EnvOptionCandidate] = []
    escape_idle = int(env._control_state.get("escape_ticks", 0)) <= 0
    collect_idle = int(env._control_state.get("collect_drop_ticks", 0)) <= 0
    surface_search_idle = int(env._control_state.get("surface_search_ticks", 0)) <= 0
    tunnel_idle = int(env._control_state.get("tunnel_recovery_ticks", 0)) <= 0
    surface_turn_idle = int(env._control_state.get("surface_turn_around_ticks", 0)) <= 0

    if escape_idle and env._should_surface_escape():
        candidates.append(
            EnvOptionCandidate(
                name="option:surface_escape",
                control_key="escape_ticks",
                recovery_event_key="surface_escape",
                planned_ticks=int(os.environ.get("XENON_ESCAPE_TICKS", "80")),
                reason={
                    "air": int(env._current_air()),
                    "health": float(env._current_health()),
                },
            )
        )

    if escape_idle and env._should_movement_escape(action, goal, prompt):
        candidates.append(
            EnvOptionCandidate(
                name="option:movement_escape",
                control_key="escape_ticks",
                recovery_event_key="movement_escape",
                planned_ticks=int(os.environ.get("XENON_ESCAPE_TICKS", "80")),
                reason={
                    "movement_stagnant_ticks": int(env._control_state.get("movement_stagnant_ticks", 0)),
                    "stale_progress_ticks": int(env._stale_progress_ticks()),
                },
            )
        )

    if escape_idle and collect_idle and surface_search_idle and env._should_collect_drops(action, goal, prompt):
        candidates.append(
            EnvOptionCandidate(
                name="option:collect_drops",
                control_key="collect_drop_ticks",
                recovery_event_key="collect_drops",
                planned_ticks=int(os.environ.get("XENON_COLLECT_DROPS_TICKS", "24")),
                reason={
                    "pending_relevant_drops": int(env._pending_relevant_drops(goal)),
                    "stale_goal_progress_ticks": int(env._stale_goal_progress_ticks()),
                },
            )
        )

    if escape_idle and collect_idle and surface_search_idle and env._should_surface_search(goal, prompt):
        candidates.append(
            EnvOptionCandidate(
                name="option:surface_search",
                control_key="surface_search_ticks",
                recovery_event_key="surface_search",
                planned_ticks=int(os.environ.get("XENON_SURFACE_SEARCH_TICKS", "90")),
                reason={
                    "stale_goal_progress_ticks": int(env._stale_goal_progress_ticks()),
                    "is_surface_resource": bool(env._is_surface_resource_acquisition(goal, prompt)),
                },
            )
        )

    if escape_idle and tunnel_idle and env._should_tunnel_recovery(action, goal, prompt):
        candidates.append(
            EnvOptionCandidate(
                name="option:tunnel_recovery",
                control_key="tunnel_recovery_ticks",
                recovery_event_key="tunnel_recovery",
                planned_ticks=int(os.environ.get("XENON_TUNNEL_RECOVERY_TICKS", "70")),
                reason={
                    "resource_stagnant_ticks": int(env._control_state.get("resource_stagnant_ticks", 0)),
                    "stale_progress_ticks": int(env._stale_progress_ticks()),
                    "is_tunnel_resource": bool(env._is_tunnel_resource_acquisition(goal, prompt)),
                },
            )
        )

    if escape_idle and collect_idle and surface_search_idle and surface_turn_idle and env._should_surface_turn_around(action, goal, prompt):
        horizontal, _ = env._position_delta()
        candidates.append(
            EnvOptionCandidate(
                name="option:surface_turn_around",
                control_key="surface_turn_around_ticks",
                recovery_event_key="surface_turn_around",
                planned_ticks=int(os.environ.get("XENON_SURFACE_TURNAROUND_TICKS", "30")),
                reason={
                    "horizontal_delta": float(horizontal),
                    "surface_stuck_ticks": int(env._control_state.get("surface_stuck_ticks", 0)),
                    "stale_goal_progress_ticks": int(env._stale_goal_progress_ticks()),
                },
            )
        )

    return candidates


def activate_env_option(env: Any, candidate: EnvOptionCandidate) -> None:
    """Activate a selected option by setting the wrapper's existing controls."""

    if candidate.name == "option:surface_turn_around":
        env._control_state["escape_turn"] *= -1
        env.cache["last_surface_turn_around_step"] = env.num_steps
        env.cache["last_goal_progress_step"] = env.num_steps
        env._control_state["attack_hold"] = 0
    elif candidate.name == "option:collect_drops":
        env._control_state["attack_hold"] = 0
        env.cache["last_collect_drop_step"] = env.num_steps
        env.cache["last_goal_progress_step"] = env.num_steps
    elif candidate.name == "option:surface_search":
        env._control_state["attack_hold"] = 0
        env.cache["last_surface_search_step"] = env.num_steps
        env.cache["last_goal_progress_step"] = env.num_steps

    env._control_state[candidate.control_key] = int(candidate.planned_ticks)
    events = env._control_state.setdefault("recovery_events", {})
    events[candidate.recovery_event_key] = int(events.get(candidate.recovery_event_key, 0)) + 1


__all__ = [
    "EnvOptionCandidate",
    "activate_env_option",
    "build_env_option_candidates",
    "build_option_context",
]
