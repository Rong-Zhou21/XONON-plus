"""Perception-action capability suite.

This module is the *single, addressable* entry point for the
environment-perception + action capabilities the agent gained in this
round of work. It is the intended innovation point: instead of asking
the user to flip eight different env vars to opt in/out, they flip
*one* — :pyenv:`XENON_PERCEPTION_ACTION_SUITE` — and the suite cascades
sane defaults into every per-feature env var the wrapper / planner
reads at runtime.

Design contract
---------------

* The suite is *opt-out* via per-feature env vars. If the user has
  already exported ``XENON_ENABLE_TREE_EXPLORE=0`` and then sets
  ``XENON_PERCEPTION_ACTION_SUITE=1``, the per-feature override wins
  (the suite uses :py:func:`os.environ.setdefault`, never overwrites).
  This is what makes ablation studies tractable.

* The suite default is ``"1"`` (ON). To run a clean baseline that
  matches the pre-perception-action era, set
  ``XENON_PERCEPTION_ACTION_SUITE=0`` before launching.

* Every feature lives in *exactly one* place (an entry in
  :py:attr:`PerceptionActionSuite._FEATURES`). Adding a new
  perception-action feature should mean: (1) add the gate-check inside
  the wrapper / planner; (2) append a row here. Nothing else.

Example
-------
::

    # main_planning.py / main_exploration.py
    from .env.perception_action import PerceptionActionSuite
    PerceptionActionSuite.apply_from_env(logger)
    # ... rest of main() ...

After the call, every gated primitive in :py:mod:`optimus1.env.wrapper`
and :py:mod:`optimus1.main_planning` reads its own env var as usual,
but the *defaults* now reflect the suite's choice.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List


# ---------------------------------------------------------------- #
#  The suite                                                        #
# ---------------------------------------------------------------- #


@dataclass(frozen=True)
class _Feature:
    """Static metadata for one perception-action capability.

    Attributes:
        env_var: the per-feature env var the wrapper / planner reads.
        on_value: value to set when the suite is ON and the env var is
            unset by the user.
        off_value: value to set when the suite is OFF and the env var
            is unset by the user.
        group: free-text grouping for the summary log line.
        title: short description of what flipping this feature does.
    """

    env_var: str
    on_value: str
    off_value: str
    group: str
    title: str


class PerceptionActionSuite:
    """Single-toggle perception-action feature manager.

    Methods are class-level on purpose — no instance state. Call
    :py:meth:`apply_from_env` once at the very start of ``main()``
    (before :py:func:`env_make`), then read the per-feature env vars as
    usual everywhere else in the codebase.
    """

    # The master env var. ``"1"`` is the project's research default;
    # ``"0"`` reproduces the pre-perception-action baseline.
    SUITE_ENV_VAR = "XENON_PERCEPTION_ACTION_SUITE"
    SUITE_DEFAULT = "1"

    # Single source of truth for which features the suite controls.
    # Order is purely cosmetic (used for the summary log).
    _FEATURES: List[_Feature] = [
        # --- Group 1: underground perception (this round) ----------------
        _Feature(
            env_var="XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT",
            on_value="1",
            off_value="0",
            group="underground",
            title=(
                "On overshoot (deeper ore observed during a target-ore "
                "sub-goal): pillar up to mid-band Y + scripted 3-block "
                "tunnel before flipping STEVE-1 to dig-forward."
            ),
        ),
        _Feature(
            env_var="XENON_ENABLE_PILLAR_UP_FOR_ORE",
            on_value="0",
            off_value="0",
            group="underground",
            title=(
                "Pre-mining height check: at sub-goal start, if agent is "
                "below the target ore band, pillar up before STEVE-1 "
                "starts. Off-by-default even within the suite (high cost "
                "for low-information benefit)."
            ),
        ),
        # --- Group 2: surface exploration -------------------------------
        _Feature(
            env_var="XENON_ENABLE_LOW_AIR_ESCAPE",
            on_value="1",
            off_value="0",
            group="surface",
            title=(
                "When O2 < threshold the agent triggers a sprint-jump "
                "escape primitive (climbs back out of water)."
            ),
        ),
        _Feature(
            env_var="XENON_ENABLE_SURFACE_SEARCH_PRIMITIVE",
            on_value="0",
            off_value="0",
            group="surface",
            title=(
                "Sprint-search primitive when surface mining is "
                "stagnant and resource ledger has no progress. "
                "Off-by-default within the suite (interferes with chop)."
            ),
        ),
        _Feature(
            env_var="XENON_ENABLE_TREE_EXPLORE",
            on_value="1",
            off_value="0",
            group="surface",
            title=(
                "When chop is stagnant the planner switches STEVE-1's "
                "prompt to 'find a tree' and back. New master gate; "
                "previously this fired unconditionally."
            ),
        ),
        # --- Group 3: inventory perception ------------------------------
        _Feature(
            env_var="XENON_ENABLE_INVENTORY_CLEANUP",
            on_value="1",
            off_value="0",
            group="inventory",
            title=(
                "Drop low-priority items when hotbar-slot pressure is "
                "high so the agent does not silently fail to pick up "
                "newly mined ores. New master gate; previously the "
                "wrapper fired this unconditionally."
            ),
        ),
        _Feature(
            env_var="XENON_ENABLE_COLLECT_DROPS",
            on_value="1",
            off_value="0",
            group="inventory",
            title=(
                "When mined ores have been broken but not picked up "
                "(falling-block / out-of-reach drops) the agent walks "
                "back over them. New master gate."
            ),
        ),
        # --- Group 4: stagnation recovery -------------------------------
        _Feature(
            env_var="XENON_ENABLE_MOVEMENT_ESCAPE",
            on_value="1",
            off_value="0",
            group="stagnation",
            title=(
                "Sprint-jump-rotate escape when horizontal motion has "
                "been stagnant for N ticks. New master gate."
            ),
        ),
        _Feature(
            env_var="XENON_ENABLE_TUNNEL_RECOVERY",
            on_value="1",
            off_value="0",
            group="stagnation",
            title=(
                "When mining a tunnel target ore and progress has been "
                "stagnant, force a forward-attack-rotate primitive to "
                "punch through the wall the policy is stuck against. "
                "New master gate."
            ),
        ),
    ]

    # ---- public API ------------------------------------------------ #

    @classmethod
    def is_on(cls) -> bool:
        """Return True if the suite is currently enabled."""
        return os.environ.get(cls.SUITE_ENV_VAR, cls.SUITE_DEFAULT) == "1"

    @classmethod
    def apply_from_env(cls, logger: logging.Logger | None = None) -> dict:
        """Cascade suite default into each per-feature env var.

        Per-feature env vars already exported by the user are *not*
        overwritten — this is what makes ablation studies sane.

        Returns a dict ``{env_var: effective_value}`` for the summary log.
        """
        suite_on = cls.is_on()
        applied: dict[str, str] = {}
        for feat in cls._FEATURES:
            target = feat.on_value if suite_on else feat.off_value
            # setdefault: don't overwrite explicit user choice
            os.environ.setdefault(feat.env_var, target)
            applied[feat.env_var] = os.environ[feat.env_var]

        if logger is not None:
            cls._log_summary(logger, suite_on, applied)
        return applied

    @classmethod
    def features(cls) -> Iterable[_Feature]:
        """Iterate all features the suite controls (for inspection)."""
        return tuple(cls._FEATURES)

    # ---- internals ------------------------------------------------- #

    @classmethod
    def _log_summary(
        cls,
        logger: logging.Logger,
        suite_on: bool,
        applied: dict[str, str],
    ) -> None:
        # Group features for readability.
        by_group: dict[str, list[_Feature]] = {}
        for feat in cls._FEATURES:
            by_group.setdefault(feat.group, []).append(feat)

        state = "ON" if suite_on else "OFF"
        logger.info(
            f"[perception_action_suite] {cls.SUITE_ENV_VAR}={state} "
            f"(set XENON_PERCEPTION_ACTION_SUITE=0 for baseline)"
        )
        for group, feats in by_group.items():
            for feat in feats:
                effective = applied.get(feat.env_var, "?")
                # 1 = on, 0 = off — no other values produced by this suite,
                # but user could have exported anything; surface it as-is.
                logger.info(
                    f"[perception_action_suite]   {group:<12} "
                    f"{feat.env_var}={effective}"
                )


__all__ = ["PerceptionActionSuite"]
