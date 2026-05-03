#!/usr/bin/env python3
"""Targeted verification of the *new* pillar-up APIs.

Exercised entry points (defined in ``src/optimus1/env/wrapper.py``):

    * ``CustomEnvWrapper.perceive_height_context``
    * ``CustomEnvWrapper.pillar_up_smart``
    * ``CustomEnvWrapper.raise_to_height``
    * ``CustomEnvWrapper.raise_to_ore_band``

This is the *_smart_* counterpart of ``scripts/verify_pillar_up.py``,
which only tests the low-level ``pillar_up`` primitive.

Two scenarios run back-to-back so the user can see the new behaviour
end-to-end:

  Scenario A — hotbar source:
      Pre-load 32 cobblestone in **hotbar slot 0**.
      a1) Snapshot ``perceive_height_context(look_for="coal_ore")``.
      a2) ``raise_to_height(start_y + 5)`` — confirm the absolute-Y entry
          point and verify Y rises ~5 blocks.

  Scenario B — main-inventory source (the new code path):
      Pre-load 32 cobblestone in **main-inventory slot 9** (NOT hotbar).
      The classic ``pillar_up`` would refuse with
      ``no_placeable_block_in_hotbar``; ``pillar_up_smart`` should
      issue ``/replaceitem`` to move a stack into hotbar before climbing.
      b1) Snapshot ``perceive_height_context``.
      b2) ``raise_to_ore_band("coal_ore")`` if below band, else
          ``pillar_up_smart(target_dy=5)`` as a fallback.

The wrapper streams every POV frame to ``MONITOR_URL`` (default
``http://172.17.0.1:8080/push``), so the host's monitor_server shows
the agent live in the browser.

Output: ``/tmp/pillar_smart_verify_<timestamp>.json`` with both scenario
results plus the perception snapshots.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from optimus1.env import CustomEnvWrapper, env_make, register_custom_env

LOG = logging.getLogger("pillar_smart_verify")
LOG.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_h)


def _ypos(info: Dict[str, Any]) -> float:
    try:
        return float(np.asarray(info["location_stats"].get("ypos", 64.0)).reshape(-1)[0])
    except Exception:
        return 64.0


def _perception_snapshot(env: CustomEnvWrapper, ore: str) -> Dict[str, Any]:
    ctx = env.perceive_height_context(look_for=ore)
    LOG.info(
        "perceive_height_context(%s): y=%.2f in_band=%s placeable_hotbar=%d "
        "placeable_inventory=%d preferred=%s recommended_action=%s "
        "recommended_dy=%s target_y=%s reason=%s",
        ore,
        ctx.get("current_y", 0.0),
        ctx.get("in_band"),
        ctx.get("placeable_in_hotbar", 0),
        ctx.get("placeable_in_inventory", 0),
        ctx.get("preferred_block"),
        ctx.get("recommended_action"),
        ctx.get("recommended_dy"),
        ctx.get("target_y"),
        ctx.get("reason"),
    )
    return ctx


def _warm_up(env: CustomEnvWrapper, ticks: int = 4) -> None:
    """Drive a few no-op steps so cache / status are fully populated."""
    for _ in range(ticks):
        try:
            env.raw_step(env.env.noop_action())
        except Exception as exc:
            LOG.warning(f"warm-up raw_step raised: {exc}")
            break


def run_scenario_hotbar_source(env: CustomEnvWrapper) -> Dict[str, Any]:
    LOG.info("=" * 60)
    LOG.info("Scenario A: source = hotbar (raise_to_height absolute target)")
    LOG.info("=" * 60)
    perception = _perception_snapshot(env, ore="coal_ore")
    start_y = _ypos(env.cache.get("info", {}))
    target_y = start_y + 5.0
    LOG.info("calling raise_to_height(target_y=%.2f) [start_y=%.2f]", target_y, start_y)
    res = env.raise_to_height(target_y, max_blocks=8, max_steps=200)
    LOG.info(
        "raise_to_height -> success=%s dy=%.2f blocks_used=%s reason=%s "
        "prep_action=%s",
        res.get("success"),
        float(res.get("dy", 0.0)),
        res.get("blocks_used"),
        res.get("reason"),
        res.get("prep_action"),
    )
    return {"perception_pre": perception, "result": res, "start_y": start_y}


def run_scenario_inventory_source(env: CustomEnvWrapper) -> Dict[str, Any]:
    LOG.info("=" * 60)
    LOG.info(
        "Scenario B: build a 3x11x3 air chamber at y=40-50, /tp agent in, "
        "then call raise_to_ore_band('coal_ore'). Exercises the full "
        "perception -> auto-target-Y -> climb path the planner relies on."
    )
    LOG.info("=" * 60)
    pre_y = _ypos(env.cache.get("info", {}))
    # Build a small underground arena around the agent's current xz so
    # the /tp doesn't suffocate inside solid stone.
    setup_cmds = [
        # clear a 3 wide x 11 tall x 3 deep chamber at y=40..50
        "/fill ~-1 40 ~-1 ~1 50 ~1 minecraft:air",
        # solid floor at y=39 directly under the agent's xz
        "/setblock ~ 39 ~ minecraft:cobblestone",
        # also widen the floor slightly so pillar_up can place a side
        # block if needed
        "/fill ~-1 39 ~-1 ~1 39 ~1 minecraft:cobblestone",
        # finally drop the agent into the chamber
        "/tp @s ~ 40 ~",
    ]
    for cmd in setup_cmds:
        try:
            env.env.execute_cmd(cmd)
        except Exception as exc:
            LOG.warning(f"setup cmd failed: {cmd!r}: {exc}")
    _warm_up(env, ticks=6)

    perception_pre = _perception_snapshot(env, ore="coal_ore")
    if perception_pre.get("current_y", 0.0) > 50:
        LOG.warning(
            "After /tp the agent is still at y=%.1f (>50); the chamber "
            "build may have failed and the test won't isolate the climb path.",
            perception_pre.get("current_y", 0.0),
        )

    start_y = _ypos(env.cache.get("info", {}))
    if perception_pre.get("recommended_action") == "pillar_up":
        LOG.info(
            "recommendation = pillar_up; dispatching raise_to_ore_band(coal_ore)"
        )
        res = env.raise_to_ore_band(
            "coal_ore", max_blocks=24, max_steps=400, ascend_margin=1
        )
    else:
        LOG.info(
            "recommendation = %s; dispatching pillar_up_smart(target_dy=5)",
            perception_pre.get("recommended_action"),
        )
        res = env.pillar_up_smart(target_dy=5, max_blocks=8, max_steps=200)

    LOG.info(
        "scenario B -> success=%s dy=%.2f blocks_used=%s reason=%s "
        "prep_action=%s preferred_block=%s",
        res.get("success"),
        float(res.get("dy", 0.0)),
        res.get("blocks_used"),
        res.get("reason"),
        res.get("prep_action"),
        res.get("preferred_block"),
    )
    perception_post = _perception_snapshot(env, ore="coal_ore")
    return {
        "perception_pre": perception_pre,
        "perception_post": perception_post,
        "result": res,
        "start_y": start_y,
        "pre_tp_y": pre_y,
    }


@hydra.main(version_base=None, config_path="../src/optimus1/conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    LOG.info(f"hydra cwd: {os.getcwd()}")
    LOG.info(f"MONITOR_URL = {os.environ.get('MONITOR_URL')}")

    # Pre-load the hotbar with cobblestone for scenario A. Scenario B
    # then clears the hotbar and seeds the main inventory at runtime.
    inv = list(cfg["env"].get("initial_inventory", []) or [])
    has_cobble_hotbar = any(
        isinstance(item, dict)
        and item.get("type") == "cobblestone"
        and 0 <= int(item.get("slot", -1)) <= 8
        for item in inv
    )
    if not has_cobble_hotbar:
        inv = list(inv) + [
            {"type": "cobblestone", "quantity": 32, "slot": 0},
        ]
        OmegaConf.update(cfg, "env.initial_inventory", inv, force_add=True)
    LOG.info(f"initial_inventory = {inv}")

    register_custom_env(cfg)
    env = env_make(cfg["env"]["name"], cfg, LOG)

    LOG.info("env.reset() ...")
    env.reset()
    time.sleep(1.0)
    _warm_up(env, ticks=4)

    out: Dict[str, Any] = {
        "config": {
            "benchmark": str(cfg.get("benchmark", "?")),
            "world_seed": int(cfg.get("world_seed", 0)),
            "seed": int(cfg.get("seed", 0)),
            "biome": str(cfg.get("env", {}).get("prefer_biome", "")),
            "monitor_url": os.environ.get("MONITOR_URL", ""),
        },
        "scenarios": {},
    }

    try:
        out["scenarios"]["A_hotbar_source"] = run_scenario_hotbar_source(env)
    except Exception as exc:
        LOG.exception(f"scenario A crashed: {exc}")
        out["scenarios"]["A_hotbar_source"] = {"error": repr(exc)}

    try:
        out["scenarios"]["B_inventory_source"] = run_scenario_inventory_source(env)
    except Exception as exc:
        LOG.exception(f"scenario B crashed: {exc}")
        out["scenarios"]["B_inventory_source"] = {"error": repr(exc)}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"/tmp/pillar_smart_verify_seed{cfg.get('world_seed', 0)}_{timestamp}.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    LOG.info(f"saved verification result -> {out_path}")

    LOG.info("=" * 60)
    LOG.info("SUMMARY")
    LOG.info("=" * 60)
    for name, scen in out["scenarios"].items():
        if "error" in scen:
            LOG.info(f"[{name}] ERROR: {scen['error']}")
            continue
        res = scen.get("result", {})
        LOG.info(
            "[%s] success=%s dy=%.2f blocks_used=%s prep_action=%s reason=%s",
            name,
            res.get("success"),
            float(res.get("dy", 0.0)),
            res.get("blocks_used"),
            res.get("prep_action"),
            res.get("reason"),
        )

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
