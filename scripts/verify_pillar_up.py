#!/usr/bin/env python3
"""Targeted verification of CustomEnvWrapper.pillar_up().

What it does (one episode):
    1. Build the MineRL env exactly the way main_planning does.
    2. Phase A — dig down: pitch the camera straight down and hold attack
       for N ticks, falling through the world while harvesting cobblestone
       and dirt naturally (no STEVE-1, no planner, no decisioner).
    3. When inventory contains >= MIN_BLOCKS placeable items, stop digging.
    4. Phase B — pillar up: call env.pillar_up(target_dy=...).
    5. Print start_y / end_y / blocks_used / reason and dump full result
       JSON to /tmp/pillar_verify_<seed>_<timestamp>.json.

The agent's frame stream is already wired into monitor_server via
wrapper._push_pov_to_monitor, so opening http://<host>:8080/ in a browser
will show the run live as long as monitor_server is up.

Usage:
    # inside container, after monitor_server is running on the host:
    xvfb-run -a python -u scripts/verify_pillar_up.py \\
        benchmark=stone seed=0 world_seed=10 server.port=9100 \\
        env.times=1 env.max_minutes=10 \\
        prefix=ours_planning exp_num=70001
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
from omegaconf import DictConfig

from optimus1.env import CustomEnvWrapper, env_make, register_custom_env

LOG = logging.getLogger("pillar_verify")
LOG.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_h)

# ---------- knobs ----------
DIG_MAX_STEPS = 1500          # cap for the dig-down phase
DIG_MIN_BLOCKS = 8            # stop digging when we have >= this many placeable blocks
PILLAR_TARGET_DY = 20         # rise this much in pillar-up
PILLAR_MAX_BLOCKS = 24
PILLAR_MAX_STEPS = 400
# ---------------------------


def _placeable_count_in_hotbar(env_status: Dict[str, Any]) -> int:
    inv = env_status.get("plain_inventory", {}) or {}
    keep = {
        "cobblestone", "cobbled_deepslate", "stone", "dirt", "andesite",
        "diorite", "granite", "sand", "gravel",
    }
    total = 0
    for slot, item in inv.items():
        try:
            si = int(slot)
        except Exception:
            continue
        if not (0 <= si <= 8):
            continue
        name = (item.get("type", "") or "").split(":")[-1]
        if name in keep:
            total += int(item.get("quantity", 0) or 0)
    return total


def _ypos(info: Dict[str, Any]) -> float:
    try:
        return float(info["location_stats"].get("ypos", 64.0))
    except Exception:
        return 64.0


def _pitch(info: Dict[str, Any]) -> float:
    try:
        return float(info["location_stats"].get("pitch", 0.0))
    except Exception:
        return 0.0


def dig_down_phase(env: CustomEnvWrapper, max_steps: int, min_blocks: int) -> Dict[str, Any]:
    """Tilt down, then hold attack while moving slightly forward to fall through.

    Returns a small summary dict.
    """
    LOG.info("=== Phase A: dig down ===")
    start_status = env.get_status()
    start_y = _ypos(start_status)
    LOG.info(f"start_y={start_y:.1f}, blocks={_placeable_count_in_hotbar(start_status)}")

    # 1. orient pitch to ~+85 (looking almost straight down)
    for _ in range(12):
        action = env.env.noop_action()
        cur_pitch = _pitch(env.get_status())
        if cur_pitch >= 80:
            break
        action["camera"] = np.array([min(90 - cur_pitch, 12), 0])
        env.raw_step(action)

    # 2. attack loop — break the block under feet, fall, repeat
    steps_used = 0
    blocks_collected = _placeable_count_in_hotbar(env.get_status())
    while steps_used < max_steps:
        action = env.env.noop_action()
        action["attack"] = np.array(1)
        # tiny camera correction every step so pitch stays down
        cur_pitch = _pitch(env.get_status())
        if cur_pitch < 85:
            action["camera"] = np.array([min(90 - cur_pitch, 5), 0])
        env.raw_step(action)
        steps_used += 1

        if steps_used % 60 == 0:
            status = env.get_status()
            cur_y = _ypos(status)
            blocks_collected = _placeable_count_in_hotbar(status)
            LOG.info(
                f"  dig step {steps_used}: y={cur_y:.1f} blocks={blocks_collected}"
            )
            if blocks_collected >= min_blocks and cur_y < start_y - 4:
                break

    end_status = env.get_status()
    end_y = _ypos(end_status)
    blocks_collected = _placeable_count_in_hotbar(end_status)
    LOG.info(
        f"end Phase A: dy={end_y - start_y:+.1f} (start={start_y:.1f} end={end_y:.1f}) "
        f"blocks={blocks_collected} steps={steps_used}"
    )
    return {
        "start_y": start_y,
        "end_y": end_y,
        "blocks": blocks_collected,
        "steps": steps_used,
    }


def pillar_phase(env: CustomEnvWrapper) -> Dict[str, Any]:
    LOG.info("=== Phase B: pillar up ===")
    pre_status = env.get_status()
    pre_y = _ypos(pre_status)
    pre_blocks = _placeable_count_in_hotbar(pre_status)
    LOG.info(f"pre-pillar: y={pre_y:.1f} blocks={pre_blocks}")

    result = env.pillar_up(
        target_dy=PILLAR_TARGET_DY,
        max_blocks=PILLAR_MAX_BLOCKS,
        max_steps=PILLAR_MAX_STEPS,
    )

    LOG.info(
        f"pillar_up: dy={result['dy']:+.1f} blocks_used={result['blocks_used']} "
        f"steps_used={result['steps_used']} reason={result['reason']} "
        f"success={result['success']}"
    )
    return result


@hydra.main(version_base=None, config_path="../src/optimus1/conf", config_name="evaluate")
def main(cfg: DictConfig):
    LOG.info(f"hydra cwd: {os.getcwd()}")
    register_custom_env(cfg)
    env = env_make(cfg["env"]["name"], cfg, LOG)

    LOG.info("env.reset()...")
    env.reset()
    time.sleep(1.0)

    # Phase A: dig down
    dig_summary = dig_down_phase(env, DIG_MAX_STEPS, DIG_MIN_BLOCKS)

    # Phase B: pillar up
    pillar_result = pillar_phase(env)

    out = {
        "config": {
            "benchmark": str(cfg.get("benchmark", "?")),
            "seed": int(cfg.get("seed", 0)),
            "world_seed": int(cfg.get("world_seed", 0)),
            "biome": str(cfg.get("env", {}).get("prefer_biome", "")),
            "dig_max_steps": DIG_MAX_STEPS,
            "dig_min_blocks": DIG_MIN_BLOCKS,
            "pillar_target_dy": PILLAR_TARGET_DY,
            "pillar_max_blocks": PILLAR_MAX_BLOCKS,
            "pillar_max_steps": PILLAR_MAX_STEPS,
        },
        "dig_phase": dig_summary,
        "pillar_phase": pillar_result,
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"/tmp/pillar_verify_seed{cfg.get('world_seed',0)}_{timestamp}.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, default=str)
    LOG.info(f"saved verification result -> {out_path}")
    LOG.info("=== SUMMARY ===")
    LOG.info(json.dumps({
        "dig_dy": dig_summary["end_y"] - dig_summary["start_y"],
        "pillar_dy": pillar_result["dy"],
        "pillar_success": pillar_result["success"],
        "pillar_reason": pillar_result["reason"],
        "blocks_used": pillar_result["blocks_used"],
    }, indent=2))

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
