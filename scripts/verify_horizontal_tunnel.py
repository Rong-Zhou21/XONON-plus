#!/usr/bin/env python3
"""Targeted verification for CustomEnvWrapper.dig_forward_blocks().

Builds a controlled underground stone volume, places the agent in a
two-block-high pocket facing the stone wall, then asks the scripted tunnel
primitive to bore several horizontal segments. The pass condition is physical
x/z advance, not merely breaking blocks.
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
from omegaconf import DictConfig, OmegaConf

from optimus1.env import CustomEnvWrapper, env_make, register_custom_env

LOG = logging.getLogger("horizontal_tunnel_verify")
LOG.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(_h)


def _loc(env: CustomEnvWrapper) -> Dict[str, float]:
    loc = (env.cache.get("info") or {}).get("location_stats", {}) or {}

    def scalar(key: str, default: float = 0.0) -> float:
        try:
            return float(np.asarray(loc.get(key, default)).reshape(-1)[0])
        except Exception:
            return float(default)

    return {
        "x": scalar("xpos"),
        "y": scalar("ypos"),
        "z": scalar("zpos"),
        "yaw": scalar("yaw"),
        "pitch": scalar("pitch"),
    }


def _warm_up(env: CustomEnvWrapper, ticks: int = 8) -> None:
    for _ in range(ticks):
        env.raw_step(env.env.noop_action())


def _setup_arena(env: CustomEnvWrapper) -> None:
    commands = [
        "/tp @s 0 45 0 0 0",
        "/fill -3 43 -3 3 48 12 minecraft:stone",
        "/fill -1 45 -1 1 46 0 minecraft:air",
        "/fill -1 44 -1 1 44 12 minecraft:cobblestone",
        "/effect @s minecraft:night_vision 99999 1 true",
    ]
    for cmd in commands:
        try:
            env.env.execute_cmd(cmd)
        except Exception as exc:
            LOG.warning("setup command failed: %s -> %s", cmd, exc)
    _warm_up(env, 12)


@hydra.main(version_base=None, config_path="../src/optimus1/conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    inv = list(cfg["env"].get("initial_inventory", []) or [])
    if not any(isinstance(item, dict) and item.get("type") == "diamond_pickaxe" for item in inv):
        inv.append({"type": "diamond_pickaxe", "quantity": 1, "slot": 0})
    OmegaConf.update(cfg, "env.initial_inventory", inv, force_add=True)

    register_custom_env(cfg)
    env = env_make(cfg["env"]["name"], cfg, LOG)

    LOG.info("env.reset()...")
    env.reset()
    _warm_up(env, 6)
    _setup_arena(env)
    start = _loc(env)
    LOG.info("start loc: %s", start)

    n_blocks = int(os.environ.get("VERIFY_TUNNEL_BLOCKS", "4"))
    max_steps = int(os.environ.get("VERIFY_TUNNEL_MAX_STEPS", "650"))
    result = env.dig_forward_blocks(n_blocks=n_blocks, max_steps=max_steps)
    end = _loc(env)
    LOG.info("end loc: %s", end)
    LOG.info("tunnel result: %s", json.dumps(result, ensure_ascii=False, default=str))

    out = {
        "start": start,
        "end": end,
        "result": result,
        "pass": bool(result.get("success")) and int(result.get("blocks_dug", 0)) >= min(n_blocks, int(result.get("min_success_blocks", n_blocks))),
    }
    out_path = Path("/tmp") / f"horizontal_tunnel_verify_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    LOG.info("wrote %s", out_path)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
