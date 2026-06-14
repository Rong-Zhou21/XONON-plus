#!/usr/bin/env python3
"""Verify the lightweight environment option decisioner.

This script does not start MineRL. It exercises the decision path that matters
for the new XENON-plus option skills:

1. Rule-gated cold-start candidates are executable.
2. Option invocation/outcome records are written to case_memory/option_events.jsonl.
3. Reloaded history changes later decisions.
4. Failed option history changes the decision trace/ranking without hard
   disabling rule-gated skills.
"""

from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from optimus1.decisioner.option_selector import OptionDecisioner


def load_env_options_module():
    module_path = REPO_ROOT / "src" / "optimus1" / "env" / "options.py"
    spec = importlib.util.spec_from_file_location("xenon_env_options", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV_OPTIONS = load_env_options_module()
EnvOptionCandidate = ENV_OPTIONS.EnvOptionCandidate


def load_env_wrapper_module():
    module_path = REPO_ROOT / "src" / "optimus1" / "env" / "wrapper.py"

    if "gym" not in sys.modules:
        gym_stub = types.ModuleType("gym")

        class DummyWrapper:
            def __init__(self, env=None):
                self.env = env

            def reset(self):
                return self.env.reset()

            def step(self, action):
                return self.env.step(action)

        gym_stub.Wrapper = DummyWrapper
        sys.modules["gym"] = gym_stub

    if "omegaconf" not in sys.modules:
        omegaconf_stub = types.ModuleType("omegaconf")
        omegaconf_stub.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_stub

    env_pkg = types.ModuleType("optimus1.env")
    env_pkg.__path__ = [str(REPO_ROOT / "src" / "optimus1" / "env")]
    sys.modules["optimus1.env"] = env_pkg

    util_pkg = types.ModuleType("optimus1.util")
    util_pkg.__path__ = [str(REPO_ROOT / "src" / "optimus1" / "util")]
    sys.modules["optimus1.util"] = util_pkg
    server_api_stub = types.ModuleType("optimus1.util.server_api")

    class DummyMultiThreadServerAPI:
        pass

    server_api_stub.MultiThreadServerAPI = DummyMultiThreadServerAPI
    sys.modules["optimus1.util.server_api"] = server_api_stub

    spec = importlib.util.spec_from_file_location("optimus1.env.wrapper", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENV_WRAPPER = load_env_wrapper_module()


def candidate(name: str) -> EnvOptionCandidate:
    key = name.split(":", 1)[1]
    return EnvOptionCandidate(
        name=name,
        control_key=f"{key}_ticks",
        recovery_event_key=key,
        planned_ticks=5,
        reason={"rule": "verify"},
    )


class FakeEnv:
    def __init__(self) -> None:
        self.num_steps = 123
        self.cache: Dict[str, Any] = {}
        self._control_state: Dict[str, Any] = {
            "attack_hold": 9,
            "escape_turn": 1,
            "recovery_events": {},
        }


class FakeCommandEnv:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute_cmd(self, command: str) -> None:
        self.commands.append(command)


def context(*, tunnel: bool = False, surface: bool = False, y: float = 45.0, stagnant: int = 120) -> Dict[str, Any]:
    return {
        "step": 100,
        "waypoint": "iron_ore" if tunnel else "log",
        "goal": ["iron_ore" if tunnel else "log", 1],
        "prompt": "collect resource",
        "inventory_slots_used": 4,
        "pending_relevant_drops": 0,
        "stale_progress_ticks": stagnant + 200,
        "stale_goal_progress_ticks": stagnant + 200,
        "movement_stagnant_ticks": stagnant if not tunnel else 0,
        "resource_stagnant_ticks": stagnant if tunnel else 0,
        "surface_stuck_ticks": stagnant if surface else 0,
        "horizontal_delta": 0.1,
        "vertical_delta": 0.0,
        "ypos": y,
        "is_surface_resource": surface,
        "is_tunnel_resource": tunnel,
        "is_resource_acquisition": True,
    }


def record_outcome(
    decisioner: OptionDecisioner,
    option: str,
    ctx: Dict[str, Any],
    success: bool,
) -> None:
    event_id = decisioner.record_invocation(
        option,
        ctx,
        {"rule": "verify"},
        {"source": "verify"},
        {"step": 0, "x": 0.0, "y": ctx["ypos"], "z": 0.0},
        planned_ticks=5,
    )
    decisioner.record_outcome(
        event_id,
        option,
        ctx,
        {"step": 0, "x": 0.0, "y": ctx["ypos"], "z": 0.0},
        {"step": 5, "x": 1.0 if success else 0.0, "y": ctx["ypos"], "z": 0.0},
        {
            "success": success,
            "elapsed_ticks": 5,
            "goal_observed_delta": 1 if success else 0,
            "goal_mined_delta": 0,
            "pending_drop_delta": 0,
            "inventory_slot_delta": 0,
            "horizontal_delta": 1.0 if success else 0.0,
            "vertical_delta": 0.0,
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xenon_option_verify_") as tmp:
        cold_ctx = context(tunnel=True, y=22.0, stagnant=140)
        tunnel = candidate("option:tunnel_recovery")

        cold = OptionDecisioner(tmp, min_score=0.35, min_context_total=1, min_option_total=1)
        cold_selection = cold.select([tunnel], cold_ctx)
        assert cold_selection.execute is True
        assert cold_selection.option == "option:tunnel_recovery"
        assert cold_selection.rankings[0].source == "cold_start_rule_gate"

        record_outcome(cold, tunnel.name, cold_ctx, success=False)

        reloaded = OptionDecisioner(tmp, min_score=0.35, min_context_total=1, min_option_total=1)
        same_context_selection = reloaded.select([tunnel], cold_ctx)
        assert same_context_selection.execute is True
        assert same_context_selection.option == "option:tunnel_recovery"
        assert same_context_selection.rankings[0].source == "option_context_history"

        new_tunnel_ctx = context(tunnel=True, y=55.0, stagnant=360)
        new_context_selection = reloaded.select([tunnel], new_tunnel_ctx)
        assert new_context_selection.execute is True
        assert new_context_selection.option == "option:tunnel_recovery"
        assert new_context_selection.rankings[0].source == "option_event_history"

        surface_ctx = context(surface=True, y=64.0, stagnant=80)
        collect = candidate("option:collect_drops")
        record_outcome(reloaded, collect.name, surface_ctx, success=True)

        reloaded_again = OptionDecisioner(tmp, min_score=0.35, min_context_total=1, min_option_total=1)
        mixed_selection = reloaded_again.select([tunnel, collect], surface_ctx)
        assert mixed_selection.execute is True
        assert mixed_selection.option == "option:collect_drops"

        fake = FakeEnv()
        collect_runtime = candidate("option:collect_drops")
        collect_runtime.control_key = "collect_drop_ticks"
        ENV_OPTIONS.activate_env_option(fake, collect_runtime)
        assert fake._control_state["collect_drop_ticks"] == collect_runtime.planned_ticks
        assert fake._control_state["attack_hold"] == 0
        assert fake.cache["last_collect_drop_step"] == fake.num_steps
        assert fake._control_state["recovery_events"]["collect_drops"] == 1

        turn_runtime = candidate("option:surface_turn_around")
        turn_runtime.control_key = "surface_turn_around_ticks"
        turn_runtime.recovery_event_key = "surface_turn_around"
        ENV_OPTIONS.activate_env_option(fake, turn_runtime)
        assert fake._control_state["surface_turn_around_ticks"] == turn_runtime.planned_ticks
        assert fake._control_state["escape_turn"] == -1
        assert fake.cache["last_surface_turn_around_step"] == fake.num_steps

        ore_env = FakeCommandEnv()
        ore_map: Dict[Any, Any] = {}
        ENV_WRAPPER.random_ore(ore_env, ore_map, 20.0, thresold=0.0, xpos=10.2, zpos=-5.3)
        ENV_WRAPPER.random_ore(ore_env, ore_map, 20.0, thresold=0.0, xpos=10.2, zpos=-5.3)
        ENV_WRAPPER.random_ore(ore_env, ore_map, 20.0, thresold=0.0, xpos=30.2, zpos=-5.3)
        assert len(ore_env.commands) == 2
        assert ENV_WRAPPER._ore_map_key(20.0, 10.2, -5.3) != ENV_WRAPPER._ore_map_key(20.0, 30.2, -5.3)

        events_path = Path(tmp) / "case_memory" / "option_events.jsonl"
        with events_path.open("r") as fp:
            event_lines = [json.loads(line) for line in fp if line.strip()]

        summary = {
            "events_path": str(events_path),
            "event_count": len(event_lines),
            "cold_selection": cold_selection.to_trace(),
            "same_context_selection": same_context_selection.to_trace(),
            "new_context_selection": new_context_selection.to_trace(),
            "mixed_selection": mixed_selection.to_trace(),
            "option_stats": reloaded_again.stats(),
            "context_stat_count": len(reloaded_again.context_stats()),
            "runtime_activation_checks": {
                "collect_drop_ticks": fake._control_state["collect_drop_ticks"],
                "surface_turn_around_ticks": fake._control_state["surface_turn_around_ticks"],
                "escape_turn": fake._control_state["escape_turn"],
                "recovery_events": fake._control_state["recovery_events"],
            },
            "dynamic_ore_checks": {
                "commands": ore_env.commands,
                "ore_map_entries": len(ore_map),
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
