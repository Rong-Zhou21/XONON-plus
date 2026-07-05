import copy
import logging
import math
import os
import random
import threading
import time
from collections import deque
from io import BytesIO
from typing import Any, Deque, Dict, List, Tuple

import gym
import numpy as np
from omegaconf import DictConfig

from ..decisioner.option_selector import OptionDecisioner
from ..util.server_api import MultiThreadServerAPI

from .mods import RecorderMod, StatusMod, TaskCheckerMod
from .options import (
    EnvOptionCandidate,
    activate_env_option,
    build_env_option_candidates,
    build_option_context,
)


# ── Monitor push: 把最新 POV 异步推送到宿主机 monitor_server ──
# 参考 Optimus-3 的 gui_server.py 设计：agent 运行时把每一帧 POV 推给前端。
#
# 用法（容器内运行实验时）:
#     export MONITOR_URL="http://172.17.0.1:8080/push"   # 宿主机 docker0 网关
#     export MONITOR_FPS=15                              # 推送帧率上限
# 禁用:
#     export MONITOR_URL=""
_MONITOR_URL = os.environ.get("MONITOR_URL", "http://172.17.0.1:8080/push")
try:
    _MONITOR_FPS = max(1, int(os.environ.get("MONITOR_FPS", "15")))
except ValueError:
    _MONITOR_FPS = 15

_monitor_latest_pov = None
_monitor_lock = threading.Lock()
_monitor_thread_started = False


def _monitor_pusher_worker():
    """后台线程：按固定节奏编码 POV 并 POST 到宿主机 monitor_server。

    永不抛异常，失败时静默（不影响实验）。
    """
    import urllib.request
    try:
        from PIL import Image
    except Exception:
        return  # 没有 PIL 就放弃推送

    interval = 1.0 / _MONITOR_FPS
    while True:
        time.sleep(interval)
        if not _MONITOR_URL:
            continue
        with _monitor_lock:
            pov = _monitor_latest_pov
        if pov is None:
            continue
        try:
            arr = pov
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            img = Image.fromarray(arr)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75)
            data = buf.getvalue()
            req = urllib.request.Request(
                _MONITOR_URL, data=data, method="POST",
                headers={"Content-Type": "image/jpeg",
                         "Content-Length": str(len(data))})
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            # 静默：monitor_server 没起、网络不通、JPEG 编码失败都不要影响实验
            pass


def _push_pov_to_monitor(observation):
    """在 env.step 之后调用：把 POV 放入共享变量，由后台线程异步推送。O(1)，不阻塞。"""
    global _monitor_latest_pov, _monitor_thread_started
    if not _MONITOR_URL:
        return
    if not isinstance(observation, dict):
        return
    pov = observation.get("pov", None)
    if pov is None or not hasattr(pov, "dtype"):
        return
    with _monitor_lock:
        _monitor_latest_pov = pov
    if not _monitor_thread_started:
        _monitor_thread_started = True
        t = threading.Thread(target=_monitor_pusher_worker, daemon=True,
                             name="xenon-monitor-pusher")
        t.start()


_DYNAMIC_ORE_MULTIPLIER_ENV = {
    "coal_ore": "XENON_RANDOM_ORE_COAL_MULTIPLIER",
    "iron_ore": "XENON_RANDOM_ORE_IRON_MULTIPLIER",
    "gold_ore": "XENON_RANDOM_ORE_GOLD_MULTIPLIER",
    "redstone_ore": "XENON_RANDOM_ORE_REDSTONE_MULTIPLIER",
    "diamond_ore": "XENON_RANDOM_ORE_DIAMOND_MULTIPLIER",
}
_RARE_DYNAMIC_ORES = {"gold_ore", "redstone_ore", "diamond_ore"}


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _dynamic_ore_multiplier(ore_name: str) -> float:
    default = _env_float("XENON_RANDOM_ORE_RARE_MULTIPLIER", 1.0) if ore_name in _RARE_DYNAMIC_ORES else 1.0
    env_var = _DYNAMIC_ORE_MULTIPLIER_ENV.get(ore_name)
    if env_var is None:
        return default
    return _env_float(env_var, default)


def _should_place_dynamic_ore(ore_name: str, thresold: float) -> bool:
    place_chance = max(0.0, min(1.0, 1.0 - thresold))
    place_chance = min(1.0, place_chance * _dynamic_ore_multiplier(ore_name))
    return random.random() > 1.0 - place_chance


def _block_coord(value: float) -> int:
    return int(math.floor(float(value)))


def _ore_map_key(ypos: float, xpos: float | None = None, zpos: float | None = None) -> Tuple[Any, ...]:
    y_key = _block_coord(ypos)
    if xpos is None or zpos is None:
        return ("global", y_key)
    return (_block_coord(xpos), y_key, _block_coord(zpos))


def random_ore(
    env,
    ORE_MAP,
    ypos: float,
    thresold: float = 0.9,
    xpos: float | None = None,
    zpos: float | None = None,
    prob_scale: float = 1.0,
):
    # Global spawn-probability scale (e.g. halved after the agent has
    # triggered the underground pillar-up skill). Applied multiplicatively
    # on top of the per-ore chance: effective = prob_scale * per_ore_chance.
    # Guarded so prob_scale == 1.0 consumes no RNG (identical to original).
    if prob_scale < 1.0 and random.random() >= max(0.0, prob_scale):
        return
    dy = random.randint(-5, -3)
    new_pos = int(ypos + dy)
    current_key = _ore_map_key(ypos, xpos, zpos)
    new_key = _ore_map_key(new_pos, xpos, zpos)
    if 45 <= ypos <= 50:  # max: 6
        # coal_ore
        if (
            current_key not in ORE_MAP
            and new_key not in ORE_MAP
            and new_pos >= 45
            and _should_place_dynamic_ore("coal_ore", thresold)
        ):
            ORE_MAP[new_key] = "coal_ore"
            ORE_MAP[current_key] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:coal_ore".format(dy))
            print(f"coal ore at {new_pos}")
    elif 26 <= ypos <= 43:  # max: 17
        if (
            current_key not in ORE_MAP
            and new_key not in ORE_MAP
            and new_pos >= 26
            and _should_place_dynamic_ore("iron_ore", thresold)
        ):
            ORE_MAP[new_key] = "iron_ore"
            ORE_MAP[current_key] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:iron_ore".format(dy))
            print(f"iron ore at {new_pos}")

    elif 14 < ypos <= 26:
        if (
            current_key not in ORE_MAP
            and new_key not in ORE_MAP
            and new_pos >= 17
            and _should_place_dynamic_ore("gold_ore", thresold)
        ):  # max: 10
            ORE_MAP[new_key] = "gold_ore"
            ORE_MAP[current_key] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:gold_ore".format(dy))
            print(f"gold ore at {new_pos}")
        elif (
            current_key not in ORE_MAP
            and new_key not in ORE_MAP
            and new_pos <= 16
            and _should_place_dynamic_ore("redstone_ore", thresold)
        ):  # max:12
            ORE_MAP[new_key] = "redstone_ore"
            ORE_MAP[current_key] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:redstone_ore".format(dy))
            print(f"redstone ore at {new_pos}")
    elif (
        ypos <= 14
        and current_key not in ORE_MAP
        and new_key not in ORE_MAP
        and new_pos >= 1
        and _should_place_dynamic_ore("diamond_ore", thresold)
    ):  # max: 14
        ORE_MAP[new_key] = "diamond_ore"
        ORE_MAP[current_key] = 1
        env.execute_cmd("/setblock ~ ~{} ~ minecraft:diamond_ore".format(dy))
        print(f"diamond ore at {new_pos}")


class BasaltTimeoutWrapper(gym.Wrapper):
    """Timeout wrapper specifically crafted for the BASALT environments"""

    def __init__(self, env):
        super().__init__(env)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        return super().reset()

    def step(self, action):
        observation, reward, done, info = super().step(action)
        _push_pov_to_monitor(observation)
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        return observation, reward, done, info


class CustomEnvWrapper(gym.Wrapper):
    _api_thread: MultiThreadServerAPI | None

    can_change_hotbar: bool = False
    can_open_inventory: bool = False

    cache: Dict[str, Any]

    logger: logging.Logger
    cfg: DictConfig

    _only_once: bool = False

    def __init__(self, env, cfg: DictConfig, logger: logging.Logger):
        super().__init__(env)
        self._current_task_finish = False

        self.cfg = cfg
        self.logger = logger
        option_cfg = (
            cfg.get("memory", {})
            .get("case_memory", {})
            .get("option_decisioner", {})
            or {}
        )
        self.option_decisioner = OptionDecisioner(
            str(cfg["memory"]["path"]),
            logger,
            enabled=bool(option_cfg.get("enabled", True)),
            min_score=float(option_cfg.get("min_score", 0.35)),
            default_score=float(option_cfg.get("default_score", 0.65)),
            prior_success=float(option_cfg.get("prior_success", 1.0)),
            prior_total=float(option_cfg.get("prior_total", 2.0)),
            min_context_total=int(option_cfg.get("min_context_total", 3)),
            min_option_total=int(option_cfg.get("min_option_total", 3)),
        )
        self._active_option_events: Dict[str, Dict[str, Any]] = {}
        self._last_option_skip_steps: Dict[str, int] = {}

        self.record_mod = RecorderMod(cfg["record"], logger)
        self.status_mod = StatusMod(cfg, logger)
        self.task_checker_mod = TaskCheckerMod(cfg)

    @property
    def current_subgoal_finish(self):
        return self._current_task_finish

    @property
    def current_task_finish(self):
        return self._current_task_finish

    def reset(self):
        self.ORE_MAP = {}
        self._current_task_finish = False
        self._api_thread = None

        self.record_mod.reset()
        self.status_mod.reset()
        self.task_checker_mod.reset()

        self.cache = {}
        self.cache["task"] = ""
        self.cache["ypos"] = {}
        self.cache["last_life_stats"] = {}
        self.cache["last_progress_step"] = 0
        self.cache["last_goal_progress_step"] = 0
        self.cache["last_target_block_step"] = 0
        self.cache["last_surface_attack_step"] = -1000000
        self.cache["surface_attack_streak"] = 0
        self.cache["last_surface_log_step"] = -1000000
        self.cache["last_surface_search_step"] = -1000000
        self.cache["last_surface_turn_around_step"] = -1000000
        self.cache["last_collect_drop_step"] = -1000000
        self.cache["position_window"] = deque(maxlen=120)
        self.cache["resource_ledger"] = {
            "last_inventory": {},
            "last_pickup_stats": {},
            "last_mine_block_stats": {},
            "max_inventory": {},
            "collected": {},
            "pickup": {},
            "mined_blocks": {},
        }

        self._control_state = {
            "attack_hold": 0,
            "escape_ticks": 0,
            "escape_turn": 1,
            "tunnel_recovery_ticks": 0,
            "surface_search_ticks": 0,
            "surface_turn_around_ticks": 0,
            "surface_stuck_ticks": 0,
            "collect_drop_ticks": 0,
            "surface_log_jump_lock_ticks": 0,
            "resource_stagnant_ticks": 0,
            "movement_stagnant_ticks": 0,
            "last_prompt": None,
            "last_health": None,
            "last_position": None,
            "policy_reset_requested": False,
            "recovery_events": {
                "surface_escape": 0,
                "surface_search": 0,
                "surface_turn_around": 0,
                "ground_pitch_clamp": 0,
                "collect_drops": 0,
                "movement_escape": 0,
                "tunnel_recovery": 0,
                "respawn_reset": 0,
                "checker_error": 0,
                "inventory_cleanup": 0,
                "inventory_cleanup_blocked": 0,
                "surface_log_jump_lock": 0,
            },
        }
        self._active_option_events = {}
        self._last_option_skip_steps = {}

        self._only_once = False

        # ====设置spawn point & env seed ==========
        commands: List[str] = self.cfg["commands"]

        obs = self.env.reset()
        import os as _os
        skip_commands = _os.environ.get("SKIP_RESET_COMMANDS", "0") == "1"
        if commands and not skip_commands:
            for cmd in commands:
                try:
                    self.env.execute_cmd(cmd)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"execute_cmd({cmd!r}) failed: {e}; aborting remaining commands."
                    )
                    # Once a cmd fails, env may be in done=True state; stop sending further cmds
                    break
        return obs

    def _action_scalar(self, action: Dict[str, Any], key: str) -> float:
        value = action.get(key, 0)
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return 0.0
            return float(arr.reshape(-1)[0])
        except Exception:
            return float(value or 0)

    def _set_button(self, action: Dict[str, Any], key: str, value: int) -> None:
        action[key] = np.array(value)

    def _button_down(self, action: Dict[str, Any], key: str) -> bool:
        return self._action_scalar(action, key) > 0

    def _action_fingerprint(self, action: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("attack", "use", "jump", "forward", "back", "left", "right", "sprint", "sneak", "drop", "inventory")
        data = {key: int(self._button_down(action, key)) for key in keys}
        camera = action.get("camera")
        try:
            data["camera"] = np.asarray(camera).reshape(-1).astype(float).round(3).tolist()
        except Exception:
            data["camera"] = []
        return data

    def _maybe_log_action_debug(
        self,
        stage: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> None:
        if os.environ.get("XENON_ACTION_DEBUG", "0") != "1":
            return
        before_fp = self._action_fingerprint(before)
        after_fp = self._action_fingerprint(after)
        interval = int(os.environ.get("XENON_ACTION_DEBUG_INTERVAL", "200"))
        changed = before_fp != after_fp
        if not changed and interval > 0 and self.num_steps % interval != 0:
            return
        self.logger.info(
            "Action debug: "
            f"stage={stage}, step={self.num_steps}, prompt={prompt}, goal={goal}, "
            f"before={before_fp}, after={after_fp}"
        )

    def _normalise_action(self, action: Dict[str, Any] | List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(action, list):
            if not action:
                return self.env.noop_action()
            return action[-1]
        return action

    def _prompt_text(self, goal: tuple[str, int] | None, prompt: str | None) -> str:
        parts = [prompt or ""]
        if goal:
            parts.append(str(goal[0]))
        return " ".join(parts).lower()

    def _is_resource_acquisition(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        text = self._prompt_text(goal, prompt)
        return any(token in text for token in ("mine", "dig", "break", "chop", "punch"))

    def _is_surface_resource_acquisition(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        text = self._prompt_text(goal, prompt)
        return any(token in text for token in ("chop", "punch", "tree", "log", "logs", "wood"))

    def _is_tunnel_resource_acquisition(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        text = self._prompt_text(goal, prompt)
        surface_tokens = ("chop", "punch", "tree", "log", "logs", "wood")
        if any(token in text for token in surface_tokens):
            return False
        return any(token in text for token in ("mine", "dig", "ore", "cobblestone", "stone", "diamond", "redstone"))

    def _lock_jump_during_attack(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        if self._is_tunnel_resource_acquisition(goal, prompt):
            return True
        return self._surface_log_jump_lock_active(goal, prompt)

    def _lock_full_movement_during_attack(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        return self._is_tunnel_resource_acquisition(goal, prompt)

    def _is_log_item(self, item_type: str) -> bool:
        item_type = self._plain_item_type(item_type)
        return item_type == "log" or item_type.endswith("_log")

    def _log_delta_total(self, deltas: Dict[str, int]) -> int:
        return sum(int(quantity) for item, quantity in deltas.items() if self._is_log_item(item))

    def _surface_log_jump_lock_active(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        return (
            self._is_surface_resource_acquisition(goal, prompt)
            and int(self._control_state.get("surface_log_jump_lock_ticks", 0)) > 0
        )

    def _apply_surface_log_jump_lock(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> Dict[str, Any]:
        if self._surface_log_jump_lock_active(goal, prompt):
            self._set_button(action, "jump", 0)
            self._set_button(action, "sprint", 0)
        return action

    def _tick_surface_log_jump_lock(self) -> None:
        ticks = int(self._control_state.get("surface_log_jump_lock_ticks", 0))
        if ticks > 0:
            self._control_state["surface_log_jump_lock_ticks"] = ticks - 1

    def _finish_stabilized_action(
        self,
        stage: str,
        before: Dict[str, Any],
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> Dict[str, Any]:
        action = self._apply_surface_log_jump_lock(action, goal, prompt)
        self._maybe_log_action_debug(stage, before, action, goal, prompt)
        self._tick_surface_log_jump_lock()
        return action

    def _option_progress_snapshot(
        self,
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> Dict[str, Any]:
        goal_items = self._goal_item_names(goal)
        ledger = self.cache.get("resource_ledger", {}) or {}
        status = self.status_mod.get_status() if self.status_mod is not None else {}
        inventory = status.get("inventory", {}) or {}
        loc = status.get("location_stats", {}) or {}
        observed_goal_total = 0
        if goal_items:
            observed_goal_total = max(
                self._items_total(inventory, goal_items),
                self._items_total(ledger.get("max_inventory", {}) or {}, goal_items),
                self._items_total(ledger.get("pickup", {}) or {}, goal_items),
                self._items_total(ledger.get("collected", {}) or {}, goal_items),
            )
        x = float(loc.get("xpos", 0.0) or 0.0)
        y = float(loc.get("ypos", 0.0) or 0.0)
        z = float(loc.get("zpos", 0.0) or 0.0)
        return {
            "step": int(getattr(self, "num_steps", 0)),
            "goal_items": goal_items,
            "goal_observed_total": int(observed_goal_total),
            "goal_mined_total": int(self._items_total(ledger.get("mined_blocks", {}) or {}, goal_items)) if goal_items else 0,
            "pending_relevant_drops": int(self._pending_relevant_drops(goal)),
            "inventory_slots_used": int(self._used_inventory_slots()),
            "x": x,
            "y": y,
            "z": z,
            "air": int(self._current_air()),
            "health": float(self._current_health()),
            "prompt": prompt or "",
        }

    def _evaluate_option_outcome(
        self,
        option_name: str,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Unified state-change (progress) function for option effectiveness.

        A skill's effect is a single scalar progress

            Δ = [Φ(s_after) − Φ(s_before)]  +  λ · disp_term

        where Φ(s) = goal_obtained(s) is the *primary potential* (number
        of goal items obtained so far — monotone, directly observable,
        zero-rollout), and ``disp_term`` is the displacement the skill
        caused (did the recovery physically relocate / unstick the
        agent). λ is small (``XENON_OPTION_ESC_WEIGHT``, default 0.01) so
        any real goal progress dominates pure movement; movement only
        decides effectiveness when the goal potential is flat — exactly
        the primary/secondary priority the old branchy logic encoded,
        now as one formula.

        A skill is effective iff Δ > τ (``XENON_OPTION_TAU``, default 0),
        and Δ itself is the reward. No air/health terms: the agent has no
        escape mechanic (it drowns and respawns), so those are not part
        of skill effectiveness.

        This is the zero-rollout embodied analogue of progress-based step
        value (PAV): instead of Monte-Carlo estimating the increment in
        final-success probability, we read an observable potential
        difference plus an observable displacement directly. The result
        keeps every legacy field so existing logs/analysis still work.
        """
        lam = float(os.environ.get("XENON_OPTION_ESC_WEIGHT", "0.01"))
        scale = float(os.environ.get("XENON_OPTION_DISP_SCALE", "5.0"))
        tau = float(os.environ.get("XENON_OPTION_TAU", "0.0"))

        # Primary potential Φ: goal items obtained so far (observed + mined).
        goal_before = int(before.get("goal_observed_total", 0)) + int(before.get("goal_mined_total", 0))
        goal_after = int(after.get("goal_observed_total", 0)) + int(after.get("goal_mined_total", 0))
        d_goal = goal_after - goal_before

        # Secondary term: 3D displacement caused by the skill (a transition
        # quantity, not a state potential), normalised by ``scale`` blocks.
        dx = float(after.get("x", 0.0)) - float(before.get("x", 0.0))
        dy = float(after.get("y", 0.0)) - float(before.get("y", 0.0))
        dz = float(after.get("z", 0.0)) - float(before.get("z", 0.0))
        horizontal_delta = (dx * dx + dz * dz) ** 0.5
        displacement = (dx * dx + dy * dy + dz * dz) ** 0.5
        disp_term = (displacement / scale) if scale > 0 else 0.0

        progress_delta = float(d_goal) + lam * disp_term
        success = progress_delta > tau
        if d_goal > 0:
            success_reason = "goal_progress"
        elif success:
            success_reason = "position_changed"
        else:
            success_reason = "none"

        # Legacy/diagnostic deltas (kept for backward-compatible logging).
        observed_delta = int(after.get("goal_observed_total", 0)) - int(before.get("goal_observed_total", 0))
        mined_delta = int(after.get("goal_mined_total", 0)) - int(before.get("goal_mined_total", 0))
        pending_drop_delta = int(after.get("pending_relevant_drops", 0)) - int(before.get("pending_relevant_drops", 0))
        slot_delta = int(after.get("inventory_slots_used", 0)) - int(before.get("inventory_slots_used", 0))
        elapsed_ticks = int(after.get("step", 0)) - int(before.get("step", 0))

        return {
            "success": bool(success),
            "success_reason": success_reason,
            # --- progress function Δ ---
            "reward": float(progress_delta),
            "progress_delta": float(progress_delta),
            "progress_goal_delta": float(d_goal),
            "progress_disp_term": float(lam * disp_term),
            "potential_before": float(goal_before),
            "potential_after": float(goal_after),
            "displacement": float(displacement),
            # --- legacy fields ---
            "elapsed_ticks": max(0, elapsed_ticks),
            "goal_observed_delta": observed_delta,
            "goal_mined_delta": mined_delta,
            "pending_drop_delta": pending_drop_delta,
            "inventory_slot_delta": slot_delta,
            "horizontal_delta": horizontal_delta,
            "vertical_delta": dy,
        }

    def _start_option_event(
        self,
        candidate: EnvOptionCandidate,
        context: Dict[str, Any],
        decision_trace: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> None:
        before = self._option_progress_snapshot(goal, prompt)
        event_id = self.option_decisioner.record_invocation(
            candidate.name,
            context,
            candidate.reason,
            decision_trace,
            before,
            candidate.planned_ticks,
        )
        self._active_option_events[candidate.control_key] = {
            "event_id": event_id,
            "candidate": candidate,
            "context": context,
            "before": before,
            "started_step": int(getattr(self, "num_steps", 0)),
        }

    def _finalize_option_events(
        self,
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> None:
        if not self._active_option_events:
            return
        for control_key, event in list(self._active_option_events.items()):
            candidate = event["candidate"]
            remaining = int(self._control_state.get(control_key, 0))
            elapsed = int(getattr(self, "num_steps", 0)) - int(event.get("started_step", 0))
            if remaining > 0 and elapsed < int(candidate.planned_ticks):
                continue
            after = self._option_progress_snapshot(goal, prompt)
            outcome = self._evaluate_option_outcome(candidate.name, event["before"], after)
            self.option_decisioner.record_outcome(
                event["event_id"],
                candidate.name,
                event["context"],
                event["before"],
                after,
                outcome,
            )
            if self.logger:
                self.logger.info(
                    "[option] outcome: option=%s success=%s reason=%s elapsed=%s observed_delta=%s mined_delta=%s",
                    candidate.name,
                    outcome["success"],
                    outcome["success_reason"],
                    outcome["elapsed_ticks"],
                    outcome["goal_observed_delta"],
                    outcome["goal_mined_delta"],
                )
            del self._active_option_events[control_key]

    def _select_and_activate_option(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> None:
        candidates = build_env_option_candidates(self, action, goal, prompt)
        if not candidates:
            return
        context = build_option_context(self, action, goal, prompt)
        selection = self.option_decisioner.select(candidates, context)
        if not selection.execute or selection.option is None:
            skip_key = "|".join(candidate.name for candidate in candidates)
            skip_interval = int(os.environ.get("XENON_OPTION_SKIP_LOG_INTERVAL_TICKS", "40"))
            last_skip = int(self._last_option_skip_steps.get(skip_key, -1000000))
            should_log_skip = self.num_steps - last_skip >= skip_interval
            if should_log_skip:
                self.option_decisioner.record_skip(context, candidates, selection)
                self._last_option_skip_steps[skip_key] = int(self.num_steps)
            if should_log_skip and self.logger:
                self.logger.info(
                    "[option] skipped candidates=%s reason=%s",
                    [candidate.name for candidate in candidates],
                    selection.reason,
                )
            return
        selected = next(
            candidate for candidate in candidates if candidate.name == selection.option
        )
        trace = selection.to_trace()
        self._start_option_event(selected, context, trace, goal, prompt)
        activate_env_option(self, selected)
        if self.logger:
            ranking = ", ".join(
                f"{item.option}={item.score:.3f}" for item in selection.rankings
            )
            self.logger.info(
                "[option] activating: option=%s ranking=[%s] reason=%s rule_reason=%s",
                selected.name,
                ranking,
                selection.reason,
                selected.reason,
            )

    def run_scheduled_option(
        self,
        name: str,
        goal: tuple[str, int] | None,
        prompt: str | None,
        action: Dict[str, Any],
        execute_fn,
        reason: Dict[str, Any] | None = None,
        recovery_event_key: str | None = None,
        mandatory: bool = True,
    ):
        """Route a synchronous, blocking skill through the option scheduler.

        Unlike the tick-based options in ``_select_and_activate_option``
        (which set a control field and are finalised later when the ticks
        expire), this path is for skills whose *policy* is a scripted,
        immediately-executing routine (e.g. the overshoot pillar-up +
        lateral-shift chain). The caller keeps its own hard trigger
        condition and must only call this when that condition is already
        true, so the rule gate stays the sole trigger.

        The decisioner genuinely selects the candidate (the skill is, in
        fact, scheduled by the option decisioner), but a ``mandatory``
        candidate is always executed -> trigger timing and execution
        effect are identical to calling ``execute_fn`` directly. The only
        additions are read-only before/after snapshots and option-event
        logging.

        Returns ``execute_fn()``'s result, or ``None`` if (only possible
        for a non-mandatory candidate) the decisioner declines.
        """
        candidate = EnvOptionCandidate(
            name=name,
            control_key=recovery_event_key or name,
            recovery_event_key=recovery_event_key or name,
            planned_ticks=0,
            reason=reason or {},
            mandatory=bool(mandatory),
        )
        context = build_option_context(self, action, goal, prompt)
        selection = self.option_decisioner.select([candidate], context)
        if not selection.execute or selection.option != candidate.name:
            self.option_decisioner.record_skip(context, [candidate], selection)
            if self.logger:
                self.logger.info(
                    "[option] scheduled-skip: option=%s reason=%s",
                    candidate.name,
                    selection.reason,
                )
            return None

        before = self._option_progress_snapshot(goal, prompt)
        event_id = self.option_decisioner.record_invocation(
            candidate.name,
            context,
            candidate.reason,
            selection.to_trace(),
            before,
            candidate.planned_ticks,
        )
        events = self._control_state.setdefault("recovery_events", {})
        events[candidate.recovery_event_key] = (
            int(events.get(candidate.recovery_event_key, 0)) + 1
        )

        result = execute_fn()

        after = self._option_progress_snapshot(goal, prompt)
        outcome = self._evaluate_option_outcome(candidate.name, before, after)
        self.option_decisioner.record_outcome(
            event_id, candidate.name, context, before, after, outcome
        )
        if self.logger:
            self.logger.info(
                "[option] scheduled outcome: option=%s success=%s reward=%.4f "
                "delta=%.4f reason=%s",
                candidate.name,
                outcome["success"],
                outcome.get("reward", 0.0),
                outcome.get("progress_delta", 0.0),
                selection.reason,
            )
        return result

    def _attack_hold_ticks(self, goal: tuple[str, int] | None, prompt: str | None) -> int:
        text = self._prompt_text(goal, prompt)
        if any(token in text for token in ("chop", "punch", "tree", "log", "logs")):
            return int(os.environ.get("XENON_ATTACK_HOLD_WOOD_TICKS", "24"))
        return int(os.environ.get("XENON_ATTACK_HOLD_MINE_TICKS", "14"))

    def _current_air(self) -> int:
        life_stats = self.cache.get("last_life_stats") or {}
        try:
            return int(np.asarray(life_stats.get("air", 300)).reshape(-1)[0])
        except Exception:
            return 300

    def _life_stat_number(self, life_stats: Dict[str, Any], names: Tuple[str, ...], default: float) -> float:
        for name in names:
            if name not in life_stats:
                continue
            try:
                return float(np.asarray(life_stats[name]).reshape(-1)[0])
            except Exception:
                continue
        return default

    def _current_health(self) -> float:
        life_stats = self.cache.get("last_life_stats") or {}
        return self._life_stat_number(life_stats, ("life", "health"), 20.0)

    def _position_delta(self) -> Tuple[float, float]:
        window: Deque[Tuple[float, float, float]] = self.cache.get("position_window")
        if not window or len(window) < 40:
            return 999.0, 999.0
        x0, y0, z0 = window[0]
        x1, y1, z1 = window[-1]
        horizontal = ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5
        vertical = abs(y1 - y0)
        return horizontal, vertical

    def _position_jump(self, previous: Tuple[float, float, float] | None, current: Tuple[float, float, float]) -> float:
        if previous is None:
            return 0.0
        return (
            (current[0] - previous[0]) ** 2
            + (current[1] - previous[1]) ** 2
            + (current[2] - previous[2]) ** 2
        ) ** 0.5

    def _inventory_counts(self, plain_inventory: Dict[int, Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in plain_inventory.values():
            if not isinstance(item, dict):
                continue
            item_type = self._plain_item_type(item.get("type", "air"))
            if item_type in ("air", "none", ""):
                continue
            quantity = self._plain_item_quantity(item.get("quantity", 0))
            if quantity <= 0:
                continue
            counts[item_type] = counts.get(item_type, 0) + quantity
        return counts

    def _plain_item_type(self, value: Any) -> str:
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                value = arr.reshape(-1)[0]
        except Exception:
            pass
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            item_type = value
        else:
            item_type = str(value)
        if ":" in item_type:
            item_type = item_type.split(":")[-1]
        return item_type

    def _plain_item_quantity(self, value: Any) -> int:
        try:
            arr = np.asarray(value)
            if arr.size == 0:
                return 0
            return int(arr.reshape(-1)[0])
        except Exception:
            try:
                return int(value or 0)
            except Exception:
                return 0

    def _goal_item_names(self, goal: tuple[str, int] | None) -> List[str]:
        if goal is None:
            return []
        try:
            return list(self.task_checker_mod._expand_item(str(goal[0])))
        except Exception:
            return [str(goal[0])]

    def _goal_quantity(self, goal: tuple[str, int] | None) -> int:
        if goal is None:
            return 0
        try:
            return int(goal[1])
        except Exception:
            return 1

    def _items_total(self, counts: Dict[str, Any], items: List[str]) -> int:
        return sum(int(counts.get(name, 0)) for name in items)

    def _relevant_inventory_delta(
        self,
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> Dict[str, int]:
        if not self.status_mod.inventory_change:
            return {}
        diff = self.status_mod.inventory_change_what()
        if not diff:
            return {}
        if goal is None:
            return {item: int(quantity) for item, quantity in diff.items() if int(quantity) > 0}

        goal_items = set(self._goal_item_names(goal))
        relevant: Dict[str, int] = {}
        for item, quantity in diff.items():
            if item in goal_items:
                relevant[item] = int(quantity)
        return relevant

    def _stat_counts(self, value: Any) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if value is None:
            return counts
        if isinstance(value, dict):
            for key, nested in value.items():
                item_type = self._plain_item_type(key)
                if isinstance(nested, dict):
                    for nested_key, nested_value in self._stat_counts(nested).items():
                        merged_key = nested_key if nested_key != "value" else item_type
                        counts[merged_key] = counts.get(merged_key, 0) + nested_value
                    continue
                quantity = self._plain_item_quantity(nested)
                if quantity > 0 and item_type not in ("air", "none", ""):
                    counts[item_type] = counts.get(item_type, 0) + quantity
            return counts
        quantity = self._plain_item_quantity(value)
        if quantity > 0:
            counts["value"] = quantity
        return counts

    def _record_stat_deltas(
        self,
        ledger: Dict[str, Any],
        current_key: str,
        last_key: str,
        total_key: str,
        observation: Dict[str, Any],
    ) -> Dict[str, int]:
        raw_stats = observation.get(current_key)
        if raw_stats is None and isinstance(observation.get("stat"), dict):
            raw_stats = observation["stat"].get(current_key)
        current = self._stat_counts(raw_stats)
        last = ledger.get(last_key, {})
        totals = ledger.setdefault(total_key, {})
        deltas: Dict[str, int] = {}
        for item, quantity in current.items():
            delta = int(quantity) - int(last.get(item, 0))
            if delta > 0:
                totals[item] = int(totals.get(item, 0)) + delta
                deltas[item] = delta
        ledger[last_key] = current
        return deltas

    def _record_resource_ledger(self, observation: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        ledger = self.cache["resource_ledger"]
        current = self._inventory_counts(self.status_mod.inventory_with_slot)
        last = ledger["last_inventory"]
        inventory_delta: Dict[str, int] = {}
        for item, quantity in current.items():
            ledger["max_inventory"][item] = max(ledger["max_inventory"].get(item, 0), quantity)
            delta = quantity - int(last.get(item, 0))
            if delta > 0:
                ledger["collected"][item] = ledger["collected"].get(item, 0) + delta
                inventory_delta[item] = delta
        ledger["last_inventory"] = current
        pickup_delta = self._record_stat_deltas(ledger, "pickup", "last_pickup_stats", "pickup", observation)
        mine_delta = self._record_stat_deltas(ledger, "mine_block", "last_mine_block_stats", "mined_blocks", observation)
        return {
            "inventory": inventory_delta,
            "pickup": pickup_delta,
            "mine_block": mine_delta,
        }

    # Items whose count NEVER decreases (tools, equipment, blocks the
    # agent never crafts back into something else). For these the
    # historical "max-ever-observed" / "ever-collected" / "ever-picked-up"
    # ledger remains a useful fallback when the live inventory query is
    # briefly stale right after a craft.
    #
    # The complementary set — raw consumables (logs, planks, sticks,
    # cobblestone, ores, ingots) — gets used up by subsequent crafts, so
    # checking historical "max-ever-observed" against the goal count
    # would trivially say "yes" forever once the agent has briefly held
    # any quantity. That trips the planner into an infinite re-plan
    # loop where every "go fetch N more logs" sub-goal claims success
    # without the agent doing anything. Those items must consult the
    # *current* inventory only.
    _LEDGER_FALLBACK_GOAL_ITEMS: frozenset[str] = frozenset({
        "crafting_table",
        "furnace",
        "blast_furnace",
        "smoker",
        "chest",
        "hopper",
        "ladder",
        "torch",
        "bowl",
        "shears",
        "stonecutter",
        "tripwire_hook",
        "wooden_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "golden_pickaxe",
        "diamond_pickaxe",
        "wooden_axe",
        "stone_axe",
        "iron_axe",
        "golden_axe",
        "diamond_axe",
        "wooden_shovel",
        "stone_shovel",
        "iron_shovel",
        "golden_shovel",
        "diamond_shovel",
        "wooden_hoe",
        "stone_hoe",
        "iron_hoe",
        "golden_hoe",
        "diamond_hoe",
        "wooden_sword",
        "stone_sword",
        "iron_sword",
        "golden_sword",
        "diamond_sword",
    })

    def _ledger_fallback_allowed(self, goal: tuple[str, int] | None) -> bool:
        """Return True only when the goal item is non-consumable.

        See `_LEDGER_FALLBACK_GOAL_ITEMS` for the rationale. Set
        `XENON_LEDGER_FALLBACK_ALL=1` to restore the legacy behaviour
        (ledger fallback for every goal) as an A/B knob.
        """
        if goal is None:
            return False
        if os.environ.get("XENON_LEDGER_FALLBACK_ALL", "0") == "1":
            return True
        try:
            item = str(goal[0]).lower().replace(" ", "_")
        except Exception:
            return False
        if item in self._LEDGER_FALLBACK_GOAL_ITEMS:
            return True
        if item.endswith("_pickaxe") or item.endswith("_axe") \
                or item.endswith("_shovel") or item.endswith("_hoe") \
                or item.endswith("_sword"):
            return True
        return False

    def _ledger_satisfies_goal(self, goal: tuple[str, int] | None) -> bool:
        if goal is None:
            return False
        # Consumable raw materials (logs, planks, sticks, cobblestone,
        # ores, ingots, ...) MUST go through the live-inventory check
        # only. The historical ledger only counts up, never down, so it
        # would falsely report "satisfied" the moment the agent ever
        # held any quantity — even after it crafted them away — which
        # caused infinite re-plan loops in the chest / iron / golden
        # tasks.
        if not self._ledger_fallback_allowed(goal):
            return False
        expanded_items = self._goal_item_names(goal)
        number = self._goal_quantity(goal)
        ledger = self.cache.get("resource_ledger", {})
        max_inventory = ledger.get("max_inventory", {})
        collected = ledger.get("collected", {})
        pickup = ledger.get("pickup", {})
        observed = self._items_total(max_inventory, expanded_items)
        gained = self._items_total(collected, expanded_items)
        picked_up = self._items_total(pickup, expanded_items)
        return observed >= int(number) or gained >= int(number) or picked_up >= int(number)

    def _used_inventory_slots(self) -> int:
        used = 0
        for item in self.status_mod.inventory_with_slot.values():
            if not isinstance(item, dict):
                continue
            item_type = self._plain_item_type(item.get("type", "air"))
            quantity = self._plain_item_quantity(item.get("quantity", 0))
            if item_type not in ("air", "none", "") and quantity > 0:
                used += 1
        return used

    def _protected_items(self, goal: tuple[str, int] | None, prompt: str | None) -> set[str]:
        protected = {
            "crafting_table",
            "furnace",
            "chest",
            "stick",
            "coal",
            "charcoal",
            "cobblestone",
            "stone",
            "smooth_stone",
            "iron_ingot",
            "gold_ingot",
        }
        protected.update(item for item in self.status_mod.inventory if item.endswith("_pickaxe"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_axe"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_shovel"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_hoe"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_sword"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_log"))
        protected.update(item for item in self.status_mod.inventory if item.endswith("_planks"))
        if goal:
            protected.update(self.task_checker_mod._expand_item(str(goal[0])))
        return protected

    def _junk_priority(self, item_type: str) -> int | None:
        low_value_tokens = (
            "leaves",
            "flower",
            "dandelion",
            "poppy",
            "tulip",
            "azure_bluet",
            "oxeye_daisy",
            "grass",
            "fern",
            "sapling",
            "seeds",
            "dead_bush",
            "vine",
            "lily_pad",
        )
        if any(token in item_type for token in low_value_tokens):
            return 0
        if item_type in {"dirt", "gravel", "sand", "clay", "flint", "rotten_flesh"}:
            return 1
        if item_type in {"granite", "diorite", "andesite", "tuff", "deepslate", "netherrack"}:
            return 2
        return None

    def _inventory_pressure_threshold(self, goal: tuple[str, int] | None, prompt: str | None) -> int:
        if self._is_resource_acquisition(goal, prompt):
            return int(os.environ.get("XENON_RESOURCE_INVENTORY_PRESSURE_SLOTS", "28"))
        return int(os.environ.get("XENON_INVENTORY_PRESSURE_SLOTS", "34"))

    def _maybe_cleanup_inventory(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> Dict[str, Any]:
        # Master gate: managed by PerceptionActionSuite (default ON).
        if os.environ.get("XENON_ENABLE_INVENTORY_CLEANUP", "1") != "1":
            return action
        used_slots = self._used_inventory_slots()
        pressure_threshold = self._inventory_pressure_threshold(goal, prompt)
        if used_slots < pressure_threshold:
            return action
        if self._control_state["escape_ticks"] > 0:
            return action
        severe_pressure = used_slots >= int(os.environ.get("XENON_SEVERE_INVENTORY_PRESSURE_SLOTS", "35"))
        stale_resource = (
            self._is_resource_acquisition(goal, prompt)
            and self._stale_progress_ticks() >= int(os.environ.get("XENON_INVENTORY_CLEANUP_STALE_TICKS", "180"))
        )
        if self._button_down(action, "attack") and not (severe_pressure or stale_resource):
            return action

        protected = self._protected_items(goal, prompt)
        candidates = []
        for raw_slot, item in self.status_mod.inventory_with_slot.items():
            if not isinstance(item, dict):
                continue
            try:
                slot = int(raw_slot)
            except Exception:
                continue
            item_type = self._plain_item_type(item.get("type", "air"))
            quantity = self._plain_item_quantity(item.get("quantity", 0))
            if slot > 8 or quantity <= 0 or item_type in protected:
                continue
            priority = self._junk_priority(item_type)
            if priority is None and (severe_pressure or stale_resource):
                priority = 20
            if priority is not None:
                candidates.append((priority, -quantity, slot, item_type))

        if not candidates:
            self._control_state["recovery_events"]["inventory_cleanup_blocked"] += 1
            return action

        _, _, slot, item_type = sorted(candidates)[0]
        self._control_state["attack_hold"] = 0
        for key in ("attack", "use", "jump", "forward", "back", "left", "right", "sprint", "sneak"):
            self._set_button(action, key, 0)
        for i in range(9):
            self._set_button(action, f"hotbar.{i+1}", 0)
        self._set_button(action, f"hotbar.{slot+1}", 1)
        self._set_button(action, "drop", 1)
        self._control_state["recovery_events"]["inventory_cleanup"] += 1
        self.logger.info(
            "Dropping non-waypoint hotbar item under inventory pressure: "
            f"slot={slot}, item={item_type}, used_slots={used_slots}, goal={goal}"
        )
        return action

    def _movement_intent(self, action: Dict[str, Any]) -> bool:
        return any(
            self._button_down(action, key)
            for key in ("forward", "back", "left", "right", "jump", "sprint")
        )

    def _stagnant_motion(self) -> bool:
        horizontal, vertical = self._position_delta()
        return (
            horizontal < float(os.environ.get("XENON_STAGNANT_HORIZONTAL_DELTA", "0.35"))
            and vertical < float(os.environ.get("XENON_STAGNANT_VERTICAL_DELTA", "0.25"))
        )

    def _stale_progress_ticks(self) -> int:
        return self.num_steps - int(self.cache.get("last_progress_step", 0))

    def _stale_goal_progress_ticks(self) -> int:
        return self.num_steps - int(self.cache.get("last_goal_progress_step", 0))

    def _relevant_delta_total(self, deltas: Dict[str, int], goal: tuple[str, int] | None) -> int:
        goal_items = self._goal_item_names(goal)
        return self._items_total(deltas, goal_items)

    def _pending_relevant_drops(self, goal: tuple[str, int] | None) -> int:
        goal_items = self._goal_item_names(goal)
        if not goal_items:
            return 0
        ledger = self.cache.get("resource_ledger", {})
        mined = self._items_total(ledger.get("mined_blocks", {}), goal_items)
        observed = max(
            self._items_total(ledger.get("max_inventory", {}), goal_items),
            self._items_total(ledger.get("pickup", {}), goal_items),
            self._items_total(ledger.get("collected", {}), goal_items),
        )
        return max(0, mined - observed)

    def _should_collect_drops(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> bool:
        # Master gate: managed by PerceptionActionSuite (default ON).
        if os.environ.get("XENON_ENABLE_COLLECT_DROPS", "1") != "1":
            return False
        if goal is None or not self._is_resource_acquisition(goal, prompt):
            return False
        if self._ledger_satisfies_goal(goal):
            return False
        if self._button_down(action, "attack") or int(self._control_state.get("attack_hold", 0)) > 0:
            return False
        if self.num_steps - int(self.cache.get("last_target_block_step", 0)) < int(
            os.environ.get("XENON_COLLECT_DROPS_AFTER_MINE_GRACE_TICKS", "100")
        ):
            return False
        if self.num_steps - int(self.cache.get("last_collect_drop_step", -1000000)) < int(
            os.environ.get("XENON_COLLECT_DROPS_COOLDOWN_TICKS", "180")
        ):
            return False
        pending = self._pending_relevant_drops(goal)
        if pending <= 0:
            return False
        return self._stale_goal_progress_ticks() >= int(os.environ.get("XENON_COLLECT_DROPS_STALE_TICKS", "70"))

    def _should_surface_search(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        if os.environ.get("XENON_ENABLE_SURFACE_SEARCH_PRIMITIVE", "0") != "1":
            return False
        if not self._is_surface_resource_acquisition(goal, prompt):
            return False
        if self._ledger_satisfies_goal(goal):
            return False
        if self.num_steps - int(self.cache.get("last_target_block_step", 0)) < int(
            os.environ.get("XENON_SURFACE_SEARCH_AFTER_MINE_GRACE_TICKS", "240")
        ):
            return False
        if self.num_steps - int(self.cache.get("last_surface_search_step", -1000000)) < int(
            os.environ.get("XENON_SURFACE_SEARCH_COOLDOWN_TICKS", "260")
        ):
            return False
        return self._stale_goal_progress_ticks() >= int(os.environ.get("XENON_SURFACE_SEARCH_STALE_TICKS", "220"))

    def _should_surface_escape(self) -> bool:
        if os.environ.get("XENON_ENABLE_LOW_AIR_ESCAPE", "0") != "1":
            return False
        return self._current_air() < int(os.environ.get("XENON_LOW_AIR_THRESHOLD", "280"))

    def _should_movement_escape(self, action: Dict[str, Any], goal: tuple[str, int] | None, prompt: str | None) -> bool:
        # Master gate: managed by PerceptionActionSuite (default ON).
        if os.environ.get("XENON_ENABLE_MOVEMENT_ESCAPE", "1") != "1":
            return False
        if self._is_tunnel_resource_acquisition(goal, prompt):
            return False
        if self._movement_intent(action) and self._stagnant_motion() and not self._button_down(action, "attack"):
            self._control_state["movement_stagnant_ticks"] += 1
        else:
            self._control_state["movement_stagnant_ticks"] = 0
        return (
            self._control_state["movement_stagnant_ticks"]
            >= int(os.environ.get("XENON_ESCAPE_STAGNANT_TICKS", "80"))
            and self._stale_progress_ticks() >= int(os.environ.get("XENON_ESCAPE_MIN_STALE_PROGRESS", "240"))
        )

    def _safe_pitch(self) -> float:
        """Best-effort read of the agent's current pitch in degrees.

        MineRL convention: positive pitch = looking down, negative = looking up.
        Falls back to 0.0 if location_stats are not yet available.
        """
        for source in (
            (self.cache.get("info") or {}).get("location_stats"),
            getattr(self.status_mod, "location_stats", None),
        ):
            if not source:
                continue
            try:
                value = source.get("pitch", 0)
            except AttributeError:
                continue
            try:
                arr = np.asarray(value).reshape(-1)
                if arr.size:
                    return float(arr[0])
            except Exception:
                continue
        return 0.0

    def _should_surface_turn_around(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None,
        prompt: str | None,
    ) -> bool:
        """Trigger a 180-degree yaw turn when surface exploration is
        horizontally stagnant — agent is wedged against terrain.

        Two-tier responsiveness so the agent reacts promptly when
        actually stuck on a hill but never while actually chopping a
        tree:

        Primary signal — per-tick stuck counter
            Increment ``surface_stuck_ticks`` on every tick where the
            agent has movement intent (forward/back/left/right/jump/
            sprint pressed) AND the recent xz-position delta is below
            ``XENON_SURFACE_TURNAROUND_HORIZONTAL_DELTA`` (default 0.6).
            Fire when the counter reaches
            ``XENON_SURFACE_TURNAROUND_STUCK_TICKS`` (default 60).

        Secondary safety — stale goal progress
            Even after the counter is full, do not fire unless no
            goal-relevant inventory delta has happened for at least
            ``XENON_SURFACE_TURNAROUND_STALE_TICKS`` (default 100)
            ticks. Prevents firing right after a successful log pickup.

        Hard guards (counter is reset to 0, never accumulates):
        * ``button_down(attack)`` — agent is actively chopping
        * ``attack_hold > 0`` — agent is in the post-attack hold window
        * ``num_steps - last_surface_attack_step < ATTACK_RECENT`` —
          between adjacent swings during a real chop sequence
          (default 30 ticks ≈ 1.5 s)
        * not in surface mode — no leakage across mode changes
        * resource ledger already considers the goal satisfied
        * within ``cooldown`` ticks of the previous turn-around

        Master gate: ``XENON_ENABLE_SURFACE_TURNAROUND`` (default ON
        via PerceptionActionSuite).
        """
        if os.environ.get("XENON_ENABLE_SURFACE_TURNAROUND", "1") != "1":
            return False
        # Out of surface mode — clear the counter so it does not leak
        # state into the next surface session.
        if not self._is_surface_resource_acquisition(goal, prompt):
            self._control_state["surface_stuck_ticks"] = 0
            return False
        # Hard guard: actively attacking right now.
        if self._button_down(action, "attack") or int(self._control_state.get("attack_hold", 0)) > 0:
            self._control_state["surface_stuck_ticks"] = 0
            return False
        # Soft guard: attack was pressed in the recent past — we are in
        # a between-swings frame of an ongoing chop sequence. Real
        # chopping cycles attack on and off as the policy alternates;
        # without this guard the counter would accumulate during those
        # off frames and falsely trigger a turn-around mid-chop.
        attack_recent = int(os.environ.get("XENON_SURFACE_TURNAROUND_ATTACK_RECENT_TICKS", "30"))
        if self.num_steps - int(self.cache.get("last_surface_attack_step", -1000000)) < attack_recent:
            self._control_state["surface_stuck_ticks"] = 0
            return False
        # Cooldown after previous turn-around.
        if self.num_steps - int(self.cache.get("last_surface_turn_around_step", -1000000)) < int(
            os.environ.get("XENON_SURFACE_TURNAROUND_COOLDOWN_TICKS", "200")
        ):
            return False
        if self._ledger_satisfies_goal(goal):
            self._control_state["surface_stuck_ticks"] = 0
            return False

        # Per-tick stuck counter ----------------------------------
        horizontal, _ = self._position_delta()
        horizontal_thresh = float(os.environ.get(
            "XENON_SURFACE_TURNAROUND_HORIZONTAL_DELTA", "0.6"
        ))
        # _position_delta returns 999.0 sentinel when the window is not
        # full yet; that case is correctly excluded by the threshold.
        if self._movement_intent(action) and horizontal < horizontal_thresh:
            self._control_state["surface_stuck_ticks"] = int(
                self._control_state.get("surface_stuck_ticks", 0)
            ) + 1
        else:
            self._control_state["surface_stuck_ticks"] = 0

        stuck_required = int(os.environ.get(
            "XENON_SURFACE_TURNAROUND_STUCK_TICKS", "60"
        ))
        if self._control_state["surface_stuck_ticks"] < stuck_required:
            return False
        # Secondary safety: don't disturb a chop that is producing
        # progress (logs going up).
        return self._stale_goal_progress_ticks() >= int(
            os.environ.get("XENON_SURFACE_TURNAROUND_STALE_TICKS", "100")
        )

    def _should_tunnel_recovery(self, action: Dict[str, Any], goal: tuple[str, int] | None, prompt: str | None) -> bool:
        # Master gate: managed by PerceptionActionSuite (default ON).
        if os.environ.get("XENON_ENABLE_TUNNEL_RECOVERY", "1") != "1":
            return False
        if not self._is_tunnel_resource_acquisition(goal, prompt):
            self._control_state["resource_stagnant_ticks"] = 0
            return False
        if self._stagnant_motion() and self._stale_progress_ticks() >= int(
            os.environ.get("XENON_TUNNEL_MIN_STALE_PROGRESS", "260")
        ):
            self._control_state["resource_stagnant_ticks"] += 1
        else:
            self._control_state["resource_stagnant_ticks"] = 0
        return self._control_state["resource_stagnant_ticks"] >= int(
            os.environ.get("XENON_TUNNEL_STAGNANT_TICKS", "100")
        )

    def _escape_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._control_state["escape_ticks"] -= 1
        if self._control_state["escape_ticks"] % 20 == 0:
            self._control_state["escape_turn"] *= -1

        for key in ("attack", "use", "back", "left", "right", "sneak", "inventory", "drop"):
            self._set_button(action, key, 0)
        for key in ("forward", "jump", "sprint"):
            self._set_button(action, key, 1)
        action["camera"] = np.array([-8, 10 * self._control_state["escape_turn"]])
        return action

    def _collect_drop_action(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ) -> Dict[str, Any]:
        self._control_state["collect_drop_ticks"] -= 1

        policy_jump = self._button_down(action, "jump")
        for key in ("attack", "use", "back", "left", "right", "sneak", "sprint", "inventory", "drop"):
            self._set_button(action, key, 0)
        self._set_button(action, "forward", 1)
        if self._lock_jump_during_attack(goal, prompt):
            self._set_button(action, "jump", 0)
        else:
            self._set_button(
                action,
                "jump",
                1 if policy_jump or self._control_state["collect_drop_ticks"] % 12 == 0 else 0,
            )
        return action

    def _surface_search_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._control_state["surface_search_ticks"] -= 1
        if self._control_state["surface_search_ticks"] % 60 == 0:
            self._control_state["escape_turn"] *= -1

        policy_jump = self._button_down(action, "jump")
        for key in ("attack", "use", "back", "left", "right", "sneak", "inventory", "drop"):
            self._set_button(action, key, 0)
        self._set_button(action, "forward", 1)
        self._set_button(action, "sprint", 1)
        self._set_button(action, "jump", 1 if policy_jump or self._control_state["surface_search_ticks"] % 12 == 0 else 0)
        action["camera"] = np.array([-1, 3 * self._control_state["escape_turn"]])
        return action

    def _surface_turn_around_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Force a 180-degree yaw rotation while keeping the agent walking
        forward, so that the next exploration step faces away from the
        obstacle the policy was wedged against.

        Pitch is gently nudged toward 0 (level horizon) so the agent
        does not finish the manoeuvre staring at the sky.
        """
        self._control_state["surface_turn_around_ticks"] -= 1
        for key in ("attack", "use", "back", "left", "right", "sneak", "inventory", "drop", "jump"):
            self._set_button(action, key, 0)
        self._set_button(action, "forward", 1)
        self._set_button(action, "sprint", 1)
        yaw_delta = float(os.environ.get("XENON_SURFACE_TURNAROUND_YAW_PER_TICK", "6.0"))
        cur_pitch = self._safe_pitch()
        # Slide pitch back toward 0 (max 4 deg / tick), bounded both ways.
        pitch_delta = max(min(0.0 - cur_pitch, 4.0), -4.0)
        action["camera"] = np.array([pitch_delta, yaw_delta * self._control_state["escape_turn"]])
        return action

    def _tunnel_recovery_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._control_state["tunnel_recovery_ticks"] -= 1
        if self._control_state["tunnel_recovery_ticks"] % 20 == 0:
            self._control_state["escape_turn"] *= -1

        for key in ("use", "back", "left", "right", "sneak", "inventory", "drop"):
            self._set_button(action, key, 0)
        self._set_button(action, "attack", 1)
        self._set_button(action, "forward", 1)
        self._set_button(action, "sprint", 0)
        self._set_button(action, "jump", 1 if self._control_state["tunnel_recovery_ticks"] % 12 == 0 else 0)
        action["camera"] = np.array([0, 12 * self._control_state["escape_turn"]])
        return action

    def _stabilize_action(
        self,
        action: Dict[str, Any] | List[Dict[str, Any]],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ) -> Dict[str, Any]:
        action = self._normalise_action(action)
        original_action = copy.deepcopy(action)
        if prompt != self._control_state.get("last_prompt"):
            self._control_state["attack_hold"] = 0
            self._control_state["movement_stagnant_ticks"] = 0
            self._control_state["resource_stagnant_ticks"] = 0
            self._control_state["tunnel_recovery_ticks"] = 0
            self._control_state["surface_search_ticks"] = 0
            self._control_state["surface_turn_around_ticks"] = 0
            self._control_state["surface_stuck_ticks"] = 0
            self._control_state["collect_drop_ticks"] = 0
            self._control_state["last_prompt"] = prompt
            self.cache["surface_attack_streak"] = 0
            self.cache["last_progress_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps

        self._select_and_activate_option(action, goal, prompt)

        if self._control_state["escape_ticks"] > 0:
            action = self._escape_action(action)
            return self._finish_stabilized_action("stabilize_escape", original_action, action, goal, prompt)
        if self._control_state["collect_drop_ticks"] > 0:
            action = self._collect_drop_action(action, goal, prompt)
            return self._finish_stabilized_action("stabilize_collect_drops", original_action, action, goal, prompt)
        if self._control_state["surface_search_ticks"] > 0:
            action = self._surface_search_action(action)
            return self._finish_stabilized_action("stabilize_surface_search", original_action, action, goal, prompt)
        if self._control_state["surface_turn_around_ticks"] > 0:
            action = self._surface_turn_around_action(action)
            return self._finish_stabilized_action("stabilize_surface_turn_around", original_action, action, goal, prompt)
        if self._control_state["tunnel_recovery_ticks"] > 0:
            action = self._tunnel_recovery_action(action)
            return self._finish_stabilized_action("stabilize_tunnel", original_action, action, goal, prompt)

        original_attack = self._button_down(action, "attack")
        if original_attack and self._is_resource_acquisition(goal, prompt):
            self._control_state["attack_hold"] = max(
                int(self._control_state["attack_hold"]),
                self._attack_hold_ticks(goal, prompt),
            )

        if self._control_state["attack_hold"] > 0:
            self._set_button(action, "attack", 1)
            if self._lock_full_movement_during_attack(goal, prompt):
                locked_keys = ("jump", "forward", "back", "left", "right", "sprint", "sneak", "use")
                if original_attack:
                    locked_keys = ("jump", "left", "right", "sprint", "sneak", "use")
            elif self._lock_jump_during_attack(goal, prompt):
                locked_keys = ("jump", "sprint", "sneak", "use")
            else:
                locked_keys = ("sneak", "use")
            for key in locked_keys:
                self._set_button(action, key, 0)
            if not original_attack:
                action["camera"] = np.array([0, 0])
            self._control_state["attack_hold"] -= 1
        elif original_attack:
            locked_keys = ("left", "right", "sneak")
            if self._lock_full_movement_during_attack(goal, prompt):
                locked_keys = ("jump", "left", "right", "sprint", "sneak")
            elif self._lock_jump_during_attack(goal, prompt):
                locked_keys = ("jump", "sprint", "sneak")
            for key in locked_keys:
                self._set_button(action, key, 0)

        if self._is_surface_resource_acquisition(goal, prompt):
            if self._button_down(action, "attack"):
                self.cache["surface_attack_streak"] = int(self.cache.get("surface_attack_streak", 0)) + 1
                self.cache["last_surface_attack_step"] = self.num_steps
            else:
                self.cache["surface_attack_streak"] = 0
        else:
            self.cache["surface_attack_streak"] = 0

        # Ground-exploration pitch clamp: when the agent is on the surface
        # looking for / chopping a tree but is *not* currently attacking,
        # bound the camera pitch into a near-horizontal band so STEVE-1
        # cannot drift its view toward the sky during navigation. This
        # only touches the regular (non-scripted) action path; underground
        # mining and all scripted recovery primitives are unaffected.
        if (
            os.environ.get("XENON_ENABLE_GROUND_PITCH_CLAMP", "1") == "1"
            and self._is_surface_resource_acquisition(goal, prompt)
            and not self._is_tunnel_resource_acquisition(goal, prompt)
            and not self._button_down(action, "attack")
            and int(self._control_state.get("attack_hold", 0)) == 0
        ):
            try:
                cam_arr = np.asarray(action.get("camera", [0.0, 0.0]), dtype=float).reshape(-1)
            except Exception:
                cam_arr = np.array([0.0, 0.0])
            if cam_arr.size >= 1:
                pitch_delta = float(cam_arr[0])
                yaw_delta = float(cam_arr[1]) if cam_arr.size >= 2 else 0.0
                cur_pitch = self._safe_pitch()
                # MineRL convention: positive pitch = looking down,
                # negative = looking up. -15 = roughly 15 deg above
                # horizon, +75 = looking down at feet.
                min_pitch = float(os.environ.get("XENON_GROUND_PITCH_MIN", "-15.0"))
                max_pitch = float(os.environ.get("XENON_GROUND_PITCH_MAX", "75.0"))
                target_pitch = cur_pitch + pitch_delta
                clamped_delta = pitch_delta
                if target_pitch < min_pitch:
                    clamped_delta = min_pitch - cur_pitch
                elif target_pitch > max_pitch:
                    clamped_delta = max_pitch - cur_pitch
                if clamped_delta != pitch_delta:
                    action["camera"] = np.array([clamped_delta, yaw_delta])
                    self._control_state["recovery_events"]["ground_pitch_clamp"] += 1

        return self._finish_stabilized_action("stabilize", original_action, action, goal, prompt)

    def _record_step_state(
        self,
        observation: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ) -> None:
        previous_health = self._control_state.get("last_health")
        previous_position = self._control_state.get("last_position")
        previous_is_alive = self._control_state.get("last_is_alive")
        life_stats = observation.get("life_stats", {})
        self.cache["last_life_stats"] = life_stats
        loc = observation.get("location_stats", {})
        try:
            pos = (
                float(np.asarray(loc["xpos"]).reshape(-1)[0]),
                float(np.asarray(loc["ypos"]).reshape(-1)[0]),
                float(np.asarray(loc["zpos"]).reshape(-1)[0]),
            )
            self.cache["position_window"].append(pos)
            self._control_state["last_position"] = pos
        except Exception:
            pos = None
            pass
        health = self._life_stat_number(life_stats, ("life", "health"), 20.0)
        self._control_state["last_health"] = health
        # Track is_alive in addition to health: lava deals enough damage
        # in one tick to skip past the previous_health <= 2 frame, so a
        # health-only detector would miss lava deaths and the planner
        # would never reset after respawn. is_alive flips False -> True
        # cleanly across any respawn, regardless of damage source.
        is_alive_raw = life_stats.get("is_alive", 1)
        try:
            is_alive = bool(np.asarray(is_alive_raw).reshape(-1)[0])
        except Exception:
            is_alive = True
        self._control_state["last_is_alive"] = is_alive

        # Two independent death-respawn detectors (any one fires a reset):
        #
        #   (a) Health-based: previous_health <= 2, current >= 19, with
        #       a position jump > XENON_RESPAWN_POSITION_JUMP. Catches
        #       gradual deaths (drowning, fall, mob) where we observed a
        #       low-health frame.
        #
        #   (b) is_alive transition: last frame had is_alive False, this
        #       frame is_alive True. Catches single-tick fatal damage —
        #       most importantly lava — where health goes 20 -> 0 -> 20
        #       fast enough that no <=2 frame is observed.
        #
        # Either path resets control state and requests a STEVE-1 policy
        # reset so the planner re-issues the current sub-goal cleanly.
        respawn_health = (
            previous_health is not None
            and pos is not None
            and previous_position is not None
            and health >= 19.0
            and float(previous_health) <= 2.0
            and self._position_jump(previous_position, pos) > float(os.environ.get("XENON_RESPAWN_POSITION_JUMP", "6.0"))
        )
        respawn_is_alive = (
            previous_is_alive is False
            and is_alive is True
        )
        if respawn_health or respawn_is_alive:
            self._control_state["attack_hold"] = 0
            self._control_state["escape_ticks"] = 0
            self._control_state["tunnel_recovery_ticks"] = 0
            self._control_state["surface_search_ticks"] = 0
            self._control_state["surface_turn_around_ticks"] = 0
            self._control_state["surface_stuck_ticks"] = 0
            self._control_state["collect_drop_ticks"] = 0
            self._control_state["surface_log_jump_lock_ticks"] = 0
            self._control_state["movement_stagnant_ticks"] = 0
            self._control_state["resource_stagnant_ticks"] = 0
            self._control_state["policy_reset_requested"] = True
            self._control_state["recovery_events"]["respawn_reset"] += 1
            self.cache["position_window"].clear()
            reason = "is_alive_transition" if respawn_is_alive else "health_low_to_full"
            self.logger.info(
                f"Detected death/respawn transition (reason={reason}); "
                f"clearing low-level control state and requesting STEVE-1 policy reset."
            )
        deltas = self._record_resource_ledger(observation)
        if self._is_surface_resource_acquisition(goal, prompt):
            log_delta = max(
                self._log_delta_total(deltas.get("inventory", {})),
                self._log_delta_total(deltas.get("pickup", {})),
            )
            if log_delta > 0:
                lock_ticks = int(os.environ.get("XENON_SURFACE_LOG_JUMP_LOCK_TICKS", "30"))
                self._control_state["surface_log_jump_lock_ticks"] = max(
                    int(self._control_state.get("surface_log_jump_lock_ticks", 0)),
                    lock_ticks,
                )
                self.cache["last_surface_log_step"] = self.num_steps
                self._control_state["recovery_events"]["surface_log_jump_lock"] += 1
                self.logger.info(
                    "Temporarily locking jump after log pickup: "
                    f"log_delta={log_delta}, lock_ticks={lock_ticks}, prompt={prompt}, goal={goal}"
                )
        relevant_inventory = self._relevant_inventory_delta(goal, prompt)
        relevant_pickup = self._relevant_delta_total(deltas.get("pickup", {}), goal)
        relevant_mined = self._relevant_delta_total(deltas.get("mine_block", {}), goal)
        if goal is None and self.status_mod.inventory_change:
            self.cache["last_progress_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps
        elif relevant_inventory or relevant_pickup:
            self.cache["last_progress_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps
        if relevant_mined:
            self.cache["last_target_block_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps

    def raw_step(self, action: Dict[str, Any]):
        action = self._stabilize_action(action)
        if not self.can_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i+1}"] = np.array(0)
        # ban drop(Q) action
        action["drop"] = 0
        # attack时不乱动
        if self._button_down(action, "attack"):
            action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
        observation, reward, done, info = self.env.step(action)
        if isinstance(info, dict) and info.get("error"):
            info["isGuiOpen"] = observation.get("isGuiOpen", False)
            self.cache["info"] = info
            return observation, reward, done, info
        # 推送 POV 到宿主机 monitor_server（异步、O(1)、失败静默）
        _push_pov_to_monitor(observation)
        self.record_mod.step(observation, None, action)
        self.status_mod.step(observation, action, self.num_steps)
        self._record_step_state(observation)
        self._finalize_option_events(None, None)

        info.update(self.status_mod.get_status())

        info["isGuiOpen"] = observation["isGuiOpen"]

        # Mirror what step() does so methods like pillar_up that read
        # self.cache["info"] (location_stats, plain_inventory) work correctly
        # when driven exclusively through raw_step.
        self.cache["info"] = info

        return observation, reward, done, info

    def step(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ):
        action = self._stabilize_action(action, goal, prompt)
        if not self.can_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i+1}"] = np.array(0)
            action["use"] = np.array(0)
            action["inventory"] = np.array(0)

            hotbar = self.find_best_pickaxe()
            if hotbar:
                action[hotbar] = np.array(1)

        if not self.can_open_inventory:
            action["inventory"] = np.array(0)
        action["drop"] = np.array(0)
        action = self._maybe_cleanup_inventory(action, goal, prompt)

        observation, reward, done, info = self.env.step(action)
        if isinstance(info, dict) and info.get("error"):
            info.update({"killed": 0})
            info["isGuiOpen"] = observation.get("isGuiOpen", False)
            return observation, reward, done, info

        if goal is not None and goal[0] != self.cache["task"]:
            self.task_checker_mod.reset(observation["inventory"])
            self.cache["task"] = goal[0]

        self.record_mod.step(observation, None, action)
        self.status_mod.step(observation, action, self.num_steps)
        self._record_step_state(observation, goal, prompt)
        self._finalize_option_events(goal, prompt)

        info.update(self.status_mod.get_status())
        info.update({"killed": 0})

        ypos = self.status_mod.get_height()
        if ypos not in self.cache["ypos"]:
            self.cache["ypos"][ypos] = 0
        self.cache["ypos"][ypos] += 1
        # stuck 检测阈值：默认提高到 100000，避免 craft（静止开 inventory）误判
        _stuck_thresh = int(os.environ.get("XENON_STUCK_THRESHOLD", "100000"))
        _stuck_disabled = os.environ.get("XENON_DISABLE_STUCK_KILL", "0") == "1"
        if not _stuck_disabled and self.cache["ypos"][ypos] > _stuck_thresh:
            self.logger.critical("Stuck....")
            _kill_ok = True
            try:
                self.env.execute_cmd("/kill")
            except Exception as _e:
                _kill_ok = False
                self.logger.warning(f"execute_cmd('/kill') failed: {_e}; skipping null_action loop")
            self.logger.warning("Kill agent because ypos > 8000")
            self.cache["ypos"] = {}
            info.update({"killed": 1})
            if _kill_ok:
                for i in range(50):
                    try:
                        self.null_action()
                    except Exception as _e2:
                        self.logger.warning(f"null_action failed in stuck-recovery: {_e2}")
                        break
            # self.cache["explore"] = 100

        if self._only_once:
            if os.environ.get("XENON_ENABLE_RANDOM_ORE_ONCE", "1") == "1":
                try:
                    xpos, _, zpos = self.status_mod.get_position()
                except Exception:
                    xpos = zpos = None
                _ore_scale = float(self.cache.get("ore_spawn_scale", 1.0))
                random_ore(self.env, self.ORE_MAP, ypos, xpos=xpos, zpos=zpos, prob_scale=_ore_scale)
            self._only_once = False

        try:
            self._current_task_finish = self.task_checker_mod.step(
                observation["inventory"], goal
            )
            if not self._current_task_finish and self._ledger_satisfies_goal(goal):
                self._current_task_finish = True
                self.logger.info(f"Goal satisfied by resource ledger: {goal}")
        except Exception as e:
            self._control_state["recovery_events"]["checker_error"] += 1
            self.logger.warning(f"Task checker failed for goal={goal}; keeping subgoal unfinished: {e}")
            self._current_task_finish = False

        if self._current_task_finish:
            self.cache["task"] = ""
        info["isGuiOpen"] = observation["isGuiOpen"]

        self.cache["info"] = info
        return observation, reward, done, info
    
    # def check_already_achieved(self, inventory, goal):
    #     if goal is not None and goal[0] != self.cache["task"]:
    #         self.task_checker_mod.reset(inventory)
    #         self.cache["task"] = goal[0]
    #     self._current_task_finish = self.task_checker_mod.check_already_achieved(
    #         inventory, goal
    #     )
    #     if self._current_task_finish:
    #         self.cache["task"] = ""
    #     return self._current_task_finish
    
    def _check_goal_inventory_state(self, inventory: Dict[str, Any], item: str, number: int) -> bool:
        candidates = [
            item,
            item.replace(" ", "_"),
            item.replace("_ore", ""),
        ]
        for candidate in candidates:
            try:
                if self.task_checker_mod.step(inventory, [candidate, number], check_original_goal=True):
                    return True
            except Exception as exc:
                self.logger.warning(f"Inventory state check failed for {candidate}: {exc}")
        return False

    def _check_goal_inventory_delta(self, inventory: Dict[str, Any], item: str, number: int) -> bool:
        candidates = [
            item,
            item.replace(" ", "_"),
            item.replace("_ore", ""),
        ]
        for candidate in candidates:
            try:
                if self.task_checker_mod.step(inventory, [candidate, number]):
                    return True
            except Exception as exc:
                self.logger.warning(f"Inventory delta check failed for {candidate}: {exc}")
        return False

    def check_original_goal_finish(self, goal: tuple[str, int] | None):
        item_str = copy.deepcopy(goal[0])
        item_num = goal[1]

        current_env_status = self.get_status()
        current_inventory = current_env_status["inventory"]

        self.logger.info(f"In check_original_goal_finish")
        self.logger.info(f"goal: {goal}")
        self.logger.info(f"item_str: {item_str}")
        self.logger.info(f'item_str.replace(" ", "_"): {item_str.replace(" ", "_")}')
        self.logger.info(f"item_num: {item_num}")
        self.logger.info(f"current_inventory: {current_inventory}")

        return self._check_goal_inventory_state(current_inventory, item_str, item_num)

    def check_waypoint_finish(self, waypoint: tuple[str, int] | None):
        item_str = copy.deepcopy(waypoint[0])
        item_num = waypoint[1] if len(waypoint) > 1 else 1

        current_env_status = self.get_status()
        current_inventory = current_env_status["inventory"]
        self.logger.info(f"In check_waypoint_finish")
        self.logger.info(f"waypoint: {waypoint}")
        self.logger.info(f"item_str: {item_str}")
        self.logger.info(f"current_inventory: {current_inventory}")

        if self._check_goal_inventory_delta(current_inventory, item_str, item_num):
            return True
        if self._ledger_satisfies_goal([item_str, item_num]):
            self.logger.info(f"Waypoint satisfied by resource ledger: {waypoint}")
            return True
        return False

    def save_video(
        self,
        task: str,
        status: str,
        is_sub_task: bool = False,
        actual_done_final_task: str = "",
        biome: str = "",
        run_uuid: str = "",
    ):
        thread = self.record_mod.save(task, status, is_sub_task, actual_done_final_task, biome, run_uuid)
        return thread

    def inventory_change(self) -> bool:
        return self.status_mod.inventory_change

    def inventory_change_what(self):
        return self.status_mod.inventory_change_what()

    def inventory_new_item(self):
        return self.status_mod.inventory_new_item

    def inventory_new_item_what(self):
        return self.status_mod.inventory_new_item_what()

    @property
    def api_thread(self) -> MultiThreadServerAPI | None:
        return self._api_thread

    @api_thread.setter
    def api_thread(self, thread: MultiThreadServerAPI | None) -> None:
        self._api_thread = thread

    def api_thread_get_result(self):
        assert self._api_thread is not None, "Need set api_thread first."
        return self._api_thread.get_result()

    def api_thread_is_alive(self) -> bool:
        assert self._api_thread is not None, "Need set api_thread first."
        return self._api_thread.is_alive()

    def consume_policy_reset_requested(self) -> bool:
        requested = bool(self._control_state.get("policy_reset_requested", False))
        self._control_state["policy_reset_requested"] = False
        return requested

    def _call_func(self, func_name: str):
        action = self.env.noop_action()
        action[func_name] = 1
        self.step(action)
        action[func_name] = 0
        for _ in range(5):
            self.step(action)

    def null_action(self):
        action = self.env.noop_action()
        self.env.step(action)

    def find_best_pickaxe(self):
        if "info" not in self.cache:
            return None
        # height = self.cache["info"]["location_stats"]["ypos"]

        inventory_id = -1
        # find pickaxe
        inventory_id_diamond = self._find_in_inventory("diamond_pickaxe")
        inventory_id_iron = self._find_in_inventory("iron_pickaxe")
        inventory_id_stone = self._find_in_inventory("stone_pickaxe")
        inventory_id_wooden = self._find_in_inventory("wooden_pickaxe")

        if inventory_id_wooden is not None:
            inventory_id = inventory_id_wooden
        if inventory_id_stone is not None:
            inventory_id = inventory_id_stone
        if inventory_id_iron is not None:
            inventory_id = inventory_id_iron
        if inventory_id_diamond is not None:
            inventory_id = inventory_id_diamond

        if inventory_id == -1:
            return None
        if inventory_id >= 0 and inventory_id <= 8:
            return f"hotbar.{inventory_id+1}"
        else:
            pass

        # if height < 70:
        #     inventory_id = -1
        #     # find pickaxe
        #     inventory_id_diamond = self._find_in_inventory("diamond_pickaxe")
        #     inventory_id_iron = self._find_in_inventory("iron_pickaxe")
        #     inventory_id_stone = self._find_in_inventory("stone_pickaxe")
        #     inventory_id_wooden = self._find_in_inventory("wooden_pickaxe")

        #     if inventory_id_wooden:
        #         inventory_id = inventory_id_wooden
        #     if inventory_id_stone:
        #         inventory_id = inventory_id_stone
        #     if inventory_id_iron:
        #         inventory_id = inventory_id_iron
        #     if inventory_id_diamond:
        #         inventory_id = inventory_id_diamond
        #     if inventory_id == -1:
        #         return None
        #     if inventory_id >= 0 and inventory_id <= 8:
        #         return f"hotbar.{inventory_id+1}"
        #     else:
        #         pass
        # return None

    def _find_in_inventory(self, item: str):
        inventory = self.cache["info"]["plain_inventory"]
        for slot, it in inventory.items():
            if self._plain_item_type(it.get("type", "")) == item:
                try:
                    return int(slot)
                except Exception:
                    return slot
        return None

    # def give_night_vision(self):
    #     self.env.execute_cmd("/effect give @a night_vision 99999 250 true")

    # ---------------------------------------------------------------- #
    #  Scripted recovery: pillar-up                                     #
    # ---------------------------------------------------------------- #

    PILLAR_PREFER_BLOCKS: tuple[str, ...] = (
        "cobblestone",
        "cobbled_deepslate",
        "stone",
        "dirt",
        "andesite",
        "diorite",
        "granite",
        "sand",
        "gravel",
    )

    def _find_placeable_block_slot(
        self, prefer: tuple[str, ...] | None = None
    ) -> tuple[int | None, str | None]:
        """Locate a hotbar (0-8) slot containing a placeable block. Returns
        (slot_index, item_name) or (None, None) if no placeable block found
        in the hotbar. Inventory beyond hotbar (slot 9-) is ignored because
        the pillar cycle needs to keep the block selected without opening
        inventory."""
        prefer = prefer or self.PILLAR_PREFER_BLOCKS
        try:
            inventory = self.cache["info"].get("plain_inventory", {})
        except Exception:
            return None, None
        for block_name in prefer:
            for slot, item in inventory.items():
                try:
                    slot_int = int(slot)
                except Exception:
                    continue
                if not (0 <= slot_int <= 8):
                    continue
                if self._plain_item_type(item.get("type", "")) != block_name:
                    continue
                if self._plain_item_quantity(item.get("quantity", 0)) <= 0:
                    continue
                return slot_int, block_name
        return None, None

    def pillar_up(
        self,
        target_dy: int = 20,
        max_blocks: int = 32,
        max_steps: int = 400,
        prefer_blocks: tuple[str, ...] | None = None,
        prepared_slot: int | None = None,
        prepared_block: str | None = None,
    ) -> Dict[str, Any]:
        previous_can_change_hotbar = self.can_change_hotbar
        self.can_change_hotbar = True
        try:
            return self._pillar_up_impl(
                target_dy=target_dy,
                max_blocks=max_blocks,
                max_steps=max_steps,
                prefer_blocks=prefer_blocks,
                prepared_slot=prepared_slot,
                prepared_block=prepared_block,
            )
        finally:
            self.can_change_hotbar = previous_can_change_hotbar

    def _pillar_up_impl(
        self,
        target_dy: int = 20,
        max_blocks: int = 32,
        max_steps: int = 400,
        prefer_blocks: tuple[str, ...] | None = None,
        prepared_slot: int | None = None,
        prepared_block: str | None = None,
    ) -> Dict[str, Any]:
        """Scripted pillar-up. Place blocks under the agent to rise vertically.

        Bypasses STEVE-1 entirely (uses raw_step). Trace logged to
        `recovery_events["pillar_up"]`.

        Returns:
            dict with keys: success, blocks_used, dy, start_y, end_y,
                            steps_used, reason, trajectory.
        """
        prefer = prefer_blocks or self.PILLAR_PREFER_BLOCKS

        # snapshot starting state
        try:
            start_y = float(self.cache["info"]["location_stats"].get("ypos", 64.0))
        except Exception:
            start_y = 64.0
        target_y = start_y + float(target_dy)
        traj = [start_y]
        prep_actions: List[str] = []

        def _resolve_placeable_slot(
            allow_inventory_refill: bool = True,
        ) -> tuple[int | None, str | None]:
            slot, block = self._find_placeable_block_slot(prefer)
            if slot is not None:
                return slot, block
            if allow_inventory_refill:
                slot, block, prep_action = self._ensure_placeable_block_in_hotbar(
                    prefer=prefer, dest_hotbar_slot=2
                )
                if prep_action not in ("hotbar_ready", "none_available"):
                    prep_actions.append(prep_action)
                if slot is not None and block is not None:
                    return slot, block
            if prepared_slot is not None and prepared_block:
                return int(prepared_slot), str(prepared_block)
            return None, None

        slot, block_name = _resolve_placeable_slot()
        if slot is None:
            res = {
                "success": False,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": start_y,
                "end_y": start_y,
                "steps_used": 0,
                "reason": "no_placeable_block_in_hotbar",
                "trajectory": traj,
                "prep_actions": prep_actions,
            }
            self._record_recovery("pillar_up", res)
            return res

        if self.logger:
            self.logger.info(
                f"[pillar_up] start: y={start_y:.1f} target_dy={target_dy} "
                f"max_blocks={max_blocks} block={block_name} slot={slot}"
            )

        steps_used = 0
        blocks_used = 0
        last_y = start_y
        stuck_count = 0

        # Tunables exposed via env vars so we can iterate without code changes.
        max_stuck_per_cycle = int(os.environ.get("XENON_PILLAR_MAX_STUCK", "8"))
        jump_place_ticks = int(os.environ.get("XENON_PILLAR_JUMP_PLACE_TICKS", "8"))
        post_jump_settle = int(os.environ.get("XENON_PILLAR_SETTLE_TICKS", "8"))

        def _hold_pitch_to(a, target_pitch: float, max_step: float = 6.0):
            try:
                cp = float(self.cache["info"]["location_stats"].get("pitch", 0))
            except Exception:
                cp = 0.0
            delta = max(min(target_pitch - cp, max_step), -max_step)
            if abs(delta) > 0.1:
                a["camera"] = np.array([delta, 0])

        def _hold_pitch_down(a, target_pitch=88.0, max_step=6.0):
            _hold_pitch_to(a, target_pitch=target_pitch, max_step=max_step)

        def _clear_overhead_for_pillar() -> int:
            """Mine upward briefly so pillar jumping has headroom in tunnels."""
            nonlocal steps_used
            ticks = int(os.environ.get("XENON_PILLAR_CLEAR_OVERHEAD_TICKS", "36"))
            if ticks <= 0:
                return 0
            try:
                pickaxe_slot = self.find_best_pickaxe()
            except Exception:
                pickaxe_slot = None
            if not pickaxe_slot:
                return 0
            used = 0
            for _ in range(4):
                if steps_used >= max_steps:
                    return used
                a_eq = self.env.noop_action()
                a_eq[pickaxe_slot] = np.array(1)
                _hold_pitch_to(a_eq, target_pitch=-82.0, max_step=12.0)
                self.raw_step(a_eq)
                steps_used += 1
                used += 1
            mined_before = self._mined_block_count()
            for _ in range(ticks):
                if steps_used >= max_steps:
                    break
                a = self.env.noop_action()
                a["attack"] = np.array(1)
                _hold_pitch_to(a, target_pitch=-82.0, max_step=12.0)
                self.raw_step(a)
                steps_used += 1
                used += 1
            if self.logger:
                self.logger.info(
                    "[pillar_up] overhead_clear: ticks=%d mined_delta=%d held=%s",
                    used,
                    max(0, self._mined_block_count() - mined_before),
                    self._plain_item_type(self.status_mod.equipment),
                )
            return used

        def _confirm_placeable_selected(slot_index: int, expected_block: str, ticks: int = 4) -> bool:
            """Select the placeable hotbar slot and verify status_mod saw it."""
            nonlocal steps_used
            confirmed = False
            for _ in range(max(1, ticks)):
                a = self.env.noop_action()
                for i in range(9):
                    a[f"hotbar.{i + 1}"] = np.array(0)
                a[f"hotbar.{slot_index + 1}"] = np.array(1)
                _hold_pitch_down(a)
                self.raw_step(a)
                steps_used += 1
                held = self._plain_item_type(self.status_mod.equipment)
                if held == expected_block:
                    confirmed = True
            return confirmed

        _clear_overhead_for_pillar()

        # Phase 1: orient pitch fully downward. Need a strong commitment to
        # ~88-90° before the first placement; otherwise `use` lands the block
        # on the side of the floor instead of its top face. We push pitch
        # higher than the original 80° threshold and verify before exiting.
        for _ in range(20):
            try:
                cur_pitch = float(self.cache["info"]["location_stats"].get("pitch", 0))
            except Exception:
                cur_pitch = 0.0
            if cur_pitch >= 88.0:
                break
            action = self.env.noop_action()
            delta = max(min(90.0 - cur_pitch, 12.0), -12.0)
            action["camera"] = np.array([delta, 0])
            self.raw_step(action)
            steps_used += 1

        # Phase 1b: hotbar pre-confirmation. Hold the placeable hotbar key
        # for a couple ticks BEFORE jumping so the held item is definitely
        # the placeable block when `use` fires. Skipping this caused
        # blocks_used=1 dy=0 stuck_no_progress in the V5 armor run.
        slot, block_name = _resolve_placeable_slot()
        if slot is None:
            res = {
                "success": False,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": start_y,
                "end_y": start_y,
                "steps_used": steps_used,
                "reason": "no_placeable_block_during_pillar",
                "trajectory": traj,
                "prep_actions": prep_actions,
            }
            self._record_recovery("pillar_up", res)
            return res
        if not _confirm_placeable_selected(slot, block_name, ticks=4):
            res = {
                "success": False,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": start_y,
                "end_y": start_y,
                "steps_used": steps_used,
                "reason": f"placeable_hotbar_not_selected:{block_name}:slot_{slot}",
                "trajectory": traj,
                "prep_actions": prep_actions,
            }
            self._record_recovery("pillar_up", res)
            if self.logger:
                self.logger.warning(
                    f"[pillar_up] aborted before jump: expected held block "
                    f"{block_name} in hotbar.{slot + 1}, got "
                    f"{self._plain_item_type(self.status_mod.equipment)}"
                )
            return res

        # Phase 2: pillar cycle (per-block placement). Each cycle is now
        # 8+ ticks with a wider `use` window, so even if the agent's jump
        # apex is offset by a tick the placement still lands.
        #
        #   pre: select placeable hotbar slot, then stop pressing hotbar
        #        (the client keeps the selection; mixing hotbar+use in the
        #        same tick can consume the right-click window)
        #   t0..tN: jump + use (continuous placement window while rising)
        #   t_settle..t_settle+M: noop   (land + settle on the new block)
        #
        # Cycle ends with a Y check; if the agent gained < 0.5 we mark
        # this as a "stuck" cycle. We allow up to ``max_stuck_per_cycle``
        # consecutive stucks (default 8) before breaking out — the
        # original 4 was too easy to hit on the first cycle when the
        # hotbar swap was still settling.
        break_reason: str | None = None
        while blocks_used < max_blocks and steps_used < max_steps:
            try:
                cur_y = float(self.cache["info"]["location_stats"].get("ypos", last_y))
            except Exception:
                cur_y = last_y
            if cur_y >= target_y:
                break

            slot, block_name = _resolve_placeable_slot()
            if slot is None:
                break_reason = "no_placeable_block_available"
                break
            if not _confirm_placeable_selected(slot, block_name, ticks=2):
                break_reason = (
                    f"placeable_hotbar_not_selected:{block_name}:slot_{slot}"
                )
                stuck_count = max_stuck_per_cycle
                if self.logger:
                    self.logger.warning(
                        f"[pillar_up] stopping: failed to select "
                        f"{block_name} in hotbar.{slot + 1} before jump; "
                        f"held={self._plain_item_type(self.status_mod.equipment)}"
                    )
                break

            # Keep use pressed throughout the jump. This mirrors manual
            # pillar-jumping more closely than a delayed one-tick use:
            # as soon as the feet leave enough space above the previous
            # block, Minecraft can place the selected block under the agent.
            for _ in range(max(1, jump_place_ticks)):
                a_use = self.env.noop_action()
                a_use["jump"] = np.array(1)
                a_use["use"] = np.array(1)
                _hold_pitch_down(a_use)
                self.raw_step(a_use)
                steps_used += 1

            # Post-placement: settle on the new block (default 4 ticks).
            for _ in range(max(1, post_jump_settle)):
                aw = self.env.noop_action()
                _hold_pitch_down(aw)
                self.raw_step(aw)
                steps_used += 1

            try:
                new_y = float(self.cache["info"]["location_stats"].get("ypos", last_y))
            except Exception:
                new_y = last_y
            traj.append(new_y)

            if new_y > last_y + 0.5:
                blocks_used += 1
                last_y = new_y
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count >= max_stuck_per_cycle:
                    break

        # Phase 3: restore pitch to roughly horizontal
        for _ in range(15):
            try:
                cur_pitch = float(self.cache["info"]["location_stats"].get("pitch", 0))
            except Exception:
                cur_pitch = 0.0
            if abs(cur_pitch) < 5.0:
                break
            action = self.env.noop_action()
            delta = max(min(-cur_pitch, 12.0), -12.0)
            action["camera"] = np.array([delta, 0])
            self.raw_step(action)
            steps_used += 1

        try:
            end_y = float(self.cache["info"]["location_stats"].get("ypos", last_y))
        except Exception:
            end_y = last_y
        success = (end_y - start_y) >= 1.0
        if blocks_used == 0 and break_reason:
            reason = break_reason
        elif blocks_used == 0:
            reason = "no_placement_succeeded"
        elif end_y >= target_y - 0.5:
            reason = "reached_target"
        elif break_reason:
            reason = break_reason
        elif stuck_count >= max_stuck_per_cycle:
            reason = "stuck_no_progress"
        elif steps_used >= max_steps:
            reason = "step_budget_exhausted"
        else:
            reason = "block_budget_exhausted"

        result = {
            "success": success,
            "blocks_used": blocks_used,
            "dy": end_y - start_y,
            "start_y": start_y,
            "end_y": end_y,
            "steps_used": steps_used,
            "reason": reason,
            "trajectory": traj,
            "prep_actions": prep_actions,
        }
        self._record_recovery("pillar_up", result)
        if self.logger:
            self.logger.info(
                f"[pillar_up] done: dy={result['dy']:.1f} blocks_used={blocks_used} "
                f"steps_used={steps_used} reason={reason} success={success}"
            )
        return result

    # ---------------------------------------------------------------- #
    #  Environment perception & raise-to-height action                  #
    #                                                                   #
    #  Higher-level wrappers around the low-level `pillar_up` primitive.#
    #  These give the planner / decisioner a single entry point for     #
    #  "lift me up to access an ore band" without having to manage      #
    #  hotbar slots, pitch, or block selection by hand.                 #
    # ---------------------------------------------------------------- #

    # Canonical ore-band map (mirrors `random_ore` near the top of this
    # file). Keys are ore names; values are (min_y, max_y) inclusive.
    # Used by `perceive_height_context` to suggest a target_y.
    ORE_HEIGHT_BANDS: Dict[str, Tuple[int, int]] = {
        "coal_ore": (45, 50),
        "iron_ore": (26, 43),
        "gold_ore": (17, 26),
        "redstone_ore": (5, 16),
        "diamond_ore": (1, 14),
    }

    def _hotbar_free_slot(self, prefer_dest: int = 0) -> int:
        """Return a hotbar slot suitable to receive a placeable block.

        Preference order:
          1. The caller-suggested `prefer_dest` if it is currently empty.
          2. The first empty hotbar slot.
          3. The first hotbar slot whose item is *not* a tool / weapon
             (so we never trash a pickaxe / axe / sword while preparing
             to pillar up).
          4. Fallback: `prefer_dest`.
        """
        try:
            inventory = self.cache["info"].get("plain_inventory", {})
        except Exception:
            inventory = {}

        def _slot_item(s: int) -> Tuple[str, int]:
            item = inventory.get(s) or inventory.get(str(s)) or {}
            if not isinstance(item, dict):
                return "air", 0
            return (
                self._plain_item_type(item.get("type", "air")),
                self._plain_item_quantity(item.get("quantity", 0)),
            )

        # 1. caller preference, if empty
        name, qty = _slot_item(prefer_dest)
        if name in ("air", "none", "") or qty <= 0:
            return prefer_dest

        # 2. any empty hotbar slot
        for slot in range(9):
            name, qty = _slot_item(slot)
            if name in ("air", "none", "") or qty <= 0:
                return slot

        # 3. first hotbar slot that does not hold a tool / weapon
        protected_suffixes = ("_pickaxe", "_axe", "_shovel", "_hoe", "_sword")
        for slot in range(9):
            name, qty = _slot_item(slot)
            if not any(name.endswith(suf) for suf in protected_suffixes):
                return slot

        # 4. give up, overwrite the requested destination
        return prefer_dest

    def _find_placeable_block_anywhere(
        self,
        prefer: tuple[str, ...] | None = None,
    ) -> Tuple[int | None, str | None, int]:
        """Search the entire inventory for a placeable block.

        Returns ``(slot_index, item_name, quantity)``. ``slot_index`` is
        the observation slot id (0-8 = hotbar, 9-35 = main inventory).
        Returns ``(None, None, 0)`` if no placeable block exists.
        """
        prefer = prefer or self.PILLAR_PREFER_BLOCKS
        try:
            inventory = self.cache["info"].get("plain_inventory", {})
        except Exception:
            return None, None, 0
        for block_name in prefer:
            best_slot: int | None = None
            best_qty = 0
            best_in_hotbar = False
            for raw_slot, item in inventory.items():
                try:
                    slot_int = int(raw_slot)
                except Exception:
                    continue
                if not (0 <= slot_int <= 35):
                    continue
                if self._plain_item_type(item.get("type", "")) != block_name:
                    continue
                qty = self._plain_item_quantity(item.get("quantity", 0))
                if qty <= 0:
                    continue
                in_hotbar = 0 <= slot_int <= 8
                # Prefer hotbar slots, otherwise pick the largest stack.
                if best_slot is None:
                    best_slot, best_qty, best_in_hotbar = slot_int, qty, in_hotbar
                elif in_hotbar and not best_in_hotbar:
                    best_slot, best_qty, best_in_hotbar = slot_int, qty, in_hotbar
                elif in_hotbar == best_in_hotbar and qty > best_qty:
                    best_slot, best_qty, best_in_hotbar = slot_int, qty, in_hotbar
            if best_slot is not None:
                return best_slot, block_name, best_qty
        return None, None, 0

    def _ensure_placeable_block_in_hotbar(
        self,
        prefer: tuple[str, ...] | None = None,
        dest_hotbar_slot: int = 0,
    ) -> Tuple[int | None, str | None, str]:
        """Guarantee a placeable block is reachable from the hotbar.

        If a placeable block already lives in the hotbar (slot 0-8) the
        method returns immediately. Otherwise it tries to relocate one
        stack from the main inventory (slot 9-35) into a hotbar slot
        using the ``/replaceitem`` cheat (the same channel that
        ``random_ore`` uses for ``/setblock``).

        Returns ``(slot, block_name, prep_action)`` where ``prep_action``
        is one of:

        * ``"hotbar_ready"`` -- a placeable block was already in hotbar.
        * ``"swapped_to_hotbar"`` -- pulled from main inventory.
        * ``"none_available"`` -- no placeable block anywhere.
        * ``"swap_failed"`` -- block exists in main inventory but the
          ``/replaceitem`` command failed (and we cannot fall back).
        """
        prefer = prefer or self.PILLAR_PREFER_BLOCKS

        # Fast path: existing pillar_up logic already finds a hotbar slot.
        slot, block_name = self._find_placeable_block_slot(prefer)
        if slot is not None:
            return slot, block_name, "hotbar_ready"

        # Search main inventory.
        any_slot, any_block, qty = self._find_placeable_block_anywhere(prefer)
        if any_slot is None or any_block is None:
            return None, None, "none_available"
        if 0 <= any_slot <= 8:
            # Should have been caught above, but be defensive.
            return any_slot, any_block, "hotbar_ready"

        target_slot = self._hotbar_free_slot(dest_hotbar_slot)
        # Cap quantity at 64 (Minecraft stack limit) so /replaceitem
        # accepts the count argument cleanly.
        qty = max(1, min(int(qty), 64))

        def _refresh_inventory(ticks: int = 3) -> None:
            for _ in range(max(1, ticks)):
                try:
                    self.raw_step(self.env.noop_action())
                except Exception:
                    break

        def _hotbar_slot_contains(slot_index: int, block_name: str) -> bool:
            try:
                inventory = self.cache["info"].get("plain_inventory", {})
                item = inventory.get(slot_index) or inventory.get(str(slot_index)) or {}
            except Exception:
                return False
            if not isinstance(item, dict):
                return False
            return (
                self._plain_item_type(item.get("type", "")) == block_name
                and self._plain_item_quantity(item.get("quantity", 0)) > 0
            )

        cmd_put = (
            f"/replaceitem entity @s hotbar.{target_slot} "
            f"minecraft:{any_block} {qty}"
        )
        cmd_clear = (
            f"/replaceitem entity @s inventory.{any_slot - 9} "
            f"minecraft:air 1"
        )
        try:
            self.env.execute_cmd(cmd_put)
            _refresh_inventory(ticks=4)
            swap_visible = _hotbar_slot_contains(target_slot, any_block)
            if swap_visible:
                self.env.execute_cmd(cmd_clear)
                _refresh_inventory(ticks=1)
        except Exception as exc:
            if self.logger:
                self.logger.warning(
                    f"[pillar_up_smart] /replaceitem failed: {exc!s}; "
                    f"cmd_put={cmd_put!r} cmd_clear={cmd_clear!r}"
                )
            return None, None, "swap_failed"

        if not swap_visible:
            if self.logger:
                self.logger.warning(
                    f"[pillar_up_smart] /replaceitem did not become visible: "
                    f"cmd_put={cmd_put!r} target_hotbar={target_slot} "
                    f"block={any_block}"
                )
            return None, None, "swap_failed_not_visible"

        if self.logger:
            self.logger.info(
                f"[pillar_up_smart] swapped {any_block} x{qty} "
                f"from inv-slot {any_slot} -> hotbar.{target_slot}"
            )
        return target_slot, any_block, "swapped_to_hotbar"

    def perceive_height_context(
        self,
        look_for: str | None = None,
        ascend_margin: int = 1,
    ) -> Dict[str, Any]:
        """Snapshot the agent's vertical environment.

        Reports the current Y, the canonical ore band the agent is
        currently in (if any), the placeable-block inventory state, and
        — when ``look_for`` is provided — a recommended ``target_dy``
        and ``target_y`` to reach the corresponding ore band.

        Args:
            look_for: optional ore name (e.g. ``"diamond_ore"``). The
                method maps this to the canonical Minecraft Y range and
                proposes a target Y. ``None`` skips the recommendation.
            ascend_margin: extra blocks to add on top of the band's max
                Y when the agent must climb up. Helps the policy reach
                ores that sit slightly above the agent.

        Returns a dict with keys:
            current_y, current_pitch, current_yaw, in_band (str|None),
            placeable_in_hotbar (int), placeable_in_inventory (int),
            placeable_total (int), preferred_block (str|None),
            target_band (Tuple[int,int]|None), target_y (float|None),
            recommend_pillar_up (bool), recommended_dy (int|None),
            recommended_action (str: "noop"/"pillar_up"/"descend"/"unsupported"),
            reason (str).
        """
        loc = (self.cache.get("info") or {}).get("location_stats", {}) or {}

        def _scalar(key: str, default: float) -> float:
            try:
                return float(np.asarray(loc.get(key, default)).reshape(-1)[0])
            except Exception:
                return float(default)

        current_y = _scalar("ypos", 64.0)
        current_pitch = _scalar("pitch", 0.0)
        current_yaw = _scalar("yaw", 0.0)

        # Tally placeable-block availability (hotbar vs. main inventory).
        try:
            inventory = self.cache["info"].get("plain_inventory", {})
        except Exception:
            inventory = {}
        placeable_in_hotbar = 0
        placeable_in_inventory = 0
        preferred_block: str | None = None
        for block_name in self.PILLAR_PREFER_BLOCKS:
            for raw_slot, item in inventory.items():
                try:
                    slot_int = int(raw_slot)
                except Exception:
                    continue
                if not (0 <= slot_int <= 35):
                    continue
                if self._plain_item_type(item.get("type", "")) != block_name:
                    continue
                qty = self._plain_item_quantity(item.get("quantity", 0))
                if qty <= 0:
                    continue
                if 0 <= slot_int <= 8:
                    placeable_in_hotbar += qty
                else:
                    placeable_in_inventory += qty
                if preferred_block is None:
                    preferred_block = block_name
        placeable_total = placeable_in_hotbar + placeable_in_inventory

        # Which canonical band does the agent stand in right now?
        in_band: str | None = None
        for ore, (lo, hi) in self.ORE_HEIGHT_BANDS.items():
            if lo <= current_y <= hi:
                in_band = ore
                break

        # Compute a recommendation toward `look_for` if requested.
        target_band: Tuple[int, int] | None = None
        target_y: float | None = None
        recommended_dy: int | None = None
        recommended_action = "noop"
        reason = "no_target_specified"

        if look_for:
            band = self.ORE_HEIGHT_BANDS.get(look_for)
            if band is None:
                recommended_action = "unsupported"
                reason = f"unknown_ore:{look_for}"
            else:
                target_band = band
                lo, hi = band
                if current_y < lo:
                    target_y = float(min(hi, lo + max(0, ascend_margin)))
                    recommended_dy = max(1, int(round(target_y - current_y)))
                    if placeable_total > 0:
                        recommended_action = "pillar_up"
                        reason = (
                            f"below_band:y={current_y:.1f}<lo={lo} "
                            f"need_dy={recommended_dy}"
                        )
                    else:
                        recommended_action = "unsupported"
                        reason = "below_band_but_no_placeable_block"
                elif current_y > hi + ascend_margin:
                    # Mining downward is out of scope for this primitive;
                    # only flag the situation so the planner can decide.
                    target_y = float(hi)
                    recommended_dy = -int(round(current_y - hi))
                    recommended_action = "descend"
                    reason = (
                        f"above_band:y={current_y:.1f}>hi={hi} "
                        f"need_dy={recommended_dy}"
                    )
                else:
                    target_y = float(current_y)
                    recommended_dy = 0
                    recommended_action = "noop"
                    reason = f"already_in_band:{look_for}"

        return {
            "current_y": current_y,
            "current_pitch": current_pitch,
            "current_yaw": current_yaw,
            "in_band": in_band,
            "placeable_in_hotbar": int(placeable_in_hotbar),
            "placeable_in_inventory": int(placeable_in_inventory),
            "placeable_total": int(placeable_total),
            "preferred_block": preferred_block,
            "target_band": target_band,
            "target_y": target_y,
            "recommend_pillar_up": recommended_action == "pillar_up",
            "recommended_dy": recommended_dy,
            "recommended_action": recommended_action,
            "reason": reason,
        }

    def pillar_up_smart(
        self,
        target_dy: int = 5,
        target_y: float | None = None,
        max_blocks: int = 32,
        max_steps: int = 400,
        prefer_blocks: tuple[str, ...] | None = None,
        dest_hotbar_slot: int = 0,
    ) -> Dict[str, Any]:
        """Inventory-aware wrapper around :py:meth:`pillar_up`.

        Differences vs. the low-level primitive:

        * Pulls a placeable block from main inventory into the hotbar
          (via ``/replaceitem``) when no placeable block is in hotbar.
        * Accepts an absolute ``target_y`` in addition to ``target_dy``;
          if both are given ``target_y`` takes precedence.
        * Returns the same dict as ``pillar_up`` plus ``prep_action``,
          ``target_y``, ``preferred_block``, and ``planned_dy`` for
          inspection by the planner / decisioner.

        Notes:
            * ``target_dy`` is clamped to ``[1, max_blocks]`` to avoid
              wasting blocks; pass a larger ``max_blocks`` for big climbs.
            * If no placeable block exists anywhere in the inventory the
              method returns ``success=False, reason="no_placeable_block"``
              without performing any environment interaction beyond the
              cache refresh that may already have happened.
        """
        # Snapshot starting Y so we can resolve target_y vs target_dy.
        try:
            start_y = float(self.cache["info"]["location_stats"].get("ypos", 64.0))
        except Exception:
            start_y = 64.0

        if target_y is not None:
            planned_dy = int(round(float(target_y) - start_y))
        else:
            planned_dy = int(target_dy)
        if planned_dy <= 0:
            res = {
                "success": True,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": start_y,
                "end_y": start_y,
                "steps_used": 0,
                "reason": "already_at_or_above_target",
                "trajectory": [start_y],
                "prep_action": "noop",
                "target_y": float(target_y) if target_y is not None else start_y,
                "planned_dy": planned_dy,
                "preferred_block": None,
            }
            self._record_recovery("pillar_up_smart", res)
            return res

        planned_dy = max(1, min(planned_dy, int(max_blocks)))

        slot, block_name, prep_action = self._ensure_placeable_block_in_hotbar(
            prefer=prefer_blocks, dest_hotbar_slot=dest_hotbar_slot
        )
        if slot is None:
            res = {
                "success": False,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": start_y,
                "end_y": start_y,
                "steps_used": 0,
                "reason": f"no_placeable_block:{prep_action}",
                "trajectory": [start_y],
                "prep_action": prep_action,
                "target_y": float(target_y) if target_y is not None else start_y + planned_dy,
                "planned_dy": planned_dy,
                "preferred_block": None,
            }
            self._record_recovery("pillar_up_smart", res)
            if self.logger:
                self.logger.warning(
                    f"[pillar_up_smart] aborted: {res['reason']} "
                    f"y={start_y:.1f} planned_dy={planned_dy}"
                )
            return res

        if self.logger:
            self.logger.info(
                f"[pillar_up_smart] start: y={start_y:.1f} planned_dy={planned_dy} "
                f"target_y={(start_y + planned_dy):.1f} block={block_name} "
                f"slot={slot} prep={prep_action}"
            )

        result = self.pillar_up(
            target_dy=planned_dy,
            max_blocks=max_blocks,
            max_steps=max_steps,
            prefer_blocks=prefer_blocks,
            prepared_slot=slot,
            prepared_block=block_name,
        )
        result = dict(result)
        result["prep_action"] = prep_action
        result["target_y"] = (
            float(target_y) if target_y is not None else start_y + planned_dy
        )
        result["planned_dy"] = planned_dy
        result["preferred_block"] = block_name
        self._record_recovery("pillar_up_smart", result)
        return result

    def raise_to_height(
        self,
        target_y: float,
        max_blocks: int = 64,
        max_steps: int = 600,
        prefer_blocks: tuple[str, ...] | None = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper: raise the agent until y >= ``target_y``.

        This is the preferred entry point for the planner / decisioner
        because the recipe-driven goal is usually expressed in absolute
        coordinates ("be at y=50 to mine coal") rather than as a delta.
        """
        return self.pillar_up_smart(
            target_y=float(target_y),
            max_blocks=max_blocks,
            max_steps=max_steps,
            prefer_blocks=prefer_blocks,
        )

    def raise_to_ore_band(
        self,
        ore: str,
        max_blocks: int = 64,
        max_steps: int = 600,
        prefer_blocks: tuple[str, ...] | None = None,
        ascend_margin: int = 1,
    ) -> Dict[str, Any]:
        """Pillar up so the agent enters the canonical Y band of ``ore``.

        Combines :py:meth:`perceive_height_context` and
        :py:meth:`pillar_up_smart` so the planner only has to say
        "I want to mine coal" and the wrapper figures out the target Y.

        Returns the dict produced by ``pillar_up_smart``, augmented with
        ``ore``, ``perception`` (the perception snapshot used) and
        ``skipped`` (True if no climb was needed).
        """
        ctx = self.perceive_height_context(look_for=ore, ascend_margin=ascend_margin)
        if ctx["recommended_action"] == "noop":
            res = {
                "success": True,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": ctx["current_y"],
                "end_y": ctx["current_y"],
                "steps_used": 0,
                "reason": ctx["reason"],
                "trajectory": [ctx["current_y"]],
                "prep_action": "noop",
                "target_y": ctx["target_y"] if ctx["target_y"] is not None else ctx["current_y"],
                "planned_dy": 0,
                "preferred_block": ctx["preferred_block"],
                "ore": ore,
                "perception": ctx,
                "skipped": True,
            }
            self._record_recovery("raise_to_ore_band", res)
            return res
        if ctx["recommended_action"] != "pillar_up" or ctx["target_y"] is None:
            res = {
                "success": False,
                "blocks_used": 0,
                "dy": 0.0,
                "start_y": ctx["current_y"],
                "end_y": ctx["current_y"],
                "steps_used": 0,
                "reason": ctx["reason"],
                "trajectory": [ctx["current_y"]],
                "prep_action": "noop",
                "target_y": ctx["target_y"] if ctx["target_y"] is not None else ctx["current_y"],
                "planned_dy": ctx["recommended_dy"] or 0,
                "preferred_block": ctx["preferred_block"],
                "ore": ore,
                "perception": ctx,
                "skipped": True,
            }
            self._record_recovery("raise_to_ore_band", res)
            return res
        result = self.pillar_up_smart(
            target_y=ctx["target_y"],
            max_blocks=max_blocks,
            max_steps=max_steps,
            prefer_blocks=prefer_blocks,
        )
        result = dict(result)
        result["ore"] = ore
        result["perception"] = ctx
        result["skipped"] = False
        self._record_recovery("raise_to_ore_band", result)
        return result

    # ---------------------------------------------------------------- #
    #  Scripted forward dig / lateral offset (post-pillar-up safety)    #
    # ---------------------------------------------------------------- #

    def _mined_block_count(self) -> int:
        """Total Minecraft mined-block stat across all item types."""
        ledger = self.cache.get("resource_ledger") or {}
        mined = ledger.get("mined_blocks") or {}
        try:
            return sum(int(v) for v in mined.values() if isinstance(v, (int, float)))
        except Exception:
            return 0

    def dig_forward_blocks(
        self,
        n_blocks: int = 3,
        max_steps: int = 240,
        max_steps_per_block: int = 80,
        prefer_pickaxe: bool = True,
    ) -> Dict[str, Any]:
        """Scripted two-block-high horizontal tunnel while preserving Y.

        Used immediately after ``raise_to_height`` so the agent explores
        horizontally at the first target-ore height before STEVE-1 may resume
        the original dig-down prompt. This primitive bypasses STEVE-1
        (uses ``raw_step``) and treats physical x/z advance as the only
        success signal. For each tunnel segment it clears the body/head block
        first, then the lower-front blocker, and only counts the segment when
        the agent has moved meaningfully from that segment's start.

        Args:
            n_blocks: how many forward blocks to mine.
            max_steps: total tick budget.
            max_steps_per_block: tick budget per block.
            prefer_pickaxe: switch to best pickaxe via ``find_best_pickaxe``
                before digging.

        Returns:
            dict {success, blocks_dug, steps_used, reason, start_x,
                  start_y, start_z, end_x, end_y, end_z}.
        """
        loc = (self.cache.get("info") or {}).get("location_stats", {}) or {}

        def _scalar(key: str, default: float) -> float:
            try:
                return float(np.asarray(loc.get(key, default)).reshape(-1)[0])
            except Exception:
                return float(default)

        start_x = _scalar("xpos", 0.0)
        start_y = _scalar("ypos", 0.0)
        start_z = _scalar("zpos", 0.0)
        start_yaw = _scalar("yaw", 0.0)

        steps_used = 0
        blocks_dug = 0

        if self.logger:
            self.logger.info(
                f"[dig_forward_blocks] start: n={n_blocks} "
                f"pos=({start_x:.1f},{start_y:.1f},{start_z:.1f})"
            )

        # Equip best pickaxe (mirrors find_best_pickaxe used elsewhere).
        if prefer_pickaxe:
            try:
                slot = self.find_best_pickaxe()
            except Exception:
                slot = None
            if slot:
                previous_can_change_hotbar = self.can_change_hotbar
                self.can_change_hotbar = True
                try:
                    for _ in range(4):
                        if steps_used >= int(max_steps):
                            break
                        a_eq = self.env.noop_action()
                        a_eq[slot] = np.array(1)
                        self.raw_step(a_eq)
                        steps_used += 1
                    if self.logger:
                        self.logger.info(
                            f"[dig_forward_blocks] equipped pickaxe via {slot}; "
                            f"held={self._plain_item_type(self.status_mod.equipment)}"
                        )
                finally:
                    self.can_change_hotbar = previous_can_change_hotbar

        # Phase 1: pitch back to ~0 (look horizontal).
        for _ in range(15):
            try:
                cur_pitch = float(
                    np.asarray(
                        (self.cache.get("info") or {}).get("location_stats", {}).get(
                            "pitch", 0
                        )
                    ).reshape(-1)[0]
                )
            except Exception:
                cur_pitch = 0.0
            if abs(cur_pitch) < 5.0:
                break
            a = self.env.noop_action()
            delta = max(min(-cur_pitch, 12.0), -12.0)
            a["camera"] = np.array([delta, 0])
            self.raw_step(a)
            steps_used += 1

        # Phase 2: tunnel boring by per-segment movement success. Each
        # segment starts from the current x/z and is counted only after the
        # agent physically advances far enough from that segment start. This
        # avoids the old multi-block bug where the first x/z displacement made
        # every later segment look successful.
        head_pitch_target = float(os.environ.get("XENON_CORRIDOR_HEAD_PITCH", "0.0"))
        blocked_front_pitch_target = float(
            os.environ.get("XENON_CORRIDOR_BLOCKED_FRONT_PITCH", "0.0")
        )
        up_pitch_target = float(os.environ.get("XENON_CORRIDOR_UP_PITCH", "-65.0"))
        raw_blocked_feet_pitch_target = float(
            os.environ.get("XENON_CORRIDOR_BLOCKED_FEET_PITCH", "55.0")
        )
        # Aim at the lower-front blocker, not the block directly under the
        # agent. Near-vertical pitch (for example 82 degrees) tends to mine
        # the support block under the feet and causes a fall/relevel loop.
        blocked_feet_pitch_cap = float(
            os.environ.get("XENON_CORRIDOR_BLOCKED_FEET_MAX_PITCH", "65.0")
        )
        blocked_feet_pitch_target = min(
            raw_blocked_feet_pitch_target,
            blocked_feet_pitch_cap,
        )
        walk_ticks_per_block = int(os.environ.get("XENON_CORRIDOR_WALK_TICKS", "6"))
        attack_forward = os.environ.get("XENON_CORRIDOR_ATTACK_FORWARD", "0") == "1"
        max_y_drop = float(os.environ.get("XENON_LATERAL_MAX_Y_DROP", "0.75"))
        max_position_jump = float(os.environ.get("XENON_CORRIDOR_MAX_POSITION_JUMP", "6.0"))
        min_move_delta = float(os.environ.get("XENON_CORRIDOR_MIN_MOVE_DELTA", "1.0"))
        segment_min_move_delta = float(
            os.environ.get("XENON_CORRIDOR_SEGMENT_MIN_MOVE_DELTA", str(min_move_delta))
        )
        max_unstuck_passes = int(os.environ.get("XENON_CORRIDOR_UNSTUCK_PASSES", "2"))
        blocked_head_budget = int(os.environ.get("XENON_CORRIDOR_BLOCKED_HEAD_BUDGET", "24"))
        blocked_feet_budget = int(os.environ.get("XENON_CORRIDOR_BLOCKED_FEET_BUDGET", "36"))
        blocked_up_budget = int(os.environ.get("XENON_CORRIDOR_BLOCKED_UP_BUDGET", "0"))
        axis_lock = os.environ.get("XENON_CORRIDOR_AXIS_LOCK", "1") == "1"
        yaw_mode = os.environ.get("XENON_CORRIDOR_YAW_MODE", "fan30").strip().lower()
        if yaw_mode == "snap90":
            aligned_yaw = round(start_yaw / 90.0) * 90.0
        elif yaw_mode == "fan30":
            aligned_yaw = start_yaw
        else:
            # Default: preserve the direction STEVE-1 was already facing.
            aligned_yaw = start_yaw
            yaw_mode = "hold"
        raw_yaw_offsets = os.environ.get(
            "XENON_CORRIDOR_YAW_OFFSETS",
            "0,30,-30" if yaw_mode == "fan30" else "0",
        )

        def _parse_yaw_offsets(raw: str) -> List[float]:
            offsets: List[float] = []
            for part in str(raw).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    offsets.append(float(part))
                except ValueError:
                    continue
            return offsets or [0.0]

        yaw_offsets = _parse_yaw_offsets(raw_yaw_offsets)
        direction_sweep_enabled = (
            os.environ.get("XENON_CORRIDOR_DIRECTION_SWEEP", "0") == "1"
        )
        direction_sweep_offsets = _parse_yaw_offsets(
            os.environ.get("XENON_CORRIDOR_DIRECTION_SWEEP_OFFSETS", "0,90,-90,180")
        )
        if not direction_sweep_offsets:
            direction_sweep_offsets = [0.0]
        sweep_index = 0
        sweep_anchor_yaw = start_yaw
        sweep_base_offset = 0.0
        if direction_sweep_enabled and axis_lock and len(direction_sweep_offsets) > 1:
            raw_index = getattr(self, "_corridor_direction_sweep_index", 0)
            try:
                sweep_index = int(raw_index) % len(direction_sweep_offsets)
            except Exception:
                sweep_index = 0
            raw_anchor = getattr(self, "_corridor_direction_sweep_anchor_yaw", None)
            try:
                sweep_anchor_yaw = (
                    float(raw_anchor) if raw_anchor is not None else float(start_yaw)
                )
            except Exception:
                sweep_anchor_yaw = float(start_yaw)
            if raw_anchor is None:
                setattr(self, "_corridor_direction_sweep_anchor_yaw", sweep_anchor_yaw)
            sweep_base_offset = float(direction_sweep_offsets[sweep_index])
            if yaw_mode == "snap90":
                aligned_yaw = round(sweep_anchor_yaw / 90.0) * 90.0 + sweep_base_offset
            else:
                aligned_yaw = sweep_anchor_yaw + sweep_base_offset
        stabilize_floor_enabled = (
            os.environ.get("XENON_CORRIDOR_STABILIZE_FLOOR", "1") == "1"
        )
        stabilize_after_full_sweep = (
            os.environ.get("XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP", "1") == "1"
        )
        stabilize_floor_requested = bool(
            getattr(self, "_corridor_floor_stabilize_requested", False)
        )
        try:
            stabilize_floor_radius = max(
                0, int(os.environ.get("XENON_CORRIDOR_STABILIZE_RADIUS", "2"))
            )
        except ValueError:
            stabilize_floor_radius = 2
        stabilize_floor_lanes = (
            os.environ.get("XENON_CORRIDOR_STABILIZE_LANES", "1") == "1"
        )
        try:
            stabilize_floor_length = max(
                0, int(os.environ.get("XENON_CORRIDOR_STABILIZE_LENGTH", "0"))
            )
        except ValueError:
            stabilize_floor_length = 0
        try:
            stabilize_floor_width = max(
                0, int(os.environ.get("XENON_CORRIDOR_STABILIZE_WIDTH", "1"))
            )
        except ValueError:
            stabilize_floor_width = 1
        stabilize_floor_block = (
            os.environ.get("XENON_CORRIDOR_STABILIZE_BLOCK", "cobblestone")
            .replace("minecraft:", "")
            .strip()
            or "cobblestone"
        )
        move_attack_ticks = int(
            os.environ.get(
                "XENON_CORRIDOR_MOVE_ATTACK_TICKS",
                str(max(walk_ticks_per_block, 8)),
            )
        )
        stop_after_first_move = os.environ.get("XENON_CORRIDOR_STOP_AFTER_MOVE", "1") == "1"
        target_blocks = max(1, int(n_blocks))
        default_min_success = "1" if stop_after_first_move else str(target_blocks)
        try:
            min_success_blocks = int(
                os.environ.get("XENON_CORRIDOR_MIN_SUCCESS_BLOCKS", default_min_success)
            )
        except ValueError:
            min_success_blocks = int(default_min_success)
        min_success_blocks = max(1, min(target_blocks, min_success_blocks))
        terminal_abort = False
        position_jump_abort = False

        def _current_y(default: float) -> float:
            try:
                return float(
                    np.asarray(
                        (self.cache.get("info") or {}).get("location_stats", {}).get(
                            "ypos", default
                        )
                    ).reshape(-1)[0]
                )
            except Exception:
                return float(default)

        def _current_xyz(
            default_x: float,
            default_y: float,
            default_z: float,
        ) -> Tuple[float, float, float]:
            loc_now = (self.cache.get("info") or {}).get("location_stats", {}) or {}
            try:
                x_now = float(np.asarray(loc_now.get("xpos", default_x)).reshape(-1)[0])
            except Exception:
                x_now = float(default_x)
            try:
                y_now = float(np.asarray(loc_now.get("ypos", default_y)).reshape(-1)[0])
            except Exception:
                y_now = float(default_y)
            try:
                z_now = float(np.asarray(loc_now.get("zpos", default_z)).reshape(-1)[0])
            except Exception:
                z_now = float(default_z)
            return x_now, y_now, z_now

        def _height_dropped() -> bool:
            return _current_y(start_y) < (start_y - max_y_drop)

        def _current_xz(default_x: float, default_z: float) -> Tuple[float, float]:
            loc_now = (self.cache.get("info") or {}).get("location_stats", {}) or {}
            try:
                x_now = float(np.asarray(loc_now.get("xpos", default_x)).reshape(-1)[0])
            except Exception:
                x_now = float(default_x)
            try:
                z_now = float(np.asarray(loc_now.get("zpos", default_z)).reshape(-1)[0])
            except Exception:
                z_now = float(default_z)
            return x_now, z_now

        def _horizontal_delta(x0: float, z0: float, x1: float, z1: float) -> float:
            return ((x1 - x0) ** 2 + (z1 - z0) ** 2) ** 0.5

        def _block_cell(x: float, z: float) -> Tuple[int, int]:
            return int(np.floor(float(x))), int(np.floor(float(z)))

        def _block_cell_changed(x0: float, z0: float, x1: float, z1: float) -> bool:
            return _block_cell(x0, z0) != _block_cell(x1, z1)

        def _stabilize_corridor_floor() -> Dict[str, Any]:
            """Patch air floor cells after every candidate direction drops.

            A local pad fixes the current foot position. Lane fill fixes the
            more common underground failure: the next horizontal tunnel opens
            into cave air, so the agent falls before it can count a clean
            horizontal displacement.
            """
            if not hasattr(self.env, "execute_cmd"):
                return {"success": False, "reason": "execute_cmd_unavailable"}
            loc_x, loc_y, loc_z = _current_xyz(start_x, start_y, start_z)
            bx = int(np.floor(loc_x))
            by = int(np.floor(loc_y)) - 1
            bz = int(np.floor(loc_z))
            radius = stabilize_floor_radius
            lane_length = (
                stabilize_floor_length
                if stabilize_floor_length > 0
                else max(target_blocks + 2, radius + 1)
            )
            cells: set[Tuple[int, int]] = set()
            commands_sent = 0
            errors: List[str] = []
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    cells.add((bx + dx, bz + dz))
            lane_yaws: List[float] = []
            if stabilize_floor_lanes:
                if (
                    direction_sweep_enabled
                    and axis_lock
                    and len(direction_sweep_offsets) > 1
                ):
                    lane_yaws = [
                        float(sweep_anchor_yaw) + float(offset)
                        for offset in direction_sweep_offsets
                    ]
                elif axis_lock:
                    lane_yaws = [float(aligned_yaw)]
                for lane_yaw in lane_yaws:
                    yaw_rad = np.deg2rad(lane_yaw)
                    forward_x = -float(np.sin(yaw_rad))
                    forward_z = float(np.cos(yaw_rad))
                    side_x = float(np.cos(yaw_rad))
                    side_z = float(np.sin(yaw_rad))
                    for dist in range(1, lane_length + 1):
                        for side in range(-stabilize_floor_width, stabilize_floor_width + 1):
                            tx = bx + int(round(forward_x * dist + side_x * side))
                            tz = bz + int(round(forward_z * dist + side_z * side))
                            cells.add((tx, tz))
            for tx, tz in sorted(cells):
                    for air_name in ("air", "cave_air"):
                        cmd = (
                            f"/execute if block {tx} {by} {tz} minecraft:{air_name} "
                            f"run setblock {tx} {by} {tz} minecraft:{stabilize_floor_block}"
                        )
                        try:
                            self.env.execute_cmd(cmd)
                            commands_sent += 1
                        except Exception as exc:
                            if len(errors) < 3:
                                errors.append(str(exc))
            result = {
                "success": commands_sent > 0 and not errors,
                "reason": "floor_stabilized" if commands_sent > 0 else "no_commands_sent",
                "center": (bx, by, bz),
                "radius": radius,
                "lane_fill": stabilize_floor_lanes,
                "lane_length": lane_length if stabilize_floor_lanes else 0,
                "lane_half_width": stabilize_floor_width if stabilize_floor_lanes else 0,
                "lane_yaws": lane_yaws,
                "cells": len(cells),
                "block": stabilize_floor_block,
                "commands_sent": commands_sent,
                "errors": errors,
            }
            self._record_recovery("corridor_floor_stabilize", result)
            if self.logger:
                self.logger.info(
                    "[dig_forward_blocks] floor stabilize after full sweep: "
                    "center=(%d,%d,%d) radius=%d lane_fill=%s lane_length=%d "
                    "lane_half_width=%d cells=%d block=%s commands=%d errors=%d",
                    bx,
                    by,
                    bz,
                    radius,
                    stabilize_floor_lanes,
                    lane_length if stabilize_floor_lanes else 0,
                    stabilize_floor_width if stabilize_floor_lanes else 0,
                    len(cells),
                    stabilize_floor_block,
                    commands_sent,
                    len(errors),
                )
            return result

        def _meaningful_lateral_move(
            x0: float,
            z0: float,
            x1: float,
            z1: float,
            threshold: float | None = None,
        ) -> bool:
            return _horizontal_delta(x0, z0, x1, z1) >= (
                min_move_delta if threshold is None else threshold
            )

        def _total_horizontal_delta() -> float:
            now_x, now_z = _current_xz(start_x, start_z)
            return _horizontal_delta(start_x, start_z, now_x, now_z)

        def _angle_delta(target: float, current: float) -> float:
            return (float(target) - float(current) + 180.0) % 360.0 - 180.0

        def _current_yaw(default: float) -> float:
            try:
                return float(
                    np.asarray(
                        (self.cache.get("info") or {}).get("location_stats", {}).get(
                            "yaw", default
                        )
                    ).reshape(-1)[0]
                )
            except Exception:
                return float(default)

        def _nudge_pitch_to(target_pitch: float, max_step: float = 8.0):
            try:
                cp = float(
                    np.asarray(
                        (self.cache.get("info") or {}).get("location_stats", {}).get(
                            "pitch", 0
                        )
                    ).reshape(-1)[0]
                )
            except Exception:
                cp = 0.0
            return max(min(target_pitch - cp, max_step), -max_step)

        def _nudge_yaw_to(target_yaw: float, max_step: float = 12.0) -> float:
            cy = _current_yaw(start_yaw)
            delta = _angle_delta(target_yaw, cy)
            return max(min(delta, max_step), -max_step)

        def _face_axis(
            target_pitch: float,
            target_yaw: float,
            budget: int = 18,
            pitch_tol: float = 3.0,
            yaw_tol: float = 5.0,
        ) -> None:
            """Rotate to a discrete underground mining direction."""
            nonlocal steps_used
            for _ in range(max(int(budget), 0)):
                if steps_used >= int(max_steps):
                    break
                pitch_delta = _nudge_pitch_to(target_pitch, max_step=10.0)
                yaw_delta = _nudge_yaw_to(target_yaw, max_step=14.0)
                if (
                    abs(pitch_delta) <= pitch_tol
                    and abs(_angle_delta(target_yaw, _current_yaw(start_yaw))) <= yaw_tol
                ):
                    break
                a = self.env.noop_action()
                a["camera"] = np.array([pitch_delta, yaw_delta])
                self.raw_step(a)
                steps_used += 1

        def _phase_attack(
            target_pitch: float,
            label: str,
            budget: int,
            force_forward: bool = False,
            sprint: bool = False,
            target_yaw: float | None = None,
        ) -> bool:
            """Hold attack at the given pitch until a block breaks or the
            budget is exhausted. Returns True if a block broke."""
            nonlocal steps_used, terminal_abort, position_jump_abort
            mined_at_phase_start = self._mined_block_count()
            ticks = 0
            while ticks < budget and steps_used < int(max_steps):
                if terminal_abort or _height_dropped():
                    return False
                a = self.env.noop_action()
                a["attack"] = np.array(1)
                if attack_forward or force_forward:
                    a["forward"] = np.array(1)
                if sprint:
                    a["sprint"] = np.array(1)
                pitch_delta = _nudge_pitch_to(target_pitch)
                yaw_delta = _nudge_yaw_to(target_yaw) if target_yaw is not None else 0.0
                if abs(pitch_delta) > 0.1 or abs(yaw_delta) > 0.1:
                    a["camera"] = np.array([pitch_delta, yaw_delta])
                prev_x, prev_y, prev_z = _current_xyz(start_x, start_y, start_z)
                _, _, done, _ = self.raw_step(a)
                steps_used += 1
                ticks += 1
                cur_x, cur_y, cur_z = _current_xyz(prev_x, prev_y, prev_z)
                jump = (
                    (cur_x - prev_x) ** 2
                    + (cur_y - prev_y) ** 2
                    + (cur_z - prev_z) ** 2
                ) ** 0.5
                if done or jump > max_position_jump:
                    terminal_abort = True
                    position_jump_abort = bool(jump > max_position_jump)
                    return False
                if self._mined_block_count() > mined_at_phase_start:
                    return True
            return False

        def _phase_move_attack(
            target_pitch: float,
            ticks: int,
            label: str,
            stop_on_move: bool = True,
            target_yaw: float | None = None,
            move_threshold: float | None = None,
        ) -> Tuple[float, int]:
            """Forward + sprint + attack. Returns (x/z displacement, mined delta)."""
            nonlocal steps_used, terminal_abort, position_jump_abort
            phase_start_x, phase_start_z = _current_xz(start_x, start_z)
            mined_at_start = self._mined_block_count()
            for _ in range(ticks):
                if steps_used >= int(max_steps):
                    break
                if terminal_abort or _height_dropped():
                    break
                a = self.env.noop_action()
                a["forward"] = np.array(1)
                a["sprint"] = np.array(1)
                a["attack"] = np.array(1)
                pitch_delta = _nudge_pitch_to(target_pitch, max_step=10.0)
                yaw_delta = _nudge_yaw_to(target_yaw, max_step=14.0) if target_yaw is not None else 0.0
                if abs(pitch_delta) > 0.1 or abs(yaw_delta) > 0.1:
                    a["camera"] = np.array([pitch_delta, yaw_delta])
                prev_x, prev_y, prev_z = _current_xyz(start_x, start_y, start_z)
                _, _, done, _ = self.raw_step(a)
                steps_used += 1
                cur_x, cur_y, cur_z = _current_xyz(prev_x, prev_y, prev_z)
                jump = (
                    (cur_x - prev_x) ** 2
                    + (cur_y - prev_y) ** 2
                    + (cur_z - prev_z) ** 2
                ) ** 0.5
                if done or jump > max_position_jump:
                    terminal_abort = True
                    position_jump_abort = bool(jump > max_position_jump)
                    break
                if stop_on_move:
                    now_x, now_z = _current_xz(phase_start_x, phase_start_z)
                    if _meaningful_lateral_move(
                        phase_start_x,
                        phase_start_z,
                        now_x,
                        now_z,
                        move_threshold,
                    ):
                        break
            phase_end_x, phase_end_z = _current_xz(phase_start_x, phase_start_z)
            return (
                _horizontal_delta(phase_start_x, phase_start_z, phase_end_x, phase_end_z),
                max(0, self._mined_block_count() - mined_at_start),
            )

        def _clear_forward_blocker(
            target_yaw: float | None,
            segment_start_x: float,
            segment_start_z: float,
        ) -> Tuple[bool, float]:
            """Clear no-displacement blockers in physical body order.

            The agent is two blocks tall. A successful lateral relocation
            may require clearing a two-block-high forward channel. We always
            attack the horizontal front block first, then the diagonal
            lower-front block. Optional upward clearing is disabled by default.
            """
            before = self._mined_block_count()
            moved_head, _ = _phase_move_attack(
                blocked_front_pitch_target,
                max(blocked_head_budget, 1),
                "blocked_front",
                target_yaw=target_yaw,
                move_threshold=segment_min_move_delta,
            )
            now_x, now_z = _current_xz(segment_start_x, segment_start_z)
            segment_delta = _horizontal_delta(segment_start_x, segment_start_z, now_x, now_z)
            if (
                segment_delta >= segment_min_move_delta
                or terminal_abort
                or _height_dropped()
                or steps_used >= int(max_steps)
            ):
                return self._mined_block_count() > before, max(moved_head, segment_delta)
            if blocked_up_budget > 0:
                _phase_attack(
                    up_pitch_target,
                    "blocked_up",
                    blocked_up_budget,
                    force_forward=False,
                    sprint=False,
                    target_yaw=target_yaw,
                )
                if terminal_abort or _height_dropped() or steps_used >= int(max_steps):
                    now_x, now_z = _current_xz(segment_start_x, segment_start_z)
                    segment_delta = _horizontal_delta(segment_start_x, segment_start_z, now_x, now_z)
                    return self._mined_block_count() > before, max(moved_head, segment_delta)
            moved_feet, _ = _phase_move_attack(
                blocked_feet_pitch_target,
                max(blocked_feet_budget, 1),
                "blocked_diagonal_down",
                target_yaw=target_yaw,
                move_threshold=segment_min_move_delta,
            )
            now_x, now_z = _current_xz(segment_start_x, segment_start_z)
            segment_delta = _horizontal_delta(segment_start_x, segment_start_z, now_x, now_z)
            if (
                segment_delta >= segment_min_move_delta
                or terminal_abort
                or _height_dropped()
                or steps_used >= int(max_steps)
            ):
                return self._mined_block_count() > before, max(moved_head, moved_feet, segment_delta)
            if not terminal_abort and not _height_dropped() and steps_used < int(max_steps):
                _phase_attack(
                    blocked_feet_pitch_target,
                    "blocked_diagonal_down_finish",
                    max(blocked_feet_budget // 3, 1),
                    force_forward=True,
                    sprint=True,
                    target_yaw=target_yaw,
                )
            now_x, now_z = _current_xz(segment_start_x, segment_start_z)
            segment_delta = _horizontal_delta(segment_start_x, segment_start_z, now_x, now_z)
            return self._mined_block_count() > before, max(moved_head, moved_feet, segment_delta)

        def _restore_pitch_to(target_pitch: float, budget: int = 12) -> None:
            nonlocal steps_used
            for _ in range(max(int(budget), 0)):
                pitch_delta = _nudge_pitch_to(target_pitch, max_step=10.0)
                if abs(pitch_delta) <= 0.5:
                    break
                a = self.env.noop_action()
                a["camera"] = np.array([pitch_delta, 0])
                self.raw_step(a)
                steps_used += 1

        floor_stabilize_result = None
        if (
            stabilize_floor_enabled
            and stabilize_after_full_sweep
            and stabilize_floor_requested
        ):
            floor_stabilize_result = _stabilize_corridor_floor()
            setattr(self, "_corridor_floor_stabilize_requested", False)

        per_block_initial = self._mined_block_count()
        reason = "ok"
        height_drop = False
        forward_move_failures = 0
        stuck_clears = 0
        walk_deltas: List[float] = []
        alignment_attempts: List[Dict[str, Any]] = []
        segment_records: List[Dict[str, Any]] = []
        success_direction = ""
        yaw_candidates: List[Tuple[str, float | None, float]] = []
        if axis_lock:
            for offset in yaw_offsets:
                yaw_candidates.append(
                    (
                        f"yaw_offset_{offset:+.0f}",
                        aligned_yaw + offset,
                        offset,
                    )
                )
        else:
            yaw_candidates.append(("unlocked", None, 0.0))

        while blocks_dug < target_blocks and steps_used < int(max_steps):
            if terminal_abort or _height_dropped():
                reason = "terminal_or_position_jump" if terminal_abort else "height_drop"
                height_drop = True
                break
            if steps_used >= int(max_steps):
                break
            segment_start_x, segment_start_z = _current_xz(start_x, start_z)
            segment_start_cell = _block_cell(segment_start_x, segment_start_z)
            segment_index = blocks_dug + 1
            relocated = False
            segment_record: Dict[str, Any] = {
                "segment": segment_index,
                "start_x": segment_start_x,
                "start_z": segment_start_z,
                "start_cell": segment_start_cell,
                "attempts": [],
            }

            def _segment_delta() -> float:
                now_x, now_z = _current_xz(segment_start_x, segment_start_z)
                return _horizontal_delta(segment_start_x, segment_start_z, now_x, now_z)

            for direction, target_yaw, yaw_offset in yaw_candidates:
                moved = 0.0
                mined_delta = 0
                if target_yaw is not None:
                    _face_axis(head_pitch_target, target_yaw)
                moved, mined_delta = _phase_move_attack(
                    head_pitch_target,
                    max(move_attack_ticks, 1),
                    f"move_probe_{direction}",
                    target_yaw=target_yaw,
                    move_threshold=segment_min_move_delta,
                )
                walk_deltas.append(moved)
                segment_delta = _segment_delta()
                now_x, now_z = _current_xz(segment_start_x, segment_start_z)
                total_delta = _total_horizontal_delta()
                attempt: Dict[str, Any] = {
                    "segment": segment_index,
                    "direction": direction,
                    "target_yaw": target_yaw,
                    "yaw_offset": yaw_offset,
                    "yaw_mode": yaw_mode,
                    "probe_delta": moved,
                    "segment_delta": segment_delta,
                    "segment_cell_changed": _block_cell_changed(
                        segment_start_x, segment_start_z, now_x, now_z
                    ),
                    "total_delta": total_delta,
                    "mined_delta": mined_delta,
                    "unstuck_passes": [],
                }
                if terminal_abort or _height_dropped():
                    alignment_attempts.append(attempt)
                    segment_record["attempts"].append(attempt)
                    reason = "terminal_or_position_jump" if terminal_abort else "height_drop"
                    height_drop = True
                    break
                if segment_delta >= segment_min_move_delta:
                    alignment_attempts.append(attempt)
                    segment_record["attempts"].append(attempt)
                    success_direction = direction
                    relocated = True
                    break
                unstuck_pass = 0
                while (
                    _segment_delta() < segment_min_move_delta
                    and unstuck_pass < max_unstuck_passes
                    and steps_used < int(max_steps)
                    and not terminal_abort
                    and not _height_dropped()
                ):
                    forward_move_failures += 1
                    stuck_clears += 1
                    if self.logger:
                        self.logger.info(
                            "[dig_forward_blocks] segment=%d %s has not advanced "
                            "(phase_delta=%.3f segment_delta=%.3f total_delta=%.3f mined_delta=%d); "
                            "clearing horizontal front then diagonal-down blockers "
                            "pass=%d/%d yaw=%s yaw_offset=%.1f front_pitch=%.1f "
                            "diagonal_pitch=%.1f",
                            segment_index,
                            direction,
                            moved,
                            _segment_delta(),
                            _total_horizontal_delta(),
                            mined_delta,
                            unstuck_pass + 1,
                            max_unstuck_passes,
                            f"{target_yaw:.1f}" if target_yaw is not None else "unlocked",
                            yaw_offset,
                            blocked_front_pitch_target,
                            blocked_feet_pitch_target,
                        )
                    cleared, clear_moved = _clear_forward_blocker(
                        target_yaw,
                        segment_start_x,
                        segment_start_z,
                    )
                    segment_delta = _segment_delta()
                    total_delta = _total_horizontal_delta()
                    now_x, now_z = _current_xz(segment_start_x, segment_start_z)
                    pass_record = {
                        "pass": unstuck_pass + 1,
                        "cleared": cleared,
                        "clear_delta": clear_moved,
                        "segment_delta_after_clear": segment_delta,
                        "segment_cell_changed_after_clear": _block_cell_changed(
                            segment_start_x, segment_start_z, now_x, now_z
                        ),
                        "total_delta_after_clear": total_delta,
                    }
                    if terminal_abort or _height_dropped() or steps_used >= int(max_steps):
                        attempt["unstuck_passes"].append(pass_record)
                        break
                    if segment_delta >= segment_min_move_delta:
                        moved = max(moved, clear_moved, segment_delta)
                        walk_deltas.append(clear_moved)
                        pass_record["retry_delta"] = segment_delta
                        attempt["unstuck_passes"].append(pass_record)
                        relocated = True
                        break
                    moved, mined_delta = _phase_move_attack(
                        head_pitch_target,
                        max(move_attack_ticks, 1),
                        f"move_retry_{direction}",
                        target_yaw=target_yaw,
                        move_threshold=segment_min_move_delta,
                    )
                    walk_deltas.append(moved)
                    segment_delta = _segment_delta()
                    total_delta = _total_horizontal_delta()
                    now_x, now_z = _current_xz(segment_start_x, segment_start_z)
                    pass_record["retry_delta"] = segment_delta
                    pass_record["retry_mined_delta"] = mined_delta
                    pass_record["segment_cell_changed_after_retry"] = _block_cell_changed(
                        segment_start_x, segment_start_z, now_x, now_z
                    )
                    pass_record["segment_delta_after_retry"] = segment_delta
                    pass_record["total_delta_after_retry"] = total_delta
                    attempt["unstuck_passes"].append(pass_record)
                    if segment_delta >= segment_min_move_delta:
                        relocated = True
                        break
                    if not cleared and segment_delta < segment_min_move_delta:
                        reason = "stuck_no_block_break"
                    unstuck_pass += 1
                attempt["final_delta"] = _segment_delta()
                attempt["final_total_delta"] = _total_horizontal_delta()
                alignment_attempts.append(attempt)
                segment_record["attempts"].append(attempt)
                if relocated:
                    success_direction = direction
                    break
            if terminal_abort or _height_dropped():
                reason = "terminal_or_position_jump" if terminal_abort else "height_drop"
                height_drop = True
                break
            if not relocated:
                reason = "stuck_no_forward_displacement"
                segment_record["success"] = False
                segment_record["reason"] = reason
                segment_record["end_x"], segment_record["end_z"] = _current_xz(
                    segment_start_x,
                    segment_start_z,
                )
                segment_record["segment_delta"] = _segment_delta()
                segment_records.append(segment_record)
                break
            blocks_dug += 1
            reason = "tunnel_segment_advanced"
            end_seg_x, end_seg_z = _current_xz(segment_start_x, segment_start_z)
            segment_record["success"] = True
            segment_record["direction"] = success_direction
            segment_record["end_x"] = end_seg_x
            segment_record["end_z"] = end_seg_z
            segment_record["end_cell"] = _block_cell(end_seg_x, end_seg_z)
            segment_record["segment_delta"] = _horizontal_delta(
                segment_start_x,
                segment_start_z,
                end_seg_x,
                end_seg_z,
            )
            segment_records.append(segment_record)
            if self.logger:
                self.logger.info(
                    "[dig_forward_blocks] segment advanced: %d/%d "
                    "segment_delta=%.2f total_delta=%.2f direction=%s "
                    "cell=%s->%s",
                    blocks_dug,
                    target_blocks,
                    segment_record["segment_delta"],
                    _total_horizontal_delta(),
                    success_direction,
                    segment_start_cell,
                    segment_record["end_cell"],
                )
            if stop_after_first_move:
                break

        provisional_success = blocks_dug >= min_success_blocks
        if not provisional_success:
            _restore_pitch_to(head_pitch_target)

        end_loc = (self.cache.get("info") or {}).get("location_stats", {}) or {}

        def _end_scalar(key: str) -> float:
            try:
                return float(np.asarray(end_loc.get(key, 0.0)).reshape(-1)[0])
            except Exception:
                return 0.0

        end_x, end_y, end_z = _end_scalar("xpos"), _end_scalar("ypos"), _end_scalar("zpos")
        horizontal_delta = _horizontal_delta(start_x, start_z, end_x, end_z)
        block_cell_changed = _block_cell_changed(start_x, start_z, end_x, end_z)
        meaningful_final_move = horizontal_delta >= min_move_delta
        success = provisional_success and not height_drop and not terminal_abort
        if success and not meaningful_final_move:
            success = False
            reason = "no_block_cell_change"
        if success:
            if blocks_dug >= target_blocks:
                reason = "tunnel_bore_complete"
            else:
                reason = "tunnel_bore_min_success"
        elif steps_used >= int(max_steps):
            reason = "step_budget_exhausted"

        next_sweep_index = sweep_index
        if direction_sweep_enabled and axis_lock and len(direction_sweep_offsets) > 1:
            should_rotate_direction = (
                not success
                and not terminal_abort
                and reason
                in {
                    "height_drop",
                    "no_block_cell_change",
                    "stuck_no_forward_displacement",
                    "stuck_no_block_break",
                    "step_budget_exhausted",
                }
            )
            if should_rotate_direction:
                next_sweep_index = (sweep_index + 1) % len(direction_sweep_offsets)
                setattr(self, "_corridor_direction_sweep_index", next_sweep_index)
                setattr(self, "_corridor_direction_sweep_anchor_yaw", sweep_anchor_yaw)
                if (
                    stabilize_floor_enabled
                    and stabilize_after_full_sweep
                    and next_sweep_index == 0
                ):
                    setattr(self, "_corridor_floor_stabilize_requested", True)
                if self.logger:
                    self.logger.info(
                        "[dig_forward_blocks] direction sweep rotate: "
                        "reason=%s index=%d->%d next_offset=%.1f anchor_yaw=%.1f",
                        reason,
                        sweep_index,
                        next_sweep_index,
                        float(direction_sweep_offsets[next_sweep_index]),
                        sweep_anchor_yaw,
                    )
            else:
                next_sweep_index = 0
                setattr(self, "_corridor_direction_sweep_index", 0)
                setattr(self, "_corridor_direction_sweep_anchor_yaw", None)
                setattr(self, "_corridor_floor_stabilize_requested", False)

        result = {
            "success": success,
            "blocks_dug": blocks_dug,
            "target_blocks": target_blocks,
            "min_success_blocks": min_success_blocks,
            "steps_used": steps_used,
            "reason": reason,
            "start_x": start_x,
            "start_y": start_y,
            "start_z": start_z,
            "end_x": end_x,
            "end_y": end_y,
            "end_z": end_z,
            "mined_total_delta": self._mined_block_count() - per_block_initial,
            "height_drop": height_drop,
            "terminal_abort": terminal_abort,
            "position_jump_abort": position_jump_abort,
            "max_position_jump": max_position_jump,
            "forward_move_failures": forward_move_failures,
            "stuck_clears": stuck_clears,
            "walk_deltas": walk_deltas,
            "alignment_attempts": alignment_attempts,
            "segment_records": segment_records,
            "success_direction": success_direction,
            "axis_lock": axis_lock,
            "yaw_mode": yaw_mode,
            "aligned_yaw": aligned_yaw if axis_lock else None,
            "yaw_offsets": yaw_offsets,
            "direction_sweep_enabled": direction_sweep_enabled,
            "direction_sweep_index": sweep_index,
            "direction_sweep_next_index": next_sweep_index,
            "direction_sweep_anchor_yaw": sweep_anchor_yaw if axis_lock else None,
            "direction_sweep_base_offset": sweep_base_offset,
            "direction_sweep_offsets": direction_sweep_offsets,
            "floor_stabilize_requested": stabilize_floor_requested,
            "floor_stabilize_result": floor_stabilize_result,
            "min_move_delta": min_move_delta,
            "segment_min_move_delta": segment_min_move_delta,
            "horizontal_delta": horizontal_delta,
            "start_block_cell": _block_cell(start_x, start_z),
            "end_block_cell": _block_cell(end_x, end_z),
            "block_cell_changed": block_cell_changed,
            "blocked_front_pitch": blocked_front_pitch_target,
            "blocked_up_pitch": up_pitch_target,
            "blocked_up_budget": blocked_up_budget,
            "blocked_feet_pitch_requested": raw_blocked_feet_pitch_target,
            "blocked_feet_pitch_cap": blocked_feet_pitch_cap,
            "blocked_feet_pitch": blocked_feet_pitch_target,
            "stop_after_first_move": stop_after_first_move,
        }
        self._record_recovery("dig_forward_blocks", result)
        if self.logger:
            self.logger.info(
                f"[dig_forward_blocks] done: blocks_dug={blocks_dug}/{target_blocks} "
                f"min_success={min_success_blocks} "
                f"steps_used={steps_used} reason={reason} "
                f"horizontal_delta={horizontal_delta:.2f} "
                f"end=({end_x:.1f},{end_y:.1f},{end_z:.1f}) "
                f"success_direction={success_direction} "
                f"forward_move_failures={forward_move_failures} "
                f"stuck_clears={stuck_clears} "
                f"direction_sweep_index={sweep_index} "
                f"direction_sweep_base_offset={sweep_base_offset:.1f}"
            )
        return result

    def dig_down_blocks(
        self,
        n_blocks: int = 5,
        max_steps: int = 320,
        max_steps_per_block: int = 70,
        prefer_pickaxe: bool = True,
        generate_ore: bool = False,
        force_ore: str | None = None,
    ) -> Dict[str, Any]:
        """Scripted vertical shaft at the current x/z.

        Used after a successful horizontal tunnel relocation. The normal
        STEVE-1 dig-down prompt can drift away from the newly chosen x/z; this
        primitive keeps the agent roughly in place and mines straight down a
        small bounded distance so ore spawned below this new column is actually
        collected.
        """
        loc = (self.cache.get("info") or {}).get("location_stats", {}) or {}

        def _scalar(key: str, default: float) -> float:
            try:
                return float(np.asarray(loc.get(key, default)).reshape(-1)[0])
            except Exception:
                return float(default)

        def _current_xyz(default_x: float, default_y: float, default_z: float) -> Tuple[float, float, float]:
            loc_now = (self.cache.get("info") or {}).get("location_stats", {}) or {}
            try:
                x_now = float(np.asarray(loc_now.get("xpos", default_x)).reshape(-1)[0])
            except Exception:
                x_now = float(default_x)
            try:
                y_now = float(np.asarray(loc_now.get("ypos", default_y)).reshape(-1)[0])
            except Exception:
                y_now = float(default_y)
            try:
                z_now = float(np.asarray(loc_now.get("zpos", default_z)).reshape(-1)[0])
            except Exception:
                z_now = float(default_z)
            return x_now, y_now, z_now

        def _nudge_pitch_to(target_pitch: float, max_step: float = 12.0) -> float:
            try:
                cp = float(
                    np.asarray(
                        (self.cache.get("info") or {}).get("location_stats", {}).get(
                            "pitch", 0
                        )
                    ).reshape(-1)[0]
                )
            except Exception:
                cp = 0.0
            return max(min(target_pitch - cp, max_step), -max_step)

        start_x = _scalar("xpos", 0.0)
        start_y = _scalar("ypos", 0.0)
        start_z = _scalar("zpos", 0.0)
        min_y = float(os.environ.get("XENON_SCRIPTED_DIGDOWN_MIN_Y", "4.0"))
        ore_threshold = float(os.environ.get("XENON_SCRIPTED_DIGDOWN_ORE_THRESHOLD", "0.0"))
        target_pitch = float(os.environ.get("XENON_SCRIPTED_DIGDOWN_PITCH", "88.0"))
        settle_ticks = int(os.environ.get("XENON_SCRIPTED_DIGDOWN_SETTLE_TICKS", "5"))

        steps_used = 0
        blocks_dug = 0
        terminal_abort = False
        position_jump_abort = False
        max_position_jump = float(os.environ.get("XENON_CORRIDOR_MAX_POSITION_JUMP", "6.0"))
        forced_ore: Dict[str, Any] | None = None

        if generate_ore and force_ore and os.environ.get("XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE", "0") == "1":
            ore_name = str(force_ore).replace("minecraft:", "")
            if ore_name == "redstone":
                ore_name = "redstone_ore"
            if ore_name == "diamond":
                ore_name = "diamond_ore"
            if not ore_name.endswith("_ore"):
                ore_name = f"{ore_name}_ore"
            band = self.ORE_HEIGHT_BANDS.get(ore_name)
            dy_candidates: List[int] = []
            for raw_dy in os.environ.get(
                "XENON_SCRIPTED_DIGDOWN_FORCE_ORE_DYS",
                "-3,-4,-5,-2,-1",
            ).split(","):
                try:
                    dy = int(raw_dy.strip())
                except ValueError:
                    continue
                if dy < 0 and dy not in dy_candidates:
                    dy_candidates.append(dy)
            if not dy_candidates:
                dy_candidates = [-3, -4, -5, -2, -1]
            forced_dy = os.environ.get("XENON_SCRIPTED_DIGDOWN_FORCE_ORE_DY")
            chosen_dy = int(forced_dy) if forced_dy is not None else dy_candidates[0]
            if band is not None:
                lo, hi = band
                for dy in dy_candidates:
                    target_y = start_y + dy
                    if float(lo) <= target_y <= float(hi):
                        chosen_dy = dy
                        break
            try:
                self.env.execute_cmd(f"/setblock ~ ~{chosen_dy} ~ minecraft:{ore_name}")
                forced_ore = {
                    "ore": ore_name,
                    "dy": chosen_dy,
                    "target_y": start_y + chosen_dy,
                    "dy_candidates": dy_candidates,
                }
                if self.logger:
                    self.logger.info(
                        "[dig_down_blocks] forced target ore: ore=%s dy=%d target_y=%.1f",
                        ore_name,
                        chosen_dy,
                        start_y + chosen_dy,
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning("[dig_down_blocks] force target ore failed: %s", exc)
        elif generate_ore:
            try:
                ore_map = {} if os.environ.get("XENON_SCRIPTED_DIGDOWN_ORE_LOCAL_MAP", "1") == "1" else self.ORE_MAP
                random_ore(
                    self.env,
                    ore_map,
                    start_y,
                    thresold=ore_threshold,
                    xpos=start_x,
                    zpos=start_z,
                    prob_scale=float(self.cache.get("ore_spawn_scale", 1.0)),
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning("[dig_down_blocks] random_ore failed: %s", exc)

        if prefer_pickaxe:
            try:
                slot = self.find_best_pickaxe()
            except Exception:
                slot = None
            if slot:
                previous_can_change_hotbar = self.can_change_hotbar
                self.can_change_hotbar = True
                try:
                    for _ in range(4):
                        if steps_used >= int(max_steps):
                            break
                        a_eq = self.env.noop_action()
                        a_eq[slot] = np.array(1)
                        self.raw_step(a_eq)
                        steps_used += 1
                finally:
                    self.can_change_hotbar = previous_can_change_hotbar

        mined_before = self._mined_block_count()
        reason = "ok"
        trajectory: List[float] = [start_y]

        for _block in range(max(0, int(n_blocks))):
            if steps_used >= int(max_steps):
                reason = "step_budget_exhausted"
                break
            _, segment_start_y, _ = _current_xyz(start_x, start_y, start_z)
            if segment_start_y <= min_y:
                reason = "min_y_reached"
                break
            moved_down = False
            for _ in range(max(1, int(max_steps_per_block))):
                if steps_used >= int(max_steps):
                    reason = "step_budget_exhausted"
                    break
                a = self.env.noop_action()
                a["attack"] = np.array(1)
                pitch_delta = _nudge_pitch_to(target_pitch)
                if abs(pitch_delta) > 0.1:
                    a["camera"] = np.array([pitch_delta, 0])
                prev_x, prev_y, prev_z = _current_xyz(start_x, start_y, start_z)
                _, _, done, _ = self.raw_step(a)
                steps_used += 1
                cur_x, cur_y, cur_z = _current_xyz(prev_x, prev_y, prev_z)
                jump = (
                    (cur_x - prev_x) ** 2
                    + (cur_y - prev_y) ** 2
                    + (cur_z - prev_z) ** 2
                ) ** 0.5
                if done or jump > max_position_jump:
                    terminal_abort = True
                    position_jump_abort = bool(jump > max_position_jump)
                    reason = "terminal_or_position_jump"
                    break
                if cur_y <= segment_start_y - 0.75:
                    moved_down = True
                    blocks_dug += 1
                    trajectory.append(cur_y)
                    for _settle in range(max(0, settle_ticks)):
                        if steps_used >= int(max_steps):
                            break
                        self.raw_step(self.env.noop_action())
                        steps_used += 1
                    break
            if terminal_abort or steps_used >= int(max_steps):
                break
            if not moved_down:
                reason = "stuck_no_vertical_displacement"
                break

        end_x, end_y, end_z = _current_xyz(start_x, start_y, start_z)
        if blocks_dug >= max(1, min(int(n_blocks), 1)) and reason == "ok":
            reason = "scripted_digdown_complete" if blocks_dug >= int(n_blocks) else "scripted_digdown_partial"
        success = blocks_dug > 0 and not terminal_abort
        result = {
            "success": success,
            "blocks_dug": blocks_dug,
            "target_blocks": int(n_blocks),
            "steps_used": steps_used,
            "reason": reason,
            "start_x": start_x,
            "start_y": start_y,
            "start_z": start_z,
            "end_x": end_x,
            "end_y": end_y,
            "end_z": end_z,
            "mined_total_delta": self._mined_block_count() - mined_before,
            "terminal_abort": terminal_abort,
            "position_jump_abort": position_jump_abort,
            "trajectory": trajectory,
            "generated_ore": bool(generate_ore),
            "forced_ore": forced_ore,
            "ore_threshold": ore_threshold,
            "min_y": min_y,
        }
        self._record_recovery("dig_down_blocks", result)
        if self.logger:
            self.logger.info(
                "[dig_down_blocks] done: blocks_dug=%d/%d steps=%d reason=%s "
                "start=(%.1f,%.1f,%.1f) end=(%.1f,%.1f,%.1f) mined_delta=%d",
                blocks_dug,
                int(n_blocks),
                steps_used,
                reason,
                start_x,
                start_y,
                start_z,
                end_x,
                end_y,
                end_z,
                result["mined_total_delta"],
            )
        return result

    def _record_recovery(self, name: str, payload: Dict[str, Any]) -> None:
        events = self._control_state.setdefault("recovery_events", {})
        log = events.setdefault(name, [])
        log.append(copy.deepcopy(payload))

    def get_status(self):
        status = self.status_mod.get_status()
        status["resource_ledger"] = copy.deepcopy(self.cache.get("resource_ledger", {}))
        status["inventory_slots_used"] = self._used_inventory_slots()
        status["recovery_events"] = copy.deepcopy(
            self._control_state.get("recovery_events", {})
        )
        status["control_state"] = {
            "surface_attack_streak": int(self.cache.get("surface_attack_streak", 0)),
            "last_surface_attack_step": int(self.cache.get("last_surface_attack_step", -1000000)),
            "last_surface_log_step": int(self.cache.get("last_surface_log_step", -1000000)),
            "surface_log_jump_lock_ticks": int(self._control_state.get("surface_log_jump_lock_ticks", 0)),
            "last_target_block_step": int(self.cache.get("last_target_block_step", 0)),
            "last_goal_progress_step": int(self.cache.get("last_goal_progress_step", 0)),
        }
        return status
