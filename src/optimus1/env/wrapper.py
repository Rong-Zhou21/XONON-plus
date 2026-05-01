import copy
import logging
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

from ..util.server_api import MultiThreadServerAPI

from .mods import RecorderMod, StatusMod, TaskCheckerMod


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


def random_ore(env, ORE_MAP, ypos: float, thresold: float = 0.9):
    prob = random.random()
    if prob <= thresold:
        return
    dy = random.randint(-5, -3)
    new_pos = int(ypos + dy)
    if 45 <= ypos <= 50:  # max: 6
        # coal_ore
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 45:
            ORE_MAP[new_pos] = "coal_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:coal_ore".format(dy))
            print(f"coal ore at {new_pos}")
    elif 26 <= ypos <= 43:  # max: 17
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 26:
            ORE_MAP[new_pos] = "iron_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:iron_ore".format(dy))
            print(f"iron ore at {new_pos}")

    elif 14 < ypos <= 26:
        if ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 17:  # max: 10
            ORE_MAP[new_pos] = "gold_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:gold_ore".format(dy))
            print(f"gold ore at {new_pos}")
        elif ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos <= 16:  # max:12
            ORE_MAP[new_pos] = "redstone_ore"
            ORE_MAP[ypos] = 1
            env.execute_cmd("/setblock ~ ~{} ~ minecraft:redstone_ore".format(dy))
            print(f"redstone ore at {new_pos}")
    elif (
        ypos <= 14 and ypos not in ORE_MAP and new_pos not in ORE_MAP and new_pos >= 1
    ):  # max: 14
        ORE_MAP[new_pos] = "diamond_ore"
        ORE_MAP[ypos] = 1
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
        self.cache["last_surface_search_step"] = -1000000
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
            "collect_drop_ticks": 0,
            "resource_stagnant_ticks": 0,
            "movement_stagnant_ticks": 0,
            "last_prompt": None,
            "last_health": None,
            "last_position": None,
            "policy_reset_requested": False,
            "recovery_events": {
                "surface_escape": 0,
                "surface_search": 0,
                "collect_drops": 0,
                "movement_escape": 0,
                "tunnel_recovery": 0,
                "respawn_reset": 0,
                "checker_error": 0,
                "inventory_cleanup": 0,
                "inventory_cleanup_blocked": 0,
            },
        }

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
        if not self._is_surface_resource_acquisition(goal, prompt):
            return False

        contact_ticks = int(
            os.environ.get(
                "XENON_SURFACE_ATTACK_LOCK_TICKS",
                os.environ.get("XENON_TREE_CONTACT_ATTACK_TICKS", "16"),
            )
        )
        return (
            int(self.cache.get("surface_attack_streak", 0)) >= contact_ticks
            and self.num_steps - int(self.cache.get("last_surface_attack_step", -1000000)) <= 2
        )

    def _lock_full_movement_during_attack(self, goal: tuple[str, int] | None, prompt: str | None) -> bool:
        return self._is_tunnel_resource_acquisition(goal, prompt)

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

    def _ledger_satisfies_goal(self, goal: tuple[str, int] | None) -> bool:
        if goal is None:
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

    def _should_tunnel_recovery(self, action: Dict[str, Any], goal: tuple[str, int] | None, prompt: str | None) -> bool:
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
            self._control_state["collect_drop_ticks"] = 0
            self._control_state["last_prompt"] = prompt
            self.cache["surface_attack_streak"] = 0
            self.cache["last_progress_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps

        if self._control_state["escape_ticks"] <= 0 and self._should_surface_escape():
            self._control_state["escape_ticks"] = int(os.environ.get("XENON_ESCAPE_TICKS", "80"))
            self._control_state["recovery_events"]["surface_escape"] += 1
            self.logger.info(
                "Activating surface recovery primitive: "
                f"air={self._current_air()}, health={self._current_health()}"
            )
        elif self._control_state["escape_ticks"] <= 0 and self._should_movement_escape(action, goal, prompt):
            self._control_state["escape_ticks"] = int(os.environ.get("XENON_ESCAPE_TICKS", "80"))
            self._control_state["recovery_events"]["movement_escape"] += 1
            self.logger.info(
                "Activating movement recovery primitive: "
                f"stagnant_ticks={self._control_state['movement_stagnant_ticks']}, "
                f"stale_progress={self._stale_progress_ticks()}"
            )
        elif (
            self._control_state["escape_ticks"] <= 0
            and self._control_state["collect_drop_ticks"] <= 0
            and self._control_state["surface_search_ticks"] <= 0
            and self._should_collect_drops(action, goal, prompt)
        ):
            self._control_state["collect_drop_ticks"] = int(os.environ.get("XENON_COLLECT_DROPS_TICKS", "24"))
            self._control_state["attack_hold"] = 0
            self.cache["last_collect_drop_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps
            self._control_state["recovery_events"]["collect_drops"] += 1
            self.logger.info(
                "Activating resource drop collection primitive: "
                f"pending={self._pending_relevant_drops(goal)}, "
                f"stale_goal_progress={self._stale_goal_progress_ticks()}, prompt={prompt}"
            )
        elif (
            self._control_state["escape_ticks"] <= 0
            and self._control_state["collect_drop_ticks"] <= 0
            and self._control_state["surface_search_ticks"] <= 0
            and self._should_surface_search(goal, prompt)
        ):
            self._control_state["surface_search_ticks"] = int(os.environ.get("XENON_SURFACE_SEARCH_TICKS", "90"))
            self._control_state["attack_hold"] = 0
            self.cache["last_surface_search_step"] = self.num_steps
            self.cache["last_goal_progress_step"] = self.num_steps
            self._control_state["recovery_events"]["surface_search"] += 1
            self.logger.info(
                "Activating surface resource search primitive: "
                f"stale_goal_progress={self._stale_goal_progress_ticks()}, prompt={prompt}, goal={goal}"
            )
        elif (
            self._control_state["escape_ticks"] <= 0
            and self._control_state["tunnel_recovery_ticks"] <= 0
            and self._should_tunnel_recovery(action, goal, prompt)
        ):
            self._control_state["tunnel_recovery_ticks"] = int(os.environ.get("XENON_TUNNEL_RECOVERY_TICKS", "70"))
            self._control_state["recovery_events"]["tunnel_recovery"] += 1
            self.logger.info(
                "Activating tunnel clearance primitive: "
                f"resource_stagnant_ticks={self._control_state['resource_stagnant_ticks']}, "
                f"stale_progress={self._stale_progress_ticks()}, prompt={prompt}"
            )

        if self._control_state["escape_ticks"] > 0:
            action = self._escape_action(action)
            self._maybe_log_action_debug("stabilize_escape", original_action, action, goal, prompt)
            return action
        if self._control_state["collect_drop_ticks"] > 0:
            action = self._collect_drop_action(action, goal, prompt)
            self._maybe_log_action_debug("stabilize_collect_drops", original_action, action, goal, prompt)
            return action
        if self._control_state["surface_search_ticks"] > 0:
            action = self._surface_search_action(action)
            self._maybe_log_action_debug("stabilize_surface_search", original_action, action, goal, prompt)
            return action
        if self._control_state["tunnel_recovery_ticks"] > 0:
            action = self._tunnel_recovery_action(action)
            self._maybe_log_action_debug("stabilize_tunnel", original_action, action, goal, prompt)
            return action

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

        self._maybe_log_action_debug("stabilize", original_action, action, goal, prompt)
        return action

    def _record_step_state(
        self,
        observation: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ) -> None:
        previous_health = self._control_state.get("last_health")
        previous_position = self._control_state.get("last_position")
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
        if (
            previous_health is not None
            and pos is not None
            and previous_position is not None
            and health >= 19.0
            and float(previous_health) <= 2.0
            and self._position_jump(previous_position, pos) > float(os.environ.get("XENON_RESPAWN_POSITION_JUMP", "6.0"))
        ):
            self._control_state["attack_hold"] = 0
            self._control_state["escape_ticks"] = 0
            self._control_state["tunnel_recovery_ticks"] = 0
            self._control_state["surface_search_ticks"] = 0
            self._control_state["collect_drop_ticks"] = 0
            self._control_state["movement_stagnant_ticks"] = 0
            self._control_state["resource_stagnant_ticks"] = 0
            self._control_state["policy_reset_requested"] = True
            self._control_state["recovery_events"]["respawn_reset"] += 1
            self.cache["position_window"].clear()
            self.logger.info(
                "Detected death/respawn transition; clearing low-level control state "
                "and requesting STEVE-1 policy reset."
            )
        deltas = self._record_resource_ledger(observation)
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
            return observation, reward, done, info
        # 推送 POV 到宿主机 monitor_server（异步、O(1)、失败静默）
        _push_pov_to_monitor(observation)
        self.record_mod.step(observation, None, action)
        self.status_mod.step(observation, action, self.num_steps)
        self._record_step_state(observation)

        info.update(self.status_mod.get_status())

        info["isGuiOpen"] = observation["isGuiOpen"]

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
            random_ore(self.env, self.ORE_MAP, ypos)
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
            "last_target_block_step": int(self.cache.get("last_target_block_step", 0)),
            "last_goal_progress_step": int(self.cache.get("last_goal_progress_step", 0)),
        }
        return status
