import copy
import logging
import os
import random
import threading
import time
from io import BytesIO
from typing import Any, Dict, List

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

    def raw_step(self, action: Dict[str, Any]):
        if not self.can_change_hotbar:
            for i in range(9):
                action[f"hotbar.{i+1}"] = np.array(0)
        # ban drop(Q) action
        action["drop"] = 0
        # attack时不乱动
        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
        observation, reward, done, info = self.env.step(action)
        # 推送 POV 到宿主机 monitor_server（异步、O(1)、失败静默）
        _push_pov_to_monitor(observation)
        self.record_mod.step(observation, None, action)
        self.status_mod.step(observation, action, self.num_steps)

        info.update(self.status_mod.get_status())

        info["isGuiOpen"] = observation["isGuiOpen"]

        return observation, reward, done, info

    def step(
        self,
        action: Dict[str, Any],
        goal: tuple[str, int] | None = None,
        prompt: str | None = None,
    ):
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

        observation, reward, done, info = self.env.step(action)

        if goal is not None and goal[0] != self.cache["task"]:
            self.task_checker_mod.reset(observation["inventory"])
            self.cache["task"] = goal[0]

        self.record_mod.step(observation, prompt, action)
        self.status_mod.step(observation, action, self.num_steps)

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
        except Exception as e:
            print("Error ", e)
            self._current_task_finish = True

        if (
            goal
            and "iron_ore" in goal[0]
            and ypos < 25
            and self._current_task_finish is False
        ):
            self.logger.critical("Return to ground....")
            _kill_ok = True
            try:
                self.env.execute_cmd("/kill")
            except Exception as _e:
                _kill_ok = False
                self.logger.warning(f"execute_cmd('/kill') failed: {_e}; skipping null_action loop")
            self.logger.warning("Kill agent because goal is iron_ore, but ypos < 25")
            self.cache["ypos"] = {}
            info.update({"killed": 1})
            if _kill_ok:
                for i in range(50):
                    try:
                        self.null_action()
                    except Exception as _e2:
                        self.logger.warning(f"null_action failed: {_e2}")
                        break
            # self.cache["explore"] = 100

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
    
    def check_original_goal_finish(self, goal: tuple[str, int] | None):
        action = self.env.noop_action()
        observation, reward, done, info = self.step(action)
        self.status_mod.step(observation, action, self.num_steps)

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

        as_it_is = self.task_checker_mod.step(current_inventory, [item_str, item_num], check_original_goal=True)
        with_underbar = self.task_checker_mod.step(current_inventory, [item_str.replace(" ", "_"), item_num], check_original_goal=True)
        without_ore = self.task_checker_mod.step(current_inventory, [item_str.replace("_ore", ""), 1], check_original_goal=True)

        return as_it_is or with_underbar or without_ore

    def check_waypoint_finish(self, waypoint: tuple[str, int] | None):
        action = self.env.noop_action()
        observation, reward, done, info = self.step(action)
        self.status_mod.step(observation, action, self.num_steps)

        item_str = copy.deepcopy(waypoint[0])

        current_env_status = self.get_status()
        current_inventory = current_env_status["inventory"]
        self.logger.info(f"In check_waypoint_finish")
        self.logger.info(f"waypoint: {waypoint}")
        self.logger.info(f"item_str: {item_str}")
        self.logger.info(f"current_inventory: {current_inventory}")

        as_it_is = self.task_checker_mod.step(current_inventory, [item_str, 1])
        with_underbar = self.task_checker_mod.step(current_inventory, [item_str.replace(" ", "_"), 1])
        without_ore = self.task_checker_mod.step(current_inventory, [item_str.replace("_ore", ""), 1])

        return as_it_is or with_underbar or without_ore

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

        if inventory_id_wooden:
            inventory_id = inventory_id_wooden
        if inventory_id_stone:
            inventory_id = inventory_id_stone
        if inventory_id_iron:
            inventory_id = inventory_id_iron
        if inventory_id_diamond:
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
            if it["type"] == item:
                return slot
        return None

    # def give_night_vision(self):
    #     self.env.execute_cmd("/effect give @a night_vision 99999 250 true")

    def get_status(self):
        return self.status_mod.get_status()