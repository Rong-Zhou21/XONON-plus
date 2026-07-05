"""Advanced-VLM comparison arm (对比模块): direct decision control by a
stronger vision-language model via an OpenAI-compatible API.

When ``XENON_ADVANCED_VLM=1``:

* Decision point D in ``make_plan`` is taken over by :meth:`AdvancedVLMClient.decide`
  — the case library, RADS decisioner and the local Qwen planner are all
  bypassed. The Oracle dependency graph still supplies the waypoint sequence;
  the VLM chooses *how* to achieve the current waypoint from the live
  first-person frame + inventory.
* The in-execution reflection point is taken over by
  :meth:`AdvancedVLMClient.reflect` — the VLM looks at the current frame and
  may intervene in real time (switch the STEVE-1 prompt), matching the
  ``reasoning_dict['need_intervention'] / ['task']`` contract of the existing
  ``render_context_aware_reasoning`` path.
* Case-memory writes are suppressed (see guards in ``case_memory.py``) so this
  arm neither pollutes nor exploits the case library.

Environment variables:
    XENON_ADVANCED_VLM        "1" to enable the arm (default off).
    XENON_VLM_BASE_URL        OpenAI-compatible endpoint, e.g. https://x.cn/v1
    XENON_VLM_API_KEY         API key for the relay/proxy station.
    XENON_VLM_MODEL           Model name, e.g. gpt-4o / claude-sonnet-4-5.
    XENON_VLM_PROXY           Optional http(s) proxy ONLY for these API calls
                              (the run scripts unset global proxies for local
                              MineRL/STEVE-1 traffic; an external API may still
                              need one, e.g. http://127.0.0.1:7897).
    XENON_VLM_TIMEOUT_SEC     Per-request timeout (default 60).
    XENON_VLM_MAX_RETRIES     Retries per call (default 3).
    XENON_VLM_TEMPERATURE     Sampling temperature (default 0.2).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from optimus1.util.image import img2base64


def advanced_vlm_enabled() -> bool:
    return os.environ.get("XENON_ADVANCED_VLM", "0") == "1"


def _strip_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of one JSON object from a model reply."""
    if not text:
        return None
    if "```" in text:
        for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


class AdvancedVLMClient:
    """Minimal OpenAI-compatible chat client with image input."""

    _singleton: "AdvancedVLMClient | None" = None

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.base_url = os.environ.get("XENON_VLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("XENON_VLM_API_KEY", "")
        self.model = os.environ.get("XENON_VLM_MODEL", "gpt-4o")
        self.timeout = float(os.environ.get("XENON_VLM_TIMEOUT_SEC", "60"))
        self.max_retries = int(os.environ.get("XENON_VLM_MAX_RETRIES", "3"))
        self.temperature = float(os.environ.get("XENON_VLM_TEMPERATURE", "0.2"))
        proxy = os.environ.get("XENON_VLM_PROXY", "")
        self.proxies = {"http": proxy, "https": proxy} if proxy else None
        self.logger = logger
        # usage counters (for the efficiency metrics of this arm)
        self.n_decide_calls = 0
        self.n_reflect_calls = 0
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "XENON_ADVANCED_VLM=1 but XENON_VLM_BASE_URL / XENON_VLM_API_KEY "
                "are not set. Export them before launching."
            )

    @classmethod
    def get(cls, logger: logging.Logger | None = None) -> "AdvancedVLMClient":
        if cls._singleton is None:
            cls._singleton = cls(logger)
        return cls._singleton

    # ---------------------------------------------------------------- #
    #  transport                                                        #
    # ---------------------------------------------------------------- #

    def _chat(self, system: str, user_text: str, pov=None) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if pov is not None:
            b64 = img2base64(pov)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=self.proxies,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 — retry any transport error
                last_err = exc
                if self.logger:
                    self.logger.warning(
                        f"ADV_VLM chat attempt {attempt}/{self.max_retries} "
                        f"failed: {exc}"
                    )
        raise RuntimeError(f"advanced VLM call failed after retries: {last_err}")

    # ---------------------------------------------------------------- #
    #  decision point D                                                 #
    # ---------------------------------------------------------------- #

    _DECIDE_SYSTEM = (
        "You are an expert Minecraft agent controller. You see the agent's "
        "first-person view and its inventory. Decide the ONE next language "
        "action for the low-level controller (STEVE-1) to achieve the current "
        "waypoint item. Reply with STRICT JSON only:\n"
        '{"task": "<short action phrase>", "goal": ["<waypoint item>", <count>]}\n'
        "Allowed action patterns (choose the appropriate one):\n"
        '  "chop a tree"                      - for logs/wood on the surface\n'
        '  "dig down and mine <item>"         - for ores/stone underground\n'
        '  "craft <item>"                     - when materials are in inventory\n'
        '  "smelt <item>"                     - when a furnace recipe is needed\n'
        "No explanations, no markdown fences."
    )

    def decide(
        self,
        pov,
        waypoint: str,
        waypoint_num: int,
        inventory: Dict[str, Any],
        final_goal: str,
    ) -> Tuple[Dict[str, Any], str]:
        """Return (subgoal_dict, language_action_str) for waypoint D-decision."""
        self.n_decide_calls += 1
        inv_str = json.dumps(inventory, ensure_ascii=True) if inventory else "{}"
        user = (
            f"Final goal: {final_goal}\n"
            f"Current waypoint to obtain: {waypoint} x{waypoint_num}\n"
            f"Inventory: {inv_str}\n"
            "Look at the screenshot and output the JSON decision."
        )
        reply = self._chat(self._DECIDE_SYSTEM, user, pov)
        parsed = _strip_json(reply)
        if not isinstance(parsed, dict):
            raise ValueError(f"unparseable VLM decision: {reply[:200]}")
        # tolerate wrapped forms, same as render_subgoal
        if "goal" not in parsed:
            for key in ("task_planning", "task_plan", "plan", "subgoal"):
                inner = parsed.get(key)
                if isinstance(inner, dict) and "goal" in inner:
                    parsed = inner
                    break
        if "task" not in parsed or "goal" not in parsed:
            raise ValueError(f"VLM decision missing task/goal: {parsed}")
        parsed["goal"] = [str(parsed["goal"][0]), int(waypoint_num)]
        return parsed, str(parsed["task"])

    # ---------------------------------------------------------------- #
    #  in-execution reflection (real-time intervention)                 #
    # ---------------------------------------------------------------- #

    _REFLECT_SYSTEM = (
        "You are monitoring a Minecraft agent that has made no progress on its "
        "current sub-goal for a while. You see its current first-person view. "
        "Decide whether to intervene with a different short action prompt for "
        "the low-level controller. Reply with STRICT JSON only:\n"
        '{"need_intervention": true/false, "task": "<new short action phrase>", '
        '"visual": "<one-line scene description>"}\n'
        'If no intervention is needed, set "task" to the current prompt. '
        "No explanations, no markdown fences."
    )

    def reflect(
        self,
        pov,
        current_prompt: str,
        waypoint: str,
    ) -> Tuple[Dict[str, Any], str]:
        """Return (reasoning_dict, visual_description) matching the
        render_context_aware_reasoning contract."""
        self.n_reflect_calls += 1
        user = (
            f"Current sub-goal prompt: {current_prompt}\n"
            f"Target waypoint item: {waypoint}\n"
            "The agent is stuck. Look at the screenshot and output the JSON."
        )
        reply = self._chat(self._REFLECT_SYSTEM, user, pov)
        parsed = _strip_json(reply)
        if not isinstance(parsed, dict) or "need_intervention" not in parsed:
            raise ValueError(f"unparseable VLM reflection: {reply[:200]}")
        task = str(parsed.get("task") or current_prompt)
        reasoning = {
            "need_intervention": bool(parsed["need_intervention"]),
            "task": task,
        }
        return reasoning, str(parsed.get("visual", ""))


__all__ = ["AdvancedVLMClient", "advanced_vlm_enabled"]
