#!/usr/bin/env python
"""Connectivity + contract test for the Advanced-VLM comparison arm.

Usage (after exporting XENON_VLM_BASE_URL / XENON_VLM_API_KEY / XENON_VLM_MODEL
and optionally XENON_VLM_PROXY):

    /home/yzb/.conda/envs/vllm_qwen2_5_vl/bin/python scripts/test_advanced_vlm.py

Sends a synthetic Minecraft-like frame and checks that decide() and reflect()
return well-formed decisions. No MineRL / GPU needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("XENON_ADVANCED_VLM", "1")

import numpy as np  # noqa: E402

from optimus1.models.advanced_vlm import AdvancedVLMClient  # noqa: E402


def main() -> int:
    # synthetic 360x640 frame: green ground + blue sky + brown "tree trunk"
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:180, :, :] = (120, 170, 255)   # sky
    img[180:, :, :] = (60, 140, 60)     # grass
    img[120:300, 300:340, :] = (90, 60, 30)  # trunk

    client = AdvancedVLMClient.get()
    print(f"endpoint : {client.base_url}")
    print(f"model    : {client.model}")
    print(f"proxy    : {client.proxies or 'none'}")

    print("\n-- decide() --")
    subgoal, action = client.decide(
        img, waypoint="logs", waypoint_num=3,
        inventory={}, final_goal="craft a wooden pickaxe",
    )
    print(f"subgoal  : {subgoal}")
    print(f"action   : {action}")
    assert isinstance(subgoal, dict) and "task" in subgoal and "goal" in subgoal
    assert subgoal["goal"][1] == 3

    print("\n-- reflect() --")
    reasoning, visual = client.reflect(img, "chop a tree", "logs")
    print(f"reasoning: {reasoning}")
    print(f"visual   : {visual}")
    assert "need_intervention" in reasoning and "task" in reasoning

    print("\nALL CHECKS PASSED — the arm is ready to launch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
