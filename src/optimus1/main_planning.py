import copy
import json
import logging
import os
import re
import shutil
import time
import traceback
from typing import Any, Dict
import sys

# Ensure files created in Docker are world-readable/writable
os.umask(0o000)

import hydra
import shortuuid
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress, TaskID, TimeElapsedColumn
# import wandb

import random
import numpy as np
import torch
import transformers

from optimus1.env import CustomEnvWrapper, PerceptionActionSuite, env_make, register_custom_env
from optimus1.helper import NewHelper
from optimus1.memories import CaseBasedMemory
from optimus1.memories import KnowledgeGraph as OracleGraph

from optimus1.monitor import Monitors, StepMonitor, SuccessMonitor
from optimus1.util import (
    ServerAPI,
    base64_to_img,
    get_evaluate_task,
    get_evaluate_task_and_goal,
    get_logger,
    pretty_result,
    render_subgoal,
    render_context_aware_reasoning
)


MINUTE = 1200
visual_info = ""


def _video_task_name(benchmark: str, task: str) -> str:
    benchmark_prefix = {
        "wooden": "Wood",
        "stone": "Stone",
        "iron": "Iron",
        "golden": "Gold",
        "diamond": "Diamond",
        "redstone": "Redstone",
        "armor": "Armor",
    }.get(benchmark, benchmark.title() if benchmark else "Unknown")
    return f"{benchmark_prefix}_{task}"


def _video_action_name(
    status: str,
    completed_subgoals: list[Dict[str, Any]],
    failed_subgoals: list[Any],
    task: str,
) -> str:
    if status != "success" and failed_subgoals:
        failed = failed_subgoals[0]
        if isinstance(failed, dict) and failed.get("task"):
            return failed["task"]
        if isinstance(failed, str):
            return failed
    if completed_subgoals:
        return completed_subgoals[-1].get("task") or task
    return task


def _malmo_log_has_fatal_error(env_malmo_logger_path: str, logger: logging.Logger) -> bool:
    """Treat MineRL log exceptions as diagnostic unless they are clearly fatal.

    The Minecraft client writes benign Java exceptions for services such as
    Realms, OpenAL, and narrator initialization. Aborting on every "Exception"
    prevents failed episodes from being recorded and saved as videos.
    """
    if not os.path.exists(env_malmo_logger_path):
        logger.warning(f"env_malmo_logger_path: {env_malmo_logger_path} does not exist.")
        return False

    try:
        with open(env_malmo_logger_path, "r", encoding="utf-8") as file:
            content = file.read()
    except OSError as exc:
        logger.warning(f"Cannot read env_malmo_logger_path {env_malmo_logger_path}: {exc}")
        return False

    fatal_patterns = [
        "OutOfMemoryError",
        "Could not reserve enough space",
        "The game crashed",
        "Minecraft has crashed",
    ]
    if any(pattern in content for pattern in fatal_patterns):
        return True

    if "Exception" in content:
        logger.warning(
            f"Ignoring non-fatal Minecraft log exception in {env_malmo_logger_path}; "
            "episode result/video will still be recorded."
        )
    else:
        logger.info(f"normal! env_malmo_logger_path: {env_malmo_logger_path} exists.")
    return False

def call_planner_with_retry(
    cfg: DictConfig,
    obs: Dict[str, Any],
    wp: str,
    wp_num: int,
    similar_wp_sg_dict: dict,
    failed_sg_list: list,
    hydra_path: str,
    run_uuid: str,
    logger: logging.Logger,
):
    attempts = 0
    max_retries = 3
    subgoal, sg_str = [], ""
    while attempts < max_retries:
        attempts += 1

        logger.info(f"Attempt: {attempts}, Just before get_decomposed_plan: ")
        logger.info(f"waypoint: {wp}")
        logger.info(f"similar_wp_sg_dict: {json.dumps(similar_wp_sg_dict)}")
        logger.info(f"failed_sg_list: {str(failed_sg_list)}")
        logger.info(f"Starting get_decomposed_plan ...\n")

        try:
            sg_str, prompt = ServerAPI.get_decomposed_plan(
                cfg["server"],
                obs,
                waypoint=wp,
                similar_wp_sg_dict=similar_wp_sg_dict,
                failed_sg_list_for_wp=failed_sg_list,
                hydra_path=hydra_path,
                run_uuid=run_uuid
            )

            logger.info(f'prompt before render_subgoal at attempt {attempts}')
            logger.info(f"{prompt}\n")
            logger.info(f'sg_str before render_subgoal at attempt {attempts}')
            logger.info(f"{sg_str}\n")

            tmp_subgoal, _, render_error = render_subgoal(copy.deepcopy(sg_str), wp_num)
            if render_error is None:
                break

            logger.warning(f"get_decomposed_plan at attempt {attempts} failed. Error message: {render_error}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_decomposed_plan.")
                return [], "", "max_tries_get_decomposed_plan"

        except Exception as e:
            logger.info(f"Error in get_decomposed_plan: {e}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_decomposed_plan.")
                return [], "", "max_tries_get_decomposed_plan"
            continue

    subgoal, language_action_str, _ = render_subgoal(sg_str, wp_num)

    return subgoal, language_action_str, None


def retrieve_waypoints(
    waypoint_generator: OracleGraph,
    item: str,
    number: int = 1,
    cur_inventory: dict = dict()
) -> str:
    item = item.lower().replace(" ", "_")
    item = item.replace("logs", "log")

    _cur_inventory = copy.deepcopy(cur_inventory)
    if item in _cur_inventory:
        del _cur_inventory[item]

    pretty_result, ordered_text, ordered_item, ordered_item_quantity = \
        waypoint_generator.compile(item.replace(" ", "_"), number, _cur_inventory)
    return pretty_result


def _normalise_waypoint_name(item: Any) -> str:
    item_name = str(item or "").lower().replace(" ", "_").strip()
    if item_name == "log" or item_name.endswith("_log"):
        return "logs"
    if item_name == "coal_ore":
        return "coal"
    if item_name == "redstone_ore":
        return "redstone"
    if item_name == "diamond_ore":
        return "diamond"
    return item_name


def _parse_waypoint_summary(wp_list_str: str) -> list[tuple[str, int, str]]:
    parsed: list[tuple[str, int, str]] = []
    for line in wp_list_str.splitlines()[1:]:
        match = re.match(r"^\s*\d+\.\s*([^:]+):\s*need\s*(\d+)", line)
        if not match:
            continue
        waypoint = _normalise_waypoint_name(match.group(1))
        required = int(match.group(2))
        parsed.append((waypoint, required, line.strip()))
    return parsed


def _inventory_count_for_waypoint(inventory: Dict[str, Any], waypoint: str) -> int:
    waypoint_name = _normalise_waypoint_name(waypoint)
    total = 0
    for item, quantity in (inventory or {}).items():
        try:
            count = int(quantity or 0)
        except Exception:
            continue
        item_name = _normalise_waypoint_name(item)
        if waypoint_name == "logs":
            if item_name == "logs" or item_name.endswith("_log"):
                total += count
            continue
        if waypoint_name == "planks":
            if item_name == "planks" or item_name.endswith("_planks"):
                total += count
            continue
        if item_name == waypoint_name:
            total += count
    return total


def _can_skip_satisfied_waypoint(waypoint: str) -> bool:
    waypoint_name = _normalise_waypoint_name(waypoint)
    if waypoint_name in {"crafting_table", "furnace", "shield"}:
        return True
    return waypoint_name.endswith((
        "_pickaxe",
        "_axe",
        "_shovel",
        "_hoe",
        "_sword",
        "_helmet",
        "_chestplate",
        "_leggings",
        "_boots",
    ))


def _select_next_planning_waypoint(
    wp_list_str: str,
    logger: logging.Logger,
    inventory: Dict[str, Any] | None = None,
) -> tuple[str, int]:
    parsed = _parse_waypoint_summary(wp_list_str)
    if not parsed:
        raise ValueError(f"Cannot parse waypoint summary: {wp_list_str}")
    if inventory:
        for waypoint, required, line in parsed:
            if (
                _can_skip_satisfied_waypoint(waypoint)
                and _inventory_count_for_waypoint(inventory, waypoint) >= required
            ):
                logger.info(
                    "Skipping already-satisfied planner waypoint: "
                    f"{line}, inventory={inventory}"
                )
                continue
            logger.info(f"Selected first unsatisfied planner waypoint: {line}")
            return waypoint, required
    logger.info(f"Selected first remaining planner waypoint: {parsed[0][2]}")
    return parsed[0][0], parsed[0][1]


def _pickaxe_prereq_for_mining(
    env_status: Dict[str, Any],
    waypoint: str,
) -> tuple[str, int] | None:
    waypoint_name = _normalise_waypoint_name(waypoint)
    if waypoint_name not in MINE_ONLY_WAYPOINTS:
        return None
    if _has_capable_pickaxe_for_target(env_status, waypoint_name):
        return None

    required_tier = ORE_REQUIRED_PICKAXE_TIER.get(waypoint_name, 1)
    counts = _normalised_inventory_counts(env_status)

    def missing(item: str, need: int) -> int:
        return max(0, need - int(counts.get(item, 0) or 0))

    if missing("stick", 2) > 0:
        return "stick", missing("stick", 2)
    if missing("crafting_table", 1) > 0:
        return "crafting_table", 1

    if required_tier >= 3:
        if counts.get("iron_ingot", 0) >= 3:
            return "iron_pickaxe", 1
        smeltable_iron = counts.get("iron_ingot", 0) + counts.get("iron_ore", 0)
        if smeltable_iron >= 3 and counts.get("furnace", 0) >= 1:
            return "iron_ingot", missing("iron_ingot", 3)
        if counts.get("furnace", 0) < 1:
            if counts.get("cobblestone", 0) >= 8:
                return "furnace", 1
            return "cobblestone", missing("cobblestone", 8)
        if not _has_capable_pickaxe_for_target(
            {"inventory": counts}, "iron_ore"
        ):
            return "stone_pickaxe", 1
        return "iron_ore", max(1, 3 - smeltable_iron)

    if required_tier == 2:
        if counts.get("cobblestone", 0) >= 3:
            return "stone_pickaxe", 1
        if not _has_capable_pickaxe_for_target(
            {"inventory": counts}, "cobblestone"
        ):
            return "wooden_pickaxe", 1
        return "cobblestone", missing("cobblestone", 3)

    return "wooden_pickaxe", 1


STONE_TOOL_MATERIALS = ("cobblestone", "blackstone", "cobbled_deepslate")
STONE_TOOL_MATERIAL_NEEDS = {
    "stone_pickaxe": 3,
    "stone_axe": 3,
    "stone_hoe": 2,
    "stone_shovel": 1,
    "stone_sword": 2,
    "furnace": 8,
}


def _stone_tool_material_count(counts: Dict[str, int]) -> int:
    return sum(int(counts.get(item, 0) or 0) for item in STONE_TOOL_MATERIALS)


def _crafting_prereq_for_waypoint(
    env_status: Dict[str, Any],
    waypoint: str,
) -> tuple[str, int] | None:
    waypoint_name = _normalise_waypoint_name(waypoint)
    material_need = STONE_TOOL_MATERIAL_NEEDS.get(waypoint_name)
    if material_need is None:
        return None

    counts = _normalised_inventory_counts(env_status)

    def missing(item: str, need: int) -> int:
        return max(0, need - int(counts.get(item, 0) or 0))

    if waypoint_name != "furnace":
        if missing("stick", 2) > 0:
            return "stick", missing("stick", 2)
        if missing("crafting_table", 1) > 0:
            return "crafting_table", 1

    valid_stone_materials = _stone_tool_material_count(counts)
    if valid_stone_materials >= material_need:
        return None

    if not _has_capable_pickaxe_for_target({"inventory": counts}, "cobblestone"):
        return "wooden_pickaxe", 1

    return "cobblestone", max(1, material_need - valid_stone_materials)


def _planning_prereq_for_waypoint(
    env_status: Dict[str, Any],
    waypoint: str,
) -> tuple[str, int, str] | None:
    craft_prereq = _crafting_prereq_for_waypoint(env_status, waypoint)
    if craft_prereq is not None:
        prereq_wp, prereq_num = craft_prereq
        return prereq_wp, prereq_num, "crafting_material"

    mining_prereq = _pickaxe_prereq_for_mining(env_status, waypoint)
    if mining_prereq is not None:
        prereq_wp, prereq_num = mining_prereq
        return prereq_wp, prereq_num, "mining_pickaxe"

    return None


def make_plan(
    original_final_goal: str,
    env_status: dict,
    action_memory: CaseBasedMemory,
    waypoint_generator: OracleGraph,
    topK: int,
    cfg: DictConfig,

    logger: logging.Logger,

    # needed for VLM call using Optimus-1's code
    obs: Dict[str, Any],
    hydra_path: str,
    run_uuid: str,
):
    inventory = env_status["inventory"]
    wp_list_str = retrieve_waypoints(waypoint_generator, original_final_goal, 1, inventory)
    logger.info(f"In make_plan")
    logger.info(f"wp_list_str: {wp_list_str}")
    wp, wp_num = _select_next_planning_waypoint(wp_list_str, logger, inventory)
    prereq_hops: list[tuple[str, int, str, str]] = []
    for _ in range(4):
        prereq = _planning_prereq_for_waypoint(env_status, wp)
        if prereq is None:
            break
        prereq_wp, prereq_num, reason = prereq
        prereq_hops.append((prereq_wp, prereq_num, reason, wp))
        logger.warning(
            "Planner selected waypoint before prerequisite was ready; "
            f"using {prereq_wp}:{prereq_num} before {wp}. "
            f"reason={reason} inventory={inventory}"
        )
        wp, wp_num = prereq_wp, prereq_num
    if len(prereq_hops) >= 4:
        logger.warning(
            "Stopped prerequisite expansion after 4 hops; "
            f"current waypoint={wp}:{wp_num}, hops={prereq_hops}"
        )

    state_snapshot = action_memory.create_state_snapshot(env_status, obs, cfg)
    case_decision = action_memory.select_case_decision(
        wp,
        wp_num,
        state_snapshot,
        topK,
        run_uuid,
        original_final_goal,
    )

    logger.info(f"In make_plan")
    logger.info(f"waypoint: {wp}, waypoint_num: {wp_num}")
    logger.info(f"case_decision: {str(case_decision is not None)}")

    if case_decision is not None:
        logger.info(f"Reuse case decision: {json.dumps(case_decision['decision_trace'])}")
        if _subgoal_action_is_feasible(wp, case_decision["subgoal"], case_decision["language_action_str"]):
            return wp, case_decision["subgoal"], case_decision["language_action_str"], None
        logger.warning(
            "Rejected infeasible case decision for waypoint "
            f"{wp}: {case_decision['language_action_str']}"
        )
        subgoal, language_action_str = _fallback_subgoal_for_waypoint(wp, wp_num)
        action_memory.record_decision(
            waypoint=wp,
            waypoint_num=wp_num,
            state_snapshot=state_snapshot,
            candidate_actions=[
                {
                    "action": case_decision["language_action_str"],
                    "source": "case_memory_rejected",
                },
                {
                    "action": language_action_str,
                    "source": "semantic_fallback",
                },
            ],
            selected_action=language_action_str,
            selected_subgoal=subgoal,
            selected_subgoal_str=json.dumps(subgoal),
            decision_trace={
                "source": "semantic_fallback",
                "rejected_action": case_decision["language_action_str"],
                "reason": "infeasible action verb for waypoint",
            },
            run_uuid=run_uuid,
            original_final_goal=original_final_goal,
        )
        return wp, subgoal, language_action_str, None

    else:
        logger.info(f"No high-confidence case for waypoint: {wp}, so, call planner to generate a plan.")

        similar_wp_sg_dict = action_memory.retrieve_similar_succeeded_waypoints(wp, topK, state_snapshot)
        failed_sg_list = action_memory.retrieve_failed_subgoals(wp) # could be empty list, i.e., []

        subgoal, language_action_str, error_message = call_planner_with_retry(
            cfg, obs, wp, wp_num, similar_wp_sg_dict, failed_sg_list, hydra_path, run_uuid, logger
        )
        if error_message is None:
            candidate_actions = [
                {
                    "action": language_action_str,
                    "source": "planner_selected",
                }
            ]
            decision_source = "planner"
            decision_reason = None
            if not _subgoal_action_is_feasible(wp, subgoal, language_action_str):
                rejected_action = language_action_str
                subgoal, language_action_str = _fallback_subgoal_for_waypoint(wp, wp_num)
                candidate_actions.append(
                    {
                        "action": language_action_str,
                        "source": "semantic_fallback",
                    }
                )
                decision_source = "semantic_fallback"
                decision_reason = f"rejected infeasible planner action: {rejected_action}"
                logger.warning(
                    "Rejected infeasible planner action for waypoint "
                    f"{wp}: {rejected_action}; using {language_action_str}."
                )
            action_memory.record_decision(
                waypoint=wp,
                waypoint_num=wp_num,
                state_snapshot=state_snapshot,
                candidate_actions=candidate_actions,
                selected_action=language_action_str,
                selected_subgoal=subgoal,
                selected_subgoal_str=json.dumps(subgoal),
                decision_trace={
                    "source": decision_source,
                    "confidence": None,
                    "retrieved_examples": similar_wp_sg_dict,
                    "failed_subgoals": failed_sg_list,
                    "reason": decision_reason,
                },
                run_uuid=run_uuid,
                original_final_goal=original_final_goal,
            )

        return wp, subgoal, language_action_str, error_message


# cfg, obs, current_sg_prompt, waypoint, hydra_path, run_uuid, logger
def call_reasoning_with_retry(
    cfg: DictConfig,
    obs: Dict[str, Any],
    current_sg_prompt: str,
    waypoint: str,
    hydra_path: str,
    run_uuid: str,
    logger: logging.Logger,
):
    attempts = 0
    max_retries = 3
    reasoning, visual_description = "", ""
    while attempts < max_retries:
        attempts += 1

        logger.info(f"Attempt: {attempts}, Just before get_context_aware_reasoning: ")
        logger.info(f"current_sg_prompt: {current_sg_prompt}")
        logger.info(f"waypoint: {waypoint}")
        logger.info(f"Starting get_context_aware_reasoning ...\n")

        try:
            reasoning, visual_description = ServerAPI.get_context_aware_reasoning(
                cfg["server"],
                obs,
                current_sg_prompt,
                waypoint,
                hydra_path=hydra_path,
                run_uuid=run_uuid
            )
            tmp_dict, render_error = render_context_aware_reasoning(copy.deepcopy(reasoning))
            if render_error is None:
                break

            logger.warning(f"get_context_aware_reasoning at attempt {attempts} failed. Error message: {render_error}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_context_aware_reasoning.")
                return dict(), "", "max_tries_get_context_aware_reasoning"
        
        except Exception as e:
            logger.info(f"Error in get_context_aware_reasoning: {e}")
            if attempts >= max_retries:
                logger.error("Max retries reached. Could not fetch get_context_aware_reasoning.")
                return dict(), "", "max_tries_get_context_aware_reasoning"
            continue
    
    reasoning_dict, render_error = render_context_aware_reasoning(reasoning)
    return reasoning_dict, visual_description, render_error


def check_waypoint_item_obtained(new_item_dict, waypoint, logger):
    if len(new_item_dict) == 0:
        logger.error("env.inventory_new_item is True, but env.inventory_new_item_what() is empty.")
        return False

    for new_item_name in new_item_dict.keys():
        if "log" in waypoint and "log" in new_item_name:
            return True
        elif "planks" in waypoint and "planks" in new_item_name:
            return True
        elif "coal" in waypoint and "coal" in new_item_name:
            return True
        elif waypoint == new_item_name:
            return True

    return False


def _is_tree_chop_subgoal(prompt: str, target: list[Any] | tuple[Any, ...] | None) -> bool:
    if not target:
        return False
    target_item = str(target[0]).lower()
    is_log_goal = target_item in {"log", "logs"} or target_item.endswith("_log")
    if not is_log_goal:
        return False
    prompt_text = (prompt or "").lower()
    return any(token in prompt_text for token in ("chop", "punch", "tree", "log", "logs"))


def _log_activity_count(env_status: Dict[str, Any]) -> int:
    total = 0
    ledger = env_status.get("resource_ledger") or {}
    for bucket_name in ("mined_blocks", "pickup", "collected", "max_inventory"):
        bucket = ledger.get(bucket_name) or {}
        for item, quantity in bucket.items():
            if isinstance(item, str) and item.endswith("_log"):
                total += int(quantity or 0)

    inventory = env_status.get("inventory") or {}
    for item, quantity in inventory.items():
        if isinstance(item, str) and item.endswith("_log"):
            total += int(quantity or 0)
    return total


ORE_LAYER_ORDER = {
    "coal": 0,
    "iron_ore": 1,
    "gold_ore": 2,
    "redstone": 3,
    "diamond": 4,
}

ORE_ALIASES = {
    "coal_ore": "coal",
    "iron": "iron_ore",
    "iron_ore": "iron_ore",
    "gold": "gold_ore",
    "gold_ore": "gold_ore",
    "redstone": "redstone",
    "redstone_ore": "redstone",
    "diamond": "diamond",
    "diamond_ore": "diamond",
}


def _normalise_ore_name(item: Any) -> str:
    return ORE_ALIASES.get(str(item).lower(), str(item).lower())


def _is_layered_mining_subgoal(prompt: str, target: list[Any] | tuple[Any, ...] | None) -> bool:
    """Return True for mining sub-goals that should run the layered
    overshoot / pillar-up logic.

    Includes both layered ores (``ORE_LAYER_ORDER``) and ``cobblestone``,
    because the agent typically dig-downs through every band while
    collecting cobblestone for stone tools — and that is exactly when a
    "deeper ore in inventory" signal indicates the agent has gone too
    far. For cobblestone the *effective* mining_target_ore is remapped
    elsewhere to ``iron_ore`` (the natural next tier) so the deeper-seen
    set and the overshoot Y band both make sense.
    """
    if not target:
        return False
    target_item = _normalise_ore_name(target[0])
    prompt_text = (prompt or "").lower()
    if target_item not in ORE_LAYER_ORDER and target_item != "cobblestone":
        return False
    return "mine" in prompt_text or "dig" in prompt_text


def _effective_mining_target_ore(planner_target: str) -> str:
    """Map a planner sub-goal target onto the ore name the pillar-up
    machinery should reference.

    Cobblestone has no Y band of its own, but stone-tier mining is the
    natural precursor to iron mining. Reusing ``iron_ore``'s band gives
    the overshoot logic a sensible reference depth (anything below
    iron's lowest valid Y minus the configured margin counts as too
    deep) without needing to peek at the upcoming waypoint chain.
    """
    if planner_target == "cobblestone":
        return os.environ.get("XENON_COBBLESTONE_PILLAR_TARGET", "iron_ore")
    return planner_target


def _ore_required_count(target: list[Any] | tuple[Any, ...] | None) -> int:
    if not target or len(target) < 2:
        return 1
    try:
        return int(target[1])
    except Exception:
        return 1


def _ore_count_in_mapping(values: Dict[str, Any], target_ore: str) -> int:
    total = 0
    for item, quantity in (values or {}).items():
        if _normalise_ore_name(item) == target_ore:
            try:
                total += int(quantity or 0)
            except Exception:
                continue
    return total


def _ore_activity_count(env_status: Dict[str, Any], target_ore: str) -> int:
    total = _ore_count_in_mapping(env_status.get("inventory") or {}, target_ore)
    ledger = env_status.get("resource_ledger") or {}
    for bucket_name in ("max_inventory", "pickup", "collected", "mined_blocks"):
        total += _ore_count_in_mapping(ledger.get(bucket_name) or {}, target_ore)
    return total


def _ore_available_count(env_status: Dict[str, Any], target_ore: str) -> int:
    inventory = _ore_count_in_mapping(env_status.get("inventory") or {}, target_ore)
    ledger = env_status.get("resource_ledger") or {}
    observed = max(
        inventory,
        _ore_count_in_mapping(ledger.get("max_inventory") or {}, target_ore),
        _ore_count_in_mapping(ledger.get("pickup") or {}, target_ore),
        _ore_count_in_mapping(ledger.get("collected") or {}, target_ore),
    )
    return observed


def _deeper_ores_seen(env_status: Dict[str, Any], target_ore: str) -> list[str]:
    target_rank = ORE_LAYER_ORDER.get(target_ore)
    if target_rank is None:
        return []
    seen: set[str] = set()
    mappings = [env_status.get("inventory") or {}]
    ledger = env_status.get("resource_ledger") or {}
    mappings.extend((ledger.get(name) or {}) for name in ("max_inventory", "pickup", "collected", "mined_blocks"))
    for values in mappings:
        for item, quantity in values.items():
            ore = _normalise_ore_name(item)
            if ore in ORE_LAYER_ORDER and ORE_LAYER_ORDER[ore] > target_rank:
                try:
                    if int(quantity or 0) > 0:
                        seen.add(ore)
                except Exception:
                    continue
    return sorted(seen, key=lambda ore: ORE_LAYER_ORDER[ore])


def _deeper_ore_available_counts(env_status: Dict[str, Any], target_ore: str) -> Dict[str, int]:
    target_rank = ORE_LAYER_ORDER.get(target_ore)
    if target_rank is None:
        return {}
    counts: Dict[str, int] = {}
    for ore, rank in ORE_LAYER_ORDER.items():
        if rank > target_rank:
            count = _ore_available_count(env_status, ore)
            if count > 0:
                counts[ore] = count
    return counts


def _forward_mining_prompt(target_ore: str) -> str:
    return f"dig forward and mine {target_ore.replace('_', ' ')}"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _lateral_shift_succeeded(result: Dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    lateral = result.get("lateral_shift")
    if not isinstance(lateral, dict):
        return False
    required_delta = _env_float("XENON_CORRIDOR_MIN_MOVE_DELTA", 0.20)
    try:
        horizontal_delta = float(lateral.get("horizontal_delta", 0.0))
    except (TypeError, ValueError):
        horizontal_delta = 0.0
    return (
        bool(lateral.get("success"))
        and horizontal_delta >= required_delta
        and not bool(lateral.get("height_drop"))
    )


PICKAXE_PRIORITY = ("diamond_pickaxe", "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
PICKAXE_TIER = {
    "wooden_pickaxe": 1,
    "stone_pickaxe": 2,
    "iron_pickaxe": 3,
    "diamond_pickaxe": 4,
}
ORE_REQUIRED_PICKAXE_TIER = {
    "cobblestone": 1,
    "coal": 1,
    "coal_ore": 1,
    "iron_ore": 2,
    "gold_ore": 3,
    "redstone": 3,
    "redstone_ore": 3,
    "diamond": 3,
    "diamond_ore": 3,
}
MINE_ONLY_WAYPOINTS = {
    "cobblestone",
    "coal",
    "coal_ore",
    "iron_ore",
    "gold_ore",
    "redstone",
    "redstone_ore",
    "diamond",
    "diamond_ore",
}


def _is_mining_action(text: str) -> bool:
    action = (text or "").lower()
    return any(token in action for token in ("mine", "dig", "break"))


def _is_pickaxe_mining_subgoal(prompt: str, target: list[Any] | tuple[Any, ...] | None) -> bool:
    if not target:
        return False
    target_item = _normalise_ore_name(target[0])
    if target_item in {"log", "logs"} or str(target[0]).endswith("_log"):
        return False
    return _is_mining_action(prompt) and (
        target_item in ORE_LAYER_ORDER or target_item in MINE_ONLY_WAYPOINTS or target_item == "cobblestone"
    )


def _subgoal_action_is_feasible(waypoint: str, subgoal: Dict[str, Any] | None, language_action: str) -> bool:
    action_text = ((subgoal or {}).get("task") or language_action or "").lower()
    waypoint_name = _normalise_ore_name(waypoint)
    if waypoint_name in MINE_ONLY_WAYPOINTS:
        return _is_mining_action(action_text) and "craft" not in action_text and "smelt" not in action_text
    if waypoint_name in {"log", "logs"} or waypoint_name.endswith("_log"):
        return any(token in action_text for token in ("chop", "punch", "tree", "log"))
    if waypoint_name.endswith("_ingot") or waypoint_name == "charcoal":
        return "smelt" in action_text
    return True


def _fallback_subgoal_for_waypoint(waypoint: str, waypoint_num: int) -> tuple[Dict[str, Any], str]:
    waypoint_name = _normalise_ore_name(waypoint)
    if waypoint_name in {"log", "logs"} or waypoint_name.endswith("_log"):
        subgoal = {"task": "chop a tree", "goal": ["logs", waypoint_num]}
    elif waypoint_name == "diamond":
        subgoal = {"task": "dig down and mine diamond", "goal": ["diamond", waypoint_num]}
    elif waypoint_name in MINE_ONLY_WAYPOINTS:
        goal_name = waypoint_name
        subgoal = {"task": f"dig down and mine {goal_name}", "goal": [goal_name, waypoint_num]}
    elif waypoint_name.endswith("_ingot") or waypoint_name == "charcoal":
        item_name = waypoint_name.replace("_ingot", " ore") if waypoint_name.endswith("_ingot") else waypoint_name
        subgoal = {"task": f"smelt {item_name}", "goal": [waypoint_name, waypoint_num]}
    else:
        subgoal = {"task": f"craft {waypoint_name}", "goal": [waypoint_name, waypoint_num]}
    return subgoal, subgoal["task"]


def _normalised_inventory_counts(env_status: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item, quantity in (env_status.get("inventory") or {}).items():
        try:
            counts[str(item)] = counts.get(str(item), 0) + int(quantity or 0)
        except Exception:
            continue
    for item in (env_status.get("plain_inventory") or {}).values():
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", ""))
        if item_type in ("", "none", "air"):
            continue
        try:
            quantity = int(item.get("quantity", 0) or 0)
        except Exception:
            quantity = 0
        if quantity > 0:
            counts[item_type] = counts.get(item_type, 0) + quantity
    return counts


def _best_pickaxe_from_status(env_status: Dict[str, Any]) -> str:
    counts = _normalised_inventory_counts(env_status)
    for pickaxe in PICKAXE_PRIORITY:
        if counts.get(pickaxe, 0) > 0:
            return pickaxe
    return ""


def _has_capable_pickaxe_for_target(env_status: Dict[str, Any], target_ore: str) -> bool:
    required_tier = ORE_REQUIRED_PICKAXE_TIER.get(_normalise_ore_name(target_ore), 1)
    best_pickaxe = _best_pickaxe_from_status(env_status)
    return PICKAXE_TIER.get(best_pickaxe, 0) >= required_tier


def _ensure_best_pickaxe_equipped(
    env: CustomEnvWrapper,
    helper: NewHelper,
    prompt: str,
    target: list[Any] | tuple[Any, ...] | None,
    pbar: Progress,
    num_step: TaskID,
    logger: logging.Logger,
) -> None:
    if not _is_pickaxe_mining_subgoal(prompt, target):
        return
    env_status = env.get_status()
    best_pickaxe = _best_pickaxe_from_status(env_status)
    if not best_pickaxe:
        return
    if env_status.get("equipment") == best_pickaxe:
        return

    previous_can_change_hotbar = env.can_change_hotbar
    previous_can_open_inventory = env.can_open_inventory
    env.can_change_hotbar = True
    env.can_open_inventory = True
    try:
        equip_prompt = f"equip {best_pickaxe}"
        helper.reset(equip_prompt, pbar, num_step, logger)
        equipped, info = helper.step(equip_prompt, [best_pickaxe, 1])
        if equipped:
            logger.info(f"Pre-mining equipment check: equipped {best_pickaxe} for {prompt}.")
        else:
            logger.warning(f"Pre-mining equipment check failed for {best_pickaxe}: {info}")
    finally:
        env.can_change_hotbar = previous_can_change_hotbar
        env.can_open_inventory = previous_can_open_inventory


# Map planner-level normalised ore names back to the canonical wrapper
# keys used by `CustomEnvWrapper.ORE_HEIGHT_BANDS`. The wrapper keeps the
# `_ore` suffix because that is what `random_ore` writes to the world.
_PILLAR_PLANNER_TO_WRAPPER_ORE: Dict[str, str] = {
    "coal": "coal_ore",
    "coal_ore": "coal_ore",
    "iron_ore": "iron_ore",
    "iron": "iron_ore",
    "gold_ore": "gold_ore",
    "gold": "gold_ore",
    "redstone": "redstone_ore",
    "redstone_ore": "redstone_ore",
    "diamond": "diamond_ore",
    "diamond_ore": "diamond_ore",
}


def _maybe_pillar_up_for_ore(
    env: CustomEnvWrapper,
    prompt: str,
    target: list[Any] | tuple[Any, ...] | None,
    logger: logging.Logger,
) -> None:
    """Perceive the current height and pillar up if it would help mining.

    This is the high-level glue between the new env-wrapper primitives
    (``perceive_height_context`` / ``raise_to_ore_band``) and the planning
    loop. It is opt-in: behaviour is unchanged unless the env var
    ``XENON_ENABLE_PILLAR_UP_FOR_ORE`` is set to ``1``. Even when enabled,
    the helper only acts when:

    1. the current sub-goal is a pickaxe-mining sub-goal,
    2. the target ore maps to a known canonical Y band,
    3. the agent currently sits below that band, and
    4. at least one placeable block (cobblestone / dirt / stone / ...)
       exists somewhere in the agent's inventory.

    All other situations are no-ops, so this helper never fights the
    existing dig-down logic for diamonds at the bottom of the world.
    """
    if os.environ.get("XENON_ENABLE_PILLAR_UP_FOR_ORE", "0") != "1":
        return
    if not _is_pickaxe_mining_subgoal(prompt, target):
        return
    if not target:
        return
    planner_ore = _normalise_ore_name(target[0])
    wrapper_ore = _PILLAR_PLANNER_TO_WRAPPER_ORE.get(planner_ore)
    if wrapper_ore is None:
        return

    try:
        ascend_margin = int(os.environ.get("XENON_PILLAR_ORE_MARGIN", "1"))
    except ValueError:
        ascend_margin = 1
    try:
        max_blocks = int(os.environ.get("XENON_PILLAR_ORE_MAX_BLOCKS", "32"))
    except ValueError:
        max_blocks = 32
    try:
        max_steps = int(os.environ.get("XENON_PILLAR_ORE_MAX_STEPS", "400"))
    except ValueError:
        max_steps = 400

    try:
        ctx = env.perceive_height_context(
            look_for=wrapper_ore, ascend_margin=ascend_margin
        )
    except Exception as exc:
        logger.warning(f"perceive_height_context({wrapper_ore}) failed: {exc!s}")
        return

    if ctx.get("recommended_action") != "pillar_up":
        logger.info(
            "[pillar_up_for_ore] skip: ore=%s recommendation=%s reason=%s",
            wrapper_ore,
            ctx.get("recommended_action"),
            ctx.get("reason"),
        )
        return
    if int(ctx.get("placeable_total", 0)) <= 0:
        logger.info(
            "[pillar_up_for_ore] skip: ore=%s no placeable block in inventory",
            wrapper_ore,
        )
        return

    logger.info(
        "[pillar_up_for_ore] activating: ore=%s y=%.1f -> target_y=%s "
        "recommended_dy=%s placeable_hotbar=%d placeable_total=%d",
        wrapper_ore,
        float(ctx.get("current_y", 0.0)),
        ctx.get("target_y"),
        ctx.get("recommended_dy"),
        int(ctx.get("placeable_in_hotbar", 0)),
        int(ctx.get("placeable_total", 0)),
    )
    try:
        result = env.raise_to_ore_band(
            wrapper_ore,
            max_blocks=max_blocks,
            max_steps=max_steps,
            ascend_margin=ascend_margin,
        )
    except Exception as exc:
        logger.warning(f"raise_to_ore_band({wrapper_ore}) failed: {exc!s}")
        return
    logger.info(
        "[pillar_up_for_ore] result: ore=%s success=%s dy=%.1f "
        "blocks_used=%s reason=%s",
        wrapper_ore,
        result.get("success"),
        float(result.get("dy", 0.0)),
        result.get("blocks_used"),
        result.get("reason"),
    )


def _ore_band_midpoint(wrapper_ore: str) -> int | None:
    """Return the integer midpoint of the wrapper's canonical ore band.

    Bands come from ``CustomEnvWrapper.ORE_HEIGHT_BANDS``. Evidence for
    these ranges (verified against both the project's ``random_ore``
    spawner and the official Minecraft pre-1.17 distribution table) is
    documented in the wrapper. Midpoints used for horizontal-mine
    targeting:

        coal_ore [45,50] -> 47
        iron_ore [26,43] -> 34
        gold_ore [17,26] -> 21
        redstone_ore [5,16] -> 10
        diamond_ore [1,14] -> 7
    """
    band = CustomEnvWrapper.ORE_HEIGHT_BANDS.get(wrapper_ore)
    if band is None:
        return None
    lo, hi = band
    return (int(lo) + int(hi)) // 2


def _maybe_relevel_for_overshoot(
    env: CustomEnvWrapper,
    planner_ore: str,
    new_deeper_seen: list[str],
    logger: logging.Logger,
    target_y: float | None = None,
) -> Dict[str, Any] | None:
    """Pillar up after an overshoot.

    Destination Y selection:
        * If the caller passes ``target_y`` (typically the Y at which
          the target ore was first encountered for this sub-goal), the
          agent is lifted to that exact height. This is the user's
          requested semantics — "回到我们第一次挖到这个矿的高度
          重新探索".
        * Otherwise, fall back to the band-midpoint heuristic so the
          first-ever pillar-up before any target ore was found still
          has a sensible destination.

    What this helper does:
        1. Maps the planner-side ore name onto the wrapper's canonical
           key.
        2. Reads ``perceive_height_context`` to learn current Y and
           placeable-block availability.
        3. If the agent is below the destination by at least
           ``XENON_OVERSHOOT_RELEVEL_MIN_DY`` blocks (default 2) AND
           there is at least one placeable block in the inventory,
           calls ``raise_to_height(dest_y)``. Near the bedrock floor, a
           smaller one-block correction is allowed so the agent does not
           keep digging down when it is just below the first target-ore Y.
        4. Logs perception + result; never raises.

    Returns the ``raise_to_height`` result dict, or ``None`` when the
    helper opted not to act.

    Behaviour is gated by ``XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT``
    (default ``"1"`` ON).
    """
    if os.environ.get("XENON_ENABLE_PILLAR_UP_FOR_OVERSHOOT", "1") != "1":
        return None

    wrapper_ore = _PILLAR_PLANNER_TO_WRAPPER_ORE.get(planner_ore)
    if wrapper_ore is None:
        return None

    try:
        ctx = env.perceive_height_context(look_for=wrapper_ore)
    except Exception as exc:
        logger.warning(
            f"[overshoot_relevel] perceive_height_context({wrapper_ore}) "
            f"failed: {exc!s}"
        )
        return None

    cur_y = float(ctx.get("current_y", 64.0))

    # Destination Y: prefer the first-encounter Y from the caller; fall
    # back to band midpoint when unknown. If the agent is already at or
    # above that destination, the "return to height" part is a no-op but
    # the v7 shaft-relocation loop should still perform the short
    # horizontal offset before resuming dig-down.
    band_mid = _ore_band_midpoint(wrapper_ore)
    if target_y is not None:
        dest_y = float(target_y)
        dest_source = "first_target_ore_y"
    elif band_mid is not None:
        dest_y = float(band_mid)
        dest_source = "band_midpoint"
    else:
        return None
    needed_dy = dest_y - cur_y

    def _apply_lateral_shift(result: Dict[str, Any]) -> Dict[str, Any]:
        # After re-levelling, do not hand control to open-ended
        # horizontal mining. Script only a short lateral offset so the
        # agent leaves the old vertical shaft, then let the original
        # "dig down" prompt resume.
        #
        # This implements the new loop:
        #   dig down -> overshoot -> pillar/no-op back to target Y ->
        #   move horizontally until x/z changes -> dig down again at a new x/z.
        #
        # Keep the old XENON_OVERSHOOT_TUNNEL_* env vars as a fallback for
        # existing runner scripts, but the default is now one successful
        # displacement; repeated overshoot triggers provide additional shifts.
        lateral_blocks = _env_int(
            "XENON_OVERSHOOT_LATERAL_BLOCKS",
            _env_int("XENON_OVERSHOOT_TUNNEL_BLOCKS", 1),
        )
        lateral_max_steps = _env_int(
            "XENON_OVERSHOOT_LATERAL_MAX_STEPS",
            _env_int("XENON_OVERSHOOT_TUNNEL_MAX_STEPS", 240),
        )
        lateral_attempts = max(1, _env_int("XENON_OVERSHOOT_LATERAL_RETRIES", 3))
        if lateral_blocks > 0 and bool(result.get("success")):
            for attempt in range(lateral_attempts):
                if attempt > 0:
                    try:
                        logger.info(
                            "[overshoot_relevel] lateral retry %d/%d: relevel to %.1f before retry",
                            attempt + 1,
                            lateral_attempts,
                            dest_y,
                        )
                        env.raise_to_height(
                            dest_y,
                            max_blocks=_env_int("XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS", 64),
                            max_steps=_env_int("XENON_OVERSHOOT_RELEVEL_MAX_STEPS", 600),
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[overshoot_relevel] retry raise_to_height({dest_y:.1f}) failed: {exc!s}"
                        )
                try:
                    lateral_shift = env.dig_forward_blocks(
                        n_blocks=lateral_blocks,
                        max_steps=lateral_max_steps,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[overshoot_relevel] lateral dig_forward_blocks failed: {exc!s}"
                    )
                    continue
                horizontal_delta = (
                    (float(lateral_shift.get("end_x", 0.0)) - float(lateral_shift.get("start_x", 0.0))) ** 2
                    + (float(lateral_shift.get("end_z", 0.0)) - float(lateral_shift.get("start_z", 0.0))) ** 2
                ) ** 0.5
                lateral_shift["horizontal_delta"] = horizontal_delta
                lateral_shift["attempt"] = attempt + 1
                logger.info(
                    "[overshoot_relevel] lateral_shift: attempt=%d/%d blocks_dug=%s/%d "
                    "steps=%s reason=%s horizontal_delta=%.2f "
                    "start=(%.1f,%.1f,%.1f) end=(%.1f,%.1f,%.1f)",
                    attempt + 1,
                    lateral_attempts,
                    lateral_shift.get("blocks_dug"),
                    lateral_blocks,
                    lateral_shift.get("steps_used"),
                    lateral_shift.get("reason"),
                    horizontal_delta,
                    float(lateral_shift.get("start_x", 0.0)),
                    float(lateral_shift.get("start_y", 0.0)),
                    float(lateral_shift.get("start_z", 0.0)),
                    float(lateral_shift.get("end_x", 0.0)),
                    float(lateral_shift.get("end_y", 0.0)),
                    float(lateral_shift.get("end_z", 0.0)),
                )
                result["lateral_shift"] = lateral_shift
                # Backward-compatible alias for older summary scripts that
                # still look for the previous post-pillar tunnel payload.
                result["tunnel"] = lateral_shift
                if _lateral_shift_succeeded(result):
                    break
        elif lateral_blocks > 0:
            logger.info(
                "[overshoot_relevel] lateral_shift skipped: pillar-up was unsuccessful"
            )
        result["lateral_success"] = (
            lateral_blocks <= 0 or _lateral_shift_succeeded(result)
        )
        return result

    try:
        min_dy = float(os.environ.get("XENON_OVERSHOOT_RELEVEL_MIN_DY", "2"))
    except ValueError:
        min_dy = 2.0
    if needed_dy <= 0:
        logger.info(
            "[overshoot_relevel] skip: ore=%s cur_y=%.1f dest_y=%.1f (%s) "
            "needed_dy=%.1f <= 0.0 (already in or above destination)",
            wrapper_ore, cur_y, dest_y, dest_source, needed_dy,
        )
        return _apply_lateral_shift(
            {
                "success": True,
                "skipped": True,
                "start_y": cur_y,
                "end_y": cur_y,
                "dy": 0.0,
                "blocks_used": 0,
                "steps_used": 0,
                "reason": "already_in_or_above_destination",
                "prep_action": "noop",
                "target_y": dest_y,
            }
        )
    if needed_dy < min_dy:
        try:
            low_y_floor = float(os.environ.get("XENON_BEDROCK_FLOOR_Y", "8.0"))
        except ValueError:
            low_y_floor = 8.0
        try:
            low_y_min_dy = float(os.environ.get("XENON_OVERSHOOT_LOW_Y_MIN_DY", "1.0"))
        except ValueError:
            low_y_min_dy = 1.0
        allow_low_y_micro_lift = cur_y <= low_y_floor and needed_dy >= low_y_min_dy
        if not allow_low_y_micro_lift:
            logger.info(
                "[overshoot_relevel] skip: ore=%s cur_y=%.1f dest_y=%.1f (%s) "
                "needed_dy=%.1f < min_dy=%.1f (already close enough to destination)",
                wrapper_ore, cur_y, dest_y, dest_source, needed_dy, min_dy,
            )
            return _apply_lateral_shift(
                {
                    "success": True,
                    "skipped": True,
                    "start_y": cur_y,
                    "end_y": cur_y,
                    "dy": 0.0,
                    "blocks_used": 0,
                    "steps_used": 0,
                    "reason": "close_enough_to_destination",
                    "prep_action": "noop",
                    "target_y": dest_y,
                }
            )
        logger.info(
            "[overshoot_relevel] allowing low-Y micro-lift: ore=%s cur_y=%.1f "
            "dest_y=%.1f needed_dy=%.1f min_dy=%.1f floor=%.1f",
            wrapper_ore, cur_y, dest_y, needed_dy, min_dy, low_y_floor,
        )
    if int(ctx.get("placeable_total", 0)) <= 0:
        logger.info(
            "[overshoot_relevel] skip: ore=%s no placeable block in inventory; "
            "leaving height unchanged.",
            wrapper_ore,
        )
        return None

    try:
        max_blocks = int(os.environ.get("XENON_OVERSHOOT_RELEVEL_MAX_BLOCKS", "64"))
    except ValueError:
        max_blocks = 64
    try:
        max_steps = int(os.environ.get("XENON_OVERSHOOT_RELEVEL_MAX_STEPS", "600"))
    except ValueError:
        max_steps = 600

    logger.info(
        "[overshoot_relevel] activating: ore=%s cur_y=%.1f -> dest_y=%.1f (%s) "
        "(band=%s) deeper_seen=%s placeable_hotbar=%d placeable_total=%d",
        wrapper_ore,
        cur_y,
        dest_y,
        dest_source,
        ctx.get("target_band"),
        new_deeper_seen,
        int(ctx.get("placeable_in_hotbar", 0)),
        int(ctx.get("placeable_total", 0)),
    )
    try:
        result = env.raise_to_height(
            dest_y, max_blocks=max_blocks, max_steps=max_steps
        )
    except Exception as exc:
        logger.warning(
            f"[overshoot_relevel] raise_to_height({dest_y:.1f}) failed: {exc!s}"
        )
        return None
    logger.info(
        "[overshoot_relevel] result: ore=%s success=%s end_y=%.1f dy=%.1f "
        "blocks_used=%s reason=%s prep_action=%s",
        wrapper_ore,
        result.get("success"),
        float(result.get("end_y", 0.0)),
        float(result.get("dy", 0.0)),
        result.get("blocks_used"),
        result.get("reason"),
        result.get("prep_action"),
    )

    return _apply_lateral_shift(result)


def new_agent_do(
    cfg: DictConfig,
    env: CustomEnvWrapper,
    logger: logging.Logger,
    monitors: Monitors,
    reset_obs: Dict[str, Any],
    action_memory: CaseBasedMemory,
    original_task: str,
    original_final_goal: str,
    run_uuid: str
):
    prefix = cfg.get("prefix")
    logger.info(f"[yellow]In agent_do(), prefix: {prefix}[/yellow]")

    oracle_knowledge_graph = OracleGraph()
    helper = NewHelper(env, oracle_knowledge_graph, prefix)
    obs = reset_obs

    # image_to_log = wandb.Image(obs["pov"], caption=f"Observation at step 0")
    # wandb.log({
    #     f"obs/0": image_to_log,
    # })
    env_status = env.get_status()
    loc = env_status["location_stats"]
    initial_xpos, initial_ypos, initial_zpos = loc["xpos"].item(), loc["ypos"].item(), loc["zpos"].item()
    # wandb.config.update({
    #     "initial_xpos": initial_xpos,
    #     "initial_ypos": initial_ypos,
    #     "initial_zpos": initial_zpos,
    # }, allow_val_change=True)

    logger.info(f"[yellow]original_final_goal: {original_final_goal}[/yellow]")

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # MineRL is unstable, so check env_malmo_logger for 'Exception' periodically
    env_malmo_logger_port = env.instances[0]._target_port - 9000
    env_malmo_logger_path = os.path.join(hydra_path.split('logs')[0], 'logs', f'mc_{env_malmo_logger_port}.log')

    if _malmo_log_has_fatal_error(env_malmo_logger_path, logger):
        return "env_malmo_logger_error", None, None, None, None

    status = ""
    original_final_goal_success = False

    waypoint = ""
    subgoal = None
    language_action_str = ""
    subgoal_done = False
    topK = cfg["memory"]["topK"]
    waypoint_generator = OracleGraph() # OracleGraph knows all recipes accurately.

    completed_subgoals = []
    completed_waypoints = []
    failed_subgoals = []
    failed_waypoints = []

    num_reasoning_intervention = 0
    step_waypoint_obtained = 0

    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        "{task.completed} of {task.total}",
        expand=True,
    ) as pbar:
        num_step = pbar.add_task("[cyan]Running...", total=env.timeout)

        progress = 0
        game_over = False

        while not game_over:
            if subgoal is None:
                # check if original_final_goal is achieved
                original_final_goal_success = env.check_original_goal_finish([original_final_goal, 1])
                if original_final_goal_success:
                    logger.info(f"[green]Original Goal: {original_final_goal} is achieved![/green]")
                    status = "success"
                    break

                env_status = env.get_status()
                waypoint, subgoal, language_action_str, error_message = make_plan(
                    original_final_goal,
                    env_status,
                    action_memory,
                    waypoint_generator,
                    topK,
                    cfg,
                    logger,
                    obs,
                    hydra_path,
                    run_uuid
                )
                if error_message is not None:
                    logger.error(f"Error message: {error_message}")
                    status = "cannot generate plan"
                    failed_subgoals = [f"achieve {waypoint}"]
                    break

                subgoal_done = False
                try:
                    waypoint_num_now = _ore_required_count((subgoal or {}).get("goal"))
                    env_status_now = env.get_status()
                    if _can_skip_satisfied_waypoint(str(waypoint)) and env._check_goal_inventory_state(
                        env_status_now.get("inventory", {}),
                        str(waypoint),
                        waypoint_num_now,
                    ):
                        logger.info(
                            "Skipping already-satisfied waypoint before execution: "
                            f"waypoint={waypoint}, need={waypoint_num_now}, "
                            f"inventory={env_status_now.get('inventory', {})}"
                        )
                        action_memory.save_success_failure(
                            waypoint,
                            language_action_str,
                            is_success=True,
                            outcome_status="success",
                            env_status=env_status_now,
                        )
                        completed_waypoints.append(waypoint)
                        subgoal = None
                        continue
                except Exception as exc:
                    logger.warning(
                        f"Pre-execution waypoint satisfaction check failed for "
                        f"{waypoint}: {exc}"
                    )
                logger.info(f"After make_plan()")
                logger.info(f"[yellow]Waypoint: {waypoint}, Subgoal: {subgoal}[/yellow]")

            current_sg = subgoal
            current_sg_prompt, current_sg_target = copy.deepcopy(current_sg["task"]), copy.deepcopy(current_sg["goal"])
            if current_sg_target[0] == "log":
                current_sg_target[0] = "logs"

            temp_sg_prompt = copy.deepcopy(current_sg_prompt)
            if "punch" in current_sg_prompt:
                current_sg_prompt = current_sg_prompt.replace("punch", "chop")
            op = current_sg_prompt.split(" ")[0]

            if "create" in current_sg_prompt:
                op = "craft"

            logger.info(f"[yellow]Subgoal Prompt: {current_sg_prompt}, Subgoal Target: {current_sg_target}[/yellow]")

            if op in ["craft", "smelt"] or "smelt" in current_sg_prompt:
                if not env.can_change_hotbar:
                    env.can_change_hotbar = True
                if not env.can_open_inventory:
                    env.can_open_inventory = True
                helper.reset(current_sg_prompt, pbar, num_step, logger)
                sg_done, info = helper.step(current_sg_prompt, current_sg_target)
                steps = helper.get_task_steps(current_sg_prompt)

                env.can_open_inventory = False
                env.can_change_hotbar = False

                monitors.update(f"{current_sg_prompt}_{progress}", sg_done, steps)
                if sg_done:
                    logger.info(f"[green]{current_sg_prompt} Success[/green]!")
                    progress += 1
                    completed_subgoals.append(current_sg)
                    subgoal_done = True

                    if "pickaxe" in waypoint:
                        env.can_change_hotbar = True
                        env.can_open_inventory = True
                        tmp_prompt = f"equip {waypoint}"
                        tmp_sg_target = [waypoint, 1]
                        helper.reset(tmp_prompt, pbar, num_step, logger)
                        sg_done, info = helper.step(tmp_prompt, tmp_sg_target)
                        env.can_open_inventory = False
                        env.can_change_hotbar = False
                else:
                    assert (
                        info is not None
                    ), "info should not be None! Because equip/craft/smelt failed!"
                    env.can_open_inventory = False
                    env.can_change_hotbar = False

                    if _malmo_log_has_fatal_error(env_malmo_logger_path, logger):
                        return "env_malmo_logger_error", None, None, None, None

                    fail_env_step = (env.num_steps >= int(cfg["env"]["max_minutes"])*MINUTE)
                    fail_monitor_step = (monitors.all_steps()  >= int(cfg["env"]["max_minutes"])*MINUTE)

                    if fail_env_step or fail_monitor_step:
                        game_over = True
                        status = "timeout_programmatic"
                        failed_subgoals = [current_sg]
                        failed_waypoints.append(waypoint)
                        break

                    if 'error_msg' not in info:
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue
                    if not ("cannot find a recipe" in info['error_msg'] or "missing material" in info['error_msg']):
                        logger.warning(f'fail for unkown reason. info: {info}')
                        continue

                    failed_waypoints.append(waypoint)
                    action_memory.save_success_failure(
                        waypoint,
                        language_action_str,
                        is_success=False,
                        outcome_status="failed",
                        env_status=env.get_status(),
                    )
                    subgoal = None

                    # NOTE: if a same waypoint is failed multiple times, then end this episode
                    # MineRL environment is not stable, so sometimes it fails to craft item even if it has enough materials
                    if failed_waypoints.count(waypoint) >= 3:
                        status = "failed"
                        failed_subgoals = [current_sg]
                        break

                    continue
            else:
                # op is not in ["craft", "smelt", "equip"]
                step_waypoint_obtained = env.num_steps
                current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                _ensure_best_pickaxe_equipped(
                    env,
                    helper,
                    current_sg_prompt,
                    current_sg_target,
                    pbar,
                    num_step,
                    logger,
                )
                _maybe_pillar_up_for_ore(
                    env,
                    current_sg_prompt,
                    current_sg_target,
                    logger,
                )
                tree_chop_active = _is_tree_chop_subgoal(current_sg_prompt, current_sg_target)
                tree_log_activity = _log_activity_count(env.get_status()) if tree_chop_active else 0
                tree_last_activity_step = env.num_steps
                tree_mode = "chop"
                tree_explore_prompt = os.environ.get("XENON_TREE_EXPLORE_PROMPT", "find a tree")
                tree_chop_stale_ticks = int(os.environ.get("XENON_TREE_CHOP_STALE_TICKS", "360"))
                tree_explore_ticks = int(os.environ.get("XENON_TREE_EXPLORE_TICKS", "420"))
                tree_contact_attack_ticks = int(os.environ.get("XENON_TREE_CONTACT_ATTACK_TICKS", "16"))
                mining_direction_active = _is_layered_mining_subgoal(current_sg_prompt, current_sg_target)
                # planner_target is the raw sub-goal item (e.g. "cobblestone");
                # mining_target_ore is the canonical ore the pillar-up
                # machinery references. They differ for "cobblestone", which
                # is remapped onto iron_ore so the overshoot Y band and the
                # deeper-ores-seen comparison both make sense.
                _planner_target = _normalise_ore_name(current_sg_target[0]) if mining_direction_active else ""
                mining_target_ore = _effective_mining_target_ore(_planner_target) if mining_direction_active else ""
                mining_required = _ore_required_count(current_sg_target)
                mining_initial_status = env.get_status()
                mining_activity = _ore_activity_count(mining_initial_status, mining_target_ore) if mining_direction_active else 0
                mining_available_at_start = _ore_available_count(
                    mining_initial_status, mining_target_ore
                ) if mining_direction_active else 0
                mining_last_activity_step = env.num_steps
                mining_mode = "dig_down"
                mining_forward_prompt = _forward_mining_prompt(mining_target_ore) if mining_direction_active else ""
                mining_deeper_counts = (
                    _deeper_ore_available_counts(mining_initial_status, mining_target_ore)
                    if mining_direction_active else {}
                )
                mining_pending_deeper_overshot: set[str] = set()
                # Track Y of first encounter with the *target* ore for this
                # subgoal. When pillar-up triggers we want to lift the agent
                # back to that Y rather than the band's mid-y, because we
                # already know that height has at least one target-ore
                # cluster within reach. Initialised lazily — set to None
                # until a target-ore inventory delta is observed.
                first_target_ore_y: float | None = None
                mining_activity_at_start = mining_activity
                # Cooldown lowered from the original 240 default to 120 ticks
                # (~6s) so the trigger can fire again sooner after the agent
                # finishes a pillar-up + brief dig-forward burst, in case it
                # has already overshot the band again.
                mining_switch_cooldown_ticks = int(os.environ.get("XENON_MINING_DIRECTION_SWITCH_COOLDOWN_TICKS", "120"))
                mining_last_switch_step = -1000000
                mining_failed_lateral_origin_xz: tuple[float, float] | None = None

                while True:
                    env._only_once = True

                    if env.consume_policy_reset_requested():
                        logger.info("Resetting action server after detected respawn/control-state reset.")
                        reset_thread = ServerAPI.reset(cfg["server"])
                        reset_thread.join()
                        current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                        step_waypoint_obtained = env.num_steps
                        if tree_chop_active:
                            tree_mode = "chop"
                            tree_log_activity = _log_activity_count(env.get_status())
                            tree_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                        if mining_direction_active:
                            mining_mode = "dig_down"
                            mining_activity = _ore_activity_count(env.get_status(), mining_target_ore)
                            mining_last_activity_step = env.num_steps
                            mining_last_switch_step = env.num_steps
                        logger.info(
                            "Waypoint-aware respawn recovery: restored STEVE-1 prompt "
                            f"to {current_sg_prompt} for waypoint {waypoint} at timestep {env.num_steps}."
                        )

                    action = ServerAPI.get_action(
                        cfg["server"], obs, current_sg_prompt, step=env.num_steps,
                        hydra_path=hydra_path, run_uuid=run_uuid
                    )
                    obs, reward, game_over, info = env.step(
                        action,
                        current_sg_target,
                        prompt=current_sg_prompt,
                    )
                    pbar.update(num_step, advance=1)
                    monitors.update(f"{temp_sg_prompt}_{progress}", env.current_subgoal_finish)

                    if tree_chop_active:
                        tree_status = env.get_status()
                        current_log_activity = _log_activity_count(tree_status)
                        control_state = tree_status.get("control_state") or {}
                        tree_contact_active = (
                            int(control_state.get("surface_attack_streak", 0)) >= tree_contact_attack_ticks
                            and env.num_steps - int(control_state.get("last_surface_attack_step", -1000000)) <= 2
                        )
                        if current_log_activity > tree_log_activity:
                            tree_log_activity = current_log_activity
                            tree_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                            if current_sg_prompt != temp_sg_prompt:
                                current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                                tree_mode = "chop"
                                logger.info(
                                    "Tree acquisition has started; switching STEVE-1 prompt back to "
                                    f"{current_sg_prompt} at timestep {env.num_steps}."
                                )
                        elif tree_contact_active:
                            tree_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                            if current_sg_prompt != temp_sg_prompt:
                                current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                                tree_mode = "chop"
                                logger.info(
                                    "Sustained tree-chopping contact detected; switching STEVE-1 prompt back to "
                                    f"{current_sg_prompt} at timestep {env.num_steps}."
                                )
                        elif (
                            os.environ.get("XENON_ENABLE_TREE_EXPLORE", "1") == "1"
                            and tree_mode == "chop"
                            and current_sg_prompt == temp_sg_prompt
                            and env.num_steps - tree_last_activity_step >= tree_chop_stale_ticks
                        ):
                            # Master gate: managed by PerceptionActionSuite
                            # (default ON). When OFF the agent stays on the
                            # original chop prompt and never temporarily
                            # diverts to "find a tree".
                            current_sg_prompt = tree_explore_prompt
                            tree_mode = "explore"
                            tree_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                            logger.info(
                                "No log-related progress from chop prompt; temporarily switching STEVE-1 "
                                f"prompt to {current_sg_prompt} at timestep {env.num_steps}."
                            )
                        elif (
                            tree_mode == "explore"
                            and env.num_steps - tree_last_activity_step >= tree_explore_ticks
                        ):
                            current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                            tree_mode = "chop"
                            tree_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                            logger.info(
                                "Tree exploration window ended; probing original STEVE-1 prompt "
                                f"{current_sg_prompt} at timestep {env.num_steps}."
                            )

                    if mining_direction_active:
                        env_status_now = env.get_status()
                        if not _has_capable_pickaxe_for_target(env_status_now, _planner_target):
                            logger.warning(
                                "Mining subgoal needs a better pickaxe; forcing replanning: "
                                f"target={_planner_target}, effective_ore={mining_target_ore}, "
                                f"inventory={env_status_now.get('inventory', {})}, "
                                f"timestep={env.num_steps}"
                            )
                            subgoal = None
                            current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                            step_waypoint_obtained = env.num_steps
                            break
                        current_activity = _ore_activity_count(env_status_now, mining_target_ore)
                        current_available = _ore_available_count(env_status_now, mining_target_ore)
                        current_available_delta = current_available - mining_available_at_start
                        deeper_counts = _deeper_ore_available_counts(env_status_now, mining_target_ore)
                        new_deeper_seen = [
                            ore
                            for ore, count in deeper_counts.items()
                            if count > mining_deeper_counts.get(ore, 0)
                        ]
                        if new_deeper_seen:
                            mining_pending_deeper_overshot.update(new_deeper_seen)
                        mining_deeper_counts = deeper_counts
                        if current_activity > mining_activity:
                            # Target-ore count just increased -> the agent
                            # successfully encountered a target-ore block at
                            # *this* Y. If we haven't yet recorded a
                            # first-target-ore Y for this subgoal, do so now.
                            # That Y becomes the destination for any
                            # subsequent pillar-up trigger ("回到我们第一次
                            # 挖到这个矿的高度重新探索").
                            if first_target_ore_y is None:
                                try:
                                    loc_now = env_status_now.get("location_stats") or {}
                                    ypos_now = loc_now.get("ypos", None)
                                    if ypos_now is not None:
                                        first_target_ore_y = float(
                                            np.asarray(ypos_now).reshape(-1)[0]
                                        )
                                        logger.info(
                                            f"[overshoot_relevel] recording first-target-ore Y for "
                                            f"{mining_target_ore}: y={first_target_ore_y:.1f}, "
                                            f"activity={current_activity}, "
                                            f"timestep={env.num_steps}"
                                        )
                                except Exception:
                                    first_target_ore_y = None
                            mining_activity = current_activity
                            mining_last_activity_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                        target_incomplete = current_available_delta < mining_required
                        # Trigger 1 (new deeper ore acquisition): only the
                        # moment a deeper ore count increases should request
                        # a lift. A deeper ore already seen earlier is not a
                        # persistent lower-bound restriction.
                        overshot_seen = sorted(
                            mining_pending_deeper_overshot,
                            key=lambda ore: ORE_LAYER_ORDER.get(ore, 999),
                        )
                        overshot_layer = len(overshot_seen) > 0

                        # Trigger 2 (optional Y-band guard): disabled by
                        # default because it treats low Y as a bottom limit
                        # even when no new deeper ore was mined. Enable only
                        # for ablations with XENON_OVERSHOOT_ENABLE_Y_TRIGGER=1.
                        y_overshoot = False
                        cur_y_value = None
                        cur_x_value = None
                        cur_z_value = None
                        target_band_min = None
                        try:
                            loc = env_status_now.get("location_stats") or {}
                            xpos_raw = loc.get("xpos", None)
                            ypos_raw = loc.get("ypos", None)
                            zpos_raw = loc.get("zpos", None)
                            if xpos_raw is not None:
                                cur_x_value = float(np.asarray(xpos_raw).reshape(-1)[0])
                            if ypos_raw is not None:
                                cur_y_value = float(np.asarray(ypos_raw).reshape(-1)[0])
                            if zpos_raw is not None:
                                cur_z_value = float(np.asarray(zpos_raw).reshape(-1)[0])
                            enable_y_trigger = os.environ.get(
                                "XENON_OVERSHOOT_ENABLE_Y_TRIGGER", "0"
                            ) == "1"
                            wrapper_ore = _PILLAR_PLANNER_TO_WRAPPER_ORE.get(mining_target_ore)
                            if enable_y_trigger and wrapper_ore is not None:
                                band = CustomEnvWrapper.ORE_HEIGHT_BANDS.get(wrapper_ore)
                                if band is not None and cur_y_value is not None:
                                    target_band_min = int(band[0])
                                    y_margin = float(os.environ.get("XENON_OVERSHOOT_Y_MARGIN", "5"))
                                    y_overshoot = cur_y_value < (target_band_min - y_margin)
                        except Exception:
                            y_overshoot = False

                        # Trigger 3 (bedrock-stuck): when the agent has dug
                        # below an absolute Y floor (default 8, just above
                        # bedrock at y=0-4) AND has had no mining activity
                        # for a long time, force a pillar-up. This catches
                        # the user's "挖到最坚硬矿石/基岩，一直挖不动" scenario
                        # where Trigger 2 cannot fire — for diamond targets
                        # the band_min is 1 so y < 1-5 = -4 is impossible.
                        # The absolute floor doesn't depend on the target
                        # band so it works for every layered subgoal.
                        bedrock_stuck = False
                        try:
                            absolute_floor_y = float(os.environ.get("XENON_BEDROCK_FLOOR_Y", "8.0"))
                            no_activity_required = int(os.environ.get("XENON_BEDROCK_STAGNANT_TICKS", "1200"))
                            if (
                                cur_y_value is not None
                                and cur_y_value <= absolute_floor_y
                                and (env.num_steps - mining_last_activity_step) >= no_activity_required
                            ):
                                bedrock_stuck = True
                        except Exception:
                            bedrock_stuck = False

                        should_pillar_up = overshot_layer or y_overshoot or bedrock_stuck
                        switch_ready = env.num_steps - mining_last_switch_step >= mining_switch_cooldown_ticks
                        forward_shift_ready = True
                        forward_shift_delta = None
                        if mining_mode == "dig_forward" and mining_failed_lateral_origin_xz is not None:
                            forward_shift_ready = False
                            if cur_x_value is not None and cur_z_value is not None:
                                dx = cur_x_value - mining_failed_lateral_origin_xz[0]
                                dz = cur_z_value - mining_failed_lateral_origin_xz[1]
                                forward_shift_delta = (dx * dx + dz * dz) ** 0.5
                                forward_shift_ready = (
                                    forward_shift_delta >= _env_float(
                                        "XENON_CORRIDOR_MIN_MOVE_DELTA",
                                        0.20,
                                    )
                                )
                        can_switch = (
                            mining_mode == "dig_down"
                            and current_sg_prompt == temp_sg_prompt
                            and switch_ready
                        )
                        can_relevel_forward = (
                            mining_mode == "dig_forward"
                            and switch_ready
                            and (overshot_layer or y_overshoot or bedrock_stuck)
                            and forward_shift_ready
                        )
                        if (can_switch or can_relevel_forward) and should_pillar_up:
                            # Pillar up to the Y where we first encountered
                            # the target ore, then require a real horizontal
                            # displacement before STEVE-1 may resume dig-down.
                            relevel_result = _maybe_relevel_for_overshoot(
                                env, mining_target_ore, overshot_seen, logger,
                                target_y=first_target_ore_y,
                            )
                            trigger_reason = []
                            if overshot_layer:
                                trigger_reason.append(f"deeper_ores={overshot_seen}")
                            if y_overshoot:
                                trigger_reason.append(
                                    f"y_overshoot(cur_y={cur_y_value:.1f}, "
                                    f"band_min={target_band_min}, "
                                    f"margin={os.environ.get('XENON_OVERSHOOT_Y_MARGIN', '5')})"
                                )
                            if bedrock_stuck:
                                trigger_reason.append(
                                    f"bedrock_stuck(cur_y={cur_y_value:.1f}, "
                                    f"no_activity_ticks={env.num_steps - mining_last_activity_step})"
                                )
                            lateral_summary = None
                            if isinstance(relevel_result, dict):
                                lateral_summary = relevel_result.get("lateral_shift")
                            lateral_ok = _lateral_shift_succeeded(relevel_result)
                            if lateral_ok:
                                current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                                mining_mode = "dig_down"
                                mining_last_switch_step = env.num_steps
                                mining_last_activity_step = env.num_steps
                                step_waypoint_obtained = env.num_steps
                                mining_failed_lateral_origin_xz = None
                                logger.info(
                                    "Mining shaft relocation: horizontal displacement succeeded; "
                                    "restoring STEVE-1 prompt as "
                                    f"{current_sg_prompt} for waypoint {waypoint}; "
                                    f"mode_before={'dig_forward' if can_relevel_forward else 'dig_down'}, "
                                    f"reason={'+'.join(trigger_reason) or 'unknown'}, "
                                    f"target={mining_target_ore} (planner_target={_planner_target}), "
                                    f"current={current_available}, start={mining_available_at_start}, "
                                    f"gained={current_available_delta}, required_delta={mining_required}, "
                                    f"relevel_success={relevel_result.get('success') if isinstance(relevel_result, dict) else None}, "
                                    f"lateral_shift={lateral_summary}, "
                                    f"timestep={env.num_steps}."
                                )
                            else:
                                # Do not resume dig-down at the old x/z. If the
                                # scripted relocation failed, hand STEVE-1 only
                                # a horizontal mining prompt. Another relevel
                                # attempt is allowed only after the agent has
                                # actually changed x/z from this failure point.
                                current_sg_prompt = mining_forward_prompt or copy.deepcopy(temp_sg_prompt)
                                mining_mode = "dig_forward" if mining_forward_prompt else "dig_down"
                                mining_last_switch_step = env.num_steps
                                if isinstance(lateral_summary, dict):
                                    try:
                                        mining_failed_lateral_origin_xz = (
                                            float(lateral_summary.get("end_x")),
                                            float(lateral_summary.get("end_z")),
                                        )
                                    except (TypeError, ValueError):
                                        mining_failed_lateral_origin_xz = None
                                elif cur_x_value is not None and cur_z_value is not None:
                                    mining_failed_lateral_origin_xz = (cur_x_value, cur_z_value)
                                else:
                                    mining_failed_lateral_origin_xz = None
                                logger.info(
                                    "Mining shaft relocation: horizontal displacement failed; "
                                    "NOT restoring dig-down prompt yet. "
                                    f"next_prompt={current_sg_prompt}, "
                                    f"mode_before={'dig_forward' if can_relevel_forward else 'dig_down'}, "
                                    f"reason={'+'.join(trigger_reason) or 'unknown'}, "
                                    f"target={mining_target_ore} (planner_target={_planner_target}), "
                                    f"current={current_available}, start={mining_available_at_start}, "
                                    f"gained={current_available_delta}, required_delta={mining_required}, "
                                    f"relevel_success={relevel_result.get('success') if isinstance(relevel_result, dict) else None}, "
                                    f"lateral_shift={lateral_summary}, "
                                    f"forward_retry_origin={mining_failed_lateral_origin_xz}, "
                                    f"forward_shift_delta={forward_shift_delta}, "
                                    f"timestep={env.num_steps}."
                                )
                            if overshot_layer:
                                mining_pending_deeper_overshot.difference_update(overshot_seen)
                        elif mining_mode == "dig_forward" and not target_incomplete:
                            current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                            mining_mode = "dig_down"
                            mining_last_switch_step = env.num_steps
                            step_waypoint_obtained = env.num_steps
                            logger.info(
                                "Mining target count is satisfied; "
                                f"restoring STEVE-1 prompt {current_sg_prompt} at timestep {env.num_steps}."
                            )


                    # if current waypoint item is not obtained over a MINUTE, then do get_context_aware_reasoning.
                    if env.inventory_change():
                        new_item_dict = env.inventory_change_what()
                        is_waypoint_obtained = check_waypoint_item_obtained(new_item_dict, waypoint, logger)
                        if is_waypoint_obtained:
                            step_waypoint_obtained = env.num_steps
                            current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                    if env.num_steps - step_waypoint_obtained >= MINUTE and not (
                        mining_direction_active and mining_mode == "dig_forward"
                    ):
                        current_sg_prompt = copy.deepcopy(temp_sg_prompt)
                        logger.info(f"Current timestep: {env.num_steps}. Calling get_context_aware_reasoning ...")
                        reasoning_dict, visual_description, render_error = call_reasoning_with_retry(
                            cfg, obs, temp_sg_prompt, waypoint, hydra_path, run_uuid, logger
                        )
                        if render_error is not None:
                            logger.error(f"Error message: {render_error}")
                            status = "cannot generate reasoning"
                            failed_subgoals = [f"achieve {waypoint}"]
                            break

                        logger.info(f"visual_description: {visual_description}")
                        logger.info(f"reasoning_dict: {str(reasoning_dict)}")
                        step_waypoint_obtained = env.num_steps

                        if reasoning_dict["need_intervention"]:
                            current_sg_prompt = reasoning_dict["task"]
                            logger.info(f"New prompt for STEVE-1: {current_sg_prompt}. timestep: {env.num_steps}\n\n")
                            num_reasoning_intervention += 1
                            # image_to_log = wandb.Image(obs["pov"], caption=f"Observation at step {env.num_steps}")
                            # wandb.log({
                            #     f"obs/{env.num_steps}": image_to_log,
                            #     "env_num_steps": env.num_steps,
                            #     "num_reasoning_intervention": num_reasoning_intervention,
                            # })

                    if env.num_steps % (MINUTE * 3) == 0:
                        if _malmo_log_has_fatal_error(env_malmo_logger_path, logger):
                            return "env_malmo_logger_error", None, None, None, None

                    if game_over:
                        if isinstance(info, dict) and info.get("error"):
                            logger.warning(f"[red]:warning: MineRL step error: {info.get('error')}[/red]")
                            status = "env_step_timeout"
                        else:
                            logger.warning("[red]:warning: Timeout![/red]")
                            status = "timeout_non_programmatic"
                        failed_subgoals = [current_sg]
                        failed_waypoints.append(waypoint)
                        break

                    if env.current_subgoal_finish:
                        # sg is achieved
                        logger.info(f"[green]{temp_sg_prompt} Success :smile: [/green]!")
                        progress += 1
                        steps = monitors.get_steps(temp_sg_prompt)
                        completed_subgoals.append(current_sg)
                        subgoal_done = True
                        break

            # current_sg is done
            if subgoal_done:
                env_status = env.get_status()
                inventory = env_status["inventory"]

                waypoint_success = env.check_waypoint_finish([waypoint, 1])

                action_memory.save_success_failure(
                    waypoint,
                    language_action_str,
                    is_success=waypoint_success,
                    outcome_status="success" if waypoint_success else "failed",
                    env_status=env_status,
                )
                if waypoint_success:
                    logger.info(f"[green]Achieved waypoint {waypoint}[/green]")
                    completed_waypoints.append(waypoint)
                else:
                    logger.info(f"[red]Subgoal is done, but failed to achieve waypoint {waypoint}[/red]")
                    failed_waypoints.append(waypoint)
                subgoal = None


        if _malmo_log_has_fatal_error(env_malmo_logger_path, logger):
            return "env_malmo_logger_error", None, None, None, None

        # end of while loop. game is done.
        if not original_final_goal_success:
            action_memory.save_success_failure(
                waypoint,
                language_action_str,
                is_success=False,
                outcome_status="failed",
                env_status=env.get_status(),
                create_if_missing=False,
            )

        if env.api_thread is not None and env.api_thread_is_alive():
            env.api_thread.join()

        # wandb.log({
        #     "env_num_steps": env.num_steps,
        #     "num_reasoning_intervention": num_reasoning_intervention,
        # })

    return status, monitors.all_steps(), completed_subgoals, failed_subgoals, failed_waypoints


@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(cfg: DictConfig):
    register_custom_env(cfg)

    logger = get_logger(__name__)

    # Cascade XENON_PERCEPTION_ACTION_SUITE into per-feature env vars so
    # the wrapper / planner gates downstream see consistent defaults.
    # Per-feature env vars exported by the user are NOT overwritten.
    PerceptionActionSuite.apply_from_env(logger)

    benchmark = ""
    if "wooden" in cfg["env"]["name"].lower():
        benchmark = "wooden"
    elif "redstone" in cfg["env"]["name"].lower():
        benchmark = "redstone"
    elif "armor" in cfg["env"]["name"].lower():
        benchmark = "armor"
    elif "stone" in cfg["env"]["name"].lower():
        benchmark = "stone"
    elif "iron" in cfg["env"]["name"].lower():
        benchmark = "iron"
    elif "golden" in cfg["env"]["name"].lower():
        benchmark = "golden"
    elif "diamond" in cfg["env"]["name"].lower():
        benchmark = "diamond"
    cfg["benchmark"] = benchmark

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # wandb.init(project="", entity="", config=OmegaConf.to_container(cfg, resolve=True), save_code=True)

    logger.info(f"main_ours_planning.py is executed.")

    logger.info(f"benchmark: {benchmark}")
    logger.info(f"cfg['benchmark']: {cfg['benchmark']}")

    is_fixed_memory = cfg["memory"]["is_fixed"]
    logger.info(f"is_fixed_memory: {is_fixed_memory}")
    if not is_fixed_memory: # if growing memory
        logger.info(f"Growing memory")
        # Save path = retrieve path
        cfg["memory"]["waypoint_to_sg"]["save_path"] = cfg["memory"]["waypoint_to_sg"]["path"]
    else:
        logger.info("Fixed memory. Only a few experiences are used, and the memory doesn't grow.")

    prefix = cfg.get("prefix")
    logger.info(f"prefix: {prefix}\n")

    env = env_make(cfg["env"]["name"], cfg, logger)

    action_memory = CaseBasedMemory(cfg, logger)

    if cfg["task"]["interactive"] and cfg["type"] != "headless":
        raise NotImplementedError("Not implemented yet!")

    running_tasks, running_goals = get_evaluate_task_and_goal(cfg)

    if len(running_tasks) == 0:
        logger.error("No tasks to evaluate.")
        # wandb.finish(exit_code=1)
        sys.exit(1)

    logger.info(f"Running Tasks: {running_tasks}")
    logger.info(OmegaConf.to_yaml(cfg))

    times = cfg["env"]["times"]
    for task, goal in zip(running_tasks, running_goals):
        monitors = []
        for run_t in range(times):
            try:
                ServerAPI._reset(cfg["server"])
                logger.info("[red]env & server reset...[/red] ")
                obs = env.reset()

            except Exception as e:
                logger.error(f"Error during reset: {e}")
                # wandb.finish(exit_code=1)
                sys.exit(1)

            logger.info("Done of reset of env and server")

            hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            run_uuid = shortuuid.uuid()
            logger.info(f"trial: {run_t}, goal: {goal}, hydra_path: {hydra_path}, run_uuid: {run_uuid}\n\n")

            visual_info = ""
            environment = cfg["env"]["prefer_biome"]

            logger.info("goal, environment. start")
            logger.info(f"goal: {goal}, environment: {environment}")
            logger.info("goal, environment. end")

            # wandb.config.update(
            #     {"task": task.replace(' ', '_').lower(), "goal": goal.replace(' ', '_').lower(),
            #      "hydra_path": hydra_path, "run_uuid": run_uuid, "benchmark": benchmark},
            #     allow_val_change=True
            # )

            action_memory.current_environment = environment

            current_monitos = Monitors([SuccessMonitor(), StepMonitor()])

            try:
                env.record_mod.step(obs, None, None)
            except Exception as e:
                logger.warning(f"Failed to record initial frame: {e}")

            # wandb.config.update({
            #     "is_fixed_memory": bool(is_fixed_memory),
            #     "biome": cfg["env"]["prefer_biome"],
            #     "prefix": prefix,
            # }, allow_val_change=True)

            completed_subgoals = []
            failed_subgoals = []
            failed_waypoints = []
            try:
                status, steps, completed_subgoals, failed_subgoals, failed_waypoints = new_agent_do(
                    cfg, env, logger, current_monitos, obs, action_memory, task, goal, run_uuid
                )
            except Exception as e:
                status = f"crash_{type(e).__name__}"
                steps = getattr(env, "num_steps", current_monitos.all_steps())
                completed_subgoals = []
                failed_subgoals = [{"task": task, "goal": goal, "error": str(e)}]
                failed_waypoints = []
                logger.error(f"Experiment crashed and will be recorded as failure: {e}")
                logger.error(traceback.format_exc())

            if status == "env_malmo_logger_error":
                logger.error("env_malmo_logger_error; recording episode as failure instead of exiting")

            steps = steps if steps is not None else getattr(env, "num_steps", current_monitos.all_steps())
            completed_subgoals = completed_subgoals or []
            failed_subgoals = failed_subgoals or []
            failed_waypoints = failed_waypoints or []

            failed_waypoints = list(set(failed_waypoints))
            failed_waypoints.sort()

            done_final_task = _video_action_name(status, completed_subgoals, failed_subgoals, task)
            biome = cfg["env"]["prefer_biome"]

            status_detailed = copy.deepcopy(status)
            early_log_start_failure = int(steps or 0) < 300 and failed_waypoints == ["logs"]
            infra_early_stop = (
                (status_detailed == "env_step_timeout" and int(steps or 0) < 300)
                or early_log_start_failure
            )
            status = "failed" if status != "success" else status
            if status != "success":
                if infra_early_stop:
                    removed_cases = action_memory.discard_pending_cases(run_uuid)
                    logger.warning(
                        "Discarded pending cases from infrastructure early stop: "
                        f"run_uuid={run_uuid}, removed_cases={removed_cases}, steps={steps}"
                    )
                else:
                    action_memory.mark_pending_cases_failed(
                        run_uuid,
                        reason=status_detailed or "failed",
                        env_status=env.get_status(),
                    )

            video_file = env.save_video(_video_task_name(benchmark, task), status, is_sub_task=False,
                                        actual_done_final_task=done_final_task, biome=biome, run_uuid=run_uuid)

            video_path = ""
            if video_file is not None:
                video_file.join()
                video_path = video_file.get_result()
                if not video_path:
                    video_path = ""

            current_planning = completed_subgoals + failed_subgoals
            final_env_status = env.get_status()
            t = action_memory.save_plan(
                task,
                visual_info,
                goal,
                status,
                current_planning,
                steps,
                run_uuid,
                video_path,
                environment=environment,
            )

            monitors.append(current_monitos)

            logger.info(f"completed_subgoals: {str(completed_subgoals)}\n")
            logger.info(f"failed_subgoals: {str(failed_subgoals)}\n")
            logger.info(f"Summary: {current_monitos.get_metric()}")

            result_file_name = f"{prefix}_{task.replace(' ', '_').lower()}_{cfg['exp_num']:003}_{status}_{biome}_{run_uuid[:4]}.json"
            result_data = {
                "run_uuid": run_uuid,
                "seed": seed,
                "prefix": prefix,
                "benchmark": benchmark,
                "task": task.replace(' ', '_').lower(),
                "goal": goal.replace(' ', '_').lower(),
                "exp_num": cfg["exp_num"],
                "biome": biome,
                "is_fixed_memory": bool(is_fixed_memory),
                "max_minutes": cfg["env"]["max_minutes"],
                "success": bool(status=="success"),
                "status_detailed": status_detailed,
                "infra_early_stop": infra_early_stop,
                "video_file": video_path,
                "steps": steps,
                "minutes": round(steps / MINUTE, 2),
                "metrics": current_monitos.get_metric(),
                "completed_subgoals": completed_subgoals,
                "completed_plans": completed_subgoals, # backward compatibility
                "failed_subgoals": failed_subgoals,
                "remain_plans": failed_subgoals, # backward compatibility
                "all_subgoals": current_planning,
                "all_plans": current_planning, # backward compatibility
                "failed_waypoints": failed_waypoints,
                "recovery_events": final_env_status.get("recovery_events", {}),
                "resource_ledger": final_env_status.get("resource_ledger", {}),
                "inventory_slots_used": final_env_status.get("inventory_slots_used"),
            }

            result_file_path = os.path.join(hydra_path, result_file_name)
            with open(result_file_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            # with open(f"{wandb.run.dir}/result.json", "w") as f:
            #     json.dump(result_data, f, indent=2)
            #     wandb.save(f"result.json")

            # wandb.log({
            #     "success": int(bool(status=="success")),
            #     "total_steps": steps,
            #     "total_failed_waypoints": len(failed_waypoints),
            #     "total_minutes": round(steps / MINUTE, 2),
            # })

            os.makedirs(cfg["results"]["path"], exist_ok=True)
            result_file_path = os.path.join(cfg["results"]["path"], result_file_name)
            with open(result_file_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            pretty_result(
                task, current_monitos.get_metric(), 1, steps=current_monitos.all_steps()
            )

            t.join()
            logger.info(f"Done of trial: {run_t}, task: {task}, hydra_path: {hydra_path}, run_uuid: {run_uuid}")

            img_dir = os.path.join(hydra_path, run_uuid, "imgs")
            shutil.rmtree(img_dir)

            # wandb.finish()

        env.close()
        all_steps = 0
        for monitor in monitors:
            logger.info(monitor.get_metric())
            all_steps += monitor.all_steps()
        logger.info(f" All Steps: {all_steps}")
    exit(0)


if __name__ == "__main__":
    main()
