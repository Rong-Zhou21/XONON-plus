import copy
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

import fcntl
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from ..util.prompt import language_action_to_subgoal, render_subgoal
from ..util.thread import MultiThreadServerAPI


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _normalise_waypoint(waypoint: str) -> str:
    waypoint = str(waypoint).lower().replace(" ", "_")
    if waypoint == "logs":
        return "log"
    return waypoint


class CaseBasedMemory:
    """Case-based replacement for waypoint action memory.

    The class intentionally keeps the small subset of DecomposedMemory's public
    API that main_planning depends on, while storing richer decision cases.
    """

    current_environment: str = ""
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        cfg: DictConfig,
        logger: logging.Logger | None = None,
        life_long_learning: bool = False,
    ) -> None:
        self.cfg = cfg
        self.version = self.cfg["version"]
        self.logger = logger
        self.plan_failure_threshold = int(cfg["memory"]["plan_failure_threshold"])

        self.root_path = self.cfg["memory"]["path"]
        os.makedirs(self.root_path, exist_ok=True)

        self.case_dir_path = os.path.join(self.root_path, "case_memory")
        os.makedirs(self.case_dir_path, exist_ok=True)
        self.case_file_path = os.path.join(self.case_dir_path, "cases.json")
        self.bootstrap_marker_path = os.path.join(self.case_dir_path, "legacy_bootstrap.done")

        self.plan_dir_save_path = os.path.join(
            self.root_path, self.cfg["memory"]["decomposed_plan"]["save_path"]
        )
        os.makedirs(self.plan_dir_save_path, exist_ok=True)

        case_cfg = cfg["memory"].get("case_memory", {})
        self.reuse_threshold = float(case_cfg.get("reuse_threshold", 0.72))
        self.retrieve_threshold = float(case_cfg.get("retrieve_threshold", 0.45))
        self.bootstrap_legacy = bool(case_cfg.get("bootstrap_legacy", True))

        self.bert_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        if self.bootstrap_legacy:
            self._bootstrap_from_legacy_waypoint_memory()
        self.cases: List[Dict[str, Any]] = self._load_cases()
        self._case_embeddings = self._encode_cases(self.cases)
        self._pending_decisions: Dict[tuple[str, str], List[str]] = {}
        self._rebuild_pending_index()

    def _load_cases(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.case_file_path):
            return []
        with open(self.case_file_path, "r") as fp:
            fcntl.flock(fp, fcntl.LOCK_SH)
            try:
                data = json.load(fp)
            finally:
                fcntl.flock(fp, fcntl.LOCK_UN)
        return data.get("cases", []) if isinstance(data, dict) else []

    def _rewrite_cases(self) -> None:
        with self._lock, open(self.case_file_path, "a+") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            fp.seek(0)
            fp.truncate()
            json.dump({"cases": self.cases}, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)

    def _encode_text(self, text: str) -> torch.Tensor:
        embedding = torch.tensor(self.bert_encoder.encode(text)).float().unsqueeze(0).to(DEVICE)
        return F.normalize(embedding, p=2, dim=1)

    def _encode_cases(self, cases: List[Dict[str, Any]]) -> torch.Tensor | None:
        texts = [case.get("similarity_text", "") for case in cases]
        if not texts:
            return None
        embeddings = torch.tensor(self.bert_encoder.encode(texts)).float().to(DEVICE)
        return F.normalize(embeddings, p=2, dim=1)

    def _refresh_embeddings(self) -> None:
        self._case_embeddings = self._encode_cases(self.cases)

    def _rebuild_pending_index(self) -> None:
        self._pending_decisions.clear()
        for case in self.cases:
            outcome = case.get("outcome", {})
            if outcome.get("status") != "pending":
                continue
            key = (_normalise_waypoint(case.get("waypoint", "")), case.get("selected_action", ""))
            self._pending_decisions.setdefault(key, []).append(case["id"])

    def create_state_snapshot(
        self,
        env_status: Dict[str, Any] | None = None,
        obs: Dict[str, Any] | None = None,
        cfg: DictConfig | None = None,
    ) -> Dict[str, Any]:
        env_status = _jsonable(env_status or {})
        location_stats = env_status.get("location_stats", {})
        compact_location = {}
        for key in ["xpos", "ypos", "zpos", "pitch", "yaw", "biome_id"]:
            if key in location_stats:
                compact_location[key] = location_stats[key]

        obs_summary = {}
        if isinstance(obs, dict):
            for key in ["compassAngle", "isGuiOpen"]:
                if key in obs:
                    obs_summary[key] = _jsonable(obs[key])

        biome = ""
        if cfg is not None:
            biome = str(cfg.get("env", {}).get("prefer_biome", ""))

        return {
            "inventory": env_status.get("inventory", {}),
            "equipment": env_status.get("equipment", "none"),
            "location_stats": compact_location,
            "plain_inventory": env_status.get("plain_inventory", {}),
            "biome": biome or self.current_environment,
            "obs_summary": obs_summary,
        }

    def similarity_text(self, waypoint: str, state_snapshot: Dict[str, Any]) -> str:
        inventory = state_snapshot.get("inventory", {})
        inventory_text = ", ".join(f"{k}:{inventory[k]}" for k in sorted(inventory)) or "empty"
        location = state_snapshot.get("location_stats", {})
        location_text = ", ".join(f"{k}:{location[k]}" for k in sorted(location)) or "unknown"
        return (
            f"waypoint: {_normalise_waypoint(waypoint)}; "
            f"inventory: {inventory_text}; "
            f"equipment: {state_snapshot.get('equipment', 'none')}; "
            f"location: {location_text}; "
            f"biome: {state_snapshot.get('biome', '')}"
        )

    def _successful_case_indices(self) -> List[int]:
        return [
            idx
            for idx, case in enumerate(self.cases)
            if case.get("outcome", {}).get("success") is True and case.get("selected_action")
        ]

    def _retrieve_cases(
        self,
        waypoint: str,
        state_snapshot: Dict[str, Any],
        topK: int = 3,
        successful_only: bool = True,
    ) -> List[Dict[str, Any]]:
        if self._case_embeddings is None:
            return []
        indices = self._successful_case_indices() if successful_only else list(range(len(self.cases)))
        if not indices:
            return []

        query_embedding = self._encode_text(self.similarity_text(waypoint, state_snapshot))
        candidate_embeddings = self._case_embeddings[indices]
        similarities = torch.matmul(candidate_embeddings, query_embedding.T).squeeze(1)
        k = min(topK, len(indices))
        scores, local_indices = torch.topk(similarities, k)

        retrieved = []
        for score, local_idx in zip(scores.tolist(), local_indices.tolist()):
            case_idx = indices[local_idx]
            case = copy.deepcopy(self.cases[case_idx])
            case["_score"] = float(score)
            retrieved.append(case)
        return retrieved

    def _bootstrap_from_legacy_waypoint_memory(self) -> None:
        """Convert copied XENON waypoint memory into decision cases once.

        This preserves the original action-library capability while making the
        runtime depend on case memory after initialization. It does not create
        new Minecraft knowledge; it only migrates existing waypoint/action
        success and failure counts into the richer case schema.
        """
        if os.path.exists(self.bootstrap_marker_path):
            return

        waypoint_dir = os.path.join(
            self.root_path, self.cfg["memory"]["waypoint_to_sg"]["path"]
        )
        if not os.path.isdir(waypoint_dir):
            return

        cases = self._load_cases()
        bootstrapped = 0
        for file_name in sorted(os.listdir(waypoint_dir)):
            if not file_name.endswith(".json"):
                continue
            waypoint = file_name.replace(".json", "")
            path = os.path.join(waypoint_dir, file_name)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            for action, history in data.get("action", {}).items():
                success_count = int(history.get("success", 0))
                failure_count = int(history.get("failure", 0))
                subgoal, subgoal_str = language_action_to_subgoal(action, waypoint)
                state_snapshot = {
                    "inventory": {},
                    "equipment": "unknown",
                    "location_stats": {},
                    "plain_inventory": {},
                    "biome": "legacy",
                    "obs_summary": {},
                    "source": "legacy_waypoint_to_sg",
                }
                cases.append(
                    {
                        "id": f"legacy:{waypoint}:{bootstrapped:06d}",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "run_uuid": "legacy",
                        "original_final_goal": "unknown",
                        "environment": "legacy",
                        "waypoint": waypoint,
                        "waypoint_num": 1,
                        "state_snapshot": state_snapshot,
                        "similarity_text": self.similarity_text(waypoint, state_snapshot),
                        "candidate_actions": [
                            {
                                "action": action,
                                "source": "legacy_waypoint_to_sg",
                            }
                        ],
                        "selected_action": action,
                        "selected_subgoal": subgoal,
                        "selected_subgoal_str": subgoal_str,
                        "decision_trace": {
                            "source": "legacy_bootstrap",
                            "success_count": success_count,
                            "failure_count": failure_count,
                        },
                        "outcome": {
                            "status": "success" if success_count > 0 else "failed",
                            "success": success_count > 0,
                            "success_count": success_count,
                            "failure_count": failure_count,
                        },
                    }
                )
                bootstrapped += 1

        with self._lock, open(self.case_file_path, "w") as fp:
            fcntl.flock(fp, fcntl.LOCK_EX)
            json.dump({"cases": cases}, fp, indent=2)
            fcntl.flock(fp, fcntl.LOCK_UN)
        with open(self.bootstrap_marker_path, "w") as fp:
            fp.write(f"bootstrapped={bootstrapped}\n")
        if self.logger:
            self.logger.info(f"Bootstrapped {bootstrapped} legacy waypoint cases")

    def select_case_decision(
        self,
        waypoint: str,
        wp_num: int,
        state_snapshot: Dict[str, Any],
        topK: int,
        run_uuid: str,
        original_final_goal: str,
    ) -> Dict[str, Any] | None:
        exact = self._best_exact_success_case(waypoint)
        if exact is not None:
            return self._decision_from_case(
                exact,
                waypoint,
                wp_num,
                state_snapshot,
                run_uuid,
                original_final_goal,
                {
                    "source": "case_memory_exact_waypoint",
                    "confidence": 1.0,
                    "selected_case_id": exact["id"],
                    "reason": "exact successful waypoint case",
                },
            )

        retrieved = self._retrieve_cases(waypoint, state_snapshot, topK=topK, successful_only=True)
        exact_waypoint = [
            case
            for case in retrieved
            if _normalise_waypoint(case.get("waypoint", "")) == _normalise_waypoint(waypoint)
        ]
        best = exact_waypoint[0] if exact_waypoint else None
        if best is None or best["_score"] < self.reuse_threshold:
            return None

        return self._decision_from_case(
            best,
            waypoint,
            wp_num,
            state_snapshot,
            run_uuid,
            original_final_goal,
            {
                "source": "case_memory",
                "confidence": round(float(best["_score"]), 6),
                "reuse_threshold": self.reuse_threshold,
                "selected_case_id": best["id"],
                "retrieved_cases": self._summarise_retrieved_cases(retrieved),
            },
        )

    def _best_exact_success_case(self, waypoint: str) -> Dict[str, Any] | None:
        action_scores: Dict[str, int] = {}
        action_success_counts: Dict[str, int] = {}
        representatives: Dict[str, Dict[str, Any]] = {}
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) != _normalise_waypoint(waypoint):
                continue
            action = case.get("selected_action")
            if not action:
                continue
            outcome = case.get("outcome", {})
            success_count = int(outcome.get("success_count", 1))
            failure_count = int(outcome.get("failure_count", 0))
            if outcome.get("success") is True:
                action_scores[action] = action_scores.get(action, 0) + success_count - failure_count
                action_success_counts[action] = action_success_counts.get(action, 0) + success_count
                representatives.setdefault(action, case)
            elif outcome.get("success") is False:
                action_scores[action] = action_scores.get(action, 0) - max(1, failure_count)

        candidates = [
            (score, action_success_counts.get(action, 0), representatives[action])
            for action, score in action_scores.items()
            if action in representatives and score > -self.plan_failure_threshold
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0][2]

    def _decision_from_case(
        self,
        case: Dict[str, Any],
        waypoint: str,
        wp_num: int,
        state_snapshot: Dict[str, Any],
        run_uuid: str,
        original_final_goal: str,
        trace: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        best = case
        subgoal_str = best.get("selected_subgoal_str") or json.dumps(best.get("selected_subgoal", {}))
        subgoal, language_action_str, render_error = render_subgoal(copy.deepcopy(subgoal_str), wp_num)
        if render_error is not None:
            subgoal, subgoal_str = language_action_to_subgoal(best["selected_action"], waypoint)
            subgoal["goal"][1] = wp_num
            language_action_str = subgoal["task"]
            subgoal_str = json.dumps(subgoal)

        self.record_decision(
            waypoint=waypoint,
            waypoint_num=wp_num,
            state_snapshot=state_snapshot,
            candidate_actions=[
                {
                    "action": language_action_str,
                    "source": trace.get("source", "case_memory"),
                }
            ],
            selected_action=language_action_str,
            selected_subgoal=subgoal,
            selected_subgoal_str=subgoal_str,
            decision_trace=trace,
            run_uuid=run_uuid,
            original_final_goal=original_final_goal,
        )

        return {
            "subgoal": subgoal,
            "language_action_str": language_action_str,
            "decision_trace": trace,
        }

    def _summarise_retrieved_cases(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for case in retrieved:
            summary.append(
                {
                    "id": case.get("id"),
                    "waypoint": case.get("waypoint"),
                    "selected_action": case.get("selected_action"),
                    "score": round(float(case.get("_score", 0.0)), 6),
                    "success": case.get("outcome", {}).get("success"),
                }
            )
        return summary

    def record_decision(
        self,
        waypoint: str,
        waypoint_num: int,
        state_snapshot: Dict[str, Any],
        candidate_actions: List[Any],
        selected_action: str,
        selected_subgoal: Dict[str, Any],
        selected_subgoal_str: str,
        decision_trace: Dict[str, Any],
        run_uuid: str,
        original_final_goal: str,
    ) -> str:
        case_id = f"{run_uuid}:{len(self.cases):06d}:{int(time.time() * 1000)}"
        case = {
            "id": case_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_uuid": run_uuid,
            "original_final_goal": original_final_goal,
            "environment": self.current_environment,
            "waypoint": waypoint,
            "waypoint_num": waypoint_num,
            "state_snapshot": _jsonable(state_snapshot),
            "similarity_text": self.similarity_text(waypoint, state_snapshot),
            "candidate_actions": self._normalise_candidate_actions(candidate_actions),
            "selected_action": selected_action,
            "selected_subgoal": _jsonable(selected_subgoal),
            "selected_subgoal_str": selected_subgoal_str,
            "decision_trace": _jsonable(decision_trace),
            "outcome": {
                "status": "pending",
                "success": None,
            },
        }
        self.cases.append(case)
        self._rewrite_cases()
        self._refresh_embeddings()

        key = (_normalise_waypoint(waypoint), selected_action)
        self._pending_decisions.setdefault(key, []).append(case_id)
        return case_id

    def _normalise_candidate_actions(self, candidate_actions: List[Any]) -> List[Dict[str, Any]]:
        normalised = []
        for item in candidate_actions:
            if isinstance(item, dict):
                action = item.get("action")
                if not action:
                    continue
                normalised.append(
                    {
                        "action": action,
                        "source": item.get("source", "unknown"),
                    }
                )
            elif item:
                normalised.append({"action": str(item), "source": "unknown"})
        return normalised

    def _find_case(self, case_id: str) -> Dict[str, Any] | None:
        for case in self.cases:
            if case.get("id") == case_id:
                return case
        return None

    def save_success_failure(
        self,
        waypoint: str,
        action_str: str,
        is_success: bool,
        outcome_status: str | None = None,
        env_status: Dict[str, Any] | None = None,
        decision_trace_update: Dict[str, Any] | None = None,
        create_if_missing: bool = True,
    ):
        if not waypoint or not action_str:
            return

        key = (_normalise_waypoint(waypoint), action_str)
        pending_ids = self._pending_decisions.get(key, [])
        case = self._find_case(pending_ids.pop() if pending_ids else "") if pending_ids else None

        if case is None:
            if not create_if_missing:
                return
            state_snapshot = self.create_state_snapshot(env_status)
            subgoal, subgoal_str = language_action_to_subgoal(action_str, waypoint)
            case_id = self.record_decision(
                waypoint=waypoint,
                waypoint_num=1,
                state_snapshot=state_snapshot,
                candidate_actions=[action_str] if action_str else [],
                selected_action=action_str,
                selected_subgoal=subgoal,
                selected_subgoal_str=subgoal_str,
                decision_trace={"source": "legacy_save_success_failure"},
                run_uuid="unknown",
                original_final_goal="unknown",
            )
            case = self._find_case(case_id)

        if case is None:
            return

        case["outcome"] = {
            "status": outcome_status or ("success" if is_success else "failed"),
            "success": bool(is_success),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state_snapshot": _jsonable(self.create_state_snapshot(env_status)) if env_status is not None else {},
        }
        if decision_trace_update:
            case.setdefault("decision_trace", {}).update(_jsonable(decision_trace_update))

        self._rewrite_cases()
        self._refresh_embeddings()

    def mark_pending_cases_failed(
        self,
        run_uuid: str | None = None,
        reason: str = "failed_incomplete_run",
        env_status: Dict[str, Any] | None = None,
    ) -> int:
        updated = 0
        for case in self.cases:
            if case.get("outcome", {}).get("status") != "pending":
                continue
            if run_uuid is not None and case.get("run_uuid") != run_uuid:
                continue
            case["outcome"] = {
                "status": reason,
                "success": False,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "state_snapshot": _jsonable(self.create_state_snapshot(env_status)) if env_status is not None else {},
            }
            case.setdefault("decision_trace", {})["finalized_from_pending"] = True
            updated += 1

        if updated:
            self._rewrite_cases()
            self._refresh_embeddings()
            self._rebuild_pending_index()
        return updated

    def discard_pending_cases(self, run_uuid: str | None = None) -> int:
        original_count = len(self.cases)
        self.cases = [
            case
            for case in self.cases
            if not (
                (
                    run_uuid is not None
                    and case.get("run_uuid") == run_uuid
                )
                or (
                    run_uuid is None
                    and case.get("outcome", {}).get("status") == "pending"
                )
            )
        ]
        removed = original_count - len(self.cases)
        if removed:
            self._rewrite_cases()
            self._refresh_embeddings()
            self._rebuild_pending_index()
        return removed

    def is_succeeded_waypoint(self, waypoint: str):
        action_scores: Dict[str, int] = {}
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) != _normalise_waypoint(waypoint):
                continue
            action = case.get("selected_action", "")
            if not action:
                continue
            success = case.get("outcome", {}).get("success")
            action_scores.setdefault(action, 0)
            action_scores[action] += 1 if success is True else -1 if success is False else 0

        valid_actions = [
            (action, score)
            for action, score in action_scores.items()
            if score > -self.plan_failure_threshold
        ]
        if not valid_actions:
            return False, None
        action, _ = sorted(valid_actions, key=lambda item: item[1], reverse=True)[0]
        _, subgoal_str = language_action_to_subgoal(action, waypoint)
        return True, subgoal_str

    def retrieve_similar_succeeded_waypoints(
        self,
        waypoint: str,
        topK: int = 3,
        state_snapshot: Dict[str, Any] | None = None,
    ):
        state_snapshot = state_snapshot or self.create_state_snapshot()
        retrieved = self._retrieve_cases(waypoint, state_snapshot, topK=max(topK * 2, topK), successful_only=True)
        wp_sg_dict = {}
        for case in retrieved:
            if case["_score"] < self.retrieve_threshold:
                continue
            case_waypoint = case.get("waypoint")
            if not case_waypoint or case_waypoint in wp_sg_dict:
                continue
            wp_sg_dict[case_waypoint] = case.get("selected_subgoal_str") or json.dumps(case.get("selected_subgoal", {}))
            if len(wp_sg_dict) >= topK:
                break
        return wp_sg_dict

    def retrieve_failed_subgoals(self, waypoint: str):
        action_scores: Dict[str, int] = {}
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) != _normalise_waypoint(waypoint):
                continue
            action = case.get("selected_action", "")
            if not action:
                continue
            success = case.get("outcome", {}).get("success")
            action_scores.setdefault(action, 0)
            action_scores[action] += 1 if success is True else -1 if success is False else 0

        failed_subgoals = []
        for action, score in action_scores.items():
            if score <= -self.plan_failure_threshold:
                _, failed_subgoal_str = language_action_to_subgoal(action, waypoint)
                failed_subgoals.append(failed_subgoal_str)
        return failed_subgoals

    def retrieve_total_failed_counts(self, waypoint: str):
        total = 0
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) != _normalise_waypoint(waypoint):
                continue
            if case.get("outcome", {}).get("success") is False:
                total += 1
        return total

    def reset_success_failure_history(self, item_name: str):
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) == _normalise_waypoint(item_name):
                case["outcome"] = {"status": "reset", "success": None}
        self._rewrite_cases()
        self._refresh_embeddings()
        self._rebuild_pending_index()
        if self.logger:
            self.logger.info(f"Reset case outcomes for {item_name}")

    def save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        run_uuid: str,
        video_path: str = "",
        environment: str = "none",
    ):
        thread = MultiThreadServerAPI(
            self._save_plan,
            args=(
                task,
                visual_info,
                goal,
                status,
                planning,
                steps,
                run_uuid,
                video_path,
                environment,
            ),
        )
        thread.start()
        return thread

    def _save_plan(
        self,
        task: str,
        visual_info: str,
        goal: str,
        status: str,
        planning: List[Dict[str, Any]],
        steps: int | float,
        run_uuid: str,
        video_path: str = "",
        environment: str = "none",
    ):
        assert status in [
            "success",
            "failed",
        ], "status should be one of success, failed"
        file_name = self.cfg["memory"]["decomposed_plan"]["file"].replace(
            "<task>", task.replace(" ", "_")
        )

        memory_path = self.plan_dir_save_path.replace("<status>", status)
        os.makedirs(memory_path, exist_ok=True)

        memory_file = os.path.join(memory_path, file_name)

        if self.logger:
            self.logger.info(f"[hot_pink]store plan of {task} to {memory_file}[/hot_pink]")
        with self._lock:
            if os.path.exists(memory_file):
                with open(memory_file, "r") as fp:
                    memory = json.load(fp)
            else:
                memory = {"plan": []}

        with self._lock, open(memory_file, "w") as fp:
            memory["plan"].append(
                {
                    "id": run_uuid,
                    "environment": environment,
                    "visual_info": visual_info,
                    "goal": goal,
                    "video": video_path,
                    "planning": planning,
                    "status": status,
                    "steps": steps,
                }
            )
            json.dump(memory, fp, indent=2)
