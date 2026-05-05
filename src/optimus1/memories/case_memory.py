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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


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
        # On-disk cache of `_case_embeddings` keyed by a hash of the case
        # similarity_text list. Encoding all 4000+ cases via the BERT
        # encoder each task start costs ~15s (140 batches at ~10it/s);
        # the cache reduces that to a single torch.load + a hash check.
        self.case_embeddings_cache_path = os.path.join(
            self.case_dir_path, "cases_embeddings.pt"
        )

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

        # ----- decisioner (RADS) runtime hook -------------------------------
        # When `memory.case_memory.decisioner.enabled` is True, every call to
        # select_case_decision() is routed through the trained decisioner
        # instead of `_best_exact_success_case` + cosine retrieval. The model
        # is loaded lazily; failure to load is non-fatal and falls back to the
        # baseline path so a misconfigured run never deadlocks the agent.
        self.decisioner = None
        self.decisioner_min_p = 0.20
        self.decisioner_evidence_topk = 5
        decisioner_cfg = case_cfg.get("decisioner") or {}
        if bool(decisioner_cfg.get("enabled", False)):
            ckpt = str(decisioner_cfg.get("checkpoint", "artifacts/decisioner/rads_v2.pt"))
            self.decisioner_min_p = float(decisioner_cfg.get("min_p_success", 0.20))
            self.decisioner_evidence_topk = int(decisioner_cfg.get("log_evidence_topk", 5))
            requested_device = str(decisioner_cfg.get("device", "cuda"))
            device = requested_device if (
                requested_device == "cpu" or torch.cuda.is_available()
            ) else "cpu"
            try:
                from ..decisioner.runtime import RADSRuntime
                self.decisioner = RADSRuntime.load(ckpt, device=device)
                if self.logger:
                    self.logger.info(
                        f"[decisioner] enabled. checkpoint={ckpt} device={device} "
                        f"min_p_success={self.decisioner_min_p}"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[decisioner] load failed ({exc}); falling back to baseline path"
                    )
                self.decisioner = None

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

        # Disk-cached fast path. The cache is keyed by a SHA-256 of the
        # joined similarity_text list. If the hash matches, we reuse
        # the cached tensor and skip the ~15s SentenceTransformer pass
        # entirely. The cache file is shared across processes by
        # `flock`-guarded open since concurrent main_planning runs may
        # try to write at the same time.
        import hashlib
        joined = "\u241e".join(texts).encode("utf-8")  # \u241e = unit separator
        cur_hash = hashlib.sha256(joined).hexdigest()
        cache_path = getattr(self, "case_embeddings_cache_path", None)
        use_cache = cache_path is not None and os.environ.get("XENON_DISABLE_EMBEDDING_CACHE", "0") != "1"
        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fp:
                    fcntl.flock(fp, fcntl.LOCK_SH)
                    try:
                        cached = torch.load(fp, map_location=DEVICE)
                    finally:
                        fcntl.flock(fp, fcntl.LOCK_UN)
                if (
                    isinstance(cached, dict)
                    and cached.get("hash") == cur_hash
                    and isinstance(cached.get("embeddings"), torch.Tensor)
                    and cached["embeddings"].shape[0] == len(texts)
                ):
                    if self.logger:
                        self.logger.info(
                            f"[case_memory] reusing cached embeddings "
                            f"({len(texts)} cases, hash={cur_hash[:12]})"
                        )
                    return cached["embeddings"].to(DEVICE)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[case_memory] embedding cache read failed: {exc}; recomputing"
                    )

        embeddings = torch.tensor(
            self.bert_encoder.encode(texts, show_progress_bar=False)
        ).float().to(DEVICE)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if use_cache:
            try:
                tmp_path = cache_path + ".tmp"
                with open(tmp_path, "wb") as fp:
                    fcntl.flock(fp, fcntl.LOCK_EX)
                    try:
                        torch.save({"hash": cur_hash, "embeddings": embeddings.cpu()}, fp)
                    finally:
                        fcntl.flock(fp, fcntl.LOCK_UN)
                os.replace(tmp_path, cache_path)
                if self.logger:
                    self.logger.info(
                        f"[case_memory] wrote embedding cache "
                        f"({len(texts)} cases, hash={cur_hash[:12]})"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[case_memory] embedding cache write failed: {exc}"
                    )

        return embeddings

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

    def _case_duplicate_key(self, case: Dict[str, Any]) -> str:
        payload = copy.deepcopy(case)
        for key in ("id", "created_at", "run_uuid"):
            payload.pop(key, None)
        outcome = payload.get("outcome")
        if isinstance(outcome, dict):
            outcome.pop("recorded_at", None)
        return _canonical_json(payload)

    def _remove_exact_duplicate_case(self, case: Dict[str, Any]) -> str | None:
        outcome = case.get("outcome", {})
        if outcome.get("status") == "pending":
            return None

        duplicate_key = self._case_duplicate_key(case)
        for existing in self.cases:
            if existing is case:
                continue
            existing_outcome = existing.get("outcome", {})
            if existing_outcome.get("status") == "pending":
                continue
            if self._case_duplicate_key(existing) != duplicate_key:
                continue

            self.cases = [item for item in self.cases if item is not case]
            if self.logger:
                self.logger.info(
                    "Skipped exact duplicate case: "
                    f"case_id={case.get('id')}, duplicate_of={existing.get('id')}"
                )
            return existing.get("id")
        return None

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
        if self.decisioner is not None:
            decision = self._select_case_decision_rads(
                waypoint, wp_num, state_snapshot, run_uuid, original_final_goal
            )
            if decision is not None:
                return decision
            # On low-confidence or empty candidate set, fall through to None
            # so the upper layer can invoke planner.
            return None

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

    def _candidate_actions_with_reps(self, waypoint: str) -> Dict[str, Dict[str, Any]]:
        """Distinct (action -> representative case) for the given waypoint.

        Prefers a successful case as the representative because its
        selected_subgoal_str is what we want to reuse downstream. Falls back to
        any case carrying that action when no success exists yet.
        """
        by_action: Dict[str, Dict[str, Any]] = {}
        for case in self.cases:
            if _normalise_waypoint(case.get("waypoint", "")) != _normalise_waypoint(waypoint):
                continue
            action = case.get("selected_action")
            if not action:
                continue
            outcome = case.get("outcome", {}) or {}
            existing = by_action.get(action)
            existing_success = (
                existing is not None
                and (existing.get("outcome", {}) or {}).get("success") is True
            )
            this_success = outcome.get("success") is True
            if existing is None or (this_success and not existing_success):
                by_action[action] = case
        return by_action

    def _select_case_decision_rads(
        self,
        waypoint: str,
        wp_num: int,
        state_snapshot: Dict[str, Any],
        run_uuid: str,
        original_final_goal: str,
    ) -> Dict[str, Any] | None:
        candidates = self._candidate_actions_with_reps(waypoint)
        if not candidates:
            return None

        position_in_run = sum(1 for c in self.cases if c.get("run_uuid") == run_uuid)

        scored: List[tuple[str, Dict[str, Any], Any]] = []
        for action, rep_case in candidates.items():
            query_case = {
                "waypoint": waypoint,
                "waypoint_num": wp_num,
                "original_final_goal": original_final_goal,
                "selected_action": action,
                "state_snapshot": state_snapshot,
                "id": f"{run_uuid}:query",
                "run_uuid": run_uuid,
            }
            # The decisioner's feature extractor reads `_position_in_run` if
            # present; we patch it onto the dict for this scoring call only.
            query_case["_position_in_run"] = position_in_run
            try:
                result = self.decisioner.score(
                    query_case,
                    topk_evidence=self.decisioner_evidence_topk,
                    exclude_run_uuid=run_uuid,
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[decisioner] score failed for action={action!r}: {exc}"
                    )
                continue
            scored.append((action, rep_case, result))

        if not scored:
            return None

        scored.sort(key=lambda x: x[2].p_success, reverse=True)
        best_action, best_rep, best_result = scored[0]

        if self.logger:
            ranking = ", ".join(
                f"{a}={r.p_success:.3f}" for a, _, r in scored
            )
            self.logger.info(
                f"[decisioner] waypoint={waypoint} ranking=[{ranking}] "
                f"best={best_action!r} p={best_result.p_success:.3f} "
                f"conf={best_result.confidence:.3f}"
            )

        if best_result.p_success < self.decisioner_min_p:
            if self.logger:
                self.logger.info(
                    f"[decisioner] best p_success={best_result.p_success:.3f} "
                    f"< min={self.decisioner_min_p}; falling back to planner"
                )
            return None

        trace = {
            "source": "rads_decisioner",
            "p_success": float(best_result.p_success),
            "confidence": float(best_result.confidence),
            "attention_concentration": float(best_result.attention_concentration),
            "min_p_success": self.decisioner_min_p,
            "selected_case_id": best_rep.get("id"),
            "candidates": [
                {
                    "action": a,
                    "p_success": float(r.p_success),
                    "confidence": float(r.confidence),
                }
                for a, _, r in scored
            ],
            "evidence": [
                {
                    "case_id": e.case_id,
                    "waypoint": e.waypoint,
                    "selected_action": e.selected_action,
                    "success": e.success,
                    "attention": float(e.attention),
                }
                for e in best_result.evidence
            ],
        }
        return self._decision_from_case(
            best_rep,
            waypoint,
            wp_num,
            state_snapshot,
            run_uuid,
            original_final_goal,
            trace,
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

        duplicate_of = self._remove_exact_duplicate_case(case)
        self._rewrite_cases()
        self._refresh_embeddings()
        if duplicate_of is not None:
            self._rebuild_pending_index()

    def mark_pending_cases_failed(
        self,
        run_uuid: str | None = None,
        reason: str = "failed_incomplete_run",
        env_status: Dict[str, Any] | None = None,
    ) -> int:
        updated = 0
        updated_cases = []
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
            updated_cases.append(case)

        if updated:
            removed_duplicates = 0
            for case in updated_cases:
                if not any(existing is case for existing in self.cases):
                    continue
                if self._remove_exact_duplicate_case(case) is not None:
                    removed_duplicates += 1
            self._rewrite_cases()
            self._refresh_embeddings()
            self._rebuild_pending_index()
            if removed_duplicates and self.logger:
                self.logger.info(f"Skipped {removed_duplicates} exact duplicate failed cases.")
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
