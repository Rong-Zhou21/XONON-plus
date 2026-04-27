# Case-Based Memory Decisioner

## Goal

`main_planning.py` now uses `CaseBasedMemory` as the waypoint-to-action decision layer. The new memory replaces the previous direct `DecomposedMemory.is_succeeded_waypoint()` reuse path:

1. Build a deterministic state snapshot from `env.get_status()` and stable `obs` fields.
2. Retrieve successful cases with SentenceTransformer embeddings and cosine similarity.
3. Reuse a high-confidence exact-waypoint case.
4. Fall back to the original planner when no case passes the threshold.
5. Store every accepted decision as a pending case, then update the same case with success or failure outcome.

The original planner call, `render_subgoal` parsing, result JSON writing, and `save_plan` behavior remain compatible.

## Replacement Point

The replacement is in `src/optimus1/main_planning.py::make_plan`.

Previous behavior:

- Query `DecomposedMemory.is_succeeded_waypoint(wp)`.
- If any successful waypoint action existed, reuse it without considering current state.
- Otherwise call `ServerAPI.get_decomposed_plan(...)`.

New behavior:

- Build `state_snapshot = action_memory.create_state_snapshot(env_status, obs, cfg)`.
- Call `CaseBasedMemory.select_case_decision(...)`.
- If a same-waypoint successful case has similarity above `reuse_threshold`, reuse its stored subgoal via `render_subgoal`.
- Otherwise retrieve similar successful cases and failed subgoals from the case store, pass them to the existing planner, then record the planner decision.

## Case Schema

Cases are stored in:

`src/optimus1/memories/${prefix}/${version}/case_memory/cases.json`

Each case contains:

```json
{
  "id": "run_uuid:000000:timestamp_ms",
  "created_at": "UTC timestamp",
  "run_uuid": "episode id",
  "original_final_goal": "task goal",
  "environment": "configured biome/environment",
  "waypoint": "current waypoint",
  "waypoint_num": 1,
  "state_snapshot": {
    "inventory": {},
    "equipment": "none",
    "location_stats": {
      "xpos": 0,
      "ypos": 0,
      "zpos": 0,
      "pitch": 0,
      "yaw": 0,
      "biome_id": 0
    },
    "plain_inventory": {},
    "biome": "forest",
    "obs_summary": {}
  },
  "similarity_text": "waypoint + inventory + equipment + location + biome",
  "candidate_actions": ["craft item", "smelt item"],
  "selected_action": "craft item",
  "selected_subgoal": {"task": "craft item", "goal": ["item", 1]},
  "selected_subgoal_str": "{\"task\": \"craft item\", \"goal\": [\"item\", 1]}",
  "decision_trace": {
    "source": "case_memory | planner",
    "confidence": 0.91,
    "retrieved_cases": []
  },
  "outcome": {
    "status": "pending | success | failed",
    "success": true,
    "recorded_at": "UTC timestamp",
    "state_snapshot": {}
  }
}
```

The state snapshot deliberately avoids model-generated summaries. It uses deterministic inventory, equipment, compact location stats, configured biome, and selected stable observation fields.

## Retrieval Logic

`CaseBasedMemory` embeds `similarity_text` with `all-MiniLM-L6-v2` and computes cosine similarity with normalized vectors.

Reuse is conservative in the first version:

- Only successful cases are candidates.
- The selected case must have the same normalized waypoint (`logs` and `log` are treated as equivalent).
- The cosine score must be at least `memory.case_memory.reuse_threshold`, default `0.72`.

Planner examples use a looser successful-case retrieval:

- Successful cases above `memory.case_memory.retrieve_threshold`, default `0.45`, are converted into the existing `similar_wp_sg_dict` format.
- Failed exact-waypoint cases are aggregated by selected action; actions whose net score is below `plan_failure_threshold` are returned as failed subgoals.

## Training Interface Later

The case schema is intended to support both retrieval and supervised decisioner training:

- Input features: `state_snapshot`, `similarity_text`, `waypoint`, `candidate_actions`.
- Label: `selected_action`.
- Outcome target: `outcome.success`, plus optional status-specific filtering.
- Trace metadata: `decision_trace.source`, confidence, retrieved case ids, planner examples, failed subgoals.

A training decisioner can read `cases.json`, filter to resolved cases, and train a scorer or classifier over `(state, waypoint, candidate_action) -> expected success`. The runtime can then replace or rerank the embedding decision in `select_case_decision` without changing the main planning loop.

## Legacy Bootstrap

On first initialization, `CaseBasedMemory` migrates copied XENON
`waypoint_to_sg/*.json` files into `case_memory/cases.json` and writes
`legacy_bootstrap.done`. After this one-time conversion, runtime retrieval reads
from the case store rather than the old waypoint action files. This preserves
original action-library capability while replacing the storage and retrieval unit
with decision cases.

The case memory does not generate new Minecraft candidates with code rules.
Runtime `candidate_actions` records the action actually selected by the planner
or by a retrieved case, plus provenance.
