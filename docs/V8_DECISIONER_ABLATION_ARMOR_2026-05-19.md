# V8 Decisioner Ablation Armor Run

## Purpose

This batch compares the effect of the RADS decisioner.

- V7 Armor runs used the `run_v7_armor_targeted.sh` default:
  `DECISIONER_ENABLED=1`, which enables `memory.case_memory.decisioner.enabled=true`
  through Hydra overrides.
- V8 uses the same XENON-plus underground/action logic but explicitly disables
  the decisioner:
  `DECISIONER_ENABLED=0`, which sets
  `memory.case_memory.decisioner.enabled=false`.

The intended comparison is therefore:

- Plus V7: case memory + RADS decisioner + current perception/action modules.
- Plus V8: case memory reuse only + current perception/action modules.
- XENON-main V8 comparison: original XENON-main under the same Armor tasks,
  seed set, resource setup, and valid-result-file criterion.

## Shared Settings

- Tasks: Armor task ids `12 10 6`.
  - `12`: `golden_chestplate`
  - `10`: `golden_leggings`
  - `6`: `diamond_chestplate`
- Trials per task: `10`.
- Seeds/world seeds: `0..9`.
- World generation: `DefaultWorldGenerator(force_reset=True)` with no custom generator options.
- Dynamic ore behavior: original random ore placement with
  `XENON_RANDOM_ORE_GOLD_MULTIPLIER=1.5`.
- Scripted target ore / forced ore: disabled.
- Command relevel/craft fallback: disabled in Plus.
- Hostile mob influence: disabled via `/gamerule doMobSpawning false` and
  `/difficulty peaceful`.
- Planner backend: vLLM `Qwen/Qwen2.5-VL-7B-Instruct`.
- Valid completion criterion: each task must have `10` parseable JSON result
  files. Attempt count alone is not sufficient.

## Run State

Supervisor script:

```bash
/home/yzb/zhourong/XENON-plus/scripts/run_v8_no_decisioner_then_main_armor_compare.sh
```

Started:

- Time: `2026-05-19 11:28:05 CST`
- Supervisor PID: `127471`
- Supervisor log: `/tmp/xenon_v8_no_decisioner_compare_20260519_112805.log`
- Launcher log: `/tmp/xenon_v8_no_decisioner_then_main_launcher_20260519_112805.log`

Planner service:

- vLLM PID: `125895`
- vLLM URL: `http://127.0.0.1:8000/v1`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- GPU: `0`

Plus V8:

- Decisioner: disabled.
- GPU: `1`.
- Exp base: `880000`.
- Results: `/home/yzb/zhourong/XENON-plus/exp_results/v8`
- Videos: `/home/yzb/zhourong/XENON-plus/videos/v8`
- Round-1 summary: `/tmp/xenon_v7_v8_no_decisioner_round1_20260519_112805_summary.log`

XENON-main V8 comparison:

- Starts only after Plus V8 reaches 10 valid JSON files per Armor task.
- GPU: `1`.
- Exp base: `890000`.
- Results: `/home/yzb/zhourong/XENON-main/exp_results/v8_xenonmain`
- Videos: `/home/yzb/zhourong/XENON-main/videos/v8_xenonmain`

## Notes

An earlier launch at `2026-05-19 11:15:32 CST` was aborted because the planner
service was not ready. A second launch at `11:26:21 CST` was aborted because
vLLM and the Plus app both used GPU0, causing CUDA OOM while loading MineCLIP.
The active run fixes this by keeping vLLM on GPU0 and running Plus/Main on GPU1.

## V7 Decisioner Evidence

The existing V7 result directory contains parseable JSON records:

- Directory: `/home/yzb/zhourong/XENON-plus/exp_results/v7`
- JSON count: `32`
- Task distribution:
  - `craft_golden_chestplate`: `10`
  - `craft_golden_leggings`: `11`
  - `craft_diamond_chestplate`: `11`
- Successful JSONs: `3`

The extra two JSONs come from duplicate exp nums:

- `craft_golden_leggings`: duplicate `551001`
- `craft_diamond_chestplate`: duplicate `550609`

Using these V7 result JSON `run_uuid` values to query
`src/optimus1/memories/ours_planning/v1/case_memory/cases.json` gives:

- Matched decision cases: `492`
- `rads_decisioner`: `485`
- `planner`: `6`
- `semantic_fallback`: `1`

By task:

| Task | RADS | Planner | Semantic fallback |
| --- | ---: | ---: | ---: |
| `craft_golden_chestplate` | 148 | 2 | 1 |
| `craft_golden_leggings` | 185 | 3 | 0 |
| `craft_diamond_chestplate` | 152 | 1 | 0 |

This confirms V7 did not merely set the decisioner flag. It actually invoked
the RADS path and wrote `decision_trace.source = "rads_decisioner"` records.

RADS candidate scoring was also present:

- RADS decision cases: `485`
- Multi-candidate RADS cases: `28`
- Example waypoint: `planks`
  - `craft planks`: `p_success ~= 0.9998`
  - `dig down and mine planks`: `p_success ~= 0.99`
  - selected action: `craft planks`

Most RADS calls had only one historical candidate action for the waypoint. In
those cases the decisioner still scored the candidate, but there was no real
multi-action competition. The strongest evidence for action selection is the
`28` multi-candidate records.

Planner generation also occurred in V7 when no suitable RADS decision was
available. Recorded planner-generated actions include:

- `golden_chestplate -> craft golden_chestplate`
- `golden_leggings -> craft golden_leggings`
- `diamond_chestplate -> craft diamond_chestplate`
- `cobblestone -> dig down and mine cobblestone`
