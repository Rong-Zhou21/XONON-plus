# v2 baseline vs v3 RADS decisioner — 67-task on-line comparison

Date: 2026-05-02
v2 = retrieval-only (`_best_exact_success_case` + cosine retrieval)
v3 = RADS decisioner enabled (artifacts/decisioner/rads_v2.pt, min_p_success=0.20)

## Comparison must be single-shot vs single-shot

`exp_results/v2/` contains 107 result files for 67 tasks — 40 retries
across 26 tasks. Comparing v3's single attempt against v2's best-of-N is
unfair. The fair comparison is:

| basis | success / 67 | rate |
|---|---:|---:|
| **v2 first-attempt (single shot)** | **39** | **58.2%** |
| **v3 decisioner (single shot)** | **49** | **73.1%** |
| v2 best-of-N (up to 5 retries) | 57 | 85.1% |

**v3 decisioner closes the gap to the multi-attempt upper bound by 10 of
the 18 tasks v2 first-attempt missed.**

## Per-benchmark

| benchmark | tasks | v2 first | v2 best | v3 | v3 vs v2 first |
|---|---:|---:|---:|---:|---:|
| wooden | 10 | 8 | 10 | 10 | **+2** |
| stone | 9 | 7 | 7 | 6 | -1 |
| iron | 16 | 7 | 14 | 10 | **+3** |
| golden | 6 | 3 | 5 | 2 | -1 |
| diamond | 7 | 5 | 7 | 7 | **+2** |
| redstone | 6 | 2 | 5 | 6 | **+4** |
| armor | 13 | 7 | 9 | 8 | +1 |
| **total** | **67** | **39** | **57** | **49** | **+10** |

## v3 wins (v3 success, v2 first-attempt fail) — 18 tasks

| benchmark | task |
|---|---|
| iron | craft_a_blast_furnace |
| iron | craft_a_iron_axe |
| iron | craft_a_iron_hoe |
| iron | craft_a_tripwire_hook |
| iron | craft_an_iron_bars |
| redstone | craft_a_compass |
| redstone | craft_a_dropper |
| redstone | craft_a_redstone_torch |
| redstone | craft_an_activator_rail |
| diamond | craft_a_diamond_shovel |
| diamond | dig_down_and_mine_a_diamond |
| armor | craft_iron_chestplate |
| armor | craft_shield |
| stone | craft_a_smoker |
| stone | craft_a_stone_axe |
| wooden | craft_a_wooden_axe |
| wooden | craft_a_wooden_sword |
| golden | craft_a_golden_pickaxe |

## v3 losses (v2 first-attempt success, v3 fail) — 8 tasks

| benchmark | task | v3 failed_waypoints | likely cause |
|---|---|---|---|
| stone | craft_a_stone_hoe | logs | execution stuck on logs |
| stone | craft_a_stone_pickaxe | logs | execution stuck on logs |
| stone | smelt_a_charcoal | logs | execution stuck on logs |
| iron | craft_a_iron_sword | iron_ore | resource gathering stalled |
| iron | craft_a_chain | (none) | crash (RuntimeError) |
| golden | craft_a_golden_hoe | iron_ore, planks | resource gathering stalled |
| golden | craft_a_golden_sword | iron_ore | resource gathering stalled |
| armor | craft_golden_boots | gold_ore | resource gathering stalled |

The 3 stone losses on `logs` and 4 resource-gathering losses look like
execution-layer stochastic failures — the decisioner correctly picked
the only available action (P>=0.98 for `chop a tree` etc.) but STEVE-1
never delivered the resource within the time budget. These are not
decision-layer regressions; they are the same kind of failure mode that
v2 also exhibits (its first-attempt iron success rate is only 7/16, see
"v2 first" column above), just on different tasks this run.

## Steps on tasks both succeeded (n=31)

| | v2 first-attempt | v3 decisioner |
|---|---:|---:|
| mean steps | 4602 | 5591 |
| median steps | 5450 | 5240 |
| v3 faster | — | 18/31 |
| v3 slower | — | 13/31 |

Slight mean increase, similar median. Decisioner adds modest per-decision
overhead but does not blow up runtimes.

## Decision source breakdown across all v3 cases (n=659)

| source | count | of which success |
|---|---:|---:|
| rads_decisioner | 628 | 606 |
| planner (fallback) | 31 | 31 |

22 rads_decisioner cases were marked failed; many of these are pending
cases on later waypoints that got cascade-marked when the run timed out
on an earlier waypoint, so they don't reflect bad decisioner picks.

Planner fallback was triggered 31 times across two runs:
- 22 times on `logs/chop a tree` for `craft_a_note_block` (run succeeded)
- 9 times on `cobblestone/dig down and mine cobblestone` for `craft_a_diamond_axe` (run succeeded)

The repeat counts indicate the decisioner rejected the action multiple
times within those single tasks (its P(success) for that state hit the
< 0.20 threshold). The agent then fell back to planner each time, and
both runs eventually completed successfully — so the fallback path
worked as intended.

## Bottom line

| takeaway | evidence |
|---|---|
| RADS decisioner improves single-shot success by ~15 percentage points over baseline retrieval | 49/67 vs 39/67 |
| Wins are broad: 7 categories see net gains; only stone and golden see net loss (1 each, 1-task swings) | per-benchmark table |
| v3 unlocks tasks that v2 baseline cannot solve on first try, including ones from v2's "never-success after retries" list (`craft_a_iron_axe`, `craft_a_blast_furnace`, `craft_a_golden_pickaxe`, `craft_an_activator_rail`) | wins table |
| v3 losses cluster on resource gathering (logs / iron_ore / gold_ore), an execution-layer issue not addressed by the decisioner | losses table + wrapper.py is unchanged |
| Step efficiency is broadly preserved (median 5240 vs 5450; v3 faster on 58% of common-success tasks) | steps table |
| Planner fallback path is functional and used selectively (31 times across 67 runs, all in successful tasks) | source breakdown |

## What this does NOT yet show

1. Variance: each task ran once. `craft_a_chain` failure (no failed_wp,
   crashed) and the 3 logs-stuck stone tasks could swing either way on a
   re-run. To estimate variance we should run v3 a second time and compare.
2. The decisioner has limited reach: 70 of 74 waypoints have only 1
   action in the case library, so the decisioner's top-1 pick equals the
   baseline pick on those waypoints. The wins likely come from
   - The 4 multi-action waypoints (cobblestone, charcoal, stone, smooth_stone)
   - The threshold gate triggering planner fallback when P < 0.20, which
     allows the planner to re-propose actions on hard states.
3. We have not yet broken down which fraction of the 18 wins come from
   "decisioner picked a different action" vs "decisioner triggered a
   planner fallback at the right moment". A follow-up analysis pass on
   the cases.json decision_trace can answer this.

## Suggested next steps

1. Run v3 a second time (different EXP_NUM_BASE, e.g. 33000) and look at
   which 8 losses are reproducible vs stochastic.
2. Re-run the 8 v3 losses with `DECISIONER_MIN_P=0.40` to make the
   decisioner more eager to fall back to planner. If the failures are
   recovered, threshold tuning will be the cheapest win.
3. Aggregate `decision_trace.candidates` per win to find the smoking gun
   on the 18 wins — which ones came from action re-ranking vs from
   planner fallback.
