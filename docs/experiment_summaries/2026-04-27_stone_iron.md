# 2026-04-27 Stone / Iron Sequential Experiment Summary

Project: `/home/yzb/zhourong/XENON-plus`  
Runtime: Docker container `xenon_plus_case`  
Model backend: local vLLM, `Qwen/Qwen2.5-VL-7B-Instruct`  
Prefix: `ours_planning`  
Version: `v1`  
Execution policy: run one task at a time; finish all tasks in one benchmark before starting the next.

## Scope

The requested benchmark order was:

1. Stone
2. Iron
3. Gold
4. Diamond
5. Redstone
6. Armor

This session completed Stone and Iron. Per user instruction, experiments were stopped immediately after Iron completed. Gold task 0 had just started and was terminated before producing a valid result JSON; it is not counted as an experiment result.

## Result Overview

| Benchmark | Task Count | Success | Failed | Missing Result | Missing Video |
|---|---:|---:|---:|---:|---:|
| Stone | 9 | 6 | 3 | 0 | 0 |
| Iron | 16 | 4 | 12 | 0 | 0 |

All valid failed experiments have failure videos.

## Stone Results

| Exp | Task | Success | Status | Steps | Failed Waypoints | Result JSON | Video |
|---:|---|---|---|---:|---|---|---|
| 1000 | `craft_a_stone_shovel` | true | `success` | 2356 | [] | `exp_results/v1/ours_planning_craft_a_stone_shovel_1000_success_plains_WpFF.json` | exists |
| 1001 | `craft_a_stone_pickaxe` | true | `success` | 2093 | [`planks`] | `exp_results/v1/ours_planning_craft_a_stone_pickaxe_1001_success_plains_g8GH.json` | exists |
| 1002 | `craft_a_stone_axe` | true | `success` | 1631 | [] | `exp_results/v1/ours_planning_craft_a_stone_axe_1002_success_plains_iQ3H.json` | exists |
| 1003 | `craft_a_stone_hoe` | true | `success` | 1642 | [] | `exp_results/v1/ours_planning_craft_a_stone_hoe_1003_success_plains_HRbf.json` | exists |
| 1004 | `smelt_a_charcoal` | false | `timeout_non_programmatic` | 7138 | [`charcoal`] | `exp_results/v1/ours_planning_smelt_a_charcoal_1004_failed_plains_2zEd.json` | exists |
| 1005 | `craft_a_smoker` | false | `failed` | 3409 | [`smoker`] | `exp_results/v1/ours_planning_craft_a_smoker_1005_failed_plains_BZdb.json` | exists |
| 1006 | `craft_a_stone_sword` | true | `success` | 2164 | [] | `exp_results/v1/ours_planning_craft_a_stone_sword_1006_success_plains_mmJG.json` | exists |
| 1007 | `craft_a_furnace` | false | `timeout_non_programmatic` | 121 | [`logs`] | `exp_results/v1/ours_planning_craft_a_furnace_1007_failed_plains_PKw6.json` | exists |
| 1008 | `craft_a_torch` | true | `success` | 2440 | [] | `exp_results/v1/ours_planning_craft_a_torch_1008_success_plains_K8vG.json` | exists |

## Iron Results

| Exp | Task | Success | Status | Steps | Failed Waypoints | Result JSON | Video |
|---:|---|---|---|---:|---|---|---|
| 2000 | `craft_a_iron_shovel` | true | `success` | 5919 | [] | `exp_results/v1/ours_planning_craft_a_iron_shovel_2000_success_plains_SKRQ.json` | exists |
| 2001 | `craft_a_iron_pickaxe` | false | `timeout_non_programmatic` | 121 | [`logs`] | `exp_results/v1/ours_planning_craft_a_iron_pickaxe_2001_failed_plains_B7wa.json` | exists |
| 2002 | `craft_a_iron_axe` | false | `timeout_non_programmatic` | 122 | [`logs`] | `exp_results/v1/ours_planning_craft_a_iron_axe_2002_failed_plains_XBAi.json` | exists |
| 2003 | `craft_a_iron_hoe` | true | `success` | 8395 | [] | `exp_results/v1/ours_planning_craft_a_iron_hoe_2003_success_plains_AWnj.json` | exists |
| 2004 | `craft_a_bucket` | false | `timeout_non_programmatic` | 11896 | [`iron_ore`] | `exp_results/v1/ours_planning_craft_a_bucket_2004_failed_plains_BZeT.json` | exists |
| 2005 | `craft_a_hopper` | false | `timeout_non_programmatic` | 11887 | [`chest`, `iron_ore`] | `exp_results/v1/ours_planning_craft_a_hopper_2005_failed_plains_D4ES.json` | exists |
| 2006 | `craft_a_rail` | true | `success` | 8064 | [`crafting_table`] | `exp_results/v1/ours_planning_craft_a_rail_2006_success_plains_QH5e.json` | exists |
| 2007 | `craft_a_iron_sword` | false | `timeout_non_programmatic` | 11893 | [`iron_ore`] | `exp_results/v1/ours_planning_craft_a_iron_sword_2007_failed_plains_AjYG.json` | exists |
| 2008 | `craft_a_shears` | false | `timeout_non_programmatic` | 132 | [`logs`] | `exp_results/v1/ours_planning_craft_a_shears_2008_failed_plains_3mhq.json` | exists |
| 2009 | `craft_a_smithing_table` | true | `success` | 6848 | [] | `exp_results/v1/ours_planning_craft_a_smithing_table_2009_success_plains_CPvJ.json` | exists |
| 2010 | `craft_a_tripwire_hook` | false | `timeout_non_programmatic` | 11891 | [`iron_ore`] | `exp_results/v1/ours_planning_craft_a_tripwire_hook_2010_failed_plains_j3iG.json` | exists |
| 2011 | `craft_a_chain` | false | `timeout_non_programmatic` | 11894 | [`iron_ore`] | `exp_results/v1/ours_planning_craft_a_chain_2011_failed_plains_QKxn.json` | exists |
| 2012 | `craft_an_iron_bars` | false | `timeout_non_programmatic` | 11888 | [`iron_ore`] | `exp_results/v1/ours_planning_craft_an_iron_bars_2012_failed_plains_HE4K.json` | exists |
| 2013 | `craft_an_iron_nugget` | false | `timeout_non_programmatic` | 120 | [`logs`] | `exp_results/v1/ours_planning_craft_an_iron_nugget_2013_failed_plains_HubS.json` | exists |
| 2014 | `craft_a_blast_furnace` | false | `timeout_non_programmatic` | 111 | [`logs`] | `exp_results/v1/ours_planning_craft_a_blast_furnace_2014_failed_plains_Sg39.json` | exists |
| 2015 | `craft_a_stonecutter` | false | `timeout_non_programmatic` | 11938 | [`stone`] | `exp_results/v1/ours_planning_craft_a_stonecutter_2015_failed_plains_byMe.json` | exists |

## Video Organization

Videos use the category-prefixed directory naming rule:

```text
videos/v1/<Category>_<Task>/<biome>/<status>/<timestamp>_<done_task>_<run_uuid>.mp4
```

Examples:

```text
videos/v1/Stone_Craft_a_stone_pickaxe/plains/success/...
videos/v1/Iron_Craft_a_bucket/plains/failed/...
```

## Case Memory State After This Stage

Case file:

```text
src/optimus1/memories/ours_planning/v1/case_memory/cases.json
```

Current distribution after finalizing stopped pending cases:

```text
total: 289
success: 255
failed: 29
failed_incomplete_run: 3
stopped_by_user: 2
pending: 0
```

The two `stopped_by_user` cases come from interrupted runs after the user requested stopping experiments after Iron. They are retained as explicit stopped records rather than left as pending.

## Notes For Analysis

Update: some failures in this first Stone/Iron batch were later identified as runtime/helper bugs rather than agent-performance failures. See `docs/experiment_summaries/2026-04-27_bugfix_reruns.md` for root-cause analysis and corrected rerun results.

- Stone failures:
  - `smelt_a_charcoal` timed out at `charcoal`.
  - `craft_a_smoker` failed at `smoker`.
  - `craft_a_furnace` timed out at `logs`.
- Iron failures frequently occur at:
  - `logs` during early resource collection.
  - `iron_ore` during mining progression.
  - `stone` for `stonecutter`.
- These failures are expected at this stage and are useful for later case analysis and decisioner improvement.
