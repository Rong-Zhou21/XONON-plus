# 2026-04-27 Bug Investigation And Rerun Summary

Project: `/home/yzb/zhourong/XENON-plus`
Scope: investigate failed Stone/Iron experiments that were caused by runtime/helper bugs rather than agent decision quality.
Video policy: videos are kept locally only and are not uploaded to GitHub.

## Problems Reported

1. Some failed tasks produced only a few seconds of video and had almost no actual execution, for example Iron `shears`.
2. Stone `smoker` obtained the required `furnace`, but `smoker` crafting was still judged failed.
3. Several Iron timeout videos were named after the last completed subgoal, usually `craft_stone_pickaxe`, even though the actual failed subgoal was mining `iron_ore`.

## Root Causes

### 1. Short Videos / No Real Execution

Affected examples:

- `craft_a_furnace`, old exp `1007`
- `craft_a_iron_pickaxe`, old exp `2001`
- `craft_a_iron_axe`, old exp `2002`
- `craft_a_shears`, old exp `2008`
- `craft_an_iron_nugget`, old exp `2013`
- `craft_a_blast_furnace`, old exp `2014`

Cause:

MineRL/Malmo sometimes returns `done=True` after a backend step timeout:

```text
Failed to take a step (error timed out). Terminating episode and sending random observation
```

The previous code treated this as a normal `timeout_non_programmatic` task failure, even when it happened at around 100-130 steps. These runs were infrastructure failures, not meaningful agent failures.

Fix:

- `main_planning.py` now distinguishes MineRL step errors from real task timeouts.
- Such runs are marked as `env_step_timeout`.
- The rerun script automatically retries `env_step_timeout` or very early `logs` failures.
- Case memory entries from these runs are marked with `outcome.status = env_step_timeout` and `success = null`, so they are not treated as decision-training failures.

### 2. Smoker Failed Despite Having Furnace

Affected old result:

- `craft_a_smoker`, old exp `1005`

Cause:

The inventory contained enough total logs for the smoker recipe:

```text
birch_log: 3
oak_log: 1
furnace: 1
```

The recipe uses the tag `minecraft:logs`, requiring four log-tag items. The old craft helper checked only one matching inventory stack. Since no single stack had four logs, it incorrectly reported missing material even though the combined tag quantity was sufficient.

Fix:

- `new_craft_helper.py` now sums quantities across all matching inventory stacks for tag ingredients.
- Shaped and shapeless crafting can place ingredients from multiple matching stacks when a tag spans several item types.
- This specifically fixes recipes like `smoker` where multiple log species can satisfy `minecraft:logs`.

### 3. Failed Videos Named `craft_stone_pickaxe`

Affected examples:

- `craft_a_bucket`, old exp `2004`
- `craft_a_hopper`, old exp `2005`
- `craft_a_iron_sword`, old exp `2007`
- `craft_a_tripwire_hook`, old exp `2010`
- `craft_a_chain`, old exp `2011`
- `craft_an_iron_bars`, old exp `2012`

Cause:

The video saving logic used the last completed subgoal as `actual_done_final_task`. For Iron tasks, the last completed subgoal before failure is often `craft stone_pickaxe`, while the real failed subgoal is `dig down and mine iron_ore`.

Fix:

- On failed runs, video naming now uses the failed subgoal when available.
- On successful runs, video naming still uses the final completed subgoal.

Example corrected path:

```text
videos/v1/Iron_Craft_a_hopper/plains/failed/..._dig_down_and_mine_iron_ore_...
```

## Rerun Results

The affected Stone failures and Iron failures were rerun after the fixes. New experiments use the `1100/2100` exp ranges.

### Stone

| Old Exp | New Exp | Task | Result After Fix | Status | Failed Waypoints | Notes |
|---:|---:|---|---|---|---|---|
| 1004 | 1104 | `smelt_a_charcoal` | success | `success` | [] | Previous timeout did not reproduce. |
| 1005 | 1105 | `craft_a_smoker` | success | `success` | [] | Confirms tag-material fix. |
| 1007 | 1107 | `craft_a_furnace` | success | `success` | [] | Previous few-second early failure did not reproduce. |

### Iron

| Old Exp | New Exp | Task | Result After Fix | Status | Failed Waypoints | Notes |
|---:|---:|---|---|---|---|---|
| 2001 | 2101 | `craft_a_iron_pickaxe` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid full run; no early failure. |
| 2002 | 2102 | `craft_a_iron_axe` | success | `success` | [] | Previous early failure did not reproduce. |
| 2004 | 2104 | `craft_a_bucket` | success | `success` | [] | Previous iron_ore timeout did not reproduce. |
| 2005 | 2105 | `craft_a_hopper` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid failure; video name fixed. |
| 2007 | 2107 | `craft_a_iron_sword` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid failure; video name fixed. |
| 2008 | 2108 | `craft_a_shears` | invalid rerun | `env_step_timeout` | [`logs`] | Infrastructure early stop; retried. |
| 2008 | 2109 | `craft_a_shears` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid full run; video name fixed. |
| 2010 | 2110 | `craft_a_tripwire_hook` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid failure; video name fixed. |
| 2011 | 2111 | `craft_a_chain` | success | `success` | [] | Previous iron_ore timeout did not reproduce. |
| 2012 | 2112 | `craft_an_iron_bars` | invalid rerun | `env_step_timeout` | [`logs`] | Infrastructure early stop; retried. |
| 2012 | 2113 | `craft_an_iron_bars` | failed | `timeout_non_programmatic` | [`iron_ore`] | Valid full run; video name fixed. |
| 2013 | 2113 | `craft_an_iron_nugget` | success | `success` | [] | Exp number collided with iron_bars retry; kept locally only. |
| 2013 | 2123 | `craft_an_iron_nugget` | failed | `timeout_non_programmatic` | [`iron_ore`] | Clean canonical rerun; valid full run. |
| 2014 | 2114 | `craft_a_blast_furnace` | failed | `timeout_non_programmatic` | [`stone`] | Valid full run; no early logs failure. |
| 2015 | 2115 | `craft_a_stonecutter` | failed | `timeout_non_programmatic` | [`iron_ore`, `stone`] | Valid full run; video name fixed. |

## Corrected Interpretation

After the bugfix reruns:

- The Stone failures from the previous batch should not be treated as final failures; all three affected Stone tasks succeeded after rerun.
- Several Iron failures that looked like immediate no-op failures were actually environment step timeouts and have now been separated from agent failures.
- Remaining Iron failures are mostly meaningful progression failures at `iron_ore` or `stone`, not immediate execution bugs.
- Old bug-affected case records were relabeled as `superseded_bug`; infrastructure early stops were relabeled as `env_step_timeout`.
- `superseded_bug` and `env_step_timeout` records should be excluded from decisioner training and final success/failure analysis.
- The canonical `craft_an_iron_nugget` rerun is exp `2123`; exp `2113` was kept locally only because its number collided with the iron bars retry.
- Video files remain local and are not part of GitHub tracking.

## Case Memory State

After reruns and infrastructure-error relabeling:

```text
total cases: 446
success: 391
failed: 31
failed_incomplete_run: 3
stopped_by_user: 2
superseded_bug: 17
env_step_timeout: 2
```

`superseded_bug` and `env_step_timeout` cases have `success = null` so they can be filtered out from decisioner training.
