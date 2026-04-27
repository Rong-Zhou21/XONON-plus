# 2026-04-27 Execution Logic Fix And Gold/Diamond Run

Project: `/home/yzb/zhourong/XENON-plus`
Scope: bottom-level execution fixes for action persistence, recovery from adverse movement states, and non-visual resource tracking; then one benchmark pass over previously unsolved Iron tasks plus Gold and Diamond.

Video policy: videos are kept locally only and are not uploaded to GitHub.

## Execution Changes

1. STEVE-1 recurrent state is reset when the language action prompt changes.
   - This prevents hidden-state inertia from one subgoal carrying into the next subgoal.
   - Targeted symptom: repeated short jumps/clicks after a task transition.

2. Non-programmatic actions now pass the current prompt into the environment wrapper.
   - The wrapper can distinguish resource-acquisition actions such as `chop`, `mine`, `dig`, and `break`.
   - This makes action stabilization depend on the current action semantics rather than on a specific task name.

3. Resource-acquisition `attack` is held across multiple ticks.
   - The first model action keeps the model's movement intent.
   - Follow-up ticks hold `attack` and suppress movement/camera drift so mining/chopping is not interrupted by single-frame clicks.
   - Targeted symptom: mining clicks were too short to complete a Minecraft block-breaking swing.

4. A movement recovery primitive was added.
   - It uses non-visual state: breath/air and short-window position deltas.
   - If the agent is losing air or has movement intent but makes no position progress, it executes a bounded forward+jump+sprint turn-out primitive.
   - After this run, the default low-air trigger was tightened from `air < 300` to `air < 280`, because logs showed `air=299` can be too sensitive.

5. A non-visual resource ledger was added.
   - It records inventory facts from `plain_inventory`: max observed inventory count and positive deltas per item.
   - Subgoal completion can be satisfied by this ledger even when visual perception misses that an item entered inventory.
   - The ledger is included in `env.get_status()` for future case-memory records.

6. Inventory pressure cleanup was added.
   - When inventory slots are nearly full, the wrapper can drop low-priority hotbar clutter such as seeds, flowers, grass, leaves, saplings, dirt, sand, and gravel.
   - This uses normal hotbar/drop actions, not `/clear` commands.
   - Protected items include tools, logs, planks, sticks, crafting tables, furnaces, ores, ingots, coal, diamonds, redstone, and the current goal item.

7. MineRL step-timeout observations are guarded.
   - MineRL sometimes returns a random observation with `info["error"]`.
   - The wrapper now returns immediately in this case instead of parsing malformed random inventory/location values.
   - These runs are classified as `env_step_timeout`, not Python crashes.

## Run Set

Command script:

```text
scripts/run_execution_logic_benchmarks.sh /tmp/xenon_plus_execution_logic_summary.log
```

Run order:

1. Iron tasks without canonical success after the previous uploaded batch: task ids `1, 5, 7, 8, 10, 12, 13, 14, 15`
2. Gold task ids `0-5`
3. Diamond task ids `0-6`

## Results

### Iron Previously Unsolved

| Exp | Task | Result | Status | Steps | Failed Waypoints | Notes |
|---:|---|---|---|---:|---|---|
| 7101 | `craft_a_iron_pickaxe` | invalid | `env_step_timeout` | 157 | [`logs`] | Environment early stop. |
| 7105 | `craft_a_hopper` | failed | `timeout_non_programmatic` | 11892 | [`iron_ore`] | Valid full run. |
| 7107 | `craft_a_iron_sword` | failed | `timeout_non_programmatic` | 11897 | [`iron_ore`] | Valid full run. |
| 7108 | `craft_a_shears` | invalid | `env_step_timeout` | 203 | [`logs`] | Environment early stop. |
| 7110 | `craft_a_tripwire_hook` | failed | `timeout_non_programmatic` | 11996 | [`logs`] | Valid full run. |
| 7112 | `craft_an_iron_bars` | success | `success` | 9553 | [] | New canonical success. |
| 7113 | `craft_an_iron_nugget` | failed | `timeout_non_programmatic` | 11996 | [`logs`] | Valid full run. |
| 7114 | `craft_a_blast_furnace` | failed | `timeout_non_programmatic` | 11996 | [`logs`] | Valid full run. |
| 7115 | `craft_a_stonecutter` | success | `success` | 11489 | [] | New canonical success. |

### Gold

| Exp | Task | Result | Status | Steps | Failed Waypoints | Notes |
|---:|---|---|---|---:|---|---|
| 7300 | `craft_a_golden_shovel` | success | `success` | 6226 | [] | Valid success. |
| 7301 | `craft_a_golden_pickaxe` | failed | `timeout_non_programmatic` | 13034 | [`iron_ore`, `stick`] | Valid failure. |
| 7302 | `craft_a_golden_axe` | invalid | `env_step_timeout` | 171 | [`logs`] | Environment early stop. |
| 7303 | `craft_a_golden_hoe` | success | `success` | 10853 | [] | Valid success. |
| 7304 | `craft_a_golden_sword` | success | `success` | 6518 | [] | Valid success. |
| 7305 | `smelt_and_craft_a_gold_ingot` | success | `success` | 5288 | [] | Valid success. |

### Diamond

| Exp | Task | Result | Status | Steps | Failed Waypoints | Notes |
|---:|---|---|---|---:|---|---|
| 7400 | `craft_a_diamond_shovel` | success | `success` | 6154 | [] | Valid success. |
| 7401 | `craft_a_diamond_pickaxe` | success | `success` | 8426 | [`stick`] | Final task succeeded after an earlier failed waypoint. |
| 7402 | `craft_a_diamond_axe` | success | `success` | 5474 | [] | Valid success. |
| 7403 | `craft_a_diamond_hoe` | success | `success` | 6517 | [`stick`] | Final task succeeded after an earlier failed waypoint. |
| 7404 | `craft_a_diamond_sword` | invalid | `env_step_timeout` | 180 | [`logs`] | Environment early stop. |
| 7405 | `dig_down_and_mine_a_diamond` | invalid | `env_step_timeout` | 189 | [`logs`] | Environment early stop. |
| 7406 | `craft_a_jukebox` | invalid | `env_step_timeout` | 185 | [`logs`] | Environment early stop. |

## Aggregate

Valid results only:

```text
Iron unsolved rerun: 2 success, 5 failed
Gold: 4 success, 1 failed
Diamond: 4 success, 0 failed
Total valid: 10 success, 6 failed
```

Environment early stops:

```text
env_step_timeout: 6
```

The `env_step_timeout` JSON files are kept locally for traceability but should be excluded from decisioner training and canonical success/failure analysis.

## Follow-up Observations

- Recovery primitive logs appeared in the Iron mining failures, especially around `iron_ore`; this confirms the new non-visual recovery path is being exercised.
- The first low-air threshold was too sensitive, so the committed code now requires stronger evidence of water/adverse state by default (`air < 280`) while keeping the stagnation-based recovery path.
- Some valid failures still happen at `logs`, which should be reviewed by video; these are not environment early stops and may indicate STEVE-1 low-level navigation/chop weakness rather than planner/case-memory failure.
