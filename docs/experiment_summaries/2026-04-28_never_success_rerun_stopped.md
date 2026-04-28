# 2026-04-28 Never-Success Rerun, Stopped

This run was started to rerun tasks that had no successful result in `exp_results/v1`.
It was stopped by user request after the second completed task.

## Never-Success Tasks Before This Run

| Category | Benchmark | Task id | Task |
| --- | --- | ---: | --- |
| Iron | `iron` | 1 | `craft_a_iron_pickaxe` |
| Iron | `iron` | 5 | `craft_a_hopper` |
| Iron | `iron` | 7 | `craft_a_iron_sword` |
| Iron | `iron` | 8 | `craft_a_shears` |
| Iron | `iron` | 14 | `craft_a_blast_furnace` |
| Diamond | `diamond` | 6 | `craft_a_jukebox` |
| Redstone | `redstone` | 0 | `craft_a_piston` |
| Redstone | `redstone` | 5 | `craft_a_note_block` |
| Armor | `armor` | 5 | `craft_diamond_helmet` |
| Armor | `armor` | 6 | `craft_diamond_chestplate` |
| Armor | `armor` | 7 | `craft_diamond_leggings` |
| Armor | `armor` | 8 | `craft_diamond_boots` |
| Armor | `armor` | 10 | `craft_golden_leggings` |
| Armor | `armor` | 11 | `craft_golden_boots` |
| Armor | `armor` | 12 | `craft_golden_chestplate` |

## Run Settings

- Script: `scripts/run_never_success_tasks.sh`
- Attempts per task: 1
- Task cooldown: 10 seconds
- Cleanup between tasks: enabled; stale MineRL/Malmo/Xvfb processes are stopped before the next task.
- Iron task limit: 10 minutes
- Diamond, Redstone, and Armor task limit in this rerun script: 20 minutes
- Videos were kept locally and were not staged for Git.

## Completed Results Before Stop

| Category | Task | Exp | Result | Status | Steps | Failed waypoint | Video |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| Iron | `craft_a_iron_pickaxe` | 10601 | failed | `timeout_non_programmatic` | 11997 | `logs` | exists |
| Iron | `craft_a_hopper` | 10605 | failed | `env_step_timeout` | 203 | `logs` | exists |

## Interrupted / No Result

| Category | Task | Exp | State |
| --- | --- | ---: | --- |
| Iron | `craft_a_iron_sword` | 10607 | interrupted by stop request; no result JSON |
| Iron | `craft_a_shears` | 10608 | script had just advanced to this task; process was cleaned before result JSON |

## Notes

- `craft_a_hopper` reproduced an early infrastructure-style stop: `env_step_timeout` at 203 steps.
- `craft_a_iron_pickaxe` ran the full 10 minute limit and failed at `logs`; this is a valid failed attempt, not an early environment stop.
- The run was not continued to Diamond, Redstone, or Armor after the stop request.
