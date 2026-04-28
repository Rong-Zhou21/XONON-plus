# XENON-plus Current Performance Sweep - 2026-04-28

Code version before run: `5dcaec3 chore: add current performance sweep runner`

Run command:

```bash
XENON_MAX_VALID_ATTEMPTS=1 bash scripts/run_current_performance_sweep.sh /tmp/xenon_plus_current_performance_sweep.log
```

Scope:

- Original never-success task list: 11 tasks
- Redstone: 6 tasks
- Armor: 13 tasks

Overall result: 30 tasks, 15 success, 15 failed.

All result JSON files were generated under `exp_results/v1/`. All referenced local videos existed after the run. `exp_results/` and `videos/` are gitignored, so this file records the success/failure summary for GitHub retention without uploading videos.

## Category Summary

| Category | Tasks | Success | Failed |
| --- | ---: | ---: | ---: |
| Original never-success list | 11 | 5 | 6 |
| Redstone | 6 | 4 | 2 |
| Armor | 13 | 6 | 7 |
| Total | 30 | 15 | 15 |

## Task Results

| Category | Task | Result | Status | Failed waypoint | Steps | Minutes |
| --- | --- | --- | --- | --- | ---: | ---: |
| Gold | `craft_a_golden_pickaxe` | success | `success` | `cobblestone` | 6608 | 5.51 |
| Gold | `craft_a_golden_axe` | success | `success` | - | 6988 | 5.82 |
| Diamond | `craft_a_diamond_sword` | success | `success` | - | 5575 | 4.65 |
| Diamond | `dig_down_and_mine_a_diamond` | success | `success` | - | 5926 | 4.94 |
| Diamond | `craft_a_jukebox` | failed | `env_step_timeout` | `logs` | 161 | 0.13 |
| Iron | `craft_a_iron_pickaxe` | failed | `crash_RuntimeError` | - | 12000 | 10.00 |
| Iron | `craft_a_hopper` | failed | `timeout_non_programmatic` | `iron_ore` | 11904 | 9.92 |
| Iron | `craft_a_iron_sword` | failed | `timeout_non_programmatic` | `logs` | 11997 | 10.00 |
| Iron | `craft_a_shears` | failed | `timeout_non_programmatic` | `iron_ore` | 11912 | 9.93 |
| Iron | `craft_a_tripwire_hook` | success | `success` | - | 10547 | 8.79 |
| Iron | `craft_a_blast_furnace` | failed | `timeout_non_programmatic` | `smooth_stone` | 11954 | 9.96 |
| Redstone | `craft_a_piston` | failed | `env_step_timeout` | `logs` | 185 | 0.15 |
| Redstone | `craft_a_redstone_torch` | success | `success` | - | 5838 | 4.87 |
| Redstone | `craft_an_activator_rail` | success | `success` | - | 7803 | 6.50 |
| Redstone | `craft_a_compass` | success | `success` | - | 6864 | 5.72 |
| Redstone | `craft_a_dropper` | success | `success` | `stick` | 7509 | 6.26 |
| Redstone | `craft_a_note_block` | failed | `timeout_non_programmatic` | `iron_ore` | 35910 | 29.93 |
| Armor | `craft_shield` | success | `success` | - | 3921 | 3.27 |
| Armor | `craft_iron_chestplate` | success | `success` | - | 5589 | 4.66 |
| Armor | `craft_iron_boots` | success | `success` | - | 6158 | 5.13 |
| Armor | `craft_iron_leggings` | success | `success` | - | 5321 | 4.43 |
| Armor | `craft_iron_helmet` | success | `success` | - | 4957 | 4.13 |
| Armor | `craft_diamond_helmet` | failed | `env_step_timeout` | `logs` | 159 | 0.13 |
| Armor | `craft_diamond_chestplate` | failed | `timeout_non_programmatic` | `diamond` | 35866 | 29.89 |
| Armor | `craft_diamond_leggings` | failed | `env_step_timeout` | `logs` | 184 | 0.15 |
| Armor | `craft_diamond_boots` | failed | `env_step_timeout` | `logs` | 179 | 0.15 |
| Armor | `craft_golden_helmet` | success | `success` | - | 9222 | 7.68 |
| Armor | `craft_golden_leggings` | failed | `timeout_non_programmatic` | `gold_ore` | 35867 | 29.89 |
| Armor | `craft_golden_boots` | failed | `timeout_non_programmatic` | `gold_ore` | 35864 | 29.89 |
| Armor | `craft_golden_chestplate` | failed | `timeout_non_programmatic` | `gold_ore` | 35866 | 29.89 |

## Notes For Review

- Early `env_step_timeout` failures were retained as-is because this sweep was intended to measure current performance with one attempt per task.
- Long timeout failures are concentrated around mining `iron_ore`, `diamond`, and `gold_ore`.
- Several successful tasks still recorded intermediate failed waypoints; these are preserved in the result JSON and case memory.
