# V7 Fair Armor One-Round Experiment Record

## Run Scope

- Repository: `/home/yzb/zhourong/XENON-plus`
- Run label: `v7_fair_horizontal_tunnel`
- Start time: `Sat May 9 20:01:36 CST 2026`
- End time: `Sun May 10 04:47:30 CST 2026`
- Summary log: `/tmp/xenon_v7_v7_fair_horizontal_tunnel_20260509_200136_summary.log`
- Results directory: `exp_results/v7`
- Videos directory: `videos/v7`
- Trial order: `golden_chestplate` x10, `golden_leggings` x10, `diamond_chestplate` x10
- Instruction followed in this run: keep existing logic, run one full round, no real-time code correction.

## Fairness Configuration

This run kept the current underground exploration logic but disabled the earlier debugging shortcuts that could affect fairness:

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`
- `XENON_ENABLE_RANDOM_ORE_ONCE=0`
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`
- `XENON_CORRIDOR_STABILIZE_FLOOR=0`
- `XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP=0`
- `DELETE_ABNORMAL_ARTIFACTS=0`
- `MAX_RETRIES_ON_CRASH=0`
- `MAX_RETRIES_ON_INFRA_EARLY_STOP=2`

Important note: no ore was spawned at the agent's digging position in this run. Failure artifacts were preserved. The only retry observed was the configured infra early-stop retry for trial 16.

## Command

```bash
RUN_LABEL=v7_fair_horizontal_tunnel \
EXP_NUM_BASE=470000 \
SEED_BASE=1 \
SKIP_DONE=0 \
MAX_RETRIES_ON_CRASH=0 \
MAX_RETRIES_ON_INFRA_EARLY_STOP=2 \
DELETE_ABNORMAL_ARTIFACTS=0 \
XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0 \
XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0 \
XENON_ENABLE_RANDOM_ORE_ONCE=0 \
XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0 \
XENON_CORRIDOR_STABILIZE_FLOOR=0 \
XENON_CORRIDOR_STABILIZE_AFTER_FULL_SWEEP=0 \
XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0 \
XENON_TUNNEL_ALLOW_LOW_SOFT_ACCEPT=1 \
bash scripts/run_v7_armor_targeted.sh
```

## Result Summary

| Task | Trials | Success | Fail | Timeout | Crash | Other failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| golden_chestplate | 10 | 0 | 10 | 8 | 1 | 1 |
| golden_leggings | 10 | 0 | 10 | 9 | 1 | 0 |
| diamond_chestplate | 10 | 0 | 10 | 8 | 2 | 0 |
| Total | 30 | 0 | 30 | 25 | 4 | 1 |

Overall success rate: `0/30 = 0%`.

## Per-Trial Results

| # | Task | Exp | Seed | Result | Status | Steps | Minutes |
| ---: | --- | ---: | ---: | --- | --- | ---: | ---: |
| 1 | golden_chestplate | 471200 | 1 | FAIL | timeout_non_programmatic | 20778 | 17.32 |
| 2 | golden_chestplate | 471201 | 2 | FAIL | timeout_non_programmatic | 21220 | 17.68 |
| 3 | golden_chestplate | 471202 | 3 | FAIL | timeout_non_programmatic | 33969 | 28.31 |
| 4 | golden_chestplate | 471203 | 4 | FAIL | failed | 25102 | 20.92 |
| 5 | golden_chestplate | 471204 | 5 | FAIL | crash_RuntimeError | 36000 | 30.00 |
| 6 | golden_chestplate | 471205 | 6 | FAIL | timeout_non_programmatic | 25384 | 21.15 |
| 7 | golden_chestplate | 471206 | 7 | FAIL | timeout_non_programmatic | 20901 | 17.42 |
| 8 | golden_chestplate | 471207 | 8 | FAIL | timeout_non_programmatic | 29846 | 24.87 |
| 9 | golden_chestplate | 471208 | 9 | FAIL | timeout_non_programmatic | 30164 | 25.14 |
| 10 | golden_chestplate | 471209 | 10 | FAIL | timeout_non_programmatic | 13918 | 11.60 |
| 11 | golden_leggings | 471000 | 1 | FAIL | timeout_non_programmatic | 32803 | 27.34 |
| 12 | golden_leggings | 471001 | 2 | FAIL | crash_RuntimeError | 36001 | 30.00 |
| 13 | golden_leggings | 471002 | 3 | FAIL | timeout_non_programmatic | 33144 | 27.62 |
| 14 | golden_leggings | 471003 | 4 | FAIL | timeout_non_programmatic | 21707 | 18.09 |
| 15 | golden_leggings | 471004 | 5 | FAIL | timeout_non_programmatic | 16868 | 14.06 |
| 16 | golden_leggings | 471005 | 6 | FAIL | timeout_non_programmatic | 35832 | 29.86 |
| 17 | golden_leggings | 471006 | 7 | FAIL | timeout_non_programmatic | 12618 | 10.52 |
| 18 | golden_leggings | 471007 | 8 | FAIL | timeout_non_programmatic | 34631 | 28.86 |
| 19 | golden_leggings | 471008 | 9 | FAIL | timeout_non_programmatic | 30754 | 25.63 |
| 20 | golden_leggings | 471009 | 10 | FAIL | timeout_non_programmatic | 14780 | 12.32 |
| 21 | diamond_chestplate | 470600 | 1 | FAIL | timeout_non_programmatic | 29219 | 24.35 |
| 22 | diamond_chestplate | 470601 | 2 | FAIL | crash_RuntimeError | 36000 | 30.00 |
| 23 | diamond_chestplate | 470602 | 3 | FAIL | timeout_non_programmatic | 35874 | 29.89 |
| 24 | diamond_chestplate | 470603 | 4 | FAIL | timeout_non_programmatic | 13142 | 10.95 |
| 25 | diamond_chestplate | 470604 | 5 | FAIL | timeout_non_programmatic | 13155 | 10.96 |
| 26 | diamond_chestplate | 470605 | 6 | FAIL | timeout_non_programmatic | 20541 | 17.12 |
| 27 | diamond_chestplate | 470606 | 7 | FAIL | timeout_non_programmatic | 15288 | 12.74 |
| 28 | diamond_chestplate | 470607 | 8 | FAIL | timeout_non_programmatic | 28729 | 23.94 |
| 29 | diamond_chestplate | 470608 | 9 | FAIL | timeout_non_programmatic | 23517 | 19.60 |
| 30 | diamond_chestplate | 470609 | 10 | FAIL | crash_RuntimeError | 36001 | 30.00 |

## Observed Underground Behavior

1. Iron acquisition is still the main bottleneck.
   Many runs completed the wood, cobblestone, furnace, and stone-pickaxe chain, then spent most of the budget in `dig down and mine iron_ore`. The agent frequently kept digging vertically through valid depth ranges without collecting enough iron.

2. Some trials reached the next mining stage, but not reliably.
   Trial 15 reached the gold stage and crafted an iron pickaxe, then failed at `dig down and mine gold_ore`. Trial 21 reached the diamond stage with an iron pickaxe, then failed at `dig down and mine diamond`.

3. The horizontal tunnel primitive can move the agent, but resource yield is sparse.
   Trial 24 collected `diamond: 1` and showed about 8.5 blocks of horizontal displacement near y=5. Trial 27 also collected `diamond: 1`. Neither trial expanded this into enough diamond collection for a chestplate.

4. Relevel overshoot remains a major failure mode.
   Several gold/iron-stage attempts tried to return to a target band but ended around y=54 instead of the intended y=21 or y=34. The later tunnel check then rejected horizontal exploration with `pre_tunnel_above_target`, so the agent resumed vertical digging instead of exploring the remembered useful height.

5. Cave or structure contact is not converted into exploration.
   One run collected `rail: 1`, suggesting contact with a mineshaft-like structure, but the policy continued with the fixed dig-down pattern and did not turn this into cave/mineshaft exploration.

6. Visual reasoning can either do nothing or misdirect.
   Logs repeatedly show visual reasoning judging that the agent should keep digging even after long stagnation. In other cases it changed the prompt to broad exploration such as `explore other areas for iron ore`, but this did not produce stable mining behavior.

7. Failure records were preserved.
   The run ended with `no_result=0`, `skipped=0`, and `fail=30`. The summary file and per-trial logs contain the failed outcomes rather than hiding them.

## Conclusion

With normal ore distribution and all ore-generation/debug shortcuts disabled, the current V7 underground exploration logic did not solve any of the 30 Armor trials in this one-round run.

The current mechanism is not unfairly succeeding by placing ore in front of the agent. In the fair environment, the remaining bottleneck is not the surface/toolchain setup but robust underground resource acquisition: reaching a target layer, staying near that layer, moving horizontally enough, and turning that movement into repeated ore discovery.
