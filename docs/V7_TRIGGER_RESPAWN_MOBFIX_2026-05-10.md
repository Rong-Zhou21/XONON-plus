# V7 Trigger, Respawn, and Mob-Fix Experiment Record

## Scope

- Repository: `/home/yzb/zhourong/XENON-plus`
- Run label: `v7_trigger_respawn_mobfix`
- Start time: `Sun May 10 16:54:13 CST 2026`
- Summary log: `/tmp/xenon_v7_v7_trigger_respawn_mobfix_20260510_165413_summary.log`
- App server log: `/tmp/xenon_v7_v7_trigger_respawn_mobfix_app_20260510_165413.log`
- Results directory: `exp_results/v7`
- Videos directory: `videos/v7`
- Trial order: `golden_chestplate` x10, `golden_leggings` x10, `diamond_chestplate` x10

## Code Changes Being Validated

1. Mob interference removal.
   `evaluate.yaml` now disables natural mob spawning and switches the world to peaceful difficulty at episode reset. This is intended to remove hostile mob death noise without changing ore distribution or adding resources.

2. Pillar-up trigger tightening.
   Underground relevel should now trigger only from deeper-stage ore discovery or bottom-layer hard-stuck behavior. The old scripted-loop relevel path is disabled by default, and low Y alone is not enough: the agent must also show both target-resource stagnation and total mined-block stagnation.

3. Respawn equipment recovery.
   After respawn, mining mode is reset to `dig_down`, target/resource counters are refreshed, and the best available pickaxe is equipped before underground mining resumes.

4. Tree-stage stale-contact guard.
   Tree contact now has a bounded grace window, so stale contact with leaves/logs cannot indefinitely block the planner from issuing a fresh tree-search action.

## Fairness Configuration

Debug or unfair shortcuts remain disabled:

- `XENON_TUNNEL_SCRIPTED_DIGDOWN_GENERATE_ORE=0`
- `XENON_SCRIPTED_DIGDOWN_FORCE_TARGET_ORE=0`
- `XENON_ENABLE_RANDOM_ORE_ONCE=0`
- `XENON_ENABLE_COMMAND_RELEVEL_FALLBACK=0`
- `XENON_ENABLE_COMMAND_CRAFT_FALLBACK=0`
- `XENON_TUNNEL_SCRIPTED_LOOP_CAN_TRIGGER_RELEVEL=0`

No ore is spawned at the agent's digging position in this run.

## Live Observations

### Trial 1

- Task: `golden_chestplate`
- Exp: `491200`
- Seed: `0`
- Result file: `exp_results/v7/ours_planning_craft_golden_chestplate_491200_failed_forest_8GRs.json`
- Video: `videos/v7/Armor_Craft_golden_chestplate/forest/failed/2026_05_10_17_06_50_dig_down_and_mine_gold_ore_8GRsbKAFWktaMgQr62iaky.mp4`
- Result: failed, `timeout_non_programmatic`
- Steps/minutes: `21028` steps, `17.52` minutes

Observed behavior:

- Surface/toolchain setup completed: logs, planks, crafting table, stick, wooden pickaxe, cobblestone, furnace, stone pickaxe, iron ore, iron ingot, and iron pickaxe.
- The iron pickaxe was later exhausted or lost during gold mining. The agent recovered by crafting another stone pickaxe, mining more iron, smelting it, and crafting a replacement iron pickaxe.
- Final failure point was `dig down and mine gold_ore`: the agent collected `gold_ore: 2` and still needed `6` more.
- Relevel triggers observed during underground mining were `bedrock_stuck(...)`; no `scripted_loop_after_digdown` trigger was observed.
- One horizontal tunnel attempt succeeded, but a later attempt aborted with `terminal_or_position_jump` after only `0.09` horizontal displacement. This should be watched across later trials.

### Trial 2

- Task: `golden_chestplate`
- Exp: `491201`
- Seed: `1`
- Status: stopped manually after reproducing the respawn-equipment bug.

Observed behavior:

- A death/respawn transition was detected at `17:09:50` with `reason=health_low_to_full`.
- The high-level prompt was restored to `dig down and mine iron_ore`, which is correct.
- The agent did not reliably re-equip the best available pickaxe after respawn. A later visual reasoning record described the agent as holding a wooden pickaxe while mining iron, despite `stone_pickaxe` being present in inventory.
- This invalidated the first restart and led to a code fix before the next clean v7 run.

## Follow-Up Fix

After the Trial 2 observation, respawn equipment recovery was made retryable:

- `_ensure_best_pickaxe_equipped(...)` now returns success/failure and logs the no-pickaxe case.
- It first tries to directly select the best pickaxe already present on the hotbar.
- If the best pickaxe is only in inventory, it falls back to helper-based `equip <pickaxe>`.
- Helper success is no longer trusted blindly; the code checks that the held item actually equals the intended pickaxe.
- After respawn, the mining loop retries equipment recovery for a short window instead of only trying at the exact respawn tick.

The invalid v7 result/video artifacts from this stopped run were cleared before restarting.

### Retry Run Observation

A later clean retry reproduced the same timing hazard on seed `1`: the respawn handler restored the mining prompt, but the immediate equipment check could still read a stale pre-respawn status and therefore stop retrying too early.

Follow-up adjustment:

- The immediate respawn equipment check no longer closes the retry window.
- The loop now requires a later post-respawn observation to confirm that the best available pickaxe is actually equipped.
- This second invalid retry was stopped and its v7 result/video artifacts were cleared before the next clean restart.

## Current Assessment

The two requested bug fixes are active in the current run:

- Hostile mob influence is disabled through normal environment commands.
- The observed underground relevel triggers are no longer firing simply because the agent reached a low Y level.

The remaining observed bottleneck is not fairness or trigger correctness. It is resource-yield efficiency under fair ore distribution, especially collecting enough gold before the time budget expires.
