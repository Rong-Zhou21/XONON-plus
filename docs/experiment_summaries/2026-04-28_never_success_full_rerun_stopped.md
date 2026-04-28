# 2026-04-28 Never-Success Full Rerun, Stopped

This run was started after the raw POV video recorder fix to rerun all tasks with no previous success, from easier tasks to harder tasks.
It was stopped by user request during the second task.

## Completed Before Stop

| Category | Task | Exp | Result | Status | Steps | Failed waypoint | Video |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| Iron | `craft_a_iron_pickaxe` | 10601 | failed | `env_step_timeout` | 198 | `logs` | exists |

## Interrupted / No New Result

| Category | Task | Exp | State |
| --- | --- | ---: | --- |
| Iron | `craft_a_hopper` | 10605 | interrupted by stop request; no new result JSON for this run |

## Successes Added By This Rerun

None. No never-success task became successful before the run was stopped.

## Notes

- The completed `craft_a_iron_pickaxe` attempt is an early environment stop, not a full agent-capability failure.
- The referenced video was written with the raw POV recorder path introduced in commit `87aafcd`.
- The run was not continued to the remaining Iron, Diamond, Redstone, or Armor tasks.
