# 2026-04-28 Never-Success Autorun, Stopped

This run used the no-prompt raw POV video recorder and the early-stop retry policy.
It was stopped by user request during `craft_a_blast_furnace`.

## Run Policy

- Early infrastructure stop is retried when `env_step_timeout` occurs before 300 steps, or when a run exits before 300 steps at `logs`.
- Early-stop pending cases are discarded from case memory instead of being marked as failed training cases.
- Maximum attempts per task: 3
- Task cooldown: 10 seconds
- Iron task limit: 10 minutes
- Later difficult task limit in the script: 20 minutes

## Completed Task Outcomes

| Category | Task | Attempts | Final counted result | Status | Steps | New success? |
| --- | --- | ---: | --- | --- | ---: | --- |
| Iron | `craft_a_iron_pickaxe` | 1 | failed | `crash_RuntimeError` | 12001 | no |
| Iron | `craft_a_hopper` | 1 | failed | `timeout_non_programmatic` | 11997 | no |
| Iron | `craft_a_iron_sword` | 3 | failed | `timeout_non_programmatic` | 11997 | no |
| Iron | `craft_a_shears` | 2 | failed | `timeout_non_programmatic` | 11997 | no |

## Retried Early Stops

| Task | Attempt | Status | Steps |
| --- | ---: | --- | ---: |
| `craft_a_iron_sword` | 1 | `env_step_timeout` | 170 |
| `craft_a_iron_sword` | 2 | `env_step_timeout` | 159 |
| `craft_a_shears` | 1 | `env_step_timeout` | 174 |

## Interrupted / No Result

| Category | Task | Exp | State |
| --- | --- | ---: | --- |
| Iron | `craft_a_blast_furnace` | 10614 | interrupted by stop request; no result JSON |

## Successes Added By This Run

None. No previously never-success task became successful before the run was stopped.
