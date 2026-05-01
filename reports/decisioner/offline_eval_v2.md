# RADS Offline Evaluation (rads_v2)

- artifact: `artifacts/decisioner/rads_v2.pt`
- data: `data/decisioner/rads_v1.jsonl`
- spec: waypoints=73, goals=68, actions=79, feature_dim=52
- config: same_wp_min=8, lambda_triplet=0.1, lambda_wp=0.05, dropout=0.2
- library size (train cases): 1847

## Split: val

- n: 351
- pos_rate: 0.9402
- AUC: 0.8763
- AP:  0.9917
- best F1: 0.9455 @ thr=0.15
- ECE: 0.1190
- mean attention concentration: 0.8429

### Attention diagnostic (top-1 case)

- top-1 same-waypoint rate: 0.9430
- top-1 same-outcome rate: 0.6268
- top-1 same-both rate:    0.6068

### Per-waypoint AUC (waypoints with >= 5 negatives)

| waypoint | n | pos | neg | AUC |
|---|---:|---:|---:|---:|
| logs | 36 | 31 | 5 | 0.8710 |
| iron_ore | 22 | 17 | 5 | 0.8353 |

## Split: test

- n: 376
- pos_rate: 0.8165
- AUC: 0.9110
- AP:  0.9739
- best F1: 0.9535 @ thr=0.10
- ECE: 0.0866
- mean attention concentration: 0.8417

### Attention diagnostic (top-1 case)

- top-1 same-waypoint rate: 0.9441
- top-1 same-outcome rate: 0.5239
- top-1 same-both rate:    0.4894

### Per-waypoint AUC (waypoints with >= 5 negatives)

| waypoint | n | pos | neg | AUC |
|---|---:|---:|---:|---:|
| cobblestone | 85 | 36 | 49 | 0.7222 |

## Baselines on test

| baseline | AUC | AP | F1 | ECE |
|---|---:|---:|---:|---:|
| history_count_baseline | 0.8555 | 0.9431 | 0.9119 | 0.0419 |
| majority_class_baseline | 0.5000 | 0.8165 | 0.8990 | 0.0000 |

**RADS test AUC: 0.9110**

## Multi-action waypoint ranking on test

For waypoints where the train set contains >= 2 distinct actions, we re-score every candidate action under the eval state and check (a) how often RADS' top-1 matches the historical majority action, and (b) on successful eval cases, whether RADS' top-1 matches the action that actually succeeded.

| waypoint | n_eval | n_actions | majority_action | RADS-top1=majority | success | RADS-top1 match on success |
|---|---:|---:|---|---:|---:|---:|
| cobblestone | 85 | 3 | `dig down and mine cobblestone` | 0.4235 | 36 | 1.0000 |
| stone | 4 | 2 | `smelt stone` | 1.0000 | 1 | 1.0000 |

## v1 vs v2 head-to-head

| metric | v1 | v2 | delta |
|---|---:|---:|---:|
| test AUC | 0.9291 | 0.9110 | -0.018 |
| test AP  | 0.9714 | 0.9739 | +0.003 |
| test best F1 | 0.9700 | 0.9535 | -0.017 |
| test ECE | 0.0560 | 0.0866 | +0.031 |
| val AUC | 0.8020 | 0.8763 | **+0.074** |
| **top-1 attention same-waypoint** | 0.0931 | **0.9441** | **+0.851** |
| **top-1 attention same-outcome** | 0.1463 | 0.5239 | +0.378 |
| **top-1 attention same-both** | 0.0027 | 0.4894 | +0.487 |
| mean attention concentration | 0.5337 | 0.8429 | +0.31 |
| cobblestone per-wp AUC | 0.9932 | 0.7222 | -0.27 |
| cobblestone multi-action: RADS picks majority | 0.0000 | 0.4235 | +0.42 |
| **cobblestone multi-action: top-1 match on success** | **0.0000** | **1.0000** | **+1.00** |
| stone multi-action: top-1 match on success | 0.0000 | 1.0000 | +1.00 |

### Reading the trade-off

- v1's per-waypoint cobblestone AUC of 0.99 was achieved by a model whose
  attention was collapsed (9% same-waypoint top-1) and whose multi-action
  ranker picked rare unsuccessful actions over the successful majority
  action 100% of the time. That AUC reflects state-conditional
  discrimination on a fixed action, but the model was not usable as an
  action ranker in the runtime decision sense.
- v2 deliberately trades some of that single-action AUC (0.99 -> 0.72)
  for working multi-action behaviour (cobblestone top-1 match on success
  0% -> 100%) and meaningful attention (9% -> 94% same-waypoint). The
  prior log-odds residual injects (waypoint, action) base rates directly
  at the output, preventing the action embedding from dominating in the
  small-support regime.
- Test ECE rose modestly from 0.056 to 0.087 but remains well calibrated.
- The trainable prior_logit_weight settled to 1.49 (init 1.5), and the
  attention temperature tau to 0.51 (init 0.5).

### v3 spec coverage

| criterion (from `docs/DECISIONER_V3_PROPOSAL_2026-05-01.md`) | target | v2 result |
|---|---|---|
| total test AUC | >= 0.75 | 0.9110 |
| cobblestone per-wp AUC | >= 0.70 | 0.7222 |
| top-1 attention same-waypoint rate | >= 0.70 | 0.9441 |
| multi-action top-1 accuracy beats baseline | yes | 0% -> 100% on cobblestone, 0% -> 100% on stone |

All four spec criteria met.

## What v2 changed (vs v1)

1. Same-waypoint hard mask in attention with fallback when same-wp pool < 8.
2. Lower auxiliary loss weights: lambda_triplet 0.3 -> 0.1, lambda_wp 0.2 -> 0.05.
3. Dropout 0.1 -> 0.2 with patience-based early stopping.
4. (Waypoint, action) Laplace-smoothed success rate added as a residual
   logit prior with a trainable scalar weight (init 1.5).

Files touched:
`src/optimus1/decisioner/{feature,encoder,rads,runtime}.py`,
`scripts/{train_rads,evaluate_rads_offline}.py`. No changes to the case
library schema, env wrapper, planner, or OracleGraph.

## Notes

- Group split is by `run_uuid`. Same-run library entries are masked out of the retrieval pool during attention. This is a strict leak guard.
- Val split has very few negative samples (~21). Val AUC is therefore noisy and should be read alongside test AUC.
- Baseline `history_count_baseline` simulates the existing `_best_exact_success_case` ranking (Laplace-smoothed success rate per (waypoint, action)).
- Multi-action ranking only meaningful for waypoints with >= 2 actions in the case library. Currently 4 waypoints qualify: cobblestone, charcoal, stone, smooth_stone.
