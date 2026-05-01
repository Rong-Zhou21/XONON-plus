# RADS Offline Evaluation v1

- artifact: `artifacts/decisioner/rads_v1.pt`
- data: `data/decisioner/rads_v1.jsonl`
- spec: waypoints=73, goals=68, actions=79, feature_dim=52
- library size (train cases): 1847

## Split: val

- n: 351
- pos_rate: 0.9402
- AUC: 0.8020
- AP:  0.9813
- best F1: 0.9435 @ thr=0.20
- ECE: 0.1110
- mean attention concentration: 0.5428

### Attention diagnostic (top-1 case)

- top-1 same-waypoint rate: 0.0769
- top-1 same-outcome rate: 0.2194
- top-1 same-both rate:    0.0000

### Per-waypoint AUC (waypoints with >= 5 negatives)

| waypoint | n | pos | neg | AUC |
|---|---:|---:|---:|---:|
| logs | 36 | 31 | 5 | 0.8452 |
| iron_ore | 22 | 17 | 5 | 0.8000 |

## Split: test

- n: 376
- pos_rate: 0.8165
- AUC: 0.9291
- AP:  0.9714
- best F1: 0.9700 @ thr=0.15
- ECE: 0.0560
- mean attention concentration: 0.5337

### Attention diagnostic (top-1 case)

- top-1 same-waypoint rate: 0.0931
- top-1 same-outcome rate: 0.1463
- top-1 same-both rate:    0.0027

### Per-waypoint AUC (waypoints with >= 5 negatives)

| waypoint | n | pos | neg | AUC |
|---|---:|---:|---:|---:|
| cobblestone | 85 | 36 | 49 | 0.9932 |

## Baselines on test

| baseline | AUC | AP | F1 | ECE |
|---|---:|---:|---:|---:|
| history_count_baseline | 0.8555 | 0.9431 | 0.9119 | 0.0419 |
| majority_class_baseline | 0.5000 | 0.8165 | 0.8990 | 0.0000 |

**RADS test AUC: 0.9291**

## Multi-action waypoint ranking on test

For waypoints where the train set contains >= 2 distinct actions, we re-score every candidate action under the eval state and check (a) how often RADS' top-1 matches the historical majority action, and (b) on successful eval cases, whether RADS' top-1 matches the action that actually succeeded.

| waypoint | n_eval | n_actions | majority_action | RADS-top1=majority | success | RADS-top1 match on success |
|---|---:|---:|---|---:|---:|---:|
| cobblestone | 85 | 3 | `dig down and mine cobblestone` | 0.0000 | 36 | 0.0000 |
| stone | 4 | 2 | `smelt stone` | 1.0000 | 1 | 1.0000 |

## Discussion

### What works

- Test AUC 0.9291 vs `history_count_baseline` 0.8555 (+0.074 absolute). The retrieval baseline is the function currently used in `case_memory._best_exact_success_case()`, so this is a direct proxy for "would RADS rank cases better than the existing logic on held-out runs?". Answer: yes.
- Test ECE 0.056. Predicted probabilities are reasonably calibrated, which matters for downstream confidence thresholding.
- **cobblestone per-waypoint AUC 0.9932** on 85 test cases (36 pos / 49 neg). This is the only waypoint with both significant negatives and multiple distinct actions in the library, so it is the most informative single number. RADS correctly distinguishes high-success states (ypos≈64) from low-success states (ypos≈40) almost perfectly.
- Val AUC 0.80 is lower but val has only 21 negatives across all waypoints; the metric is high variance.

### What raises a flag

- **Top-1 attention same-waypoint rate is 9.3% on test** (and 7.7% on val). The attention is not behaving like a case retriever. The model attains its AUC mostly through the query encoder; the cross-attention context vector has collapsed to summarising a few "popular" library cases regardless of query waypoint. Options for the next round:
  - Stronger waypoint reconstruction weight, or a margin-based attention regulariser that explicitly pushes top-attended cases toward same-waypoint.
  - Two-stage retrieval: hard-restrict the attention pool to same-waypoint cases (with cross-waypoint as a fallback when the pool is small).
  - Investigate whether attention temperature is collapsing (`tau` is trainable; check final value).
- **Multi-action ranking on cobblestone is 0% match with the historical majority action** (`dig down and mine cobblestone`). Yet per-state P(success) for the dig-down action tracks the truth (AUC 0.99). This means: when scoring the same state with different actions, the rare actions (`craft cobblestone`, `smelt cobblestone`) get a higher score than the dominant action. Both rare actions only have 2 train cases each, both failed, so the model has not seen successful counterexamples and overestimates them via cross-state generalisation. This is exactly the "70/74 single-action waypoint" data limitation showing up. Two pragmatic mitigations:
  - At runtime restrict the candidate set to actions with ≥ N train successes for that waypoint (filter step before RADS rerank).
  - Up-weight loss on under-represented actions, or explicitly down-weight predictions for rare actions.

### Comparison to baselines

| metric | majority | history_count | RADS |
|---|---:|---:|---:|
| test AUC | 0.5000 | 0.8555 | **0.9291** |
| test AP  | 0.8165 | 0.9431 | **0.9714** |
| test F1  | 0.8990 | 0.9119 | **0.9700** |
| test ECE | 0.0000 | 0.0419 | 0.0560 |

`history_count_baseline` is well calibrated by construction (its prediction is empirical success rate); RADS gives up a bit of calibration to gain a meaningful chunk of ranking power.

### Recommended next round (out of scope this round)

1. Add an attention regulariser to push top-attended cases toward same-waypoint; re-evaluate top-1 attention rate and per-waypoint AUC.
2. At runtime, filter candidate actions by minimum train support before RADS rerank, to avoid the "rare action overestimation" failure mode observed on cobblestone.
3. Plumb the runtime hook in `case_memory.select_case_decision()` behind a config switch (`decisioner.enabled: false` by default), and run a v3 in-game A/B against v2 retrieval-only.

## Notes

- Group split is by `run_uuid`. Same-run library entries are masked out of the retrieval pool during attention. This is a strict leak guard.
- Val split has very few negative samples (~21). Val AUC is therefore noisy and should be read alongside test AUC.
- Baseline `history_count_baseline` simulates the existing `_best_exact_success_case` ranking (Laplace-smoothed success rate per (waypoint, action)).
- Multi-action ranking only meaningful for waypoints with >= 2 actions in the case library. In the train split four waypoints qualify, but only `cobblestone` and `stone` appear in the test split.
