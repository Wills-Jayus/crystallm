# MP-20-hgb-ensemble Fast K50 Score Route Sweep

Created: 2026-06-25T15:37:36+00:00

Validation-only sweep from saved OOF scores. Samples not routed keep baseline K20 order.

## Baseline

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 72.731% | 29.135% |
| match@5 | 83.088% | 44.291% |
| match@20 | 87.631% | 55.571% |

## Top Routes

| score file | strategy | anchor_keep | rule | routed | d@1 | rows7 d@1 | d@5 | rows7 d@5 | d@20 | rows7 d@20 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.9 | 905 | 1.249 pp | 0.900 pp | -0.077 pp | 0.000 pp | -0.066 pp | -0.138 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | unconstrained | None | best_rank_le_5 | 3001 | 0.464 pp | -0.069 pp | -0.144 pp | 0.346 pp | -0.122 pp | -0.138 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.5 | 4524 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.420 pp | 1.246 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_rank_ge_3 | 7015 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.409 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.4 | 3619 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.409 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.2 | 7237 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.25 | 6785 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_rank_ge_2 | 7578 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.55 | 4976 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.246 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.75 | 6785 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.65 | 5880 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.6 | 5428 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.398 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.3 | 6333 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.387 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.35 | 5880 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.387 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_rank_ge_5 | 6319 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.387 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.7 | 6333 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.387 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.4 | 5428 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.315 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_ge_q0 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q1 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.98 | 8870 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.95 | 8594 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.92 | 8329 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.9 | 8143 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.85 | 7690 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.8 | 7237 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_le_q1 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.02 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.05 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.08 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.1 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.15 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.5 | 4524 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.6 | 3619 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_rank_le_50 | 9047 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.376 pp | 1.176 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | unconstrained | None | best_rank_le_3 | 2408 | 0.376 pp | -0.138 pp | -0.066 pp | 0.346 pp | -0.111 pp | -0.208 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_score_le_q0.3 | 2714 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.365 pp | 1.246 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 12 | best_minus_rank1_ge_q0.45 | 4976 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.354 pp | 1.107 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 14 | best_minus_rank1_ge_q0.2 | 7237 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.343 pp | 0.969 pp |
| mp20_val_k50_hgb_mean_seed012_scores.jsonl | conservative | 14 | best_minus_rank1_ge_q0.25 | 6785 | 0.000 pp | 0.000 pp | 0.000 pp | 0.000 pp | 0.343 pp | 0.969 pp |

## Best Gate Candidate

- Scores: /data/users/xsw/autodlmini/model/New_model/opentry_10/features/k50_selector_oof_scores_ensemble/mp20_val_k50_hgb_mean_seed012_scores.jsonl
- Model: hgb_mean_seed012 seed=12
- Strategy: unconstrained anchor_keep=None
- Route: best_minus_rank1_ge_q0.9 threshold=0.20066793518000028
- Routed samples: 905
- match@1 delta: 1.249 pp; rows>=7: 0.900 pp
- match@5 delta: -0.077 pp; rows>=7: 0.000 pp
- match@20 delta: -0.066 pp; rows>=7: -0.138 pp
