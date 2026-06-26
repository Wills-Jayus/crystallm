# MPTS-52-K30 Fast K50 Score Route Sweep

Created: 2026-06-25T06:14:16+00:00

Validation-only sweep from saved OOF scores. Samples not routed keep baseline K20 order.

## Baseline

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 30.020% | 5.323% |
| match@5 | 40.480% | 9.991% |
| match@20 | 48.000% | 14.747% |

## Top Routes

| score file | strategy | anchor_keep | rule | routed | d@1 | rows7 d@1 | d@5 | rows7 d@5 | d@20 | rows7 d@20 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.15 | 4250 | 2.940 pp | 0.698 pp | 1.680 pp | 0.000 pp | 0.680 pp | 0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.2 | 4000 | 2.940 pp | 0.611 pp | 1.720 pp | -0.131 pp | 0.640 pp | 0.000 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.05 | 4750 | 2.920 pp | 0.654 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.08 | 4600 | 2.920 pp | 0.654 pp | 1.640 pp | -0.087 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.1 | 4500 | 2.920 pp | 0.654 pp | 1.640 pp | -0.087 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_le_q1 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.02 | 4900 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_le_q0.95 | 4750 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_le_q1 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.02 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_le_50 | 5000 | 2.900 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.15 | 4250 | 2.900 pp | 0.567 pp | 1.720 pp | -0.087 pp | 0.640 pp | 0.000 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.05 | 4750 | 2.900 pp | 0.611 pp | 1.700 pp | -0.087 pp | 0.600 pp | -0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_ge_2 | 4774 | 2.900 pp | 0.611 pp | 1.700 pp | -0.087 pp | 0.600 pp | -0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.08 | 4600 | 2.900 pp | 0.567 pp | 1.700 pp | -0.087 pp | 0.600 pp | -0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_le_q0.98 | 4900 | 2.880 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.1 | 4500 | 2.880 pp | 0.567 pp | 1.720 pp | -0.044 pp | 0.600 pp | -0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_ge_3 | 4616 | 2.880 pp | 0.567 pp | 1.620 pp | -0.131 pp | 0.500 pp | -0.131 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.5 | 2500 | 2.860 pp | 0.785 pp | 1.180 pp | -0.131 pp | 0.440 pp | -0.131 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.45 | 2750 | 2.820 pp | 1.047 pp | 1.700 pp | -0.087 pp | 0.840 pp | 0.131 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.820 pp | 0.567 pp | 1.240 pp | 0.000 pp | 0.320 pp | -0.087 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.2 | 4000 | 2.800 pp | 1.047 pp | 1.660 pp | -0.175 pp | 0.940 pp | 0.305 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_le_q0.92 | 4600 | 2.800 pp | 0.611 pp | 1.660 pp | -0.044 pp | 0.660 pp | 0.044 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.1 | 4500 | 2.780 pp | 0.916 pp | 1.760 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.3 | 3500 | 2.780 pp | 0.960 pp | 1.700 pp | -0.175 pp | 0.880 pp | 0.218 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_ge_5 | 4266 | 2.780 pp | 0.611 pp | 1.600 pp | -0.175 pp | 0.540 pp | 0.131 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.780 pp | 0.698 pp | 1.180 pp | -0.044 pp | 0.420 pp | -0.044 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.1 | 4500 | 2.760 pp | 0.960 pp | 1.740 pp | 0.000 pp | 0.940 pp | 0.305 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.08 | 4600 | 2.760 pp | 0.960 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.08 | 4600 | 2.760 pp | 0.916 pp | 1.760 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.15 | 4250 | 2.760 pp | 1.003 pp | 1.780 pp | 0.000 pp | 0.920 pp | 0.218 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.760 pp | 0.567 pp | 1.300 pp | 0.131 pp | 0.300 pp | -0.087 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.15 | 4250 | 2.740 pp | 0.916 pp | 1.700 pp | -0.087 pp | 0.940 pp | 0.305 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0 | 5000 | 2.740 pp | 0.916 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_le_q1 | 5000 | 2.740 pp | 0.916 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.02 | 4900 | 2.740 pp | 0.916 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.05 | 4750 | 2.740 pp | 0.916 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0 | 5000 | 2.740 pp | 0.916 pp | 1.740 pp | 0.000 pp | 0.920 pp | 0.262 pp |

## Best Gate Candidate

- Scores: /data/users/xsw/autodlmini/model/New_model/opentry_10/features/mpts52_k50_selector_oof_scores_rf_seed1/mpts52_val_k50_rf_seed1_scores.jsonl
- Model: rf seed=1
- Strategy: unconstrained anchor_keep=None
- Route: best_score_ge_q0.15 threshold=0.03690556341
- Routed samples: 4250
- match@1 delta: 2.940 pp; rows>=7: 0.698 pp
- match@5 delta: 1.680 pp; rows>=7: 0.000 pp
- match@20 delta: 0.680 pp; rows>=7: 0.087 pp
