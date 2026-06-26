# MPTS-52-rf-ensemble Fast K50 Score Route Sweep

Created: 2026-06-25T15:37:34+00:00

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
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.7 | 1500 | 2.360 pp | 0.698 pp | 1.000 pp | -0.087 pp | 0.400 pp | -0.175 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.9 | 500 | 2.180 pp | 0.611 pp | 0.720 pp | -0.087 pp | 0.260 pp | 0.087 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.6 | 2000 | 2.160 pp | 0.698 pp | 0.680 pp | -0.044 pp | 0.020 pp | -0.044 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.6 | 2000 | 2.000 pp | 0.698 pp | 1.080 pp | -0.175 pp | 0.520 pp | -0.087 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.7 | 1500 | 1.520 pp | 0.131 pp | 0.160 pp | -0.175 pp | -0.020 pp | 0.000 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_rank_ge_20 | 3090 | 1.400 pp | 0.742 pp | 0.540 pp | -0.480 pp | 0.460 pp | -0.175 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_rank_ge_21 | 2996 | 1.320 pp | 0.654 pp | 0.460 pp | -0.524 pp | 0.420 pp | -0.175 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.3 | 3500 | 2.460 pp | 1.003 pp | 1.000 pp | -0.654 pp | 0.440 pp | -0.393 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.460 pp | 0.654 pp | 0.820 pp | -0.262 pp | 0.280 pp | -0.305 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.25 | 3750 | 2.440 pp | 0.960 pp | 1.020 pp | -0.654 pp | 0.500 pp | -0.262 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.2 | 4000 | 2.420 pp | 1.003 pp | 1.000 pp | -0.698 pp | 0.400 pp | -0.436 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.4 | 3000 | 2.380 pp | 0.916 pp | 1.060 pp | -0.567 pp | 0.520 pp | -0.218 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.35 | 3250 | 2.380 pp | 0.829 pp | 1.040 pp | -0.698 pp | 0.500 pp | -0.349 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.15 | 4250 | 2.380 pp | 0.916 pp | 1.000 pp | -0.698 pp | 0.420 pp | -0.393 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.3 | 3500 | 2.360 pp | 0.873 pp | 1.000 pp | -0.742 pp | 0.460 pp | -0.393 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.35 | 3250 | 2.340 pp | 0.742 pp | 1.000 pp | -0.567 pp | 0.420 pp | -0.436 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.08 | 4600 | 2.340 pp | 0.829 pp | 0.980 pp | -0.742 pp | 0.380 pp | -0.480 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.1 | 4500 | 2.340 pp | 0.829 pp | 0.980 pp | -0.742 pp | 0.380 pp | -0.480 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.4 | 3000 | 2.340 pp | 0.742 pp | 1.000 pp | -0.611 pp | 0.380 pp | -0.480 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.25 | 3750 | 2.320 pp | 0.785 pp | 0.980 pp | -0.829 pp | 0.400 pp | -0.567 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_rank_ge_2 | 4852 | 2.320 pp | 0.785 pp | 1.000 pp | -0.742 pp | 0.380 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_le_q1 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.02 | 4900 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.05 | 4750 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_le_q0.95 | 4750 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_le_q1 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.02 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_rank_le_50 | 5000 | 2.320 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.5 | 2500 | 2.300 pp | 0.829 pp | 0.980 pp | -0.305 pp | 0.420 pp | -0.218 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.05 | 4750 | 2.300 pp | 0.785 pp | 0.960 pp | -0.742 pp | 0.380 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_le_q0.98 | 4900 | 2.300 pp | 0.785 pp | 0.960 pp | -0.785 pp | 0.360 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_ge_q0.45 | 2750 | 2.300 pp | 0.785 pp | 0.900 pp | -0.436 pp | 0.320 pp | -0.393 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.08 | 4600 | 2.280 pp | 0.742 pp | 0.980 pp | -0.742 pp | 0.380 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.2 | 4000 | 2.280 pp | 0.742 pp | 0.940 pp | -0.829 pp | 0.340 pp | -0.654 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_rank_ge_3 | 4757 | 2.280 pp | 0.698 pp | 1.000 pp | -0.742 pp | 0.300 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.1 | 4500 | 2.260 pp | 0.742 pp | 0.960 pp | -0.742 pp | 0.360 pp | -0.567 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_score_le_q0.92 | 4600 | 2.240 pp | 0.785 pp | 0.980 pp | -0.785 pp | 0.380 pp | -0.524 pp |
| mpts52_val_k50_rf_mean_seed012_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.15 | 4250 | 2.240 pp | 0.742 pp | 0.980 pp | -0.742 pp | 0.380 pp | -0.567 pp |

## Best Gate Candidate

- Scores: /data/users/xsw/autodlmini/model/New_model/opentry_10/features/mpts52_k50_selector_oof_scores_ensemble/mpts52_val_k50_rf_mean_seed012_scores.jsonl
- Model: rf_mean_seed012 seed=12
- Strategy: unconstrained anchor_keep=None
- Route: best_minus_rank1_ge_q0.7 threshold=0.12348572495999996
- Routed samples: 1500
- match@1 delta: 2.360 pp; rows>=7: 0.698 pp
- match@5 delta: 1.000 pp; rows>=7: -0.087 pp
- match@20 delta: 0.400 pp; rows>=7: -0.175 pp
