# MPTS-52 Fast K50 Score Route Sweep

Created: 2026-06-25T04:42:31+00:00

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
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.7 | 1500 | 2.420 pp | 0.654 pp | 1.200 pp | -0.044 pp | 0.580 pp | -0.131 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.5 | 2500 | 2.340 pp | 0.829 pp | 0.920 pp | -0.218 pp | 0.520 pp | -0.175 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.9 | 500 | 2.120 pp | 0.567 pp | 0.740 pp | 0.000 pp | 0.220 pp | 0.000 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.6 | 2000 | 2.120 pp | 0.698 pp | 0.660 pp | 0.044 pp | -0.020 pp | -0.044 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.9 | 500 | 2.060 pp | 0.611 pp | 0.700 pp | -0.087 pp | 0.280 pp | 0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.9 | 500 | 2.060 pp | 0.524 pp | 0.740 pp | -0.087 pp | 0.240 pp | 0.044 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.6 | 2000 | 2.040 pp | 0.480 pp | 1.140 pp | -0.131 pp | 0.840 pp | 0.087 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.6 | 2000 | 2.040 pp | 0.654 pp | 1.140 pp | -0.175 pp | 0.500 pp | -0.087 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.6 | 2000 | 1.940 pp | 0.611 pp | 1.120 pp | -0.131 pp | 0.580 pp | -0.175 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.6 | 2000 | 1.860 pp | 0.698 pp | 0.560 pp | 0.000 pp | 0.040 pp | -0.044 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.45 | 2750 | 1.840 pp | 0.654 pp | 0.980 pp | -0.349 pp | 0.580 pp | -0.175 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_score_ge_q0.6 | 2000 | 1.740 pp | 0.611 pp | 0.680 pp | -0.087 pp | 0.040 pp | -0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.7 | 1500 | 1.580 pp | 0.131 pp | 0.120 pp | -0.131 pp | -0.060 pp | 0.000 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_score_ge_q0.7 | 1500 | 1.320 pp | 0.175 pp | 0.080 pp | -0.131 pp | -0.020 pp | 0.000 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_le_20 | 1940 | 1.200 pp | 0.218 pp | 0.520 pp | -0.218 pp | 0.120 pp | -0.175 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_score_ge_q0.7 | 1500 | 1.180 pp | 0.131 pp | 0.200 pp | -0.175 pp | -0.040 pp | 0.000 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_rank_ge_20 | 3145 | 1.020 pp | 0.393 pp | 0.560 pp | -0.480 pp | 0.560 pp | -0.175 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.540 pp | 0.654 pp | 0.960 pp | -0.087 pp | 0.340 pp | -0.218 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.440 pp | 0.654 pp | 0.860 pp | -0.131 pp | 0.260 pp | -0.305 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.3 | 3500 | 2.400 pp | 0.829 pp | 1.160 pp | -0.480 pp | 0.640 pp | -0.393 pp |
| mpts52_val_k50_rf_seed0_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.8 | 1000 | 2.400 pp | 0.567 pp | 0.780 pp | -0.305 pp | 0.320 pp | -0.218 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.25 | 3750 | 2.360 pp | 0.829 pp | 1.140 pp | -0.524 pp | 0.740 pp | -0.175 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.2 | 4000 | 2.320 pp | 0.829 pp | 1.100 pp | -0.611 pp | 0.700 pp | -0.262 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.35 | 3250 | 2.320 pp | 0.611 pp | 1.220 pp | -0.567 pp | 0.680 pp | -0.262 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.3 | 3500 | 2.320 pp | 0.611 pp | 1.200 pp | -0.698 pp | 0.680 pp | -0.349 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.25 | 3750 | 2.300 pp | 0.567 pp | 1.100 pp | -0.785 pp | 0.680 pp | -0.393 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.2 | 4000 | 2.300 pp | 0.611 pp | 1.140 pp | -0.698 pp | 0.620 pp | -0.436 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.35 | 3250 | 2.280 pp | 0.654 pp | 1.200 pp | -0.436 pp | 0.620 pp | -0.393 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.45 | 2750 | 2.280 pp | 0.829 pp | 1.020 pp | -0.305 pp | 0.480 pp | -0.349 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.7 | 1500 | 2.280 pp | 0.742 pp | 1.120 pp | 0.044 pp | 0.380 pp | -0.218 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.4 | 3000 | 2.260 pp | 0.611 pp | 1.200 pp | -0.480 pp | 0.760 pp | -0.087 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.15 | 4250 | 2.240 pp | 0.654 pp | 1.100 pp | -0.611 pp | 0.700 pp | -0.262 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.4 | 3000 | 2.240 pp | 0.611 pp | 1.020 pp | -0.524 pp | 0.520 pp | -0.480 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.15 | 4250 | 2.220 pp | 0.567 pp | 1.160 pp | -0.698 pp | 0.660 pp | -0.349 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.35 | 3250 | 2.220 pp | 0.916 pp | 1.160 pp | -0.436 pp | 0.440 pp | -0.305 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.08 | 4600 | 2.200 pp | 0.567 pp | 1.060 pp | -0.698 pp | 0.660 pp | -0.349 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_score_ge_q0.1 | 4500 | 2.200 pp | 0.567 pp | 1.060 pp | -0.698 pp | 0.660 pp | -0.349 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.08 | 4600 | 2.200 pp | 0.524 pp | 1.120 pp | -0.698 pp | 0.580 pp | -0.480 pp |
| mpts52_val_k50_rf_seed1_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.05 | 4750 | 2.200 pp | 0.524 pp | 1.100 pp | -0.742 pp | 0.580 pp | -0.480 pp |
| mpts52_val_k50_rf_seed2_scores.jsonl | unconstrained | None | best_minus_rank1_ge_q0.4 | 3000 | 2.200 pp | 1.003 pp | 1.200 pp | -0.305 pp | 0.480 pp | -0.218 pp |

## Best Gate Candidate

- Scores: /data/users/xsw/autodlmini/model/New_model/opentry_10/features/mpts52_k50_selector_oof_scores_rf_seed1/mpts52_val_k50_rf_seed1_scores.jsonl
- Model: rf seed=1
- Strategy: unconstrained anchor_keep=None
- Route: best_minus_rank1_ge_q0.7 threshold=0.12608004456999994
- Routed samples: 1500
- match@1 delta: 2.420 pp; rows>=7: 0.654 pp
- match@5 delta: 1.200 pp; rows>=7: -0.044 pp
- match@20 delta: 0.580 pp; rows>=7: -0.131 pp
