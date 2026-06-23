# MP-20 Test Hybrid-Prior K5 Evaluation

## Setup

- Dataset: MP-20 structured test split, `data/structured_symcif_v4_mp20/test.jsonl`
- Samples: 8,893 structured test samples
- Method: frozen train-only `hybrid_prior_train_only` WA selector from the MP-20 K<=5 dev round
- Raw candidate pool: `reports/symcif_v4_table3_fix_audit/wa_search_audit_full_stablekey/mp20/hybrid_top700_full/test_hybrid_candidates.jsonl`
- Raw cap: first 100 candidates per sample before selector rerank
- Renderer/evaluator: `run_symcif_v4_geometry_model_eval.py`, full mode, top-k 5, 64 eval workers, CUDA device for geometry model
- No test-label tuning was done in this run.

## Main Result

| scope | match@1 | match@5 | RMSE@1 | RMSE@5 | matched@1 | matched@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structured MP-20 test | 60.90% | 73.68% | 0.0573 | 0.0615 | 5416/8893 | 6552/8893 |
| original-test adjusted denominator | 59.87% | 72.43% | 0.0573 | 0.0615 | 5416/9046 | 6552/9046 |

The second row treats the 153 non-structured original MP-20 test samples as failures, matching the older full-test denominator convention. RMSE is unchanged because it is averaged over matched samples with available RMS.

## Candidate Coverage

| selector | WA_hit@1 | WA_hit@5 | skeleton_hit@1 | skeleton_hit@5 |
| --- | ---: | ---: | ---: | ---: |
| previous/current order | 49.80% | 71.27% | - | - |
| frozen hybrid-prior selector | 59.02% | 81.66% | 71.36% | 84.45% |

The new selector mainly improves the WA candidate coverage. The observed match@5 is 73.68%, still below WA_hit@5 = 81.66%, so the remaining gap is mostly geometry/render/evaluator quality rather than candidate absence.

## Comparison

| reference | match@1 | match@5 | RMSE@1 | RMSE@5 |
| --- | ---: | ---: | ---: | ---: |
| previous MP-20 structured WA5_geom4 current order | 51.29% | 66.11% | 0.0668 | 0.0742 |
| this run, structured test | 60.90% | 73.68% | 0.0573 | 0.0615 |
| this run, original-test adjusted | 59.87% | 72.43% | 0.0573 | 0.0615 |
| baseline threshold mentioned earlier | 55.58% | - | - | - |

Against the previous structured WA5_geom4 order, this is +9.61 pp match@1, +7.56 pp match@5, and lower RMSE at both @1 and @5. Against the 55.58 match@1 baseline threshold, this run is above baseline under both structured-test and original-test-adjusted denominators.

## Validity And Failure Notes

| metric | @1 | @5 |
| --- | ---: | ---: |
| readable | 97.72% | 82.75% |
| formula_ok | 97.72% | 82.75% |
| atom_count_ok | 100.00% | 99.10% |
| SG_ok | 97.72% | 82.75% |
| strict_valid attempt rate | 74.01% | 50.01% |
| strict_valid any@k | 74.01% | 87.24% |
| eval_timeout | 0.00% | 0.00% |

The evaluator reported no parse/SG/matcher/sample timeouts at @1 or @5. Top candidate-level errors were invalid CIF/no readable structure and missing candidates. This means the run itself looks mechanically clean; the main remaining bottleneck is not timeout or evaluation failure, but generated geometry quality and candidate validity after the improved WA selector.

## Artifacts

- Predictions: `reports/symcif_v4_mp20_test_hybrid_prior_k5/test_one_fix_hybrid_prior_predictions.jsonl`
- Build audit: `reports/symcif_v4_mp20_test_hybrid_prior_k5/test_hybrid_prior_build_audit.json`
- Full eval summary: `reports/symcif_v4_mp20_test_hybrid_prior_k5/eval_gpu/full_eval_summary.json`
- Full eval breakdown: `reports/symcif_v4_mp20_test_hybrid_prior_k5/eval_gpu/full_eval_breakdown.csv`
- Metrics: `reports/symcif_v4_mp20_test_hybrid_prior_k5/eval_gpu/metrics/baseline_per_generation_metrics.jsonl`
