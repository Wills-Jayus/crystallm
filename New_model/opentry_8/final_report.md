# opentry_8 Final Report

Protocol: CrystaLLM Table 3 official MP-20 and MPTS-52 test splits, 20 final candidate slots per official sample, StructureMatcher ltol=0.3/stol=0.5/angle_tol=10, and missing/failed candidates counted as failures.

The opentry_8 strategy/fusion line is a coverage-repaired frozen strategy. It preserves CrystaLLM-a GT-SG candidate order as the anchor and uses stablekey only to fill absent GT-SG slots. It does not use test StructureMatcher feedback, oracle selection, or post-test threshold changes.

The opentry_8 pure structural line was not official full-tested. No structural checkpoint passed a validation-generation gate in this run, so reporting a full-test pure result would be misleading.

## MP-20

| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | rows>=7 positive-any | candidate budget | fusion/ranking | pure model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| opentry_7 reproduced CrystaLLM-a composition-only | official full-test K20 | 62.17 | 74.98 | 81.84 | 0.0442 | 0.0403 | 0.0402 | 53.74 | 68.29 | 76.36 | 4607 | 20 | no | no |
| opentry_7 CrystaLLM-a GT-SG | official full-test K20 | 71.67 | 83.08 | 87.81 | 0.0509 | 0.0449 | 0.0431 | 62.37 | 76.35 | 82.61 | 4984 | 20 | no | no |
| opentry_7 strategy/fusion | official full-test K20, incomplete source coverage | 50.44 | 65.06 | 69.11 | 0.0670 | 0.0745 | 0.0745 | 36.13 | 52.83 | 57.33 | 3459 | 20 final | yes | no |
| opentry_7 pure model | official full-test K20 | 60.58 | 70.67 | 77.96 | 0.0726 | 0.0775 | 0.0744 | 47.72 | 58.78 | 67.88 | 4095 | 20 | no | yes |
| opentry_8 strategy/fusion | official full-test K20, 9046/9046 samples covered | 71.67 | 83.08 | 87.81 | 0.0509 | 0.0449 | 0.0431 | 62.37 | 76.35 | 82.61 | 4984 | 20 final | yes, fallback only | no |
| opentry_8 pure structural MVP | not official full-tested | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no | intended yes |

## MPTS-52

| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | rows>=7 positive-any | candidate budget | fusion/ranking | pure model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| opentry_7 reproduced CrystaLLM-a composition-only | official full-test K20 | 18.86 | 28.08 | 35.17 | 0.1056 | 0.1058 | 0.1174 | 16.69 | 25.71 | 32.76 | 2498 | 20 | no | no |
| opentry_7 CrystaLLM-a GT-SG | official full-test K20, 8095/8096 samples with candidates | 25.23 | 36.46 | 43.96 | 0.1211 | 0.1257 | 0.1334 | 22.49 | 33.37 | 41.04 | 3130 | 20 | no | no |
| opentry_7 strategy/fusion | official full-test K20, incomplete source coverage | 17.63 | 27.99 | 32.99 | 0.1326 | 0.1370 | 0.1419 | 15.50 | 24.69 | 29.61 | 2258 | 20 final | yes | no |
| opentry_7 pure model | official full-test K20 | 17.18 | 24.35 | 31.52 | 0.1655 | 0.1702 | 0.1746 | 14.32 | 20.94 | 27.88 | 2126 | 20 | no | yes |
| opentry_8 strategy/fusion | official full-test K20, 8096/8096 samples covered | 25.23 | 36.46 | 43.97 | 0.1211 | 0.1257 | 0.1334 | 22.49 | 33.37 | 41.04 | 3130 | 20 final | yes, fallback only | no |
| opentry_8 pure structural MVP | not official full-tested | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no | intended yes |

## Answers

- Does opentry_8 strategy/fusion exceed CrystaLLM-a GT-SG? No in a meaningful sense. MP-20 is identical to the GT-SG anchor. MPTS-52 only changes coverage from 8095/8096 to 8096/8096 and match@20 moves from 43.96 to 43.97.
- Does opentry_8 strategy/fusion exceed opentry_7 strategy/fusion? Yes. MP-20 match@20 improves from 69.11 to 87.81, and MPTS-52 match@20 improves from 32.99 to 43.97. This comes from coverage repair and using the stronger GT-SG anchor, not from a new W/A or geometry breakthrough.
- Does opentry_8 pure structural model exceed opentry_7 pure or CrystaLLM-a GT-SG? Not evaluated. No opentry_8 structural checkpoint passed validation-generation gates, so official full test was not run.
- Where does the improvement come from? Coverage and anchor choice. opentry_7 strategy used missing stablekey top20 as the final backbone. opentry_8 uses full GT-SG K20 coverage and only uses stablekey fallback for absent GT-SG slots. It is not evidence of improved W/A, geometry, or ranking.
- Which results are final official full-test K20? All numeric rows in the two tables are official full-test K20 under the unified evaluator. The opentry_8 pure structural MVP row is not a full-test result.

## Artifacts

- `experiment_log.md`
- `configs/unified_evaluator.json`
- `frozen_strategy/config.json`
- `frozen_strategy/strategy_fusion_build_manifest.json`
- `generations/strategy_fusion_mp_20_test_k20_candidates.jsonl`
- `generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl`
- `generations/strategy_fusion_mp_20_test_k20.jsonl`
- `generations/strategy_fusion_mpts_52_test_k20.jsonl`
- `metrics/strategy_fusion_mp_20_test_k20.json`
- `metrics/strategy_fusion_mpts_52_test_k20.json`
- `frozen_pure_model/pure_structural_mvp_status.json`
- `reports/pure_structural_mvp_report.md`

