# MP-20 Selector Branch Status

Timestamp: 2026-06-24 11:26 UTC

## Completed Evidence

The MP-20 K50 label pass is complete:

- Candidate-label rows: 452350 / 452350.
- Materials: 9047 / 9047.
- Material hard timeouts: 6.
- Candidate label status counts: 447453 ok, 4589 parse_error, 300 material_hard_timeout, 6 rmsd_error, 2 rmsd_timeout.

Per-sample K50 labels confirm that the K50 pool has real oracle headroom:

| budget | match_rate | RMSE | rows>=7 match_rate |
| --- | ---: | ---: | ---: |
| K1 | 0.7273129214 | 0.0514681112 | 0.2913494810 |
| K5 | 0.8308831657 | 0.0436475313 | 0.4429065744 |
| K20 | 0.8763125898 | 0.0404914181 | 0.5557093426 |
| K50 | 0.8991931027 | 0.0400885497 | 0.6207612457 |

K50 rescues 207 complete-validation materials over K20. In the rows>=7 bucket, K50 improves over K20 by +6.505pp, so the expanded pool is especially relevant for larger structures.

## Rerank-Only OOF Result

The first Phase 2A rerank-only OOF search is complete for logistic regression and HistGradientBoosting, with 5-fold GroupKFold by sample_id and 3 seeds.

Best current result:

- Model: HistGradientBoosting seed 0.
- match@1: 0.7343870896, delta +0.707pp.
- match@5: 0.8296672930, delta -0.122pp.
- match@20: 0.8763125898, delta 0.000pp.
- rows>=7 match@1: delta +0.554pp.
- rows>=7 match@5: delta +1.246pp.
- rows>=7 match@20: delta 0.000pp.
- bootstrap 95% CI for match@1 delta: +0.044pp to +1.349pp.

This is useful signal but not enough to freeze a rerank-only official-test system. It does not meet the >=1pp match improvement target, and match@5 declines slightly on all tested HGB seeds.

## Decision

Do not freeze rerank-only yet.

Proceed to a K50 conservative selector/fusion OOF search:

- preserve a fixed anchor prefix or anchor quota;
- allow only tail-slot replacement from ranks 21-50;
- gate on full validation OOF match@1/5/20 and rows>=7;
- require match@20 not to drop materially.

MPTS-52 validation shard generation should continue in parallel.

## K50 Conservative Selector Result

The first K50 conservative selector OOF search is complete for HistGradientBoosting seeds 0/1/2. A full anchor_keep sweep from 0 to 20 was also evaluated using the OOF scores.

Best result:

- Model: HistGradientBoosting seed 0.
- Strategy: conservative K50 selector with anchor_keep=14.
- match@1 delta: 0.000pp.
- match@5 delta: 0.000pp.
- match@20 delta: +0.343pp.
- rows>=7 match@20 delta: +0.969pp.
- mean supplemental slots per sample: 3.481.

The selector confirms that the K50 rescue signal is usable: replacing tail K20 slots can improve match@20 without hurting match@1/5. However, the improvement is still below the formal >=1pp total-match threshold, so this is not yet a freeze candidate.

Updated decision:

- Do not freeze the current MP-20 strategy.
- The next MP-20 strategy step should be residual/routed fusion rather than fixed global anchor_keep. The fixed quota leaves most of the +2.288pp K50 oracle headroom unrecovered.
- MPTS-52 remains important because MP-20 alone has not yet produced a validation-approved frozen system.

## Residual Route Sweep

A lightweight route sweep was run from the saved HGB OOF scores:

- report: `reports/mp20_k50_residual_route_sweep_from_hgb_scores.md`;
- metrics: `metrics/mp20_k50_residual_route_sweep_from_hgb_scores.json`;
- routed strategies used best supplemental score thresholds with keep_if_routed in 5/8/10/12/14/16/18.

The best routed strategy is still equivalent to fixed keep=14 on all samples. Thresholding the route reduces the number of modified samples but also reduces the match@20 gain. This means the current HGB candidate score is not sufficient as a residual router; it can rank tail replacements but does not reliably identify which samples should be routed.

Updated MP-20 conclusion:

- K50 contains useful additional correct candidates.
- Simple rerank-only is insufficient.
- Fixed conservative tail replacement is positive but too small for the formal gate.
- Score-threshold residual routing does not improve over fixed keep=14.
- A stronger sample-level residual classifier, new candidate source, or MPTS/joint evidence is needed before freezing.

## Sample-Level Residual Classifier

A sample-level OOF residual classifier was tested from the HGB K50 OOF scores:

- report: `reports/mp20_k50_residual_classifier_oof.md`;
- metrics: `metrics/mp20_k50_residual_classifier_oof.json`;
- target: predict samples where keep14 tail replacement rescues match@20.

Observed labels for keep14:

- benefit samples: 68;
- harm samples: 37.

Best routed classifier:

- model: HGB sample classifier;
- routed samples: 4524;
- match@20 delta: +0.376pp;
- rows>=7 match@20 delta: +0.900pp;
- match@1/5 unchanged by construction.

This is a small improvement over fixed keep14 (+0.343pp), but still below the formal >=1pp match gate. The route improves precision slightly, but the recoverable rescue count from this candidate score remains too small.

MP-20 branch conclusion at this point:

- Validation evidence is scientifically useful and shows real K50 headroom.
- Current selector methods recover only about 0.34-0.38pp of total match@20.
- MP-20 alone is not ready for official full test.
- Continue MPTS-52 validation generation and pursue either stronger candidate scoring or additional candidate sources before freezing.
