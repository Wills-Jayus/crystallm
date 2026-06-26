# MP-20 Validation Decision

Timestamp: 2026-06-24 10:49 UTC

## Decision

Proceed with the MP-20 selector/reranker branch immediately while MPTS-52 validation shards continue in parallel.

The MP-20 validation anchor generation, assembly, export, and aggregate K1/K5/K20/K50/K100 evaluation are complete. The reliable aggregate evidence is strong enough to justify model/selector work without waiting for MPTS-52:

| budget | match_rate | RMSE | hard timeouts | attempted candidates |
| --- | ---: | ---: | ---: | ---: |
| K1 | 0.7275339892 | 0.0515230847 | 0 | 9047 |
| K5 | 0.8311042334 | 0.0436973151 | 0 | 45235 |
| K20 | 0.8764231237 | 0.0404888525 | 3 | 180880 |
| K50 | 0.8991931027 | 0.0400885497 | 6 | 452050 |

The reliable K50 oracle gain over K20 is +2.277 percentage points, with a small RMSE improvement. That is enough headroom for a leakage-safe selector to try to recover at least part of the K50 gain into a frozen K20 output.

## Interpretation

MP-20 is not blocked on candidate generation. The limiting question is now selection, not whether the CrystaLLM GT-SG checkpoint can produce additional correct structures.

The practical target is:

- keep or nearly keep the K20 oracle coverage;
- improve rank placement or replace low-value tail candidates using inference-visible features only;
- recover at least 1 pp validation match improvement in an OOF setting before freezing any official-test candidate.

The K20 validation match rate is close to the historical MP-20 official-test anchor scale, so the K50 headroom is experimentally meaningful. It is not an official-test success claim; it only supports entering the selector/reranker route.

## K100 Caveat

Do not use the current MP-20 K100 aggregate metric as an oracle gate.

The recorded K100 result is timeout-censored:

- K100 match_rate=0.7562727976.
- K100 RMSE=0.0484667947.
- K100 hard timeouts=1435.
- K100 attempted candidates=761200 rather than the full 904700.

Because match@K should be non-decreasing for a faithfully evaluated fixed-prefix candidate list, K100 < K50 is an evaluator-pressure diagnostic. The K100 candidate bank remains useful raw material, but K100 aggregate metrics should not control the next scientific decision until per-candidate/checkpointed matching is available.

## Work Already Started

The MP-20 K50 selector feature table is complete:

- `features/mp20_val_k50_candidate_features.jsonl`
- 452350 rows.
- 9047 samples/materials.
- complete 50-candidate coverage per sample.

The MP-20 K50 per-candidate label pass is running:

- current snapshot: 230281 / 452350 rows written;
- output: `labels/mp20_val_k50_candidate_labels.jsonl`;
- summary target: `metrics/mp20_val_k50_candidate_label_summary.json`.

This label pass is required before any formal rows>=7 or OOF selector conclusion. The current labeler does not yet write target rows>=7 into each label row, so the rows>=7 bucket should be merged in a post-label summary step from the target CIFs before selector gates are reported.

## Parallel MPTS-52 Status

MPTS-52 validation K100 shard generation is continuing in parallel:

- current snapshot: 19 completed, 1 running, 59 pending out of 79 shards;
- currently running shard: 0018;
- latest observed shard raw CIFs are being written under `generations/crystallm_gt_sg_val_anchor_symprec0p1_shards/mpts52/`.

MPTS-52 remains mandatory for cross-dataset validation and possible joint strategy selection, but it should not block MP-20 analysis and selector preparation.

## Next MP-20 Steps

1. Let the full K50 label pass finish.
2. Summarize per-sample K1/K5/K20/K50 labels, RMSE, status counts, and target rows>=7 buckets.
3. Train/evaluate rerank-only and conservative selector models with deterministic group OOF splits by sample_id.
4. Gate only on full validation OOF metrics; do not use official test labels or per-sample test target structure before freezing.
5. If MP-20 selector validation succeeds, freeze a candidate system while MPTS-52 validation continues.
