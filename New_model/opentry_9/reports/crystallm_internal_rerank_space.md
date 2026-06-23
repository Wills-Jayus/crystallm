# CrystaLLM Internal Rerank Space

Required validation K20 per-rank labels were not available. The table below is a post-hoc diagnostic from existing official test sample metrics only; it is not used for tuning.

| dataset | match@1 | match@5 | match@20 | top1 fail but top5 hit | top1 fail but top20 hit | top5 fail but top20 hit | oracle rerank@1 upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp_20_test_posthoc | 71.67% | 83.08% | 87.81% | 1032 | 1460 | 428 | 87.81% |
| mpts_52_test_posthoc | 25.23% | 36.46% | 43.96% | 909 | 1516 | 607 | 43.96% |

Exact per-rank cumulative gain could not be recomputed because the historical sample_metrics files only expose @1/@5/@20 booleans, not per-candidate match labels.

## opentry_9 Partial Validation Anchor Exact Rerank Space

The rows below are exact per-sample cumulative K20 labels for completed validation-anchor shards only.

| dataset | samples | match@1 | match@5 | match@20 | top1 fail but top5 hit | top1 fail but top20 hit | top5 fail but top20 hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mp_20 | 23 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
| mpts_52 | 8 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |
