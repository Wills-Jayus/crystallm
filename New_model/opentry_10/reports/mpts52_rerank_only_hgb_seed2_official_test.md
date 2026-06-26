# MPTS-52 rerank-only HGB seed2 official test

Status: not successful.

Frozen system:

- `mpts52_rerank_only_hgb_seed2`
- Frozen before official evaluation in `frozen_strategy/mpts52_rerank_only_hgb_seed2/`
- Freeze declaration: `reports/pre_test_freeze_declaration.md`

Official MPTS-52 CrystaLLM-a GT-SG anchor:

| Metric | Anchor | This system | Delta |
|---|---:|---:|---:|
| match@1 | 25.23% | 24.938% | -0.292pp |
| match@5 | 36.46% | 35.931% | -0.529pp |
| match@20 | 43.96% | 43.960% | -0.000pp |

The system does not meet the formal success criterion because no match metric exceeds the anchor by at least 1.0 percentage point. match@20 is effectively unchanged and therefore satisfies the non-degradation guard, but K1 and K5 are below anchor.

Raw metric artifacts:

- `metrics/official_test/mpts52_rerank_only_hgb_seed2_k1.raw.txt`
- `metrics/official_test/mpts52_rerank_only_hgb_seed2_k5.raw.txt`
- `metrics/official_test/mpts52_rerank_only_hgb_seed2_k20.raw.txt`
- `metrics/official_test/mpts52_rerank_only_hgb_seed2_summary.json`

Policy note:

This official test result must not be used to tune thresholds, rerun the same method, or claim success. The next work should continue from validation-only evidence on a distinct frozen route or switch to the pure structural route.
