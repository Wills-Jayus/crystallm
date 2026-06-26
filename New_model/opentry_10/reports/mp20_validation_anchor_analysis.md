# MP-20 Validation Anchor Analysis

Timestamp: 2026-06-24 10:15 UTC

## Status

The MP-20 CrystaLLM GT-SG validation anchor is complete:

- Generated/postprocessed CIFs: 904700 / 904700.
- Unified candidate bank: `candidates/crystallm_gt_sg_mp20_val_k100.jsonl`.
- Candidate rows: 904700.
- Metrics copy: `metrics/crystallm_gt_sg_mp20_val.json`.

## Reliable Validation Metrics

| budget | match_rate | RMSE | hard timeouts | candidate attempts |
| --- | ---: | ---: | ---: | ---: |
| K1 | 0.7275339892 | 0.0515230847 | 0 | 9047 |
| K5 | 0.8311042334 | 0.0436973151 | 0 | 45235 |
| K20 | 0.8764231237 | 0.0404888525 | 3 | 180880 |
| K50 | 0.8991931027 | 0.0400885497 | 6 | 452050 |

Reliable deltas:

- K1 -> K5: +10.357pp match, RMSE improves by 0.007826.
- K5 -> K20: +4.532pp match, RMSE improves by 0.003208.
- K20 -> K50: +2.277pp match, RMSE improves by 0.000400.

## K100 Caveat

The recorded K100 metric is timeout-censored and must not be interpreted as a true monotonic oracle:

- K100 match_rate=0.7562727976, RMSE=0.0484667947.
- K100 hard timeouts=1435.
- K100 candidate attempts=761200 rather than the full 904700.

Because match@K should be non-decreasing when the same candidate prefix is evaluated faithfully, K100 < K50 is a diagnostic signal, not a scientific conclusion. The cause is pathological `StructureMatcher` work in K100 combined with material-level hard-timeout handling. K100 is useful for preserving candidates and diagnosing evaluator pressure, but not for gate decisions until a per-candidate or checkpointed evaluator is implemented.

## Experimental Conclusion

MP-20 has real validation headroom beyond the reproduced K20 anchor. The K20 validation anchor is close to the historical official-test anchor scale, and expanding the candidate budget to K50 gives +2.277pp additional oracle coverage with essentially unchanged RMSE among matches. This is large enough to justify an MP-20 reranker/selector branch.

The strongest near-term path is not to expand MP-20 generation further. It is to train or construct a leakage-safe selector over the reliable K50 pool and evaluate whether a frozen top-20 ordering can recover a meaningful fraction of the K50 oracle gain. The K100 pool can remain available as raw material, but it should not drive selection gates until timeout-safe labels are available.

## Immediate MP-20 Next Steps

1. Generate leakage-safe validation labels/features for the MP-20 K50 pool.
2. Include rank/config features, parser/validity/sensible diagnostics, composition/cell sanity features, and optional retrieval priors.
3. Train a selector/reranker on official train-derived signals only or validation labels only for model selection, then freeze before any official test.
4. Gate on full MP-20 validation rows, including rows>=7, against the K20 anchor ordering.
5. Run official test only after a frozen MP-20 or joint MP-20/MPTS strategy satisfies the prompt's validation gate.

## Parallel Work

MPTS-52 validation K100 shard generation should continue in parallel. It is mandatory for cross-dataset validation and for deciding whether the strategy is MP-20-only or joint MP-20/MPTS.
