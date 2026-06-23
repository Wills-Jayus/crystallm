# Strategy Oracle Union Audit

## Validation Gate Status

- CrystaLLM-a GT-SG validation K20 candidate bank was not found in opentry_5/6/7/8 or symcif_experiment.
- opentry_7 cache contains official validation target CIF tarballs, not K20 CrystaLLM candidates; these are treated as references and were not used for candidate fusion.
- Therefore the required validation oracle-union gate cannot be passed. Phase B selector/ranker and new official test are not run.

## Available Validation Diagnostics

- MP-20 SymCIF-v4 K<=5 baseline: match@1/match@5 = 44.12% / 63.42%; WA_hit@1/WA_hit@5 = 38.63% / 65.11%.
- MP-20 one-fix prior selector: selected WA_hit@5 improves to 79.52% and match@5 to 71.58% in the historical validation artifact.
- MP-20 GT-WA geometry K<=5: match@1/match@5 = 77.16% / 82.94%; rows>=7/n_sites>=6 match@1/match@5 = 59.22% / 69.46%.

## Historical Test-Only Post-Hoc Overlap

These numbers use already-computed official test metrics only as diagnosis. They were not used to tune or freeze any strategy.

### mp20_anchor_vs_stablekey

- A_hit@20: 7943 (87.81%)
- B_hit@20: 6252 (69.11%)
- B exclusive rescue@20: 246 (2.72%)
- Oracle union@20: 8189 (90.53%)
- rows>=7 oracle union@20: 5204 (86.26%); B exclusive rescue: 220 (3.65%)

### mpts52_anchor_vs_stablekey

- A_hit@20: 3559 (43.96%)
- B_hit@20: 2671 (32.99%)
- B exclusive rescue@20: 551 (6.81%)
- Oracle union@20: 4110 (50.77%)
- rows>=7 oracle union@20: 3663 (48.03%); B exclusive rescue: 533 (6.99%)

### mp20_anchor_vs_strategy_fusion

- A_hit@20: 7943 (87.81%)
- B_hit@20: 7943 (87.81%)
- B exclusive rescue@20: 0 (0.00%)
- Oracle union@20: 7943 (87.81%)
- rows>=7 oracle union@20: 4984 (82.61%); B exclusive rescue: 0 (0.00%)

### mpts52_anchor_vs_strategy_fusion

- A_hit@20: 3559 (43.96%)
- B_hit@20: 3560 (43.97%)
- B exclusive rescue@20: 1 (0.01%)
- Oracle union@20: 3560 (43.97%)
- rows>=7 oracle union@20: 3130 (41.04%); B exclusive rescue: 0 (0.00%)

## opentry_9 Partial Validation Anchor Progress

A resumable real CrystaLLM GT-SG validation-anchor generation was started in opentry_9. The completed shards were evaluated as a partial K20 anchor only; this does not satisfy the full validation oracle-union gate.

| dataset | completed shards | samples | candidates | match@1 | match@5 | match@20 | gate use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mp_20 | 1 | 23 | 460 | 0.0000 | 0.0000 | 0.0000 | diagnostic only |
| mpts_52 | 1 | 8 | 160 | 0.0000 | 0.0000 | 0.0000 | diagnostic only |

Gate decision remains unchanged: do not train/freeze a selector and do not run a new official test until full validation K20 anchor plus validation-source union is complete.
