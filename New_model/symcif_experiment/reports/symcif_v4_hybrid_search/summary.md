# SymCIF-v4 Hybrid WA Search and Structured WA Report

## Scope

- Data root: `data/structured_symcif_v4_reextracted`.
- Did not run full match@5; did not use v3 text sampler; did not train coords/lattice losses.
- Hybrid candidate generation combines policy search and old frequency search, deduplicates by canonical WA key, and keeps `source_labels`, `policy_rank`, and `old_rank`.
- Fallback flags are recorded for timeout, empty policy candidates, `SG=2`, `n_sites>=6`, and `num_elements>=4`; fallback adds old-search candidates without post-hoc candidate repair.

## Main Test Metrics

| Method | WA@20 | WA@100 | WA@200 | Skeleton@200 | Nonempty | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| policy only | 83.40% | 91.40% | 93.40% | 94.60% | 99.00% | 8 |
| old only | 69.60% | 80.20% | 85.80% | 87.00% | 100.00% | 0 |
| hybrid union | 84.20% | 92.20% | 94.20% | 95.40% | 100.00% | 0 |
| orbit-aware rerank fusion | 86.60% | 92.60% | 94.40% | 95.60% | 100.00% | 0 |
| structured multitask rerank | 81.00% | 88.80% | 91.00% | 94.20% | 100.00% | 0 |

## Runtime

- Policy search runtime p50/p90/max: 0.0293s / 3.0526s / 109.0897s; policy timeout samples: 8.
- Old search runtime p50/p90/max: 0.0107s / 0.0779s / 0.7476s; timeout samples: 0.
- Hybrid effective runtime p50/p90/max: 0.0414s / 3.1291s / 109.5695s; final timeout samples: 0.

## Required Breakdowns

| Subset | Samples | WA@20 | WA@100 | WA@200 | Skeleton@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| n_sites>=6 | 175 | 72.57% | 82.86% | 87.43% | 89.14% |
| num_elements>=4 | 192 | 83.85% | 91.67% | 93.75% | 93.75% |
| SG=2 | 28 | 89.29% | 96.43% | 96.43% | 96.43% |
| SG=65 | 12 | 33.33% | 33.33% | 41.67% | 58.33% |
| SG=71 | 9 | 44.44% | 66.67% | 66.67% | 66.67% |
| SG=127 | 8 | 62.50% | 62.50% | 62.50% | 75.00% |

## Acceptance

- WA@20 > 83%: PASS (86.60%).
- WA@100 >= 92%: PASS (92.60%).
- WA@200 >= 95%: FAIL (best 94.40%).
- Skeleton@200 >= 97%: FAIL (best 95.60%).
- n_sites>=6 WA@200 >= 88%: FAIL (87.43%).
- Timeout <= 1: PASS (0).

## Analysis

- Hybrid fixes runtime reliability: policy-only had 8 test timeouts, while union/fallback has 0 and 100% nonempty candidate sets.
- Candidate pool depth is now the main ceiling. Test union contains the GT WA somewhere in the 700-candidate pool for about 96.2% of samples, but only 94.2% land in the first 200 under policy-first ranking.
- The orbit-aware Transformer reranker improves early precision and passes WA@20/WA@100, but does not reliably pull enough deep candidates into top200. The conservative fusion preserves more coverage than pure model ranking.
- Hard cases remain concentrated in `n_sites>=6` and SG 65/71/127. SG=2 is comparatively strong after fallback, reaching WA@200 96.43%.
- The WA-only structured multitask run is functional with the specified weighted losses. In this short run, the step-policy head improves, but the multitask reranker branch underperforms the standalone reranker on test WA@200.

## Artifacts

- `reports/symcif_v4_hybrid_search/hybrid_search_summary.json`
- `reports/symcif_v4_hybrid_search/hybrid_breakdown_per_sg.csv`
- `reports/symcif_v4_hybrid_search/hybrid_breakdown_per_nsites.csv`
- `reports/symcif_v4_hybrid_search/test_hybrid_candidates.jsonl`
- `reports/symcif_v4_hybrid_search/test_reranked_predictions.jsonl`
- `reports/symcif_v4_hybrid_search/reranker_summary.json`
- `reports/symcif_v4_hybrid_search/structured_multitask_wa_summary.json`
