# MPTS-52 Selector Branch Status

Updated: 2026-06-25 04:16 UTC

## Completed Artifacts

- MPTS-52 validation K50 features: `features/mpts52_val_k50_candidate_features.jsonl`
- MPTS-52 validation K50 labels: `labels/mpts52_val_k50_candidate_labels.jsonl`
- MPTS-52 K50 label summary: `metrics/mpts52_val_k50_candidate_label_metrics.json`
- MPTS-52 rerank-only OOF: `metrics/mpts52_rerank_oof_results.json`
- MPTS-52 rerank-only HGB seed2 bootstrap: `metrics/mpts52_rerank_hgb_seed2_bootstrap.json`
- MPTS-52 K50 selector OOF HGB seed0/1/2 and RF seed0/1/2:
  - `metrics/mpts52_k50_selector_scores_eval_hgb_seed0_seed1.json`
  - `metrics/mpts52_k50_conservative_selector_oof_hgb_seed2.json`
  - `metrics/mpts52_k50_conservative_selector_oof_rf_seed0.json`
  - `metrics/mpts52_k50_conservative_selector_oof_rf_seed1.json`
  - `metrics/mpts52_k50_conservative_selector_oof_rf_seed2.json`

## Validation Findings

- K50 oracle headroom exists: K20 match is 48.00%, K50 match is 52.72%, giving +4.72pp oracle headroom and 236 K50-only rescues.
- Rerank-only HGB seed2 is the strongest clean candidate so far:
  - match@1: 30.02% -> 31.24%, +1.22pp.
  - match@5: 40.48% -> 41.16%, +0.68pp.
  - match@20: unchanged at 48.00%.
  - 10k bootstrap CI for match@1 delta: +0.38pp to +2.10pp.
  - rows>=7 match@20: unchanged.
  - rows>=7 match@5: 9.99% -> 9.29%, -0.70pp, so this remains a risk for freeze.
- K50 conservative selector did not meet the >=1pp validation gate:
  - Best HGB conservative result: match@20 +0.30pp.
  - Best RF conservative result: match@20 +0.70pp.
- RF unconstrained selector improved top metrics but was not clean enough:
  - RF seed1 unconstrained: match@1 +2.18pp, match@5 +1.04pp, match@20 +0.66pp.
  - It also reduced rows>=7 match@20 by 0.35pp, so it should not be frozen directly.

## Decision

Do not freeze the K50 selector branch as-is.

The current primary freeze candidate is MPTS-52 rerank-only HGB seed2, subject to deciding whether its rows>=7@5 drop is acceptable or whether a guarded/routed reranker is required before official test.

