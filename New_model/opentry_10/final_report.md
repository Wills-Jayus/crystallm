# opentry_10 Final Report

## Bottom line

opentry_10 completed frozen-system official full-test evaluation, but no frozen system exceeded the CrystaLLM-a GT-SG anchor under the formal success standard.

This report must not claim "successfully exceeded". The paper-usable conclusion is negative: the validation-selected rerank/fusion systems did not transfer to official test with a >=1.0 percentage-point match improvement while preserving match@20 and rows>=7 performance.

## Formal success check

Formal standard from `prompt.md`:

- at least one official-test match metric improves over the corresponding CrystaLLM-a GT-SG anchor by >=1.0 pp;
- the same system's match@20 does not drop by more than 0.2 pp;
- rows>=7 corresponding metrics do not clearly drop;
- no test target, test StructureMatcher label, or test oracle selection is used for training/selection.

Result: `false` for every frozen official-test line evaluated in opentry_10.

## Official full-test results

| dataset | frozen system | match@1 | delta | match@5 | delta | match@20 | delta | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MP-20 | `mp20_k50_hgb_mean_seed012_margin_route` | 70.230% | -1.440 pp | 81.870% | -1.210 pp | 87.486% | -0.324 pp | failed |
| MPTS-52 | `mpts52_rerank_only_hgb_seed2` | 24.938% | -0.292 pp | 35.931% | -0.529 pp | 43.960% | -0.000 pp | failed |
| MPTS-52 | `mpts52_k30_rf_seed1_bestscore_route` | 26.075% | +0.845 pp | 36.228% | -0.232 pp | 44.059% | +0.099 pp | failed |
| MPTS-52 | `mpts52_k50_rf_seed1_margin_route` | 25.791% | +0.561 pp | 36.265% | -0.195 pp | 43.824% | -0.136 pp | failed |

Anchors:

- MP-20 CrystaLLM-a GT-SG: match@1/5/20 = 71.67% / 83.08% / 87.81%.
- MPTS-52 CrystaLLM-a GT-SG: match@1/5/20 = 25.23% / 36.46% / 43.96%.

## rows>=7 results

| dataset | frozen system | rows>=7 match@1 | delta | rows>=7 match@5 | delta | rows>=7 match@20 | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MP-20 | `mp20_k50_hgb_mean_seed012_margin_route` | 27.200% | -35.170 pp | 43.273% | -33.077 pp | 55.273% | -27.337 pp |
| MPTS-52 | `mpts52_k50_rf_seed1_margin_route` | 23.053% | +0.563 pp | 33.228% | -0.142 pp | 40.939% | -0.101 pp |

rows>=7 anchors:

- MP-20 CrystaLLM-a GT-SG: match@1/5/20 = 62.37% / 76.35% / 82.61%.
- MPTS-52 CrystaLLM-a GT-SG: match@1/5/20 = 22.49% / 33.37% / 41.04%.

## RMSE

| dataset | frozen system | RMSE@1 | RMSE@5 | RMSE@20 |
| --- | --- | ---: | ---: | ---: |
| MP-20 | `mp20_k50_hgb_mean_seed012_margin_route` | 0.049965 | 0.043623 | 0.042699 |
| MPTS-52 | `mpts52_rerank_only_hgb_seed2` | 0.123947 | 0.124546 | 0.133401 |
| MPTS-52 | `mpts52_k30_rf_seed1_bestscore_route` | 0.121638 | 0.122237 | 0.131714 |
| MPTS-52 | `mpts52_k50_rf_seed1_margin_route` | 0.126404 | 0.125088 | 0.132897 |

## Validation and search work completed

- Built and evaluated MP-20 validation CrystaLLM GT-SG K100 anchor bank: `candidates/crystallm_gt_sg_mp20_val_k100.jsonl`.
- Built MP-20 K50 candidate features/labels and OOF rerank/fusion searches.
- Completed MP-20 rerank-only OOF over logistic regression and HistGradientBoosting seeds; best rerank-only validation improved match@1 by +0.707 pp but did not meet a freeze-worthy official gate.
- Completed MP-20 K50 conservative selector and residual selector OOF. Best fixed/quota route improved validation match@20 by less than +1 pp.
- Completed MPTS-52 validation/fusion branches and froze/evaluated three MPTS-52 official-test lines.
- Completed MP-20 K50 test generation, postprocessing, frozen K20 construction, and official evaluation for the best validation-selected MP-20 HGB ensemble route.

## Candidate generation volume

- MP-20 validation K100: 904,700 generated validation CIF members.
- MP-20 test K50 supplemental generation completed all blocks 21-50 and built final K20 for 9,046 official test samples; final manifest had 9,046 samples with exactly 20 slots and no placeholder slots.
- MPTS-52 official frozen lines used 8,096 official test samples; the original MPTS anchor line contains one known missing sample filled by placeholders, so that line was not treated as a meaningful success source.

## Leakage and test feedback policy

No frozen threshold or route in this report was selected using official-test per-sample labels. Official aggregate results were read only after frozen artifacts were built and registered. The failed official results must not be used to retune the same thresholds/routes and rerun them as if pre-registered.

## Paper-usable conclusion

Usable as a negative result: validation-selected CrystaLLM GT-SG rerank/fusion routes showed limited validation headroom but did not produce a formal official-test exceed over CrystaLLM-a GT-SG. The MPTS-52 K30 route came closest, with +0.845 pp match@1, but it still missed the >=1.0 pp requirement and lowered match@5.

Not usable as a success claim: none of the opentry_10 frozen official-test systems.

## Primary artifacts

- MP-20 official summary: `metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json`
- MPTS-52 official summaries: `metrics/official_test/mpts52_rerank_only_hgb_seed2_summary.json`, `metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json`, `metrics/official_test/mpts52_k50_rf_seed1_margin_route_summary.json`
- Official-test reports: `reports/*official_test.md`
- Frozen registry: `state/frozen_registry.json`
- Experiment log: `experiment_log.md`

---

# Archived copied artifact below

The remaining content below was copied from `opentry_9` earlier to preserve mistakenly written artifacts. It is retained only as archive text and is not the opentry_10 final conclusion.

# INVALID COPIED ARTIFACT - NOT AN OPENTRY_10 FINAL REPORT

This file was copied from `opentry_9` at the user's request to preserve the mistakenly written artifacts.
It is not a valid `opentry_10` final report and must not be used as evidence that the `opentry_10`
prompt is complete. The active `opentry_10` prompt permits a real `final_report.md` only after
full frozen-system official test evaluation is complete.

# opentry_9 Final Report

## What Was Executed

- Created the requested opentry_9 directory structure and local copies of historical evaluator utilities.
- Built a candidate source manifest and unified JSONL candidate files for available anchor/fusion/stablekey/SymCIF validation sources.
- Audited strategy/fusion feasibility without training a selector and without launching a new official test.
- Reused SymCIF-v4 structured caches for pure structural canonicalization and wrote opentry_9 cache copies.
- Reused existing MP-20 validation GT-WA geometry and WA coverage artifacts for pure bottleneck diagnosis.

## Strategy/Fusion Answer

strategy/fusion does not have a validation-proven meaningful exceed path in opentry_9. The required CrystaLLM-a GT-SG validation K20 candidate bank was not found, so validation oracle union and reranker upper-bound gates cannot be passed. Consequently no selector/ranker was trained, no frozen strategy was produced, and no new official test was run.

Historical opentry_8 evidence remains unchanged: MP-20 strategy_fusion is exactly the CrystaLLM-a GT-SG anchor; MPTS-52 repaired only one missing primary sample. That is coverage repair, not a method breakthrough.

## Validation Oracle Union

The required validation oracle union against CrystaLLM-a GT-SG K20 could not be computed. Available validation diagnostics are MP-20 SymCIF K<=5 only: baseline match@1/@5 = 44.12% / 63.42%; GT-WA geometry match@1/@5 = 77.16% / 82.94%; one-fix selection match@5 = 71.58%. These are pure/structural diagnostics, not CrystaLLM fusion gates.

## CrystaLLM K20 Internal Rerank Space

Only official test @1/@5/@20 sample metrics are available for CrystaLLM K20; per-rank validation labels were not found. Post-hoc test diagnostics show substantial @20-vs-@1 room, but this was not used for tuning.

## Pure Model Bottleneck

The pure structural bottleneck is both WA coverage/selection and geometry. WA selection is weak at K<=5 (baseline WA_hit@5 = 65.11%, raw top100 WA_hit = 87.85%), while GT-WA geometry is not saturated (MP-20 match@5 = 82.94%, rows>=7 proxy match@5 = 69.46%).

No WA decoder checkpoint was trained in opentry_9 and no pure K20 full-test was run.

## Required Explicit Answers

- strategy/fusion meaningful exceed CrystaLLM-a GT-SG: no. MP-20 is identical to the anchor; MPTS-52 has only a tiny match@20 change from one repaired missing sample.
- pure model meaningful exceed CrystaLLM-a GT-SG: no. No frozen pure structural K20 model passed validation, and no official pure test was run.
- Source of any observed gain: coverage repair only for historical MPTS-52 strategy_fusion; no validated reranking gain, residual rescue gain, or geometry repair gain was frozen.
- Test leakage risk: low for opentry_9 actions. Existing test metrics were read only for post-hoc reporting; they were not used for training, threshold selection, reranking, or deciding to run a new test.
- Paper-usable conclusion: opentry_8 fusion should be reported as a negative/diagnostic result, not as a new method result; pure structural work can be reported only as a bottleneck diagnosis.

## MP-20

| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | candidate budget | fusion/ranking | pure model |
| ------ | ----- | ------: | ------: | -------: | -----: | -----: | ------: | --------------: | --------------: | ---------------: | ---------------: | -------------- | ---------- |
| CrystaLLM-a GT-SG | official test, historical opentry_7 | 71.67% | 83.08% | 87.81% | 0.0509 | 0.0449 | 0.0431 | 62.37% | 76.35% | 82.61% | K20 | anchor | no |
| opentry_8 strategy_fusion | official test, historical | 71.67% | 83.08% | 87.81% | 0.0509 | 0.0449 | 0.0431 | 62.37% | 76.35% | 82.61% | K20 | coverage repair | no |
| opentry_7 stablekey hybrid | official test, historical | 50.44% | 65.06% | 69.11% | 0.0670 | 0.0745 | 0.0745 | 36.13% | 52.83% | 57.33% | K20 | stablekey/SymCIF | no |

## MPTS-52

| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | candidate budget | fusion/ranking | pure model |
| ------ | ----- | ------: | ------: | -------: | -----: | -----: | ------: | --------------: | --------------: | ---------------: | ---------------: | -------------- | ---------- |
| CrystaLLM-a GT-SG | official test, historical opentry_7 | 25.23% | 36.46% | 43.96% | 0.1211 | 0.1257 | 0.1334 | 22.49% | 33.37% | 41.04% | K20 | anchor | no |
| opentry_8 strategy_fusion | official test, historical | 25.23% | 36.46% | 43.97% | 0.1211 | 0.1257 | 0.1334 | 22.49% | 33.37% | 41.04% | K20 | coverage repair | no |
| opentry_7 stablekey hybrid | official test, historical | 17.63% | 27.99% | 32.99% | 0.1326 | 0.1370 | 0.1419 | 15.50% | 24.69% | 29.61% | K20 | stablekey/SymCIF | no |

## Gate Decisions

- strategy selector/ranker trained: no
- frozen strategy passed validation gate: no
- opentry_9 official test run: no
- official test exceed over CrystaLLM-a GT-SG: no new claim
- pure model meaningful exceed CrystaLLM-a GT-SG: no
- test leakage risk: low for opentry_9 actions; historical test metrics were read only for reporting and were not used for tuning

## Paper-Usable vs Diagnostic

Paper-usable: the negative result that opentry_8 fusion was coverage repair rather than meaningful exceed, plus the pure structural bottleneck diagnosis if described as validation/diagnostic.

Diagnostic only: all test-overlap/exclusive-rescue numbers, the MP-20 K<=5 SymCIF one-fix result, and any stablekey hybrid official-test comparison from historical runs.

Routes not worth continuing as-is: opentry_8-style coverage repair and opentry_7 byte-level CIF pure model. The next best experiment is a validation-first CrystaLLM K20/K50 candidate bank plus an inference-feasible selector, and for pure structural work an exact-cover WA decoder/ranker coupled with geometry-quality reranking.

## 2026-06-22 Validation Anchor Addendum

After the initial diagnostic report, opentry_9 started real CrystaLLM-a GT-SG validation-anchor generation under a resumable shard controller. Two completed shards were exported and evaluated as partial K20 evidence.

| dataset | completed shards | samples | candidates | match@1 | match@5 | match@20 | use in decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mp_20 | 1 | 23 | 460 | 0.0000 | 0.0000 | 0.0000 | diagnostic only, not full validation gate |
| mpts_52 | 1 | 8 | 160 | 0.0000 | 0.0000 | 0.0000 | diagnostic only, not full validation gate |

This addendum does not change the gate decision: the full validation K20 CrystaLLM anchor and validation-source oracle union are still incomplete, so no selector/ranker was trained or frozen and no new official test was run. The partial candidates are useful only for checking that the reproduction path is working and resumable.
