# opentry_8 Experiment Log

Created: 2026-06-22 14:53:13 UTC

Scope: write only under `/data/users/xsw/autodlmini/model/New_model/opentry_8`.
Historical directories under `/data/users/xsw/autodlmini/model/New_model` are read-only inputs.
Runtime requirement: `crystallm_env`.

## 2026-06-22 14:53:13 UTC initial audit

Files read or inspected:

- `opentry_7/GPTprompt.md`
- `opentry_7/opentry_7_experiment_log.md`
- `opentry_7/opentry_7_final_report.md`
- `opentry_5/reports/opentry_5_final_summary.md`
- `opentry_5/reports/no_ranking_audit.md`
- `opentry_6/opentry_6_final_summary.md`
- `symcif_experiment/Log_GPT/round_20260529_01_mp20_minicfjoint_v2/comprehensive_mp20_minicfjoint_v2_report.md`
- `symcif_experiment/Log_GPT/round_20260529_02_mp20_minicfjoint_v2_goal_2026530/comprehensive_experiment_analysis_report.md`
- `symcif_experiment/Log_GPT/round_20260530_02_symcif_v5_multidataset_wa_decoder/symcif_v5_multidataset_wa_decoder_experiment_analysis_report.md`
- `symcif_experiment/Log_GPT/round_20260530_03_symcif_v5_full_test/symcif_v5_mp20_mpts52_full_test_report.md`
- `symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md`

opentry_7 pure-model failure diagnosis:

- The opentry_7 pure line was a CrystaLLM-style byte-level CIF GPT fine-tuned with composition + GT-SG prompts, not a structured model over row_count, Wyckoff skeleton, element assignment, lattice, and free parameters.
- Checkpoint selection used validation token loss only. It did not select by validation generation metrics such as match@1/5/20, rows>=7 match, RMSE, or validity.
- The fixed candidate order was honest, but the model did not inherit the exact-cover W/A decoder, row_count prior, skeleton-first decoding, or explicit geometry head required by the opentry_8 prompt.
- The result was therefore expected to trail CrystaLLM-a GT-SG: MP-20 60.58/70.67/77.96 vs GT-SG 71.67/83.08/87.81, and MPTS-52 17.18/24.35/31.52 vs GT-SG 25.23/36.46/43.96.

opentry_7 strategy/fusion failure diagnosis:

- The opentry_7 strategy line directly converted historical stablekey top20 predictions as the final K20 system.
- Coverage was incomplete before evaluation: MP-20 stablekey had 8893/9046 official samples, missing 153; MPTS-52 had 7663/8096, missing 433.
- The converted stablekey artifacts also had many missing CIF files during conversion: MP-20 10176 missing CIFs, MPTS-52 7780 missing CIFs.
- This violates the opentry_8 constraint that every official test sample must have 20 final candidate slots and that missing official samples cannot be silently dropped.
- It also explains the large gap to CrystaLLM-a GT-SG: opentry_7 strategy was lower by 18.69 pp match@20 on MP-20 and 10.97 pp match@20 on MPTS-52.

Historical comparability audit:

- opentry_5 MiniCFJoint E8009-E8038 were grouped-fold and smoke-style experiments, not official Table 3 full-test K20. They are useful for mechanism diagnosis only.
- opentry_5 showed exact-cover/fixed-order W/A generation can be honest, but rows>=7 match often remained zero even when aggregate match@20 looked high. Aggregate fold metrics are not directly comparable to official full-test metrics.
- opentry_6 reports stage-level rows>=7 new positives and geometry/refiner diagnostics, but many stages have match@20 `not_run`; they are not final official full-test K20 systems.
- SymCIF-v5 full-test reports use train-only retrieval/index instances with GT-SG and structured W/A+geometry, not a single pure neural model checkpoint. They are strategy/structured-generator evidence, not pure-model evidence.
- Validation, subset, clean_val, K5, K50/K100 raw-pool, common-subset, and structured-only metrics must not be presented as official full-test K20 results.

Most credible opentry_8 route:

- For strategy/fusion, repair coverage first. Use opentry_7 CrystaLLM-a GT-SG official K20 candidates as the primary full-coverage anchor, because it is the strongest audited full-test K20 legal source.
- Use historical stablekey/SymCIF candidates only as supplemental fallback for absent GT-SG slots, not as the final system backbone.
- Freeze a coverage-repaired strategy before full-test evaluation: no oracle selection, no StructureMatcher feedback, no post-test threshold changes.
- For pure structural model, do not repeat byte-level CIF GPT. A real next model must implement composition + GT-SG -> row_count/Wyckoff skeleton -> element W/A exact-cover -> lattice/free parameters -> CIF rendering, with rows>=7 curriculum and validation generation metrics for checkpoint selection.

## 2026-06-22 14:53:13 UTC evaluator setup

- Copied `opentry_7/scripts/opentry7_tools.py` to `opentry_8/scripts/opentry8_tools.py`.
- Updated write guard and log path so new outputs stay inside opentry_8.
- Copied official test target caches from opentry_7:
  - `cache/mp_20_test_targets.jsonl`
  - `cache/mpts_52_test_targets.jsonl`
- Copied unified evaluator config and updated metadata to `opentry_8_unified_csp_evaluator`.

## 2026-06-22 14:55:03 UTC build strategy_fusion mp20/test
- config: `model/New_model/opentry_8/frozen_strategy/config.json`
- output: `model/New_model/opentry_8/generations/strategy_fusion_mp_20_test_k20_candidates.jsonl`
- official samples: 9046; candidate rows: 180920
- samples with 20 slots: 9046
- primary slots: 180920; supplemental slots: 0; placeholder slots: 0

## 2026-06-22 14:55:21 UTC build strategy_fusion mpts52/test
- config: `model/New_model/opentry_8/frozen_strategy/config.json`
- output: `model/New_model/opentry_8/generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl`
- official samples: 8096; candidate rows: 161920
- samples with 20 slots: 8096
- primary slots: 161900; supplemental slots: 20; placeholder slots: 0

## 2026-06-22 15:06:56 UTC evaluate strategy_fusion mp20/test
- candidate source: `model/New_model/opentry_8/generations/strategy_fusion_mp_20_test_k20_candidates.jsonl`
- official samples: 9046; with candidates: 9046
- all match@1/5/20: 0.7166703515365908 / 0.8307539243864691 / 0.8780676542118063
- all RMSE@1/5/20: 0.0509480997149395 / 0.04485415674824324 / 0.043076585142923056
- rows>=7 samples: 6033; match@1/5/20: 0.62373611801757 / 0.7634675948947456 / 0.8261229902204542; positive-any: 4984

## 2026-06-22 15:16:20 UTC evaluate strategy_fusion mpts52/test
- candidate source: `model/New_model/opentry_8/generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl`
- official samples: 8096; with candidates: 8096
- all match@1/5/20: 0.252346837944664 / 0.36462450592885376 / 0.4397233201581028
- all RMSE@1/5/20: 0.12110278576321938 / 0.12573424156501198 / 0.1333635891060911
- rows>=7 samples: 7626; match@1/5/20: 0.22488853920797272 / 0.33372672436401785 / 0.41043797534749543; positive-any: 3130

## 2026-06-22 15:20:00 UTC pure structural MVP decision

- No opentry_8 pure structural checkpoint was trained that satisfies the prompt gate.
- opentry_7 pure remains a historical byte-level CIF GPT control, not the requested structural W/A+geometry model.
- MiniCFJoint-v2 historical validation evidence shows the exact-cover factorized decoder is viable, but GT-W/A geometry failed validation gates.
- SymCIF/v5 and stablekey candidates are strategy/retrieval/index systems and are not eligible as pure model outputs.
- Official full test for opentry_8 pure structural model is therefore not run; reporting it as a full-test result would be misleading.
- Artifacts written:
  - `frozen_pure_model/pure_structural_mvp_status.json`
  - `reports/pure_structural_mvp_report.md`
