# opentry_4 Requirements Audit

Time: 2026-06-18T18:45:35Z

| requirement | status | evidence |
|---|---|---|
| write only under opentry_4 | pass | New scripts, data, cache, eval, reports, checkpoints and manifest entries are under `/data/users/xsw/autodlmini/model/New_model/opentry_4`; opentry_3 was read-only. |
| use crystallm_env | pass | E7006-E7010 commands were executed with `conda run -n crystallm_env python`; wrappers compile under that env. |
| freeze/import one-time E718 full test | pass | `reports/E718_frozen_before_full_test.md`, `eval/E718_full_test_result.json`, `reports/E718_full_test_report.md`; opentry_4 did not rerun full test. |
| hard-negative diagnosis | pass | `eval/hard_negative_diagnosis.json` plus top20/top50 `eval/hard_negative_diagnosis_v2.json`. |
| geometry energy model train-dev/val512 | pass | `checkpoints/geometry_energy_model_sklearn_e7006.joblib`, train-dev AUC 0.8873, val512 AUC 0.8665. |
| joint free-param/lattice proposal generator train-dev/val512 | pass | Train-dev generated candidates 208 over 26 rows>=7 samples with match@20 34.62%; val512 E700/E7008 new positives beyond E424 top20 = 2. |
| smoke then val512 generator schedule | pass | E677 val64 smoke, E685 aligned55, E700 val512, plus E7009/E7010 train-dev evaluation are reported in `reports/joint_geometry_generator_report.md`. |
| anchor-safe replacement after ceiling gain | pass | E7008 anchor-safe insertion runs only after val512 new-positive gate passes; config in `configs/e7008_anchor_safe_selector_config.json`. |
| full test tuning avoided | pass | No test labels or test GT CIFs are used for training/tuning after frozen E734c import. |
| two metrics +5pp vs GT-SG baseline | fail | E718 full test: match@1 +1.04 pp, match@5 -0.73 pp, match@20 -5.01 pp. |

Termination condition 2 is now satisfied with direct evidence: frozen E718 full test is reported, the hard-negative geometry energy model has train-dev/val512 evaluation, and the constrained joint free-param/lattice proposal generator has train-dev/val512 evaluation. The scientific result remains weak: train-dev adds no new positives beyond its baseline, while val512 adds only 2 rows>=7 new positives and anchor-safe insertion does not improve match@20.

## Revalidation 2026-06-19T14:45:00Z

Rechecked under `conda run -n crystallm_env python`:

* Required final deliverables exist and are non-empty: experiment log, manifest, E718 full-test report, hard-negative report, geometry energy report, joint generator report, final summary, and this audit.
* `manifests/opentry_4_manifest.jsonl` parses as JSONL with 15 records.
* All opentry_4 scripts compile with `python -m py_compile`.
* Core eval JSON files parse: `E718_full_test_result.json`, `hard_negative_diagnosis.json`, `geometry_energy_model_eval.json`, and `joint_geometry_generator_eval.json`.
* Frozen full-test tuning flag remains `no`.
* E7006 sklearn energy model evidence remains val512 AUC 0.8665 and rows>=7 pairwise accuracy 0.7114.
* Joint generator evidence remains train-dev new positives beyond baseline = 0 and val512 new positives beyond E424 top20 = 2.
* E7008 anchor-safe val512 metrics remain match@1 33.40%, match@5 41.30%, match@20 46.05%, rows>=7 match@20 19.00%.

Conclusion: prompt termination condition 2 remains satisfied. The two-main-metrics +5pp scientific target remains unmet, so future work should continue from the stronger pre-render residual/mixture generator route described in the final summary rather than from selector tuning.
