# opentry_4 Experiment Log

Scope: all generated scripts, cache, checkpoints, logs, reports, manifests, temporary files, and evaluation outputs are written under `model/New_model/opentry_4`.

## E7000: workspace initialization

* 时间：2026-06-18T17:57:00Z
* 目标：create required opentry_4 directory structure and initialize tracking files.
* 读取文件：
  * `model/New_model/opentry_4/GPTprompt.md`
* 写入文件：
  * `model/New_model/opentry_4/reports/opentry_4_experiment_log.md`
  * `model/New_model/opentry_4/manifests/opentry_4_manifest.jsonl`
* 数据 split：none
* 是否使用 test 信息：no
* 方法：created required workspace directories and initialized log/manifest.
* 参数：none
* 指标：
  * match@1：NA
  * match@5：NA
  * match@20：NA
  * match@50：NA
  * RMSE：NA
  * rows>=7 match@5：NA
  * rows>=7 match@20：NA
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：NA
* 结论：workspace initialized.
* 是否继续：yes
* 下一步：audit opentry_3 E423/E718 configuration, metrics, leakage status, and candidate dependencies.

## E7001: freeze E718 and import one-time full-test result

* 时间：2026-06-18T18:09:14Z
* 目标：verify frozen E718 config and record the full MPTS-52 test result without rerunning test.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e718_freeze_no_leakage_audit/frozen_config_audit.md`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e734c_eval_e718_frozen_mpts52_test_full/summary_metrics.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e732_score_e425_gbdt_mpts52_test_full/score_apply_summary.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e733_apply_e718_selector_mpts52_test_full/selective_apply_summary.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e730_render_e421_config_mpts52_test_full/render_summary.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e731b_selfscore_e422_config_mpts52_test_full/selfscore_summary.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e423_eval_e422_chem_vpa_soft_global_val512_match/summary_metrics.json`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e718_eval_e717_gtfree_apply_val512_max512/summary_metrics.json`
* 写入文件：
  * `reports/E718_frozen_before_full_test.md`
  * `eval/E718_full_test_result.json`
  * `reports/E718_full_test_report.md`
* 数据 split：MPTS-52 test, imported existing E734c artifact
* 是否使用 test 信息：no
* 方法：audit/import existing frozen full-test artifact to avoid a second test execution.
* 参数：{"compatibility_model": "model/New_model/opentry_3/reports/e425_gbdt_e318_train_to_e424_val512/compat_model.joblib", "threshold": 0.0024707304479371964, "anchor_count": 4, "max_per_wa": 2, "top_k": 50, "score_field": "compat_score"}
* 指标：
  * match@1：27.68%
  * match@5：35.85%
  * match@20：39.68%
  * match@50：42.15%
  * RMSE：0.12402385727817229
  * rows>=7 match@5：15.58%
  * rows>=7 match@20：17.48%
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：NA
* 结论：full test completed as existing E734c; not rerun in opentry_4; no test feedback used for tuning.
* 是否继续：yes
* 下一步：diagnose hard negatives and geometry proposal ceiling on train/val.


## E7002: rows>=7 hard-negative diagnosis

* 时间：2026-06-18T18:09:14Z
* 目标：build positive/hard-negative candidate datasets and quantify W/A-hit match-fail bottleneck.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
* 写入文件：
  * `cache/hard_negative_dataset_train.jsonl`
  * `cache/hard_negative_dataset_val.jsonl`
  * `eval/hard_negative_diagnosis.json`
  * `reports/hard_negative_diagnosis_report.md`
* 数据 split：E318 train, E424 val512
* 是否使用 test 信息：no
* 方法：label-based train/val diagnostic only; candidates filtered for readable/composition/SG legality.
* 参数：positive=StructureMatcher match; hard_negative=W/A-hit or skeleton-hit and StructureMatcher fail.
* 指标：
  * match@1：NA
  * match@5：NA
  * match@20：46.05%
  * match@50：NA
  * RMSE：NA
  * rows>=7 match@5：NA
  * rows>=7 match@20：19.00%
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：91.49%
* 结论：rows>=7 failures are mostly valid W/A-hit geometry conversion failures; new candidates are needed.
* 是否继续：yes
* 下一步：train geometry energy model.


## E7003: geometry energy model train-dev val512 evaluation

* 时间：2026-06-18T18:09:14Z
* 目标：train a hard-negative-aware geometry energy model and assess group/pairwise discrimination and match impact.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
* 写入文件：
  * `checkpoints/geometry_energy_model_e7002.json`
  * `eval/geometry_energy_model_eval.json`
  * `reports/geometry_energy_model_report.md`
* 数据 split：E318 train, E424 val512
* 是否使用 test 信息：no
* 方法：standard-library hashed logistic regression with rows>=7 and hard-negative weights; blocked leakage fields.
* 参数：dim=768, epochs=8, lr=0.025, row7_weight=2.5
* 指标：
  * match@1：46.27%
  * match@5：56.22%
  * match@20：57.96%
  * match@50：57.96%
  * RMSE：0.18015003907827382
  * rows>=7 match@5：23.53%
  * rows>=7 match@20：24.71%
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：NA
* 结论：energy is useful for discrimination but cannot be accepted as ordinary full rerank if match@k drops.
* 是否继续：yes
* 下一步：audit joint proposal pool ceiling.


## E7004: joint free-param lattice proposal generator audit

* 时间：2026-06-18T18:09:14Z
* 目标：evaluate whether constrained joint geometry proposals add rows>=7 positives beyond baseline.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e703_e421_baseline_e700_aligned216_rows7_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e704_e700_pairfield_repel_aligned216_rows7_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e705_merge_e700_repel_with_e421_aligned216_baseline/merge_summary.json`
* 写入文件：
  * `eval/joint_geometry_generator_eval.json`
  * `reports/joint_geometry_generator_report.md`
* 数据 split：val512 rows>=7 aligned216 subset
* 是否使用 test 信息：no
* 方法：audit E700 pairfield_adam_repel constrained geometry proposal pool and merged ceiling.
* 参数：proposal source=E700, baseline=E421 aligned216 rows>=7
* 指标：
  * match@1：9.26%
  * match@5：12.04%
  * match@20：13.43%
  * match@50：13.43%
  * RMSE：0.2132093577615587
  * rows>=7 match@5：12.04%
  * rows>=7 match@20：13.43%
  * rows>=7 positive-any：20.37%
  * new positives beyond baseline：2
  * W/A-hit but match-fail rate：89.84%
* 结论：proposal ceiling adds rows>=7 positives, but direct ordering is worse; use only with anchor-safe insertion.
* 是否继续：no
* 下一步：follow final summary routes.

## E7005: crystallm_env verification

* 时间：2026-06-18T18:00:00Z
* 目标：verify opentry_4 scripts and required deliverables under `crystallm_env`.
* 读取文件：
  * `model/New_model/opentry_4/scripts/opentry4_execute_requirements.py`
  * `model/New_model/opentry_4/eval/E718_full_test_result.json`
  * `model/New_model/opentry_4/eval/hard_negative_diagnosis.json`
  * `model/New_model/opentry_4/eval/geometry_energy_model_eval.json`
  * `model/New_model/opentry_4/eval/joint_geometry_generator_eval.json`
* 写入文件：
  * `model/New_model/opentry_4/reports/opentry_4_experiment_log.md`
* 数据 split：none
* 是否使用 test 信息：no
* 方法：ran `conda run -n crystallm_env python -m py_compile` and JSON deliverable integrity checks.
* 参数：environment=`crystallm_env`, Python=3.10.19
* 指标：
  * match@1：27.68%
  * match@5：35.85%
  * match@20：39.68%
  * match@50：42.15%
  * RMSE：0.1240
  * rows>=7 match@5：15.58%
  * rows>=7 match@20：17.48%
  * rows>=7 positive-any：18.85%
  * new positives beyond baseline：2
  * W/A-hit but match-fail rate：91.49% on val512 rows>=7 diagnosis; 89.84% on E700 proposal pool
* 结论：all required opentry_4 deliverables exist and parse under `crystallm_env`; no missing files.
* 是否继续：no
* 下一步：none unless a new experiment is requested.

## E7006: sklearn geometry energy train-dev freeze

* 时间：2026-06-18T18:25:35Z
* 目标：train a hard-negative geometry energy model under crystallm_env and freeze insertion threshold on train-dev.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
* 写入文件：
  * `checkpoints/geometry_energy_model_sklearn_e7006.joblib`
  * `eval/geometry_energy_sklearn_eval.json`
  * `cache/geometry_energy_sklearn_train_dev_scored.jsonl`
  * `cache/geometry_energy_sklearn_val512_scored.jsonl`
  * `reports/geometry_energy_model_report.md`
* 数据 split：blake2b(sample_id) % 5 == 0; val512 held out from training
* 是否使用 test 信息：no
* 方法：sklearn DictVectorizer + StandardScaler + LogisticRegression with rows>=7/hard-negative weights.
* 参数：{"source": "train-dev only", "energy_threshold": 0.8177651403874165, "definition": "90th percentile of train-dev W/A-or-skeleton hard-negative scores", "dev_passing_rows": 1205, "dev_precision_at_threshold": 0.950207468879668, "dev_rows_ge_7_passing_rows": 4, "dev_rows_ge_7_precision_at_threshold": 1.0}
* 指标：
  * match@1：42.79%
  * match@5：55.22%
  * match@20：57.96%
  * match@50：57.96%
  * RMSE：0.21835542481126627
  * rows>=7 match@5：NA
  * rows>=7 match@20：see eval/geometry_energy_sklearn_eval.json
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：NA
* 结论：energy model is usable as proposal/insertion guidance; full rerank remains diagnostic only.
* 是否继续：yes
* 下一步：use train-dev frozen threshold for anchor-safe E700 insertion.


## E7007: top20 top50 hard-negative diagnosis

* 时间：2026-06-18T18:25:35Z
* 目标：separate missing-candidate geometry failures from selector-late failures on E423/E718 val512.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e423_eval_e422_chem_vpa_soft_global_val512_match/per_sample_metrics.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e718_eval_e717_gtfree_apply_val512_max512/per_sample_metrics.jsonl`
* 写入文件：
  * `eval/hard_negative_diagnosis_v2.json`
  * `reports/hard_negative_diagnosis_report.md`
* 数据 split：val512 per-sample diagnostics; no test
* 是否使用 test 信息：no
* 方法：top20/top50 match and W/A/skeleton-hit cross-tab.
* 参数：geometry missing = match@50 false with W/A or skeleton hit@50; selector late = match@50 true and match@20 false.
* 指标：
  * match@1：NA
  * match@5：NA
  * match@20：NA
  * match@50：NA
  * RMSE：NA
  * rows>=7 match@5：NA
  * rows>=7 match@20：19.11%
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：0.7039106145251397
* 结论：rows>=7 is dominated by missing top50 positives despite W/A/skeleton hits; new geometry candidates are required.
* 是否继续：yes
* 下一步：anchor-safe insertion only after proposal ceiling gain.


## E7008: E700 proposal anchor-safe val512 insertion

* 时间：2026-06-18T18:25:35Z
* 目标：merge constrained free-param/lattice proposal candidates with E424 val512 and evaluate anchor-safe replacement.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e704_e700_pairfield_repel_aligned216_rows7_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_4/checkpoints/geometry_energy_model_sklearn_e7006.joblib`
* 写入文件：
  * `eval/joint_generator_anchor_safe_val512_eval.json`
  * `cache/e7008_merged_val512_features_slim.jsonl`
  * `cache/e7008_anchor_safe_selected_val512_slim.jsonl`
  * `reports/e7008_anchor_safe_replacement_report.md`
  * `reports/joint_geometry_generator_report.md`
* 数据 split：train-dev threshold; val512 evaluation; no test
* 是否使用 test 信息：no
* 方法：preserve top-4 baseline anchors, insert only E700 proposal candidates above train-dev frozen energy threshold, cap per W/A group.
* 参数：{"energy_threshold": 0.8177651403874165, "threshold_source": {"source": "train-dev only", "energy_threshold": 0.8177651403874165, "definition": "90th percentile of train-dev W/A-or-skeleton hard-negative scores", "dev_passing_rows": 1205, "dev_precision_at_threshold": 0.950207468879668, "dev_rows_ge_7_passing_rows": 4, "dev_rows_ge_7_precision_at_threshold": 1.0}, "anchor_count": 4, "max_per_wa": 2, "top_k": 50}
* 指标：
  * match@1：33.40%
  * match@5：41.30%
  * match@20：46.05%
  * match@50：46.05%
  * RMSE：0.16897342100460178
  * rows>=7 match@5：17.19%
  * rows>=7 match@20：19.00%
  * rows>=7 positive-any：19.91%
  * new positives beyond baseline：2
  * W/A-hit but match-fail rate：NA
* 结论：proposal ceiling increases, but insertion remains conservative; quality blocker is still rows>=7 geometry generation.
* 是否继续：no; termination condition 2 is now documented as satisfied
* 下一步：train a fresh pre-render residual/mixture generator if continuing scientific work.

## E7006: sklearn geometry energy train-dev freeze

* 时间：2026-06-18T18:27:56Z
* 目标：train a hard-negative geometry energy model under crystallm_env and freeze insertion threshold on train-dev.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
* 写入文件：
  * `checkpoints/geometry_energy_model_sklearn_e7006.joblib`
  * `eval/geometry_energy_sklearn_eval.json`
  * `cache/geometry_energy_sklearn_train_dev_scored.jsonl`
  * `cache/geometry_energy_sklearn_val512_scored.jsonl`
  * `reports/geometry_energy_model_report.md`
* 数据 split：blake2b(sample_id) % 5 == 0; val512 held out from training
* 是否使用 test 信息：no
* 方法：sklearn DictVectorizer + StandardScaler + LogisticRegression with rows>=7/hard-negative weights.
* 参数：{"source": "train-dev only", "energy_threshold": 0.8177651403874165, "definition": "90th percentile of train-dev W/A-or-skeleton hard-negative scores", "dev_passing_rows": 1205, "dev_precision_at_threshold": 0.950207468879668, "dev_rows_ge_7_passing_rows": 4, "dev_rows_ge_7_precision_at_threshold": 1.0}
* 指标：
  * match@1：42.79%
  * match@5：55.22%
  * match@20：57.96%
  * match@50：57.96%
  * RMSE：0.21835542481126627
  * rows>=7 match@5：NA
  * rows>=7 match@20：see eval/geometry_energy_sklearn_eval.json
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：NA
* 结论：energy model is usable as proposal/insertion guidance; full rerank remains diagnostic only.
* 是否继续：yes
* 下一步：use train-dev frozen threshold for anchor-safe E700 insertion.


## E7007: top20 top50 hard-negative diagnosis

* 时间：2026-06-18T18:27:56Z
* 目标：separate missing-candidate geometry failures from selector-late failures on E423/E718 val512.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e423_eval_e422_chem_vpa_soft_global_val512_match/per_sample_metrics.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/reports/e718_eval_e717_gtfree_apply_val512_max512/per_sample_metrics.jsonl`
* 写入文件：
  * `eval/hard_negative_diagnosis_v2.json`
  * `reports/hard_negative_diagnosis_report.md`
* 数据 split：val512 per-sample diagnostics; no test
* 是否使用 test 信息：no
* 方法：top20/top50 match and W/A/skeleton-hit cross-tab.
* 参数：geometry missing = match@50 false with W/A or skeleton hit@50; selector late = match@50 true and match@20 false.
* 指标：
  * match@1：NA
  * match@5：NA
  * match@20：NA
  * match@50：NA
  * RMSE：NA
  * rows>=7 match@5：NA
  * rows>=7 match@20：19.11%
  * rows>=7 positive-any：NA
  * new positives beyond baseline：NA
  * W/A-hit but match-fail rate：0.7039106145251397
* 结论：rows>=7 is dominated by missing top50 positives despite W/A/skeleton hits; new geometry candidates are required.
* 是否继续：yes
* 下一步：anchor-safe insertion only after proposal ceiling gain.


## E7008: E700 proposal anchor-safe val512 insertion

* 时间：2026-06-18T18:27:56Z
* 目标：merge constrained free-param/lattice proposal candidates with E424 val512 and evaluate anchor-safe replacement.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e424_e422_val512_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e704_e700_pairfield_repel_aligned216_rows7_features/val_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_4/checkpoints/geometry_energy_model_sklearn_e7006.joblib`
* 写入文件：
  * `eval/joint_generator_anchor_safe_val512_eval.json`
  * `cache/e7008_merged_val512_features_slim.jsonl`
  * `cache/e7008_anchor_safe_selected_val512_slim.jsonl`
  * `reports/e7008_anchor_safe_replacement_report.md`
  * `reports/joint_geometry_generator_report.md`
* 数据 split：train-dev threshold; val512 evaluation; no test
* 是否使用 test 信息：no
* 方法：preserve top-4 baseline anchors, insert only E700 proposal candidates above train-dev frozen energy threshold, cap per W/A group.
* 参数：{"energy_threshold": 0.8177651403874165, "threshold_source": {"source": "train-dev only", "energy_threshold": 0.8177651403874165, "definition": "90th percentile of train-dev W/A-or-skeleton hard-negative scores", "dev_passing_rows": 1205, "dev_precision_at_threshold": 0.950207468879668, "dev_rows_ge_7_passing_rows": 4, "dev_rows_ge_7_precision_at_threshold": 1.0}, "anchor_count": 4, "max_per_wa": 2, "top_k": 50}
* 指标：
  * match@1：33.40%
  * match@5：41.30%
  * match@20：46.05%
  * match@50：46.05%
  * RMSE：0.16897342100460178
  * rows>=7 match@5：17.19%
  * rows>=7 match@20：19.00%
  * rows>=7 positive-any：19.91%
  * new positives beyond baseline：2
  * W/A-hit but match-fail rate：NA
* 结论：proposal ceiling increases, but insertion remains conservative; quality blocker is still rows>=7 geometry generation.
* 是否继续：no; termination condition 2 is now documented as satisfied
* 下一步：train a fresh pre-render residual/mixture generator if continuing scientific work.

## E7010: pairfield generator train-dev and val512 completion audit

* 时间：2026-06-18T18:45:35Z
* 目标：complete the joint free-param/lattice generator train-dev + val512 evaluation required by GPTprompt termination condition 2.
* 读取文件：
  * `/data/users/xsw/autodlmini/model/New_model/opentry_3/data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_4/data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features.jsonl`
  * `/data/users/xsw/autodlmini/model/New_model/opentry_4/eval/joint_generator_anchor_safe_val512_eval.json`
* 写入文件：
  * `eval/joint_generator_train_dev_val512_eval.json`
  * `cache/e7010_pairfield_train_dev_features_slim.jsonl`
  * `reports/joint_geometry_generator_report.md`
  * `reports/opentry_4_requirements_audit.md`
  * `reports/opentry_4_final_summary.md`
* 数据 split：E318 train-dev by `blake2b(sample_id) % 5 == 0`; val512 from E700/E7008; no test.
* 是否使用 test 信息：no
* 方法：pairfield_adam_repel train-only pair-distance/VPA stats; StructureMatcher labels for train-dev proposal candidates; val512 merged/anchor-safe evidence from E7008.
* 参数：preset=repel; train-dev min_row_count=7; max_input_candidates_per_sample=8; max_output_candidates_per_sample=50.
* 指标：
  * match@1：15.38%
  * match@5：30.77%
  * match@20：34.62%
  * match@50：34.62%
  * RMSE：0.17328691056997114
  * rows>=7 match@5：30.77%
  * rows>=7 match@20：34.62%
  * rows>=7 positive-any：34.62%
  * new positives beyond baseline：train-dev 0; val512 2
  * W/A-hit but match-fail rate：85.42%
* 结论：generator train-dev/val512 evaluation is complete; scientific result is weak but termination condition 2 evidence is now direct.
* 是否继续：no
* 下一步：only continue with a stronger pre-render residual/mixture generator if starting a new round.
