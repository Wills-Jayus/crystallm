# E718 Frozen Before Full Test

Frozen note time: 2026-06-18T18:09:06Z

Important audit note: opentry_3 already contains a frozen E718 full-test artifact (`e734c_eval_e718_frozen_mpts52_test_full`). To respect the "full test only once" rule, opentry_4 imports and reports that artifact instead of rerunning test.

## Frozen Config

- Compatibility model: `model/New_model/opentry_3/reports/e425_gbdt_e318_train_to_e424_val512/compat_model.joblib`
- Scorer: `model/New_model/opentry_3/scripts/opentry_score_rendered_candidates_apply.py`
- Selector: `model/New_model/opentry_3/scripts/opentry_apply_selective_replacement.py`
- Candidate generation: E724 skeleton infer, E725d/E726c assignment infer, E727 merge, E728 renderer predictions, E730 E421 renderer config, E731b E422 selfscore top50.
- Evaluator: `model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py`
- Data split: MPTS-52 test eval/infer under opentry_3.
- Test label tuning: no.
- Parameters: threshold=0.0024707304479371964, anchor_count=4, max_per_wa=2, top_k=50.
