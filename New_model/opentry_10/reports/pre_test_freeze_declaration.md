# Pre-Test Freeze Declaration

Created: 2026-06-25T04:27:27+00:00

This declaration freezes the primary official-test line before any new per-sample test evaluation is read.

## Primary Strategy

- ID: mpts52_rerank_only_hgb_seed2
- Dataset: MPTS-52
- Candidate source: CrystaLLM-a GT-SG official test anchor K20
- Final K budget: 20
- Strategy: rerank-only; no supplemental candidates and no test target labels.
- Ranker: HistGradientBoostingClassifier seed=2, trained on full MPTS-52 validation K20 candidate labels after OOF model selection.

## Validation Evidence

- match@1 delta: 1.220 pp; bootstrap CI95 [0.380, 2.100] pp
- match@5 delta: 0.680 pp; bootstrap CI95 [-0.040, 1.400] pp
- match@20 delta: 0.000 pp
- rows>=7 match@1 delta: -0.131 pp
- rows>=7 match@5 delta: -0.698 pp
- rows>=7 match@20 delta: 0.000 pp

Rows>=7@5 is a validation risk, but the formal primary gain is match@1 and the corresponding rows>=7@1 change is small with CI crossing zero.

## Frozen Artifacts

- Config: `/data/users/xsw/autodlmini/model/New_model/opentry_10/frozen_strategy/mpts52_rerank_only_hgb_seed2/config.json`
- Ranker: `/data/users/xsw/autodlmini/model/New_model/opentry_10/frozen_strategy/mpts52_rerank_only_hgb_seed2/ranker_hgb_seed2.joblib`
- Test candidates JSONL: `/data/users/xsw/autodlmini/model/New_model/opentry_10/generations/mpts52_rerank_only_hgb_seed2_test_k20_candidates.jsonl`
- Test generated tar: `/data/users/xsw/autodlmini/model/New_model/opentry_10/generations/mpts52_rerank_only_hgb_seed2_test_k20/tars/generated_data_atomtype_gt_sg.tar.gz`
- Test true tar prepared for post-freeze evaluation: `/data/users/xsw/autodlmini/model/New_model/opentry_10/generations/mpts52_official_test/tars/true.tar.gz`
