# opentry_4 Final Summary

Time: 2026-06-18T18:45:35Z

## Completed Experiments

1. E7001 froze/audited E718 and imported the existing one-time full MPTS-52 test result from opentry_3 E734c without rerunning test.
2. E7002/E7007 built train/val hard-negative datasets and added top20/top50 rows>=7 diagnosis.
3. E7006 trained a sklearn hard-negative geometry energy model with a deterministic E318 train/train-dev split and evaluated it on val512.
4. E7004/E7008/E7009/E7010 audited the constrained pairfield free-param/lattice proposal generator on train-dev and val512, then evaluated train-dev-frozen anchor-safe insertion on val512.

## Final E718 Full-Test Metrics

| metric | value | delta vs GT-SG CrystaLLM |
|---|---:|---:|
| match@1 | 27.68% | +1.04 pp |
| match@5 | 35.85% | -0.73 pp |
| match@20 | 39.68% | -5.01 pp |
| match@50 | 42.15% | NA |
| RMSE@20 | 0.1240 | NA |

Rows>=7 full test: match@5=15.58%, match@20=17.48%, match@50=18.85%. The two-metrics +5pp target is not met (`False`), and no full-test feedback was used for tuning.

## New Method Results

| item | result |
|---|---:|
| energy train-dev AUC | 0.8873 |
| energy val512 AUC | 0.8665 |
| energy val512 rows>=7 pairwise accuracy | 0.7114285714285714 |
| generator train-dev candidates / samples | 208 / 26 |
| generator train-dev match@20 | 34.62% |
| generator train-dev new positives beyond baseline | 0 |
| generator val512 new positives beyond E424 top20 | 2 |
| anchor-safe val512 match@20 | 46.05% |
| anchor-safe rows>=7 match@20 | 19.00% |

## Effective

- E718 is frozen and full-test-reported without leakage, but does not clear the GT-SG +5pp target.
- Hard-negative diagnosis shows rows>=7 failures are mainly W/A/skeleton-hit geometry failures: E718 rows>=7 top50 missing samples = 179, with W/A/skeleton hit = 126.
- The constrained pairfield generator creates val512 rows>=7 new positives beyond baseline, proving the ceiling can move.

## Ineffective / Still Weak

- On train-dev, pairfield direct proposals underperform the E318 baseline and add 0 new positive samples beyond that baseline.
- On val512, direct proposal ordering is weak; anchor-safe insertion inserts conservatively and does not improve match@20.
- The system remains in a local optimum around rows>=7 coupled lattice/free-parameter/site-mapping failures.

## Next Routes

1. Train a real pre-render residual or mixture generator over lattice plus free parameters, using E700 positive cases as teacher/supervision but evaluating full val512 ceiling before selector work.
2. Use the sklearn energy model as generation guidance or rejection scoring, not as an ordinary full reranker.

Do not continue ordinary full rerank selectors, source-prior-only tuning, single-row free-param copy, source-free random priors, direct MSE-only regression, or post-render coordinate surgery as primary routes.
