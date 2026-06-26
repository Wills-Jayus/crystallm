# MPTS-52 K50 Conservative Selector OOF Search

Created: 2026-06-25T04:05:50+00:00

The selector is trained with K50 validation labels in 5-fold GroupKFold by sample_id. Evaluation is out-of-fold only.

## Baseline K20

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 30.020% | 5.323% |
| match@5 | 40.480% | 9.991% |
| match@20 | 48.000% | 14.747% |

## Strategy Results

| model | seed | strategy | anchor_keep | match@1 delta | match@5 delta | match@20 delta | rows>=7 @20 delta | supp slots/sample | CI95 delta@20 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rf | 1 | unconstrained | None | 2.180 pp | 1.040 pp | 0.660 pp | -0.349 pp | 12.008 | NA |
| rf | 1 | conservative | 10 | 0.000 pp | 0.000 pp | 0.700 pp | 0.131 pp | 7.518 | NA |
| rf | 1 | conservative | 12 | 0.000 pp | 0.000 pp | 0.620 pp | 0.262 pp | 6.309 | NA |
| rf | 1 | conservative | 14 | 0.000 pp | 0.000 pp | 0.460 pp | 0.218 pp | 4.967 | NA |
| rf | 1 | conservative | 16 | 0.000 pp | 0.000 pp | 0.560 pp | 0.218 pp | 3.477 | NA |
| rf | 1 | conservative | 18 | 0.000 pp | 0.000 pp | 0.460 pp | 0.175 pp | 1.847 | NA |

## Best Gate Candidate

No strategy satisfied the conservative validation gate in this run.

This report is validation-only and does not freeze an official-test strategy by itself.
