# MP-20 K50 Conservative Selector OOF Search

Created: 2026-06-24T11:46:59+00:00

The selector is trained with K50 validation labels in 5-fold GroupKFold by sample_id. Evaluation is out-of-fold only.

## Baseline K20

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 72.731% | 29.135% |
| match@5 | 83.088% | 44.291% |
| match@20 | 87.631% | 55.571% |

## Strategy Results

| model | seed | strategy | anchor_keep | match@1 delta | match@5 delta | match@20 delta | rows>=7 @20 delta | supp slots/sample | CI95 delta@20 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hgb | 0 | unconstrained | None | -0.796 pp | -1.271 pp | -0.442 pp | -0.208 pp | 8.105 | NA |
| hgb | 0 | conservative | 14 | 0.000 pp | 0.000 pp | 0.343 pp | 0.969 pp | 3.481 | NA |
| hgb | 0 | conservative | 16 | 0.000 pp | 0.000 pp | 0.177 pp | 0.623 pp | 2.501 | NA |
| hgb | 0 | conservative | 18 | 0.000 pp | 0.000 pp | 0.188 pp | 0.277 pp | 1.364 | NA |
| hgb | 1 | unconstrained | None | -0.785 pp | -1.028 pp | -0.365 pp | -0.415 pp | 7.992 | NA |
| hgb | 1 | conservative | 14 | 0.000 pp | 0.000 pp | 0.287 pp | 0.415 pp | 3.452 | NA |
| hgb | 1 | conservative | 16 | 0.000 pp | 0.000 pp | 0.243 pp | 0.692 pp | 2.482 | NA |
| hgb | 1 | conservative | 18 | 0.000 pp | 0.000 pp | 0.177 pp | 0.208 pp | 1.366 | NA |
| hgb | 2 | unconstrained | None | -1.050 pp | -1.293 pp | -0.376 pp | -0.208 pp | 7.949 | NA |
| hgb | 2 | conservative | 14 | 0.000 pp | 0.000 pp | 0.254 pp | 0.900 pp | 3.449 | NA |
| hgb | 2 | conservative | 16 | 0.000 pp | 0.000 pp | 0.144 pp | 0.208 pp | 2.481 | NA |
| hgb | 2 | conservative | 18 | 0.000 pp | 0.000 pp | 0.111 pp | -0.069 pp | 1.359 | NA |

## Best Gate Candidate

No strategy satisfied the conservative validation gate in this run.

This report is validation-only and does not freeze an official-test strategy by itself.
