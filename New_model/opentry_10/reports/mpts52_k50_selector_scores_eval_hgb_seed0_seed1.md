# MPTS-52 K50 Selector Score Evaluation

Created: 2026-06-25T04:05:02+00:00

This report evaluates already generated OOF score files. It does not train a model.

## Baseline K20

| metric | value | rows>=7 |
| --- | ---: | ---: |
| match@1 | 30.020% | 5.323% |
| match@5 | 40.480% | 9.991% |
| match@20 | 48.000% | 14.747% |

## Strategies

| model | seed | strategy | anchor_keep | delta@1 | delta@5 | delta@20 | rows>=7 delta@20 | supp slots/sample |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgb | 0 | unconstrained | None | 0.360 pp | -0.500 pp | -0.100 pp | -0.611 pp | 10.815 |
| hgb | 0 | conservative | 10 | 0.000 pp | 0.000 pp | 0.020 pp | -0.393 pp | 6.845 |
| hgb | 0 | conservative | 12 | 0.000 pp | 0.000 pp | 0.220 pp | -0.262 pp | 5.791 |
| hgb | 0 | conservative | 14 | 0.000 pp | 0.000 pp | 0.120 pp | 0.000 pp | 4.610 |
| hgb | 0 | conservative | 16 | 0.000 pp | 0.000 pp | 0.240 pp | 0.349 pp | 3.281 |
| hgb | 0 | conservative | 18 | 0.000 pp | 0.000 pp | 0.160 pp | 0.044 pp | 1.757 |
| hgb | 1 | unconstrained | None | 0.300 pp | -0.320 pp | -0.120 pp | -0.349 pp | 11.269 |
| hgb | 1 | conservative | 10 | 0.000 pp | 0.000 pp | 0.260 pp | -0.131 pp | 7.234 |
| hgb | 1 | conservative | 12 | 0.000 pp | 0.000 pp | 0.140 pp | -0.349 pp | 6.125 |
| hgb | 1 | conservative | 14 | 0.000 pp | 0.000 pp | 0.100 pp | -0.262 pp | 4.853 |
| hgb | 1 | conservative | 16 | 0.000 pp | 0.000 pp | 0.240 pp | 0.044 pp | 3.441 |
| hgb | 1 | conservative | 18 | 0.000 pp | 0.000 pp | 0.160 pp | 0.044 pp | 1.831 |

## Best Gate Candidate

No evaluated score file satisfied the validation gate.
