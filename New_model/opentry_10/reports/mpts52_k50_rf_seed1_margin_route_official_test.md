# MPTS-52 mpts52_k50_rf_seed1_margin_route Official Test

| metric | anchor | official | delta | rows>=7 anchor | rows>=7 official | rows>=7 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| match@1 | 25.23% | 25.791% | 0.561 pp | 22.49% | 23.053% | 0.563 pp |
| match@5 | 36.46% | 36.265% | -0.195 pp | 33.37% | 33.228% | -0.142 pp |
| match@20 | 43.96% | 43.824% | -0.136 pp | 41.04% | 40.939% | -0.101 pp |

Success standard met: `false`.

Failure reasons:
- no overall match metric improves by >= 1.0 pp
