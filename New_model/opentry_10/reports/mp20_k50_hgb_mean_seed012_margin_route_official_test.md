# MP-20 mp20_k50_hgb_mean_seed012_margin_route Official Test

| metric | anchor | official | delta | rows>=7 anchor | rows>=7 official | rows>=7 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| match@1 | 71.67% | 70.230% | -1.440 pp | 62.37% | 27.200% | -35.170 pp |
| match@5 | 83.08% | 81.870% | -1.210 pp | 76.35% | 43.273% | -33.077 pp |
| match@20 | 87.81% | 87.486% | -0.324 pp | 82.61% | 55.273% | -27.337 pp |

Success standard met: `false`.

Failure reasons:
- no overall match metric improves by >= 1.0 pp
- match@20 drops by more than 0.2 pp
