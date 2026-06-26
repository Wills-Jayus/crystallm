# MPTS-52 K30 RF Best-Score Route Official Test

System: `mpts52_k30_rf_seed1_bestscore_route`

| metric | anchor | official | delta |
| --- | ---: | ---: | ---: |
| match@1 | 25.23% | 26.075% | 0.845 pp |
| match@5 | 36.46% | 36.228% | -0.232 pp |
| match@20 | 43.96% | 44.059% | 0.099 pp |

Formal success standard is not met: no match metric reaches +1.0 pp over the official anchor. match@20 is not degraded, but K1 is only +0.845 pp and K5 is below anchor.

Artifacts:
- `model/New_model/opentry_10/metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json`
- `metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_k1.raw.txt`
- `metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_k5.raw.txt`
- `metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_k20.raw.txt`
