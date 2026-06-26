# MP-20 K50 Label Summary

Created: 2026-06-25T03:39:13+00:00

- Label rows: 250000 / 250000
- Complete materials: 5000 / 5000
- Coverage complete: True
- Target rows>=7 materials: 2292
- Complete rows>=7 materials: 2292
- K50 rescues over K20 among complete materials: 236

## Complete-Material Metrics

| budget | match_rate | RMSE | hits/materials | rows>=7 match_rate | rows>=7 hits/materials |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1 | 30.020% | 0.1255473167 | 1501/5000 | 5.323% | 122/2292 |
| K5 | 40.480% | 0.1202741142 | 2024/5000 | 9.991% | 229/2292 |
| K20 | 48.000% | 0.1291643317 | 2400/5000 | 14.747% | 338/2292 |
| K50 | 52.720% | 0.1342423268 | 2636/5000 | 18.325% | 420/2292 |

If coverage is incomplete, these are progress diagnostics over complete materials only. Formal selector gates require coverage_complete=true.

## Label Status Counts

```json
{
  "material_hard_timeout": 250,
  "ok": 244804,
  "parse_error": 4942,
  "rmsd_error": 1,
  "skipped_large": 3
}
```
