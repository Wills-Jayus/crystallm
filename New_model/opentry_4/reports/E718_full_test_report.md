# E718 Full Test Report

Time: 2026-06-18T18:09:06Z

Full test source: imported existing frozen E734c artifact from opentry_3; opentry_4 did not rerun test. Test information used for tuning: no.

## Full MPTS-52 Test

| metric | E718 full test | GT-SG CrystaLLM baseline | delta |
|---|---:|---:|---:|
| match@1 | 27.68% | 26.64% | +1.04 pp |
| match@5 | 35.85% | 36.58% | -0.73 pp |
| match@20 | 39.68% | 44.69% | -5.01 pp |
| match@50 | 42.15% | NA | NA |
| RMSE@1 | 0.1020 | NA | NA |
| RMSE@5 | 0.1128 | NA | NA |
| RMSE@20 | 0.1240 | NA | NA |
| RMSE@50 | 0.1313 | NA | NA |

Rows>=7 full test: match@5=15.58%, match@20=17.48%, match@50=18.85%, RMSE@20=0.1465.

At least two +5pp metrics achieved: False. Individual +5pp pass flags: {"match@1": false, "match@5": false, "match@20": false}.

## Validation Comparison

| system | match@1 | match@5 | match@20 | match@50 | RMSE@20 |
|---|---:|---:|---:|---:|---:|
| E423 val512 | 33.01% | 40.82% | 45.51% | 48.24% | 0.1309 |
| E718 val512 | 33.01% | 41.99% | 45.90% | 48.24% | 0.1318 |
| E718 full test | 27.68% | 35.85% | 39.68% | 42.15% | 0.1240 |

E718 improves E423 on val512 mainly at match@5 (+1.17 pp) and match@20 (+0.39 pp), but full test is lower than val512 across all main match metrics. This is consistent with validation overfit or split shift. Per protocol, no full-test feedback is used to alter the frozen config.
