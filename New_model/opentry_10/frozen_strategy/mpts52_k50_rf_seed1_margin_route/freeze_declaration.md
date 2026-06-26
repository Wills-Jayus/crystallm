# Freeze Declaration: mpts52_k50_rf_seed1_margin_route

Status: frozen pending test K50 generation.

This candidate is selected from validation-only OOF evidence in `reports/mpts52_k50_score_route_sweep_fast_rf.md`.

Frozen route:

- Score model: RF seed1
- Route rule: `best_score - rank1_score >= 0.12608004456999994`
- Routed action: rank all K50 candidates by RF score and keep top 20
- Non-routed action: keep baseline CrystaLLM-a GT-SG K20 order

Validation OOF result:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| match@1 | 30.020% | 32.440% | +2.420pp |
| match@5 | 40.480% | 41.680% | +1.200pp |
| match@20 | 48.000% | 48.580% | +0.580pp |
| rows>=7 match@1 | 5.323% | 5.977% | +0.654pp |
| rows>=7 match@5 | 9.991% | 9.948% | -0.044pp |
| rows>=7 match@20 | 14.747% | 14.616% | -0.131pp |

The official test dependency is not yet complete: opentry_10 currently has only MPTS-52 test K20 anchor candidates. Ranks 21-50 must be generated before this frozen route can produce official K20 output.

This declaration does not claim official success.
