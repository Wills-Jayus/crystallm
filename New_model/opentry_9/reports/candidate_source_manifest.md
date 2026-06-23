# Candidate Source Manifest

All rows are based on read-only historical artifacts. Test match labels are not used for selector training or threshold selection.

| source name | dataset | split | candidate count | sample coverage | per-sample slots | official full-test | validation | pure model | strategy/retrieval/fusion | train selector | final test fusion | notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| crystallm_a_gt_sg_anchor_mp20_test | mp_20 | test | 180920 | 9046 | 20/20.0/20 | yes | no | no | no | no | yes | Anchor candidates recovered from opentry_8 strategy_fusion rows; MP-20 strategy_fusion is exactly this anchor. |
| crystallm_a_gt_sg_anchor_mpts52_test | mpts_52 | test | 161900 | 8095 | 20/20/20 | yes | no | no | no | no | yes | Anchor candidates recovered from opentry_8 strategy_fusion rows; one sample was missing in primary source. |
| strategy_fusion_mp20_test | mp_20 | test | 180920 | 9046 | 20/20.0/20 | yes | no | no | yes | no | no | Historical post-test artifact; no validation gate in opentry_9. |
| strategy_fusion_mpts52_test | mpts_52 | test | 161920 | 8096 | 20/20.0/20 | yes | no | no | yes | no | no | Historical post-test artifact; only one sample was coverage-repaired. |
| strategy_stablekey_hybrid_mp20_test | mp_20 | test | 177860 | 8893 | 20/20/20 | yes | no | no | yes | no | no | Historical test-only stablekey/SymCIF hybrid. Useful for post-hoc complement diagnosis only. |
| strategy_stablekey_hybrid_mpts52_test | mpts_52 | test | 153260 | 7663 | 20/20/20 | yes | no | no | yes | no | no | Historical test-only stablekey/SymCIF hybrid. Useful for post-hoc complement diagnosis only. |
| symcif_v4_mp20_val_baseline_k5 | mp_20 | val | 44370 | 8874 | 5/5.0/5 | no | yes | yes | no | yes | no | MP-20 validation K<=5 SymCIF-v4 structural candidate bank; not CrystaLLM anchor. |
| pure_gt_wa_geometry_mp20_val_k5 | mp_20 | val | 44370 | 8874 | 5/5.0/5 | no | yes | yes | no | no | no | GT-WA geometry upper-bound artifact, K<=5 only. |
| one_fix_hybrid_prior_mp20_val_k5 | mp_20 | val | 44370 | 8874 | 5/5.0/5 | no | yes | yes | no | yes | no | Validation-only one-fix prior selector artifact; diagnostic, not frozen for test. |
| symcif_v4_exact_cover_val_wa_candidates | mixed | val | 500 | NA | NA | no | yes | yes | no | yes | no | WA candidates without rendered official K20 CIF/eval; mechanism-only. |
| opentry_5_fixed_order_geometry_smokes | fold/dev | fold | NA | NA | NA | no | no | yes | no | no | no | Mechanism-analysis smoke/fold generations only. |
| opentry_6_stage_geometry_exactcover_folds | fold/dev | fold | NA | NA | NA | no | no | yes | no | no | no | Mechanism-analysis fold generations only. |
| crystallm_gt_sg_k50_k100_raw_pool | mp_20/mpts_52 | test/val | NA | NA | NA | no | no | no | no | no | no | No CrystaLLM-a GT-SG K50/K100 full raw pool was found; only opentry_2 partial atomtype K50 diagnostics exist. |

## opentry_9 Partial Validation Anchor Progress

| source name | dataset | split | candidate count | sample coverage | per-sample slots | validation | final test fusion | notes |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| crystallm_gt_sg_anchor_mp20_val_partial_k20 | mp_20 | val | 460 | 23 | 20/20/20 | yes, partial | no | Completed real validation-anchor shards only; not a full validation gate. |
| crystallm_gt_sg_anchor_mpts52_val_partial_k20 | mpts_52 | val | 160 | 8 | 20/20/20 | yes, partial | no | Completed real validation-anchor shards only; not a full validation gate. |
