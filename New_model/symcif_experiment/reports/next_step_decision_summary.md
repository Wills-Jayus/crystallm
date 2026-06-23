# Next Step Decision Summary

## 1. 当前 v4 end-to-end 结果

| experiment | match@1 | match@5 | match@20 | RMSE@20 | readable@20 | formula_ok@20 | atom_count_ok@20 | SG_ok@20 | bond_score@20 | bond_reasonable@20 | valid/strict_valid | WA hit@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 current | 34.8% | 52.0% | 56.6% | 0.1030 | 69.9% | 69.9% | 77.2% | 69.9% | 0.3344 | 15.7% | 0.0%/0.0% | 86.6% |
| WA upper-bound | 52.4% | 75.2% | 82.4% | 0.0157 | 70.4% | 70.4% | 77.0% | 70.4% | 0.3227 | 11.7% | 0.0%/0.0% | 86.6% |
| GT-WA + current geometry | 56.4% | 56.4% | 56.4% | 0.0973 | 99.6% | 99.6% | 98.6% | 99.6% | 0.7738 | 59.2% | 0.0%/0.0% | 100.0% |

v4 current 已经超过同等 evaluator 条件下的 baseline match@20=44.6%，也超过 baseline_minprompt match@20=47.8%。

## 2. Breakdown

| subset | samples | current match@20 | current WA hit@20 | current RMSE | WA upper match@20 | GT-WA/current-geometry match@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 500 | 56.6% | 86.6% | 0.1030 | 82.4% | 56.4% |
| n_sites>=6 | 175 | 13.1% | 72.6% | 0.1916 | 61.1% | 17.7% |
| num_elements>=4 | 192 | 47.9% | 83.9% | 0.1059 | 77.1% | 47.9% |
| SG=2 | 28 | 14.3% | 89.3% | 0.1002 | 67.9% | 10.7% |
| SG=65 | 12 | 41.7% | 33.3% | 0.1437 | 50.0% | 50.0% |
| SG=71 | 9 | 44.4% | 44.4% | 0.0475 | 44.4% | 77.8% |
| SG=127 | 8 | 75.0% | 62.5% | 0.0746 | 62.5% | 75.0% |

## 3. 瓶颈判断

- WA candidate/search/ranking：不是当前主瓶颈。当前 WA@20=86.6%，WA upper-bound match@20=82.4%；WA 命中后到 evaluator 成功只损失约 4.2 个百分点。
- geometry/free_params/lattice：是主瓶颈。GT WA + current geometry 的 match@20=56.4%，几乎等于 v4 current 的 56.6%；说明即使 WA 完全正确，当前 retrieval/fallback geometry 也只能做到约 56%。
- renderer/validator：需要修，但不是重建 match 的首要瓶颈。v4 expanded CIF 的 `valid/strict_valid` 为 0，主要因为 legacy evaluator 的 multiplicity loop 检查不适配 OrbitEngine 展开 CIF；readable/formula/SG/atom_count 与 StructureMatcher 结果仍然可用于本轮决策。
- data scale：复杂子集仍弱，尤其 n_sites>=6 current match@20=13.1%，WA upper-bound=61.1%，GT-WA/current-geometry=17.7%；这说明复杂结构需要 geometry model 加 oversampling/data scale 验证。

## 4. v2/full-data 大模型结论

新 v2 full_large：match@1=17.4%，match@5=35.4%，match@20=47.2%，RMSE@20=0.1468。它略高于同条件 baseline n20=44.6%，但低于同条件 baseline_minprompt n20=47.8%，也低于既有 v2_constrained n20=49.2% 和 v4 current=56.6%。训练曲线显示 step 750 后明显过拟合。

## 5. 是否继续 v4

继续 v4。按判定规则，current v4 full match@20 已超过 baseline；同时 WA upper-bound 高、full pipeline 低，最短路径不是继续深挖 WA@200，而是补 orbit-level geometry/free_params/lattice model。

## 6. 下一步最短路径

1. 训练 orbit-level geometry model：condition on formula + SG + WA，预测 lattice 与 free parameters；先用 GT WA 评测，目标把 GT-WA geometry bottleneck 从 56.4% 提到 70-75% 以上。
2. 接入 predicted WA top20：用同一 evaluator 跑 match@1/5/20，直接对比当前 v4 full 56.6% 和 WA upper-bound 82.4%。
3. 修 renderer/evaluator 兼容：OrbitEngine expanded CIF 增加或适配 multiplicity 信息，避免 `valid/strict_valid` 被 legacy 格式检查全部打零。
4. 对复杂子集做 data scale/oversampling 验证：优先 n_sites>=6、num_elements>=4、SG=65/71/127；不要只加局部规则。
5. v2 路线保留为 baseline/control：若继续训练大模型，使用早停和更强正则；但当前不应把它作为主线。

## 7. Artifacts

- v4 current：`reports/symcif_v4_full_eval_current/`
- WA upper-bound：`reports/symcif_v4_wa_upper_bound/`
- geometry bottleneck：`reports/symcif_v4_geometry_bottleneck/`
- full model comparison：`reports/full_model_vs_crystallm_small/summary.md`
- same-condition baseline：`eval_runs/baseline_reeval_same_as_v4_20260522/`
- same-condition baseline_minprompt：`eval_runs/baseline_minprompt_reeval_same_as_v4_20260522/`
- new v2 large eval：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/`
