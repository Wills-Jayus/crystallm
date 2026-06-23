# Opentry-3 Final Summary

时间：2026-06-14 UTC

## 目标与边界

本轮只写入：

```text
/data/users/xsw/autodlmini/model/New_model/opentry_3
```

目标是用 CrystalFormer/WyFormer 风格的对称感知中间表示推进：

```text
formula + GT-SG -> Wyckoff skeleton / W-A sequence -> CIF
```

本轮没有使用 test GT CIF、test GT W/A、StructureMatcher match/rms、row_count 训练/排序/过滤/调参；test 未被读取用于模型训练或模型选择。当前已经进入 validation CIF rendering smoke，但尚未冻结 full-test config，也没有跑 full test。

## 最新状态快照

截至 E172，opentry_3 仍未达到 full-test freeze gate，但 validation 指标已有实质推进：

| best item | run | value |
|---|---|---:|
| best full match@1 | E151/E166-E172 | 32.03% |
| best full match@5 | E166 `row_aligned_chem_quality + soft VPA global` | 40.62% |
| best full match@20 | E156/E161 `row_aligned_chem_quality global` | 45.31% |
| best full match@50 | E156/E157/E161/E162 | 45.31% |
| best rows>=7 match@5 | E156/E157/E161/E162/E166-E172 | 18.33% |
| best rows>=7 match@20/50 | E156/E157/E161/E162/E166-E172 | 20.00% / 20.00% |

目标状态：

- match@1 validation best 已超过 `31.64%` target。
- match@5 validation best `40.62%` 仍低于 `41.58%` target。
- match@20 validation best `45.31%` 仍低于 `49.69%` target。
- 因此尚未满足至少两个 match 指标 +5pp 的冻结条件。
- 没有跑 full test。

当前方法判断：

- `row_aligned_chem_quality` 是有效方向：它提升 K5/K20 和 rows>=7 转化。
- `source_vpa_calibrated_soft + global` 进一步把 K5 推到 40.62%，并改善 RMSE@1 到 0.0942，但仍没有打开 full-test gate。
- 下一步应继续 chemical analogue source family，并做 source-conditioned free-param / multimodal source proposal，尤其针对 rows>=7。

## 已完成实验

| id | 内容 | 结论 |
|---|---|---|
| E00 | 审计 `opentry` / `opentry_2` / `Log_GPT` | 旧路线瓶颈是 W/A recall；K50/RF、低 LR SFT、fixed fusion 不是本轮主线。 |
| E01 | 构建 MPTS-52 train/val canonical Wyckoff representation | 成功；train 25,998 条、val 4,727 条；train-only priors 已输出。 |
| E02 | 128 train / 16 val tiny debug | pipeline 可运行，但仅调试意义。 |
| E03 | 2k train learned A/B/C | skeleton@5=37.50%，W/A=0，说明 assignment 是瓶颈。 |
| E04 | 加入 train-only composition-exact prior assignment | W/A@5 提到 6.25%，仍低。 |
| E05 | 8k train smoke | skeleton@5=65.62%，W/A union@5=6.25%；W/A 未随 skeleton 提升。 |
| E06 | K50 prior-only diagnostic | 非向量化 beam 太慢，中止。 |
| E07 | fixed-skeleton exact-cover DP assignment | 证明 assignment decoder 上限高；GT skeleton 下 val32 W/A@50=68.75%。 |
| E08 | E05 skeleton candidates 扩到 val128 | skeleton@5=64.84%，rows>=7=65.00%。 |
| E09 | predicted skeleton + DP assignment val128 | symbolic W/A gate 局部通过：W/A@50=42.19%，W/A@100=45.31%。 |
| E10-E15 | E09 W/A -> collision-aware deterministic CIF val128 | CIF closure 100%，但 match@50 只有 21.09%，rows>=7 为 0。 |
| E16-E19 | e08 row-conditioned geometry val128 | closure 100%，full match@50=30.47%，rows>=7 match@50=8.33%；仍不足以冻结 full test。 |
| E20-E26 | 修正 canonical W/A key，重跑 e08 val128 | canonical W/A@50=57.03%，match@50=31.25%。 |
| E27-E33 | train-only skeleton prior augmentation + e08 | skeleton@50=87.50%，W/A@50=75.00%，best val128 match@50=39.84%。 |
| E34-E38 | count-aware model-only skeleton generator + budgeted DP | learned skeleton@50=87.50%，rows>=7=85.00%；canonical W/A@5=66.41%，rows>=7 W/A@50=66.67%。 |
| E39-E42 | count-aware W/A candidates -> e08 renderer diagnostics | W/A-diverse match@50=39.06%；geometry interleave 更差；仍不冻结 full test。 |
| E43-E47 | fixed-orbit duplicate mask 前移到 assignment DP | conversion duplicate fixed drops 从 2,512 降到 0；full W/A@50=74.22%，但 match@50 仍 39.06%。 |
| E48-E52 | hybrid geometry plan 与 W/A-to-match 诊断 | hybrid 未改善 K50；rows>=7 W/A-to-match 转化率约 17.5%，几何/free-param 是主瓶颈。 |

## 关键指标

早期 E05 小样本 gate：

| subset | skeleton@1 | skeleton@5 | W/A prior@5 | W/A C@5 | W/A union@5 | unique W/A union |
|---|---:|---:|---:|---:|---:|---:|
| full val32 | 34.38% | 65.62% | 3.12% | 6.25% | 6.25% | 3.47 |
| rows>=7 | 38.89% | 66.67% | 0.00% | 5.56% | 5.56% | 3.78 |
| atoms>=12 | 36.67% | 63.33% | 3.33% | 6.67% | 6.67% | 3.37 |

E09 raw DP 后的 symbolic gate 来自 val128，但 raw DP key 后续被发现存在 repeated-orbit 排列问题，需以 E24/E29 canonical W/A key 为准：

| branch | subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | W/A@100 | unique W/A |
|---|---|---:|---:|---:|---:|---:|---:|
| predicted skeleton + DP | full val128 | 24.22% | 33.59% | 35.16% | 42.19% | 45.31% | 67.80 |
| predicted skeleton + DP | rows>=7 | 16.67% | 16.67% | 16.67% | 23.33% | 25.00% | 90.83 |
| predicted skeleton + DP | atoms>=12 | 23.28% | 29.31% | 30.17% | 37.93% | 39.66% | 71.68 |
| GT skeleton diagnostic | full val128 | 46.09% | 59.38% | 64.84% | 72.66% | 75.00% | 48.85 |

Skeleton val128：

| subset | skeleton@1 | skeleton@5 | unique skeleton |
|---|---:|---:|---:|
| full val128 | 39.06% | 64.84% | 4.66 |
| rows>=7 | 48.33% | 65.00% | 4.32 |
| atoms>=12 | 38.79% | 61.21% | 4.63 |

对比旧基线：

- opentry_2 E04/E07 policy search 的 W/A@100 约 12-13%。
- opentry_3 E09 predicted-skeleton DP 当前 W/A@50=42.19%，W/A@100=45.31%，已经明显超过旧 W/A recall。
- rows>=7 W/A@50=23.33%，仍弱但已经高于旧路线。
- 因此 symbolic validation gate 当前局部通过，可以进入 validation CIF rendering smoke；仍不能 full test。

Canonical key 修正后的当前 best symbolic gate：

| source | subset | skeleton@50 | W/A@5 | W/A@20 | W/A@50 | unique W/A |
|---|---|---:|---:|---:|---:|---:|
| E24 canonical E09 | full val128 | 59.38% | 53.12% | 57.03% | 57.03% | 4.61 |
| E24 canonical E09 | rows>=7 | 55.00% | 48.33% | 50.00% | 50.00% | 2.90 |
| E29 augmented skeleton | full val128 | 87.50% | 57.03% | 75.00% | 75.00% | 9.16 |
| E29 augmented skeleton | rows>=7 | 85.00% | 51.67% | 61.67% | 61.67% | 6.38 |

## Validation CIF Smoke

E15 collision-aware deterministic renderer:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 8.59% | 17.19% | 19.53% | 21.09% | 0.1070 | 0.1106 | 0.1533 | 0.1643 |
| rows>=7 | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | NA | NA |

E19 e08 row-conditioned geometry with composition-exact filtering:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 17.19% | 24.22% | 29.69% | 30.47% | 0.1345 | 0.1095 | 0.1274 | 0.1328 | 52.34% | 3.23 |
| rows>=7 | 6.67% | 6.67% | 8.33% | 8.33% | 0.0964 | 0.0964 | 0.0999 | 0.0999 | 45.00% | 1.30 |
| atoms>=12 | 12.93% | 17.24% | 22.41% | 23.28% | 0.0989 | 0.1210 | 0.1539 | 0.1609 | 49.14% | 2.82 |

结论：

- e08 geometry 确实改善 match，但 comp-filter 后 W/A 多样性很低。
- E27-E33 的 train-only skeleton prior augmentation 明显改善了 skeleton/W-A recall，并带来当前 best validation CIF smoke：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 21.88% | 32.03% | 37.50% | 39.84% | 0.1611 | 0.1681 | 0.1325 | 0.1215 |
| rows>=7 | 10.00% | 11.67% | 11.67% | 13.33% | 0.1946 | 0.1731 | 0.1660 | 0.1614 |
| atoms>=12 | 18.97% | 29.31% | 31.03% | 33.62% | 0.1606 | 0.1606 | 0.1464 | 0.1442 |

- 当前 validation CIF 指标仍明显低于 full-test 冻结要求，不允许跑 full test。
- E35 已把 train-prior skeleton recall 转成模型侧 generator 能力；下一步应集中在 B-stage assignment 去重/排序和 rows>=7 geometry/free-param 转化。

E35 count-aware model-only skeleton generator 已经把 E27 的 train-prior skeleton recall 转成模型侧能力：

| subset | skeleton@1 | skeleton@5 | skeleton@20 | skeleton@50 | unique skeleton@50 |
|---|---:|---:|---:|---:|---:|
| full val128 | 45.31% | 70.31% | 81.25% | 87.50% | 45.23 |
| rows>=7 | 50.00% | 68.33% | 81.67% | 85.00% | 42.65 |
| atoms>=12 | 42.24% | 67.24% | 79.31% | 86.21% | 44.74 |

E45 canonical W/A gate after moving duplicate fixed-orbit masking into DP:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|
| full val128 | 45.31% | 66.41% | 74.22% | 74.22% | 12.59 |
| rows>=7 | 48.33% | 66.67% | 66.67% | 66.67% | 6.55 |
| atoms>=12 | 43.10% | 63.79% | 71.55% | 71.55% | 11.66 |

Validation CIF diagnostics:

| run | geometry plan | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---|---:|---:|---:|---:|---:|
| E40 | W/A-diverse e08 | 17.97% | 27.34% | 35.94% | 39.06% | 11.67% |
| E42 | geometry-interleaved e08 | 17.97% | 21.88% | 27.34% | 36.72% | 11.67% |
| E47 | fixed-mask W/A-diverse e08 | 17.97% | 27.34% | 35.94% | 39.06% | 11.67% |
| E49 | fixed-mask hybrid5x3 e08 | 17.97% | 24.22% | 36.72% | 39.06% | 11.67% |
| E51 | fixed-mask hybrid3x2 e08 | 17.97% | 23.44% | 35.94% | 39.06% | 11.67% |

Interpretation:

- A-stage skeleton generation is now a genuine model-side improvement.
- Moving fixed-orbit duplicate masking into DP is a retained B-stage improvement: post-conversion duplicate fixed drops went from 2,512 to 0.
- CIF match is still limited by geometry/free-parameter transfer. Geometry interleave and hybrid geometry schedules either reduce W/A diversity or only improve K20 by <1 pp, without improving K50.
- No full-test config is frozen.

## 是否达到 CrystaLLM +5pp

尚未达到。

原因是 validation CIF smoke 仍低于冻结标准。当前 best val128 为：`match@1=32.03%`、`match@5=40.62%`、`match@20=45.31%`、`match@50=45.31%`，rows>=7 best 仍只有 `18.33/20.00/20.00`。本轮仍没有 frozen config 进入 full test，因此没有合法的 MPTS-52 test match/RMSE@1/5/20。按无泄露协议，不能用 test 来反调。

参考目标仍为：

| metric | CrystaLLM GT-SG | +5pp target |
|---|---:|---:|
| MPTS-52 match@1 | 26.64% | 31.64% |
| MPTS-52 match@5 common-subset | 36.58% | 41.58% |
| MPTS-52 match@20 | 44.69% | 49.69% |

## 污染风险审计

- 训练数据：只使用 MPTS-52 train representation。
- train priors：只从 train split 构建。
- validation：只用于 symbolic gate 和调参判断。
- test：未用于本轮模型训练、排序、过滤、调参或 full test。
- 未使用 StructureMatcher label 训练/rank/filter。
- 未使用 CrystaLLM test predictions 作为 primary candidate source。
- `row_count` 只作为分析字段输出，没有作为 generation input。

## 当前判断

这条路线的 A 阶段现在有更强的模型侧信号：E35 count-aware skeleton generator 在 val128 达到 `skeleton@50=87.50%`、rows>=7 `85.00%`，基本复现 E27 train-prior augmentation，但不再依赖 train-prior skeleton 作为外部候选补丁。

Validation CIF smoke 暴露了下一层瓶颈：W/A recall 已明显提高，但 geometry/free-param/lattice 尤其在 rows>=7 上仍不能稳定转化为 match。E47 full `W/A@50=74.22%` 只转成 `match@50=39.06%`；rows>=7 `W/A@50=66.67%` 只转成 `match@50=11.67%`，W/A-to-match 转化率约 17.5%。下一步不应 full test，应继续：

1. 在 B-stage 保留 fixed-orbit duplicate mask，并继续提高 canonical W/A@50，同时保住 E45 的 W/A@1/5。
2. 改进 rows>=7 geometry/free-param transfer；单纯增加 e08 geometry ranks、interleave 或 hybrid schedule 都不够。
3. 训练 learned geometry/free-param head 或更好的 train-only geometry proposal，使 W/A 命中能转化为 StructureMatcher match。
4. 优化 skeleton beam/DP 速度；E28 augmented DP val128 已需约 209 秒。
5. 只有 val CIF match 达到冻结标准后，才允许一次 full test。

## 2026-06-13 追加更新

新增 geometry diagnostics 后，full test 仍不能冻结。

| run | change | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| E54 | W/A/skeleton-priority train source retrieval | 17.97% | 27.34% | 35.16% | 37.50% | 11.67% | worse than E47 |
| E57 | train-only one-shot geometry net | 10.94% | 17.97% | 21.88% | 22.66% | 0.00% | rejected |
| E60 | e08 + wrapped jitter hybrid5x5 | 17.97% | 21.09% | 30.47% | 38.28% | 11.67% | rejected |
| E62 | e08 + wrapped jitter hybrid2x3 | 17.97% | 25.00% | 35.16% | 37.50% | 11.67% | rejected |

E55c trained a train-only geometry net (`formula+GT-SG+W/A rows -> lattice/free params`) with best val loss at epoch 4, but CIF conversion failed: E56 rank1 SG-ok was only 27.20%, and E57 rows>=7 match was 0.00%.

E59-E62 tested small deterministic multimodal jitter around e08 retrieval. It preserved CIF readability/composition but did not improve K50 or rows>=7, and it reduced early W/A diversity.

Current conclusion is sharper: the symbolic A/B stages are useful, but deterministic learned geometry and simple local perturbations are not enough. The next viable geometry direction needs a stronger multimodal or constraint-aware free-param/lattice proposal, not another single MSE regression head, source-priority retrieval tweak, or small jitter schedule.

## 2026-06-13 B-stage neural assignment update

E63-E74 returned to the requested CrystalFormer/WyFormer-style symbolic route.

New model-side component:

- `opentry_train_assignment_scorer.py`
- train-only assignment scorer: `formula + GT-SG + skeleton orbit + remaining composition -> element`
- trained on MPTS-52 train only
- best validation-state metrics: top1 `60.84%`, top5 `86.86%`, MRR `72.72%`

Best reduced-budget val128 symbolic result:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E45 prior full-budget reference | 45.31% | 66.41% | 74.22% | 74.22% | 66.67% | 12.59 |
| E71c neural assignment s10p10 | 47.66% | 70.31% | 75.78% | 75.78% | 76.67% | 10.00 |

Interpretation:

- The neural assignment scorer clearly helps rows>=7 W/A recall and early W/A ranking.
- It does not yet justify CIF rendering as a frozen symbolic gate: full W/A@50 is only `+1.56 pp` over E45, and unique W/A@50 is lower.
- Full-budget neural DP (`s20p20`) is still too slow in the current implementation and was interrupted; no CIF render or full test was run from E63-E74.

Current frozen status remains unchanged:

- No opentry_3 config is frozen for full MPTS-52 test.
- No legal MPTS-52 test match/RMSE@1/5/20 has been produced in opentry_3.
- The best next step is to optimize neural exact-DP so full-budget W/A@50 and unique W/A both clearly exceed E45 before re-entering CIF validation.

## 2026-06-13 neural-first merge update

After E63-E74, I added a label-free symbolic merge/rescore pass over exact DP candidates.

New scripts:

- `opentry_rescore_dp_candidates.py`
- `opentry_merge_dp_candidates_priority.py`

The best B-stage result is now E85 neural-first merge:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E45 fixed-mask reference | 45.31% | 66.41% | 74.22% | 74.22% | 66.67% | 12.59 |
| E85 neural-first merge | 47.66% | 70.31% | 77.34% | 78.91% | 76.67% | 13.74 |

This cleared the symbolic val gate, so I ran one validation-only CIF conversion with the same e08 row-conditioned geometry protocol as E47.

E87 validation CIF result:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 21.09% | 28.12% | 35.94% | 40.62% | 0.0927 | 0.1123 | 0.1397 | 0.1488 |
| rows>=7 | 6.67% | 10.00% | 15.00% | 16.67% | 0.0964 | 0.1566 | 0.1689 | 0.1997 |
| atoms>=12 | 16.38% | 21.55% | 29.31% | 34.48% | 0.1138 | 0.1179 | 0.1631 | 0.1758 |

Current interpretation:

- E87 is the best opentry_3 validation K50 result so far: full match@50 `40.62%`, rows>=7 match@50 `16.67%`.
- It improves E47 full match@1, match@5, match@50 and rows>=7 match@50.
- It still is not enough to freeze full test; rows>=7 conversion is far below W/A recall, and match@5 remains weak.
- No test split was used and no full MPTS-52 test was launched.

Next step:

- Treat E83/E85 as the current symbolic W/A backbone.
- Use E84/E86 as the input for geometry conversion experiments.
- Do not switch back to RF rankers, CrystaLLM candidate fusion, low-LR CIF SFT, or fixed K50 candidate hybrid as the main route.

## 2026-06-13 hybrid geometry recheck

E88-E89 retested `hybrid_top_wa` on the stronger E84/E85 neural-first W/A set.

| run | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | full W/A@5 | full W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E87 W/A-diverse e08 | 28.12% | 35.94% | 40.62% | 16.67% | 72.66% | 78.91% |
| E89 hybrid5x3 e08 | 27.34% | 35.94% | 40.62% | 16.67% | 57.03% | 78.91% |

Conclusion:

- Hybrid geometry scheduling does not improve K20/K50 or rows>=7 conversion.
- It reduces early W/A diversity, so E87 remains the best validation CIF result.
- No full MPTS-52 test config is frozen yet.
- The next useful direction is a new geometry proposal/scoring family, not more `hybrid_top_wa`, interleave, jitter, RF ranker, CrystaLLM candidate fusion, or low-LR CIF SFT.

## 2026-06-13 GT-free generated-CIF self-score update

E90-E93 added a generated-CIF self-score ordering layer over E86 candidates. The score uses only candidate-internal signals: parser/readability, formula and composition exactness, detected SG consistency, min interatomic distance, volume per atom, geometry distance, and original rank tie-breaks. It does not use StructureMatcher labels or test data for sorting.

Best current validation CIF result is now E91 self-score diverse:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 28.91% | 32.81% | 38.28% | 40.62% | 0.1096 | 0.1152 | 0.1349 | 0.1488 |
| rows>=7 | 11.67% | 11.67% | 16.67% | 16.67% | 0.1884 | 0.1884 | 0.1997 | 0.1997 |
| atoms>=12 | 23.28% | 25.86% | 31.90% | 34.48% | 0.1089 | 0.1236 | 0.1591 | 0.1758 |

Against E87, E91 improves full match@1/5/20 by `+7.81/+4.69/+2.34 pp`, while match@50 remains `40.62%`. This means the self-score layer improves early ordering but does not solve the geometry proposal ceiling.

Current status:

- No full MPTS-52 test config is frozen.
- E91 is still below the target gate for full test, especially match@5 and rows>=7.
- Next step should generate additional multimodal train-only geometry proposals for the strong E85 W/A set, then apply the GT-free self-score layer before validation.

## 2026-06-13 larger geometry pool update

E94-E96 expanded the e08 geometry proposal pool before applying the same GT-free self-score selection.

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E91 K50-pool self-score | 28.91% | 32.81% | 38.28% | 40.62% | 16.67% | 12.80 |
| E96 K100-pool self-score | 29.69% | 32.81% | 39.84% | 39.84% | 16.67% | 13.20 |

Interpretation:

- E96 is the current early-rank validation best.
- E91 remains the K50-ceiling validation best.
- The result is still below the full-test freeze gate; no full test was run.
- Next step should preserve E96 early-rank gains while recovering E91 tail recall, likely through a proposal/selection rule that mixes self-score quality with explicit W/A/geometry diversity.

## 2026-06-13 prefix-tail selection update

E97-E100 merged the E96 early-rank ordering with the E91 K50 tail using a deterministic GT-free prefix-tail policy. The selected current best is E100 (`prefix_k=20`).

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E91 K50 self-score | 28.91% | 32.81% | 38.28% | 40.62% | 16.67% | 12.80 |
| E96 K100 self-score | 29.69% | 32.81% | 39.84% | 39.84% | 16.67% | 13.20 |
| E100 prefix20-tail | 29.69% | 32.81% | 39.84% | 40.62% | 16.67% | 12.80 |

E100 combines the best early ranks with the best K50 ceiling seen so far in opentry_3 validation:

- full match@1/5/20/50 = `29.69 / 32.81 / 39.84 / 40.62`
- rows>=7 match@1/5/20/50 = `11.67 / 11.67 / 16.67 / 16.67`
- atoms>=12 match@1/5/20/50 = `24.14 / 25.86 / 33.62 / 34.48`

Status remains not frozen:

- match@5 is still far below the `41.58%` target.
- rows>=7 conversion is still poor.
- No full MPTS-52 test was run.
- Next work should add new rows>=7 geometry proposal capacity rather than further tuning deterministic selection over the same candidates.

## 2026-06-13 row-prototype geometry update

E101-E106 added a train-only row-wise geometry prototype source. Standalone row-prototype rendering was weak, but using it only as a conservative tail-fill improved the current best.

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E100 prefix20-tail | 29.69% | 32.81% | 39.84% | 40.62% | 16.67% | 12.80 |
| E104 row-prototype standalone | 17.97% | 24.22% | 28.12% | 30.47% | 1.67% | 13.59 |
| E106 E100 + row-prototype tail-fill | 29.69% | 32.81% | 40.62% | 41.41% | 18.33% | 12.90 |

Current best validation CIF result is E106:

- full match@1/5/20/50 = `29.69 / 32.81 / 40.62 / 41.41`
- rows>=7 match@1/5/20/50 = `11.67 / 11.67 / 18.33 / 18.33`
- atoms>=12 match@1/5/20/50 = `24.14 / 25.86 / 34.48 / 35.34`
- complex_flag match@1/5/20/50 = `24.37 / 27.73 / 36.13 / 36.97`

Interpretation:

- Row-prototype geometry adds a small amount of tail recall but is too noisy as a standalone renderer.
- It improves match@20/50 and rows>=7 K50 slightly.
- The result still does not pass the freeze gate: match@1 and match@5 remain below target, and rows>=7 conversion is still weak.
- No full MPTS-52 test was run.

## 2026-06-13 source-consistent row alignment update

E108-E113 added `row_aligned_knn`, a source-consistent geometry strategy:

- source selection remains train-only e08 row-conditioned KNN;
- lattice is copied/scaled from one source;
- free params are copied after aligning target rows to rows within that same source by orbit/element/free-symbol/multiplicity/site-symmetry cost;
- no StructureMatcher labels or test data are used in proposal or ranking.

This is the current best validation result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | atoms>=12 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E106 previous best | 29.69% | 32.81% | 40.62% | 41.41% | 18.33% | 35.34% | 12.90 |
| E111 row_aligned_knn | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 36.21% | 13.20 |

E111 details:

- full match@1/5/20/50 = `31.25 / 35.16 / 41.41 / 42.19`
- full RMSE@1/5/20/50 = `0.1137 / 0.1174 / 0.1382 / 0.1396`
- rows>=7 match@1/5/20/50 = `13.33 / 13.33 / 18.33 / 18.33`
- atoms>=12 match@1/5/20/50 = `25.86 / 28.45 / 35.34 / 36.21`
- complex_flag match@1/5/20/50 = `26.05 / 30.25 / 36.97 / 37.82`

Freeze status:

- E111 nearly reaches the MPTS-52 match@1 +5pp target: `31.25%` vs required `31.64%`.
- It still does not pass two +5pp metrics.
- match@5 remains far below `41.58%`, and rows>=7 conversion is still weak.
- No full MPTS-52 test was run.

## 2026-06-13 row-aligned source scoring update

E114-E121 tested source-priority and scoring variants around the E111 `row_aligned_knn` renderer.

`row_aligned_priority` did not improve the validation match gate:

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@50 |
|---|---:|---:|---:|---:|---:|
| E111 row_aligned_knn diverse | 31.25% | 35.16% | 41.41% | 42.19% | 0.1396 |
| E117 row_aligned_priority | 28.91% | 33.59% | 39.06% | 40.62% | 0.1317 |

It improves RMSE but loses match, so it is not the new main path.

Global GT-free self-score over the E109 row-aligned pool improved K50:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E111 diverse self-score | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 13.20 |
| E119 global self-score | 31.25% | 35.16% | 41.41% | 43.75% | 18.33% | 10.62 |
| E121 global prefix20 + diverse tail | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 13.01 |

Current best records:

- Best K50 validation ceiling: E119, full match/RMSE@1/5/20/50 = `31.25/35.16/41.41/43.75` and `0.1137/0.1039/0.1320/0.1457`.
- Best diversity-preserving validation result: E111, full match/RMSE@1/5/20/50 = `31.25/35.16/41.41/42.19` and `0.1137/0.1174/0.1382/0.1396`, unique W/A@50 `13.20`.
- Best compromise ordering: E121, same early match as E119/E111 with unique W/A@50 `13.01` and lower RMSE@50 than E111.

Gate status:

- Full test remains unfrozen.
- No opentry_3 config has passed two +5pp metrics.
- Current best match@1 `31.25%` is still below the `31.64%` target.
- Current best match@5 `35.16%` is far below `41.58%`.
- Current best match@20 `41.41%` is far below `49.69%`.
- The dominant bottleneck is rows>=7 geometry conversion: symbolic rows>=7 W/A recall remains high, but rows>=7 CIF match@50 is only `18.33%`.

Pollution and leakage check:

- E114-E121 used train-only source/prototype statistics and validation-only StructureMatcher evaluation.
- Test split was not used for training, sorting, filtering, tuning, or full evaluation.
- No CrystaLLM test predictions were used as primary candidates.
- No RF ranker, test-label rerank, fallback, GT-WA input, or low-LR CIF SFT was used as the main route.

Next step:

- Continue within the source-consistent row-aligned geometry family.
- Add train-only multimodal lattice/free-parameter proposal capacity for rows>=7, then apply GT-free self-score and validate on val before any full test.

## 2026-06-13 source-cluster lattice negative result

E122-E125 tested a train-only `source_cluster_quantile` lattice proposal:

- row-aligned W/A and free-param transfer stayed unchanged;
- lattice was aggregated from train-only row-aligned source pool quantiles;
- GT-free diverse self-score selected top50;
- validation StructureMatcher was used only for evaluation.

Result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | full W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E111 row_aligned source lattice | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 78.91% |
| E125 source_cluster_quantile lattice | 25.78% | 31.25% | 35.16% | 35.94% | 10.00% | 78.91% |

Gain/loss vs E111:

- full K50 gained 2 samples and lost 10;
- rows>=7 K50 gained 0 and lost 5.

Conclusion:

- Independent lattice quantile aggregation is rejected.
- The symbolic W/A set is not the failure source here; W/A@50 stayed `78.91%`.
- The lattice must remain compatible with the same source-conditioned free params, especially for rows>=7.
- Current best is unchanged:
  - E119 for K50 ceiling: `31.25/35.16/41.41/43.75`;
  - E111 for diversity-preserving baseline: `31.25/35.16/41.41/42.19`.
- Full test remains unfrozen.

## 2026-06-13 source-health row-aligned update

E126-E131 added `row_aligned_quality`, which keeps coherent source lattice/free-param transfer but penalizes train sources with weak structured geometry health:

- failed row expansion;
- failed free-param re-extraction when free symbols are present;
- missing declared free params.

This is train-only and GT-free; no StructureMatcher label or test data is used for sorting.

Result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E119 previous K50 best | 31.25% | 35.16% | 41.41% | 43.75% | 15.00% | 18.33% | 10.62 |
| E129 quality diverse | 31.25% | 35.16% | 40.62% | 44.53% | 15.00% | 18.33% | 12.80 |
| E131 quality global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |

Current best validation model:

- E131 full match/RMSE@1/5/20/50 = `31.25/35.94/41.41/44.53` and `0.1056/0.1047/0.1376/0.1562`.
- E131 rows>=7 match/RMSE@1/5/20/50 = `13.33/16.67/18.33/18.33` and `0.1284/0.1205/0.1450/0.1450`.
- E131 atoms>=12 match@1/5/20/50 = `25.86/29.31/35.34/38.79`.
- E131 complex_flag match@1/5/20/50 = `26.05/31.09/36.97/40.34`.

Target status:

- Not achieved.
- Best match@1 `31.25%` is still below `31.64%`.
- Best match@5 `35.94%` is still below `41.58%`.
- Best match@20 `41.41%` is still below `49.69%`.
- No full test was run.

Updated next step:

- Continue from `row_aligned_quality`.
- Improve rows>=7 source-conditioned geometry modes; current rows>=7 match@50 remains `18.33%` despite rows>=7 W/A@50 `76.67%`.
- Do not switch to RF ranker, low-LR CIF SFT, CrystaLLM test predictions, or test-time candidate fusion as the main route.

## 2026-06-13 wider row_aligned_quality pool

E132-E136 widened `row_aligned_quality` rendering to K100/g10 over the same E84 neural-first W/A candidates, then applied GT-free self-score top50.

Result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E131 K50/global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |
| E134 K100/global | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 10.66 |
| E136 K100/diverse | 31.25% | 34.38% | 39.84% | 43.75% | 13.33% | 18.33% | 13.02 |

Current best split:

- Best early/K5 validation result: E131, full match@1/5/20/50 = `31.25/35.94/41.41/44.53`.
- Best K20/K50 validation ceiling: E134, full match@1/5/20/50 = `31.25/35.16/42.19/45.31`.
- Best rows>=7 K5 remains E131 at `16.67%`; rows>=7 K50 remains stuck at `18.33%`.

Target status remains not achieved:

- match@1: best `31.25%` vs target `31.64%`;
- match@5: best `35.94%` vs target `41.58%`;
- match@20: best `42.19%` vs target `49.69%`.

No full test was run. The next real bottleneck is rows>=7 source-conditioned geometry quality, not just candidate-pool width or post-hoc ordering.

## 2026-06-13 train-only VPA prior self-score

E137-E140 added an optional train-only volume-per-atom prior to the generated-CIF self-score. The prior uses only MPTS-52 train structured JSONL statistics and is disabled by default, so earlier scoring remains reproducible.

Motivation:

- rows>=7 W/A-hit/match cases have higher VPA and slightly lower geometry distance than W/A-hit/no-match cases;
- no-W/A rows have substantially worse source health;
- a train-only VPA prior was a low-cost, leakage-safe check for whether ordering was missing a simple geometry-scale signal.

Validation results:

| run | train VPA weight | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | RMSE@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E131 K50/global | 0.0 | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 0.1047 |
| E134 K100/global | 0.0 | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 0.1040 |
| E138 VPA prior | 2.0 | 30.47% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% | 0.1131 |
| E140 VPA prior | 0.5 | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 0.1011 |

Conclusion:

- VPA prior does not change the current best split.
- E138 recovers E131 K5 and E134 K20/K50 but lowers match@1.
- E140 is essentially E134 with slightly better RMSE@5.
- rows>=7 match@50 remains stuck at `18.33%`.
- Full test remains unfrozen; no opentry_3 config meets two +5pp targets.

Current best split:

- Best early/K5 validation result: E131, full match@1/5/20/50 = `31.25/35.94/41.41/44.53`.
- Best K20/K50 validation ceiling: E134, full match@1/5/20/50 = `31.25/35.16/42.19/45.31`.
- Best achieved match@1 remains below `31.64%`; best match@5 remains below `41.58%`; best match@20 remains below `49.69%`.

Next step:

- Move away from scalar self-score tweaks.
- Add stronger source-conditioned multimodal lattice/free-param generation for rows>=7 while keeping train/val/test separation and W/A-first modeling.

## 2026-06-13 hybrid geometry allocation

E141-E152 tested whether high-confidence W/A candidates need more coherent source-conditioned geometry modes before W/A diversity is expanded.

Important protocol note:

- E144/E145 were invalid because `--max-records 128` was omitted, making the denominator full val while candidates covered only val128. They are not used below.
- Valid comparable results are E146/E147/E151/E152.

Results:

| run | plan | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | W/A@50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E131 | K50 baseline | global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 78.91% | 12.80 |
| E134 | K100 baseline | global | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 76.56% | 10.66 |
| E146 | hybrid10x10 | global | 28.12% | 32.81% | 39.06% | 42.19% | 15.00% | 18.33% | 74.22% | 7.45 |
| E147 | hybrid10x10 | diverse | 28.12% | 32.03% | 38.28% | 42.19% | 13.33% | 18.33% | 76.56% | 8.66 |
| E151 | hybrid5x5 | global | 32.03% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% | 76.56% | 10.41 |
| E152 | hybrid5x5 | diverse | 32.03% | 35.16% | 40.62% | 43.75% | 13.33% | 18.33% | 78.91% | 13.02 |

Updated best validation result:

- Best combined ordering is now E151:
  - full match/RMSE@1 = `32.03% / 0.1109`;
  - full match/RMSE@5 = `35.94% / 0.1064`;
  - full match/RMSE@20 = `42.19% / 0.1393`;
  - full match/RMSE@50 = `45.31% / 0.1530`.
- E151 clears the match@1 target on validation, but this is not enough to freeze full test.

Gate status:

- Still not achieved.
- match@5 remains below `41.58%`.
- match@20 remains below `49.69%`.
- rows>=7 match@50 remains `18.33%`.
- No full test was run.

Conclusion:

- Conservative front-loaded geometry allocation helps rank-1.
- Aggressive geometry allocation harms W/A diversity and match.
- The next step remains a real rows>=7 source-conditioned geometry proposal improvement, not another scalar scorer or rank-budget tweak.

## 2026-06-13 chemical analogue source proposal

E153-E162 added `row_aligned_chem_quality`, a train-only chemical-analogue source selector for the renderer.

What changed:

- Source selection now includes a periodic-table formula chemistry distance instead of relying mainly on exact formula overlap.
- The source pool is still train-only.
- Lattice and free params are still transferred coherently from one source.
- StructureMatcher is used only for validation evaluation.
- No test data, GT W/A input, RF ranker, oracle rerank, or CrystaLLM prediction source is used.

Results:

| run | method | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@20/50 | W/A@50 | unique W/A@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E151 | quality + hybrid5x5 global | 32.03% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% / 18.33% | 76.56% | 10.41 |
| E156 | chem quality global | 31.25% | 39.06% | 45.31% | 45.31% | 18.33% | 20.00% / 20.00% | 76.56% | 10.57 |
| E157 | chem quality diverse | 31.25% | 39.84% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | 78.91% | 13.23 |
| E161 | chem + hybrid5x5 global | 31.25% | 39.06% | 45.31% | 45.31% | 18.33% | 20.00% / 20.00% | 76.56% | 10.39 |
| E162 | chem + hybrid5x5 diverse | 31.25% | 39.84% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | 78.91% | 13.23 |

Updated best validation metrics:

- Best match@1: E151, `32.03%`, RMSE@1 `0.1109`.
- Best match@5: E157/E162, `39.84%`, RMSE@5 about `0.133`.
- Best match@20: E156/E161, `45.31%`, RMSE@20 about `0.155`.
- Best rows>=7 match@5/20/50: `18.33% / 20.00% / 20.00%`.

Gate status:

- Still not achieved.
- One validation metric clears target: match@1 `32.03%` vs target `31.64%`.
- match@5 remains below target: `39.84%` vs `41.58%`.
- match@20 remains below target: `45.31%` vs `49.69%`.
- No full test was run.

Interpretation:

- Chemical analogue source proposal is the first recent change that materially improves K5/K20 and rows>=7 conversion.
- The cost is worse RMSE and loss of E151 rank-1 strength.
- Next work should keep the chemical analogue source family but add source-conditioned lattice/free-param calibration to improve precision.

## 2026-06-14 source VPA calibration update

E163-E172 added train-only source VPA calibration to `row_aligned_chem_quality`.

What changed:

- Expected VPA is estimated from train-only chemical analogues and fallback buckets.
- Lattice/free params still come from one coherent train source.
- Soft calibration rescales source lattice volume with strength 0.5; hard calibration uses strength 1.0.
- Sorting remains GT-free generated-CIF self-score; StructureMatcher is only used for validation evaluation.

Results:

| run | method | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E156 | chem quality | global | 31.25% | 39.06% | 45.31% | 45.31% | 0.1100 | 0.1273 | 20.00% | 10.57 |
| E157 | chem quality | diverse | 31.25% | 39.84% | 44.53% | 45.31% | 0.1100 | 0.1332 | 20.00% | 13.23 |
| E166 | chem + soft VPA | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 20.00% | 10.59 |
| E167 | chem + soft VPA | diverse | 32.03% | 39.06% | 42.97% | 45.31% | 0.0942 | 0.1219 | 20.00% | 13.23 |
| E171 | chem + hard VPA | global | 32.03% | 39.84% | 44.53% | 45.31% | 0.0969 | 0.1252 | 20.00% | 10.55 |
| E172 | chem + hard VPA | diverse | 32.03% | 37.50% | 42.19% | 45.31% | 0.0969 | 0.1154 | 20.00% | 13.23 |

Updated best validation metrics:

- Best match@1: `32.03%`.
- Best match@5: E166, `40.62%`, now 0.96 pp below the `41.58%` target.
- Best match@20: E156/E161, `45.31%`, still below the `49.69%` target.
- Best rows>=7 match@20/50 remains `20.00% / 20.00%`.

Gate status:

- Still not achieved.
- Only one match metric clears +5pp target.
- No full MPTS-52 test was run.

Interpretation:

- Soft VPA calibration is useful and should be retained as the current K5-best validation default.
- Hard VPA over-corrects and hurts K5.
- The remaining gap is not simple volume scale; rows>=7 needs better source-conditioned free-param/source-mode compatibility before full test can be frozen.

## 2026-06-14 source-mode allocation check

E173-E182 checked whether soft-VPA chemical geometry benefits from allocating extra coherent train-source modes to the top W/A candidates.

Results:

| run | plan | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | rows>=7 match@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E166 | rank0-only | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 20.00% |
| E176 | hybrid5x5 | global | 30.47% | 40.62% | 45.31% | 45.31% | 0.0864 | 20.00% |
| E177 | hybrid5x5 | diverse | 30.47% | 37.50% | 42.97% | 45.31% | 0.0864 | 20.00% |
| E181 | hybrid2x2 | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0938 | 20.00% |
| E182 | hybrid2x2 | diverse | 32.03% | 39.06% | 42.97% | 45.31% | 0.0938 | 20.00% |

Conclusion:

- `hybrid5x5` is too heavy: it can recover K20 but loses match@1.
- `hybrid2x2` ties E166 on match and slightly improves RMSE@1, but does not improve the gate.
- E166/E181 remain below the match@5 target (`40.62%` vs `41.58%`).

Diagnostic:

- In E166 full K5, W/A hits are 88/128 but match hits are only 52/128.
- In rows>=7 K5, W/A hits are 42/60 but match hits are only 11/60.
- 44 full samples have W/A@5 true but match@5 false; 32 are rows>=7.

Current next step:

- Stop spending effort on source allocation/hybrid scheduling.
- Focus on train-only source-conditioned free-param/source-mode compatibility or joint lattice/free-param proposal for rows>=7.
- Full test remains closed.

## 2026-06-14 free-param prior source penalty

E183-E189 tested a train-only free-param manifold penalty inside chemical source selection.

Implementation:

- Added `row_aligned_chem_param_quality` to the renderer.
- The penalty uses only train free-param distributions by orbit/element/free-symbol context.
- Full K100 attempts were too slow initially; after binary-search optimization I ran a K20 val128 smoke.

Results:

| run | method | full match@1 | full match@5 | full match@20 | rows>=7 match@5 | rows>=7 match@20 |
|---|---|---:|---:|---:|---:|---:|
| E166 | current best soft-VPA global | 32.03% | 40.62% | 44.53% | 18.33% | 20.00% |
| E188 | param-quality global K20 smoke | 29.69% | 36.72% | 40.62% | 16.67% | 18.33% |
| E189 | param-quality diverse K20 smoke | 29.69% | 37.50% | 40.62% | 18.33% | 18.33% |

Conclusion:

- The train-only free-param prior is compliant but negative.
- It should not be expanded to K100 or full test.
- Current best remains E166/E181, with validation full match@1/5/20/50 = `32.03 / 40.62 / 44.53 / 45.31`.
- The next viable geometry work needs a joint source/free-param proposal, not an independent scalar prior.

## 2026-06-14 strict physical self-score

E190-E193 tested a stronger GT-free physical self-score over the E163 soft-VPA candidate pool.

Results:

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 | standard global | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% | 20.00% |
| E192 | strict physical global | 25.00% | 35.94% | 45.31% | 45.31% | 16.67% | 20.00% |
| E193 | strict physical diverse | 25.00% | 39.06% | 44.53% | 45.31% | 15.00% | 20.00% |

Conclusion:

- Strict physical scoring improves candidate-internal min-distance/VPA but damages early W/A/skeleton ranking.
- It is not adopted.
- Current best remains E166/E181.
- Full test remains closed.

## 2026-06-14 physical source selection

E194-E198 added `row_aligned_chem_physical_select`, a GT-free renderer strategy that renders the top 3 chemical train sources for each rank-0 W/A candidate and chooses the generated CIF with the best internal health score.

Results:

| run | method | full match@1 | full match@5 | full match@20 | RMSE@1 | RMSE@5 | rows>=7 match@5 | W/A@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E166 | current best soft-VPA global | 32.03% | 40.62% | 44.53% | 0.0942 | 0.1295 | 18.33% | 76.56% |
| E197 | physical-select global | 30.47% | 34.38% | 39.84% | 0.1050 | 0.1123 | 16.67% | 77.34% |
| E198 | physical-select diverse | 30.47% | 33.59% | 39.84% | 0.1050 | 0.1180 | 15.00% | 77.34% |

Conclusion:

- The selector improves generated-CIF health, but damages validation match.
- It is not adopted and will not be expanded.
- Current best remains E166/E181, with only match@1 clearing the +5pp target.
- Full test remains closed.

## 2026-06-14 geometry-net lattice batching

E199-E204 revisited the old geometry-net lattice diagnostic. E199 confirmed the original per-candidate inference path was still impractical; E200 added batched per-sample lattice inference in the renderer.

Results:

| run | lattice mode | full match@1 | full match@5 | full match@20 | rows>=7 match@5 | rows>=7 match@20 |
|---|---|---:|---:|---:|---:|---:|
| E166 | source VPA soft | 32.03% | 40.62% | 44.53% | 18.33% | 20.00% |
| E203 | geometry-net global | 28.12% | 32.03% | 36.72% | 8.33% | 10.00% |
| E204 | geometry-net diverse | 28.12% | 32.03% | 36.72% | 8.33% | 10.00% |

Conclusion:

- Batching fixed the runtime blocker.
- The existing geometry-net lattice model is harmful for validation match.
- It is not adopted.
- The next geometry model should be source-conditioned residual transfer, not direct W/A-only lattice prediction.
- Full test remains closed.

## 2026-06-14 source-conditioned residual geometry

E205-E213 implemented a first source-conditioned residual geometry route.

Artifacts:

- `scripts/opentry_train_source_residual_geometry.py`
- `scripts/opentry_render_source_residual_geometry.py`
- `runs/e209_source_residual_geometry_fastsrc_1k_5epoch`
- `reports/e210b_render_e84_source_residual_e209_val32_k5_smoke`
- `reports/e213_eval_e211_source_residual_global_val32_match`

Results:

| method | subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | 78.12% | 2.88 |
| E213 residual self-score | full | 25.00% | 25.00% | 0.3035 | 0.2736 | 81.25% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | 83.33% | 2.72 |
| E213 residual self-score | rows>=7 | 5.56% | 5.56% | 0.3770 | 0.3770 | 83.33% | 4.56 |

Conclusion:

- The residual model is leakage-safe and aligned with the requested source-conditioned geometry direction.
- The current small model is not adopted: match is much worse than E166 and composition/readability drop to about 94-96%.
- The route needs cached source-pair data and validity-preserving residual constraints before larger training.
- Current best remains E166/E181.
- Full test remains closed.

## 2026-06-14 cached source-residual update

New artifacts:

- `scripts/opentry_build_source_residual_examples.py`
- Updated `scripts/opentry_train_source_residual_geometry.py`
- Updated `scripts/opentry_render_source_residual_geometry.py`
- `data/source_residual_geometry_mpts52/e214_cache_train1024_val256_fulltrain_selector`
- `runs/e215_source_residual_cached_constrained_1k_3epoch`
- `runs/e220_source_residual_cached_tight_1k_3epoch`
- `reports/e224_eval_e223_source_residual_tight_global_val32_match`

Leakage status:

- E214 source selector uses only MPTS-52 train records.
- E214 cache targets are train/val only.
- No test records, test GT, StructureMatcher labels, row_count input, oracle rerank, or fallback were used for training or sorting.
- StructureMatcher was used only for val evaluation.

Result summary:

| method | subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | 78.12% | 2.88 |
| E213 residual self-score | full | 25.00% | 25.00% | 0.3035 | 0.2736 | 81.25% | 4.38 |
| E224 tight cached residual | full | 34.38% | 37.50% | 0.1579 | 0.1789 | 81.25% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | 83.33% | 2.72 |
| E224 tight cached residual | rows>=7 | 16.67% | 16.67% | 0.2118 | 0.2118 | 83.33% | 4.56 |

Conclusion:

- Cached source-pair training is now functional and should remain as infrastructure.
- Tight delta constraints improve over E213 and recover first-32 match@1, but match@5 remains below the E166 reference.
- This branch is not adopted and will not be scaled to val128/full test.
- Current best remains E166/E181: val128 full `32.03 / 40.62 / 44.53 / 45.31`; only match@1 clears the +5pp target.
- Full test remains closed.

Next:

- Build cached predicted-W/A source-pair examples and train a source-mode/no-move or coherent-source mixture objective.
- Keep the main route Wyckoff-symbolic + source-conditioned geometry; do not return to K50/RF/low-LR SFT as primary routes.

## 2026-06-14 source-mode selector smoke

New artifacts:

- `scripts/opentry_build_source_mode_examples.py`
- `scripts/opentry_train_source_mode_selector.py`
- `data/source_mode_geometry_mpts52/e225_source_mode_train1024_val256_k8`
- `runs/e226_source_mode_selector_train1024_val256_k8`

Leakage status:

- Source pool uses train records only.
- Training labels are train lattice/free-param transfer errors, not StructureMatcher match/rms.
- Val labels are used only for model selection/reporting.
- No test records, no test GT, no oracle rerank, no fallback.

Result summary:

| run | subset | heuristic error | selected error | oracle best error | selected gain |
|---|---|---:|---:|---:|---:|
| E225 oracle | val full | 1.7472 | - | 0.4064 | 1.3408 oracle |
| E226 best epoch 13 | val full | 1.7472 | 1.5610 | 0.4064 | +0.1862 |
| E225 oracle | val rows>=7 | 3.1732 | - | 0.6461 | 2.5270 oracle |
| E226 best epoch 13 | val rows>=7 | 3.1732 | 2.7459 | 0.6461 | +0.4272 |

Conclusion:

- Source-mode choice has real headroom; the heuristic rank0 source is best in only about 24% of val groups.
- The first listwise MLP gives a positive transfer-label smoke, including rows>=7, but it has not yet been rendered to CIF.
- Current best validation CIF model remains E166/E181; full test remains closed.

Next:

- Integrate E226 source-mode selector into a val32 renderer smoke before any larger training.
- If CIF match improves, scale source-mode examples and add no-move/source-mixture calibration.

## 2026-06-14 source-mode renderer integration

New artifacts:

- Updated `scripts/opentry_render_source_residual_geometry.py`
- `reports/e227_render_e84_source_residual_e220_sourcemode_e226_val32_k5`
- `reports/e228_eval_e227_source_residual_sourcemode_raw_val32_match`
- `reports/e229_selfscore_e227_source_residual_sourcemode_global_val32_k5`
- `reports/e230_eval_e229_source_residual_sourcemode_global_val32_match`

Leakage status:

- Source-mode selector was trained from train/val transfer-error labels only.
- Rendering/evaluation used validation first 32 only.
- No test data, test GT, StructureMatcher labels for sorting/training, oracle rerank, fallback, or CrystaLLM primary candidates were used.

Result summary:

| method | subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | 78.12% | 2.88 |
| E224 tight cached residual | full | 34.38% | 37.50% | 0.1579 | 0.1789 | 81.25% | 4.38 |
| E230 source-mode residual | full | 34.38% | 37.50% | 0.2722 | 0.2522 | 81.25% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | 83.33% | 2.72 |
| E230 source-mode residual | rows>=7 | 11.11% | 16.67% | 0.2356 | 0.2505 | 83.33% | 4.56 |

Conclusion:

- E226 source-mode selection does not improve CIF top-5 after integration.
- Self-score restores full match@1 to the E224/E166 first-32 level, but K5 remains below E166 and rows>=7 is worse.
- Current best validation CIF model remains E166/E181; full test remains closed.

Next:

- Keep source-mode code paths and datasets.
- The next aligned step is a stronger no-move/source-mixture geometry objective trained from train-only source pools, with val used only for selection.

## 2026-06-14 source-mode margin calibration

New artifacts:

- Updated `scripts/opentry_render_source_residual_geometry.py`
- `scripts/opentry_tune_source_mode_margin.py`
- `reports/e231_tune_source_mode_margin_e226`
- `reports/e232_render_e84_source_residual_e220_sourcemode_e226_margin075_val32_k5`
- `reports/e235_eval_e234_source_residual_sourcemode_margin075_global_val32_match`

Leakage status:

- Threshold was selected from train/val transfer-label examples only.
- StructureMatcher was used only for val32 evaluation after rendering.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Result summary:

| method | subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | 78.12% | 2.88 |
| E230 pure source-mode residual | full | 34.38% | 37.50% | 0.2722 | 0.2522 | 81.25% | 4.38 |
| E235 margin source-mode residual | full | 34.38% | 37.50% | 0.2634 | 0.2497 | 81.25% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | 83.33% | 2.72 |
| E235 margin source-mode residual | rows>=7 | 11.11% | 16.67% | 0.2356 | 0.2505 | 83.33% | 4.56 |

Conclusion:

- Margin calibration improves transfer-label source selection but does not improve rendered CIF match.
- Current best validation CIF model remains E166/E181, with val128 full `32.03 / 40.62 / 44.53 / 45.31`.
- Full test remains closed; the two +5pp objective is not yet met.

Next:

- Train a stronger coherent source-mixture/no-move geometry model rather than adding more thresholds to E226.

## 2026-06-14 source-mode mixture smoke

New artifacts:

- Updated `scripts/opentry_render_source_residual_geometry.py`
- `reports/e236_render_e84_source_residual_e220_sourcemode_mixture3_val32_k7`
- `reports/e237_eval_e236_source_residual_sourcemode_mixture3_raw_val32_match`
- `reports/e238_selfscore_e236_source_residual_sourcemode_mixture3_global_val32`
- `reports/e239_eval_e238_source_residual_sourcemode_mixture3_global_val32_match`
- `reports/e240_selfscore_e236_source_residual_sourcemode_mixture3_diverse_val32`
- `reports/e241_eval_e240_source_residual_sourcemode_mixture3_diverse_val32_match`

Leakage status:

- The source-mode selector uses train/val transfer-label data only.
- StructureMatcher was used only for validation evaluation after rendering.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Result summary:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | W/A@20 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | - | - | 78.12% | - | 2.88 |
| E235 margin source-mode | full | 34.38% | 37.50% | - | 0.2634 | 0.2497 | - | 81.25% | - | 4.38 |
| E239 mixture global | full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1946 | 0.1901 | 81.25% | 84.38% | 3.16 |
| E241 mixture diverse | full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1954 | 0.1901 | 81.25% | 84.38% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | - | - | 83.33% | - | 2.72 |
| E241 mixture diverse | rows>=7 | 16.67% | 16.67% | 16.67% | 0.2039 | 0.2039 | 0.2039 | 83.33% | 83.33% | 4.56 |

Conclusion:

- Source-mode mixture improves full K5 over E235, but remains below the E166 first-32 reference and does not improve rows>=7.
- High W/A recall with weak rows>=7 match confirms the current bottleneck is coherent geometry under predicted W/A.
- Current best validation CIF model remains E166/E181, with val128 full `32.03 / 40.62 / 44.53 / 45.31`.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Keep source-mixture rendering code.
- Train a stronger joint source-mixture/no-move residual geometry model before any larger validation or final full test.

## 2026-06-14 train-time source-mixture residual smoke

New artifacts:

- `scripts/opentry_build_source_mixture_residual_examples.py`
- `data/source_residual_geometry_mpts52/e242_mixture_rank0_top2_train1024_val256`
- `runs/e243_source_residual_mixture_rank0_top2_1k_4epoch`
- `reports/e244_render_e84_source_residual_e243_mixturetrained_sourcemode_mixture3_val32_k7`
- `reports/e245_eval_e244_source_residual_e243_mixturetrained_raw_val32_match`
- `reports/e247_eval_e246_source_residual_e243_mixturetrained_global_val32_match`
- `reports/e249_eval_e248_source_residual_e243_mixturetrained_diverse_val32_match`

Leakage status:

- E242 labels are lattice/free-param transfer-error labels from train/val source-mode examples, not StructureMatcher match/rms.
- Source records come from train only.
- Validation StructureMatcher is evaluation-only.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Result summary:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E239 E220 mixture global | full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1946 | 0.1901 | 81.25% | 3.16 |
| E241 E220 mixture diverse | full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1954 | 0.1901 | 81.25% | 4.38 |
| E247 E243 mixture-trained global | full | 31.25% | 37.50% | 40.62% | 0.1526 | 0.2019 | 0.2064 | 71.88% | 3.03 |
| E249 E243 mixture-trained diverse | full | 31.25% | 37.50% | 40.62% | 0.1526 | 0.2038 | 0.2064 | 75.00% | 4.38 |
| E241 E220 mixture diverse | rows>=7 | 16.67% | 16.67% | 16.67% | 0.2039 | 0.2039 | 0.2039 | 83.33% | 4.56 |
| E249 E243 mixture-trained diverse | rows>=7 | 11.11% | 11.11% | 16.67% | 0.1937 | 0.1937 | 0.2347 | 72.22% | 4.56 |

Conclusion:

- E242/E243 is aligned with the requested source-mixture/no-move direction, but it does not improve validation CIF match.
- Training the residual head equally on rank0 and transfer-top sources weakens early-rank match, likely because transfer-error top-source supervision is not a reliable rendered-CIF-quality proxy.
- Current best validation CIF model remains E166/E181, with val128 full `32.03 / 40.62 / 44.53 / 45.31`.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Keep E242 as infrastructure.
- Replace equal regression over transfer-top sources with a source-conditioned no-op/gating residual head or a GT-free rendered-quality source-context proxy before scaling.

## 2026-06-14 gated source-mixture residual smoke

New artifacts:

- Updated `scripts/opentry_train_source_residual_geometry.py`
- Updated `scripts/opentry_render_source_residual_geometry.py`
- `runs/e250_source_residual_gated_mixture_rank0_top2_1k_4epoch`
- `reports/e251_render_e84_source_residual_e250_gated_sourcemode_mixture3_val32_k7`
- `reports/e252_eval_e251_source_residual_e250_gated_raw_val32_match`
- `reports/e254_eval_e253_source_residual_e250_gated_global_val32_match`
- `reports/e256_eval_e255_source_residual_e250_gated_diverse_val32_match`

Leakage status:

- Training uses the E242 train/val source-mixture residual cache only.
- Labels are lattice/free-param transfer errors, not StructureMatcher match/rms.
- Validation StructureMatcher is evaluation-only.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Result summary:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | - | - | - | 78.12% | 2.88 |
| E241 E220 mixture diverse | full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1954 | 0.1901 | 81.25% | 4.38 |
| E249 E243 mixture-trained diverse | full | 31.25% | 37.50% | 40.62% | 0.1526 | 0.2038 | 0.2064 | 75.00% | 4.38 |
| E254 E250 gated global | full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1534 | 0.1534 | 75.00% | 2.91 |
| E256 E250 gated diverse | full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1710 | 0.1534 | 78.12% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | - | - | - | 83.33% | 2.72 |
| E256 E250 gated diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 77.78% | 4.56 |

Conclusion:

- The gated no-op residual head is a real improvement over equal source-mixture regression: full val32 K1/K5 reaches `37.50/43.75`.
- It still does not pass the complex-subset gate. rows>=7 remains `11.11/16.67/16.67`, below the E166 rows>=7 prefix reference.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Keep the gated residual implementation.
- Do not scale E250 directly.
- Train a source/geometry quality proxy or mixture-of-experts source selector aimed at complex W/A geometry compatibility before larger validation.

## 2026-06-14 rendered-health source-mode selector diagnostic

New artifacts:

- `scripts/opentry_build_source_render_quality_examples.py`
- `data/source_render_quality_mpts52/e257_e250_render_health_train512_val256_pool8`
- `runs/e258_source_mode_render_health_e250_train512_val256`
- `reports/e259_render_e84_source_residual_e250_sourcemode_e258_renderhealth_mixture3_val32_k7`
- `reports/e260_eval_e259_source_residual_e250_sourcemode_e258_renderhealth_raw_val32_match`
- `reports/e262_eval_e261_source_residual_e258_renderhealth_global_val32_match`
- `reports/e264_eval_e263_source_residual_e258_renderhealth_diverse_val32_match`
- `reports/e265_tune_source_mode_margin_e258_renderhealth`

Leakage status:

- E257 uses train/val records only.
- Source records come from train only.
- Labels are GT-free rendered-health proxy scores, not StructureMatcher match/rms.
- Validation StructureMatcher is evaluation-only.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Proxy signal:

| item | value |
|---|---:|
| train / val source groups | 512 / 256 |
| source_pool_k | 8 |
| residual geometry used for labels | E250 gated residual |
| val full oracle health gain | 67.82 |
| val rows>=7 oracle health gain | 104.23 |
| E258 best val full selected health gain | +1.42 |
| E258 best val rows>=7 selected health gain | +3.06 |

Result summary:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E254 E250 gated global | full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1534 | 0.1534 | 75.00% | 2.91 |
| E256 E250 gated diverse | full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1710 | 0.1534 | 78.12% | 4.38 |
| E262 E258 render-health global | full | 34.38% | 40.62% | 43.75% | 0.1155 | 0.1295 | 0.1532 | 81.25% | 2.78 |
| E264 E258 render-health diverse | full | 34.38% | 40.62% | 43.75% | 0.1155 | 0.1490 | 0.1532 | 84.38% | 4.38 |
| E256 E250 gated diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 77.78% | 4.56 |
| E264 E258 render-health diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 83.33% | 4.56 |

Conclusion:

- E257 is a useful diagnostic: source choice has real GT-free rendered-health oracle space, especially for rows>=7.
- E258 does not convert that space into CIF match. Full K1/K5 are below E254/E256 on the same validation prefix, and rows>=7 match is unchanged.
- Margin tuning does not rescue the branch.
- E258 is not adopted, and it should not be scaled to val128 or full test.
- Current best validation CIF model remains E166/E181, with val128 full `32.03 / 40.62 / 44.53 / 45.31`.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Keep the rendered-health source builder.
- Move from source-only numeric selector features to direct rendered-candidate quality scoring or a stronger per-rendered-candidate model with GT-free features.
- Continue targeting complex W/A geometry compatibility before any full test.

## 2026-06-14 source-expanded rendered-candidate self-score diagnostic

New artifacts:

- Updated `scripts/opentry_render_source_residual_geometry.py`
- Updated `scripts/opentry_selfscore_rendered_cifs.py`
- `reports/e266_render_e84_source_residual_e250_sourceexpand8_val32_k7`
- `reports/e269_eval_e268_sourceexpand8_global_val32_match`
- `reports/e272_render_e84_source_residual_e250_sourceexpand8_val64_k7`
- `reports/e275_eval_e273_sourceexpand8_global_val64_match`
- `reports/e276_render_e84_source_residual_e250_sourceexpand8_val128_k7`
- `reports/e279_eval_e277_sourceexpand8_global_val128_match`

Leakage status:

- Source records are train split only.
- Validation W/A predictions come from the existing opentry_3 W/A model output.
- Ranking uses GT-free rendered-CIF self-score only.
- StructureMatcher is validation evaluation-only.
- No test data, test GT, oracle rerank, fallback, or primary CrystaLLM candidates were used.

Method:

- Add `--source-expand-k` to render multiple train-source contexts per predicted W/A.
- Use E250 gated residual geometry on each expanded source context.
- Reorder the rendered pool with global GT-free self-score.
- Check val32, val64, and val128 without changing the config between prefixes.

Result summary:

| method | prefix | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source-expand8 global | 32 | full | 34.38% | 40.62% | 46.88% | 0.1172 | 0.1342 | 0.1729 | 78.12% | 2.25 |
| source-expand8 global | 32 | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.0443 | 0.0443 | 77.78% | 2.22 |
| source-expand8 global | 64 | full | 35.94% | 43.75% | 46.88% | 0.1337 | 0.1531 | 0.1685 | 71.88% | 2.33 |
| source-expand8 global | 64 | rows>=7 | 13.33% | 20.00% | 20.00% | 0.1480 | 0.1056 | 0.1054 | 73.33% | 2.27 |
| source-expand8 global | 128 | full | 27.34% | 35.94% | 41.41% | 0.1128 | 0.1514 | 0.1812 | 64.06% | 2.28 |
| source-expand8 global | 128 | rows>=7 | 11.67% | 16.67% | 18.33% | 0.1465 | 0.1128 | 0.1448 | 65.00% | 2.32 |

Conclusion:

- Source expansion is a useful diagnostic: it increases rendered candidate diversity and can improve short-prefix K20.
- The gain is prefix-sensitive. At val128, full `27.34/35.94/41.41` is below the current best validation model E166/E181 (`32.03/40.62/44.53/45.31`).
- rows>=7 remains weak and does not pass the complex gate.
- This branch is not adopted as a frozen config and does not enter full test.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Keep `--source-expand-k` as infrastructure.
- Do not continue by only increasing source expansion or tuning the fixed self-score.
- Train a per-rendered-candidate quality model with train/val labels and actual rendered-CIF features, or return to improving complex W/A generator recall.

## 2026-06-14 canonical W/A source-priority decoder diagnostic

New artifacts:

- `scripts/opentry_merge_renderer_predictions_balanced.py`
- `reports/e280_renderer_predictions_e69c_neural_canonical`
- `reports/e285_renderer_predictions_e69d_neural_canonical`
- `reports/e288_priority_e69c_e69d_prior_canonical_merge_val128`
- `reports/e289_wa_eval_e288_priority_e69c_e69d_prior_val128`
- `reports/e290_render_e288_chem_vpa_soft_e08_g10_val128_k100`
- `reports/e292_eval_e291_e288_chem_vpa_soft_global_val128_match`
- `reports/e294_eval_e293_e288_chem_vpa_soft_diverse_val128_match`

Leakage status:

- W/A sources are model-side validation predictions from train-trained DP/scorers.
- Merge uses canonical W/A keys only; no StructureMatcher labels or test data.
- Renderer uses train-only geometry/source records.
- StructureMatcher is validation evaluation only.

Result summary:

| run | subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 |
|---|---|---:|---:|---:|---:|---:|
| E85 old raw merge | full | 47.66% | 70.31% | 77.34% | 78.91% | 13.74 |
| E289 canonical priority | full | 47.66% | 70.31% | 77.34% | 79.69% | 15.13 |
| E85 old raw merge | rows>=7 | 51.67% | 73.33% | 76.67% | 76.67% | 8.37 |
| E289 canonical priority | rows>=7 | 51.67% | 73.33% | 76.67% | 76.67% | 10.38 |

Validation CIF result:

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E166 reference | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 20.00% |
| E292 E289 candidates | global | 32.03% | 39.84% | 44.53% | 45.31% | 0.0942 | 0.1314 | 20.00% |
| E294 E289 candidates | diverse | 32.03% | 38.28% | 42.97% | 45.31% | 0.0942 | 0.1237 | 20.00% |

Conclusion:

- Canonical source-priority merge is a valid symbolic decoder improvement: it increases full W/A@50 and unique W/A without losing W/A@1/5/20.
- The current renderer/self-score does not convert the extra W/A tail into better validation match.
- E166/E181 remain the best validation CIF configs.
- Full test remains closed; the two +5pp objective is not met.

Next:

- Use canonical merge instead of raw merge for future W/A-tail diagnostics.
- Do not scale E289/E292/E294 to full test.
- Prioritize rows>=7 W/A recall and W/A-to-geometry compatibility; extra full-set unique tail alone is insufficient.

## 2026-06-14 E69b-first canonical decoder follow-up

E295-E300 tested E69b as the first canonical priority source.

Symbolic result:

| run | full W/A@1 | full W/A@5 | full W/A@50 | rows>=7 W/A@1 | rows>=7 W/A@5 | rows>=7 W/A@50 | full unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E289 E69c-first | 47.66% | 70.31% | 79.69% | 51.67% | 73.33% | 76.67% | 15.13 |
| E297 E69b-first | 49.22% | 70.31% | 79.69% | 53.33% | 73.33% | 76.67% | 15.23 |

CIF result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E166 reference | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 20.00% |
| E300 E69b-first | 31.25% | 39.06% | 44.53% | 45.31% | 0.0955 | 0.1273 | 20.00% |

Conclusion:

- E297 is the best symbolic W/A ordering but not the best CIF config.
- The symbolic rank1 gain is lost after rendered-CIF self-score and geometry selection.
- Current best validation CIF remains E166/E181.
- Full test remains closed; the two +5pp objective is not met.

## 2026-06-14 geometry compatibility selector batch

E301-E312 tested the requested geometry compatibility / source-free-param selector hypothesis without test access.

Artifacts:

- `data/geometry_compat_mpts52/e307_train512_val128_e166_features`
- `reports/e308_geometry_compat_gbdt_train512_val128`
- `reports/e310_eval_e308_gbdt_compat_val128_match`
- `reports/e309_geometry_compat_mlp_train512_val128`
- `reports/e311_eval_e309_mlp_compat_val128_match`
- `reports/e312_geometry_compat_gbdt_wa_group_selector_val128`

Train candidates were generated from train split only, with self-source geometry excluded. The train label pool used `atom_count>=12` train512, top20 candidates, and train StructureMatcher labels. Validation labels used existing E166 val128 top50 candidates only.

Result:

| run | selector | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@20 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E166 reference | global self-score | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% | 20.00% | keep best |
| E310 | GBDT pointwise compatibility | 30.47% | 36.72% | 42.19% | 45.31% | 15.00% | 16.67% | reject |
| E311 | MLP pointwise compatibility | 26.56% | 36.72% | 44.53% | 45.31% | 15.00% | 20.00% | reject |
| E312 | GBDT W/A-group geometry selector | 23.02% | 33.33% | 35.71% | 38.10% | 13.56% | 13.56% | reject |

Interpretation:

- Candidate-level compatibility signal exists: GBDT achieved validation candidate AUC `0.894`.
- The signal is misaligned with top-k structure generation: sorting all candidates by pointwise probability destroys W/A diversity and lowers match@5/match@20.
- Preserving W/A group order while selecting one geometry per W/A increased unique W/A, but removed useful alternate geometry candidates and dropped match further.
- Current conclusion is not “compatibility model impossible”; it is “pointwise probability reranking is the wrong objective.”

Status remains unchanged: no full-test config is frozen, and the two +5pp target is still unmet. Next viable step is a group/listwise selector trained directly for hit@5/hit@20 while preserving W/A diversity, with a more representative train candidate pool before scaling to val512.

## 2026-06-14 balanced train listwise selector update

E313-E322 built a more representative train-only compatibility pool and tested two group/listwise selectors.

Train pool:

- selected 1,024 train records across four atom-count buckets, 256 each; rows>=7=170 and atoms>=12=512;
- E313 skeleton eval-only reached full `skeleton@50=93.26%`, rows>=7 `87.06%`;
- E314 assignment DP reached full `W/A@50=81.64%`, rows>=7 `58.24%`;
- E316/E317 rendered and GT-free self-scored 44,460 train top50 candidates with self-source excluded;
- E318 train top20 labels had positive rate `44.91%`, rows>=7 positive rate `10.30%`, and W/A-hit/match-fail rate `20.07%`.

Validation selector result:

| run | selector | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@20/50 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E166 reference | global self-score | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | keep |
| E321 | pairwise W/A-cap | 31.25% | 37.50% | 43.75% | 45.31% | 18.33% | 20.00% / 20.00% | reject |
| E322 | group-bundle linear | 29.69% | 38.28% | 42.97% | 43.75% | 16.67% | 18.33% / 18.33% | reject |

Interpretation:

- The balanced train pool is healthier than the previous atom>=12-only pool, but current GT-free selector features still do not generalize into better validation top-k.
- Pairwise ranking preserved the K50 ceiling but hurt K5; group-bundle retained two geometry variants per W/A group but still reduced match and rows>=7.
- Stop this shallow selector family. The next viable direction is richer source/free-param compatibility modeling or new geometry proposal features, not bundle-size or scalar weight tuning.

Current best validation CIF remains E166/E181: `match@1/5/20/50=32.03/40.62/44.53/45.31`; only match@1 clears the +5pp target. Full test remains closed.

## 2026-06-14 enriched source/rendered compatibility selector update

E323-E327 added GT-free source-pair and rendered-cell context features to the balanced train1024 compatibility pool, then trained two dynamic selectors against the same E166 val128 candidate set. The feature schema blocks labels, target row_count/keys, candidate hit flags, raw CIF, sample/material/source IDs, and uses only train-source metadata plus target formula/SG context.

Artifacts:

- `data/geometry_compat_mpts52/e323_enriched_train_balanced1024_val128`
- `reports/e324_dynamic_gbdt_enriched_train_balanced1024_val128`
- `reports/e325_dynamic_pairwise_enriched_train_balanced1024_val128`
- `reports/e326_eval_e324_dynamic_gbdt_enriched_val128_match`
- `reports/e327_eval_e325_dynamic_pairwise_enriched_val128_match`

| run | selector | candidate AUC/AP | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5/20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 reference | global self-score | - | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 0.1534 | 18.33% / 20.00% |
| E324/E326 | enriched GBDT | 0.942 / 0.753 | 33.59% | 35.94% | 41.41% | 45.31% | 0.1221 | 0.1086 | 0.1332 | 18.33% / 20.00% |
| E325/E327 | enriched pairwise | 0.566 / 0.102 | 28.91% | 35.16% | 42.19% | 45.31% | 0.1045 | 0.1096 | 0.1394 | 16.67% / 20.00% |

Interpretation:

- Enriched GBDT learns a strong candidate-level label signal and improves rank1, but it damages K5/K20, so it does not satisfy the top-k objective.
- Pairwise ordering also lowers K1/K5/K20 and gives no rows>=7 gain.
- The same-candidate reranking family is still misaligned with hit@5/hit@20. Current best remains E166/E181; no full-test config is frozen.

Next direction: stop re-ranking this fixed rendered candidate set with current GT-free feature family. The viable path is to change the source/free-param proposal itself, or train a rendered-candidate quality model with richer context and an explicit top-k/diversity objective.

## 2026-06-14 sourceexpand proposal-quality smoke

E328-E333 tested a changed geometry proposal family: E250 gated source-residual geometry with source expansion, using train W/A predictions for train labels and existing E276 val128 sourceexpand predictions for validation. The renderer was updated with `--exclude-self-source` so train targets cannot use their own CIF as the source geometry.

Operational note:

- Train256 sourceexpand CPU rendering had no output after about 4 minutes and was interrupted.
- Train32 excl-self sourceexpand3 completed and was used only as a small smoke, not a formal model-selection training set.

Feature/proposal health:

| item | value |
|---|---:|
| train32 rendered candidates | 363 |
| train32 feature rows / positive rate | 363 / 62.26% |
| train32 rows>=7 samples | 2 |
| val128 feature rows / positive rate | 5,502 / 8.72% |
| raw val128 sourceexpand match@1/5/20/50 | 23.02 / 28.57 / 38.89 / 44.44% |
| raw val128 rows>=7 match@5/20/50 | 10.17 / 15.25 / 18.64% |

Selector results:

| run | selector | candidate AUC/AP | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5/20/50 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E166 reference | global self-score | - | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 0.1534 | 18.33 / 20.00 / 20.00% | keep |
| E330/E332 | sourceexpand GBDT | 0.827 / 0.397 | 14.06% | 27.34% | 41.41% | 43.75% | 0.1790 | 0.1404 | 0.1941 | 11.67 / 18.33 / 18.33% | reject |
| E331/E333 | sourceexpand MLP | 0.876 / 0.461 | 20.31% | 29.69% | 41.41% | 43.75% | 0.1749 | 0.1580 | 0.1999 | 8.33 / 15.00 / 18.33% | reject |

Interpretation:

- This changed proposal family has a near-E166 K50 ceiling but weak early ranks. The raw sourceexpand K5 is only `28.57%`.
- Tiny train32 quality models learn some candidate signal but destroy early W/A/skeleton ordering and remain far below E166 K5/K20.
- Scaling this exact train32 quality model is not justified. A viable next source/free-param attempt needs a larger excl-self train pool with progress/caching, or a generator that raises the sourceexpand pool quality rather than selecting inside a weak early-rank pool.

Status remains unchanged: best validation CIF remains E166/E181 `32.03/40.62/44.53/45.31`; only match@1 clears the +5pp target; no full test was run.

## 2026-06-14 merged proposal-pool selector update

E334-E338 tested a merged candidate-pool diagnostic: current E166 candidates plus source-residual source-expand candidates.

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@20/50 | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| E166 reference | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | keep |
| E337 merged-pool GBDT | 30.47% | 37.50% | 42.19% | 47.66% | 16.67% | 18.33% / 20.00% | reject |
| E338 merged-pool pairwise | 24.22% | 35.16% | 42.97% | 47.66% | 11.67% | 18.33% / 20.00% | reject |

Interpretation:

- The residual pool adds only 3 new validation-positive samples beyond E166 in val128.
- Pool-aware selectors can exploit the added tail enough to raise K50 to `47.66%`, but they damage K5/K20 and do not improve rows>=7.
- This closes the current selector-over-existing-candidates branch. The remaining useful work must improve the generated geometry pool itself, not merely select among weak tails.
- Full test remains closed; current best is still E166/E181 `32.03/40.62/44.53/45.31`, with only match@1 above the +5pp target.

## 2026-06-14 rows>=7 sourceexpand selector smoke

E340-E343 revisited sourceexpand selector training with a harder train-only rows>=7 pool rather than the tiny train32 pool. The renderer was updated with `--start-index`, `--min-row-count`, progress reporting, and partial JSONL resume support. Artifact directory names include `e334/e336/e338b`, but the logical experiment IDs are E340-E343 to avoid collision with the merged-pool batch above.

Train pool health:

| item | value |
|---|---:|
| rows>=7 train samples | 64 |
| rendered candidates | 852 |
| render rank1 SG-ok | 87.50% |
| render overall composition/readable | 94.01% |
| feature positive rate | 9.15% |
| train match@1/5/20 | 21.88 / 26.56 / 32.81% |
| W/A-hit but match-fail rate | 80.00% |

Validation result on sourceexpand val128:

| run | selector | candidate AUC/AP | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5/20/50 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E166 reference | global self-score | - | 32.03% | 40.62% | 44.53% | 45.31% | 18.33 / 20.00 / 20.00% | keep |
| raw sourceexpand | current order | - | 23.02% | 28.57% | 38.89% | 44.44% | 10.17 / 15.25 / 18.64% | weak baseline |
| E342 rows7 GBDT | GBDT | 0.828 / 0.295 | 21.88% | 28.12% | 42.19% | 43.75% | 13.33 / 16.67 / 18.33% | reject |
| E343 rows7 MLP | MLP | 0.755 / 0.238 | 22.66% | 32.81% | 40.62% | 43.75% | 13.33 / 16.67 / 18.33% | reject |

Interpretation:

- Rows>=7 labels are more relevant than train32 and improve W/A/diversity ordering, but they still do not convert to validation StructureMatcher match.
- The GBDT raises W/A@5 to `63.28%`, yet match@5 stays `28.12%`; this is another W/A-to-geometry conversion failure.
- The MLP gives the best sourceexpand K5 in this batch (`32.81%`) but remains far below E166/E181 K5 `40.62%` and does not improve rows>=7 enough.
- Early e338/e339 evaluator runs without `--max-records 128` used the full-val denominator and are invalid; e338b/e339b are the valid val128 results.

Status remains unchanged: full test is still closed, current best validation CIF remains E166/E181 `32.03/40.62/44.53/45.31`, and only match@1 clears the +5pp target. Stop scaling selectors over this sourceexpand pool until the raw proposal early-rank quality improves; before larger sparse train pools, add sample-id filtering to the feature builder to avoid scanning the full split.

## 2026-06-14 free-pattern compatibility smoke

E344-E348 tested one changed source/free-param proposal and two sparse train-label compatibility models.

- Free-pattern proposal (`row_aligned_chem_freepattern_quality`) was healthy but not better: val32 full `match@1/5/20/50=34.38/43.75/43.75/43.75`, rows>=7 `22.22/22.22/22.22/22.22`, matching the E166 first-32 baseline.
- Feature-builder filtering is now available: `--sample-ids-file`, `--restrict-to-rendered-samples`, and `--min-row-count`. E346 processed 31 rendered samples / 596 top20 rows instead of scanning unused split records.
- E346 confirms the bottleneck remains W/A-to-geometry: full positive rate `18.79%`, rows>=7 positive rate `3.87%`, W/A-hit but match-fail `70.00%`.
- Sparse SGD compatibility selectors learned candidate signal but hurt top-k: E347 `25.81/38.71/45.16`, E348 rows-weighted `29.03/38.71/45.16` on the 31-sample free-pattern feature smoke, below the current order `35.48/45.16/45.16`.
- Decision: reject and do not scale to val128. Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; full test remains closed.

Next useful direction: stop reranking weak proposal pools. Build a source/free-param proposal model that improves raw val128 K5/K20 first, then apply compatibility selection only after the candidate pool itself is stronger.

## 2026-06-14 free-pattern val128 follow-up

I ran one val128 follow-up for `row_aligned_chem_freepattern_quality` to check whether the tied val32 result was prefix-specific. It was a small proposal-level improvement, but not enough for the gate.

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1/5/20 | rows>=7 match@5/20/50 | decision |
|---|---:|---:|---:|---:|---|---:|---|
| E166/E181 reference | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00 / 20.00% | keep |
| E350 free-pattern proposal | 32.03% | 41.41% | 44.53% | 45.31% | 0.0942 / 0.1357 / 0.1496 | 18.33 / 18.33 / 20.00% | reject for gate |
| E352 free-pattern + GBDT | 31.25% | 35.94% | 42.97% | 45.31% | 0.0883 / 0.1026 / 0.1447 | 18.33 / 18.33 / 20.00% | reject |
| E353 free-pattern + MLP | 28.12% | 35.94% | 42.97% | 45.31% | 0.0868 / 0.1078 / 0.1495 | 16.67 / 18.33 / 20.00% | reject |

Diagnostics:

- E350 raises K5 by `+0.78 pp` over E166, but `41.41%` remains below the `41.58%` +5pp target and rows>=7 K20 regresses.
- E351 candidate labels show the same bottleneck: W/A-hit but match-fail rate is `71.09%`; rows>=7 positive rate is only `1.32%`.
- E352/E353 have high candidate AUC/AP (`0.940/0.747`, `0.903/0.621`) but damage official top-k match, so candidate-level compatibility scores still do not optimize hit@5/hit@20.

No full-test gate: still only match@1 clears the +5pp target. Next work should change the proposal generator itself, not apply another selector to this free-pattern pool.

## 2026-06-14 learned source-mode proposal update

E354-E363 tested existing learned source-mode selectors as render-time source/free-param proposal generators with E250 gated residual geometry. This changed the proposal before CIF rendering, rather than reranking an already rendered pool.

| run | source-mode | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E166/E181 reference | current best | 32.03% | 40.62% | 44.53% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00% | keep |
| E357 | E226 source-transfer selector | 30.47% | 36.72% | 39.84% | 0.1471 / 0.1589 / 0.1740 | 16.67 / 18.33% | reject |
| E361 | E258 rendered-health selector | 28.91% | 35.94% | 41.41% | 0.1277 / 0.1527 / 0.1858 | 16.67 / 18.33% | reject |

Feature diagnostics on the self-scored pools show why this branch fails: positive rate is only about `9.7%`, rows>=7 positive rate is `2.8-2.9%`, and W/A-hit but match-fail remains `66-67%`. Existing source-mode selectors do not solve geometry/free-param conversion.

Status is unchanged: full test remains closed, current best remains E166/E181, and only match@1 clears the +5pp target. Next work should train or construct a stronger source/free-param proposal generator with raw val128 K5/K20 and rows>=7 improvement before any more selector training.

## 2026-06-14 source-success prior update

E364-E374 tested a train-only source-success prior as a render-time source/free-param proposal change, including a free-pattern variant. The best result was E372 full val128 `match@1/5/20=31.25/39.84/43.75%`, RMSE `0.0748/0.1329/0.1452`, rows>=7 `13.33/18.33/18.33%`. This is below E166/E181 `32.03/40.62/44.53%`.

Feature diagnostics on E373/E374 show global positive rate about `14.8%`, but rows>=7 positive rate only `2.8%`, with W/A-hit but match-fail `68-69%`. The source prior helps candidate density but not complex geometry conversion.

No full-test config is frozen. Current best remains E166/E181; only match@1 clears the +5pp target. The next direction should be a new source/free-param proposal model for rows>=7 source/W-A compatibility, not more scalar source-prior/free-pattern tuning.

## 2026-06-14 rows>=7 residual geometry update

E375-E385 trained a rows>=7-only source-conditioned residual geometry model, because earlier residual caches had very few complex train examples. The new cache had 512 rows>=7 train examples and 256 rows>=7 validation examples, with no missing source.

The trained E376 model did not transfer to CIF match. One-source rendering gave E380 full val128 `match@1/5/20=29.69/35.94/36.72%`, rows>=7 `13.33/16.67/16.67%`. Sourceexpand3 gave E384 `28.91/33.59/39.84%`, rows>=7 `10.00/15.00/16.67%`. Both are below E166/E181, and RMSE is much worse.

E385 diagnostics keep the same story: rows>=7 positive rate is only `2.60%`, and W/A-hit but match-fail is `64.13%`. Rows>=7-only MSE residual training is therefore rejected; the next proposal model needs a different objective or richer multimodal labels.

## 2026-06-14 anchored group selector update

E386-E392 tested a more conservative fixed-pool geometry compatibility selector: keep the first three candidates as anchors, then use a GT-free logreg group selector over `wa_source` groups, capped at two candidates per W/A. The trainer blocks labels, target keys/row_count, ids, CIF text, and hit flags from features.

| run | pool | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E166/E181 reference | current best | 32.03% | 40.62% | 44.53% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00% | keep |
| E391 | anchored E166 logreg | 32.03% | 39.06% | 42.97% | 0.0942 / 0.1172 / 0.1448 | 18.33 / 20.00% | reject |
| E392 | anchored free-pattern logreg | 32.03% | 40.62% | 44.53% | 0.0942 / 0.1297 / 0.1508 | 18.33 / 20.00% | reject/no gain |

E392 ties the E166/E181 full match numbers but does not exceed them, does not recover E350's tiny K5 bump, and still misses the K5 target `41.58%`. Rows>=7 remains unchanged. The anchored fixed-pool selector family is therefore rejected.

Current best is still E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears the +5pp target and no full test is frozen. The next useful direction is not another selector over E166/free-pattern candidates. It must improve the raw geometry/free-param proposal pool before self-score, especially for rows>=7 W/A-to-match conversion.

## 2026-06-14 match-aware source-mode proposal

E393-E399 tested a render-time proposal change rather than a fixed-pool rerank: source-mode examples were built from E323 train/val rendered labels, then E394 selected source/free-param contexts before rendering.

| run | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---|---:|---|
| E166/E181 reference | 32.03% | 40.62% | 44.53% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00% | keep |
| E396 raw E394 proposal | 21.09% | 31.25% | 39.84% | 0.1316 / 0.1730 / 0.1751 | 13.33 / 18.33% | reject |
| E398 self-scored E394 proposal | 28.12% | 36.72% | 39.06% | 0.1161 / 0.1603 / 0.1693 | 16.67 / 18.33% | reject |

E394 did not learn to outperform the heuristic source choice: final validation selected gain was `-0.0018`, with `0.00%` improved-over-heuristic rate. E399 labels show the same conversion bottleneck: positive rate `9.00%`, rows>=7 positive rate `2.08%`, and W/A-hit but match-fail `69.57%`.

Decision: reject this match-aware source-mode branch. Current runtime source features are insufficient; the next viable proposal model needs richer geometry/free-param generation or labels, not another selector over the same source candidate features. Full test remains closed.

## 2026-06-14 train-positive source-bank proposal

E400-E407 tested a train-only source/template bank built from E318 StructureMatcher-positive rendered candidates. The bank contains 8,884 positive source/template entries from train labels only, including 310 rows>=7 positives.

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---:|---|---:|---|
| E166/E181 reference | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00% | keep |
| E403 successbank replacement | 21.09% | 28.91% | 32.03% | 35.16% | 0.0942 / 0.1557 / 0.1707 | 3.33 / 5.00% | reject |
| E406 anchored successbank tail | 32.03% | 39.84% | 43.75% | 45.31% | 0.0942 / 0.1230 / 0.1484 | 18.33 / 20.00% | reject |

The direct bank replacement loses coverage and badly damages complex rows. The anchored tail preserves rank1 and raises W/A diversity, but still reduces match@5/20. E407 feature labels show full positive rate `14.54%`, rows>=7 positive rate `2.74%`, and W/A-hit but match-fail `64.84%`.

Conclusion: train-positive source IDs alone are not enough. The next proposal needs row-level free-parameter/lattice template learning or generation, not another source-ID selector. Full test remains closed; current best is still E166/E181.

## 2026-06-15 row-level template bank result

E408-E414 tested a stronger version of the train-positive bank idea: instead of only copying successful source IDs, the renderer tried to graft train-positive row-level free-parameter templates into predicted W/A candidates.

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---:|---|---:|---|
| E166/E181 reference | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 / 0.1295 / 0.1534 | 18.33 / 20.00% | keep |
| E410 rowbank quality | 14.06% | 21.88% | 26.56% | 29.69% | 0.1900 / 0.1719 / 0.1808 | 0.00 / 0.00% | reject |
| E413 rowbank freepattern | 14.06% | 21.88% | 27.34% | 29.69% | 0.1887 / 0.1724 / 0.1887 | 0.00 / 0.00% | reject |

Both rowbank variants render only 107/128 val samples and collapse rows>=7 match to zero. E414 diagnostics show positive rate `9.98%`, rows>=7 positive rate `0.00%`, and W/A-hit but match-fail `68.92%`.

Conclusion: direct cross-sample row-level template grafting is worse than source-ID transfer. It can preserve some W/A/skeleton hits, but the copied row coordinates are not source-consistent with the lattice and full orbit context. Current best remains E166/E181; no val512 or full-test gate is met.

## 2026-06-15 val512 stability and selector result

E415 showed that the old E84 prediction file only contained 128 validation rows, so it cannot be used as val512 evidence. E416-E420 regenerated val512 W/A predictions from train-trained skeleton/assignment models only, then E421-E423 evaluated the E166-style geometry configuration on 512 validation samples.

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---:|---|---:|---|
| E423 val512 baseline | 33.01% | 40.82% | 45.51% | 48.24% | 0.1095 / 0.1201 / 0.1309 | 16.89 / 18.67% | keep as larger-val reference |
| E427 GBDT selector | 33.59% | 40.62% | 45.51% | 45.51% | 0.1274 / 0.1273 / 0.1309 | 17.33 / 18.67% | reject |
| E428 MLP selector | 28.71% | 38.28% | 45.51% | 45.51% | 0.1377 / 0.1237 / 0.1309 | 14.67 / 18.67% | reject |

The val512 baseline confirms the same pattern as val128: match@1 clears the CrystaLLM+5pp target, but match@5 and match@20 do not. E424 labels show positive rate `17.83%`, rows>=7 positive rate only `3.04%`, and W/A-hit but match-fail `64.68%`.

Conclusion: fixed-pool pointwise compatibility selection is not enough. GBDT only moves some positives to rank1 while reducing K5; MLP damages K1/K5. The next route must increase the raw positive density of geometry proposals for rows>=7, not rescore the same pool.

## 2026-06-15 match-aware source-mixture residual result

E429-E437 tested the next proposal-side idea: use E393 match-aware source-mode groups to build a source-consistent residual geometry cache, then train a gated residual model that adjusts lattice/free parameters before CIF rendering.

| run | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---|---:|---|
| E423 val512 reference | 33.01% | 40.82% | 45.51% | 0.1095 / 0.1201 / 0.1309 | 16.89 / 18.67% | keep |
| E432 E430 rank0 raw val128 | 17.19% | 26.56% | 28.91% | 0.1939 / 0.2188 / 0.2147 | 6.67 / 6.67% | reject |
| E434 E430 sourceexpand3 raw val128 | 17.19% | 24.22% | 30.47% | 0.1939 / 0.2156 / 0.2310 | 5.00 / 6.67% | reject |
| E437 E430 sourceexpand3 + global self-score | 22.66% | 29.69% | 33.59% | 0.1893 / 0.2205 / 0.2304 | 5.00 / 6.67% | reject |

E435 diagnostics show this is not just ordering: full positive rate is `9.05%`, rows>=7 positive rate is `1.55%`, and W/A-hit but match-fail is `71.53%`. This is worse than the E424 val512 baseline feature density (`17.83%` full, `3.04%` rows>=7).

Conclusion: residual MSE transfer from source examples is the wrong objective for the current bottleneck. It preserves some W/A hits but damages geometry conversion. Current best remains E423 val512; no full-test config is frozen. Next work should target match-positive source/free-param/lattice proposal generation with group/listwise objectives, not another residual-loss continuation or fixed-pool scorer.

## 2026-06-15 enriched val512 listwise selector result

E438-E442 tested whether a richer GT-free feature schema plus listwise/group objectives can rescue the E423/E424 fixed pool. E438 added source formula/context, source/candidate W/A relation, and rendered-cell-shape features to all 9,717 E424 val512 candidate rows; source metadata came only from train representations.

| run | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---:|---:|---:|---|---:|---|
| E423 val512 reference | 33.01% | 40.82% | 45.51% | 0.1095 / 0.1201 / 0.1309 | 16.89 / 18.67% | keep |
| E441 pairwise_mmr | 27.73% | 36.91% | 45.51% | 0.1491 / 0.1274 / 0.1309 | 14.67 / 18.67% | reject |
| E442 group_bundle | 31.25% | 40.23% | 43.75% | 0.1339 / 0.1430 / 0.1306 | 16.89 / 17.78% | reject |

The pairwise selector only preserves the K20 ceiling while damaging K1/K5. The group-bundle selector is closer, but still below E423 on K1/K5 and regresses K20, atoms>=12, and rows>=7 K20.

Conclusion: fixed-pool listwise selection is also insufficient. The remaining gap is not mainly a selector objective issue; the pool needs more match-positive geometry proposals, particularly for rows>=7. Full test remains closed.

## 2026-06-15 pre-render learned source proposal smoke

E443-E451 moved one step earlier than fixed-pool ranking: the source/free-param context is selected before rendering. The E443 model was trained from train source-pair candidate labels and blocked post-render fields, raw IDs, target row_count/keys, W/A-hit fields, and all labels from inference features.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E450 direct sourceproposal | val64 | 32.81% | 40.62% | 40.62% | 0.1465 / 0.1553 / 0.1472 | 23.33 / 23.33% | reject |
| E451 anchored-tail sourceproposal | val64 | 32.81% | 42.19% | 43.75% | 0.1194 / 0.1438 / 0.1510 | 23.33 / 23.33% | reject/no scale |

The direct variant damages full K5/K20. The anchored-tail variant preserves the baseline on this small subset and gives a small rows>=7 bump, but it does not improve the overall match metrics and is only val64 evidence. The larger cache attempt was also too slow because source alignment feature construction is expensive.

Conclusion: this source-only pre-render proposal is not enough. If this direction continues, it needs a faster rows>=7-focused generator and must propose source-consistent free parameters/lattice, not just reorder source IDs. Current best remains E423 val512; full test remains closed.

## 2026-06-15 train-manifold free-param/lattice variants

E452-E457 tested whether keeping the same source context but sweeping free parameters toward train-set orbit/symbol parameter quantiles could increase match-positive geometry density. A second variant also swept lattice scale toward train VPA quantiles.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E456 manifold params | val64 | 31.25% | 37.50% | 40.62% | 0.1127 / 0.1470 / 0.1356 | 16.67 / 16.67% | reject |
| E457 manifold params + lattice | val64 | 31.25% | 37.50% | 40.62% | 0.1127 / 0.1432 / 0.1356 | 16.67 / 16.67% | reject |

Both variants preserve readable/composition-exact candidates, but reduce match@1/5/20 and rows>=7 match. W/A-hit but match-fail at K20 worsens from `51.02%` in the baseline subset to `56.00%`.

Conclusion: unconditional train-manifold parameter/lattice sweeps are not a useful generator. They perturb source-consistent coordinates away from match-positive regions. Current best remains E423 val512; full test remains closed.

## 2026-06-15 microvariant selector smoke

E458-E477 converted the rejected manifold variants into a supervised microvariant pool: train64 plus a train rows>=7 subset, each with baseline/params/lattice candidates. This produced 127 train samples and 11,515 labeled rows. The validation pool used the E449 val64 baseline plus E456/E457 variants.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E480 microvariant GBDT | val64 | 20.31% | 31.25% | 37.50% | 0.1875 / 0.1469 / 0.1488 | 20.00 / 20.00% | reject |
| E481 microvariant pairwise MMR | val64 | 31.25% | 35.94% | 42.19% | 0.1256 / 0.1277 / 0.1406 | 20.00 / 20.00% | reject |

The key diagnostic is that validation microvariants added `0` new positive samples beyond the baseline pool. The merged pool's K50 ceiling stayed `44.44%`, and rows>=7 K20 stayed `20.00%`. With no new positive geometry to select, both selector models mostly damaged the early ranks.

Conclusion: do not scale microvariant selectors on unconditional variant pools. The next useful step must make the generator/proposal add new match-positive source-consistent free-param/lattice candidates before applying another selector. Current best remains E423 val512; full test remains closed.

## 2026-06-15 train-positive free-param bank smoke

E482-E491 tested a stricter variant of the free-param idea: build a bank only from train StructureMatcher-positive rendered candidates, then move source-copied free parameters and lattice VPA toward those train-positive quantiles.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E490 positive params | val64 | 32.81% | 37.50% | 40.63% | 0.1148 / 0.1331 / 0.1376 | 16.67 / 16.67% | reject |
| E491 positive params + lattice | val64 | 32.81% | 37.50% | 40.63% | 0.1148 / 0.1331 / 0.1376 | 16.67 / 16.67% | reject |

The decisive diagnostic is E489: after merging baseline plus both positive-param variants, there were `0` new positive samples beyond the baseline pool. So the train-positive bank changes coordinates, but does not create new StructureMatcher-positive validation geometries.

Conclusion: aggregate positive-quantile free-param/lattice variants are another dead end. Current best remains E423 val512 `33.01/40.82/45.51/48.24`, with only match@1 above the CrystaLLM+5pp target. Full test remains closed.

## 2026-06-15 collision-relief lattice smoke

E492-E494 tested a train-diagnostic-motivated GT-free proposal: complex W/A-hit positives have larger `self_min_distance` than hard negatives, so collided candidates were expanded by scaling CIF cell lengths while keeping fractional coordinates fixed.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E493b collision-relief variants only | val64 | 20.69% | 20.69% | 22.41% | 0.2160 / 0.2151 / 0.2240 | 13.33 / 13.33% | reject |

E494 merged the baseline with the collision-relief variants and again found `0` new positive samples beyond baseline. The merged K50 ceiling stayed `44.44%`; rows>=7 K20 stayed `20.00%`.

Conclusion: lattice-only collision relief is not enough. It increases W/A-hit match-fail to `82.16%`, so the next generator must couple fractional-coordinate changes with lattice changes instead of scaling the cell independently.

## 2026-06-15 coordinate+lattice relaxation smoke

E495-E497b tested that next obvious step: a GT-free post-render relaxation that updates fractional coordinates with PBC pair-repulsion while lightly scaling the lattice. The full variant budget was too slow, so the actual smoke used one variant per candidate on the same E449/E473 val64 IDs.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E496b coord+lattice relax | val64 | 20.00% | 24.00% | 26.00% | 0.1673 / 0.2124 / 0.2324 | 13.79 / 13.79% | reject |

E497b merged the baseline with the coord-relax variants and again found `0` new positive samples beyond baseline. The merged K50 ceiling stayed `44.44%`, and rows>=7 K20 stayed `20.00%`.

Conclusion: post-render coordinate surgery is not the needed source-consistent generator. The next viable route should change source/free-param selection before rendering, with train-label evidence, rather than modifying finished CIF coordinates.

## 2026-06-15 row-aligned sourceproposal smoke

E498-E501 returned to render-time source/free-param context selection. The cache builder was patched with per-record timeout so long-tail source scoring no longer requires manual interruption. A row-aligned source pool was scored by the existing train-label E443 source proposal model, then rendered in anchored-tail mode.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449 baseline subset | val64 | 32.81% | 42.19% | 43.75% | 0.1199 / 0.1449 / 0.1455 | 20.00 / 20.00% | reference |
| E500 row-aligned sourceproposal tail | val64 | 25.40% | 30.16% | 38.10% | 0.0549 / 0.0840 / 0.1403 | 10.00 / 10.00% | reject |

E501 merged the baseline with the sourceproposal-tail pool and again found `0` new positive samples beyond baseline. The merged K50 ceiling stayed `44.44%`, and rows>=7 K20 stayed `20.00%`.

Conclusion: source-ID/source-context proposal variants are not enough by themselves. The gap requires a new free-param generator coupled to source context, not more source allocation or post-render coordinate edits.

## 2026-06-15 source row-shift free-param diagnostic

E502-E505 tested a narrower source-conditioned generator: keep the selected source, but shift each target row's free parameters to near-tie compatible rows inside that same source. This changes free-param mapping before CIF rendering, without using test data or StructureMatcher labels at inference.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449/E473 baseline subset | val64 | 28.57% | 39.68% | 42.86% | 0.0741 / 0.1501 / 0.1408 | 16.67 / 20.00% | reference |
| E503 row-shift raw | val64 | 20.63% | 22.22% | 33.33% | 0.0491 / 0.0333 / 0.1246 | 0.00 / 0.00% | reject |
| E505 row-shift merged + GBDT | val64 | 31.75% | 38.10% | 39.68% | 0.0957 / 0.1331 / 0.1414 | 20.00 / 20.00% | reject |

E504 is the useful diagnostic: merging row-shift with baseline adds `1` new positive sample beyond baseline, the first recent generator to do so. But the new positive is not a rows>=7 gain, and the selector lowers K5/K20 while only improving K50 ceiling to `46.03%`.

Conclusion: source-conditioned free-param changes are a plausible direction, but exact row-shift is not enough. Current best remains E423 val512 `33.01/40.82/45.51/48.24`, with only match@1 above target. Full test remains closed.

## 2026-06-15 rows>=7 positive-nearest free-param smoke

E506-E509 narrowed the train-positive bank to complex rows only. The renderer then used `positive_nearest_complex`: for rows>=7 candidates, each source-copied free parameter was moved toward nearby train-positive values for the same SG/orbit/free-symbol bucket, rather than toward global quantiles.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E449/E473 baseline subset | val64 | 28.57% | 39.68% | 42.86% | 0.0741 / 0.1501 / 0.1408 | 16.67 / 20.00% | reference |
| E508 rows7 positive-nearest raw | val64 | 25.40% | 28.57% | 38.10% | 0.0549 / 0.0618 / 0.1399 | 10.00 / 10.00% | reject |

E509 merged the baseline and rows7-positive-nearest pools. It found `0` new positive samples beyond baseline; full K50 stayed `44.44%` and rows>=7 K20 stayed `20.00%`.

Conclusion: complex-only positive parameter lookup improves raw positive rate versus row-shift but does not create new positive samples. Stop positive-nearest/quantile bank variants; the next useful route needs a row-level geometry feasibility model, not more parameter-bank lookup around copied source parameters.

## 2026-06-15 row-feasibility histogram-bank result

E510-E517 tested a harder train signal than positive-only lookup: only rows>=7 W/A-hit train candidates were used, with StructureMatcher positives contrasted against hard negatives. The learned object was a row/free-symbol histogram of positive-vs-negative log-odds.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E512 fine histogram raw | val64 | 25.40% | 28.57% | 38.10% | 0.0549 / 0.0618 / 0.1399 | 10.00 / 10.00% | reject |
| E516 coarse histogram raw | val64 | 25.40% | 28.57% | 38.10% | 0.0549 / 0.0618 / 0.1399 | 10.00 / 10.00% | reject |

Both E513 and E517 merge diagnostics found `0` new positive samples beyond baseline. The full K50 ceiling stayed `44.44%`; rows>=7 K20 stayed `20.00%`.

Conclusion: scalar row/free-param feasibility bins are insufficient. The remaining geometry gap likely depends on coupled row source context and lattice/source consistency, not independent parameter substitutions.

## 2026-06-15 row-pair feasibility bank result

E518-E525 moved from scalar free-param bins to joint target-row/source-row matching. A train-only row-pair bank was built from rows>=7 positive vs hard-negative rendered candidates, then `row_pair_feasible` changed row matching inside the selected source before copying free params.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E520 strict row-pair raw | val64 | 25.40% | 28.57% | 38.10% | 0.0549 / 0.0618 / 0.1399 | 10.00 / 10.00% | reject |
| E524 broad row-pair raw | val64 | 25.40% | 28.57% | 38.10% | 0.0549 / 0.0618 / 0.1399 | 10.00 / 10.00% | reject |

Both merge diagnostics added `0` positive samples beyond baseline. The broad bank used more train negatives, but produced the same validation candidates as the strict bank.

Conclusion: same-source row rematching is not enough. The next generator must jointly alter source selection, row mapping, and lattice context before rendering.

## 2026-06-15 row-pair-aware source selection result

E526-E533 tested that next joint step: the renderer now has `row_pair_source_feasible_quality`, which uses train-only row-pair feasibility while choosing the source record itself, then applies `row_pair_feasible` row mapping.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E527 strict source-feasible | val64 | 25.00% | 37.50% | 42.19% | 0.0549 / 0.1410 / 0.1408 | 16.67 / 20.00% | 0 | reject |
| E531 broad source-feasible | val64 | 25.00% | 37.50% | 42.19% | 0.0549 / 0.1408 / 0.1409 | 16.67 / 20.00% | 0 | reject |

Both strict and broad merges left the full K50 ceiling at `44.44%` and rows>=7 positive-any at `20.00%`. The branch improves over same-source row-pair raw K5, but still duplicates baseline positives.

Current best remains E423 val512 `match@1/5/20/50=33.01/40.82/45.51/48.24` with RMSE@1/5/20 `0.1095/0.1201/0.1309`. Only match@1 clears the MPTS-52 +5pp target; full test remains closed.

Conclusion: stop row-pair-aware source-selection variants. The next route must create genuinely new geometry-positive candidates, likely by changing lattice/source context or learning source-free-param proposals, before any selector/ranker is worth training.

## 2026-06-15 train-positive geometry bundle transfer

E534-E536 added renderer metadata export and built a train-only bank of StructureMatcher-positive geometry bundles. The bank is healthy as data: train256 had 2,729 positive rows, 795 unique W/A keys, and 37 rows>=7 positive rows.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E538 exact bundle | val64 | 25.00% | 40.63% | 43.75% | 0.0699 / 0.1654 / 0.1531 | 16.67 / 20.00% | 0 | reject |
| E542 broad bundle | val64 | 29.69% | 35.94% | 39.06% | 0.1583 / 0.1565 / 0.1431 | 16.67 / 16.67% | 1 | reject |

E540 exact merge added `0` positives beyond baseline. E544 broad merge added `1` full positive but no rows>=7 positive; rows>=7 positive-any stayed `20.00%`.

Conclusion: train-positive geometry bundle transfer should not be scaled or used for selector training. It copies useful-looking train geometry but does not create new complex-row positives. Current best remains E423 val512 `33.01/40.82/45.51/48.24`, with only match@1 above target; full test remains closed.

## 2026-06-15 rows>=7-only geometry bundle transfer

E545 rebuilt the geometry bundle bank using only train rows>=7 positives. This made the bank rows-specific but very sparse: 37 entries across 5 SGs and 12 W/A keys.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E547 rows7 exact bundle | val64 | 25.00% | 39.06% | 42.19% | 0.0549 / 0.1526 / 0.1409 | 16.67 / 20.00% | 0 | reject |
| E551 rows7 broad bundle | val64 | 25.00% | 37.50% | 40.63% | 0.0549 / 0.1395 / 0.1285 | 16.67 / 20.00% | 0 | reject |

Both E549 and E553 merge diagnostics found `0` new positive samples beyond baseline; rows>=7 positive-any stayed `20.00%`.

Conclusion: rows>=7-only bundle transfer also fails. The issue is not just selecting train-positive geometry from the right subset; the model needs to generate or choose coupled source/free-param/lattice context that is new enough to create complex-row positives. Full test remains closed.

## 2026-06-15 rows>=7 positive-geometry residual model

E554-E556 built a train/val cached dataset from StructureMatcher-positive rendered candidates with geometry metadata. This changed the residual target from raw GT CIF geometry to "geometry that already produced a match" on train/val. The rows>=7 dataset was very small: 37 train examples and 15 val examples.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E559 positive residual expand1 | val first64 | 26.98% | 36.51% | 39.68% | 0.1330 / 0.1732 / 0.1759 | 13.33 / 13.33% | 0 | reject |
| E564 positive residual expand3 | val first64 | 26.56% | 34.38% | 39.06% | 0.1330 / 0.1455 / 0.1659 | 10.00 / 16.67% | 0 | reject |

E558 with `--min-row-count 7` is invalid for first64 comparison because that flag filters the full val split before applying `max_records`; it was not used for merge or selection. E560 and E563 merge diagnostics both found `0` new positive samples beyond baseline, and rows>=7 positive-any stayed `20.00%`.

Conclusion: the tiny positive-residual generator raises candidate positive rate in places but does not raise sample-level ceiling and hurts rows>=7 top-k. It should not be scaled as configured.

## 2026-06-15 larger rows>=7 positive-residual retrain

E565-E566b tested whether the previous residual failure was just a data-size problem. A train1024 top20 metadata render was rebuilt and rows>=7 candidates were relabeled from train GT only. Reusing old E318 labels was rejected because candidate content no longer matched despite matching `sample_id/rank`.

The resulting rows>=7 train label table had 166 samples, 3,009 candidates, and 209 positive rows, expanding train positive residual examples from 37 to 209. E567 cached 209 train and 15 val examples; E568 retrained the same gated residual model and improved best val loss from 0.20137 to 0.18437.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E572 larger positive residual expand3 | val first64 | 26.56% | 34.38% | 39.06% | 0.11695 / 0.13390 / 0.15321 | 10.00 / 16.67% | 0 | reject |

E571 merge diagnostics found `0` new positive samples beyond E473 baseline. Rows>=7 positive-any stayed `20.00%`. W/A@20 was still `76.56%` and skeleton@20 `81.25%`, while W/A-hit match-fail remained high at `63.27%`.

Conclusion: simply increasing rows>=7 positive residual examples improves training loss and RMSE, but not validation coverage or match@5/20. This residual objective should be stopped in its current form. Full test remains closed; current best is still E423 val512, with only match@1 above the +5pp target.

## 2026-06-15 pair-delta geometry transfer

E573-E576 tested a different target from residual regression and bundle copying: train same-signature negative-to-positive geometry deltas were stored, then applied to validation candidates with matching SG/row/free-param signatures.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E576 pair-delta signature | val first64 sparse | 0.00% | 0.00% | 0.00% | null / null / null | 0.00 / 0.00% | 0 | reject |

The delta bank had 909 raw train pairs and 263 kept pairs, but validation coverage was only 27 input rows and 5 samples. E574 found `0` positives among 200 rendered variants; E575 merge left full positive-any at `44.44%` and rows>=7 positive-any at `20.00%`.

Conclusion: pair-delta transfer is too sparse and does not solve the W/A-hit but match-fail geometry bottleneck. Full test remains closed; current best is still E423 val512, with only match@1 above the +5pp target.

## 2026-06-15 same-sample geometry recombination

E577-E580 tested a label-free intra-sample donor/recipient recombination: use one candidate's lattice/free-params to render another candidate's W/A in the same formula+GT-SG sample.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E580 same-sample recombine | val first64 | 25.00% | 26.56% | 35.94% | 0.0605 / 0.0719 / 0.1410 | 6.67 / 10.00% | 1 full, 0 rows>=7 | reject |

This route had broad coverage: 3,108 variants across 63 samples. E578 labels found a high candidate positive rate (`21.07%`) and E579 added one full positive sample beyond baseline. But rows>=7 positive-any stayed `20.00%`, and W/A diversity collapsed (`unique_WA_mean@20=1.58` in official eval).

Conclusion: intra-sample recombination can create easy positives, but not the complex-row positives needed for match@5/@20. It should not feed a selector unless a later variant raises rows>=7 sample-level positive-any.

## 2026-06-15 direct GeometryNet smoke

E581-E592 tested direct train-only geometry prediction from W/A rows to lattice/free parameters, instead of selecting or copying source geometry. Two models were tried: a full-data 4096-train smoke and a rows>=7-only 4096-train contrast.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E586 full-data GeometryNet | val first64 | 15.63% | 18.75% | 20.31% | 0.2309 / 0.2425 / 0.2217 | 0.00 / 0.00% | 0 | reject |
| E592 rows7-only GeometryNet | val first64 | 7.81% | 10.94% | 18.75% | 0.1279 / 0.1850 / 0.1571 | 0.00 / 0.00% | 0 | reject |

The rows>=7-only model achieved much lower validation regression loss (`0.8209` vs `3.7938` for full-data), but both produced zero rows>=7 positive candidates after rendering. E585 and E591 merges added no positives beyond baseline.

Conclusion: plain supervised GT-geometry regression is not enough. Future geometry models need a match-aware or physically constrained pre-render objective, not just lower lattice/free-param MSE.

## 2026-06-15 jitter and bond-length physics smokes

E593-E595 tested wrapped jitter/lattice variants on the E554 val64 baseline setup. It improved SG health but did not add any positive sample beyond baseline.

E596-E599 added a new train-stat physical objective: `opentry_bond_length_refine_variants.py` estimates element-pair distance windows from 1,024 train CIFs, then refines rendered candidates without GT labels, W/A labels, StructureMatcher labels, row_count labels, or test data at inference.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E594 wrapped jitter lattice | val first64 raw | 25.40% | 39.68% | 41.27% | 0.0549 / 0.1542 / 0.1298 | 16.67 / 16.67% | 0 | reject |
| E599 bond-length refine | val first64 direct | 23.44% | 31.25% | 37.50% | 0.0870 / 0.1305 / 0.1563 | 10.00 / 16.67% | 1 full, 0 rows>=7 | reject |

Bond-length refinement raised candidate positive rate and found one easy full positive beyond baseline, but rows>=7 positive-any stayed `20.00%` and direct K5/K20 were worse than baseline. These pools should not feed selector/ranker training. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp and full test remains closed.

## 2026-06-15 complex-weighted assignment and W/A merge

E600-E620 returned to the W/A generator side, but only after the recent geometry/refinement routes failed to add complex positives. The test split remained closed.

| run | scope | match@1 | match@5 | match@20 | RMSE@1/5/20 | rows>=7 K5/20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E605 narrow complex8 DP | val512 | 30.27% | 35.16% | 40.63% | 0.0887 / 0.0952 / 0.1223 | 15.11 / 16.44% | reject |
| E610 wide complex8 DP | val512 | 33.01% | 39.84% | 45.31% | 0.1105 / 0.1141 / 0.1332 | 15.11 / 18.22% | reject |
| E615 E420+E607 round-robin | val512 | 33.40% | 40.63% | 45.90% | 0.1081 / 0.1144 / 0.1331 | 16.44 / 18.22% | reject |
| E620 E420+E607 priority | val512 | 33.01% | 40.63% | 46.09% | 0.1071 / 0.1144 / 0.1347 | 16.44 / 18.67% | reject |

E600 improved assignment action prediction (`val top1/top5=64.57/89.30%`). E606 wide DP also improved W/A recall and diversity: the priority-merged E616 predictions reached W/A@5/20 `68.55/77.93%` and rows>=7 W/A@5/20 `67.56/72.89%` before rendering.

The gain did not convert into the needed CIF metrics. E620 is the best K20 diagnostic in this batch, but it still misses match@5 target `41.58%` and match@20 target `49.69%`, and rows>=7 does not improve beyond E423. Current best for the main gate remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.

## 2026-06-15 coarse pair-delta coverage test

E621-E628 tested whether E573 failed only because exact pair-delta signatures were too sparse. A new `opentry_coarse_pair_delta_geometry_variants.py` bank used train-only StructureMatcher labels but matched validation candidates through broader GT-free keys: SG/crystal-system, row-count buckets, and free-symbol multisets.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| E627 coarse pair-delta lattice-only | val first64 | 4.69% | 4.69% | 4.69% | 0.1576 / 0.1575 / 0.1575 | 10.00 / 10.00% | 60.00 / 60.00% | 0 | reject |
| E628 coarse pair-delta full-param | val first64 | 4.69% | 4.69% | 4.69% | 0.2851 / 0.2753 / 0.1819 | 10.00 / 10.00% | 60.00 / 60.00% | 0 | reject |

The coarse key fixed coverage: both variants generated candidates for all 30 rows>=7 samples in the first64 set, compared with only 5 samples for exact pair-delta E573. But coverage did not become useful positives. Lattice-only had rows>=7 candidate positive rate `4.55%`; full-param was worse at `2.88%`; both E625/E626 merges added `0` new positive samples and kept rows>=7 positive-any at `20.00%`.

Conclusion: pair-delta/coarse-delta geometry transfer should stop. It can move many candidates but still leaves W/A-hit match-fail very high (`90%+`) and does not raise the complex-row sample ceiling. Current best remains E423 val512; full test remains closed.

## 2026-06-15 source-free train-prior geometry sampler

E629-E636 tested a route that no longer copies validation source geometry or transfers deltas. The new `opentry_source_free_prior_geometry_sampler.py` builds train-only priors from train-positive rendered rows, then samples lattice/free parameters for predicted W/A keys on validation rows>=7 candidates.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E633 source-free uniform | val first64 | 0.00% | 0.00% | 0.00% | null / null / null | 76.67 / 80.00% | 0 | reject |
| E634 source-free orbit prior | val first64 | 0.00% | 0.00% | 0.00% | null / null / null | 76.67 / 80.00% | 0 | reject |

Both variants rendered 765 candidates across all 30 rows>=7 samples and used 209 train-positive rows as the prior source. They preserved nontrivial W/A and skeleton recall, but E631/E632 found `0` positives and `100%` W/A-hit match-fail. Merging with baseline added `0` positives and kept rows>=7 positive-any at `20.00%`.

Conclusion: naive source-free train-prior lattice/parameter sampling is not enough. The next route needs a stronger match-aware generation objective, not independent prior sampling. Current best remains E423 val512; full test remains closed.

## 2026-06-15 compatibility-guided source-free proposal scoring

E637-E645 tested the next stronger variant of the source-free route: train a GT-free geometry compatibility GBDT from train-only StructureMatcher labels, score source-free train-prior proposals during generation, and only persist top candidates.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| E645 E554 baseline rerun | val first64 | 25.00% | 39.06% | 42.19% | 0.0549 / 0.1526 / 0.1409 | 16.67 / 20.00% | 80.00 / 83.33% | - | reference |
| E639 E637 compat-guided GBDT | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 0.00 / 0.00% | 73.33 / 80.00% | 0 | reject |
| E640 E638 compat-guided strict GBDT | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 0.00 / 0.00% | 73.33 / 76.67% | 0 | reject |

Both guided variants covered all 30 rows>=7 samples and retained 1,330 candidates, but E641/E642 labels found `0` positives. W/A-hit rows were abundant (`308` and `305`) and failed StructureMatcher at `100%`. E643/E644 merges added no positive sample beyond baseline, leaving rows>=7 positive-any at `20.00%`.

Conclusion: compatibility scoring cannot rescue independently sampled source-free train-prior geometry. This branch should stop unless the generator/objective itself changes; the current best remains E423 val512 `33.01/40.82/45.51/48.24`, with only match@1 above the +5pp target and full test still closed.

## 2026-06-15 compatibility-guided local geometry search

E646c-E653 moved the compatibility signal from independent source-free proposal scoring into a local geometry search around E554 baseline candidates. The model was trained only on train labels; inference used GT-free features from rendered CIFs. An overwide E646 was interrupted, and a duplicate-selection bug found by E646b was fixed before the valid runs.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| E645 E554 baseline rerun | val first64 | 25.00% | 39.06% | 42.19% | 0.0549 / 0.1526 / 0.1409 | 16.67 / 20.00% | 80.00 / 83.33% | - | reference |
| E648 E646c joint local search | val first64 rows>=7 proposals | 4.69% | 7.81% | 7.81% | 0.2476 / 0.2619 / 0.2412 | 16.67 / 16.67% | 76.67 / 80.00% | 0 | reject |
| E649 E647 param-only local search | val first64 rows>=7 proposals | 6.25% | 7.81% | 7.81% | 0.3172 / 0.2391 / 0.2355 | 16.67 / 16.67% | 76.67 / 80.00% | 0 | reject |

Both local-search variants produced 40 positive candidate rows, unlike the source-free prior runs, but E652/E653 showed those positives are all within samples already positive under baseline. rows>=7 positive-any stayed `20.00%`, and W/A-hit match-fail remained high (`89.36%`).

Conclusion: local perturb/search around E554 geometry does not raise the candidate ceiling. Future work must create positives for previously negative rows>=7 samples, not local variants of already covered samples. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.

## 2026-06-15 contrastive geometry prototype transfer

E654-E661 tested a stronger source-free representation before rendering. `opentry_contrastive_geometry_prototype_sampler.py` builds train-only positive/negative prototype contexts from geometry compatibility labels, then renders predicted rows>=7 W/A candidates with prototype lattice/free parameters and GT-free compatibility scoring.

| run | scope | full match@1 | full match@5 | full match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| E656/E658 strict prototype | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 0.00 / 0.00% | 80.00 / 83.33% | 0 | reject |
| E657/E659 broad prototype | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 0.00 / 0.00% | 80.00 / 83.33% | 0 | reject |

The broad variant improved rows>=7 readability and composition-exact coverage (`readable_any@50=56.67%`, `composition_exact_any@50=50.00%`), but both variants had `0` StructureMatcher positives and `100%` W/A-hit match-fail. E660/E661 merge audits added `0` positive samples beyond E554 baseline and rows>=7 positive-any stayed `20.00%`.

Conclusion: contrastive prototype/source-free representation transfer should stop in this configuration. It preserves symbolic W/A recall but does not solve W/A-to-geometry conversion. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp and full test remains closed.

## 2026-06-15 positive row-sequence geometry decoder

E662-E666 tested a different geometry objective rather than another source-copy or selector variant. `opentry_positive_rowseq_geometry.py` trains a probabilistic row-sequence decoder from formula + GT-SG + predicted W/A rows to lattice/free parameters, using train rendered StructureMatcher-positive candidates. Validation positives are used only for checkpoint selection; test remains closed.

| run | scope | match@1 | match@5 | match@20 | RMSE@1/5/20 | rows>=7 W/A@20 / skeleton@20 | rows>=7 positive-any after merge | decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| E664/E665 positive row-seq geometry | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 80.00 / 83.33% | 20.00% | reject |

The model rendered 462/462 attempted CIFs across all 30 rows>=7 samples, and rows>=7 candidates were composition-exact/readable at `100%` by sample. But StructureMatcher positives were `0`, and 72 W/A-hit rows all failed match. E666 merge added `0` positive samples beyond E554 baseline.

Conclusion: positive-only row-sequence geometry is still not enough. It keeps the symbolic representation healthy, but cannot place lattice/free parameters into the narrow geometry basin required for MPTS-52 rows>=7. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.

## 2026-06-15 negative-aware row-sequence energy sampler

E667-E676 added negatives to the row-sequence geometry route. `opentry_contrastive_rowseq_energy_sampler.py` trains a binary energy model on rendered train candidates, then scores samples from the E662 probabilistic geometry decoder before rendering.

| run | energy train scope | val proposal scope | match@1 | match@5 | match@20 | RMSE@1/5/20 | rows>=7 W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| E667-E671 | rows>=7 train labels | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 73.33 / 76.67% | 0 | reject |
| E672-E676 | full train labels | val first64 rows>=7 proposals | 0.00% | 0.00% | 0.00% | null / null / null | 73.33 / 76.67% | 0 | reject |

Both variants rendered 894/894 proposals over the 30 rows>=7 samples. Both produced `0` StructureMatcher positives, `100%` W/A-hit match-fail, and no new positive samples after merging with E554 baseline. The full-energy model had more train data, but its fixed-pool rows>=7 K5/K20 remained only `10.00/20.00%`.

Conclusion: adding a negative-aware energy score on top of positive row-sequence samples still does not enter the correct geometry basin. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.

## 2026-06-15 pairfield Adam constrained geometry

E677-E691 tested a materially different geometry route: optimize rendered candidates with a differentiable physical objective rather than scoring sampled geometry. `opentry_pairfield_adam_refine.py` uses only train CIF pair-distance and volume-per-atom statistics; val StructureMatcher labels are used only for diagnostics and selection.

| run | scope | match@1 | match@5 | match@20 | RMSE@1/5/20 | rows>=7 match@5/20 | W/A@20 / skeleton@20 | new positives vs baseline | decision |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| E679/E681/E683 repel | E554 val64 rows>=7 | 4.69% | 7.81% | 9.38% | 0.2970 / 0.2901 / 0.3181 | 16.67 / 20.00% | 80.00 / 83.33% | 1 | auxiliary only |
| E680/E682/E684 pairfield | E554 val64 rows>=7 | 4.69% | 7.81% | 7.81% | 0.3157 / 0.3185 / 0.3161 | 16.67 / 16.67% | 80.00 / 83.33% | 0 | reject |
| E687/E690/E691 repel aligned55 | E421 rows>=7 aligned subset | 10.91% | 14.55% | 16.36% | 0.1409 / 0.1770 / 0.1710 | 14.55 / 16.36% | 72.73 / 76.36% | 1 | auxiliary only |
| E688/E689 baseline aligned55 | same subset | 12.73% | 18.18% | 25.45% | 0.0617 / 0.0920 / 0.0733 | 18.18 / 25.45% | 74.55 / 78.18% | reference | reference |

Important caveat: the first E685 scale eval was invalid because the E421 rendered pool order did not match `val.jsonl --max-records 128`; only 9 of 55 output sample IDs were in that repr prefix. E687/E688/E691 use an aligned 55-sample repr subset and are the valid comparison.

Conclusion: repel-style differentiable physical optimization is not a good ordered system: direct K5/K20 and RMSE are worse than baseline. But it repeatedly added `1` baseline-missing positive sample on val rows>=7, raising all-candidate positive-any from `20.00%` to `23.33%` on val64 and from `25.45%` to `27.27%` on the aligned55 subset. This is the first recent geometry branch with a small candidate-ceiling gain, so it should continue only as auxiliary candidate generation with better train-only objectives or a validation-selected compatibility selector. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp and full test remains closed.

## 2026-06-15 compatibility selector on baseline+repel pool

E692-E699 tested whether train-label GT-free compatibility selectors can exploit the small repel candidate-ceiling gain. Training used only train feature labels; the aligned55 validation labels were used only to compare selectors.

| run | train labels | aligned55 match@1 | match@5 | match@20 | RMSE@1/5/20 | W/A@20 / skeleton@20 | decision |
|---|---|---:|---:|---:|---|---:|---|
| E688 baseline | none | 12.73% | 18.18% | 25.45% | 0.0617 / 0.0920 / 0.0733 | 74.55 / 78.18% | reference |
| E696 E318 GBDT | broad train | 10.91% | 18.18% | 25.45% | 0.0748 / 0.0884 / 0.1037 | 74.55 / 78.18% | weak |
| E697 E318 pairwise | broad train | 10.91% | 20.00% | 25.45% | 0.1088 / 0.0834 / 0.0730 | 72.73 / 76.36% | weak |
| E698 rows7 GBDT | rows>=7 train | 14.55% | 20.00% | 25.45% | 0.0829 / 0.0687 / 0.0981 | 74.55 / 78.18% | continue |
| E699 rows7 pairwise | rows>=7 train | 12.73% | 21.82% | 27.27% | 0.1748 / 0.1275 / 0.1089 | 74.55 / 78.18% | continue |

Conclusion: the rows>=7-specific selector is directionally useful on this small aligned subset: E699 improves K5 and K20 over the same-subset baseline, and E698 improves K1/K5 with better K5 RMSE. The evidence is still too narrow for full test, and RMSE tradeoffs remain. Next step is to scale the rows>=7 selector validation to a larger aligned E421/E423 val subset with the same no-leakage rules.

## 2026-06-15 larger aligned216 selector scale check

E700-E709 scaled the same baseline+repel idea from aligned55 to a larger 216-sample rows>=7 validation subset. This was still validation-only; test remained untouched.

| run | scope | match@1 | match@5 | match@20 | match@50 | RMSE@1/5/20 | W/A@20 / skeleton@20 | decision |
|---|---|---:|---:|---:|---:|---|---:|---|
| E701 baseline | E421 aligned216 rows>=7 | 11.57% | 15.28% | 18.52% | 19.44% | 0.1370 / 0.1280 / 0.1171 | 70.83 / 78.70% | reference |
| E702 repel direct | E700 repel aligned216 | 9.26% | 12.04% | 13.43% | 13.43% | 0.2062 / 0.2063 / 0.2132 | 68.98 / 76.39% | reject |
| E708 rows7 GBDT | E705 merged pool | 11.57% | 14.81% | 18.98% | 20.37% | 0.0942 / 0.0965 / 0.1322 | 70.37 / 78.24% | reject for scale |
| E709 rows7 pairwise | E705 merged pool | 7.87% | 13.89% | 18.98% | 20.37% | 0.1927 / 0.1434 / 0.1305 | 70.37 / 78.24% | reject for scale |

E705 showed why the selector had little room to help: repel added only `2` baseline-missing positive samples on 216 complex validation samples, raising all-candidate positive-any from `19.44%` to `20.37%`. The W/A-hit match-fail rate stayed above `90%` for both baseline and repel labels.

Conclusion: the aligned55 selector improvement did not generalize. The current failure is not mostly ranking; it is the predicted W/A to matched CIF conversion, especially creating new rows>=7 StructureMatcher-positive geometry candidates. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, so full test remains closed.

## 2026-06-15 selective replacement gate result

E710-E718 revisited fixed-pool ranking with a stricter risk-control rule rather than a full rerank. The selector preserves the first four original candidates, then inserts only high-confidence E425 GBDT candidates under a frozen threshold and max-per-WA cap.

| run | apply path | match@1 | match@5 | match@20 | match@50 | RMSE@1/5/20 | rows>=7 K5/K20 | decision |
|---|---|---:|---:|---:|---:|---|---:|---|
| E423 baseline | E420/E421/E422 original | 33.01% | 40.82% | 45.51% | 48.24% | 0.1095 / 0.1201 / 0.1309 | 16.89 / 18.67% | previous best |
| E712b | E425 val feature scores + E710 selective | 33.01% | 41.80% | 45.51% | 45.51% | 0.1095 / 0.1275 / 0.1309 | 17.78 / 18.67% | passes K1/K5 |
| E718 | GT-free rendered-top50 scoring + E710 selective | 33.01% | 41.99% | 45.90% | 48.24% | 0.1095 / 0.1281 / 0.1318 | 18.22 / 19.11% | frozen candidate |

E718 is now the best validation system: it clears the MPTS-52 CrystaLLM GT-SG +5pp targets for match@1 and match@5, while match@20 still misses. It uses `reports/e425_gbdt_e318_train_to_e424_val512/compat_model.joblib`, `opentry_score_rendered_candidates_apply.py`, and `opentry_apply_selective_replacement.py` with frozen params `threshold=0.0024707304479371964`, `anchor_count=4`, `max_per_wa=2`, `top_k=50`.

No full test has been run for this frozen config. The no-leakage audit is in `reports/e718_freeze_no_leakage_audit/frozen_config_audit.md`. Next step is a single frozen full-test attempt after generating the E420/E421/E422-style full MPTS-52 test candidate pool; no full-test feedback may be used for retuning.
