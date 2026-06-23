# Composition-Exact Wyckoff Table Generator / SymCIF-v4 Frontend Summary

日期：2026-05-21

## 1. 本轮做了什么

本轮基于 `data/structured_symcif_v3` 实现并跑通了第一版 SymCIF-v4 结构化前端：

`full conventional formula + fixed SG -> exact-cover skeleton/WA-table candidates -> WA-table scorer -> coords/lattice fallback -> CIF -> CIF-level closure diagnostics`

新增代码：

- `src/symcif_v4/formula.py`
- `src/symcif_v4/wyckoff_table.py`
- `src/symcif_v4/exact_cover.py`
- `src/symcif_v4/wa_table.py`
- `src/symcif_v4/render_cif.py`
- `src/symcif_v4/scorer.py`
- `scripts/audit_full_formula_composition.py`
- `scripts/build_composition_exact_candidates.py`
- `scripts/compress_wa_candidates.py`
- `scripts/train_wa_table_scorer.py`
- `scripts/eval_composition_exact_frontend.py`
- `scripts/render_composition_exact_cifs.py`
- `scripts/run_composition_exact_v1.sh`

主要产物：

- `reports/composition_exact_v1/formula_audit.json`
- `reports/composition_exact_v1/candidate_generation_summary.json`
- `reports/composition_exact_v1/candidate_generation_per_sg*.csv`
- `reports/composition_exact_v1/candidate_generation_per_nsites*.csv`
- `reports/composition_exact_v1/test_candidates.jsonl`
- `reports/composition_exact_v1/test_wa_candidates.jsonl`
- `reports/composition_exact_v1/wa_scorer_eval_summary.json`
- `reports/composition_exact_v1/wa_scorer_breakdown_per_sg_neural.csv`
- `reports/composition_exact_v1/wa_scorer_breakdown_per_nsites_neural.csv`
- `reports/composition_exact_v1/render_cif_summary.json`
- `reports/composition_exact_v1/rendered_test_topk.jsonl`
- `reports/composition_exact_v1_gt_oracle/render_cif_summary.json`

`test_candidates.jsonl` 是 `test_wa_candidates.jsonl` 的兼容 hardlink，避免重复占用 1.9GB 存储。

## 2. Formula Audit

本轮严格使用完整 conventional cell formula counts 作为 exact-cover target，不使用 reduced formula，不枚举 latent Z。`reduced_formula` 和 `z_from_counts` 只用于日志和分桶。

| 指标 | 数值 |
| --- | ---: |
| total_records | 6000 |
| train / val / test | 5000 / 500 / 500 |
| parsed_equals_structured_rate | 100.00% |
| z_consistency_rate | 99.9833% |
| formula parse fail | 0 |
| formula count mismatch | 0 |

唯一 `z` warning：

- `mpts_52_test_orig__mp-1066100`, formula=`O4`, structured_z=2, z_from_counts=4

这不影响实验，因为 target_counts 始终是 full conventional formula_counts。

## 3. Exact-Cover Candidate Coverage

最终稳定跑法：

- skeleton cap: `max_skeleton_candidates=20000`
- eval WA cap: `max_wa_candidates=5000`
- train WA cap: `max_wa_candidates=1000`
- per-sample timeout: `30s`

说明：按 prompt 中 `max_wa_candidates=50000` 的原始规模试跑时，test 中途文件已超过十几 GB 且长时间无推进；本轮按“及时介入长时间卡顿”的要求改为稳定 cap，并把 cap/truncation 全部记录到 summary。

### 3.1 Split Summary

| split | samples | skeleton coverage | WA coverage | skeleton mean/median/p90/max | WA mean/median/p90/max | skeleton truncated | WA truncated | timeout |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| train | 5000 | 99.50% | 78.42% | 534.3 / 16 / 406 / 20000 | 455.3 / 176 / 1000 / 1000 | 61 | 1889 | 0 |
| val | 500 | 99.40% | 89.80% | 642.2 / 20 / 451 / 20000 | 1700.5 / 166 / 5000 / 5000 | 8 | 136 | 0 |
| test | 500 | 99.20% | 86.60% | 611.8 / 18.5 / 440 / 20000 | 1897.8 / 228 / 5000 / 5000 | 7 | 156 | 0 |

结论：

- skeleton candidate coverage 99.2%，明显超过旧 train-catalog upper bound 80.6%，并达到目标 >=95%。
- WA exhaustive coverage test=86.6%，未达到目标 90%。
- test 中 WA miss 共 67 个样本：63 个是 WA cap 截断导致 GT 没进入前 5000；4 个是 skeleton cap 截断导致 GT skeleton 未进入候选。
- zero skeleton candidate = 0，zero WA candidate = 0，timeout = 0。说明 exact-cover 可行性本身不是问题，主要是候选爆炸和枚举顺序/cap。

### 3.2 Breakdown

按 n_sites：

| n_sites | samples | skeleton coverage | WA coverage | 说明 |
| ---: | ---: | ---: | ---: | --- |
| 1-4 | 256 | 100.00% | 100.00% | 小结构完全可覆盖 |
| 5 | 69 | 100.00% | 95.65% | 开始出现 WA cap |
| 6 | 45 | 100.00% | 66.67% | WA 候选爆炸明显 |
| 7 | 17 | 94.12% | 52.94% | skeleton/WA cap 都开始影响 |
| 8 | 15 | 100.00% | 53.33% | WA cap 影响大 |
| 9 | 23 | 91.30% | 56.52% | skeleton cap 和 WA cap 都影响 |

按元素数：

| num_elements | samples | skeleton coverage | WA coverage |
| ---: | ---: | ---: | ---: |
| 1 | 50 | 100.00% | 100.00% |
| 2 | 55 | 100.00% | 96.36% |
| 3 | 203 | 99.01% | 90.15% |
| 4 | 150 | 98.67% | 80.00% |
| 5 | 37 | 100.00% | 67.57% |
| 6 | 5 | 100.00% | 40.00% |

主要规律：结构越大、元素越多，WA assignment 组合数越大，`max_wa_candidates=5000` 很快成为主限制。

## 4. WA-Table Scorer

训练实现了两个 scorer：

- Baseline scorer：SG prior、site token frequency、multiplicity pattern、element@site、anonymous formula、nsites prior。
- Neural scorer：formula vector + SG embedding + WA rows DeepSets pooling，candidate set CE / sampled negatives。

由于完整候选文件过大，训练使用 `reports/composition_exact_v1_trimmed`：每个样本保留 baseline top200，并强制保留 GT candidate（如果 GT 原本在 candidates 中）。训练集 GT in candidates = 3921/5000。

| scorer | skeleton top1 | skeleton top5 | skeleton top20 | WA top1 | WA top5 | WA top20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 66.8% | 72.4% | 77.8% | 41.2% | 63.0% | 70.0% |
| neural | 59.2% | 72.0% | 80.2% | 38.2% | 61.4% | 72.4% |

结论：

- exact-cover 候选空间已经突破 train-catalog skeleton ceiling，但 scorer 没有充分利用这个空间。
- neural skeleton top20=80.2%，没有显著超过旧 80.6% ceiling。
- neural WA top20=72.4%，没有超过之前组合式 assignment 的 joint_pair_top20=76.8%。
- assignment 在 oracle skeleton 下曾达到 top20=96.6%，所以“assignment 能否闭合”不是主瓶颈；当前瓶颈是候选爆炸后的排序和 CIF 渲染闭合。

按 n_sites 的 neural WA top20：

| n_sites | WA top20 |
| ---: | ---: |
| 1 | 100.0% |
| 2 | 100.0% |
| 3 | 93.5% |
| 4 | 81.3% |
| 5 | 47.8% |
| 6 | 62.2% |
| 7 | 47.1% |
| 8 | 46.7% |
| 9 | 43.5% |

复杂结构排序明显掉线。

## 5. Rendered CIF Closure

本轮必须区分三件事：

- intermediate WA-table closure：按 multiplicity 计数，by construction formula-exact。
- rendered CIF formula_ok：CIF 被 pymatgen 读回后，展开结构的 composition 是否等于 full conventional target_counts。
- rendered CIF SG_ok：pymatgen/spglib 识别出的 SG 是否等于输入 SG。

### 5.1 Retrieval Render

`coord_mode=retrieval`, top20 neural predictions：

| subset | generations | readable | formula_ok | SG_ok | samples_with_formula_ok | samples_with_sg_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank1 | 500 | 86.6% | 81.2% | 83.4% | 406/500 | 417/500 |
| top5 | 2332 | 86.15% | 80.49% | 82.38% | 431/500 | 443/500 |
| top20 | 8303 | 84.04% | 78.68% | 79.56% | 436/500 | 446/500 |

按坐标来源：

| coord_source | generations | readable | formula_ok | SG_ok |
| --- | ---: | ---: | ---: | ---: |
| train_same_wa | 100 | 99.0% | 95.0% | 95.0% |
| train_same_skeleton | 3448 | 97.7% | 94.5% | 95.0% |
| sg_median_lattice | 4715 | 74.0% | 67.0% | 68.1% |
| default_safe | 40 | 55.0% | 55.0% | 55.0% |

### 5.2 GT Oracle Render Diagnostic

`coord_mode=gt_oracle` 只作为渲染链路诊断，不作为生成方法。

| subset | readable | formula_ok | SG_ok |
| --- | ---: | ---: | ---: |
| rank1 | 87.6% | 82.2% | 84.8% |
| top5 | 86.45% | 80.75% | 82.76% |
| top20 | 84.10% | 78.72% | 79.66% |

`gt_oracle_exact_wa` 子集：

| coord_source | generations | readable | formula_ok | SG_ok |
| --- | ---: | ---: | ---: | ---: |
| gt_oracle_exact_wa | 362 | 93.1% | 85.1% | 90.3% |

关键结论：即使 WA-table 等于 GT 并使用 GT coords/lattice，CIF-level formula_ok 也只有 85.1%。这证明剩余主要问题不是 WA-table 不闭合，而是 Wyckoff lookup/坐标 setting 与当前 `pymatgen Structure.from_spacegroup` 渲染链路不完全一致。

## 6. Render Failure Root Cause

本轮修掉了一个明确 bug：

- 旧 `default_safe` 只按 `free_mask/fixed_values` 填坐标，忽略 `representative_expr`。
- 对 SG 194 的 `12k: x,2x,z` 这类带参数关系的 Wyckoff 位点，直接填 `(x,y,z)` 会被渲染成 general multiplicity 24，而不是 12。
- 修复后 retrieval rank1 formula_ok 从 75.2% 提升到 81.2%。

修复后仍未达 98%，原因更深：

1. **Wyckoff lookup 和 pymatgen setting/origin 不一致。**  
   例子：SG 227 `Fd-3m` 中，lookup/structured data 认为 `8a=(1/8,1/8,1/8)`、`16d=(1/2,1/2,1/2)`；但 `pymatgen Structure.from_spacegroup('Fd-3m', ...)` 对这些坐标的展开 multiplicity 与 lookup 标注不一致，导致 CIF 读回后元素计数被交换或 multiplicity 变化。

2. **当前 `render_standard_cif_v3` 依赖 pymatgen 内置 SG setting。**  
   exact-cover 用的是 CrystalFormer/Wyckoff lookup 的 letter/multiplicity 约定；渲染用的是 pymatgen 的 space-group operations。两者 setting 一旦不一致，WA-table 中间闭合不等于 CIF-level formula_ok。

3. **当没有 same skeleton / same WA 训练样本可检索时，`sg_median_lattice + default coords` 的 closure 明显差。**  
   `sg_median_lattice` 占 4715/8303 generations，formula_ok 只有 67.0%，是 top20 整体 formula_ok 的最大拖累。

4. **spglib SG 识别本身也不完全稳定。**  
   top20 中 `sg_detect` 错误 27 次；另外 1325 个 CIF 解析为 no structures，主要来自坐标/setting 导致的不可读或重合/异常结构。

## 7. 是否满足验收标准

| 验收项 | 结果 | 是否通过 |
| --- | --- | --- |
| 使用 full conventional formula_counts | parsed_equals_structured_rate=100% | 通过 |
| 不使用 reduced formula / latent Z | 代码和日志均确认 | 通过 |
| skeleton candidates 不限制 train catalog | exact-cover 组合生成 | 通过 |
| skeleton coverage 超过 80.6% | test=99.2% | 通过 |
| skeleton coverage >=95% | test=99.2% | 通过 |
| WA exhaustive coverage >=90% | test=86.6% | 未通过 |
| WA top20 >76.8% | neural=72.4% | 未通过 |
| rendered readable >=90% | rank1=86.6%, top20=84.0% | 未通过 |
| rendered formula_ok >=98% | rank1=81.2%, top20=78.7% | 未通过 |
| full match@5 | 未跑 | 正确，因为 gating 未满足 |

## 8. 是否建议进入 coords/lattice 模型阶段

暂时不建议。

理由：

- 离散 skeleton candidate coverage 已经解决到 99.2%，但 WA ranker top20 只有 72.4%，排序还没过线。
- CIF-level formula_ok 只有 81.2% rank1，且 `gt_oracle_exact_wa` 也只有 85.1%，说明渲染链路本身有 setting/operation 约定问题。
- 在 renderer/lookup convention 没修正前，训练 coords/lattice 模型会把系统性渲染错误混进连续变量学习，实验信号会很脏。

## 9. 是否建议跑 full match@5

不建议。

按 prompt 的 gating 条件：

- skeleton coverage 已通过。
- WA top20 未超过 76.8%。
- rendered formula_ok 远低于 98%。
- readable 未达到 90%。
- SG_ok 仍有明显系统性错误。

因此 full StructureMatcher match@5 现在不会提供清晰结论，只会把 WA ranking、renderer setting、坐标/lattice 三类错误混在一起。

## 10. 下一步建议

1. **先修 renderer/lookup convention，而不是训练新坐标模型。**  
   需要让 exact-cover 使用的 Wyckoff letter/multiplicity/coordinate template 与 CIF 渲染使用的 symmetry operations 完全同源。优先方案是不要再通过 `Structure.from_spacegroup(record.sg_symbol, ...)` 间接渲染，而是基于同一份 CrystalFormer/Wyckoff operations 构造 CIF symmetry operations 和 atom loop，或建立 lookup -> pymatgen setting 的确定性映射。

2. **为 Wyckoff representative expression 建立单元测试。**  
   对每个 SG/letter，使用默认参数生成坐标，展开后 multiplicity 必须等于 lookup multiplicity。当前 SG 194 `12k` 已修复，但 SG 227 等 setting 问题仍存在。

3. **优化 WA candidate 枚举和排序。**  
   test 中所有 WA miss 都来自 cap：63 个 WA cap、4 个 skeleton cap。应把 WA 枚举改成 streaming beam/priority search，而不是先生成巨量 JSONL；同时训练 scorer 应使用 hard negatives 和更强 candidate-set objective。

4. **WA scorer 需要重新训练更强版本。**  
   当前 neural 只训练 3 epochs，且在 top200 trimmed candidates 上训练；下一版应减少 IO 瓶颈后用更多候选、更长训练、更强特征，包括 Wyckoff operation/setting feature、anonymous formula compatibility、site symmetry group feature。

5. **renderer closure 达到阈值后再进入 coords/lattice。**  
   目标是 retrieval/default_safe 下 formula_ok >=98%、readable >=90%、SG_ok 无系统错误；然后再训练连续坐标/lattice 模型，最后才跑 full match@5。

