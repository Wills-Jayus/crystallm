# SymCIF-v4 Free Param Re-extraction Report

日期：2026-05-22

本轮只做 source CIF -> OrbitEngine canonical free_params 重提取和 Gate 1 roundtrip 验证；没有训练模型、没有修 scorer、没有 jitter、没有 relabel、没有删除样本、没有跑 full match@5。

## 1. 实现内容

新增/更新：

- `src/symcif_v4/free_param_extractor.py`
- `scripts/reextract_symcif_v4_free_params_from_source.py`
- `scripts/audit_symcif_v4_reextracted_gate1.py`

核心做法：

1. 对每个 structured v3 WA row，根据 `sample_id` 回到原始 source CIF。
2. 从 source CIF 的 atom_site representative row 读取元素、multiplicity、fractional coordinate。
3. 对目标 OrbitToken 遍历 symmetry operations，将 source coordinate 反解为 canonical representative 的 free_params。
4. 只接受 residual 合法、source coordinate 能在展开 orbit 中复现、且 expanded unique count 等于 declared multiplicity 的参数。
5. 找不到合法重提取时记录 failure，并保留 old params 作为显式 fallback；不静默设 0。

## 2. Re-extraction 统计

输入：`data/structured_symcif_v3/{train,val,test}.jsonl`

输出：`data/structured_symcif_v4_reextracted/{train,val,test}.jsonl`

| 指标 | 数值 |
| --- | ---: |
| samples | 6000 |
| WA rows | 33757 |
| row extraction success | 32827 / 33757 = 97.2450% |
| row expansion_ok after reextract | 33757 / 33757 = 100.0000% |
| sample row_expansion_all_ok | 6000 / 6000 = 100.0000% |

Extraction method 分布：

| method | rows |
| --- | ---: |
| direct_coordinate_solve | 24520 |
| fixed_orbit | 8049 |
| linear_lstsq_combination_expr | 258 |
| fallback_old_params_no_valid_source_extraction | 930 |

fallback 的 930 行没有导致 row collapse；这些行后续可以继续改善 source matching，但不阻塞 Gate 1。

## 3. Top Failed SG/Letter 回归

| SG/letter | before failed/rows | after degenerate/occurrences | after expansion_ok |
| --- | ---: | ---: | ---: |
| 225\|24e | 154/154 | 0/154 | 100.00% |
| 189\|3f | 79/79 | 0/79 | 100.00% |
| 189\|3g | 80/80 | 0/80 | 100.00% |
| 193\|6g | 51/51 | 0/51 | 100.00% |
| 216\|16e | 0/86 | 0/86 | 100.00% |
| 216\|24f | 25/25 | 0/25 | 100.00% |

结论：上一轮 Gate 1 的系统性塌缩已经消失。SG225 24e、SG189 3f/3g、SG193 6g、SG216 24f 的 source-coordinate 重提取都能恢复 declared multiplicity。

## 4. Gate 1 Roundtrip

输入：`data/structured_symcif_v4_reextracted`

输出：

- `reports/symcif_v4_reextract/gate1_roundtrip_summary.json`
- `reports/symcif_v4_reextract/gate1_failures.jsonl`
- `reports/symcif_v4_reextract/gate1_breakdown_per_sg_letter.csv`

| 指标 | 数值 | 验收线 | 是否通过 |
| --- | ---: | ---: | --- |
| readable | 100.00% | >= 99% | yes |
| formula_ok | 100.00% | >= 99% | yes |
| atom_count_ok | 100.00% | >= 99% | yes |
| row_expansion_ok | 100.00% | >= 99% | yes |
| SG_ok | 99.65% | >= 98% | yes |

Gate 1：通过。

剩余 failed samples：21 / 6000，全部是 readable/formula_ok/atom_count_ok/row_expansion_ok 成立后 SG detection 未通过，属于 detection-only/setting-sensitive 残差。

剩余失败 target SG 分布：

| target SG | failed samples |
| ---: | ---: |
| 88 | 5 |
| 137 | 4 |
| 166 | 3 |
| 125 | 3 |
| 227 | 3 |
| 141 | 2 |
| 70 | 1 |

## 5. 结论

1. free_params 重提取修复了 row_degeneracy：row_expansion_ok 从旧数据的系统性失败恢复到 100%。
2. Gate 1 已通过：readable/formula_ok/atom_count_ok/SG_ok 均超过验收线。
3. 当前可以进入 streaming WA search / orbit-aware scorer 阶段，但必须使用 `data/structured_symcif_v4_reextracted`，不要继续使用旧的 `data/structured_symcif_v4` free_params。
4. 后续若要进一步打磨数据层，应优先处理 930 个 fallback rows 的 source-row matching/linear expression coverage；这不是 Gate 1 blocker。
