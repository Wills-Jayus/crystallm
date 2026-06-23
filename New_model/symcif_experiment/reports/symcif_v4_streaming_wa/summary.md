# SymCIF-v4 Streaming WA Stage Report

日期：2026-05-22

本轮继续执行 `2026522_01.md` 的后续阶段：Streaming Composition-Exact WA Generator、orbit-aware WA ranker、OrbitEngine CIF rendering。没有继续修 v2/v3 文本 sampler，没有训练 coords/lattice，没有跑 full match@5。

## 1. 实现内容

新增/改造：

- `src/symcif_v4/wa_search.py`
- `scripts/run_streaming_wa_search.py`
- `scripts/train_orbit_aware_wa_scorer.py`
- `scripts/eval_streaming_wa_generator.py`
- `scripts/render_symcif_v4_cifs.py`
- `scripts/run_symcif_v4_stage1.sh`

数据输入统一切换为：

- `data/structured_symcif_v4_reextracted`

原因：旧 `data/structured_symcif_v4` 的 free_params 存在 canonical representative 映射错误；reextracted 数据已经通过 Gate 1。

## 2. OrbitEngine / Convention

OrbitEngine 数据源：

- Wyckoff lookup: `artifacts/wyckoff_lookup_full.json`
- canonical orbit id: `setting=crystalformer|sg=...|<multiplicity><letter>|enum=...|sym=...`
- CIF rendering: OrbitEngine 自己展开 full atom positions，不使用 `Structure.from_spacegroup` 作为核心 renderer。

Orbit multiplicity unit test：

| 指标 | 数值 |
| --- | ---: |
| total_orbits | 1731 |
| passed | 1731 |
| failed | 0 |
| pass_rate | 100.00% |

Canonical dataset：

| 指标 | 数值 |
| --- | ---: |
| train / val / test | 5000 / 500 / 500 |
| map failures | 0 |
| unique canonical skeletons | 1706 |
| unique canonical WA | 5493 |

## 3. Gate 1: GT WA + GT coords/lattice Roundtrip

基于 `data/structured_symcif_v4_reextracted`：

| 指标 | 全量 6000 |
| --- | ---: |
| readable | 100.00% |
| formula_ok | 100.00% |
| atom_count_ok | 100.00% |
| row_expansion_ok | 100.00% |
| SG_ok | 99.65% |

Gate 1：通过。

注意区分：

- intermediate WA closure：WA multiplicity 能 exact-cover formula。
- OrbitEngine expanded atom count：OrbitEngine 展开后 atom 数等于 full conventional formula count。
- rendered CIF formula_ok：CIF 经 pymatgen 读回后 composition 正确。
- rendered CIF SG_ok：CIF 经 spglib 检测后 SG 正确。
- StructureMatcher match：本轮没有跑。

## 4. Gate 2: Streaming WA Search

配置：

- split: test
- beam_size: 1024
- top_k: 200
- candidate_multiplier: 8
- max_expanded_states: 1000000
- timeout_per_sample: 30s

Search 结果：

| 指标 | test |
| --- | ---: |
| candidate_nonempty | 100.00% |
| timeout_samples | 0 |
| truncated_samples | 145 |
| mean candidate count | 121.188 |
| median expanded states | 1298 |
| skeleton top20 / top100 / top200 | 72.40% / 82.40% / 87.00% |
| WA top20 / top100 / top200 | 69.60% / 80.20% / 85.80% |

Gate 2 search：未通过。

主要原因不是 timeout，而是 top200 candidate coverage 不够；复杂结构和候选截断仍然是主要瓶颈。

## 5. Orbit-Aware WA Ranker

实现的是第一版 orbit-aware MLP ranker，使用候选 WA 的 orbit-level 聚合特征：

- element@orbit train frequency
- orbit train frequency
- multiplicity / site_symmetry / letter / free_dof features
- candidate row count、fixed/free rows、total free DOF
- full formula atom count、num elements、SG

训练输出：

- `runs/orbit_aware_wa_scorer_v1/ckpt.pt`
- `reports/symcif_v4_streaming_wa/test_ranked_wa_predictions.jsonl`

训练说明：

- 当前 CUDA 在会话内不可见，使用 CPU。
- 为避免 1.3GB train candidates 全量特征构建长时间卡顿，本轮用前 1500 个 train candidate sets 做快速训练。
- train samples with positive: 1329
- val samples with positive: 422

Ranker test 指标：

| 指标 | test |
| --- | ---: |
| candidate_nonempty | 100.00% |
| skeleton top1 / top5 / top20 / top100 / top200 | 69.00% / 78.60% / 84.40% / 86.80% / 87.00% |
| WA top1 / top5 / top20 / top100 / top200 | 52.40% / 76.20% / 82.00% / 85.40% / 85.80% |

复杂子集：

| subset | samples | WA top20 | WA top100 | WA top200 |
| --- | ---: | ---: | ---: | ---: |
| n_sites >= 6 | 175 | 65.14% | 68.00% | 68.57% |
| num_elements >= 4 | 192 | 78.13% | 79.69% | 79.69% |

Gate 2 ranker：

- WA top20 > 76.8%：通过，达到 82.00%。
- skeleton top20 > 80.6%：通过，达到 84.40%。
- WA top100 >= 90%：未通过，当前 85.40%。

结论：ranker 能把 top20 拉过上一阶段基线，但受 candidate coverage 上限限制，top100/top200 达不到 Gate 2 强目标。

## 6. Gate 3: Rendered CIF Closure

### gt_oracle

Mode: GT WA + GT free_params + GT lattice

| 指标 | test |
| --- | ---: |
| readable | 100.00% |
| formula_ok | 100.00% |
| atom_count_ok | 100.00% |
| SG_ok | 100.00% |

gt_oracle render：通过。

### retrieved_geometry

Mode: predicted WA + retrieved train geometry / deterministic fallback

| 指标 | rank1 | all rendered top20 rows |
| --- | ---: | ---: |
| readable | 94.80% | 74.09% |
| formula_ok | 94.40% | 73.69% |
| atom_count_ok | 94.40% | 73.69% |
| SG_ok | 92.40% | 70.24% |

retrieved_geometry Gate 3：未通过。

原因：WA table 虽然 formula-exact，但把不匹配的 free_params/lattice 从 train skeleton/WA 迁移到新 WA 时，多个 orbit rows 之间会发生坐标碰撞或 symmetry detection 退化，导致 CIF 读回后 atom_count/formula/SG 下降。这不是 OrbitEngine 的 GT renderer 问题，而是 geometry retrieval/fallback 还不稳定。

## 7. Gate 状态

| Gate | 目标 | 状态 |
| --- | --- | --- |
| Gate 1 OrbitEngine roundtrip | formula_ok >=99%, SG_ok >=98%, readable >=99% | 通过 |
| Gate 2 streaming search | skeleton top200 >=99%, WA top200 >=90% | 未通过 |
| Gate 2 ranker | WA top20 >76.8%, WA top100 >=90% | 部分通过 |
| Gate 3 gt_oracle render | readable/formula/SG 过线 | 通过 |
| Gate 3 retrieved_geometry | formula_ok >=98%, readable >=90% | 未通过 |
| Gate 4 full match | 前三关过线后再跑 | 未执行 |

## 8. 是否进入下一阶段

不建议进入 coords/lattice continuous model，也不建议跑 full match@5。

理由：

1. Gate 2 candidate coverage 未过，WA top200 只有 85.8%。
2. 复杂结构 n_sites>=6 的 WA top100 只有 68.0%，说明 search policy 对复杂 exact-cover 仍然弱。
3. retrieved_geometry rank1 formula_ok 只有 94.4%，top20 全量只有 73.7%，说明几何迁移策略还不能稳定闭合 CIF。

## 9. 下一步建议

1. 改 streaming search 的状态评分，从 action frequency 升级为 step-level policy model，直接预测 `element@OrbitToken` next action，并在 beam search 内使用合法 action mask。
2. 提升 candidate coverage：对 truncated samples 做定向分析，尤其 n_sites>=6 和大 formula count 样本，避免高频 action 过早占满 topK。
3. hard negatives 应集中在 same skeleton wrong assignment、same multiplicity wrong site_symmetry、same SG same nsites wrong orbit token。
4. geometry 阶段不要简单搬运同 skeleton free_params；应训练 orbit free-parameter model 或至少做 orbit-level collision-aware deterministic coordinate selection。
5. 在 Gate 2 WA top100 >=90% 且 retrieved_geometry formula_ok >=98% 前，不跑 full match@5。
