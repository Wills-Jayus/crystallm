# Opentry-3 Wyckoff Model Iterative Experiment Log

开始时间：2026-06-13 UTC

固定工作目录：

```text
/data/users/xsw/autodlmini/model/New_model/opentry_3
```

## 0. 写入边界与实验原则

本轮只允许写入：

```text
/data/users/xsw/autodlmini/model/New_model/opentry_3
```

执行约束：

- 可以读取历史报告、数据、脚本、模型和已有 run/report 产物。
- 不改写、不删除 `opentry_3` 之外的任何文件。
- 所有脚本、数据、runs、reports、cache、checkpoint、tmp、logs 都放到 `opentry_3` 下。
- Python 运行时使用 `PYTHONDONTWRITEBYTECODE=1`，避免在外部源码目录生成新的 `__pycache__`。
- 只用 train split 训练/统计/构建 prior/augmentation；val 只用于模型选择和调参；test 只用于 frozen config 的最终评估。
- 禁止用 test GT CIF、test GT W/A、row_count、StructureMatcher match/rms 训练、排序、过滤或调参。
- 禁止 source CIF fallback、oracle rerank、GT-WA input、CrystaLLM test predictions primary candidate source。
- 主线是 CrystalFormer/WyFormer 风格中间表示：`formula + GT-SG -> Wyckoff skeleton / W-A sequence -> CIF`。
- 不以 K50/K100 candidate stacking、RF ranker、低 LR CIF next-token SFT、固定候选融合为主线。

## 1. E00 - Prior-result audit

时间：2026-06-13 UTC

目的：

- 审计 `opentry`、`opentry_2` 和 `symcif_experiment/Log_GPT` 中与本轮相关的结果。
- 固定本轮 baseline、已知最佳结果、失败路线和 W/A coverage 瓶颈。

### 1.1 CrystaLLM GT-SG reference and operational target

| dataset/reference | metric | CrystaLLM GT-SG | opentry_3 +5pp target |
|---|---:|---:|---:|
| MPTS-52 aggregate | match@1 | 26.64% | 31.64% |
| MPTS-52 aggregate | match@20 | 44.69% | 49.69% |
| MPTS-52 common-subset | match@1 | 26.27% | 31.27% |
| MPTS-52 common-subset | match@5 | 36.58% | 41.58% |
| MP-20 aggregate | match@1 | 72.95% | 77.95% |
| MP-20 aggregate | match@20 | 87.69% | 92.69% |

本轮优先 MPTS-52。要求至少两个 match 指标超过同 GT-SG CrystaLLM baseline 5pp，并记录 RMSE。

### 1.2 Best pure SymCIF / W-A decoder evidence

| source | method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | skeleton@5 | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `opentry` E04 | `v5_e1_geometry_distance_ranking_e08` | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | pure SymCIF rank-1 locally beats old CrystaLLM K1 but not +5pp target. |
| `opentry` E13 | e1 top4 + consensus unique | 28.08% | 34.53% | 36.64% | 0.1029 | 0.1105 | 0.1166 | 6.75% | 30.67% | cached non-oracle ensemble only gives small K5/K20 gains. |
| `opentry` E14 | e1 top3 + consensus unique | 28.08% | 34.46% | 36.64% | 0.1029 | 0.1103 | 0.1166 | 6.73% | 30.64% | similar to E13; not a new generator. |
| `opentry` A4 | exact-cover diversity | 25.49% | 31.57% | 32.48% | 0.1182 | 0.1087 | 0.1084 | 6.83% | 31.44% | diversity alone did not improve match. |

结论：

- 当前 pure SymCIF 的主要上限不是 geometry；是 W/A recall 与复杂结构 coverage。
- 旧 E13/E14 ensemble 属于候选重排/缓存互补，不能作为本轮主线。

### 1.3 opentry_2 model-side attempts

| source | route | key result | conclusion |
|---|---|---|---|
| E04 policy search | E03 step-policy checkpoint, val512 | skeleton top20 39.26%, W/A top20 12.50%, W/A top100 12.89% | step policy search recall very low. |
| E08 broad search | E06 full-train checkpoint, val256, top700 | W/A top100/top200/top700 all 12.11% | correct W/A is usually not merely deeper in beam. |
| E14 SFT full MPTS-52 | conservative CrystaLLM continuation | full test 27.88/37.04/43.70 | ordinary CIF next-token SFT did not reach target. |
| E35 | E14 K50 + validation RF ranker | 32.20/39.51/44.95 | only match@1 passes +5pp; ranker route remains short on K5/K20. |
| E47 | structured-val K50 ranker | 32.07/39.16/45.07 | K50-aware RF did not improve K5 enough. |

结论：

- 低 LR SFT、K50 pool + RF ranker、fixed fusion 都是低收益或不符合本轮主线的路线。
- 本轮必须直接训练 `formula+GT-SG -> canonical skeleton/W-A` generator，并先用 val W/A recall gate 约束是否进入 CIF。

### 1.4 Bottleneck summary for opentry_3

- `rows>=7` / `atom_count>=12` 复杂子集仍是核心瓶颈；旧 rows>=7 match@5 常在 11-14%。
- 旧 pure SymCIF skeleton@5 约 30%，但 W/A@5 只有约 6-7%，说明 element assignment 与 exact composition sequence 是更细瓶颈。
- opentry_2 policy search的 W/A@100 约 12-13%，远低于进入 CIF 评估所需 recall。
- 训练和评估必须先拆成：
  - A：`formula + GT-SG -> skeleton`
  - B：`formula + GT-SG + skeleton -> element/W-A`
  - C：`formula + GT-SG -> canonical W/A`

## 2. E01 - Canonical Wyckoff representation build plan

时间：2026-06-13 UTC

目的：

- 从 `structured_symcif_v4_mpts52/train.jsonl` 和 `val.jsonl` 构建本轮 canonical Wyckoff representation。
- 输出目录：`model/New_model/opentry_3/data/wyckoff_repr_mpts52/`。

字段要求：

- `formula`、`formula_counts`、`sg`、`sg_symbol`
- `canonical_skeleton_key`、`canonical_wa_key`
- `skeleton_sequence`、`assignment_sequence`、`wa_sequence`
- `element_assignment`、`multiplicities`
- `row_count` 只作为 label/analysis 字段，不作为 test-time feature
- `atom_count`、`complex_flag`、`complex_reasons`
- `keys`：sample/material/source/split keys
- `schema_audit` 与 split summaries

设计：

- row canonical：按 `(multiplicity, letter, enumeration, site_symmetry, element)` 稳定排序。
- skeleton 与 assignment 拆开保存。
- 只从 train 统计 priors、vocab、augmentation/repeat limits；val 不进入统计。
- test 暂不构建 representation，直到 frozen final eval 需要读取 evaluator GT。

## 3. E01 - Canonical Wyckoff representation build result

时间：2026-06-13 UTC

新增脚本：

- `model/New_model/opentry_3/scripts/build_wyckoff_repr_mpts52.py`

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_3/scripts/build_wyckoff_repr_mpts52.py \
  --structured-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --out-dir model/New_model/opentry_3/data/wyckoff_repr_mpts52 \
  --splits train,val
```

输出：

- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/train.jsonl`
- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/val.jsonl`
- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/schema_audit_train.jsonl`
- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/schema_audit_val.jsonl`
- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/train_priors.json`
- `model/New_model/opentry_3/data/wyckoff_repr_mpts52/build_summary.json`

结果：

| split | records | unique SG | unique skeleton | unique W/A | rows>=7 | atoms>=12 | row_count mean/median/p90 | atom_count mean/median/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 25,998 | 197 | 4,122 | 25,660 | 6,863 | 19,540 | 5.64 / 4 / 11 | 25.09 / 20 / 48 |
| val | 4,727 | 160 | 1,717 | 4,689 | 2,197 | 4,113 | 7.63 / 6 / 14 | 32.31 / 28 / 56 |

Schema audit:

| split | expanded atom mismatch | row expansion not ok | source skeleton key changed by row canonical | source W/A key changed by row canonical |
|---|---:|---:|---:|---:|
| train | 0 | 0 | 14,798 | 20,180 |
| val | 0 | 0 | 2,892 | 4,242 |

判断：

- canonical representation 构建成功，且所有 train/val 记录的 expanded atom count 与 formula atom_count 一致。
- key change 数量较大是预期内结果：opentry_3 使用新的 deterministic row canonical order，而历史 source key 部分保留原 site order。
- `train_priors.json` 只由 train split 构建；val 未进入统计。
- 下一步训练 A/B/C 轻量 generator，并优先用 val skeleton@k / W/A@k gate 判断，不进入 CIF/full test。

## 4. E02 - Wyckoff A/B/C tiny debug

时间：2026-06-13 UTC

目的：

- 验证 `formula+GT-SG -> skeleton`、`formula+GT-SG+skeleton -> W/A`、`formula+GT-SG -> W/A` 三个轻量模型的训练/评估链路。
- 只跑 tiny debug，不作为模型选择结论。

新增脚本：

- `model/New_model/opentry_3/scripts/train_wyckoff_sequence_models.py`

配置：

| item | value |
|---|---:|
| train records | 128 |
| val records | 16 |
| epochs | 1 |
| skeleton beam / W-A beam | 5 / 5 |
| device | CPU |

结果：

| subset | skeleton@5 | W/A union@5 | unique skeleton mean | unique W/A union mean |
|---|---:|---:|---:|---:|
| full val16 | 6.25% | 6.25% | 0.50 | 0.25 |
| rows>=7 | 0.00% | 0.00% | 0.50 | 0.20 |

判断：

- 脚本功能闭环可用。
- tiny 数据过小，结果只说明 pipeline 能运行。

## 5. E03 - 2k Wyckoff sequence smoke

时间：2026-06-13 UTC

目的：

- 扩大到 2,048 train records，检查 skeleton generator 是否可学。
- 初始 learned B / direct C 组合不使用 train-prior assignment decoder。

输出：

- `model/New_model/opentry_3/reports/e03_wyckoff_seq_smoke_2k_val16/training_summary.json`
- `model/New_model/opentry_3/runs/e03_wyckoff_seq_smoke_2k_val16/best.pt`

结果：

| subset | skeleton@1 | skeleton@5 | W/A AB@5 | W/A C@5 | W/A union@5 | unique skeleton mean | unique W/A union mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| full val16 | 31.25% | 37.50% | 0.00% | 0.00% | 0.00% | 4.19 | 2.25 |
| rows>=7 | 40.00% | 40.00% | 0.00% | 0.00% | 0.00% | 3.70 | 2.10 |

判断：

- A skeleton generator 出现有效学习信号。
- B assignment 和 C direct W/A 都未形成 exact W/A hit，下一步优先修 composition-exact assignment decoder。

## 6. E04 - Train-only prior assignment decoder

时间：2026-06-13 UTC

目的：

- 增加只由 train split 构建的 element@orbit / SG@element@orbit prior。
- 在 predicted skeleton 上做 composition-exact assignment beam，避免用 val/test label 排序。

工程改动：

- `train_wyckoff_sequence_models.py` 增加 `decoder_priors_train_only.json`。
- evaluator 报告 `wa_ab`、`wa_prior`、`wa_c`、`wa_union`。

输出：

- `model/New_model/opentry_3/reports/e04_wyckoff_seq_prior_2k_val16/training_summary.json`
- `model/New_model/opentry_3/runs/e04_wyckoff_seq_prior_2k_val16/decoder_priors_train_only.json`

结果：

| subset | skeleton@1 | skeleton@5 | W/A prior@5 | W/A union@5 | unique W/A prior mean | unique W/A union mean |
|---|---:|---:|---:|---:|---:|---:|
| full val16 | 31.25% | 37.50% | 6.25% | 6.25% | 1.63 | 2.75 |
| rows>=7 | 40.00% | 40.00% | 10.00% | 10.00% | 1.80 | 2.70 |

判断：

- train-only prior decoder 恢复了少量 exact W/A recall。
- 但 W/A@5/50 仍低于 opentry_2 E04/E07 policy search 的 W/A@100 约 12-13% 基线，不能进入 CIF。

## 7. E05 - 8k Wyckoff sequence smoke

时间：2026-06-13 UTC

目的：

- 扩大到 8,192 train records，检查 skeleton/W-A 是否随数据规模提高。
- 仍只做小 validation gate，不触碰 test。

输出：

- `model/New_model/opentry_3/reports/e05_wyckoff_seq_prior_8k_val32/training_summary.json`
- `model/New_model/opentry_3/runs/e05_wyckoff_seq_prior_8k_val32/best.pt`

训练统计：

| item | value |
|---|---:|
| train records | 8,192 |
| val records | 32 |
| skeleton states | 41,023 |
| assignment states | 32,831 |
| direct W/A states | 41,023 |
| vocab orbits / pairs / SGs | 849 / 7,297 / 171 |
| epochs | 4 |
| elapsed | 273.8 s |

结果：

| subset | skeleton@1 | skeleton@5 | W/A prior@5 | W/A C@5 | W/A union@5 | unique skeleton mean | unique W/A union mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| full val32 | 34.38% | 65.62% | 3.12% | 6.25% | 6.25% | 4.75 | 3.47 |
| rows>=7 | 38.89% | 66.67% | 0.00% | 5.56% | 5.56% | 4.56 | 3.78 |
| atoms>=12 | 36.67% | 63.33% | 3.33% | 6.67% | 6.67% | 4.73 | 3.37 |

判断：

- Skeleton generation 随训练规模显著改善，val32 skeleton@5 达到 65.62%。
- W/A exact recall 未随 skeleton 提升同步提高，full W/A union@5 仍只有 6.25%。
- 当前瓶颈已经从 A skeleton 转向 B/C assignment sequence。

## 8. E06 - K50 prior-only decoder diagnostic aborted

时间：2026-06-13 UTC

目的：

- 检查 8k skeleton model + train-only prior assignment 在 K50 下是否能提高 W/A@50。
- 禁用 learned AB 与 direct C，只保留 prior assignment，避免 ranker/DPO/CIF。

尝试：

- `e06_wyckoff_seq_prior_8k_val64_k50`: val64, skeleton beam 50, W/A beam 50，因非向量化 autoregressive skeleton beam 太慢，停止。
- `e06_wyckoff_seq_prior_8k_val16_k50`: val16 同配置，仍过慢，停止。

判断：

- 当前 Python per-beam 神经解码器不适合 K50/K100 gate，需要 vectorized beam search 或 DP/skeleton candidate generator。
- 已有 K5 gate 已显示 W/A recall 低于旧基线，因此不应进入 CIF rendering 或 full test。

## 9. Current gate decision

时间：2026-06-13 UTC

Val gate status:

| gate | status | evidence |
|---|---|---|
| skeleton@50 healthy | partial / blocked | K5 skeleton 已到 65.62%，但 K50 非向量化评估过慢。 |
| W/A@50 clearly exceeds old baseline | fail | best measured W/A union@5=6.25%；K50 blocked，且未见超过旧 W/A@100=12-13% 的迹象。 |
| rows>=7 W/A improves | fail | rows>=7 W/A union@5=5.56%，仍很低。 |
| composition exact | partial | generated nonempty candidates composition exact，但 candidate_nonempty 只有 75%。 |
| enter CIF rendering | no | W/A gate 未通过。 |
| full test | no | config 未由 val gate 冻结，不能触碰 test。 |

## 10. E07 - Fixed-skeleton exact-cover DP assignment diagnostic

时间：2026-06-13 UTC

目的：

- 不重新训练，直接诊断 assignment decoder 的上限。
- 在固定 skeleton 下，用 train-only element@orbit / SG@element@orbit prior 做 exact-cover DP top-K assignment。
- 两个分支：
  - GT skeleton diagnostic：只用于 val 上定位 assignment capacity，不作为正式候选输入。
  - predicted skeleton diagnostic：使用 E05 已生成的 skeleton candidates，不使用 val/test label 排序。

新增脚本：

- `model/New_model/opentry_3/scripts/diagnose_wyckoff_assignment_dp.py`

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_3/scripts/diagnose_wyckoff_assignment_dp.py \
  --data-dir model/New_model/opentry_3/data/wyckoff_repr_mpts52 \
  --vocab-json model/New_model/opentry_3/runs/e05_wyckoff_seq_prior_8k_val32/vocab.json \
  --priors-json model/New_model/opentry_3/runs/e05_wyckoff_seq_prior_8k_val32/decoder_priors_train_only.json \
  --predicted-candidates model/New_model/opentry_3/reports/e05_wyckoff_seq_prior_8k_val32/val_candidates.jsonl \
  --out-dir model/New_model/opentry_3/reports/e07_assignment_dp_diagnostic_val32 \
  --max-val-records 32 \
  --top-k 100 \
  --state-beam 100 \
  --max-skeletons 5 \
  --per-skeleton 50
```

输出：

- `model/New_model/opentry_3/reports/e07_assignment_dp_diagnostic_val32/assignment_dp_summary.json`
- `model/New_model/opentry_3/reports/e07_assignment_dp_diagnostic_val32/gt_skeleton_assignment_per_sample.jsonl`
- `model/New_model/opentry_3/reports/e07_assignment_dp_diagnostic_val32/predicted_skeleton_assignment_per_sample.jsonl`

结果：

| branch | subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | W/A@100 | unique mean |
|---|---|---:|---:|---:|---:|---:|---:|
| GT skeleton DP diagnostic | full val32 | 37.50% | 56.25% | 62.50% | 68.75% | 68.75% | 51.84 |
| GT skeleton DP diagnostic | rows>=7 | 22.22% | 27.78% | 33.33% | 44.44% | 44.44% | 87.78 |
| predicted skeleton DP | full val32 | 21.88% | 31.25% | 34.38% | 43.75% | 50.00% | 71.84 |
| predicted skeleton DP | rows>=7 | 11.11% | 11.11% | 11.11% | 22.22% | 27.78% | 94.44 |

判断：

- E07 证明此前 E05 的 W/A failure 主要是 decoder/beam 实现问题，而不是 representation 上限。
- 固定 GT skeleton 时，train-only prior DP 可以把 full val32 W/A@50 推到 68.75%。
- 使用 predicted skeleton candidates 时，full val32 W/A@50=43.75%、W/A@100=50.00%，明显超过 opentry_2 policy-search W/A@100 约 12-13%。
- rows>=7 仍显著较弱，复杂子集需要继续强化。

## 11. E08 - E05 skeleton candidate generation expanded to val128

时间：2026-06-13 UTC

目的：

- 扩大 E05 skeleton model 的 validation evidence。
- 不重新训练；加载 E05 checkpoint，对 val 前 128 条生成 top-5 skeleton candidates。

新增脚本：

- `model/New_model/opentry_3/scripts/generate_wyckoff_skeleton_candidates.py`

输出：

- `model/New_model/opentry_3/reports/e08_skeleton_candidates_e05_val128/skeleton_candidates.jsonl`
- `model/New_model/opentry_3/reports/e08_skeleton_candidates_e05_val128/skeleton_per_sample.jsonl`
- `model/New_model/opentry_3/reports/e08_skeleton_candidates_e05_val128/skeleton_summary.json`

结果：

| subset | samples | candidate nonempty | skeleton@1 | skeleton@5 | unique skeleton mean |
|---|---:|---:|---:|---:|---:|
| full val128 | 128 | 99.22% | 39.06% | 64.84% | 4.66 |
| rows>=7 | 60 | 98.33% | 48.33% | 65.00% | 4.32 |
| atoms>=12 | 116 | 99.14% | 38.79% | 61.21% | 4.63 |

判断：

- Skeleton generator 的 val signal 在 128 条上基本稳定。
- top-5 skeleton recall 接近 65%，但非向量化 beam 生成 128 条仍耗时约 205 秒；后续 K50/K100 skeleton beam 需要优化。

## 12. E09 - Predicted-skeleton DP assignment expanded to val128

时间：2026-06-13 UTC

目的：

- 将 E07 的 DP assignment 诊断扩到 val128。
- 使用 E08 的 predicted skeleton candidates 和 train-only priors。
- 不使用 StructureMatcher，不触碰 test。

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_3/scripts/diagnose_wyckoff_assignment_dp.py \
  --data-dir model/New_model/opentry_3/data/wyckoff_repr_mpts52 \
  --vocab-json model/New_model/opentry_3/runs/e05_wyckoff_seq_prior_8k_val32/vocab.json \
  --priors-json model/New_model/opentry_3/runs/e05_wyckoff_seq_prior_8k_val32/decoder_priors_train_only.json \
  --predicted-candidates model/New_model/opentry_3/reports/e08_skeleton_candidates_e05_val128/skeleton_candidates.jsonl \
  --out-dir model/New_model/opentry_3/reports/e09_assignment_dp_diagnostic_val128 \
  --max-val-records 128 \
  --top-k 100 \
  --state-beam 100 \
  --max-skeletons 5 \
  --per-skeleton 50
```

输出：

- `model/New_model/opentry_3/reports/e09_assignment_dp_diagnostic_val128/assignment_dp_summary.json`
- `model/New_model/opentry_3/reports/e09_assignment_dp_diagnostic_val128/gt_skeleton_assignment_per_sample.jsonl`
- `model/New_model/opentry_3/reports/e09_assignment_dp_diagnostic_val128/predicted_skeleton_assignment_per_sample.jsonl`

结果：

| branch | subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | W/A@100 | unique mean |
|---|---|---:|---:|---:|---:|---:|---:|
| GT skeleton DP diagnostic | full val128 | 46.09% | 59.38% | 64.84% | 72.66% | 75.00% | 48.85 |
| GT skeleton DP diagnostic | rows>=7 | 25.00% | 26.67% | 33.33% | 45.00% | 50.00% | 89.90 |
| predicted skeleton DP | full val128 | 24.22% | 33.59% | 35.16% | 42.19% | 45.31% | 67.80 |
| predicted skeleton DP | rows>=7 | 16.67% | 16.67% | 16.67% | 23.33% | 25.00% | 90.83 |
| predicted skeleton DP | atoms>=12 | 23.28% | 29.31% | 30.17% | 37.93% | 39.66% | 71.68 |

判断：

- E09 使 symbolic W/A gate 从失败转为局部通过：full val128 predicted-skeleton `W/A@50=42.19%` / `W/A@100=45.31%`，明显超过旧 opentry_2 W/A@100 约 12-13%。
- rows>=7 也从旧约低个位/十几百分点提升到 `W/A@50=23.33%`，但仍是主要弱点。
- 该结果仍是 symbolic recall，不是 CIF match。按实验边界，下一步可以进入 validation CIF rendering，但不能 full test。

## 13. Updated val gate decision after E09

时间：2026-06-13 UTC

| gate | status | evidence |
|---|---|---|
| skeleton recall | pass for top-5 smoke | val128 skeleton@5=64.84%，rows>=7=65.00%。 |
| W/A@50 exceeds old baseline | pass | val128 predicted-skeleton DP W/A@50=42.19%，W/A@100=45.31%；old policy W/A@100≈12-13%。 |
| rows>=7 improves | partial pass | rows>=7 W/A@50=23.33%，仍远低于 full，但比旧路线明显更高。 |
| composition exact | pass for generated candidates | predicted DP candidate_nonempty/eligible rate 99.22%，assignment is composition exact by construction. |
| enter CIF rendering | yes, validation only | 可进入 val CIF rendering smoke，复用 e07/e08 geometry/renderer。 |
| full test | no | 尚未做 val StructureMatcher，不可冻结 full-test config。 |

## 14. E10-E15 - E09 W/A candidates to validation CIF, collision-aware deterministic renderer

时间：2026-06-13 UTC

目的：

- 将 E09 predicted-skeleton DP W/A candidates 转成 SymCIF-v4 renderer input。
- 用 validation only 的 CIF smoke 检查 W/A recall 能否转化为 StructureMatcher match。
- 不触碰 test；renderer geometry index/metadata 只从 train 读取。

新增脚本：

- `model/New_model/opentry_3/scripts/build_renderer_predictions_from_dp.py`
- `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs.py`
- `model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py`

E10 conversion:

| item | value |
|---|---:|
| input samples | 128 |
| samples with candidates | 127 |
| output candidates | 5,111 |
| orbit metadata source | train repr only |
| missing orbit ids | 0 |

初始 render 结果：

| renderer | readable | formula/atom exact | SG ok |
|---|---:|---:|---:|
| old retrieved geometry top50 overall | 88.50% | 85.85% | 79.87% |
| old retrieved geometry rank1 | 93.70% | 90.55% | 87.40% |

问题定位：

- DP 允许同一个全固定 orbit 重复分配给多个元素，例如同一 `2c` 同时给 Cs/Gd。
- renderer 会将重复固定 orbit 展开到完全相同坐标，pymatgen 解析失败或 atom_count 变少。
- 这是 decoder mask 缺失，不是 StructureMatcher 训练信号。

修正：

- `build_renderer_predictions_from_dp.py` 默认过滤 duplicate fixed orbit candidates。
- `opentry_render_wyckoff_cifs.py` 对无 train geometry reference 的 free-param rows 使用 collision-aware deterministic parameter search，避免重复 free orbit 之间坐标碰撞。

E12/E14 closure 结果：

| item | value |
|---|---:|
| fixed-orbit dropped candidates | 626 |
| post-filter samples with candidates | 123 / 128 |
| rendered CIFs | 4,631 |
| readable / formula / atom exact overall | 100% / 100% / 100% |
| readable / formula / atom exact rank1 | 100% / 100% / 100% |
| SG ok overall / rank1 | 91.69% / 92.68% |

E15 validation StructureMatcher:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | skeleton@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 8.59% | 17.19% | 19.53% | 21.09% | 0.1070 | 0.1106 | 0.1533 | 0.1643 | 42.19% | 46.09% | 36.18 |
| rows>=7 | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | NA | NA | 23.33% | 46.67% | 49.17 |
| atoms>=12 | 5.17% | 10.34% | 12.07% | 12.93% | 0.1145 | 0.1813 | 0.2212 | 0.2139 | 37.93% | 41.38% | 38.28 |

判断：

- symbolic gate 确实能产生合法 CIF，但 simple retrieved/collision-aware deterministic geometry 无法把复杂 W/A 命中转为 match。
- rows>=7 W/A@50=23.33% 但 match@50=0，说明复杂样本主要受 geometry/free-param/lattice 影响。

## 15. E16-E19 - e08 row-conditioned geometry renderer smoke

时间：2026-06-13 UTC

目的：

- 将 opentry_3 W/A candidates 接入旧 e07/e08 row-conditioned geometry path。
- `GeometryIndex` 只由 train split 构建；val 只作 evaluator。
- 对比 e08 geometry 是否改善 validation StructureMatcher。

新增脚本：

- `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`

E16 old e08 render:

| item | value |
|---|---:|
| rendered rows | 623 |
| samples with rendered candidates | 123 / 128 |
| overall readable/formula/atom exact | 87.32% |
| rank1 readable/formula/atom exact | 94.31% |

E17 old e08 validation StructureMatcher:

| subset | match@1 | match@5 | match@20/50 | RMSE@1 | RMSE@5 | RMSE@20/50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 17.19% | 22.66% | 25.00% | 0.1345 | 0.1015 | 0.1053 | 52.34% | 3.45 |
| rows>=7 | 6.67% | 6.67% | 6.67% | 0.0964 | 0.0964 | 0.0964 | 45.00% | 1.52 |

修正：

- e08 wrapper 改成多 geometry-rank 尝试。
- 默认跳过非 composition-exact CIF，再继续补候选。

E18 comp-filter e08 closure:

| item | value |
|---|---:|
| rendered rows | 1,870 |
| samples with rendered candidates | 120 / 128 |
| overall readable/formula/atom exact | 100% |
| rank1 readable/formula/atom exact | 100% |
| SG ok overall / rank1 | 95.45% / 94.17% |

E19 comp-filter e08 validation StructureMatcher:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | skeleton@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 17.19% | 24.22% | 29.69% | 30.47% | 0.1345 | 0.1095 | 0.1274 | 0.1328 | 52.34% | 55.47% | 3.23 |
| rows>=7 | 6.67% | 6.67% | 8.33% | 8.33% | 0.0964 | 0.0964 | 0.0999 | 0.0999 | 45.00% | 50.00% | 1.30 |
| atoms>=12 | 12.93% | 17.24% | 22.41% | 23.28% | 0.0989 | 0.1210 | 0.1539 | 0.1609 | 49.14% | 51.72% | 2.82 |

判断：

- e08 row-conditioned geometry 明显优于 simple collision-aware deterministic renderer：full match@50 从 21.09% 提升到 30.47%，rows>=7 从 0 提升到 8.33%。
- 但 e08 comp-filter 后 unique W/A@50 很低，平均只有 3.23；大量 top-k budget 被同一 W/A 的 geometry variants 占用。
- 当前还不能冻结 full test config。下一步应合并 e08 geometry 优势和 E14 collision-aware renderer 的 W/A 多样性：优先 e08 composition-exact candidates，再用 collision-aware deterministic fallback 补足不同 W/A。

## 16. E20-E26 - canonical W/A key correction and trusted val128 e08 result

时间：2026-06-13 UTC

问题：

- E09/E12 的 DP `wa_key` 保留了 repeated identical orbit 上的元素排列顺序。
- canonical representation 的 W/A key 会按 `(multiplicity, letter, enumeration, site_symmetry, element, orbit_id)` 排序。
- 因此旧 E09/E12 的 `unique W/A` 和 W/A@k 解释不一致：raw permutations 既会高估 unique，也会低估 target hit。

修正：

- `build_renderer_predictions_from_dp.py` 现在从 candidate rows 重新计算 canonical skeleton/W-A key。
- 用 canonical W/A key 去重。
- 保留 `raw_dp_wa_key` / `raw_dp_skeleton_key` 仅作审计字段。

E24 canonical conversion from E09:

| item | value |
|---|---:|
| samples | 128 |
| samples with candidates | 123 |
| canonical output candidates | 590 |
| dropped duplicate fixed orbit candidates | 1,210 |
| missing orbit ids | 0 |

canonical W/A recall:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A mean |
|---|---:|---:|---:|---:|---:|
| full val128 | 38.28% | 53.12% | 57.03% | 57.03% | 4.61 |
| rows>=7 | 41.67% | 48.33% | 50.00% | 50.00% | 2.90 |
| atoms>=12 | 38.79% | 50.86% | 52.59% | 52.59% | 4.16 |

E26 canonical E09 + e08 comp-filter validation match:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | skeleton@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 17.19% | 25.00% | 30.47% | 31.25% | 0.1345 | 0.1121 | 0.1238 | 0.1286 | 57.03% | 59.38% | 4.15 |
| rows>=7 | 6.67% | 8.33% | 10.00% | 10.00% | 0.0964 | 0.1158 | 0.1156 | 0.1156 | 50.00% | 55.00% | 2.33 |
| atoms>=12 | 12.93% | 18.10% | 23.28% | 24.14% | 0.0989 | 0.1245 | 0.1554 | 0.1621 | 52.59% | 55.17% | 3.67 |

判断：

- 修正后的 canonical W/A@50 比旧 raw 统计更可信：full 57.03%、rows>=7 50.00%。
- CIF match 仍远低于 W/A recall，说明 geometry/free-param 是主要转化瓶颈之一。
- 但 unique W/A 很低，说明 skeleton diversity 仍不足。

## 17. E27-E33 - train-only skeleton prior augmentation and improved val128 CIF smoke

时间：2026-06-13 UTC

目的：

- E05 model beam 只有约 5 个 skeleton，`skeleton@20` 与 `skeleton@5` 一样，限制 W/A diversity。
- 加入 train-only skeleton prior augmentation：同 SG + atom_count exact，按 formula distance 排序。
- 不使用 row_count input，不使用 val/test label 排序。

新增脚本：

- `model/New_model/opentry_3/scripts/augment_skeleton_candidates_train_prior.py`

E27 skeleton augmentation:

| subset | skeleton@1 | skeleton@5 | skeleton@20 | skeleton@50 | unique skeleton mean |
|---|---:|---:|---:|---:|---:|
| full val128 | 39.84% | 65.62% | 87.50% | 87.50% | 7.45 |
| rows>=7 | 50.00% | 66.67% | 85.00% | 85.00% | 7.38 |
| atoms>=12 | 39.66% | 62.07% | 86.21% | 86.21% | 7.50 |

E28 raw DP over augmented skeletons:

- runtime: 208.8 s
- raw `pred_dp@50`: full 48.44%, rows>=7 16.67%
- raw key is not the final metric due canonical ordering issue.

E29 canonical conversion from E28:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A mean |
|---|---:|---:|---:|---:|---:|
| full val128 | 32.03% | 57.03% | 75.00% | 75.00% | 9.16 |
| rows>=7 | 33.33% | 51.67% | 61.67% | 61.67% | 6.38 |
| atoms>=12 | 31.90% | 59.48% | 72.41% | 72.41% | 8.66 |

E31 augmented skeleton + canonical W/A + e08 comp-filter validation match (`geometry_ranks_per_wa=5`):

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | skeleton@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full val128 | 21.88% | 32.03% | 37.50% | 39.06% | 0.1612 | 0.1682 | 0.1327 | 0.1165 | 75.00% | 80.47% | 8.63 |
| rows>=7 | 10.00% | 11.67% | 11.67% | 13.33% | 0.1946 | 0.1731 | 0.1660 | 0.1614 | 61.67% | 73.33% | 5.67 |
| atoms>=12 | 18.97% | 29.31% | 31.03% | 32.76% | 0.1607 | 0.1607 | 0.1466 | 0.1371 | 72.41% | 78.45% | 8.09 |

E33 same W/A candidates with `geometry_ranks_per_wa=10`:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@50 |
|---|---:|---:|---:|---:|---:|
| full val128 | 21.88% | 32.03% | 37.50% | 39.84% | 0.1215 |
| rows>=7 | 10.00% | 11.67% | 11.67% | 13.33% | 0.1614 |
| atoms>=12 | 18.97% | 29.31% | 31.03% | 33.62% | 0.1442 |

判断：

- train-only skeleton prior augmentation 是本轮最大有效改动：full W/A@50 从 57.03% 提到 75.00%，rows>=7 从 50.00% 提到 61.67%。
- CIF match 也随之提升：full match@50 从 31.25% 到 39.06/39.84%，rows>=7 从 10.00% 到 13.33%。
- 增加 geometry ranks 主要只改善 K50，不改善 K20；e08 source diversity 基本饱和。
- 当前 val128 `match@20=37.50%`、`match@50=39.84%` 仍不足以冻结 full test。下一步应训练/蒸馏 skeleton generator，让 train-prior augmentation 的高 recall 变成模型侧能力，并继续改进 rows>=7 geometry。

## 18. E34-E35 - count-aware A-stage skeleton generator

时间：2026-06-13 UTC

目的：

- 把 E27 train-only same-SG/atom-count skeleton prior 的收益转成模型侧 A-stage skeleton generator。
- 修正旧 skeleton net 的输入缺口：旧 `formula_vec` 只保留元素比例，绝对 atom_count 主要只通过 legal mask 间接出现；E27 的收益说明 atom_count exact 是关键。
- 新模型输入为 formula fraction + absolute element counts + SG + decoding state；`row_count` 仍不作为输入，只用于训练权重和分组报告。

新增/修改脚本：

- `model/New_model/opentry_3/scripts/train_count_aware_skeleton_model.py`
- 追加 eval-only 模式和 batched beam decoding。

训练配置：

| item | value |
|---|---:|
| train records | 25,998 |
| train skeleton states | 172,594 |
| epochs | 5 |
| hidden | 256 |
| device | CPU |
| complex weight | 2.5 |
| count scale | 64 |

训练结果：

| epoch | loss |
|---:|---:|
| 1 | 2.7457 |
| 2 | 0.7055 |
| 3 | 0.5027 |
| 4 | 0.4281 |
| 5 | 0.3905 |

工程记录：

- 初始逐 beam forward 的 val128 `beam=50/branch=12` 评估过慢，已中断。
- 将同一步多个 beams 合并成 batch forward 后，val128 `beam=50/branch=12` 评估耗时约 12.3 秒。

E35e val128 skeleton result：

| subset | skeleton@1 | skeleton@5 | skeleton@20 | skeleton@50 | unique skeleton@50 |
|---|---:|---:|---:|---:|---:|
| full | 45.31% | 70.31% | 81.25% | 87.50% | 45.23 |
| rows>=7 | 50.00% | 68.33% | 81.67% | 85.00% | 42.65 |
| atoms>=12 | 42.24% | 67.24% | 79.31% | 86.21% | 44.74 |

对比：

| source | full skeleton@50 | rows>=7 skeleton@50 |
|---|---:|---:|
| E08 old model-only | 64.84% | 65.00% |
| E27 model + train prior | 87.50% | 85.00% |
| E35e count-aware model-only | 87.50% | 85.00% |

判断：

- count-aware A-stage 模型基本复现了 E27 train-prior skeleton recall，但不再需要把 train-prior skeleton 作为候选补丁。
- 这是本轮第一个明确的 model-side A-stage 改进。
- 下一步接 assignment DP，确认高 skeleton recall 能否转为 canonical W/A recall。

## 19. E36-E38 - count-aware skeleton + budgeted assignment DP canonical W/A gate

时间：2026-06-13 UTC

目的：

- 使用 E35e model-only skeleton candidates。
- 使用 train-only assignment priors 和 exact-cover DP 生成 W/A。
- 不使用 StructureMatcher label，不使用 test，不使用 row_count 排序。

工程记录：

- E36 full `50 skeleton x 50 assignment` DP 过慢，已中断。
- 给 `diagnose_wyckoff_assignment_dp.py` 增加：
  - `--max-active-paths`：每步保留 top active partial assignment paths。
  - `--skip-gt-skeleton`：跳过 GT-skeleton diagnostic，只评估 predicted skeleton。
- E36e 使用 `max_skeletons=20`、`per_skeleton=20`、`state_beam=50`、`max_active_paths=1000`，耗时 87.3 秒。
- E37 canonical conversion 过滤 duplicate fixed-orbit candidates。

E37 conversion：

| item | value |
|---|---:|
| samples | 128 |
| samples with candidates | 126 |
| input DP candidates seen | 10,559 |
| output canonical candidates | 1,406 |
| dropped duplicate fixed-orbit candidates | 2,512 |
| missing orbit ids | 0 |

E38 canonical W/A result：

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 | skeleton@50 |
|---|---:|---:|---:|---:|---:|---:|
| full | 45.31% | 66.41% | 73.44% | 73.44% | 10.98 | 79.69% |
| rows>=7 | 48.33% | 66.67% | 66.67% | 66.67% | 5.52 | 78.33% |
| atoms>=12 | 43.10% | 63.79% | 70.69% | 70.69% | 10.19 | 77.59% |

判断：

- E35/E38 是 model-side skeleton 生成路线的有效推进：早排 W/A 明显强于 E29，rows>=7 W/A@50 也从 E29 的 61.67% 提升到 66.67%。
- full W/A@50 低于 E29 的 75.00%，说明 reduced/budgeted DP 和 duplicate fixed-orbit 过滤仍损失一部分 full recall。
- 由于 W/A gate 已明显超过旧 opentry_2 policy-search 约 12-13% W/A@100，可进入 validation CIF smoke。

## 20. E39-E40 - count-aware W/A candidates with e08 W/A-diverse rendering

时间：2026-06-13 UTC

目的：

- 将 E37 canonical W/A candidates 接入 e08 train-only row-conditioned geometry renderer。
- 使用 W/A-diverse geometry plan：先为每个 W/A 尝试 geometry rank0，保持 W/A 多样性。
- 只跑 val128 StructureMatcher，不跑 test。

E39 render closure：

| item | value |
|---|---:|
| samples with prediction rows | 128 |
| samples with rendered candidates | 125 |
| rendered rows | 5,106 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG ok overall / rank1 | 93.63% / 90.40% |

E40 validation StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 27.34% | 35.94% | 39.06% | 0.0621 | 0.1022 | 0.1295 | 0.1342 | 73.44% |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 0.0964 | 0.1039 | 0.1053 | 0.1053 | 66.67% |
| atoms>=12 | 13.79% | 21.55% | 29.31% | 32.76% | 0.0716 | 0.1016 | 0.1494 | 0.1617 | 70.69% |

对比 E33：

| metric | E33 augmented prior + e08 | E40 count-aware model + e08 |
|---|---:|---:|
| full match@1 | 21.88% | 17.97% |
| full match@5 | 32.03% | 27.34% |
| full match@20 | 37.50% | 35.94% |
| full match@50 | 39.84% | 39.06% |
| rows>=7 match@50 | 13.33% | 11.67% |

判断：

- 虽然 E38 W/A early ranks 更强，E40 CIF match 没有超过 E33。
- W/A@1=47.66% 但 match@1=17.97%，说明 geometry/free-param/lattice 转化仍是主要瓶颈。
- 当前不冻结 full test。

## 21. E41-E42 - geometry-interleaved renderer diagnostic

时间：2026-06-13 UTC

目的：

- 检查是否因为只尝试 geometry rank0 导致前排 W/A 无法转化为 match。
- 新增 `--geometry-plan-mode geometry_interleave`：对前排 W/A 优先尝试多个 e08 geometry ranks。
- 仍只在 validation 上诊断。

E41 render：

| item | value |
|---|---:|
| rendered rows | 5,106 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG ok overall / rank1 | 92.36% / 89.60% |

E42 validation StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 21.88% | 27.34% | 36.72% | 67.97% | 4.65 |
| rows>=7 | 6.67% | 8.33% | 10.00% | 11.67% | 66.67% | 4.17 |
| atoms>=12 | 12.93% | 17.24% | 22.41% | 30.17% | 65.52% | 4.63 |

判断：

- Geometry interleave 降低了 W/A diversity，并使 full match@5/20/50 全面低于 E40。
- 当前不是简单增加 geometry ranks 的问题；需要更好的 geometry/free-param proposal 或 learned geometry head，同时保持 W/A 多样性。
- full test 仍未达冻结标准，不能运行。

## 22. E43-E47 - fixed-orbit duplicate mask moved into assignment DP

时间：2026-06-13 UTC

目的：

- E37 conversion 显示 10,559 个 raw DP candidates 中有 2,512 个 duplicate fixed-orbit candidates 被后处理丢弃。
- 将 fixed-orbit duplicate 约束前移到 assignment DP，避免无效 skeleton/candidate 抢占 DP budget。
- `row_count` 仍不作为输入；fixed-orbit metadata 只从 train representation 构建。

脚本改动：

- `diagnose_wyckoff_assignment_dp.py`
  - 新增 `--train-repr`
  - 新增 `--allow-duplicate-fixed-skeletons`
  - 默认构建 train-only fixed orbit set，并跳过 duplicate fixed-orbit skeleton。
  - `max_skeletons` 现在统计有效 skeleton，而不是原始前 N 个 skeleton。

E43 DP 配置：

| item | value |
|---|---:|
| skeleton source | E35e count-aware model-only skeleton |
| max skeletons | 20 valid skeletons |
| per skeleton | 20 |
| state beam | 50 |
| max active paths | 1000 |
| fixed orbits from train | 447 |
| elapsed | 89.0 s |

E44 canonical conversion：

| item | E37 old | E44 fixed-mask |
|---|---:|---:|
| input candidates seen | 10,559 | 8,941 |
| dropped duplicate fixed-orbit candidates | 2,512 | 0 |
| output canonical candidates | 1,406 | 1,612 |
| samples with candidates | 126 | 126 |

E45 canonical W/A result:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 | skeleton@50 |
|---|---:|---:|---:|---:|---:|---:|
| full | 45.31% | 66.41% | 74.22% | 74.22% | 12.59 | 80.47% |
| rows>=7 | 48.33% | 66.67% | 66.67% | 66.67% | 6.55 | 78.33% |
| atoms>=12 | 43.10% | 63.79% | 71.55% | 71.55% | 11.66 | 78.45% |

对比 E38：

| metric | E38 | E45 |
|---|---:|---:|
| full W/A@20/50 | 73.44% / 73.44% | 74.22% / 74.22% |
| full unique W/A@50 | 10.98 | 12.59 |
| rows>=7 W/A@50 | 66.67% | 66.67% |
| rows>=7 unique W/A@50 | 5.52 | 6.55 |

判断：

- 前移 fixed-orbit duplicate mask 是正确工程改动：后处理不再丢弃 duplicate fixed candidates，canonical candidate 数和 unique W/A 均提升。
- 命中率提升有限，尤其 rows>=7 W/A 命中未变。

E46 render：

| item | value |
|---|---:|
| samples with rendered candidates | 126 / 128 |
| rendered rows | 5,355 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG ok overall / rank1 | 93.76% / 91.27% |

E47 validation StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 27.34% | 35.94% | 39.06% | 0.0621 | 0.1027 | 0.1324 | 0.1381 | 74.22% |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 0.0964 | 0.1039 | 0.1053 | 0.1053 | 66.67% |
| atoms>=12 | 13.79% | 21.55% | 29.31% | 32.76% | 0.0716 | 0.1016 | 0.1495 | 0.1632 | 71.55% |

判断：

- Fixed-mask DP 改善 W/A candidate health，但没有改善 CIF match。
- 当前 renderer 的 W/A-diverse plan 在 top50 budget 下基本只使用 geometry rank0；E42 的 full interleave 又过度牺牲 W/A diversity。
- 下一步应尝试 hybrid geometry plan：给前排少量 W/A 多个 geometry ranks，同时保留后排 W/A diversity；若仍无明显提升，则必须转向 learned geometry/free-param proposal。

## 23. E48-E51 - hybrid geometry plan diagnostics

时间：2026-06-13 UTC

目的：

- 在不使用 StructureMatcher label 排序的前提下，测试更温和的 geometry plan。
- `hybrid_top_wa`：给前排少量 W/A 多个 e08 geometry ranks，同时保留其余 W/A 的 rank0 多样性。
- 仍只在 val128 诊断。

脚本改动：

- `opentry_render_wyckoff_cifs_e07e08.py`
  - 新增 `--geometry-plan-mode hybrid_top_wa`
  - 新增 `--hybrid-geometry-wa`
  - 新增 `--hybrid-geometry-ranks`

E48/E49 hybrid5x3:

| subset | match@1 | match@5 | match@20 | match@50 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 24.22% | 36.72% | 39.06% | 55.47% | 72.66% | 74.22% | 11.23 |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 58.33% | 66.67% | 66.67% | 5.57 |

E50/E51 hybrid3x2:

| subset | match@1 | match@5 | match@20 | match@50 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 23.44% | 35.94% | 39.06% | 64.84% | 74.22% | 74.22% | 11.52 |
| rows>=7 | 6.67% | 8.33% | 11.67% | 11.67% | 63.33% | 66.67% | 66.67% | 5.57 |

对比 W/A-diverse E47:

| run | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---:|---:|---:|---:|
| E47 W/A-diverse | 27.34% | 35.94% | 39.06% | 11.67% |
| E49 hybrid5x3 | 24.22% | 36.72% | 39.06% | 11.67% |
| E51 hybrid3x2 | 23.44% | 35.94% | 39.06% | 11.67% |

判断：

- Hybrid geometry plan 没有改善 K50，也牺牲 K5。
- E49 只把 full match@20 从 35.94% 小幅推到 36.72%，不足以改变路线判断。
- 简单 geometry rank 排序/调参已基本排除。

## 24. E52 - W/A-to-match conversion diagnostic

时间：2026-06-13 UTC

目的：

- 定量确认当前主要瓶颈是否为 W/A -> geometry match 转化。
- 只读已有 val evaluator per-sample metrics，不引入新训练或 test。

结果摘要：

| run/subset | K | W/A hit | match | match / W-A hit |
|---|---:|---:|---:|---:|
| E33 full | 50 | 75.00% | 39.84% | 53.1% |
| E47 full | 50 | 74.22% | 39.06% | 52.6% |
| E47 rows>=7 | 50 | 66.67% | 11.67% | 17.5% |
| E47 atoms>=12 | 50 | 71.55% | 32.76% | 45.8% |
| E49 full | 20 | 72.66% | 36.72% | 50.5% |
| E51 full | 50 | 74.22% | 39.06% | 52.6% |

判断：

- Full subset 上 W/A hit 到 StructureMatcher match 的转化率约 50%。
- rows>=7 只有约 17.5%，是当前最大转化瓶颈。
- 继续调 skeleton/W-A 排序仍可能改善 symbolic recall，但很难单独把 val CIF match 推到冻结线。
- 下一步应训练或构造 learned geometry/free-param proposal，尤其针对 rows>=7；不能继续靠简单 e08 geometry-rank plan。

## 25. E53-E54 - train-only W/A/skeleton-priority geometry source diagnostic

时间：2026-06-13 UTC

目的：

- 检查 E52 暴露的 W/A->geometry 转化瓶颈是否来自 e08 retrieval source pruning。
- 在 `opentry_3` renderer 内新增 train-only source selector：
  - exact canonical W/A source；
  - exact skeleton source；
  - same free-signature source；
  - same SG+row_count source；
  - fallback 到原 `row_conditioned_knn`。
- 该 selector 只读 train records，不使用 val/test StructureMatcher label。

脚本改动：

- `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`
  - 新增 `--geometry-source-strategy row_conditioned_knn|wa_skeleton_priority`
  - 默认仍为 `row_conditioned_knn`，保持 E46/E48/E50 可复现。

E53 render：

| item | value |
|---|---:|
| source strategy | `wa_skeleton_priority` |
| samples with candidates | 126 / 128 |
| rendered rows | 5,439 |
| composition exact | 100.00% |
| SG ok overall / rank1 | 93.79% / 89.68% |

E54 validation StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 27.34% | 35.16% | 37.50% | 0.0616 | 0.1115 | 0.1232 | 0.1270 | 74.22% |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 0.0964 | 0.1039 | 0.1053 | 0.1053 | 66.67% |
| atoms>=12 | 12.93% | 20.69% | 29.31% | 31.03% | 0.0718 | 0.1199 | 0.1484 | 0.1501 | 71.55% |

对比 E47：

| metric | E47 row-conditioned | E54 W/A/skeleton-priority | delta |
|---|---:|---:|---:|
| full match@50 | 39.06% | 37.50% | -1.56 pp |
| rows>=7 match@50 | 11.67% | 11.67% | 0.00 pp |
| atoms>=12 match@50 | 32.76% | 31.03% | -1.72 pp |

判断：

- Source-priority retrieval 不是解决方案；精确 W/A/skeleton source 反而略降 full K50。
- 说明当前 e08 row-conditioned source selection 已不只是被 pruning 限制，rows>=7 转化瓶颈更可能来自 free-param/lattice proposal 本身。

## 26. E55-E57 - train-only geometry net proposal diagnostic

时间：2026-06-13 UTC

目的：

- 训练一个轻量 geometry head：`formula+GT-SG+W/A rows -> lattice/free_params`。
- 只用 MPTS-52 train split 构建 vocab、lattice stats、训练权重；val 只用于 checkpoint selection 和 CIF validation。
- 检查 learned geometry/free-param head 是否比 e08 retrieval 更能把 E44 W/A candidates 转成 StructureMatcher match。

新增脚本：

- `model/New_model/opentry_3/scripts/opentry_train_geometry_net.py`
  - 复用 upstream `GeometryNet` 架构，但 vocab/statistics 只来自 train。
  - 所有 checkpoint/summary 写入 `opentry_3`。
- `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_geometry_net.py`
  - 加载 train-only geometry checkpoint。
  - 对 E44 W/A candidates 预测 lattice/free_params 并渲染 CIF。
  - 支持 label-free geometry variants；本次使用 W/A-diverse order，最多 5 variants/WA。

运行说明：

- E55 初始 `num_workers=4` 触发 sandbox PyTorch DataLoader multiprocessing socket restriction，已中断，不产生有效 checkpoint。
- E55b 40 epoch 试跑到 epoch 15 后出现明显 overfit：epoch 5 val loss `0.9060`，epoch 15 val loss `1.0205`，已中断，仅作为观察。
- E55c 重新跑 5 epoch 完整训练，产生可审计 summary 和 best checkpoint。

E55c training：

| item | value |
|---|---:|
| train / val records | 25,998 / 4,727 |
| vocab/stat source | train only |
| device | CPU |
| epochs | 5 |
| complex_weight / coord_weight | 4.0 / 4.0 |
| best epoch | 4 |
| best val loss | 0.9080 |
| checkpoint | `model/New_model/opentry_3/runs/e55c_geometry_net_fulltrain_cw4_coord4_5epoch/ckpt_best.pt` |

E56 render：

| item | value |
|---|---:|
| samples with candidates | 125 / 128 |
| rendered rows | 4,183 |
| composition exact | 100.00% |
| SG ok overall / rank1 | 46.67% / 27.20% |

E57 validation StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 10.94% | 17.97% | 21.88% | 22.66% | 0.1903 | 0.2661 | 0.2786 | 0.2802 | 71.88% |
| rows>=7 | 0.00% | 0.00% | 0.00% | 0.00% | NA | NA | NA | NA | 61.67% |
| atoms>=12 | 5.17% | 11.21% | 15.52% | 16.38% | 0.2670 | 0.3347 | 0.3398 | 0.3412 | 68.97% |

判断：

- One-shot geometry net 明显劣于 e08 retrieval；rows>=7 完全没有 match。
- SG detection 大幅下降，说明该 deterministic head 生成的 free params / lattice 虽 composition exact，但无法维持有效对称/几何。
- 训练/val MSE 下降不能保证 StructureMatcher match；与 CrystaLLM SFT 路线的经验一致。
- 下一步如果继续 learned geometry，必须使用多模态/分布式 geometry proposal 或 collision/symmetry-aware loss，而不是单点回归。

## 27. E58 - e08 params + geometry-net lattice diagnostic aborted

时间：2026-06-13 UTC

目的：

- 分解 E57 失败来源：保留 e08/source free params，只用 E55c geometry net lattice。

脚本改动：

- `opentry_render_wyckoff_cifs_e07e08.py`
  - 新增 `--geometry-lattice-mode source|geometry_net`
  - 新增 `--geometry-lattice-ckpt`
  - 默认仍为 source lattice。

状态：

- E58 render 因当前实现对每个 candidate 单独做 geometry-net lattice inference，val128 运行过慢，已中断。
- 不产生有效 `render_summary` 或 StructureMatcher 指标。

判断：

- E58 不计入模型结论。
- 若后续要继续 lattice-only diagnostic，应先把 renderer 改成 per-sample/candidate batch inference。
- 但 E57 已足够说明当前 one-shot geometry head 不适合作为替换 e08 retrieval 的主路线。

## 28. E59-E62 - label-free e08 free-param/lattice jitter diagnostics

时间：2026-06-13 UTC

目的：

- 在不使用 StructureMatcher label 排序/过滤的前提下，测试简单 multimodal geometry proposal。
- 保留 e08 train retrieval source，给前排 W/A 的 free params 和 lattice lengths 加 deterministic wrapped jitter。
- 仍只在 val128 上诊断，不进入 full test。

脚本改动：

- `opentry_render_wyckoff_cifs_e07e08.py`
  - 新增 `--geometry-param-variant-mode none|wrapped_jitter|wrapped_jitter_lattice`
  - 默认 `none`，保持历史 runs 可复现。
  - `wrapped_jitter_lattice` 对 source free params 加小幅 wrapped offset，并对 a/b/c 做小幅 scale。

E59/E60 hybrid5x5：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@5 | W/A@20 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 21.09% | 30.47% | 38.28% | 0.0773 | 0.0796 | 0.1395 | 0.1315 | 46.88% | 64.84% | 74.22% |
| rows>=7 | 6.67% | 8.33% | 10.00% | 11.67% | 0.0964 | 0.1084 | 0.1226 | 0.1114 | 48.33% | 65.00% | 66.67% |
| atoms>=12 | 12.93% | 16.38% | 23.28% | 31.90% | 0.0959 | 0.1065 | 0.1597 | 0.1544 | 44.83% | 62.07% | 71.55% |

E61/E62 hybrid2x3：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@5 | W/A@20 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 25.00% | 35.16% | 37.50% | 0.0773 | 0.0960 | 0.1254 | 0.1273 | 55.47% | 74.22% | 74.22% |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 0.0964 | 0.1226 | 0.1114 | 0.1114 | 56.67% | 66.67% | 66.67% |
| atoms>=12 | 12.93% | 18.97% | 28.45% | 31.03% | 0.0959 | 0.1116 | 0.1448 | 0.1518 | 53.45% | 71.55% | 71.55% |

对比 E47：

| run | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---:|---:|---:|---:|
| E47 e08 W/A-diverse | 27.34% | 35.94% | 39.06% | 11.67% |
| E60 jitter hybrid5x5 | 21.09% | 30.47% | 38.28% | 11.67% |
| E62 jitter hybrid2x3 | 25.00% | 35.16% | 37.50% | 11.67% |

判断：

- 简单 wrapped jitter 没有提高 K50，也没有改善 rows>=7。
- jitter 主要牺牲前排 W/A diversity；hybrid2x3 比 hybrid5x5 保守，但仍低于 E47。
- 当前可以排除三类低成本 geometry tweak：
  1. source priority；
  2. one-shot MSE geometry net；
  3. deterministic jitter around e08 retrieval。
- 下一步若继续 geometry，必须显式建模多模态/碰撞/对称有效性，而不是 schedule 或 small jitter。

## 29. E63 - train-only neural assignment scorer for exact W/A DP

时间：2026-06-13 UTC

目的：

- 回到 B-stage symbolic assignment，而不是继续 geometry tweak。
- 训练一个 train-only `formula + GT-SG + skeleton orbit + remaining composition -> element` assignment scorer。
- 将该 scorer 作为 exact composition DP 的可选 score term，用于改善 W/A 排序和复杂 rows>=7。

新增/修改文件：

- 新增 `model/New_model/opentry_3/scripts/opentry_train_assignment_scorer.py`
- 修改 `model/New_model/opentry_3/scripts/diagnose_wyckoff_assignment_dp.py`
  - 新增 `--assignment-model-ckpt`
  - 新增 `--assignment-model-weight`
  - 新增 `--prior-weight`
  - 新增 per-record/step batched neural score cache
  - 默认 `assignment_model_weight=0.0`，保留 prior-only 历史行为

训练设置：

| item | value |
|---|---:|
| train records | 25,998 |
| val records | 1,024 |
| train assignment states | 146,596 |
| val assignment states | 7,821 |
| hidden | 256 |
| epochs | 6 |
| complex weight | 3.0 |
| device | CPU |
| checkpoint | `model/New_model/opentry_3/runs/e63_assignment_scorer_fulltrain/best.pt` |

训练结果：

| epoch | val top1 | val top3 | val top5 | val MRR | val loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.32% | 36.99% | 42.45% | 34.77% | 3.3626 |
| 2 | 27.34% | 43.03% | 49.10% | 39.13% | 3.0933 |
| 3 | 47.88% | 61.82% | 66.99% | 57.44% | 2.4165 |
| 4 | 53.77% | 71.27% | 76.99% | 64.80% | 1.9934 |
| 5 | 58.09% | 77.69% | 82.89% | 69.67% | 1.7080 |
| 6 | 60.84% | 82.18% | 86.86% | 72.72% | 1.5220 |

判断：

- E63 证明 assignment head 有清晰 train/val 可学习信号。
- 这是 train-only 模型侧 scorer，不使用 test，也不使用 StructureMatcher label。
- 直接把 scorer 接入 full `s20p20` exact DP 太慢；已加入 per-step batched scorer cache，但大预算仍需要进一步工程优化。

## 30. E64-E74 - neural assignment scorer in composition-exact DP

时间：2026-06-13 UTC

目的：

- 在 E35 count-aware skeleton candidates 上测试 E63 scorer 是否能提高 canonical W/A recall。
- 只在 val 前 128 条上调参；不渲染 CIF，除非 symbolic gate 明显通过。

工程观察：

- full `s20p20` DP budget 即使 batched neural score 仍过慢，中断：
  - E64a/E64b/E64c/E64d
  - E65a/E65b
- 改用 reduced exact-DP diagnostic budget：
  - `max_skeletons=10`
  - `per_skeleton=10`
  - `state_beam=30`
  - `max_active_paths=500`
  - `top_k=50`
- 该 budget 不等价于 E45 full `s20p20` best，但可公平比较 prior-only 与 neural scorer。

Reduced-budget val32 diagnostic：

| config | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 |
|---|---:|---:|---:|---:|---:|
| E68a prior-only | 40.62% | 75.00% | 78.12% | 78.12% | 66.67% |
| E68b prior + 0.5 neural | 53.12% | 81.25% | 87.50% | 87.50% | 83.33% |

Reduced-budget val128 canonical W/A：

| config | prior weight | neural weight | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@1 | rows>=7 W/A@5 | rows>=7 W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E71a prior-only s10p10 | 1.0 | 0.0 | 45.31% | 68.75% | 71.88% | 71.88% | 48.33% | 68.33% | 68.33% | 8.80 |
| E71b neural s10p10 | 1.0 | 0.5 | 49.22% | 70.31% | 75.00% | 75.00% | 53.33% | 73.33% | 75.00% | 9.68 |
| E71c neural s10p10 | 1.0 | 1.0 | 47.66% | 70.31% | 75.78% | 75.78% | 51.67% | 73.33% | 76.67% | 10.00 |
| E71d neural s10p10 | 0.5 | 1.0 | 47.66% | 67.97% | 75.00% | 75.78% | 48.33% | 70.00% | 75.00% | 10.16 |
| E74 neural s15p15 | 1.0 | 1.0 | 47.66% | 67.97% | 73.44% | 75.00% | 51.67% | 70.00% | 70.00% | 10.67 |
| E45 prior full-budget reference | - | - | 45.31% | 66.41% | 74.22% | 74.22% | 48.33% | 66.67% | 66.67% | 12.59 |

判断：

- E63 neural assignment scorer provides real B-stage signal:
  - best reduced-budget full W/A@50 reaches `75.78%`, slightly above E45 `74.22%`;
  - rows>=7 W/A@50 improves from E45 `66.67%` to `76.67%`;
  - W/A@5 improves from E45 `66.41%` to `70.31%`.
- However, this is not yet a clean CIF gate pass:
  - full W/A@50 gain over E45 is only `+1.56 pp`, not a decisive margin;
  - unique W/A@50 is lower than E45 (`10.00` vs `12.59`);
  - reduced `s10p10` budget is not directly comparable to the prior full `s20p20` setting;
  - larger `s15p15` did not improve W/A@50.
- Therefore no CIF rendering or full test was run from E63-E74.

Next step:

- Keep E63 assignment scorer as promising B-stage component.
- Optimize neural exact-DP full-budget implementation before rendering:
  - batch candidate-state scoring more aggressively;
  - add per-sample DP profiling;
  - preserve E45 fixed-orbit duplicate mask;
  - target a full `s20p20` neural run with full W/A@50 and unique W/A both clearly above E45.

## 31. E75-E87 - neural assignment rescoring / priority merge and val CIF check

时间：2026-06-13 UTC

目的：

- 避免重新跑慢速 full neural DP，直接利用已生成的 exact-cover W/A candidates。
- 用 E63 train-only assignment scorer 对 full-budget E43 candidates 和 reduced neural E69c candidates 进行 label-free B-stage 组合/排序诊断。
- 只使用 val symbolic/CIF evaluator，不使用 test，不使用 StructureMatcher label 排序。

新增文件：

- `model/New_model/opentry_3/scripts/opentry_rescore_dp_candidates.py`
  - 对 complete W/A sequence 计算 train-prior score + E63 neural assignment score。
  - 可合并多个 DP candidate files。
- `model/New_model/opentry_3/scripts/opentry_merge_dp_candidates_priority.py`
  - label-free priority merge。
  - 保留第一个 symbolic decoder 的 early order，再用后续 candidate file 补 tail diversity。

E75/E76 rescoring：

| run | input | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 | unique W/A@50 | judgment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E79 | rescore E43 full prior candidates | 28.12% | 59.38% | 72.66% | 74.22% | 66.67% | 12.59 | hurts early ranks |
| E80 | rescore union E43+E69c | 32.81% | 59.38% | 76.56% | 78.12% | 76.67% | 14.05 | strong tail, weak early |

E81/E82 CIF check for E80 rescored union:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 14.84% | 25.00% | 35.94% | 41.41% | 0.0888 | 0.1320 | 0.1464 | 0.1619 | 78.12% |
| rows>=7 | 6.67% | 11.67% | 15.00% | 16.67% | 0.0964 | 0.1405 | 0.1689 | 0.1997 | 76.67% |
| atoms>=12 | 12.93% | 19.83% | 29.31% | 35.34% | 0.1017 | 0.1588 | 0.1622 | 0.1921 | 75.86% |

判断：

- E80/E82 提高 full match@50 和 rows>=7 match@50，但 early rank 太弱。
- 说明 pure rescoring 不适合作为最终 ordering：neural scorer 有 recall/tail 信号，但会把 early true candidates 往后推。

E83 priority merge：

- source 0: E69c neural DP `s10p10`
- source 1: E43 full-budget prior DP `s20p20`
- 规则：保留 source 0 order，再用 source 1 补足 unique raw W/A。
- 该步骤不使用 label，只是 symbolic decoder order/coverage merge。

E85 canonical W/A:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | skeleton@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| full | 47.66% | 70.31% | 77.34% | 78.91% | 82.03% | 13.74 |
| rows>=7 | 51.67% | 73.33% | 76.67% | 76.67% | 81.67% | 8.37 |
| atoms>=12 | 45.69% | 68.10% | 75.00% | 76.72% | 80.17% | 12.93 |

对比 E45：

| metric | E45 | E85 | delta |
|---|---:|---:|---:|
| full W/A@5 | 66.41% | 70.31% | +3.91 pp |
| full W/A@20 | 74.22% | 77.34% | +3.12 pp |
| full W/A@50 | 74.22% | 78.91% | +4.69 pp |
| full unique W/A@50 | 12.59 | 13.74 | +1.15 |
| rows>=7 W/A@50 | 66.67% | 76.67% | +10.00 pp |

判断：

- E85 finally clears the symbolic W/A gate strongly enough for val-only CIF rendering:
  - full and rows>=7 W/A@50 both clearly exceed E45;
  - unique W/A@50 also exceeds E45;
  - composition exact remains 98.44% due two missing rendered samples, not invalid top candidates.

E86/E87 val CIF check:

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 21.09% | 28.12% | 35.94% | 40.62% | 0.0927 | 0.1123 | 0.1397 | 0.1488 | 78.91% |
| rows>=7 | 6.67% | 10.00% | 15.00% | 16.67% | 0.0964 | 0.1566 | 0.1689 | 0.1997 | 76.67% |
| atoms>=12 | 16.38% | 21.55% | 29.31% | 34.48% | 0.1138 | 0.1179 | 0.1631 | 0.1758 | 76.72% |

E47/E87 comparison:

| metric | E47 fixed-mask e08 | E87 neural-first merge e08 | delta |
|---|---:|---:|---:|
| full match@1 | 17.97% | 21.09% | +3.12 pp |
| full match@5 | 27.34% | 28.12% | +0.78 pp |
| full match@20 | 35.94% | 35.94% | +0.00 pp |
| full match@50 | 39.06% | 40.62% | +1.56 pp |
| rows>=7 match@50 | 11.67% | 16.67% | +5.00 pp |
| full W/A@50 | 74.22% | 78.91% | +4.69 pp |
| rows>=7 W/A@50 | 66.67% | 76.67% | +10.00 pp |

判断：

- E87 is the current best opentry_3 validation CIF at match@50 and rows>=7 match@50.
- It still does not justify full test:
  - match@5 remains below E33/E47-era best enough to be risky;
  - rows>=7 match remains poor despite W/A recall jump;
  - geometry/free-param conversion is still the limiting factor.
- No full MPTS-52 test run was launched.

Next step:

- Keep E83 neural-first merge as the current best B-stage symbolic candidate.
- Focus next on geometry conversion for the stronger W/A set:
  - not source priority / one-shot MSE / small jitter;
  - consider multimodal geometry proposal or collision/symmetry-aware geometry scoring;
  - use E84/E86 as the validation W/A input for geometry experiments.

## 32. E88-E89 - hybrid geometry on neural-first W/A set

时间：2026-06-13 UTC

目的：

- 检查 E84/E85 更强 early W/A 是否能让 `hybrid_top_wa` geometry plan 变得有用。
- 这不是 test，也不使用 label 排序；只是在 val128 上对同一 E84 W/A candidates 换 e08 geometry plan。

配置：

| item | value |
|---|---|
| W/A input | `reports/e84_renderer_predictions_e83_neural_first_merge/renderer_predictions.jsonl` |
| geometry | e08 row-conditioned KNN |
| plan | `hybrid_top_wa` |
| hybrid W/A / geometry ranks | 5 / 3 |
| top_k | 50 |

E88 render integrity：

| item | value |
|---|---:|
| rendered rows | 5,539 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG ok overall / rank1 / rank<=5 | 94.29% / 90.48% / 92.81% |

E89 StructureMatcher：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@5 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 21.09% | 27.34% | 35.94% | 40.62% | 0.0927 | 0.1196 | 0.1299 | 0.1446 | 57.03% | 78.91% | 12.37 |
| rows>=7 | 6.67% | 10.00% | 15.00% | 16.67% | 0.0964 | 0.1180 | 0.1689 | 0.1997 | 56.67% | 76.67% | 7.42 |
| atoms>=12 | 16.38% | 19.83% | 29.31% | 34.48% | 0.1138 | 0.1307 | 0.1577 | 0.1736 | 54.31% | 76.72% | 11.68 |

对比 E87 W/A-diverse：

| metric | E87 W/A-diverse | E89 hybrid5x3 | delta |
|---|---:|---:|---:|
| full match@5 | 28.12% | 27.34% | -0.78 pp |
| full match@20 | 35.94% | 35.94% | 0.00 pp |
| full match@50 | 40.62% | 40.62% | 0.00 pp |
| full W/A@5 | 72.66% | 57.03% | -15.62 pp |
| rows>=7 match@50 | 16.67% | 16.67% | 0.00 pp |

判断：

- 即使在 E84/E85 更强 W/A set 上，简单 hybrid geometry schedule 仍不改善 match@50，并牺牲 early W/A diversity。
- 与 E48-E51 一致，geometry-rank scheduling 已基本排除。
- 下一步不应继续调 `hybrid_top_wa` / `geometry_interleave` / jitter；需要真正的 geometry proposal/scoring：
  - 多模态 train-only geometry proposal；
  - collision/symmetry-aware filtering or scoring based on generated CIF features, not StructureMatcher labels；
  - 针对 rows>=7 的 free-param/lattice conversion。

## 33. E90-E93 - GT-free generated-CIF self-score ordering

时间：2026-06-13 UTC

目的：

- 在不使用 StructureMatcher label 的前提下，对 E86 rendered CIF candidates 做 generated-CIF 自洽排序。
- 检查可读性、composition exact、SG consistency、最短原子距离、体积/原子、geometry distance 等 GT-free features 是否能改善 early match。
- 继续只使用 val128；不读取 test，不用 match/rms 训练或排序。

新增脚本：

- `model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py`

方法：

- 读取 E86 `rendered_topk.jsonl`。
- 使用 pymatgen 解析 generated CIF，计算：
  - `self_min_distance`
  - `self_volume_per_atom`
  - parse status / site count
- self-score 只使用 generated candidate 自身信息：
  - readable / formula_ok / atom_count_ok / composition_exact / sg_ok
  - min-distance sanity
  - volume-per-atom sanity
  - geometry_distance
  - original rank tie-break
- E90 使用 `mode=diverse`：每个 W/A 先取最高自洽 geometry，再补后续 geometry，保留 W/A diversity。
- E92 使用 `mode=global`：全局按 self-score 排序，作为对照。

E90 self-score integrity：

| item | value |
|---|---:|
| input rows | 5,539 |
| output rows | 5,539 |
| samples | 126 |
| mode | diverse |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.36% |
| rank<=20 SG-ok | 97.49% |
| rank1 min-distance mean | 1.8757 |
| rank<=5 min-distance mean | 1.6901 |

E91 diverse self-score StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@5 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 28.91% | 32.81% | 38.28% | 40.62% | 0.1096 | 0.1152 | 0.1349 | 0.1488 | 71.88% | 78.91% | 12.80 |
| rows>=7 | 11.67% | 11.67% | 16.67% | 16.67% | 0.1884 | 0.1884 | 0.1997 | 0.1997 | 73.33% | 76.67% | 7.42 |
| atoms>=12 | 23.28% | 25.86% | 31.90% | 34.48% | 0.1089 | 0.1236 | 0.1591 | 0.1758 | 68.97% | 76.72% | 11.91 |

E93 global self-score对照：

| subset | match@1 | match@5 | match@20 | match@50 | W/A@5 | W/A@20 | unique W/A@5 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 28.91% | 32.81% | 38.28% | 40.62% | 68.75% | 75.00% | 3.02 | 8.02 |
| rows>=7 | 11.67% | 13.33% | 16.67% | 16.67% | 68.33% | 75.00% | 2.67 | 5.95 |

对比 E87：

| metric | E87 W/A-diverse e08 | E91 self-score diverse | delta |
|---|---:|---:|---:|
| full match@1 | 21.09% | 28.91% | +7.81 pp |
| full match@5 | 28.12% | 32.81% | +4.69 pp |
| full match@20 | 35.94% | 38.28% | +2.34 pp |
| full match@50 | 40.62% | 40.62% | +0.00 pp |
| rows>=7 match@1 | 6.67% | 11.67% | +5.00 pp |
| rows>=7 match@5 | 10.00% | 11.67% | +1.67 pp |
| rows>=7 match@50 | 16.67% | 16.67% | +0.00 pp |

判断：

- GT-free generated-CIF self-score 能显著改善 early ordering，尤其 full match@1/5/20。
- K50 上限不变，说明它主要解决 geometry ordering，而不是新 candidate recall。
- diverse 与 global 的 full match 相同，但 global 明显损失 W/A diversity；后续主线保留 E91 diverse。
- E91 仍不满足 full test freeze：val128 full match@5 只有 32.81%，rows>=7 match@50 仍只有 16.67%。
- 下一步应把 self-score 作为 geometry proposal 的 GT-free scoring layer，并生成更多 train-only multimodal geometry proposals，而不是只重排现有 E86 K50。

## 34. E94-E96 - larger e08 geometry proposal pool + self-score top50

时间：2026-06-13 UTC

目的：

- 在 E90-E93 证明 GT-free self-score 能改善 early ordering 后，扩大 geometry proposal pool。
- 对同一 E84 W/A predictions 渲染 `top_k=100`，允许更多 e08 train-source geometry ranks 进入候选池。
- 再用 E90 的 diverse self-score 从 larger pool 选回 top50。
- 仍只跑 val128，不使用 test，不使用 StructureMatcher label 排序。

E94 render：

| item | value |
|---|---:|
| input W/A | E84 neural-first merge |
| geometry | e08 row-conditioned KNN |
| top_k | 100 |
| geometry ranks per W/A | 10 |
| rendered rows | 8,905 |
| samples with rendered candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.27% / 91.27% / 96.01% / 95.20% |

E95 self-score top50：

| item | value |
|---|---:|
| input rows | 8,905 |
| output rows | 5,539 |
| mode | diverse |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.36% |
| rank<=20 SG-ok | 97.79% |
| overall SG-ok | 96.12% |
| rank1 min-distance mean | 1.8969 |

E96 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 29.69% | 32.81% | 39.84% | 39.84% | 0.1176 | 0.1141 | 0.1476 | 0.1445 | 13.20 |
| rows>=7 | 11.67% | 11.67% | 16.67% | 16.67% | 0.1884 | 0.1884 | 0.1997 | 0.1997 | 7.42 |
| atoms>=12 | 24.14% | 25.86% | 33.62% | 33.62% | 0.1198 | 0.1213 | 0.1739 | 0.1736 | 12.35 |

对比 E91：

| metric | E91 K50 pool self-score | E96 K100 pool self-score | delta |
|---|---:|---:|---:|
| full match@1 | 28.91% | 29.69% | +0.78 pp |
| full match@5 | 32.81% | 32.81% | +0.00 pp |
| full match@20 | 38.28% | 39.84% | +1.56 pp |
| full match@50 | 40.62% | 39.84% | -0.78 pp |
| unique W/A@50 | 12.80 | 13.20 | +0.40 |
| atoms>=12 match@20 | 31.90% | 33.62% | +1.72 pp |

判断：

- Larger geometry pool + self-score improves early ranks and unique W/A diversity, especially match@1 and match@20.
- E96 lowers match@50 slightly, so it is not a uniform replacement for E91.
- Current validation best should be tracked as:
  - early-rank best: E96 (`match@1/5/20 = 29.69/32.81/39.84`)
  - K50 ceiling best: E91 (`match@50 = 40.62`)
- Still no full test freeze: match@5 is far below 41.58% target and rows>=7 remains weak.
- Next useful step is not simply increasing top_k again; it should add a better proposal/selection balance that preserves E96 early-rank gains without losing E91 K50 tail.

## 35. E97-E100 - prefix-tail geometry selection policy

时间：2026-06-13 UTC

目的：

- 合并 E96 的 early-rank gain 与 E91 的 K50 tail。
- 不使用 StructureMatcher label 排序；只按两个已有 GT-free ordering 做 deterministic merge：
  - prefix: E95/E96 larger-pool self-score ordering；
  - tail: E90/E91 K50-pool self-score ordering。
- 评估 prefix length 10 和 20。

新增脚本：

- `model/New_model/opentry_3/scripts/opentry_merge_rendered_prefix_tail.py`

方法：

- 对每个 sample：
  - 先取 prefix list 前 `prefix_k`；
  - 再用 tail list 补齐 top50；
  - 跳过 exact CIF duplicate；
  - 若不足，再从 prefix 后续补齐。
- 该合并只使用 generated-CIF candidates 与已有 GT-free rank，不读取 target CIF 或 match/rms。

E97 prefix=10 合并完整性：

| item | value |
|---|---:|
| rows | 5,580 |
| samples | 126 |
| prefix / tail / prefix_fill rows | 1,239 / 4,315 / 26 |
| SG-ok | 94.87% |
| unique W/A mean | 12.92 |

E99 prefix=20 合并完整性：

| item | value |
|---|---:|
| rows | 5,580 |
| samples | 126 |
| prefix / tail / prefix_fill rows | 2,394 / 3,183 / 3 |
| SG-ok | 95.39% |
| unique W/A mean | 13.00 |

E98/E100 StructureMatcher result：

| run | prefix_k | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E98 | 10 | 29.69% | 32.81% | 39.84% | 40.62% | 0.1176 | 0.1141 | 0.1462 | 0.1489 | 12.72 |
| E100 | 20 | 29.69% | 32.81% | 39.84% | 40.62% | 0.1176 | 0.1141 | 0.1476 | 0.1489 | 12.80 |

E100 complex subsets：

| subset | match@1 | match@5 | match@20 | match@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| rows>=7 | 11.67% | 11.67% | 16.67% | 16.67% | 76.67% | 7.42 |
| atoms>=12 | 24.14% | 25.86% | 33.62% | 34.48% | 76.72% | 12.05 |
| complex_flag | 24.37% | 27.73% | 35.29% | 36.13% | 77.31% | 12.45 |

对比当前参考：

| metric | E91 K50-pool self-score | E96 K100-pool self-score | E100 prefix20-tail |
|---|---:|---:|---:|
| full match@1 | 28.91% | 29.69% | 29.69% |
| full match@5 | 32.81% | 32.81% | 32.81% |
| full match@20 | 38.28% | 39.84% | 39.84% |
| full match@50 | 40.62% | 39.84% | 40.62% |
| unique W/A@50 | 12.80 | 13.20 | 12.80 |

判断：

- E100 成功合并了 E96 的 early-rank 与 E91 的 K50 tail，是当前 best validation CIF ordering。
- 但仍不满足 full test freeze：
  - full match@5 只有 32.81%，低于 41.58%；
  - full match@50 仍只有 40.62%；
  - rows>=7 match@50 仍只有 16.67%。
- 当前瓶颈已经不是简单排序，而是 geometry proposal ceiling / rows>=7 free-param conversion。
- 下一步应针对 rows>=7 生成新的 train-only multimodal geometry proposals，或训练非 SM-label的 geometry validity/proposal model，而不是继续 prefix/tail 调参。

## 36. E101-E106 - train-only row-prototype geometry proposals

时间：2026-06-13 UTC

目的：

- 增加真正不同于 e08 whole-source copy 的 geometry proposal capacity。
- 针对每个 predicted W/A row，从 train split 中按 `(SG, orbit_id, element, free_symbols)` 等层级取 row-level free-parameter prototype。
- lattice 仍使用 e08 row-conditioned source lattice；free params 改为逐行组合。
- 不使用 test、不使用 StructureMatcher label 排序；StructureMatcher 只用于 val128 评估。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`
- 新增 `--geometry-source-strategy row_prototype`
- 原理：
  - 初始化时只从 train records 构建 row prototype index；
  - render 时查表取 row-level prototypes；
  - 参数不足时 fallback 到 train-only free-param quantile；
  - 所有输出仍 composition-exact 过滤。

性能修正：

- 初版每个 row 动态排序大 prototype pool，smoke 运行过慢，已中断。
- 后续改成按匹配层级直接取小池，`E101b` K5 smoke 可运行。

E102 row-prototype K100 render：

| item | value |
|---|---:|
| rendered rows | 9,136 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 85.63% / 85.71% / 89.52% / 88.10% |

E103 self-score on row-prototype pool：

| item | value |
|---|---:|
| output rows | 5,587 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 97.78% |
| rank<=20 SG-ok | 94.61% |
| overall SG-ok | 90.69% |
| rank1 min-distance mean | 1.5257 |

E104 row-prototype standalone result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 17.97% | 24.22% | 28.12% | 30.47% | 0.1674 | 0.1635 | 0.1798 | 0.1865 | 13.59 |
| rows>=7 | 1.67% | 1.67% | 1.67% | 1.67% | 0.3851 | 0.3851 | 0.3851 | 0.3851 | 8.12 |
| atoms>=12 | 12.07% | 16.38% | 20.69% | 23.28% | 0.1483 | 0.2008 | 0.2175 | 0.2302 | 12.77 |

判断：

- row-prototype standalone 明显差于 E100/E91，尤其 rows>=7。
- 逐行 free-param 组合会破坏全局几何一致性；不能作为主 renderer。

E105/E106 conservative tail-fill：

- prefix: E100 current best ordering，完整保留已有顺序；
- tail: E103 row-prototype self-score candidates；
- `prefix_k=50`，只在 sample 不足 top50 或 exact CIF duplicate 后补 row-prototype tail。

E105 merge integrity：

| item | value |
|---|---:|
| rows | 5,902 |
| samples | 126 |
| prefix / tail rows | 5,580 / 322 |
| SG-ok | 95.36% |
| unique W/A mean | 13.10 |

E106 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 29.69% | 32.81% | 40.62% | 41.41% | 0.1176 | 0.1141 | 0.1522 | 0.1521 | 12.90 |
| rows>=7 | 11.67% | 11.67% | 18.33% | 18.33% | 0.1884 | 0.1884 | 0.2166 | 0.2166 | 7.55 |
| atoms>=12 | 24.14% | 25.86% | 34.48% | 35.34% | 0.1198 | 0.1213 | 0.1792 | 0.1794 | 12.16 |
| complex_flag | 24.37% | 27.73% | 36.13% | 36.97% | 0.1203 | 0.1279 | 0.1802 | 0.1804 | 12.56 |

对比 E100：

| metric | E100 | E106 | delta |
|---|---:|---:|---:|
| full match@1 | 29.69% | 29.69% | +0.00 pp |
| full match@5 | 32.81% | 32.81% | +0.00 pp |
| full match@20 | 39.84% | 40.62% | +0.78 pp |
| full match@50 | 40.62% | 41.41% | +0.78 pp |
| rows>=7 match@20 | 16.67% | 18.33% | +1.67 pp |
| rows>=7 match@50 | 16.67% | 18.33% | +1.67 pp |

E107 raw row-prototype tail-fill：

- 用 E102 raw row-prototype pool 继续补 E105 空位，但没有新增非重复候选；E107 是 no-op。

判断：

- Row-prototype 不适合单独排序，但作为保守 tail-fill 能补少量 e08 tail 缺口。
- E106 是当前 opentry_3 validation CIF best：
  - early rank 保持 E100；
  - match@20 / match@50 小幅提高；
  - rows>=7 有小幅改善。
- 仍不能 full test freeze：
  - match@1 仍低于 31.64%；
  - match@5 仍低于 41.58%；
  - match@20/50 远低于 +5pp K20 target；
  - rows>=7 conversion 仍很弱。
- 下一步需要让 row-level geometry proposal 保持全局一致性，例如按 source-cluster/row-prototype mixture 生成，而不是完全逐行独立组合。

## 37. E108-E113 - source-consistent row-aligned geometry

时间：2026-06-13 UTC

目的：

- 修复 e08 `params_from_source` 的潜在 row-index mismatch：e08 从一个 train source 复制 free params，但按 canonical row index 对齐；当 predicted W/A rows 与 source rows 不同构时，容易把 free params 复制到错误 row。
- 保留全局 source consistency：同一个 train source 提供 lattice 和 free-param source pool。
- 在该 source 内做 target/source row alignment，再复制 free params，避免 E102 row-prototype 的逐行独立混搭噪声。

工程改动：

- 在 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py` 新增：
  - `--geometry-source-strategy row_aligned_knn`
  - `OpentryGeometrySelector.source_aligned_params`
- row alignment cost 使用：
  - orbit_id mismatch
  - element mismatch
  - free_symbols mismatch
  - multiplicity mismatch
  - site_symmetry mismatch
- source selection 仍来自 train-only row-conditioned KNN。
- 不使用 StructureMatcher label、不使用 test。

E108 smoke：

| item | value |
|---|---:|
| top_k / geometry ranks | 5 / 2 |
| rendered rows | 575 |
| readable / composition exact | 100% / 100% |
| SG-ok rank1 / rank<=5 | 90.40% / 95.65% |

E109 K100 render：

| item | value |
|---|---:|
| rendered rows | 8,892 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.69% / 89.68% / 95.53% / 95.08% |

E110 self-score：

| item | value |
|---|---:|
| output rows | 5,542 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.36% |
| rank<=20 SG-ok | 97.96% |
| overall SG-ok | 96.37% |
| rank1 min-distance mean | 1.8906 |

E111 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 41.41% | 42.19% | 0.1137 | 0.1174 | 0.1382 | 0.1396 | 78.91% | 13.20 |
| rows>=7 | 13.33% | 13.33% | 18.33% | 18.33% | 0.1689 | 0.1689 | 0.1516 | 0.1450 | 76.67% | 7.42 |
| atoms>=12 | 25.86% | 28.45% | 35.34% | 36.21% | 0.1145 | 0.1252 | 0.1604 | 0.1653 | 76.72% | 12.35 |
| complex_flag | 26.05% | 30.25% | 36.97% | 37.82% | 0.1151 | 0.1309 | 0.1627 | 0.1643 | 77.31% | 12.76 |

E112/E113 prefix-tail check：

- prefix: E111 row-aligned self-score top20
- tail: E106 previous best tail
- result: same match@1/5/20/50 as E111, lower unique W/A@50, slightly lower RMSE@50.

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E111 row_aligned | 31.25% | 35.16% | 41.41% | 42.19% | 0.1396 | 13.20 |
| E113 E111 prefix + E106 tail | 31.25% | 35.16% | 41.41% | 42.19% | 0.1366 | 12.54 |

对比 E106：

| metric | E106 | E111 | delta |
|---|---:|---:|---:|
| full match@1 | 29.69% | 31.25% | +1.56 pp |
| full match@5 | 32.81% | 35.16% | +2.34 pp |
| full match@20 | 40.62% | 41.41% | +0.78 pp |
| full match@50 | 41.41% | 42.19% | +0.78 pp |
| rows>=7 match@1 | 11.67% | 13.33% | +1.67 pp |
| rows>=7 match@50 | 18.33% | 18.33% | +0.00 pp |

判断：

- E111 是当前 opentry_3 best validation CIF match result。
- match@1=31.25% 已接近 MPTS-52 +5pp target 31.64%，但仍差 0.39 pp。
- match@5=35.16% 仍低于 41.58% target；match@20/50 也仍低于 49.69% K20 target。
- rows>=7 仍弱，说明复杂结构 geometry conversion 仍是核心瓶颈。
- 不能 freeze full test。
- 下一步优先围绕 source-consistent row-aligned strategy 做更好的 source selection / lattice proposal，而不是回到 independent row prototype 或 deterministic tail merge。

## 38. E114-E117 - row-aligned source-priority geometry check

时间：2026-06-13 UTC

目的：

- 在 E111 `row_aligned_knn` 的基础上，尝试更强的 source selection。
- 新策略 `row_aligned_priority` 仍保持 source-consistent row alignment，但先用 target/source row alignment cost 对 train-only e08 source pool 重排。
- 不使用 StructureMatcher label、不使用 test、不用 GT-WA input；排序只使用候选内部/训练源结构信息。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`
- 新增：
  - `--geometry-source-strategy row_aligned_priority`
  - `OpentryGeometrySelector.row_aligned_candidates`
  - `source_alignment_cost`
- 初版 K100 source priority 扫描过慢，`E115` 被中断，不计为有效结果。
- 后续优化为只对小 e08 source pool 重排，避免大规模 source 枚举。

E114 smoke：

| item | value |
|---|---:|
| rendered rows | 583 |
| samples with candidates | 125 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 | 95.71% / 88.80% / 95.71% |

E115c K50 render：

| item | value |
|---|---:|
| rendered rows | 4,550 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.12% / 90.48% / 95.69% / 94.48% |

E116 GT-free self-score：

| item | value |
|---|---:|
| output rows | 4,550 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.20% |
| rank<=20 SG-ok | 96.82% |
| overall SG-ok | 93.12% |

E117 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 28.91% | 33.59% | 39.06% | 40.62% | 0.1018 | 0.0952 | 0.1242 | 0.1317 |
| rows>=7 | 11.67% | 13.33% | 16.67% | 16.67% | 0.1796 | 0.1321 | 0.1422 | 0.1422 |
| atoms>=12 | 24.14% | 27.59% | 32.76% | 34.48% | 0.1038 | 0.1042 | 0.1482 | 0.1584 |
| complex_flag | 23.53% | 28.57% | 34.45% | 36.13% | 0.1043 | 0.1099 | 0.1505 | 0.1557 |

判断：

- `row_aligned_priority` 的 RMSE 明显更低，但 match 指标低于 E111。
- 该策略可能更偏向局部几何质量/SG consistency，而没有提升 StructureMatcher recall。
- E111 仍是 early match best；E117 不进入 full test freeze。

## 39. E118-E119 - global self-score over row_aligned_knn pool

时间：2026-06-13 UTC

目的：

- 在 E109/E110 `row_aligned_knn` K100 pool 上改用 global self-score，检查能否提高 K50 ceiling。
- 仍只用 GT-free generated-CIF self-score，不使用 StructureMatcher label 排序。

E118 scoring：

| item | value |
|---|---:|
| input | E109 row_aligned_knn rendered pool |
| mode | global |
| output rows | 5,542 |
| rank1 / rank<=5 / rank<=20 SG-ok | 100.00% / 100.00% / 99.54% |
| overall SG-ok | 98.39% |

E119 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 41.41% | 43.75% | 0.1137 | 0.1039 | 0.1320 | 0.1457 | 75.78% | 10.62 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1689 | 0.1351 | 0.1516 | 0.1450 | 71.67% | 5.75 |
| atoms>=12 | 25.86% | 28.45% | 35.34% | 37.93% | 0.1145 | 0.1147 | 0.1539 | 0.1724 | 73.28% | 9.71 |
| complex_flag | 26.05% | 30.25% | 36.97% | 39.50% | 0.1151 | 0.1194 | 0.1560 | 0.1718 | 74.79% | 10.25 |

对比 E111：

| metric | E111 diverse | E119 global | delta |
|---|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | +0.00 pp |
| full match@5 | 35.16% | 35.16% | +0.00 pp |
| full match@20 | 41.41% | 41.41% | +0.00 pp |
| full match@50 | 42.19% | 43.75% | +1.56 pp |
| full W/A@50 | 78.91% | 75.78% | -3.12 pp |
| unique W/A@50 | 13.20 | 10.62 | -2.58 |

判断：

- E119 是当前 best validation CIF K50 ceiling：full match@50 `43.75%`。
- early ranks 与 E111 相同，K50 提升来自 geometry/order tail，而不是 W/A recall 增加。
- global self-score 明显降低 W/A diversity，因此不宜直接作为唯一主线；但它说明 row_aligned pool 内仍有更好的 tail geometry 可被 GT-free score 前移。
- 仍不能 full test freeze：match@1 未达到 31.64%，match@5 仍远低于 41.58%，match@20 低于 49.69%。

## 40. E120-E121 - global prefix plus diverse tail merge

时间：2026-06-13 UTC

目的：

- 尝试保留 E119 global self-score 的 early/top20，同时用 E111 diverse ordering 恢复 W/A diversity tail。
- 这是 deterministic GT-free merge，不使用 StructureMatcher label 或 test tuning。

E120 merge：

| item | value |
|---|---:|
| prefix source | E118 global self-score top50 |
| tail source | E110 diverse self-score top50 |
| prefix_k | 20 |
| output rows | 5,542 |
| samples | 126 |
| prefix / tail rows | 2,397 / 3,145 |
| SG-ok overall | 96.57% |
| unique W/A mean | 13.01 |

E121 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 41.41% | 42.19% | 0.1137 | 0.1039 | 0.1320 | 0.1369 | 13.01 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1689 | 0.1351 | 0.1516 | 0.1450 | 7.37 |
| atoms>=12 | 25.86% | 28.45% | 35.34% | 36.21% | 0.1145 | 0.1147 | 0.1539 | 0.1600 | 12.25 |
| complex_flag | 26.05% | 30.25% | 36.97% | 37.82% | 0.1151 | 0.1194 | 0.1560 | 0.1595 | 12.66 |

判断：

- E121 恢复了多数 W/A diversity，并保持 E119 的 K1/K5/K20，但 K50 回落到 E111 水平。
- 当前 best 需要分开记录：
  - best K50 match ceiling：E119 global self-score，full `31.25/35.16/41.41/43.75`；
  - best diversity-preserving early result：E111 diverse row_aligned，full `31.25/35.16/41.41/42.19`，unique W/A@50 `13.20`；
  - E121 是折中结果：同 E111/E119 early ranks，RMSE@50 更低，unique W/A@50 `13.01`。
- 三者都不能进入 full test：
  - match@1 仍差目标 `31.64%` 约 `0.39 pp`；
  - match@5 仍差 `41.58%` 约 `6.42 pp`；
  - match@20 仍差 `49.69%` 约 `8.28 pp`。

## 41. Current freeze status after E121

时间：2026-06-13 UTC

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/opentry_merge_rendered_prefix_tail.py
```

结果：compile 通过。

## 45. Experiment E214-E224 - Cached source-residual geometry with delta constraints

时间：2026-06-14 UTC

目的：

- 将 source-conditioned residual geometry 的 train/val source-pair 构建从在线训练中拆出，缓存到 `opentry_3`，为后续更大训练做准备。
- 显式约束 residual lattice/free-param delta，测试“少移动 source geometry”是否比早期 one-source residual 更稳。
- 仍只使用 MPTS-52 train/val；selector source pool 只来自 train split；StructureMatcher 只用于 val evaluation。

工程改动：

- 新增 `model/New_model/opentry_3/scripts/opentry_build_source_residual_examples.py`。
- 修改 `model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py`：
  - 支持 `--cached-examples-dir` / `--train-examples` / `--val-examples`。
  - 修复 JSONL cache 中 `base_params` integer keys 被字符串化后的读取。
  - 增加 `--lattice-delta-scale` 与 `--coord-delta-scale`；默认保持旧行为。
- 修改 `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`：
  - 从 checkpoint config 读取 delta scale，兼容新模型。

E214 cache：

| item | value |
|---|---:|
| output | `model/New_model/opentry_3/data/source_residual_geometry_mpts52/e214_cache_train1024_val256_fulltrain_selector` |
| selector records | 25,998 train |
| train target examples | 1,024 / 1,024 |
| val target examples | 256 / 256 |
| skipped no source | 0 / 0 |
| train unique source ids | 894 |
| val unique source ids | 230 |
| train mean / max sample weight | 2.209 / 7.0 |
| leakage guard | train selector, train targets, val targets, no test, no StructureMatcher labels |

E215 constrained residual model：

| item | value |
|---|---:|
| cached examples | E214 train1024 / val256 |
| lattice_delta_scale / coord_delta_scale | 0.35 / 0.15 |
| epochs | 3 CPU |
| best val loss | 4.985 |
| best val lattice / coord loss | 4.718 / 0.0668 |

E216-E219 val32/top5 result：

| run | scorer | full match@1 | full match@5 | RMSE@1 | RMSE@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E217 | raw | 18.75% | 31.25% | 0.2849 | 0.2570 | 5.56% | 5.56% | 81.25% | 4.38 |
| E219 | global self-score | 31.25% | 31.25% | 0.2601 | 0.2570 | 5.56% | 5.56% | 81.25% | 4.38 |

E220 tighter residual model：

| item | value |
|---|---:|
| cached examples | E214 train1024 / val256 |
| lattice_delta_scale / coord_delta_scale | 0.15 / 0.05 |
| epochs | 3 CPU |
| best val loss | 5.029 |
| best val lattice / coord loss | 4.763 / 0.0667 |

E221-E224 val32/top5 result：

| run | scorer | full match@1 | full match@5 | RMSE@1 | RMSE@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E222 | raw | 21.88% | 37.50% | 0.1469 | 0.1789 | 5.56% | 16.67% | 81.25% | 4.38 |
| E224 | global self-score | 34.38% | 37.50% | 0.1579 | 0.1789 | 16.67% | 16.67% | 81.25% | 4.38 |

Same-prefix reference：

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% | 78.12% | 2.88 |
| E224 tight residual | 34.38% | 37.50% | 16.67% | 16.67% | 81.25% | 4.38 |

判断：

- E214 解决了训练侧 source-pair 构建不可复用的问题；缓存可审计、只含 train/val，并记录 no-test/no-StructureMatcher-label leakage guard。
- 更紧 residual constraints 明显优于早期 E213 residual (`25.00/25.00`)，并恢复了 first-32 match@1。
- 但 E224 match@5 仍低于 E166 first-32，rows>=7 也低于 E166；不能扩大到 val128 或 full test。
- W/A@5 与 unique W/A@5 仍高，说明失败仍是 geometry/source-mode compatibility，而不是 symbolic W/A recall。
- 下一步如果继续 residual 路线，应缓存 inference-time predicted-W/A source pairs，并训练 source-mode/no-move 或 mixture objective；不要继续简单 one-source residual regression。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_build_source_residual_examples.py \
  model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E214 summary、E215 training summary、E224 summary TSV 均存在且非空。

## 55. E205-E213 - source-conditioned residual geometry smoke

时间：2026-06-14 UTC

目的：

- 针对 E166/E197/E203 均显示的 geometry/source compatibility bottleneck，新增一个模型侧 residual geometry 路线。
- 训练目标不是 CIF next-token，也不是 RF/ranker；输入为 formula+GT-SG+canonical W/A rows + train-source aligned params/lattice，输出 residual free params/lattice。
- 训练只用 train split；val 只用于模型选择/评估；不使用 test。

新增脚本：

- `model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py`
- `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`

方法：

- 对每个 train target，从 train split 中选一个非自身 source。
- source selector 使用 bounded train-only chemical/row/source-quality features：
  - SG/row-count、SG/atom-bucket、SG、crystal-system/atom-bucket、row-conditioned kNN；
  - 每个 pool 先按 cheap chemical distance 限制，再做 row/chem alignment scoring；
  - 不使用 StructureMatcher label、test GT、GT-WA input。
- base geometry = source-aligned free params + source VPA soft lattice。
- neural residual model 输入：
  - formula weighted element embedding；
  - SG embedding；
  - canonical W/A row embeddings；
  - base per-row free params/masks；
  - base lattice normalized values；
  - source distance / alignment cost。
- 输出：
  - lattice residual；
  - row free-param residual。

Training attempts：

| run | setting | status / result |
|---|---|
| E205 | 512 train / 128 val / 1 epoch | pipeline passes; val loss `16.5889`, val lattice loss `16.3003` |
| E206 | 4k train / 512 val / 5 epoch / slow selector | interrupted after >2 min before training header |
| E207 | 2k train / 512 val / 5 epoch / slow selector | interrupted after >90 s before training header |
| E208 | 2k train / 512 val / 5 epoch / fast selector | interrupted after >90 s before training header |
| E209 | 1k train / 256 val / 5 epoch / fast selector | completed; best val loss `9.2103`, val lattice loss `8.9151`, val coord loss `0.0738` |

E210 val128 render attempt：

- `E209` checkpoint with sequential residual renderer and top20 was interrupted after >2 min with no output file.
- 原因是 per-candidate source selection/render still too slow without source-pair caching/parallel workers.

E210b-E213 val32/top5 smoke：

| item | value |
|---|---:|
| rows | first 32 val predictions |
| rendered samples | 31 / 32 |
| rendered candidates | 140 |
| readable / formula / atom / composition exact overall | 95.71% / 95.71% / 95.71% / 95.71% |
| SG-ok overall / rank1 | 94.29% / 93.55% |

StructureMatcher, raw E210b:

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | skeleton@5 | W/A@5 | unique W/A@5 | composition exact all@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val32 | 15.62% | 25.00% | 0.2922 | 0.2736 | 84.38% | 81.25% | 4.38 | 90.62% |
| rows>=7 | 5.56% | 5.56% | 0.3770 | 0.3770 | 83.33% | 83.33% | 4.56 | 88.89% |

StructureMatcher, self-scored E213:

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | skeleton@5 | W/A@5 | unique W/A@5 | composition exact all@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full val32 | 25.00% | 25.00% | 0.3035 | 0.2736 | 84.38% | 81.25% | 4.38 | 90.62% |
| rows>=7 | 5.56% | 5.56% | 0.3770 | 0.3770 | 83.33% | 83.33% | 4.56 | 88.89% |

E166 first-32 reference from existing per-sample metrics:

| subset | match@1 | match@5 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|
| full val32 | 34.38% | 43.75% | 81.25% | 78.12% | 2.88 |
| rows>=7 | 22.22% | 22.22% | 83.33% | 83.33% | 2.72 |

判断：

- Source-conditioned residual model branch is implemented and leakage-safe, but the current 1k smoke is weak.
- It preserves/improves W/A@5 and unique W/A@5 on val32, but geometry match collapses versus E166 first-32.
- Composition/readability are also worse than E166, indicating unconstrained residual coordinates create invalid/overlapping expansions.
- Do not expand E209 to val128/full.
- Next residual-geometry work requires:
  - cached train/val source-pair examples to make >2k training practical;
  - composition/readability-preserving render constraints or residual clipping;
  - likely train source-pair hard negatives, not current one-source regression.
- Current best remains E166/E181. Full test remains closed.

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py
```

结果：compile 通过。

## 54. E199-E204 - batched geometry-net lattice diagnostic

时间：2026-06-14 UTC

目的：

- 重新检查 E58 曾经中断的 lattice-only geometry-net diagnostic。
- 保持 W/A candidates、source-aligned free params 和 self-score protocol 与 E166 方向一致，仅把 lattice 从 `source_vpa_calibrated_soft` 换成 train-only `geometry_net` lattice。
- 仍只跑 val128 K20 smoke，不运行 full test。

E199：

- 原始 per-candidate geometry-net lattice inference 超过约 95 秒仍无输出文件。
- 手动中断。
- 结论：旧实现的 E58 速度问题仍存在。

实现修复：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`。
- 新增 worker-local geometry-net lattice cache。
- 在 `process_payload` 开始时，对同一样本的前 `top_k` W/A candidates 批量做 lattice inference。
- `render_from_source` 后续按 `(sample_id, canonical_wa_key)` 读取缓存 lattice。
- 该改动只影响 `--geometry-lattice-mode geometry_net`，不改变 E166/E181 默认 source-VPA 路径。

E200 batched render：

| item | value |
|---|---:|
| split / records | val / 128 |
| top_k / geometry ranks | 20 / 10 |
| rendered rows | 2,398 |
| samples with rendered candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 95.33% / 92.86% / 96.19% / 95.33% |

E203/E204 StructureMatcher val128：

| run | scorer | full match@1 | full match@5 | full match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5 | rows>=7 match@20 | W/A@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 | source VPA soft global | 32.03% | 40.62% | 44.53% | 0.0942 | 0.1295 | 0.1534 | 18.33% | 20.00% | 76.56% |
| E203 | geometry-net lattice global | 28.12% | 32.03% | 36.72% | 0.1185 | 0.1303 | 0.1520 | 8.33% | 10.00% | 77.34% |
| E204 | geometry-net lattice diverse | 28.12% | 32.03% | 36.72% | 0.1185 | 0.1322 | 0.1520 | 8.33% | 10.00% | 77.34% |

判断：

- Batching fixes the practical E58 runtime blocker for lattice-net inference.
- The existing E55c train-only geometry net lattice is harmful when paired with the strong source-aligned free-param path.
- W/A@20 stays comparable, while match and rows>=7 collapse; this is a lattice/geometry quality issue, not W/A recall.
- Do not adopt geometry-net lattice for current validation gate.
- If geometry learning is revisited, it should be a source-conditioned residual/free-param+lattice model, not direct mean lattice prediction.

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py
```

结果：compile 通过。

## 53. E194-E198 - GT-free physical source selection diagnostic

时间：2026-06-14 UTC

目的：

- 针对 E166 中大量 `W/A@5=true` 但 StructureMatcher 不匹配的问题，测试一个更局部的 joint source/free-param/lattice 选择策略。
- 不是 RF/ranker/fixed candidate fusion，也不使用 StructureMatcher label；只在 renderer 内部对同一 W/A 的若干 train source geometry 进行 GT-free generated-CIF health 选择。
- 仍只用 MPTS-52 val128 做 gate，不运行 full test。

实现：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`。
- 新增 `row_aligned_chem_physical_select`：
  - 复用 `row_aligned_chem_quality` 的 train-only source pool；
  - 对 `geometry_rank=0` 的 top source 取前 3 个候选；
  - 分别渲染后用 generated CIF 自身指标打分：readable/formula/atom/composition/SG、min-distance、volume-per-atom、geometry distance；
  - 选择 health score 最高的 source/free-param/lattice 组合。
- 该策略不读取 test，不使用 StructureMatcher match/rms，不使用 GT-WA input。

E194 render smoke：

| item | value |
|---|---:|
| split / records | val / 128 |
| top_k / geometry ranks | 20 / 10 |
| rendered rows | 2,398 |
| samples with rendered candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 96.21% / 92.06% / 96.03% / 96.21% |

E195/E196 GT-free self-score：

| run | mode | rank1 SG-ok | rank1 min-distance mean | rank<=5 SG-ok | rank<=5 min-distance mean |
|---|---|---:|---:|---:|---:|
| E195 | global | 100.00% | 1.8368 | 99.52% | 1.7393 |
| E196 | diverse | 100.00% | 1.8368 | 99.37% | 1.6677 |

E197/E198 StructureMatcher val128：

| run | scorer | full match@1 | full match@5 | full match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5 | rows>=7 match@20 | W/A@20 | unique W/A@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 | chem + soft VPA global | 32.03% | 40.62% | 44.53% | 0.0942 | 0.1295 | 0.1534 | 18.33% | 20.00% | 76.56% | 6.85 |
| E197 | physical-select global | 30.47% | 34.38% | 39.84% | 0.1050 | 0.1123 | 0.1410 | 16.67% | 20.00% | 77.34% | 9.98 |
| E198 | physical-select diverse | 30.47% | 33.59% | 39.84% | 0.1050 | 0.1180 | 0.1410 | 15.00% | 20.00% | 77.34% | 9.98 |

判断：

- Physical source selection improves generated-CIF health at early ranks, especially rank1 SG-ok and min-distance.
- However, validation StructureMatcher match drops sharply versus E166, especially full match@5 (`40.62% -> 34.38%`) and match@20 (`44.53% -> 39.84%`).
- W/A@20 and unique W/A@20 are not worse, so the failure is geometry/source compatibility rather than W/A recall.
- Do not expand this branch to K100 or full test.
- Current best remains E166/E181:
  - full validation match@1/5/20/50 = `32.03 / 40.62 / 44.53 / 45.31`;
  - only match@1 clears CrystaLLM GT-SG +5pp target;
  - full test remains closed.

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py
```

结果：compile 通过。

## 50. E173-E182 - soft-VPA coherent source-mode allocation check

时间：2026-06-14 UTC

目的：

- 在当前 best K5 configuration `row_aligned_chem_quality + source_vpa_calibrated_soft` 上，检查是否需要给前排 W/A 分配少量额外 coherent train-source geometry modes。
- 这不是 RF/ranker/test fusion；仍只使用 E84/E85 model-side W/A candidates、train-only source geometry 和 GT-free self-score。
- 只跑 val128，不跑 full test。

E173 hybrid5x5 render：

| item | value |
|---|---:|
| geometry plan | `hybrid_top_wa` |
| hybrid W/A / geometry ranks | 5 / 5 |
| rendered rows | 9,141 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.75% / 91.27% / 91.27% / 94.35% |

E174/E176 hybrid5x5 global：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 30.47% | 40.62% | 45.31% | 45.31% | 0.0864 | 0.1298 | 0.1558 | 0.1551 | 76.56% | 10.35 |
| rows>=7 | 11.67% | 18.33% | 20.00% | 20.00% | 0.1460 | 0.1423 | 0.1705 | 0.1705 | 75.00% | 6.72 |
| atoms>=12 | 24.14% | 34.48% | 39.66% | 39.66% | 0.1061 | 0.1650 | 0.1933 | 0.1926 | 74.14% | 9.88 |
| complex_flag | 25.21% | 36.13% | 41.18% | 41.18% | 0.1050 | 0.1552 | 0.1829 | 0.1822 | 74.79% | 10.14 |

E175/E177 hybrid5x5 diverse：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 30.47% | 37.50% | 42.97% | 45.31% | 0.0864 | 0.1131 | 0.1462 | 0.1551 | 78.91% | 13.23 |
| rows>=7 | 11.67% | 16.67% | 20.00% | 20.00% | 0.1460 | 0.1376 | 0.1705 | 0.1705 | 76.67% | 7.33 |

E178 hybrid2x2 render：

| item | value |
|---|---:|
| geometry plan | `hybrid_top_wa` |
| hybrid W/A / geometry ranks | 2 / 2 |
| rendered rows | 9,141 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.73% / 92.06% / 94.76% / 95.38% |

E179/E181 hybrid2x2 global：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 40.62% | 44.53% | 45.31% | 0.0938 | 0.1295 | 0.1534 | 0.1549 | 76.56% | 10.56 |
| rows>=7 | 15.00% | 18.33% | 20.00% | 20.00% | 0.1636 | 0.1423 | 0.1705 | 0.1705 | 75.00% | 6.80 |
| atoms>=12 | 25.86% | 34.48% | 38.79% | 39.66% | 0.1137 | 0.1644 | 0.1912 | 0.1923 | 74.14% | 10.03 |
| complex_flag | 26.89% | 36.13% | 40.34% | 41.18% | 0.1122 | 0.1548 | 0.1807 | 0.1819 | 74.79% | 10.31 |

E180/E182 hybrid2x2 diverse：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 39.06% | 42.97% | 45.31% | 0.0938 | 0.1219 | 0.1463 | 0.1551 | 78.91% | 13.23 |
| rows>=7 | 15.00% | 18.33% | 20.00% | 20.00% | 0.1636 | 0.1448 | 0.1705 | 0.1705 | 76.67% | 7.33 |

E166 diagnostic on geometry vs symbolic bottleneck:

| subset | budget | match hits | W/A hits | skeleton hits | W/A-hit but no match | match without W/A |
|---|---:|---:|---:|---:|---:|---:|
| full | 5 | 52 | 88 | 94 | 44 | 8 |
| full | 50 | 58 | 98 | 102 | 45 | 5 |
| rows>=7 | 5 | 11 | 42 | 45 | 32 | 1 |
| rows>=7 | 50 | 12 | 45 | 48 | 34 | 1 |

Additional K5/K50 diagnostic:

- E166 has 6 samples that miss match@5 but hit match@50.
- Among those 6, only 1 is rows>=7.
- E166 has 44 samples with W/A@5 true but match@5 false; 32 of those are rows>=7.

判断：

- hybrid5x5 is too aggressive:
  - K5 stays at `40.62%`, but match@1 drops from `32.03%` to `30.47%`;
  - diverse K5 drops to `37.50%`.
- hybrid2x2 is a near no-op:
  - global exactly ties E166 on match (`32.03 / 40.62 / 44.53 / 45.31`) with slightly better RMSE@1 (`0.0938` vs `0.0942`);
  - it does not move K5 over the `41.58%` gate.
- The diagnostic confirms the next bottleneck:
  - W/A@5 is already much higher than match@5;
  - rows>=7 has many W/A-hit/no-match cases;
  - source allocation alone does not fix free-param/source-mode compatibility.
- Full-test gate remains closed.
- Next work should be a real train-only source-conditioned free-param / joint lattice-free-param proposal, not more hybrid scheduling.

## 51. E183-E189 - train-only free-param manifold source penalty

时间：2026-06-14 UTC

目的：

- 针对 E166 诊断中的 W/A-hit/no-match 问题，尝试让 chemical source selection 关注 free-param compatibility。
- 新增 train-only free-param manifold penalty：
  - 从 train records 中按 `(SG, orbit_id, element, free_symbols, symbol)` 等层级收集 free-param 分布；
  - source selection 时，把 target row 对齐到 source row 后，检查将复制的 free-param 是否接近 train 中同类参数流形；
  - 使用 circular distance，不使用 StructureMatcher、test GT、validation match label 或 GT-WA input。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`。
- 新增 `--geometry-source-strategy row_aligned_chem_param_quality`。
- 首版 E183/E184 full K100 尝试过慢，已中断，不计入指标。
- 性能修正：
  - train free-param buckets 初始化后排序；
  - nearest lookup 改为 binary search；
  - param penalty 只对 chem score top-N source 做二阶段 refinement。

E185 top20 smoke render：

| item | value |
|---|---:|
| top_k / geometry ranks | 20 / 10 |
| rendered rows | 2,394 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 95.41% / 92.06% / 95.87% / 95.41% |

E186/E188 global self-score/result:

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@20 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 29.69% | 36.72% | 40.62% | 0.1002 | 0.1336 | 0.1632 | 77.34% | 9.69 |
| rows>=7 | 13.33% | 16.67% | 18.33% | 0.1294 | 0.1120 | 0.1454 | 76.67% | 6.45 |
| atoms>=12 | 23.28% | 31.03% | 34.48% | 0.1098 | 0.1550 | 0.1830 | 75.00% | 9.23 |
| complex_flag | 24.37% | 31.93% | 36.13% | 0.1234 | 0.1630 | 0.1954 | 75.63% | 9.48 |

E187/E189 diverse self-score/result:

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@20 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 29.69% | 37.50% | 40.62% | 0.1002 | 0.1445 | 0.1632 | 77.34% | 9.69 |
| rows>=7 | 13.33% | 18.33% | 18.33% | 0.1294 | 0.1487 | 0.1454 | 76.67% | 6.45 |
| atoms>=12 | 23.28% | 31.90% | 34.48% | 0.1098 | 0.1648 | 0.1830 | 75.00% | 9.23 |
| complex_flag | 24.37% | 32.77% | 36.13% | 0.1234 | 0.1721 | 0.1954 | 75.63% | 9.48 |

对比当前 best E166：

| metric | E166 soft-VPA global | E188 param-quality global | E189 param-quality diverse |
|---|---:|---:|---:|
| full match@1 | 32.03% | 29.69% | 29.69% |
| full match@5 | 40.62% | 36.72% | 37.50% |
| full match@20 | 44.53% | 40.62% | 40.62% |
| rows>=7 match@5 | 18.33% | 16.67% | 18.33% |
| rows>=7 match@20 | 20.00% | 18.33% | 18.33% |

判断：

- The train-only param manifold penalty is compliant but negative.
- It appears to over-prefer common/free-param-manifold-compatible sources at the expense of target W/A/source geometry compatibility:
  - full W/A@1 drops from E166 `53.91%` to `49.22%`;
  - full match@5 drops by 3-4 pp.
- Do not expand this strategy to K100/full val or full test.
- Keep the code branch for reproducibility, but current active geometry default remains E166/E181:
  - `row_aligned_chem_quality + source_vpa_calibrated_soft + global self-score`.
- Next attempt should not use a scalar param prior alone; it needs a joint source/free-param proposal that preserves W/A ranking and source compatibility.

## 52. E190-E193 - strict physical GT-free self-score diagnostic

时间：2026-06-14 UTC

目的：

- E166 诊断显示 W/A-hit/no-match 样本具有更短 `self_min_distance`、更低 VPA、更高 `geometry_distance`。
- 在不使用 StructureMatcher label 排序的前提下，新增一个更强的 candidate-internal physical self-score profile：
  - 更强惩罚短距离；
  - 更强惩罚明显过低/过高 volume-per-atom；
  - 不改 renderer、不改 W/A candidates、不用 test。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py`。
- 新增 `--score-profile strict_physical`。
- 默认 `standard` 不变，因此旧实验可复现。

E190 strict-physical global self-score over E163：

| item | standard E164 | strict E190 |
|---|---:|---:|
| rank1 min-distance mean | 1.8698 | 2.0190 |
| rank<=5 min-distance mean | 1.8065 | 1.9145 |
| rank1 VPA mean | 17.1840 | 17.4006 |
| rank<=5 SG-ok | 100.00% | 100.00% |

E191 strict-physical diverse self-score over E163：

| item | standard E165 | strict E191 |
|---|---:|---:|
| rank1 min-distance mean | 1.8698 | 2.0190 |
| rank<=5 min-distance mean | 1.7361 | 1.8384 |
| rank1 VPA mean | 17.1840 | 17.4006 |
| rank<=5 SG-ok | 99.37% | 99.37% |

E192/E193 StructureMatcher result：

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@5 | rows>=7 match@20/50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 | standard global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 18.33% | 20.00% / 20.00% |
| E192 | strict global | 25.00% | 35.94% | 45.31% | 45.31% | 0.1107 | 0.1218 | 16.67% | 20.00% / 20.00% |
| E193 | strict diverse | 25.00% | 39.06% | 44.53% | 45.31% | 0.1107 | 0.1361 | 15.00% | 20.00% / 20.00% |

判断：

- strict physical profile improves internal min-distance/VPA surrogate but damages W/A/skeleton early ranking:
  - full W/A@1 drops from E166 `53.91%` to `42.19%`;
  - full match@1 drops from `32.03%` to `25.00%`;
  - match@5 remains below E166.
- It can recover K20 in global mode but only by sacrificing early ranks; this does not satisfy the target.
- Do not adopt strict physical self-score as default.
- The negative result reinforces the same conclusion:
  - candidate-internal physical filtering alone cannot solve W/A-to-match conversion;
  - next work needs joint source/free-param/lattice proposal that preserves W/A ranking.

## 49. E163-E172 - train-only source VPA calibrated chemical geometry

时间：2026-06-14 UTC

目的：

- 沿 E153-E162 的 `row_aligned_chem_quality` 继续改进 geometry precision。
- 用 train-only chemical analogue 估计目标结构的 expected volume-per-atom (VPA)，对 source lattice 做软/硬体积校准。
- 仍保持 source consistency：lattice shape 和 free params 来自同一个 train source，只对 lattice 体积尺度做 train-only calibrated rescale。
- 不使用 test、StructureMatcher label、GT W/A、RF ranker 或 CrystaLLM candidate source。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`。
- 新增 `cell_volume` import 与 train-only VPA 统计：
  - `(SG, atom_bucket)`；
  - `(crystal_system, atom_bucket)`；
  - `atom_bucket`；
  - global train fallback。
- 新增 `record_volume_per_atom()`、`expected_vpa()`、`vpa_calibrated_lattice()`。
- 新增 `--geometry-lattice-mode`：
  - `source_vpa_calibrated_soft`：strength 0.5；
  - `source_vpa_calibrated`：strength 1.0。

E163 soft-VPA render：

| item | value |
|---|---:|
| rendered rows | 9,141 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.73% / 92.06% / 95.87% / 95.30% |

E164/E166 soft-VPA global：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 0.1534 | 0.1549 | 76.56% | 10.59 |
| rows>=7 | 15.00% | 18.33% | 20.00% | 20.00% | 0.1636 | 0.1423 | 0.1705 | 0.1705 | 75.00% | 6.80 |
| atoms>=12 | 25.86% | 34.48% | 38.79% | 39.66% | 0.1143 | 0.1644 | 0.1912 | 0.1923 | 74.14% | 10.04 |
| complex_flag | 26.89% | 36.13% | 40.34% | 41.18% | 0.1127 | 0.1548 | 0.1807 | 0.1819 | 74.79% | 10.34 |

E165/E167 soft-VPA diverse：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 39.06% | 42.97% | 45.31% | 0.0942 | 0.1219 | 0.1466 | 0.1551 | 78.91% | 13.23 |
| rows>=7 | 15.00% | 18.33% | 20.00% | 20.00% | 0.1636 | 0.1448 | 0.1705 | 0.1705 | 76.67% | 7.33 |
| atoms>=12 | 25.86% | 32.76% | 37.07% | 39.66% | 0.1143 | 0.1495 | 0.1796 | 0.1925 | 76.72% | 12.37 |
| complex_flag | 26.89% | 34.45% | 38.66% | 41.18% | 0.1127 | 0.1437 | 0.1725 | 0.1821 | 77.31% | 12.79 |

E168 hard-VPA render：

| item | value |
|---|---:|
| rendered rows | 9,141 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.82% / 92.06% / 95.87% / 95.34% |

E169/E171 hard-VPA global：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 39.84% | 44.53% | 45.31% | 0.0969 | 0.1252 | 0.1540 | 0.1553 | 76.56% | 10.55 |
| rows>=7 | 13.33% | 18.33% | 20.00% | 20.00% | 0.1387 | 0.1423 | 0.1705 | 0.1705 | 75.00% | 6.80 |
| atoms>=12 | 25.86% | 33.62% | 38.79% | 39.66% | 0.1179 | 0.1598 | 0.1919 | 0.1929 | 74.14% | 10.03 |
| complex_flag | 26.89% | 35.29% | 40.34% | 41.18% | 0.1161 | 0.1503 | 0.1813 | 0.1825 | 74.79% | 10.30 |

E170/E172 hard-VPA diverse：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 37.50% | 42.19% | 45.31% | 0.0969 | 0.1154 | 0.1417 | 0.1556 | 78.91% | 13.23 |
| rows>=7 | 13.33% | 18.33% | 20.00% | 20.00% | 0.1387 | 0.1657 | 0.1705 | 0.1705 | 76.67% | 7.33 |
| atoms>=12 | 25.86% | 31.03% | 36.21% | 39.66% | 0.1179 | 0.1424 | 0.1742 | 0.1931 | 76.72% | 12.37 |
| complex_flag | 26.89% | 32.77% | 37.82% | 41.18% | 0.1161 | 0.1368 | 0.1672 | 0.1827 | 77.31% | 12.79 |

对比：

| metric | E157 chem/diverse | E156 chem/global | E166 soft-global | E171 hard-global |
|---|---:|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | 32.03% | 32.03% |
| full match@5 | 39.84% | 39.06% | 40.62% | 39.84% |
| full match@20 | 44.53% | 45.31% | 44.53% | 44.53% |
| full match@50 | 45.31% | 45.31% | 45.31% | 45.31% |
| full RMSE@1 | 0.1100 | 0.1100 | 0.0942 | 0.0969 |
| rows>=7 match@5 | 18.33% | 18.33% | 18.33% | 18.33% |
| rows>=7 match@50 | 20.00% | 20.00% | 20.00% | 20.00% |

判断：

- `source_vpa_calibrated_soft + global` 是当前 validation K5 best：
  - full match@1/5/20/50 = `32.03 / 40.62 / 44.53 / 45.31`；
  - match@5 距离 `41.58%` target 还差 `0.96 pp`。
- soft-VPA 显著改善 RMSE@1 (`0.0942`) 并恢复 E151 的 rank1，同时保留 chemical analogue 的 K5 增益。
- hard-VPA 校准过强，K5 回落，不采用。
- rows>=7 match 仍卡在 `15.00 / 18.33 / 20.00 / 20.00`，说明 remaining bottleneck 仍是复杂子集 geometry/free-param 转化，而不是简单体积尺度。
- Full-test gate 仍关闭：
  - match@1 已过 `31.64%`；
  - match@5 未过 `41.58%`；
  - match@20 未过 `49.69%`。
- 下一步应保留 soft-VPA 作为 geometry default 候选，并继续做 rows>=7/source-conditioned free-param proposal；不要因为 K5 接近目标就启动 full test。

## 46. E137-E140 - train-only volume-per-atom prior self-score diagnostic

时间：2026-06-13 UTC

目的：

- 诊断 E131/E134 中 rows>=7 已命中 W/A 但 CIF 仍未 match 的样本。
- 在不使用 StructureMatcher label 训练/排序、不使用 test 的前提下，测试一个 train-only volume-per-atom prior 是否能改善 geometry self-score 排序。
- 该实验只在 validation 128 subset 上评估，不触发 full test。

Rows>=7 诊断摘要：

| source | group | n | mean candidates | unique W/A | SG-ok | source-good | mean min-distance | mean VPA | mean geometry distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E131 | W/A@50 and match@50 | 10 | 26.20 | 5.60 | 98.20% | 94.60% | 1.3126 | 19.04 | 2.28 |
| E131 | W/A@50 but no match@50 | 36 | 27.53 | 5.89 | 97.53% | 92.34% | 1.1954 | 15.65 | 2.66 |
| E131 | no W/A@50 and no match@50 | 13 | 33.31 | 11.85 | 90.88% | 72.53% | 0.8057 | 15.00 | 4.99 |
| E134 | W/A@50 and match@50 | 10 | 38.40 | 5.60 | 100.00% | 96.60% | 1.3504 | 19.07 | 2.77 |
| E134 | W/A@50 but no match@50 | 35 | 41.26 | 5.71 | 99.71% | 93.09% | 1.2697 | 16.03 | 2.89 |
| E134 | no W/A@50 and no match@50 | 14 | 42.64 | 10.57 | 97.44% | 73.54% | 1.0377 | 16.39 | 5.10 |

判断：

- W/A-hit 但 no-match 的 rows>=7 样本通常仍有较好的 SG/source health；失败更多来自 geometry scale/free-param compatibility。
- no-W/A 样本 source-good 明显更差，说明 symbolic/source quality 仍是另一类失败，但不是本次 VPA prior 能解决的问题。
- W/A-hit no-match 的 VPA 低于 W/A-hit match，因此可以用 train-only VPA prior 做一个 GT-free 排序 smoke。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py`。
- 新增参数：
  - `--train-jsonl`
  - `--train-volume-prior-weight`
- prior 只从 MPTS-52 train structured JSONL 统计：
  - `(space_group, atom_bucket)`
  - `(crystal_system, atom_bucket)`
  - `atom_bucket`
  - global
- 默认 weight 为 `0.0`，因此旧实验可复现。
- 输出新增 `self_train_volume_prior_score` 和 summary 中的 train prior metadata。

E137 self-score：

| item | value |
|---|---:|
| source rendered pool | E132 K100/g10 row_aligned_quality |
| mode | global |
| top_k | 50 |
| train VPA prior weight | 2.0 |
| train VPA buckets | 359 |
| output rows | 5,569 |
| rank1 / rank<=5 / rank<=20 SG-ok | 100.00% / 100.00% / 99.50% |
| rank1 / overall VPA mean | 18.62 / 17.80 |

E138 validation result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 30.47% | 35.94% | 42.19% | 45.31% | 0.1011 | 0.1131 | 0.1394 | 0.1526 | 76.56% | 10.54 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1284 | 0.1225 | 0.1450 | 0.1450 | 75.00% | 6.80 |
| atoms>=12 | 25.86% | 29.31% | 36.21% | 39.66% | 0.1031 | 0.1309 | 0.1663 | 0.1806 | 74.14% | 10.05 |
| complex_flag | 26.05% | 31.09% | 37.82% | 41.18% | 0.1040 | 0.1357 | 0.1652 | 0.1788 | 74.79% | 10.33 |

E139 self-score：

| item | value |
|---|---:|
| source rendered pool | E132 K100/g10 row_aligned_quality |
| mode | global |
| top_k | 50 |
| train VPA prior weight | 0.5 |
| train VPA buckets | 359 |
| output rows | 5,569 |
| rank1 / rank<=5 / rank<=20 SG-ok | 100.00% / 100.00% / 99.50% |
| rank1 / overall VPA mean | 19.09 / 17.92 |

E140 validation result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 42.19% | 45.31% | 0.1056 | 0.1011 | 0.1394 | 0.1526 | 76.56% | 10.66 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1284 | 0.1225 | 0.1450 | 0.1450 | 75.00% | 6.83 |
| atoms>=12 | 25.86% | 28.45% | 36.21% | 39.66% | 0.1037 | 0.1172 | 0.1663 | 0.1806 | 74.14% | 10.12 |
| complex_flag | 26.05% | 30.25% | 37.82% | 41.18% | 0.1046 | 0.1229 | 0.1652 | 0.1788 | 74.79% | 10.44 |

对比当前 best：

| metric | E131 best early | E134 best ceiling | E138 VPA w2 | E140 VPA w0.5 |
|---|---:|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | 30.47% | 31.25% |
| full match@5 | 35.94% | 35.16% | 35.94% | 35.16% |
| full match@20 | 41.41% | 42.19% | 42.19% | 42.19% |
| full match@50 | 44.53% | 45.31% | 45.31% | 45.31% |
| rows>=7 match@5 | 16.67% | 15.00% | 15.00% | 15.00% |
| rows>=7 match@50 | 18.33% | 18.33% | 18.33% | 18.33% |

判断：

- Train-only VPA prior 是合规的 GT-free 排序信号，但没有提升 validation gate。
- weight 2.0 恢复 E131 的 K5 并保持 E134 的 K20/K50，但损失 match@1。
- weight 0.5 基本等同 E134，只小幅改善 RMSE@5。
- rows>=7 match@50 仍固定在 `18.33%`，说明体积 prior 不是复杂子集的关键瓶颈。
- 当前 best split 不变：
  - E131 是 best early/K5 validation result；
  - E134 是 best K20/K50 validation ceiling；
  - E138/E140 不进入 freeze。
- Full test gate 仍关闭；下一步应提升 rows>=7 source-conditioned geometry generation capacity，而不是继续调 VPA/scorer。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py

test -s model/New_model/opentry_3/reports/e138_eval_e137_quality_k100_global_train_vpa_w2_val128_match/summary_metrics.tsv
test -s model/New_model/opentry_3/reports/e140_eval_e139_quality_k100_global_train_vpa_w05_val128_match/summary_metrics.tsv
```

结果：compile 与产物存在性检查均通过。

## 47. E141-E152 - source-conditioned hybrid geometry allocation

时间：2026-06-13 UTC

目的：

- 检查 E132/E134 的 K100 `wa_diverse` 计划是否过度优先 W/A diversity，而没有给前排 high-confidence W/A 足够多的 coherent source geometry modes。
- 保持同一 E84 neural-first W/A candidates 和 `row_aligned_quality` source strategy。
- 只改变 geometry allocation：
  - E141: top10 W/A each gets up to 10 geometry ranks (`hybrid10x10`)；
  - E148: top5 W/A each gets up to 5 geometry ranks (`hybrid5x5`)。
- 排序仍使用 GT-free self-score；StructureMatcher 只用于 val128 evaluation。
- 不使用 test、StructureMatcher label 训练/排序、RF ranker 或 CrystaLLM candidate source。

注意：

- E144/E145 首次 evaluator 命令漏了 `--max-records 128`，分母变成 full val，不可与 E131/E134 比较；该结果作废，不纳入结论。
- 正式可比结果为 E146/E147/E151/E152。

E141 render (`hybrid10x10`)：

| item | value |
|---|---:|
| top_k / geometry ranks | 100 / 10 |
| hybrid W/A / ranks | 10 / 10 |
| rendered rows | 9,013 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.90% / 91.27% / 89.98% / 92.28% |

E142/E146 `hybrid10x10` global self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 28.12% | 32.81% | 39.06% | 42.19% | 0.0979 | 0.0921 | 0.1314 | 0.1351 | 74.22% | 7.45 |
| rows>=7 | 11.67% | 15.00% | 18.33% | 18.33% | 0.1405 | 0.1225 | 0.1450 | 0.1450 | 75.00% | 6.53 |
| atoms>=12 | 21.55% | 25.86% | 32.76% | 36.21% | 0.0845 | 0.1101 | 0.1592 | 0.1701 | 71.55% | 7.45 |
| complex_flag | 22.69% | 27.73% | 34.45% | 37.82% | 0.0951 | 0.1138 | 0.1583 | 0.1604 | 72.27% | 7.50 |

E143/E147 `hybrid10x10` diverse self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 28.12% | 32.03% | 38.28% | 42.19% | 0.0979 | 0.0974 | 0.1263 | 0.1455 | 76.56% | 8.66 |
| rows>=7 | 11.67% | 13.33% | 18.33% | 18.33% | 0.1405 | 0.1284 | 0.1516 | 0.1450 | 76.67% | 7.03 |
| atoms>=12 | 21.55% | 25.00% | 31.90% | 36.21% | 0.0845 | 0.0979 | 0.1513 | 0.1743 | 74.14% | 8.53 |
| complex_flag | 22.69% | 26.89% | 33.61% | 37.82% | 0.0951 | 0.1069 | 0.1512 | 0.1724 | 74.79% | 8.67 |

判断 E141-E147：

- `hybrid10x10` 太激进：多 geometry ranks 提前后，W/A diversity 和 SG stability 均下降。
- Full match@1/5/20/50 全面低于 E131/E134。
- rows>=7 match@50 仍为 `18.33%`，没有解决核心瓶颈。

E148 render (`hybrid5x5`)：

| item | value |
|---|---:|
| top_k / geometry ranks | 100 / 10 |
| hybrid W/A / ranks | 5 / 5 |
| rendered rows | 9,013 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.86% / 91.27% / 90.30% / 94.15% |

E149/E151 `hybrid5x5` global self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 35.94% | 42.19% | 45.31% | 0.1109 | 0.1064 | 0.1393 | 0.1530 | 76.56% | 10.41 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1284 | 0.1225 | 0.1450 | 0.1450 | 75.00% | 6.80 |
| atoms>=12 | 25.86% | 29.31% | 36.21% | 39.66% | 0.1037 | 0.1272 | 0.1665 | 0.1814 | 74.14% | 9.95 |
| complex_flag | 26.89% | 31.09% | 37.82% | 41.18% | 0.1115 | 0.1291 | 0.1654 | 0.1795 | 74.79% | 10.24 |

E150/E152 `hybrid5x5` diverse self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 32.03% | 35.16% | 40.62% | 43.75% | 0.1109 | 0.1114 | 0.1320 | 0.1459 | 78.91% | 13.02 |
| rows>=7 | 13.33% | 13.33% | 18.33% | 18.33% | 0.1284 | 0.1284 | 0.1516 | 0.1450 | 76.67% | 7.22 |
| atoms>=12 | 25.86% | 28.45% | 34.48% | 37.93% | 0.1037 | 0.1170 | 0.1520 | 0.1722 | 76.72% | 12.13 |
| complex_flag | 26.89% | 30.25% | 36.13% | 39.50% | 0.1115 | 0.1233 | 0.1549 | 0.1708 | 77.31% | 12.55 |

对比当前 best：

| metric | E131 early best | E134 ceiling best | E151 hybrid5x5/global |
|---|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | 32.03% |
| full match@5 | 35.94% | 35.16% | 35.94% |
| full match@20 | 41.41% | 42.19% | 42.19% |
| full match@50 | 44.53% | 45.31% | 45.31% |
| rows>=7 match@5 | 16.67% | 15.00% | 15.00% |
| rows>=7 match@50 | 18.33% | 18.33% | 18.33% |

判断：

- E151 是当前 best-combined val128 ordering：
  - full match@1/5/20/50 = `32.03 / 35.94 / 42.19 / 45.31`；
  - 它组合了 E131 的 K5、E134 的 K20/K50，并将 match@1 推过 `31.64%` 目标线。
- 但 E151 仍未达到 full-test freeze gate：
  - match@5 仍远低于 `41.58%`；
  - match@20 仍远低于 `49.69%`；
  - rows>=7 match@50 仍卡在 `18.33%`。
- Geometry allocation 的保守版本有助于 early rank，但没有提升 rows>=7 转化上限。
- 下一步不能只调 geometry plan；需要真正增强 rows>=7 的 source-conditioned lattice/free-param proposal 或让 W/A candidates 带 geometry compatibility 信号。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py

test -s model/New_model/opentry_3/reports/e151_eval_e149_hybrid5x5_global_val128_match/summary_metrics.tsv
test -s model/New_model/opentry_3/reports/e152_eval_e150_hybrid5x5_diverse_val128_match/summary_metrics.tsv
```

结果：compile 与有效产物存在性检查均通过。

## 48. E153-E162 - train-only chemical-analogue source-conditioned geometry

时间：2026-06-13 UTC

目的：

- 针对 rows>=7 geometry/source bottleneck，新增一个真正改变 train source proposal 的策略，而不是继续调 scalar self-score 或 rank-budget。
- 现有 e08 `row_condition_distance` 使用 exact formula overlap；当目标结构需要化学类似但元素不同的 train analogue 时，该距离可能过度惩罚 source。
- 新策略用元素周期表特征构建 formula chemistry distance，选择化学类似、SG/row/atom bucket 接近且 source health 好的 train sources。
- 仍然保持 source consistency：lattice 和 free params 都来自同一个 coherent train source；不做 independent lattice aggregation。
- 不使用 StructureMatcher label、test GT、test W/A、RF ranker 或 CrystaLLM prediction source。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`。
- 新增 `--geometry-source-strategy row_aligned_chem_quality`。
- 新增 GT-free source scoring components：
  - formula chemical vector：元素 `Z/period/group/electronegativity/atomic_radius` 的加权均值与方差；
  - `chem_distance(target, source)`；
  - `row_condition_chem_distance`：弱化 exact formula overlap，保留 SG、row_count、orbit、multiplicity、free-symbol 约束；
  - chemical row alignment cost：元素 mismatch 用 element-property distance 替代硬 exact mismatch。
- source pool 只来自 train records：
  - same `(SG, row_count)`；
  - same `(SG, atom_bucket)`；
  - same SG；
  - same `(crystal_system, atom_bucket)`；
  - train fallback。
- 首次 E153 运行失败：`OpentryGeometrySelector` 漏建 `by_sg` index。已修复并重跑；失败结果不计入指标。

E153 render (`row_aligned_chem_quality`, K100/g10, wa_diverse)：

| item | value |
|---|---:|
| rendered rows | 9,141 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.71% / 92.06% / 95.87% / 95.22% |

E154/E156 global self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 39.06% | 45.31% | 45.31% | 0.1100 | 0.1273 | 0.1554 | 0.1549 | 76.56% | 10.57 |
| rows>=7 | 13.33% | 18.33% | 20.00% | 20.00% | 0.1835 | 0.1423 | 0.1705 | 0.1705 | 75.00% | 6.82 |
| atoms>=12 | 25.86% | 32.76% | 39.66% | 39.66% | 0.1337 | 0.1600 | 0.1930 | 0.1923 | 74.14% | 9.97 |
| complex_flag | 26.05% | 34.45% | 41.18% | 41.18% | 0.1336 | 0.1534 | 0.1825 | 0.1819 | 74.79% | 10.32 |

E155/E157 diverse self-score/result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 39.84% | 44.53% | 45.31% | 0.1100 | 0.1332 | 0.1545 | 0.1551 | 78.91% | 13.23 |
| rows>=7 | 13.33% | 18.33% | 20.00% | 20.00% | 0.1835 | 0.1657 | 0.1705 | 0.1705 | 76.67% | 7.33 |
| atoms>=12 | 25.86% | 33.62% | 38.79% | 39.66% | 0.1337 | 0.1636 | 0.1882 | 0.1925 | 76.72% | 12.37 |
| complex_flag | 26.05% | 35.29% | 40.34% | 41.18% | 0.1336 | 0.1569 | 0.1808 | 0.1821 | 77.31% | 12.79 |

E158-E162 `row_aligned_chem_quality + hybrid5x5`：

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E161 | global | 31.25% | 39.06% | 45.31% | 45.31% | 18.33% | 20.00% | 10.39 |
| E162 | diverse | 31.25% | 39.84% | 44.53% | 45.31% | 18.33% | 20.00% | 13.23 |

对比上一轮 best：

| metric | E151 previous best-combined | E156 chem/global | E157 chem/diverse |
|---|---:|---:|---:|
| full match@1 | 32.03% | 31.25% | 31.25% |
| full match@5 | 35.94% | 39.06% | 39.84% |
| full match@20 | 42.19% | 45.31% | 44.53% |
| full match@50 | 45.31% | 45.31% | 45.31% |
| rows>=7 match@5 | 15.00% | 18.33% | 18.33% |
| rows>=7 match@20/50 | 18.33% / 18.33% | 20.00% / 20.00% | 20.00% / 20.00% |
| full unique W/A@50 | 10.41 | 10.57 | 13.23 |

判断：

- `row_aligned_chem_quality` 是本轮有效进展：
  - full match@5 从 `35.94%` 提升到 `39.84%`；
  - full match@20 从 `42.19%` 提升到 `45.31%`；
  - rows>=7 match@5 从 `16.67%`/`15.00%` 提升到 `18.33%`；
  - rows>=7 match@20/50 从 `18.33%` 提升到 `20.00%`。
- 代价：
  - match@1 从 E151 的 `32.03%` 回落到 `31.25%`；
  - RMSE 明显变差，尤其复杂子集，说明化学 analogue source 增加了粗匹配和 tail hits，但几何精细度不足。
- `chem + hybrid5x5` 没有优于纯 chem source selector，说明本轮收益来自 source proposal，而不是 geometry budget scheduling。
- Full-test gate 仍关闭：
  - match@1 只有 E151 过 `31.64%`；
  - best match@5 `39.84%` 仍低于 `41.58%`；
  - best match@20 `45.31%` 仍低于 `49.69%`。
- 下一步应沿 chemical analogue source proposal 继续改进 geometry precision，例如 source-conditioned lattice/free-param calibration，而不是回退到 RF ranker、low-LR CIF SFT 或 test-time fusion。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py

test -s model/New_model/opentry_3/reports/e156_eval_e154_chem_quality_global_val128_match/summary_metrics.tsv
test -s model/New_model/opentry_3/reports/e157_eval_e155_chem_quality_diverse_val128_match/summary_metrics.tsv
test -s model/New_model/opentry_3/reports/e161_eval_e159_chem_hybrid5x5_global_val128_match/summary_metrics.tsv
test -s model/New_model/opentry_3/reports/e162_eval_e160_chem_hybrid5x5_diverse_val128_match/summary_metrics.tsv
```

结果：compile 与产物存在性检查均通过。

## 45. E132-E136 - wider row_aligned_quality geometry pool

时间：2026-06-13 UTC

目的：

- 检查 E131 的 `row_aligned_quality` 是否受 geometry pool 宽度限制。
- 保持同一 E84 neural-first W/A candidates；不新增 CrystaLLM/test candidates。
- 渲染 K100/g10 后用 GT-free self-score 选 top50，再做 validation-only StructureMatcher。
- 不使用 test、StructureMatcher label 排序或 validation match 训练。

E132 render：

| item | value |
|---|---:|
| top_k / geometry ranks | 100 / 10 |
| rendered rows | 9,013 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.84% / 92.86% / 96.66% / 95.53% |

E133 global self-score：

| item | value |
|---|---:|
| output rows | 5,569 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 100.00% |
| rank<=20 SG-ok | 99.50% |
| unique-reduced behavior | lower W/A diversity than diverse mode |

E134 global result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 42.19% | 45.31% | 0.1056 | 0.1040 | 0.1394 | 0.1526 | 76.56% | 10.66 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1284 | 0.1242 | 0.1450 | 0.1450 | 75.00% | 6.83 |
| atoms>=12 | 25.86% | 28.45% | 36.21% | 39.66% | 0.1037 | 0.1172 | 0.1663 | 0.1806 | 74.14% | 10.13 |
| complex_flag | 26.05% | 30.25% | 37.82% | 41.18% | 0.1046 | 0.1266 | 0.1652 | 0.1788 | 74.79% | 10.45 |

E135/E136 diverse result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 34.38% | 39.84% | 43.75% | 0.1056 | 0.1033 | 0.1249 | 0.1455 | 78.91% | 13.02 |
| rows>=7 | 13.33% | 13.33% | 18.33% | 18.33% | 0.1284 | 0.1284 | 0.1516 | 0.1450 | 76.67% | 7.22 |
| atoms>=12 | 25.86% | 27.59% | 33.62% | 37.93% | 0.1037 | 0.1061 | 0.1442 | 0.1716 | 76.72% | 12.13 |
| complex_flag | 26.05% | 29.41% | 35.29% | 39.50% | 0.1046 | 0.1136 | 0.1477 | 0.1703 | 77.31% | 12.55 |

对比 E131：

| metric | E131 K50/global | E134 K100/global | E136 K100/diverse |
|---|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | 31.25% |
| full match@5 | 35.94% | 35.16% | 34.38% |
| full match@20 | 41.41% | 42.19% | 39.84% |
| full match@50 | 44.53% | 45.31% | 43.75% |
| rows>=7 match@5 | 16.67% | 15.00% | 13.33% |
| rows>=7 match@50 | 18.33% | 18.33% | 18.33% |
| unique W/A@50 | 12.80 | 10.66 | 13.02 |

Gain/loss diagnostic E134 vs E131：

| budget | full gained | full lost | rows>=7 gained | rows>=7 lost |
|---:|---:|---:|---:|---:|
| K1 | 0 | 0 | 0 | 0 |
| K5 | 0 | 1 | 0 | 1 |
| K20 | 2 | 1 | 0 | 0 |
| K50 | 1 | 0 | 0 | 0 |

判断：

- E134 是当前 K20/K50 validation ceiling best：
  - full match@20/50 = `42.19% / 45.31%`。
- E131 仍是 K5 best：
  - full match@5 = `35.94%`；
  - rows>=7 match@5 = `16.67%`。
- 更宽 geometry pool 可以补少量 tail match，但 global scoring 会牺牲 W/A diversity 和 K5。
- Diverse scoring 恢复 W/A diversity，但 match@5/20/50 均低于 E131/E134。
- 仍不能 freeze full test：
  - best match@1 `31.25%` < `31.64%`；
  - best match@5 `35.94%` < `41.58%`；
  - best match@20 `42.19%` < `49.69%`；
  - rows>=7 match@50 仍卡在 `18.33%`。
- 下一步不能只继续放大 pool；需要提高 rows>=7 的 coherent source-conditioned geometry proposal ceiling 或回到 symbolic side 增加 rows>=7 W/A/geometry-coupled candidates。

关键产物检查：

- `model/New_model/opentry_3/reports/e119_eval_e118_row_aligned_global_selfscore_val128_match/summary_metrics.tsv`
- `model/New_model/opentry_3/reports/e121_eval_e120_global_prefix20_diverse_tail_val128_match/summary_metrics.tsv`

结果：均存在且非空。

结论：

- 当前仍不冻结 full MPTS-52 test。
- opentry_3 的主要进展是：
  - symbolic W/A recall 已由 E85/E111 维持在高位；
  - source-consistent row alignment 把 validation match@1 推到 `31.25%`，接近但未过 +5pp target；
  - global self-score 把 validation match@50 推到 `43.75%`。
- 主要瓶颈已经从 W/A recall 转为 rows>=7 geometry/source/lattice proposal：
  - rows>=7 W/A@50 仍在 `70%+`；
  - rows>=7 match@50 只有 `18.33%`。
- 下一步应继续做 source-consistent lattice/free-param proposal 或 train-only generative geometry head；不应转向 full-test rerank、RF ranker、CrystaLLM candidate source、低 LR CIF SFT 或固定 K50 融合。

## 42. E122-E125 - train-only source-cluster lattice proposal

时间：2026-06-13 UTC

目的：

- 在 E111 `row_aligned_knn` 的 source-consistent free-param transfer 基础上，测试 train-only source-cluster lattice proposal。
- 动机是 rows>=7 仍可能受 lattice/source transfer 影响；新方法只改变 lattice，不改变 W/A generator。
- 不使用 StructureMatcher label 排序，不使用 test，不使用 test GT/W-A。

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`
- 新增 `--geometry-lattice-mode source_cluster_quantile`
- 实现：
  - 对当前 predicted W/A，从 train-only row-aligned source pool 取多个 source；
  - 对每个 source 先做 `volume_scaled_source` lattice；
  - 按 geometry rank 取 train source cluster 的 lattice 分位数；
  - 用 `lattice_from_target` 重新施加 SG crystal-system constraints；
  - free params 仍来自 row-aligned same-source transfer。
- 同时修复当前脚本中 `row_aligned_knn` 在 `OpentryGeometrySelector.candidates` 分发缺少显式分支的问题；该修复恢复已有 row-aligned 路径的预期行为，不改默认参数。

E122b smoke：

| item | value |
|---|---:|
| top_k / geometry ranks | 5 / 2 |
| rendered rows | 575 |
| samples with candidates | 125 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 | 95.83% / 90.40% / 95.83% |

判断：basic render health 足够进入 K50 validation smoke。

E123 render：

| item | value |
|---|---:|
| top_k / geometry ranks | 50 / 5 |
| rendered rows | 4,453 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 93.78% / 90.48% / 95.68% / 95.02% |

E124 diverse self-score：

| item | value |
|---|---:|
| output rows | 4,453 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.04% |
| rank<=20 SG-ok | 97.20% |
| min-distance mean rank1 / overall | 1.8081 / 1.1541 |

E125 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 25.78% | 31.25% | 35.16% | 35.94% | 0.1164 | 0.1403 | 0.1685 | 0.1518 | 78.91% | 12.65 |
| rows>=7 | 6.67% | 8.33% | 8.33% | 10.00% | 0.1962 | 0.1698 | 0.1671 | 0.1575 | 76.67% | 7.20 |
| atoms>=12 | 19.83% | 24.14% | 28.45% | 29.31% | 0.1471 | 0.1683 | 0.2074 | 0.1838 | 76.72% | 11.74 |
| complex_flag | 20.17% | 26.05% | 30.25% | 31.09% | 0.1465 | 0.1706 | 0.2061 | 0.1844 | 77.31% | 12.16 |

对比 E111：

| metric | E111 | E125 | delta |
|---|---:|---:|---:|
| full match@1 | 31.25% | 25.78% | -5.47 pp |
| full match@5 | 35.16% | 31.25% | -3.91 pp |
| full match@20 | 41.41% | 35.16% | -6.25 pp |
| full match@50 | 42.19% | 35.94% | -6.25 pp |
| rows>=7 match@50 | 18.33% | 10.00% | -8.33 pp |
| full W/A@50 | 78.91% | 78.91% | +0.00 pp |

Gain/loss diagnostic vs E111：

| budget | full gained | full lost | rows>=7 gained | rows>=7 lost |
|---:|---:|---:|---:|---:|
| K1 | 1 | 8 | 0 | 4 |
| K5 | 1 | 6 | 0 | 3 |
| K20 | 2 | 10 | 0 | 6 |
| K50 | 2 | 10 | 0 | 5 |

判断：

- Source-cluster lattice quantiles keep W/A recall unchanged but substantially hurt CIF match.
- The failure is not symbolic recall; it is lattice aggregation breaking source-specific geometry compatibility.
- This rejects independent lattice aggregation over row-aligned source pools.
- Do not expand this run to K100 or full test.
- Current best remains:
  - E119 for K50 ceiling: full `31.25/35.16/41.41/43.75`;
  - E111 for diversity-preserving row-aligned baseline: full `31.25/35.16/41.41/42.19`.

## 43. Current status after E125

时间：2026-06-13 UTC

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过。

## 46. Latest status after E214-E224

时间：2026-06-14 UTC

说明：

- E214-E224 的详细记录见本日志第 45 节。
- 本轮新增 source-residual cache 和 constrained residual smoke：
  - `scripts/opentry_build_source_residual_examples.py`
  - `data/source_residual_geometry_mpts52/e214_cache_train1024_val256_fulltrain_selector`
  - `runs/e215_source_residual_cached_constrained_1k_3epoch`
  - `runs/e220_source_residual_cached_tight_1k_3epoch`
  - `reports/e224_eval_e223_source_residual_tight_global_val32_match`
- E224 val32/top5 self-scored result:
  - full `match@1/5 = 34.38% / 37.50%`, RMSE `0.1579 / 0.1789`;
  - rows>=7 `match@1/5 = 16.67% / 16.67%`;
  - W/A@5 `81.25%`, unique W/A@5 `4.38`。
- E166 first-32 reference remains better on K5:
  - full `34.38% / 43.75%`;
  - rows>=7 `22.22% / 22.22%`。

结论：

- Cached source-pair training is useful infrastructure, but E215/E220 are not adopted.
- Current best remains E166/E181 on val128.
- Full test gate remains closed.
- Next aligned move: predicted-W/A source-pair cache plus source-mode/no-move or coherent-source mixture objective; no fallback, no test tuning.

结论：

- full test gate 仍关闭。
- E125 说明 lattice proposal 不能脱离 source-specific free-param compatibility 做分位数聚合。
- 下一步若继续 geometry，应该尝试 source-consistent multimodal source pairs / source-conditioned generative free-param-lattice head，而不是 independent lattice quantile、small jitter、single MSE geometry_net 或 source-priority sorting。

## 44. E126-E131 - train-source quality aware row-aligned selector

时间：2026-06-13 UTC

目的：

- 在 E111 `row_aligned_knn` 基础上加入 train source 自身质量信号。
- 只使用 train structured schema 中的 GT-free source health：
  - `row_expansion_all_ok`
  - `free_param_reextract_all_success`
  - row free-symbol 是否有对应 free params
- 不使用 StructureMatcher label、test GT、test W/A 或 validation match 来训练/排序。

训练源质量审计：

| subset | records | quality-ok records | rate |
|---|---:|---:|---:|
| train all | 25,998 | 21,007 | 80.80% |
| train rows>=7 | 6,863 | 5,844 | 85.15% |

E111 selected source diagnostic：

| budget | candidate good-source rate | matched samples with any good source |
|---:|---:|---:|
| K1 | 87.30% | 37 / 40 |
| K5 | 84.35% | 40 / 45 |
| K20 | 81.98% | 48 / 53 |
| K50 | 79.43% | 49 / 54 |

工程改动：

- 修改 `model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py`
- 新增 `--geometry-source-strategy row_aligned_quality`
- source score：
  - base `row_condition_distance`
  - `0.35 * source_alignment_cost`
  - train source quality penalty
- lattice/free params 仍来自同一 coherent source；没有独立 lattice aggregation。

E126 smoke：

| item | value |
|---|---:|
| top_k / geometry ranks | 5 / 2 |
| rendered rows | 581 |
| samples with candidates | 125 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 | 96.39% / 92.80% / 96.39% |

E127 render：

| item | value |
|---|---:|
| top_k / geometry ranks | 50 / 5 |
| rendered rows | 4,561 |
| samples with candidates | 126 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 94.19% / 92.06% / 96.48% / 95.44% |

E128 diverse self-score：

| item | value |
|---|---:|
| output rows | 4,561 |
| rank1 SG-ok | 100.00% |
| rank<=5 SG-ok | 99.36% |
| rank<=20 SG-ok | 97.48% |
| min-distance mean rank1 / overall | 1.9117 / 1.2279 |

E129 StructureMatcher result：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.16% | 40.62% | 44.53% | 0.1056 | 0.1025 | 0.1334 | 0.1562 | 78.91% | 12.80 |
| rows>=7 | 13.33% | 15.00% | 18.33% | 18.33% | 0.1284 | 0.1239 | 0.1516 | 0.1450 | 76.67% | 7.15 |
| atoms>=12 | 25.86% | 28.45% | 34.48% | 38.79% | 0.1037 | 0.1055 | 0.1543 | 0.1814 | 76.72% | 11.90 |
| complex_flag | 26.05% | 30.25% | 36.13% | 40.34% | 0.1046 | 0.1129 | 0.1571 | 0.1822 | 77.31% | 12.33 |

E130/E131 global self-score：

| subset | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 | RMSE@50 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 35.94% | 41.41% | 44.53% | 0.1056 | 0.1047 | 0.1376 | 0.1562 | 78.91% | 12.80 |
| rows>=7 | 13.33% | 16.67% | 18.33% | 18.33% | 0.1284 | 0.1205 | 0.1450 | 0.1450 | 76.67% | 7.15 |
| atoms>=12 | 25.86% | 29.31% | 35.34% | 38.79% | 0.1037 | 0.1167 | 0.1598 | 0.1814 | 76.72% | 11.90 |
| complex_flag | 26.05% | 31.09% | 36.97% | 40.34% | 0.1046 | 0.1258 | 0.1621 | 0.1822 | 77.31% | 12.33 |

对比 E119：

| metric | E119 | E131 | delta |
|---|---:|---:|---:|
| full match@1 | 31.25% | 31.25% | +0.00 pp |
| full match@5 | 35.16% | 35.94% | +0.78 pp |
| full match@20 | 41.41% | 41.41% | +0.00 pp |
| full match@50 | 43.75% | 44.53% | +0.78 pp |
| rows>=7 match@5 | 15.00% | 16.67% | +1.67 pp |
| rows>=7 match@50 | 18.33% | 18.33% | +0.00 pp |

Gain/loss diagnostic E131 vs E119：

| budget | full gained | full lost | rows>=7 gained | rows>=7 lost |
|---:|---:|---:|---:|---:|
| K1 | 0 | 0 | 0 | 0 |
| K5 | 1 | 0 | 1 | 0 |
| K20 | 2 | 2 | 0 | 0 |
| K50 | 2 | 1 | 0 | 0 |

判断：

- E131 是当前 opentry_3 best validation match ordering：
  - full match@1/5/20/50 = `31.25 / 35.94 / 41.41 / 44.53`;
  - rows>=7 match@1/5/20/50 = `13.33 / 16.67 / 18.33 / 18.33`。
- 它相对 E119 小幅改善 K5/K50，并改善 rows>=7 K5。
- 仍不能 freeze full test：
  - match@1 仍低于 `31.64%`；
  - match@5 仍低于 `41.58%`；
  - match@20 仍低于 `49.69%`；
  - rows>=7 match@50 仍只有 `18.33%`。
- 该实验支持“source coherence + source health”方向，但增益仍小；下一步需要更强的 source-conditioned multimodal geometry proposal，而不是进一步调全局排序。

脚本检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_wyckoff_cifs_e07e08.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过。
## 47. EOF latest status after E214-E224

时间：2026-06-14 UTC

- E214-E224 的详细记录已写入本日志第 45 节；本节作为文件末尾状态锚点。
- E224 val32/top5 self-scored result:
  - full `match@1/5 = 34.38% / 37.50%`, RMSE `0.1579 / 0.1789`;
  - rows>=7 `match@1/5 = 16.67% / 16.67%`;
  - W/A@5 `81.25%`, unique W/A@5 `4.38`。
- E166 first-32 reference remains better on K5:
  - full `34.38% / 43.75%`;
  - rows>=7 `22.22% / 22.22%`。
- E214 cache and E215/E220 constrained residual models are useful infrastructure, but not adopted as best model.
- Current best remains E166/E181 on val128; full test gate remains closed.
- Next aligned move: predicted-W/A source-pair cache plus source-mode/no-move or coherent-source mixture objective; no fallback, no test tuning.

## 48. Experiment E225-E226 - Train-only source-mode selector smoke

时间：2026-06-14 UTC

目的：

- 对 E224 失败来源继续拆解：同一个 predicted/target W/A 下，当前 heuristic source mode 是否经常不是最接近目标几何的 source。
- 构建 train/val source-mode group，每个 target 保留 8 个 train-source candidates。
- 用 train/val lattice + free-param transfer error 作为 source-mode label；不使用 StructureMatcher match/rms，不使用 test。
- 训练一个轻量 listwise MLP source-mode selector，判断能否在 val transfer-label 上超过当前 heuristic source rank0。

新增文件：

- `model/New_model/opentry_3/scripts/opentry_build_source_mode_examples.py`
- `model/New_model/opentry_3/scripts/opentry_train_source_mode_selector.py`

E225 data：

| item | value |
|---|---:|
| output | `model/New_model/opentry_3/data/source_mode_geometry_mpts52/e225_source_mode_train1024_val256_k8` |
| selector records | 25,998 train |
| train groups | 1,024 |
| val groups | 256 |
| source candidates / group | 8 |
| label | lattice + 4.0 * free-param transfer error |
| test records used | 0 |
| StructureMatcher labels used | false |

E225 oracle gap：

| subset | heuristic best-source rate | heuristic error mean | oracle best error mean | oracle gain mean |
|---|---:|---:|---:|---:|
| train full | 24.41% | 0.2914 | 0.0816 | 0.2098 |
| train rows>=7 | 32.61% | 2.3378 | 0.4022 | 1.9356 |
| val full | 23.83% | 1.7472 | 0.4064 | 1.3408 |
| val rows>=7 | 27.56% | 3.1732 | 0.6461 | 2.5270 |

判断：

- source-mode selection has real headroom, especially for rows>=7.
- The current heuristic rank0 source is best in only about one quarter of val groups.
- This supports the strategy shift from scalar self-score or one-source residual regression to a learned source-mode/no-move or mixture objective.

E226 source-mode MLP：

| item | value |
|---|---:|
| train / val groups | 1,024 / 256 |
| hidden_dim | 96 |
| epochs | 20 CPU |
| best epoch by val selected error | 13 |
| best checkpoint | `model/New_model/opentry_3/runs/e226_source_mode_selector_train1024_val256_k8/ckpt_best.pt` |

Best E226 validation result：

| subset | heuristic error mean | selected error mean | selected gain | oracle gain | hit best rate | improved over heuristic |
|---|---:|---:|---:|---:|---:|---:|
| val full | 1.7472 | 1.5610 | +0.1862 | 1.3408 | 22.66% | 21.88% |
| val rows>=7 | 3.1732 | 2.7459 | +0.4272 | 2.5270 | 25.20% | 22.05% |

判断：

- E226 is a positive smoke on transfer-label metrics: it reduces val selected source transfer error vs heuristic, including rows>=7.
- The model still captures only a small fraction of oracle headroom and does not improve best-source hit rate much.
- This is not yet a CIF result and is not adopted as best model.
- Next step: integrate the source-mode selector into a val32 renderer smoke, or train a stronger listwise objective with more train examples and explicit no-move/source-mixture calibration.

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_build_source_mode_examples.py \
  model/New_model/opentry_3/scripts/opentry_train_source_mode_selector.py
```

结果：compile 通过；E225 summary 与 E226 training summary 均存在且非空。

## 49. Experiment E227-E230 - Source-mode selector integration into residual renderer

时间：2026-06-14 UTC

目的：

- 将 E226 train-only source-mode selector 接入 source-conditioned residual renderer，验证 transfer-label 改善是否能转化为 validation CIF match。
- 仍只使用 validation 前 32 条做 StructureMatcher 评估；不使用 test。
- 排序/选择不使用 StructureMatcher match/rms；E229 只使用 GT-free global geometry self-score。

工程改动：

- 更新 `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`。
- 新增 `--source-mode-ckpt` / `--source-mode-pool-k`。
- renderer 可加载 E226 listwise MLP，从 train-source pool 中选择 source mode。
- 输出记录 `source_rank`、`source_mode_score`、`source_mode_used`，summary 记录 source-mode checkpoint 与 best val selected error。

E227 render：

| item | value |
|---|---:|
| model | E220 tight residual geometry |
| source-mode ckpt | E226 best epoch 13 |
| split / max records | val / 32 |
| W/A top-k | 5 |
| source-mode pool k | 8 |
| rendered rows | 140 |
| samples with rendered candidates | 31 |
| overall readable / composition exact | 95.71% / 95.71% |
| overall SG-ok | 92.14% |
| rank1 readable / composition exact | 93.55% / 93.55% |
| rank1 SG-ok | 87.10% |

E228 raw order StructureMatcher：

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| full | 18.75% | 37.50% | 0.1763 | 0.2522 | 81.25% | 4.38 |
| rows>=7 | 0.00% | 16.67% | - | 0.2505 | 83.33% | 4.56 |
| atoms>=12 | 20.00% | 33.33% | - | - | 83.33% | - |
| complex_flag | 19.35% | 35.48% | - | - | 83.87% | - |

E229 global self-score：

| item | value |
|---|---:|
| output rows | 140 |
| samples | 31 |
| overall readable / composition exact | 95.71% / 95.71% |
| overall SG-ok | 92.14% |
| rank1 readable / composition exact | 100.00% / 100.00% |
| rank1 SG-ok | 100.00% |
| rank1 min-distance mean | 1.526 |

E230 self-scored StructureMatcher：

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@1 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 37.50% | 0.2722 | 0.2522 | 56.25% | 81.25% | 4.38 |
| rows>=7 | 11.11% | 16.67% | 0.2356 | 0.2505 | 66.67% | 83.33% | 4.56 |
| atoms>=12 | 30.00% | 33.33% | 0.2288 | 0.2158 | 60.00% | 83.33% | 4.33 |
| complex_flag | 32.26% | 35.48% | 0.2535 | 0.2395 | 58.06% | 83.87% | 4.35 |

Same-prefix comparison：

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% | 78.12% | 2.88 |
| E224 tight cached residual | 34.38% | 37.50% | 16.67% | 16.67% | 81.25% | 4.38 |
| E230 source-mode + residual + self-score | 34.38% | 37.50% | 11.11% | 16.67% | 81.25% | 4.38 |

判断：

- E226 的 transfer-label 改善没有转化为 validation CIF top-5 改善。
- E230 self-score 将 rank1 恢复到 `34.38%`，但 K5 仍停在 `37.50%`，低于 E166 first-32 reference 的 `43.75%`。
- rows>=7 仍弱，且 rank1 比 E224 更低，说明当前 source-mode selector 对复杂样本的 source 几何兼容性仍不足。
- W/A@5 与 unique W/A@5 没有下降，失败主要仍在 source geometry / free-param / lattice transfer，而非 symbolic W/A recall。
- 不满足 val gate；不进入 val128/full test；E227-E230 不采用为 best model。

下一步：

- 保留 source-mode selector 接口和 E225/E226 数据作为基础设施。
- 若继续该路线，应扩大 train-only source-mode examples，并加入 no-move/source-mixture calibration 或直接建模 coherent source mixture，而不是继续调 scalar self-score。
- 仍不使用 fallback、oracle rerank、test tuning 或 CrystaLLM predictions 作为 primary candidates。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_build_source_mode_examples.py \
  model/New_model/opentry_3/scripts/opentry_train_source_mode_selector.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E230 `summary_metrics.tsv` 存在且非空。

## 50. Experiment E231-E235 - Source-mode no-move margin calibration

时间：2026-06-14 UTC

目的：

- 在 E230 没有改善 CIF top-5 后，测试一个更保守的 no-move calibration。
- 如果 source-mode 模型选择非 rank0 source，但相对 heuristic rank0 的 score margin 不足，则保留 rank0。
- 阈值只用 E225 train/val transfer-label examples 与 E226 checkpoint 调参，不使用 StructureMatcher match/rms，不使用 test。

新增/修改：

- 更新 `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`：
  - 新增 `--source-mode-min-margin`；
  - 输出 `source_mode_margin`、`source_mode_raw_best_idx`、`source_mode_overrode_to_rank0`。
- 新增 `model/New_model/opentry_3/scripts/opentry_tune_source_mode_margin.py`。

E231 transfer-label threshold tuning：

| threshold | val selected error full | val gain full | val selected error rows>=7 | val gain rows>=7 | overrode to rank0 full |
|---|---:|---:|---:|---:|---:|
| pure model | 1.5610 | +0.1862 | 2.7459 | +0.4272 | 0.00% |
| margin>=0.75 | 1.4991 | +0.2481 | 2.6644 | +0.5088 | 21.09% |
| heuristic rank0 | 1.7472 | +0.0000 | 3.1732 | +0.0000 | 48.44% model choices suppressed |

判断：

- `margin>=0.75` 在 validation transfer-label 上优于纯 E226 selector，因此作为 E232 CIF smoke 的 frozen threshold。
- 该阈值仍是 validation selection，不触碰 test。

E232 render with margin>=0.75：

| item | value |
|---|---:|
| rendered rows | 140 |
| samples with rendered candidates | 31 |
| overall readable / composition exact | 95.71% / 95.71% |
| overall SG-ok | 92.86% |
| rank1 readable / composition exact | 93.55% / 93.55% |
| rank1 SG-ok | 87.10% |

E233 raw StructureMatcher：

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | W/A@1 | W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| full | 18.75% | 37.50% | 0.1762 | 0.2497 | 50.00% | 81.25% |
| rows>=7 | 0.00% | 16.67% | - | 0.2505 | 38.89% | 83.33% |
| atoms>=12 | 20.00% | 33.33% | 0.1762 | 0.2158 | 53.33% | 83.33% |
| complex_flag | 19.35% | 35.48% | 0.1762 | 0.2395 | 51.61% | 83.87% |

E234 global self-score：

| item | value |
|---|---:|
| output rows | 140 |
| samples | 31 |
| overall readable / composition exact | 95.71% / 95.71% |
| overall SG-ok | 92.86% |
| rank1 readable / composition exact | 100.00% / 100.00% |
| rank1 SG-ok | 100.00% |
| rank1 min-distance mean | 1.5779 |

E235 self-scored StructureMatcher：

| subset | match@1 | match@5 | RMSE@1 | RMSE@5 | skeleton@1 | W/A@1 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 37.50% | 0.2634 | 0.2497 | 62.50% | 56.25% | 81.25% | 4.38 |
| rows>=7 | 11.11% | 16.67% | 0.2356 | 0.2505 | 66.67% | 66.67% | 83.33% | 4.56 |
| atoms>=12 | 30.00% | 33.33% | 0.2288 | 0.2158 | 60.00% | 60.00% | 83.33% | 4.33 |
| complex_flag | 32.26% | 35.48% | 0.2535 | 0.2395 | 61.29% | 58.06% | 83.87% | 4.35 |

Same-prefix comparison：

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 |
|---|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% |
| E224 tight cached residual | 34.38% | 37.50% | 16.67% | 16.67% |
| E230 pure source-mode self-score | 34.38% | 37.50% | 11.11% | 16.67% |
| E235 margin-calibrated source-mode self-score | 34.38% | 37.50% | 11.11% | 16.67% |

判断：

- No-move margin calibration improves transfer-label selected error, but does not improve CIF match.
- E235 rank1 is recovered by self-score, but K5 remains `37.50%`, below E166 first-32 `43.75%`.
- rows>=7 remains weak and below E166/E224.
- This confirms the current source-mode selector is not sufficient; the next model needs stronger coherent source-mixture/no-move training, not just margin thresholding.
- Val gate remains closed; no val128/full test run.

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_tune_source_mode_margin.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E231 summary 与 E235 summary metrics 均存在且非空。

## 51. Experiment E236-E241 - Source-mode mixture residual geometry smoke

时间：2026-06-14 UTC

目的：

- 在不改变 symbolic W/A candidate source 的前提下，测试 coherent source-mode mixture 是否能缓解残差几何的 source mismatch。
- 仍使用 E84 Wyckoff model predictions、E220 tight cached residual geometry、E226 source-mode selector。
- 本实验只在 MPTS-52 validation first 32 上做 smoke；不触碰 test，不使用 StructureMatcher label 做训练/排序。

工程改动：

- 更新 `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`。
- 新增 `--source-mode-mixture-size` 与 `--source-mode-include-rank0`。
- `--top-k` 现在按 unique W/A candidate 计数；每个 W/A 可渲染多个 source-mode proposal。
- 每条输出记录新增 `source_mode_choice_rank`、`source_mode_mixture_size`、`wa_rank`。
- 默认参数保持旧行为：`source_mode_mixture_size=1`，不强制包含 rank0。

E236 render 配置：

| item | value |
|---|---:|
| split | val first 32 |
| W/A predictions | E84 |
| residual geometry | E220 |
| source-mode selector | E226 |
| unique W/A top-k | 7 |
| source-mode pool-k | 8 |
| mixture size | 3 |
| include rank0 | yes |

E236 render health：

| item | value |
|---|---:|
| rendered rows | 540 |
| samples with rendered candidates | 31 |
| overall readable / formula / atom / composition exact | 95.00% / 95.00% / 95.00% / 95.00% |
| overall SG-ok | 92.22% |
| rank1 readable / formula / atom / composition exact | 93.55% / 93.55% / 93.55% / 93.55% |
| rank1 SG-ok | 93.55% |

E237 raw StructureMatcher：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@1 | W/A@5 | W/A@20 | unique W/A@5 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 18.75% | 31.25% | 40.62% | 0.1016 | 0.1631 | 0.1901 | 50.00% | 59.38% | 84.38% | 1.88 | 5.62 |
| rows>=7 | 0.00% | 5.56% | 16.67% | - | 0.1866 | 0.2039 | 38.89% | 44.44% | 83.33% | - | - |

E238/E239 global self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@1 | skeleton@5 | skeleton@20 | W/A@1 | W/A@5 | W/A@20 | unique W/A@5 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1946 | 0.1901 | 62.50% | 84.38% | 87.50% | 56.25% | 81.25% | 84.38% | 3.16 | 5.62 |
| rows>=7 | 16.67% | 16.67% | 16.67% | 0.2039 | 0.2039 | 0.2039 | 66.67% | 83.33% | 88.89% | 66.67% | 83.33% | 83.33% | 3.39 | 5.67 |

E240/E241 diverse self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@1 | skeleton@5 | skeleton@20 | W/A@1 | W/A@5 | W/A@20 | unique W/A@5 | unique W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 40.62% | 40.62% | 0.1536 | 0.1954 | 0.1901 | 62.50% | 84.38% | 87.50% | 56.25% | 81.25% | 84.38% | 4.38 | 5.62 |
| rows>=7 | 16.67% | 16.67% | 16.67% | 0.2039 | 0.2039 | 0.2039 | 66.67% | 83.33% | 88.89% | 66.67% | 83.33% | 83.33% | 4.56 | 5.67 |

Same-prefix comparison：

| method | full match@1 | full match@5 | full match@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | - | 22.22% | 22.22% | - |
| E235 margin source-mode self-score | 34.38% | 37.50% | - | 11.11% | 16.67% | - |
| E239 mixture global self-score | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |
| E241 mixture diverse self-score | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |

判断：

- Source-mode mixture improves full K5 over E235 (`37.50% -> 40.62%`) and keeps K1 at `34.38%`.
- It still does not beat the E166 first-32 K5 reference (`43.75%`) and does not improve rows>=7 beyond E224/E166.
- Diverse self-score preserves match while increasing full unique W/A@5 from `3.16` to `4.38`, but this diversity does not translate into additional CIF matches.
- W/A recall is high (`W/A@20=84.38%` full, `83.33%` rows>=7), so the immediate bottleneck remains geometry/source compatibility under predicted W/A.
- This is useful infrastructure, not a frozen config. No val128 or full test run is justified.

下一步：

- Do not scale E236-E241 as-is.
- Train a stronger mixture/no-move geometry objective rather than only increasing inference-time source proposals.
- Keep the full test gate closed until validation rows>=7 and full K5 clearly exceed the E166/E181 reference.

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py
```

结果：compile 通过；E239/E241 summary metrics 已生成。

## 52. Experiment E242-E249 - Train-time source-mixture residual geometry smoke

时间：2026-06-14 UTC

目的：

- 将 E236-E241 的 inference-time source-mode mixture 进一步推进到训练侧。
- 不再只用 single selected source 训练 residual geometry，而是构建 rank0 no-move anchor + transfer-error top sources 的多 source-context 样本。
- 目标是让 residual net 在训练时见到 coherent source mixture/no-move contexts，再用同样的 E236 source-mode mixture protocol 做 validation CIF smoke。

新增文件：

- `model/New_model/opentry_3/scripts/opentry_build_source_mixture_residual_examples.py`

E242 数据构建：

| item | value |
|---|---:|
| source-mode examples | E225 train1024 / val256 / pool8 |
| train groups / examples | 1,024 / 2,645 |
| val groups / examples | 256 / 672 |
| train rank0 / transfer_top examples | 1,024 / 1,621 |
| val rank0 / transfer_top examples | 256 / 416 |
| train rows>=7 groups | 46 |
| val rows>=7 groups | 127 |
| missing records / sources | 0 / 0 |
| label kind | lattice/free-param transfer error, not StructureMatcher |
| test records used | 0 |

E243 training：

| item | value |
|---|---:|
| model | source residual geometry net |
| train / val examples | 2,645 / 672 |
| epochs | 4 |
| device | CPU |
| lattice_delta_scale / coord_delta_scale | 0.15 / 0.05 |
| best val loss | 2.3287 |
| best epoch | 4 |

E244 render health：

| item | value |
|---|---:|
| predictions | E84 |
| geometry ckpt | E243 |
| source-mode ckpt | E226 |
| unique W/A top-k | 7 |
| source-mode mixture size | 3 + rank0 included |
| rendered rows | 540 |
| samples with rendered candidates | 31 |
| overall readable / composition exact | 95.19% / 95.19% |
| overall SG-ok | 91.48% |
| rank1 readable / composition exact | 93.55% / 93.55% |
| rank1 SG-ok | 90.32% |

E245 raw StructureMatcher：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@1 | W/A@5 | W/A@20 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 18.75% | 31.25% | 40.62% | 0.1312 | 0.1749 | 0.2064 | 50.00% | 59.38% | 84.38% | 1.88 |
| rows>=7 | 0.00% | 5.56% | 16.67% | - | 0.1623 | 0.2347 | 38.89% | 44.44% | 83.33% | 1.89 |

E246/E247 global self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 37.50% | 40.62% | 0.1526 | 0.2019 | 0.2064 | 75.00% | 71.88% | 3.03 |
| rows>=7 | 11.11% | 11.11% | 16.67% | 0.1937 | 0.1937 | 0.2347 | 72.22% | 72.22% | 3.33 |

E248/E249 diverse self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 31.25% | 37.50% | 40.62% | 0.1526 | 0.2038 | 0.2064 | 78.12% | 75.00% | 4.38 |
| rows>=7 | 11.11% | 11.11% | 16.67% | 0.1937 | 0.1937 | 0.2347 | 72.22% | 72.22% | 4.56 |

Same-prefix comparison：

| method | full match@1 | full match@5 | full match@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | - | 22.22% | 22.22% | - |
| E239 E220 mixture global | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |
| E241 E220 mixture diverse | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |
| E247 E243 mixture-trained global | 31.25% | 37.50% | 40.62% | 11.11% | 11.11% | 16.67% |
| E249 E243 mixture-trained diverse | 31.25% | 37.50% | 40.62% | 11.11% | 11.11% | 16.67% |

判断：

- E242/E243 是对主线更贴近的一次训练侧推进：rank0 + transfer-top source contexts 进入 residual geometry training。
- 但 CIF 层表现退化：full K1/K5 从 E239/E241 的 `34.38/40.62` 降到 `31.25/37.50`；rows>=7 从 `16.67/16.67` 降到 `11.11/11.11`。
- Diverse self-score 仍能提高 unique W/A@5，但 match 不随之提升。
- 说明 lattice/free-param transfer-error top-source label 仍不是足够好的 rendered CIF proxy；简单把 transfer-top source 加入训练会让模型更平均化，反而弱化 no-move/source geometry。
- E243 不采用，不进入 val128 或 full test。

下一步：

- 保留 E242 builder 作为基础设施。
- 后续不应继续只扩大 transfer-error mixture 数据；需要训练能判别 source context 是否会产生 valid rendered CIF 的 GT-free/self-supervised proxy，或把 residual head 改为 source-conditioned gating/no-op delta，而不是对多 source 做同一回归平均。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_build_source_mixture_residual_examples.py \
  model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E245/E247/E249 summary metrics 均存在。

## 53. Experiment E250-E256 - Gated source-mixture residual geometry smoke

时间：2026-06-14 UTC

目的：

- 延续 E242-E249 的 train-time source-mixture residual 路线，但避免对 rank0 和 transfer-top source 做同权平均回归。
- 在 residual geometry head 中加入 source-conditioned delta gate，让模型可学习 no-op / small-delta 行为。
- 继续使用 E84 symbolic W/A candidates + E226 source-mode mixture3 + rank0 protocol，在 MPTS-52 validation first 32 做 CIF smoke。

代码改动：

- `model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py`
  - `SourceResidualGeometryNet` 新增可选 `enable_delta_gate`。
  - gated 模式下新增 lattice/coord gate head，使用 sigmoid gate 缩放 residual delta。
  - 新增 `--enable-delta-gate`、`--gate-bias-init`、`--gate-l1-weight`。
  - loss 记录 `gate_l1_loss`、`lattice_gate_mean`、`coord_gate_mean`。
- `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`
  - 支持读取 gated checkpoint config。
  - 兼容旧 checkpoint；未启用 `enable_delta_gate` 时行为不变。

数据与无泄露边界：

- 训练数据仍为 E242 train/val source residual mixture cache：
  `model/New_model/opentry_3/data/source_residual_geometry_mpts52/e242_mixture_rank0_top2_train1024_val256`
- labels 为 lattice/free-param transfer error，不是 StructureMatcher match/rms。
- source pool 来自 train split；val 只用于 checkpoint selection 和 CIF smoke。
- 未使用 test 数据、test GT CIF、oracle rerank、fallback 或 CrystaLLM prediction 作为 primary candidate source。

E250 gated training：

| item | value |
|---|---:|
| train / val examples | 2,645 / 672 |
| epochs | 4 |
| device | CPU |
| lattice_delta_scale / coord_delta_scale | 0.15 / 0.05 |
| gate_bias_init / gate_l1_weight | -1.5 / 0.02 |
| best val loss | 2.3394 |
| best epoch | 3 |
| epoch4 val lattice_gate_mean | 0.3448 |
| epoch4 val coord_gate_mean | 0.0315 |
| epoch4 val gate_l1_loss | 0.1882 |

判断：

- Gate 没有训练崩溃，且 coord gate 明显偏小，说明模型确实学到接近 no-op 的 coordinate residual。
- Lattice gate 仍有中等幅度，说明 lattice transfer 仍在发挥作用。

E251 render health：

| item | value |
|---|---:|
| predictions | E84 |
| geometry ckpt | E250 best |
| source-mode ckpt | E226 |
| unique W/A top-k | 7 |
| source-mode mixture size | 3 + rank0 included |
| rendered rows | 540 |
| samples with rendered candidates | 31 |
| overall readable / composition exact | 95.00% / 95.00% |
| overall SG-ok | 91.48% |
| rank1 readable / composition exact | 93.55% / 93.55% |
| rank1 SG-ok | 93.55% |

E252 raw StructureMatcher：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@1 | W/A@5 | W/A@20 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 21.88% | 34.38% | 43.75% | 0.1033 | 0.1479 | 0.1534 | 50.00% | 59.38% | 84.38% | 1.88 |
| rows>=7 | 0.00% | 5.56% | 16.67% | - | 0.0083 | 0.0443 | 38.89% | 44.44% | 83.33% | 1.89 |

E253/E254 global self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1534 | 0.1534 | 78.12% | 75.00% | 2.91 |
| rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.0443 | 0.0443 | 77.78% | 77.78% | 3.11 |

E255/E256 diverse self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 37.50% | 43.75% | 43.75% | 0.1455 | 0.1710 | 0.1534 | 81.25% | 78.12% | 4.38 |
| rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 77.78% | 77.78% | 4.56 |

Same-prefix comparison：

| method | full match@1 | full match@5 | full match@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | - | 22.22% | 22.22% | - |
| E239 E220 mixture global | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |
| E241 E220 mixture diverse | 34.38% | 40.62% | 40.62% | 16.67% | 16.67% | 16.67% |
| E247 E243 mixture-trained global | 31.25% | 37.50% | 40.62% | 11.11% | 11.11% | 16.67% |
| E249 E243 mixture-trained diverse | 31.25% | 37.50% | 40.62% | 11.11% | 11.11% | 16.67% |
| E254 E250 gated global | 37.50% | 43.75% | 43.75% | 11.11% | 16.67% | 16.67% |
| E256 E250 gated diverse | 37.50% | 43.75% | 43.75% | 11.11% | 16.67% | 16.67% |

判断：

- E250 gated residual 比 E243 equal-regression mixture 明显更稳，full K1/K5 回到或略超过 E166 first-32 reference。
- 但复杂子集没有改善：rows>=7 仍只有 `11.11/16.67/16.67`，低于 E166 rows>=7 `22.22/22.22`。
- E252 raw K20 上限提升到 43.75%，说明 gating 对候选池尾部有帮助；self-score 能把 full K1/K5 前移，但不能解决 rows>=7 coherent geometry。
- Diverse self-score 提高 unique W/A@5 到 4.38，但 match 与 global 相同，说明当前失败不是单纯 W/A 多样性不足。
- E250 不进入 val128 或 full test；full test gate 继续关闭。

下一步：

- 保留 gated residual head 和 source-mixture rendering support。
- 不继续简单扩大 E242/E250 同类 transfer-error mixture 数据。
- 下一轮应训练更直接的 geometry-quality proxy 或 mixture-of-experts source/geometry selector，使复杂 rows>=7 的 source geometry 与 predicted W/A 一致，而不是只靠 GT-free self-score 后排。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_train_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E250 checkpoint 与 E252/E254/E256 summary metrics 均存在。

## 54. Experiment E257-E265 - Rendered-health source-mode selector diagnostic

时间：2026-06-14 UTC

目的：

- 针对 E250-E256 暴露出的 rows>=7 source/W-A geometry compatibility 瓶颈，尝试把 source-mode selector 的训练标签从 transfer-error 改为 rendered CIF GT-free health。
- 标签只来自 train/val 的 formula/GT-SG validity、composition exact、SG-ok、pymatgen parse/min-distance、source distance 等可在推理时计算的健康指标。
- 不使用 StructureMatcher match/rms 作为训练、排序、过滤或调参标签。

新增文件：

- `model/New_model/opentry_3/scripts/opentry_build_source_render_quality_examples.py`

E257 数据构建：

| item | value |
|---|---:|
| target split | MPTS-52 train512 / val256 |
| source pool | train split only |
| source_pool_k | 8 |
| residual geometry for labels | E250 gated residual |
| train / val groups | 512 / 256 |
| train candidate count mean | 8.0 |
| val candidate count mean | 8.0 |
| val full heuristic health / best health | 4876.49 / 4944.31 |
| val full oracle health gain | 67.82 |
| val rows>=7 groups | 127 |
| val rows>=7 heuristic health / best health | 4817.83 / 4922.07 |
| val rows>=7 oracle health gain | 104.23 |
| StructureMatcher labels used | 0 |
| test records used | 0 |

判断：

- rendered-health label 有非平凡 source 选择空间，尤其 val rows>=7 的 oracle health gain 更大。
- 这是一个无泄露的 source/geometry proxy 数据集，可用于训练 source-mode selector。

E258 source-mode training：

| item | value |
|---|---:|
| examples | E257 train512 / val256 |
| model | same SourceModeScorer |
| epochs | 30 |
| complex_weight | 5.0 |
| best epoch | 16 |
| best val full selected health gain vs rank0 | +1.42 |
| best val rows>=7 selected health gain vs rank0 | +3.06 |
| final val full selected health gain vs rank0 | -0.12 |
| final val rows>=7 selected health gain vs rank0 | -1.12 |

判断：

- E258 能在 best checkpoint 上取得很小的 validation proxy gain，但泛化很弱。
- full/rows>=7 proxy gain 远低于 oracle space，说明现有 numeric source features 对 rendered-health 标签解释力不足。

E259 render health：

| item | value |
|---|---:|
| predictions | E84 |
| geometry ckpt | E250 best |
| source-mode ckpt | E258 best |
| unique W/A top-k | 7 |
| source-mode mixture size | 3 + rank0 included |
| rendered rows | 540 |
| samples with rendered candidates | 31 |
| overall readable / composition exact | 94.81% / 94.81% |
| overall SG-ok | 93.15% |
| rank1 readable / composition exact | 93.55% / 93.55% |
| rank1 SG-ok | 93.55% |

E260 raw StructureMatcher：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | W/A@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 21.88% | 31.25% | 43.75% | 0.0755 | 0.1149 | 0.1532 | 59.38% | 84.38% |
| rows>=7 | 5.56% | 5.56% | 16.67% | 0.0083 | 0.0083 | 0.0443 | 44.44% | 83.33% |

E261/E262 global self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 40.62% | 43.75% | 0.1155 | 0.1295 | 0.1532 | 84.38% | 81.25% | 2.78 |
| rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.0443 | 0.0443 | 83.33% | 83.33% | 3.00 |

E263/E264 diverse self-score：

| subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | skeleton@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 34.38% | 40.62% | 43.75% | 0.1155 | 0.1490 | 0.1532 | 87.50% | 84.38% | 4.38 |
| rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 83.33% | 83.33% | 4.56 |

E265 margin tuning:

| item | value |
|---|---:|
| best validation label | pure_model |
| best threshold | none |
| best val full selected health gain | +1.42 |
| best val rows>=7 selected health gain | +3.06 |

Same-prefix comparison：

| method | full match@1 | full match@5 | full match@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 |
|---|---:|---:|---:|---:|---:|---:|
| E254 E250 gated global | 37.50% | 43.75% | 43.75% | 11.11% | 16.67% | 16.67% |
| E256 E250 gated diverse | 37.50% | 43.75% | 43.75% | 11.11% | 16.67% | 16.67% |
| E262 E258 render-health global | 34.38% | 40.62% | 43.75% | 11.11% | 16.67% | 16.67% |
| E264 E258 render-health diverse | 34.38% | 40.62% | 43.75% | 11.11% | 16.67% | 16.67% |

判断：

- E257 数据构建是有价值的诊断：rows>=7 的 source choice 在 GT-free rendered-health proxy 下确实有较大 oracle space。
- 但 E258 selector 没有把这个 space 转为 CIF match；full K1/K5 低于 E254/E256，rows>=7 没有改善。
- margin tuning 不能补救，best validation proxy config 仍是 pure E258 model。
- E258 不采用，不进入 val128 或 full test。
- 当前问题不是没有 source 选择空间，而是现有 source numeric features + listwise classifier 不足以识别“复杂 W/A 下哪个 source geometry 真正兼容”。

下一步：

- 保留 E257 builder 作为 source/geometry proxy 数据基础。
- 不继续只调 source-mode margin 或扩大同一 selector。
- 下一步应把 rendered CIF health features 或 per-candidate parse/min-distance proxy 直接进入 inference-time scoring，或训练更强的 per-rendered-candidate quality model；同时保持 test 禁用 StructureMatcher label。

验证：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python -m py_compile \
  model/New_model/opentry_3/scripts/opentry_build_source_render_quality_examples.py \
  model/New_model/opentry_3/scripts/opentry_train_source_mode_selector.py \
  model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py \
  model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py \
  model/New_model/opentry_3/scripts/evaluate_rendered_wyckoff_cifs.py
```

结果：compile 通过；E257/E258/E260/E262/E264/E265 关键产物均存在。

## 55. Experiment E266-E279 - Source-expanded rendered-candidate self-score diagnostic

时间：2026-06-14 UTC

目的：

- 承接 E257-E265 的结论：source choice 有空间，但 E258 source-only selector 不能识别兼容 source。
- 改为不训练 selector，直接对每个 W/A 展开多个 train-source geometry context，渲染后用 GT-free rendered-CIF self-score 重排。
- 检查 source-expanded rendered candidate pool 是否能改善 CIF match，尤其 rows>=7。

工程改动：

- 更新 `model/New_model/opentry_3/scripts/opentry_render_source_residual_geometry.py`：
  - 新增 `--source-expand-k`，默认 `1`，旧行为不变。
  - 当不使用 source-mode checkpoint 且 `source_expand_k>1` 时，对每个 unique W/A 展开 top-N fast chemistry-quality train sources。
  - 每个 source context 经过同一个 E250 gated residual geometry model 渲染。
  - 输出新增 `source_choice_rank`，保留 `source_rank/source_sample_id/geometry_distance/align_cost`。
- 更新 `model/New_model/opentry_3/scripts/opentry_selfscore_rendered_cifs.py`：
  - 在 `pymatgen Structure.from_str` 周围抑制 warning，避免 bad CIF 把完整 CIF 文本刷进日志。

无泄露边界：

- source records 只来自 train split。
- W/A predictions 来自既有 val E84 prediction file。
- rendered-candidate ranking 只用 GT-free self-score：readable/formula/composition/SG、pymatgen parse、min-distance、volume、geometry distance、rank prior。
- StructureMatcher 只用于 validation evaluation。
- 没有使用 test、test GT、StructureMatcher label 排序、oracle rerank、fallback 或 CrystaLLM test candidates。

固定配置：

| item | value |
|---|---|
| W/A predictions | E84 renderer predictions |
| geometry model | E250 gated residual |
| split | MPTS-52 val prefix |
| W/A top-k before source expansion | 7 unique W/A |
| source_pool_k / source_expand_k | 24 / 8 |
| source-mode ckpt | none |
| self-score | global, standard profile, geometry_distance_weight=5.0, original_rank_weight=0.02 |

Render health:

| experiment | val rows | rendered rows | samples with candidates | overall composition exact | overall SG-ok |
|---|---:|---:|---:|---:|---:|
| E266 | 32 | 1,440 | 31 | 95.35% | 92.92% |
| E272 | 64 | 2,968 | 63 | 93.90% | 90.46% |
| E276 | 128 | 6,000 | 126 | 94.70% | 90.47% |

Validation first-32 results:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E267 raw source-expand8 | full | 21.88% | 28.12% | 43.75% | 0.0724 | 0.1271 | 0.1814 | 50.00% | 0.97 |
| E269 global self-score | full | 34.38% | 40.62% | 46.88% | 0.1172 | 0.1342 | 0.1729 | 78.12% | 2.25 |
| E271 diverse self-score | full | 34.38% | 40.62% | 40.62% | 0.1172 | 0.1479 | 0.1300 | 84.38% | 4.38 |
| E269 global self-score | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.0443 | 0.0443 | 77.78% | 2.22 |
| E271 diverse self-score | rows>=7 | 11.11% | 16.67% | 16.67% | 0.0623 | 0.1195 | 0.0443 | 83.33% | 4.56 |

Validation first-64 results:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E274 raw source-expand8 | full | 25.00% | 31.25% | 42.19% | 0.1023 | 0.1341 | 0.1694 | 50.00% | 0.98 |
| E275 global self-score | full | 35.94% | 43.75% | 46.88% | 0.1337 | 0.1531 | 0.1685 | 71.88% | 2.33 |
| E275 global self-score | rows>=7 | 13.33% | 20.00% | 20.00% | 0.1480 | 0.1056 | 0.1054 | 73.33% | 2.27 |

Validation first-128 results:

| method | subset | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E278 raw source-expand8 | full | 22.66% | 28.12% | 38.28% | 0.1393 | 0.1676 | 0.1919 | 47.66% | 0.98 |
| E279 global self-score | full | 27.34% | 35.94% | 41.41% | 0.1128 | 0.1514 | 0.1812 | 64.06% | 2.28 |
| E279 global self-score | rows>=7 | 11.67% | 16.67% | 18.33% | 0.1465 | 0.1128 | 0.1448 | 65.00% | 2.32 |

对比与判断：

- source-expand8 + global self-score 在 val32/val64 有正向信号，尤其 val64 full `35.94/43.75/46.88`。
- 但 val128 回落到 full `27.34/35.94/41.41`，低于当前 best val128 E166/E181 full `32.03/40.62/44.53/45.31`。
- rows>=7 仍弱：val128 只有 `11.67/16.67/18.33`，未解决复杂子集。
- direct rendered self-score 可以把更多 rendered candidates 的 GT-free validity 推到前排，但它仍不能稳定识别 StructureMatcher 正例。
- E266-E279 不进入 full test，不作为 frozen config。

下一步：

- 保留 `--source-expand-k`，它是有用的 rendered-candidate pool/diagnostic 工具。
- 不继续只扩大 source_expand_k 或只调 self-score weight。
- 更合理的下一步是训练 per-rendered-candidate quality model：输入实际 rendered CIF 的 GT-free features + W/A/source metadata，label 仅来自 train/val，目标直接改善 top-5 source/W/A geometry compatibility；或回到 W/A generator 提升复杂子集 recall。

## 56. Experiment E280-E294 - Canonical source-priority W/A decoder diagnostic

时间：2026-06-14 UTC

目的：

- 回到 B-stage W/A decoder，而不是继续扩大 rendered source pool。
- 诊断 E83/E85 neural-first raw merge 中 raw permutation duplicate 是否浪费 canonical W/A budget。
- 在 canonical W/A 层合并多个 assignment-DP source，保留 model-side W/A generator 主线。
- 只使用 MPTS-52 validation；不使用 test、StructureMatcher label、row_count 排序、oracle rerank、RF ranker 或 CrystaLLM candidates。

新增/修改：

- 新增 `model/New_model/opentry_3/scripts/opentry_merge_renderer_predictions_balanced.py`。
- 支持 canonical prediction 层 `round_robin` 与 `priority` 合并。
- 合并 key 使用 `canonical_wa_key`，避免 raw W/A permutation duplicate 抢占 top-k。
- 输出仍为 `renderer_predictions.jsonl`，可直接接 `evaluate_wa_predictions.py` 或 renderer。

Canonical conversion / merge artifacts:

| experiment | source | output candidates | notes |
|---|---|---:|---|
| E280 | E69c neural assignment DP `p1_m1` | 1,280 | canonicalized neural source |
| E281/E282 | round-robin E69c + E43 prior | 1,871 | more unique, but W/A@5 slightly lower |
| E283/E284 | canonical priority E69c + E43 prior | 1,871 | preserves E85 W/A@5 and improves unique |
| E285 | E69d neural assignment DP `p05_m1` | 1,301 | stronger full W/A@50 tail |
| E286/E287 | priority E69d + E43 prior | 1,919 | better W/A@50, worse W/A@5 |
| E288/E289 | priority E69c + E69d + E43 prior | 1,937 | best symbolic tradeoff |

Symbolic W/A comparison on val128:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | full unique W/A@50 | rows>=7 W/A@5 | rows>=7 W/A@50 | rows>=7 unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E85 old raw neural-first merge | 47.66% | 70.31% | 77.34% | 78.91% | 13.74 | 73.33% | 76.67% | 8.37 |
| E284 canonical priority E69c+prior | 47.66% | 70.31% | 77.34% | 78.91% | 14.62 | 73.33% | 76.67% | 9.60 |
| E287 canonical priority E69d+prior | 47.66% | 67.97% | 77.34% | 79.69% | 14.99 | 70.00% | 76.67% | 10.12 |
| E289 canonical priority E69c+E69d+prior | 47.66% | 70.31% | 77.34% | 79.69% | 15.13 | 73.33% | 76.67% | 10.38 |

判断：

- E289 是最好的 symbolic decoder diagnostic：
  - 保住 E85 的 full W/A@1/5/20；
  - full W/A@50 从 `78.91%` 提升到 `79.69%`；
  - full unique W/A@50 从 `13.74` 提升到 `15.13`；
  - rows>=7 unique W/A@50 从 `8.37` 提升到 `10.38`。
- rows>=7 W/A@50 没有提升，仍为 `76.67%`。
- 改动是 decoder/canonical-selection 层面的合规 model-side 改进，但幅度较小。

CIF validation:

- 用 E289/E288 W/A candidates 复用当前 K5-best renderer：
  - `e08`
  - `wa_diverse`
  - `geometry_ranks_per_wa=10`
  - `row_aligned_chem_quality`
  - `source_vpa_calibrated_soft`
  - GT-free global/diverse self-score
- 只跑 val128，不跑 test。

Render health E290:

| item | value |
|---|---:|
| rendered rows | 9,856 |
| samples with rendered candidates | 126 / 128 |
| readable / formula / atom / composition exact | 100% / 100% / 100% / 100% |
| SG-ok overall / rank1 / rank<=5 / rank<=20 | 94.25% / 92.06% / 95.87% / 95.68% |

Validation CIF results:

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@5 | rows>=7 match@20/50 | full W/A@50 | full unique W/A@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 reference | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 18.33% | 20.00% / 20.00% | 76.56% | 10.59 |
| E292 E289 candidates | global | 32.03% | 39.84% | 44.53% | 45.31% | 0.0942 | 0.1314 | 18.33% | 20.00% / 20.00% | 77.34% | 11.59 |
| E294 E289 candidates | diverse | 32.03% | 38.28% | 42.97% | 45.31% | 0.0942 | 0.1237 | 18.33% | 20.00% / 20.00% | 79.69% | 14.52 |

判断：

- Canonical priority decoder improves symbolic W/A tail and diversity, but current renderer/self-score does not convert it into higher match.
- E292 ties E166 on match@1/20/50 but loses `0.78 pp` on match@5.
- E294 preserves W/A diversity but loses match@5/20.
- Current best validation CIF remains E166/E181:
  - full `32.03 / 40.62 / 44.53 / 45.31`;
  - rows>=7 `15.00 / 18.33 / 20.00 / 20.00` depending reference ordering;
  - only match@1 clears the +5pp target.
- Full test remains closed.

下一步：

- Keep `opentry_merge_renderer_predictions_balanced.py`; canonical source-priority should replace raw merge when a W/A-tail diagnostic is needed.
- Do not render full test from E289/E292/E294.
- The next useful B-stage improvement must raise rows>=7 W/A@50, not only unique W/A.
- If continuing CIF conversion, target source/W-A geometry compatibility directly; extra W/A tail alone is not enough under the current renderer.

## 57. Experiment E295-E300 - E69b-first canonical W/A priority follow-up

时间：2026-06-14 UTC

目的：

- E69b raw DP had better rows>=7 early assignment recall than E69c/E69d.
- Test whether using E69b as the first canonical priority source improves rows>=7 W/A@1/5 and whether that transfers to CIF match.
- Still validation-only, no StructureMatcher label sorting, no test, no row_count input.

Artifacts:

- `reports/e295_renderer_predictions_e69b_neural_canonical`
- `reports/e296_priority_e69b_e69c_e69d_prior_canonical_merge_val128`
- `reports/e297_wa_eval_e296_priority_e69b_e69c_e69d_prior_val128`
- `reports/e298_render_e296_chem_vpa_soft_e08_g10_val128_k100`
- `reports/e299_selfscore_e298_e296_chem_vpa_soft_global_gtfree_top50`
- `reports/e300_eval_e299_e296_chem_vpa_soft_global_val128_match`

Symbolic W/A result:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | full unique W/A@50 | rows>=7 W/A@1 | rows>=7 W/A@5 | rows>=7 W/A@50 | rows>=7 unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E289 E69c-first | 47.66% | 70.31% | 77.34% | 79.69% | 15.13 | 51.67% | 73.33% | 76.67% | 10.38 |
| E297 E69b-first | 49.22% | 70.31% | 77.34% | 79.69% | 15.23 | 53.33% | 73.33% | 76.67% | 10.57 |

判断：

- E69b-first is the best symbolic ordering so far:
  - full W/A@1 improves by `+1.56 pp` over E289/E85;
  - rows>=7 W/A@1 improves by `+1.67 pp`;
  - unique W/A@50 improves slightly.
- W/A@5/20/50 and rows>=7 W/A@5/50 do not improve.

CIF validation with the E166 renderer/global self-score:

| run | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@5 | rows>=7 match@50 | full W/A@50 after self-score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E166 reference | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 18.33% | 20.00% | 76.56% |
| E292 E69c-first global | 32.03% | 39.84% | 44.53% | 45.31% | 0.0942 | 0.1314 | 18.33% | 20.00% | 77.34% |
| E300 E69b-first global | 31.25% | 39.06% | 44.53% | 45.31% | 0.0955 | 0.1273 | 18.33% | 20.00% | 77.34% |

判断：

- E69b-first symbolic W/A@1 gain does not survive rendered-CIF self-score and validation matching.
- E300 loses match@1 and match@5 against E166 and E292.
- The current GT-free self-score selects similar rendered W/A coverage at top50 (`77.34%`) and does not exploit the E69b symbolic rank1 gain.
- Full test remains closed.

下一步：

- Keep E297 as the best symbolic W/A ordering record.
- Do not render more canonical-source-order permutations unless they improve rows>=7 W/A@5/50, not only rank1/unique tail.
- The next work should either:
  - improve rows>=7 assignment/skeleton recall directly; or
  - train a rendered-candidate quality model that can preserve good symbolic W/A while choosing source geometry.
## 2026-06-14 Batch E301-E312 - geometry compatibility selector negative result

- Initialized lightweight state files because `EXEC_STATE.json` / `NEXT_ACTION.md` / `experiments.jsonl` were absent; no repeated E00 audit.
- Built train-only atom>=12 compatibility pool: E302 skeleton train512 `skeleton@50=90.63%`, rows>=7 `86.96%`.
- E303 neural assignment DP on the same train pool reached full `W/A@1/5/20/50=55.47/72.85/82.62/84.38%`; rows>=7 `43.48/54.35/63.04/67.39%`.
- E305 rendered train candidates with E166-like `row_aligned_chem_quality + source_vpa_calibrated_soft + e08 g10`, with `--exclude-self-source`; E306 global self-score produced 18,289 train candidates.
- E307 feature table used train/val StructureMatcher labels only. Train top20 positive rate was 43.56%; val128 top50 positive rate was 9.47%. Val W/A-hit-but-match-fail candidate rate was 71.39%, confirming geometry conversion remains the bottleneck.
- E308/E310 GBDT compatibility selector learned candidate-level signal (`val AUC=0.894`, AP=0.561) but worsened official val128 match to `30.47/36.72/42.19/45.31`; rows>=7 became `13.33/15.00/16.67/20.00`.
- E309/E311 MLP also failed: official val128 `26.56/36.72/44.53/45.31`; rows>=7 `8.33/15.00/20.00/20.00`.
- E312 W/A-preserving group selector improved unique W/A but label match dropped to `23.02/33.33/35.71/38.10` on nonempty val samples.
- Decision: reject pointwise compatibility probability sorting and current group selector. No full test; E166/E181 remains best validation CIF config.
- Next: build more representative train feature pool and train group/listwise hit@5/hit@20 objective that preserves W/A diversity before any val512/full-test gate.

## 2026-06-14 Batch E313-E322 - balanced train pool and listwise/group selector negative result

- Built `data/wyckoff_repr_mpts52_train_atom_balanced_1024`: 4 atom-count buckets x 256 train records; `rows>=7=170`, `atoms>=12=512`, no test access.
- E313 E35 count-aware skeleton eval on this train pool: full `skeleton@1/5/20/50=53.91/76.95/87.40/93.26%`; rows>=7 `47.06/67.06/83.53/87.06%`.
- E314 E63 assignment DP on E313 candidates: full `W/A@1/5/20/50=53.71/70.70/79.30/81.64%`; rows>=7 `35.29/48.82/55.88/58.24%`.
- E315b converted DP candidates with full-train orbit metadata: missing orbit references = 0, output candidates = 13,096, samples with candidates = 1,023/1,024.
- E316 rendered train candidates with E166-like `row_aligned_chem_quality + source_vpa_calibrated_soft + e08 g10`, `--exclude-self-source`: 75,257 rendered rows, rank<=20 SG-ok 92.31%.
- E317 global GT-free self-score produced 44,460 train top50 rows; rank1 SG-ok 99.02%, rank<=20 SG-ok 98.23%.
- E318 train top20 StructureMatcher feature labels: 19,850 rows, positive rate 44.91%, full match@20 77.06%, rows>=7 match@20 34.94%, W/A-hit/match-fail rate 20.07%.
- E319/E321 pairwise W/A-cap selector official val128: full `match@1/5/20/50=31.25/37.50/43.75/45.31%`; rows>=7 `15.00/18.33/20.00/20.00%`; reject.
- Pairwise-MMR and GBDT group-bundle variants were worse: MMR `26.56/35.94/42.97/45.31%`, GBDT group-bundle `29.69/33.59/43.75/43.75%`; reject.
- E320b/E322 group-bundle linear selector official val128: full `match@1/5/20/50=29.69/38.28/42.97/43.75%`; rows>=7 `13.33/16.67/18.33/18.33%`; reject.
- Decision: balanced train pool is useful, but shallow pairwise/group selectors still misalign with E166 validation top-k. E166/E181 remains best; full test remains closed.

## 2026-06-14 E323-E327 enriched source/rendered compatibility selector

目的：在不触碰 test 的前提下，检查更丰富的 source/free-param/rendered-CIF GT-free 特征是否能把 E166 val128 top50 候选排序成更高 K5/K20。

产物：
- `data/geometry_compat_mpts52/e323_enriched_train_balanced1024_val128/`
- `reports/e324_dynamic_gbdt_enriched_train_balanced1024_val128`
- `reports/e325_dynamic_pairwise_enriched_train_balanced1024_val128`
- `reports/e326_eval_e324_dynamic_gbdt_enriched_val128_match`
- `reports/e327_eval_e325_dynamic_pairwise_enriched_val128_match`

E323 特征表：train rows `19,850`，val rows `5,599`，source found rate `100%`；新增 source formula similarity、source row/atom/SG context、source-candidate skeleton/W-A relation、rendered cell shape。训练脚本显式屏蔽 label、target row_count/keys、candidate hit flags、sample/material/source ids 和 raw CIF。

| run | selector | candidate AUC/AP | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5/20 | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| E166 baseline | global self-score | - | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% / 20.00% | keep |
| E324/E326 | enriched GBDT | 0.942 / 0.753 | 33.59% | 35.94% | 41.41% | 45.31% | 18.33% / 20.00% | reject |
| E325/E327 | enriched pairwise | 0.566 / 0.102 | 28.91% | 35.16% | 42.19% | 45.31% | 16.67% / 20.00% | reject |

判断：
- GBDT 候选级分类信号很强，且 rank1 match 变高，但 K5/K20 明显低于 E166；点式概率仍与 top-k 结构生成目标错位。
- Pairwise enriched selector 也未改善 K5/K20，说明当前 GT-free feature family 不能靠同池重排解决 geometry compatibility。
- 当前 best 仍是 E166/E181 `32.03/40.62/44.53/45.31`；full test 继续关闭。

## 2026-06-14 E328-E333 sourceexpand proposal-quality smoke

- Changed candidate realization with source-residual `sourceexpand`, not fixed E166 reranking. Added `--exclude-self-source` to `opentry_render_source_residual_geometry.py`.
- Initial train256 sourceexpand CPU render had no output after about 4 minutes, so it was interrupted; train32 excl-self smoke completed with `363` rendered rows, rank1 SG-ok `90.63%`.
- E329 train32 feature table: rows `363`, positive rate `62.26%`, full match@5 `84.38%`; rows>=7 has only 2 samples, so this is not a formal training set.
- E329 val128 sourceexpand feature table from existing E276: rows `5,502`, positive rate `8.72%`, raw sourceexpand `match@1/5/20/50=23.02/28.57/38.89/44.44%`, rows>=7 `10.17/10.17/15.25/18.64%`.
- E330/E332 sourceexpand GBDT quality: AUC/AP `0.827/0.397`, official full `14.06/27.34/41.41/43.75%`, rows>=7 `5.00/11.67/18.33/18.33%`.
- E331/E333 sourceexpand MLP quality: AUC/AP `0.876/0.461`, official full `20.31/29.69/41.41/43.75%`, rows>=7 `6.67/8.33/15.00/18.33%`.
- Both are below E166/E181 `32.03/40.62/44.53/45.31`; no val512/full-test gate.
- Decision: sourceexpand changed-proposal pool is currently weaker in early ranks; quality models trained from tiny train32 labels damage W/A early-rank. Next step needs larger excl-self train sourceexpand pool or a true source/free-param proposal generator.

## 2026-06-14 Batch E334-E338 - merged E166 + source-residual pool diagnostic

- Added `opentry_merge_compat_feature_tables.py` and made `proposal_pool` an explicit dynamic-selector categorical feature.
- E334 merged E323 enriched E166 features with E328 source-residual features; writes only under `opentry_3`.
- Val merged pool has 11,101 rows over 126 nonempty val128 samples. Source-residual adds only 3 new positive samples beyond E166; all-candidate positive-any ceiling is 48.41%.
- Pool-order diagnostic is `23.02/37.30/41.27/46.03`, so the added tail mainly affects K50 rather than K5/K20.
- E335/E337 pool-aware GBDT: AUC/AP `0.930/0.721`; official val128 `match@1/5/20/50=30.47/37.50/42.19/47.66`, rows>=7 `13.33/16.67/18.33/20.00`.
- E336/E338 pool-aware pairwise: AUC/AP `0.883/0.444`; official val128 `24.22/35.16/42.97/47.66`, rows>=7 `6.67/11.67/18.33/20.00`.
- Decision: reject mixed-pool selector for freeze gate. It can recover a little K50 tail, but K5/K20 remain below E166/E181 `32.03/40.62/44.53/45.31`.
- Full test remains closed; no test labels, oracle rerank, or CrystaLLM test candidates were used.
- Next: stop selector-over-existing-tail work. Improve generated geometry proposals, especially rows>=7 source-conditioned free-param/lattice compatibility with caching/progress.

## 2026-06-14 Batch E340-E343 - rows>=7 sourceexpand selector smoke

- Avoided ID collision with the prior merged-pool E334-E338 batch; artifact dirs keep existing names, logical experiment IDs are E340-E343.
- Updated `opentry_render_source_residual_geometry.py` with `--start-index`, `--min-row-count`, `--progress-every`, and partial JSONL resume support.
- E340 rows>=7 train64 sourceexpand3/K5 render: 64 samples, 852 rows, rank1 SG-ok `87.50%`, overall composition/readable `94.01%`; total eligible rows>=7 train predictions in this file: 170.
- E341 train feature labels: 852 rows, 64 rows>=7 samples, positive rate `9.15%`, train `match@1/5/20=21.88/26.56/32.81%`, W/A-hit but match-fail rate `80.00%`.
- Validation sourceexpand baseline remains raw val128 `23.02/28.57/38.89/44.44%`, rows>=7 `10.17/10.17/15.25/18.64%`.
- E342 rows7 GBDT: AUC/AP `0.828/0.295`; official val128 full `21.88/28.12/42.19/43.75%`, rows>=7 `10.00/13.33/16.67/18.33%`.
- E343 rows7 MLP: AUC/AP `0.755/0.238`; official val128 full `22.66/32.81/40.62/43.75%`, rows>=7 `8.33/13.33/16.67/18.33%`.
- Decision: reject. Rows>=7 train labels are more relevant than train32, but selectors still trail E166/E181 K5/K20 and do not fix rows>=7.
- Note: early e338/e339 evaluator runs without `--max-records 128` used full-val denominator and are invalid; use e338b/e339b only.
- Next: stop sourceexpand selector scaling until raw proposal early-rank quality improves; feature builder needs sample-id filtering before larger sparse train pools.

## 2026-06-14 Batch E344-E348 - free-pattern proposal and sparse compatibility smoke

- E344/E345 completed `row_aligned_chem_freepattern_quality` val32: self-scored top50 had rank<=20 SG-ok `100%`, but StructureMatcher matched E166 first-32 exactly at full `34.38/43.75/43.75/43.75` and rows>=7 `22.22/22.22/22.22/22.22`.
- Decision: do not scale free-pattern source penalty to val128; it does not improve early match or rows>=7 on the aligned prefix.
- E346 patched feature building with `--sample-ids-file`, `--restrict-to-rendered-samples`, `--min-row-count`; filtered smoke processed only 31 rendered samples / 596 top20 rows.
- E346 diagnostics: full positive rate `18.79%`, rows>=7 positive rate `3.87%`, W/A-hit match-fail rate `70.00%`.
- E347/E348 added sparse `sgd` selector support to avoid dense GBDT stalls, trained on train-only E318 labels and evaluated on E346 free-pattern val32.
- E347 standard SGD: AUC/AP `0.755/0.360`, but match dropped from baseline `35.48/45.16/45.16` to `25.81/38.71/45.16`; rows>=7 K5 dropped to `16.67%`.
- E348 rows-weighted SGD: AUC/AP `0.707/0.342`, match `29.03/38.71/45.16`; rows>=7 `5.56/16.67/22.22`.
- Decision: reject. The selector improves W/A/diversity at top5 but damages match, so train-E166 compatibility signals do not transfer to the free-pattern proposal pool.
- Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears the +5pp target, no full test.
- Next: stop re-ranking weak proposal pools; train or construct a proposal generator that improves raw K5/K20 on val128 before selector scaling. Use new sample filtering for sparse feature builds.

## 2026-06-14 Batch E349-E353 - free-pattern val128 follow-up

- Despite the val32 reject, ran one val128 follow-up to measure whether the free-pattern source/free-param proposal has a small-prefix artifact.
- E349 rendered/self-scored `row_aligned_chem_freepattern_quality` val128 K100 -> top50: 126/128 samples, 5,603 rows, rank<=5 SG-ok `100%`, composition exact `100%`.
- E350 official val128: full `match@1/5/20/50=32.03/41.41/44.53/45.31%`, RMSE@1/5/20 `0.0942/0.1357/0.1496`.
- E350 rows>=7: `match@5/20/50=18.33/18.33/20.00%`; this regresses versus E166 rows>=7 `18.33/20.00/20.00%`.
- E351 feature labels on E350 pool: 5,603 rows, positive rate `9.42%`, rows>=7 positive rate `1.32%`, W/A-hit but match-fail `71.09%`.
- E352 GBDT trained on train-only E318 labels: AUC/AP `0.940/0.747`, but official full `31.25/35.94/42.97/45.31%`; reject.
- E353 MLP trained on the same labels: AUC/AP `0.903/0.621`, official full `28.12/35.94/42.97/45.31%`; reject.
- Interpretation: free-pattern proposal gives only a tiny K5 bump below the `41.58%` target and hurts rows>=7; pointwise compatibility selectors again overfit candidate-level labels and damage top-k match.
- Full test remains closed. Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears the +5pp target.
- Next: stop free-pattern+selector scaling; change the proposal generator so raw val128 K5/K20 and rows>=7 improve before any new selector.

## 2026-06-14 Batch E354-E363 - learned source-mode proposal negative result

- Tested two render-time source/free-param proposal variants, not post-hoc reranking: E226 source-transfer selector and E258 rendered-health selector, both with E250 gated residual geometry, source-mode mix3+rank0, val128 K7.
- E354/E226 render health: 2,250 rows, 126/128 nonempty, rank<=5 SG-ok `92.62%`, composition exact `96.72%`; weaker than E166-style rendering.
- E357 E226 global self-score official val128: full `match@1/5/20=30.47/36.72/39.84%`, RMSE `0.1471/0.1589/0.1740`; rows>=7 `11.67/16.67/18.33%`.
- E358/E258 render health: 2,250 rows, 126/128 nonempty, rank<=5 SG-ok `93.28%`, composition exact `97.38%`.
- E361 E258 global self-score official val128: full `match@1/5/20=28.91/35.94/41.41%`, RMSE `0.1277/0.1527/0.1858`; rows>=7 `11.67/16.67/18.33%`.
- Both variants are below current E166/E181 `32.03/40.62/44.53%`; no val512/full-test gate.
- E362/E363 feature diagnostics: positive rate about `9.7%`, rows>=7 positive rate only `2.8-2.9%`, W/A-hit but match-fail `66-67%`.
- Decision: reject existing source-mode selectors as a proposal solution. Next work needs a new source/free-param proposal generator that improves raw val128 K5/K20 and rows>=7 before any more selector training.

## 2026-06-14 Batch E364-E374 - train source-success prior negative result

- Built E364 train-only source success prior from E318 train labels: train positive rate `44.91%`, rows>=7 positive rate `10.30%`, 4,502 kept source ids; no test data used.
- E365/E368 applied the prior in render-time source selection (`row_aligned_chem_sourceprior_quality`), then GT-free global self-score on val128.
- E368 official val128: full `match@1/5/20=31.25/38.28/43.75%`, RMSE `0.0748/0.1196/0.1499`; rows>=7 `13.33/18.33/20.00%`.
- E369/E372 added free-pattern compatibility to the same source prior.
- E372 official val128: full `match@1/5/20=31.25/39.84/43.75%`, RMSE `0.0748/0.1329/0.1452`; rows>=7 `13.33/18.33/18.33%`.
- Both are below E166/E181 `32.03/40.62/44.53%`; no val512/full-test gate.
- E373/E374 feature diagnostics: global positive rate about `14.8%`, but rows>=7 positive rate only `2.8%`; W/A-hit but match-fail remains `68-69%`.
- Decision: reject this scalar source-success-prior family. It increases global candidate density but does not fix complex geometry/free-param conversion.
- Current best remains E166/E181 val128, with only match@1 above +5pp. Full test remains closed.
- Next: train or construct a genuinely new source/free-param proposal model for rows>=7 source/W-A compatibility; do not continue source penalty/free-pattern scalar tweaks.

## 2026-06-14 Batch E375-E385 - rows>=7 residual geometry negative result

- Patched `opentry_build_source_residual_examples.py` with train/val row/atom filters and built E375 rows>=7 residual examples: train512/val256, no skipped source, train rows>=7=512.
- Trained E376 gated source-conditioned residual geometry model on E375 for 5 epochs; best rows>=7 val loss at epoch 5 (`1.9446`).
- E377/E380 rendered E84 val128 with E376 and one source: global self-score full `match@1/5/20=29.69/35.94/36.72%`, RMSE `0.2317/0.2435/0.2486`; rows>=7 `13.33/16.67/16.67%`.
- E381/E384 expanded top-3 sources with the same model: global self-score full `28.91/33.59/39.84%`, RMSE `0.2015/0.2143/0.2384`; rows>=7 `10.00/15.00/16.67%`.
- E385 diagnostics on E384 pool: positive rate `9.92%`, rows>=7 positive rate `2.60%`, W/A-hit but match-fail `64.13%`.
- Decision: reject rows>=7-only residual MSE training and sourceexpand3. It hurts RMSE and does not improve complex match.
- Current best remains E166/E181; full test remains closed.
- Next: residual geometry needs a different objective or richer multimodal proposal labels; do not continue small rows7 MSE residual variants.

## 2026-06-14 Batch E386-E392 - anchored group selector negative result

- Added `opentry_train_anchored_group_selector.py`: group-level selector over `wa_source`/`wa`/`skeleton_source`, with first-rank anchors and `max_per_wa`; it reuses the dynamic selector feature blocklist, excluding labels, target keys/row_count, ids, CIF text and hit flags.
- E386 enriched the free-pattern val128 feature table with the same GT-free source/rendered context features as E323: 5,603 rows, source found rate `100%`, no missing source rows.
- E387/E388 GBDT anchored-group attempts on E166/free-pattern pools were interrupted because they produced no useful output after waiting; no metrics are counted.
- E389 logreg anchored E166 pool, official E391 val128: full `match@1/5/20=32.03/39.06/42.97%`, RMSE `0.0942/0.1172/0.1448`; rows>=7 `15.00/18.33/20.00%`.
- E390 logreg anchored free-pattern pool, official E392 val128: full `match@1/5/20=32.03/40.62/44.53%`, RMSE `0.0942/0.1297/0.1508`; rows>=7 `15.00/18.33/20.00%`.
- E392 ties E166/E181 on full K1/K5/K20 but does not recover E350's tiny K5 bump and remains below the K5 +5pp target `41.58%`.
- The fixed-pool anchored selector preserves rank1 but still cannot improve hit@5/hit@20 or complex rows; W/A-hit but match-fail remains about `45.9%` globally and `75.6%` for rows>=7.
- Decision: reject this anchored fixed-pool selector family. Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears +5pp and full test remains closed.
- Next: stop selecting within E166/free-pattern candidate pools. Build a raw proposal generator/objective that improves geometry/free-param compatibility before self-score, especially rows>=7.

## 2026-06-14 Batch E393-E399 - match-aware source-mode proposal negative result

- E393 converted E323 train/val rendered labels into source-mode examples with runtime source features only; train positive-any `44.23%`, val positive-any `14.59%`, val oracle source gain mean `0.313`.
- E394 trained a match-aware source-mode model for 12 epochs, but final val selected gain vs heuristic was `-0.0018` and improved-over-heuristic rate was `0.00%`.
- E395/E396 rendered E84 val128 with E394 + E250 residual geometry before self-score; raw full `match@1/5/20=21.09/31.25/39.84%`, rows>=7 `6.67/13.33/18.33%`.
- E397/E398 global self-score improved health but remained weak: full `28.12/36.72/39.06%`, RMSE `0.1161/0.1603/0.1693`, rows>=7 `13.33/16.67/18.33%`.
- E399 diagnostics: positive rate `9.00%`, rows>=7 positive rate `2.08%`, W/A-hit but match-fail `69.57%`.
- Decision: reject match-aware source-mode from existing runtime features. It changes proposals before rendering but still cannot beat E166/E181 or improve rows>=7.
- Current best remains E166/E181; no val512/full-test gate.

## 2026-06-14 Batch E400-E407 - train-positive source-bank proposal negative result

- E400 built a train-only source/template bank from E318 train rendered-candidate labels: 19,850 train rows, 8,915 positives, 310 rows>=7 positives, 8,884 bank entries, 4,040 unique sources, 2,406 W/A keys.
- Patched `opentry_render_wyckoff_cifs_e07e08.py` to support `row_aligned_chem_successbank_quality` and anchored `row_aligned_chem_successbank_tail_quality`; source-bank JSON must live under `opentry_3`.
- E401/E403 direct source-bank replacement was a hard negative: 112/128 rendered samples, full `match@1/5/20/50=21.09/28.91/32.03/35.16%`, rows>=7 `1.67/3.33/5.00%` for K1/5/20.
- E404 initial anchored tail run was interrupted at 112/128 because source-bank retrieval was too broad; it produced no final JSONL and is not used for metrics.
- After reducing source-bank search limits, E404b/E406 anchored tail completed with 126/128 rendered samples.
- E406 official val128: full `match@1/5/20/50=32.03/39.84/43.75/45.31%`, RMSE `0.0942/0.1230/0.1484`, rows>=7 `15.00/18.33/20.00%`.
- E406 improved W/A diversity (`W/A@5=70.31%`, unique W/A@5=3.06) but still dropped match@5/20 below E166/E181 `40.62/44.53%`.
- E407 diagnostics on E405 top20: full positive rate `14.54%`, rows>=7 positive rate `2.74%`, W/A-hit but match-fail `64.84%`.
- Decision: reject source-ID success-bank replacement/tail. It transfers train-positive source IDs but not the row-level free-param/lattice compatibility needed for StructureMatcher match.
- Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears +5pp. Full test remains closed.

## 2026-06-15 Batch E408-E414 - row-level template bank negative result

- Patched `opentry_render_wyckoff_cifs_e07e08.py` with train-only row-level success-template pools and two strategies: `row_aligned_chem_rowbank_quality` and `row_aligned_chem_rowbank_freepattern_quality`.
- E408/E410 rowbank-quality rendered 8,151 rows but only 107/128 samples survived; rank1 SG-ok was `82.24%`.
- E410 official val128: full `match@1/5/20/50=14.06/21.88/26.56/29.69%`, RMSE `0.1900/0.1719/0.1808`; rows>=7 `match@1/5/20/50=0/0/0/0%`.
- E411/E413 rowbank-freepattern had the same coverage health: 8,151 rows, 107/128 samples, rank1 SG-ok `82.24%`.
- E413 official val128: full `match@1/5/20/50=14.06/21.88/27.34/29.69%`, RMSE `0.1887/0.1724/0.1887`; rows>=7 still `0/0/0/0%`.
- E414 diagnostics on E412 top20: positive rate `9.98%`, rows>=7 positive rate `0.00%`, W/A-hit but match-fail `68.92%`.
- Decision: reject cross-sample row-level free-param template grafting. It increases W/A/skeleton presence but breaks source-consistent geometry and complex-row matching.
- Current best remains E166/E181 val128 `32.03/40.62/44.53/45.31`; only match@1 clears +5pp. Full test remains closed.
- Next: build source-consistent row/lattice/free-param generation or joint templates, not another rowbank scalar/freepattern variant.

## 2026-06-15 Batch E415-E428 - val512 baseline and pointwise selectors

- E415 attempted a val512 render with E84, but E84 only had 128 prediction rows; E415 is invalid as val512 evidence.
- E416 regenerated count-aware skeleton candidates for val512: full skeleton@50 `88.28%`, rows>=7 skeleton@50 `87.11%`.
- E417b budgeted neural assignment DP: full W/A@5/20 `42.19/54.10%`, rows>=7 W/A@20 `35.11%`.
- E418 fixedmask prior DP: full W/A@50 `52.93%`, rows>=7 W/A@50 `22.22%`.
- E419/E420 merged neural-first DP candidates and built 512-row renderer predictions; 509/512 samples had candidates.
- E421/E423 E166-style val512 baseline: full `match@1/5/20/50=33.01/40.82/45.51/48.24%`, RMSE `0.1095/0.1201/0.1309`; rows>=7 `16.89/18.67%` at K5/20.
- E424 val512 feature labels: positive rate `17.83%`, rows>=7 positive rate `3.04%`, W/A-hit but match-fail `64.68%`.
- E425/E427 GBDT selector: full `33.59/40.62/45.51%`, rows>=7 `17.33/18.67%`; K1 rises but K5 drops and K20 is unchanged.
- E426/E428 MLP selector: full `28.71/38.28/45.51%`, rows>=7 `14.67/18.67%`; clearly worse.
- Decision: val512 confirms only match@1 clears +5pp; fixed-pool pointwise selectors do not solve K5/K20 or rows>=7. Full test remains closed.
- Next: stop pointwise GBDT/MLP selector scaling on this pool; improve proposal/generator positive density for rows>=7 before more selector work.

## 2026-06-15 Batch E429-E437 - match-aware source-mixture residual geometry negative result

- E429 built a new source-consistent residual cache from E393 match-aware source-mode examples: train 2,048 groups / 4,379 examples, val 512 groups / 1,053 examples.
- E429 has more complex coverage than old E242: rows>=7 train/val groups `286/182`; atoms>=12 train/val groups `951/481`.
- Leakage guard: source records from train, train targets train, val targets val, no test records, no StructureMatcher labels in residual target.
- E430 trained a gated source residual geometry model on E429; best val loss `1.80265` at epoch 3.
- E431/E432 one-source val128 raw render: 126/128 samples, 1,256 rows, full `match@1/5/20=17.19/26.56/28.91%`, RMSE `0.1939/0.2188/0.2147`.
- E432 rows>=7 was very poor: `match@1/5/20=3.33/6.67/6.67%`, despite full W/A@20 `77.34%`.
- E433/E434 sourceexpand3 raw render: 126/128 samples, 3,768 rows, full `17.19/24.22/30.47%`, rows>=7 `3.33/5.00/6.67%`.
- E436/E437 global self-score improved health and full metrics to `22.66/29.69/33.59%`, RMSE `0.1893/0.2205/0.2304`, but rows>=7 remained `1.67/5.00/6.67%`.
- E435 diagnostics on E433 top20: full positive rate `9.05%`, rows>=7 positive rate `1.55%`, atoms>=12 positive rate `5.83%`, W/A-hit but match-fail `71.53%`.
- Decision: reject E429/E430 residual MSE mixture. It lowers raw positive density below E424 and cannot be fixed by self-score.
- Current best remains E423 val512 `match@1/5/20/50=33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: stop residual MSE transfer scaling; train/build match-positive source/free-param/lattice proposals with group/listwise objectives on train labels.

## 2026-06-15 Batch E438-E442 - enriched val512 listwise selector negative result

- E438 enriched E424 val512 features with GT-free source/rendered context: 9,717 rows, source found rate `100%`, no missing sources.
- Enrichment uses val formula/SG and train-source metadata only; target row_count/keys remain label/reporting fields and are blocked by selector feature policy.
- E439 pairwise_mmr trained from E323 enriched train to E438 val512; fit used 76,792 pair rows from 583 samples with positive/negative pairs.
- E441 official val512 pairwise_mmr: full `match@1/5/20=27.73/36.91/45.51%`, RMSE `0.1491/0.1274/0.1309`.
- E441 rows>=7: `match@5/20=14.67/18.67%`; atoms>=12 `31.22/39.52%`.
- Decision E441: reject. K20 only ties E423, while K1/K5 and rows>=7 K5 are worse.
- E440 group_bundle trained 6,163 W/A groups with train group positive rate `39.06%`.
- E442 official val512 group_bundle: full `31.25/40.23/43.75%`, RMSE `0.1339/0.1430/0.1306`.
- E442 rows>=7: `16.89/17.78%`; atoms>=12 `34.50/37.77%`.
- Decision E442: reject. It is close on K1/K5 but below E423 and regresses K20/atoms>=12/rows>=7 K20.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; no full-test gate.
- Next: stop fixed-pool listwise selector scaling on E423/E424; move to raw proposal generation that increases match-positive density before selection.

## 2026-06-15 Batch E443-E451 - pre-render learned source proposal smoke

- Added `opentry_train_source_proposal_model.py` and patched `opentry_render_wyckoff_cifs_e07e08.py` with `row_aligned_chem_sourceproposal_quality` and `row_aligned_chem_sourceproposal_tail_quality`.
- E443 trained a HistGBDT source proposal model from E323 train source-pair labels using only pre-render GT-free source/W-A metadata; post-render fields, IDs, target row_count/keys, W/A-hit fields, and labels are blocked.
- E443 internal val128 source-pair metrics: AP `0.7909`, ROC-AUC `0.9415`, hit@1/5/20 `33.33/38.10/42.86%`; rows>=7 hit@5/20 `18.64/20.34%`.
- Initial larger cache attempts were too slow because per-source alignment features are expensive; E444 completed a val64 smoke cache with cheap-KNN source pool: 64 selected records, 63 samples with entries, 449 W/A entries, 3,592 scored source rows.
- E445 direct sourceproposal render and E447 global self-score stayed healthy but did not improve match.
- Same-sample val64 comparison:
  - E449 baseline subset: full `match@1/5/20=32.81/42.19/43.75%`, RMSE `0.1199/0.1449/0.1455`, rows>=7 `20.00/20.00%` at K5/20.
  - E450 direct sourceproposal: full `32.81/40.62/40.62%`, RMSE `0.1465/0.1553/0.1472`, rows>=7 `23.33/23.33%`.
  - E451 anchored-tail sourceproposal: full `32.81/42.19/43.75%`, RMSE `0.1194/0.1438/0.1510`, rows>=7 `23.33/23.33%`.
- Decision: reject this exact cheap-KNN pre-render sourceproposal for gate. Direct damages full K5/K20; tail only ties baseline overall and the rows>=7 bump is small val64 smoke evidence.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: if source proposal continues, make cache construction much faster and rows>=7-focused, or move to a source-consistent free-param/lattice generator rather than another source-only proposal.

## 2026-06-15 Batch E452-E457 - train-manifold param/lattice variants

- Patched renderer with `geometry-param-variant-mode=manifold_params` and `manifold_params_lattice`.
- Idea: keep source context fixed, but generate multiple free-parameter variants by cyclic interpolation from copied source params toward train-set orbit/symbol parameter quantiles; lattice variant also sweeps train VPA quantiles.
- Same val64 subset as E449 baseline:
  - E449 baseline: full `match@1/5/20=32.81/42.19/43.75%`, RMSE `0.1199/0.1449/0.1455`, rows>=7 `20.00/20.00%`.
  - E456 `manifold_params`: full `31.25/37.50/40.62%`, RMSE `0.1127/0.1470/0.1356`, rows>=7 `16.67/16.67%`.
  - E457 `manifold_params_lattice`: full `31.25/37.50/40.62%`, RMSE `0.1127/0.1432/0.1356`, rows>=7 `16.67/16.67%`.
- W/A-hit but match-fail at K20 worsened from E449 `51.02%` to `56.00%` for both manifold variants.
- Decision: reject. Moving source-copied params toward unconditional train quantiles breaks match-positive geometry more often than it helps.
- Current best remains E423 val512; full test remains closed.
- Next: avoid unconditional param/lattice sweeps. If free-param generation continues, it must be learned from match-positive/hard-negative source-consistent examples directly.

## 2026-06-15 Batch E458-E481 - microvariant feature pool and selector smoke

- Added `opentry_filter_jsonl_by_repr_condition.py` to build train-only rows>=7 prediction subsets without using row_count as a model feature.
- E458-E477 built a merged train microvariant pool: train64 baseline/params/lattice plus train rows>=7 baseline/params/lattice; 127 train samples, 11,515 rows after CIF dedupe.
- The merged val64 pool used the E449 same-sample baseline plus E456/E457 microvariants: 63 rendered samples, 5,791 rows.
- Critical diagnostic: val64 microvariants added `0` new positive samples beyond baseline; all-candidate K50 remained `44.44%`, rows>=7 K20 stayed `20.00%`.
- E478/E480 rows>=7-weighted GBDT selector official val64: full `match@1/5/20/50=20.31/31.25/37.50/42.19%`, RMSE `0.1875/0.1469/0.1488`; rows>=7 `20.00/20.00%`.
- E479/E481 pairwise MMR selector official val64: full `31.25/35.94/42.19/42.19%`, RMSE `0.1256/0.1277/0.1406`; rows>=7 `20.00/20.00%`.
- Same-sample reference E449 remains better: `32.81/42.19/43.75%`, rows>=7 `20.00/20.00%`.
- Decision: reject microvariant selector scaling. When variants do not add new validation positives, selectors mostly damage top-k ordering.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: require the generator/proposal itself to add new val positives before training more selectors; focus on source-consistent free-param/lattice generation rather than selecting among unconditional variants.

## 2026-06-15 Batch E482-E491 - train-positive free-param bank negative result

- Added `opentry_build_positive_param_bank.py`; it extracts free-param and VPA values only from train StructureMatcher-positive rendered candidates.
- E482 bank from E318 train labels: 19,850 input rows, 8,915 positive rows extracted, 3,977 param buckets, 196 VPA buckets, 19,857 row-param values.
- Patched renderer with `positive_params` and `positive_params_lattice` modes, blending copied source params/lattice toward train-positive quantiles.
- E483/E484 rendered E420 val64 K50 with healthy CIF output: 63/64 samples, 2,170 rows, composition exact/readable essentially unchanged.
- Raw feature labels:
  - E487 `positive_params`: `match@1/5/20/50=25.40/39.68/41.27/41.27%`, positive rate `8.43%`, rows>=7 `10.00/16.67/16.67/16.67%`.
  - E488 `positive_params_lattice`: same match ceiling, positive rate `8.39%`, rows>=7 unchanged.
- E489 merged baseline + positive-param variants: `0` new positive samples beyond the baseline pool; merged ceiling stayed `match@50=44.44%`, rows>=7 K20 stayed `20.00%`.
- E490/E491 official global self-score eval for both variants: full `match@1/5/20/50=32.81/37.50/40.63/40.63%`, RMSE `0.1148/0.1331/0.1376`, rows>=7 K5/K20 `16.67/16.67%`.
- Same-sample E449 baseline remains better at `32.81/42.19/43.75%`, rows>=7 `20.00/20.00%`.
- Decision: reject train-positive-param quantile variants. They preserve W/A/skeleton recall but do not create new StructureMatcher-positive geometries.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: stop aggregate positive-quantile variants; use train positive-vs-hard-negative evidence to build a source-consistent generator that can add new val positives before selector training.

## 2026-06-15 Batch E492-E494 - collision-relief lattice negative result

- Added `opentry_lattice_collision_relief_variants.py`, a GT-free post-render proposal that scales CIF cell lengths when candidate `self_min_distance` is below a train-derived collision threshold.
- Train diagnostic behind the smoke: rows>=7/atom>=12 W/A-hit positives have higher min-distance than hard negatives, so the first test used `trigger=1.65`, `target=1.75`, `max_scale=1.16`.
- E492b generated variants on the same val64 IDs as E449/E473: 2,399 rows, 58 samples with output, 626 triggered source candidates.
- E493b variants-only raw labels were poor: full `match@1/5/20/50=20.69/20.69/22.41/24.14%`, RMSE `0.2160/0.2151/0.2240`, rows>=7 K5/K20 `13.33/13.33%`.
- W/A-hit but match-fail worsened to `82.16%`, so cell scaling alone often keeps the symbolic hit but breaks geometry match.
- E494 merged E473 baseline + collision-relief variants: `0` new positive samples beyond baseline; merged full K50 stayed `44.44%`, rows>=7 K20 stayed `20.00%`.
- Decision: reject lattice-only collision relief and do not train selectors on this pool.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: a viable proposal likely needs coupled fractional-coordinate and lattice moves learned from positive-vs-hard-negative examples, not independent aggregate parameter quantiles or scalar cell expansion.

## 2026-06-15 Batch E495-E497b - coordinate+lattice relax negative result

- Added `opentry_coordinate_relax_lattice_variants.py`, a GT-free post-render proposal that applies PBC pair-repulsion to fractional coordinates while lightly scaling lattice.
- Full E495 generation was too slow with 3 variants/candidate and 12 relaxation steps, so it was interrupted before output.
- E495b small budget used the same E449/E473 val64 IDs, top20 input candidates, 1 variant/candidate, `scale=1.03`, `cutoff=1.60`, 6 steps, step size `0.06`.
- E495b generated 555 variants across 50 samples; inference used only candidate CIF, atom count, and self min-distance.
- E496b variants-only raw labels: full `match@1/5/20/50=20.00/24.00/26.00/26.00%`, RMSE `0.1673/0.2124/0.2324`, rows>=7 K5/K20 `13.79/13.79%`.
- W/A-hit but match-fail remained very high at `80.99%`.
- E497b merged E473 baseline + coord-relax variants: `0` new positive samples beyond baseline; full K50 stayed `44.44%`, rows>=7 K20 stayed `20.00%`.
- Decision: reject coordinate/lattice relaxation heuristics and do not train selectors on this pool.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: stop post-render coordinate surgery; move to a source/free-param proposal model or faster source-free-param cache that changes the rendered source context before CIF creation.

## 2026-06-15 Batch E498-E501 - row-aligned sourceproposal timeout/tooling and negative result

- Patched `opentry_train_source_proposal_model.py` with `--record-timeout-seconds`; timed-out records roll back partial entries and cache writing still completes.
- Two exploratory cache attempts were interrupted before patch/final output: row-aligned val64 and cheap48 val64 both hit long-tail source scoring.
- E498d rebuilt a row-aligned sourceproposal cache with 45 s record timeout: 64 records, 449 entries, 3,592 scored rows, 0 timeout/failed records.
- E499 rendered sourceproposal-tail K50 using E498d cache: 63/64 samples, 2,767 rows, readable/formula/atom/composition exact all `1.0`.
- E500 raw labels: full `match@1/5/20/50=25.40/30.16/38.10/42.86%`, RMSE `0.0549/0.0840/0.1403`, rows>=7 K5/K20 `10.00/10.00%`.
- E500 W/A-hit but match-fail was `69.08%`, worse than needed for the geometry bottleneck.
- E501 merged E473 baseline + E499 sourceproposal-tail: `0` new positive samples beyond baseline; full K50 stayed `44.44%`, rows>=7 K20 stayed `20.00%`.
- Decision: reject this row-aligned sourceproposal-tail branch and do not train selectors on it.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: source-ID/context proposals are exhausted unless paired with a genuinely new free-param generator; do not keep rerunning source allocation variants.

## 2026-06-15 Batch E502-E505 - source row-shift free-param diagnostic

- Patched `opentry_render_wyckoff_cifs_e07e08.py` with `source_row_shift`, which keeps the selected source but shifts row-to-row free-param assignment among near-tie compatible source rows.
- E502 rendered E420 val64 K50: 63/64 samples, 2,798 rows, readable/formula/atom/composition exact all `1.0`, overall SG-ok `96.39%`.
- E503 raw labels were weak: full `match@1/5/20/50=20.63/22.22/33.33/38.10%`, RMSE `0.0491/0.0333/0.1246`, rows>=7 `0.00/0.00/0.00/6.67%`.
- E503 W/A-hit but match-fail remained high at `71.97%`; rows>=7 positive rate was only `0.16%`.
- E504 merged baseline + row-shift: `1` new positive sample beyond baseline, full positive-any ceiling rose to `46.03%`, but rows>=7 positive-any stayed `20.00%`.
- E505 GBDT selector on the merged pool: full `match@1/5/20/50=31.75/38.10/39.68/46.03%`, RMSE `0.0957/0.1331/0.1414`, rows>=7 K5/K20 `20.00/20.00%`.
- Decision: reject for gate. Row-shift proves source-conditioned free-param perturbation can add a rare positive, but it does not improve K5/K20 or complex rows.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp. Full test remains closed.
- Next: do not scale exact row-shift; build a rows>=7-targeted source/free-param generator that adds complex positives before another selector.

## 2026-06-15 Batch E506-E509 - rows>=7 positive-nearest free-param smoke

- Patched `opentry_build_positive_param_bank.py` with `--min-target-row-count`; patched renderer with `positive_nearest_complex`.
- E506 bank used only train `target_row_count>=7` positive candidates: 310 positive rows, 899 parameter buckets, 38 VPA buckets, 5,991 row-param values.
- E507 rendered val64 K50: 63 samples, 2,765 rows, composition exact/readable `100%`, overall SG-ok `94.07%`.
- E508 raw labels: full `match@1/5/20/50=25.40/28.57/38.10/42.86%`, RMSE `0.0549/0.0618/0.1399`, rows>=7 K5/K20/K50 `10.00/10.00/20.00%`.
- E508 rows>=7 positive rate rose to `1.24%`, but W/A-hit match-fail stayed high at `69.28%`.
- E509 merged baseline + rows7-positive-nearest: `0` new positive samples beyond baseline; full K50 stayed `44.44%`, rows>=7 K20 stayed `20.00%`.
- Decision: reject. Nearest-to-source positive params reproduce baseline-positive candidates but do not create new complex positives.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.
- Next: stop positive-nearest/quantile parameter-bank variants. Need a generator that changes row-level geometry feasibility, not another bank lookup around copied source params.

## 2026-06-15 Batch E510-E517 - row-feasibility histogram-bank negative result

- Added `opentry_build_row_feasibility_bank.py`; it uses only train rows>=7 W/A-hit candidates and contrasts StructureMatcher positives against hard negatives.
- Added renderer mode `row_feasibility_complex`, which shifts copied source free params toward high log-odds row/free-symbol bins before CIF rendering.
- E510 fine bank: 775 train W/A-hit candidates, 180 positives, 595 negatives, 1,542 kept keys, 25,080 row values.
- E512 fine raw labels: full `25.40/28.57/38.10/41.27%`, RMSE `0.0549/0.0618/0.1399`, rows>=7 K5/K20/K50 `10.00/10.00/16.67%`.
- E513 merged fine bank with baseline: `0` new positive samples; full K50 stayed `44.44%`, rows>=7 K20 stayed `20.00%`.
- E514 coarse bank increased kept keys to 2,058 but used the same 180/595 positive/hard-negative train candidates.
- E516/E517 repeated the same outcome: full `25.40/28.57/38.10/41.27%`, `0` new positives beyond baseline, rows>=7 ceiling unchanged.
- Decision: reject and stop row-feasibility histogram/free-param-bank variants after two negative val runs.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.
- Next: a viable row model must jointly choose row source context/lattice context, not substitute scalar free-param bins around copied source parameters.

## 2026-06-15 Batch E518-E525 - row-pair feasibility bank negative result

- Added `opentry_build_row_pair_feasibility_bank.py`; it learns train-only target-row/source-row log-odds from rows>=7 StructureMatcher positives vs hard negatives.
- Added renderer mode `row_pair_feasible`, which changes target-row to source-row matching inside the selected source before copying free params.
- E518 strict bank used rows>=7 W/A-hit train candidates: 775 candidates, 180 positives, 595 negatives, 8,336 row pairs, 401 kept keys.
- E520 strict raw labels: full `25.40/28.57/38.10/42.86%`, RMSE `0.0549/0.0618/0.1399`, rows>=7 K5/K20/K50 `10.00/10.00/20.00%`.
- E521 strict merge: `0` new positives beyond baseline; full K50 `44.44%`, rows>=7 K20 `20.00%`.
- E522 broad bank removed W/A-hit filtering: 3,009 train candidates, 310 positives, 2,699 negatives, 31,986 row pairs, 1,299 kept keys.
- E524/E525 broad repeated strict output and again added `0` positives beyond baseline.
- Decision: reject and stop row-pair feasibility-bank variants. Same-source row rematching mostly duplicates baseline candidates.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.
- Next: a useful joint model must alter source selection, row mapping, and lattice/source context together, not only rematch rows inside a fixed selected source.

## 2026-06-15 Batch E526-E533 - row-pair-aware source selection negative result

- Added `row_pair_source_feasible_quality` to the renderer. It uses train-only row-pair feasibility to rank source records before rendering, then uses `row_pair_feasible` for source-row mapping.
- E526 strict-bank render: 63/64 samples, 2,163 rows, composition exact/readable `100%`, SG-ok `94.04%`.
- E527/E528 strict raw labels: full `match@1/5/20/50=25.00/37.50/42.19/42.19%`, RMSE `0.0549/0.1410/0.1408`, rows>=7 K5/K20/K50 `16.67/20.00/20.00%`.
- E529 strict merge with baseline: `0` new positive samples; full K50 ceiling `44.44%`, rows>=7 positive-any `20.00%`.
- E530 broad-bank render repeated the same candidate count with SG-ok `94.22%`.
- E531/E532 broad raw labels: full `25.00/37.50/42.19/42.19%`, RMSE `0.0549/0.1408/0.1409`, rows>=7 K5/K20/K50 `16.67/20.00/20.00%`.
- E533 broad merge with baseline: `0` new positive samples; full K50 ceiling and rows>=7 positive-any unchanged.
- Decision: reject and stop row-pair-aware source-selection variants. It improves over same-source row-pair raw K5 but still duplicates baseline positives.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, so full test remains closed.
- Next: do not train selectors on E526/E530 pools. Need a generator that changes lattice/source context or learns source-free-param proposals that add validation positives beyond baseline.

## 2026-06-15 Batch E534-E544 - train-positive geometry bundle transfer negative result

- Patched renderer to optionally write `geometry_lattice` and `geometry_params`, then rendered E315b train256 K50 with E423-style geometry and metadata.
- E534/E535 train metadata pool was healthy: 8,308 rows, train full `match@1/5/20/50=66.02/78.52/84.38/86.33%`, rows>=7 `23.53/35.29/41.18/41.18%`.
- E536 built a train-only geometry bundle bank from positives: 2,729 entries, 58 SGs, 309 skeleton keys, 795 W/A keys, 37 rows>=7 positive rows.
- E537/E538 exact bundle val64: 63/64 samples, 2,169 rows, full `25.00/40.63/43.75/43.75%`, RMSE `0.0699/0.1654/0.1531`, rows>=7 K5/K20/K50 `16.67/20.00/20.00%`.
- E539/E540 exact labels/merge: W/A-hit match-fail `69.41%`, `0` new positives beyond baseline, full positive-any `44.44%`, rows>=7 positive-any `20.00%`.
- E541/E542 broad bundle val64: 63/64 samples, 2,071 rows, full `29.69/35.94/39.06/39.06%`, RMSE `0.1583/0.1565/0.1431`, rows>=7 K5/K20/K50 `16.67/16.67/16.67%`.
- E543/E544 broad labels/merge: W/A-hit match-fail `70.37%`, `1` new full positive beyond baseline, but rows>=7 positive-any stayed `20.00%`.
- Decision: reject both exact and broad bundle transfer for gate. Broad gives one easy full positive but damages direct K5/K20/RMSE and does not improve complex rows.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, full test closed.
- Next: stop train256 bundle scaling/selectors. A useful next route must be rows>=7-specific and increase raw complex positive density before selector/ranker training.

## 2026-06-15 Batch E545-E553 - rows>=7-only geometry bundle transfer negative result

- E545 rebuilt the geometry bundle bank using only train rows>=7 StructureMatcher-positive candidates from E535. The bank is very sparse: 37 entries, 5 SGs, 11 skeleton keys, 12 W/A keys.
- E546/E547 exact rows7-bundle val64: 63/64 samples, 2,169 rows, full `match@1/5/20/50=25.00/39.06/42.19/42.19%`, RMSE `0.0549/0.1526/0.1409`, rows>=7 K5/K20/K50 `16.67/20.00/20.00%`.
- E548/E549 exact labels/merge: W/A-hit match-fail `69.41%`, `0` new positives beyond baseline, full positive-any `44.44%`, rows>=7 positive-any `20.00%`.
- E550/E551 broad rows7-bundle val64: 63/64 samples, 2,167 rows, full `25.00/37.50/40.63/40.63%`, RMSE `0.0549/0.1395/0.1285`, rows>=7 K5/K20/K50 `16.67/20.00/20.00%`.
- E552/E553 broad labels/merge: positive rate `8.86%`, W/A-hit match-fail `69.41%`, `0` new positives beyond baseline, rows>=7 positive-any still `20.00%`.
- Decision: reject and stop rows>=7-only geometry-bundle transfer. Restricting train-positive bundles to complex rows does not create new complex positives and worsens K5/K20.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, full test closed.
- Next: do not train selectors on E546/E550. Need a learned or constructed complex-row generator that changes coupled source/free-param/lattice context beyond copying train-positive bundles.

## 2026-06-15 Batch E554-E564 - rows>=7 positive-geometry residual model negative result

- Added `opentry_build_positive_geometry_residual_examples.py`. It builds cached residual examples from train/val StructureMatcher-positive rendered candidates with geometry metadata; no test labels or records are used.
- E554/E555 produced val64 baseline-style candidates with metadata and labels. E556 rows>=7 positive examples were sparse: train `37`, val `15`.
- E557 trained a small gated source-residual net on E556. Best val loss was epoch 2 (`0.20137`); last val loss rose to `0.20765`, so overfit risk is high.
- E558 with `--min-row-count 7` was a protocol diagnostic only: the renderer filters full val before `max_records`, so it is not E473 first64-comparable.
- E558b/E559 expand1 first64: 631 rows, full `match@1/5/20=26.98/36.51/39.68%`, RMSE `0.1330/0.1732/0.1759`, rows>=7 K5/K20 `13.33/13.33%`, W/A-hit match-fail `58.00%`.
- E560 merge with baseline: `0` new positive samples; full positive-any `44.44%`, rows>=7 positive-any `20.00%`.
- E561/E564 expand3 first64: 1,893 rows, official `match@1/5/20=26.56/34.38/39.06%`, RMSE `0.1330/0.1455/0.1659`, rows>=7 K5/K20 `10.00/16.67%`.
- E563 merge with baseline: `0` new positive samples; full positive-any and rows>=7 positive-any unchanged.
- Decision: reject and stop this tiny positive-residual generator. It raises raw positive rate but not sample-level ceiling and damages complex top-k.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, full test closed.

## 2026-06-15 Batch E565-E572 - larger rows>=7 positive residual negative result

- E565/E565b rebuilt train1024 metadata renders; old E318 label reuse was rejected because `sample_id/rank` aligned but 18,668 core candidate fields mismatched.
- E566b relabeled only train rows>=7 from E565b using train GT: 166 samples, 3,009 candidates, 209 positives, positive rate `6.95%`, no timeout.
- E567 cached 209 train / 15 val rows>=7 positive residual examples; no missing metadata/source and no test records.
- E568 retrained the same gated residual model. Best val loss improved to `0.18437` at epoch 1, but this was only a training sanity signal.
- E569/E572 expand3 first64: official `match@1/5/20=26.56/34.38/39.06%`, RMSE `0.11695/0.13390/0.15321`, rows>=7 K5/K20 `10.00/16.67%`.
- W/A@20 stayed `76.56%`, skeleton@20 `81.25%`, W/A-hit match-fail `63.27%`: the failure remains geometry conversion, not W/A recall.
- E571 merge with E473 baseline added `0` new positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject larger positive-residual branch. More rows>=7 positive examples improved loss/RMSE but did not increase validation positive coverage or match@5/20.

## 2026-06-15 Batch E573-E576 - train pair-delta geometry transfer negative result

- Added `opentry_pair_delta_geometry_variants.py`: train-only same-signature negative->positive lattice/free-param deltas are applied to val candidates; inference uses GT-free candidate metadata plus formula/GT-SG from repr.
- E573 built a sparse delta bank from E566b train labels: 909 raw pairs, 263 kept pairs, 41 signatures.
- Val64 coverage was too narrow: 2,147 input rows seen, only 27 signature-covered rows, 200 rendered variants across 5 samples.
- E574 labels: `0` positives / 200 variants, rows>=7 positives `0`, W/A-hit match-fail `100%`.
- E575 merge with E473 baseline: `0` new positive samples; full positive-any stayed `44.44%`, rows>=7 positive-any stayed `20.00%`.
- E576 official sparse eval: match@1/5/20 all `0.00%`; RMSE null; W/A@20 and skeleton@20 both `3.13%` over the full 64 sample denominator.
- Decision: reject pair-delta signature transfer. It changes the objective versus residual/bundle copy, but train-positive signatures do not cover val complex candidates enough and produce no usable positives.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; only match@1 clears +5pp, full test closed.

## 2026-06-15 Batch E577-E580 - same-sample geometry recombination diagnostic

- Added `opentry_same_sample_geometry_recombine.py`. It recombines donor lattice/free-params from candidates in the same sample; inference is GT-free and uses no labels.
- E577 covered 63/64 samples, 3,108 rendered variants, 732 same-signature donor/recipient pairs.
- E578 labels: full `match@1/5/20/50=25.40/26.98/36.51/44.44%`, positive rate `21.07%`, W/A-hit match-fail `61.66%`.
- rows>=7 remained weak: `match@5/20/50=6.67/10.00/20.00%`, positive rate `3.98%`.
- E579 merge with E473 baseline added `1` full positive sample, raising full positive-any to `46.03%`, but rows>=7 positive-any stayed `20.00%`.
- E580 official direct eval: full `match@1/5/20=25.00/26.56/35.94%`, RMSE `0.0605/0.0719/0.1410`; rows>=7 K5/K20 `6.67/10.00%`.
- Decision: reject for gate. Same-sample recombination has coverage and one easy full positive, but collapses W/A diversity and does not improve complex rows.
- Current best remains E423 val512; full test closed.

## 2026-06-15 Batch E581-E592 - direct GeometryNet smoke negative result

- Tested a train-only direct GeometryNet (`W/A rows -> lattice/free params`) to avoid copying source geometry.
- E581 full-data smoke: 4,096 train / 512 val, best val loss `3.79375` at epoch 4. Initial DataLoader workers failed due sandbox multiprocessing socket permission; reran with `num_workers=0`.
- E583b rendered val64 top20/variant1: 407 rows, 43 samples, SG ok `59.21%`.
- E586 official full `match@1/5/20=15.63/18.75/20.31%`, RMSE `0.2309/0.2425/0.2217`; rows>=7 K5/K20 `0/0%`.
- E587-E588 rows>=7-only contrast: filtered 4,096 train / 512 val rows>=7, best val loss `0.82088`.
- E589/E592 rows7 model rendered 378 rows / 40 samples; official full `7.81/10.94/18.75%`, RMSE `0.1279/0.1850/0.1571`; rows>=7 K5/K20 still `0/0%`.
- E585/E591 merges added `0` positives beyond E473; rows>=7 positive-any stayed `20.00%`.
- Decision: reject direct GeometryNet regression in this form. Lower regression loss does not produce complex-row StructureMatcher positives.

## 2026-06-15 Batch E593-E599 - jitter and bond-length physics smokes

- E593 wrapped_jitter_lattice used the same E420 val64/E554-style setup, GT-free at inference.
- E594 raw: `match@1/5/20/50=25.40/39.68/41.27/41.27%`, rows>=7 `10.00/16.67/16.67%`, W/A-hit match-fail `69.86%`.
- E595 merge with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Added `opentry_bond_length_refine_variants.py`: train-only element-pair distance stats from 1,024 train CIFs guide coordinate/lattice refinement without GT labels at inference.
- E596 produced 2,675 variants across 61 samples; E597 raw `match@1/5/20/50=24.59/32.79/39.34/44.26%`, rows>=7 `10.34/17.24/20.69%`.
- E598 merge added `1` full positive sample but `0` rows>=7 positives; rows>=7 positive-any remained `20.00%`.
- E599 official direct eval: full `23.44/31.25/37.50%`, RMSE `0.0870/0.1305/0.1563`, rows>=7 K5/K20 `10.00/16.67%`, W/A@20 `73.44%`, skeleton@20 `78.13%`.
- Decision: reject both as gates fail. They do not raise complex-row positive-any, so no selector/ranker is allowed on these pools.

## 2026-06-15 Batch E600-E620 - complex-weighted assignment and W/A merge

- E600 trained a full-train AssignmentNet with `complex_weight=8`; best val action top1/top5/MRR = `64.57/89.30/75.96%`.
- E601 narrow DP (`s5p5`) improved symbolic W/A but canonical conversion collapsed to 3,037 unique W/A candidates; E605 CIF eval fell to `30.27/35.16/40.63%`, rows>=7 K5/K20 `15.11/16.44%`.
- E606 wide DP (`s20p20`) restored symbolic W/A diversity: full W/A@5/20/50 `48.83/59.57/67.58%`, rows>=7 `41.78/47.56/53.33%`; conversion kept 8,891 candidates.
- E610 wide-DP CIF eval: full `33.01/39.84/45.31%`, RMSE `0.1105/0.1141/0.1332`, rows>=7 K5/K20 `15.11/18.22%`. This remains below E423 on K5 and rows>=7.
- E611/E615 round-robin merged E420 baseline + E607 wide W/A. Pre-render W/A@5/20 rose to `67.58/77.34%`, but CIF was only `33.40/40.63/45.90%`, rows>=7 `16.44/18.22%`.
- E616/E620 priority merge preserved E420 first, then appended E607. It gave the batch-best K20: `33.01/40.63/46.09%`, RMSE `0.1071/0.1144/0.1347`, rows>=7 `16.44/18.67%`.
- Decision: reject for full-test gate. W/A recall and unique W/A improve, but match@5 still misses `41.58%`, match@20 misses `49.69%`, and rows>=7 does not improve beyond E423.
- Current best remains E423 val512 `33.01/40.82/45.51/48.24`; E620 is only a K20 diagnostic. Full test remains closed.

## 2026-06-15 Batch E621-E628 - coarse pair-delta coverage test

- Added `opentry_coarse_pair_delta_geometry_variants.py`: train-only negative->positive deltas keyed by coarse SG/crystal-system, row bucket, and free-symbol multiset. This deliberately relaxes E573 exact signature matching.
- E621 lattice-only and E622 full-param both covered all 30 rows>=7 samples in val first64, producing 1,494 variants each; E573 had only 5 samples.
- E623 lattice-only labels: rows>=7 `match@1/5/20/50=10.00/10.00/10.00/16.67%`, positive rate `4.55%`, W/A-hit match-fail `90.00%`.
- E624 full-param labels were worse: rows>=7 K50 `13.33%`, positive rate `2.88%`, W/A-hit match-fail `92.55%`.
- E625/E626 merges with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- E627 official lattice-only eval over first64: full `match@1/5/20=4.69/4.69/4.69%`, RMSE `0.1576/0.1575/0.1575`; rows>=7 W/A@20 `60.00%`, skeleton@20 `60.00%`.
- Decision: reject. Coarse deltas solve coverage sparsity but do not create new complex positive samples beyond baseline.
- Current best remains E423 val512; full test remains closed.

## 2026-06-15 Batch E629-E636 - source-free train-prior geometry sampler

- Added `opentry_source_free_prior_geometry_sampler.py`: no val source lattice/free-param copying; it samples lattice/free params from train-positive rendered rows and predicted W/A keys.
- Two variants were tested on val first64 rows>=7 candidates: E629 `uniform` free params and E630 `orbit_prior` free params; both used 209 train-positive rows only.
- Both covered 30 rows>=7 samples, 153 input W/A candidates, and rendered 765 CIFs.
- E631/E632 labels: `0` positives for both variants; W/A-hit match-fail `100%`.
- E633/E634 official eval: full and rows>=7 match@1/5/20 all `0.00%`; RMSE null. rows>=7 W/A@20 `76.67%`, skeleton@20 `80.00%`, unique W/A@20 `3.50`.
- E635/E636 merge with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject. Independent source-free train-prior sampling preserves W/A recall but cannot produce matched geometry.
- Current best remains E423 val512; full test remains closed.

## 2026-06-15 Batch E637-E645 - compatibility-guided source-free proposal scoring

- Added `opentry_compat_guided_source_free_sampler.py`: train-label GBDT compatibility model scores source-free train-prior proposals during generation; inference features remain GT-free through `compat_selector.feature_dict`.
- E637 used orbit-prior source-free proposals + GBDT trained on E566b train rows>=7 labels. It scored 1,932 proposals, retained 1,330 candidates across all 30 val64 rows>=7 samples.
- E639/E641 result: full and rows>=7 match@1/5/20 all `0.00%`; rows>=7 W/A@20 `73.33%`, skeleton@20 `80.00%`, unique W/A@20 `4.70`; 308 W/A-hit rows, W/A-hit match-fail `100%`.
- E638 repeated with stricter GBDT weights (`positive=8`, `row7=3`, `atom12=1.5`). It also retained 1,330 candidates across all 30 rows>=7 samples.
- E640/E642 result: full and rows>=7 match@1/5/20 all `0.00%`; rows>=7 W/A@20 `73.33%`, skeleton@20 `76.67%`; 305 W/A-hit rows, W/A-hit match-fail `100%`.
- E645 baseline rerun on the same val64 reference: full `25.00/39.06/42.19%`, RMSE `0.0549/0.1526/0.1409`, rows>=7 `10.00/16.67/20.00%`.
- E643/E644 merges with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject and stop this branch. Train-label compatibility scoring cannot rescue independent source-free train-prior proposals; next route must change the geometry objective, not just proposal selection.

## 2026-06-15 Batch E646c-E653 - compatibility-guided local geometry search

- Added `opentry_compat_local_geometry_search.py`: train-label GBDT compatibility model guides local lattice/free-param search around E554 rendered candidates. Initial overwide E646 was interrupted; E646b exposed duplicate fallback in `select_diverse`, fixed before valid runs.
- E646c joint search used 6 source candidates/sample, 1 iteration, 6 neighbors, mutating both lattice and free params. It rendered 1,026 unique candidates across all 30 rows>=7 samples.
- E648/E650 joint result: full `match@1/5/20=4.69/7.81/7.81%`, RMSE `0.2476/0.2619/0.2412`; rows>=7 `10.00/16.67/16.67%`; W/A@20 `76.67%`, skeleton@20 `80.00%`; W/A-hit match-fail `89.36%`.
- E647 param-only kept the baseline lattice fixed and searched only free params. It also rendered 1,026 unique candidates.
- E649/E651 param-only result: full `6.25/7.81/7.81%`, RMSE `0.3172/0.2391/0.2355`; rows>=7 `13.33/16.67/16.67%`; W/A@20 `76.67%`; W/A-hit match-fail `89.36%`.
- E652/E653 merges with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject and stop local perturb/search around E554 geometry as configured. It creates positives only in samples already covered by baseline and does not raise the complex-row ceiling.

## 2026-06-15 Batch E654-E661 - contrastive geometry prototype transfer

- Added `opentry_contrastive_geometry_prototype_sampler.py`: train-only positive/negative feature labels build prototype contexts, then predicted val rows>=7 W/A candidates are rendered with prototype lattice/free params and GT-free compatibility scoring.
- E654 strict prototype rendered 616 candidates for all 30 rows>=7 val64 samples. E656/E658: full and rows>=7 match@1/5/20 all `0.00%`; rows>=7 W/A@20 `80.00%`, skeleton@20 `83.33%`, unique W/A@20 `4.43`; W/A-hit match-fail `100%`.
- E655 broad prototype rendered 1,118 candidates. E657/E659: full and rows>=7 match@1/5/20 all `0.00%`; rows>=7 W/A@20 `80.00%`, skeleton@20 `83.33%`, unique W/A@20 `4.47`; W/A-hit match-fail `100%`.
- Broad mode improved rows>=7 composition/readable any@50 from `40.00/43.33%` to `50.00/56.67%`, but still produced `0` StructureMatcher positives.
- E660/E661 merges with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject and stop contrastive prototype/source-free representation transfer as configured. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.

## 2026-06-15 Batch E662-E666 - positive row-sequence geometry decoder

- Added `opentry_positive_rowseq_geometry.py`: train-positive rendered CIFs supervise a probabilistic row-sequence geometry decoder from formula + GT-SG + predicted W/A rows to lattice/free parameters. It does not read test labels.
- E662 trained on 4,096 train positive rendered candidates; only 172 train examples and 23 val examples were rows>=7. Best epoch was 1 (`val_loss=0.4983`), then val lattice NLL overfit sharply by epoch 6.
- E663 rendered val first64 rows>=7 candidates from E554 W/A keys: 462 render attempts, 462 render_ok, 30 samples with output, 161 deduped W/A.
- E664 official eval: full and rows>=7 `match@1/5/20/50=0.00/0.00/0.00/0.00%`, RMSE null. rows>=7 W/A@5/20 `60.00/80.00%`, skeleton@20 `83.33%`, unique W/A@20 `4.83`.
- rows>=7 composition_exact_any@50 and readable_any@50 were both `100.00%`, so the failure is not parse/composition/W/A coverage; it is geometry placement.
- E665 labels found `0` positives among 462 rows; 72 W/A-hit rows all failed match (`W/A-hit match-fail=100%`).
- E666 merge with E555 baseline added `0` positive samples; rows>=7 positive-any stayed `20.00%`.
- Decision: reject this positive-only row-sequence decoder as configured. Current best remains E423 val512 `33.01/40.82/45.51/48.24`; full test remains closed.

## 2026-06-15 Batch E667-E676 - negative-aware row-sequence energy sampler

- Added `opentry_contrastive_rowseq_energy_sampler.py`: a train-label binary energy model sees formula + GT-SG + predicted W/A rows + candidate lattice/free params, with positive and negative rendered candidates. It then scores samples from the E662 row-sequence geometry decoder before rendering.
- E667 trained rows>=7-only energy on E566b: 3,009 train rows, 209 positives. Best fixed-pool val rows>=7 K5/K20 was `13.33/20.00%`, below E555 baseline `16.67/20.00%`.
- E668 rendered 894 energy-filtered rowseq proposals across all 30 rows>=7 val64 samples. E669/E670: match@1/5/20/50 all `0.00%`, RMSE null, 144 W/A-hit rows all failed match.
- E671 merge added `0` positives beyond baseline; rows>=7 positive-any stayed `20.00%`.
- E672 trained full energy on 12,000 E318 train rows. Its fixed-pool full K5/K20 was `25.40/39.68%`, but rows>=7 stayed weak at `10.00/20.00%`.
- E673 rendered the same 894 proposal budget with full-energy filtering. E674/E675 again had match@1/5/20/50 all `0.00%`, 144 W/A-hit rows and `100%` W/A-hit match-fail.
- E676 merge added `0` positives; rows>=7 positive-any stayed `20.00%`.
- Decision: reject energy-over-positive-rowseq-samples as configured. Negative-aware scoring of sampled NLL geometry does not change the geometry basin enough.

## 2026-06-15 Batch E677-E691 - pairfield Adam constrained geometry

- Added `opentry_pairfield_adam_refine.py`: differentiable Adam refinement of fractional coordinates plus isotropic scale, using only train CIF pair-distance and volume-per-atom stats. Inference uses rendered CIFs and predicted W/A metadata only.
- E677 repel and E678 pairfield refined E554 val64 rows>=7 candidates: each output 211 rows over 29 samples, with no candidate failures.
- E679 repel direct eval: full `match@1/5/20/50=4.69/7.81/9.38/9.38%`, RMSE `0.2970/0.2901/0.3181`; rows>=7 `10.00/16.67/20.00/20.00%`; W/A@20 `80.00%`, skeleton@20 `83.33%`.
- E680 pairfield direct eval was worse at K20: full `4.69/7.81/7.81/7.81%`, rows>=7 `10.00/16.67/16.67/16.67%`.
- E681/E683 found repel has 8 positives, W/A-hit match-fail `89.09%`, and adds `1` positive sample beyond E555 baseline, raising rows>=7 all-candidate positive-any from `20.00%` to `23.33%`.
- E682/E684 found pairfield has 9 positives but adds `0` new baseline samples; rows>=7 positive-any stays `20.00%`. Reject pairfield preset.
- E685 initially scaled repel to E421 val128, but E686 was invalid because only 9/55 output sample IDs were in `val.jsonl` first128; this is a sample-order mismatch, not a model result.
- Built aligned repr subset from E685 rendered IDs: 55 rows>=7 samples. E687 repel direct: `match@1/5/20/50=10.91/14.55/16.36/16.36%`, RMSE `0.1409/0.1770/0.1710`.
- Same aligned subset E688 baseline: `12.73/18.18/25.45/25.45%`, RMSE `0.0617/0.0920/0.0733`; baseline remains better as an ordered system.
- E691 merge: repel adds `1` positive sample beyond baseline; all-candidate rows>=7 positive-any rises from `25.45%` to `27.27%`, but pool-order K20 stays `25.45%`.
- Decision: continue only as auxiliary constrained candidate generation. Do not use repel as final ranking and do not full test.

## 2026-06-15 Batch E692-E699 - train-label compatibility selectors on baseline+repel

- Trained GT-free compatibility selectors on train labels only, then applied them to E691 aligned55 baseline+repel val pool. Val labels are diagnostic/selection only; test remains closed.
- E692 dynamic GBDT from broad E318 train: AUC/AP `0.7645/0.2553`; E696 official `match@1/5/20/50=10.91/18.18/25.45/27.27%`, RMSE `0.0748/0.0884/0.1037`.
- E693 pairwise/listwise from broad E318 train: E697 official `10.91/20.00/25.45/27.27%`, RMSE `0.1088/0.0834/0.0730`.
- Broad E318 selectors recover the K50 ceiling and one improves K5, but neither improves K20 over baseline.
- E694 rows>=7 GBDT from E566b train rows>=7 labels: AUC/AP `0.8704/0.3256`; E698 official `14.55/20.00/25.45/27.27%`, RMSE `0.0829/0.0687/0.0981`.
- E695 rows>=7 pairwise/listwise: E699 official `12.73/21.82/27.27/27.27%`, RMSE `0.1748/0.1275/0.1089`; W/A@20 and skeleton@20 both `74.55/78.18%`.
- Baseline E688 on the same 55 rows>=7 samples was `12.73/18.18/25.45/25.45%`; rows7 pairwise improves K5/K20 but worsens RMSE@1/5.
- Decision: continue rows>=7-specific compatibility selector on a larger aligned val subset. No full test.

## 2026-06-15 Batch E700-E709 - larger aligned216 rows>=7 selector scale check

- E700 scaled the repel constrained-geometry auxiliary source to all E421 val512 rows>=7 candidates with output, then built an aligned 216-sample repr subset from rendered IDs.
- E701 baseline on aligned216: `match@1/5/20/50=11.57/15.28/18.52/19.44%`, RMSE `0.1370/0.1280/0.1171`; W/A@20 `70.83%`, skeleton@20 `78.70%`.
- E702 direct repel was worse: `9.26/12.04/13.43/13.43%`, RMSE `0.2062/0.2063/0.2132`; W/A@20 `68.98%`, skeleton@20 `76.39%`.
- E703/E704 labels: baseline had 133 positives from 9,428 rows; repel had 46 positives from 1,721 rows. W/A-hit match-fail stayed very high: `93.75%` baseline and `90.16%` repel.
- E705 merge added only `2` positive samples beyond baseline; all-candidate positive-any rose from `19.44%` to only `20.37%`.
- E706 rows7 GBDT had AUC/AP `0.8652/0.2965`; E708 official `11.57/14.81/18.98/20.37%`, RMSE `0.0942/0.0965/0.1322`.
- E707 rows7 pairwise had E709 official `7.87/13.89/18.98/20.37%`, RMSE `0.1927/0.1434/0.1305`.
- Decision: reject scaling. The aligned55 selector gain did not generalize; GBDT only adds `+0.46 pp` at K20 while lowering K5, and pairwise damages K1/K5.
- Full test remains closed. The current bottleneck is not selector capacity on this baseline+repel pool, but the lack of new StructureMatcher-positive geometry candidates for rows>=7.

## 2026-06-15 Batch E710-E718 - selective replacement clears val gate

- E710 added risk-controlled selective replacement on E425 GBDT scores; E711 repeated with E426 MLP. Val labels were used only to choose global `anchor_count/threshold/max_per_wa`.
- Accidental E712/E713 full-val-denominator runs omitted `--max-records 512`; those are protocol sanity only and not used for model comparison.
- Correct E712b GBDT val512: `match@1/5/20/50=33.01/41.80/45.51/45.51%`, RMSE `0.1095/0.1275/0.1309`; rows>=7 K5/K20 `17.78/18.67%`.
- Correct E713b MLP val512: `33.01/41.60/45.51/45.51%`, RMSE `0.1095/0.1222/0.1309`; rows>=7 K5/K20 `17.33/18.67%`.
- E714/E715 introduced fixed-parameter apply-only selective replacement and reproduced E712b exactly: no label grid search is needed at inference.
- E716/E717/E718 introduced the full-test-applicable GT-free scorer path from rendered top50 candidates: E718 val512 `33.01/41.99/45.90/48.24%`, RMSE `0.1095/0.1281/0.1318`.
- E718 rows>=7 K5/K20/K50 `18.22/19.11/20.44%`; atoms>=12 K5/K20/K50 `36.24/39.96/42.58%`; W/A@20 `71.09%`, skeleton@20 `75.78%`, unique W/A@20 `8.15`.
- Decision: freeze E718 for one full-test attempt after generating the corresponding full test candidate pool. It clears match@1 and match@5 targets on val512; match@20 still misses.
- No-leakage audit written to `reports/e718_freeze_no_leakage_audit/frozen_config_audit.md`. Full test has not been run.
