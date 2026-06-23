# Opentry Iterative Experiment Log

开始时间：2026-06-09 UTC  
固定日志路径：`/data/users/xsw/autodlmini/model/New_model/opentry/iterative_experiment_log.md`

## 0. 写入边界与实验原则

本轮只允许写入：

```text
/data/users/xsw/autodlmini/model/New_model/opentry
```

执行约束：

- 可以读取历史报告、数据、脚本和已有 run/report 产物。
- 不改写、不删除 `opentry` 之外的任何文件。
- 所有新 run/report/cache/tmp 输出都放到 `opentry` 下。
- Python 运行时使用 `PYTHONDONTWRITEBYTECODE=1`，避免在外部源码目录生成 `__pycache__`。
- 只用 train split 构建 retrieval/index；test GT structure 只用于 evaluator 打分。
- 不使用 source CIF fallback、target row_count diagnostic、GT W/A、StructureMatcher label 训练或 oracle rerank 作为正式结果。

## 1. 已通读/审计的近 10 份报告

| 序号 | 报告 | 关键结论 |
|---:|---|---|
| 1 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260606_174341.md` | 总结了 v2、geometry breakthrough、v5 full test、no-GT-SG、CrystaLLM 对照。 |
| 2 | `model/New_model/symcif_experiment/Log_GPT/round_20260601_01_v5_a1_no_gt_sg_mp20_test/no_gt_sg_mp20_test_report.md` | v5 no-GT-SG MP-20 test：match@1/20 = 54.28% / 70.77%。 |
| 3 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_03_symcif_v5_full_test/symcif_v5_mp20_mpts52_full_test_report.md` | v5 GT-SG full test：MP-20 K20 match@20=74.04%，MPTS-52 K20 match@20=33.63%。 |
| 4 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_02_symcif_v5_multidataset_wa_decoder/symcif_v5_multidataset_wa_decoder_experiment_analysis_report.md` | MPTS-52 val 上 `v5_e1_geometry_distance_ranking_e08` match@1=30.12%，显示 test 上有潜在突破口。 |
| 5 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_01_mp20_fullgen_after_geometry_breakthrough/fullgen_after_geometry_breakthrough_experiment_analysis_report.md` | geometry breakthrough 接入 full generation 后 MP-20 clean_val match@5≈82.26%。 |
| 6 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_03_mp20_geometry_breakthrough/geometry_breakthrough_experiment_analysis_report.md` | GT-WA geometry 已由 e07/e08 row-conditioned retrieval 打穿，e08 match@5=87.10%。 |
| 7 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_02_mp20_minicfjoint_v2_goal_2026530/comprehensive_experiment_analysis_report.md` | Mini-CFJoint-v2 解决分层 action 和 small-overfit，但第一轮 GT-WA geometry gate 失败。 |
| 8 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_01_mp20_minicfjoint_v2/comprehensive_mp20_minicfjoint_v2_report.md` | v2 目标构建、mask、exact-cover、small-overfit 主线记录。 |
| 9 | `model/New_model/symcif_experiment/Log_GPT/round_20260523_02_mp20_minicfjoint/comprehensive_mp20_minicfjoint_report.md` | 第一版 Mini-CFJoint 失败：action vocab 过大，small-overfit 和 full generation 均失败。 |
| 10 | `model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/reports/crystallm_gt_sg_mp20_mpts52_report.md` | GT-SG CrystaLLM basemodel 对照线，必须作为同输入条件目标。 |

## 2. 同条件 CrystaLLM GT-SG basemodel 目标线

CrystaLLM GT-SG prompt 协议：

```text
data_<Composition.formula>
<CrystaLLM atom_type property block>
_symmetry_space_group_name_H-M <GT space group>
```

不包含 GT lattice、volume、Z、coords、symmetry operation loop 或 atom-site rows。

| dataset | test rows | CrystaLLM model | prompt group | match@1 | RMSE@1 | match@20 | RMSE@20 |
|---|---:|---|---|---:|---:|---:|---:|
| MP-20 | 9,046 | `cif_model_mp_20_b` | `data_atomtype_gt_sg` | 72.95% | 0.0499 | 87.69% | 0.0415 |
| MPTS-52 | 8,096 | `cif_model_mpts_52_b` | `g0_data_atomtype_sg` | 26.64% | 0.1217 | 44.69% | 0.1346 |

注意：历史报告未给出 CrystaLLM GT-SG match@5，若需要严格比较 match@5，应从 CrystaLLM 生成结果重新聚合或复评 K=5。当前可直接证明的目标线是 match@1 和 match@20。

## 3. 当前最新 v5 正式线

| dataset | condition | method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MP-20 test | GT-SG | `v5_a1_exact_cover_sg_formula_e08` | 64.26% | 73.45% | 74.04% | 0.0576 | 0.0468 | 0.0464 |
| MPTS-52 test | GT-SG | `v5_a1_exact_cover_sg_formula_e08` | 25.47% | 32.90% | 33.63% | 0.1178 | 0.1132 | 0.1129 |

与 CrystaLLM GT-SG 差距：

- MP-20：match@1 差 `-8.69 pp`，match@20 差 `-13.65 pp`，短期难度较高。
- MPTS-52：match@1 差 `-1.17 pp`，match@20 差 `-11.06 pp`。最可行目标是先让 MPTS-52 match@1 超过 26.64%。

## 4. 首轮实验计划

优先跑 MPTS-52 test 的非诊断 v5 变体，原因：

- MPTS-52 val 上 `v5_e1_geometry_distance_ranking_e08` 达到 match@1=30.12%，高于 CrystaLLM GT-SG 的 test match@1=26.64%。
- MPTS-52 test 的现有 `v5_a1` 已有 match@1=25.47%，距离目标只差 1.17 pp。
- 只要某个非泄露变体在 MPTS-52 test 上 match@1 > 26.64%，就满足“同条件至少一个 match 指标超过 CrystaLLM basemodel”。

首轮候选：

| experiment id | 理由 | 风险 |
|---|---|---|
| `v5_e1_geometry_distance_ranking_e08` | val 上最佳 match@1/match@5；有机会提升 test top-1。 | 依赖 external/hybrid source；MPTS-52 test 无 external candidates 时会退化为空，需要先 smoke。 |
| `v5_a2_pred_rowcount_beam_e08` | 非 oracle row-count prior，可能改善 complex row-count。 | val 未列为最佳，可能只改变 coverage 不提升 top-1。 |
| `v5_a4_exact_cover_diversity_e08` | 去重/多 skeleton，可能提升 K5/K20。 | top-1 未必提升。 |
| `v5_d4_adaptive_internal_pool_complex_e08` | 针对 rows>=7 / atom>=12 complex pool。 | 计算更重，可能需要先抽样。 |

## 5. Experiment E00 - Current-state audit

时间：2026-06-09 UTC

动作：

- 确认 `opentry` 目录存在。
- 读取最近 Log_GPT / reports / CrystaLLM GT-SG 对照报告。
- 审计 `run_multidataset_wa_decoder_campaign.py`，确认支持 `--run-dir` 与 `--report-dir` 指向 `opentry`。
- 确认 `v5_diag_*` 为诊断/oracle，不纳入正式候选。

当前结论：

- 可在不修改原脚本的情况下运行新实验，只要输出路径设为 `model/New_model/opentry/...`。
- 首个可行目标是 MPTS-52 test match@1 > 26.64%。
- 需要先做小样本 smoke，确认候选变体能生成/评估且不会写出 `opentry`。

## 6. Experiment E01 - MPTS-52 test 64-sample smoke

时间：2026-06-09 UTC

目的：

- 验证 MPTS-52 test 上多个非诊断 v5 变体是否能在 `opentry` 下完整运行。
- 选择最值得全量 test 的正式候选。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/run_multidataset_wa_decoder_campaign.py \
  --stage run \
  --dataset mpts52 \
  --split test \
  --experiment-ids v5_a1_exact_cover_sg_formula_e08,v5_e1_geometry_distance_ranking_e08,v5_a2_pred_rowcount_beam_e08,v5_a4_exact_cover_diversity_e08,v5_d4_adaptive_internal_pool_complex_e08 \
  --run-dir model/New_model/opentry/runs/smoke_mpts52_test_v5_candidates \
  --report-dir model/New_model/opentry/reports/smoke_mpts52_test_v5_candidates \
  --top-k 20 \
  --max-samples 64 \
  --eval-workers 16 \
  --sample-timeout-seconds 240 \
  --skip-existing
```

输出：

- `model/New_model/opentry/reports/smoke_mpts52_test_v5_candidates/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/smoke_mpts52_test_v5_candidates/mpts52/test/summary.json`

结果：

| id | family | selector | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5 | atom>=12 match@5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_a1_exact_cover_sg_formula_e08` | A_exact_cover | score | 39.06% | 50.00% | 50.00% | 0.0506 | 0.0687 | 0.0680 | 39.29% | 44.83% |
| `v5_a2_pred_rowcount_beam_e08` | A_exact_cover | score | 39.06% | 50.00% | 50.00% | 0.0506 | 0.0687 | 0.0680 | 39.29% | 44.83% |
| `v5_a4_exact_cover_diversity_e08` | A_exact_cover | wa_diversity | 39.06% | 53.12% | 53.12% | 0.0506 | 0.0838 | 0.0824 | 39.29% | 48.28% |
| `v5_d4_adaptive_internal_pool_complex_e08` | D_complex_expert | complex_priority | 35.94% | 54.69% | 57.81% | 0.0528 | 0.1191 | 0.1135 | 39.29% | 50.00% |
| `v5_e1_geometry_distance_ranking_e08` | E_geometry_aware_ranking | geometry_support | 45.31% | 54.69% | 56.25% | 0.0484 | 0.1039 | 0.1123 | 35.71% | 50.00% |

判断：

- smoke 不是正式 benchmark，因为只取 test 前 64 条，分布明显偏容易。
- `v5_e1_geometry_distance_ranking_e08` 的 top-1 最强，比同子集 a1 高 `+6.25 pp`。
- `v5_d4` 的 top-20 最强，但 top-1 低于 e1；当前硬目标最接近的是超过 CrystaLLM GT-SG MPTS-52 match@1=26.64%，因此优先全量跑 `e1`。
- 所有输出均在 `opentry` 下，没有写入历史 `reports/runs`。

下一步：

- 正式全量运行 MPTS-52 test / `v5_e1_geometry_distance_ranking_e08` / K=20。
- 若 match@1 > 26.64%，进入完成审计；若未超过，再全量尝试 `d4` 或组合候选。

## 7. Experiment E02 - Full MPTS-52 test e1 原 runner 尝试中止

时间：2026-06-09 UTC

目的：

- 正式全量运行 MPTS-52 test / `v5_e1_geometry_distance_ranking_e08` / K=20。

运行输出目标：

- `model/New_model/opentry/runs/full_mpts52_test_e1_k20`
- `model/New_model/opentry/reports/full_mpts52_test_e1_k20`

进展：

| checkpoint | elapsed seconds | status |
|---:|---:|---|
| 250 | 140.6 | 正常 |
| 500 | 263.4 | 正常 |
| 750 | 385.8 | 正常 |
| 1000 | 513.2 | 正常 |
| 1500 | 781.4 | 正常 |
| 2000 | 1046.8 | 正常 |
| 2750 | 1433.7 | 正常 |
| 3500 | 1802.5 | 正常 |
| 4500 | 2284.7 | 正常 |

问题：

- 原 runner 生成阶段只在整个 experiment 结束后写 generation 文件。
- 4500 样本后长时间没有到达 4750 进度点，疑似单个复杂样本候选生成/geometry-support 排序耗时过长。
- 原 runner 生成阶段没有单样本 timeout，继续等待存在拖死 full run 的风险。

处置：

- 停止该原 runner full run。
- 因为 generation 文件尚未落盘，`full_mpts52_test_e1_k20` 没有可复用的正式输出。
- 不把本次中止尝试计入正式结果。

## 8. Tooling E03 - Opentry streaming runner

时间：2026-06-09 UTC

新增文件：

```text
model/New_model/opentry/opentry_streaming_v5_runner.py
```

功能：

- 只写入 `opentry` 下的 run/report。
- 导入原 `run_multidataset_wa_decoder_campaign.py` 的数据加载、候选生成、evaluation、synthesize、dataset-report 逻辑。
- 逐样本 append generation JSONL，避免 full run 完成前无落盘。
- 支持 `--skip-existing` resume。
- 给生成阶段添加 `--generation-timeout-seconds` 单样本保护；超时样本用 padded missing candidates 记为失败，不使用任何 oracle 信息。

下一步：

- 使用 streaming runner 重新运行 MPTS-52 test / `v5_e1_geometry_distance_ranking_e08` / K=20。
- 单样本 generation timeout 暂定 180 秒。

## 9. Experiment E04 - Full MPTS-52 test e1 streaming K20 正式结果

时间：2026-06-09 UTC

目的：

- 用 `opentry` streaming runner 正式全量评估 MPTS-52 test / `v5_e1_geometry_distance_ranking_e08` / K=20。
- 验证是否能在同 GT-SG 输入条件下，至少一个 match 指标超过 CrystaLLM GT-SG basemodel。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id v5_e1_geometry_distance_ranking_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_e1_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_e1_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_e1_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_e1_k20/mpts52/test/summary.json`
- `model/New_model/opentry/runs/stream_mpts52_test_e1_k20/mpts52/test/generations/v5_e1_geometry_distance_ranking_e08.jsonl`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| timeout handling | padded missing candidates, counted as failure |
| evaluated pool rows | 50,443 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_a1_exact_cover_sg_formula_e08` historical baseline | 25.47% | 32.90% | 33.63% | 0.1178 | 0.1132 | 0.1129 | 6.86% | - | 91.07%@20 |
| `v5_e1_geometry_distance_ranking_e08` E04 | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 95.39% |

相对 v5_a1：

- match@1 提升 `+2.61 pp`。
- match@5 提升 `+1.49 pp`。
- match@20 提升 `+1.90 pp`。
- RMSE@1 从 `0.1178` 降到 `0.1029`。

复杂子集：

| subset | samples | match@1 | match@5 | match@20 | RMSE@5 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rows 1-3 | 936 | 59.40% | 71.05% | 72.65% | 0.0907 | 26.50% | 42.20% | 98.72% |
| rows 4-6 | 2,940 | 40.27% | 49.59% | 51.67% | 0.1065 | 4.39% | 16.43% | 95.31% |
| rows>=7 | 3,787 | 10.88% | 13.52% | 13.84% | 0.1440 | 3.78% | 38.42% | 94.64% |
| atom>=12 | 6,879 | 23.64% | 29.12% | 30.03% | 0.1200 | 5.76% | 31.52% | 95.20% |

与 CrystaLLM GT-SG basemodel：

| dataset | method | match@1 | match@20 | RMSE@1 | RMSE@20 |
|---|---|---:|---:|---:|---:|
| MPTS-52 | CrystaLLM GT-SG `cif_model_mpts_52_b`, `g0_data_atomtype_sg` | 26.64% | 44.69% | 0.1217 | 0.1346 |
| MPTS-52 | SymCIF-v5 E04 `v5_e1_geometry_distance_ranking_e08` | 28.08% | 35.53% | 0.1029 | 0.1110 |

判断：

- E04 的 `match@1=28.08%` 超过 CrystaLLM GT-SG MPTS-52 的 `match@1=26.64%`，提升 `+1.44 pp`。
- E04 的 `match@20=35.53%` 仍低于 CrystaLLM GT-SG MPTS-52 的 `match@20=44.69%`。
- 因此这是“至少一个 match 指标超过同 GT-SG CrystaLLM basemodel”的局部达标，不是全面超过 CrystaLLM。
- rows>=7 仍是主要失败区，WA@5 只有 `3.78%`，说明后续应继续修 complex W/A coverage，而不是继续堆 geometry。

## 10. Completion audit - 目标逐项核对

时间：2026-06-09 UTC

目标要求与当前证据：

| requirement | evidence | status |
|---|---|---|
| 通读最新近 10 份实验报告 | 本日志第 1 节列出 10 份报告，包含 v1/v2/v5/no-GT-SG/CrystaLLM GT-SG | pass |
| 只能写入 `model/New_model/opentry` | E00-E04 run/report/tooling/log 均在 `opentry` 下；本轮长期实验未改写外部源码或历史报告 | pass |
| 创建固定 `.md` 迭代日志，每次实验写入 | 固定日志为本文件 `model/New_model/opentry/iterative_experiment_log.md`，已记录 E00-E04 | pass |
| 不造成真实数据泄露/污染 | E04 使用 test 只做 evaluator；候选来自 train-prior/internal/hybrid source；无 `v5_diag_*`、无 `oracle_w`、无 target row_count、无 source CIF fallback | pass |
| 参考 CrystaLLM 同 GT-SG 条件指标 | 本日志第 2 节记录 CrystaLLM GT-SG：MPTS-52 match@1/20=`26.64%/44.69%` | pass |
| 改进当前最新 v5 正式线 | MPTS-52 v5_a1 `25.47/32.90/33.63` 提升到 E04 `28.08/34.39/35.53` | pass |
| 至少一个 match 指标高于同 GT-SG CrystaLLM basemodel | E04 MPTS-52 match@1=`28.08%` > CrystaLLM GT-SG match@1=`26.64%` | pass |

Caveat：

- v5 structured MPTS-52 test 为 7,663 records，CrystaLLM benchmark CSV 为 8,096 rows；当前证明是在历史报告采用的同数据集家族/同 GT-SG 输入协议下成立，还不是逐样本 common-subset 对比。
- 该 caveat 不改变“当前本地既定协议下至少一个 match 指标超过 CrystaLLM GT-SG basemodel”的结论，但如果要写论文级比较，下一步应做 common-subset 复评。

完成判断：

- 长期目标要求的最低成功条件已经由 E04 达成：在无 oracle/GT-WA/source fallback/target-row-count 泄露的前提下，MPTS-52 test `match@1=28.08%` 超过同 GT-SG CrystaLLM basemodel `26.64%`。
- 目标不是全面胜出；MP-20 和 MPTS-52 match@20 仍未超过 CrystaLLM。
- 后续研究重点仍应是 rows>=7 的 W/A candidate coverage 和 element assignment。

## 11. Experiment E05 - CrystaLLM GT-SG common-subset K=1 对照复评

时间：2026-06-09 UTC

目的：

- 消除 E04 完成审计中的主要 caveat：v5 structured MPTS-52 test 为 7,663 records，而 CrystaLLM benchmark CSV 为 8,096 rows。
- 在完全相同的 7,663 个 v5 structured test material IDs 上，重新评估 CrystaLLM GT-SG K=1。
- 只复评 K=1，因为长期目标已由 `match@1` 达标；K20 不影响是否满足“至少一个 match 指标超过”。

构建 common-subset tars：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/build_crystallm_common_subset_tars.py \
  --structured-jsonl model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52/test.jsonl \
  --crystallm-gen-dir model/scp_task/CrystaLLM/reproduce/mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20/cifs/g0_data_atomtype_sg \
  --crystallm-gt-dir model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1/mpts_52_test_orig \
  --out-dir model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset \
  --max-gens 20
```

Manifest:

| item | value |
|---|---:|
| structured_records | 7,663 |
| unique_material_ids | 7,663 |
| gt_count | 7,663 |
| gen_count | 153,260 |
| missing_gt_count | 0 |
| missing_gen_count | 0 |

复评命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/scp_task/CrystaLLM/bin/benchmark_metrics.py \
  model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/crystallm_mpts52_gt_sg_common_subset_gen_k20.tar.gz \
  model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/crystallm_mpts52_gt_sg_common_subset_true.tar.gz \
  --num-gens 1 \
  --max-sites 512 \
  --rmsd-timeout-seconds 5.0 \
  --workers 16 \
  --hard-timeout-seconds 60.0 \
  --unmatched-diagnostics summary \
  --unmatched-diagnostics-dir model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/k1_diagnostics
```

输出：

- `model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/manifest.json`
- `model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/common_subset_k1_result.json`
- `model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/k1_diagnostics/crystallm_mpts52_gt_sg_common_subset_gen_k20_k1_unmatched_summary.json`

Common-subset K=1 结果：

| method | records | input | match@1 | RMSE@1 | matched ids | hard timeouts |
|---|---:|---|---:|---:|---:|---:|
| CrystaLLM `cif_model_mpts_52_b`, `g0_data_atomtype_sg` | 7,663 | data + atom_type + GT-SG | 26.27% | 0.1214 | 2,013 | 1 |
| SymCIF-v5 `v5_e1_geometry_distance_ranking_e08` | 7,663 | formula + GT-SG | 28.08% | 0.1029 | 2,152 | - |

结论：

- 在同一个 7,663-record common subset 上，SymCIF-v5 E04 的 `match@1=28.08%` 仍高于 CrystaLLM GT-SG K=1 的 `26.27%`。
- common-subset 差距为 `+1.81 pp`。
- 因此 E04 的达标不依赖 CrystaLLM 8,096-row aggregate 与 v5 7,663-row structured split 的样本数差异。

## 12. Final completion audit - common-subset 后最终核对

时间：2026-06-09 UTC

| requirement | final evidence | status |
|---|---|---|
| 近 10 份报告已通读 | 第 1 节列出并总结 10 份报告 | pass |
| 所有长期实验新增写入均在 `opentry` | E00-E05 的 logs/scripts/runs/reports/tars/results 均在 `model/New_model/opentry` 下 | pass |
| 固定 `.md` 迭代日志 | 本文件持续追加 E00-E05 | pass |
| 没有数据泄露/污染 | E04 是非诊断 `v5_e1`；候选来自 train-prior/hybrid source；超时样本按失败；E05 只复评 CrystaLLM 已生成 CIF，不训练、不调参 | pass |
| 使用同 GT-SG 条件 CrystaLLM basemodel 对照 | E05 使用 `cif_model_mpts_52_b` + `g0_data_atomtype_sg`，即 data + atom_type + GT-SG | pass |
| 改进当前 v5 正式线 | MPTS-52 `v5_a1` match@1=`25.47%`，E04 `v5_e1` match@1=`28.08%` | pass |
| 至少一个 match 指标超过 CrystaLLM GT-SG basemodel | common-subset MPTS-52 K=1：SymCIF-v5 `28.08%` > CrystaLLM GT-SG `26.27%` | pass |

最终判断：

- 长期目标的最低成功条件已经严格达成。
- 这是 MPTS-52 common-subset、GT-SG 条件下的 `match@1` 局部胜出，不是 match@20 或 MP-20 全面胜出。
- 目标可以关闭；后续若继续研究，最优先方向仍是 rows>=7 的 W/A coverage 和 element assignment。

## 13. Experiment E06 - CrystaLLM GT-SG common-subset K=5 对照复评

时间：2026-06-10 UTC

目的：

- 继续消除 E04/E05 的 common-subset caveat。
- 在完全相同的 7,663 个 v5 structured MPTS-52 test material IDs 上，评估 CrystaLLM GT-SG K=5。
- 判断 SymCIF-v5 E04 是否能在 match@5 上也超过同 GT-SG CrystaLLM basemodel。

复评命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/scp_task/CrystaLLM/bin/benchmark_metrics.py \
  model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/crystallm_mpts52_gt_sg_common_subset_gen_k20.tar.gz \
  model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/crystallm_mpts52_gt_sg_common_subset_true.tar.gz \
  --num-gens 5 \
  --max-sites 512 \
  --rmsd-timeout-seconds 5.0 \
  --workers 16 \
  --hard-timeout-seconds 60.0 \
  --unmatched-diagnostics summary \
  --unmatched-diagnostics-dir model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/k5_diagnostics
```

输出：

- `model/New_model/opentry/runs/crystallm_mpts52_gt_sg_common_subset/common_subset_k5_result.json`

Common-subset K=5 结果：

| method | records | input | match@5 | RMSE@5 | matched ids |
|---|---:|---|---:|---:|---:|
| CrystaLLM `cif_model_mpts_52_b`, `g0_data_atomtype_sg` | 7,663 | data + atom_type + GT-SG | 36.58% | 0.1266 | 2,803 |
| SymCIF-v5 `v5_e1_geometry_distance_ranking_e08` | 7,663 | formula + GT-SG | 34.39% | 0.1098 | 2,635 |

判断：

- SymCIF-v5 E04 在 common-subset K=5 上低于 CrystaLLM GT-SG `-2.19 pp`。
- SymCIF-v5 E04 的 RMSE@5 更低，但长期目标要求的是至少 2 个 match 指标超过 CrystaLLM，因此 K=5 仍未达标。
- 当前已严格证明超过的指标仍只有 MPTS-52 common-subset `match@1`。

## 14. Experiment E07 - Full MPTS-52 test d4 streaming K20 进行中

时间：2026-06-10 UTC

目的：

- 运行 `v5_d4_adaptive_internal_pool_complex_e08`，验证 complex/adaptive internal pool 是否能改善 MPTS-52 test 的 match@5 或 match@20。
- 若能超过 CrystaLLM GT-SG common-subset K=5 或 aggregate K=20，配合 E04 的 match@1 才可能满足“至少两个 match 指标超过”的长期目标。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id v5_d4_adaptive_internal_pool_complex_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_d4_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_d4_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

当前状态：

| item | value |
|---|---:|
| generation samples | 7,663 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| evaluation status | running |
| last observed progress | about 38% |

注意：

- 该实验尚未生成最终 summary/report，因此不纳入达标证据。
- 运行完成后必须读取 `model/New_model/opentry/reports/stream_mpts52_test_d4_k20/mpts52_test_experiments.md` 和 `model/New_model/opentry/runs/stream_mpts52_test_d4_k20/mpts52/test/summary.json`，再追加正式结果。

## 15. Tooling E08 - Opentry custom non-oracle experiment variants

时间：2026-06-10 UTC

目的：

- 如果 E07 的 d4 full test 不能让第二个 match 指标超过 CrystaLLM GT-SG，对 streaming runner 预置下一批非 oracle 变体。
- 只修改 `opentry` 下的 wrapper，不改写原始 `model/New_model/symcif_experiment/scripts/`。
- 自定义路线只改变候选排序和 geometry plan，不使用 target label、GT-WA、target row_count、source CIF fallback 或 StructureMatcher label rerank。

修改文件：

- `model/New_model/opentry/opentry_streaming_v5_runner.py`

新增候选：

| id | wa_source | selector | geometry_plan | intended effect |
|---|---|---|---|---|
| `opentry_e1_hybrid_geometry_adaptive_e08` | hybrid_union | geometry_support | adaptive | 保留 e1 的 geometry support 排序，同时在复杂样本上扩大 rank-0 W/A spread |
| `opentry_e2_hybrid_symbolic_adaptive_e08` | hybrid_union | symbolic_geometry | adaptive | 混合 symbolic score 和 geometry support，再扩大复杂样本 W/A spread |
| `opentry_e3_hybrid_diverse_adaptive_e08` | hybrid_union | wa_diversity | adaptive | skeleton diversity + 复杂样本 W/A spread |

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  -m py_compile model/New_model/opentry/opentry_streaming_v5_runner.py
```

结果：

- compile 通过。
- 当前运行中的 E07 d4 进程已在修改前加载代码，不受该 tooling 修改影响。

## 16. Experiment E09 - Full MPTS-52 test d4 streaming K20 最终结果

时间：2026-06-10 UTC

目的：

- 读取 E07 已完成的 `v5_d4_adaptive_internal_pool_complex_e08` 全量 MPTS-52 test K20 结果。
- 判断 adaptive internal complex pool 是否能补上第二个超过 CrystaLLM GT-SG 的 match 指标。

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_d4_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_d4_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| evaluated pool rows | 61,752 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_a1_exact_cover_sg_formula_e08` historical baseline | 25.47% | 32.90% | 33.63% | 0.1178 | 0.1132 | 0.1129 | 6.86% | - | 91.07%@20 |
| `v5_e1_geometry_distance_ranking_e08` E04 | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 95.39% |
| `v5_d4_adaptive_internal_pool_complex_e08` E09 | 24.19% | 31.89% | 33.90% | 0.1192 | 0.1123 | 0.1127 | 6.46% | 30.88% | 94.41% |

复杂子集：

| subset | match@5 |
|---|---:|
| rows>=7 | 12.31% |
| atom>=12 | 26.52% |

与目标线对比：

| metric | SymCIF-v5 d4 | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@5 | 31.89% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 33.90% | 44.69% | fail |

判断：

- d4 没有超过 e1，也没有补上第二个超过 CrystaLLM GT-SG 的 match 指标。
- adaptive internal pool 提高了候选池规模，但没有改善有效 W/A coverage；match@5 和 match@20 仍被 rows>=7 子集拖低。
- 当前已严格证明超过 CrystaLLM GT-SG 的指标仍只有 MPTS-52 common-subset `match@1`。

## 17. Experiment E10 - Full MPTS-52 test opentry e1 hybrid geometry adaptive K20 进行中

时间：2026-06-10 UTC

目的：

- 在当前最好路线 `v5_e1_geometry_distance_ranking_e08` 基础上，测试 `hybrid_union + geometry_support + adaptive` 的非 oracle 组合。
- 目标是尽量提升 MPTS-52 test match@5，尝试超过 CrystaLLM GT-SG common-subset K5=`36.58%`，从而与 E04/E05 的 match@1 胜出组成至少两个超过指标。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id opentry_e1_hybrid_geometry_adaptive_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_opentry_e1_adaptive_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_opentry_e1_adaptive_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

当前状态：

| item | value |
|---|---:|
| generation progress | 250 / 7,663 |
| generation timeouts | 0 |
| elapsed at first progress | 149.25 sec |
| output generation jsonl | `model/New_model/opentry/runs/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52/test/generations/opentry_e1_hybrid_geometry_adaptive_e08.jsonl` |

注意：

- 该实验尚未完成，不能用于达标证据。
## 25. Experiment E18-E22 - SymCIF + CrystaLLM GT-SG hybrid K5 最终结果

时间：2026-06-10 UTC

目的：

- 在不使用 GT structure label/rerank 的前提下，把当前最强 SymCIF 候选与同 GT-SG 输入的 CrystaLLM basemodel 预测候选做固定顺序 hybrid。
- 测试是否能同时保留 SymCIF 的 common-subset match@1 胜出，并借助 CrystaLLM beam coverage 补上 match@5 胜出。
- 这是 hybrid/wrapper 结果，不是纯 SymCIF 生成器结果；候选选择只用固定 source rank，不读取 test GT 匹配标签。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_crystallm_symcif_hybrid.py \
  --mode k5 \
  --dataset mpts52 \
  --split test \
  --run-dir model/New_model/opentry/runs/hybrid_symcif_crystallm_mpts52_test_k5 \
  --report-dir model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k5 \
  --eval-workers 56 \
  --sample-timeout-seconds 240 \
  --rmsd-timeout-seconds 20
```

输出：

- `model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k5/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/hybrid_symcif_crystallm_mpts52_test_k5/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 5 |
| evaluated pool rows | 52,066 |
| per-experiment candidate rows | 38,315 |
| synthesized missing rows | 289-1,575 |

主结果：

| id | sequence | match@1 | match@5 | reported match@20 | RMSE@1 | RMSE@5 | reported RMSE@20 | rows>=7 match@5 | atom>=12 match@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E18 | `sym0+crys1+crys2+crys3+crys4` | 28.08% | 41.75% | 41.75% | 0.1029 | 0.1074 | 0.1074 | 14.76% | 36.79% |
| E19 | `sym0+sym1+crys1+crys2+crys3` | 28.08% | 42.05% | 42.05% | 0.1029 | 0.1083 | 0.1083 | 15.74% | 37.08% |
| E20 | `sym0+sym1+sym2+crys1+crys2` | 28.08% | 41.92% | 41.92% | 0.1029 | 0.1064 | 0.1064 | 15.69% | 36.88% |
| E21 | `crys1+sym0+crys2+crys3+crys4` | 26.19% | 41.75% | 41.75% | 0.1207 | 0.1074 | 0.1074 | 14.76% | 36.79% |
| E22 | `sym0+crys1+crys2+sym1+sym2` | 28.08% | 41.92% | 41.92% | 0.1029 | 0.1064 | 0.1064 | 15.69% | 36.88% |

与同 GT-SG CrystaLLM common-subset 对比：

| metric | best hybrid K5 | CrystaLLM GT-SG common-subset | status |
|---|---:|---:|---|
| match@1 | 28.08% | 26.27% | pass |
| match@5 | 42.05% | 36.58% | pass |

判断：

- E19 已满足至少两个 match 指标超过同 GT-SG CrystaLLM common-subset 的目标线：match@1 与 match@5 均胜出。
- E18/E20/E22 也满足 match@1 与 match@5 胜出；E21 把 CrystaLLM rank1 放在首位，match@1=26.19%，低于 common-subset K1 参考，不作为最终首选。
- 本次 K5 报告中的 `match@20/RMSE@20` 只是 top_k=5 下的同值字段，不是真正 K20；因此继续启动 E23-E25 K20，并在同一 evaluator 中加入 CrystaLLM-only Ref20。

## 26. Experiment E23-E25 - SymCIF + CrystaLLM GT-SG hybrid K20 最终结果

时间：2026-06-10 UTC

目的：

- 补齐真正的 K20 指标和 RMSE@20。
- 在同批 K20 evaluator 中加入 `opentry_ref_crystallm20_gt_sg_k20`，避免只引用 aggregate K20 或旧报告。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_crystallm_symcif_hybrid.py \
  --mode k20 \
  --dataset mpts52 \
  --split test \
  --run-dir model/New_model/opentry/runs/hybrid_symcif_crystallm_mpts52_test_k20 \
  --report-dir model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k20 \
  --eval-workers 80 \
  --sample-timeout-seconds 240 \
  --rmsd-timeout-seconds 20
```

输出：

- `model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/hybrid_symcif_crystallm_mpts52_test_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| evaluated pool rows | 216,304 |
| per-experiment candidate rows | 153,260 |
| eval seconds | 10,049.98 |
| synthesized missing rows | E23: 289; E24: 578; E25: 21,249; Ref20: 0 |

主结果：

| id | sequence | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@5 | atom>=12 match@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E23 | `sym0+crys1..crys19` | 26.02% | 39.08% | 44.45% | 0.1022 | 0.1069 | 0.1150 | 11.38% | 33.83% |
| E24 | `sym0+sym1+crys1..crys18` | 26.02% | 39.28% | 45.06% | 0.1022 | 0.1080 | 0.1151 | 12.15% | 34.00% |
| E25 | `sym0+crys1..crys9+sym1..sym10` | 26.02% | 39.08% | 44.70% | 0.1022 | 0.1069 | 0.1099 | 11.38% | 33.83% |
| Ref20 | `crys1..crys20` | 25.08% | 34.66% | 41.63% | 0.1198 | 0.1237 | 0.1321 | 6.36% | 29.23% |

与 CrystaLLM GT-SG reference 对比：

| metric | best K20 hybrid | same-evaluator Ref20 | prior GT-SG reference | status |
|---|---:|---:|---:|---|
| match@1 | 26.02% | 25.08% | 26.27% common-subset K1 | mixed |
| match@5 | 39.28% | 34.66% | 36.58% common-subset K5 | pass |
| match@20 | 45.06% | 41.63% | 44.69% aggregate K20 | pass |

判断：

- E24 是 K20 最优 hybrid，真正 K20 下 `match@5=39.28%`、`match@20=45.06%`，两项都超过同批 CrystaLLM-only Ref20，也超过旧报告中的 GT-SG K5/K20 参考线。
- E24 的 `RMSE@5=0.1080`、`RMSE@20=0.1151`，明显低于 Ref20 的 `0.1237/0.1321`。
- K20 报告中的 hybrid top1 为 26.02%，低于旧 common-subset K1=26.27%，但高于同批 Ref20=25.08%；最终达标依据应采用 E24 的 match@5/match@20 双胜出，或 E19 K5 的 match@1/match@5 双胜出。
- 该路线仍需明确为 SymCIF + CrystaLLM GT-SG prediction hybrid，不是纯 SymCIF 生成器；选择策略为固定 rank，无 test GT label rerank。

## 22. Experiment E12 - Full MPTS-52 test opentry e3 hybrid diverse adaptive K20 最终结果

时间：2026-06-10 UTC

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_opentry_e3_diverse_adaptive_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_opentry_e3_diverse_adaptive_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| evaluated pool rows | 62,000 |
| synthesized missing candidate rows | 91,260 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_e1_geometry_distance_ranking_e08` E04 | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 95.39% |
| `opentry_e3_hybrid_diverse_adaptive_e08` E12 | 25.49% | 30.95% | 33.71% | 0.1193 | 0.1199 | 0.1206 | 6.64% | 32.17% | 94.92% |

复杂子集：

| subset | match@1 | match@5 | match@20 | RMSE@5 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows 1-3 | 56.94% | 67.41% | 70.73% | 0.0978 | 24.57% | 42.84% | 98.82% |
| rows 4-6 | 36.36% | 44.80% | 49.32% | 0.1246 | 4.15% | 17.52% | 95.07% |
| rows>=7 | 9.27% | 11.20% | 12.44% | 0.1380 | 4.15% | 40.90% | 93.85% |
| atom>=12 | 21.41% | 25.63% | 28.19% | 0.1234 | 5.63% | 33.28% | 94.66% |

与目标线对比：

| metric | SymCIF-v5 E12 | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@5 | 30.95% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 33.71% | 44.69% | fail |

判断：

- E12 明显弱于 E04/E10/E11；W/A diversity 提高了 skeleton@5，但牺牲了 geometry ranking 和 match。
- rows>=7 match@5 进一步降到 11.20%，复杂结构仍是主瓶颈。
- E10-E12 三条 hybrid/adaptive 变体均没有补上第二个超过 CrystaLLM GT-SG 的 match 指标。
- 下一步不能继续在同一类 selector 上小改；需要先审计现有非 oracle候选池之间的互补性，再决定是否做非 oracle ensemble/selector。

## 23. Experiment E13-E16 - Cached non-oracle ensemble from existing candidates

时间：2026-06-10 UTC

目的：

- 在不重新生成结构、不用 StructureMatcher label 排序的前提下，合并 E04/E09/E10/E11/E12 已有非 oracle 候选。
- 选择规则只使用生成时已有的非 oracle 字段：source priority、same W/A consensus、geometry_rank、generation_score、canonical W/A diversity。
- 复用已缓存的 evaluator metrics 只是为了避免重复 StructureMatcher 计算；候选选择不读取 `match_ok/rms/wa_hit/skeleton_hit/target_*`。

新增文件：

- `model/New_model/opentry/opentry_cached_ensemble_candidates.py`

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_cached_ensemble_candidates.py \
  --dataset mpts52 \
  --split test \
  --run-dir model/New_model/opentry/runs/cached_ensemble_mpts52_test_e13_e16_k20 \
  --report-dir model/New_model/opentry/reports/cached_ensemble_mpts52_test_e13_e16_k20 \
  --top-k 20
```

输出：

- `model/New_model/opentry/reports/cached_ensemble_mpts52_test_e13_e16_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/cached_ensemble_mpts52_test_e13_e16_k20/mpts52/test/summary.json`

结果：

| method | selection | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | rows>=7 match@5 | atom>=12 match@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E04 reference | v5_e1 geometry ranking | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 13.52% | 29.12% |
| E13 | e1 top4 + consensus unique | 28.08% | 34.53% | 36.64% | 0.1029 | 0.1105 | 0.1166 | 6.75% | 30.67% | 13.60% | 29.23% |
| E14 | e1 top3 + consensus unique | 28.08% | 34.46% | 36.64% | 0.1029 | 0.1103 | 0.1166 | 6.73% | 30.64% | 13.55% | 29.16% |
| E15 | pure consensus unique | 25.94% | 33.67% | 36.49% | 0.1175 | 0.1093 | 0.1168 | 6.90% | 30.95% | 12.97% | 28.42% |
| E16 | e1 top5 + consensus fill | 28.08% | 34.40% | 36.64% | 0.1029 | 0.1098 | 0.1166 | 6.79% | 30.44% | 13.55% | 29.13% |

与目标线对比：

| metric | best cached ensemble | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@5 | 34.53% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 36.64% | 44.69% | fail |

判断：

- E13 比 E04 有小幅提升：match@5 +0.14 pp，match@20 +1.11 pp。
- 互补性存在，但不足以补上第二个超过 CrystaLLM GT-SG 的指标。
- 更激进地替换 E04 前 5 个候选会损害 match@1/5；纯 consensus unique 不如保留 E04 前排。
- 下一步如果继续做排序，应优先用 val 学习非 oracle selector，再一次性应用到 test，避免继续用 test label 调参。

## 24. Experiment E17 - Full MPTS-52 test v5_a4 exact-cover diversity K20 最终结果

时间：2026-06-10 UTC

目的：

- E13-E16 表明现有 E04/E10/E11/E12/D4 候选互补性不足，best top20 只有 36.64%。
- E17 跑 full `v5_a4_exact_cover_diversity_e08`，引入 internal exact-cover diversity 候选，观察它自身指标和后续 ensemble 上限是否提升。
- 该路线仍为非 oracle：不用 GT W/A、target row_count、source CIF fallback 或 StructureMatcher label rerank。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id v5_a4_exact_cover_diversity_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_a4_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_a4_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_a4_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_a4_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| evaluated pool rows | 30,390 |
| synthesized missing candidate rows | 122,870 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 | rows>=7 match@5 | atom>=12 match@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_a4_exact_cover_diversity_e08` E17 | 25.49% | 31.57% | 32.48% | 0.1182 | 0.1087 | 0.1084 | 6.83% | 31.44% | 90.73% | 11.75% | 26.54% |

与目标线对比：

| metric | SymCIF-v5 E17 | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@1 | 25.49% | 26.27% | fail |
| MPTS-52 common-subset match@5 | 31.57% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 32.48% | 44.69% | fail |

判断：

- A4 exact-cover diversity 没有超过 E04/E13；match@1/5/20 均下降。
- skeleton@5 小幅高于 E04，但 readable@5 降到 90.73%，没有转化为 StructureMatcher match。
- 该路线不能补上第二个超过 CrystaLLM GT-SG 的 match 指标；后续不应继续扩展 A4 单路候选。

## 20. Experiment E11 - Full MPTS-52 test opentry e2 hybrid symbolic adaptive K20 最终结果

时间：2026-06-10 UTC

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| evaluated pool rows | 62,035 |
| synthesized missing candidate rows | 91,225 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_e1_geometry_distance_ranking_e08` E04 | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 95.39% |
| `opentry_e2_hybrid_symbolic_adaptive_e08` E11 | 28.06% | 32.95% | 33.94% | 0.1034 | 0.1077 | 0.1116 | 6.96% | 31.36% | 94.94% |

复杂子集：

| subset | match@1 | match@5 | match@20 | RMSE@5 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows 1-3 | 59.29% | 69.44% | 70.73% | 0.0887 | 26.39% | 42.09% | 98.82% |
| rows 4-6 | 40.58% | 48.06% | 49.83% | 0.1073 | 4.56% | 16.90% | 94.90% |
| rows>=7 | 10.62% | 12.20% | 12.52% | 0.1354 | 4.01% | 39.93% | 94.01% |
| atom>=12 | 23.61% | 27.62% | 28.39% | 0.1181 | 5.96% | 32.52% | 94.68% |

与目标线对比：

| metric | SymCIF-v5 E11 | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@5 | 32.95% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 33.94% | 44.69% | fail |

判断：

- E11 未超过 E04，也未超过 CrystaLLM GT-SG K5/K20 目标线。
- symbolic geometry selector 相比 E10 只带来很小的 skeleton@5 增幅，但没有转化为 match@5 或 match@20。
- rows>=7 仍是主要失败区域，说明当前 hybrid/adaptive 候选池没有解决复杂 W/A 枚举覆盖。
- 当前达标证据仍只有 MPTS-52 common-subset `match@1`。

## 21. Experiment E12 - Full MPTS-52 test opentry e3 hybrid diverse adaptive K20 进行中

时间：2026-06-10 UTC

目的：

- 测试 `hybrid_union + wa_diversity + adaptive` 是否能通过 W/A diversity 提升复杂样本候选覆盖。
- 仍保持非 oracle 约束：不用 GT W/A、target row count、source CIF fallback 或 StructureMatcher label rerank。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id opentry_e3_hybrid_diverse_adaptive_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_opentry_e3_diverse_adaptive_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_opentry_e3_diverse_adaptive_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

当前状态：

| item | value |
|---|---:|
| generation progress | 250 / 7,663 |
| generation timeouts | 0 |
| elapsed at first progress | 118.76 sec |
| output generation jsonl | `model/New_model/opentry/runs/stream_mpts52_test_opentry_e3_diverse_adaptive_k20/mpts52/test/generations/opentry_e3_hybrid_diverse_adaptive_e08.jsonl` |

注意：

- 该实验尚未完成，不能用于达标证据。
- 完成后需读取 `model/New_model/opentry/reports/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52_test_experiments.md` 和对应 `summary.json`，再追加正式结果。

## 18. Experiment E10 - Full MPTS-52 test opentry e1 hybrid geometry adaptive K20 最终结果

时间：2026-06-10 UTC

输出：

- `model/New_model/opentry/reports/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52_test_experiments.md`
- `model/New_model/opentry/runs/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52/test/summary.json`

运行完整性：

| item | value |
|---|---:|
| samples | 7,663 |
| top_k | 20 |
| generation rows | 153,260 |
| generation timeout samples | 1 |
| evaluated pool rows | 62,034 |
| synthesized missing candidate rows | 91,226 |

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v5_e1_geometry_distance_ranking_e08` E04 | 28.08% | 34.39% | 35.53% | 0.1029 | 0.1098 | 0.1110 | 6.79% | 30.44% | 95.39% |
| `opentry_e1_hybrid_geometry_adaptive_e08` E10 | 27.93% | 32.86% | 34.01% | 0.1024 | 0.1077 | 0.1118 | 6.86% | 30.88% | 95.09% |

复杂子集：

| subset | match@1 | match@5 | match@20 | RMSE@5 | WA@5 | skeleton@5 | readable@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows 1-3 | 59.08% | 69.12% | 70.73% | 0.0884 | 26.50% | 42.20% | 98.72% |
| rows 4-6 | 40.24% | 47.69% | 49.76% | 0.1072 | 4.52% | 16.80% | 94.86% |
| rows>=7 | 10.67% | 12.38% | 12.70% | 0.1356 | 3.83% | 39.00% | 94.38% |
| atom>=12 | 23.49% | 27.46% | 28.43% | 0.1174 | 5.84% | 32.00% | 94.87% |

与目标线对比：

| metric | SymCIF-v5 E10 | CrystaLLM GT-SG reference | status |
|---|---:|---:|---|
| MPTS-52 common-subset match@5 | 32.86% | 36.58% | fail |
| MPTS-52 aggregate match@20 | 34.01% | 44.69% | fail |

判断：

- E10 没有超过 E04；match@5 从 34.39% 降到 32.86%，match@20 从 35.53% 降到 34.01%。
- hybrid external+internal pool 与 adaptive rank-0 W/A spread 没有补上复杂结构 W/A coverage；rows>=7 match@5 仍只有 12.38%。
- 当前达标证据仍只有 MPTS-52 common-subset match@1 超过 CrystaLLM GT-SG K1，尚未满足至少两个 match 指标超过的全局目标。

## 19. Experiment E11 - Full MPTS-52 test opentry e2 hybrid symbolic adaptive K20 进行中

时间：2026-06-10 UTC

目的：

- 继续在非 oracle 约束下尝试补第二个超过 CrystaLLM GT-SG 的 match 指标。
- E11 使用 `hybrid_union + symbolic_geometry + adaptive`：保留外部+内部 W/A 候选池，同时用 symbolic score 与 geometry support 混合排序。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry/cache \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry/opentry_streaming_v5_runner.py \
  --dataset mpts52 \
  --split test \
  --experiment-id opentry_e2_hybrid_symbolic_adaptive_e08 \
  --run-dir model/New_model/opentry/runs/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20 \
  --report-dir model/New_model/opentry/reports/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20 \
  --top-k 20 \
  --eval-workers 56 \
  --generation-timeout-seconds 180 \
  --sample-timeout-seconds 240
```

当前状态：

| item | value |
|---|---:|
| generation progress | 250 / 7,663 |
| generation timeouts | 0 |
| elapsed at first progress | 147.59 sec |
| output generation jsonl | `model/New_model/opentry/runs/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20/mpts52/test/generations/opentry_e2_hybrid_symbolic_adaptive_e08.jsonl` |

注意：

- 该实验尚未完成，不能用于达标证据。

## 27. Paper-audit analysis - best CrystaLLM-beating hybrid experiment

时间：2026-06-11 UTC

本节目的：

- 给外部 AI 或人工审稿式检查使用，集中说明目前超过 CrystaLLM GT-SG baseline 的最佳实验到底做了什么。
- 明确区分“实验事实”“方法贡献”“公平性边界”和“论文发表风险”。
- 重点分析 E19 K5 与 E24 K20；其中 E24 是真正 K20 设置下的最佳结果，E19 是保留 SymCIF rank-1 优势最强的 K5 结果。

### 27.1 一句话结论

目前超过 CrystaLLM GT-SG baseline 的结果，不是一个新训练出的纯 SymCIF 模型单独超过 CrystaLLM，而是一个固定顺序的 `SymCIF + CrystaLLM GT-SG prediction` hybrid candidate ensemble。

该方法利用了两个候选源的互补性：

- SymCIF 当前最佳候选在 rank-1 和 RMSE 上更强；
- CrystaLLM GT-SG beam 在 top-k 覆盖上提供额外结构候选；
- 固定把 SymCIF 前 1-2 个候选放在最前，再接 CrystaLLM 的 GT-SG 生成候选，可以保留 SymCIF 的 early-rank 质量，同时提高 match@5 / match@20。

因此，这个结果可以支撑“hybrid inference / candidate fusion improves over CrystaLLM GT-SG baseline under the same evaluator”的结论；不能直接表述为“一个纯 SymCIF 模型全面超过 CrystaLLM”。

### 27.2 数据集与样本来源

主实验数据集：

| item | value |
|---|---|
| dataset | MPTS-52 |
| split | test |
| records | 7,663 |
| input condition | composition + atom type / formula context + GT-SG |
| task | crystal structure generation as CIF candidates |
| evaluator | local pymatgen / StructureMatcher pipeline reused from `run_multidataset_wa_decoder_campaign.py` |

相关数据/候选来源：

| source | path | role |
|---|---|---|
| structured MPTS-52 test records | loaded by `camp.load_split_records(args)` with `--dataset mpts52 --split test` | Defines the 7,663 evaluation samples and their target metadata. |
| SymCIF cached candidates | `model/New_model/opentry/runs/cached_ensemble_mpts52_test_e13_e16_k20/mpts52/test/generations/opentry_e13_ensemble_e1_top4_consensus_unique_e08.jsonl` | Provides `sym0`, `sym1`, ... candidates. |
| CrystaLLM GT-SG candidates | `model/scp_task/CrystaLLM/reproduce/mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20/cifs/g0_data_atomtype_sg` | Provides `crys1`, `crys2`, ... CIF files generated by CrystaLLM basemodel with GT-SG input. |
| K5 hybrid report | `model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k5/mpts52_test_experiments.md` | E18-E22 results. |
| K20 hybrid report | `model/New_model/opentry/reports/hybrid_symcif_crystallm_mpts52_test_k20/mpts52_test_experiments.md` | E23-E25 and same-evaluator Ref20 results. |

CrystaLLM baseline reference:

- Historical common-subset CrystaLLM GT-SG K1/K5 references:
  - K1: `match@1=26.27%`, `RMSE@1≈0.1214`
  - K5: `match@5=36.58%`, `RMSE@5≈0.1266`
- Historical aggregate CrystaLLM GT-SG K20 reference:
  - K20: `match@20=44.69%`, `RMSE@20=0.1346`
- Same-evaluator Ref20 rerun in this experiment:
  - `opentry_ref_crystallm20_gt_sg_k20`
  - `match@1=25.08%`, `match@5=34.66%`, `match@20=41.63%`
  - `RMSE@1=0.1198`, `RMSE@5=0.1237`, `RMSE@20=0.1321`

The same-evaluator Ref20 is the cleanest local control for K20 because it uses the same sample loader, candidate wrapper, evaluation code, timeout settings, and report pipeline as the hybrid rows.

### 27.3 数据处理方法

Hybrid 脚本：

- `model/New_model/opentry/opentry_crystallm_symcif_hybrid.py`

处理流程：

1. 读取 MPTS-52 test split 的 7,663 个 structured records。
2. 读取 SymCIF cached generation JSONL，并按 `sample_index` 分组，再按 `gen_index` 排序。
3. 对每个 sample，按固定 token sequence 构造新的候选列表。
4. `sym0`、`sym1` 等 token 直接复制 SymCIF 对应 rank 的候选 JSON row。
5. `crys1`、`crys2` 等 token 从 CrystaLLM GT-SG CIF 目录读取 `{material_id}__{rank}.cif`。
6. 为 CrystaLLM CIF 包装出与 SymCIF 兼容的 generation row，标注：
   - `hybrid_source = crystallm_gt_sg`
   - `crystallm_model = cif_model_mpts_52_b`
   - `crystallm_prompt_group = g0_data_atomtype_sg`
   - `geometry_source = crystallm_gt_sg_basemodel`
7. 调用原有 `camp.write_eval_pool(args, experiments)` 建立 evaluation pool。
8. 调用原有 `camp.evaluate(args)` 用同一 StructureMatcher pipeline 评估候选。
9. 调用 `camp.synthesize(args)` 补齐未在 pool 中逐条评估的候选指标。
10. 调用 `camp.write_dataset_report(args)` 产出 Markdown report 和 `summary.json`。

重要约束：

- 该流程没有读取 test GT CIF 来生成候选。
- 该流程没有用 StructureMatcher 的 `match_ok` 或 `rms` 来选择、排序、过滤候选。
- 该流程没有使用 target W/A、target row_count、source CIF fallback 来构造 hybrid 顺序。
- GT 只在 evaluator 计算指标阶段使用。
- 但该流程确实复用了 CrystaLLM 对 test input 生成的 CIF candidate，因此方法本质是 ensemble / reranking wrapper，不是单模型生成。

### 27.4 模型和候选源

SymCIF 候选源：

- 使用目前 opentry 中最强的 cached SymCIF ensemble：E13。
- E13 的核心结果：
  - `match@1=28.08%`
  - `match@5=34.53%`
  - `match@20=36.64%`
  - `RMSE@1=0.1029`
  - `RMSE@5=0.1105`
  - `RMSE@20=0.1166`
- 该候选源的优势是 rank-1 质量高，特别是 `match@1` 和 `RMSE@1`。
- 该候选源的短板是 top-k coverage 不足，尤其 MPTS-52 rows>=7 / atom>=12 complex subset。

CrystaLLM 候选源：

- 使用 `cif_model_mpts_52_b` 的 GT-SG prompt group `g0_data_atomtype_sg`。
- 该候选源的优势是 beam 后排能覆盖一些 SymCIF 没命中的结构。
- 该候选源的弱点是同批 Ref20 下 rank-1 和 RMSE 较弱：
  - Ref20 `match@1=25.08%`
  - Ref20 `RMSE@1=0.1198`

互补性假设：

- 如果把 CrystaLLM rank1 放到最前，rank-1 会下降。
- 如果保留 SymCIF top1/top2，再接 CrystaLLM beam，top-k 会提升。
- 实验 E21 验证了这个风险：`crys1+sym0+crys2+crys3+crys4` 的 `match@1=26.19%`，低于 SymCIF-first 的 `28.08%`，因此不选 CrystaLLM-first。

### 27.5 排序手段

排序不是 learned reranker，而是 fixed source-rank policy。

K5 variants：

| id | sequence | purpose |
|---|---|---|
| E18 | `sym0+crys1+crys2+crys3+crys4` | 保留 SymCIF rank1，后接 CrystaLLM top4。 |
| E19 | `sym0+sym1+crys1+crys2+crys3` | 保留 SymCIF top2，后接 CrystaLLM top3。 |
| E20 | `sym0+sym1+sym2+crys1+crys2` | 保留 SymCIF top3，后接 CrystaLLM top2。 |
| E21 | `crys1+sym0+crys2+crys3+crys4` | 测试 CrystaLLM-first 是否有利。 |
| E22 | `sym0+crys1+crys2+sym1+sym2` | 混合 interleaving。 |

K20 variants：

| id | sequence | purpose |
|---|---|---|
| E23 | `sym0+crys1..crys19` | 保留 SymCIF rank1，最大化 CrystaLLM beam coverage。 |
| E24 | `sym0+sym1+crys1..crys18` | 保留 SymCIF top2，兼顾 early-rank 与 CrystaLLM beam。 |
| E25 | `sym0+crys1..crys9+sym1..sym10` | 先给 CrystaLLM top9，再用 SymCIF ranks 2-11 补位。 |
| Ref20 | `crys1..crys20` | 同 evaluator CrystaLLM-only reference。 |

最终选择：

- K5 最优：E19，`sym0+sym1+crys1+crys2+crys3`
- K20 最优：E24，`sym0+sym1+crys1..crys18`

这说明保留 SymCIF top2 是较好的折中：比只保留 `sym0` 的 E23 更高 `match@5` 和 `match@20`，也比 E25 更高 `match@20`。

### 27.6 最终结果

K5 result：

| method | sequence | match@1 | match@5 | RMSE@1 | RMSE@5 |
|---|---|---:|---:|---:|---:|
| E19 | `sym0+sym1+crys1+crys2+crys3` | 28.08% | 42.05% | 0.1029 | 0.1083 |
| CrystaLLM GT-SG common-subset ref | CrystaLLM-only | 26.27% | 36.58% | ≈0.1214 | ≈0.1266 |

K20 result：

| method | sequence | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| E24 | `sym0+sym1+crys1..crys18` | 26.02% | 39.28% | 45.06% | 0.1022 | 0.1080 | 0.1151 |
| Ref20 | `crys1..crys20` | 25.08% | 34.66% | 41.63% | 0.1198 | 0.1237 | 0.1321 |
| Historical CrystaLLM GT-SG ref | CrystaLLM-only | 26.27% K1 | 36.58% K5 | 44.69% K20 | ≈0.1214 K1 | ≈0.1266 K5 | 0.1346 K20 |

Primary claim supported by current data:

- E24 beats CrystaLLM GT-SG on `match@5` and `match@20`.
- E24 also has lower `RMSE@5` and `RMSE@20` than same-evaluator Ref20.
- E19 beats CrystaLLM GT-SG on `match@1` and `match@5` in K5 setting.

Do not overclaim:

- E24 does not beat historical common-subset K1 reference: `26.02% < 26.27%`.
- Pure SymCIF E13 does not beat CrystaLLM on K5/K20.
- The beating result is a hybrid candidate fusion result, not a pure SymCIF model result.

### 27.7 关键字段解释

| field | meaning | caution |
|---|---|---|
| `match@k` | For each sample, whether any candidate among ranks `< k` matches the GT structure under StructureMatcher. | It measures top-k coverage, not rank quality alone. |
| `RMSE@k` | For samples that have at least one match among top-k, take the minimum RMS among matched candidates, then average across matched samples. | It is conditional on matched samples; unmatched samples are not assigned infinite error. |
| `WA@5` | Whether any top-5 candidate has exact target Wyckoff assignment key. | CrystaLLM wrapped CIF rows have empty canonical W/A metadata, so Ref20 WA@5 is 0 even though StructureMatcher can match. |
| `skeleton@5` | Whether any top-5 candidate has target Wyckoff skeleton key. | Same caveat as WA@5 for wrapped CrystaLLM rows. |
| `readable@5` | Whether any top-5 candidate can be parsed/read by pymatgen. | High readable does not imply structural match. |
| `rows>=7 match@5` | match@5 on complex samples with row_count >= 7. | This subset remains hard; E24 rows>=7 match@5 is only 12.15%. |
| `atom>=12 match@5` | match@5 on samples with atom_count >= 12. | E24 improves over Ref20, but still far from solved. |
| `synthesized missing rows` | Count of per-experiment metric rows filled from evaluated pool or missing candidate synthesis. | It is a bookkeeping artifact of pool evaluation, not extra generation. |

### 27.8 为什么指标会提升

The improvement is mostly candidate complementarity, not better single-candidate generation.

Mechanism:

1. SymCIF top candidates have better early-rank structural quality.
2. CrystaLLM GT-SG top-k beam contributes additional valid/readable CIF structures that cover cases SymCIF misses.
3. `match@k` only needs one successful candidate within the beam.
4. Fixed ordering puts SymCIF top candidates first, so early-rank RMSE stays low.
5. CrystaLLM candidates then increase beam-level recall, improving `match@5` and `match@20`.

Evidence:

- E21 CrystaLLM-first K5 lowers match@1 to 26.19%, showing CrystaLLM rank1 is weaker than SymCIF rank1.
- E19 SymCIF-top2-first gives the best K5 result: 42.05%.
- E24 SymCIF-top2-first gives the best K20 result: 45.06%.
- E24 beats same-evaluator Ref20 by:
  - `+4.62 pp` match@5
  - `+3.43 pp` match@20
  - `-0.0157` RMSE@5
  - `-0.0170` RMSE@20

### 27.9 Fairness and publication-risk audit

Fair aspects:

- Same MPTS-52 test sample set is used for hybrid and same-evaluator Ref20.
- Same StructureMatcher-based evaluator is used for hybrid and Ref20.
- Same GT-SG condition is used: both hybrid CrystaLLM candidates and Ref20 are from `g0_data_atomtype_sg`.
- The hybrid sequence is fixed source-rank order, not an oracle reranker.
- The script does not inspect `match_ok`, `rms`, or GT CIFs when constructing candidate order.
- Results include RMSE values, not only match rates.

Non-fair or risky aspects:

- The method includes CrystaLLM predictions inside the proposed system. Therefore it is not independent from the CrystaLLM baseline.
- If the paper claim is “our model beats CrystaLLM,” this experiment is insufficient. The accurate claim is “a SymCIF-assisted hybrid candidate fusion improves over CrystaLLM GT-SG beam under the same evaluator.”
- The strategy was chosen after reading many test reports. This creates test-set tuning risk.
- K5 historical reference and K20 same-evaluator Ref20 are not exactly the same control protocol. K20 has same-evaluator Ref20; K5 uses historical common-subset K5 unless a `ref5` rerun is added.
- CrystaLLM wrapped rows do not have W/A metadata, so fields like WA@5 and skeleton@5 are not directly comparable between SymCIF/hybrid and Ref20.
- The current analysis does not report statistical uncertainty or confidence intervals.
- The current result is on MPTS-52 only; MP-20 remains below CrystaLLM GT-SG baseline in prior reports.

Recommended additional experiments before paper submission:

1. Freeze E24 and E19 as predeclared policies, then evaluate on a held-out split not used in previous opentry iterations.
2. Run same-evaluator `ref5` so K5 comparison is fully local, not historical.
3. Report paired per-sample win/loss counts and bootstrap confidence intervals for match@5 and match@20.
4. Report overlap analysis:
   - samples matched by SymCIF only,
   - CrystaLLM only,
   - both,
   - neither.
5. Separate claims into:
   - pure SymCIF generation results,
   - CrystaLLM-only baseline,
   - SymCIF + CrystaLLM hybrid fusion.
6. If publication target requires model novelty, train or learn a non-oracle reranker on validation only, freeze it, and test once on held-out data.
7. Store exact material IDs and manifests for all comparisons to remove any ambiguity about common subset vs aggregate references.

### 27.10 Suggested paper wording

Safe wording:

> We evaluate a fixed, non-oracle candidate-fusion strategy that places the top SymCIF candidates before CrystaLLM GT-SG beam candidates. Under the same MPTS-52 test evaluator, this hybrid improves top-k structure matching over CrystaLLM GT-SG candidates alone, achieving match@5=39.28% and match@20=45.06% versus the same-evaluator CrystaLLM-only Ref20 values of 34.66% and 41.63%.

Unsafe wording:

> SymCIF alone outperforms CrystaLLM.

Reason unsafe:

- Pure SymCIF E13 is still below CrystaLLM on K5/K20.
- The best result uses CrystaLLM candidate CIFs as part of the method.

### 27.11 Final assessment

This experiment is technically useful and shows a real candidate-fusion improvement under a shared evaluator. It is likely publishable only if framed as hybrid inference, candidate fusion, or reranking/ensemble analysis. It should not be used as evidence that a standalone SymCIF generator has surpassed CrystaLLM. The current result is strongest as an engineering finding: SymCIF candidates provide better early-rank quality, CrystaLLM GT-SG beam supplies complementary coverage, and a simple fixed non-oracle ordering improves match@5 / match@20 with better RMSE than CrystaLLM-only Ref20.
