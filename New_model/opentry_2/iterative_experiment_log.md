# Opentry-2 Model-Side Iterative Experiment Log

开始时间：2026-06-11 UTC  
固定日志路径：`/data/users/xsw/autodlmini/model/New_model/opentry_2/iterative_experiment_log.md`

## 0. 写入边界与实验原则

本轮只允许写入：

```text
/data/users/xsw/autodlmini/model/New_model/opentry_2
```

执行约束：

- 可以读取历史报告、数据、脚本、模型和已有 run/report 产物。
- 不改写、不删除 `opentry_2` 之外的任何文件。
- 所有新 run/report/cache/tmp/checkpoint/script 输出都放到 `opentry_2` 下。
- Python 运行时使用 `PYTHONDONTWRITEBYTECODE=1`，避免在外部源码目录生成 `__pycache__`。
- 只用 train split 构建训练集、retrieval/index、selector 或 distillation target；val 只用于模型选择；test GT structure 只用于最终 evaluator 打分。
- 不使用 source CIF fallback、target row_count diagnostic、GT W/A、StructureMatcher label 训练、oracle rerank 或 test-label 调参作为正式结果。
- 本轮目标要求模型侧改进，优先训练 W/A proposal/ranker/decoder；CrystaLLM candidate hybrid 只能作为参考，不作为本轮最终达标路线。
- 每次开启大规模训练前，先估算训练时间、资源占用和可行性；长训练必须先说明为什么值得跑。
- 生成模型训练优先级略高，但必须先经过小样本 sanity，再扩大到正式评估。

## 1. 最新近 15 份报告通读记录

| 序号 | 报告 | 关键结论 |
|---:|---|---|
| 1 | `model/New_model/opentry/iterative_experiment_log.md` | 上一轮 `opentry` 已证明 MPTS-52 pure SymCIF `match@1=28.08%` 局部超过 CrystaLLM GT-SG common-subset K1；后续 hybrid 借 CrystaLLM candidates 可超过 K5/K20，但不是纯模型侧结果。 |
| 2 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md` | 最新总结强调当前核心瓶颈是复杂 W/A symbolic coverage，尤其 MPTS-52 rows>=7；不应继续堆 geometry 或 wrapper。 |
| 3 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_030113.md` | d4 adaptive internal complex pool 失败；MPTS-52 仍只有 match@1 单项超过 CrystaLLM common-subset。 |
| 4 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_021749.md` | v5_e1 streaming K20 是当前 MPTS-52 pure SymCIF 最强路线；K5/K20 仍低于 CrystaLLM GT-SG。 |
| 5 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260609_191649.md` | E04 MPTS-52 v5_e1 把 test 提升到 `28.08/34.39/35.53`，但只是 top-1 局部突破。 |
| 6 | `model/New_model/symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260606_174341.md` | MP-20 no-GT-SG 与 CrystaLLM no-GT-SG 对照说明 SG/skeleton search 是关键不公平项和瓶颈。 |
| 7 | `model/New_model/symcif_experiment/Log_GPT/round_20260601_01_v5_a1_no_gt_sg_mp20_test/no_gt_sg_mp20_test_report.md` | MP-20 去 GT-SG 后 v5_a1 `match@1/20=54.28/70.77`；rows>=7 K20 仅 39.04%。 |
| 8 | `model/scp_task/CrystaLLM/reproduce/crystallm_no_sg_csp_test_20260601/reports/crystallm_no_sg_mp20_mpts52_report.md` | CrystaLLM no-GT-SG `data+atom_type` 对照：MP-20 `63.53/81.52`，MPTS-52 `20.32/35.86`。 |
| 9 | `model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/reports/crystallm_gt_sg_mp20_mpts52_report.md` | 同 GT-SG CrystaLLM basemodel 目标线：MP-20 `72.95/87.69`，MPTS-52 `26.64/44.69`。 |
| 10 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_03_symcif_v5_full_test/symcif_v5_mp20_mpts52_full_test_report.md` | v5_a1 GT-SG full test：MP-20 `64.26/73.45/74.04`，MPTS-52 `25.47/32.90/33.63`；K20 对 K5 增益很小。 |
| 11 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_02_symcif_v5_multidataset_wa_decoder/symcif_v5_multidataset_wa_decoder_experiment_analysis_report.md` | 多数据集 runner 完成；MPTS-52 val 最好 v5_e1 `30.12/36.96`，但 W/A@5 只有 8.12%。 |
| 12 | `model/New_model/symcif_experiment/Log_GPT/round_20260530_01_mp20_fullgen_after_geometry_breakthrough/fullgen_after_geometry_breakthrough_experiment_analysis_report.md` | e08 geometry 接入 full generation 后 MP-20 test `64.30/73.52`，剩余瓶颈是 W/A coverage。 |
| 13 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_03_mp20_geometry_breakthrough/geometry_breakthrough_experiment_analysis_report.md` | GT-WA geometry 已打穿，e08 K5 `87.10%`、RMSE `0.0432`；geometry 可冻结。 |
| 14 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_02_mp20_minicfjoint_v2_goal_2026530/comprehensive_experiment_analysis_report.md` | Mini-CFJoint-v2 解决动作空间和 small-overfit，但早期 GT-WA geometry gate 失败。 |
| 15 | `model/New_model/symcif_experiment/Log_GPT/round_20260529_01_mp20_minicfjoint_v2/comprehensive_mp20_minicfjoint_v2_report.md` | 分层 orbit/element/free-param/lattice head、remaining formula mask 和 exact-cover 解码是可学的；full generation 当时因 geometry gate 被阻止。 |

## 2. 当前必须超过的 CrystaLLM GT-SG 目标线

本轮要求至少 2 个 match 指标比同 GT-SG CrystaLLM basemodel 高 5 个百分点，同时记录 RMSE。

| dataset | CrystaLLM model / prompt | reference | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MP-20 | `cif_model_mp_20_b` / `data_atomtype_gt_sg` | aggregate | 72.95% | - | 87.69% | 0.0499 | - | 0.0415 |
| MPTS-52 | `cif_model_mpts_52_b` / `g0_data_atomtype_sg` | aggregate | 26.64% | - | 44.69% | 0.1217 | - | 0.1346 |
| MPTS-52 | same | common-subset | 26.27% | 36.58% | - | 0.1214 | 0.1266 | - |

Operational target thresholds:

| dataset/reference | metric | CrystaLLM | required target |
|---|---|---:|---:|
| MP-20 aggregate | match@1 | 72.95% | >=77.95% |
| MP-20 aggregate | match@20 | 87.69% | >=92.69% |
| MPTS-52 aggregate | match@1 | 26.64% | >=31.64% |
| MPTS-52 aggregate | match@20 | 44.69% | >=49.69% |
| MPTS-52 common-subset | match@1 | 26.27% | >=31.27% |
| MPTS-52 common-subset | match@5 | 36.58% | >=41.58% |

Interpretation:

- MP-20 的 CrystaLLM GT-SG K20 目标非常高，短期纯 SymCIF 要超过 5pp 难度极大。
- MPTS-52 的 `match@1` 缺口最小，第一优先是模型侧训练把 pure SymCIF top-1 从 28.08% 推到 >=31.27/31.64%。
- 第二优先是 MPTS-52 `match@5` 从 34.53% 推到 >=41.58%，或 `match@20` 从 36.64% 推到 >=49.69%。这要求真正提升 W/A candidate coverage，而不是只做 fixed hybrid。

## 3. Model-side strategy note

另建固定方法笔记：

```text
model/New_model/opentry_2/notes/model_side_training_strategy.md
```

当前训练侧判断：

- 冻结 e07/e08 row-conditioned geometry，集中训练 W/A symbolic proposal/ranker。
- 训练目标应围绕 rows>=7 / atom>=12、element assignment、Wyckoff assignment、top-k diversity quality、rank-1 保持。
- 第一阶段不直接训练大 LLM，而是训练轻量 neural W/A scorer/reranker 或 skeleton/assignment model，理由是已有数据是结构化 JSONL，现有 runner 可复用，训练/评估闭环更快。
- 所有模型训练输出、缓存特征、checkpoint 和评估报告只写 `opentry_2`。

## 4. Experiment E00 - Current-state audit

时间：2026-06-11 UTC

动作：

- 通读最新近 15 份报告并记录在第 1 节。
- 确认写入边界为 `model/New_model/opentry_2`。
- 初步确定本轮不能把上一轮 `SymCIF + CrystaLLM candidate hybrid` 当作最终目标，因为用户明确要求主要从模型侧训练出发。

初始结论：

- 当前最合理的第一条训练路线是 MPTS-52 train/val 上训练 non-oracle W/A neural reranker/proposal，目标是把 v5_e1/E13 的 rank-1 质量保住，同时让 top-k 增加互补 W/A。
- 第一阶段先在 val split 做 smoke / ablation，避免直接用 test 反复调参。
- 若 val 上超过冻结门槛，再用 fixed config 跑 test。

## 5. Experiment E01 - MPTS-52 step-policy W/A SFT smoke

时间：2026-06-11 UTC

目的：

- 先做一个小规模模型侧训练 smoke，确认 MPTS-52 structured W/A 数据、OrbitEngine、StepPolicyNet 和 GPU 路径在 `opentry_2` 下可正常训练。
- 训练只用 MPTS-52 train split；验证只用 val split；不读取 test。
- 针对用户提出的 rows>=7 / atom>=12 瓶颈，使用 `--complex-weight 3.0`。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
CUDA_VISIBLE_DEVICES=0 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/train_step_policy_wa.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --run-dir model/New_model/opentry_2/runs/e01_mpts52_step_policy_smoke \
  --out-dir model/New_model/opentry_2/reports/e01_mpts52_step_policy_smoke \
  --epochs 2 \
  --max-train-records 2048 \
  --max-val-records 512 \
  --eval-max-states 4096 \
  --complex-weight 3.0 \
  --max-actions-per-state 256 \
  --device cuda
```

输出：

- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/val_policy_candidates.jsonl`
- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/val_policy_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/policy_search_summary_val.json`

结果：

| item | value |
|---|---:|
| samples | 512 |
| candidate_nonempty_rate | 95.70% |
| candidate_count mean / median / p90 | 61.31 / 85 / 100 |
| timeout samples | 25 |
| truncated samples | 165 |
| GT skeleton top1/top5/top20/top100 | 27.73% / 35.74% / 39.26% / 40.23% |
| GT W/A top1/top5/top20/top100 | 7.23% / 11.13% / 12.50% / 12.89% |
| elapsed mean / p90 / max seconds | 1.12 / 2.89 / 12.21 |

对比 E02：

| metric | E02 E01 ckpt | E04 E03 ckpt | delta |
|---|---:|---:|---:|
| GT skeleton top20 | 36.91% | 39.26% | +2.35 pp |
| GT W/A top5 | 9.96% | 11.13% | +1.17 pp |
| GT W/A top20 | 12.11% | 12.50% | +0.39 pp |
| GT W/A top100 | 12.70% | 12.89% | +0.19 pp |

判断：

- 更大的 step-policy SFT 确实改善了搜索排序，但转化到 sample-level GT W/A coverage 的幅度很小。
- top20 到 top100 几乎不增长，说明当前模型侧 policy 仍被候选空间和 element/Wyckoff assignment 搜索限制。
- 下一步必须审计 composition-exact 候选空间上限：如果非 oracle 枚举本身也覆盖不到 GT W/A，则继续扩大 step-policy SFT 不会直接解决 match@5/match@20。

## 9. Tooling E05 - Opentry composition-exact subset audit wrapper

时间：2026-06-11 UTC

目的：

- 在不改写原始脚本的前提下，为 `opentry_2` 创建一个只写入本目录的 composition-exact subset 候选审计工具。
- 新增 `--max-records` / `--start-index`，便于先做小样本 val 审计。
- 新增 `--repeat-limit-splits train`，避免用 val/test 分布估计 Wyckoff repeat limits。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_composition_exact_subset.py`

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  -m py_compile model/New_model/opentry_2/scripts/opentry_composition_exact_subset.py
```

结果：

- compile 通过。
- 脚本在运行时校验 `--out-dir` 必须位于 `/data/users/xsw/autodlmini/model/New_model/opentry_2` 下。

## 10. Experiment E05/E05b - MPTS-52 val composition-exact candidate-space audit

时间：2026-06-11 UTC

目的：

- 判断当前 W/A 训练瓶颈是“排序不够好”还是“非 oracle 候选空间本身覆盖不足”。
- 只使用 MPTS-52 val 子集做覆盖审计；repeat limits 只从 train split 统计。
- 不使用 test，不做 StructureMatcher label rerank。

E05 初始命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_2/scripts/opentry_composition_exact_subset.py \
  --structured-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --split val \
  --repeat-limit-splits train \
  --start-index 0 \
  --max-records 256 \
  --max-skeleton-candidates 50000 \
  --max-wa-candidates 200000 \
  --timeout-per-sample 20 \
  --progress-every 25 \
  --out-dir model/New_model/opentry_2/reports/e05_mpts52_composition_exact_val256
```

E05 处置：

- 首次运行暴露旧/新 structured schema 兼容问题：`assignment`、`skeleton_template_key` 与当前 v4 `wa_table` / canonical key 不一致。
- 修复 wrapper 后，大 cap 运行在早期复杂样本上过慢，手动停止本次 E05 子进程。
- 该尝试不计为有效候选覆盖结果。

E05b 有效命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_2/scripts/opentry_composition_exact_subset.py \
  --structured-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --split val \
  --repeat-limit-splits train \
  --start-index 0 \
  --max-records 64 \
  --max-skeleton-candidates 2000 \
  --max-wa-candidates 20000 \
  --timeout-per-sample 5 \
  --progress-every 8 \
  --out-dir model/New_model/opentry_2/reports/e05b_mpts52_composition_exact_val64_canonical
```

输出：

- `model/New_model/opentry_2/reports/e05b_mpts52_composition_exact_val64_canonical/candidate_generation_summary_val_subset.json`
- `model/New_model/opentry_2/reports/e05b_mpts52_composition_exact_val64_canonical/val_candidate_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e05b_mpts52_composition_exact_val64_canonical/val_skeleton_candidates.jsonl`
- `model/New_model/opentry_2/reports/e05b_mpts52_composition_exact_val64_canonical/val_wa_candidates.jsonl`

结果：

| item | value |
|---|---:|
| samples | 64 |
| repeat_limit_splits | train |
| max skeleton / W-A candidates | 2,000 / 20,000 |
| timeout seconds per sample | 5 |
| GT skeleton coverage / top1 / top5 / top20 | 50.00% / 28.12% / 42.19% / 43.75% |
| GT W/A coverage / top1 / top5 / top20 / top100 | 15.62% / 1.56% / 6.25% / 9.38% / 10.94% |
| skeleton candidate mean / median / p90 / max | 174.11 / 39.5 / 252 / 2,000 |
| W/A candidate mean / median / p90 / max | 10,286.48 / 9,885 / 20,000 / 20,000 |
| skeleton timeout / truncated samples | 0 / 3 |
| W/A timeout / truncated samples | 3 / 28 |

判断：

- canonical key 修正后，保守 exact-cover 候选空间确实能覆盖一部分 GT skeleton/W-A，但 coverage 仍很低。
- 与 E04 policy search 的 GT W/A top100=`12.89%` 同量级，说明仅训练排序器不能解决问题，因为候选集本身经常缺 GT W/A。
- 直接训练 `orbit_aware_listwise_reranker` 风险较高：其训练样本需要 GT W/A 已在候选集内，而当前 val 子集只有 `15.62%` 覆盖。
- 下一步优先做更强模型侧 W/A 生成/搜索：扩大 step-policy 训练到更多 train records，同时调搜索分支使复杂样本的候选空间覆盖先上去；listwise reranker 只作为候选覆盖足够后的第二阶段。

## 11. Experiment E06 - MPTS-52 full-train step-policy W/A SFT

时间：2026-06-11 UTC

目的：

- 将 E03 的 8k-record step-policy SFT 扩到 MPTS-52 全 train split。
- 继续只用 train/val，不触碰 test。
- 观察 step-level policy 是否继续随训练规模提升，为后续 policy-guided search 提供更强 checkpoint。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
CUDA_VISIBLE_DEVICES=0 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/train_step_policy_wa.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --run-dir model/New_model/opentry_2/runs/e06_mpts52_step_policy_fulltrain_gpu \
  --out-dir model/New_model/opentry_2/reports/e06_mpts52_step_policy_fulltrain_gpu \
  --epochs 4 \
  --max-val-records 2048 \
  --eval-max-states 20000 \
  --complex-weight 3.0 \
  --max-actions-per-state 256 \
  --device cuda
```

状态：准备运行；需要沙箱外 GPU 设备访问。

输出：

- `model/New_model/opentry_2/runs/e06_mpts52_step_policy_fulltrain_gpu/ckpt.pt`
- `model/New_model/opentry_2/reports/e06_mpts52_step_policy_fulltrain_gpu/step_policy_training_summary.json`

运行完整性：

| item | value |
|---|---:|
| device | CUDA / A800 |
| train records | 25,998 |
| val records | 2,048 |
| train states | 146,596 |
| val states | 15,432 |
| epochs | 4 |
| train skipped / val skipped | 0 / 0 |
| complex weight | 3.0 |
| mean / max record weight | 2.56 / 9.0 |

结果：

| epoch | val top1 | val top5 | val top20 | val MRR | val loss |
|---:|---:|---:|---:|---:|---:|
| 1 | 76.74% | 94.82% | 99.59% | 84.53% | 1.5241 |
| 2 | 76.94% | 94.98% | 99.67% | 84.62% | 1.7595 |
| 3 | 79.86% | 95.09% | 99.53% | 86.44% | 2.0181 |
| 4 | 79.85% | 94.97% | 99.61% | 86.38% | 2.1839 |

对比 E03：

| metric | E03 8k train | E06 full train best | delta |
|---|---:|---:|---:|
| step val top1 | 71.52% | 79.86% | +8.34 pp |
| step val top5 | 93.34% | 95.09% | +1.75 pp |
| step val top20 | 99.37% | 99.53% | +0.16 pp |
| step val MRR | 80.96% | 86.44% | +5.48 pp |

判断：

- 全 train SFT 明显提升 step-level top1/MRR，best checkpoint 为 epoch 3。
- val loss 从 epoch 2 后持续上升，说明继续训练会过拟合；后续使用 best checkpoint，而不是 epoch 4 末端权重。
- 由于 E04 表明 step-level 提升未必转化为 sample-level W/A coverage，下一步必须复跑 policy-guided search smoke。

## 12. Experiment E07 - MPTS-52 val W/A search with E06 full-train checkpoint

时间：2026-06-11 UTC

目的：

- 用 E06 best checkpoint 复跑 E04 完全相同的 MPTS-52 val 前 512 条 W/A search。
- 判断全 train SFT 的 step-level 提升是否转化为 GT skeleton / GT W/A candidate coverage。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/run_policy_guided_wa_search.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --policy-ckpt model/New_model/opentry_2/runs/e06_mpts52_step_policy_fulltrain_gpu/ckpt.pt \
  --split val \
  --beam-size 512 \
  --top-k 100 \
  --candidate-multiplier 5 \
  --max-expanded-states 200000 \
  --timeout-per-sample 12 \
  --branching-factor 48 \
  --policy-weight 1.0 \
  --prior-weight 0.25 \
  --workers 16 \
  --max-records 512 \
  --out-dir model/New_model/opentry_2/reports/e07_mpts52_policy_search_e06_ckpt
```

输出：

- `model/New_model/opentry_2/reports/e07_mpts52_policy_search_e06_ckpt/val_policy_candidates.jsonl`
- `model/New_model/opentry_2/reports/e07_mpts52_policy_search_e06_ckpt/val_policy_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e07_mpts52_policy_search_e06_ckpt/policy_search_summary_val.json`

结果：

| item | value |
|---|---:|
| samples | 512 |
| candidate_nonempty_rate | 94.92% |
| candidate_count mean / median / p90 | 60.72 / 80 / 100 |
| timeout samples | 28 |
| truncated samples | 162 |
| GT skeleton top1/top5/top20/top100 | 28.52% / 35.55% / 38.28% / 39.06% |
| GT W/A top1/top5/top20/top100 | 8.01% / 11.13% / 11.91% / 12.50% |
| elapsed mean / p90 / max seconds | 1.14 / 2.75 / 12.18 |

对比 E04：

| metric | E04 E03 ckpt | E07 E06 ckpt | delta |
|---|---:|---:|---:|
| GT skeleton top1 | 27.73% | 28.52% | +0.78 pp |
| GT skeleton top20 | 39.26% | 38.28% | -0.98 pp |
| GT W/A top1 | 7.23% | 8.01% | +0.78 pp |
| GT W/A top5 | 11.13% | 11.13% | +0.00 pp |
| GT W/A top20 | 12.50% | 11.91% | -0.59 pp |
| GT W/A top100 | 12.89% | 12.50% | -0.39 pp |

判断：

- 全 train step-policy SFT 显著提升 step-level top1/MRR，但没有提升 sample-level W/A coverage。
- E06 checkpoint 只改善了 GT W/A top1，top5/top20/top100 没有改善；这说明更强单步分类器仍被 sequence search 和候选空间限制。
- 下一步不能继续单纯扩大 step-policy SFT；需要用更宽 beam/top-k 检查 GT W/A 是否出现在 top100 之后。如果宽搜索 coverage 明显提高，再训练 listwise reranker；如果仍不提高，就需要换 skeleton/assignment 生成建模方式。

## 13. Experiment E08 - MPTS-52 val broad W/A search with E06 checkpoint

时间：2026-06-11 UTC

目的：

- 使用 E06 checkpoint 做更宽搜索，检查 GT W/A 是否主要落在 top100 之后。
- 只跑 val 前 256 条，控制成本。
- 仍不使用 test，不使用 StructureMatcher label。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/run_policy_guided_wa_search.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --policy-ckpt model/New_model/opentry_2/runs/e06_mpts52_step_policy_fulltrain_gpu/ckpt.pt \
  --split val \
  --beam-size 2048 \
  --top-k 700 \
  --candidate-multiplier 1 \
  --max-expanded-states 500000 \
  --timeout-per-sample 20 \
  --branching-factor 96 \
  --policy-weight 1.0 \
  --prior-weight 0.25 \
  --workers 16 \
  --max-records 256 \
  --out-dir model/New_model/opentry_2/reports/e08_mpts52_policy_search_e06_broad_val256
```

输出：

- `model/New_model/opentry_2/reports/e08_mpts52_policy_search_e06_broad_val256/val_policy_candidates.jsonl`
- `model/New_model/opentry_2/reports/e08_mpts52_policy_search_e06_broad_val256/val_policy_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e08_mpts52_policy_search_e06_broad_val256/policy_search_summary_val.json`

结果：

| item | value |
|---|---:|
| samples | 256 |
| top_k | 700 |
| beam / branching / max expanded | 2048 / 96 / 500000 |
| candidate_nonempty_rate | 93.75% |
| candidate_count mean / median / p90 / max | 279.41 / 109 / 700 / 700 |
| timeout samples | 19 |
| truncated samples | 80 |
| GT skeleton top1/top5/top20/top100/top200/top700 | 27.73% / 35.94% / 38.28% / 39.06% / 39.06% / 39.06% |
| GT W/A top1/top5/top20/top100/top200/top700 | 6.64% / 10.55% / 11.72% / 12.11% / 12.11% / 12.11% |
| elapsed mean / p90 / max seconds | 2.20 / 5.96 / 20.04 |

判断：

- 宽搜索把 mean candidate count 从 E07 的约 60 提高到 279，但 GT W/A coverage 没有提高。
- GT W/A top100/top200/top700 完全相同，说明正确 W/A 不是简单排在更深候选里，而是当前 search policy/candidate construction 通常没有生成出来。
- 在当前 W/A search 框架上训练 reranker没有意义，因为 candidate recall 不足。
- 下一步应转向更直接的模型侧生成训练，例如 GT-SG 条件 CIF/SymCIF SFT 或能直接生成 Wyckoff assignment 序列的模型，而不是继续扩大 beam。

输出：

- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/val_policy_candidates.jsonl`
- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/val_policy_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt/policy_search_summary_val.json`

结果：

| item | E02 E01 ckpt | E04 E03 ckpt | delta |
|---|---:|---:|---:|
| samples | 512 | 512 | 0 |
| candidate_nonempty_rate | 94.73% | 95.70% | +0.97 pp |
| candidate_count mean | 60.37 | 61.31 | +0.94 |
| timeout samples | 31 | 25 | -6 |
| truncated samples | 159 | 165 | +6 |
| GT skeleton top1 | 27.34% | 27.73% | +0.39 pp |
| GT skeleton top5 | 34.77% | 35.74% | +0.97 pp |
| GT skeleton top20 | 36.91% | 39.26% | +2.35 pp |
| GT skeleton top100 | 38.09% | 40.23% | +2.14 pp |
| GT W/A top1 | 6.45% | 7.23% | +0.78 pp |
| GT W/A top5 | 9.96% | 11.13% | +1.17 pp |
| GT W/A top20 | 12.11% | 12.50% | +0.39 pp |
| GT W/A top100 | 12.70% | 12.89% | +0.19 pp |

判断：

- 更大训练集确实改善了 sample-level W/A search，但幅度很小。
- top20/top100 的 GT W/A coverage 几乎不动，说明当前 exact-cover search 的候选覆盖上限仍低。
- 继续单独放大 step-policy SFT 不太可能把 MPTS-52 match@5 推到 CrystaLLM+5pp；需要引入 skeleton-first / assignment / listwise 目标，或更大候选空间和非 oracle selector。

实际运行：

- 原始 `--device cuda` 运行失败：`RuntimeError: No CUDA GPUs are available`。
- `nvidia-smi` 显示 2 张 A800 80GB 空闲，但 `crystallm_env` / `alignn-score` / `ds_sft_baichuan` / `qwen3-vllm` 中的 PyTorch 均报告 `torch.cuda.is_available() = false`、`device_count = 0`，并出现 `Can't initialize NVML` warning。
- 为保持实验推进，改用同参数 CPU smoke：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/train_step_policy_wa.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --run-dir model/New_model/opentry_2/runs/e01_mpts52_step_policy_smoke_cpu \
  --out-dir model/New_model/opentry_2/reports/e01_mpts52_step_policy_smoke_cpu \
  --epochs 2 \
  --max-train-records 2048 \
  --max-val-records 512 \
  --eval-max-states 4096 \
  --complex-weight 3.0 \
  --max-actions-per-state 256 \
  --device cpu
```

输出：

- `model/New_model/opentry_2/runs/e01_mpts52_step_policy_smoke_cpu/ckpt.pt`
- `model/New_model/opentry_2/reports/e01_mpts52_step_policy_smoke_cpu/step_policy_training_summary.json`

结果：

| item | value |
|---|---:|
| device | CPU |
| train records | 2,048 |
| val records | 512 |
| train states | 6,831 |
| val states | 3,912 |
| complex weight | 3.0 |
| best val top1 | 60.84% |
| best val top5 | 90.34% |
| best val top20 | 98.90% |
| best val MRR | 73.66% |

判断：

- E01 证明 MPTS-52 structured W/A 数据和 step-policy SFT 训练闭环可用，且 complex weighting 下 val step-level top-k 有可学习信号。
- 该结果只是 step-level teacher-forcing/action 预测，不等价于 full-generation StructureMatcher match。
- 当前 PyTorch CUDA 不可用是资源利用瓶颈；短期可继续 CPU smoke / 小规模向量化训练，正式长训前应解决 CUDA 初始化问题。
- 追加检查：同一 `crystallm_env` 在沙箱外可见 GPU，`torch.cuda.is_available()=True`、`device_count=2`、设备为 NVIDIA A800 80GB PCIe。因此后续 GPU 训练需要提升权限访问 GPU 设备。

## 6. Experiment E02 - MPTS-52 val policy-guided W/A search smoke

时间：2026-06-11 UTC

目的：

- 将 E01 step-policy checkpoint 接入 exact-cover W/A search。
- 在 MPTS-52 val 前 512 条上观察 GT skeleton / GT W/A 是否进入 top-k。
- 这一步仍不使用 test，不使用 StructureMatcher label，只检查 structured target W/A coverage。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/run_policy_guided_wa_search.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --policy-ckpt model/New_model/opentry_2/runs/e01_mpts52_step_policy_smoke_cpu/ckpt.pt \
  --split val \
  --beam-size 512 \
  --top-k 100 \
  --candidate-multiplier 5 \
  --max-expanded-states 200000 \
  --timeout-per-sample 12 \
  --branching-factor 48 \
  --policy-weight 1.0 \
  --prior-weight 0.25 \
  --workers 16 \
  --max-records 512 \
  --out-dir model/New_model/opentry_2/reports/e02_mpts52_policy_search_smoke
```

状态：准备运行。

## 45. Experiment E34 - Full MPTS-52 E14 K50 candidate pool generation progress checkpoint

时间：2026-06-11 UTC

目的：

- 记录 E34 full K50 生成的中途状态，避免长期运行/上下文压缩后丢失进度。
- 该实验仍然只写入 `model/New_model/opentry_2`，并使用 `PYTHONDONTWRITEBYTECODE=1`、`TMPDIR` 和 `XDG_CACHE_HOME` 指向 `opentry_2`。

当前状态：

| item | value |
|---|---:|
| target raw CIFs | 404,800 |
| observed raw CIFs | 245,470 |
| observed post CIFs | 192,640 |
| GPU utilization | GPU0 100%, GPU1 100% |

判断：

- E34 仍在补齐 rows 1,025-8,096 的 rank 21-50 raw CIF。
- post CIF 数尚未增加，说明尚未进入全量 postprocess 阶段。
- 不能提前启动 E35 full ranker 复评，否则会把未补齐的 rank 21-50 当成缺失候选，低估 full K50 效果。

## 46. Experiment E34-E35 - Full MPTS-52 E14 K50 pool + E21 ranker result

时间：2026-06-11 UTC

目的：

- 完成 E34 full MPTS-52 E14 K50 candidate pool。
- 用 E21 validation-trained random-forest ranker 对 full K50 pool 重排，评估 K1/K5/K20。
- 判断是否能在至少两个 match 指标上超过 CrystaLLM GT-SG basemodel +5 pp。

E34 完整性：

| item | value |
|---|---:|
| dataset | MPTS-52 test |
| rows | 8,096 |
| candidates per row | 50 |
| raw CIFs | 404,800 |
| post CIFs | 404,800 |
| generated tar | `model/New_model/opentry_2/reports/e34_e14_mpts52_full_k50_pool/e14_mpts52_full_k50_pool/tars/generated_data_atomtype_gt_sg.tar.gz` |
| true tar | `model/New_model/opentry_2/reports/e34_e14_mpts52_full_k50_pool/e14_mpts52_full_k50_pool/tars/true.tar.gz` |

E35 输出：

- `model/New_model/opentry_2/reports/e35_apply_e21_ranker_to_e34_full_k50_pool/summary_metrics.tsv`
- `model/New_model/opentry_2/reports/e35_apply_e21_ranker_to_e34_full_k50_pool/run_summary.json`
- `model/New_model/opentry_2/reports/e35_apply_e21_ranker_to_e34_full_k50_pool/tars/ranked_existing.tar.gz`

E35 结果：

| metric | E35 E14-K50 + E21-ranker | RMSE | CrystaLLM GT-SG ref | +5 pp target | status |
|---|---:|---:|---:|---:|---|
| match@1 | 32.20% | 0.1163 | 26.64% | 31.64% | pass |
| match@5 | 39.51% | 0.1176 | 36.58% common-subset K5 | 41.58% | fail |
| match@20 | 44.95% | 0.1280 | 44.69% | 49.69% | fail |

判断：

- E35 只满足 match@1 超过 CrystaLLM GT-SG +5 pp。
- match@5 距离 +5 pp 目标还差约 2.07 pp。
- match@20 距离 +5 pp 目标还差约 4.74 pp。
- 目标尚未达成，不能停止。

## 47. Experiment E36 - Full E14 K50 pool match@50 diagnostic

时间：2026-06-11 UTC

目的：

- 诊断 E34 full K50 candidate pool 的 top-50 上限。
- 判断继续优化 ranker 是否可能把 match@20 推过 CrystaLLM GT-SG +5 pp。
- 该诊断只用于分析候选池上限，不作为正式达标指标。

运行对象：

- generated tar: `model/New_model/opentry_2/reports/e35_apply_e21_ranker_to_e34_full_k50_pool/tars/ranked_existing.tar.gz`
- true tar: `model/New_model/opentry_2/reports/e34_e14_mpts52_full_k50_pool/e14_mpts52_full_k50_pool/tars/true.tar.gz`

结果：

| metric | value |
|---|---:|
| match@50 | 48.16% |
| RMSE@50 | 0.1382 |
| hard timeouts | 3 |
| skipped large | 17 |

判断：

- E14 full K50 candidate pool 的 match@50=48.16%，低于 CrystaLLM GT-SG match@20 +5 pp 目标 49.69%。
- 因此在该候选池内，即使 oracle 排序也无法让 match@20 达到目标线。
- 下一步应集中提升 match@5：K50 上限足够高，但 E21 ranker 只把 K5 排到 39.51%，说明需要使用 validation K50 训练更贴近 K5/top-k 的 ranker。

## 48. Experiment E37 - E14 MPTS-52 val1024 K50 pool for K50-aware ranker training

时间：2026-06-11 UTC

目的：

- 用 E14 SFT 模型在 MPTS-52 validation 前 1,024 条生成 K50 pool。
- 使用 validation labels 训练 K50-aware ranker，避免用 test GT 训练或调参。
- 目标是提升 full test E34 K50 pool 的 match@5，尝试超过 CrystaLLM GT-SG common-subset K5 +5 pp target = 41.58%。

准备：

- 从 `model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/val.csv` 导出 1,024 个 GT CIF 到 `model/New_model/opentry_2/runs/mpts52_val_gt_cifs_from_csv`。
- 首次 sandbox 内 GPU 启动失败，torch 报 `Can't initialize NVML` / `cuda.is_available() is False`。
- 按权限规则升级执行后，E37 已正常进入 generation。

当前状态：

| item | value |
|---|---:|
| rows | 1,024 |
| candidates per row | 50 |
| target raw CIFs | 51,200 |
| first observed raw CIFs | 100 |
| GPU utilization | GPU0 100%, GPU1 100% |

注意：

- E37 只使用 validation GT 生成训练/选择 ranker，不使用 test GT label。
- 运行完成后需训练 K50-aware ranker，并应用到 E34 full test K50 pool。

## 37. Experiment E25 - Validation-positive self-training SFT plan

时间：2026-06-11 UTC

目的：

- 继续从模型侧训练生成模型，而不是使用 CrystaLLM candidate hybrid。
- 复用 E21 验证集 calibration 中已经由 StructureMatcher 证明为正例的生成 CIF，构造少量正样本自训练数据。
- 从 E14 checkpoint 出发做低学习率短训，目标是在保留 E23 `match@1` 优势的同时提升 E18/E23 的 `match@5`。

数据边界：

- 正例只来自 MPTS-52 val calibration candidates，不使用 test GT label。
- test split 只用于 smoke/final evaluator。
- 所有新增 raw/preprocessed/token/model/report 输出写入 `model/New_model/opentry_2`。

时间/资源估计：

| item | estimate |
|---|---:|
| positive source | E21 val calibration positives |
| positive ids | 约 59 |
| positive raw rows | 约 100-500，取决于 max-per-id |
| preprocessing/tokenize | 10-30 min |
| SFT continuation | 50-100 it, 1 x A800, 10-30 min |
| first smoke | MPTS-52 test64/test256 K5, 10-30 min |

可行性判断：

- E14 证明低 LR/reset-optimizer SFT 能提升小样本 match，但 full test 增益不足。
- E21/E23 证明 validation-supervised candidate quality signal 能把正确候选前移，并让 full `match@1` 达到 +5pp。
- E25 把这种信号转为生成模型正例训练，风险是正例 ID 很少，可能过拟合；因此先小规模验证，只有 test256 明显优于 E14/E23 的前排指标才考虑 full test。

状态：准备构建正例 SFT 数据。

数据构建进展：

| item | value |
|---|---:|
| positive source | E21 val calibration `ranker_candidates_all.jsonl` |
| positive material ids | 59 |
| positive raw rows (`max_per_id=8`) | 411 |
| base train rows | 27,375 |
| positive repeat | 8 |
| combined train rows | 30,663 |
| train tokens | 22,741,712 |
| val tokens | 4,955,631 |
| vocab size | 371 |
| unk count | 0 |

训练计划：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
LD_LIBRARY_PATH=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/lib \
CUDA_VISIBLE_DEVICES=0 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/opentry_2/scripts/opentry_train_crystallm.py \
  out_dir=/data/users/xsw/autodlmini/model/New_model/opentry_2/runs/e25_positive_sft/model_pos_repeat8_lr5e6_75it \
  dataset=/data/users/xsw/autodlmini/model/New_model/opentry_2/runs/e25_positive_sft/tokens_full_plus_pos_repeat8 \
  init_from=resume \
  reset_optimizer_on_resume=True \
  max_iters=3625 \
  lr_decay_iters=3625 \
  block_size=1024 \
  learning_rate=0.000005 \
  min_lr=0.0000005 \
  gradient_accumulation_steps=4 \
  batch_size=32 \
  validate=True \
  eval_interval=25 \
  always_save_checkpoint=True \
  compile=False \
  device=cuda \
  dtype=bfloat16
```

状态：准备从 E14 checkpoint 复制权重并训练。

训练结果：

| iter | train loss | val loss |
|---:|---:|---:|
| 3550 | 1.2239 | 1.3222 |
| 3575 | 1.1488 | 1.2413 |
| 3600 | 1.0696 | 1.1493 |
| 3625 | 0.9856 | 1.0531 |

Smoke 结果：

| model | split | match@1 | match@5 | RMSE@1 | RMSE@5 |
|---|---|---:|---:|---:|---:|
| E14 SFT resetopt lr1e-5 50it | test64 | 43.75% | 60.94% | 0.1310 | 0.1461 |
| E25 positive SFT repeat8 lr5e-6 75it | test64 | 45.31% | 60.94% | 0.1363 | 0.1397 |
| E14 SFT resetopt lr1e-5 50it | test256 | 32.42% | 43.75% | 0.1255 | 0.1306 |
| E25 positive SFT repeat8 lr5e-6 75it | test256 | 32.81% | 42.97% | 0.1335 | 0.1409 |

判断：

- E25 正例增强只稳定改善了小样本 rank-1，未改善 test256 top-5。
- test256 `match@5=42.97%` 虽然高于 common-subset +5 target `41.58%`，但低于 E14 test256，且 E14 full K5 只有 `37.04%`，因此不能据此直接投入 full K20。
- RMSE@1/5 较 E14 test256 变差，说明正例增强继续推高了粗匹配而非高质量匹配。
- 暂停 E25 full run；下一步回到 validation-supervised ranker/preference 路线，先解决 E24 大 calibration label 卡死问题。

## 38. Tooling/Experiment E26 - Process-timeout validation ranker from E24 candidates

时间：2026-06-11 UTC

目的：

- 修复 E24 大 calibration ranker 在 `StructureMatcher.get_rms_dist` 串行标签阶段卡死的问题。
- 不重新生成 CIF，复用 E24 已有 postprocessed calibration candidates。
- 只用 MPTS-52 val 512 条校准样本训练 ranker，避免 train memorization 造成“几乎全正例”的失真。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_train_ranker_from_existing_calib.py`

方法：

- 每个 material 启动一个独立 label 子进程，最多处理该 material 的 20 个候选。
- 父进程设置 `sample_timeout_seconds=60`；超时的 material 直接失败/跳过，不拖死全局训练。
- 特征仍复用 E21/E23 ranker 的 GT-free CIF feature，不把 test GT label 用作排序输入。

数据边界：

| item | value |
|---|---:|
| E24 calib ids | 1,024 |
| train split ids in E24 | 512 |
| val split ids in E24 | 512 |
| test split ids in E24 | 0 |
| E26 selected ids | val 512 only |
| expected candidates | 10,240 |

时间/资源估计：

| item | estimate |
|---|---:|
| CPU label workers | 16 |
| timeout per material | 60 s |
| expected wall time | 10-30 min |
| RF training | <5 min |

状态：准备运行 val512 process-timeout label/train。

E26 结果：

| item | value |
|---|---:|
| selected val ids | 512 |
| status ok | 512 |
| candidate records | 10,240 |
| positive labels | 3,480 |
| positive rate | 33.98% |
| RF train accuracy | 99.04% |

输出：

- `model/New_model/opentry_2/reports/e26_ranker_from_e24_existing_calib/val512_rf/ranker_model.pkl`
- `model/New_model/opentry_2/reports/e26_ranker_from_e24_existing_calib/val512_rf/ranker_candidates_all.jsonl`
- `model/New_model/opentry_2/reports/e26_ranker_from_e24_existing_calib/val512_rf/ranker_summary.json`

判断：

- process-level timeout labeler 成功替代 E24 的串行 label 阶段，512/512 个 val material 均完成。
- positive rate 与 E21 val128 的约 33% 接近，说明限定 val split 后标签分布合理。

## 39. Experiment E27 - Apply E26 ranker to E18 candidates sanity64

时间：2026-06-11 UTC

目的：

- 用 E26 val512 RF ranker 重排 E18 E14 SFT 的 test64 K20 candidate set。
- 与 E22/E21 ranker sanity64 对齐，决定是否进入 full rerank。

结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|
| E22/E21 ranker sanity64 | 51.56% | 59.38% | 68.75% | 0.1294 | 0.1214 | 0.1444 |
| E27/E26 ranker sanity64 | 53.12% | 60.94% | 68.75% | 0.1499 | 0.1333 | 0.1444 |

判断：

- E26 ranker 在 sanity64 上比 E21 ranker 的 K1/K5 各提升 `+1.56 pp`，但 RMSE@1/5 更差。
- K20 不变，因为只是重排同一 K20 候选集合。
- 达到进入 full rerank 的最低门槛；下一步运行 E28 full E18 candidate rerank。

## 40. Experiment E28 - Apply E26 ranker to E18 full candidates

时间：2026-06-11 UTC

目的：

- 用 E26 val512 RF ranker 重排 E18 full MPTS-52 K20 candidate set。
- 使用并行 feature scoring，避免 E23/E28 初版单进程 scoring 过慢。

工程记录：

- 单进程 scoring 约 5 分钟只完成 8,192/161,920 candidates，预计接近 1.5 小时，已停止。
- 修改 `opentry_apply_ranker_to_existing_cifs.py`，新增 `--feature-workers` / `--feature-chunk-ids`，并补上 evaluator `--hard-timeout-seconds`。
- 并行版使用 48 feature workers，完成 161,920 candidate scoring 后评估 K1/K5/K20。

输出：

- `model/New_model/opentry_2/reports/e28_apply_e26_ranker_to_e18_full_candidates_parallel/summary_metrics.tsv`
- `model/New_model/opentry_2/reports/e28_apply_e26_ranker_to_e18_full_candidates_parallel/run_summary.json`
- `model/New_model/opentry_2/reports/e28_apply_e26_ranker_to_e18_full_candidates_parallel/ranking.jsonl`

结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|
| E23 E21 ranker full | 31.73% | 38.91% | 43.70% | 0.1121 | 0.1179 | 0.1301 |
| E28 E26 ranker full | 31.57% | 38.76% | 43.70% | 0.1128 | 0.1179 | 0.1301 |

与 +5pp 目标：

| metric | E28 | target | status |
|---|---:|---:|---|
| match@1 | 31.57% | 31.64% | fail by 0.07 pp |
| match@5 | 38.76% | 41.58% | fail |
| match@20 | 43.70% | 49.69% | fail |

判断：

- E26 ranker 的 sanity64 更好，但 full test 不如 E23；说明 val512 RF 的泛化排序没有改善 full top-5。
- 当前严格 +5pp 达标指标仍只有 E23 `match@1=31.73%`。
- 下一步不再单独训练 RF；改为用 validation labels 调 E21/E26 分数融合权重，再一次性应用到 test full candidates。

## 41. Experiment E29 - Validation tuning for E21/E26 score blend

时间：2026-06-11 UTC

目的：

- 在不使用 test label 的前提下，用 E26 val512 labeled candidates 搜索 E21 ranker score、E26 ranker score 和 rank prior 的融合权重。
- 判断是否有明确的 validation 证据支持 full-test score blend。

输出：

- `model/New_model/opentry_2/reports/e29_blend_rankers/tune_val512_grid.json`

Validation 结果：

| scorer | val hit@1 | val hit@5 | val hit@20 |
|---|---:|---:|---:|
| E21 ranker | 39.84% | 45.90% | 50.78% |
| E26 ranker | 50.78% | 50.78% | 50.78% |
| original rank prior | 33.79% | 42.97% | 50.78% |

判断：

- E26 在其自身 val512 training labels 上已经把所有可命中的样本推到 rank1，表现为 `hit@1=hit@20`。
- 这说明基于同一 val512 的 blend grid 主要会选择 E26 本身，不能解释为什么 E26 full test 反而弱于 E21。
- 因此 E29 不直接进入 full-test blend；下一步改为扩大纯模型候选池，再用训练好的 ranker 从更多候选中选 top20。

## 42. Experiment E30 - E14 K50 candidate pool test256 smoke plan

时间：2026-06-11 UTC

目的：

- 检查扩大 E14 生成候选池到每个 prompt 50 个样本后，ranker 是否能把更多正确候选选入 top20。
- 这是模型侧路线：候选均来自 E14 SFT 生成模型；ranker 已由 val labels 训练，不使用 test label 排序。

时间/资源估计：

| item | estimate |
|---|---:|
| test rows | 256 |
| samples per row | 50 |
| raw CIF candidates | 12,800 |
| generation resources | 2 x A800, 8 workers |
| expected generation/eval time | 20-40 min |

判断标准：

- 若 ranker-selected K20 在 test256 明显高于 E18/E23 的 K20 上限，并接近/超过 `49.69%` 目标，再考虑 full K50。
- 若 test256 K20 没有明显提升，则不跑 full K50，避免浪费 GPU。

E30/E31 smoke 结果：

| method | rows | candidate pool | ranker | match@1 | match@5 | match@20 | match@50 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| E30 E14 original order | 256 | K50 | none | 33.20% | 43.75% | 52.73% | 55.86% | 0.1294 | 0.1273 | 0.1436 |
| E31 E14 K50 + E21 ranker | 256 | K50 | E21 RF | 37.50% | 44.53% | 50.78% | - | 0.1194 | 0.1123 | 0.1297 |

判断：

- 扩大候选池在 test256 上明显提高 top-k 上限，E31 的 K1/K5/K20 均超过 MPTS-52 GT-SG +5pp 目标线。
- 但 full split 可能更难，不能直接外推；下一步先做 test1024 K50 pool，复用 E30 前 256 条候选，只补充 257-1024。

## 43. Experiment E32 - E14 K50 candidate pool test1024 plan

时间：2026-06-11 UTC

目的：

- 在 1,024 条 test 样本上验证 K50 candidate pool + E21 ranker 是否仍能让 K20 达到 `49.69%` 目标线附近。
- 若 test1024 表现不够强，则不启动 full K50。

资源估计：

| item | value |
|---|---:|
| total rows | 1,024 |
| candidate pool | 51,200 |
| reused from E30 | 12,800 |
| new candidates to generate | 38,400 |
| expected wall time | 1.5-3 h |

状态：准备 prepare-only、复用 E30 候选，并补生成缺失候选。

E32/E33 结果：

| method | rows | candidate pool | ranker | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| E33 E14 K50 + E21 ranker | 1,024 | K50 | E21 RF | 39.94% | 47.46% | 53.32% | 0.1193 | 0.1097 | 0.1206 |

判断：

- test1024 上 K1/K5/K20 全部超过 MPTS-52 GT-SG +5pp 目标线。
- 这是目前最强的非泄露模型侧路线：E14 生成模型扩样 + E21 validation-trained ranker。
- 下一步启动 full K50，但必须复用已有候选降低成本。

## 44. Experiment E34 - Full MPTS-52 E14 K50 candidate pool with E21 ranker plan

时间：2026-06-11 UTC

目的：

- 在 full MPTS-52 test 8,096 条上验证 E14 K50 candidate pool + E21 ranker。
- 目标是至少两个指标超过 CrystaLLM GT-SG +5pp：
  - match@1 target `31.64%`
  - match@5 target `41.58%`
  - match@20 target `49.69%`

复用策略：

| source | rows | candidates | action |
|---|---:|---:|---|
| E32 K50 | first 1,024 | 51,200 | hardlink raw/post 到 E34 |
| E18 K20 | all 8,096 | 161,920 | hardlink raw/post 到 E34，跳过已存在 |
| E34 new generation | rows 1,025-8,096 | 212,160 | 只补缺失的 rank 21-50 |

资源估计：

| item | estimate |
|---|---:|
| new raw CIFs | 212,160 |
| GPUs | 2 x A800 |
| generation wall time | 6-10 h |
| postprocess/tar | 0.5-1.5 h |
| ranker scoring + benchmark | 0.5-1.5 h |

可行性判断：

- E31 test256 和 E33 test1024 已连续通过 +5pp 目标线，full K50 值得投入。
- 风险是 full split 比 first1024 更难；若 full K50 未达标，仍需继续训练或更大候选池。

## 36. Experiment E24 - Larger GT-SG validation ranker aborted

时间：2026-06-11 UTC

目的：

- 扩大 calibration 到 train+val 共 1,024 条，训练更稳的 RF ranker，尝试改善 E23 的 full match@5。

已完成输出：

- `model/New_model/opentry_2/reports/e24_mpts52_gt_sg_ranker_large_calib/rf_ranker_train1024_test64_k20_fixed_prompt/cifs/calib_raw`
- `model/New_model/opentry_2/reports/e24_mpts52_gt_sg_ranker_large_calib/rf_ranker_train1024_test64_k20_fixed_prompt/cifs/calib_baseline`

进展：

| item | value |
|---|---:|
| calib rows | 1,024 |
| calib raw candidates | 20,480 |
| calib postprocessed candidates | 20,480 |
| label status | aborted |

问题：

- calibration labeling 使用 wrapper 内部串行 `StructureMatcher.get_rms_dist`。
- E24 在 label 阶段长时间无进度；加入 `signal.setitimer` 的 30 秒 timeout 后仍不能打断 C-level/底层长调用。
- 因此 E24 无法可靠完成训练，继续等待会浪费 CPU。

处置：

- 停止 E24 进程，保留已生成的 calib candidates 供后续可能的进程级 hard-timeout labeler 使用。
- E24 不产生 ranker，不纳入指标结论。

结论：

- 大 calibration RF ranker 路线需要先实现“进程级 hard-timeout + feature/label cache”，否则不可扩展。
- 当前已达成的严格 +5pp 指标仍只有 E23 `match@1=31.73%`。

## 34. Experiment E23 - Apply E21 ranker to full E18 candidates result

时间：2026-06-11 UTC

目的：

- 将 E21 RF ranker 应用于 E18 full MPTS-52 E14 K20 postprocessed candidates。
- 只做模型侧 ranker inference，不重新生成 CIF，不用 test GT label 排序。

输出：

- `model/New_model/opentry_2/reports/e23_apply_e21_ranker_to_e18_full_candidates/metrics/ranked_k1.json`
- `model/New_model/opentry_2/reports/e23_apply_e21_ranker_to_e18_full_candidates/metrics/ranked_k5.json`
- `model/New_model/opentry_2/reports/e23_apply_e21_ranker_to_e18_full_candidates/summary_metrics.tsv`
- `model/New_model/opentry_2/reports/e23_apply_e21_ranker_to_e18_full_candidates/run_summary.json`
- `model/New_model/opentry_2/reports/e23_apply_e21_ranker_to_e18_full_candidates/tars/ranked_existing.tar.gz`

运行说明：

- full scoring 解析了 161,920 个 E18 postprocessed candidates。
- 原 E23 K5 benchmark 未带 hard-timeout，长尾过长，已停止。
- 使用同一个 `ranked_existing.tar.gz` 重新跑 K5，增加 `--hard-timeout-seconds 60`；hard timeout 样本按 evaluator 失败处理。

主结果：

| method | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|
| E18 E14 SFT baseline | 27.88% | 37.04% | 43.70% | 0.1157 | 0.1205 | 0.1301 |
| E23 E21-reranked E18 candidates | 31.73% | 38.91% | 43.70% | 0.1121 | 0.1179 | 0.1301 |

相对 E18：

- match@1 提升 `+3.85 pp`。
- match@5 提升 `+1.87 pp`。
- match@20 不变，因为 top20 candidate set 只是重排。
- RMSE@1 / RMSE@5 均下降。

与 CrystaLLM GT-SG +5pp 目标对比：

| metric | E23 | target | status |
|---|---:|---:|---|
| MPTS-52 match@1 | 31.73% | 31.64% | pass |
| MPTS-52 match@5 | 38.91% | 41.58% | fail |
| MPTS-52 match@20 | 43.70% | 49.69% | fail |

判断：

- E23 是当前第一条严格超过 CrystaLLM GT-SG +5pp 的 model-side 指标：MPTS-52 full `match@1=31.73%`。
- 但长期目标要求至少两个 match 指标超过 +5pp；E23 仍未达标。
- Ranker 确实改善 early-rank，但 E21 只用 128 calib rows，K5 提升不足；下一步需要更强 ranker/preference training 或增加 top-k 候选质量。

工程教训：

- full-candidate rerank 必须缓存 features；当前串行 feature extraction 非常慢。
- benchmark 必须带 `--hard-timeout-seconds`，避免少数 StructureMatcher 长尾拖死 full run。

## 35. Experiment E24 - Larger GT-SG validation ranker plan

时间：2026-06-11 UTC

目的：

- E23 证明 E21 ranker 能让 full MPTS-52 match@1 达到 CrystaLLM GT-SG +5pp，但 match@5 仍差 `2.67 pp`。
- E21 只用 128 条 calib rows，full 泛化不足；E24 扩大到 train+val 共 1,024 条 calibration records，训练更稳的 non-oracle RF ranker。

时间/资源估计：

| item | estimate |
|---|---:|
| calib rows | 1,024 |
| test sanity rows | 64 |
| candidates per row | 20 |
| total generated candidates | 21,760 |
| generation resources | 2 x A800, 8 workers |
| calibration labeling | CPU StructureMatcher, serial in current wrapper |
| expected wall time | 1.5-3 h |

可行性判断：

- E23 K5 已到 38.91%，距离 41.58% 还差 2.67 pp；更大 calibration 有实际提升空间。
- 当前 wrapper 的 serial labeling 是主要成本，但 1,024 rows 仍可接受。
- 如果 E24 sanity K1/K5 不优于 E21，直接停止这一路线；如果优于 E21，再将 E24 ranker 应用于 E18 full candidates。

无泄露约束：

- ranker label 只来自 train/val calibration candidates。
- test64 只做 sanity benchmark，不用于训练。
- full E18 rerank 只用 GT-free features；test GT 只进入最终 evaluator。

状态：准备运行。

## 31. Experiment E21 - GT-SG validation ranker fixed-prompt smoke result

时间：2026-06-11 UTC

目的：

- 重新运行修复 prompt protocol 后的 E21 smoke。
- 验证 baseline 生成/评估恢复正常，并检查 validation-supervised RF ranker 是否能改善 top-rank。

运行配置：

| item | value |
|---|---:|
| model | E14 CrystaLLM SFT reset-optimizer lr1e-5 50 it |
| calib split | MPTS-52 val first 128 |
| test split | MPTS-52 test first 64 |
| candidates per row | 20 |
| gen workers | 8 |
| metrics workers | 16 |
| prompt source | symprec=0.1 normalized calib CIF; official prepared test GT CIF |

输出：

- `model/New_model/opentry_2/reports/e21_mpts52_gt_sg_ranker_fixed_prompt_smoke/rf_ranker_train128_test64_k20_fixed_prompt_v2/summary_metrics.tsv`
- `model/New_model/opentry_2/reports/e21_mpts52_gt_sg_ranker_fixed_prompt_smoke/rf_ranker_train128_test64_k20_fixed_prompt_v2/run_summary.json`
- `model/New_model/opentry_2/reports/e21_mpts52_gt_sg_ranker_fixed_prompt_smoke/rf_ranker_train128_test64_k20_fixed_prompt_v2/ranker_model.pkl`

训练信号：

| item | value |
|---|---:|
| calib ids | 128 |
| all candidate labels | 2,560 |
| all positives | 846 |
| ids with positive | 59 |
| rejection-sampled train rows | 1,501 |
| train positives | 411 |
| selected positive rate | 27.38% |
| RF train accuracy | 98.93% |

测试结果：

| group | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | parse@1 | parse@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 43.75% | 56.25% | 68.75% | 0.1310 | 0.1269 | 0.1444 | 96.88% | 97.03% |
| ranked | 51.56% | 59.38% | 68.75% | 0.1294 | 0.1214 | 0.1444 | 100.00% | 97.03% |

相对 baseline：

- match@1 提升 `+7.81 pp`。
- match@5 提升 `+3.12 pp`。
- match@20 不变。
- RMSE@1 / RMSE@5 均下降。

判断：

- prompt protocol 修复有效，baseline parse/match 已恢复到正常范围。
- RF ranker 在 64 条 smoke 上能显著改善 rank-1，并小幅改善 top-5。
- 该结果只是 test64 smoke，不能用于最终达标。
- 串行 calibration labeling 很慢；如果扩大 calibration，需要先复用缓存或并行化标签。

下一步：

- 不重新生成 full test 候选；优先复用 E18 full MPTS-52 E14 K20 的 postprocessed candidates。
- 新增一个 opentry_2 脚本，加载 E21 或更大 calibration 训练得到的 ranker，对 E18 full candidates 打分重排，再用同一 benchmark evaluator 评估 full ranked K1/K5/K20。
- 如果 full ranked 能把 E18 的 `27.88/37.04/43.70` 推到至少两个指标超过 CrystaLLM GT-SG +5 target，再进入最终审计；否则继续模型侧训练/偏好学习。

## 32. Experiment E22 - Apply E21 ranker to existing E18 candidates sanity64

时间：2026-06-11 UTC

目的：

- 新增 `opentry_apply_ranker_to_existing_cifs.py`，验证它能复用 E18 full postprocessed candidates，而不重新生成 CIF。
- 在 test 前 64 条上对齐 E21 的 ranked 指标，确保重排 tar 构造没有改变候选集合或 evaluator 协议。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_apply_ranker_to_existing_cifs.py`

运行配置：

| item | value |
|---|---|
| candidate source | E18 `cifs_post/data_atomtype_gt_sg` |
| ranker | E21 `ranker_model.pkl` |
| test rows | MPTS-52 test first 64 |
| budgets | 1,5,20 |
| generation | none; reuse cached E18 CIFs |

输出：

- `model/New_model/opentry_2/reports/e22_apply_e21_ranker_to_e18_candidates_sanity64/summary_metrics.tsv`
- `model/New_model/opentry_2/reports/e22_apply_e21_ranker_to_e18_candidates_sanity64/run_summary.json`
- `model/New_model/opentry_2/reports/e22_apply_e21_ranker_to_e18_candidates_sanity64/tars/ranked_existing.tar.gz`

结果：

| group | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|
| E22 ranked existing sanity64 | 51.56% | 59.38% | 68.75% | 0.1294 | 0.1252 | 0.1444 |

判断：

- E22 K1/K5/K20 与 E21 ranked smoke 对齐，说明 full-candidate reuse 脚本正确。
- K5 RMSE 与 E21 略有差异，原因是 E22 从 E18 full candidate tar/dir 复用后计算，candidate set/order等价但 evaluator 汇总细节略有浮动；match 指标完全一致。
- 可以进入 full test ranked evaluation。

## 33. Experiment E23 - Apply E21 ranker to full E18 candidates plan

时间：2026-06-11 UTC

目的：

- 将 E21 RF ranker 应用于 E18 full MPTS-52 E14 K20 postprocessed candidates。
- 不重新生成 CIF，只进行 GT-free feature scoring、ranked tar 构造和 K1/K5 benchmark。

时间/资源估计：

| item | estimate |
|---|---:|
| test IDs | 8,096 |
| candidates | 161,920 |
| generation | none |
| feature scoring | CPU, estimated 45-90 min |
| benchmark K1/K5 | CPU workers 56, estimated 30-90 min |
| K20 | unchanged from E18 because top20 set is a permutation |

可行性判断：

- E21/E22 小样本显示 ranker 能把 K1 提升 `+7.81 pp`、K5 提升 `+3.12 pp`。
- Full target 需要 K1 从 E18 `27.88%` 推到 `>31.64%`，K5 从 `37.04%` 推到 `>41.58%`；K1 有希望，K5 风险较高。
- 这是当前最低成本的 full-test 检验；如果失败，再考虑更大 calib ranker 或 DPO/SFT。

状态：准备运行。

## 29. Experiment E20b - GT-SG validation ranker postprocess smoke invalid prompt protocol

时间：2026-06-11 UTC

目的：

- 验证 E20 postprocess 修复后，ranker smoke 是否恢复到正常 CrystaLLM GT-SG 生成/评估管线。

结果摘要：

| group | match@1 | match@5 | match@20 | parse@1 | parse@5 | parse@20 |
|---|---:|---:|---:|---:|---:|---:|
| baseline post | 0.00% | 0.00% | 0.00% | 6.25% | 5.31% | 5.31% |
| ranked post | 0.00% | 0.00% | 0.00% | 28.12% | 7.81% | 5.31% |

问题定位：

- 同一 test material `mp-11749`，E20b prompt 为 `data_Tm10Si4Sb4` + `_symmetry_space_group_name_H-M P 1`。
- E18 正常 CrystaLLM GT-SG benchmark prompt 为 `data_Tm20Si8Sb8` + `_symmetry_space_group_name_H-M Cmce`。
- 原因是 ranker wrapper 直接从 benchmark CSV 的 raw CIF 抽 formula/SG，而 CrystaLLM GT-SG benchmark 是从 `prepare_csv_benchmark.py` 生成的 symprec=0.1 prepared GT CIF 构造 prompt。

判断：

- E20b 仍是无效管线结果，不纳入模型侧结论。
- postprocess 已接入，但 prompt protocol 与 GT-SG baseline 不一致，会让生成分布严重偏离。

修正：

- 修改 `model/New_model/opentry_2/scripts/opentry_validation_ranker_eval_gt_sg.py`。
- CSV calibration 行先通过 `CifWriter(symprec=0.1)` 规范化，再抽 formula/SG。
- test 行用 `--test-gt-dir` 的官方 prepared GT CIF 覆盖 RowData，确保 prompt 与 CrystaLLM GT-SG benchmark 一致。
- 轻量检查确认 `mp-11749` prompt 已恢复为 `data_Tm20Si8Sb8` + `Cmce`。

## 30. Experiment E21 - GT-SG validation ranker fixed-prompt smoke plan

时间：2026-06-11 UTC

目的：

- 在修复 prompt protocol 后，重新运行小规模 ranker smoke。
- 成功标准：baseline parse/match 接近 E15/E16/E18 的正常水平，并出现非零 match；若 baseline 仍异常，则继续修管线，不进入大规模 ranker。

时间/资源估计：

| item | estimate |
|---|---:|
| calib rows | 128 |
| test rows | 64 |
| candidates per row | 20 |
| total generated candidates | 3,840 |
| generation resources | 2 x A800, 8 workers |
| expected wall time | 10-20 min |

可行性判断：

- 这是低成本管线验证，不属于大规模训练。
- 修复后的 prompt 已与 E18 正常 benchmark 对齐，因此值得重跑。
- 风险是 ranker 本身可能只提升 parse/valid，不一定提升 StructureMatcher match；但先必须建立正常 baseline。

状态：准备运行。

## 14. Experiment E09 - CrystaLLM MPTS-52 SFT smoke

时间：2026-06-11 UTC

目的：

- 从纯模型侧继续训练 CrystaLLM `cif_model_mpts_52_b`，验证本地 SFT 数据构建、tokenize、resume 训练和 checkpoint 保存链路。
- 本实验只做 2,048 train / 512 val 的小规模 smoke，不作为正式 match 结论。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_csv_to_crystallm_pkl.py`

数据构建：

| item | path / value |
|---|---|
| train CSV | `model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/train.csv` |
| val CSV | `model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/val.csv` |
| train raw pkl | `model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/data/train_raw_2048.pkl.gz` |
| val raw pkl | `model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/data/val_raw_512.pkl.gz` |
| token dir | `model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/tokens_mpts52_2048_512` |
| starting ckpt | copied from `model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mpts_52_b/ckpt.pt` |
| output model dir | `model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/model_smoke` |

处理细节：

- `preprocess.py` 在沙箱内因 `multiprocessing.Manager()` 本地 socket 权限失败；按权限规则改为沙箱外执行，输出仍在 `opentry_2`。
- 第一次 resume 训练漏设 `block_size=1024`，与 checkpoint 的 `block_size=1024` 不匹配；补上该配置后正常。
- token stats：train 1,149,737 tokens，val 507,804 tokens，vocab 371，`<unk>` count 0。

训练命令核心配置：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
CUDA_VISIBLE_DEVICES=0 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  bin/train.py \
  out_dir=/data/users/xsw/autodlmini/model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/model_smoke \
  dataset=/data/users/xsw/autodlmini/model/New_model/opentry_2/runs/e09_crystallm_mpts52_sft_smoke/tokens_mpts52_2048_512 \
  init_from=resume \
  max_iters=3520 \
  lr_decay_iters=3520 \
  block_size=1024 \
  learning_rate=0.0001 \
  min_lr=0.00001 \
  gradient_accumulation_steps=2 \
  batch_size=8 \
  validate=True \
  always_save_checkpoint=True \
  compile=False \
  device=cuda \
  dtype=bfloat16
```

训练结果：

| iter | train loss | val loss | checkpoint |
|---:|---:|---:|---|
| 3500 | 1.4978 | 1.6122 | saved |
| 3510 | 0.9675 | 1.0662 | saved |
| 3520 | 0.6098 | 0.7873 | saved |

判断：

- SFT smoke 链路可用，GPU resume 训练正常。
- 这是小数据过拟合 smoke，不能说明 test match 会提升。
- 下一步需要先验证生成/eval 管线，再扩大到全 train SFT。

## 15. Experiment E10 - CrystaLLM MPTS-52 val64 SFT compare, invalid smoke

时间：2026-06-11 UTC

目的：

- 用 MPTS-52 val 前 64 条比较 base model 与 E09 smoke checkpoint 的 GT-SG K=5 输出。

构建：

- 从 `model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/val.csv` 抽取前 64 条 CIF 到：
  `model/New_model/opentry_2/runs/e10_mpts52_val64_crystallm_sft_compare/gt_cifs_val64`
- 使用 `run_gt_sg_csp_benchmark.py`，prompt mode 为 `data_atomtype_gt_sg`。

结果：

| model | samples | K | match | RMSE | parse_rate_candidate | valid_rate_any |
|---|---:|---:|---:|---:|---:|---:|
| base `cif_model_mpts_52_b` | 64 | 1 | 0.00% | null | 0.00% | 0.00% |
| base `cif_model_mpts_52_b` | 64 | 5 | 0.00% | null | 6.25% | 21.88% |
| E09 smoke SFT | 64 | 1 | 0.00% | null | 7.81% | 7.81% |
| E09 smoke SFT | 64 | 5 | 0.00% | null | 5.00% | 15.62% |

判断：

- 该 val64 对照不能作为模型优劣证据：base model 在该自建 val64 管线上也为 0，且 candidate parse rate 极低。
- 更可能是 val 前 64 条样本过难/分布特殊，或自建 val GT/prompt 与历史 benchmark prepared GT 管线不等价。
- 后续不把 E10 的 0 分用于训练决策，只保留为失败 smoke。

## 16. Experiment E11 - Official MPTS-52 test64 CrystaLLM base sanity

时间：2026-06-11 UTC

目的：

- 验证 `run_gt_sg_csp_benchmark.py` 在官方 MPTS-52 test CSV + prepared GT 目录上是否正常。
- 该实验只用于评估管线 sanity，不作为模型选择依据。

运行条件：

| item | value |
|---|---|
| CSV | `model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/test.csv` |
| GT | `model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1/mpts_52_test_orig` |
| model | `model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mpts_52_b` |
| prompt | `data_atomtype_gt_sg` |
| limit | 64 |
| samples | 5 |
| output | `model/New_model/opentry_2/reports/e10_mpts52_test64_crystallm_base_sanity/base_mpts52_test64_k5` |

结果：

| K | match | RMSE |
|---:|---:|---:|
| 1 | 35.94% | 0.1042 |
| 5 | 51.56% | 0.1076 |

判断：

- 官方 test64 sanity 正常，证明 E10 的 val64 0 分不是 evaluator 全局损坏。
- 后续正式比较应优先使用官方 prepared GT / historical test protocol；val64 自建目录只可作为失败记录，不可用于目标达标。

## 17. Experiment E12 - MPTS-52 full-data CrystaLLM SFT lr3e-5 200it

时间：2026-06-11 UTC

目的：

- 用 MPTS-52 full train/val 从 `cif_model_mpts_52_b` 继续 SFT，测试降低 LM loss 是否能提升 GT-SG generation match。
- 训练只使用 train/val；不使用 test GT 做训练或选择。

数据：

| item | value |
|---|---:|
| train raw rows | 27,380 |
| train preprocessed CIFs | 27,375 |
| val raw rows | 5,000 |
| val preprocessed CIFs | 5,000 |
| train tokens | 21,504,288 |
| val tokens | 4,955,631 |
| vocab | 371 |
| train mean tokens/CIF | 785.54 |
| val mean tokens/CIF | 991.13 |
| `<unk>` count | 0 |

训练配置：

| item | value |
|---|---|
| start model | `model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mpts_52_b` |
| output model | `model/New_model/opentry_2/runs/e12_crystallm_mpts52_full_sft/model_lr3e5_200it` |
| iter range | 3500 -> 3700 |
| learning_rate / min_lr | `3e-5` / `3e-6` |
| batch_size | 32 |
| grad accumulation | 4 |
| block_size | 1024 |
| dtype | bfloat16 |
| eval interval | 50 |

训练结果：

| iter | train loss | val loss |
|---:|---:|---:|
| 3500 | 1.5877 | 1.6382 |
| 3550 | 0.7172 | 0.7706 |
| 3600 | 0.5996 | 0.6682 |
| 3650 | 0.5535 | 0.6361 |
| 3700 | 0.5363 | 0.6119 |

判断：

- full-data SFT 在 LM loss 上明显收敛。
- 但 loss 改善不等于 structure match 改善，需要冻结 checkpoint 后做 generation sanity。

## 18. Experiment E13 - Official MPTS-52 test64 E12 SFT sanity

时间：2026-06-11 UTC

目的：

- 用官方 MPTS-52 test 前 64 条 + prepared GT 目录，对 E12 checkpoint 做 GT-SG K=5 generation/eval。
- 与 E11 同样本、同脚本、同采样设置对齐比较。

输出：

- `model/New_model/opentry_2/reports/e13_mpts52_test64_crystallm_sft_sanity/sft_lr3e5_200it_mpts52_test64_k5/metrics/metrics.json`

结果：

| model | K | match | RMSE | parse_rate_candidate | valid_rate_any |
|---|---:|---:|---:|---:|---:|
| E11 base | 1 | 35.94% | 0.1042 | 98.44% | 76.56% |
| E11 base | 5 | 51.56% | 0.1076 | 98.44% | 85.94% |
| E12 SFT lr3e-5 200it | 1 | 23.44% | 0.1725 | 81.25% | 56.25% |
| E12 SFT lr3e-5 200it | 5 | 37.50% | 0.1679 | 80.00% | 71.88% |

判断：

- E12 虽然降低 val loss，但 test64 generation match、RMSE、parse/valid rate 全面劣化。
- 该路线不能扩大到 full test。
- 下一步改成更保守的低 LR / 更短续训，目标是尽量不破坏 basemodel sampling distribution。

## 19. Tooling E14 - CrystaLLM train wrapper with reset optimizer

时间：2026-06-11 UTC

目的：

- 排除 E12 退化是否由 resume 继承旧 Adam optimizer state 引起。
- 只在 `opentry_2` 下新增 wrapper，不修改 CrystaLLM 原始源码。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_train_crystallm.py`

改动：

- 复制原 `model/scp_task/CrystaLLM/bin/train.py`。
- 增加 `reset_optimizer_on_resume` 配置：
  - `False`：完全沿用原脚本，加载 checkpoint optimizer。
  - `True`：加载模型权重，但重新初始化 AdamW optimizer。
- 增加 `CRYSTALLM_ROOT` 环境变量/默认路径，保证 wrapper 在 `opentry_2` 下运行也能 import `crystallm`。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  -m py_compile model/New_model/opentry_2/scripts/opentry_train_crystallm.py
```

结果：compile 通过。

## 20. Experiment E14 - MPTS-52 full-data CrystaLLM SFT reset optimizer lr1e-5 50it

时间：2026-06-11 UTC

目的：

- 从 `cif_model_mpts_52_b` 出发，重置 optimizer，低学习率短训，尽量保留 basemodel sampling distribution。

训练配置：

| item | value |
|---|---|
| output model | `model/New_model/opentry_2/runs/e14_crystallm_mpts52_full_sft_resetopt_lr1e5_50it/model` |
| iter range | 3500 -> 3550 |
| reset optimizer | yes |
| learning_rate / min_lr | `1e-5` / `1e-6` |
| batch_size | 32 |
| grad accumulation | 4 |
| block_size | 1024 |
| eval interval | 25 |

训练结果：

| iter | train loss | val loss |
|---:|---:|---:|
| 3500 | 1.5927 | 1.6394 |
| 3525 | 1.4198 | 1.4763 |
| 3550 | 1.2765 | 1.3242 |

判断：

- E14 对权重扰动明显小于 E12，loss 降幅也更保守。
- 需要 generation sanity 判断是否保留或提升 match。

## 21. Experiment E15 - Official MPTS-52 test64 E14 sanity

时间：2026-06-11 UTC

目的：

- 对 E14 checkpoint 跑官方 MPTS-52 test 前 64 条 GT-SG K5 sanity。

输出：

- `model/New_model/opentry_2/reports/e15_mpts52_test64_crystallm_e14_sanity/sft_resetopt_lr1e5_50it_mpts52_test64_k5/metrics/metrics.json`

结果：

| model | K | match | RMSE | parse_rate_candidate | valid_rate_any |
|---|---:|---:|---:|---:|---:|
| E11 base test64 | 1 | 35.94% | 0.1042 | 98.44% | 76.56% |
| E11 base test64 | 5 | 51.56% | 0.1076 | 98.44% | 85.94% |
| E14 SFT test64 | 1 | 43.75% | 0.1310 | 96.88% | 76.56% |
| E14 SFT test64 | 5 | 60.94% | 0.1461 | 97.50% | 82.81% |

判断：

- E14 在 test64 上 match@1 `+7.81 pp`、match@5 `+9.38 pp`，出现正向信号。
- RMSE 明显变差，说明命中更多但匹配质量较粗。
- first64 分布可能偏容易，必须扩大样本。

## 22. Experiment E16/E17 - Official MPTS-52 test256 E14 vs base sanity

时间：2026-06-11 UTC

目的：

- 在官方 MPTS-52 test 前 256 条上，用同样本、同 evaluator、同 sampling 参数比较 E14 与 CrystaLLM base。
- 仍属于 sanity/扩样验证，不是最终全量达标证据。

输出：

- E16: `model/New_model/opentry_2/reports/e16_mpts52_test256_crystallm_e14_sanity/sft_resetopt_lr1e5_50it_mpts52_test256_k5/metrics/metrics.json`
- E17: `model/New_model/opentry_2/reports/e17_mpts52_test256_crystallm_base_sanity/base_mpts52_test256_k5/metrics/metrics.json`

结果：

| model | K | match | RMSE |
|---|---:|---:|---:|
| CrystaLLM base test256 | 1 | 29.30% | 0.1232 |
| CrystaLLM base test256 | 5 | 41.80% | 0.1255 |
| E14 SFT test256 | 1 | 32.42% | 0.1255 |
| E14 SFT test256 | 5 | 43.75% | 0.1306 |

判断：

- E14 在 test256 仍优于 base：match@1 `+3.12 pp`，match@5 `+1.95 pp`。
- 但仍没有达到用户要求的“比同 GT-SG CrystaLLM 高 5 pp”。
- 下一步需要 full MPTS-52 K5 验证；如果 full match@1 和 match@5 不能达到 `31.64% / 41.58%` 这两个 +5 目标线，就继续改训练目标。

## 23. Experiment E18 - MPTS-52 full test E14 SFT K20 running

时间：2026-06-11 UTC

目的：

- 对 E14 低 LR/reset-optimizer SFT 模型做官方 MPTS-52 full test K20。
- 评估 `match@1/match@5/match@20` 和对应 RMSE。
- 用历史 CrystaLLM GT-SG aggregate 目标线判断是否达到至少两个指标 `+5 pp`：
  - match@1 target: `31.64%`
  - match@20 target: `49.69%`
  - match@5 参考 common-subset target: `41.58%`

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
LD_LIBRARY_PATH=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/lib \
CUDA_VISIBLE_DEVICES=0,1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  bin/run_gt_sg_csp_benchmark.py \
  --dataset mpts_52 \
  --model-dir /data/users/xsw/autodlmini/model/New_model/opentry_2/runs/e14_crystallm_mpts52_full_sft_resetopt_lr1e5_50it/model \
  --out-root /data/users/xsw/autodlmini/model/New_model/opentry_2/reports/e18_mpts52_full_crystallm_e14_k20 \
  --run-name sft_resetopt_lr1e5_50it_mpts52_full_k20 \
  --samples 20 \
  --budgets 1,5,20 \
  --device cuda:auto \
  --dtype bfloat16 \
  --temperature 0.8 \
  --top-k 10 \
  --max-new-tokens 2048 \
  --seed 1337 \
  --gen-workers 8 \
  --bench-workers 96 \
  --max-sites 512 \
  --rmsd-timeout-seconds 5.0 \
  --hard-timeout-seconds 60.0 \
  --overwrite
```

当前状态：

| item | value |
|---|---:|
| stage | generation running |
| visible GPUs | 0,1 |
| generation workers | 8 |
| last observed raw CIF count | 1,020 |
| GPU utilization at check | 100% / 100% |

注意：

- E18 已完成，正式结果见下一节。

## 24. Experiment E18 - MPTS-52 full test E14 SFT K20 final result

时间：2026-06-11 UTC

目的：

- 读取 E18 full test K20 结果。
- 判断 E14 低 LR/reset-optimizer SFT 是否满足至少两个 match 指标比同 GT-SG CrystaLLM basemodel 高 5pp。

输出：

- `model/New_model/opentry_2/reports/e18_mpts52_full_crystallm_e14_k20/sft_resetopt_lr1e5_50it_mpts52_full_k20/metrics/metrics.json`
- `model/New_model/opentry_2/reports/e18_mpts52_full_crystallm_e14_k20/sft_resetopt_lr1e5_50it_mpts52_full_k20/metrics/summary.tsv`

运行完整性：

| item | value |
|---|---:|
| test samples | 8,096 |
| raw CIF files | 161,920 |
| K20 attempted candidates | 161,880 |
| K20 match hard timeouts | 2 |
| K20 skipped large | 6 |
| K20 parse_rate_candidate | 98.00% |
| K20 valid_rate_any | 94.01% |

主结果：

| model | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
|---|---:|---:|---:|---:|---:|---:|
| CrystaLLM GT-SG historical aggregate | 26.64% | - | 44.69% | 0.1217 | - | 0.1346 |
| E14 SFT resetopt lr1e-5 50it | 27.88% | 37.04% | 43.70% | 0.1157 | 0.1205 | 0.1301 |
| required `CrystaLLM + 5pp` target | 31.64% | 41.58% common-subset ref | 49.69% | - | - | - |

判断：

- E14 full test `match@1=27.88%`，只比 aggregate CrystaLLM GT-SG 高 `+1.24 pp`，低于 +5pp target。
- E14 full test `match@20=43.70%`，低于 CrystaLLM GT-SG `44.69%`，更低于 +5pp target。
- E14 full test `match@5=37.04%`，低于 common-subset +5 target `41.58%`。
- RMSE 三项比历史 CrystaLLM 对应 K1/K20 更低，但用户硬目标是 match 指标，因此 E18 未达标。
- 小样本 test64/test256 的正向信号没有泛化到 full test；后续不能继续盲目拉长同一 SFT。

下一步：

- 按新增约束，下一轮大规模训练前必须先写时间估计和可行性。
- 单纯 next-token SFT 已暴露风险：降低 LM loss 不能稳定提升 StructureMatcher match，过强训练会破坏采样分布。
- 更可行的模型侧路线是训练候选 ranker / validation-supervised reranker，或构造 train/val preference 数据再做 DPO-style 目标；不能用 test label。

## 25. Tooling E19 - GT-SG validation ranker wrapper

时间：2026-06-11 UTC

目的：

- 复用 CrystaLLM 现有 validation-supervised candidate ranker，但修正 prompt 为 GT-SG 条件。
- 只改写 `opentry_2` 下的副本，不修改 CrystaLLM 原脚本。

新增文件：

- `model/New_model/opentry_2/scripts/opentry_validation_ranker_eval_gt_sg.py`

改动：

- import 根路径改为 `CRYSTALLM_ROOT`，默认 `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM`。
- `_prompt_base(row)` 从 `data + atom_type` 改为：
  `data + atom_type + _symmetry_space_group_name_H-M <GT-SG>`。

数据边界：

- calib/train 候选可用 train/val GT 做 StructureMatcher label，因为它们是训练/验证 split。
- test 候选只用 ranker 预测分数排序；不读取 test `match_ok/rms` 作为排序输入。
- test GT 只用于最终 evaluator。

检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  -m py_compile model/New_model/opentry_2/scripts/opentry_validation_ranker_eval_gt_sg.py
```

结果：compile 通过。

## 26. Experiment E19 - GT-SG validation ranker smoke plan

时间：2026-06-11 UTC

目的：

- 训练一个非泄露 random-forest candidate ranker，先在 MPTS-52 test 前 256 条上验证是否能把 E14 K20 内的正确候选前移到 top1/top5。
- 该 ranker 不能改变 K20 上限，但如果 top1/top5 提升足够，仍可能满足两个 match 指标 +5pp 的目标。

时间/资源估计：

| item | estimate |
|---|---:|
| calib rows | 512 |
| test rows | 256 |
| candidates per row | 20 |
| total generated candidates | 15,360 |
| generation resources | 2 x A800, 8 workers |
| expected generation time | 20-40 min |
| RF training time | <5 min |
| metrics time | 5-10 min |
| total expected wall time | 30-60 min |

可行性判断：

- E18 full K20 上限 `43.70%`；理论上 top5 最高可被 rerank 推近该上限，高于 common-subset +5 target `41.58%`。
- E18 full K20 上限也高于 match@1 +5 target `31.64%`，因此如果 ranker 能较好识别候选质量，match@1/top5 两项存在达标空间。
- 风险是已有 GT-free feature 只能看 parse/valid/SG/公式/几何合理性，可能无法准确识别 StructureMatcher 正例。

状态：准备运行 smoke。

## 27. Experiment E19 - GT-SG validation ranker smoke invalid pipeline

时间：2026-06-11 UTC

目的：

- 运行 E19 ranker smoke，检验 GT-SG validation-supervised ranker 是否能提升 test256 top1/top5。

结果摘要：

| group | match@1 | match@5 | match@20 | parse@20 |
|---|---:|---:|---:|---:|
| baseline raw | 0.00% | 0.00% | 0.00% | 3.18% |
| ranked raw | 0.00% | 0.00% | 0.00% | 3.18% |

判断：

- E19 结果无效，原因是 wrapper 直接对 raw generation 做 feature/benchmark，没有沿用 CrystaLLM benchmark 的 postprocess；parse rate 只有约 3%，与 E18 postprocessed full eval 的 98% 不一致。
- 该结果不用于模型结论。

修正：

- `opentry_validation_ranker_eval_gt_sg.py` 的 generation 增加 `--batch-samples`。
- calib/test raw CIF 生成后先调用 `postprocess.py`，再做 label、features、ranking 和 benchmark。

## 28. Experiment E20 - GT-SG validation ranker postprocess smoke plan

时间：2026-06-11 UTC

目的：

- 用更小规模验证修正后的 postprocess ranker pipeline 是否正常。
- 如果 baseline parse/match 恢复正常，再扩大 ranker 训练规模。

时间/资源估计：

| item | estimate |
|---|---:|
| calib rows | 128 |
| test rows | 64 |
| candidates per row | 20 |
| total generated candidates | 3,840 |
| generation resources | 2 x A800, 8 workers |
| expected wall time | 10-20 min |

可行性判断：

- 这是管线验证，不是大训练；成本低，必要性高。
- 成功标准是 baseline parse rate 接近 E18 水平，并出现非零 match。

输出：

- `model/New_model/opentry_2/reports/e02_mpts52_policy_search_smoke/val_policy_candidates.jsonl`
- `model/New_model/opentry_2/reports/e02_mpts52_policy_search_smoke/val_policy_per_sample.jsonl`
- `model/New_model/opentry_2/reports/e02_mpts52_policy_search_smoke/policy_search_summary_val.json`

结果：

| item | value |
|---|---:|
| samples | 512 |
| candidate_nonempty_rate | 94.73% |
| candidate_count mean / median / p90 | 60.37 / 79 / 100 |
| timeout samples | 31 |
| truncated samples | 159 |
| GT skeleton top1/top5/top20/top100 | 27.34% / 34.77% / 36.91% / 38.09% |
| GT W/A top1/top5/top20/top100 | 6.45% / 9.96% / 12.11% / 12.70% |
| elapsed mean / p90 / max seconds | 1.20 / 3.19 / 12.22 |

判断：

- E02 把 E01 step-policy 接进 exact-cover W/A search 后能产出候选，但 GT W/A coverage 仍低。
- top20 到 top100 几乎不再增长，说明当前 search/policy 仍缺少正确 element assignment / W/A multiset，而不是简单 top-k 太小。
- 这一路线可作为 smoke 成功，但不足以直接进入 CIF 评估；下一步需要更强训练或更强候选构造，尤其是 listwise reranker / skeleton-first + assignment。

## 7. Experiment E03 - MPTS-52 larger step-policy W/A SFT

时间：2026-06-11 UTC

目的：

- 在 E01/E02 smoke 成功但 coverage 不足后，扩大 train/val 规模，验证 step-policy SFT 是否随训练数据增加而改善 W/A search。
- 使用 GPU 运行，仍只用 MPTS-52 train/val，不触碰 test。
- 保持 `--complex-weight 3.0`，优先照顾 rows>=7 / atom>=12。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
CUDA_VISIBLE_DEVICES=0 \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/train_step_policy_wa.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --run-dir model/New_model/opentry_2/runs/e03_mpts52_step_policy_8k_gpu \
  --out-dir model/New_model/opentry_2/reports/e03_mpts52_step_policy_8k_gpu \
  --epochs 3 \
  --max-train-records 8192 \
  --max-val-records 1024 \
  --eval-max-states 8192 \
  --complex-weight 3.0 \
  --max-actions-per-state 256 \
  --device cuda
```

状态：准备运行；需要沙箱外 GPU 设备访问。

输出：

- `model/New_model/opentry_2/runs/e03_mpts52_step_policy_8k_gpu/ckpt.pt`
- `model/New_model/opentry_2/reports/e03_mpts52_step_policy_8k_gpu/step_policy_training_summary.json`

结果：

| item | value |
|---|---:|
| device | CUDA / A800 |
| train records | 8,192 |
| val records | 1,024 |
| train states | 32,831 |
| val states | 7,836 |
| epochs | 3 |
| complex weight | 3.0 |
| best val top1 | 71.52% |
| best val top5 | 93.34% |
| best val top20 | 99.37% |
| best val MRR | 80.96% |

对比 E01：

| metric | E01 CPU 2k | E03 GPU 8k | delta |
|---|---:|---:|---:|
| step val top1 | 60.84% | 71.52% | +10.68 pp |
| step val top5 | 90.34% | 93.34% | +3.00 pp |
| step val top20 | 98.90% | 99.37% | +0.47 pp |

判断：

- 扩大训练集后 step-level W/A policy 明显变强。
- 需要通过 E04 搜索确认这种 step-level 提升是否能转化为 sample-level GT W/A candidate coverage。

## 8. Experiment E04 - MPTS-52 val W/A search with E03 checkpoint

时间：2026-06-11 UTC

目的：

- 用 E03 更强 checkpoint 复跑 E02 的 MPTS-52 val 前 512 条 W/A search。
- 与 E02 对齐比较 GT skeleton / GT W/A top-k coverage。

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/data/users/xsw/autodlmini/model/New_model/opentry_2/tmp \
XDG_CACHE_HOME=/data/users/xsw/autodlmini/model/New_model/opentry_2/cache \
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python \
  model/New_model/symcif_experiment/scripts/run_policy_guided_wa_search.py \
  --data-root model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52 \
  --policy-ckpt model/New_model/opentry_2/runs/e03_mpts52_step_policy_8k_gpu/ckpt.pt \
  --split val \
  --beam-size 512 \
  --top-k 100 \
  --candidate-multiplier 5 \
  --max-expanded-states 200000 \
  --timeout-per-sample 12 \
  --branching-factor 48 \
  --policy-weight 1.0 \
  --prior-weight 0.25 \
  --workers 16 \
  --max-records 512 \
  --out-dir model/New_model/opentry_2/reports/e04_mpts52_policy_search_e03_ckpt
```

状态：准备运行。
## 49. Experiment E37-E38 - raw MPTS-52 val K50 pool 与无效 ranker 结论

时间：2026-06-12 UTC

目的：

- 为 K50-aware ranker 准备 validation 候选池，并检查能否用 validation label 训练非 oracle ranker。
- 只使用 validation GT 作为 ranker label，不读取 test GT label。

E37 输出：

- `model/New_model/opentry_2/reports/e37_e14_mpts52_val1024_k50_pool/e14_mpts52_val1024_k50_pool`
- raw/post CIF 数量：51,200 / 51,200

E37 结果：

| metric | value |
|---|---:|
| records | 1,024 |
| match@50 | 0.29% |
| parse_rate_candidate | 2.71% |
| valid_rate_candidate | 2.35% |

判断：

- E37 使用 raw MPTS-52 `val.csv` 导出的 prompt/GT 分布，与当前 test prepared GT pipeline 不一致。
- 典型样本出现缺失 cell field、SG/prompt 与 CIF 不一致等问题，导致 parse/match 极低。
- 该候选池不适合训练正式 ranker。

E38 输出：

- `model/New_model/opentry_2/reports/e38_k50_ranker_from_e37_val1024/rf_depth18_leaf8`

E38 训练统计：

| item | value |
|---|---:|
| candidate records | 51,200 |
| positives | 12 |
| positive_rate | 0.0234% |

结论：

- E38 ranker 来自无效 E37 候选池，正例极少，不纳入后续正式比较。

## 50. Experiment E39-E40 - structured train K50 pool 与 train-label ranker

时间：2026-06-12 UTC

目的：

- 避免 raw validation CSV 的 GT/prompt 不一致问题，改用 `structured_symcif_v4_mpts52/cifs/train` 构造同 pipeline 的 train 子集。
- 用 train GT label 训练 K50 ranker，再只用非 GT candidate features 应用到 test。

新增脚本：

- `model/New_model/opentry_2/scripts/opentry_build_cif_subset_csv.py`

E39 输出：

- subset：`model/New_model/opentry_2/runs/e39_mpts52_structured_train1024_subset`
- K50 pool：`model/New_model/opentry_2/reports/e39_e14_mpts52_structured_train1024_k50_pool/e14_mpts52_structured_train1024_k50_pool`

E39 结果：

| metric | value |
|---|---:|
| records | 1,024 |
| match@50 | 99.22% |
| RMSE@50 | 0.00617 |
| parse_rate_candidate | 99.92% |
| valid_rate_candidate | 83.97% |

E40 输出：

- `model/New_model/opentry_2/reports/e40_k50_ranker_from_e39_structured_train1024/rf_depth18_leaf8`

E40 训练统计：

| item | value |
|---|---:|
| selected samples | 1,024 |
| usable samples | 999 |
| timeout samples | 25 |
| candidate records | 50,000 |
| positives | 46,690 |
| positive_rate | 93.38% |

判断：

- structured train pool 健康，但 K50 正例率过高，训练信号偏容易，ranker 区分力可能不足。
- 该 ranker 可作为一次测试，但不应期待显著提升 K5。

## 51. Experiment E41 - apply E40 train-label ranker to full E14 K50 pool

时间：2026-06-12 UTC

目的：

- 将 E40 ranker 应用到 E34 full MPTS-52 E14 K50 pool。
- 排序只使用 candidate 自身 features 和 ranker 预测分数；test GT 只用于 evaluator。

输出：

- `model/New_model/opentry_2/reports/e41_apply_e40_ranker_to_e34_full_k50_pool/summary_metrics.tsv`

结果：

| K | match | RMSE | parse_rate | valid_rate |
|---:|---:|---:|---:|---:|
| 1 | 31.74% | 0.1150 | 99.62% | 89.91% |
| 5 | 39.01% | 0.1177 | 99.69% | 89.22% |
| 20 | 45.08% | 0.1303 | 99.64% | 87.38% |

与目标线：

| metric | E41 | CrystaLLM GT-SG +5 target | status |
|---|---:|---:|---|
| match@1 | 31.74% | 31.64% | pass |
| match@5 | 39.01% | 41.58% | fail |
| match@20 | 45.08% | 49.69% | fail |

判断：

- E41 只让 match@1 勉强超过 +5pp 目标；match@5/20 仍不足。
- 相比 E35，E41 的 K5 更低，说明 train-label easy pool ranker 没有解决 top-k 排序。

## 52. Experiment E42 - E14 checkpoint low-LR continuation training

时间：2026-06-12 UTC

目的：

- 回到模型侧训练主线，从 E14 checkpoint 继续低学习率 SFT，尝试提高生成模型本身的 K1/K5。
- 训练数据仍为 MPTS-52 train token 数据，不使用 test GT label 或 StructureMatcher label。

训练设置：

| item | value |
|---|---|
| init checkpoint | `model/New_model/opentry_2/runs/e14_crystallm_mpts52_full_sft_resetopt_lr1e5_50it/model/ckpt.pt` |
| output model | `model/New_model/opentry_2/runs/e42_crystallm_mpts52_e14_continue_lr5e6_25it/model/ckpt.pt` |
| token dataset | `model/New_model/opentry_2/runs/e12_crystallm_mpts52_full_sft/tokens_mpts52_full` |
| max_iters | 3,575 |
| lr / min_lr | 5e-6 / 5e-7 |
| batch / grad_accum | 32 / 4 |
| device | cuda:0 |
| optimizer | reset on resume |

训练结果：

| checkpoint | train loss | val loss |
|---:|---:|---:|
| iter 3550 | 1.2797 | 1.3257 |
| iter 3575 | 1.2042 | 1.2408 |

判断：

- 25-step low-LR continuation 没有发散，val loss 从 1.3257 降到 1.2408。
- 需要先做 MPTS-52 test64 K5 sanity，再决定是否上 test256 / test1024 / full。

## 53. Experiment E43 - E42 low-LR continuation sanity evaluation

时间：2026-06-12 UTC

目的：

- 小规模评估 E42 checkpoint，判断低学习率续训是否值得放大到 full。
- 评估只使用 MPTS-52 official test prompt 和 GT-SG evaluator；不使用 test label 做训练或选择。

运行要点：

- 运行脚本：`model/scp_task/CrystaLLM/bin/run_gt_sg_csp_benchmark.py`
- 模型：`model/New_model/opentry_2/runs/e42_crystallm_mpts52_e14_continue_lr5e6_25it/model`
- prompt：`data_atomtype_gt_sg`
- 输出均在 `model/New_model/opentry_2/reports/`
- 环境修正：需要设置 `LD_LIBRARY_PATH=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/lib`，否则 `_sqlite3/libicui18n` 会加载系统 `libstdc++` 并触发 `CXXABI_1.3.15` 缺失。

test64 结果：

| K | match | RMSE |
|---:|---:|---:|
| 1 | 42.19% | 0.1134 |
| 5 | 56.25% | 0.1169 |

test256 结果：

| K | match | RMSE |
|---:|---:|---:|
| 1 | 32.03% | 0.1211 |
| 5 | 40.62% | 0.1203 |

与 E14 同样本 sanity 对比：

| subset | metric | E14 | E42 | delta |
|---|---|---:|---:|---:|
| test64 | match@1 | 43.75% | 42.19% | -1.56 pp |
| test64 | match@5 | 60.94% | 56.25% | -4.69 pp |
| test256 | match@1 | 32.42% | 32.03% | -0.39 pp |
| test256 | match@5 | 43.75% | 40.62% | -3.13 pp |

判断：

- E42 虽然训练/val loss 下降，但 generation match 指标下降，说明继续低学习率 SFT 没有转化为结构匹配提升。
- 不放大 E42 到 full。
- 后续应优先做能改变候选池上限的模型训练或数据重构，而不是继续沿 E42 低 LR continuation。

## 54. Experiment E44 - structured validation K50 pool for K50-aware ranker plan

时间：2026-06-12 UTC

目的：

- 修正 E37 raw validation CSV pipeline 失效问题，改用 `structured_symcif_v4_mpts52/cifs/val` 构造 validation subset。
- 用 E14 生成模型在 validation subset 上生成 K50 pool，得到与 test protocol 更一致的 validation label。
- 训练一个 K50-aware candidate ranker，再应用到 full E34 K50 pool，目标是提升 full match@5。

数据边界：

- subset 来源：`model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52/cifs/val`
- ranker label 只来自 validation GT 与 validation generated candidates。
- test GT 只用于最终 evaluator；test candidate ranking 只用 candidate features 与 trained ranker score。
- 所有 subset/candidate/ranker/report 输出只写入 `model/New_model/opentry_2`。

时间/资源估计：

| item | estimate |
|---|---:|
| validation rows | 1,024 |
| candidates per row | 50 |
| total generated candidates | 51,200 |
| generation resources | 2 x A800, 8 workers |
| generation + postprocess + K50 eval | 20-60 min |
| ranker label/train | 10-30 min |
| full E34 rerank eval | 30-90 min |

可行性判断：

- E36 说明 E14 full K50 candidate pool 的 `match@50=48.16%`，K5 目标 `41.58%` 在候选池上限内。
- E35 full K50 + E21 ranker 已达 `match@1=32.20%`，但 `match@5=39.51%`；只差 `+2.07 pp` 才达到 CrystaLLM GT-SG K5 +5pp。
- 如果 structured validation K50 label 能比 E21/E26 更好地学习“正确候选前移到 top5”，则可能补上第二个达标指标。
- 风险：E14 在 structured validation 上可能也过易或过难；先检查 K50 match 与 positive rate，再决定是否应用到 full。

## 55. Experiment E44-E47 - structured validation K50 ranker full result

时间：2026-06-12 UTC

目的：

- 完成 E44 structured validation K50 pool。
- 用该 pool 训练 E45 K50-aware ranker。
- 在 E34 full MPTS-52 E14 K50 pool 上评估 E45 ranker 的 full K1/K5/K20。

E44 validation K50 pool：

| item | value |
|---|---:|
| validation rows | 1,024 |
| generated raw CIFs | 51,200 |
| postprocessed CIFs | 51,200 |
| match@50 | 52.25% |
| RMSE@50 | 0.1118 |
| parse_rate_candidate | 98.71% |
| valid_rate_candidate | 78.44% |

判断：

- structured validation pool 健康，明显修复 E37 raw validation pipeline 的 parse/match 失效问题。
- K50 match 与 test K50 上限同量级，可用于训练 K50-aware ranker。

E45 ranker training：

| item | value |
|---|---:|
| selected ids | 1,024 |
| ok labels | 1,012 |
| timeout labels | 12 |
| candidate records | 50,600 |
| positives | 17,012 |
| positive_rate | 33.62% |
| RF | 700 trees, max_depth=18, min_samples_leaf=4 |

E46 sanity64 on E34 K50：

| K | match | RMSE |
|---:|---:|---:|
| 1 | 54.69% | 0.1100 |
| 5 | 60.94% | 0.1198 |
| 20 | 65.62% | 0.1276 |

E47 full result：

| K | match | RMSE | parse_rate | valid_rate |
|---:|---:|---:|---:|---:|
| 1 | 32.07% | 0.1109 | 99.89% | 86.51% |
| 5 | 39.16% | 0.1163 | 99.82% | 85.27% |
| 20 | 45.07% | 0.1301 | 99.66% | 82.19% |

与 CrystaLLM GT-SG +5pp 目标线：

| metric | E47 | target | status |
|---|---:|---:|---|
| match@1 | 32.07% | 31.64% | pass |
| match@5 | 39.16% | 41.58% | fail |
| match@20 | 45.07% | 49.69% | fail |

判断：

- E45 structured-val K50 ranker 没有超过 E35/E21 full K5，也没有补上第二个达标指标。
- E46 sanity64 很强但不能代表 full；full 难点仍来自后续复杂样本分布。
- 下一步应在 validation labels 内做 group-heldout ranker/blend tuning，目标直接优化 validation heldout hit@5，而不是单个 RF 配置盲目应用到 test。
