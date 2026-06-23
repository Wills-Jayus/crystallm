# SymCIF-v2 + Constrained Decoding

本目录记录 `/data/users/xsw/autodlmini/model/New_model/GPT-Prompt/2026520_02.md` 对应实验。

## 1. 构建 SymCIF-v2 数据

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/build_symcif_v2_corpus.py \
  --source-mode symcif-v1 \
  --out-dir data/symcif_v2 \
  --reports-dir reports \
  --train-size 5000 \
  --val-size 500 \
  --test-size 500
```

输出：

- `data/symcif_v2/train.txt`
- `data/symcif_v2/val.txt`
- `data/symcif_v2/test.txt`
- `reports/symcif_v2_conversion_report.json`

## 2. Tokenize

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/tokenize_symcif_v2.py \
  --corpus-dir data/symcif_v2 \
  --out-dir data/tokens_symcif_v2 \
  --artifacts-dir artifacts \
  --reports-dir reports
```

输出：

- `data/tokens_symcif_v2/`
- `artifacts/tokenizer_symcif_v2.json`
- `reports/tokenizer_symcif_v2_report.json`

当前 tokenizer 结论：

- vocab size: 405
- unk rate: 0.00%
- element coverage: 86/86
- Wyckoff letter coverage: 22/22
- supports constrained decoding mask: true

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python external/CrystaLLM_code/bin/train.py \
  --config configs/exp_symcif_v2.yaml
```

输出：

- `runs/exp_symcif_v2/ckpt_best.pt`
- `runs/exp_symcif_v2/ckpt_last.pt`
- `runs/exp_symcif_v2/ckpt.pt`

当前 best checkpoint：

- step: 750
- best val loss: 0.655507

## 4. Constrained Sampling

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/sample_symcif_v2_constrained.py \
  --out-dir eval_runs/symcif_v2_constrained_eval_t1_topk10_n20 \
  --test-limit 500 \
  --num-gens 20 \
  --seed 1337 \
  --devices cuda:0,cuda:1 \
  --dtype bfloat16 \
  --temperature 1.0 \
  --top-k 10
```

Constrained sampler 只使用 prompt 中允许的 formula 和 space group，以及预构建 Wyckoff lookup。不使用 GT CIF、GT Wyckoff、GT 坐标或 GT lattice。

## 5. Raw / Constrained 评估

Raw generation:

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes symcif_v2_raw \
  --out-dir eval_runs/symcif_v2_constrained_eval_t1_topk10_n20 \
  --temperature 1.0 \
  --top-k 10 \
  --num-gens 20 \
  --seed 1337 \
  --max-new-tokens 2048 \
  --devices cuda:0,cuda:1 \
  --dtype bfloat16 \
  --test-limit 500 \
  --skip-evaluation \
  --overwrite \
  --date-tag 20260520_symcif_v2_full
```

Evaluation:

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes symcif_v2_raw symcif_v2_constrained \
  --out-dir eval_runs/symcif_v2_constrained_eval_t1_topk10_n20 \
  --generation-dir eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/generations \
  --skip-generation \
  --test-limit 500 \
  --num-gens 20 \
  --eval-workers 16 \
  --sample-timeout-seconds 60 \
  --bond-timeout-seconds 8 \
  --valid-timeout-seconds 8 \
  --match-timeout-seconds 8 \
  --max-match-sites 96 \
  --max-eval-sites 96 \
  --date-tag 20260520_symcif_v2_full
```

输出：

- `eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/summary.csv`
- `eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/summary.json`
- `eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/metrics/`
- `eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/generations/`
- `eval_runs/symcif_v2_constrained_eval_t1_topk10_n20/standard_cifs/`

## 6. 当前结果

| method | match@20 | formula_ok | readable | valid | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_minprompt | 48.20% | 92.99% | 92.43% | 23.04% | 0.1182 |
| symcif_v1 raw | 35.20% | 22.62% | 74.56% | 23.60% | 0.1134 |
| symcif_v1 extended rejection | 43.80% | - | - | - | 0.1322 |
| symcif_v2_raw | 36.40% | 19.19% | 75.38% | 25.79% | 0.1280 |
| symcif_v2_constrained | 49.20% | 89.22% | 87.60% | 26.64% | 0.1358 |

结论：

- 已超过 `symcif_v1 extended rejection`：`49.20% > 43.80%`。
- 已边界超过 `baseline_minprompt`：`49.20% > 48.20%`。
- Formula closure 明显改善：`89.22% vs symcif_v1 raw 22.62%`。
- 当前主要不足是 RMSE 和晶胞/几何质量。

## 7. 2026-05-21 match@5 快速迭代

### fixed_cell

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/sample_symcif_v2_constrained.py \
  --out-dir eval_runs/symcif_v2_fixed_cell_match5_20260521 \
  --test-limit 500 \
  --num-gens 5 \
  --seed 1337 \
  --devices cuda:0,cuda:1 \
  --dtype bfloat16 \
  --temperature 1.0 \
  --top-k 10 \
  --mode-name symcif_v2_constrained_fixed_cell \
  --overwrite
```

评估：

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes baseline_minprompt symcif_v2_constrained_old symcif_v2_constrained_fixed_cell \
  --out-dir eval_runs/symcif_v2_fixed_cell_match5_20260521 \
  --skip-generation \
  --test-limit 500 \
  --num-gens 5 \
  --eval-workers 64 \
  --sample-timeout-seconds 240 \
  --bond-timeout-seconds 10 \
  --valid-timeout-seconds 10 \
  --match-timeout-seconds 10 \
  --max-match-sites 96 \
  --max-eval-sites 96 \
  --date-tag 20260521_fixed_cell_match5
```

结果：

| method | match@1 | match@5 | RMSE | formula_ok | sg_ok | readable | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_minprompt | 22.40% | 38.60% | 0.1055 | 93.08% | 92.88% | 92.48% | 22.48% |
| symcif_v2_constrained_old | 24.20% | 41.00% | 0.1518 | 93.96% | 90.40% | 92.24% | 27.76% |
| symcif_v2_constrained_fixed_cell | 26.20% | 41.80% | 0.1445 | 93.96% | 90.16% | 92.40% | 28.40% |

fixed_cell 修复了 angle hardcode bug，monoclinic beta 不再全部为 100，triclinic alpha/gamma 不再全部为 90。该版本超过 old constrained match@5，且 RMSE 更好，应作为后续 cell 生成基础。

### skeleton_beam

运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/sample_symcif_v2_skeleton_beam.py \
  --out-dir eval_runs/symcif_v2_skeleton_beam_match5_20260521 \
  --test-limit 500 \
  --num-gens 5 \
  --seed 1337 \
  --devices cuda:0,cuda:1 \
  --dtype bfloat16 \
  --temperature 1.0 \
  --top-k 10 \
  --beam-size 32 \
  --top-skeletons 5 \
  --max-sites 64 \
  --max-expansions-per-beam 16 \
  --overwrite
```

评估：

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes baseline_minprompt symcif_v2_constrained_old symcif_v2_constrained_fixed_cell symcif_v2_constrained_skeleton_beam \
  --out-dir eval_runs/symcif_v2_skeleton_beam_match5_20260521 \
  --skip-generation \
  --test-limit 500 \
  --num-gens 5 \
  --eval-workers 64 \
  --sample-timeout-seconds 240 \
  --bond-timeout-seconds 10 \
  --valid-timeout-seconds 10 \
  --match-timeout-seconds 10 \
  --max-match-sites 96 \
  --max-eval-sites 96 \
  --date-tag 20260521_skeleton_beam_match5
```

结果：

| method | match@1 | match@5 | RMSE | formula_ok | sg_ok | readable | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| symcif_v2_constrained_old | 24.20% | 41.00% | 0.1518 | 93.96% | 90.40% | 92.24% | 27.76% |
| symcif_v2_constrained_fixed_cell | 26.20% | 41.80% | 0.1445 | 93.96% | 90.16% | 92.40% | 28.40% |
| symcif_v2_constrained_skeleton_beam | 22.20% | 43.00% | 0.1549 | 94.20% | 89.56% | 92.96% | 24.36% |

skeleton_beam 超过 old constrained match@5，也达到 43.00%。但 match@1、conditional match、RMSE 和 valid 退化，说明当前 beam ranking 不够稳。它值得作为探索性方案补跑 match@20，尤其可用于 rank 2-20 扩展；正式默认方案建议先做 hybrid：保留 fixed_cell 的高质量 rank-1，再引入 beam skeleton 扩展后续候选。

报告：

- `Log_GPT/symcif_v2_fixed_cell_match5_report.md`
- `Log_GPT/symcif_v2_skeleton_beam_match5_report.md`

## 8. 2026-05-21 诊断、Oracle 与 Skeleton-Rank

本轮对应 `/data/users/xsw/autodlmini/model/New_model/GPT-Prompt/2026521_03.md`。原则：

- 默认主线仍为 `symcif_v2_constrained_fixed_cell`。
- `oracle` 诊断可以使用 GT skeleton / assignment / cell，但不作为正式生成结果。
- 正式 `skeleton_rank` 不使用 GT，不根据 StructureMatcher 反筛。

### 8.1 Failure diagnosis

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/diagnose_symcif_v2_failures.py \
  --compute-rank \
  --overwrite-rank \
  --test-limit 500 \
  --n 5 \
  --out-dir eval_runs/symcif_v2_diagnosis_20260521 \
  --devices cuda:0,cuda:1 \
  --rank-beam-size 64 \
  --rank-candidate-limit 128 \
  --rank-max-expansions-per-beam 24
```

输出：

- `eval_runs/symcif_v2_diagnosis_20260521/diagnosis_summary.csv`
- `eval_runs/symcif_v2_diagnosis_20260521/per_sample_diagnosis.csv`
- `eval_runs/symcif_v2_diagnosis_20260521/per_generation_diagnosis.csv`
- `Log_GPT/symcif_v2_diagnosis_report.md`

关键诊断：

- fixed_cell: skeleton@5 = 70.20%，assignment@5 = 61.00%。
- GT skeleton missing rate = 6.20%，GT skeleton rank <= 5 = 65.20%。
- 主要不是 lookup/枚举大面积缺失，而是当前 SymCIF-v2 LM 对 skeleton / assignment 的排序不够稳。

### 8.2 Oracle diagnosis

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_symcif_v2_oracle_diagnosis.py \
  --test-limit 500 \
  --num-gens 5 \
  --devices cuda:0,cuda:1 \
  --out-dir eval_runs/symcif_v2_oracle_diagnosis_20260521 \
  --overwrite

/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes symcif_v2_oracle_gt_skeleton symcif_v2_oracle_gt_skeleton_gt_cell \
  --out-dir eval_runs/symcif_v2_oracle_diagnosis_20260521 \
  --skip-generation \
  --num-gens 5 \
  --test-limit 500 \
  --eval-workers 64 \
  --max-match-sites 96 \
  --max-eval-sites 96 \
  --date-tag 20260521_oracle

/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_symcif_v2_oracle_diagnosis.py \
  --test-limit 500 \
  --num-gens 5 \
  --out-dir eval_runs/symcif_v2_oracle_diagnosis_20260521 \
  --summarize-only
```

结果：

| oracle | match@1 | match@5 | RMSE |
| --- | ---: | ---: | ---: |
| GT skeleton + generated coord/cell | 40.40% | 51.60% | 0.1534 |
| GT skeleton + GT cell + generated coord | 45.00% | 58.60% | 0.1645 |

Oracle A 到 B 提升 7.00 个百分点，说明 cell 是重要拖累；但 GT skeleton + GT cell 仍只有 58.60%，说明 free coordinate 也是明显瓶颈。

### 8.3 Formal skeleton-rank generation

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/sample_symcif_v2_skeleton_rank.py \
  --test-limit 500 \
  --num-gens 5 \
  --devices cuda:0,cuda:1 \
  --out-dir eval_runs/symcif_v2_skeleton_rank_match5_20260521 \
  --beam-size 64 \
  --candidate-limit 128 \
  --top-skeletons 5 \
  --max-expansions-per-beam 24 \
  --temperature 1.0 \
  --top-k 10 \
  --overwrite

/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python scripts/run_generation_eval.py \
  --modes symcif_v2_constrained_skeleton_rank \
  --out-dir eval_runs/symcif_v2_skeleton_rank_match5_20260521 \
  --skip-generation \
  --num-gens 5 \
  --test-limit 500 \
  --eval-workers 64 \
  --max-match-sites 96 \
  --max-eval-sites 96 \
  --date-tag 20260521_skeleton_rank_match5
```

最终 match@5 对比：

| method | match@1 | match@5 | RMSE | formula_ok | sg_ok | readable | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_minprompt | 22.40% | 38.60% | 0.1055 | 93.08% | 92.88% | 92.48% | 22.48% |
| fixed_cell | 26.20% | 41.80% | 0.1445 | 93.96% | 90.16% | 92.40% | 28.40% |
| skeleton_beam_old | 22.20% | 43.00% | 0.1549 | 94.20% | 89.56% | 92.96% | 24.36% |
| skeleton_rank | 21.20% | 40.20% | 0.1545 | 91.28% | 86.24% | 92.04% | 25.12% |

Skeleton-rank 判定：不成功。它没有超过 fixed_cell，skeleton@5 从 70.20% 降到 65.40%，skeleton_wrong 从 827 增到 1126，RMSE 也变差。

### 8.4 当前失败主因与 v3 判断

当前失败主因：

1. 正式生成中 top skeleton / assignment 排序不稳，是最大瓶颈。
2. GT skeleton 候选覆盖缺失只有 6.20%，lookup/setting/converter 不是主要问题。
3. Oracle 表明即使 skeleton 已知，cell 和 free coordinate 仍会限制 match 上限。

建议进入 SymCIF-v3 设计，但本轮不实现。方向为：

```text
SG + formula -> Wyckoff skeleton -> element assignment -> free coords -> cell
```

也就是更接近 CrystalFormer / WyckoffTransformer 的 W -> A -> X -> L 顺序。

综合报告：

- `Log_GPT/symcif_v2_skeleton_rank_match5_report.md`
- `Log_GPT/round_20260521_03_diagnosis_oracle_rank/comprehensive_experiment_report.md`
