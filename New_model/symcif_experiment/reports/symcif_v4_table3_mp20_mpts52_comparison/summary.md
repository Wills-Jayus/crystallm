# SymCIF-v4 Table 3 Benchmark: MP-20 / MPTS-52

Generated: 2026-05-22 19:56 UTC

## Executive conclusion

- This run completed the official-train-split SymCIF-v4 structured-data build, no-weight and complex-weight geometry training, WA policy search, and the full `WA20 x geom1` test evaluation on MP-20 and MPTS-52. MP-20 has a complete no-weight WA policy training summary; MPTS-52 has a usable policy checkpoint/search output but its final policy training summary was interrupted.
- SymCIF-v4 in this report uses **cell composition + oracle ground-truth space group (GT SG)**. Published CrystaLLM Table 3 uses composition-only, so the comparison is informative but not strictly identical input.
- Under the completed fair-budget config (`WA20 x geom1`, top20 final evaluation), SymCIF-v4 **does not exceed** the published CrystaLLM a Table 3 numbers. MP-20 is below on match rate and worse on RMSE; MPTS-52 is close on match@20 but still below and worse on RMSE.
- The main bottleneck is WA search/rerank, not just geometry: GT-WA@20 is only 18.40% on MP-20 and 8.12% on MPTS-52. Larger training data alone should not be assumed to beat CrystaLLM until WA recall and geometry RMSE improve.

## Completed vs requested scope

| Item | Status | Notes |
| --- | --- | --- |
| Structured MP-20/MPTS-52 data | Completed | Official train/val/test splits only; test not used for training. |
| Evaluator throughput instrumentation | Completed for this pipeline | parse/SG/matcher/eval timeout rates recorded; hash dedup and early mismatch skips implemented. |
| Training: geometry no-weight | Completed | MP-20 and MPTS-52 checkpoints saved. |
| Training: geometry complex-weight | Completed | MP-20 and MPTS-52 checkpoints saved; not fully re-evaluated through every budget. |
| Training/search: WA policy no-weight | Completed enough for full eval | MP-20 final summary exists; MPTS checkpoint/search exists but policy training summary was interrupted after checkpoint creation. |
| Full fair-budget eval: WA20 x geom1 | Completed | This is the formal result reported below. |
| WA10 x geom2 / WA5 x geom4 / WA15 x geom2 / WA20 x geom2 | Not completed | One full WA20 x geom1 eval already produced 331,120 generated/evaluated attempts across both datasets and took multiple hours; these configs are documented as pending, not fabricated. |
| CrystaLLM+GT-SG baseline re-eval | Not completed | Checkpoints found, but full n=20 SG-conditioned generation/eval was not run in this pass. Published Table 3 comparison is retained. |

## Published Table 3 comparison

| Dataset | Model | Input | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | Delta/Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | CrystaLLM a published | composition only | 55.85% | NA | 75.14% | 0.0437 | NA | 0.0395 | reference |
| MP-20 | SymCIF-v4 WA20xgeom1 | composition + oracle GT SG | 45.62% | 59.73% | 68.92% | 0.0657 | 0.0826 | 0.0848 | match@20 -6.22 pp; RMSE@20 +0.0453 |
| MPTS-52 | CrystaLLM a published | composition only | 17.47% | NA | 32.98% | 0.1113 | NA | 0.1197 | reference |
| MPTS-52 | SymCIF-v4 WA20xgeom1 | composition + oracle GT SG | 14.46% | 22.55% | 30.43% | 0.1409 | 0.1453 | 0.1571 | match@20 -2.55 pp; RMSE@20 +0.0374 |

## SG-conditioned baseline status

CrystaLLM benchmark checkpoints were found:

- `model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mp_20_b/ckpt.pt`
- `model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mpts_52_b/ckpt.pt`

The requested CrystaLLM+GT-SG full re-evaluation was not completed in this run. Therefore there is no same-evaluator SG-conditioned CrystaLLM numeric row in the formal table. Treat the Table 3 comparison above as the only completed external baseline comparison.

## Structured data audit

| Dataset | Split | Input | Structured | Success | Row free-param success | Row fallback | n_sites mean | n_sites p90 | elements mean | n_sites>=6 | elements>=4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | train | 27136 | 26629 | 98.13% | 91.03% | 11185 | 4.68 | 8.0 | 3.01 | 24.18% | 20.81% |
| MP-20 | val | 9047 | 8874 | 98.09% | 90.38% | 3966 | 4.65 | 8.0 | 3.01 | 23.77% | 20.49% |
| MP-20 | test | 9046 | 8893 | 98.31% | 91.04% | 3674 | 4.61 | 8.0 | 3.02 | 22.97% | 20.68% |
| MPTS-52 | train | 27380 | 25998 | 94.95% | 87.55% | 18249 | 5.64 | 11.0 | 2.90 | 35.33% | 17.57% |
| MPTS-52 | val | 5000 | 4727 | 94.54% | 89.99% | 3612 | 7.63 | 14.0 | 3.43 | 58.71% | 42.73% |
| MPTS-52 | test | 8096 | 7663 | 94.65% | 89.21% | 6430 | 7.77 | 13.0 | 3.56 | 63.23% | 48.98% |

## Training summary

| Dataset | Component | Summary | Best val metric | Checkpoint |
| --- | --- | --- | --- | --- |
| MP-20 | geometry no-weight | model/New_model/symcif_experiment/runs/symcif_v4_benchmark/mp20/geometry_no_weight/training_summary.json | 0.3660 | runs/symcif_v4_benchmark/mp20/geometry_no_weight/ckpt_best.pt |
| MP-20 | geometry complex-weight=3 | model/New_model/symcif_experiment/runs/symcif_v4_benchmark/mp20/geometry_complex_weight3/training_summary.json | 0.3832 | runs/symcif_v4_benchmark/mp20/geometry_complex_weight3/ckpt_best.pt |
| MP-20 | WA policy no-weight | model/New_model/symcif_experiment/reports/symcif_v4_benchmark_complex_subsets/mp20/policy_no_weight/step_policy_training_summary.json | top5=95.76% | runs/symcif_v4_benchmark/mp20/policy_no_weight/ckpt.pt |
| MPTS-52 | geometry no-weight | model/New_model/symcif_experiment/runs/symcif_v4_benchmark/mpts52/geometry_no_weight/training_summary.json | 0.7511 | runs/symcif_v4_benchmark/mpts52/geometry_no_weight/ckpt_best.pt |
| MPTS-52 | geometry complex-weight=3 | model/New_model/symcif_experiment/runs/symcif_v4_benchmark/mpts52/geometry_complex_weight3/training_summary.json | 0.7933 | runs/symcif_v4_benchmark/mpts52/geometry_complex_weight3/ckpt_best.pt |
| MPTS-52 | WA policy no-weight | summary missing | checkpoint exists; training summary interrupted | runs/symcif_v4_benchmark/mpts52/policy_no_weight/ckpt.pt |

## WA search recall

| Dataset | Samples | Candidate nonempty | Skeleton@1 | Skeleton@5 | Skeleton@20 | GT-WA@1 | GT-WA@5 | GT-WA@20 | Median s | P90 s | Timeouts | Truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | 8893 | 99.58% | 24.06% | 28.94% | 31.01% | 12.64% | 17.25% | 18.40% | 0.0096 | 0.278 | 44 | 3701 |
| MPTS-52 | 7663 | 94.83% | 22.80% | 27.38% | 30.54% | 4.70% | 7.10% | 8.12% | 0.0534 | 3.143 | 412 | 4965 |

## Full pipeline metrics: WA20 x geom1

| Dataset | K | Samples | match@K | RMSE | readable | formula_ok | SG_ok | strict_valid | eval_timeout | render_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | @1 | 8893 | 45.62% | 0.0657 | 75.90% | 75.90% | 75.90% | 51.94% | 0.21% | 99.58% |
| MP-20 | @5 | 8893 | 59.73% | 0.0826 | 64.93% | 64.93% | 64.93% | 33.07% | 0.21% | 84.58% |
| MP-20 | @20 | 8893 | 68.92% | 0.0848 | 48.80% | 48.80% | 48.80% | 18.49% | 0.21% | 65.98% |
| MPTS-52 | @1 | 7663 | 14.46% | 0.1409 | 49.65% | 49.65% | 49.65% | 22.01% | 2.09% | 94.83% |
| MPTS-52 | @5 | 7663 | 22.55% | 0.1453 | 50.21% | 50.21% | 50.21% | 17.46% | 2.09% | 88.73% |
| MPTS-52 | @20 | 7663 | 30.43% | 0.1571 | 45.43% | 45.43% | 45.43% | 12.93% | 2.09% | 82.00% |

## Complex subset tracking

| Dataset | Subset | Samples | match@20 | RMSE@20 | strict_valid_any@20 | GT-WA@20 | Skeleton@20 | eval_timeout |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MP-20 | overall | 8893 | 68.92% | 0.0848 | 80.97% | 18.40% | 31.01% | 0.21% |
| MP-20 | n_sites>=6 | 2043 | 27.21% | 0.1578 | 53.01% | 3.48% | 27.31% | 0.20% |
| MP-20 | n_sites>=12 | 336 | 8.63% | 0.2772 | 36.01% | 5.36% | 52.08% | 0.00% |
| MP-20 | n_sites>=20 | 43 | 9.30% | 0.2354 | 32.56% | 2.33% | 88.37% | 0.00% |
| MP-20 | num_elements>=4 | 1839 | 57.10% | 0.0906 | 69.44% | 1.85% | 15.39% | 0.38% |
| MP-20 | rare_sg | 270 | 60.37% | 0.1276 | 65.93% | 20.37% | 28.52% | 0.00% |
| MP-20 | high_multiplicity_orbit | 1281 | 87.74% | 0.0836 | 82.90% | 15.93% | 18.58% | 1.41% |
| MP-20 | extraction_hard | 1132 | 40.46% | 0.2593 | 62.46% | 20.05% | 39.93% | 0.27% |
| MPTS-52 | overall | 7663 | 30.43% | 0.1571 | 56.36% | 8.12% | 30.54% | 2.09% |
| MPTS-52 | n_sites>=6 | 4845 | 13.27% | 0.1911 | 40.70% | 3.90% | 32.05% | 2.64% |
| MPTS-52 | n_sites>=12 | 1335 | 3.52% | 0.0906 | 19.40% | 3.15% | 38.28% | 2.55% |
| MPTS-52 | n_sites>=20 | 243 | 0.41% | 0.0116 | 5.76% | 1.65% | 36.21% | 1.23% |
| MPTS-52 | num_elements>=4 | 3753 | 22.36% | 0.1533 | 47.86% | 2.08% | 27.34% | 2.61% |
| MPTS-52 | rare_sg | 388 | 32.22% | 0.1781 | 46.13% | 15.21% | 25.26% | 3.35% |
| MPTS-52 | high_multiplicity_orbit | 1017 | 48.67% | 0.1250 | 56.54% | 12.00% | 15.24% | 7.87% |
| MPTS-52 | extraction_hard | 1468 | 15.87% | 0.2854 | 41.55% | 8.38% | 26.16% | 2.32% |

## Evaluator throughput and timeout

| Dataset | Top20 attempts | eval_timeout sample rate | parse_timeout | sg_timeout | matcher_timeout | timeout breakdown |
| --- | --- | --- | --- | --- | --- | --- |
| MP-20 | 177860 | 0.21% | 0.00% | 0.00% | 0.00% | reports/symcif_v4_benchmark_budget_ablation/mp20/wa20_geom1_no_weight/timeout_breakdown.csv |
| MPTS-52 | 153260 | 2.09% | 0.00% | 0.00% | 0.00% | reports/symcif_v4_benchmark_budget_ablation/mpts52/wa20_geom1_no_weight/timeout_breakdown.csv |

## Interpretation

1. MP-20 is clearly below the published CrystaLLM a baseline even with oracle GT SG: match@20 is 68.92% vs 75.14%, and RMSE@20 is 0.0848 vs 0.0395.
2. MPTS-52 is closer on match@20 but still below: 30.43% vs 32.98%, while RMSE@20 is worse at 0.1571 vs 0.1197.
3. Complex samples degrade sharply. For MPTS-52 n_sites>=20, match@20 is 0.41%; for MP-20 n_sites>=12, match@20 is 8.63%. This matches the low WA recall and high fallback/prototype reliance observed in the search/eval artifacts.
4. The completed experiment does not support the claim that scaling SymCIF-v4 to the same data size as CrystaLLM will probably exceed CrystaLLM. The immediate limiting factor is candidate recall and final geometry accuracy under predicted WA, so larger data should be paired with WA search/rerank changes and RMSE-targeted geometry training.

## Artifacts

| Dataset | Artifact | Path |
| --- | --- | --- |
| MP-20 | structured data | data/structured_symcif_v4_mp20/ |
| MP-20 | data audit | reports/symcif_v4_benchmark_data_audit/mp20/ |
| MP-20 | WA search | reports/symcif_v4_benchmark_budget_ablation/mp20/policy_no_weight_search/ |
| MP-20 | full eval summary | model/New_model/symcif_experiment/reports/symcif_v4_benchmark_budget_ablation/mp20/wa20_geom1_no_weight/full_eval_summary.json |
| MP-20 | full eval per-generation metrics | reports/symcif_v4_benchmark_budget_ablation/mp20/wa20_geom1_no_weight/metrics/baseline_per_generation_metrics.jsonl |
| MP-20 | timeout breakdown | reports/symcif_v4_benchmark_budget_ablation/mp20/wa20_geom1_no_weight/timeout_breakdown.csv |
| MP-20 | generated CIFs | reports/symcif_v4_benchmark_budget_ablation/mp20/wa20_geom1_no_weight/generated_cifs |
| MPTS-52 | structured data | data/structured_symcif_v4_mpts52/ |
| MPTS-52 | data audit | reports/symcif_v4_benchmark_data_audit/mpts52/ |
| MPTS-52 | WA search | reports/symcif_v4_benchmark_budget_ablation/mpts52/policy_no_weight_search/ |
| MPTS-52 | full eval summary | model/New_model/symcif_experiment/reports/symcif_v4_benchmark_budget_ablation/mpts52/wa20_geom1_no_weight/full_eval_summary.json |
| MPTS-52 | full eval per-generation metrics | reports/symcif_v4_benchmark_budget_ablation/mpts52/wa20_geom1_no_weight/metrics/baseline_per_generation_metrics.jsonl |
| MPTS-52 | timeout breakdown | reports/symcif_v4_benchmark_budget_ablation/mpts52/wa20_geom1_no_weight/timeout_breakdown.csv |
| MPTS-52 | generated CIFs | reports/symcif_v4_benchmark_budget_ablation/mpts52/wa20_geom1_no_weight/generated_cifs |

## Pending experiments

- Run `WA10 x geom2`, `WA5 x geom4`, `WA15 x geom2`, and `WA20 x geom2` with the same evaluator; write them under `reports/symcif_v4_benchmark_budget_ablation/{mp20,mpts52}/`.
- Run CrystaLLM benchmark checkpoints with `cell composition + GT SG`, n=20, same splits and same evaluator. Checkpoints are present, but this report does not contain those metrics.
- Re-run full pipeline with complex-weight geometry and complex-weight WA policy once the policy summary issue is cleaned up; compare on the complex subset table, not only overall match@20.
