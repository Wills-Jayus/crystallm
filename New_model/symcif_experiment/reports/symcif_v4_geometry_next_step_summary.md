# symcif v4 geometry next-step decision summary

## 1. 总结论

本轮实验完成了 renderer/evaluator 修复、GT-WA geometry model、predicted-WA full pipeline 接入、复杂子集 oversampling 对照。结论是：v4 路线值得继续，而且已经具备进入更大数据规模正式实验的条件，但必须把 geometry diversity 和复杂子集加权作为主配置的一部分。

最关键的结果是 full pipeline best 达到 match@20=62.8%，高于同条件 CrystaLLM baseline 的 44.6%/47.8%，也高于修复前 v4 current 的 56.6%。GT-WA geometry 从 56.4% 提升到 70.4%，说明 geometry/free_params/lattice 子模块是有效突破点。

## 2. 核心指标

| experiment | match@1 | match@5 | match@20 | RMSE@20 | strict_valid@20 | artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CrystaLLM baseline same-as-v4 | 16.2% | 33.8% | 44.6% | 0.1164 | 19.7% | eval_runs/baseline_reeval_same_as_v4_20260522 |
| CrystaLLM baseline_minprompt same-as-v4 | 22.4% | 38.6% | 47.8% | 0.1188 | 23.0% | eval_runs/baseline_minprompt_reeval_same_as_v4_20260522 |
| v4 current before fix | 34.8% | 52.0% | 56.6% | 0.1030 | 0.0% | reports/symcif_v4_full_eval_current/ |
| v4 current after renderer fix | 34.4% | 51.4% | 55.6% | 0.1003 | 14.7% | reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/ |
| GT-WA + old geometry | 56.4% | 56.4% | 56.4% | 0.0973 | 0.0% | reports/symcif_v4_geometry_bottleneck/ |
| GT-WA + geometry model no-over | 54.2% | 65.8% | 70.2% | 0.0835 | 44.1% | reports/symcif_v4_geometry_model_gtwa/no_oversampling/ |
| GT-WA + geometry model over | 54.2% | 65.8% | 70.4% | 0.0833 | 44.0% | reports/symcif_v4_geometry_model_gtwa/oversampling/ |
| full geometry no-over WA20x1 | 36.8% | 54.0% | 59.4% | 0.0924 | 18.4% | reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/ |
| full geometry over WA20x1 | 36.8% | 54.2% | 59.2% | 0.0913 | 18.3% | reports/symcif_v4_full_pipeline_geometry_model/oversampling/ |
| full geometry no-over WA15x2 best | 36.8% | 54.4% | 62.8% | 0.0887 | 23.0% | reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/ |
| WA upper-bound | 52.4% | 75.2% | 82.4% | 0.0157 | 0.0% | reports/symcif_v4_wa_upper_bound/ |

## 3. 判定

- renderer/evaluator: 已修。`valid/strict_valid` 不再全 0；但 timeout 增加，需要后续优化 evaluator 性能。
- geometry model: 成功。GT-WA match@20 从 56.4% 到 70.2%/70.4%，达到原计划 70-75% 区间下沿。
- full pipeline: 成功但有条件。`WA20 x geom1` 只有 59.2%-59.4%；`WA15 x geom2` 达到 62.8%，说明 top20 输出预算必须分配给 geometry 多样性。
- complex subset: 仍是主风险。`n_sites>=6` 在 GT-WA 下从 17.7% 到 32.0%，但 full best 只有 19.4%，距离 WA upper-bound 61.1% 还很远。
- oversampling: 当前小规模只提供局部收益。建议保留加权策略，但不要把它视为单独解法。

## 4. 下一步建议

1. 进入大规模训练 pilot：用 CrystaLLM-small 同量级数据、2M 结构数据或 MPTS-52 数据训练 v4 geometry/WA pipeline，保持同 evaluator 对比。
2. 固定两个 full-pipeline 配置一起跑：`WA20 x geom1` 用于公平单 geometry 对照，`WA15 x geom2` 用于最佳 end-to-end 结果。
3. 数据策略保留复杂样本加权：重点覆盖 `n_sites>=6`、`num_elements>=4`、`SG=65/71/127`，并记录每个子集的 WA hit、GT-WA geometry、full pipeline 三段损失。
4. 优化 evaluator/renderer throughput：修复后 `eval_timeout@20` 对 current full 到 9.0%，会干扰小幅增益判断。
5. 保留 CrystaLLM baseline/minprompt 作为同条件 control；当前 best 已超过 baseline，但大数据训练能否超过 CrystaLLM 原模型仍需要正式同数据、同 prompt/evaluator 验证。

## 5. 分报告

- renderer/evaluator fix: `reports/symcif_v4_renderer_evaluator_fix/summary.md`
- GT-WA geometry model: `reports/symcif_v4_geometry_model_gtwa/summary.md`
- full pipeline geometry model: `reports/symcif_v4_full_pipeline_geometry_model/summary.md`
- complex subset oversampling: `reports/symcif_v4_complex_subset_oversampling/summary.md`
