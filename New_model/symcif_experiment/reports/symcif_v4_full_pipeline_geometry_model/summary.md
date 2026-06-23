# symcif v4 full pipeline + geometry model 实验报告

## 1. 结论

接入 geometry model 后，full pipeline 的最优配置达到 match@20=62.8%，超过当前 v4 full 的 56.6%，也超过 60% 的阶段目标。这个最优来自 `WA top15 + 每个 WA 最多 2 个 geometry variant` 的输出预算。

如果只做 `WA top20 x 每个 WA 1 个 geometry`，match@20 为 59.4%/59.2%，还没稳定越过 60%。因此 full pipeline 的有效增益不只是单点 geometry 质量，而是需要在 top20 预算里保留 geometry diversity。

## 2. Full pipeline 对比

| run | k | match | RMSE | valid | strict_valid | strict_any | WA hit | skeleton hit | eval_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4 current before renderer fix | @1 | 34.8% | 0.1149 | 0.0% | 0.0% | 0.0% | 49.6% | 63.2% | 3.0% |
| v4 current before renderer fix | @5 | 52.0% | 0.1043 | 0.0% | 0.0% | 0.0% | 77.2% | 82.0% | 3.0% |
| v4 current before renderer fix | @20 | 56.6% | 0.1030 | 0.0% | 0.0% | 0.0% | 86.6% | 89.4% | 3.0% |
| v4 current after renderer fix | @1 | 34.4% | 0.1121 | 47.6% | 47.6% | 47.6% | 49.6% | 63.2% | 9.0% |
| v4 current after renderer fix | @5 | 51.4% | 0.1038 | 30.6% | 30.6% | 64.6% | 77.2% | 82.0% | 9.0% |
| v4 current after renderer fix | @20 | 55.6% | 0.1003 | 14.7% | 14.7% | 68.2% | 86.6% | 89.4% | 9.0% |
| WA upper-bound | @1 | 52.4% | 0.0433 | 0.0% | 0.0% | 0.0% | 49.6% | 63.2% | 2.4% |
| WA upper-bound | @5 | 75.2% | 0.0222 | 0.0% | 0.0% | 0.0% | 77.2% | 82.0% | 2.4% |
| WA upper-bound | @20 | 82.4% | 0.0157 | 0.0% | 0.0% | 0.0% | 86.6% | 89.4% | 2.4% |
| geometry no-over, WA20 x geom1 | @1 | 36.8% | 0.1039 | 49.6% | 49.6% | 49.6% | 49.6% | 63.2% | 0.4% |
| geometry no-over, WA20 x geom1 | @5 | 54.0% | 0.0815 | 33.6% | 33.6% | 67.8% | 77.2% | 82.0% | 0.4% |
| geometry no-over, WA20 x geom1 | @20 | 59.4% | 0.0924 | 18.4% | 18.4% | 74.8% | 86.6% | 89.4% | 0.4% |
| geometry oversampling, WA20 x geom1 | @1 | 36.8% | 0.1036 | 49.0% | 49.0% | 49.0% | 49.6% | 63.2% | 0.4% |
| geometry oversampling, WA20 x geom1 | @5 | 54.2% | 0.0827 | 33.3% | 33.3% | 67.4% | 77.2% | 82.0% | 0.4% |
| geometry oversampling, WA20 x geom1 | @20 | 59.2% | 0.0913 | 18.3% | 18.3% | 73.6% | 86.6% | 89.4% | 0.4% |
| geometry no-over, WA15 x geom2 | @1 | 36.8% | 0.1039 | 49.6% | 49.4% | 49.4% | 49.6% | 63.2% | 0.6% |
| geometry no-over, WA15 x geom2 | @5 | 54.4% | 0.0824 | 34.8% | 34.6% | 68.2% | 77.2% | 82.0% | 0.6% |
| geometry no-over, WA15 x geom2 | @20 | 62.8% | 0.0887 | 23.1% | 23.0% | 77.6% | 85.2% | 88.2% | 0.6% |

## 3. 与上限的距离

- 当前最佳 full pipeline：match@20=62.8%，RMSE@20=0.0887。
- WA upper-bound：match@20=82.4%，RMSE@20=0.0157。
- 剩余差距约 19.6 个 match 点，主要来自复杂结构 geometry/free_params 仍弱，以及 `WA15 x geom2` 预算下 WA hit@20 从 86.6% 降到 85.2%。

## 4. Artifacts

- no-over WA20 x geom1: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling/`
- oversampling WA20 x geom1: `reports/symcif_v4_full_pipeline_geometry_model/oversampling/`
- best WA15 x geom2: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/`
- current fixed reference: `reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/`
