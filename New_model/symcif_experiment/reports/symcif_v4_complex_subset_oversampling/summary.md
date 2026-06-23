# symcif v4 complex subset oversampling 实验报告

## 1. 结论

复杂子集的主要瓶颈仍是 geometry，而不是 WA：`n_sites>=6` 的当前 WA hit@20 有 72.6%，WA upper-bound match@20 有 61.1%，但原 geometry bottleneck 只有 17.7%。新 geometry model 在 GT-WA 下把该子集提升到 31.4%/32.0%，证明方向有效，但 full pipeline 仍只有 19.4%。

oversampling 不是单独的充分解。它对 `n_sites>=6` 从 31.4% 到 32.0%，对 `SG=65` 从 75.0% 到 83.3%，但对 `num_elements>=4` 从 56.8% 降到 55.7%。更合理的结论是：保留复杂样本加权，但需要更大训练集和更多 geometry variants，而不是仅靠当前小规模 oversampling。

## 2. Subset 对比

| subset | samples | current match@20 | current WA hit@20 | WA upper match@20 | old GT-WA geometry | GT-WA no-over | GT-WA over | full best | full best WA hit | full best RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 500 | 56.6% | 86.6% | 82.4% | 56.4% | 70.2% | 70.4% | 62.8% | 85.2% | 0.0887 |
| n_sites>=6 | 175 | 13.1% | 72.6% | 61.1% | 17.7% | 31.4% | 32.0% | 19.4% | 70.9% | 0.2010 |
| num_elements>=4 | 192 | 47.9% | 83.9% | 77.1% | 47.9% | 56.8% | 55.7% | 52.1% | 82.3% | 0.0736 |
| SG=2 | 28 | 14.3% | 89.3% | 67.9% | 10.7% | 32.1% | 32.1% | 25.0% | 89.3% | 0.1676 |
| SG=65 | 12 | 41.7% | 33.3% | 50.0% | 50.0% | 75.0% | 83.3% | 41.7% | 33.3% | 0.1880 |
| SG=71 | 9 | 44.4% | 44.4% | 44.4% | 77.8% | 100.0% | 100.0% | 44.4% | 44.4% | 0.0170 |
| SG=127 | 8 | 75.0% | 62.5% | 62.5% | 75.0% | 87.5% | 87.5% | 75.0% | 62.5% | 0.1170 |

## 3. Oversampling effect under GT-WA

| subset | no-over match@20 | over match@20 | delta | no-over RMSE | over RMSE | no-over valid | over valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 70.2% | 70.4% | +0.2pt | 0.0835 | 0.0833 | 44.2% | 44.1% |
| n_sites>=6 | 31.4% | 32.0% | +0.6pt | 0.2212 | 0.2280 | 15.3% | 15.5% |
| num_elements>=4 | 56.8% | 55.7% | -1.0pt | 0.0655 | 0.0572 | 33.1% | 32.8% |
| SG=2 | 32.1% | 32.1% | +0.0pt | 0.2095 | 0.2095 | 27.7% | 27.7% |
| SG=65 | 75.0% | 83.3% | +8.3pt | 0.1351 | 0.1539 | 32.5% | 32.9% |
| SG=71 | 100.0% | 100.0% | +0.0pt | 0.0509 | 0.0499 | 55.6% | 55.6% |
| SG=127 | 87.5% | 87.5% | +0.0pt | 0.1735 | 0.1642 | 32.5% | 33.1% |

## 4. Artifacts

- comparison CSV: `reports/symcif_v4_complex_subset_oversampling/comparison_rows.csv`
- GT-WA no-over: `reports/symcif_v4_geometry_model_gtwa/no_oversampling/full_eval_breakdown.csv`
- GT-WA oversampling: `reports/symcif_v4_geometry_model_gtwa/oversampling/full_eval_breakdown.csv`
- full best: `reports/symcif_v4_full_pipeline_geometry_model/no_oversampling_wa15_geom2/full_eval_breakdown.csv`
