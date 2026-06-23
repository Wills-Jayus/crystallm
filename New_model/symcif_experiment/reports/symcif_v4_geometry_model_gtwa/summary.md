# symcif v4 GT-WA geometry model 实验报告

## 1. 结论

GT-WA 条件下，新的 orbit-level geometry model 明显突破原 geometry bottleneck：match@20 从 56.4% 提升到 70.2%/70.4%，RMSE@20 从 0.0973 降到约 0.0833。说明 WA 已知时，lattice/free_params 的预测和原型补全确实是有效增益来源。

oversampling 的整体收益很小：overall match@20 只从 70.2% 到 70.4%；但在 `n_sites>=6`、`SG=65` 等目标复杂子集有局部改善，仍值得在更大数据规模下继续验证。

## 2. 训练摘要

| variant | complex_weight | best_epoch | best_val_loss | val_lattice_loss | val_coord_loss | checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no oversampling | 1.0 | 60 | 0.4099 | 0.2954 | 0.0573 | `runs/symcif_v4_geometry_model_no_oversampling/ckpt_best.pt` |
| complex oversampling | 3.0 | 60 | 0.4360 | 0.3201 | 0.0580 | `runs/symcif_v4_geometry_model_oversampling/ckpt_best.pt` |

## 3. GT-WA evaluation

| run | k | match | RMSE | readable | formula_ok | atom_count_ok | SG_ok | valid | strict_valid | bond_reasonable | bond_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT-WA + current geometry | @1 | 56.4% | 0.0973 | 99.6% | 99.6% | 98.6% | 99.6% | 0.0% | 0.0% | 59.2% | 0.7738 |
| GT-WA + current geometry | @5 | 56.4% | 0.0973 | 99.6% | 99.6% | 98.6% | 99.6% | 0.0% | 0.0% | 59.2% | 0.7738 |
| GT-WA + current geometry | @20 | 56.4% | 0.0973 | 99.6% | 99.6% | 98.6% | 99.6% | 0.0% | 0.0% | 59.2% | 0.7738 |
| GT-WA + geometry model no-over | @1 | 54.2% | 0.0607 | 72.2% | 72.2% | 100.0% | 72.2% | 57.4% | 57.4% | 57.6% | 0.6553 |
| GT-WA + geometry model no-over | @5 | 65.8% | 0.0751 | 72.6% | 72.6% | 100.0% | 72.6% | 51.5% | 51.5% | 51.7% | 0.6182 |
| GT-WA + geometry model no-over | @20 | 70.2% | 0.0835 | 73.3% | 73.3% | 100.0% | 73.3% | 44.2% | 44.1% | 44.5% | 0.5713 |
| GT-WA + geometry model oversampling | @1 | 54.2% | 0.0618 | 72.2% | 72.2% | 100.0% | 72.2% | 56.8% | 56.8% | 57.0% | 0.6495 |
| GT-WA + geometry model oversampling | @5 | 65.8% | 0.0749 | 72.3% | 72.3% | 100.0% | 72.3% | 51.2% | 51.1% | 51.4% | 0.6138 |
| GT-WA + geometry model oversampling | @20 | 70.4% | 0.0833 | 73.0% | 73.0% | 100.0% | 73.0% | 44.1% | 44.0% | 44.4% | 0.5677 |

## 4. Complex subset breakdown

| subset | samples | no-over match@20 | no-over RMSE | no-over valid | no-over strict_any | over match@20 | over RMSE | over valid | over strict_any |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 500 | 70.2% | 0.0835 | 44.2% | 83.2% | 70.4% | 0.0833 | 44.1% | 84.0% |
| n_sites>=6 | 175 | 31.4% | 0.2212 | 15.3% | 61.1% | 32.0% | 0.2280 | 15.5% | 64.0% |
| num_elements>=4 | 192 | 56.8% | 0.0655 | 33.1% | 76.6% | 55.7% | 0.0572 | 32.8% | 77.1% |
| SG=2 | 28 | 32.1% | 0.2095 | 27.7% | 85.7% | 32.1% | 0.2095 | 27.7% | 85.7% |
| SG=65 | 12 | 75.0% | 0.1351 | 32.5% | 83.3% | 83.3% | 0.1539 | 32.9% | 91.7% |
| SG=71 | 9 | 100.0% | 0.0509 | 55.6% | 88.9% | 100.0% | 0.0499 | 55.6% | 88.9% |
| SG=127 | 8 | 87.5% | 0.1735 | 32.5% | 75.0% | 87.5% | 0.1642 | 33.1% | 75.0% |

## 5. Artifacts

- no-over eval: `reports/symcif_v4_geometry_model_gtwa/no_oversampling/`
- oversampling eval: `reports/symcif_v4_geometry_model_gtwa/oversampling/`
- no-over checkpoint: `runs/symcif_v4_geometry_model_no_oversampling/ckpt_best.pt`
- oversampling checkpoint: `runs/symcif_v4_geometry_model_oversampling/ckpt_best.pt`
- training script: `scripts/train_symcif_v4_geometry_model.py`
- eval script: `scripts/run_symcif_v4_geometry_model_eval.py`
