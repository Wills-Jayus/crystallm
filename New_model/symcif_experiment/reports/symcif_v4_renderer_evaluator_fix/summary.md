# symcif v4 renderer/evaluator 兼容性修复报告

## 1. 结论

修复已生效：v4 expanded CIF 不再被 legacy evaluator 的 multiplicity/formula 检查全部打成 `valid=0`。`strict_valid@20` 从 0.0% 恢复到 14.7%，`strict_valid_any@20` 恢复到 68.2%。

代价是兼容检查后评测更重，`eval_timeout@20` 从 3.0% 升到 9.0%，所以 match@20 从 56.6% 小幅降到 55.6%。这个下降更像 evaluator timeout/检查成本变化，不是 WA 搜索能力退化。

## 2. Before / After

| run | k | match | RMSE | readable | formula_ok | atom_count_ok | SG_ok | valid | strict_valid | strict_any | WA hit | eval_timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before renderer/evaluator fix | @1 | 34.8% | 0.1149 | 95.6% | 95.6% | 98.0% | 95.6% | 0.0% | 0.0% | 0.0% | 49.6% | 3.0% |
| after renderer/evaluator fix | @1 | 34.4% | 0.1121 | 89.6% | 89.6% | 98.0% | 89.6% | 47.6% | 47.6% | 47.6% | 49.6% | 9.0% |
| before renderer/evaluator fix | @5 | 52.0% | 0.1043 | 83.5% | 83.5% | 89.5% | 83.5% | 0.0% | 0.0% | 0.0% | 77.2% | 3.0% |
| after renderer/evaluator fix | @5 | 51.4% | 0.1038 | 77.5% | 77.5% | 89.5% | 77.5% | 30.6% | 30.6% | 64.6% | 77.2% | 9.0% |
| before renderer/evaluator fix | @20 | 56.6% | 0.1030 | 69.9% | 69.9% | 77.2% | 69.9% | 0.0% | 0.0% | 0.0% | 86.6% | 3.0% |
| after renderer/evaluator fix | @20 | 55.6% | 0.1003 | 64.2% | 64.2% | 77.2% | 64.2% | 14.7% | 14.7% | 68.2% | 86.6% | 9.0% |

## 3. 修改点

- `src/symcif_v4/orbit_engine.py` 的 CIF data block 改为包含 formula id，避免 sample id 与 formula 不一致。
- 渲染输出新增 `_chemical_formula_structural`、`_cell_formula_units_Z`、`_cell_volume`。
- `_atom_site` loop 新增 `_atom_site_symmetry_multiplicity`，展开坐标行统一写 1，使 legacy evaluator 的 multiplicity loop 能正常读取。

## 4. Artifacts

- after-fix eval: `reports/symcif_v4_renderer_evaluator_fix/symcif_v4_full_eval_current/`
- summary json: `reports/symcif_v4_renderer_evaluator_fix/summary.json`
- before-fix reference: `reports/symcif_v4_full_eval_current/`
