# MP-20 Val K<=5 Baseline Summary

## Overall

本表只统计 MP-20 val split，使用当前 hybrid WA 排序和现有 geometry model/prototype renderer。按本轮约束，只报告 @1/@5，不产出 match@20。

| metric | @1 | @5 |
| --- | ---: | ---: |
| match | 44.12% | 63.42% |
| RMSE | 0.0840 | 0.0828 |
| WA_hit | 38.63% | 65.11% |
| skeleton_hit | 47.32% | 70.39% |
| readable | 96.96% | 87.11% |
| formula_ok | 96.96% | 87.11% |
| atom_count_ok | 100.00% | 99.18% |
| SG_ok | 96.96% | 87.11% |
| strict_valid | 56.52% | 39.73% |
| strict_valid_any | 56.52% | 79.96% |
| eval_timeout | 0.00% | 0.00% |

## Complex Subsets

| subset | samples | match@1 | WA_hit@1 | skeleton_hit@1 | match@5 | WA_hit@5 | skeleton_hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 8874 | 44.12% | 38.63% | 47.32% | 63.42% | 65.11% | 70.39% |
| n_sites>=6 | 2109 | 11.19% | 20.10% | 22.62% | 24.13% | 33.95% | 39.02% |
| n_sites>=12 | 350 | 11.43% | 37.14% | 38.57% | 18.29% | 39.14% | 42.57% |
| n_sites>=20 | 31 | 25.81% | 83.87% | 83.87% | 35.48% | 83.87% | 83.87% |
| num_elements>=4 | 1818 | 35.86% | 32.62% | 42.74% | 52.92% | 53.96% | 62.65% |
| rare_sg | 291 | 31.62% | 30.58% | 35.40% | 52.92% | 56.01% | 60.14% |
| high_multiplicity_orbit | 1251 | 64.91% | 51.40% | 63.95% | 84.81% | 82.65% | 86.49% |
| extraction_hard | 1236 | 15.21% | 29.94% | 36.49% | 36.17% | 67.96% | 73.71% |

## Diagnosis

- 当前 baseline 的 K<=5 主要短板是候选选择：WA_hit@5 只有 65.11%，说明很多样本在前 5 个候选里没有选到 GT WA。
- 复杂结构更明显：`n_sites>=6` 的 match@5 只有 24.13%，WA_hit@5 也只有 33.95%。
- readable/formula_ok/SG_ok 在 @5 低于 @1，说明后排候选质量更差；但 eval_timeout 为 0，不是 evaluator 超时导致的主要问题。
