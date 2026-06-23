# MP-20 Val One-Fix Experiment Summary

本轮只选择一个 fix 分支：branch A，基于 train split 统计先验和原搜索 rank 的 hybrid-prior top5 candidate selection。该 scorer 不使用 StructureMatcher label、不使用 val/test match label，也没有训练新模型。

| metric | baseline @1 | one-fix @1 | delta @1 | baseline @5 | one-fix @5 | delta @5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| match | 44.12% | 59.06% | 14.94% | 63.42% | 71.58% | 8.16% |
| RMSE | 0.0840 | 0.0591 | -0.0249 | 0.0828 | 0.0646 | -0.0183 |
| WA_hit | 38.63% | 56.91% | 18.28% | 65.11% | 79.52% | 14.41% |
| skeleton_hit | 47.32% | 69.78% | 22.46% | 70.39% | 82.51% | 12.13% |
| readable | 96.96% | 97.40% | 0.44% | 87.11% | 83.26% | -3.84% |
| formula_ok | 96.96% | 97.40% | 0.44% | 87.11% | 83.26% | -3.84% |
| atom_count_ok | 100.00% | 100.00% | 0.00% | 99.18% | 99.18% | 0.00% |
| SG_ok | 96.96% | 97.40% | 0.44% | 87.11% | 83.26% | -3.84% |
| strict_valid | 56.52% | 72.64% | 16.11% | 39.73% | 49.24% | 9.51% |

## Diagnosis

- 只改候选排序后，match@1 提升 14.94 pp，match@5 提升 8.16 pp，说明主瓶颈确实在 K<=5 candidate selection。
- one-fix WA_hit@5 = 79.52%，仍低于 raw top100 WA_hit = 87.85%，说明候选选择还有继续改进空间。
- one-fix RMSE 明显改善，但仍弱于 GT-WA geometry RMSE@5 = 0.0388，说明后续不能只停在 WA scorer，还需要非 oracle geometry-quality scorer/reranker。
- @5 readable/formula_ok/SG_ok 比 baseline 低 3.84 pp，说明排序偏向常见/高先验 WA 后，后排候选质量需要额外约束。
