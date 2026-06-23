# MP-20 K<=5 Dev Final Root Cause Summary

## Main Answer

当前主要瓶颈是 K<=5 预算下的 WA 候选选择；第二瓶颈是 geometry 质量。当前 val K<=5 pipeline 没有发现会导致结论失真的致命工程 bug。

## Evidence

- baseline val match@1/@5 = 44.12% / 63.42%；WA_hit@1/@5 = 38.63% / 65.11%。
- raw top100 WA_hit = 87.85%，但当前 selected top5 WA_hit 只有 65.11%。raw top100 里有 GT WA 但 top5 没选进去的样本有 2018 个。
- 只用 train-prior/hybrid-prior 改候选选择后，selected WA_hit@5 提升到 79.52%，实际 match@5 从 63.42% 提升到 71.58%。
- GT-WA geometry match@1/@5 = 77.16% / 82.94%，说明即使 WA 完全正确，geometry/RMSE 仍然是硬上限。

## Engineering Bug Audit

- 本轮没有发现新的致命工程 bug：CIF render、stable multiset key 命中统计、formula/SG/atom_count 检查和 evaluator 输出是自洽的。
- 旧的 ordered-key/stablekey 问题会低估 WA coverage，但本轮报告已使用 stable multiset key，因此不是当前结论的主因。
- eval_timeout = 0.00%，missing_candidate 规模也不足以解释主要性能差距。

## Root Cause

1. 当前 top5 排序没有充分利用 raw candidate pool。raw top100 已经包含大量 GT WA，但 selected top5 丢失了其中 2018 个。
2. 复杂结构仍然困难。`n_sites>=6` baseline match@5 = 24.13%，one-fix 也只解决了一部分。
3. geometry 质量仍然不足。GT-WA 只能到 82.94% match@5，说明后续必须加入 inference-feasible 的 geometry quality scoring/rerank。

## Recommendation

下一步继续 branch A：把候选 scorer 做成真正 inference-feasible、可泛化的版本，并叠加非 oracle geometry-quality scorer。暂时不要扩大到 MPTS-52，不要做 match@20 主线，不要在 MP-20 test 上反复调参。

## Artifacts

- `val_baseline_summary.md/json`
- `wa_top5_coverage_audit.md/json`
- `geometry_source_audit.md/json`
- `gtwa_geometry_k5_eval.md/json`
- `one_fix_experiment_summary.md/json`
