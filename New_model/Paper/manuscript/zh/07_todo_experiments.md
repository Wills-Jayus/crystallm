# 07 TODO Experiments

## Priority 1: selector ablation

### Purpose

证明 hybrid-prior 的提升不是单一排序偶然，而是 train-only prior 与 candidate rank 结合带来的可复现增益。

### Minimal variants

- current order
- policy rank only
- old rank only
- train WA frequency only
- train skeleton frequency only
- hybrid-prior full
- oracle GT-WA diagnostic

### Metrics

- WA_hit@1/@5
- skeleton_hit@1/@5
- match@1/@5
- RMSE@1/@5
- strict_valid@1/@5

## Priority 1: bootstrap confidence intervals

### Purpose

给 match@1、match@5 和与 CrystaLLM-a reference 的差距提供统计不确定性。

### Protocol

- 对 8,893 structured test samples bootstrap 10,000 次。
- 对 9,046 original-test adjusted samples bootstrap 10,000 次。
- 报告 95% CI。

TODO: 若 pipeline deterministic，说明 bootstrap 是 sample-level uncertainty，而不是 seed variance。

## Priority 1: input-condition fairness experiment

### Purpose

回应 SymCIF 使用 oracle ground-truth space group，而 CrystaLLM-a published reference 是 composition-only 的公平性问题。

### Options

- Route A: 给 CrystaLLM-a 同样 oracle SG prompt，并用同一 evaluator 对比。
- Route B: 让 SymCIF 从 composition-only 预测或检索 SG，再报告性能下降。
- Route C: 明确把论文定位为 symmetry-conditioned generation，不做 same-input SOTA claim。

当前 skeleton 采用 Route C 的写法。若要提升主张强度，建议补 Route A 或 B。

## Priority 2: geometry bottleneck ablation

### Purpose

解释 RMSE 落后和 WA_hit@5 > match@5 的原因。

### Suggested variants

- selected WA + current geometry
- GT-WA + current geometry
- selected WA + diagnostic best geometry
- validity-first diagnostic rerank
- geometry-quality diagnostic rerank

### Metrics

- match@1/@5
- RMSE@1/@5
- strict_valid@1/@5
- render_success@1/@5

RISK: diagnostic rerank 如果使用 evaluator-derived fields，必须明确标为 oracle/diagnostic，不能当真实 inference result。

## Priority 2: train-frequency and novelty analysis

### Purpose

判断 hybrid-prior 是否只在训练集中常见 WA pattern 上有效。

### Buckets

- GT WA seen in train vs unseen
- GT skeleton seen in train vs unseen
- train WA frequency quantiles
- common SG vs rare SG
- low/high n_sites
- low/high num_elements

### Metrics

- WA_hit@5
- match@5
- RMSE@5
- failed_with_WA_hit / failed_without_WA_hit

## Priority 3: MPTS-52 or external generalization

### Purpose

验证 SymCIF 的 claim 是否可以从 MP-20 扩展到其他 benchmark。

### Caution

如果 MPTS-52 结果弱，应作为 boundary analysis，而不是主 claim。当前不建议在没有最终流程重跑结果前写 “across benchmarks”。

## Priority 3: qualitative examples

### Purpose

帮助读者理解 WA correct、geometry fail 和 WA missing fail 的区别。

### Suggested cases

- top-1 matched and low RMSE
- top-5 matched after reranking
- WA correct but geometry fail
- WA missing fail
- extraction-hard fail

## Priority 3: DFT stability

### Purpose

如果论文要讨论物理稳定性，需要 DFT relaxation 或能量稳定性评估。

### Current status

当前没有可写入主文的 DFT stability 结果。

RISK: 在补实验前，不要写 “DFT stability improved” 或 “generated structures are thermodynamically stable”。

