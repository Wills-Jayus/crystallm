# 03 Results

## Results Storyline

本节按 evidence ladder 组织：先定义 SymCIF pipeline，再给 MP-20 主结果，然后解释 gain 来自 WA reranking，接着指出 geometry bottleneck，最后用 failure audit 明确方法边界。

## 3.1 SymCIF representation and pipeline

### Claim

SymCIF 将 CIF 生成从 file-record order 重组为 symmetry-conditioned structured generation，使 Wyckoff assignment selection 和 geometry rendering 可以被分别建模和审计。

### Evidence

- 原始 CIF order: data block -> cell parameters -> formula -> symmetry operations -> atom_site loop。
- SymCIF order: composition -> space group -> Wyckoff assignment -> free parameters / coordinates -> lattice -> CIF reconstruction -> validation。
- SymCIF record 包含 `formula_counts`、`sg`、`sg_symbol`、`wa_table`、`free_params`、`lattice`、`canonical_skeleton_key` 和 `canonical_wa_key`。

### Interpretation

该表示把离散对称性约束放在连续几何之前，使错误可以被定位到 WA coverage、element assignment、rendering、validity 或 StructureMatcher outcome。

### Draft paragraph

我们首先将 CIF language modeling 中的线性文件记录转换为 SymCIF 的结构化表示。与按 data block、cell parameters、formula、symmetry operations 和 atom_site loop 生成 CIF 的流程不同，SymCIF 从 composition 和 space group 出发，先确定 Wyckoff assignment，再渲染连续坐标和晶格，最后重建 CIF 并进行验证。每个 SymCIF record 保留 `formula_counts`、space group、Wyckoff table、free parameters、lattice 以及 canonical skeleton/WA keys，使候选覆盖、元素分配和几何渲染能够被分开检查。

TODO: 加 Figure 1。  
TODO: 加一个真实样本的 SymCIF record snippet。  
RISK: 不把 representation 本身写成已证明提升，提升证据要留给后续实验。

## 3.2 MP-20 main benchmark

### Claim

在 composition + oracle ground-truth space group 条件下，SymCIF hybrid-prior 在 MP-20 上达到强 top-1/top-5 recovery。

### Evidence

| Method | Input condition | Scope | match@1 | match@5 | RMSE@1 | RMSE@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| SymCIF current order | composition + oracle SG | structured test | 51.29% | 66.11% | 0.0668 | 0.0742 |
| SymCIF hybrid-prior | composition + oracle SG | structured test | 60.90% | 73.68% | 0.0573 | 0.0615 |
| SymCIF hybrid-prior | composition + oracle SG | original-test adjusted | 59.87% | 72.43% | 0.0573 | 0.0615 |
| CrystaLLM-a published reference | composition only | published MP-20 reference | 55.85% | TODO | 0.0437 | TODO |

### Interpretation

SymCIF hybrid-prior 明显优于 SymCIF current order，并且在 oracle-SG 条件下 match@1 高于 CrystaLLM-a published reference。但该比较不是 same-input comparison，且 RMSE 仍更差。

### Draft paragraph

在 MP-20 structured test 的 8,893 个样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。相比 SymCIF current order，match@1 提高 9.61 个百分点，match@5 提高 7.57 个百分点，同时 RMSE 下降。若按原始 MP-20 test 的 9,046 个样本计入 153 个 structured extraction failures，SymCIF hybrid-prior 仍达到 match@1 = 59.87%、match@5 = 72.43%。与 CrystaLLM-a published MP-20 reference 比较时，必须注意 SymCIF 使用 composition + oracle ground-truth space group，而 CrystaLLM-a reference 是 composition-only；因此该结果支持 oracle-SG 设置下的强 top-1 recovery，而不是 same-input superiority claim。

TODO: 补 CrystaLLM-a match@5 是否有公开数值；若没有，不填。  
TODO: 补 bootstrap 95% CI。  
RISK: 表格必须列 Input condition。

## 3.3 WA reranking explains the gain

### Claim

Hybrid-prior 的性能提升主要来自 train-only、inference-feasible 的 Wyckoff assignment reranking 提高了 top-k WA coverage。

### Evidence

| Selector | WA_hit@1 | WA_hit@5 | match@1 | match@5 |
| --- | ---: | ---: | ---: | ---: |
| current order | 49.80% | 71.27% | 51.29% | 66.11% |
| hybrid-prior | 59.02% | 81.66% | 60.90% | 73.68% |

### Interpretation

WA_hit 与 match 同步提升，说明 gain 与候选排序质量相关。Hybrid-prior 使用训练集先验，因此不能写成独立于训练分布。

### Draft paragraph

为了定位性能提升来源，我们比较了 current order 和 hybrid-prior selector 的 WA coverage。Current order 的 WA_hit@1/@5 为 49.80%/71.27%，而 hybrid-prior 提高到 59.02%/81.66%。对应地，match@1/@5 从 51.29%/66.11% 提高到 60.90%/73.68%。这表明，在 K<=5 的预算下，将正确 Wyckoff assignment 更早排入候选列表，是 SymCIF top-1/top-5 提升的主要原因。由于 hybrid-prior 使用 train-only Wyckoff statistics，该结果说明训练分布中存在可利用的 assignment prior，但不应被解释为 distribution-independent generalization。

TODO: 补 Table 2 selector ablation：policy rank only、train WA prior only、skeleton prior only、hybrid full。  
TODO: 补 train-frequency bucket analysis。  
RISK: “explains the gain” 需要 ablation 支持；现有 WA_hit/match 同步提升是强诊断，但还不是完整因果证明。

## 3.4 Geometry remains the bottleneck

### Claim

WA coverage 尚未完全转化为 match，且 RMSE 仍差于 CrystaLLM-a reference，说明 geometry rendering/refinement 是当前主要瓶颈。

### Evidence

- hybrid-prior WA_hit@5 = 81.66%，match@5 = 73.68%，存在约 8 pp coverage-to-match gap。
- SymCIF RMSE@1 = 0.0573，CrystaLLM-a published RMSE@1 = 0.0437。
- SymCIF RMSE@5 = 0.0615，CrystaLLM-a published match@20 RMSE = 0.0395。

### Interpretation

即使候选集中已有正确 WA，坐标自由参数、晶格、validity 或 StructureMatcher 判据仍可能导致失败。后续应优先做 geometry-quality scoring 和 geometry refinement。

### Draft paragraph

Hybrid-prior 提高了 WA coverage，但并未完全解决结构恢复。WA_hit@5 达到 81.66%，而 match@5 为 73.68%，说明部分样本已经包含正确或等价 Wyckoff assignment，但最终结构仍未通过匹配。与此同时，SymCIF 的 RMSE@1 = 0.0573，仍高于 CrystaLLM-a published reference 的 0.0437。这一差距表明，当前 SymCIF 更擅长将正确离散对称性候选排到前列，而连续坐标、晶格参数和几何质量仍限制最终结构精度。

TODO: 补 geometry_source audit 主表或图。  
TODO: 补 GT-WA + current geometry / diagnostic geometry rerank 对比。  
RISK: 不写 “geometry 已解决”。

## 3.5 Failure audit and method boundary

### Claim

SymCIF 的 failure audit 说明主结果工程可靠，同时明确了复杂结构、extraction-hard cases 和 geometry mismatch 的方法边界。

### Evidence

- MP-20 full evaluation 覆盖 8,893/8,893 structured samples。
- @1/@5 eval_timeout = 0。
- @5 有 44,465 条 candidate-level records。
- render_success@1 = 100.00%，render_success@5 = 99.10%。
- 复杂子集如 `n_sites>=6`、`n_sites>=12`、`extraction_hard` 较弱；high-multiplicity orbit 子集相对较强。

### Interpretation

结果不是 timeout 或 incomplete evaluation artifact。失败模式不是单一“复杂度”可解释，而是与 WA coverage、geometry source、validity 和结构复杂度共同相关。

### Draft paragraph

最后，我们对 MP-20 full evaluation 进行 failure audit。评估覆盖所有 8,893 个 structured test samples，@1/@5 均无 evaluation timeout，top-5 评估包含 44,465 条 candidate-level records，render_success@1 和 render_success@5 分别为 100.00% 和 99.10%。这些结果表明主结果不是由不完整运行或大规模渲染失败造成的。与此同时，`n_sites>=6`、`n_sites>=12` 和 `extraction_hard` 等子集仍显著更难，而 high-multiplicity orbit 子集表现相对较强，说明失败并不能由单一结构复杂度解释。SymCIF 的可审计性使这些边界能够被明确定位，为后续 geometry-aware reranking 和 refinement 提供依据。

TODO: 补复杂子集具体数字。  
TODO: 补 Supplementary Table S1/S2。  
RISK: 不把 failure audit 写成泛化已证明。

