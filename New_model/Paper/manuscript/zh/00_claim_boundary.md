# 00 Claim Boundary

## 写作口径

本稿采用 **medium-conservative** framing：SymCIF 被定位为一个 symmetry-conditioned crystal generation framework，而不是已经全面优于所有晶体生成方法的 state-of-the-art 系统。

一句话主张：

> SymCIF 将晶体生成分解为空间群条件下的 Wyckoff assignment 选择和几何渲染，在 MP-20 上实现了较强的 top-1 recovery；同时，实验诊断显示连续几何质量仍是主要瓶颈。

## 当前可以主张的内容

1. SymCIF 将 CIF language modeling 中隐式耦合的组成、空间群、Wyckoff 位点、坐标和晶格关系，重组为更接近晶体学约束传播顺序的结构化流程。
2. 在 MP-20 structured test 的 8,893 个样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。
3. 按原始 MP-20 test 的 9,046 个样本计入 153 个 structured extraction failures 后，SymCIF hybrid-prior 仍达到 match@1 = 59.87%、match@5 = 72.43%。
4. 相比 SymCIF current order，hybrid-prior 将 structured test match@1 从 51.29% 提高到 60.90%，match@5 从 66.11% 提高到 73.68%。
5. WA_hit@1/@5 从 49.80%/71.27% 提高到 59.02%/81.66%，说明主要增益来自更好的 Wyckoff assignment reranking。
6. MP-20 full evaluation 覆盖 8,893/8,893 structured test samples，@1/@5 eval timeout = 0，render_success@1 = 100.00%，render_success@5 = 99.10%，主结果不是由大规模评估缺失造成的。

## 必须同时声明的边界

1. SymCIF 当前主结果使用 **composition + oracle ground-truth space group**。
2. CrystaLLM-a published MP-20 reference 是 **composition-only**。
3. 因此，SymCIF 与 CrystaLLM-a 的公开结果不是 same-input comparison。
4. 可以写 “under a ground-truth-space-group setting, SymCIF exceeds the published CrystaLLM-a match@1 reference”，但不能写 “SymCIF outperforms CrystaLLM-a under the same input setting”。
5. SymCIF 的 RMSE@1 = 0.0573，仍差于 CrystaLLM-a published RMSE@1 = 0.0437；因此不能声称几何质量优于 CrystaLLM-a。
6. Hybrid-prior 是 train-only 且 inference-feasible，但它仍依赖训练分布中的 Wyckoff assignment 统计规律，不能写成 distribution-independent selector。
7. 当前没有系统 DFT stability 结果，不能写任何 DFT 稳定性提升。

## RISK 声明

- RISK: “SymCIF achieves state-of-the-art crystal generation.”  
  原因：输入条件与 CrystaLLM-a 不同，MPTS-52/generalization 证据不充分，RMSE 仍落后。

- RISK: “SymCIF surpasses CrystaLLM-a.”  
  原因：除非同句说明 “while using composition + oracle ground-truth space group”，否则容易被理解为 same-input superiority。

- RISK: “Hybrid-prior proves causal improvement.”  
  原因：当前证据支持 WA coverage 和 match 同步提升，但若要强因果论证，还需要 selector ablation、bootstrap CI 和 train-frequency/novelty analysis。

- RISK: “Geometry bottleneck is fully solved.”  
  原因：WA_hit@5 = 81.66% 高于 match@5 = 73.68%，且 RMSE 落后于 CrystaLLM-a。

## 建议统一术语

- `SymCIF representation`: 对称性条件结构化表示。
- `Wyckoff assignment`: 元素到 Wyckoff orbit 的占据分配。
- `WA_hit`: top-k 候选中是否包含 ground-truth 或等价 Wyckoff assignment。
- `skeleton_hit`: top-k 候选中是否包含正确的 Wyckoff orbit skeleton，不要求元素分配完全一致。
- `hybrid-prior selector`: 使用 train-only Wyckoff prior 和候选排序信息的 inference-feasible reranking。
- `structured test`: 成功转换为 SymCIF 记录的 MP-20 test 子集。
- `original-test adjusted`: 原始 MP-20 test 分母口径，structured extraction failures 计为失败。

