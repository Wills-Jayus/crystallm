# 05 Discussion

## Discussion 逻辑

1. Central advance: SymCIF 将晶体生成分解为 WA selection 和 geometry rendering。
2. Evidence meaning: MP-20 top-1 recovery 提升来自 WA coverage/reranking。
3. Relation to prior CIF language modeling: 不是否定 CIF LM，而是补充一种 symmetry-first structured route。
4. Limitation: oracle SG input、RMSE worse、train-distribution prior、MP-20-centered evidence。
5. Future direction: composition-only SG prediction、geometry-quality predictor、fairness baseline、generalization。

## 中文讨论骨架

SymCIF 的主要意义在于将晶体结构生成中的对称性约束显式化。与直接按 CIF 文件顺序生成完整文本不同，SymCIF 先在 composition 和 space group 条件下选择 Wyckoff assignment，再进行连续几何渲染和 CIF reconstruction。这一分解不仅改善了 MP-20 上的 top-1 recovery，也使错误来源能够被拆解为 WA coverage、skeleton coverage、validity 和 geometry mismatch。

实验结果表明，当前最主要的性能增益来自 Wyckoff assignment reranking。Hybrid-prior selector 将 WA_hit@1/@5 从 49.80%/71.27% 提升到 59.02%/81.66%，并带来 match@1/@5 的同步提升。这说明，在 K<=5 的候选预算下，更早地选择正确的离散对称性候选，比单纯扩大 geometry variants 更重要。由于该 selector 使用 train-only prior，该结果也表明 MP-20 训练分布中的 Wyckoff assignment 规律可以被有效利用。

然而，SymCIF 的结果也明确显示 geometry quality 仍是主要瓶颈。WA_hit@5 高于 match@5，说明一部分正确 WA 未能转化为最终结构匹配。进一步地，SymCIF RMSE@1 仍高于 CrystaLLM-a published reference，说明当前方法尚未在连续坐标和晶格几何质量上达到更强基线水平。因此，SymCIF 的当前优势应被理解为 top-1 recovery 和可诊断性，而不是几何精度的全面领先。

与 CrystaLLM-a 的比较需要谨慎解释。SymCIF 使用 composition + oracle ground-truth space group，而 CrystaLLM-a published reference 是 composition-only。这个设置差异意味着当前结果不能证明 same-input superiority。更合理的结论是：在给定真实空间群的 symmetry-conditioned setting 下，SymCIF 取得了强 MP-20 top-1 recovery，并通过 failure audit 揭示了 remaining bottleneck。

后续工作应沿三个方向推进。第一，将空间群预测或检索纳入 SymCIF，使方法能够从 composition-only 输入出发，并与 CrystaLLM-a 等方法进行公平比较。第二，开发不依赖 evaluation oracle 的 geometry-quality predictor 或 refinement module，提高正确 WA 候选的最终匹配率并降低 RMSE。第三，在 MPTS-52 或其他外部数据集上重跑最终流程，并按 train-frequency、rare SG、unseen WA 和 complex subsets 进行泛化分析。

## RISK

- RISK: Discussion 不要把 “train prior 有效” 写成 “模型学会了普适晶体规律”。
- RISK: 不要写 “SymCIF solves crystal generation”。
- RISK: 不要用 DFT stability 做 implication，除非后续补数据。

## TODO

- TODO: 补 same-input fairness experiment 后，更新 Discussion 中的 baseline 关系。
- TODO: 补 bootstrap CI 后，讨论统计显著性。
- TODO: 补 generalization 实验后，决定是否扩大主张。

