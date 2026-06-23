# 01 Title and Abstract

## 候选题目

### 推荐题目

SymCIF: symmetry-conditioned Wyckoff assignment for diagnosable crystal structure generation

中文暂译：

SymCIF：用于可诊断晶体结构生成的对称性条件 Wyckoff 分配方法

### 更保守题目

SymCIF: a symmetry-conditioned representation for crystal structure generation

中文暂译：

SymCIF：面向晶体结构生成的对称性条件表示

### 更强调机制的题目

Decomposing crystal generation into Wyckoff assignment and geometry rendering

中文暂译：

将晶体生成分解为 Wyckoff 分配与几何渲染

## 摘要骨架

### Context

晶体结构生成需要同时满足组成、空间群、Wyckoff 位点、坐标和晶格几何之间的耦合约束。基于 CIF 的语言模型提供了统一的文本生成接口，但 CIF 文件记录顺序并不等同于晶体学约束顺序。

### Gap

在标准 CIF token sequence 中，Wyckoff assignment 和 geometry recovery 通常被混合建模，导致模型失败时很难判断错误来自空间群约束、位点占据、自由坐标参数还是晶格几何。

### Approach

我们提出 SymCIF，将晶体生成重组为 composition + space group 条件下的 Wyckoff assignment 选择和 geometry rendering。SymCIF 使用结构化字段表示 `formula_counts`、space group、Wyckoff table、free parameters 和 lattice，并在生成后重建 CIF 进行验证。

### Key result

在 MP-20 structured test 的 8,893 个样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。按原始 MP-20 test 的 9,046 个样本计入 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。

### Comparison and boundary

该结果是在 **composition + oracle ground-truth space group** 条件下取得的；公开 CrystaLLM-a MP-20 reference 是 **composition-only**，因此二者不是 same-input comparison。SymCIF 的 match@1 高于 CrystaLLM-a published reference 的 55.85%，但 RMSE@1 仍差于 CrystaLLM-a 的 0.0437。

### Implication

这些结果表明，显式的 symmetry decomposition 可以提高 MP-20 上的 top-1 recovery，并使错误来源可被审计；同时，连续几何质量仍是后续改进的关键瓶颈。

## 中文摘要草稿

晶体结构生成需要同时满足化学组成、空间群、Wyckoff 位点、原子坐标和晶格几何之间的耦合约束。基于 CIF 的语言模型为晶体生成提供了统一的文本接口，但 CIF 文件的记录顺序主要服务于数据交换，并不等同于晶体学约束传播顺序。因此，组成、空间群、Wyckoff assignment 和几何参数往往被混合在同一个自回归序列中学习，使得生成失败难以被定位和修复。

本文提出 SymCIF，一种 symmetry-conditioned crystal generation framework，将晶体生成分解为 Wyckoff assignment selection 和 geometry rendering。SymCIF 将 CIF 转换为包含 `formula_counts`、space group、Wyckoff table、free parameters 和 lattice 的结构化记录；生成阶段先在组成和空间群约束下选择 Wyckoff assignment，再渲染连续坐标与晶格并重建 CIF。该表示使候选覆盖、结构有效性、空间群一致性、match rate 和 RMSE 能够被分别评估。

在 MP-20 structured test 的 8,893 个样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615；按原始 MP-20 test 的 9,046 个样本计入 153 个 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。与 SymCIF current order 相比，hybrid-prior 将 WA_hit@1/@5 从 49.80%/71.27% 提高到 59.02%/81.66%，说明主要增益来自 train-only、inference-feasible 的 Wyckoff assignment reranking。与公开 CrystaLLM-a MP-20 reference 比较时，SymCIF 使用 composition + oracle ground-truth space group，而 CrystaLLM-a reference 是 composition-only；因此该比较不构成 same-input superiority。尽管 SymCIF 的 match@1 高于 CrystaLLM-a published reference，RMSE 仍然更差，表明 geometry quality 是当前主要瓶颈。

## TODO

- TODO: 补引用：CIF language modeling、CrystaLLM、MP-20、StructureMatcher、Wyckoff representations。
- TODO: 若后续补 bootstrap CI，需要在摘要中加入不确定性范围。
- TODO: 若补 same-input fairness experiment，可更新比较语句。

## RISK

- RISK: 摘要不能写 “state-of-the-art”。
- RISK: 摘要中的 CrystaLLM-a 比较必须同句说明输入条件不同。
- RISK: 不写 DFT stability，除非后续有完整 DFT 数据。

