# 02 Introduction

## 逻辑蓝图

1. 晶体结构生成的难点不是写出合法文本，而是同时满足组成、对称性、位点占据和几何约束。
2. CIF language modeling 有价值，但 CIF file-record order 不等于 crystallographic constraint order。
3. 根本技术问题是 Wyckoff assignment 与 geometry recovery 被混合在长 token 序列中，错误不可诊断。
4. SymCIF 的核心 insight 是将晶体生成拆成 symmetry-conditioned Wyckoff assignment selection 和 geometry rendering。
5. Hybrid-prior selector 利用 train-only 分布信息进行 inference-feasible reranking，提高 WA coverage。
6. MP-20 结果支持 top-1 recovery 提升，但 RMSE 和 input-condition fairness 仍限制主张范围。

## 引言骨架

### 第 1 段：任务和挑战

晶体结构生成旨在从化学组成或其他条件出发恢复三维周期结构。该任务对材料筛选和结构候选生成有价值，但其输出必须同时满足多个约束：元素计数、空间群、Wyckoff 位点、多重度、坐标自由度和晶格几何。

TODO: 补材料发现/晶体结构生成引用。  
RISK: 不写 “直接解决材料发现”。

### 第 2 段：已有 CIF language modeling 的价值

基于 CIF 的语言模型将结构表示为文本序列，使生成模型可以直接学习标准结构文件并输出可解析 CIF。这个方向的优势是接口统一、工程实现直接、可复用自回归语言模型。

TODO: 补 CrystaLLM 等引用。  
RISK: 不否定 CrystaLLM；要承认它展示了 CIF generation 的有效性。

### 第 3 段：未解决的 representation mismatch

CIF 文件顺序是 record order，而不是 crystallographic causal order。文件中 cell parameters、formula、symmetry operations 和 atom_site loop 的排列，未必对应 composition -> space group -> Wyckoff assignment -> coordinates -> lattice 的约束传播关系。模型若按文件顺序直接学习，必须隐式保持多个字段之间的一致性。

### 第 4 段：本文方法

本文提出 SymCIF，将晶体结构表示为 formula counts、space group、Wyckoff assignment table、free parameters 和 lattice。该表示先确定 composition 和 oracle ground-truth space group，再选择 Wyckoff assignment，最后进行 geometry rendering 和 CIF reconstruction。

RISK: “causal order” 只能写为“更接近晶体学约束传播顺序”，不能写成真实晶体形成物理因果。

### 第 5 段：方法优势

SymCIF 的优势在于可约束、可重排、可诊断。它将最终 match/RMSE 拆解为 WA_hit、skeleton_hit、formula_ok、SG_ok、render_success、strict_valid 和 geometry mismatch 等环节，使性能提升和失败机制可被追踪。

### 第 6 段：结果预告和边界

在 MP-20 structured test 上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%。按原始 test 分母折算后，match@1 = 59.87%、match@5 = 72.43%。这些结果是在 composition + oracle ground-truth space group 条件下取得的；公开 CrystaLLM-a reference 是 composition-only。SymCIF match@1 高于其 published reference，但 RMSE 仍更差，因此本文重点是 symmetry-conditioned top-1 recovery 和 failure diagnosis，而不是 same-input SOTA claim。

## 中文引言草稿骨架

晶体结构生成的目标不仅是生成一个语法合法的结构文件，还要生成一个满足晶体学约束的三维周期结构。给定化学组成后，合理结构必须同时满足元素计数、空间群对称性、Wyckoff 位点多重度、原子坐标自由度和晶格几何之间的耦合关系。这些约束共同决定生成结构是否能够被解析、是否保持正确空间群、是否与真实结构匹配，以及匹配后的几何误差是否足够小。

CIF 为晶体结构提供了通用文本表示，因此自然成为语言模型生成晶体结构的接口。已有 CIF language modeling 方法表明，自回归模型可以学习 CIF 文件中的 cell parameters、formula、symmetry operations 和 atom_site loop，并在给定组成等条件时生成完整结构。该范式的优势是统一、灵活并且工程路径清晰。然而，CIF 文件的线性记录顺序主要服务于存储和交换，并不等同于晶体学约束的生成顺序。

这种顺序错位使 CIF token generation 面临一个具体困难：组成、空间群、Wyckoff assignment 和几何参数之间的依赖关系需要在长序列中被隐式学习。空间群决定合法 Wyckoff orbits 和 site symmetries；Wyckoff multiplicities 限制元素计数如何被覆盖；坐标自由参数只有在给定 orbit 后才有明确意义；晶格与展开坐标共同决定最终结构匹配误差。当这些关系被混合在普通 CIF 序列中时，生成失败很难被归因到错误的 skeleton、错误的元素分配、坐标参数失败或几何质量不足。

为解决这一 representation mismatch，我们提出 SymCIF，一种 symmetry-conditioned structured representation and generation pipeline。SymCIF 将 CIF 转换为包含 `formula_counts`、space group、Wyckoff assignment table、free parameters 和 lattice 的结构化记录。在该流程中，composition 和 oracle ground-truth space group 首先限定候选空间；Wyckoff assignment 决定离散对称骨架和元素占据；free parameters 和 lattice 再用于渲染连续几何；最终结构被重建为 CIF 并接受公式、空间群、match 和 RMSE 评估。

SymCIF 的关键优势是将生成过程变得可审计。每个候选不仅有最终 CIF，还保留 canonical skeleton key、canonical WA key、render status、validity checks 和 matching outcome。因此，模型改进可以围绕具体失败来源进行：top-k 中是否包含正确 WA，正确 WA 是否转化为 match，match 后 RMSE 是否足够低。本文进一步使用 train-only hybrid-prior selector 对候选 WA 进行 inference-feasible reranking，以提高正确 assignment 进入 top-1/top-5 的概率。

在 MP-20 上，SymCIF hybrid-prior 在 8,893 个 structured test 样本上达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。按原始 9,046 个 MP-20 test 样本计入 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。该结果显示，在 composition + oracle ground-truth space group 条件下，SymCIF 具有较强 top-1 recovery，并高于 CrystaLLM-a published MP-20 match@1 reference。与此同时，该比较不是 same-input comparison，且 SymCIF RMSE 仍差于 CrystaLLM-a reference。因此，本文的中心结论是：显式的 symmetry decomposition 和 WA reranking 能够提升 MP-20 top-1 recovery，并揭示 geometry quality 是剩余瓶颈。

