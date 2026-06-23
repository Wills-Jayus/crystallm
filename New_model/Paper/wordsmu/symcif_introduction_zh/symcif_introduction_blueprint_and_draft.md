# SymCIF Introduction 蓝图与中文初稿

## 6 段逻辑蓝图

### 第 1 段：任务背景与问题入口

**段落任务：** 从晶体结构生成的重要性进入，但不把工作包装成已经解决材料发现。  
**核心信息：** 晶体生成需要同时满足化学组成、对称性、原子坐标和晶格几何；CIF 是常用结构表达，但生成合法 CIF 不等于生成晶体学一致的结构。  
**边界控制：** 只说“为生成和评估晶体结构提供基础”，不要说“直接实现材料发现”或“解决逆向设计”。

### 第 2 段：已有 CIF language modeling 的价值与局限

**段落任务：** 公平承认 CrystaLLM 等 CIF language model 的意义，再指出本文要解决的结构性问题。  
**核心信息：** CIF language modeling 将晶体结构作为文本序列生成，具有统一、灵活、可扩展的优势；但 CIF 文件顺序是记录顺序，不是晶体学约束顺序。  
**边界控制：** 不否定 CrystaLLM；只说其 representation 没有显式分离 Wyckoff assignment 和 geometry recovery。

### 第 3 段：根本技术瓶颈

**段落任务：** 把问题收窄为 SymCIF 实际解决的 technical challenge。  
**核心信息：** 在晶体结构中，composition、space group、Wyckoff multiplicity、site symmetry、coordinates 和 lattice 是耦合的；如果在长 CIF token 序列中隐式学习，错误很难归因。  
**边界控制：** 不声称所有失败都来自 token order；说“这一顺序会使约束关系更难显式建模和诊断”。

### 第 4 段：提出 SymCIF representation

**段落任务：** 用 Methods 中的 representation 反推本文方法。  
**核心信息：** SymCIF 将结构表示为 formula_counts、space group、wa_table、free parameters、lattice 和 canonical keys；先确定离散对称骨架和 element-Wyckoff assignment，再渲染连续坐标与晶格。  
**边界控制：** 只说“符合 crystallographic causal order”或“更接近约束传播顺序”，不要说“完全模拟真实晶体形成过程”。

### 第 5 段：候选选择、重排与可审计评估

**段落任务：** 连接 representation 与 Results 中的主要机制证据。  
**核心信息：** SymCIF 不只输出结构，还能把失败拆解为 WA coverage、skeleton coverage、validity、render success 和 geometry mismatch；train-only hybrid-prior reranking 提升 WA coverage，并带来 match 提升。  
**边界控制：** 强调 train-only、inference-feasible；不要暗示它完全摆脱训练分布 prior。

### 第 6 段：本文贡献和有边界的结果预告

**段落任务：** 说明本文做了什么、证明到哪里、还没有证明什么。  
**核心信息：** 在 MP-20 structured test 上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%；原始分母折算 match@1 = 59.87%；在 symmetry-conditioned setting 下高于 published CrystaLLM-a match@1 reference，但 RMSE 仍落后，说明 geometry quality 是后续瓶颈。  
**边界控制：** 必须写明使用 composition + oracle ground-truth space group；不能写成同输入条件全面超过 CrystaLLM-a。

## 中文 Introduction 初稿

晶体结构生成的目标不仅是产生语法合法的结构文件，还要同时满足化学组成、空间群对称性、Wyckoff 位点占据、原子坐标和晶格几何之间的耦合约束。CIF 为晶体结构提供了通用的文本表达，因此很自然地成为语言模型生成晶体结构的接口。然而，对于材料建模而言，一个可读的 CIF 文件并不必然对应一个晶体学一致、可匹配且几何合理的结构。如何在生成过程中同时保持文本可解析性、对称性一致性和结构几何质量，仍然是晶体生成模型需要面对的核心问题。

近年来，基于 CIF 的 language modeling 方法展示了将晶体结构作为序列数据建模的潜力。这类方法可以直接学习 CIF 文件中的 cell parameters、chemical formula、symmetry operations 和 atom_site loop，并在给定组成等条件时生成完整结构。它们的优势在于形式统一、工程路径清晰，并且能够复用通用自回归建模框架。但 CIF 文件的记录顺序主要服务于数据存储和交换，并不等价于晶体学约束的因果顺序。也就是说，文件中先后出现的字段，未必反映 composition、space group、Wyckoff assignment、coordinates 和 lattice 在结构构造中的依赖关系。

这一错位会带来一个具体的技术困难：模型需要在长序列中隐式恢复多个相互耦合的晶体学约束。给定 composition 后，space group 决定可用的 Wyckoff orbits 和 site symmetries；Wyckoff multiplicities 又限制元素如何覆盖 conventional cell 中的原子计数；在选定 orbit 后，坐标自由度才具有明确含义；最终 lattice 与展开后的 fractional coordinates 共同决定结构是否能通过匹配和有效性检查。如果这些关系被混合在普通 CIF token 序列中，生成错误往往难以判断是来自错误的 Wyckoff skeleton、错误的元素分配、坐标自由参数失败，还是来自 lattice/geometry 质量不足。

为此，我们提出 SymCIF，一种面向晶体结构生成的 symmetry-conditioned structured representation。SymCIF 不直接以 CIF 文件顺序作为模型对象，而是将结构拆解为 `formula_counts`、space group、Wyckoff assignment table、free parameters、lattice 和 canonical keys。具体而言，`wa_table` 中的每一行表示一个元素占据的 Wyckoff orbit，并记录 multiplicity、site symmetry、enumeration、representative coordinate expression 和自由参数。`canonical_skeleton_key` 描述结构选择了哪些 Wyckoff orbits，`canonical_wa_key` 进一步加入元素分配。这样的表示使生成流程从 composition 和 space group 出发，先确定离散的对称骨架和元素占据，再渲染连续坐标与晶格，最后重构并验证 CIF。

这一表示也使生成过程更容易被诊断和改进。由于 Wyckoff skeleton、element-Wyckoff assignment、free-parameter extraction、render success 和 StructureMatcher outcome 都可以被单独记录，SymCIF 能够把总的 match/RMSE 结果分解为候选覆盖、结构有效性和几何质量等不同来源。在此基础上，我们引入 train-only hybrid-prior reranking，对候选 Wyckoff assignments 进行 inference-feasible 重排。该重排不使用 test label 或 StructureMatcher oracle，而是利用训练集中可见的 assignment 频率和已有候选排序信息，提高正确 WA 候选进入 top-1/top-5 的概率。

我们在 MP-20 上评估 SymCIF 的这一结构化生成流程。在 8,893 条 MP-20 structured test 样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615；若按原始 MP-20 test 的 9,046 条分母折算，并将 153 条 structured extraction failure 计为失败，match@1 和 match@5 仍为 59.87% 和 72.43%。这些结果表明，在 composition + oracle ground-truth space group 的 symmetry-conditioned 设置下，显式的 symmetry decomposition 和 WA reranking 可以带来较强的 top-1 recovery。与此同时，SymCIF 的 RMSE 仍高于 published CrystaLLM-a reference，且 WA_hit@5 与 match@5 之间仍存在差距，说明当前方法尚未解决几何精度问题。因此，本文的重点不是声称 SymCIF 在所有输入条件下全面优于既有 CIF language model，而是展示一种可约束、可重排、可审计的晶体生成路径，并明确指出 geometry-quality modeling 是下一步的主要瓶颈。

## Claim-evidence map

- Claim: CIF file-record order 与晶体学约束顺序不同。  
  Evidence: Methods representation 中将 CIF 转换为 formula_counts -> SG -> wa_table -> free_params/lattice -> CIF reconstruction。  
  Status: supported by implementation.

- Claim: SymCIF 可以把生成错误拆解为 WA coverage、validity 和 geometry mismatch。  
  Evidence: Results 草稿和 evidence table 中包含 WA_hit、skeleton_hit、render_success、eval_timeout、RMSE 和 failure audit。  
  Status: supported.

- Claim: hybrid-prior reranking 提升 MP-20 top-1/top-5 match。  
  Evidence: structured test match@1/@5 = 60.90%/73.68%，current order 为 51.29%/66.11%。  
  Status: supported.

- Claim: SymCIF 全面优于 CrystaLLM-a。  
  Evidence: 不充分；输入条件不同，RMSE 仍落后。  
  Status: should not be claimed.

## Assumptions and missing inputs

- TODO: Introduction 中涉及 prior CIF language models、crystal generation 和 symmetry-based crystal representation 的句子需要补正式引用。
- TODO: 如果主文要更强地说 “causal order improves performance”，需要在 Results 中明确链接 representation ablation 或 coverage/match 诊断。
- TODO: 如果目标期刊要求 Introduction 不出现太多结果数字，可以将最后一段的数值压缩到一句，并把细节移到 Results。
