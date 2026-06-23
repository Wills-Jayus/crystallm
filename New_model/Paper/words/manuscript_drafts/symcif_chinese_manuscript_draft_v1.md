# SymCIF 论文中文初稿 v1

> 本稿基于当前 `Paper/words` 与 `Paper/wordsmu` 中已有的路线图、证据表、Introduction、Methods representation 和 Results 草稿整理而成。文中带 `TODO` 的位置表示仍需要补充引用、实验细节或最终数值。当前版本刻意避免把 SymCIF 的结果写成“同输入设定下全面超过 CrystaLLM-a”，因为当前 SymCIF 主结果使用 composition + ground-truth space group，而公开 CrystaLLM-a 参考是 composition-only。

## 题目

SymCIF：面向晶体结构生成的对称性条件表示与可诊断生成流程

## 摘要

晶体结构生成通常以 CIF 文本为输出目标，因此语言模型需要同时学习化学组成、空间群、Wyckoff 位点、坐标、晶胞参数以及 CIF 文件序列之间的复杂关系。然而，标准 CIF 文件的记录顺序主要服务于数据存储和交换，并不等同于晶体结构形成中的约束顺序。具体而言，晶胞参数和原子位点在文件中可能先于或穿插于对称性信息出现，而在晶体学上，组成、空间群、Wyckoff 位点和自由坐标参数之间存在更强的层级约束。本文提出 SymCIF，一种面向晶体生成的对称性条件表示，将原始 CIF 解析为由组成、空间群、Wyckoff 模板、位点对称性、自由坐标参数和晶胞组成的结构化记录，并在生成后重建为 CIF 进行验证。

在 MP-20 测试集上，SymCIF 在使用真实空间群作为条件的设置下取得了较强的 top-1 结构恢复性能。基于 hybrid prior 的 SymCIF 流程在 8,893 个成功结构化解析的 MP-20 测试样本上达到 match@1 = 60.90%、match@5 = 73.68%，对应 RMSE@1 = 0.0573、RMSE@5 = 0.0615；按原始 9,046 个 MP-20 测试样本计入 153 个结构化失败样本后，match@1 = 59.87%、match@5 = 72.43%。相较于当前顺序的 SymCIF 结果，hybrid prior 明显提高 Wyckoff assignment 覆盖率，并带来 top-1 和 top-5 match 的同步提升。与此同时，SymCIF 的 RMSE 仍高于公开 CrystaLLM-a 参考值，说明当前主要改进来自离散对称性候选的覆盖和排序，连续几何质量仍是限制因素。本文的结果表明，将 CIF 语言建模问题改写为遵循晶体学因果顺序的结构化生成任务，可以提升生成流程的可诊断性和 top-1 恢复能力，但仍需要在空间群预测、公平同输入比较和几何精修方面进一步完善。

TODO: 补充 CrystaLLM-a、MP-20、CIF、StructureMatcher、晶体生成模型等引用。

## 1. 引言

晶体结构生成是材料发现中的基础问题之一。一个生成模型若能从化学组成或更少的先验条件出发给出合理的三维晶体结构，就可以为后续的结构筛选、性质预测和第一性原理计算提供候选空间。近年来，基于语言模型的晶体生成方法将 CIF 视为一种可序列化的结构文本，并通过自回归建模直接生成 CIF 文件。这个范式具有实现简单、数据来源丰富、可利用大规模序列模型等优势，但也引入了一个关键问题：CIF 文件的文本顺序并不天然等同于晶体学约束的生成顺序。

标准 CIF 的记录顺序主要服务于人类可读性、软件兼容性和数据交换。一个 CIF 文件通常包含 data block、晶胞参数、化学式、对称操作和 atom_site loop 等字段。语言模型按文件顺序生成这些字段时，需要在局部文本预测中隐式维持全局一致性。例如，化学式需要与原子位点计数一致，空间群需要与对称操作和 Wyckoff 位点一致，坐标需要落在相应 Wyckoff 轨道允许的自由参数形式内，晶胞参数还需要与最终三维结构共同决定匹配误差。若这些关系只通过普通 token 序列学习，模型错误很容易表现为公式不一致、空间群不一致、位点展开失败或生成结构无法匹配真实结构。

从晶体学角度看，晶体结构具有更明确的约束层级。化学组成给出元素和原子计数目标；空间群决定允许的对称操作、Wyckoff 字母、位点多重度和位点对称性；Wyckoff 轨道分配决定哪些元素占据哪些对称轨道；在此基础上，自由坐标参数才具有确定含义；最后，晶胞参数与展开后的原子坐标共同构成可写回 CIF 的三维结构。这一顺序并不是普通文件记录顺序，而是更接近结构生成中应满足的约束传播顺序。

本文提出 SymCIF，目标不是简单替换 CIF 格式，而是将 CIF 语言建模问题拆解为一个遵循晶体学约束顺序的结构化生成与重建流程。SymCIF 将原始 CIF 转换为包含组成、空间群、Wyckoff 表、自由参数和晶胞信息的中间表示；生成阶段优先处理离散对称性选择，再处理连续几何参数；最终再将结构化结果渲染回 CIF，并用公式、原子数、空间群、StructureMatcher match 和 RMSE 等指标进行验证。这种设计使错误不再只是“生成文本不对”，而可以被定位到 Wyckoff 覆盖、轨道分配、自由参数提取、几何渲染或结构匹配等具体环节。

我们在 MP-20 上评估 SymCIF 的结构恢复能力。当前最强结果来自使用真实空间群作为条件的 hybrid-prior SymCIF 流程。在结构化成功的 8,893 个 MP-20 测试样本上，该流程取得 match@1 = 60.90%、match@5 = 73.68%；将 153 个结构化失败的原始测试样本计为失败后，match@1 = 59.87%、match@5 = 72.43%。这些结果说明，当空间群条件给定时，显式建模 Wyckoff assignment 并进行基于训练集先验的候选排序，可以显著改善 top-1 恢复表现。

本文同时强调该结果的边界。公开 CrystaLLM-a MP-20 参考是 composition-only 设置，而当前 SymCIF 使用 composition + ground-truth space group，因此二者不是严格同输入比较。SymCIF 的 match@1 在该设置下高于公开 CrystaLLM-a 参考的 55.85%，但其 RMSE@1 = 0.0573 仍高于 CrystaLLM-a 参考的 0.0437，说明连续结构质量尚未达到更强基线水平。因此，本文的主要结论应表述为：SymCIF 证明了晶体学因果顺序和对称性条件表示能够提高生成流程的可诊断性，并在给定空间群条件下改善 MP-20 的 top-1 结构恢复；它尚不能被表述为同输入条件下全面超过现有方法。

## 2. 结果

### 2.1 SymCIF 将 CIF 文本生成重排为晶体学约束顺序

原始 CIF 语言建模流程通常按照文件记录顺序展开：data block 之后写入晶胞参数、化学式、对称操作和 atom_site loop，然后由自回归语言模型继续预测后续 token。这种顺序对文件存储是自然的，但对结构生成并不理想。模型必须在已经写出的文本和后续全局约束之间维持一致性，任何局部错误都可能在后续字段中放大。

SymCIF 采用不同的结构化顺序：composition → space group → Wyckoff template → site symmetry / enumeration → coordinates → lattice → CIF reconstruction → validation。该顺序首先确定离散约束，再生成连续参数，最后重建 CIF。与普通文本顺序相比，这一流程把晶体学中最强的组合约束提前，使得候选生成可以在合法 Wyckoff 轨道和元素计数约束内进行。

这一设计的直接价值是可诊断性。对于每个候选结构，SymCIF 不仅能给出最终 CIF，还能记录 Wyckoff assignment 是否命中、skeleton 是否命中、公式和空间群是否一致、坐标展开是否成功、渲染是否成功以及 StructureMatcher 是否匹配。因此，实验失败可以被分解为离散位点覆盖不足、候选排序错误、连续几何质量不足或评估失败，而不是仅停留在“生成 CIF 未匹配”的层面。

### 2.2 Hybrid prior 提高了 MP-20 上的 Wyckoff assignment 覆盖

在 MP-20 测试集中，当前顺序的 SymCIF 结果已经显示出一个明显瓶颈：match@5 与 Wyckoff assignment 覆盖高度相关。当前顺序下，结构化测试集的 WA_hit@1/@5 分别为 49.80% 和 71.27%，match@1/@5 分别为 51.29% 和 66.11%。这说明 top-k 中是否包含正确或等价的 Wyckoff assignment，是 match 能否提升的主要前提之一。

Hybrid-prior 方法通过利用训练集中的 Wyckoff 组合先验和候选排序信息，提高了测试样本中正确 Wyckoff assignment 的前排覆盖。最终结果中，WA_hit@1/@5 提升到 59.02% 和 81.66%，对应 match@1/@5 提升到 60.90% 和 73.68%。这一变化说明，SymCIF 当前 top-1 性能的关键提升不是来自重新训练大型语言模型，而是来自对离散对称性候选空间的更有效排序。

这一结果也揭示了剩余瓶颈。Hybrid-prior 的 WA_hit@5 为 81.66%，但 match@5 为 73.68%，二者之间仍存在约 8 个百分点的差距。这部分差距意味着：即使 top-5 中已经出现正确或接近正确的 Wyckoff assignment，坐标参数、晶胞参数、结构渲染质量或 StructureMatcher 判据仍可能导致最终结构不匹配。因此，进一步提升 match@5 不能只依赖增加 WA 覆盖，还需要改进 geometry scoring、自由参数生成和候选重排。

### 2.3 在给定空间群条件下，SymCIF 在 MP-20 上提高 top-1 结构恢复

我们在 MP-20 测试集上评估了 SymCIF 的主流程。对于 8,893 个能够成功转换为 SymCIF 表示的测试样本，hybrid-prior SymCIF 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。若以原始 MP-20 测试集的 9,046 个样本为分母，并将 153 个结构化失败样本计为失败，则 match@1 = 59.87%、match@5 = 72.43%。

与当前顺序的 SymCIF baseline 相比，hybrid-prior 的提升是明确的。当前顺序在结构化测试集上为 match@1 = 51.29%、match@5 = 66.11%、RMSE@1 = 0.0668、RMSE@5 = 0.0742。Hybrid-prior 将 match@1 提高 9.61 个百分点，将 match@5 提高 7.57 个百分点，同时降低 RMSE。这说明，重新排序候选并优先恢复更可能的 Wyckoff assignment，可以同时改善离散匹配率和匹配结构的平均几何误差。

与公开 CrystaLLM-a MP-20 参考相比，SymCIF 的 match@1 数值高于其 55.85% 的公开结果。然而，这一比较必须谨慎解释：SymCIF 当前主结果使用真实空间群作为输入条件，而 CrystaLLM-a 公开参考是 composition-only。更准确的表述是，在 symmetry-conditioned 设置下，SymCIF 的 MP-20 top-1 match 超过了公开 composition-only CrystaLLM-a 参考值；这不能证明 SymCIF 在完全相同输入条件下全面优于 CrystaLLM-a。

### 2.4 连续几何质量仍然是主要短板

尽管 SymCIF 提高了 match@1 和 match@5，但 RMSE 结果显示连续结构质量仍未达到更强基线水平。Hybrid-prior SymCIF 的 RMSE@1/@5 为 0.0573/0.0615，而公开 CrystaLLM-a 的 MP-20 RMSE@1 为 0.0437，match@20 对应 RMSE 为 0.0395。换言之，SymCIF 更容易在 top-1 找到可匹配结构，但匹配结构与真实结构之间的几何偏差仍然偏大。

这一现象与错误审计一致。SymCIF 的离散 Wyckoff assignment 覆盖已经较高，尤其在 top-5 下 WA_hit 达到 81.66%，但 match@5 仍停留在 73.68%。因此，后续提升应优先集中在不依赖评估标签的 geometry-quality predictor、坐标自由参数重排、晶胞参数精修以及候选结构的物理合理性筛选上。单纯继续扩大 WA 候选数量可能会提高候选覆盖，但如果几何质量不改善，RMSE 和有效 match 的提升会受到限制。

### 2.5 评估覆盖较完整，但结论仍以 MP-20 为主

本轮 MP-20 full-test 评估覆盖了 8,893/8,893 个结构化样本；top-5 评估对应 44,465 条候选记录。评估中 eval_timeout@1/@5 为 0，render_success@1 为 100.00%，render_success@5 为 99.10%。这些结果说明当前 MP-20 主表的工程执行较稳定，主要结论不是由大规模超时或渲染缺失造成的。

同时，当前证据仍以 MP-20 为主。MPTS-52 上的早期 WA20xgeom1 结果并未构成强正向泛化证据，其 match@20 低于公开 CrystaLLM-a 参考，且 RMSE 更差。因此，本文不应声称 SymCIF 已经在多个数据集上普遍超过现有方法。更稳妥的结论是：MP-20 上的结果支持 SymCIF 表示和 hybrid-prior 排序的有效性；跨数据集泛化仍需要用同一套修复后的流程重新评估。

## 3. 方法

### 3.1 方法概览

SymCIF 的核心思想是将 CIF 文本中的隐式晶体学约束显式化。给定一个 CIF 文件，流程首先将其解析为标准化晶体结构，并提取化学组成、空间群、Wyckoff 位点、自由坐标参数和晶胞参数。随后，模型或候选生成器在这一结构化空间中生成或重排候选。每个候选再被渲染回标准 CIF，并通过结构解析、公式一致性、空间群一致性、原子数一致性、StructureMatcher match 和 RMSE 进行评估。

### 3.2 SymCIF 表示

一个 SymCIF 记录包含以下主要字段。

`formula_counts` 记录常规晶胞中的元素计数，用于定义元素种类和计数约束。该字段是后续 Wyckoff 多重度组合的目标，而不是仅作为文本中的化学式出现。

`sg` 和 `sg_symbol` 分别记录空间群编号和符号。空间群决定可用的对称操作、Wyckoff 字母、多重度和位点对称性，是 SymCIF 离散候选空间的核心条件。当前主实验使用 ground-truth space group 作为输入条件。

`wa_table` 记录 Wyckoff assignment。每一行通常包含 `element`、`sg`、`letter`、`multiplicity`、`site_symmetry`、`enumeration`、`orbit_id`、`representative_expr`、`free_symbols` 和 `free_params`。其中，`element` 表示元素占据，`letter` 和 `multiplicity` 表示 Wyckoff 字母及其多重度，`site_symmetry` 表示位点对称性，`free_symbols` 和 `free_params` 表示该轨道允许的自由坐标参数及其具体数值。

为支持审计，记录中还保留 `source_coord`、`mapped_coord`、`extraction_method`、`extraction_residual`、`expansion_count_after_reextract`、`expansion_ok`、`extraction_success` 和 `fallback_reason` 等字段。这些字段不一定都作为模型输入，但用于确认从 CIF 到 SymCIF 的转换是否可靠。

SymCIF 还定义两个规范化 key。`canonical_skeleton_key` 由排序后的 orbit id 构成，不包含元素信息，用于比较两个结构是否具有相同的 Wyckoff skeleton。`canonical_wa_key` 在 orbit id 基础上加入元素信息，用于比较完整 Wyckoff assignment 是否一致。这两个 key 是分析 WA_hit 和 skeleton_hit 的基础。

### 3.3 从 CIF 到 SymCIF 的转换流程

转换首先读取 CIF 并构造晶体结构。结构随后被标准化为常规晶胞表示，以减少不同 CIF 写法造成的表面差异。接着，流程从标准化结构中计算 `formula_counts`，并通过空间群分析器识别空间群编号和空间群符号。

在空间群确定后，流程对结构中的等价原子位点进行分组，识别每组对应的 Wyckoff 字母、多重度和位点对称性。每个等价位点组被映射到预先构建的 `OrbitToken` 或等价轨道定义中，从而得到稳定的 `orbit_id` 和 `representative_expr`。

对于含有自由坐标的 Wyckoff 轨道，流程从原始坐标中反推出自由参数。其基本思想是在空间群对称操作和模 1 周期边界下，将观察到的坐标映射到代表性表达式允许的自由变量形式，并选择残差最小的参数解。随后，流程重新展开轨道，检查展开原子数是否等于 Wyckoff 多重度，检查元素计数是否闭合，并记录提取残差和失败原因。

最后，所有轨道行、晶胞参数和全局元数据被写入 SymCIF 结构化记录。若转换失败，样本会被标记为 structured extraction failure；在原始测试集分母下，这些样本被计为失败。

TODO: 补充具体使用的 pymatgen/spglib 接口名称、容差参数、标准化选项和 free-parameter 求解细节。

### 3.4 为什么该表示符合晶体学因果顺序

SymCIF 的字段顺序并不是任意工程拆分，而是对应晶体结构约束的自然传播顺序。化学组成首先定义元素和计数目标；空间群随后定义允许的对称操作和 Wyckoff 轨道集合；Wyckoff skeleton 和 assignment 决定哪些多重度组合能够满足组成；在具体轨道确定后，自由坐标参数才具有合法表达式；最后，晶胞参数和展开后的坐标共同形成可写回 CIF 的三维结构。

相比之下，原始 CIF 文本顺序可能先给出晶胞参数，再给出化学式和对称操作，最后列出 atom_site loop。这种顺序虽然适合文件记录，但不适合把合法结构的约束逐级传递给生成模型。SymCIF 将离散对称性约束前置，减少了组成、空间群、Wyckoff 位点、坐标和晶胞之间的错配空间。

### 3.5 候选生成与 hybrid-prior 排序

当前实验中的 SymCIF 流程对每个测试样本生成多个候选 Wyckoff assignment 和几何变体。候选首先需要满足空间群和组成计数约束，然后根据训练集先验、候选排序和结构化规则进行选择。Hybrid-prior 方法的目标是提高 top-k 中正确 Wyckoff assignment 的覆盖率，尤其是提高 top-1 的离散对称性命中概率。

实验结果显示，hybrid-prior 将结构化测试集上的 WA_hit@1/@5 从 49.80%/71.27% 提升到 59.02%/81.66%，并带来 match@1/@5 的同步提升。这说明训练集中的 Wyckoff 组合统计规律对 MP-20 测试集具有可利用的分布内信息。

TODO: 补充 hybrid-prior 的精确 scoring 公式、训练集统计项、平滑方式、tie-breaking 规则，以及是否使用验证集调参。

### 3.6 CIF 重建与验证

候选 SymCIF 记录生成后，renderer 根据空间群、Wyckoff 轨道、自由坐标参数和晶胞参数展开完整原子坐标，并写回 CIF。重建后会重新读取 CIF，检查可读性、公式一致性、原子数一致性、空间群一致性和 strict validity。通过这些基本检查后，候选结构与真实结构进行 StructureMatcher 比较；若匹配成功，则记录 RMS/RMSE。

Top-k match 指标按样本聚合：只要前 k 个候选中至少一个与真实结构匹配，该样本即计为 match@k 成功。RMSE 在匹配样本上计算。对于原始 MP-20 测试集中未能转换为结构化记录的样本，match 率分母中将其计为失败，但 RMSE 仍只在具有匹配 RMS 的样本上统计。

TODO: 补充 StructureMatcher 参数、RMSE 计算定义、超时设置和随机种子。

## 4. 讨论

本文的主要贡献是把 CIF 语言建模中的隐式结构约束改写为显式的对称性条件表示。SymCIF 并不否认 CIF 文本作为晶体结构交换格式的价值，而是指出 CIF 文件顺序不应被直接等同于结构生成顺序。通过将组成、空间群、Wyckoff 位点、自由参数和晶胞参数分离，SymCIF 使生成错误可以被定位到具体晶体学环节，从而更容易设计候选生成、排序和修复策略。

MP-20 结果支持这一设计。在给定真实空间群条件下，hybrid-prior SymCIF 在结构化测试集上达到 match@1 = 60.90%，高于当前顺序 SymCIF 的 51.29%，也高于公开 CrystaLLM-a MP-20 match@1 = 55.85% 的参考值。更重要的是，这一提升伴随 WA_hit 的提高，说明性能增益来自可解释的离散对称性覆盖改善，而不是评估偶然性或隐藏的渲染问题。

不过，SymCIF 的局限同样清楚。首先，当前主结果使用 ground-truth space group，因此与 composition-only 方法不是同输入比较。若要证明 SymCIF 作为端到端生成方法优于现有 composition-only 基线，需要补充空间群预测或在所有方法中统一输入空间群。其次，RMSE 仍高于 CrystaLLM-a，说明当前匹配结构的连续几何质量仍不够强。第三，当前强证据主要来自 MP-20，跨数据集泛化尚未充分建立。第四，hybrid-prior 利用训练集分布信息，后续需要通过验证集调参、bootstrap 置信区间和分布外测试确认其稳定性。

未来工作可以沿三个方向推进。第一，将空间群预测纳入 SymCIF，使流程从 composition-only 条件出发完成公平比较。第二，开发不依赖 StructureMatcher 标签的 geometry-quality predictor，用于在生成阶段选择更好的自由坐标和晶胞参数。第三，扩展到 MPTS-52 或其他材料数据集，并补充 DFT relaxation 或能量稳定性评估，从而判断 SymCIF 生成结构是否不仅能匹配数据库结构，也能通过物理稳定性检验。

## 5. 结论

SymCIF 将晶体 CIF 生成从文件记录顺序重排为晶体学约束顺序，使组成、空间群、Wyckoff assignment、自由坐标和晶胞参数之间的关系显式可控。MP-20 实验表明，在使用真实空间群作为条件的设置下，SymCIF 结合 hybrid prior 可以显著提升 Wyckoff assignment 覆盖，并将结构化测试集 match@1 提高到 60.90%、match@5 提高到 73.68%。这一结果说明，对称性条件表示是改善晶体结构生成可诊断性和 top-1 恢复能力的有效方向。与此同时，当前 SymCIF 尚未解决连续几何质量不足和同输入公平比较问题，因此其最稳妥的定位是一个有解释力的 symmetry-conditioned generation framework，而不是已经全面超过所有 composition-only 基线的最终方法。

## 当前不能过度声称的内容

1. 不能声称 SymCIF 在同输入条件下全面超过 CrystaLLM-a，因为当前 SymCIF 使用 ground-truth space group，而公开 CrystaLLM-a 参考是 composition-only。
2. 不能声称 SymCIF 的几何精度超过 CrystaLLM-a，因为当前 RMSE@1 = 0.0573，仍高于 CrystaLLM-a 参考 RMSE@1 = 0.0437。
3. 不能声称 SymCIF 已经是通用 SOTA，因为强结果主要来自 MP-20，MPTS-52 尚未用最终流程建立强正向证据。
4. 不能声称 DFT stability 已被证明，除非补充完整 DFT relaxation 或稳定性评估。
5. 不能把 diagnostic/oracle rerank 的提升写成真实 inference 提升，除非该 rerank 不使用评估标签或真实结构信息。

## 下一版需要补充的 TODO

1. 补充完整引用：CIF 标准、MP-20、CrystaLLM、晶体生成、StructureMatcher、spglib/pymatgen。
2. 补充 hybrid-prior 的精确定义和可复现实验命令。
3. 补充 StructureMatcher 和 RMSE 的参数表。
4. 补充 bootstrap 置信区间，至少覆盖 match@1、match@5、RMSE@1、RMSE@5。
5. 增加 figure/table 编号：方法示意图、MP-20 主结果表、WA coverage 分析图、错误分解图。
6. 明确所有实验的输入条件：composition-only、composition + oracle SG、或其他条件。
7. 若要投高水平期刊，需要补充 fairness baseline：给 CrystaLLM-a 同样的 oracle SG，或让 SymCIF 从 composition-only 预测 SG。
