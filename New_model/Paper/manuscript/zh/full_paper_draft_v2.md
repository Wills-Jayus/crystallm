# SymCIF：面向晶体结构生成的对称性条件表示与 Wyckoff 分配重排

# 摘要

晶体结构生成要求模型同时满足化学组成、空间群对称性、Wyckoff 位点占据、原子坐标和晶胞参数之间的耦合约束。基于 CIF 的语言模型为这一任务提供了直接而通用的文本生成接口，但标准 CIF 的字段顺序主要服务于结构记录和数据交换，并不等同于晶体学约束的组织顺序。当晶体结构被作为普通长文本序列生成时，模型需要在隐式状态中同时维持组成、空间群、Wyckoff 位点和几何参数之间的一致性；一旦生成失败，错误往往难以归因到离散对称性选择、元素占据、坐标参数还是晶胞几何。

本文提出 SymCIF，一种用于晶体结构生成的对称性条件结构化表示。SymCIF 将原始 CIF 中的信息重组为化学组成、空间群、Wyckoff assignment、自由坐标参数和晶格参数，并将生成过程分解为 Wyckoff assignment selection 与 geometry rendering 两个相互衔接的阶段。该表示不是简单的 prompt engineering，也不是对 CIF 文本格式的表层改写，而是将晶体生成中的信息组织方式从文件记录顺序调整为更接近晶体学约束传播的顺序。

在 MP-20 条件晶体结构生成/结构重建任务上，SymCIF 在 composition + oracle ground-truth space group 设置下取得了较强的 top-1 恢复表现。对于 8,893 个成功结构化转换的测试样本，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。按原始 MP-20 test 的 9,046 个样本计入 153 个结构化转换失败样本后，match@1 = 59.87%、match@5 = 72.43%。与 SymCIF current order 相比，hybrid-prior 将 WA_hit@1/@5 从 49.80%/71.27% 提高到 59.02%/81.66%，并同步提高 match@1/@5。与公开 CrystaLLM-a MP-20 reference 比较时，SymCIF 的 match@1 高于其 55.85%，但该比较并非同输入设置，因为 CrystaLLM-a reference 是 composition-only，而 SymCIF 使用真实空间群作为条件；同时，SymCIF 的 RMSE@1 仍差于 CrystaLLM-a reference 的 0.0437。上述结果表明，显式的对称性分解和 Wyckoff 分配重排可以改善 MP-20 上的 top-1 recovery，但连续几何质量仍是主要瓶颈。

# 引言

晶体结构生成是材料建模中的基础问题之一。与一般分子或文本生成不同，晶体结构具有周期性、空间群对称性和 Wyckoff 位点约束。一个生成结果即使能够被写成合法的 CIF 文件，也不一定代表它在晶体学上自洽，更不一定能够与目标结构匹配。对于条件晶体结构生成任务，模型不仅需要生成正确元素种类和原子数，还需要恢复合理的空间群、位点多重度、元素到 Wyckoff orbit 的分配、自由坐标参数和晶胞几何。match rate 和 RMSE 因此比单纯的文本可读性或格式有效性更能反映结构恢复能力。

CIF language modeling 为晶体生成提供了一条重要路径。该类方法将晶体结构表示为标准 CIF 文本，并使用自回归语言模型学习 CIF 文件中的晶胞参数、化学式、对称操作和 atom_site loop。CrystaLLM 等工作证明，直接以 CIF 为生成对象可以产生可解析的晶体结构候选，并为 composition-conditioned crystal generation 建立了有影响力的基线。本文并不否定这一范式；相反，SymCIF 以 CIF 生成的可行性为基础，进一步讨论 CIF 的线性文件顺序是否适合作为晶体学约束的建模顺序。

标准 CIF 的字段顺序主要面向结构记录，而不是生成模型的约束传播。CIF 文件通常先后包含 data block、cell parameters、formula、symmetry operations 和 atom_site loop 等字段。这样的记录顺序对数据库和软件解析是合理的，但在晶体学上，组成、空间群、Wyckoff 位点、坐标自由度和晶格之间存在更明确的依赖关系。空间群决定可用的 Wyckoff orbits 和 site symmetries；Wyckoff multiplicities 决定元素计数能否被合法覆盖；坐标自由参数只有在具体 orbit 被选定后才具有定义；晶胞参数与展开后的原子坐标共同决定最终结构匹配误差。若这些关系全部混合在一个长 CIF token sequence 中，模型可能生成格式正确但晶体学不一致的结构。

这一错位带来的直接问题是错误不可诊断。一个失败的 CIF 候选可能拥有正确化学式，却选择了错误的 Wyckoff/site 组合；也可能写入了正确空间群字段，但重新解析后实际结构对称性不一致；还可能在离散位点组合上接近目标结构，但因自由坐标参数或晶胞几何偏差而无法通过 StructureMatcher。单纯依靠后处理或 reranking 很难解决所有这些问题，因为不同失败来源对应不同改进方向。若正确 Wyckoff assignment 从未被采样到，后处理无法凭空恢复正确离散骨架；若正确 assignment 已经出现但几何渲染不足，则继续增加离散候选数量也难以降低 RMSE。

本文提出 SymCIF，将晶体生成重新组织为对称性条件下的结构化流程。给定 composition 和当前实验中的 oracle ground-truth space group，SymCIF 首先构建以 Wyckoff assignment 为核心的结构表示，再将自由坐标和晶格参数渲染回 CIF。该流程将晶体生成分解为两个可分别分析的环节：离散的 Wyckoff assignment selection，以及连续的 geometry rendering。通过 canonical skeleton key 和 canonical WA key，SymCIF 能够判断候选是否覆盖了正确的 Wyckoff orbit skeleton 和元素占据；通过 formula_ok、SG_ok、render_success、match rate 和 RMSE，SymCIF 能够进一步定位结构失败是否来自公式、空间群、渲染或几何匹配。

本文在 MP-20 上评估 SymCIF。需要提前说明的是，当前主实验使用 composition + oracle ground-truth space group，而公开 CrystaLLM-a MP-20 reference 使用 composition-only 条件。因此，本文不主张 SymCIF 在同输入条件下全面超过 CrystaLLM-a，也不将当前结果表述为晶体生成任务的 state-of-the-art。本文的核心主张更具体：在真实空间群给定的条件下，SymCIF 的结构化表示和 train-only hybrid-prior reranking 能够提高 MP-20 top-1 recovery，并揭示当前剩余误差主要来自连续几何质量，而不是单纯来自 CIF 文本格式或评估缺失。

# 结果

## 问题定义与基线诊断

本文将 MP-20 作为条件晶体结构生成/结构重建任务来评估，而不是作为无条件新材料发现任务。每个测试样本的目标是在给定组成和真实空间群条件下生成候选晶体结构，并判断候选是否能够与目标结构匹配。候选首先被重建为 CIF，再检查可读性、公式一致性、原子数一致性、空间群一致性和渲染成功率。最终恢复能力由 StructureMatcher match 和 RMSE 衡量。valid 或 strict_valid 可以说明候选在格式或部分几何检查上合理，但不能单独证明结构预测正确。

基线诊断显示，传统 CIF 语言建模的核心困难并不是“无法写出 CIF”，而是很难显式维持 CIF 字段背后的晶体学一致性。CrystaLLM 类方法已经证明 CIF 语言建模是可行的；但按照文件记录顺序生成时，composition、space group、Wyckoff sites、coordinates 和 lattice 之间的耦合关系仍主要由模型隐式学习。这样的生成过程可能产生几类典型失败：化学式复制正确但 Wyckoff/site 组合错误；空间群字段存在但生成结构实际对称性不一致；局部坐标或原子位点使 StructureMatcher 无法匹配；晶胞参数与坐标共同导致较大 RMSE。这些失败模式提示，评价晶体生成方法时必须区分文本有效性、对称性一致性、候选覆盖和几何质量。

基于这一诊断，本文将问题拆为两个层次。第一，模型是否在 top-k 候选中产生了正确或等价的 Wyckoff assignment；第二，在正确或接近正确的 assignment 条件下，坐标和晶格是否足以重建匹配目标结构的三维几何。这个分解为 SymCIF 表示和后续机制分析提供了实验基础。

## SymCIF 表示重构晶体生成过程

SymCIF 将晶体生成从 CIF file-record order 重组为 crystallographic-constraint-oriented order。原始 CIF 语言模型通常按 data block、cell parameters、formula、symmetry operations 和 atom_site loop 的顺序生成文本。SymCIF 则按照 composition、space group、Wyckoff assignment、free coordinates、lattice、CIF reconstruction 和 validation 的顺序组织信息。这里的“顺序”并不声称模拟真实晶体形成过程，而是指工程上更接近晶体学约束传播的表示顺序。

在 SymCIF 中，一个晶体结构被表示为包含 composition、space group、Wyckoff assignment table、free parameters 和 lattice 的结构化记录。`formula_counts` 记录 conventional cell 中每种元素的整数计数；`sg` 和 `sg_symbol` 记录空间群编号与符号；`wa_table` 的每一行对应一个元素占据的 Wyckoff orbit，并记录 Wyckoff letter、multiplicity、site symmetry、enumeration、orbit identifier、representative coordinate expression 和自由坐标参数；`lattice` 记录晶胞长度、角度和体积。为支持候选比较，SymCIF 还定义 `canonical_skeleton_key` 和 `canonical_wa_key`。前者只比较选中了哪些 Wyckoff orbits，后者进一步比较元素到 orbit 的分配。

这种表示使生成过程从自由文本采样转化为可审计的结构候选生成。一个候选不仅可以给出最终 CIF，还可以回答更细的问题：它是否覆盖了目标 skeleton；是否覆盖了目标 WA；是否保持公式和原子数；是否保持空间群；是否能成功渲染；是否通过 StructureMatcher；若匹配成功，RMSE 如何。这些中间指标使性能提升和失败机制可以被分开分析。SymCIF 因此不是普通 prompt engineering，而是将晶体生成中的信息组织方式重新编码为对称性条件下的离散选择与连续渲染。

## MP-20 上的结构恢复结果

在 MP-20 structured test 的 8,893 个样本上，SymCIF current order 达到 match@1 = 51.29%、match@5 = 66.11%，RMSE@1 = 0.0668、RMSE@5 = 0.0742。使用 hybrid-prior selector 后，SymCIF 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。相较于 current order，match@1 提高 9.61 个百分点，match@5 提高 7.57 个百分点，并且匹配样本上的 RMSE 同时下降。

为了避免只报告结构化成功子集，本文同时采用 original-test adjusted 口径。原始 MP-20 test 包含 9,046 个样本，其中 153 个样本在当前 SymCIF 转换中属于 structured extraction failures。将这些样本计为失败后，SymCIF hybrid-prior 仍达到 match@1 = 59.87%、match@5 = 72.43%。这说明主结果不是简单由移除结构化失败样本造成的。

与公开 CrystaLLM-a MP-20 reference 比较时，输入条件必须被明确列出。CrystaLLM-a published reference 是 composition-only 设置，报告 match@1 = 55.85%、match@20 = 75.14%、RMSE@1 = 0.0437、RMSE@20 = 0.0395。SymCIF 当前主结果使用 composition + oracle ground-truth space group。因此，SymCIF hybrid-prior 的 match@1 虽高于 CrystaLLM-a published reference，但该比较不是 same-input comparison。更重要的是，SymCIF 的 RMSE@1 = 0.0573，仍差于 CrystaLLM-a 的 0.0437。这一结果支持的结论应被限定为：在真实空间群给定的设置下，SymCIF 提供了更强的 MP-20 top-1 recovery；但当前方法尚未在几何精度上优于公开 CrystaLLM-a reference。

表 1. MP-20 主结果。SymCIF 使用 composition + oracle ground-truth space group；CrystaLLM-a published reference 使用 composition-only，因此不是同输入对比。

| 方法 | 输入条件 | 测试范围 | 样本数 | match@1 ↑ | match@5/20 ↑ | RMSE@1 ↓ | RMSE@5/20 ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SymCIF current order | composition + oracle SG | structured test | 8,893 | 51.29% | 66.11% @5 | 0.0668 | 0.0742 @5 |
| SymCIF hybrid-prior | composition + oracle SG | structured test | 8,893 | 60.90% | 73.68% @5 | 0.0573 | 0.0615 @5 |
| SymCIF hybrid-prior | composition + oracle SG | original-test adjusted | 9,046 | 59.87% | 72.43% @5 | 0.0573 | 0.0615 @5 |
| CrystaLLM-a published reference | composition only | published reference | 公开参考 | 55.85% | 75.14% @20 | 0.0437 | 0.0395 @20 |

## Wyckoff 分配重排解释主要增益

为了判断性能提升来自何处，本文比较了 current order 与 hybrid-prior 在 Wyckoff assignment coverage 上的差异。Current order 的 WA_hit@1/@5 为 49.80%/71.27%，而 hybrid-prior 提高到 59.02%/81.66%。与此同步，match@1/@5 从 51.29%/66.11% 提高到 60.90%/73.68%。这种同步变化表明，在当前 K<=5 的候选预算下，正确或等价 Wyckoff assignment 是否进入候选前列，是 top-1/top-5 恢复能力的重要决定因素。

Hybrid-prior selector 的设计原则是 train-only 和 inference-feasible。它使用训练集中可见的 WA/skeleton 统计规律，并结合已有候选排序信息，对候选进行重排；推理阶段不使用测试标签、StructureMatcher 结果或 oracle GT-WA。因此，它不同于 evaluator-derived diagnostic rerank。与此同时，该 prior 仍然依赖训练分布中的 assignment pattern，不能被解释为完全与训练分布无关的泛化机制。现有结果能够支持“WA coverage 改善与 match 改善一致”这一机制解释，但更细粒度的 selector ablation 仍需在后续版本中补充。

表 2. Current order 与 hybrid-prior 的覆盖和匹配对比。数值显示 WA coverage 与 match rate 同步提高。

| Selector | WA_hit@1 ↑ | WA_hit@5 ↑ | match@1 ↑ | match@5 ↑ | RMSE@1 ↓ | RMSE@5 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current order | 49.80% | 71.27% | 51.29% | 66.11% | 0.0668 | 0.0742 |
| hybrid-prior | 59.02% | 81.66% | 60.90% | 73.68% | 0.0573 | 0.0615 |

## 几何质量仍是主要瓶颈

WA coverage 的提升并未完全转化为最终结构匹配。Hybrid-prior 的 WA_hit@5 达到 81.66%，而 match@5 为 73.68%，二者之间存在约 8 个百分点的差距。这说明一部分样本虽然已经在 top-5 候选中包含正确或等价的 Wyckoff assignment，但最终仍因坐标自由参数、晶格参数、结构有效性或匹配判据而失败。

RMSE 结果进一步支持这一判断。SymCIF hybrid-prior 的 RMSE@1 为 0.0573，而 CrystaLLM-a published reference 的 RMSE@1 为 0.0437。虽然二者输入条件不同，不能直接作为同输入优劣判断，但这一差距足以说明 SymCIF 当前更强的部分主要在离散对称性候选排序和 top-1 recovery，而不是连续几何精度。后续若要进一步提升结构质量，需要引入 geometry-quality scoring、validity-aware reranking 或 geometry refinement，而不应只依赖增加 WA 候选数量。

## 失败模式与方法边界

MP-20 full evaluation 的工程覆盖较完整。当前评估覆盖 8,893/8,893 个 structured test samples；top-5 评估包含 44,465 条 candidate-level records；@1/@5 evaluation timeout 均为 0；render_success@1 = 100.00%，render_success@5 = 99.10%。这些结果说明主表不是由大规模渲染缺失、超时或不完整运行造成的。

失败分析显示，SymCIF 的困难并不由单一复杂度指标解释。现有审计显示，`n_sites>=6`、`n_sites>=12` 和 `extraction_hard` 等子集更难，而 high-multiplicity orbit 子集相对更强。这提示失败来源可能同时涉及候选覆盖、元素到 orbit 的分配、坐标自由度、晶格参数和结构标准化。换言之，SymCIF 的可审计性使失败位置更清楚，但并不意味着所有失败都能由简单后处理修复。

从方法边界看，当前 SymCIF 最可靠的结论是改善了真实空间群条件下的 MP-20 top-1 recovery，并提供了区分 WA coverage 和 geometry bottleneck 的分析框架。当前结果尚不足以支持跨 benchmark 普适优势、composition-only 条件下的同输入优势或 DFT 稳定性提升。

# 方法

## 数据集与任务定义

本文以 MP-20 为主要评估数据集。原始 MP-20 test 包含 9,046 个样本，其中 8,893 个样本成功转换为 SymCIF structured test，153 个样本属于 structured extraction failures。本文同时报告两种口径：structured-only 以 8,893 个结构化成功样本为分母，original-test adjusted 以 9,046 个原始测试样本为分母并将 153 个转换失败样本计为失败。

本文任务定义为 symmetry-conditioned crystal structure generation/reconstruction。输入为 chemical composition 和 oracle ground-truth space group，输出为一个或多个候选晶体结构。每个候选被重建为 CIF 后，与目标结构进行匹配评估。该设置用于检验真实空间群给定时，显式 Wyckoff assignment selection 和 geometry rendering 是否能够改善结构恢复；它不同于 composition-only crystal generation。

## CrystaLLM 基线设置

CrystaLLM-a 在本文中作为公开 CIF language modeling reference。当前稿件采用其 published MP-20 reference 数值：match@1 = 55.85%、match@20 = 75.14%、RMSE@1 = 0.0437、RMSE@20 = 0.0395。该 reference 的输入条件为 composition-only，而 SymCIF 主结果使用 composition + oracle ground-truth space group。因此，本文所有与 CrystaLLM-a 的对照均被表述为相对于公开 reference 的比较，而不是 same-input baseline comparison。

当前版本尚未完成给 CrystaLLM-a 加入相同 oracle space group 条件的重跑，也尚未完成 SymCIF 在 composition-only 条件下的完整评估。因此，CrystaLLM-a 在本文中的作用是提供公开参考水平，并帮助说明 SymCIF 当前结果的优势和边界。

## SymCIF 表示构建

SymCIF 将晶体结构表示为由组成、空间群、Wyckoff assignment、自由参数和晶格组成的结构化记录。每条记录包含 `formula_counts`、`sg`、`sg_symbol`、`wa_table`、`lattice`、`canonical_skeleton_key` 和 `canonical_wa_key`。其中，`formula_counts` 记录 conventional cell 中各元素整数计数；`sg` 和 `sg_symbol` 记录空间群；`lattice` 记录晶胞长度、角度和体积；`wa_table` 记录元素占据的 Wyckoff orbit。

`wa_table` 的每一行对应一个元素在一个 Wyckoff orbit 上的占据。该行记录 `element`、`letter`、`multiplicity`、`site_symmetry`、`enumeration`、`orbit_id`、`representative_expr`、`free_symbols` 和 `free_params`。其中，`representative_expr` 定义代表坐标表达式，`free_params` 给出从原始结构坐标反解得到的自由参数。为了支持转换审计，记录中还保存 source coordinate、mapped coordinate、extraction method、extraction residual 和 expansion correctness 等信息。

从 CIF 到 SymCIF 的转换首先读取 benchmark split 中的 CIF，并使用空间群分析器将结构转换为 conventional standard cell。随后，流程提取 conventional cell 中的元素计数，识别空间群编号、空间群符号和 symmetrized equivalent sites。每个等价位点组被映射到 Wyckoff lookup table 中对应的 orbit token，再从源分数坐标反解自由参数。反解后重新展开该 orbit，检查展开数量、源坐标包含关系、残差和元素计数闭合。通过检查的样本写入 structured SymCIF 数据集，失败样本在 original-test adjusted 口径中计为失败。

## 模型生成与候选重排

在当前实验中，生成流程被分解为 Wyckoff assignment candidate generation、hybrid-prior reranking 和 geometry rendering。给定 composition 与 oracle ground-truth space group，候选生成器产生多个 Wyckoff assignments 及其几何变体。Hybrid-prior selector 使用 train split 中的 Wyckoff assignment 或 skeleton 统计规律，并结合已有候选排序信息，对候选进行重排。该 selector 不使用测试标签、StructureMatcher outcome 或 oracle GT-WA，因此属于推理阶段可用的重排策略。

底层候选生成模型沿用当前项目中的 CrystaLLM/SymCIF 实验管线。本文不把模型训练细节作为新的方法贡献；主要关注表示转换、候选重排和渲染评估对结构恢复的影响。相关模型权重、训练数据划分、采样温度、候选数、随机种子和硬件设置记录于实验配置文件，并应随论文补充材料一并报告。

## CIF 重建、有效性检查与结构匹配

给定一个候选 SymCIF record，renderer 根据 `orbit_id` 找回对应 orbit token，并利用 `free_params` 计算代表坐标和 symmetry-expanded fractional coordinates。展开后的 atom rows 与 `formula_counts`、`sg`、`sg_symbol` 和 `lattice` 一起写回 CIF。重建后的 CIF 被重新解析，并检查 readable、formula_ok、atom_count_ok、SG_ok、valid/strict_valid 和 render_success。

结构恢复能力通过 match@k 和 RMSE@k 评估。对于每个样本，若 top-k 候选中至少一个候选通过 StructureMatcher 与目标结构匹配，则该样本计为 match@k 成功。RMSE@k 在 matched samples 上统计结构匹配误差。WA_hit@k 表示 top-k 中是否包含 ground-truth 或等价 canonical WA；skeleton_hit@k 表示 top-k 中是否包含正确 canonical skeleton。当前稿件中 StructureMatcher 的具体参数、RMSE 计算细节和超时设置以实验配置为准，应在最终版本中完整列出。

## 消融与审计设置

本文当前报告的核心消融是 SymCIF current order 与 SymCIF hybrid-prior 的比较。该比较同时报告 WA_hit、match 和 RMSE，用于判断 candidate reranking 是否提高正确 Wyckoff assignment 的前排覆盖，并观察覆盖变化是否转化为结构匹配提升。

Failure audit 记录 evaluation coverage、candidate-level record 数量、timeout、render success 和复杂子集表现。该审计用于区分工程执行问题、离散候选覆盖问题和连续几何质量问题。更细粒度的 selector ablation、geometry bottleneck ablation 和 train-frequency novelty analysis 属于后续需要补充的实验，不作为当前主结论的依据。

# 讨论

SymCIF 的主要意义在于改变晶体生成中的信息组织方式。传统 CIF language modeling 将结构作为文件顺序上的文本序列处理，而 SymCIF 将晶体表示为 composition、space group、Wyckoff assignment 和 geometry rendering 的组合。这样的重构使离散对称性选择和连续几何误差可以被分开观察，从而把“生成失败”拆解为更具体的晶体学问题。

MP-20 结果表明，这一表示在真实空间群给定时具有实际效果。Hybrid-prior 提高 WA_hit@1/@5，并同步提高 match@1/@5，说明正确 Wyckoff assignment 的前排覆盖是当前 top-1 recovery 的关键因素。该结果也解释了为什么 SymCIF 不应被理解为普通后处理：它的改进发生在候选结构的对称性分配层面，而不是仅仅在最终 CIF 文本上做格式修复。

同时，本文的结果也给出了清晰边界。SymCIF 的 RMSE 仍差于 CrystaLLM-a published reference，WA_hit@5 与 match@5 之间仍存在差距。这说明正确的离散位点组合只是结构恢复的必要条件之一，而不是充分条件。坐标自由参数、晶胞参数和几何合理性仍需要更强的建模。后续工作若要提高整体结构质量，应重点发展不依赖 evaluator oracle 的 geometry-quality predictor、validity-aware reranking 或结构精修模块。

与 CrystaLLM-a 的关系也需要审慎表述。CrystaLLM-a 展示了 CIF language modeling 的有效性，是本文的重要公开参考。SymCIF 在 oracle-SG setting 下取得更高 match@1，但输入条件不同，且 RMSE 落后。因此，当前结果更适合被理解为 symmetry-conditioned generation 的证据，而不是对 composition-only CIF language model 的全面替代。

# 局限性与未来工作

当前工作的首要局限是输入条件。SymCIF 主结果使用 composition + oracle ground-truth space group，而公开 CrystaLLM-a reference 使用 composition-only。该设置能够回答“空间群已知时，显式对称性分解是否有用”这一问题，但不能回答 SymCIF 是否在同输入条件下优于 CrystaLLM-a。后续需要补充同输入实验：或者给 CrystaLLM-a 同样的 oracle SG 条件，或者让 SymCIF 从 composition-only 条件预测或检索空间群。

第二个局限是连续几何质量。当前 SymCIF 的 match@1 表现较强，但 RMSE 仍落后于公开 CrystaLLM-a reference。WA_hit@5 与 match@5 的差距也表明，部分正确 Wyckoff assignment 没有转化为最终匹配。未来需要将 geometry rendering 从当前候选生成和重排中进一步分离出来，发展可推理使用的 geometry scoring 和 refinement。

第三个局限是分布依赖。Hybrid-prior 使用训练集中可见的 WA/skeleton 统计规律，因此可能更适合训练分布中常见的 assignment pattern。后续应按 train WA frequency、seen/unseen skeleton、rare/common space group、n_sites 和 num_elements 等维度分组评估，以判断该方法在低频或分布外结构上的可靠性。

第四个局限是统计和泛化证据仍需加强。当前主结果来自 MP-20 full-test evaluation，但尚未报告 bootstrap confidence intervals 或多随机种子结果。MPTS-52 或外部 benchmark 上的最终流程结果也尚未成为本文主证据。因此，本文不声称跨数据集普适优势，也不讨论 DFT stability 或热力学稳定性提升。

# 结论

本文提出 SymCIF，将晶体结构生成从标准 CIF 文件顺序重组为对称性条件下的 Wyckoff assignment selection 与 geometry rendering。该表示显式分离 composition、space group、Wyckoff orbit、free parameters 和 lattice，使候选生成和失败分析能够围绕离散对称性覆盖与连续几何质量展开。

在 MP-20 上，SymCIF hybrid-prior 在 8,893 个 structured test 样本上达到 match@1 = 60.90%、match@5 = 73.68%；按原始 9,046 个 test 样本计入 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。相对于 SymCIF current order，这一提升与 WA_hit@1/@5 的同步提高一致，表明 train-only Wyckoff assignment reranking 是当前收益的主要来源。

这些结果支持一个克制但明确的结论：显式的对称性条件表示能够改善真实空间群给定时的 MP-20 top-1 recovery，并提供比普通 CIF 文本生成更可诊断的错误分析路径。当前方法仍受 oracle space group、训练分布 prior 和连续几何质量限制；要进一步发展为更通用的晶体生成方法，还需要同输入比较、geometry refinement 和跨数据集验证。

# 图表说明

图 1. SymCIF 方法示意图。该图应展示原始 CIF language-modeling sequence 与 SymCIF sequence 的差异。原始流程按 data block、cell parameters、formula、symmetry operations 和 atom_site loop 生成；SymCIF 流程按 composition + oracle SG、Wyckoff assignment、free parameters/coordinates、lattice、CIF reconstruction 和 validation 组织。

图 2. MP-20 主结果比较。该图应展示 SymCIF current order、SymCIF hybrid-prior structured、SymCIF hybrid-prior original-test adjusted 与 CrystaLLM-a published reference 的 match@1 和 RMSE@1。图中必须标注输入条件，避免将 oracle-SG setting 误读为 same-input comparison。

图 3. Wyckoff assignment coverage 与 match 的关系。该图应展示 current order 与 hybrid-prior 的 WA_hit@1/@5 和 match@1/@5，并突出 WA_hit@5 与 match@5 之间的差距。

图 4. Failure audit 与 geometry bottleneck。该图应展示 evaluation coverage、render success、coverage-to-match gap 和复杂子集表现，用于说明主结果的工程可靠性和剩余失败来源。
