# SymCIF：面向可诊断晶体结构生成的对称性条件表示

# 摘要

晶体结构生成不仅要求模型输出可解析的结构文本，还要求生成结构同时满足化学组成、空间群、Wyckoff 位点、原子坐标和晶胞参数之间的耦合约束。基于 CIF 的语言模型为晶体生成提供了统一的文本接口，并已证明直接生成晶体结构文件具有可行性[REF]。然而，标准 CIF 的字段顺序主要服务于数据记录和交换，并不等同于晶体学约束传播顺序。将晶体结构作为普通长文本序列生成时，空间群、Wyckoff assignment、坐标自由度和晶格几何之间的关系往往只能被模型隐式学习，从而使错误来源难以定位。

本文提出 SymCIF，一种面向晶体结构生成的对称性条件结构化表示。SymCIF 将 CIF 中的结构信息重新组织为化学组成、空间群、Wyckoff assignment table、自由坐标参数和晶格参数，并将生成过程分解为 Wyckoff assignment selection 与 geometry rendering。该流程先在 composition + space group 条件下确定离散对称骨架和元素占据，再渲染连续坐标与晶格并重建 CIF。由此，生成候选不仅可以用 match rate 和 RMSE 评价，还可以按 Wyckoff 覆盖、空间群一致性、公式一致性、渲染成功率和几何误差进行分解诊断。

在 MP-20 structured test 的 8,893 个样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。按原始 MP-20 test 的 9,046 个样本计入 153 个 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。与 SymCIF current order 相比，hybrid-prior 将 WA_hit@1/@5 从 49.80%/71.27% 提高到 59.02%/81.66%，并带来 match@1/@5 的同步提升。需要强调的是，当前 SymCIF 主结果使用 composition + oracle ground-truth space group，而公开 CrystaLLM-a MP-20 reference 是 composition-only；因此该比较不构成 same-input superiority。尽管 SymCIF 的 match@1 高于 CrystaLLM-a published reference 的 55.85%，其 RMSE@1 仍差于 CrystaLLM-a reference 的 0.0437，说明连续几何质量仍是主要瓶颈。

# 引言

从化学组成生成三维晶体结构是计算材料研究中的核心任务之一。该任务可以为结构候选生成、数据库补全和后续性质计算提供输入，但它本身并不等同于无条件新材料发现。一个可用的晶体生成模型必须同时处理离散与连续约束：元素计数需要与原子位点一致，空间群需要与对称操作和 Wyckoff 位点一致，Wyckoff 多重度需要覆盖化学组成，坐标自由参数需要落在对应轨道允许的形式中，晶胞参数还需要与展开坐标共同决定最终结构是否能够匹配目标结构。

CIF 是晶体结构最常用的文本表达之一，因此将 CIF 作为语言模型的生成目标是自然且有效的技术路线。CrystaLLM 等 CIF language model 证明了自回归模型可以学习 CIF 文件中的 cell parameters、formula、symmetry operations 和 atom_site loop，并在给定组成等条件时生成完整晶体结构[REF]。这一方向的重要价值在于，它把复杂的周期结构转化为统一的文本建模问题，并能够复用通用语言模型的训练和采样框架。

然而，CIF 文件的记录顺序与晶体学约束顺序并不一致。标准 CIF 通常按照 data block、cell parameters、chemical formula、symmetry operations 和 atom_site loop 等字段组织文件；这种顺序适合存储和软件交换，但并不反映结构构造中 composition、space group、Wyckoff assignment、coordinates 和 lattice 的依赖关系。对于自回归模型而言，这意味着多个全局约束需要在长文本序列中被隐式维持。一旦模型在某个字段中产生局部错误，后续字段可能仍然语法合法，但整体结构却可能在空间群、位点占据或几何匹配上失败。

这一 representation mismatch 是本文关注的技术问题。给定 composition 后，space group 限定合法的 Wyckoff orbits、site symmetries 和 symmetry operations；Wyckoff multiplicities 决定元素计数是否可以被合法覆盖；只有在选定 orbit 后，坐标自由参数才具有明确的物理和几何含义；最后，lattice 与 symmetry-expanded coordinates 共同决定 StructureMatcher 是否能识别生成结构与目标结构匹配。若这些依赖关系全部混合在普通 CIF token sequence 中，生成失败往往难以归因：错误可能来自 Wyckoff skeleton 缺失、元素分配错误、空间群字段与实际结构不一致、坐标参数不合理，或晶胞几何质量不足。

为此，本文提出 SymCIF，将晶体结构生成从自由 CIF token generation 重组为对称性条件下的结构化生成流程。SymCIF 不把 CIF 文件顺序视为生成顺序，而是将结构表示为 `formula_counts`、space group、Wyckoff assignment table、free parameters 和 lattice。生成流程先确定 composition 和 oracle ground-truth space group，再进行 Wyckoff assignment selection，随后渲染连续坐标与晶格，最终重建 CIF 并接受公式、空间群、validity、match rate 和 RMSE 评估。这里的“因果顺序”并不指真实晶体形成过程的物理因果，而是指更接近晶体学约束传播的工程表示顺序。

本文在 MP-20 条件晶体结构生成/结构重建任务上评估 SymCIF。当前实验设置使用 composition + oracle ground-truth space group，目标是在给定这些条件的情况下恢复与测试集目标结构匹配的候选。该设置不同于公开 CrystaLLM-a MP-20 reference 的 composition-only 条件，因此本文不声称 SymCIF 在同输入条件下全面优于 CrystaLLM-a。本文的中心结论更窄也更具体：在 ground-truth-space-group setting 下，SymCIF 通过显式的 Wyckoff assignment selection 和 train-only hybrid-prior reranking 改善 MP-20 top-1 recovery；同时，WA coverage 与 match rate 之间的差距以及 RMSE 落后表明，连续几何质量仍是后续工作的主要瓶颈。

# 结果

## 结果一：问题定义与基线诊断

本文研究的任务是 MP-20 上的条件晶体结构生成/结构重建，而不是无条件材料发现。对于每个测试样本，模型需要在给定化学组成和当前实验中的 oracle ground-truth space group 条件下生成候选晶体结构。候选结构被重建为 CIF 后，首先检查是否可读、公式是否一致、原子数是否一致、空间群是否一致以及渲染是否成功；随后通过 StructureMatcher 判断生成结构是否与目标结构匹配，并在匹配样本上统计 RMSE。因而，match rate 与 RMSE 是本文的主要性能指标，validity 或 render success 只能说明候选能够被解析或满足部分结构检查，不能单独证明结构预测正确。

传统 CIF language modeling 的基线问题可以概括为：模型沿文件记录顺序生成结构，但需要在该顺序之外维持晶体学约束。以 CrystaLLM 类方法为代表的 CIF 语言模型证明了这一建模范式的可行性；它们不是无效模型，也不是本文要否定的对象。本文关注的是另一个层面的问题：当 composition、space group、Wyckoff sites、coordinates 和 lattice 被混合在同一自回归序列中时，模型的失败模式很难被拆解。例如，化学式字段可能与输入条件一致，但 Wyckoff/site 组合无法覆盖正确的原子计数；空间群字段可能被写入 CIF，但生成结构重新解析后的实际对称性与该字段不一致；局部坐标或 atom-site loop 的细小偏差可能导致 StructureMatcher 不匹配；晶胞参数即使格式正确，也可能与坐标共同造成较高的 RMSE。

这种诊断困难限制了单纯后处理和 reranking 的作用。如果采样阶段从未产生正确或近似正确的 Wyckoff assignment，后处理很难从错误的离散候选中恢复正确结构；如果正确 assignment 已经出现但几何渲染不足，则单纯增加 Wyckoff 候选数量也不能解决 RMSE 和匹配失败。因此，本文首先将生成错误拆成两个相互关联但可区分的问题：离散的 Wyckoff assignment 是否被正确覆盖，以及在正确或接近正确的 assignment 条件下，连续坐标和晶格是否足以通过结构匹配。这个划分构成后续 SymCIF 表示和实验分析的基础。

## 结果二：SymCIF 表示如何重构晶体生成过程

SymCIF 的核心改变是重新组织晶体结构信息，而不是简单修改 prompt 或增加后处理规则。图 1 概述了该流程。原始 CIF 生成通常遵循 file-record order，即 data block、cell parameters、formula、symmetry operations 和 atom_site loop 的线性顺序。SymCIF 则采用 composition、space group、Wyckoff assignment、free parameters/coordinates、lattice、CIF reconstruction 和 validation 的结构化顺序。该顺序将离散对称约束置于连续几何渲染之前，使模型或搜索器能够先处理合法 Wyckoff 轨道和元素占据，再处理坐标与晶胞参数。

在数据结构上，每个 SymCIF record 包含 `formula_counts`、`sg`、`sg_symbol`、`wa_table`、`free_params`、`lattice`、`canonical_skeleton_key` 和 `canonical_wa_key`。其中，`formula_counts` 描述 conventional cell 中每种元素的整数计数；`sg` 和 `sg_symbol` 描述空间群条件；`wa_table` 的每一行对应一个元素占据的 Wyckoff orbit，而不是 CIF 中的单个 atom-site token。每行记录 Wyckoff letter、multiplicity、site symmetry、enumeration、orbit identifier、representative coordinate expression 以及自由坐标参数。`canonical_skeleton_key` 只比较 orbit skeleton，`canonical_wa_key` 进一步加入元素分配，因此可以分别判断候选是否选中了正确的对称骨架和完整元素占据。

这一表示使晶体生成的失败来源更容易被拆解。对于每个候选，SymCIF 可以记录 WA_hit、skeleton_hit、formula_ok、atom_count_ok、SG_ok、render_success、strict_valid、match outcome 和 RMSE。由此，最终的 match failure 不再只是一个黑箱结果，而可以被解释为未覆盖正确 WA、覆盖了 WA 但几何失败、空间群重新解析不一致、渲染失败或 StructureMatcher 不匹配。该设计也为候选重排提供了更明确的目标：先提高正确 Wyckoff assignment 进入 top-k 的概率，再改进正确 assignment 条件下的 geometry rendering。

需要注意的是，SymCIF 表示本身并不自动保证性能提升。它的作用是将生成空间重排为更符合晶体学约束的形式，并使候选错误可以被测量。实际性能是否提高，需要通过 MP-20 主结果、WA coverage 变化和失败审计来验证。

## 结果三：MP-20 上与 CrystaLLM 的比较结果

表 1 汇总了 MP-20 上的主结果。SymCIF current order 在 structured test 上达到 match@1 = 51.29%、match@5 = 66.11%，RMSE@1 = 0.0668、RMSE@5 = 0.0742。使用 hybrid-prior 后，SymCIF 在同一 8,893 个 structured test 样本上达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。也就是说，相比 current order，hybrid-prior 将 match@1 提高 9.61 个百分点，将 match@5 提高 7.57 个百分点，并降低了匹配样本上的平均结构误差。

原始 MP-20 test 包含 9,046 个样本，其中 153 个样本在当前 SymCIF 转换流程中属于 structured extraction failures。为了避免只报告结构化成功子集带来的偏差，本文同时给出 original-test adjusted 口径：将这 153 个样本计为失败后，SymCIF hybrid-prior 仍达到 match@1 = 59.87%、match@5 = 72.43%。这一结果说明，主结果并不完全依赖于剔除结构化失败样本。

与公开 CrystaLLM-a MP-20 reference 比较时，必须区分输入条件。CrystaLLM-a published reference 是 composition-only 设置，报告 match@1 = 55.85%、match@20 = 75.14%、RMSE@1 = 0.0437、RMSE@20 = 0.0395。SymCIF 当前主结果使用 composition + oracle ground-truth space group，因此二者不是 same-input comparison。在这一前提下，SymCIF hybrid-prior 的 match@1 高于 CrystaLLM-a published reference，但其 RMSE@1 = 0.0573 仍差于 CrystaLLM-a 的 0.0437。图 2 应当将 input condition 明确标注在同一图中，以避免把 oracle-SG setting 下的 top-1 recovery 写成同输入条件下的全面超越。

这些结果支持一个中等保守的结论：SymCIF 在给定真实空间群的条件下可以显著改善 MP-20 top-1/top-5 结构恢复，且相对于当前 SymCIF 排序有清晰增益；但它尚不能被描述为全面 state-of-the-art crystal generation，也不能证明其在 composition-only 设置下优于 CrystaLLM-a。

表 1. MP-20 主结果表。SymCIF 使用 composition + oracle ground-truth space group；CrystaLLM-a published reference 使用 composition-only，二者不是 same-input comparison。

| 方法 | 输入条件 | 测试范围 | 样本数 | match@1 ↑ | match@5/20 ↑ | RMSE@1 ↓ | RMSE@5/20 ↓ | 说明 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SymCIF current order | composition + oracle SG | structured test | 8,893 | 51.29% | 66.11% @5 | 0.0668 | 0.0742 @5 | baseline ordering |
| SymCIF hybrid-prior | composition + oracle SG | structured test | 8,893 | 60.90% | 73.68% @5 | 0.0573 | 0.0615 @5 | main result |
| SymCIF hybrid-prior | composition + oracle SG | original-test adjusted | 9,046 | 59.87% | 72.43% @5 | 0.0573 | 0.0615 @5 | 153 extraction failures counted as failures |
| CrystaLLM-a published reference | composition only | published MP-20 reference | 待补充 | 55.85% | 75.14% @20 | 0.0437 | 0.0395 @20 | not same input |

## 结果四：消融实验与机制分析

为了理解 SymCIF hybrid-prior 的增益来源，本文比较了 current order 与 hybrid-prior 在 Wyckoff assignment coverage 上的差异。Current order 的 WA_hit@1/@5 为 49.80%/71.27%，而 hybrid-prior 提高到 59.02%/81.66%。对应地，match@1/@5 从 51.29%/66.11% 提高到 60.90%/73.68%。WA_hit 和 match 的同步提升说明，在当前 K<=5 预算下，正确或等价 Wyckoff assignment 能否更早进入候选列表，是 top-1/top-5 结构恢复的主要因素之一。

Hybrid-prior selector 被设计为 train-only 且 inference-feasible。它使用训练集中可见的 Wyckoff assignment 或 skeleton 统计规律，并结合已有候选排序信息，对候选进行重排。该过程不使用 test label、StructureMatcher outcome 或 oracle GT-WA，因此可以作为推理阶段可用的 selector。与此同时，它仍然依赖训练分布中出现的 assignment pattern，不能被表述为与训练分布无关的泛化机制。更细粒度的 selector ablation，例如 policy rank only、train WA prior only、train skeleton prior only 和 hybrid-prior full，目前仍需补充完整数值；因此本文当前只能将 WA_hit/match 的同步提升解释为强诊断证据，而不是完整因果证明。

WA coverage 的提升并未完全转化为结构匹配。Hybrid-prior 的 WA_hit@5 为 81.66%，而 match@5 为 73.68%，二者之间仍有约 8 个百分点差距。这意味着一部分样本已经在 top-5 中包含正确或等价的 Wyckoff assignment，但坐标自由参数、晶格参数、结构有效性或 StructureMatcher 判据仍导致最终匹配失败。进一步地，SymCIF 的 RMSE@1 仍差于 CrystaLLM-a published reference，说明当前方法更擅长改善离散对称性候选排序，而不是已经解决连续几何精度问题。

这一机制分析将后续改进方向限定得更清楚。若目标是提高 match@1，WA reranking 仍然重要；若目标是降低 RMSE 或提升 match@5 到接近 WA coverage 的上限，则需要 geometry-quality scoring、validity-aware reranking 或 geometry refinement。仅仅增加后处理规则或扩大 WA candidate 数量，很难从根本上解决已选 WA 条件下的坐标和晶胞误差。

表 2. 已完成与待补充的 selector 分析。当前可写入正文的完整数值是 current order 与 hybrid-prior 的比较；其他 selector ablation 仍为待补充实验。

| Selector | Prior source | Inference-feasible | WA_hit@1 ↑ | WA_hit@5 ↑ | match@1 ↑ | match@5 ↑ | RMSE@1 ↓ | RMSE@5 ↓ | 说明 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current order | none/current rank | yes | 49.80% | 71.27% | 51.29% | 66.11% | 0.0668 | 0.0742 | baseline |
| hybrid-prior | train split + candidate rank | yes | 59.02% | 81.66% | 60.90% | 73.68% | 0.0573 | 0.0615 | main selector |
| policy rank only | 待补充 | yes | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| train WA prior only | train split | yes | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | distribution-dependent |
| train skeleton prior only | train split | yes | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | distribution-dependent |
| oracle GT-WA diagnostic | test label | no | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | diagnostic only |

## 结果五：失败案例与边界条件分析

Failure audit 的第一层作用是确认主结果是否受到工程伪影影响。当前 MP-20 full evaluation 覆盖 8,893/8,893 个 structured test samples；top-5 评估包含 44,465 条 candidate-level records；@1/@5 evaluation timeout 均为 0；render_success@1 = 100.00%，render_success@5 = 99.10%。这些数字表明，主结果不是由大规模渲染缺失、超时或不完整运行造成的。

Failure audit 的第二层作用是区分不同失败来源。当前材料显示，`n_sites>=6`、`n_sites>=12` 和 `extraction_hard` 等子集更难，而 high-multiplicity orbit 子集相对更强。这说明失败不能简单归因于单一“结构复杂度”指标。对于一些样本，模型可能已经复制或保持了正确化学式，但 Wyckoff/site 组合无法形成正确结构；另一些样本可能在 CIF 字段中包含空间群信息，但生成结构重新解析后无法保持同一空间群；还有一些样本的离散 WA 已经命中，但局部坐标、晶格几何或原子位点排列导致 StructureMatcher 不匹配。这些失败类型需要在后续图 4 和补充表中进一步量化。

这种边界分析也说明，SymCIF 的可诊断性不等于失败已经被完全解决。当前 top-k candidate reranking 可以提高正确 WA 进入候选集的概率，但如果采样空间没有产生合理几何，或者 lattice 与坐标之间的组合不稳定，后处理很难把错误候选修复为正确结构。相反，如果正确 WA 已经出现但几何失败，继续增加离散候选的收益会逐步下降。因而，本文将 geometry rendering/refinement 视为后续最重要的技术问题，而不是把当前结果解释为完整解决晶体结构生成。

# 方法

## 数据集与任务定义

本文使用 MP-20 作为主要评估数据集[REF]。当前实验口径中，原始 MP-20 test 包含 9,046 个样本，其中 8,893 个样本成功转换为 SymCIF structured test，153 个样本属于 structured extraction failures。结果部分同时报告 structured-only 与 original-test adjusted 两种口径。Structured-only 指标以 8,893 个成功结构化样本为分母；original-test adjusted 指标以 9,046 个原始测试样本为分母，并将 153 个 structured extraction failures 计为失败。

本文任务定义为 symmetry-conditioned crystal structure generation/reconstruction。给定 chemical composition 和 oracle ground-truth space group，方法生成一个或多个候选晶体结构。每个候选被重建为 CIF 后，与目标结构进行结构匹配评估。该设置应与 composition-only generation 区分开来；它用于检验在真实空间群已知时，显式 Wyckoff assignment selection 和 geometry rendering 是否能够改善结构恢复。

## CrystaLLM 基线设置

CrystaLLM-a 在本文中作为公开 CIF language modeling reference 使用。当前稿件采用其 published MP-20 reference 数值：match@1 = 55.85%、match@20 = 75.14%、RMSE@1 = 0.0437、RMSE@20 = 0.0395。该 reference 的输入条件是 composition-only，而 SymCIF 主结果使用 composition + oracle ground-truth space group。因此，本文所有与 CrystaLLM-a 的比较均被表述为相对于公开参考值的对照，而不是 same-input baseline comparison。

当前版本尚未完成给 CrystaLLM-a 加入相同 oracle space group 条件的重跑，也尚未完成 SymCIF 在 composition-only 条件下的完整评估。因此，CrystaLLM 相关结果在本文中主要用于定位公开参考水平和说明输入条件差异。若后续补充 same-input fairness experiment，应在方法和结果中更新这一小节。

## SymCIF 表示构建

SymCIF 将晶体表示为结构化记录。每条记录包含样本标识、组成、空间群、Wyckoff assignment table、晶格参数以及诊断字段。组成由 `formula` 和 `formula_counts` 表示，其中 `formula_counts` 记录 conventional cell 中每种元素的整数原子数。空间群由 `sg` 和 `sg_symbol` 表示。晶格由 `lattice` 字段表示，包括 `a`、`b`、`c`、`alpha`、`beta`、`gamma` 和 `volume`。结构的对称性核心由 `wa_table` 表示；其中每一行对应一个元素占据的 Wyckoff orbit。

`wa_table` 每行包含 `element`、`sg`、`letter`、`multiplicity`、`site_symmetry`、`enumeration`、`orbit_id`、`representative_expr`、`free_symbols` 和 `free_params`。其中，`orbit_id` 是稳定标识一个 Wyckoff orbit 的 canonical identifier；`representative_expr` 描述代表坐标表达式；`free_symbols` 和 `free_params` 描述该 orbit 的连续自由度。为支持审计，记录中还保存 `source_coord`、`mapped_coord`、`extraction_method`、`extraction_residual`、`expansion_count_after_reextract`、`expansion_ok`、`extraction_success` 和 `fallback_reason` 等字段。

从 CIF 到 SymCIF 的转换首先使用 CIF parser 读取 benchmark split 中的 CIF，并通过空间群分析器将结构转换为 conventional standard cell。随后，从 conventional cell 中提取整数 `formula_counts`，识别空间群编号、空间群符号和 symmetrized equivalent sites。每个等价位点组被映射到 Wyckoff lookup table 中的 `OrbitToken`，并从 source fractional coordinates 反解得到该 orbit 的 `free_params`。反解后，流程重新展开 orbit，检查 expansion count、source inclusion、residual 和 formula closure。只有通过这些检查的样本才写入 structured SymCIF 数据集；失败样本在 original-test adjusted 口径中计为失败。

具体的 `symprec`、angle tolerance、free-parameter tolerance、Wyckoff lookup table 版本和 fallback 细节仍需在最终投稿版本中补充。Partial occupancy、disorder、non-integer composition 和 mixed-element equivalent groups 的处理策略也需要在补充方法中明确。

## 模型训练与生成流程

当前 SymCIF 实验将生成过程分解为候选 Wyckoff assignment 选择和 geometry rendering。给定 composition 与 oracle ground-truth space group，候选生成器产生多个 Wyckoff assignments 和对应几何变体。Hybrid-prior selector 使用 train split 中的 Wyckoff assignment 或 skeleton 统计规律，并结合已有候选排序信息，对候选进行 train-only、inference-feasible reranking。该 selector 不使用 test label、StructureMatcher outcome 或 oracle GT-WA，但依赖训练分布中的 assignment 频率，因此其泛化边界需要通过 train-frequency 和 unseen-skeleton 分析进一步评估。

本文当前没有完整列出底层候选生成模型的所有训练超参数。正式投稿前需要补充模型目录、训练数据划分、tokenization/representation 输入输出、采样温度、top-k/top-p 设置、候选数量、geometry variant 生成方式、随机种子和硬件配置。若相关步骤由既有 CrystaLLM 框架或已有模型权重完成，应明确写出是否重新训练、是否只做推理、以及哪些模块参与了 hybrid-prior reranking。

## 结构有效性检查与匹配评估

给定一个候选 SymCIF record，renderer 根据 `orbit_id` 找回对应 orbit token，并用 `free_params` 计算代表坐标和 symmetry-expanded fractional coordinates。随后，renderer 将展开后的 atom rows 与 `formula_counts`、`sg`、`sg_symbol` 和 `lattice` 一起写回 CIF。重建后的 CIF 被重新解析，并检查 readable、formula_ok、atom_count_ok、SG_ok、valid/strict_valid 和 render_success。

最终结构恢复能力通过 match@k 和 RMSE@k 评估。对于每个样本，若 top-k 候选中至少一个候选通过 StructureMatcher 与目标结构匹配，则该样本计为 match@k 成功。RMSE@k 在 matched samples 上统计匹配误差。WA_hit@k 表示 top-k 中是否包含 ground-truth 或等价 canonical WA；skeleton_hit@k 表示 top-k 中是否包含正确 canonical skeleton。StructureMatcher 参数、RMSE 定义、evaluation timeout 设置和 deterministic/seed 设置仍需在最终方法中补充。

## 消融实验设置

当前已完成并可写入正文的核心消融是 SymCIF current order 与 SymCIF hybrid-prior 的对比。该对比同时报告 WA_hit@1/@5、match@1/@5 和 RMSE@1/@5，用于判断 candidate reranking 是否提高正确 Wyckoff assignment 的前排覆盖，并观察 coverage 变化是否与 match 改善一致。

更细粒度的 selector ablation 仍待补充，包括 policy rank only、old rank only、train WA frequency only、train skeleton frequency only、hybrid-prior full 和 oracle GT-WA diagnostic。未来这些实验应报告 WA_hit、skeleton_hit、match、RMSE、strict_valid 和 render_success，并明确哪些设置是 inference-feasible，哪些仅为 diagnostic/oracle analysis。

# 讨论

本文的主要贡献是将 CIF language modeling 中隐式混合的晶体学约束重组为 SymCIF 的结构化表示。与直接按 CIF 文件顺序生成完整文本相比，SymCIF 先处理 composition 和 space group 条件下的 Wyckoff assignment，再进行连续坐标和晶格渲染。这一表示没有否定 CIF 作为结构交换格式的价值，也没有否定 CrystaLLM 类方法的有效性；相反，它指出 CIF 文件顺序与晶体学约束顺序之间存在 mismatch，并给出一种可诊断的表示重构方式。

MP-20 结果表明，这种重构在 ground-truth-space-group setting 下能够带来明确收益。Hybrid-prior 将 WA_hit@1/@5 从 49.80%/71.27% 提高到 59.02%/81.66%，同时将 match@1/@5 从 51.29%/66.11% 提高到 60.90%/73.68%。这种同步变化支持一个机制性解释：在当前候选预算下，更好的 Wyckoff assignment reranking 是 top-1 recovery 提升的主要来源。由于 hybrid-prior 使用 train-only 信息，它是推理可行的，但其效果也可能与 MP-20 训练分布中的 Wyckoff pattern 相关。

SymCIF 的结果也显示了一个重要限制：离散对称性候选排序改善并不等于连续几何质量已经解决。WA_hit@5 高于 match@5，说明部分正确 WA 没有转化为最终匹配；RMSE@1 仍差于 CrystaLLM-a published reference，说明匹配结构的几何精度仍有差距。因此，本文的结果更适合被解释为 symmetry-conditioned top-1 recovery 和 failure diagnosis 的改进，而不是晶体生成质量的全面领先。

从方法发展角度看，SymCIF 的价值在于把未来改进方向变得更具体。如果失败来自 WA missing，应改进候选覆盖和 assignment selector；如果失败来自 WA correct but geometry fail，应改进坐标自由参数、晶胞生成和 geometry-quality scoring；如果失败来自空间群字段与实际结构不一致，应加强 reconstruction 和 validation。这样的可审计性对于构建更可靠的晶体生成系统是有用的，但它仍需要更多公平比较和泛化实验支持。

# 局限性与未来工作

当前工作的第一项局限是输入条件。SymCIF 主结果使用 composition + oracle ground-truth space group，而 CrystaLLM-a published reference 是 composition-only。该设置可以用于评估在空间群已知时的结构恢复能力，但不能证明 SymCIF 在同输入条件下优于 CrystaLLM-a。后续需要补充两类实验之一：给 CrystaLLM-a 同样的 oracle SG prompt 并用同一 evaluator 对比，或让 SymCIF 从 composition-only 条件预测/检索 space group 并报告性能下降。

第二项局限是几何质量。SymCIF 的 RMSE@1 = 0.0573，仍高于 CrystaLLM-a published RMSE@1 = 0.0437。WA_hit@5 与 match@5 之间的差距也说明，正确离散候选并不总能产生正确几何结构。未来需要引入不依赖 evaluator oracle 的 geometry-quality predictor、validity-aware reranking 或 geometry refinement module，并将 diagnostic rerank 与真实 inference-feasible rerank 严格区分。

第三项局限是分布依赖和泛化证据不足。Hybrid-prior 使用 train split 中的 Wyckoff assignment 统计规律，因此可能在训练集中常见的 WA pattern 上更有效。后续需要按 GT WA seen/unseen、GT skeleton seen/unseen、train WA frequency quantiles、common/rare SG、n_sites 和 num_elements 分组报告 WA_hit、match 和 RMSE。MPTS-52 或其他外部数据集上的最终流程结果也需要补充；在这些实验完成前，本文不应声称跨 benchmark 的普适优势。

第四项局限是统计不确定性和完整消融仍需加强。当前主结果来自 full-test deterministic evaluation，但尚未报告 bootstrap confidence intervals 或多随机种子结果。Selector ablation、geometry bottleneck ablation 和 qualitative examples 也需要补充，才能更充分地支持机制解释。当前没有系统 DFT stability 结果，因此本文不讨论生成结构的热力学稳定性，也不声称生成结果可以直接用于材料发现。

# 结论

本文提出 SymCIF，将晶体结构生成从 CIF 文件记录顺序重组为对称性条件下的 Wyckoff assignment selection 与 geometry rendering。该表示把 composition、space group、Wyckoff orbit、free parameters 和 lattice 分离为可审计字段，使候选生成和失败分析能够围绕离散对称性覆盖与连续几何质量展开。

在 MP-20 上，SymCIF hybrid-prior 在 8,893 个 structured test 样本上达到 match@1 = 60.90%、match@5 = 73.68%；按原始 9,046 个 test 样本计入 structured extraction failures 后，match@1 = 59.87%、match@5 = 72.43%。相对于 SymCIF current order，这一提升与 WA_hit@1/@5 的同步提高一致，表明 train-only Wyckoff assignment reranking 是当前收益的主要来源。与此同时，SymCIF 使用 oracle ground-truth space group，且 RMSE 仍差于 CrystaLLM-a published reference，因此本文不主张 same-input superiority 或 state-of-the-art crystal generation。

总体而言，SymCIF 提供了一条更可约束、可重排、可诊断的晶体结构生成路径。当前证据支持它在 ground-truth-space-group setting 下改善 MP-20 top-1 recovery，并清楚揭示 geometry quality 是下一阶段瓶颈。未来工作需要在 same-input fairness、geometry refinement、统计置信度和跨数据集泛化上补齐证据。

# 图表说明

图 1. SymCIF framework schematic。该图应对比原始 CIF language-modeling sequence 与 SymCIF sequence：前者按 data block、cell parameters、formula、symmetry operations 和 atom_site loop 生成；后者按 composition + oracle SG、Wyckoff template、WA selection/reranking、free parameters/coordinates、lattice、CIF reconstruction 和 validation 组织流程。图中还应标注 WA_hit、skeleton_hit、formula_ok、SG_ok、match 和 RMSE 等诊断输出。

图 2. MP-20 benchmark comparison。该图应展示 SymCIF current order、SymCIF hybrid-prior structured、SymCIF hybrid-prior original-test adjusted 与 CrystaLLM-a published reference 的 match@1 和 RMSE@1 对比。图中必须明确标注输入条件：SymCIF 为 composition + oracle ground-truth SG，CrystaLLM-a published reference 为 composition-only。

图 3. WA reranking and coverage mechanism。该图应展示 current order 与 hybrid-prior 的 WA_hit@1/@5、match@1/@5，以及 WA_hit@5 与 match@5 的 coverage-to-match gap。若后续补齐 selector ablation，可加入 policy rank only、train prior only 和 hybrid-prior 的对比。

图 4. Failure audit / geometry bottleneck。该图应展示 WA_hit@5 与 match@5 的差距、failed_with_WA_hit 与 failed_without_WA_hit、geometry source failure breakdown，以及 complex subsets 中的表现。当前复杂子集的具体数值仍需从最终 failure audit 表中补齐。

表 1. MP-20 main benchmark table with input condition。该表应列出方法、输入条件、测试范围、样本数、match@1、match@5 或 match@20、RMSE@1、RMSE@5 或 RMSE@20，并明确 CrystaLLM-a 与 SymCIF 不是 same-input comparison。

表 2. Selector ablation table。当前主文可填 current order 与 hybrid-prior；policy rank only、train WA prior only、train skeleton prior only 和 oracle GT-WA diagnostic 仍为待补充。

补充表 S1. Dataset and extraction audit。该表应列出 MP-20 original test = 9,046、structured test = 8,893、structured extraction failures = 153、full evaluation coverage = 8,893/8,893、top-5 candidate records = 44,465、eval_timeout@1/@5 = 0、render_success@1 = 100.00%、render_success@5 = 99.10%。

补充表 S2. Complex subset performance。该表应按 overall、n_sites>=6、n_sites>=12、num_elements>=4、rare_sg、high_multiplicity_orbit 和 extraction_hard 等子集报告 match、RMSE、failed_with_WA_hit 和 failed_without_WA_hit。

# 待补充信息清单

1. 真实参考文献：CIF 标准、MP-20、CrystaLLM、StructureMatcher、pymatgen/spglib、Wyckoff 表示和晶体生成相关工作。
2. 具体训练与推理超参数：模型权重来源、是否重新训练、训练数据划分、tokenization 或结构化输入格式、采样温度、top-k/top-p、候选数量、随机种子和硬件环境。
3. SymCIF 转换参数：`symprec`、angle tolerance、free-parameter tolerance、Wyckoff lookup table 来源与版本、fallback extraction 策略和失败样本处理规则。
4. 完整数据划分：MP-20 train/validation/test 样本数、structured extraction 成功率、153 个 extraction failures 的失败类型统计。
5. 对比方法完整数值：CrystaLLM-a 的公开 reference 来源、是否存在 match@5 数值、是否补充 oracle-SG prompt 的 same-input baseline。
6. Selector ablation：policy rank only、old rank only、train WA frequency only、train skeleton frequency only、hybrid-prior full 和 oracle GT-WA diagnostic。
7. Geometry bottleneck ablation：GT-WA + current geometry、selected WA + diagnostic best geometry、validity-first diagnostic rerank、geometry-quality diagnostic rerank。
8. 统计显著性：match@1、match@5、RMSE@1、RMSE@5 的 bootstrap 95% confidence intervals 或多随机种子评估。
9. 泛化分析：MPTS-52 或其他外部数据集上的最终流程结果，以及 train-frequency、rare SG、unseen WA、unseen skeleton 和 complex subset 分组分析。
10. 图表最终编号和图源文件：图 1-4、表 1-2、补充表 S1-S2 的最终数据、caption 和可编辑源文件。
11. DFT 或物理稳定性评估：当前没有可写入正文的 DFT stability 结果；若要讨论稳定性，需要补充完整计算设置和结果。
