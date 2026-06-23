# SymCIF 中文论文作战地图

## 0. 当前证据口径

本作战地图基于当前已经完成的 MP-20 实验结果。核心 test 结果来自 MP-20 structured test 全量 8,893 条；同时给出原始 MP-20 test 9,046 条分母折算口径，其中 153 条 structured extraction failure 计为失败。

重要口径边界：

- SymCIF 当前结果使用 `composition + oracle ground-truth space group`。
- CrystaLLM-a published reference 是 `composition-only`。
- 因此，当前结果可以支持“在本实验设定下优于公开 CrystaLLM-a 的 match@1”，但不能直接写成“同输入条件下全面超过 CrystaLLM-a”。
- 当前最强主结果：SymCIF hybrid-prior 在 MP-20 structured test 上 match@1 = 60.90%，match@5 = 73.68%，RMSE@1 = 0.0573，RMSE@5 = 0.0615。
- 原始 test 分母折算：match@1 = 59.87%，match@5 = 72.43%。
- CrystaLLM-a published MP-20 reference：match@1 = 55.85%，match@20 = 75.14%，RMSE@1 = 0.0437，RMSE@20 = 0.0395。

当前可写的中心判断是：

> SymCIF 将晶体生成问题分解为空间群约束下的 Wyckoff assignment 选择与几何渲染，在 MP-20 上实现了强 top-1 match 表现；但结构几何精度仍落后于 CrystaLLM-a，论文必须把“匹配成功率提升”和“几何质量仍需改进”分开叙述。

## 1. 核心贡献

### 贡献 1：把晶体生成从自由 CIF token generation 改写为 symmetry-aware structured generation

传统 CIF 生成容易在空间群、Wyckoff position、原子数和几何有效性上同时犯错。SymCIF 的核心技术路线是先解析并显式表示空间群、Wyckoff assignment、site multiplicity 和 free parameters，再生成结构。这使得模型输出从自由文本生成转为带物理/对称约束的结构化生成。

可写贡献：

- 提出 SymCIF，一种面向晶体结构生成的 symmetry-constrained representation and generation pipeline。
- 将空间群、Wyckoff assignment 和几何参数解耦，降低生成空间的复杂度。
- 通过 structured extraction、stable key 和 candidate reranking，使候选结构更可控、更可诊断。

### 贡献 2：引入 train-prior hybrid reranking，大幅提升 top-K Wyckoff assignment 覆盖

当前最有效的工程突破不是扩大采样，而是用 train-only prior 对候选 WA 进行重排。该方法不使用 test label，也不依赖 StructureMatcher oracle。

关键数字：

- current order WA_hit@1/@5 = 49.80% / 71.27%。
- hybrid-prior WA_hit@1/@5 = 59.02% / 81.66%。
- structured test match@1 从 51.29% 提升到 60.90%。
- structured test match@5 从 66.11% 提升到 73.68%。

可写贡献：

- 证明 WA selection 是当前 MP-20 top-1/top-5 性能的主要瓶颈之一。
- 提出一个 inference-feasible、train-only 的 hybrid-prior selector。
- 显示 candidate coverage 的提升能直接转化为 match@1 和 match@5 提升。

### 贡献 3：建立可解释的 failure audit，把候选覆盖瓶颈和几何质量瓶颈分离

SymCIF 的优势不只是结果数值，还在于每个失败可以分解到 WA coverage、skeleton coverage、render success、strict validity、formula/SG consistency、StructureMatcher no-match 等层面。

当前诊断：

- hybrid-prior WA_hit@5 = 81.66%，但 match@5 = 73.68%，仍有约 8 pp 未转化为 match。
- 这说明下一阶段瓶颈主要是 geometry/render/validity，而不是单纯缺少正确 WA 候选。
- RMSE@1 = 0.0573，仍高于 CrystaLLM-a 的 0.0437，进一步证明几何质量是短板。

可写贡献：

- 提供从候选生成到结构匹配的可审计 pipeline。
- 把性能提升的来源和剩余失败机制拆开，而不是只报告总分。
- 为后续改进提供明确方向：geometry scoring/reranking 和 validity-aware generation。

## 2. 最强证据

### 证据 A：MP-20 full structured test 上 match@1 超过 CrystaLLM-a published reference

最适合作为摘要、Introduction 末段、Results 第一主表的数字：

| method | scope | match@1 | match@5 | RMSE@1 | RMSE@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| SymCIF current order | structured test | 51.29% | 66.11% | 0.0668 | 0.0742 |
| SymCIF hybrid-prior | structured test | 60.90% | 73.68% | 0.0573 | 0.0615 |
| SymCIF hybrid-prior | original-test adjusted | 59.87% | 72.43% | 0.0573 | 0.0615 |
| CrystaLLM-a published | published reference | 55.85% | n/a | 0.0437 | n/a |

建议写法：

> On the MP-20 structured test set, SymCIF hybrid-prior reaches 60.90% match@1, exceeding the published CrystaLLM-a match@1 reference of 55.85%, while reducing RMSE@1 from 0.0668 to 0.0573 relative to the previous SymCIF ordering.

中文逻辑：

- 先说“structured test 全量 8,893 条”。
- 再说“原始 9,046 分母折算后仍为 59.87%”。
- 最后说“但是 RMSE 仍低于 CrystaLLM-a，后文诊断几何质量瓶颈”。

### 证据 B：WA coverage 提升解释了 match 提升

最强的 mechanism/diagnostic 证据：

| selector | WA_hit@1 | WA_hit@5 | match@1 | match@5 |
| --- | ---: | ---: | ---: | ---: |
| current order | 49.80% | 71.27% | 51.29% | 66.11% |
| hybrid-prior | 59.02% | 81.66% | 60.90% | 73.68% |

这组证据说明：

- 提升不是来自随机采样或 evaluator 偶然性，而是来自候选排序质量提升。
- top5 WA coverage 比 match@5 高，说明正确候选已经更多进入 top5。
- coverage-to-match gap 是后续 geometry 模块的改进空间。

### 证据 C：评估过程机械可靠

本次 MP-20 test full eval：

- 8,893/8,893 structured test samples 全量跑完。
- @1/@5 eval timeout = 0。
- @5 metrics 有 44,465 条 candidate-level records。
- render_success@1 = 100.00%，render_success@5 = 99.10%。

这可以支撑“结果不是小样本、不完整运行或 timeout artifact”。

### 证据 D：复杂子集诊断能指出方法边界

当前旧报告显示复杂子集仍弱：

- `n_sites>=6`、`n_sites>=12`、`extraction_hard` 等子集 match 明显低于整体。
- high-multiplicity orbit 子集表现反而较强，说明并不是所有复杂度都同等困难。

这部分适合 Results 后半段或 Supplementary，用于增强可信度和 reviewer trust。

## 3. 最弱证据

### 弱点 1：SymCIF 和 CrystaLLM-a 输入条件不完全公平

SymCIF 使用 oracle GT space group；CrystaLLM-a published reference 是 composition-only。这个差异必须主动写清楚，否则 reviewer 会质疑比较不公平。

当前不应写：

- “SymCIF outperforms CrystaLLM-a under the same input setting.”
- “SymCIF is universally superior to CrystaLLM-a.”
- “SymCIF achieves state-of-the-art crystal generation.”

更稳妥写法：

- “under a symmetry-conditioned setting”
- “relative to the published CrystaLLM-a reference”
- “while using ground-truth space-group conditioning”
- “showing that explicit symmetry decomposition can deliver strong top-1 recovery”

### 弱点 2：RMSE 仍落后

虽然 match@1 超过 CrystaLLM-a，但 RMSE@1 = 0.0573，高于 CrystaLLM-a 0.0437。RMSE@5 = 0.0615 也不够强。

这意味着论文不能只讲“超过 baseline”，必须讲清：

- SymCIF 当前更擅长恢复正确拓扑/匹配。
- 几何坐标和 lattice refinement 仍需加强。
- 后续可通过 geometry-quality predictor、validity-aware reranking 或 diffusion/refinement module 改进。

### 弱点 3：目前主结果集中在 MP-20

如果 MPTS-52 或更难数据集没有同等级强结果，论文主张必须收窄为：

- MP-20 上的强验证。
- 方法学上具有可解释性和可拓展性。
- 泛化结果作为边界或补充，而不是主 claim。

### 弱点 4：hybrid-prior selector 可能被质疑为 train-frequency prior

Reviewer 可能会问：

- 是否只是记忆训练集常见 WA pattern？
- 对 rare SG、rare composition、novel skeleton 是否有效？
- 和 retrieval baseline 的区别是什么？

需要补实验或分析：

- 按 train WA frequency bucket 分组。
- rare SG / unseen WA / unseen skeleton 子集表现。
- train-prior only vs policy rank only vs hybrid score only ablation。

### 弱点 5：缺少统计置信度或 seed variance

当前是 deterministic full-test evaluation，没有 bootstrap CI 或 seed variation。对于论文图表，建议补：

- bootstrap 95% CI for match@1/@5。
- 或者说明 deterministic pipeline 无 seed variation，并提供 bootstrap CI 作为 uncertainty estimate。

## 4. 论文主线

### 一句话主线

> 晶体 CIF 生成的关键难点不是简单生成合法文本，而是在空间群约束下同时恢复正确的 Wyckoff assignment 和几何参数；SymCIF 通过结构化对称表示、train-prior WA reranking 和几何渲染，将生成过程变成可控、可诊断的 symmetry-conditioned pipeline，并在 MP-20 上取得强 top-1 match 表现。

### Introduction 逻辑

1. 晶体生成需要同时满足成分、空间群、Wyckoff position、multiplicity 和几何合理性。
2. 自由 CIF token generation 虽然灵活，但容易把这些约束混在一起，导致错误不可诊断。
3. 现有 baseline 如 CrystaLLM 展示了 composition-conditioned CIF generation 的潜力，但没有显式把 Wyckoff assignment 和 geometry recovery 分离。
4. 本文提出 SymCIF：将晶体生成拆解为 structured symmetry representation、WA candidate generation/reranking、geometry rendering/evaluation。
5. 在 MP-20 上，SymCIF hybrid-prior 达到 60.90% match@1，超过 CrystaLLM-a published match@1 reference，同时揭示 geometry quality 仍是主要短板。

### Results 逻辑

#### Result 1：SymCIF pipeline and representation

要回答：SymCIF 是什么，为什么不是普通 CIF 生成？

证据：

- method schematic。
- structured extraction success rate。
- stable key / WA representation examples。

#### Result 2：MP-20 main benchmark

要回答：它是否有效？

证据：

- structured test match@1/@5/RMSE。
- original-test denominator adjusted result。
- 与 current order、CrystaLLM-a reference 对比。

#### Result 3：WA reranking explains the gain

要回答：性能提升从哪里来？

证据：

- current vs hybrid-prior WA_hit@1/@5。
- match improvement and coverage ceiling。
- top-K budget replay：K<=5 下 WA diversity 比 geometry variants 更关键。

#### Result 4：Geometry remains the bottleneck

要回答：为什么还没有全面超过 CrystaLLM？

证据：

- RMSE@1 仍高于 CrystaLLM-a。
- WA_hit@5 > match@5 的 gap。
- strict_valid/readable/geometry_source failure audit。

#### Result 5：Failure modes and boundaries

要回答：方法在什么情况下失败？

证据：

- n_sites、num_elements、rare SG、extraction_hard、high_multiplicity 子集。
- failed_with_WA_hit vs failed_without_WA_hit。
- invalid CIF/no match/error breakdown。

### Discussion 逻辑

1. SymCIF 证明显式 symmetry decomposition 可以显著提升 top-1 recovery。
2. 该方法的价值在于可控性和可诊断性，而不仅是一个总分。
3. 当前强项是 WA selection；短板是 geometry quality。
4. 与 CrystaLLM-a 的比较要保守：输入条件不同，RMSE 仍落后。
5. 下一步是 symmetry-conditioned geometry refinement 和 inference-feasible geometry-quality reranking。

## 5. 图表清单

### 主文 Figure 1：SymCIF method schematic

目的：解释方法，不放太多数字。

建议 panel：

- a. 输入：composition + space group。
- b. Symmetry parser / structured representation。
- c. WA candidate generation and stable-key reranking。
- d. Geometry rendering and CIF reconstruction。
- e. Evaluation outputs：match、RMSE、validity、failure audit。

当前已有初步图目录：

- `Paper/photo/symcif_method_schematic/`

### 主文 Figure 2：MP-20 benchmark comparison

目的：主结果图。

当前已生成：

- `Paper/photo/mp20_symcif_crystallm_comparison.*`

建议后续移动/重生成到二级目录：

- `Paper/photo/mp20_benchmark_comparison/`

当前图包含：

- a. match@1 对比。
- b. RMSE@1 对比。
- c. top-K comparison。
- d. WA coverage ceiling。

### 主文 Figure 3：Ablation and mechanism of hybrid-prior reranking

目的：证明 gain 来自 WA reranking，而不是偶然。

建议 panel：

- current order vs hybrid-prior WA_hit@1/@5。
- policy rank only / old rank only / train prior only / hybrid-prior ablation。
- WA diversity vs geometry diversity top5 budget replay。
- raw top100 coverage to selected top5 conversion。

需要补齐：

- selector ablation table。
- train-frequency bucket analysis。

### 主文 Figure 4：Failure mode and geometry bottleneck

目的：主动承认短板，并证明方法可诊断。

建议 panel：

- WA_hit@5 vs match@5 gap。
- failed_without_WA_hit vs failed_with_WA_hit。
- geometry_source success rate。
- strict_valid/readable/RMSE by subset。

### Table 1：Main benchmark table

列：

- Method
- Input condition
- Test scope
- match@1
- match@5 或 match@20
- RMSE@1
- RMSE@5 或 RMSE@20
- Notes

重点：

- CrystaLLM-a 的 @5 不可用，不要硬填。
- SymCIF 和 CrystaLLM-a input condition 必须单独列。

### Table 2：Ablation table

行：

- current order。
- policy rank only。
- train WA prior only。
- skeleton prior only。
- hybrid-prior。
- oracle GT-WA diagnostic。

列：

- WA_hit@1/@5。
- match@1/@5。
- RMSE@1/@5。
- strict_valid_any@5。

### Supplementary Table S1：Dataset and extraction audit

内容：

- original MP-20 test = 9,046。
- structured test = 8,893。
- extraction failure = 153。
- render/eval success。
- timeout summary。

### Supplementary Table S2：Complex subset performance

内容：

- n_sites>=6。
- n_sites>=12。
- num_elements>=4。
- rare_sg。
- high_multiplicity_orbit。
- extraction_hard。

## 6. 必补实验

### 必补 1：selector ablation

目的：证明 hybrid-prior 每个组成部分有贡献。

最少需要：

- current order。
- policy_rank only。
- old_rank only。
- train WA frequency only。
- train skeleton frequency only。
- hybrid-prior full。

指标：

- WA_hit@1/@5。
- match@1/@5。
- RMSE@1/@5。

优先级：最高。

### 必补 2：bootstrap confidence interval

目的：让 match@1 超过 CrystaLLM-a 的结论更稳。

建议：

- 对 8,893 structured test samples bootstrap 10,000 次。
- 报告 match@1、match@5 的 95% CI。
- 原始分母折算也给 CI。

优先级：最高。

### 必补 3：input-condition fairness experiment

目的：回应 SymCIF 使用 GT space group 的公平性问题。

可选路线：

- Route A：跑 CrystaLLM + GT-SG prompt，同 evaluator 对比。
- Route B：跑 SymCIF without oracle SG，用 predicted/retrieved SG，报告性能下降。
- Route C：明确把论文定位为 symmetry-conditioned generation，不追求 same-input SOTA claim。

优先级：最高，至少要选一个。

### 必补 4：geometry bottleneck ablation

目的：解释 RMSE 为什么落后。

建议：

- GT-WA + current geometry。
- selected WA + oracle/best geometry diagnostic。
- geometry_quality diagnostic rerank。
- strict_valid-first diagnostic rerank。

已有部分 diagnostic 结果，但需要整理成正式表。

优先级：高。

### 必补 5：train-frequency / novelty analysis

目的：防止 reviewer 认为只是 train prior 记忆。

建议分桶：

- GT WA seen in train vs unseen。
- GT skeleton seen in train vs unseen。
- rare SG vs common SG。
- train WA frequency quantiles。

指标：

- WA_hit@5。
- match@5。
- RMSE@5。

优先级：高。

### 必补 6：MPTS-52 或 external generalization

目的：增强论文范围。

如果 MPTS-52 结果弱：

- 作为边界分析，不放主 claim。
- 解释 MP-20 与 MPTS-52 分布差异、复杂度差异。

如果 MPTS-52 结果强：

- 可升级为 “across benchmarks” claim。

优先级：中。

### 必补 7：qualitative examples

目的：帮助读者理解 match 成功/失败是什么样子。

建议：

- 2 个成功案例：top1 matched, low RMSE。
- 1 个 WA correct but geometry fail。
- 1 个 WA missing fail。
- 1 个 complex subset fail。

优先级：中。

## 7. 三种写法策略

### 7.1 保守版

适合当前证据不补太多实验时使用。

核心 claim：

> SymCIF is a symmetry-conditioned and diagnostically transparent pipeline for crystal structure generation. On MP-20 structured test samples, it improves top-1 recovery over a previous SymCIF ordering and exceeds the published CrystaLLM-a match@1 reference under a space-group-conditioned setting, while revealing geometry quality as the remaining bottleneck.

中文主张：

- 我们提出一个 symmetry-conditioned pipeline。
- 在 MP-20 structured test 上 match@1 很强。
- 与 CrystaLLM-a 比较时主动说明输入不同。
- 不声称全面 SOTA。
- 强调可诊断性和后续 geometry 改进空间。

优点：

- 最安全。
- 不容易被公平性问题打穿。
- 适合先写论文初稿。

缺点：

- 冲击力较弱。
- 可能被认为只是工程 pipeline + reranking。

### 7.2 中版

适合补齐 selector ablation、bootstrap CI、fairness 说明后使用。

核心 claim：

> By decomposing crystal generation into Wyckoff assignment selection and geometry rendering, SymCIF achieves strong MP-20 recovery, raising match@1 from 51.29% to 60.90% and outperforming the published CrystaLLM-a match@1 reference even after accounting for structured-extraction failures. Diagnostic ablations show that the gain is primarily driven by improved Wyckoff assignment coverage, whereas geometry refinement remains the main source of residual error.

中文主张：

- SymCIF 的关键是 decomposition。
- hybrid-prior 把 current match@1 提升 9.61 pp。
- 原始分母折算后仍超过 CrystaLLM-a match@1。
- ablation 证明 WA coverage 是主要提升来源。
- RMSE 仍落后，作为诚实边界。

优点：

- 有强主结果。
- 也有机制解释。
- 比保守版更像完整方法论文。

风险：

- 仍需处理 input condition fairness。
- 需要把 CrystaLLM-a comparison 写得非常严谨。

### 7.3 强版

只有在补齐 GT-SG fairness / MPTS-52 / geometry ablation 后才建议使用。

核心 claim：

> SymCIF establishes a symmetry-first alternative to language-model-only CIF generation, achieving superior top-1 recovery on MP-20 and providing a transparent route to diagnose and improve failures through separable Wyckoff assignment and geometry modules.

中文主张：

- SymCIF 是 language-model-only CIF generation 的 symmetry-first 替代方案。
- 不只是一个 reranker，而是完整 structured generation framework。
- 在 MP-20 上 top-1 recovery 优于强 published reference。
- 在 MPTS-52 或 fairness setting 上也能证明泛化/稳健性。
- failure audit 证明该框架可持续改进。

优点：

- 论文冲击力最强。
- 适合投高水平方法/材料 AI 方向。

风险：

- 如果没有 same-input baseline 或泛化实验，强版容易被 reviewer 攻击。
- RMSE 落后会削弱“全面优于”的叙述。

## 8. 推荐当前写作决策

当前最推荐采用“中版偏保守”的写法：

1. 主标题和摘要不要写 “state-of-the-art crystal generation”。
2. 可以写 “strong top-1 recovery on MP-20 under symmetry-conditioned generation”。
3. 主结果突出 match@1 超过 CrystaLLM-a reference。
4. 同一段内必须说明输入条件不同和 RMSE 仍落后。
5. Results 用 WA coverage ablation 解释提升。
6. Discussion 把 geometry refinement 作为明确下一步，而不是回避。

## 9. 当前最该先写的论文骨架

建议先写 5 个文件：

1. `01_title_and_abstract.md`
2. `02_introduction_outline.md`
3. `03_results_storyline.md`
4. `04_methods_outline.md`
5. `05_limitations_and_claim_boundaries.md`

这五个文件足够形成第一版 manuscript skeleton，然后再根据补实验填数字。

