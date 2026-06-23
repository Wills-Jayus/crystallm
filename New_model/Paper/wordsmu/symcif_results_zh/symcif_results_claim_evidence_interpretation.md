# SymCIF 中文 Results 草稿

## 小节标题建议

1. SymCIF 将 CIF 生成重构为对称性约束的结构化流程
2. Train-prior hybrid reranking 提升 Wyckoff assignment 覆盖
3. SymCIF 在 MP-20 上取得更强的 top-1/top-5 结构匹配表现
4. 候选覆盖与匹配率之间的差距揭示几何质量仍是主要瓶颈
5. Failure audit 将方法边界定位到复杂结构与几何渲染质量

## Claim-Evidence-Interpretation

### 1. SymCIF 将 CIF 生成重构为对称性约束的结构化流程

传统 CIF language modeling 按照文件记录顺序生成 data block、cell parameters、formula、symmetry operations 和 atom_site loop，这一顺序并不等价于晶体形成中的约束顺序，因而容易把 composition、space group、Wyckoff sites、coordinates 和 lattice 之间的依赖关系交给自回归模型隐式学习。SymCIF 将这一过程改写为 composition -> space group -> Wyckoff assignment -> coordinates/lattice -> CIF reconstruction -> validation 的结构化流程，使离散的空间群和 Wyckoff 约束先于连续几何生成被确定。这个结果说明，SymCIF 的核心贡献不是简单生成更像 CIF 的文本，而是把晶体生成问题转化为一个可约束、可重排、可诊断的 symmetry-conditioned generation pipeline。

### 2. Train-prior hybrid reranking 显著改善 Wyckoff assignment 的 top-K 覆盖

在 MP-20 structured test 上，原始 current order 的 WA_hit@1/@5 为 49.80%/71.27%；使用 train-only hybrid-prior selector 后，WA_hit@1/@5 提升到 59.02%/81.66%。对应地，structured test match@1 从 51.29% 提升到 60.90%，match@5 从 66.11% 提升到 73.68%。这一组结果表明，当前 K<=5 设置下，主要增益来自更好的 Wyckoff assignment 排序，而不是简单扩大采样预算；正确的 WA 候选更早进入 top-1/top-5 后，结构匹配率随之提高。

### 3. SymCIF 在 MP-20 全量 structured test 上取得强 top-1/top-5 match 表现

在 8,893 条 MP-20 structured test 样本上，SymCIF hybrid-prior 达到 match@1 = 60.90%、match@5 = 73.68%，RMSE@1 = 0.0573、RMSE@5 = 0.0615。若按原始 MP-20 test 的 9,046 条分母折算，并将 153 条 structured extraction failure 计为失败，match@1 和 match@5 仍分别为 59.87% 和 72.43%。与公开 CrystaLLM-a MP-20 reference 的 match@1 = 55.85% 相比，SymCIF 在当前 symmetry-conditioned 设置下显示出更强的 top-1 recovery 能力；同时，相对旧 SymCIF current order，新的 reranking 也明显降低了 RMSE@1 和 RMSE@5。

### 4. WA 覆盖尚未完全转化为 match，说明几何渲染仍是瓶颈

虽然 hybrid-prior selector 的 WA_hit@5 已达到 81.66%，但最终 match@5 为 73.68%，两者之间仍有约 8 个百分点的差距。这说明一部分样本已经选中了正确的 Wyckoff assignment，但坐标、晶格或结构有效性仍不足以通过 StructureMatcher。进一步地，SymCIF 的 RMSE@1 = 0.0573，仍高于 CrystaLLM-a reference 的 RMSE@1 = 0.0437，表明当前方法更擅长提升结构匹配成功率，而不是已经解决了几何精度问题。因此，后续改进重点应放在 geometry-quality scoring、validity-aware reranking 和 geometry refinement，而不是单纯继续增加 WA 候选数量。

### 5. Failure audit 提供了比总分更清晰的方法边界

本次 full evaluation 覆盖 8,893/8,893 structured test 样本，@1/@5 eval timeout 均为 0，render_success@1 为 100.00%，render_success@5 为 99.10%，说明主结果不是由小样本、不完整运行或 timeout artifact 造成的。与此同时，复杂子集如 n_sites 较大、extraction_hard、部分多元素结构仍表现较弱；而 high-multiplicity orbit 子集表现相对较强，说明“复杂度”并不是单一因素。这个诊断结果支持将 SymCIF 描述为一个可审计的生成框架：它不仅报告最终 match/RMSE，还能把失败拆解为 WA coverage、skeleton coverage、validity、render success 和 geometry mismatch 等具体来源。

## 不能过度声称的内容

- 不能写成 “SymCIF 在同输入条件下全面超过 CrystaLLM-a”，因为当前 SymCIF 使用 composition + oracle ground-truth space group，而 CrystaLLM-a reference 是 composition-only。
- 不能声称 “SymCIF 已经全面达到 state-of-the-art crystal generation”，目前主证据集中在 MP-20，且输入条件与 baseline 不完全一致。
- 不能说 “SymCIF 的几何精度优于 CrystaLLM-a”，因为 RMSE@1 仍高于 CrystaLLM-a reference。
- 不能把 8,893 条 structured test 直接写成原始 MP-20 全量 9,046 条全部成功运行；需要说明 153 条 structured extraction failure 在原始分母折算中计为失败。
- 不能声称 DFT stability 已经被系统性改善，除非后续补充完整 DFT 稳定性实验。
- 不能声称 train-prior hybrid selector 完全不依赖训练分布；它是 train-only prior，仍需要 rare SG、unseen WA、unseen skeleton 等分组分析来证明泛化边界。
- 不能把 WA_hit 提升直接等同于最终结构正确；WA_hit@5 与 match@5 之间的 gap 必须解释为几何/validity 瓶颈。
