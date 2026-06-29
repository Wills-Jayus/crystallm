你现在的目标不是继续做普通调参，也不是继续做简单 rerank/fusion，而是完成一组围绕“对称感知晶体生成”的系统实验。请把下面所有实验都做完，每个实验都要给出结果、结论、失败原因和下一步判断。不要把某个实验的成败作为是否执行后续实验的条件，所有实验都需要独立完成。

核心目标是：在 MP-20 和 MPTS-52 上，尽量让 GT-SG 条件下的 CrystaLLM / SymCIF 生成结果超过 baseline，至少在两个 match 指标上达到 +5pp。重点关注 match@1、match@5、match@20，同时必须单独分析 rows>=7 的复杂结构表现。

实验 1：Track A frozen official 泛化验证

先把当前最好的 Track A 版本视为已经冻结的候选方案，不要继续调参数。
这个实验只回答一个问题：Track A 在 validation OOF 上的提升，能不能泛化到 MPTS-52 official full-test。

需要比较：

原 GT-SG baseline；
当前 frozen Track A；
overall match@1 / match@5 / match@20；
rows>=7 match@1 / match@5 / match@20。

实验逻辑是：
如果 Track A 只提升 validation，但 official 不提升，说明它只是 validation 局部有效；如果 official 也稳定提升，说明它可以作为后续系统里的辅助结构质量模块。但无论结果如何，都不要再回头调 Track A。

实验 2：Track A 失败归因分析

只在 validation 上做 Track A 的错误分析。
不要为了提升指标而做这个实验，它的目的只是搞清楚 Track A 到底救了哪些样本，又害了哪些样本。

需要把样本分成几类：

baseline 错、Track A 对；
baseline 对、Track A 错；
两者都对；
两者都错；
rows<7 与 rows>=7 分开分析；
formula 对但结构错；
SG 对但 Wyckoff 错；
Wyckoff skeleton 对但 geometry 错；
lattice / free parameter / site mapping / collision 导致失败。

这个实验的目标是弄清楚：当前瓶颈到底是候选排序问题，还是候选池 coverage 问题，还是 skeleton 命中后 geometry 没转化成 StructureMatcher match。

实验 3：Wyckoff exact-cover 诊断实验

构造一个可以分析候选结构 Wyckoff skeleton 是否合理的实验。
重点不是生成新结构，而是判断现有候选在 composition + GT-SG 条件下是否满足合理的 Wyckoff exact-cover。

需要分析：

候选是否满足 composition；
是否满足 GT-SG；
Wyckoff multiplicity 是否能 exact-cover；
rows 数是否合理；
equivalent positions 是否一致；
skeleton feasible 但最终 match 失败的比例；
skeleton 不 feasible 但被原模型排到前面的比例。

实验逻辑是：
如果大量错误候选其实连 exact-cover 都不满足，说明后续需要 stronger crystallographic constraint；如果很多候选 exact-cover 满足但 match 失败，说明后续重点应该放在 geometry repair。

实验 4：hard-negative structural scorer v2

在 Track A 的基础上，做一个更接近晶体学本质的 structural scorer。
它不能只是学习原始 rank、CIF 长度、atom rows、cell 参数这些浅层特征，而要重点区分“看起来合理但晶体学上错误”的 hard negatives。

需要让 scorer 学会区分：

formula 正确但 geometry 错；
SG 正确但 Wyckoff 错；
exact-cover 正确但 site mapping 错；
没有明显 collision 但 StructureMatcher 不 match；
top-rank 错误候选和 lower-rank 正确候选。

这个实验的目标不是只提升 match@1，而是同时提升 match@5、match@20，尤其要提升 rows>=7。
如果它只涨 K1，不涨 K5/K20，就说明它仍然只是浅层排序器，不足以作为主方法。

实验 5：symmetry-preserving geometry repair

这是最重要的实验之一。
目标是在固定 composition + GT-SG + Wyckoff skeleton 的前提下，修复 lattice、fractional free parameters、site mapping、collision 和局部几何，使 skeleton-hit 的候选更容易变成 StructureMatcher match。

实验逻辑是：

先找到 skeleton 已经合理但最终 match 失败的候选；
不改变 SG 和 Wyckoff skeleton；
只修几何部分；
比较修复前后 StructureMatcher match 是否提升；
单独统计 rows>=7 的修复成功率。

这个实验要重点看：

skeleton-hit-to-match conversion；
collision-free rate；
lattice plausibility；
match@5 / match@20；
rows>=7 match@20。

如果这个实验能显著提升 K20 和 rows>=7，才说明我们不是在做普通 rerank，而是真的解决了晶体生成瓶颈。

实验 6：rows>=7 复杂结构专门实验

把 rows>=7 作为独立实验对象，不要只放在附录分析里。
当前复杂结构是主要瓶颈，所以需要单独看它为什么失败。

需要分析：

rows>=7 的正确候选是否存在于 top20 / top50；
正确候选一般排在什么位置；
是 skeleton 没命中，还是 skeleton 命中了但 geometry 失败；
复杂结构是否更容易出现 collision；
lattice error 是否更大；
free parameter 是否更难；
site mapping 是否更混乱。

然后对 rows>=7 单独做：

specialized scorer；
specialized geometry repair；
specialized skeleton proposal；
与普通 rows<7 结构分开比较。

目标是 rows>=7 的 match@5 / match@20 明显提升，而不是只让简单结构变好。

实验 7：exact-cover constrained skeleton proposal

做一个生成侧实验，不要只在现有候选里排序。
目标是提高正确 skeleton 的 coverage，尤其是 rows>=7 的候选 coverage。

实验逻辑是：

在 composition + GT-SG 条件下生成或筛选 Wyckoff skeleton；
强制满足 multiplicity exact-cover；
为每个 skeleton 生成多个 geometry proposal；
再交给 geometry repair 和 structural scorer；
比较是否真正提高 match@20。

需要特别注意：
只提高 W/A recall 没有意义，必须证明 exact-cover skeleton 最终能转化成 StructureMatcher match。

实验 8：整合消融实验

最后做一个统一比较，明确每个模块到底贡献了什么。

至少比较这些版本：

原 GT-SG baseline；
baseline + Track A scorer；
baseline + exact-cover filter；
baseline + hard-negative structural scorer v2；
baseline + geometry repair；
skeleton proposal + geometry repair；
skeleton proposal + geometry repair + structural scorer；
rows>=7 specialized route。

每组都要同时报告：

overall match@1 / match@5 / match@20；
rows>=7 match@1 / match@5 / match@20；
valid rate；
formula consistency；
SG consistency；
exact-cover feasible rate；
skeleton-hit-to-match conversion。

最终结论要回答：

哪个模块真正提升了 coverage；
哪个模块只是改变排序；
哪个模块对 rows>=7 最有效；
是否达到至少两个 match 指标 +5pp；
如果没达到，主要瓶颈还剩什么。

整体原则：

不要把普通排序、候选融合、threshold tuning 包装成主贡献。
真正的主线应该是：Wyckoff exact-cover constrained skeleton + symmetry-preserving geometry repair + rows>=7 specialized generation/repair。
Track A 只能作为辅助结构质量模块，不要继续围绕它做大量局部调参。