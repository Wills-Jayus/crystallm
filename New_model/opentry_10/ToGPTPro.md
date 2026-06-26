# opentry_10 当前实验情况总结

更新日期：2026-06-26

本文档的目的不是罗列文件路径，而是把 `opentry_10` 到目前为止的实验逻辑、做法、结果、失败原因和下一步判断讲清楚。重点是帮助外部判断：当前方向是否合理，为什么 selector/rerank 没有稳定超过 CrystaLLM SG-GT baseline，以及下一步应不应该继续沿着普通 ranking 策略做。

## 1. 最终目标

`opentry_10` 的唯一目标是：在 CrystaLLM Table 3 official full-test K20 协议下，做出一个冻结系统，真正超过 CrystaLLM-a GT-SG baseline。

当前 official-test baseline 是：

| dataset | baseline | match@1 | match@5 | match@20 | rows>=7 match@20 |
| --- | --- | ---: | ---: | ---: | ---: |
| MP-20 | CrystaLLM-a GT-SG | 71.67% | 83.08% | 87.81% | 82.61% |
| MPTS-52 | CrystaLLM-a GT-SG | 25.23% | 36.46% | 43.96% | 41.04% |

正式成功标准：

- 至少一个冻结系统在 official test 上有至少一个 match 指标超过对应 GT-SG baseline `>= 1.0 percentage point`；
- 同一个系统的 match@20 不能下降超过 `0.2pp`；
- rows>=7 子集不能明显变差；
- 不能靠补齐缺失样本刷分；
- 不能用 test target、test StructureMatcher label 或 test oracle selection 调参。

截至 2026-06-26，`opentry_10` 还没有 official-test 成功系统。

## 2. 为什么 opentry_10 走 selector / rerank 路线

历史 SymCIF / WyFormer / CrystalFormer 风格路线有过正向信号，但没有稳定超过 CrystaLLM SG-GT baseline。主要问题是：

- space group、Wyckoff skeleton、Wyckoff assignment 可以变得更可控；
- 但最终 StructureMatcher 是否匹配，强依赖连续几何、晶胞参数、自由坐标、site mapping；
- rows>=7 大结构尤其困难；
- 很多结构化方法在局部子集或符号层有进步，但 match@20、RMSE、rows>=7 不稳。

因此 `opentry_10` 采用较保守路线：

1. 生成完整 validation CrystaLLM SG-GT K100 候选池；
2. 评估 K20 以外是否存在更多正确结构；
3. 如果 K50/K100 里有额外正确候选，训练 selector/reranker 把它们排进 top1/top5/top20；
4. 只有 validation OOF 足够强，才冻结策略跑 official test。

这条路线的前提是：候选池中已经有正确结构，但原始顺序没有把它们放到前面。

## 3. 白话术语

### Validation Anchor

Validation Anchor 是在 validation set 上生成大量候选，然后用真实结构评估。

它回答的是：多生成候选以后，候选池里到底有没有更多正确结构？

例如 K20 到 K50 明显提升，说明第 21 到 50 个候选里确实有额外正确答案。这个阶段只证明“池子里有鱼”，不等于最终系统能把鱼捞出来。

### Selector / Rerank

Selector / Rerank 是训练一个排序或筛选策略，把 K50/K100 里的候选重新排进 top1/top5/top20。

它回答的是：能不能在不看 test label 的情况下，把 validation 里发现的额外正确候选稳定挑出来？

正式判断必须用 group OOF：同一个材料的候选必须在同一个 fold 里，避免泄漏。

### Official Test

Official Test 是最终交卷。

流程必须是：validation 上选好策略 -> 冻结策略 -> 跑 official test -> 看是否超过 baseline。跑完 test 后不能再根据结果调同一个方法，否则就是 test leakage。

## 4. Phase 0：候选池、prompt 和可恢复控制器

已经完成：

- 验证 official train/validation/test split 数量；
- 追溯 CrystaLLM SG-GT benchmark model、tokenizer、prompt 生成方式和采样参数；
- 构建 shard 级生成、断点续跑、日志记录、失败重试；
- 重建 validation SG-GT CIF cache，使用 `symprec=0.1` conventionalize；
- MP-20 和 MPTS-52 均完成 validation K100 候选池生成；
- MP-20 和 MPTS-52 均完成 K50 label pass。

这个阶段的主要价值是补齐以前缺失的 validation candidate bank。没有完整 validation CrystaLLM SG-GT 候选池，selector/ranker 不能合法训练和冻结。

## 5. MP-20 最新结果

### 5.1 Validation anchor

MP-20 validation K100 生成已完成：

- validation samples：9047；
- K100 候选总量：904700；
- K50 label pass：452350/452350 完成。

可靠 validation 指标如下：

| budget | match_rate | RMSE | rows>=7 match_rate |
| --- | ---: | ---: | ---: |
| K1 | 72.731% | 0.05147 | 29.135% |
| K5 | 83.088% | 0.04365 | 44.291% |
| K20 | 87.631% | 0.04049 | 55.571% |
| K50 | 89.919% | 0.04009 | 62.076% |

结论：

- K20 基本复现 SG-GT anchor 水平；
- K50 比 K20 高 `+2.288pp`；
- K50 相比 K20 多救回 207 个 validation material；
- rows>=7 从 K20 的 `55.571%` 到 K50 的 `62.076%`，提升 `+6.505pp`。

MP-20 候选池有真实 headroom，但整体 headroom 不大。

### 5.2 MP-20 selector / route sweep

做过的主要实验：

- rerank-only：只重排原始 K20；
- conservative K50 selector：保留 anchor prefix，只用 K21-K50 替换 tail slots；
- residual route sweep；
- sample-level residual classifier；
- HGB 多 seed ensemble fast route sweep。

最新 validation 现象：

| 方法 | 最好 validation 现象 | 问题 |
| --- | --- | --- |
| rerank-only HGB | match@1 +0.707pp；match@5 -0.122pp；match@20 不变 | 不够冻结 |
| conservative K50 keep14 | match@20 +0.343pp；rows>=7 match@20 +0.969pp | 整体提升太小 |
| residual classifier | match@20 +0.376pp；rows>=7 match@20 +0.900pp | 仍不够 |
| HGB ensemble margin route | match@1 +1.249pp；rows>=7 match@1 +0.900pp | match@5 -0.077pp；match@20 -0.066pp |

MP-20 当前结论：

- K50 候选池确实有用；
- 但普通 selector 只能回收一小部分；
- MP-20 目前没有可冻结 official-test 系统；
- MP-20 更像是 shallow selector 的上限不够，而不是单纯阈值没调好。

## 6. MPTS-52 最新结果

### 6.1 Validation anchor

MPTS-52 validation K100 生成已完成：

- validation samples：5000；
- K100 候选总量：500000；
- K50 label pass：250000/250000 完成。

Validation anchor 指标：

| budget | match_rate | RMSE | rows>=7 match_rate |
| --- | ---: | ---: | ---: |
| K1 | 30.020% | 0.12555 | 5.323% |
| K5 | 40.480% | 0.12027 | 9.991% |
| K20 | 48.000% | 0.12916 | 14.747% |
| K50 | 52.720% | 0.13424 | 18.325% |

结论：

- K50 比 K20 高 `+4.720pp`；
- K50 相比 K20 多救回 236 个 validation material；
- rows>=7 从 `14.747%` 到 `18.325%`；
- MPTS-52 的候选池 headroom 比 MP-20 更大。

### 6.2 Rerank-only official test

Rerank-only 只重排原始 K20，不引入 K21-K50。

Validation OOF 最好结果是 HGB seed2：

| metric | baseline val | rerank val | delta |
| --- | ---: | ---: | ---: |
| match@1 | 30.020% | 31.240% | +1.220pp |
| match@5 | 40.480% | 41.160% | +0.680pp |
| match@20 | 48.000% | 48.000% | 0 |

但 official test 失败：

| metric | GT-SG baseline | rerank-only HGB seed2 | delta |
| --- | ---: | ---: | ---: |
| match@1 | 25.23% | 24.94% | -0.29pp |
| match@5 | 36.46% | 35.93% | -0.53pp |
| match@20 | 43.96% | 43.96% | 0 |

结论：

- validation 上的 rerank-only 信号没有泛化到 official test；
- match@20 没掉，但 match@1 和 match@5 低于 baseline；
- 不能再根据这个 official-test 结果调同一个 rerank-only 方法。

### 6.3 K50 / K30 score route validation sweep

最新 validation-only route sweep 里，RF seed1 的 K30 route 最强：

| route | validation delta@1 | validation delta@5 | validation delta@20 | rows>=7 delta@20 |
| --- | ---: | ---: | ---: | ---: |
| MPTS-52 K30 RF seed1 best_score route | +2.940pp | +1.680pp | +0.680pp | +0.087pp |
| MPTS-52 K50 RF ensemble margin route | +2.360pp | +1.000pp | +0.400pp | -0.175pp |

这说明 RF score 有真实信号，但 rows>=7 和 validation/test 稳定性仍有风险。

### 6.4 K30 official test

已冻结并跑完：

| metric | anchor | official | delta |
| --- | ---: | ---: | ---: |
| match@1 | 25.23% | 26.075% | +0.845pp |
| match@5 | 36.46% | 36.228% | -0.232pp |
| match@20 | 43.96% | 44.059% | +0.099pp |

结论：

- K30 RF route 是目前 official test 上最接近成功的系统；
- match@1 距离 +1pp 只差约 `0.155pp`；
- 但正式标准没有达到；
- match@5 低于 anchor，因此不能宣称成功。

### 6.5 K50 official test

已冻结并跑完：

| metric | anchor | official | delta | rows>=7 anchor | rows>=7 official | rows>=7 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| match@1 | 25.23% | 25.791% | +0.561pp | 22.49% | 23.053% | +0.563pp |
| match@5 | 36.46% | 36.265% | -0.195pp | 33.37% | 33.228% | -0.142pp |
| match@20 | 43.96% | 43.824% | -0.136pp | 41.04% | 40.939% | -0.101pp |

结论：

- K50 RF margin route 也没有成功；
- match@1 有正向提升，但小于 +1pp；
- match@5 和 match@20 均略低于 anchor；
- rows>=7 没有大崩，但也没有形成足够正向收益。

## 7. 底层数据诊断

### 7.1 原始 rank 不是质量排序

对 validation labels 直接统计：

- MP-20 rank1 match 约 `72.7%`，rank20 约 `72.9%`；
- MPTS-52 rank1 match 约 `30.0%`，rank20 约 `30.3%`。

rank 1 到 rank 20 的单候选 match 率基本是平的。也就是说，CrystaLLM 的原始生成顺序不是 StructureMatcher 质量顺序。它更像是生成/采样顺序，而不是“好结构优先”的排序。

这解释了为什么普通 rerank 会有效但不稳：它不是在修一个已有强排序，而是在从很弱的原始顺序里重新找质量信号。

### 7.2 K50 有 headroom，但 selector 没有可靠回收

MP-20：

- K20 -> K50 oracle headroom：`+2.288pp`；
- 现有 selector 回收通常只有 `+0.3pp` 到 `+0.4pp`，或者提升 top1 但伤害 K5/K20。

MPTS-52：

- K20 -> K50 oracle headroom：`+4.720pp`；
- RF score-sort validation 可带来 top1/top5 提升；
- official test 中 K30 只保住 `+0.845pp` match@1，没过 +1pp 门槛；
- K50 official 更弱。

结论：候选池不是完全没正确结构，但当前 scorer 没能稳定识别正确候选。

### 7.3 当前 scorer 的特征太浅

当前 scorer 主要用：

- rank；
- generation config；
- CIF 字符数/行数；
- atom_site_rows；
- declared SG；
- cell 参数、volume、Z；
- parse_ok、valid、sensible；
- rank inverse 等简单派生特征。

它没有真正使用：

- 目标组成 vs 候选组成的一致性；
- prompt SG vs 候选 SG 的显式一致性；
- Wyckoff multiplicity exact-cover；
- 候选独立位点数是否满足公式和 SG 的可行轨道组合；
- 最短距离、碰撞、局部配位、bond-valence；
- energy/relaxation proxy；
- graph/局部环境 embedding；
- rows>=7 专门质量信号。

因此它训练的是“CIF 看起来像不像会命中”的模型，而不是“这个结构是否真的接近目标”的模型。

### 7.4 组成和 atom rows 是强信号，但当前没有用好

底层统计：

MPTS-52：

- 候选公式等于目标公式时，候选 match 率约 `30.844%`；
- 候选公式不等于目标公式时，match 率只有 `1.130%`；
- atom-site 行数等于真实结构行数时，match 率约 `41.797%`；
- atom-site 行数不等时，match 率只有 `0.886%`。

MP-20：

- 候选公式等于目标公式时，match 率约 `73.834%`；
- 候选公式不等于目标公式时，match 率只有 `6.200%`；
- atom-site 行数等于真实结构行数时，match 率约 `80.366%`；
- atom-site 行数不等时，match 率只有 `2.225%`。

注意：真实 atom-site 行数不能在 test 直接使用，否则泄漏。但可以从 composition + SG 出发，做 Wyckoff exact-cover / feasible orbit count / predicted independent-site-count 这类 inference-safe 特征。

### 7.5 只靠公式和 rows 仍不够，下一层是局部几何

对 score top 错误样本诊断：

MPTS-52 中，K50 里至少有一个正样本的 2636 个 validation sample，RF scorer 有 1026 个 sample 把错误候选排到最高分。

这些错误 top 候选中：

- `97.5%` 公式仍然是对的；
- `72.3%` atom-site 行数也等于目标行数；
- 很多错误候选和正确候选在浅层约束上都像是“合理结构”。

这说明下一步必须加入局部几何、距离、配位、碰撞、Wyckoff/free-param consistency、energy proxy。否则 scorer 无法区分“公式/SG/行数都对但几何错”的 hard negatives。

## 8. 历史根因与当前失败的对应关系

opentry_4 / opentry_5 / opentry_6 的核心结论并没有过时：

- rows>=7 很多不是 W/A 完全没有，而是 W/A 或 skeleton 有信号后，连续几何仍失败；
- 常见失败包括 free parameter、site mapping、短距离碰撞、晶胞体积/密度异常；
- selector / energy / anchor insertion 可以诊断局部收益，但不能解决候选池里缺少 match 几何的问题；
- rows>=7 如果 K50/K100 里没有真正 match 候选，任何 ranker 都无能为力。

当前 `opentry_10` 进一步证明：

- CrystaLLM K50 确实有额外正确候选；
- 但 shallow scorer 不足以稳定把正确候选排到前面；
- official test 的 validation-to-test 衰减是真实风险；
- 继续只调 HGB/RF/threshold/anchor_keep，大概率仍然在同一个局部最优里打转。

## 9. 当前已经成立的结论

1. `opentry_10` 已经补齐以前缺失的 validation candidate bank。
2. MP-20 和 MPTS-52 都证明 K50 候选池有真实额外正确结构。
3. MPTS-52 的 K50 oracle headroom 比 MP-20 更大。
4. 普通 rerank/selector 有信号，但不能稳定达到 official-test 成功标准。
5. MPTS-52 K30 RF route 是目前 official test 上最接近成功的系统，但仍失败。
6. 当前失败不是“完全没有正确候选”，而是“缺少能识别优质结构的结构质量信号”。
7. rows>=7 不能再当普通子集处理，它需要专门候选生成和几何质量建模。

## 10. 还不能成立的结论

1. 不能说已经超过 CrystaLLM SG-GT baseline。
2. 不能说 selector/rerank 路线已经成功。
3. 不能说 K30/K50 route 只差调阈值。
4. 不能用已经跑过的 official test 继续反向调同一个方法。
5. 不能把 rows>=7 的失败归因成普通排序噪声。

## 11. 下一步最靠谱的技术路线

不要继续只换 HGB/RF seed、threshold、anchor_keep。下一步应该做“结构质量 scorer”，而不是普通 selector。

### 11.1 候选池仍可用

最快路线仍然可以从 CrystaLLM K50/K100 候选池开始，因为 MPTS-52 K50 有 `+4.720pp` validation oracle headroom，K30 official 已经做到 match@1 `+0.845pp`。

这说明不一定要立刻重训大模型，先增强 scorer 有现实价值。

### 11.2 必须新增的 inference-safe 特征

建议新增：

- candidate formula vs prompt formula 的归一化一致性；
- candidate SG vs prompt SG 的一致性；
- Wyckoff multiplicity exact-cover；
- 候选 atom-site rows 是否满足 composition + SG 可行轨道组合；
- orbit multiplicity gap、element-orbit assignment consistency；
- 最短原子距离、元素对距离下界、碰撞数；
- volume/atom、density、Z、packing 合理性；
- 局部配位数、bond-valence 异常；
- graph/local-environment descriptor；
- 可选 energy 或 relaxation proxy；
- rows>=7 专门特征和单独 gate。

这些特征都必须只依赖 prompt 条件和候选 CIF，不能依赖 test true CIF。

### 11.3 训练目标要改成 hard-negative pairwise/listwise ranking

不要再做普通 candidate 二分类。

应该在同一个 material 内构造 pair：

- positive：StructureMatcher match；
- hard negative：parse_ok、valid、formula 对、SG 对、rows/轨道可行，但 StructureMatcher 不 match；
- 目标：positive score 必须高于 hard negative。

rows>=7 应该单独加权或单独训练 scorer，因为它的失败模式和普通样本不同。

### 11.4 冻结策略要保守

建议两阶段：

1. 硬过滤明显不可能候选：公式错、SG 错、Wyckoff cover 不可行、严重碰撞、密度离谱；
2. 用结构 scorer 排 K50/K100；
3. 只有 scorer margin 足够大时替换 anchor；
4. rows>=7 用单独 guard，不能为了整体 top1 牺牲 rows>=7 K5/K20。

### 11.5 rows>=7 需要回到生成 / geometry refiner

如果 rows>=7 样本的 K50/K100 里没有 match 候选，ranker 没意义。

这部分应该回到：

- Wyckoff exact-cover 生成；
- continuous geometry refiner；
- free-param / lattice / site mapping 联合修正；
- collision-aware sampling；
- row>=7 专门训练数据和评估。

## 12. 当前一句话状态

`opentry_10` 已经证明 CrystaLLM SG-GT K50 候选池存在真实 headroom，但现有 selector/rerank 没有稳定把 headroom 转成 official-test 胜利。MPTS-52 K30 RF route 已经非常接近成功，但仍未超过 +1pp 门槛；K50 route 也失败。根本问题不是“再调一个 rank 策略”，而是当前 scorer 缺少能解释 StructureMatcher match 的结构质量信号。

## 13. 附录：根本原因分析原文

下面是 2026-06-26 对当前所有实验和底层数据重新检查后的根因判断。

结论：之前很多后期实验确实在“换 rank 策略”和“换模型路线”之间摆动，但没有持续抓住根因。真正深入到底层的是 opentry_4/6，那时已经发现 rows>=7 的核心失败不是排序，而是 W/A 有了以后几何、自由参数、site mapping、碰撞、晶格仍然错。opentry_10 又回到了浅层 selector，所以 validation 看着有收益，official test 就衰减。

我看到的根因是这个：

1. 原始 rank 不是质量排序。MPTS-52 验证集 rank1 match 30.0%，rank20 30.3%；MP-20 rank1 72.7%，rank20 72.9%。基本是平的。也就是说 CrystaLLM 的生成顺序不是 StructureMatcher 质量顺序，不能指望 rank 自带“好结构靠前”。

2. 当前 selector 看不到关键变量。现在的 scorer 主要用 rank、生成配置、CIF 长度、cell 参数、atom_site_rows、SG、valid/parse_ok。它没有真正使用目标组成一致性、Wyckoff exact-cover、局部配位、最短距离、bond valence、碰撞、局部环境、energy/relaxation proxy。底层数据说明这些变量很关键：MPTS-52 里候选公式匹配目标时 match 率 30.84%，不匹配只有 1.13%；atom-site 行数等于真实行数时 match 率 41.80%，不等时只有 0.89%。这些不是小信号，是强因果信号。

3. 当前 ranker 会把“看起来合理但几何错”的结构排前面。MPTS-52 中 K50 有正样本的 2636 个样本里，RF scorer 有 1026 个样本把错误候选排到最高分。更关键的是，这些错误 top 候选里 97.5% 公式也对，72.3% atom-site 行数也对。说明只加公式/行数还不够，下一层必须加局部几何和物理合理性特征。

4. rows>=7 不是简单 rerank 问题。历史 opentry_4 已经说明 rows>=7 里很多是 W/A 或骨架有 hit，但 match 失败率极高，主要落在 free-param/site mapping、collision/short distance、lattice volume 这些桶里。也就是说候选本身经常没有真正 match 的几何结构，排序不可能把不存在的好结构排到前面。

最快、最靠谱的路线不是继续换 HGB/RF 阈值，而是做一个“结构质量 scorer”，不是普通 selector：

1. 先继续用 CrystaLLM K50/K100 候选池，不急着重训大模型。MPTS-52 K50 validation oracle headroom 是 +4.72pp，说明这里还有可挖空间；official K30 已经做到 match@1 +0.845pp，离 +1pp 只差 0.155pp，但现有浅层特征不稳。

2. 重建 scorer 特征，必须加入这些 inference-safe 结构特征：目标公式 vs 候选 formula_sum/structural 的归一化一致性；候选 SG 是否等于 prompt SG；Wyckoff multiplicity exact-cover；候选 atom-site rows 是否满足由公式+SG推导出的可行轨道数；最短原子距离、元素对距离下界、碰撞数；volume/atom、density、Z 合理性；局部配位数和 bond-valence 异常；可选 MLIP/CHGNet/M3GNet 单点能量或 relaxation proxy。

3. 训练方式改成 pairwise/listwise hard-negative ranking：不要再做普通 candidate 二分类。每个 material 内，让 true match 候选必须排过“valid=true、formula 对、rows 对但 match=false”的 hard negatives。rows>=7 单独加权或单独训练一个 scorer，因为它的失败机制和普通样本不同。

4. 决策策略用两阶段：第一阶段硬过滤明显不可能的候选：公式错、SG 错、Wyckoff cover 不可行、严重碰撞、密度离谱。第二阶段用结构 scorer 排 K50/K100。最后只在 scorer margin 足够大时替换 anchor，避免 K5/K20 被误伤。

5. 如果目标是 rows>=7，大方向要回到生成/几何 refiner：对 K50 里没有 match 的 rows>=7 样本，ranker 没意义。需要 Wyckoff exact-cover 生成 + continuous geometry refiner，把 W/A、自由参数、lattice、site mapping 一起修。WyFormer/CrystalFormer 的启发就在这里：它们用 Wyckoff/SG 条件化表示，不是在 CIF 文本顺序上硬排。CrystaLLM 本身也提到用 formation-energy predictor/MCTS 改善生成质量。

我不能诚实地说“99% 一定成功”，因为 official test 已经证明 validation 到 test 有衰减，任何没跑过的方案都不能给这种概率。但如果只问现在最应该做什么：不要再继续浅层 rank 策略微调。最快超过 MPTS-52 SG-GT CrystaLLM baseline 的高概率路线，是在现有 K50/K100 候选上补结构质量特征 + hard-negative pairwise scorer；同时把 rows>=7 作为几何生成/refiner 问题处理，而不是继续让 selector 背锅。

