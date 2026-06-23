你需要在 `/data/users/xsw/autodlmini/model/New_model` 下新建并完整执行 `opentry_9` 实验。目标是基于此前 opentry_7 / opentry_8 的结论，系统性推进 strategy/fusion 或 pure structural model，使其在官方 CrystaLLM Table 3 MP-20 和 MPTS-52 full-test K20 口径下，真正超过 CrystaLLM-a GT-SG baseline。

请严格按下面方案执行，不要直接盲目跑 full test，不要重复大量阅读历史报告，不要把覆盖修复误判为方法突破。

# 0. 总目标与已知 baseline

官方统一评估协议：

* CrystaLLM Table 3 official MP-20 / MPTS-52 splits。
* 每个 official test sample 最终 20 个 candidate slots。
* StructureMatcher 参数：`ltol=0.3, stol=0.5, angle_tol=10`。
* RMSE 使用 normalized `get_rms_dist(...)[0]`。
* missing / failed candidates 计为失败。
* test label、test StructureMatcher 结果、test oracle WA 不允许用于训练、调参、reranking、threshold selection。
* validation 可以用于训练 selector / ranker / checkpoint selection，但最终 test 前必须冻结全部配置。

当前强基线：

MP-20 CrystaLLM-a GT-SG：

* match@1 = 71.67
* match@5 = 83.08
* match@20 = 87.81
* RMSE@1/5/20 = 0.0509 / 0.0449 / 0.0431
* rows>=7 match@1/5/20 = 62.37 / 76.35 / 82.61

MPTS-52 CrystaLLM-a GT-SG：

* match@1 = 25.23
* match@5 = 36.46
* match@20 = 43.96
* RMSE@1/5/20 = 0.1211 / 0.1257 / 0.1334
* rows>=7 match@1/5/20 = 22.49 / 33.37 / 41.04

opentry_8 结论：

* strategy/fusion 只是 CrystaLLM-a GT-SG anchor + coverage repair，MP-20 完全等同 anchor，MPTS-52 只补了 1 个 sample，因此不是实质突破。
* pure structural MVP 没有通过 validation-generation gate，不能 full-test。
* opentry_7 pure model 本质是 byte-level CIF GPT，不是真正的 structural W/A + geometry model，不能继续沿这条路简单加训。

本轮必须回答两个关键问题：

1. strategy/fusion 是否存在超过 CrystaLLM-a GT-SG 的真实候选互补空间？
2. pure model 的主要瓶颈到底在 W/A 覆盖，还是在 GT-WA 条件下的 geometry renderer？

# 1. 硬性工程约束

1. 所有新脚本、缓存、日志、checkpoint、candidate、metrics、report 只能写入：

`/data/users/xsw/autodlmini/model/New_model/opentry_9`

2. 历史目录只读，包括但不限于：

* `opentry_7`
* `opentry_8`
* `opentry_6`
* `opentry_5`
* `symcif_experiment`

3. 必须使用已有 `crystallm_env` 环境。

4. 禁止事项：

* 禁止用 test StructureMatcher 结果训练 selector。
* 禁止 test-label tuning。
* 禁止 post-test threshold changes。
* 禁止根据 test 结果反复改 strategy 再重跑 test。
* 禁止把 validation/subset/smoke/fold/common-subset/structured-only 结果伪装成 official full-test K20。
* 禁止把 strategy/retrieval/fusion 结果说成 pure model。
* 禁止把没有 full-test 的 pure model 写成正式结果。
* 禁止为了消耗时间而重复读大报告。只读必要 artifact、config、manifest、candidate、metrics。

5. 日志要求：

每完成一个阶段，向 `opentry_9/experiment_log.md` 追加简洁记录：

* 读取了哪些关键文件；
* 生成了哪些 artifact；
* 主要指标；
* 是否通过 gate；
* 下一步动作。

# 2. 目录结构

请创建：

```text
opentry_9/
  configs/
  scripts/
  cache/
  candidates/
  generations/
  eval/
  metrics/
  reports/
  checkpoints/
  frozen_strategy/
  frozen_pure_model/
  experiment_log.md
  final_report.md
```

优先复用 opentry_7 / opentry_8 中已经验证过的 evaluator 工具，但复制到 opentry_9/scripts 后再修改路径保护。

# 3. Phase A：strategy/fusion 可行性审计

这是本轮最高优先级。不要直接训练 selector。先判断候选池是否有超过 CrystaLLM-a GT-SG 的理论空间。

## 3.1 建立候选源清单

在历史目录中查找并整理候选源，至少包括：

1. opentry_7 CrystaLLM-a GT-SG candidates；
2. opentry_8 strategy_fusion candidates，也就是 coverage-repaired GT-SG anchor；
3. opentry_7 strategy/stablekey candidates；
4. symcif_experiment 中 historical stablekey / SymCIF / exact-cover / v5 full-test 或 validation candidates；
5. opentry_5 / opentry_6 中可用于机制分析的 candidates；
6. 如果存在 CrystaLLM GT-SG K50/K100 raw pool，也加入；
7. 如果不存在 K50/K100，则先明确报告缺失，不要假装存在。

输出：

`reports/candidate_source_manifest.md`

内容包括：

* source name；
* dataset；
* split；
* candidate count；
* sample coverage；
* per-sample candidate slots；
* 是否 official full-test；
* 是否 validation；
* 是否 pure model；
* 是否 strategy/retrieval/fusion；
* 是否可用于训练 selector；
* 是否可用于最终 test fusion。

## 3.2 统一候选格式

将所有可用候选转换成统一 JSONL schema：

```json
{
  "dataset": "mp_20 or mpts_52",
  "split": "train/val/test",
  "sample_id": "...",
  "source": "...",
  "source_rank": 0,
  "candidate_id": "...",
  "cif": "...",
  "metadata": {}
}
```

输出到：

`candidates/unified_<dataset>_<split>_<source>.jsonl`

同时计算基础诊断：

* readable；
* formula_ok；
* atom_count_ok；
* SG consistency；
* valid CIF；
* row_count；
* atom_site rows；
* volume/atom；
* shortest distance；
* obvious geometry error；
* canonical_skeleton_key；
* canonical_wa_key；
* candidate-candidate duplicate cluster id。

不要使用 test target match 结果做这些诊断。

## 3.3 validation 上计算 standalone、overlap、exclusive rescue 和 oracle union

只在 validation 上做完整审计。

对每个 dataset 计算：

1. 每个 source 的 standalone match@1/5/20、RMSE@1/5/20；
2. 以 CrystaLLM-a GT-SG anchor 为 A，其他 source 为 B，计算：

```text
A_hit
B_hit
A_and_B_hit
B_exclusive_rescue = B_hit and not A_hit
A_exclusive_loss = A_hit and not B_hit
oracle_union = A_hit or B_hit
```

3. 多 source union：

```text
CrystaLLM anchor
+ stablekey
+ SymCIF exact-cover
+ retrieval/template
+ geometry-refined candidates
```

4. 分桶统计：

* all samples；
* rows>=7；
* high element count；
* low-frequency SG；
* high candidate duplication；
* CrystaLLM top20 failed；
* CrystaLLM top1 failed but top20 hit；
* CrystaLLM top20 failed and rows>=7。

输出：

* `reports/strategy_oracle_union_audit.md`
* `metrics/strategy_oracle_union_audit.json`

关键判断：

* 如果 validation oracle union@20 相比 anchor 提升 < 1pp，则说明当前候选池没有明显覆盖互补，不要跑 official test fusion。
* 如果 validation oracle union@20 提升 >= 2pp，进入 Phase B。
* 如果 validation match@1 rerank upper bound 相比 anchor 提升 >= 5pp，即使 union@20 不高，也进入 Phase B reranking。
* 如果 rows>=7 exclusive rescue 明显高于 all samples，也进入 Phase B residual strategy。

## 3.4 CrystaLLM K20 内部排序空间审计

对 CrystaLLM-a GT-SG validation K20 计算：

* match@1；
* match@5；
* match@20；
* top1 fail but top5 hit；
* top1 fail but top20 hit；
* top5 fail but top20 hit；
* 每个 rank 的 cumulative gain；
* 理论 oracle rerank@1；
* 理论 oracle rerank@5。

输出：

`reports/crystallm_internal_rerank_space.md`

判断：

* 如果 top20 - top1 空间很大，优先训练 ranker。
* 如果 top20 - top5 空间很小，不要强行追求 match@5 大幅提升。
* 如果 match@20 已接近饱和，MP-20 不要把 +5pp match@20 当成第一目标。

# 4. Phase B：strategy selector / reranker

只有 Phase A gate 通过后才执行。

## 4.1 候选去重与多样性选择

对每个 sample 的 candidate pool 做去重：

* exact CIF duplicate；
* canonical WA key duplicate；
* canonical skeleton key duplicate；
* candidate-candidate StructureMatcher duplicate；
* near-duplicate geometry cluster。

构造 final K20 时，不要简单按分数取前 20。使用边际覆盖选择：

```text
score(candidate | selected_set)
= p_match(candidate)
+ novelty_bonus(candidate, selected_set)
+ source_complement_bonus
+ rows_complex_rescue_bonus
- duplicate_penalty
- geometry_risk_penalty
```

要求：

* candidate 0 不一定必须是 CrystaLLM rank0；
* final K20 不能全部被 CrystaLLM 原始顺序占满；
* 对 CrystaLLM K20 高重复样本，用 stablekey/SymCIF/retrieval/geometry variant 替换冗余尾部槽位；
* 对 CrystaLLM top20 failed 高风险样本，给 residual candidates 更多槽位；
* 对普通低风险样本，尽量保守保留 CrystaLLM anchor。

## 4.2 训练 validation-supervised reranker

训练一个 candidate-level ranker，目标是预测 StructureMatcher match label。

训练数据只能来自 train/validation candidate bank，不能用 test match label。

建议模型从简单开始：

1. Logistic Regression / LightGBM / XGBoost / sklearn GBDT；
2. 如果环境不支持复杂库，用 sklearn HistGradientBoosting 或 RandomForest；
3. 不要一开始训练复杂神经网络。

特征至少包括：

* source；
* source_rank；
* CrystaLLM logprob 或可用 score；
* candidate validity；
* formula_ok；
* SG consistency；
* canonical_wa_key train frequency；
* canonical_skeleton_key train frequency；
* candidate cluster size；
* row_count；
* atom_count；
* volume/atom；
* density proxy；
* shortest distance；
* lattice anisotropy；
* composition complexity；
* SG frequency；
* whether duplicate with another source；
* whether source consensus；
* predicted rows>=7 / high-complexity bucket。

训练目标：

* candidate match label；
* 可选辅助目标：RMSE regression 或 low-RMSE classification。

保存：

* `frozen_strategy/ranker.pkl`
* `frozen_strategy/feature_config.json`
* `reports/strategy_ranker_validation_report.md`

## 4.3 validation 冻结策略

在 validation 上比较：

1. CrystaLLM-a GT-SG anchor；
2. simple rerank only；
3. dedup + diversity selector；
4. residual selector only；
5. selector + reranker；
6. selector + reranker + geometry-risk filter。

选择最优 frozen strategy，标准：

* match@1 优先；
* match@20 不能明显下降；
* rows>=7 不得下降；
* RMSE 不得严重变差；
* 至少一个 match 指标超过 CrystaLLM-a GT-SG validation baseline >= 1pp，才允许 official test；
* 若目标是强结果，至少两个 match 指标超过 validation baseline >= 2pp，再跑 official test；
* 若 validation 无任何正增益，停止 strategy 路线并报告原因，不跑 test。

冻结后写入：

* `frozen_strategy/config.json`
* `frozen_strategy/strategy_build_manifest.json`
* `frozen_strategy/no_test_feedback_declaration.md`

## 4.4 official full-test

只有 frozen strategy 通过 validation gate 后，才能对 official test 跑一次。

输出：

* `generations/strategy_<dataset>_test_k20_candidates.jsonl`
* `eval/strategy_<dataset>_test_per_sample.jsonl`
* `metrics/strategy_<dataset>_test_k20.json`
* `reports/strategy_official_test_report.md`

报告必须明确：

* 是否超过 CrystaLLM-a GT-SG；
* 超过多少 pp；
* 是否是 meaningful exceed，而不是 coverage repair；
* 增益来自 reranking、coverage、residual rescue、还是 geometry repair；
* rows>=7 是否提升；
* match@20 是否真的新增了 CrystaLLM 没有覆盖的样本。

# 5. Phase C：pure structural model

不要重复 opentry_7 的 byte-level CIF GPT。pure model 必须是结构化模型。

只有在工程时间允许时执行；但至少必须完成 pure model 的瓶颈诊断。

## 5.1 CIF → SymCIF canonicalization

构建或复用 CIF 到结构化记录的转换，记录：

* formula_counts；
* SG；
* conventional cell multiplier Z；
* row_count；
* Wyckoff skeleton；
* element-WA assignment；
* multiplicity；
* site symmetry；
* enumeration；
* lattice；
* free parameters；
* canonical_skeleton_key；
* canonical_wa_key；
* renderable CIF。

输出：

* `cache/symcif_train.jsonl`
* `cache/symcif_val.jsonl`
* `cache/symcif_test_targets.jsonl`
* `reports/symcif_canonicalization_audit.md`

必须报告 conversion coverage。无法转换的样本不得静默丢弃。

## 5.2 GT-WA geometry upper-bound 实验

这是 pure model 的第一关键实验。

输入：

```text
composition + GT-SG + GT-WA
```

输出：

```text
lattice + free parameters + rendered CIF
```

目标是判断：在离散 WA 已经正确时，geometry renderer 是否足够强。

训练 geometry model：

* lattice head：按 crystal system 预测最小独立 lattice 参数；
* volume/atom 使用 log-space；
* free parameter 使用周期损失；
* 对称等价代表坐标使用 min-over-equivalent loss；
* 辅助 loss：shortest distance、pair distance、volume、local environment；
* 低对称、多自由参数、rows>=7 样本过采样。

validation 生成 K20 geometry variants：

* same WA 多个 geometry samples；
* 不改变 WA；
* render CIF；
* 用统一 evaluator 评估 match/RMSE。

输出：

* `reports/pure_gt_wa_geometry_report.md`
* `metrics/pure_gt_wa_geometry_val.json`

判断：

* 如果 GT-WA geometry match@20 仍明显低于 CrystaLLM-a GT-SG，pure model 的主要瓶颈是 geometry，不要急着训练 WA decoder。
* 如果 GT-WA geometry match@20 很高，说明 geometry 可行，进入 WA decoder。
* 必须报告 rows>=7 的 GT-WA geometry 表现。

## 5.3 WA decoder

实现：

```text
composition + GT-SG
→ Z / conventional cell multiplier
→ row_count
→ Wyckoff skeleton
→ element assignment with exact-cover constraint
```

要求：

* 不能普通字符串生成 WA；
* 必须 exact-cover constrained decoding；
* 每一步 mask 不可能动作；
* EOS 只有当所有元素计数闭合时允许；
* row order 需要 canonical；
* 支持 beam search；
* 输出 top-k diverse canonical WA/skeleton；
* 记录 WA_hit@1/5/20。

训练目标：

* Z classification；
* row_count classification；
* skeleton set prediction；
* element assignment exact-cover；
* WA key top-k coverage。

输出：

* `reports/pure_wa_decoder_val_report.md`
* `metrics/pure_wa_decoder_val.json`
* `checkpoints/wa_decoder_best.pt`

判断：

* 如果 WA_hit@20 不够高，优先改 WA decoder，不要 end-to-end full test。
* 如果 WA_hit 高但 match 低，回到 geometry。

## 5.4 End-to-end pure structural K20

组合：

```text
composition + GT-SG
→ top diverse WA/skeleton
→ exact-cover element assignment
→ geometry variants
→ CIF rendering
→ final K20 order
```

K20 分配规则：

* 先保证 WA/skeleton 多样性；
* 高概率 WA 分配多个 geometry variants；
* 低概率但互补的 WA 保留至少一个候选；
* candidate 0 由 joint score 选择，不用 token greedy；
* final K20 不允许使用 CrystaLLM candidates；
* 不允许 strategy/retrieval candidates 混入 pure model。

checkpoint selection 必须用 validation generation metrics：

* match@1/5/20；
* WA_hit@1/5/20；
* rows>=7 match@1/5/20；
* RMSE；
* render success；
* formula closure；
* SG consistency；
* candidate diversity。

通过 gate 后才允许 official test：

* validation 至少一个 match 指标超过 CrystaLLM-a GT-SG validation baseline >= 1pp；
* 或者至少两个 match 指标接近 baseline 且 rows>=7 明显提升；
* 若没有通过，不跑 test，只写诊断报告。

输出：

* `frozen_pure_model/config.json`
* `frozen_pure_model/checkpoint_manifest.json`
* `reports/pure_structural_val_report.md`
* 若通过 gate，再输出 official test metrics。

# 6. 最终报告要求

无论是否成功，都必须写：

`final_report.md`

必须包含：

1. 本轮到底做了哪些实验；
2. strategy/fusion 是否有真实超过空间；
3. validation oracle union 结果；
4. CrystaLLM K20 内部 rerank 空间；
5. 是否训练了 strategy selector/ranker；
6. frozen strategy 是否通过 validation gate；
7. 是否跑了 official test；
8. official test 是否超过 CrystaLLM-a GT-SG；
9. pure model 的主要瓶颈是 WA 还是 geometry；
10. GT-WA geometry upper-bound 结果；
11. WA decoder 结果；
12. 哪些结果可以写进论文；
13. 哪些结果只能作为诊断；
14. 哪些路线已经证明不值得继续；
15. 下一步最优实验建议。

最终表格至少包括：

## MP-20

| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | candidate budget | fusion/ranking | pure model |
| ------ | ----- | ------: | ------: | -------: | -----: | -----: | ------: | --------------: | --------------: | ---------------: | ---------------: | -------------- | ---------- |

## MPTS-52

同上。

还必须明确写：

* “strategy/fusion 是否 meaningful exceed CrystaLLM-a GT-SG”；
* “pure model 是否 meaningful exceed CrystaLLM-a GT-SG”；
* “超过来自真实互补覆盖、排序改进、几何修复，还是仅仅 coverage repair”；
* “是否存在 test leakage 风险”；
* “是否满足论文可用结论”。

# 7. 成功判定

强成功：

* strategy/fusion 或 pure model 在 official full-test K20 上至少两个 match 指标超过 CrystaLLM-a GT-SG >= 5pp；
* rows>=7 不下降；
* RMSE 不显著恶化；
* 无 test leakage。

中等成功：

* 至少一个 match 指标超过 CrystaLLM-a GT-SG >= 1pp；
* 且能够证明增益来自 reranking/residual rescue/geometry repair，而不是 coverage repair；
* rows>=7 有正收益。

诊断成功：

* 没有 official exceed，但清楚证明：

  1. 当前候选池 oracle union 是否有超过空间；
  2. CrystaLLM K20 是否存在 reranking 空间；
  3. pure model 的瓶颈在 WA 还是 geometry；
  4. 下一步应该集中优化哪一个模块。

失败但可接受：

* validation gate 没过，因此没有跑 official test；
* 但 final_report 清楚说明为什么不跑，避免了 misleading result。

请现在开始执行 opentry_9，严格遵守上述流程。不要等待我确认。若某个历史候选源不存在，记录缺失并继续执行可用部分；不要因为缺少某个源就停止整个实验。
