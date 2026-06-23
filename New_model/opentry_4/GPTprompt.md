通读最新的近 15 份实验报告、opentry_3 中的实验日志、E718 相关脚本与结果后，我需要你继续推进晶体生成实验。本轮只能在下面目录中写入：

/data/users/xsw/autodlmini/model/New_model/opentry_4

你可以读取、复制、参考其他目录中的文件，包括 opentry_2、opentry_3、历史实验报告、已有模型、已有数据、已有脚本、已有 checkpoint，但严禁改写、删除、覆盖 /data/users/xsw/autodlmini/model/New_model/opentry_4 之外的任何文件。所有新脚本、缓存、checkpoint、日志、报告、manifest、临时文件、评估结果都必须写入 opentry_4。

本轮任务的核心目标不是继续做大量局部 selector 调参，而是完成两件事：

1. 冻结并验证当前最优系统 E718，跑一次真正的 full MPTS-52 test。
2. 针对之前反复暴露的 rows>=7 几何生成瓶颈，训练或构造新的模型侧 geometry proposal / geometry energy / joint free-param+lattice generator，目标是创造 baseline 没有的新 StructureMatcher-positive candidates，尤其是 rows>=7 new positives，从根上提升 match@20。

请严格按下面流程执行。

---

# 0. 总原则

1. 禁止数据污染

   * 不允许使用 test target labels、test GT CIF、test StructureMatcher hit flag、test sample id 相关泄露信息来训练、调参、选阈值、选模型。
   * full test 只能在系统冻结后执行一次，不能根据 full test 结果反向调参。
   * 所有阈值、模型选择、selector 参数、generator 参数必须在 train-dev / val split 上确定。

2. 不要重复走已经证明无效的弯路
   除非是为了复现或做 sanity check，否则不要把主要时间浪费在下面方向：

   * 普通 full rerank compatibility selector；
   * 只调 source ID / source prior / source mode；
   * 单纯 free-param bank lookup；
   * 单行 row-level copy / template bundle copy；
   * source-free random prior sampler；
   * 直接 MSE lattice/coordinate regression；
   * post-render CIF coordinate surgery；
   * 只优化 RMSE 但不增加 StructureMatcher positives 的方法。

3. 本轮优先级
   优先级从高到低：

   * P0：冻结 E718 并跑 full test，得到当前可汇报结果。
   * P1：建立 rows>=7 hard-negative 诊断集，确认瓶颈。
   * P2：训练 geometry energy model，区分 W/A-hit positive geometry 和 W/A-hit match-fail hard negative。
   * P3：训练或构造 SG/Wyckoff 约束下的 joint free-param+lattice proposal generator。
   * P4：只在新 proposal pool 确实增加 new positives 后，再做 anchor-safe selective replacement / selector。

4. 本轮成功标准
   主指标仍然是 match@1、match@5、match@20，同时记录 RMSE。
   由于输入含有 GT-SG，比较对象必须是同样条件下含 GT-SG 的 CrystaLLM baseline，而不是原始不含 SG 的 CrystaLLM。
   最终目标是至少 2 个指标比 GT-SG CrystaLLM baseline 高 5 个点。
   但本轮更关键的过程 gate 是：

   * rows>=7 new positives beyond baseline 必须 > 0；
   * rows>=7 positive-any ceiling 必须高于 E423/E718 baseline；
   * W/A-hit but StructureMatcher-fail rate 必须下降；
   * match@20 必须优先优化；
   * selector AUC、RMSE 不能作为唯一成功依据。

---

# 1. 建立 opentry_4 工作区

在 /data/users/xsw/autodlmini/model/New_model/opentry_4 下创建以下结构：

* reports/
* scripts/
* configs/
* logs/
* checkpoints/
* cache/
* eval/
* manifests/
* tmp/

创建一个固定实验日志：

/data/users/xsw/autodlmini/model/New_model/opentry_4/reports/opentry_4_experiment_log.md

每做一个实验都必须追加写入该日志，格式如下：

## Exxxx: 实验标题

* 时间：
* 目标：
* 读取文件：
* 写入文件：
* 数据 split：
* 是否使用 test 信息：必须明确写 no
* 方法：
* 参数：
* 指标：

  * match@1：
  * match@5：
  * match@20：
  * match@50：
  * RMSE：
  * rows>=7 match@5：
  * rows>=7 match@20：
  * rows>=7 positive-any：
  * new positives beyond baseline：
  * W/A-hit but match-fail rate：
* 结论：
* 是否继续：
* 下一步：

同时创建 manifest：

/data/users/xsw/autodlmini/model/New_model/opentry_4/manifests/opentry_4_manifest.jsonl

记录每个实验产生的脚本、配置、checkpoint、结果文件、日志路径、git hash 或文件 hash，保证之后可以复现。

---

# 2. 第一阶段：复盘并冻结 E718

先读取 opentry_3 的实验日志和 E718 相关结果，确认以下信息：

* E423 val512 baseline 指标；
* E718 val512 指标；
* E718 使用的 compatibility model；
* E718 使用的 selective replacement 参数；
* E718 no-leakage audit；
* E718 依赖的 candidate pool；
* E718 的 sample-id alignment 是否正确；
* E718 是否使用了 test 信息。

根据之前报告，E718 大致配置应为：

* compatibility model：E425 GBDT compatibility model；
* replacement 策略：selective replacement；
* threshold = 0.0024707304479371964；
* anchor_count = 4；
* max_per_wa = 2；
* top_k = 50；
* 目标：保留原始 top-4 anchor，只允许高置信 compatibility candidates 做有限插入，避免普通 rerank 破坏 early rank。

请不要盲信这些参数，必须从 opentry_3 文件中重新核对，并把核对结果写入 opentry_4_experiment_log.md。

---

# 3. 第二阶段：冻结 E718 跑 full MPTS-52 test

在不使用 test label 调参的前提下，生成 E420/E421/E422-style 的 full MPTS-52 test candidate pool，并使用冻结的 E718 selective replacement 配置跑 full test。

要求：

1. full test 前必须写一个 freeze note：
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/E718_frozen_before_full_test.md

   内容包括：

   * 使用的模型；
   * 使用的参数；
   * 使用的候选池生成脚本；
   * 使用的评估脚本；
   * 使用的数据 split；
   * 是否使用 test label 调参：no；
   * 当前 frozen 配置；
   * frozen 时间。

2. full test 只能执行一次。

3. full test 结果写入：
   /data/users/xsw/autodlmini/model/New_model/opentry_4/eval/E718_full_test_result.json
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/E718_full_test_report.md

4. 报告必须包含：

   * match@1、match@5、match@20、match@50；
   * RMSE；
   * rows>=7 match@5、match@20；
   * 和 GT-SG CrystaLLM baseline 的差值；
   * 是否达到至少两个指标 +5pp；
   * 和 E423 / E718 val512 的差异；
   * 是否可能存在 val overfit；
   * 不能根据 full test 结果继续调参。

如果 full test 失败或结果低于预期，不允许在 test 上继续调。只能回到 train-dev / val 做新方法。

---

# 4. 第三阶段：构建 rows>=7 hard-negative 诊断集

本轮真正要解决的问题是：

模型经常已经生成了正确或接近正确的 W/A skeleton，但最终 StructureMatcher 失败，尤其 rows>=7 复杂结构失败严重。

请从 train / train-dev / val 中构造诊断集，不能用 test 泄露训练或调参。

至少构建以下集合：

1. Positive geometry candidates

   * StructureMatcher positive；
   * readable CIF；
   * composition correct；
   * SG/Wyckoff 合法。

2. Hard negative geometry candidates

   * W/A-hit 或 skeleton-hit；
   * readable CIF；
   * composition correct；
   * 但 StructureMatcher fail；
   * 优先 rows>=7。

3. Baseline-missing samples

   * E423/E718 在 top20/top50 中没有 positive；
   * 但 W/A recall 较高；
   * 这是下一阶段 generator 最该突破的样本。

输出文件：

* cache/hard_negative_dataset_train.jsonl
* cache/hard_negative_dataset_val.jsonl
* eval/hard_negative_diagnosis.json
* reports/hard_negative_diagnosis_report.md

诊断报告必须回答：

* rows>=7 中 W/A-hit but match-fail 的比例是多少？
* match-fail 主要来自 lattice、free-param、inter-row distance、volume/VPA、collision、site mapping 还是其他问题？
* baseline missing 的样本是否真的缺少候选，还是 selector 没排上去？
* 当前候选池 ceiling 是多少？
* 要提升 match@20，是应该优先生成新候选还是重排旧候选？

---

# 5. 第四阶段：训练 geometry energy model

训练一个 geometry energy / compatibility-hard-negative model，用来区分：

* 同一 target / 同一 W/A group 下的 StructureMatcher-positive geometry；
* W/A-hit but StructureMatcher-fail hard negative geometry。

注意：这不是普通 selector，不要只追求全局 AUC。它必须服务于 geometry proposal 和 anchor-safe replacement。

模型输入特征可以包括但不限于：

* formula / element embedding；
* SG；
* row_count；
* Wyckoff letters；
* site multiplicity；
* free-symbol pattern；
* source row mapping；
* lattice parameters；
* VPA；
* fractional free parameters；
* min pair distance；
* element-pair distance statistics；
* row-pair distance statistics；
* orbit-level distance windows；
* baseline rank / self-score；
* candidate source type；
* 是否 rows>=7。

禁止使用：

* target CIF 原始几何；
* test label；
* sample id 直接作为特征；
* material id 直接作为特征；
* candidate hit flag；
* 任何泄露 StructureMatcher label 的字段。

训练目标建议：

* group-wise ranking loss；
* pairwise positive-vs-hard-negative loss；
* focal / hard-negative weighted BCE；
* rows>=7 加权；
* 不要只做普通 pointwise BCE。

评估指标必须包括：

* group AUC / pairwise accuracy；
* rows>=7 hard-negative pairwise accuracy；
* positive candidate top-rank rate；
* 对 match@k 的真实影响；
* new positives beyond baseline 是否增加。

如果 energy model 只提高 AUC，但一合并 candidate pool 后 match@k 下降，则不能作为最终方法，只能作为 proposal guidance 或 anchor-safe insertion 的辅助分数。

---

# 6. 第五阶段：训练或构造 SG/Wyckoff 约束下的 joint free-param+lattice proposal generator

这是本轮最重要的新方向。

不要再独立复制单行 free-param，不要只查 bank，不要只调 source ID。要尝试联合生成：

输入：

* target formula；
* GT-SG；
* predicted or candidate W/A rows；
* source candidate context；
* source lattice；
* source free parameters；
* row mapping；
* row_count；
* element/multiplicity/orbit/free-symbol 信息。

输出：

* target-compatible lattice；
* target-compatible free parameters；
* optional row-level correction；
* 多个 diverse geometry candidates。

要求：

1. 必须在 SG/Wyckoff 约束下生成，不能生成明显非法结构。
2. 优先优化 rows>=7。
3. 训练只用 train / train-dev，不用 test。
4. 先小规模 smoke test，再扩大到 val512。
5. 每个 generator 先做 candidate ceiling audit，只有产生 new positives beyond baseline，才进入 selector / replacement 阶段。

可以尝试的方法包括：

A. Conditional residual generator
以 baseline/source geometry 为初始化，预测 lattice/free-param residual，而不是从零生成。

B. Mixture density / quantile generator
输出多个可能的 free-param+lattice modes，解决同一 W/A skeleton 对应多个 geometry basin 的问题。

C. Energy-guided local search
在 free-param+lattice 空间中搜索，而不是在 post-render CIF 坐标上做 surgery。
用 geometry energy model + physical constraints 作为目标。

D. Repel/physical constraint proposal
之前 repel-style 实验虽然排序不好，但少数情况下能产生 rows>=7 new positive。
保留这个方向，但必须改成 pre-render free-param/lattice constrained optimization。

E. Distillation from train-positive geometry
从 train positive candidates 中蒸馏 geometry basin，但必须避免简单 nearest/copy。
重点是学习 row coupling、lattice scale、inter-row distance 的联合分布。

每个方法必须记录：

* generated candidates 数量；
* readable rate；
* composition valid rate；
* SG/Wyckoff valid rate；
* W/A-hit rate；
* StructureMatcher positive rate；
* rows>=7 positive rate；
* new positives beyond E423/E718 baseline；
* match@1/5/20/50；
* RMSE；
* 失败样本类型。

---

# 7. 第六阶段：只在候选池 ceiling 提升后，再做 anchor-safe replacement

如果新 generator 在 val 上确实产生了 new positives beyond baseline，才允许进入 selector 阶段。

selector 不允许普通 full rerank，必须采用 E718 风格的风险控制策略：

* 保留原始 top anchor；
* 只允许高置信新候选插入；
* 限制每个 W/A group 的候选数量；
* 限制同质候选重复；
* 优先提升 match@20，同时不能明显伤害 match@1/match@5。

可以尝试：

* anchor_count in {3,4,5}；
* max_per_wa in {1,2,3}；
* threshold 从 train-dev 冻结；
* rows>=7-aware insertion；
* energy score + original self-score 混合；
* 只对 baseline-missing / low-confidence samples 做 replacement。

评估时必须区分：

* pool ceiling 提升；
* selector 排序提升；
* match@k 提升；
* rows>=7 提升；
* 是否只是 validation overfit。

---

# 8. 实验调度和资源使用

1. 先跑小规模：

   * val64 / train-dev64 smoke；
   * 确认脚本、数据对齐、无泄露、指标正常。

2. 再跑中规模：

   * aligned216；
   * val512。

3. 最后才跑大规模：

   * full validation；
   * frozen full test。

4. 尽量榨干 CPU/GPU：

   * CPU 用于 candidate rendering、StructureMatcher 并行评估、数据预处理；
   * GPU 用于模型训练、energy model、generator；
   * 使用多进程时必须写入 opentry_4/logs，并避免覆盖外部缓存；
   * 每个长任务都要保存中间结果，支持 resume；
   * 每次启动前检查已有文件，避免重复计算；
   * 大规模 StructureMatcher 评估必须分 shard，写 manifest。

---

# 9. 必须输出的最终交付物

本轮结束前，至少输出以下文件：

1. 总日志
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/opentry_4_experiment_log.md

2. manifest
   /data/users/xsw/autodlmini/model/New_model/opentry_4/manifests/opentry_4_manifest.jsonl

3. E718 frozen full test 报告
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/E718_full_test_report.md

4. hard-negative 诊断报告
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/hard_negative_diagnosis_report.md

5. geometry energy model 报告
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/geometry_energy_model_report.md

6. joint geometry generator 报告
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/joint_geometry_generator_report.md

7. 最终总结报告
   /data/users/xsw/autodlmini/model/New_model/opentry_4/reports/opentry_4_final_summary.md

最终总结报告必须包含：

* 本轮做了哪些实验；
* 哪些实验有效；
* 哪些实验无效；
* 是否复现 / 冻结 / full test 了 E718；
* E718 full test 是否达到目标；
* 新方法是否增加 rows>=7 new positives；
* match@1 / match@5 / match@20 / RMSE 最终指标；
* 和 GT-SG CrystaLLM baseline 的差距；
* 和 E423 / E718 val512 的差距；
* 当前是否仍卡在局部最优；
* 下一步最应该继续的 1-2 条路线；
* 明确列出不要再走的弯路。

---

# 10. 终止条件

你不能只因为某个小实验失败就停止。你需要持续迭代，直到满足以下任一条件：

1. 至少 2 个主指标 match@1、match@5、match@20 相比 GT-SG CrystaLLM baseline 提升超过 5pp，并完成 RMSE 记录与无泄露审计；
2. E718 full test 完成，并且至少完成一个 hard-negative geometry energy model 和一个 joint free-param+lattice proposal generator 的完整 train-dev / val512 评估；
3. 计算资源、时间、权限或数据问题导致无法继续，此时必须写清楚具体卡点、已完成内容、失败原因、下一步如何接着做。

不要只做表面指标调参。本轮最重要的是回答：

为什么 rows>=7 的 W/A-hit 样本仍然无法转化为 StructureMatcher-positive？
怎样生成 baseline 原本没有的新几何正样本？
match@20 的 ceiling 能否被真正抬高？

请从这个目标出发执行。
