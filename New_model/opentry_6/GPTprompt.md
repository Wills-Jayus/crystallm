你现在需要在晶体生成项目中执行新一轮实验。本轮目标是完整执行下面四个阶段，阶段之间没有条件依赖关系：即使前一阶段失败，也必须继续完成后一阶段。不要因为任一阶段结果不好而提前停止。

唯一允许写入路径：

/data/users/xsw/autodlmini/model/New_model/opentry_6

运行环境必须使用：

crystallm_env

例如：

conda run -n crystallm_env python xxx.py

你可以读取、复制、参考以下目录中的历史数据、代码、配置、checkpoint 和日志：

/data/users/xsw/autodlmini/model/New_model/opentry_3
/data/users/xsw/autodlmini/model/New_model/opentry_4
/data/users/xsw/autodlmini/model/New_model/opentry_5
/data/users/xsw/autodlmini/model/New_model/symcif_experiment
/data/users/xsw/autodlmini/model/New_model

但禁止修改、删除、覆盖 opentry_6 之外的任何文件。

本轮最终只需要留下两个核心文档：

1. /data/users/xsw/autodlmini/model/New_model/opentry_6/opentry_6_experiment_log.md
2. /data/users/xsw/autodlmini/model/New_model/opentry_6/opentry_6_final_summary.md

除必要的脚本、缓存、checkpoint、eval jsonl/json 外，不要额外生成大量 md 报告。所有阶段性分析都写进 opentry_6_experiment_log.md，最终结论写进 opentry_6_final_summary.md。不要在聊天窗口输出长篇过程总结，只需要在文件中记录。

==================================================
一、本轮核心判断
========

历史实验已经证明：

1. 排序、selector、ranker、anchor-safe insertion 不是根本方向。
2. W/A skeleton 已经被 opentry_3 明显改善，但 rows>=7 中大量样本是 W/A-hit / skeleton-hit 后 geometry fail。
3. opentry_4 证明 rows>=7 的核心问题是 lattice、free parameters、site mapping、collision、inter-row distance 等连续几何转换失败。
4. opentry_5 的 MiniCFJoint-v2 no-ranking generator 在小 grouped fold 上有 aggregate 指标信号，但 rows>=7 基本没有稳定 positive，而且陷入了小样本 fold_b 调参局部。
5. 下一步必须完整实验 CrystalFormer 式 continuous geometry 建模，而不是继续调 MiniCFJoint-v2 小参数。

本轮核心目标：

用 CrystalFormer-style autoregressive multimodal geometry generator + symmetry-space learned refiner，验证 rows>=7 的连续几何能否被真正学会。

==================================================
二、绝对禁止事项
========

本轮禁止所有排序类、选择类、打分类优化，包括但不限于：

* selector
* reranker
* ranker
* compatibility score
* geometry energy score
* GBDT / RF / LR 候选评分
* anchor-safe insertion
* selective replacement
* source prior / source score
* energy-based rejection
* rejection sampling
* beam rescoring
* top-k reranking
* oracle selection
* 根据 StructureMatcher label 选择候选
* 根据 RMSE、collision、energy、logprob、validity 分数筛选候选
* 生成很多候选后挑最好的

允许固定顺序生成：

candidate 0：确定性 decode
candidate 1-19：固定 seed 采样

候选顺序只能由 generation_index 和固定 seed 决定。无效候选必须保留原 slot 并计为失败，不允许删除后前移。

禁止再次运行 full test。禁止读取 test labels 或 test per-sample StructureMatcher 结果。val512 只允许在四阶段都完成后作为一次外层诊断使用；如果算力不足，可以不跑 val512，但必须完成四个阶段的 train/dev 评估。

==================================================
三、输出要求
======

只维护两个 md 文件：

1. opentry_6_experiment_log.md

每个阶段必须追加如下内容：

## Stage X / Exxxx: 名称

* 时间：
* 是否使用 crystallm_env：
* 读取文件：
* 写入文件：
* 是否写入 opentry_6 之外：必须 no
* 是否读取 test：必须 no
* 是否使用排序/筛选/打分：必须 no
* candidate 顺序：
* 数据范围：
* 模型结构：
* 训练目标：
* 关键参数：
* readable：
* composition exact：
* SG/Wyckoff legal：
* match@1：
* match@5：
* match@20：
* match@50：
* RMSE@1/5/20：
* rows>=7 match@1/5/20/50：
* rows>=7 positive-any：
* rows>=7 new positives：
* W/A-hit match-fail：
* skeleton-hit match-fail：
* collision-like rate：
* 结论：
* 失败原因：
* 下一步：

2. opentry_6_final_summary.md

最终只总结：

* 四个阶段是否全部完成；
* 每个阶段核心结果；
* 哪个阶段最有效；
* rows>=7 是否真正提升；
* 是否仍然卡在 continuous geometry；
* 与 opentry_4 / opentry_5 相比是否进步；
* 下一轮最应该继续的唯一方向；
* 明确不要再走的弯路。

不要再生成很多分散报告。

==================================================
四、必须完成的四个阶段
===========

注意：四个阶段没有条件依赖。无论前一个阶段是否失败，后一个阶段都必须执行。可以复用前一阶段中间产物，但不能因为前一阶段失败而跳过后一阶段。

---

## 阶段 1：GT-W/A 条件下的 CrystalFormer-style geometry-only model

目标：

先把 W/A 预测问题剥离，只测试连续几何是否能被学会。

输入：

* formula / composition
* GT-SG
* GT W/A skeleton
* GT species-to-orbit mapping

模型只预测：

* lattice independent parameters
* Wyckoff free parameters

不预测 W/A，不预测元素分配，不做 joint generation。

模型要求：

1. 使用 CrystalFormer-style autoregressive orbit decoder。
2. 每个 orbit 的 geometry 必须依赖：

   * formula
   * GT-SG
   * 当前 orbit 的 W/A、element、multipity、site symmetry、enumeration、active DOF mask
   * 所有前序 orbit 的 generated free parameters
   * 全局 lattice context
3. 周期 free parameters 不能用普通 MSE-only。
   必须至少实现一种周期多模态分布：

   * von Mises mixture；或
   * circular logistic mixture；或
   * sin/cos + mixture density。
4. lattice 使用 Gaussian mixture 或 scale/shape 分解，不能只做单点 MSE。
5. context length 不要限制为 20。至少支持 rows>=64，或者根据训练数据最大 rows 自动设定。
6. 只预测真正 free 的 coordinate DOF，fixed coordinates 不参与预测。
7. 生成后必须通过 Wyckoff manifold projection / render 还原 CIF。
8. 候选按照固定 generation_index 输出，不允许排序。

数据要求：

* 使用 opentry_5 已构建的 canonical data / grouped split 作为起点；
* 不要再只用 147 个 unique rows>=7；
* 必须尽量使用全部 train 中 rows>=7 unique samples；
* 如果全量太慢，至少构建一个大于 opentry_5 的复杂样本训练集，要求 unique rows>=7 >= 1000；如果达不到，必须在日志中写明原因。

评估要求：

至少在 fold_a 和 fold_b 上分别评估：

* all samples；
* rows>=7 samples；
* rows 7-9；
* rows 10-14；
* rows 15+。

必须报告：

* GT-W/A geometry-only match@1/5/20；
* rows>=7 match@1/5/20；
* collision-like rate；
* W/A-hit match-fail rate；
* 是否比 opentry_5 MiniCFJoint-v2 的 rows>=7≈0 有实质提升。

---

## 阶段 2：全量复杂样本数据训练，不允许重复少量复杂样本制造虚假数据量

目标：

验证使用全部 unique complex rows>=7 训练数据是否能改善复杂结构，而不是像 opentry_5 一样重复 147 个 unique complex samples。

任务：

1. 读取 opentry_5 grouped split / canonical data。
2. 统计全部 train/dev 中：

   * rows>=7 数量；
   * unique formula；
   * unique SG；
   * unique Wyckoff pattern；
   * rows 7-9 / 10-14 / 15+ 分布。
3. 构建 opentry_6 complex-focused training dataset：

   * 尽量包含全部 unique rows>=7；
   * 同时保留一定比例 rows<7 简单结构，避免模型退化；
   * 使用 balanced batch sampler，而不是简单复制少数样本。
4. 用阶段 1 的 geometry-only 模型结构，在这个 complex-focused dataset 上训练一版。
5. 固定生成顺序，在 fold_a / fold_b 上评估。

禁止：

* 只重复少量 rows>=7；
* 只在 fold_b 调参；
* 只报告最好 fold；
* 用 val512 调参；
* 用排序或筛选补救结果。

必须报告：

* unique rows>=7 train count；
* 是否比 opentry_5 的 147 unique rows>=7 明显扩大；
* fold_a / fold_b rows>=7 match@20；
* rows>=7 new positives；
* collision-like rate；
* 如果 rows>=7 仍为 0，分析到底是数据覆盖、模型结构还是 loss 的问题。

---

## 阶段 3：固定步数 symmetry-space learned geometry refiner

目标：

训练一个 pre-render parameter-space learned refiner，用来统一修正 free parameters 和 lattice，而不是 post-render coordinate surgery，也不是候选选择。

输入：

* formula
* GT-SG
* W/A skeleton
* initial free parameters
* initial lattice

输出：

* corrected free parameters
* corrected lattice

约束：

1. refiner 只在 symmetry parameter space 中工作。
2. 保持 formula、SG、W/A、multiplicity 不变。
3. 对每个 candidate 统一执行固定步数，例如 4 或 8 步。
4. 不允许判断“是否修正”。
5. 不允许根据修正后质量筛选 candidate。
6. 不允许按 refiner loss / energy / collision 分数排序。

训练数据：

至少包含：

* synthetic corruption from clean canonical geometry；
* opentry_4 / opentry_5 hard negative 中 W/A-hit but match-fail 的 train-only 样本；
* collision / short contact corruption；
* lattice / VPA corruption；
* free-param / site mapping corruption；
* inter-row distance corruption。

训练目标至少包含部分：

* periodic coordinate loss；
* lattice metric / VPA loss；
* pair-distance loss；
* element-pair distance consistency；
* collision / short-contact penalty；
* row-pair distance consistency。

评估方式：

把 refiner 接到阶段 1 / 阶段 2 的输出后面：

raw generation
→ fixed-step refiner
→ render CIF
→ fixed-order evaluation

必须报告 raw vs refined：

* match@1/5/20；
* rows>=7 match@1/5/20；
* collision-like rate；
* W/A-hit match-fail；
* RMSE；
* 是否因为 refiner 使简单结构变坏。

即使阶段 1/2 模型效果差，也必须训练和评估 refiner，可以用 GT-W/A + corrupted GT geometry 作为诊断输入完成该阶段。

---

## 阶段 4：接入 predicted / OOF / exact-cover W/A 条件，完成完整系统

目标：

在前面 geometry-only 之外，验证真实推理条件下的 W/A condition gap。

必须分别跑三种输入条件：

A. GT W/A + geometry model
B. OOF / predicted W/A + geometry model
C. exact-cover / opentry_3-style W/A + geometry model，如果相关文件可用

如果 C 缺少现成文件，可以从 opentry_3/opentry_5 中寻找可复用的 W/A 预测或 exact-cover 结果；如果确实找不到，必须在日志中写清楚并用 OOF W/A 替代，但不能跳过阶段 4。

任务：

1. 统一同一批 fold_a / fold_b 样本；
2. 分别使用 GT W/A、OOF W/A、exact-cover W/A；
3. 使用同一个阶段 1/2 geometry generator；
4. 可接同一个阶段 3 fixed-step refiner；
5. 固定 generation_index 和 seeds；
6. 不排序，不筛选，不 replacement。

必须回答：

* GT W/A 成功但 OOF 失败，是否说明 W/A condition gap 是主瓶颈？
* GT W/A 也失败，是否说明 continuous geometry 仍是主瓶颈？
* exact-cover W/A 是否比 OOF 更适合作为 geometry 条件？
* rows>=7 的失败主要来自 skeleton 错，还是 geometry 错？

必须报告：

* A/B/C 三种条件的 all-sample match@1/5/20；
* A/B/C 三种条件的 rows>=7 match@1/5/20；
* A/B/C 的 W/A exact 或 skeleton exact；
* A/B/C 的 W/A-hit match-fail；
* A/B/C 的 collision-like rate；
* A/B/C 的差异解释。

==================================================
五、最小实验规模要求
==========

不要再只做 32/64 个 eval 样本的 smoke 作为结论。

最低要求：

1. fold_a 和 fold_b 都必须跑。
2. 每个 fold 的 eval 样本尽量 >= 256。
3. 每个 fold 的 rows>=7 eval 样本尽量 >= 100。
4. 如果算力不足，可以降低 candidate 数为 K=10，但必须记录；默认 K=20。
5. 如果数据量或脚本限制导致无法达到上述规模，必须写明具体原因，并尽量使用最大可行规模。

==================================================
六、确定性和复现要求
==========

opentry_5 中出现过同 checkpoint eval-only 指标不一致的问题，因此本轮必须做确定性检查。

至少选择一个阶段的一个 checkpoint，连续生成两次：

* generations 文件 hash 必须一致；
* metrics 必须一致；
* 如果不一致，必须定位原因：

  * seed 未固定；
  * dataloader 顺序；
  * torch deterministic；
  * StructureMatcher 并行；
  * cache；
  * invalid candidate 处理；
  * 其他。

如果无法完全一致，也必须在实验日志中写明原因和影响，不能假装指标可靠。

==================================================
七、环境和资源要求
=========

运行前检查并写入日志：

* conda env 是否为 crystallm_env；
* python 版本；
* torch 版本；
* cuda 是否可用；
* nvidia-smi；
* CPU 核数；
* 内存；
* 磁盘空间。

所有脚本必须能 resume。长任务必须保存 checkpoint。StructureMatcher 评估需要分 shard 或支持中断恢复。

==================================================
八、终止条件
======

你不能因为某个阶段失败就停止。必须完成四个阶段：

1. GT-W/A CrystalFormer-style geometry-only model；
2. 全量 complex rows>=7 数据训练；
3. fixed-step symmetry-space learned geometry refiner；
4. GT / OOF / exact-cover W/A 条件对比完整系统。

四个阶段都完成后，写最终总结文件：

/data/users/xsw/autodlmini/model/New_model/opentry_6/opentry_6_final_summary.md

如果因为环境、权限、数据缺失或算力问题无法完成某一阶段，也不能直接停止。你必须：

1. 写明阻塞原因；
2. 尝试至少一个降级替代方案；
3. 完成该阶段的最小可行版本；
4. 继续执行后续阶段。

最终总结必须明确给出：

* 四个阶段是否都完成；
* 哪个阶段真正带来 rows>=7 改善；
* 是否证明 GT-W/A 下 geometry 可以学会；
* predicted/OOF W/A 是否是主要瓶颈；
* fixed-step refiner 是否有用；
* 是否仍然陷入局部；
* 下一轮唯一最值得继续的路线。

不要输出长篇解释，把过程和结果写进 opentry_6_experiment_log.md 和 opentry_6_final_summary.md。
