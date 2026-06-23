在以下目录完成一次可用于论文正式结果的全量实验：

/data/users/xsw/autodlmini/model/New_model/opentry_7

只能写入 opentry_7，运行环境必须使用 crystallm_env。可以读取其他历史目录，但不能修改 opentry_7 之外的任何文件。

本轮目标：在 CrystaLLM Table 3 完全相同的 MP-20 和 MPTS-52 数据、split、test sample list、生成预算和 evaluator 下，完成两条实验线：

1. strategy / fusion line：只追求最终指标最好。允许使用 SymCIF、CrystaLLM beam、opentry 历史模型、candidate fusion、selector、ranker、reranker、energy model、anchor insertion、hybrid 等任何非泄露策略。
2. pure model line：输入严格只能是 composition + GT-SG，不能使用 GT W/A、GT geometry、CrystaLLM candidates、SymCIF candidates、外部候选融合、selector/ranker/reranker，也不能用任何额外测试信息。模型必须自己端到端生成 CIF。

无论哪条线，都禁止 test 泄露、oracle selection、用 test label 调参、根据 test StructureMatcher 结果反向修改模型或策略。所有模型选择、阈值、ranker、selector、fusion 参数只能在 train/validation 上确定。final test 配置冻结后只能执行一次。

==================================================
一、统一数据和 evaluator
=================

必须先复现并冻结 CrystaLLM 官方 Table 3 协议：

* MP-20 使用 CrystaLLM 论文相同官方 split。
* MPTS-52 使用 CrystaLLM 论文相同官方 split。
* 使用官方 StructureMatcher 参数：ltol=0.3，stol=0.5 Å，angle_tol=10°。
* 使用 CrystaLLM 论文相同 normalized RMSE 定义。
* 每个 test composition 生成 20 个候选。
* 不得删除生成失败样本，失败候选保留并计为失败。
* 输出 match@1、match@5、match@20、RMSE@1、RMSE@5、RMSE@20。
* 额外输出 rows>=7 的 match@1、match@5、match@20 和 positive-any。

先建立统一评估脚本，然后对下面所有系统使用同一 evaluator。

==================================================
二、必须评估的 baseline
================

至少完成以下 baseline：

A. published CrystaLLM-a composition-only 数字记录
只作为论文 reference，不需要重新训练。

B. reproduced CrystaLLM-a composition-only
使用官方 small/a 模型、官方 prompt、官方采样参数复现 Table 3，用来确认 evaluator 和数据一致。

C. CrystaLLM-a GT-SG
与 B 完全相同，只把输入从 composition-only 改成 composition + GT-SG。
这是后续两条线最重要的对比 baseline。

CrystaLLM 使用论文设置：

* top-k=10
* temperature=1.0
* 每个样本 20 generations

==================================================
三、实验线 1：strategy / fusion line
==============================

目标：不限制是否排序或融合，只追求最终 match 指标最高。

可以使用：

* CrystaLLM GT-SG beam candidates；
* SymCIF candidates；
* opentry、opentry_2、opentry_3、opentry_4、opentry_5、opentry_6 中所有合法模型或候选；
* candidate fusion；
* selector；
* ranker；
* reranker；
* energy model；
* anchor-safe insertion；
* hybrid strategy；
* validation-trained filtering / scoring；
* K50/K100 candidate pool 再压缩到 final top20。

要求：

1. 所有策略参数只能用 train/validation 确定，不能用 test labels。
2. 最终 test 只输出 top20，并按最终策略顺序计算 match@1/5/20。
3. 如果使用超过 20 个原始候选池，例如 K50/K100，需要在报告中明确标注为 strategy-large-pool，不能和 strict K20 结果混淆。
4. 必须分别在 MP-20 和 MPTS-52 full test 上完成。
5. 最终报告中要明确说明该线不是独立模型胜出，而是 strategy / fusion 胜出。

优先复核历史最强策略：

* opentry E19/E24 hybrid；
* opentry_2 E35/E41/E47 ranker；
* opentry_3/E718-style selective replacement；
* opentry_4 energy / pairfield / anchor-safe；
* 其他能在 validation 上带来最高 match@1/5/20 的组合。

最终选 validation 最强 strategy，冻结后跑 full test。

==================================================
四、实验线 2：pure model line
=======================

目标：训练和评估严格输入 composition + GT-SG 的独立模型。

严格禁止：

* GT W/A；
* GT site mapping；
* GT geometry；
* CrystaLLM candidates；
* SymCIF candidates；
* candidate fusion；
* selector；
* ranker；
* reranker；
* oracle；
* test label；
* test-time energy selection；
* 任何外部候选作为输入。

允许：

* 使用 train split 中的真实 CIF 训练；
* 使用 validation 选择 checkpoint；
* 使用 opentry_5 / opentry_6 的模型代码和数据处理思路；
* 使用 CrystalFormer-style autoregressive geometry；
* 使用 WyFormer-style site symmetry / enumeration 表示；
* 使用 learned refiner，但 refiner 必须对每个生成候选固定执行，不能选择性使用。

候选顺序必须固定：

candidate 0：deterministic decode
candidate 1-19：fixed seeds sampling

不得根据 logprob、RMSE、validity、energy、collision 或任何质量分数重排。

优先复用并扩展当前最好的纯模型方案：

* opentry_5 E8034/E8036 MiniCFJoint-v2 no-ranking generator；
* opentry_6 CrystalFormer-style geometry-only / refiner 代码；
* 结合 site symmetry、enumeration、autoregressive free-param+lattice generation；
* 但最终模型输入只能是 composition + GT-SG。

必须在官方 MP-20 和 MPTS-52 train split 上重新训练，不允许只用 smoke subset。
必须在官方 full test 上完整评估，不能用子集结果代替。

==================================================
五、主表输出
======

最终报告必须给出两个数据集的完整表格：

MP-20：

* published CrystaLLM-a composition-only
* reproduced CrystaLLM-a composition-only
* CrystaLLM-a GT-SG
* best strategy/fusion line
* best pure model line

MPTS-52：

* published CrystaLLM-a composition-only
* reproduced CrystaLLM-a composition-only
* CrystaLLM-a GT-SG
* best strategy/fusion line
* best pure model line

每一行报告：

* match@1
* match@5
* match@20
* RMSE@1
* RMSE@5
* RMSE@20
* rows>=7 match@1
* rows>=7 match@5
* rows>=7 match@20
* rows>=7 positive-any
* generation budget
* 是否使用 candidate fusion / ranking
* 是否为 pure model

必须计算：

* strategy/fusion line 相对 CrystaLLM-a GT-SG 的差值；
* pure model line 相对 CrystaLLM-a GT-SG 的差值；
* pure model line 是否至少两个指标超过 GT-SG CrystaLLM 5pp；
* strategy/fusion line 是否达到当前项目历史最好。

==================================================
六、最终文件
======

至少保存：

* opentry_7_experiment_log.md
* opentry_7_final_report.md
* configs/
* checkpoints/
* eval/
* generations/
* metrics/
* frozen_strategy/
* frozen_pure_model/

其中：

* generations/ 保存每个系统、每个数据集、每个 test sample 的 20 个候选，保持最终顺序。
* metrics/ 保存 machine-readable json/jsonl 指标。
* frozen_strategy/ 保存最终 strategy/fusion 配置。
* frozen_pure_model/ 保存最终 pure model checkpoint 和 inference config。

==================================================
七、执行要求
======

不要只做 smoke。不要只跑 MPTS-52。不要只跑 validation。
MP-20 和 MPTS-52 的 full train / validation / test 实验都必须完成。

执行顺序：

1. 建立统一 evaluator。
2. 复现 CrystaLLM-a composition-only。
3. 跑 CrystaLLM-a GT-SG baseline。
4. 做 strategy/fusion line，validation 选最强配置，冻结后 full test。
5. 做 pure model line，validation 选 checkpoint，冻结后 full test。
6. 输出最终论文表格和结论。

如果算力或环境阻塞，必须写入 opentry_7_experiment_log.md，并尽最大可能完成降级版本；但最终目标仍然是完整全量实验。

从现在开始执行，所有过程写入 opentry_7_experiment_log.md，最终结果写入 opentry_7_final_report.md。
