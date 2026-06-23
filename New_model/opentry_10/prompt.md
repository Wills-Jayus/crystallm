你现在负责执行一个长程、全量、可恢复的晶体结构生成研究任务：

/data/users/xsw/autodlmini/model/New_model/opentry_10

这不是审计任务，也不是整理历史报告的任务。你的职责是实际生成候选、训练模型、训练排序器、运行完整 validation，并在验证指标充分时运行冻结后的 official full test。

不得因为缺少某个历史 artifact、checkpoint、validation candidate bank 或依赖而提前结束。缺失 artifact 是需要构建的任务，不是停止理由。

============================================================
一、唯一目标
============================================================

至少产出以下两类系统中的一个：

A. strategy/fusion system
B. pure structural model

并使至少一个冻结系统在 CrystaLLM Table 3 official full-test K20 协议下真正超过 CrystaLLM-a GT-SG。

当前 test anchor：

MP-20 CrystaLLM-a GT-SG：
- match@1 = 71.67
- match@5 = 83.08
- match@20 = 87.81
- RMSE@1/5/20 = 0.0509 / 0.0449 / 0.0431
- rows>=7 match@1/5/20 = 62.37 / 76.35 / 82.61

MPTS-52 CrystaLLM-a GT-SG：
- match@1 = 25.23
- match@5 = 36.46
- match@20 = 43.96
- RMSE@1/5/20 = 0.1211 / 0.1257 / 0.1334
- rows>=7 match@1/5/20 = 22.49 / 33.37 / 41.04

最低正式成功标准：

- 至少一个冻结系统在 official test 上至少一个 match 指标超过相应 CrystaLLM-a GT-SG anchor >= 1.0 percentage point；
- 同一系统 match@20 不得下降超过 0.2pp；
- rows>=7 对应指标不得明显下降；
- 增益不能来自补齐缺失 sample；
- 不得使用 test target、test StructureMatcher label 或 test oracle selection。

强成功标准：

- 至少两个 match 指标超过 anchor；
- 或 MP-20 和 MPTS-52 各至少一个指标超过；
- rows>=7 同时提升；
- bootstrap 95% CI 下界大于 0。

+5pp 是 stretch goal，但不能为了追求 +5pp 使用 test tuning。

只有满足正式成功标准，才能在 final_report.md 中写“成功超过”。

============================================================
二、禁止提前结束
============================================================

以下情况均不允许作为 final completion：

- validation CrystaLLM GT-SG candidates 不存在；
- K50/K100 candidates 不存在；
- 历史 checkpoint 路径不清楚；
- 某个 Python package 不存在；
- 某个训练 checkpoint 没有通过第一轮 gate；
- 当前候选池 oracle union 不足；
- 某次训练 OOM；
- 某个数据转换脚本失败；
- 当前 pure model 没有超过 baseline；
- 只能得到诊断结果。

遇到上述情况时必须自动执行对应修复：

- 搜索 checkpoint/config/manifest；
- 从已有 checkpoint 重新生成；
- 必要时重新训练；
- 安装依赖到 opentry_10 专属环境；
- 减小 batch size、使用 gradient accumulation；
- 分 shard 生成并支持断点续跑；
- 切换模型、候选源、排序器或训练目标；
- 转入另一条路线继续。

只有以下两种情况允许结束：

1. 至少一个正式系统完成 official full test；
2. 发生无法修复的硬件或存储故障，并且已经实际尝试过恢复。

如果发生第二种情况，只能写 blocked_report.md，不得写 final_report.md，也不得把它称为 diagnostic success。

============================================================
三、数据、评估与泄漏规则
============================================================

使用官方 MP-20 和 MPTS-52 train/validation/test split。

必须从官方 CSV 实际读取并验证 sample 数量，不得依赖手写数量。

统一评估：

- final K20；
- StructureMatcher ltol=0.3；
- stol=0.5；
- angle_tol=10；
- normalized get_rms_dist(...)[0]；
- missing/failed candidate 计失败；
- validity 只作诊断；
- rows>=7 由 target CIF 原始 _atom_site rows 计算。

工程 sanity check 可以使用极小样本，但只能验证脚本能运行，不能用于模型选择、超参数选择或任何结论。

所有正式决策必须基于：

- 完整 official train；
- 完整 official validation；
- 完整 candidate budget；
- 完整 rows>=7 分桶。

official test 在配置冻结前不得读取 per-sample target 或 match label。

历史 aggregate test 指标可以作为已知 baseline，但不得据此训练或修改阈值。

============================================================
四、工作目录与长程执行控制器
============================================================

只写入：

/data/users/xsw/autodlmini/model/New_model/opentry_10

历史目录只读：

- opentry_2 到 opentry_9
- symcif_experiment
- 其他历史实验目录

创建：

opentry_10/
  configs/
  scripts/
  state/
  logs/
  cache/
  vendor/
  candidates/
  generations/
  features/
  labels/
  checkpoints/
  frozen_strategy/
  frozen_pure_model/
  eval/
  metrics/
  reports/
  experiment_log.md

必须实现一个可恢复控制器：

scripts/run_opentry10.py --resume

控制器需要维护：

state/controller_state.json
state/jobs.jsonl
state/artifact_registry.json
state/frozen_registry.json

每个阶段必须：

- 有唯一 stage id；
- 有输入 checksum；
- 有输出 checksum；
- 支持 resume；
- 已完成阶段不得重新初始化；
- 失败任务记录 stderr 并自动重试；
- OOM 自动降低 batch；
- 大规模生成按 shard 保存；
- shard 完成后校验 sample coverage；
- 不得静默丢样本。

不要只启动 nohup/tmux/background job 就结束会话。必须监控任务完成，检查输出完整性，然后继续下一阶段。

日志只保留一个 run id，不得像 opentry_9 一样重复写多次 start。

============================================================
五、Phase 0：资源与历史资产追溯
============================================================

先检查：

- GPU 型号、数量和显存；
- CPU 核数；
- 剩余磁盘；
- crystallm_env；
- 本地 CrystaLLM repo；
- opentry_7 GT-SG test generation 的 checkpoint；
- tokenizer/meta.pkl；
- generation config；
- prompt construction；
- seeds；
- temperatures；
- top-k；
- candidate order；
- official train/val/test CSV。

输出：

reports/resource_audit.md
reports/baseline_provenance.md

重点任务：

追溯 opentry_7 CrystaLLM-a GT-SG test candidates 是由哪个 checkpoint、哪个脚本、哪个 tokenizer、哪些参数生成的。

按以下顺序搜索：

1. opentry_7 configs/manifests/logs；
2. opentry_7 checkpoints；
3. 上层 New_model 目录；
4. CrystaLLM 官方模型目录；
5. 本地下载缓存；
6. 若仍缺失，使用官方下载工具获取 benchmark model；
7. 若无法下载，则在 official train split 上重新训练同规格 CrystaLLM-a。

不得在“checkpoint 未找到”处停止。

用少量 prompt 只进行 deterministic reproduction sanity check，确认生成格式、prompt 和 candidate order；随后立即执行完整 validation generation。

============================================================
六、Phase 1：必须生成完整 CrystaLLM validation anchor
============================================================

这是 mandatory stage，不得跳过。

对 MP-20 和 MPTS-52 完整 validation：

1. 构建 composition + GT-SG prompts；
2. 使用与 opentry_7 test anchor 相同的 checkpoint；
3. 使用相同 tokenizer 和 preprocessing；
4. 保留候选生成 log probability；
5. 第一组 20 candidates 严格复现 anchor generation policy；
6. 额外生成扩展 candidate pool。

输出至少：

- MP-20 validation K100；
- MPTS-52 validation K100。

如果初始 K100 oracle coverage 不足，后续扩展到 K200。

K100 初始采样应包含：

- greedy candidate；
- baseline 固定 seeds；
- 多个 sampling seeds；
- temperatures 至少覆盖 0.7、0.85、1.0、1.15；
- top-k 至少覆盖 5、10、20；
- 每种配置记录 generation provenance；
- 对每个候选保存 normalized token logprob。

不要改变 anchor K20 原始顺序；扩展候选放在 K20 之后。

完整评估：

- K1/K5/K20/K50/K100；
- per-rank cumulative match；
- valid/readable/formula/SG；
- rows>=7；
- candidate duplication；
- canonical WA/skeleton diversity。

输出：

candidates/crystallm_gt_sg_mp20_val_k100.jsonl
candidates/crystallm_gt_sg_mpts52_val_k100.jsonl
metrics/crystallm_gt_sg_mp20_val.json
metrics/crystallm_gt_sg_mpts52_val.json
reports/crystallm_validation_anchor_report.md

在这些文件完整生成前，不允许进入 final reporting。

============================================================
七、Phase 2A：rerank-only，必须执行，不受 union gate 限制
============================================================

先只重排原始 CrystaLLM anchor K20。

这是最安全的 strategy：

- candidate set 不变；
- match@20 必须完全不变；
- 只尝试提高 match@1 和 match@5。

使用完整 validation candidate labels 训练 ranker。

不能把同一 validation sample 同时用于训练和评价。使用 deterministic 5-fold group cross-validation：

- group = sample_id；
- 每个 sample 的全部 candidates 必须位于同一 fold；
- 所有正式 validation 指标必须来自 out-of-fold prediction；
- 最终 test ranker 在超参数冻结后使用全部 validation labels 重新训练。

至少尝试以下模型：

1. regularized logistic regression；
2. HistGradientBoosting / RandomForest；
3. XGBoost/LightGBM/CatBoost ranker，哪个可用就使用哪个；
4. pairwise ranker；
5. shallow MLP 或 graph-based scorer，若简单模型效果饱和。

至少尝试 3 个随机种子。

候选特征至少包含：

- source rank；
- greedy/sample 标记；
- temperature；
- top-k；
- normalized token logprob；
- token length；
- token entropy；
- readable；
- formula closure；
- atom count consistency；
- SG consistency；
- row_count；
- atom count；
- canonical WA key；
- canonical skeleton key；
- train WA frequency；
- train skeleton frequency；
- volume/atom；
- density proxy；
- shortest pair distance；
- very-short-bond count；
- lattice anisotropy；
- element count；
- stoichiometry pattern；
- SG frequency；
- candidate duplicate cluster size；
- candidate WA cluster size；
- multiple samplers/checkpoints consensus；
- candidate-to-group-medoid distance；
- optional pretrained energy score。

StructureMatcher candidate label只可用于 train/validation supervision，不得作为 test feature。

评估：

- OOF match@1/5/20；
- rows>=7；
- RMSE；
- bootstrap CI；
- 各 fold delta；
- MP-20 和 MPTS-52 分开模型与联合模型均需尝试。

输出：

reports/rerank_model_search.md
metrics/rerank_oof_results.json
checkpoints/rerank_models/
features/

如果 rerank-only 在完整 OOF validation 上产生稳定 match@1 增益，则冻结至少一个 rerank-only candidate。

不得因为 oracle union@20 不足而跳过 reranking。

============================================================
八、Phase 2B：扩大真实候选覆盖
============================================================

计算完整 validation 上：

- anchor K20 hit；
- expanded CrystaLLM K100/K200 hit；
- stablekey hit；
- SymCIF hit；
- exact-cover candidate hit；
- geometry-refined candidate hit；
- 每个 source 对 anchor 的 exclusive rescue；
- source overlap；
- pairwise union；
- all-source oracle union；
- rows>=7 exclusive rescue。

如果 expanded K100 oracle match@20 相对 anchor K20：

- 提升 >= 2pp：进入 selector 训练；
- 提升 < 2pp：不得停止，继续扩展候选来源。

扩展顺序：

1. 同一 CrystaLLM checkpoint 更多 seeds/temperatures；
2. 相邻或其他 validation-loss 良好的 CrystaLLM checkpoints；
3. 至少 3 个不同训练 seed 的 CrystaLLM ensemble；
4. historical SymCIF/stablekey candidates；
5. 新 exact-cover WA candidates；
6. train-only prototype/retrieval candidates；
7. pure structural model candidates；
8. symmetry-preserving geometry variants。

所有 CrystaLLM ensemble model 使用相同 official benchmark train split。

不得使用额外数据库训练主公平比较模型，除非单独标记为 extra-data exploratory line。

如果没有可用的多个 checkpoint：

- 从 official train 分别训练至少 3 个不同 seed；
- checkpoint selection 使用 full validation generation metrics，不只使用 token loss。

输出：

reports/full_oracle_union_report.md
metrics/full_oracle_union.json

============================================================
九、Phase 2C：coverage selector 与 conservative fusion
============================================================

从 K100/K200 pool 选择 final K20。

同时训练并保留三类策略：

S1：rerank-only
- 只重排 anchor K20；
- match@20 与 anchor 完全一致。

S2：conservative fusion
- 保留至少 14–16 个 anchor candidates；
- 只用 supplemental candidates 替换重复或低价值尾部 slots。

S3：residual fusion
- 根据推理时可见特征识别 anchor 高风险 sample；
- 对高风险 sample 分配更多 supplemental slots；
- 普通 sample 保守保留 anchor。

去重层次：

- exact CIF duplicate；
- canonical WA duplicate；
- canonical skeleton duplicate；
- candidate-candidate StructureMatcher duplicate；
- near-geometry duplicate。

candidate-candidate StructureMatcher 不使用 target，可以用于推理。

selector 目标不是单候选分数排序，而是最大化集合边际覆盖：

marginal_score =
  predicted_match_probability
  + source_consensus_bonus
  + new_WA_bonus
  + new_skeleton_bonus
  + residual_rescue_bonus
  - duplicate_penalty
  - geometry_risk_penalty

使用完整 5-fold OOF validation 选择：

- 保留 anchor slots 数量；
- diversity penalty；
- source quota；
- residual routing threshold；
- score calibration。

不得直接在全 validation 上训练并评价同一 selector。

输出：

reports/selector_search.md
metrics/selector_oof_results.json

============================================================
十、Phase 2D：如果简单 scorer 饱和，训练条件结构判别器
============================================================

如果出现：

- oracle union 高；
- 但 selector 无法把新增正确候选选入 final K20；
- 或 rerank top1 提升不足；

则训练一个 candidate correctness scorer。

输入：

- composition；
- GT-SG；
- candidate crystal graph / structured representation；
- candidate generation features。

输出：

- candidate 与未知目标结构匹配的概率；
- 可选 RMSE quality score。

训练标签来自 official train/validation candidate bank 的 StructureMatcher。

可以使用：

- crystal graph encoder；
- candidate CIF sequence encoder；
- pretrained structure embedding；
- handcrafted feature + neural hybrid。

必须 group cross-validation。

不能输入 target structure。

============================================================
十一、Phase 3：pure structural route
============================================================

如果 strategy 在完整 validation 上仍未达到稳定提升，必须继续 pure structural route；不得以“当前 model 未通过 gate”结束。

pure model 推理输入严格为：

composition + GT-SG

不得读取：

- CrystaLLM candidates；
- stablekey candidates；
- retrieval candidates；
- test label。

允许使用：

- official train split；
- validation 进行模型选择；
- benchmark-train-only pretraining；
- teacher distillation，但必须明确标记。

模型必须显式实现：

composition + SG
→ conventional-cell multiplier Z
→ row_count
→ Wyckoff skeleton
→ element-WA exact cover
→ lattice/free parameters
→ CIF rendering

优先顺序：

1. 搜索本地是否已有 CrystalFormer、WyFormer 或类似实现；
2. 若网络可用，可将官方代码 clone 到 opentry_10/vendor，并记录 commit；
3. 只复用架构和代码，主公平模型在 official benchmark train 上重新训练；
4. 若外部代码无法运行，则基于现有 SymCIF/MiniCFJoint 实现 structural decoder。

不能继续使用普通 byte-level CIF GPT 作为 pure structural 主模型。

------------------------------------------------------------
Phase 3A：全量 canonicalization
------------------------------------------------------------

对完整 official train/val/test target 做结构化转换，保存：

- composition/formula_counts；
- SG；
- Z；
- row_count；
- Wyckoff orbit；
- multiplicity；
- element assignment；
- lattice；
- free parameters；
- canonical WA/skeleton keys；
- renderer audit。

conversion failure 不得静默丢弃。

------------------------------------------------------------
Phase 3B：WA generator
------------------------------------------------------------

实现 permutation-aware 或 permutation-invariant decoder。

强制 exact-cover：

- 每个元素剩余计数实时更新；
- 不可能 multiplicity action 被 mask；
- EOS 只在全部计数归零时允许；
- beam search 保持不同 canonical WA；
- 不允许同一 WA 因 row permutation 重复占据 top-k。

完整 validation 评估：

- WA_hit@1/5/20/50/100/200；
- rows>=7；
- seen/unseen skeleton；
- SG frequency buckets；
- raw coverage 与 selected coverage。

当前历史 raw top100 WA_hit=87.85% 不足以对 MP-20 match@20 形成可靠超越 margin。

因此 MP-20 WA generator 目标：

- raw WA_hit@100 >= 93%；
- 优先争取 >=95%；
- selected WA_hit@20 >=90%。

如果未达到：

- 增大 beam；
- 改 exact-cover search；
- 增加 row_count/Z auxiliary heads；
- 使用 set loss；
- 使用 train-frequency prior；
- ensemble 多 seed；
- 继续训练。

不得只写诊断然后停止。

------------------------------------------------------------
Phase 3C：GT-WA geometry renderer
------------------------------------------------------------

使用 GT-WA 输入，重新训练 geometry model，不得只复用历史报告。

模型：

- 按 crystal system 输出独立 lattice parameters；
- log volume/atom；
- lattice shape；
- periodic free-coordinate loss；
- minimum over symmetry-equivalent coordinates；
- pair-distance auxiliary loss；
- short-bond penalty；
- volume/density loss；
- rows>=7 oversampling；
- low-symmetry oversampling；
- multiple geometry modes。

完整 validation 生成至少 K20 geometry variants。

评估：

- GT-WA match@1/5/20；
- RMSE；
- rows>=7；
- 按自由参数数目分桶；
- 按晶格各向异性分桶。

MP-20 目标：

- GT-WA geometry match@20 >= 93%；
- 优先争取 >=95%；
- rows>=7 明显高于历史 69.46%@5。

如果 geometry 仍弱：

- 增加 mixture density head；
- symmetry-preserving refinement；
- geometry-quality scorer；
- constrained ML-potential relaxation；
- lattice/free-param ensemble。

------------------------------------------------------------
Phase 3D：pure end-to-end K20
------------------------------------------------------------

K20 分配：

- 先保证不同 WA/skeleton；
- 高概率 WA 分配多个 geometry variants；
- candidate 0 使用 joint score；
- 不使用 token greedy；
- 同一 WA 的 geometry variants 不得占满全部 slots。

joint score 可包含：

- WA log probability；
- geometry likelihood；
- physical validity；
- predicted energy；
- geometry confidence；
- train-only prior。

checkpoint selection 必须基于完整 validation generation：

- match@1/5/20；
- WA_hit；
- rows>=7；
- RMSE；
- render coverage；
- candidate diversity。

如果 pure model standalone 仍低于 anchor，它仍可作为 strategy/fusion 的 residual candidate source，但不能标记为 pure success。

============================================================
十二、自动决策循环
============================================================

实现自动循环，不允许单次 gate failure 结束：

while no_frozen_line_passes_validation:
    if validation anchor missing:
        generate/retrain anchor
    elif rerank-only has exploitable K20 gap:
        improve ranker
    elif expanded pool oracle gain is insufficient:
        generate more diverse candidates or train ensemble
    elif oracle union is high but selected K20 is weak:
        improve selector/candidate scorer
    elif WA raw coverage is weak:
        improve WA generator/exact-cover search
    elif GT-WA geometry is weak:
        improve geometry renderer/refinement
    else:
        train a new complementary model seed/architecture

至少完成：

- 一次完整 MP-20 validation K100；
- 一次完整 MPTS-52 validation K100；
- rerank-only 全量 OOF 搜索；
- 至少 4 类 ranker；
- 至少 3 个 ranker seeds；
- conservative fusion 全量 OOF；
- residual fusion 全量 OOF；
- oracle union 全量审计；
- 如果 K100 union 不足，完整 K200 扩展；
- 如果 strategy 未通过，至少一次 full-train pure structural training；
- 至少三个 major full-validation iteration cycles。

没有完成这些 mandatory work，不得生成 final_report.md。

============================================================
十三、冻结与 official test
============================================================

在读取新的 official test per-sample evaluation 前，必须冻结并登记：

- primary strategy；
- secondary rerank-only strategy；
- secondary conservative fusion；
- optional pure model。

写入：

state/frozen_registry.json
frozen_strategy/*
frozen_pure_model/*
reports/pre_test_freeze_declaration.md

登记内容：

- exact config；
- model checksum；
- candidate source；
- source quota；
- ranker checksum；
- selector threshold；
- seeds；
- temperature；
- K budget；
- validation OOF metrics；
- primary/secondary designation。

测试时：

1. 对所有 official test samples 生成完整 candidate pool；
2. 在不读取 target 的情况下构建 final K20；
3. 校验每个 sample 恰好 20 slots；
4. 再统一评估；
5. 不得在看到 test 结果后修改配置重跑同一方法。

可以一次性评估多个事先登记的 frozen lines，但必须明确 primary line，不能事后把大量尝试中的最好结果冒充唯一预注册结果。

============================================================
十四、最终报告
============================================================

只有 official full-test 完成后才能写：

final_report.md

必须包含：

- 实际训练时长和资源；
- 完整候选生成数量；
- full validation anchor；
- OOF reranking 结果；
- oracle union；
- selector；
- pure WA coverage；
- GT-WA geometry；
- end-to-end pure；
- official full-test；
- rows>=7；
- RMSE；
- bootstrap CI；
- 与 CrystaLLM-a GT-SG 的逐指标 delta；
- 增益来源；
- 是否使用额外训练数据；
- 是否存在 leakage；
- 哪个结果可以用于论文。

如果没有达到 official exceed，必须如实报告，但在达到 mandatory execution 工作量以前不得结束。

============================================================
十五、立即执行
============================================================

现在开始 opentry_10。

第一项实际任务不是写报告，而是：

1. 追溯 opentry_7 CrystaLLM-a GT-SG checkpoint；
2. 构建完整 MP-20 和 MPTS-52 validation GT-SG prompts；
3. 生成完整 validation K100 anchor；
4. 运行完整 evaluator；
5. 训练 rerank-only OOF ranker。

不要等待确认，不要只输出计划，不要因为历史 validation bank 缺失而停止。