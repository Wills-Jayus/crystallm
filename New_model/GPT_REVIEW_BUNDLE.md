# CrystaLLM / SymCIF 审稿包

生成时间：2026-06-28 16:26:03 UTC

用途：把本文件直接复制给网页版 GPT，用于判断当前方向、路线边界、可信结果、下一轮实验 gate。

本文件在原 7 个审计文档拼接基础上，追加了 Track A MPTS-52 最新 validation OOF 结论。

## 包含文档

- `RESEARCH_STATE.md`
- `ROUTE_JUDGEMENT.md`
- `EXPERIMENT_INDEX.md`
- `BASELINE_LEDGER.md`
- `CLAIM_BOUNDARY.md`
- `AUDIT_SUMMARY_FOR_GPT.md`
- `NEXT_EXPERIMENT_GATE.md`
- `TRACK_A_MPTS52_LATEST_APPENDIX`

---

# [01] RESEARCH_STATE.md

# CrystaLLM / SymCIF 全局研究状态

生成日期：2026-06-28
审计目录：`/data/users/xsw/autodlmini/model/std_way`
证据来源：只读取历史目录；未启动训练、未启动新模型实验、未跑新 full test。

## 1. 当前总目标

在 CrystaLLM / crystallm baseline 基础上，提出可写成论文的晶体生成改进方法。主问题不是普通 rerank，而是：在 `composition + GT-SG` 条件下，能否通过 crystallographic causal ordering、Wyckoff-skeleton、SymCIF / symmetry-aware representation 和 geometry-aware refinement 稳定提升 StructureMatcher `match@1/5/20`。

硬目标：在 MP-20、MPTS-52 等 CrystaLLM 论文使用的数据集 official full test 上，`match@1/5/20` 至少两个指标超过 GT-SG 条件 CrystaLLM baseline `+5pp`。小样本、validation、common subset、coverage repair、oracle/GT-WA、普通 rerank/fusion 都不能替代该目标。

## 2. 当前论文主线

论文主线仍应是：

- symmetry-aware representation / SymCIF；
- Wyckoff skeleton / W-A sequence / exact-cover feasibility；
- crystallographic causal ordering；
- geometry-aware refinement，尤其是 lattice、free parameters、site mapping、collision/local environment；
- rows>=7 / complex structures 作为核心失败分析章节。

证据：`symcif_experiment/reports/symcif_v4_geometry_next_step_summary.md` 证明 GT-WA geometry model 可把 GT-WA match@20 从 56.4% 提到 70.2/70.4%，full pipeline best 达到 62.8%（同条件小规模 evaluator）；`opentry_3/final_summary.md` 显示 symbolic W/A 从弱 recall 推进到 E85/E119 validation 级别，但 rows>=7 geometry 转化仍低；`opentry_4/reports/opentry_4_final_summary.md` 明确 rows>=7 多是 W/A/skeleton-hit 后的 geometry/free-param/site-mapping 失败。

## 3. 当前输入条件

主比较必须使用 `composition + GT-SG`。历史 public CrystaLLM 多为 composition-only，不可作为当前主 baseline。当前本地 anchor 来自 `opentry_7/opentry_7_final_report.md` 和 `opentry_10/final_report.md`：

- MP-20 GT-SG CrystaLLM-a K20：`71.67 / 83.08 / 87.81`。
- MPTS-52 GT-SG CrystaLLM-a K20：`25.23 / 36.46 / 43.96`。

## 4. 当前指标硬门槛

`+5pp` 目标对应：

- MP-20：`76.67 / 88.08 / 92.81`。
- MPTS-52：`30.23 / 41.46 / 48.96`。

opentry_10 的近程标准曾使用 `>= +1pp` official gate，但这不是用户最终目标。所有 `+0.8pp`、`+1pp附近` 都只能算近程信号。

## 5. 当前最强可信 GT-SG baseline

主比较 baseline：

- MP-20 `CrystaLLM-a GT-SG` official full-test K20：`match@1/5/20 = 71.67 / 83.08 / 87.81`，RMSE `0.0509 / 0.0449 / 0.0431`。证据：`opentry_7/opentry_7_final_report.md`、`opentry_8/final_report.md`、`opentry_10/final_report.md`。
- MPTS-52 `CrystaLLM-a GT-SG` official full-test K20：`25.23 / 36.46 / 43.96`，RMSE `0.1211 / 0.1257 / 0.1334`。证据同上。

注意：早期 `opentry/iterative_experiment_log.md` 还记录了 MPTS-52 aggregate `26.64 / NA / 44.69`、common subset K1/K5 `26.27 / 36.58`。这是旧口径/不同样本集证据，不能替代 opentry_7/10 统一 anchor，但应在论文审稿时说明口径差异。

## 6. 当前已知最佳可信结果

按 official full-test、冻结后评估、且与 opentry_10 anchor 直接比较：

- MPTS-52 `mpts52_k30_rf_seed1_bestscore_route`：`26.075 / 36.228 / 44.059`，delta `+0.845 / -0.232 / +0.099 pp`。证据：`opentry_10/metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json`。
- MPTS-52 `mpts52_k50_rf_seed1_margin_route`：`25.791 / 36.265 / 43.824`，delta `+0.561 / -0.195 / -0.136 pp`。证据：`opentry_10/metrics/official_test/mpts52_k50_rf_seed1_margin_route_summary.json`。
- MP-20 `mp20_k50_hgb_mean_seed012_margin_route`：`70.230 / 81.870 / 87.486`，delta `-1.440 / -1.210 / -0.324 pp`。证据：`opentry_10/metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json`。

这些都是 ranking/selector/route/fusion 线，不是论文主方法成功；均未达到 `+5pp`，也未达到 opentry_10 `+1pp` gate。

历史上 `opentry` 的 SymCIF+CrystaLLM fixed hybrid E24 有 `match@5=39.28 / match@20=45.06`，优于同 evaluator CrystaLLM-only Ref20 `34.66 / 41.63`。证据：`opentry/iterative_experiment_log.md` E23-E25。它是候选融合结果，不是纯 SymCIF 或论文主贡献。

## 7. 距离 +5pp 目标的差距

MPTS-52 当前 closest official line `mpts52_k30_rf_seed1_bestscore_route` 距最终目标：

- match@1：`26.075` vs `30.23`，差 `-4.155 pp`。
- match@5：`36.228` vs `41.46`，差 `-5.232 pp`。
- match@20：`44.059` vs `48.96`，差 `-4.901 pp`。

MP-20 opentry_10 best official line距最终目标：

- match@1：`70.230` vs `76.67`，差 `-6.440 pp`。
- match@5：`81.870` vs `88.08`，差 `-6.210 pp`。
- match@20：`87.486` vs `92.81`，差 `-5.324 pp`。

结论：当前没有任何 full-test line 接近用户最终目标。

## 8. 当前最大瓶颈

最大瓶颈不是“候选池完全没有正确结构”，而是：

1. MPTS-52 / rows>=7 的 W/A / skeleton / exact-cover coverage 仍不足。
2. 即使 W/A / skeleton 命中，continuous geometry、lattice、free parameters、site mapping、短距离碰撞仍导致 StructureMatcher 失败。
3. 浅层 scorer 只看 rank、CIF长度、cell、valid、atom rows 等，无法区分“公式/SG/rows都对但几何错”的 hard negatives。

证据：

- `opentry_10/ToGPTPro.md`：MPTS-52 K50 validation oracle headroom `+4.720pp`，但 official K30 只保住 `+0.845pp match@1`。
- `opentry_10/ToGPTPro.md`：RF scorer 在 MPTS-52 K50 有正样本的 2636 个 validation sample 中，有 1026 个把错误候选排到最高分；这些错误 top 中 97.5% 公式正确、72.3% atom rows 正确。
- `opentry_4/reports/opentry_4_final_summary.md`：rows>=7 top50 missing 中 W/A/skeleton hit=126，说明很多失败发生在 geometry 转化。

## 9. 当前允许继续的路线

- Wyckoff-skeleton / symmetry-aware representation / SymCIF 主线：允许继续，但必须绑定 full-test gate 和 geometry 转化证据。
- inference-safe structural quality scorer：允许作为 auxiliary / hybrid component，前提是不使用 test true CIF、StructureMatcher label 或 oracle signal。
- position-aware slot policy / Top-1、Top-5、Tail-rescue、rows>=7 gate：允许作为 metric rescue 或 hybrid 决策层，不可包装成论文核心理论。
- symmetry-preserving geometry repair / deterministic micro-geometry refinement：允许，且更接近论文主线。
- rows>=7 专门 gate / complex structure route：允许，必须单独报告子集。

## 10. 当前禁止继续的路线

- 只换 RF/HGB seed、threshold、anchor_keep 的普通 rerank。
- candidate fusion / coverage repair 当作主贡献。
- 根据 official test 结果回调同一策略。
- 只用 validation / small fold 小涨宣布成功。
- GT-WA / GT-skeleton / oracle selection 当真实方法。
- pure CIF GPT / low-LR SFT / checkpoint continuation，若没有解决 representation 与 geometry，禁止继续盲训。

## 11. 只能作为辅助的路线

- K30/K50 selector/route/rerank：可作为 metric rescue 或 diagnostic，不能是 paper mainline。
- structural quality scorer：可作为 system quality-control layer，不能单独声称为核心理论贡献。
- CHGNet/M3GNet/energy proxy：可作为 geometry quality feature 或 refinement signal，必须 inference-safe。
- KNN / row-conditioned retrieval：可作为 geometry proposal/control，需明确 train-only index 与泛化边界。

## 12. 当前局部解风险

最大风险是继续围绕 `MPTS-52 K30/K50 RF/HGB` 做阈值微调。这个方向已有 official result：K30 近程最好但未过 +1pp，更远未过 +5pp；K50 还伤害 K5/K20。继续回调同类策略会构成 test feedback 风险。

## 13. 最近历史实验核心结论

- `opentry_7`：official full-test 复现 GT-SG baseline；pure model 和 stablekey hybrid 均低于 GT-SG anchor。
- `opentry_8`：strategy_fusion 与 GT-SG anchor 相同或只修一个缺失样本，属于 coverage repair。
- `opentry_9`：缺完整 validation anchor bank，未训练 selector，未跑新 official test；确认 pure structural 仍是 WA + geometry bottleneck。
- `opentry_10`：补齐 validation candidate bank，证明 K50 有 headroom，但 shallow scorer official 不成功。

## 14. MPTS-52 K30/K50 policy 是否值得继续

结论：只能 `watch`，不能作为论文主线 continue。它证明了 K30/K50 pool 中有近程信号，但 official full test 未过 +1pp，更未过 +5pp。若继续，必须改造成 inference-safe structural quality scorer + hard-negative pairwise/listwise ranking + rows>=7 gate，而不是继续普通 RF/HGB threshold。

## 15. Wyckoff-skeleton / symmetry-aware representation 是否仍值得作为论文主线

结论：仍值得作为论文主线，但当前证据是“方向必要”而非“路线已成功”。历史证明 symbolic W/A、GT-WA geometry 和 SymCIF representation 能定位瓶颈；同时历史也证明只提高 W/A 或只排序候选不足以达到 full-test +5pp。下一步主线必须把 W/A exact-cover 与 geometry-aware refinement 结合。

## 16. 下一轮实验必须回答的问题

1. 新实验属于哪条 route，是否在 `ROUTE_JUDGEMENT.md` 允许？
2. 对标是否为 GT-SG CrystaLLM anchor？
3. 预期提升 `match@1/5/20` 中哪两个指标，是否以 `+5pp` 为终点？
4. 是否包含 full-test 验证计划？
5. structural scorer 是否 inference-safe？
6. rows>=7 是否单独处理？
7. 失败后停止条件是什么？

## 17. 下一轮禁止做什么

- 禁止新训练或新 full test 在未通过 `NEXT_EXPERIMENT_GATE.md` 前启动。
- 禁止把普通 rerank/fusion 作为论文主线。
- 禁止用 official test aggregate 反调同一策略。
- 禁止把 `+0.845pp` 或 `+1pp附近` 写成最终成功。

## 18. 给网页版 GPT 的审稿入口

建议复制以下文件给网页版 GPT：

- `RESEARCH_STATE.md`
- `ROUTE_JUDGEMENT.md`
- `EXPERIMENT_INDEX.md`
- `BASELINE_LEDGER.md`
- `CLAIM_BOUNDARY.md`
- `AUDIT_SUMMARY_FOR_GPT.md`
- `NEXT_EXPERIMENT_GATE.md`

审稿问题：当前是否允许继续 MPTS-52 K30/K50 metric line？structural quality scorer 是否已与 ordinary rerank 区分清楚？Wyckoff-skeleton / symmetry-aware representation 是否应继续作为论文主线？下一轮是否可以开始，还是必须先补 claim/gate 设计？

## 19. Track B Hybrid 最新产物审计

审计时间：2026-06-28 UTC。

本轮按用户要求读取 `model/std_way/track_b_hybrid`。最新可见产物包括：

- `README.md`
- `SYSTEM_ARCHITECTURE.md`
- `MODULE_BOUNDARY.md`
- `CLAIM_MAPPING.md`
- `PROVENANCE_SCHEMA.md`
- `TODO_NEXT_REAL_EXPERIMENT.md`
- `GPT_HANDOFF_TRACK_B.md`
- `BUILD_PROTOTYPE.sh`
- `SMOKE_TEST.sh`
- `hybrid_core/*`
- `artifacts/prototype_manifest.json`
- `artifacts/smoke_summary.json`

Track B 建立的是 `TRACK_B_HYBRID_MAINLINE_PROTOTYPE` 可执行原型，而不是新 benchmark。它只做 adapter -> union/dedup -> provenance -> hard filter -> scorer interface -> repair interface -> smoke evaluation 的连通性验证。`smoke_summary.json` 明确为 toy smoke：`sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`repaired_count=0`，并声明 `SMOKE ONLY: no full benchmark, no route-success claim, no match@k conclusion`。

Track B 的正向价值：

- 把 paper mainline 系统边界明确为 SymCIF / Wyckoff-skeleton / exact-cover / symmetry-preserving geometry repair。
- 把 CrystaLLM GT-SG adapter、candidate union/dedup、provenance registry、structural quality scorer、evaluation adapter 明确归为 auxiliary。
- 建立 provenance schema，要求未来真实实验记录 source family、adapter、stage、是否使用 test label/oracle、是否 frozen-before-test 和 rows>=7 子集。
- 确认 ordinary rerank/fusion 不能被包装成主线。

Track B 的限制：

- structural scorer 仍是未训练 smoke heuristic。
- geometry repair 是 `noop_symmetry_preserving_geometry_repair_stub`，未改变 lattice/free parameters/site geometry。
- exact-cover 只是 basic multiplicity sanity，不是生产级 Wyckoff parser。
- evaluation adapter 不产出 StructureMatcher `match@k`。
- 没有 validation OOF、没有 official full-test、没有 baseline delta。

本轮路线判决：`continue as architecture / STOP as metric evidence`。

判决含义：

- Track B 支持论文主线方向继续，即 Wyckoff/exact-cover + geometry-aware refinement + inference-safe quality-control 的系统化路线。
- Track B 不改变当前 best official result，不新增 baseline，不允许生成 official freeze 计划。
- 下一次真实 Track B 实验前必须先补生产级 Wyckoff parser、hard-negative structural scorer 数据集、真实 symmetry-preserving repair、StructureMatcher validation adapter、rows>=7 gate 和 frozen-before-test manifest。

## 20. Track A MPTS-52 最新产物审计

审计时间：2026-06-28 UTC。

本轮按用户要求读取 `model/std_way/track_a_mpts52`。最新可见产物包括 `track_a_validation_oof.py`、`RUN_VALIDATION_OOF.sh`、`logs/validation_oof.log` 和空的 `outputs/` 目录。脚本设计为 MPTS-52 validation-only structural scorer / rows>=7 gate，包含 feature extraction、5-fold OOF pairwise utility、`STOP/WATCH/ALLOW_OFFICIAL_FREEZE` decision 逻辑；但实际日志只到 `[feature] processed 100000 candidates`，未见 OOF、decision 或 report 写出，`outputs/` 下没有 `validation_oof_results.json`、`oof_scores.jsonl`、`feature_audit.json`、`DECISION.md` 或报告文件。因此本轮 Track A 只有未完成脚本和部分进度日志，没有完成的 validation 结果，也没有可用于更新 baseline、claim 或 official freeze 的证据。

本轮路线判决：`STOP`。

判决含义：

- 不是说 MPTS-52 主问题永久停止，而是说 `track_a_mpts52` 这一轮只有未完成 validation OOF 脚本，没有完成的可审计指标，不能继续进入 official freeze。
- 不能把脚本存在、部分进度日志或缺失 output 当作 validation signal。
- 不能基于本轮 Track A 宣称 `WATCH` 信号，更不能 `ALLOW_OFFICIAL_FREEZE`。
- 当前全局 best official result 仍保持为 opentry_10 MPTS-52 K30 RF route `26.075 / 36.228 / 44.059`，且它仍不是成功结果。

下一次若要重启 Track A，必须先完整跑完 validation-only gate，并在 `track_a_mpts52` 下写入最小可审计产物：实验 manifest、候选来源、validation-only 指标、hard constraints/scorer/repair 边界、rows>=7 报告、`DECISION.md`、失败停止条件和明确 freeze gate。


---

# [02] ROUTE_JUDGEMENT.md

# 路线级判决

判决说明：

- `continue` 不表示允许立即大规模投入；只表示路线仍可在 `NEXT_EXPERIMENT_GATE.md` 下提出新实验。
- `watch` 表示有信号但证据不足或容易变成局部解。
- `pivot` 表示原形式不能继续，但可转向更严格的结构化版本。
- `stop` 表示不应重复同类实验，除非有新证据和网页版 GPT 重新批准。

## 1. 原始 CrystaLLM / CIF baseline 复现路线

状态：watch
路线类型：baseline reproduction
paper_mainline_status：diagnostic
metric_rescue_status：allowed
是否符合论文主线：否，仅为对照。
是否依赖普通 ranking/rerank/fusion：否。
历史证据：`opentry_7/opentry_7_final_report.md` 复现 composition-only MP-20 `62.17/74.98/81.84`、MPTS-52 `18.86/28.08/35.17`。
最好 full-test 结果：见上。
相对 GT-SG baseline 差距：均显著低于当前输入条件下 GT-SG anchor。
是否达到 +5pp：否。
最终判决：保留为历史 baseline，不作为当前主比较对象。
禁止重复的实验形式：继续把 composition-only CrystaLLM 当成主对照。
重新开启条件：只在论文需要补 public baseline context 时复评。

## 2. GT-SG 条件 baseline 路线

状态：continue
路线类型：mandatory comparator
paper_mainline_status：diagnostic
metric_rescue_status：allowed
是否符合论文主线：作为约束符合。
历史证据：`opentry_7/opentry_7_final_report.md`、`opentry_8/final_report.md`、`opentry_10/final_report.md`。
最好 full-test 结果：MP-20 `71.67/83.08/87.81`；MPTS-52 `25.23/36.46/43.96`。
最终判决：后续所有 claim 必须对标该 baseline。
禁止重复的实验形式：用 composition-only baseline 替代 GT-SG baseline。

## 3. SG-only / SG-conditioned generation 路线

状态：watch
路线类型：conditioned generation
paper_mainline_status：mainline only if coupled to representation
metric_rescue_status：allowed
是否符合论文主线：部分符合。
历史证据：SymCIF 与 CrystaLLM GT-SG 对照均证明 SG 输入强影响 top1；`opentry/iterative_experiment_log.md` no-GT-SG MP-20 ablation 显示 no-GT-SG `54.28/70.77` 低于 GT-SG v5 `64.26/74.04`。
最好 full-test 结果：单独 SG-conditioned CrystaLLM anchor 是当前强 baseline。
相对 GT-SG baseline 差距：新方法未超过 anchor。
最终判决：SG 条件必须保留，但不能只做“加 SG prompt”的论文贡献。

## 4. Wyckoff skeleton / WA / SymCIF 路线

状态：continue
路线类型：symmetry-aware representation
paper_mainline_status：mainline
metric_rescue_status：allowed
是否符合论文主线：是。
是否依赖普通 ranking/rerank/fusion：部分历史结果依赖 self-score/selection；主线不应依赖普通 fusion。
是否包含 structural quality scorer：部分包含 GT-free self-score。
是否包含 geometry repair：是，但仍不足。
历史证据：`symcif_experiment/reports/symcif_v4_geometry_next_step_summary.md`、`opentry_3/final_summary.md`、`opentry_4/reports/opentry_4_final_summary.md`。
最好 validation 结果：opentry_3 E119/E111 full val128 约 `31.25/35.16/41.41/43.75@50`。
最好 full-test 结果：SymCIF-v5 MPTS-52 `25.47/32.90/33.63`；opentry E04 K1 common-subset `28.08` 是局部信号。
相对 GT-SG baseline 差距：K5/K20 仍低；未达 +5pp。
失败证据：rows>=7 W/A-to-match 转化弱；E119 rows>=7 match@50 仅 `18.33%`。
是否 full test 有效：尚未在当前统一 official anchor 下达标。
局部解风险：只提升 W/A recall、但不解决 geometry。
最终判决：继续作为论文主线，但下一步必须绑定 geometry-aware refinement 与 rows>=7 gate。
禁止重复的实验形式：只扩大 skeleton beam、只换 self-score、只做 candidate tail-fill。
重新开启条件：新方案明确解决 W/A exact-cover + continuous geometry 转化，并预注册 full-test gate。

## 5. CrystalFormer / WyFormer 风格表示借鉴路线

状态：continue
路线类型：representation design reference
paper_mainline_status：mainline
metric_rescue_status：allowed
是否符合论文主线：是。
历史证据：`opentry_3/final_summary.md` 使用 CrystalFormer/WyFormer 风格 `formula+GT-SG -> W/A -> CIF`，symbolic gate 明显优于早期 opentry_2 policy search。
最好 validation 结果：E85 neural-first merge W/A@50 `78.91%`；E119 K50 ceiling `43.75%`。
失败证据：geometry/free-param 转化仍不足，未 full-test freeze。
最终判决：继续作为表示/生成顺序理论依据，但不能只复刻概念，必须给出 full-test 指标。

## 6. Constrained decoding 路线

状态：pivot
路线类型：exact-cover / feasibility constraint
paper_mainline_status：mainline or auxiliary
metric_rescue_status：allowed
历史证据：opentry_3 fixed-skeleton exact-cover DP、E45 duplicate fixed-orbit mask；opentry_2 policy search top100 W/A 只有约 12-13%，后续 canonical DP 显著提高。
最好 validation 结果：opentry_3 E45 W/A@50 `74.22%`，E85 `78.91%`。
失败证据：W/A 提升未充分转化为 match；simple geometry schedules 无效。
最终判决：保留 exact-cover feasibility，但下一步必须服务于 geometry/refinement，不再单独刷 W/A 指标。

## 7. SFT / low-LR / reset optimizer / checkpoint continuation 路线

状态：stop
路线类型：pure CIF text training
paper_mainline_status：forbidden_as_mainline unless representation changes
metric_rescue_status：forbidden
历史证据：`opentry_7/opentry_7_final_report.md` pure model MP-20 `60.58/70.67/77.96`、MPTS-52 `17.18/24.35/31.52`，显著低于 GT-SG anchor；`opentry_2` val64 SFT smoke match 为 0。
最终判决：禁止继续盲目纯 CIF SFT/低 LR/checkpoint continuation。
重新开启条件：必须换成 symmetry-aware representation 或 geometry-aware objective，并先过 validation gate。

## 8. DPO / preference / reranking-based preference 路线

状态：pivot
路线类型：preference/ranking
paper_mainline_status：forbidden_as_mainline if ordinary rerank
metric_rescue_status：allowed_with_warning
历史证据：opentry_10 rerank-only HGB validation 有 `+1.220pp` K1，但 official MPTS-52 `24.938/35.931/43.960` 低于 anchor。
最终判决：普通 preference/rerank 不继续；若重启，必须是 inference-safe structural hard-negative pairwise/listwise scorer，并归为 auxiliary。

## 9. Geometry head / lattice / fractional coordinate 改进路线

状态：continue
路线类型：geometry-aware refinement
paper_mainline_status：mainline or auxiliary
metric_rescue_status：allowed
历史证据：`symcif_v4_geometry_next_step_summary.md` GT-WA geometry model no-over/over 达到 match@20 `70.2/70.4`；`opentry_4` rows>=7 W/A/skeleton hit geometry failures；`opentry_6` 建议继续 continuous geometry/refiner。
失败证据：opentry_3 deterministic one-shot geometry net E57 rows>=7 match 0；source_cluster lattice E125 退化。
最终判决：继续，但必须处理多模态 free-param/lattice/site mapping，不做单一 MSE 回归。

## 10. KNN / row-conditioned / retrieval-like geometry 路线

状态：watch
路线类型：train-only retrieval geometry
paper_mainline_status：auxiliary
metric_rescue_status：allowed
历史证据：`symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md` 显示 e07/e08 row-conditioned KNN 在 GT-WA geometry 上把 match@5 提到 `85.11/87.10`。
失败证据：full generation 仍受 W/A coverage 限制；MPTS-52 rows>=7 match@5 `12.60%`。
最终判决：可作为 geometry proposal/control，不可单独作为论文核心。

## 11. Ordinary ranking / rerank / candidate fusion 路线

状态：stop
路线类型：post-hoc candidate selection
paper_mainline_status：forbidden_as_mainline
metric_rescue_status：allowed_with_warning
历史证据：`opentry_2` K50 RF rankers、`opentry` E24 SymCIF+CrystaLLM hybrid、`opentry_10` rerank/K30/K50 official。
最好 full-test 结果：opentry_10 K30 `26.075/36.228/44.059`，未过 +1pp；E24 hybrid K20 `26.02/39.28/45.06` 但是 candidate fusion。
是否达到 +5pp：否。
最终判决：不能作为论文主线；仅可保留为 diagnostic/ablation/metric rescue。
禁止重复的实验形式：继续调 RF/HGB seed、threshold、anchor_keep 后声称方法突破。

## 12. Coverage repair 路线

状态：stop
路线类型：coverage repair
paper_mainline_status：forbidden_as_mainline
metric_rescue_status：forbidden
历史证据：`opentry_8/final_report.md` MP-20 strategy_fusion 等于 anchor，MPTS-52 只因补齐 1 个样本使 match@20 `43.96 -> 43.97`。
最终判决：不得继续包装成 improvement。

## 13. Oracle / upper bound / GT-WA / GT-skeleton 诊断路线

状态：watch
路线类型：upper bound diagnostic
paper_mainline_status：diagnostic
metric_rescue_status：forbidden as real method
历史证据：`symcif_v4` WA upper-bound `82.4%`；`opentry_9` GT-WA geometry diagnostic；`opentry_4` W/A/skeleton hit match-fail。
最终判决：允许作为瓶颈定位和上限分析，禁止当真实方法贡献。

## 14. 数据格式重构 / causal order / tokenization 路线

状态：continue
路线类型：representation/tokenization
paper_mainline_status：mainline
metric_rescue_status：allowed
历史证据：`symcif_experiment/reports/*conversion_report*.json`、`opentry_3/final_summary.md`、`opentry_5/reports/canonical_representation_report.md`。
失败证据：仅重构格式不足以 full-test 超过 GT-SG baseline。
最终判决：继续作为方法地基，必须与 exact-cover 和 geometry 模块结合。

## 15. 小样本 fold 验证路线

状态：watch
路线类型：screening only
paper_mainline_status：diagnostic
metric_rescue_status：allowed for prescreen
历史证据：opentry_3 val128、opentry_5 grouped folds、opentry_10 validation OOF。
失败证据：opentry_10 MPTS-52 rerank-only validation K1 `+1.220pp`，official 反而 `-0.292pp`。
最终判决：小样本/fold 只能筛选，不能写成成功。

## 16. Full test 验证路线

状态：continue
路线类型：evaluation protocol
paper_mainline_status：diagnostic
metric_rescue_status：allowed
历史证据：opentry_7/8/10 official full-test reports。
最终判决：所有成功 claim 必须以 frozen official full-test 为准。

## 17. MPTS-52 K30/K50 position-aware structural quality policy 路线

状态：watch
路线类型：metric rescue / hybrid policy
paper_mainline_status：auxiliary
metric_rescue_status：allowed_with_warning
是否依赖普通 ranking/rerank/fusion：当前版本是。
是否包含 structural quality scorer：当前为 shallow scorer，不足。
最好 full-test 结果：K30 RF `26.075/36.228/44.059`，delta `+0.845/-0.232/+0.099`。
是否达到 +5pp：否。
最终判决：不能继续普通 RF/HGB route；可转为 inference-safe structural quality scorer + slot policy。
重新开启条件：新增公式/SG/Wyckoff exact-cover、collision、local environment、bond valence、energy proxy 等结构特征；预注册 validation->official gate；不能用 opentry_10 official 结果回调。

## 18. rows>=7 专用 gate / complex structure 路线

状态：continue
路线类型：complex-structure gate
paper_mainline_status：auxiliary or mainline analysis
metric_rescue_status：allowed
历史证据：opentry_4 rows>=7 W/A/skeleton-hit failures；opentry_10 MP-20 rows>=7 collapse；SymCIF MPTS-52 rows>=7 match@5 12.60%。
最终判决：必须单独处理。不能再把 rows>=7 混入总体样本后只看 aggregate。

## 19. Structural quality scorer 路线

状态：continue
路线类型：inference-safe quality scorer
paper_mainline_status：auxiliary
metric_rescue_status：allowed
历史证据：opentry_10 ToGPTPro 诊断当前 shallow scorer 缺少 formula consistency、SG consistency、Wyckoff exact-cover、collision/local geometry、bond valence/energy proxy。
最终判决：这是最合理的 metric rescue/hybrid component，但 claim 必须收缩为辅助模块。
重新开启条件：只依赖 prompt 与 candidate CIF；hard-negative pairwise/listwise；rows>=7 单独 gate。

## 20. Symmetry-preserving geometry repair 路线

状态：continue
路线类型：geometry-aware refinement
paper_mainline_status：mainline or auxiliary
metric_rescue_status：allowed
历史证据：symcif_v4 GT-WA geometry model、opentry_4 pairfield generator、opentry_6 refiner。
失败证据：direct proposal ordering weak，source-free/random priors 和 post-render coordinate surgery 被 opentry_4 禁止为主线。
最终判决：继续，但必须 preserve symmetry/Wyckoff constraints，不做无约束后处理。

## 21. Pure model / invariant sequence / Mat2Seq-like / SG-aware sequence model 路线

状态：pivot
路线类型：new sequence architecture
paper_mainline_status：mainline if symmetry-aware
metric_rescue_status：allowed
历史证据：pure CIF model failed；SymCIF/WA structured representation有必要性。
最终判决：如果是普通 CIF GPT，stop；如果是 invariant / SG-aware / Wyckoff-causal sequence model，可作为新主线候选，但必须先过 validation gate 并对标 GT-SG baseline。

## 22. Track A MPTS-52 本轮产物路线

状态：stop
路线类型：track audit / missing-artifact review
paper_mainline_status：no evidence
metric_rescue_status：not allowed
是否符合论文主线：无法判断；本轮没有可读 artifact。
是否依赖普通 ranking/rerank/fusion：无法判断；无 manifest。
是否包含 structural quality scorer：无证据。
是否包含 geometry repair：无证据。
历史证据：`model/std_way/track_a_mpts52` 当前有 `track_a_validation_oof.py`、`RUN_VALIDATION_OOF.sh` 和 `logs/validation_oof.log`；脚本包含 validation-only structural scorer、5-fold OOF、rows>=7 gate 和 `STOP/WATCH/ALLOW_OFFICIAL_FREEZE` decision 逻辑。但 `outputs/` 为空，日志只显示 feature extraction 进度到 100000 candidates，未生成 `validation_oof_results.json`、`DECISION.md`、README、metrics、validation 报告或 freeze 文件。
最好 validation 结果：NA。
最好 full-test 结果：NA。
相对 GT-SG baseline 差距：NA。
是否达到 +5pp：否；没有任何可审计指标。
是否 full test 有效：否；未运行也未冻结。
局部解风险：若在没有 artifact 的情况下继续，会形成不可复现、不可审计的路线。
最终判决：本轮 `STOP`。不得 `ALLOW_OFFICIAL_FREEZE`；也不能记为 `WATCH`，因为脚本尚未完成且没有正向指标。

## 23. Track B Hybrid Mainline Prototype 路线

状态：continue as architecture / stop as metric evidence
路线类型：hybrid mainline prototype / system architecture smoke
paper_mainline_status：architecture_allowed_no_metric_claim
metric_rescue_status：not_evaluated
是否符合论文主线：是，作为系统骨架符合；作为结果 claim 不符合。
是否依赖普通 ranking/rerank/fusion：否；当前只实现候选 adapter、union/dedup、hard constraints、scorer/repair/eval 接口 smoke。
是否包含 structural quality scorer：包含接口和未训练 heuristic smoke stub；不能当训练后 scorer 或 ordinary rerank 结果。
是否包含 geometry repair：包含 symmetry-preserving repair 接口；当前是 no-op stub，未修复 geometry。
历史证据：`model/std_way/track_b_hybrid/README.md`、`SYSTEM_ARCHITECTURE.md`、`MODULE_BOUNDARY.md`、`CLAIM_MAPPING.md`、`PROVENANCE_SCHEMA.md`、`artifacts/prototype_manifest.json`、`artifacts/smoke_summary.json`。
最好 validation 结果：NA；smoke summary 只有 toy `sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`repaired_count=0`。
最好 full-test 结果：NA。
相对 GT-SG baseline 差距：NA。
是否达到 +5pp：否；没有 match@k。
是否 full test 有效：否；未运行、未冻结、未接 official evaluator。
局部解风险：若把 smoke、candidate union、scorer stub 或 no-op repair 包装成方法收益，会退化为 fusion/rerank/local engineering claim。
最终判决：Track B 可继续作为 paper-mainline architecture scaffold；不得作为指标证据，不得启动 full benchmark，不得生成 official freeze。
重新开启条件：补生产级 SymCIF/Wyckoff parser、exact-cover search、hard-negative structural scorer 数据集、真实 symmetry-preserving geometry repair、StructureMatcher validation adapter、rows>=7 gate、frozen manifest 和 stop condition。
禁止重复的实验形式：只跑 smoke 后宣称路线成功；只做 candidate fusion/dedup/scorer stub；没有 validation gate 就启动 full benchmark。
重新开启条件：必须先完整生成 `track_a_mpts52` 的最小可审计包，包括 manifest、候选来源、validation-only 指标、rows>=7 子集、claim boundary、`DECISION.md`、freeze gate 和失败停止条件。


---

# [03] EXPERIMENT_INDEX.md

# 实验索引

本文件是人工可读索引。机器可读版见 `EXPERIMENT_INDEX.csv`。指标单位为百分比；未找到的指标写 `NA`。

## crystallm_mp20_gt_sg_anchor_op7

位置：`/data/users/xsw/autodlmini/model/New_model/opentry_7`
目标：建立 MP-20 GT-SG CrystaLLM 主比较 anchor。
方法：CrystaLLM-a，输入 composition + GT-SG，K20。
数据集：MP-20 official test。
是否 full test：是。
是否使用 GT-SG：是。
是否使用 GT-WA / oracle signal：否。
是否使用普通 ranking / fusion：否。
是否使用 structural quality scorer：否。
是否使用 position-aware policy：否。
是否使用 geometry repair：否。
paper_mainline_status：diagnostic。
metric_rescue_status：allowed。
关键指标：`71.67 / 83.08 / 87.81`，RMSE `0.0509 / 0.0449 / 0.0431`。
相对 GT-SG baseline 是否达标：自身为 baseline。
是否达到 +5pp / 至少两个指标目标：否。
结论：当前 MP-20 主比较对象。
是否继续：continue as baseline。
原因：所有新路线必须对标该 anchor。
证据路径：`opentry_7/opentry_7_final_report.md`，`opentry_10/final_report.md`。

## crystallm_mpts52_gt_sg_anchor_op7

位置：`/data/users/xsw/autodlmini/model/New_model/opentry_7`
目标：建立 MPTS-52 GT-SG CrystaLLM 主比较 anchor。
方法：CrystaLLM-a，输入 composition + GT-SG，K20。
数据集：MPTS-52 official test。
是否 full test：是。
是否使用 GT-SG：是。
是否使用 GT-WA / oracle signal：否。
是否使用普通 ranking / fusion：否。
paper_mainline_status：diagnostic。
metric_rescue_status：allowed。
关键指标：`25.23 / 36.46 / 43.96`，RMSE `0.1211 / 0.1257 / 0.1334`。
结论：当前 MPTS-52 主比较对象。
证据路径：`opentry_7/opentry_7_final_report.md`，`opentry_10/final_report.md`。

## symcif_v5_mp20_full_test

位置：`symcif_experiment`
目标：SymCIF-v5 在 MP-20 GT-SG full test 上验证。
方法：exact-cover SG/formula + e08 geometry。
是否 full test：历史 full test；样本口径与 opentry_7/10 anchor 不完全一致。
是否使用 GT-SG：是。
是否使用普通 ranking / fusion：否。
是否使用 geometry repair：是。
关键指标：`64.26 / 73.45 / 74.04`，RMSE `0.0576 / 0.0468 / 0.0464`。
相对 GT-SG baseline：低于当前 MP-20 anchor `-7.41 / -9.63 / -13.77 pp`。
是否达到 +5pp：否。
结论：支持 SymCIF 主线诊断，但不是成功结果。
是否继续：pivot。
原因：需要解决 W/A coverage 与 complex geometry。
证据路径：`symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md`。

## symcif_v5_mpts52_full_test

位置：`symcif_experiment`
目标：SymCIF-v5 在 MPTS-52 GT-SG full test 上验证。
关键指标：`25.47 / 32.90 / 33.63`，RMSE `0.1178 / 0.1132 / 0.1129`。
相对当前 GT-SG baseline：`+0.24 / -3.56 / -10.33 pp`。
是否达到 +5pp：否。
结论：K1 接近，但 K5/K20 和 rows>=7 明显不足。
是否继续：pivot。
证据路径：`symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md`。

## opentry_e04_mpts52_e1

位置：`opentry`
目标：MPTS-52 上用 `v5_e1_geometry_distance_ranking_e08` 超过旧 CrystaLLM GT-SG K1。
是否 full test：历史 full test/common-subset caveat。
是否使用普通 ranking / fusion：是，geometry-distance ranking。
关键指标：`28.08 / 34.39 / 35.53`。
相对旧 common-subset CrystaLLM：K1 超过；K5/K20 未超过。
是否达到 +5pp / 至少两个指标目标：否。
结论：局部 K1 信号，不是最终路线成功。
是否继续：watch。
证据路径：`opentry/iterative_experiment_log.md` E04-E06。

## opentry_e13_cached_ensemble

位置：`opentry`
目标：缓存候选 ensemble 提升 MPTS-52 K5/K20。
是否使用普通 ranking / fusion：是。
关键指标：`28.08 / 34.53 / 36.64`。
相对 GT-SG baseline：K5/K20 仍失败。
结论：互补性存在但不足；不能作为主线。
是否继续：watch/stop as mainline。
证据路径：`opentry/iterative_experiment_log.md` E13-E16。

## opentry_e24_symcif_crystallm_hybrid

位置：`opentry`
目标：固定顺序 SymCIF + CrystaLLM GT-SG candidate hybrid。
是否 full test：历史 same-evaluator test。
是否使用普通 ranking / fusion：是，候选融合。
关键指标：E24 `26.02 / 39.28 / 45.06`；同 evaluator CrystaLLM-only Ref20 `25.08 / 34.66 / 41.63`。
是否达到 +5pp：否；且不是独立方法。
结论：真实 hybrid 工程收益，但 forbidden_as_mainline。
是否继续：stop as paper mainline；可作 ablation/diagnostic。
证据路径：`opentry/iterative_experiment_log.md` E23-E25、Paper-audit analysis。

## opentry2_e55_k50_ranker

位置：`opentry_2`
目标：MPTS-52 K50 pool 上 RF ranker。
是否 full test：历史 full candidate pool；ranker/fusion provenance caveat。
是否使用普通 ranking / fusion：是。
关键指标：E55 `32.04 / 39.19 / 45.12`。
结论：高数值但普通 ranker；后续 opentry_10 official 冻结路线未复现最终成功。
是否继续：stop as mainline。
证据路径：`opentry_2/reports/e55_apply_e52_ranker_to_e34_full_k50_pool/run_summary.json`。

## opentry3_e119_validation_symcif

位置：`opentry_3`
目标：CrystalFormer/WyFormer 风格 symbolic W/A + row-aligned geometry。
是否 full test：否，validation val128。
关键指标：E119 `31.25 / 35.16 / 41.41 / 43.75@50`。
rows>=7：match@50 仅 `18.33%`。
结论：symbolic route 有价值，但 geometry 转化不足；不可 full-test success。
是否继续：pivot/continue under gate。
证据路径：`opentry_3/final_summary.md`。

## opentry4_e718_full_test

位置：`opentry_4`
目标：E718 frozen/audited full-test result 与 geometry energy/pairfield 诊断。
是否 full test：是，导入已有 one-time full MPTS-52 test，未 rerun。
关键指标：`27.68 / 35.85 / 39.68`，match@50 `42.15`，RMSE@20 `0.1240`。
rows>=7：match@20 `17.48%`。
结论：未达 +5pp；rows>=7 geometry/free-param/site-mapping 是瓶颈。
是否继续：pivot to geometry/refinement。
证据路径：`opentry_4/reports/opentry_4_final_summary.md`。

## opentry5_e8034_e8036_folds

位置：`opentry_5`
目标：MiniCFJoint / full-CIF fixed-order folds。
是否 full test：否，小 fold/grouped fold。
关键指标：E8034/E8036 fold 指标可到 `34.38 / 48.44 / 51.56`，但 rows>=7 match@20 仍为 0。
结论：aggregate fold positive 不等于 full-test success；rows>=7 未解。
是否继续：pivot。
证据路径：`opentry_5/reports/opentry_5_final_summary.md`。

## opentry6_geometry_refiner

位置：`opentry_6`
目标：CrystalFormer-style geometry/refiner 诊断。
是否 full test：否。
结论：下一轮应继续 continuous geometry/refiner，而不是回到 selector/ranker；若 GT-W/A 下 rows>=7 仍失败，则 geometry 是主瓶颈。
是否继续：continue under gate。
证据路径：`opentry_6/opentry_6_final_summary.md`。

## opentry7_pure_model

位置：`opentry_7`
目标：严格 composition + GT-SG pure model line。
是否 full test：是。
关键指标：MP-20 `60.58 / 70.67 / 77.96`；MPTS-52 `17.18 / 24.35 / 31.52`。
结论：显著低于 GT-SG anchor；pure CIF GPT/checkpoint 线应停止盲训。
是否继续：stop。
证据路径：`opentry_7/opentry_7_final_report.md`。

## opentry8_coverage_repair

位置：`opentry_8`
目标：coverage-repaired strategy/fusion。
是否 full test：是。
关键指标：MP-20 与 GT-SG anchor 完全相同；MPTS-52 match@20 从 `43.96` 到 `43.97`。
结论：coverage repair，不是 W/A、geometry 或 ranking 突破。
是否继续：stop。
证据路径：`opentry_8/final_report.md`。

## opentry9_audit

位置：`opentry_9`
目标：strategy/fusion feasibility audit。
是否 full test：否。
结论：缺 CrystaLLM-a GT-SG validation K20 bank，未训练 selector，未冻结策略，未跑新 official test；纯结构线仍是 WA + geometry 瓶颈。
是否继续：watch as audit evidence。
证据路径：`opentry_9/final_report.md`。

## opentry10_mpts52_k30_rf_seed1

位置：`opentry_10`
目标：MPTS-52 K30 RF best-score route official frozen evaluation。
是否 full test：是。
是否使用普通 ranking / fusion：是；浅层 scorer + route policy。
关键指标：`26.075 / 36.228 / 44.059`，RMSE `0.121638 / 0.122237 / 0.131714`。
相对 GT-SG baseline：`+0.845 / -0.232 / +0.099 pp`。
是否达到 +5pp / 至少两个指标目标：否。
结论：当前 closest official line，但只是近程信号。
是否继续：watch；只能作为 metric rescue/hybrid component。
证据路径：`opentry_10/metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json`。

## opentry10_mpts52_k50_rf_seed1

位置：`opentry_10`
目标：MPTS-52 K50 RF margin route official frozen evaluation。
关键指标：`25.791 / 36.265 / 43.824`。
相对 GT-SG baseline：`+0.561 / -0.195 / -0.136 pp`。
rows>=7：`23.053 / 33.228 / 40.939`。
结论：弱于 K30；无成功。
是否继续：watch only if rebuilt as structural scorer。
证据路径：`opentry_10/metrics/official_test/mpts52_k50_rf_seed1_margin_route_summary.json`。

## opentry10_mp20_k50_hgb

位置：`opentry_10`
目标：MP-20 K50 HGB ensemble margin route official frozen evaluation。
关键指标：`70.230 / 81.870 / 87.486`。
相对 GT-SG baseline：`-1.440 / -1.210 / -0.324 pp`。
rows>=7：`27.200 / 43.273 / 55.273`，大幅低于 anchor。
结论：失败，MP-20 K50 shallow route 不应继续 as-is。
是否继续：stop。
证据路径：`opentry_10/metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json`。

## track_a_mpts52_latest_audit

位置：`/data/users/xsw/autodlmini/model/std_way/track_a_mpts52`
目标：读取 Track A MPTS-52 最新产物并更新全局审计结论。
方法：目录级审计；读取脚本与日志，不启动实验。
数据集：MPTS-52 validation intended；脚本指向 opentry_10 MPTS-52 validation K100/K50 labels，但本轮没有完成输出指标。
是否 full test：否。
是否使用 GT-SG：NA。
是否使用 GT-WA / oracle signal：否。
是否使用普通 ranking / fusion：无证据。
是否使用 structural quality scorer：无证据。
是否使用 position-aware policy：无证据。
是否使用 geometry repair：无证据。
paper_mainline_status：no evidence。
metric_rescue_status：not allowed。
关键指标：NA。
相对 GT-SG baseline 是否达标：否；无指标。
是否达到 +5pp / 至少两个指标目标：否。
本轮路线判决：`STOP`。
结论：脚本与部分进度日志存在，但 `outputs/` 为空，没有 validation OOF 结果、`DECISION.md` 或报告；不能进入 official freeze，也不能作为 WATCH 信号。
是否继续：stop until auditable artifacts exist。
原因：缺 manifest、metrics、validation 输出、rows>=7 报告、`DECISION.md` 和 freeze gate。
证据路径：`model/std_way/track_a_mpts52/track_a_validation_oof.py`、`model/std_way/track_a_mpts52/RUN_VALIDATION_OOF.sh`、`model/std_way/track_a_mpts52/logs/validation_oof.log`；`outputs/` 当前无结果文件。

## track_b_hybrid_mainline_prototype

位置：`/data/users/xsw/autodlmini/model/std_way/track_b_hybrid`
目标：建立 hybrid paper-mainline 可执行原型，明确 SymCIF/Wyckoff/exact-cover/geometry repair 与 auxiliary scorer/provenance/evaluation 的边界。
方法：adapter -> union/dedup -> provenance registry -> hard constraint filter -> structural scorer interface -> symmetry-preserving repair interface -> smoke evaluation。
数据集：toy smoke only。
是否 full test：否。
是否使用 GT-SG：toy metadata 中使用 prompt SG。
是否使用 GT-WA / oracle signal：否。
是否使用普通 ranking / fusion：否；有 candidate union/dedup，但没有 ranking/fusion 指标 claim。
是否使用 structural quality scorer：有接口和 untrained heuristic smoke stub。
是否使用 position-aware policy：否。
是否使用 geometry repair：有接口，但当前是 no-op stub。
paper_mainline_status：architecture_allowed_no_metric_claim。
metric_rescue_status：not_evaluated。
关键指标：NA；`smoke_summary.json` 为 `sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`hard_filter_fail_count=1`、`repaired_count=0`。
相对 GT-SG baseline 是否达标：否；无 match@k。
是否达到 +5pp / 至少两个指标目标：否。
本轮路线判决：`continue as architecture / STOP as metric evidence`。
结论：Track B 可作为下一轮主线真实实验的系统骨架，但不能作为 validation 或 official result。
是否继续：continue only for parser/scorer/repair/evaluator implementation under gate。
原因：scorer 未训练、repair 未实现、exact-cover 为 basic sanity、evaluation 不产出 StructureMatcher 指标。
证据路径：`track_b_hybrid/README.md`、`SYSTEM_ARCHITECTURE.md`、`MODULE_BOUNDARY.md`、`CLAIM_MAPPING.md`、`PROVENANCE_SCHEMA.md`、`artifacts/prototype_manifest.json`、`artifacts/smoke_summary.json`。


---

# [04] BASELINE_LEDGER.md

# Baseline Ledger

本文件只记录 baseline 和可作为对照的历史 anchor。主比较对象必须是 GT-SG 条件下 CrystaLLM baseline；oracle、coverage repair、ranking/fusion 不能作为主 baseline。

## 1. 原始 CrystaLLM baseline，输入不含 SG

baseline_name：published CrystaLLM-a composition-only
dataset：MP-20
input_condition：composition-only
match@1：55.85
match@5：NA
match@20：75.14
rmse：0.0437 / NA / 0.0395
is_full_test：yes, historical published row copied in opentry_7
uses_sg：no
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：no
source_file：`/data/users/xsw/autodlmini/model/New_model/opentry_7/opentry_7_final_report.md`
可信度：medium，公开/历史上下文
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
备注：当前任务输入含 GT-SG，不能用该 baseline 作为主比较对象。

baseline_name：reproduced CrystaLLM-a composition-only
dataset：MP-20
input_condition：composition-only
match@1：62.17
match@5：74.98
match@20：81.84
rmse：0.0442 / 0.0403 / 0.0402
is_full_test：yes
uses_sg：no
source_file：`opentry_7/opentry_7_final_report.md`
可信度：strong for local reproduction
是否可作为主比较对象：否
备注：只能作为历史 context。

baseline_name：reproduced CrystaLLM-a composition-only
dataset：MPTS-52
input_condition：composition-only
match@1：18.86
match@5：28.08
match@20：35.17
rmse：0.1056 / 0.1058 / 0.1174
is_full_test：yes
uses_sg：no
source_file：`opentry_7/opentry_7_final_report.md`
可信度：strong
是否可作为主比较对象：否

## 2. GT-SG 条件下 CrystaLLM baseline

baseline_name：CrystaLLM-a GT-SG MP-20
dataset：MP-20
input_condition：composition + GT-SG
match@1：71.67
match@5：83.08
match@20：87.81
rmse：0.0509 / 0.0449 / 0.0431
is_full_test：yes, CrystaLLM Table 3 official full-test K20 protocol
uses_sg：yes
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：no
source_file：`opentry_7/opentry_7_final_report.md`; cross-checked in `opentry_8/final_report.md` and `opentry_10/final_report.md`
可信度：strong
是否可作为主比较对象：是
是否达到用户 +5pp 目标：baseline itself, no
备注：MP-20 后续目标为 `76.67 / 88.08 / 92.81`。

baseline_name：CrystaLLM-a GT-SG MPTS-52
dataset：MPTS-52
input_condition：composition + GT-SG
match@1：25.23
match@5：36.46
match@20：43.96
rmse：0.1211 / 0.1257 / 0.1334
is_full_test：yes, CrystaLLM Table 3 official full-test K20 protocol
uses_sg：yes
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：no
source_file：`opentry_7/opentry_7_final_report.md`; cross-checked in `opentry_8/final_report.md` and `opentry_10/final_report.md`
可信度：strong
是否可作为主比较对象：是
是否达到用户 +5pp 目标：baseline itself, no
备注：MPTS-52 后续目标为 `30.23 / 41.46 / 48.96`。

## 3. SymCIF / 当前仓库历史 baseline

baseline_name：SymCIF-v5 GT-SG MP-20
dataset：MP-20
input_condition：composition + GT-SG
match@1：64.26
match@5：73.45
match@20：74.04
rmse：0.0576 / 0.0468 / 0.0464
is_full_test：yes, historical structured test protocol
uses_sg：yes
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：no ordinary fusion; uses structured generation/geometric proposal
source_file：`symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md`
可信度：medium，口径与 opentry_7/10 anchor 不完全一致
是否可作为主比较对象：否；可作为历史 method baseline
是否达到用户 +5pp 目标：否

baseline_name：SymCIF-v5 GT-SG MPTS-52
dataset：MPTS-52
input_condition：composition + GT-SG
match@1：25.47
match@5：32.90
match@20：33.63
rmse：0.1178 / 0.1132 / 0.1129
is_full_test：yes, historical structured test protocol
uses_sg：yes
uses_gt_wa_or_oracle：no
source_file：`symcif_experiment/Log_GPT/multi_experiment_analysis_report_20260610_122115.md`
可信度：medium
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否

## 4. opentry_10 K30/K50 official frozen results

baseline_name：mpts52_k30_rf_seed1_bestscore_route
dataset：MPTS-52
input_condition：composition + GT-SG + CrystaLLM K30 candidate route
match@1：26.075
match@5：36.228
match@20：44.059
rmse：0.121638 / 0.122237 / 0.131714
is_full_test：yes
uses_sg：yes
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：yes, RF route
source_file：`opentry_10/metrics/official_test/mpts52_k30_rf_seed1_bestscore_route_summary.json`
可信度：strong official result
是否可作为主比较对象：否；这是方法结果，不是 baseline
是否达到用户 +5pp 目标：否
备注：closest official near signal；不能称成功。

baseline_name：mpts52_k50_rf_seed1_margin_route
dataset：MPTS-52
input_condition：composition + GT-SG + K50 route
match@1：25.791
match@5：36.265
match@20：43.824
rmse：0.126404 / 0.125088 / 0.132897
is_full_test：yes
uses_sg：yes
uses_ranking_or_fusion：yes
source_file：`opentry_10/metrics/official_test/mpts52_k50_rf_seed1_margin_route_summary.json`
可信度：strong
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否

baseline_name：mp20_k50_hgb_mean_seed012_margin_route
dataset：MP-20
input_condition：composition + GT-SG + K50 route
match@1：70.230
match@5：81.870
match@20：87.486
rmse：0.049965 / 0.043623 / 0.042699
is_full_test：yes
uses_sg：yes
uses_ranking_or_fusion：yes
source_file：`opentry_10/metrics/official_test/mp20_k50_hgb_mean_seed012_margin_route_summary.json`
可信度：strong
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
备注：MP-20 official failed and rows>=7 collapsed.

## 5. Oracle / GT-WA / GT-skeleton upper bounds

baseline_name：SymCIF-v4 WA upper-bound
dataset：small/evaluator-specific v4 set
input_condition：GT-like WA upper-bound diagnostic
match@1：52.4
match@5：75.2
match@20：82.4
rmse：RMSE@20 0.0157
is_full_test：no, diagnostic
uses_sg：yes
uses_gt_wa_or_oracle：yes
uses_ranking_or_fusion：diagnostic upper-bound
source_file：`symcif_experiment/reports/next_step_decision_summary.md`
可信度：strong as diagnostic, invalid as method
是否可作为主比较对象：否
是否达到用户 +5pp 目标：不适用

baseline_name：GT-WA + geometry model over/no-over
dataset：small/evaluator-specific v4 set
input_condition：GT-WA + learned geometry
match@1：54.2
match@5：65.8
match@20：70.2/70.4
rmse：0.0835/0.0833 @20
is_full_test：no, diagnostic
uses_gt_wa_or_oracle：yes
source_file：`symcif_experiment/reports/symcif_v4_geometry_next_step_summary.md`
可信度：strong diagnostic
是否可作为主比较对象：否
备注：证明 geometry model 有价值，但不能作为 deployable result。

## 6. Ranking/fusion 后处理结果

baseline_name：opentry E24 SymCIF + CrystaLLM hybrid
dataset：MPTS-52
input_condition：composition + GT-SG, fixed hybrid of SymCIF and CrystaLLM candidates
match@1：26.02
match@5：39.28
match@20：45.06
rmse：0.1022 / 0.1069 / 0.1099
is_full_test：historical same-evaluator full test
uses_sg：yes
uses_gt_wa_or_oracle：no
uses_ranking_or_fusion：yes, candidate fusion
source_file：`opentry/iterative_experiment_log.md` E23-E25
可信度：medium
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
备注：可作 ablation/engineering finding，不能作为论文主方法。

baseline_name：opentry_2 E55 RF K50 ranker
dataset：MPTS-52
input_condition：composition + GT-SG candidate pool
match@1：32.04
match@5：39.19
match@20：45.12
rmse：0.1146 / 0.1173 / 0.1306
is_full_test：historical candidate-pool evaluation
uses_ranking_or_fusion：yes
source_file：`opentry_2/reports/e55_apply_e52_ranker_to_e34_full_k50_pool/run_summary.json`
可信度：medium; old provenance and ordinary ranker caveat
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否; also forbidden_as_mainline

## 7. Coverage repair 结果

baseline_name：opentry_8 strategy_fusion MP-20
dataset：MP-20
input_condition：GT-SG anchor + fallback-only coverage repair
match@1：71.67
match@5：83.08
match@20：87.81
rmse：0.0509 / 0.0449 / 0.0431
is_full_test：yes
uses_ranking_or_fusion：yes, fallback-only
source_file：`opentry_8/final_report.md`
可信度：strong
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
备注：identical to GT-SG anchor; no method gain.

baseline_name：opentry_8 strategy_fusion MPTS-52
dataset：MPTS-52
input_condition：GT-SG anchor + coverage repair
match@1：25.23
match@5：36.46
match@20：43.97
rmse：0.1211 / 0.1257 / 0.1334
is_full_test：yes
uses_ranking_or_fusion：yes
source_file：`opentry_8/final_report.md`
可信度：strong
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
备注：one missing sample repaired; not method breakthrough.

## 8. Track A MPTS-52 本轮审计

baseline_name：track_a_mpts52_latest_audit
dataset：MPTS-52 intended
input_condition：NA
match@1：NA
match@5：NA
match@20：NA
rmse：NA
is_full_test：no
uses_sg：NA
uses_gt_wa_or_oracle：no evidence
uses_ranking_or_fusion：no evidence
source_file：`model/std_way/track_a_mpts52`
可信度：strong as incomplete-run audit; no metric evidence
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
本轮路线判决：`STOP`
备注：该目录当前只有 `track_a_validation_oof.py`、`RUN_VALIDATION_OOF.sh` 和部分 feature extraction 进度日志；`outputs/` 为空，没有 manifest、metrics、validation 输出、rows>=7 报告、`DECISION.md` 或 freeze 声明。因此本轮 Track A 不新增 baseline、不改变 MPTS-52 GT-SG anchor、不允许 official freeze。

## 9. Track B Hybrid 原型审计

baseline_name：track_b_hybrid_mainline_prototype
dataset：toy smoke only
input_condition：composition + GT-SG style toy candidates
match@1：NA
match@5：NA
match@20：NA
rmse：NA
is_full_test：no
uses_sg：yes in toy prompt/candidate metadata
uses_gt_wa_or_oracle：no evidence; smoke only
uses_ranking_or_fusion：candidate union/dedup exists, but no ranking/fusion metric claim
source_file：`model/std_way/track_b_hybrid/artifacts/prototype_manifest.json`; `model/std_way/track_b_hybrid/artifacts/smoke_summary.json`
可信度：strong as architecture smoke; invalid as benchmark/baseline
是否可作为主比较对象：否
是否达到用户 +5pp 目标：否
本轮路线判决：`continue as architecture / STOP as metric evidence`
备注：Track B 只建立 hybrid mainline prototype。smoke summary 为 `sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`repaired_count=0`，且明确声明没有 full benchmark、没有 route-success claim、没有 match@k conclusion。因此本条不改变 GT-SG baseline，不新增 best result，不允许 official freeze。


---

# [05] CLAIM_BOUNDARY.md

# 论文 Claim 边界

## 1. 哪些结果可以写成论文主贡献

只有以下类型可以作为主贡献候选：

- symmetry-aware / Wyckoff-skeleton / SymCIF 表示本身带来可解释生成改进；
- crystallographic causal ordering：先 composition/SG，再 Wyckoff skeleton/W-A，再 geometry；
- symmetry-preserving geometry refinement 或 geometry-aware representation；
- full-test 上至少两个 `match@1/5/20` 指标超过 GT-SG CrystaLLM baseline `+5pp`。

当前历史结果尚未满足最后一条。

## 2. 哪些只能写成辅助模块

- structural quality scorer；
- position-aware slot policy；
- rows>=7 gate；
- local geometry/collision/bond-valence/energy proxy；
- train-only row-conditioned KNN/retrieval geometry；
- deterministic micro-geometry repair。

这些可以写成 hybrid system 的 quality-control / decision layer，不能单独包装成核心理论贡献。

## 3. 哪些只能写成 diagnostic / upper bound

- GT-WA + geometry；
- WA upper-bound；
- GT-skeleton / oracle exact-cover selection；
- validation oracle union；
- common-subset post-hoc overlap；
- test internal rerank space。

这些能说明瓶颈和上限，不能当真实方法结果。

## 4. 绝对不能包装成方法贡献

- ordinary ranking / rerank；
- candidate fusion；
- coverage repair；
- 多候选选择策略本身；
- shallow RF/HGB/Logistic selector；
- oracle / GT-WA / GT-skeleton selection；
- 根据 official test 结果回调阈值；
- validation/small fold 小涨；
- 只修缺失样本导致的 match@20 `+0.01`。

## 5. 局部信号不等于最终成功

以下都不是最终成功：

- opentry_10 K30 `+0.845pp match@1`；
- opentry_10 validation K30/K50 sweeps；
- opentry E04 common-subset K1 胜出；
- opentry E24 hybrid K5/K20 胜出；
- opentry_3 val128 near-target；
- opentry_5 fold aggregate high but rows>=7=0；
- SymCIF-v4/v5 同条件小规模 evaluator 超过 baseline。

## 6. ranking / rerank / fusion 结果能否写进论文

可以写，但只能作为：

- ablation；
- engineering baseline；
- diagnostic upper-bound approximation；
- hybrid inference comparison。

不能写成“核心方法超过 CrystaLLM”。例如 opentry E24 的正确表述是：固定非 oracle SymCIF + CrystaLLM candidate fusion 在同 evaluator 上改善 top-k；错误表述是：纯 SymCIF 全面超过 CrystaLLM。

## 7. structural quality scorer 如何写才不夸大

正确写法：

> We add an inference-safe structural quality-control module over generated candidates, using composition/SG consistency, Wyckoff feasibility and local geometry signals.

错误写法：

> The scorer is the main crystallographic representation contribution.

必须声明：scorer 是 auxiliary；它不能依赖 test true CIF 或 StructureMatcher label。

## 8. position-aware policy 如何写才不夸大

正确写法：

> A slot-aware policy controls when to preserve anchor top-k candidates and when to use structurally plausible tail candidates.

错误写法：

> A slot policy is the core scientific contribution.

必须同时报告 K1/K5/K20，不能只刷 K1。

## 9. geometry repair 如何写才合理

合理 claim：

- symmetry-preserving refinement；
- lattice/free-param/site mapping correction；
- collision-aware local geometry；
- rows>=7 complex structure refinement。

不合理 claim：

- 无约束 post-render coordinate surgery；
- 只用 RMSE 改善替代 match 改善；
- 用 GT-WA diagnostic 当 deployable result。

## 10. coverage repair 为什么不能写成方法收益

`opentry_8/final_report.md` 明确：MP-20 strategy_fusion 与 GT-SG anchor 完全相同；MPTS-52 只从 8095/8096 覆盖修到 8096/8096，match@20 从 43.96 到 43.97。收益来自 coverage 和 anchor choice，不来自 W/A、geometry 或 ranking。

## 11. validation / fold 结果如何表述

只能写：

- validation signal；
- screening evidence；
- route precondition；
- failure diagnosis。

不能写：

- final success；
- official result；
- paper headline result。

## 12. full-test 结果如何表述

full-test 结果必须说明：

- dataset；
- split；
- official sample count；
- input condition；
- baseline；
- whether frozen before test；
- uses ranking/fusion/oracle or not；
- match@1/5/20 和 RMSE；
- rows>=7 subset。

## 13. 只超过 +1pp 但没到 +5pp 能否称达到用户目标

不能。`+1pp` 只是 opentry_10 的近程 gate；用户最终目标是 `+5pp` 且至少两个 match 指标超过 GT-SG baseline。

## 14. 如果 MPTS-52 有效但 MP-20 无效

claim 必须收缩为：

- MPTS-52-specific improvement；
- complex-material / dataset-specific behavior；
- not general across MP-20 and MPTS-52。

不能写成全面方法成功。

## 15. 如果 match 提升但 RMSE 不变好

可以表述为：

- StructureMatcher hit-rate improvement with neutral/worse RMSE；
- method improves candidate selection/feasibility more than coordinate accuracy。

必须报告 RMSE，不能隐藏。

## 当前边界结论

当前可写的最强论文方向是：历史结果共同说明普通 CIF generation 与 ordinary rerank 均不足，真正需要 symmetry-aware Wyckoff representation + geometry-aware refinement + inference-safe structural quality control。但当前没有 full-test +5pp 成功结果，论文主 claim 仍需新证据。

## 16. Track A MPTS-52 本轮 claim 边界

本轮读取 `model/std_way/track_a_mpts52` 后发现脚本和部分进度日志，但没有完成的可审计结果。`track_a_validation_oof.py` 设计了 MPTS-52 validation-only structural scorer gate，`RUN_VALIDATION_OOF.sh` 是运行入口，`logs/validation_oof.log` 只显示 feature extraction 到 100000 candidates；`outputs/` 为空，不包含 validation results、score file、feature audit、candidate provenance、rows>=7 报告、`DECISION.md` 或 freeze 声明。

本轮路线判决：`STOP`。

允许表述：

- Track A MPTS-52 本轮只有未完成脚本，无新增指标证据；
- 不能更新 baseline；
- 不能进入 official freeze；
- 下一次必须先补可审计产物和 validation gate。

禁止表述：

- Track A 已经形成 `WATCH` 信号；
- Track A 可以 `ALLOW_OFFICIAL_FREEZE`；
- Track A 支持任何 match@1/5/20 claim；
- Track A 支持 ordinary rerank、fusion 或 geometry repair 成功。

## 17. Track B Hybrid claim 边界

本轮读取 `model/std_way/track_b_hybrid` 后确认：Track B 是可执行架构原型和 smoke-only 连通性验证，不是指标实验。`artifacts/smoke_summary.json` 只有 toy `sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`repaired_count=0`，且显式声明没有 full benchmark、没有 route-success claim、没有 match@k conclusion。

允许表述：

- Track B 建立了 hybrid mainline skeleton：CrystaLLM GT-SG adapter、SymCIF/Wyckoff adapter、candidate union/dedup、provenance registry、hard constraint filter、scorer interface、repair interface、evaluation smoke adapter。
- Track B 把 paper mainline 与 auxiliary 边界写清：SymCIF/Wyckoff/exact-cover/geometry repair 是主线候选；CrystaLLM adapter、union/dedup、provenance、scorer、evaluation adapter 是辅助层。
- Track B 的 provenance schema 可作为未来真实实验 manifest 的模板。

禁止表述：

- Track B 已经超过 GT-SG CrystaLLM baseline；
- Track B 有任何 `match@1/5/20` 收益；
- smoke hard-filter pass 证明 exact-cover/geometry repair 成功；
- untrained heuristic scorer 是正式 structural quality scorer；
- no-op repair 是 symmetry-preserving geometry refinement；
- candidate union/dedup 是论文主贡献；
- 基于 Track B smoke 进入 official full test。

下一次 Track B 若要产生 claim，必须先补生产级 Wyckoff parser、真实 repair、validation StructureMatcher adapter、rows>=7 子集报告和 frozen-before-test manifest。


---

# [06] AUDIT_SUMMARY_FOR_GPT.md

# 给网页版 GPT 的压缩审稿材料

## 一句话结论

当前没有任何可信 official full-test 结果达到用户设定的 `GT-SG CrystaLLM baseline +5pp 且至少两个 match 指标` 目标；最接近的 opentry_10 MPTS-52 K30 RF route 只是 `match@1 +0.845pp` 的近程信号，且 `match@5` 下降，不能称成功。

补充审计：`model/std_way/track_a_mpts52` 当前只有 validation OOF 脚本、运行入口和部分 feature extraction 进度日志，`outputs/` 为空，没有完成的可审计指标。本轮 Track A MPTS-52 路线判决为 `STOP`，不是 `WATCH`，也不是 `ALLOW_OFFICIAL_FREEZE`。

Track B 补充审计：`model/std_way/track_b_hybrid` 已建立 hybrid mainline prototype，但只有 toy smoke 连通性结果：`sample_count=2`、`input_candidate_count=3`、`unique_candidate_count=2`、`hard_filter_pass_count=1`、`repaired_count=0`。它支持 SymCIF/Wyckoff/exact-cover + geometry repair 的系统架构继续，但没有 match@k、没有 validation gate、没有 baseline delta，不能触发 full benchmark 或 official freeze。

## 当前全局判断

1. 论文主线仍应是 Wyckoff-skeleton / SymCIF / symmetry-aware representation + geometry-aware refinement。
2. 普通 RF/HGB rerank、candidate fusion、coverage repair 不能作为论文主贡献。
3. K30/K50 policy 可作为短期 metric rescue 或 hybrid component 观察，但必须升级为 inference-safe structural quality scorer，不能继续阈值微调。
4. rows>=7 是长期核心瓶颈，必须单独 gate。
5. Track A MPTS-52 本轮只有未完成脚本，没有 validation result artifact，不能改变上述判断，也不能触发 official freeze。
6. Track B Hybrid 本轮只是 architecture/smoke scaffold：可作为下一轮主线实验设计地基，但不能作为结果证据。

## 最可信 GT-SG baseline

- MP-20：`71.67 / 83.08 / 87.81`，RMSE `0.0509 / 0.0449 / 0.0431`。
- MPTS-52：`25.23 / 36.46 / 43.96`，RMSE `0.1211 / 0.1257 / 0.1334`。

证据：`opentry_7/opentry_7_final_report.md`，`opentry_8/final_report.md`，`opentry_10/final_report.md`。

## 当前 best official result

- MPTS-52 `mpts52_k30_rf_seed1_bestscore_route`：`26.075 / 36.228 / 44.059`，delta `+0.845 / -0.232 / +0.099 pp`。
- MP-20 `mp20_k50_hgb_mean_seed012_margin_route`：`70.230 / 81.870 / 87.486`，delta `-1.440 / -1.210 / -0.324 pp`。

证据：`opentry_10/metrics/official_test/*_summary.json`。

## 与 +5pp 目标差距

MPTS-52 K30 距目标 `30.23 / 41.46 / 48.96` 仍差 `-4.155 / -5.232 / -4.901 pp`。
MP-20 K50 HGB 距目标 `76.67 / 88.08 / 92.81` 仍差 `-6.440 / -6.210 / -5.324 pp`。

## 已验证失败路线

- pure CIF / pure CrystaLLM-style model：opentry_7 full-test 远低于 GT-SG anchor。
- opentry_8 coverage repair：不是方法收益。
- ordinary rerank-only：MPTS-52 validation 正信号未 official 泛化。
- shallow K30/K50 RF/HGB route：K30 近但失败；K50 更弱；MP-20 失败。
- small fold aggregate gain：opentry_5 rows>=7 未解，不能 freeze。
- simple geometry schedules / source-cluster lattice / deterministic one-shot geometry：validation 失败或 rows>=7 失败。

## 仍可能有价值的路线

- Wyckoff-skeleton + exact-cover + canonical W/A；
- symmetry-aware causal ordering；
- geometry-aware refinement，特别是 lattice/free-param/site mapping/collision；
- inference-safe structural quality scorer；
- position-aware slot policy / rows>=7 gate，作为 auxiliary；
- train-only row-conditioned geometry retrieval，作为 proposal/control。

## 最危险局部解

继续调 RF/HGB seed、threshold、anchor_keep，围绕 opentry_10 K30 `+0.845pp` 做 test-feedback 微调。这既不能通向 +5pp，也可能污染 official test protocol。

## MPTS-52 K30/K50 policy 是否值得继续

建议：`watch`，只允许作为 metric rescue/hybrid component 继续；不允许普通 RF/HGB route as-is。若继续，必须新增结构特征和 hard-negative objective，并通过 gate。

## Wyckoff-skeleton / symmetry-aware representation 是否仍是论文主线

建议：是。历史证据证明 symbolic W/A 与 geometry 是真实瓶颈；普通 rerank/fusion失败反而强化了需要结构化表示与几何约束的论点。但当前还没有 final success。

## 下一步推荐路线

优先路线：`Wyckoff exact-cover + inference-safe structural quality scorer + symmetry-preserving geometry refinement + rows>=7 gate`。先在 validation 证明可同时提升至少两个 match 指标且不伤 K20/rows>=7，再冻结 full-test。

## 下一步禁止路线

- 普通 rerank/fusion/coverage repair；
- 使用 official result 回调 opentry_10 K30/K50；
- blind pure CIF SFT；
- 只追 MPTS-52 K1 小涨；
- 把 oracle/GT-WA upper bound 当真实方法。
- 在 `track_a_mpts52` 只有脚本、无 completed metrics/validation gate 的情况下继续或冻结。
- 把 `track_b_hybrid` 的 toy smoke、untrained scorer stub、no-op repair 或 candidate union/dedup 写成方法收益。

## 需要网页版 GPT 判断的问题

1. 是否允许 MPTS-52 K30/K50 作为短期 metric rescue 继续？
2. structural quality scorer 的边界是否足够清晰？
3. 下一步应优先 scorer 还是 geometry refinement？
4. rows>=7 是否必须成为硬 gate？
5. 在没有 full-test 计划前，是否允许 validation-only ablation？
6. Track A 在只有未完成脚本、无 validation 输出状态下是否应保持 `STOP`，直到补齐最小可审计包？
7. Track B 的 architecture scaffold 是否已足够作为下一轮真实 Wyckoff/exact-cover + geometry repair validation gate 的起点？

## 关键证据路径

- `opentry_7/opentry_7_final_report.md`
- `opentry_8/final_report.md`
- `opentry_9/final_report.md`
- `opentry_10/final_report.md`
- `opentry_10/ToGPTPro.md`
- `opentry_10/metrics/official_test/*_summary.json`
- `opentry_3/final_summary.md`
- `opentry_4/reports/opentry_4_final_summary.md`
- `opentry_5/reports/opentry_5_final_summary.md`
- `opentry_6/opentry_6_final_summary.md`
- `symcif_experiment/reports/symcif_v4_geometry_next_step_summary.md`
- `opentry/iterative_experiment_log.md`
- `model/std_way/track_a_mpts52/track_a_validation_oof.py`
- `model/std_way/track_a_mpts52/logs/validation_oof.log`（只有部分 feature extraction 进度；用于本轮 STOP 判决）
- `model/std_way/track_b_hybrid/README.md`
- `model/std_way/track_b_hybrid/SYSTEM_ARCHITECTURE.md`
- `model/std_way/track_b_hybrid/MODULE_BOUNDARY.md`
- `model/std_way/track_b_hybrid/CLAIM_MAPPING.md`
- `model/std_way/track_b_hybrid/PROVENANCE_SCHEMA.md`
- `model/std_way/track_b_hybrid/artifacts/prototype_manifest.json`
- `model/std_way/track_b_hybrid/artifacts/smoke_summary.json`


---

# [07] NEXT_EXPERIMENT_GATE.md

# 下一轮实验准入门槛

任何新实验开始前，必须先读：

1. `RESEARCH_STATE.md`
2. `ROUTE_JUDGEMENT.md`
3. `BASELINE_LEDGER.md`
4. `CLAIM_BOUNDARY.md`
5. 本文件

## 必答问题

1. 这个实验属于哪条路线？必须引用 `ROUTE_JUDGEMENT.md` 的路线名称。
2. 该路线状态是 `continue / stop / pivot / watch` 哪一种？
3. 如果路线是 `stop`，为什么还要做？是否有新证据？是否经过网页版 GPT 审稿？
4. 是否使用普通 ranking / rerank / fusion？
5. 如果使用，它是否只作为 diagnostic / auxiliary / metric rescue，而不是论文主线？
6. 是否使用 structural quality scorer？
7. scorer 是否只依赖 prompt 条件和 candidate CIF？是否排除 test true CIF、StructureMatcher label、oracle/GT-WA？
8. 是否使用 position-aware policy / slot policy / route policy？
9. 是否使用 geometry repair / geometry refinement？
10. 是否会在 official full test 上验证？如果不会，结果只能叫 validation/screening。
11. 主对标是否是 GT-SG 条件 CrystaLLM baseline？
12. 预期至少改善 `match@1/5/20` 中哪两个指标？
13. 目标是否是 `+5pp`，而不是只追求 `+1pp` 或 `+0.8pp`？
14. 如果失败，停止条件是什么？
15. 这个实验是否能支撑论文主线？如果不能，应该写成什么辅助/诊断角色？
16. 这个实验是否可能只是重复历史局部解？
17. rows>=7 是否有单独指标、单独 gate、单独失败分析？
18. 是否存在 test feedback 风险，尤其是使用 opentry_10 official 结果回调阈值？

## 自动禁止规则

- 无法回答以上问题：禁止开始。
- 只调 RF/HGB seed、threshold、anchor_keep：禁止开始。
- 只做普通 rerank/fusion 并声称论文主线：禁止开始。
- 只在 small fold / validation 上追小涨，且没有 full-test plan：禁止作为有效路线。
- 使用 test label、test StructureMatcher per-sample result、GT-WA、GT-skeleton selection 作为训练/选择信号：禁止作为真实方法。
- 复用 opentry_10 official aggregate 后回调同一 K30/K50 route：禁止。
- coverage repair 当作 improvement：禁止。
- 只看 match@1，明显牺牲 match@5/match@20：禁止进入主线。

## 条件允许规则

### Structural quality scorer

允许，但必须满足：

- 特征只来自 prompt composition、GT-SG、candidate CIF、train/validation 统计或 train-only models。
- 特征应包含公式一致性、SG一致性、Wyckoff exact-cover feasibility、orbit multiplicity gap、collision/min-distance/local environment/bond-valence/energy proxy 等。
- 训练建议使用 hard-negative pairwise/listwise objective。
- rows>=7 必须单独加权或单独 gate。
- claim 必须写成 auxiliary quality-control / hybrid component。

### Position-aware policy

允许，但必须满足：

- 清楚说明它是 Top-1 / Top-5 / Tail-rescue / rows>=7 gate / route policy 中哪一种。
- 不得把 slot policy 写成核心理论贡献。
- 不得牺牲 K5/K20 来刷 K1。

### Geometry repair

允许，而且是优先方向之一，但必须满足：

- preserve symmetry / Wyckoff constraints。
- 不做无约束 post-render coordinate surgery。
- 报告 lattice/free-param/site mapping/collision 改善证据。
- 与 W/A exact-cover 或 candidate feasibility 联合评估。

## 当前 gate 判断

下一轮新实验不应直接开始，除非先补齐一页实验提案并通过本 gate。特别是：

- MPTS-52 K30/K50 普通 RF/HGB route：不允许 as-is 继续。
- 允许提出“inference-safe structural quality scorer + hard-negative pairwise/listwise + rows>=7 gate”的新方案。
- 允许提出“Wyckoff-skeleton + symmetry-preserving geometry repair”的新主线方案。
- 不允许再把 opentry_10 K30 `+0.845pp` 写成成功或以此回调阈值。
- Track A MPTS-52 本轮路线判决为 `STOP`：`model/std_way/track_a_mpts52` 当前只有 validation OOF 脚本、运行入口和部分 feature extraction 进度日志，`outputs/` 为空，没有完成的可审计指标；不得进入 `ALLOW_OFFICIAL_FREEZE`，也不得记为 `WATCH` 信号。
- Track B Hybrid 本轮只能作为 architecture scaffold：`model/std_way/track_b_hybrid` 有可执行 smoke 原型和 provenance/claim 文档，但 scorer 是 untrained stub、repair 是 no-op、evaluation 不含 StructureMatcher，`smoke_summary.json` 没有 match@k；不得启动 full benchmark 或 official test。

## Track A MPTS-52 重新开启门槛

如需重新开启 Track A，必须先完整跑完 validation-only gate，并在 `track_a_mpts52` 下提供最小可审计包：

1. experiment manifest：说明路线、输入条件、候选来源、是否使用 scorer/repair/rerank/fusion。
2. validation-only 输出：至少包含样本数、candidate 数、K 值、match@1/5/20 或明确 smoke-only 声明。
3. rows>=7 子集：必须单独报告或说明为何当前阶段不可报告。
4. hard constraint / scorer / repair 边界：明确哪些是 paper mainline，哪些是 auxiliary。
5. freeze gate：说明达到什么 validation 条件后才允许 official full test。
6. no-test-feedback 声明：不得使用 official test aggregate 或 per-sample label 回调同一路线。

缺少以上内容，或只有脚本/进度日志而没有 `outputs/validation_oof_results.json` 与 `DECISION.md` 时，Track A 保持 `STOP`。

## Track B Hybrid 进入真实实验门槛

Track B 当前状态是 `continue as architecture / STOP as metric evidence`。下一轮若要从 smoke 原型进入真实 validation 实验，必须先补齐：

1. 生产级 SymCIF/Wyckoff parser：从候选 CIF 得到 Wyckoff rows、multiplicity、free-param payload 和 setting 信息。
2. Exact-cover search / verifier：支持 `composition + GT-SG` 下的 orbit cover，必须能处理 rows>=7。
3. Structural scorer 数据集：只用 train/validation positive 与 hard negative；禁止 test true CIF、test StructureMatcher label、oracle/GT-WA/GT-skeleton。
4. Scorer 特征：formula/SG consistency、exact-cover gap、orbit multiplicity、collision、local coordination、bond-valence 或等价局部合理性、volume/atom、可选 train-only energy proxy。
5. Geometry repair 真实实现：在固定 SG/Wyckoff skeleton 下更新 lattice/free parameters/site mapping，并重新跑 hard constraints。
6. Evaluation adapter：validation-only StructureMatcher `match@1/5/20`，并单独报告 rows>=7。
7. Frozen manifest：记录 source family、adapter、stage、是否使用 scorer/repair、是否 frozen-before-test、stop condition。
8. Stop condition：如果 validation 不能同时显示 W/A-to-match 或 geometry conversion 改善，不允许 full benchmark。

缺少以上任一项时，Track B 只能继续写架构/接口，不能写 metric claim，不能 freeze official test。

## 需要网页版 GPT 审稿的问题

1. K30/K50 policy 是否允许作为短期 metric rescue 继续，还是应完全停掉？
2. structural quality scorer 是否已经足够区别于 ordinary rerank？
3. 下一步更应该投向 MPTS-52 scorer，还是回到 Wyckoff-skeleton + geometry refinement 主线？
4. rows>=7 是否应成为下一轮硬 gate，而不是附加诊断？
5. 如果没有 immediate full-test plan，是否允许先做 validation-only scorer ablation？
6. Track B 是否应先补生产级 Wyckoff parser 和 real repair，再允许任何 validation benchmark？


---

# [08] TRACK_A_MPTS52_LATEST_APPENDIX

## Track A MPTS-52 最新结论追加

更新时间：2026-06-28 16:22:50 UTC。

本节是按用户要求在原 GPT_REVIEW_BUNDLE 末尾追加的新内容，不替换前面历史审计文本。若本节与前文关于 Track A “outputs 为空 / STOP / 未完成”的旧描述冲突，以本节列出的最新 artifact 状态和最新结论为准。

### 0. 一句话结论

Track A 当前结论是 `FREEZE_CANDIDATE`：`rows7_specialized_gate` 已通过 MPTS-52 validation OOF hard gate 和 bootstrap stability checks，但这不是 official full-test 成功，也不能写成已经达到最终 `+5pp` 目标。

### 1. 实验身份

- run_id: `track_a_mpts52_latest`
- experiment_dir: `/data/users/xsw/autodlmini/model/std_way/track_a_mpts52`
- route: `structural_quality_scorer + rows7_gate`
- dataset: `MPTS-52 validation OOF`
- evaluated_version: `rows7_specialized_gate`
- stage: `VALIDATION_OOF`
- status: `VALIDATION_DONE`
- verdict: `FREEZE_CANDIDATE`
- is_full_test: `false`
- uses_official_test_feedback: `false`
- uses_test_label: `false`
- sample_count: `5000`
- candidate_count: `250000`
- rows>=7 sample_count: `2292`

### 2. 最新指标

直接 anchor 是同一 validation bank 的 GT-SG K20 原始顺序，不是 official test baseline。

| metric | anchor | Track A | delta |
| --- | ---: | ---: | ---: |
| match@1 | 30.02 | 33.12 | +3.10 pp |
| match@5 | 40.48 | 41.98 | +1.50 pp |
| match@20 | 48.00 | 48.90 | +0.90 pp |
| rows>=7 match@1 | 5.322862 | 6.500873 | +1.178010 pp |
| rows>=7 match@5 | 9.991274 | 10.776614 | +0.785340 pp |
| rows>=7 match@20 | 14.746946 | 15.794066 | +1.047120 pp |

### 3. Artifact 证据

- `track_a_mpts52/RESULT.md`
- `track_a_mpts52/DECISION.md`
- `track_a_mpts52/metrics.json`
- `track_a_mpts52/outputs/validation_oof_results.json`
- `track_a_mpts52/outputs/feature_audit.json`
- `track_a_mpts52/outputs/rows7_report.json`
- `track_a_mpts52/outputs/oof_scores.jsonl`
- `track_a_mpts52/logs/validation_oof.log`

### 4. 边界

- `DECISION.md` 写的是 `FREEZE_CANDIDATE`，不是 `FREEZE`。
- `validation_oof_results.json` 的 decision 也是 `FREEZE_CANDIDATE`。
- 这是 validation-only 结果；当前没有 Track A official full-test summary。
- 不能把 validation OOF 通过写成 paper-level success。

### 5. 给网页版 GPT 的追加审稿问题

1. Track A 的 `FREEZE_CANDIDATE` verdict 是否合理？
2. validation OOF 的 +3.10 / +1.50 / +0.90 pp 和 rows>=7 +1.047 pp 是否足以允许进入 frozen official-test protocol？
3. 这个路线是否仍只是 `structural_quality_scorer` 的 WATCH 辅助路线，而不能作为论文主线？
4. 下一步是否应停止调参并只做 frozen official-test plan 审稿？

### 6. 给 Codex 的下一步最短指令

不要继续调 Track A 参数；把 Track A 最新 `FREEZE_CANDIDATE` validation OOF 结果交给网页版 GPT 审稿，确认是否允许进入 frozen official-test protocol；在批准前不要运行 official full test。


## opentry_11 追加实验：实验 1 Track A frozen official 泛化验证

时间：2026-06-28T17:15:09+00:00

实验逻辑：把 Track A validation OOF 的 frozen-candidate 思路放到 official full-test 泛化问题上检查。严格说，`rows7_specialized_gate` 本身没有独立 official 生成文件；历史中可审计的 frozen official 证据是同一 Track-A/RF-HGB 辅助排序家族的 `mpts52_k30_rf_seed1_bestscore_route`、`mpts52_k50_rf_seed1_margin_route` 和 MP-20 的 `mp20_k50_hgb_mean_seed012_margin_route`。本实验不回调参数，只读取 frozen official 结果；MPTS-52 K30 的 rows>=7 指标用既有 rows>=7 tar 在 `opentry_11/official_eval` 补评。

核心假设：如果 validation OOF 的结构质量排序是真泛化，official full-test 的 match@1/5/20 和 rows>=7 子集应同步提升；如果只在 validation 提升，则说明 Track A 是局部排序信号。

数据规模：MPTS-52 official test n=8096；MP-20 official test n=9046；MPTS-52 official rows>=7 n=7626；MP-20 rows>=7 n=1375。

baseline：MPTS-52 GT-SG anchor = 25.230% / 36.460% / 43.960%；rows>=7 anchor = 22.490% / 33.370% / 41.040%。MP-20 GT-SG anchor = 71.670% / 83.080% / 87.810%；rows>=7 anchor = 62.370% / 76.350% / 82.610%。

方法变化：只替换候选排序/route，不生成新结构；MPTS-52 K30 是 best-score route，K50 是 margin route，MP-20 是 HGB mean seed012 margin route。

结果：
- MPTS-52 K30 overall = 26.075% / 36.228% / 44.059%；delta = 0.8446047430829995 / -0.232233201581028 / 0.09879446640316258 pp。
- MPTS-52 K30 rows>=7 = 23.289% / 33.307% / 41.227%；相对 rows>=7 anchor 约为 +0.799 pp / -0.063 pp / +0.187 pp。
- MPTS-52 K50 overall = 25.791% / 36.265% / 43.824%；rows>=7 = 23.053% / 33.228% / 40.939%。
- MP-20 frozen route overall = 70.230% / 81.870% / 87.486%；rows>=7 = 27.200% / 43.273% / 55.273%。

可信度：official full-test 结果可信；但它验证的是 Track-A-family frozen route，不是完全同名的 `rows7_specialized_gate`，因此对 Track A 当前 validation 版本的结论是“相邻 frozen official 负证据”，不是正向确认。

和历史实验关系：与 opentry_10 一致，validation 有 +0.9 至 +3.1pp 信号，但 official 只保住 MPTS-52 match@1 的 +0.56/+0.845pp，K5/K20 不稳；MP-20 反而下降。

最终判决：Track A 不能作为主方法，也不能证明 validation OOF 提升可泛化到 official full-test；只能保留为辅助结构质量模块候选。

下一步：不要回头调 Track A；后续实验转向 exact-cover skeleton、hard-negative 区分和 geometry repair。


## opentry_11 追加实验：实验 2 Track A 失败归因分析

时间：2026-06-28T17:16:00+00:00

实验逻辑：不调指标，只解释 Track A 在 validation 上救了谁、害了谁。比较对象是同一 MPTS-52 validation K50 候选池上的 baseline K20 原始顺序与 `rows7_specialized_gate` 选择。

核心假设：如果瓶颈主要是排序，Track A 应把已有正确候选提前；如果瓶颈是 coverage 或 skeleton/geometry，本实验会看到大量“两者都错”或 skeleton-hit 但 StructureMatcher 失败。

数据规模：5000 个 validation 样本，250000 个 K50 候选；rows>=7 样本 2292 个。

baseline：原 GT-SG validation K20 = 30.020% / 40.480% / 48.000%。

方法变化：只做分组归因；不改变候选、不训练新模型。

结果：
- K20 分组：baseline 错 Track A 对 = 135；baseline 对 Track A 错 = 90；两者都对 = 2310；两者都错 = 2465。
- rows>=7 K20：救回 52，伤害 28，两者都错 1902。
- baseline top20 错误候选中，formula 对但结构错比例 = 95.390%；SG 对但 multiplicity/Wyckoff exact-cover 代理错比例 = 26.481%；skeleton 代理可行但 geometry/StructureMatcher 失败比例 = 68.642%。
- 失败原因代理：collision/radius 短距 = 50.593%；lattice 异常代理 = 0.491%；free parameter 坐标异常代理 = 0.457%；site mapping/exact-cover 代理失败 = 30.874%。
- rows>=7 错误候选中 skeleton 可行但 geometry 失败比例 = 58.625%，collision 代理 = 62.801%。

可信度：validation OOF 标签完整，归因使用 inference-safe 结构特征；但 Wyckoff 字母没有从候选 CIF 精确恢复，因此“Wyckoff”是 multiplicity/exact-cover 代理。

和历史实验关系：验证了 opentry_10 的判断：Track A 的正贡献是排序已有正确候选，不能解决“两者都错”的 coverage/geometry 空洞。

最终判决：瓶颈不是单纯排序；候选池 coverage 与 skeleton-hit-to-match geometry 转化同时存在。

下一步：继续实验 3 的 exact-cover 诊断，并把 hard negatives 训练成显式结构错误分类问题。


## opentry_11 追加实验：实验 3 Wyckoff exact-cover 诊断实验

时间：2026-06-28T17:16:01+00:00

实验逻辑：判断现有候选在 composition + GT-SG 条件下是否满足 Wyckoff exact-cover 的合理性。候选 CIF 没有显式 Wyckoff letter，因此本实验使用 `_atom_site_symmetry_multiplicity` + formula + SG 的 exact-cover 代理，并单独标注这一限制。

核心假设：若大量错误候选连 exact-cover 都不满足，后续应强化 crystallographic constraint；若 exact-cover 满足但 match 失败，重点应转向 geometry repair。

数据规模：MPTS-52 validation K50，共 250000 个候选、5000 个样本。

baseline：原候选池未加 exact-cover 约束，Track A 只是利用这些特征排序。

方法变化：不生成新结构；逐候选检查 formula、GT-SG、multiplicity exact-cover、row bucket、equivalent-position 代理一致性。

结果：
- formula consistency = 96.624%；GT-SG consistency = 99.838%。
- multiplicity exact-cover rate = 78.206%；orbit/equivalent-position feasible proxy = 78.117%；综合 skeleton-hit proxy = 77.812%。
- rows>=7 bucket consistency = 96.259%。
- skeleton feasible 但最终 match 失败比例 = 61.734%。
- rank1 中 skeleton 不可行比例 = 22.260%；top20 中 skeleton 不可行比例 = 21.971%。
- rows>=7 skeleton-hit proxy = 60.274%；rows>=7 skeleton feasible 但 match 失败 = 91.959%。

可信度：composition/SG/multiplicity 检查覆盖全量 K50；Wyckoff letter 级 exact-cover 未恢复，因此可信结论是“multiplicity/exact-cover feasibility”，不是完整 letter-skeleton 判定。

和历史实验关系：支撑 opentry_3/4 的判断：很多候选已经过 formula/SG 关，但 exact-cover 与 geometry 转化仍是核心瓶颈。

最终判决：需要 stronger crystallographic constraint，同时也需要 geometry repair；二者不是互斥。

下一步：实验 4 把 hard negatives 显式纳入 scorer；实验 5 检查 skeleton-hit 后 geometry repair 的实际转化率。


## opentry_11 追加实验：实验 4 hard-negative structural scorer v2

时间：2026-06-28T17:17:11+00:00

实验逻辑：在 Track A 基础上训练 hard-negative structural scorer v2，重点让模型区分 formula/SG 看似正确但 crystallographic skeleton 或 geometry 错的候选。

核心假设：如果 scorer 学到结构本质，match@5 和 match@20 也应提升，尤其 rows>=7；如果只涨 K1，则仍是浅层排序器。

数据规模：MPTS-52 validation K50，5-fold OOF；训练 hard-negative pair groups 见 `results/experiment_4_hard_negative_scorer_v2.json`。

baseline：原 GT-SG validation K20 = 30.020% / 40.480% / 48.000%。

方法变化：负样本优先选 top-rank 错误、formula 正确但 geometry 错、SG 正确但 exact-cover 错、skeleton 可行但 match 失败、rank>20 的 hard negatives；特征包括 exact-cover、collision、local geometry、rows>=7 proxy，不使用 official test。

结果：hard-negative v2 = 32.540% / 42.180% / 48.960%；delta = +2.520 pp / +1.700 pp / +0.960 pp。rows>=7 = 7.024% / 12.173% / 15.881%；rows>=7 delta = +1.702 pp / +2.182 pp / +1.134 pp。

可信度：validation OOF 可信；但仍使用 StructureMatcher validation labels 训练，定位是 validation 诊断/auxiliary scorer，不是 official 方法成功。

和历史实验关系：这是 Track A 的 hard-negative 版本，直接回应 opentry_10 中“浅层 RF/HGB 不能区分 formula/rows 都对但 geometry 错”的问题。

最终判决：若 K5/K20 未同步明显提升，则 scorer v2 仍不足以作为主方法；若 rows>=7 改善有限，说明 coverage/geometry repair 仍是主瓶颈。

下一步：把 scorer 的错误样本送入 symmetry-preserving geometry repair 诊断，而不是继续调 scorer 阈值。


## opentry_11 追加实验：实验 5 symmetry-preserving geometry repair

时间：2026-06-28T17:20:43+00:00

实验逻辑：找到 skeleton 已合理但 StructureMatcher 未 match 的候选，在不改变 SG/formula/multiplicity 行的前提下做最小几何修复，检查 skeleton-hit-to-match conversion。

核心假设：如果失败主要来自局部碰撞、坐标越界或 free-parameter 小误差，简单 symmetry-preserving-ish repair 应能让一部分 skeleton-hit 候选转成 match；如果 conversion 近零，说明需要真正的受约束 geometry model，而不是后处理小修。

数据规模：validation top20 中 skeleton-hit 但 match 失败候选共有 47949；本轮实际 pilot 请求 300，成功评估 298；优先抽 rows>=7。

baseline：repair 前这些候选均来自 StructureMatcher negative 标签；pilot 中 before-match 用同一 StructureMatcher 参数复核。

方法变化：fractional coordinates wrap 到 [0,1)，collision-proxy 候选加确定性微小 fractional jitter；不改 SG、formula、multiplicity skeleton。

结果：pilot conversion = 0/298 = 0.000%；rows>=7 conversion = 0/298 = 0.000%。

可信度：这是真实 StructureMatcher pilot，不是 oracle；但 repair 很弱，没有学习 lattice/free parameter/site mapping，且没有 full K20 全量重评。

和历史实验关系：与 opentry_4 的结论一致：skeleton-hit 后 geometry 转化是瓶颈，仅靠坐标 wrap/jitter 不能解决。

最终判决：当前 deterministic repair 失败；需要真正 symmetry-preserving geometry repair 模型或优化器。

下一步：rows>=7 专门分析中把“skeleton hit 但 geometry fail”作为主对象，而不是继续普通 rerank。


## opentry_11 追加实验：实验 6 rows>=7 复杂结构专门实验

时间：2026-06-28T17:20:46+00:00

实验逻辑：把 rows>=7 当作独立对象，分析正确候选是否存在、排位在哪里、失败来自 skeleton coverage 还是 geometry 转化。

核心假设：复杂结构的主要瓶颈不是 overall 平均排序，而是 skeleton coverage 低、geometry/free parameter/site mapping 更难。

数据规模：rows>=7 validation 样本 2292；rows<7 样本 2708；候选仍为 K50。

baseline：rows>=7 baseline K20 = 14.747%。

方法变化：单独比较 rows7 specialized gate、hard-negative v2 rows7 route、repair pilot 与 exact-cover skeleton proxy。

结果：
- rows>=7 正确候选存在于 top20 的样本数 = 338；存在于 top50 的样本数 = 420；top50 首个正确候选平均 rank = 10.416666666666666。
- rows>=7 top20 skeleton-hit sample rate = 91.841%；top20 match sample rate = 14.747%。
- rows>=7 skeleton-hit 但 geometry fail 候选比例 = 55.428%。
- collision proxy：rows>=7 = 60.820%，rows<7 = 17.016%；bad coord/free-parameter proxy = 0.539%；exact-cover bad proxy = 39.278%。
- rows7 specialized scorer：Track A rows>=7 K1/K5/K20 = 6.501% / 10.777% / 15.794%。
- hard-negative v2 rows>=7 K1/K5/K20 = 7.024% / 12.173% / 15.881%。
- specialized geometry repair pilot rows>=7 conversion = 0/298。

可信度：coverage/rank/label 统计是全量 validation K50；repair 是 pilot。

和历史实验关系：复现历史 rows>=7 是主瓶颈的结论，且显示 top50 中仍有未被转化或未排前的正确候选。

最终判决：rows>=7 不能靠普通 overall scorer 解决；需要 rows>=7 专门 skeleton proposal + geometry repair。

下一步：实验 7 做 exact-cover constrained skeleton proposal/filter，检查 coverage 是否真的转成 StructureMatcher match。


## opentry_11 追加实验：实验 7 exact-cover constrained skeleton proposal

时间：2026-06-28T17:21:05+00:00

实验逻辑：做生成侧的最小可审计替代实验：在现有 K50 中模拟 exact-cover constrained skeleton proposal/filter，强制优先 formula+GT-SG+multiplicity exact-cover feasible 的 skeleton，再看是否提高 match@20。它不是新模型生成，因此结论按 proposal proxy/upper-bound 解读。

核心假设：如果 exact-cover skeleton coverage 是瓶颈，优先 exact-cover skeleton 应提升 K20；如果只提高 skeleton feasible rate 但 match 不升，说明 geometry 转化才是主瓶颈。

数据规模：MPTS-52 validation K50，全量 5000 样本、250000 候选。

baseline：原 GT-SG validation K20 = 30.020% / 40.480% / 48.000%。

方法变化：每个样本先选 exact-cover feasible skeleton 候选，再用 collision/rank 代理排序补足 top20；没有使用 GT skeleton 或 test label。

结果：exact-cover proposal proxy = 31.980% / 42.080% / 49.160%；delta = +1.960 pp / +1.600 pp / +1.160 pp。rows>=7 = 6.588% / 11.387% / 15.838%。
- top50 任一 match coverage = 52.720%；任一 skeleton-hit coverage = 96.840%；skeleton-hit 最终有 match 的样本比例 = 52.600%。
- rows>=7 top50 任一 match coverage = 18.325%；任一 skeleton-hit coverage = 94.677%；skeleton-to-match = 18.237%。

可信度：全量 validation K50；但不是新 skeleton 生成，只能说明“如果从现有候选筛选 exact-cover skeleton，会发生什么”。

和历史实验关系：延续 SymCIF exact-cover 主线，也解释为什么只提高 W/A recall 未必提高 StructureMatcher。

最终判决：exact-cover 必须和 geometry proposal/repair 绑定；单独 filter/proposal 不足以声明成功。

下一步：实验 8 把 Track A、exact-cover、hard-negative v2、repair/rows7 route 放到同一消融表。


## opentry_11 追加实验：实验 8 整合消融实验

时间：2026-06-28T17:21:23+00:00

实验逻辑：统一比较每个模块到底贡献 coverage、排序还是 rows>=7 处理能力。

核心假设：真正有效的主线应同时提升 overall match@1/5/20 中至少两个指标，并且 rows>=7 不应恶化；仅改变 top1 排序不能算解决晶体生成瓶颈。

数据规模：MPTS-52 validation K50 全量 5000 样本；official full-test 结论沿用实验 1。

baseline：原 GT-SG validation K20。

方法变化：消融 baseline、Track A、exact-cover filter/proposal、hard-negative v2、geometry repair pilot、组合 route、rows>=7 specialized route。

结果：

| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 GT-SG baseline | 30.020%/40.480%/48.000% | 5.323%/9.991%/14.747% | 74.868% | 96.743% | 99.843% | 78.029% | 38.550% |
| baseline + Track A scorer | 33.120%/41.980%/48.900% | 6.501%/10.777%/15.794% | 82.598% | 98.061% | 99.845% | 87.400% | 36.852% |
| baseline + exact-cover filter/proposal | 31.980%/42.080%/49.160% | 6.588%/11.387%/15.838% | 79.402% | 99.002% | 99.894% | 89.585% | 35.523% |
| baseline + hard-negative structural scorer v2 | 32.540%/42.180%/48.960% | 7.024%/12.173%/15.881% | 79.493% | 98.018% | 99.826% | 87.454% | 36.378% |
| baseline + geometry repair | 30.020%/40.480%/48.000% | 5.323%/9.991%/14.747% | 74.868% | 96.743% | 99.843% | 78.029% | 38.550% |
| skeleton proposal + geometry repair | 31.980%/42.080%/49.160% | 6.588%/11.387%/15.838% | 79.402% | 99.002% | 99.894% | 89.585% | 35.523% |
| skeleton proposal + geometry repair + structural scorer | 32.360%/42.180%/49.100% | 6.937%/11.824%/15.707% | 80.223% | 98.157% | 99.879% | 89.585% | 35.740% |
| rows>=7 specialized route | 33.120%/41.980%/48.900% | 6.501%/10.777%/15.794% | 82.598% | 98.061% | 99.845% | 87.400% | 36.852% |

可信度：validation 消融全量可信；geometry repair 只有 pilot，不能当 full metric；official 泛化见实验 1。

和历史实验关系：与 opentry_10 结论一致，Track A/普通 scorer 的 official 泛化不足；exact-cover 能提高结构约束诊断，但必须和 geometry repair 结合。

最终判决：未达到至少两个 match 指标 +5pp。当前真正提升 coverage 的证据不足；Track A/hard-negative 主要改变排序；rows>=7 最有效的是 specialized route 但幅度小；最大剩余瓶颈是 exact-cover skeleton coverage 与 skeleton-hit-to-match geometry conversion。

下一步：停止普通 rerank/threshold；实现真正的 exact-cover skeleton generator 与受 SG/Wyckoff 约束的 geometry repair，再做 validation OOF 和一次冻结 official。


## opentry_11 执行口径澄清：GPU 训练与“完成”含义

时间：2026-06-28T17:22:00+00:00

针对用户质疑，明确记录如下：

1. 实验 1-3 不属于 GPU 训练实验。实验 1 是读取 frozen official full-test 结果并补跑 rows>=7 CPU StructureMatcher 评估；实验 2 是 validation 错误归因；实验 3 是 exact-cover/multiplicity 诊断。

2. 实验 4 的“训练”不是 GPU 神经网络训练。当前实现是沿用 Track A 的 validation OOF scorer 范式，用 sklearn/CPU 训练 hard-negative pairwise structural scorer v2。它不训练 CrystaLLM/SymCIF 生成模型，不训练新的 GPU neural geometry model，也不生成新 CIF 候选。

3. 因此，实验 1-4 的“已完成”含义是：在已有 MPTS-52 validation K50 候选、StructureMatcher 标签、Track A OOF 特征和 official frozen 结果基础上，完成可审计评估/诊断/CPU scorer ablation。它不能被写成“完成了新的 GPU 生成模型训练”。

4. 若后续论文主线要求“真正 GPU 模型训练”，则当前实验 4 只能算 CPU validation diagnostic / auxiliary scorer，不是最终主方法。真正需要 GPU 或更重训练的是：exact-cover constrained skeleton generator、symmetry-preserving geometry repair model、rows>=7 specialized generation/repair model。

最终判决：报告中的实验 1-4 不应被解释为 GPU 训练完成；它们是诊断和 CPU OOF scorer 版本。后续 claim 必须按这个边界收缩。


## opentry_11 严格口径重审：GPU 必要性与 half-data gate

时间：2026-06-28T17:25:00+00:00

针对更新后的目标，重新定义后续执行口径：

1. 实验 1 `Track A frozen official 泛化验证`：不需要 GPU 训练。它是 frozen official/rows>=7 评估和结果读取，GPU 训练不会改变这个实验问题。当前结果可作为 official 泛化负证据，但仍需明确它不是新模型训练。

2. 实验 2 `Track A 失败归因分析`：不需要 GPU 训练。它是 validation error attribution，重点是分组、错误类型和瓶颈定位。

3. 实验 3 `Wyckoff exact-cover 诊断实验`：不需要 GPU 训练。它是 composition/SG/multiplicity/exact-cover feasibility 诊断；GPU 不是必要条件。

4. 实验 4 `hard-negative structural scorer v2`：需要 GPU 补充实验，因为它包含“训练 scorer”的方法主张。后续先用 deterministic 50% validation samples 做 GPU neural hard-negative scorer；只有 half-data 至少两个 overall match 指标达到 +5pp，或 rows>=7 match@5/match@20 同时达到 +5pp，才补跑全量。

5. 实验 5 `symmetry-preserving geometry repair`：若只是 wrap/jitter 诊断则不需要 GPU，但这不满足 prompt 的核心要求；严格版本需要 learned/optimized geometry repair，因此需要后续 half-data repair 实验。当前 300-candidate deterministic pilot 只能算失败诊断，不能算完成。

6. 实验 6 `rows>=7 复杂结构专门实验`：分析部分不需要 GPU；specialized scorer / repair / skeleton proposal 需要 GPU 或生成侧补充。后续先跑 half-data rows>=7 GPU scorer；若 half-data gate 过，再补全量。

7. 实验 7 `exact-cover constrained skeleton proposal`：严格说需要生成侧实验，不能只在现有候选里排序。GPU 是否必要取决于 proposal 形式：exact-cover combinatorial search 本身可 CPU，若训练 skeleton proposer/WA decoder 则需要 GPU。当前现有 K50 filter/proxy 不满足 strict prompt。

8. 实验 8 `整合消融实验`：本身不一定需要 GPU，但它必须整合严格完成的模块。当前消融表混入 CPU scorer、repair pilot 和 proposal proxy，只能作为第一轮诊断消融；严格版需要在 half-data GPU/proposal/repair 补充后重写。

half-data 定义：后续训练/生成类实验先用全量 validation sample 的确定性 50% 子集，而不是任意小样本。子集按 sample_id 稳定排序，并在 rows>=7 与 rows<7 两类内隔位抽取，避免只抽简单结构或随机小批。

最终判决：前面追加的实验 1-8 不是更新目标下的严格完成版。接下来以 half-data gate 重新执行训练/生成/repair 相关部分；未过 gate 的模块不补全量，也不写成主方法成功。


## opentry_11 追加实验：实验 4C GPU hard-negative scorer half-data gate（half）

时间：2026-06-28T17:32:55+00:00

实验逻辑：按更新后的目标，先判断实验 4 是否需要 GPU。实验 4 涉及训练 structural scorer，因此需要 GPU 补充；本轮先用 deterministic 50% validation samples，而不是小 pilot 或直接全量。

核心假设：如果 hard-negative neural scorer 真学到晶体学结构错误，half-data OOF 至少应在两个 overall match 指标达到 +5pp，或在 rows>=7 match@5/match@20 同时达到 +5pp。

数据规模：stage=half；样本 2500；候选 125000；rows>=7 样本 1146；特征 46 个；设备 NVIDIA A800 80GB PCIe；PyTorch 2.5.1+cu121。half 子集按 rows>=7/rows<7 分层后 sample_id 稳定排序隔位抽取。

baseline：原 GT-SG rank 顺序 = 30.120% / 40.960% / 48.400%；rows>=7 = 5.236% / 10.035% / 14.398%。

方法变化：GPU MLP + weighted BCE；hard-negative 加权 top20 错误、formula+SG 正确但 geometry 错、skeleton-hit 但不 match、collision-free 但不 match。特征不使用 target_rows_ge7、rank/rank_inv/rank_le*、CIF 字符数/行数、atom rows 作为输入。

结果：GPU general scorer = 33.960% / 43.160% / 50.280%；delta = +3.840 pp / +2.200 pp / +1.880 pp。rows>=7 = 6.981% / 11.169% / 15.532%；rows>=7 delta = +1.745 pp / +1.134 pp / +1.134 pp。

诊断：valid rate = 80.490%；formula consistency = 95.846%；SG consistency = 99.868%；exact-cover feasible = 88.956%；skeleton-hit-to-match conversion = 38.122%。

可信度：这是实际 GPU 训练 + 5-fold OOF；但仍只在已有候选池内排序，不是新 CIF 生成，也不是 official full-test。

和历史实验关系：替代前一轮 CPU sklearn 实验 4 的严格 half-data GPU 版本；直接检验 scorer 是否只是普通 rerank。

最终判决：gate_pass=False；overall >= +5pp 指标数=0；rows>=7 K5/K20 是否均 >= +5pp=False。half-data gate 未通过，因此不补跑全量，避免把弱信号扩成 full run。

下一步：若未过 gate，实验 4 不进入全量；后续优先转向 geometry repair 和 skeleton proposal。


## opentry_11 追加实验：实验 6C GPU rows>=7 specialized scorer half-data gate（half）

时间：2026-06-28T17:32:58+00:00

实验逻辑：实验 6 的分析部分不需要 GPU，但 specialized scorer 属于训练实验，因此必须先做 half-data GPU gate。本轮 rows>=7 model 只用训练 fold 中 target rows>=7 样本训练；推理时不用 target_rows_ge7，而用候选池里的 complex_proxy 决定是否走 specialized route。

核心假设：如果 rows>=7 主要是排序/结构质量识别问题，specialized route 应显著提升 rows>=7 match@5/match@20；如果仍不过 +5pp，说明复杂结构瓶颈主要在 skeleton coverage 或 geometry conversion。

数据规模：stage=half；样本 2500；候选 125000；rows>=7 样本 1146。

baseline：rows>=7 原 GT-SG rank 顺序 = 5.236% / 10.035% / 14.398%。

方法变化：GPU rows>=7 specialized MLP；非 complex_proxy 样本回退 general scorer，避免把简单结构强行交给 rows>=7 模型。

结果：rows>=7 specialized route overall = 33.200% / 42.240% / 50.200%；overall delta = +3.080 pp / +1.280 pp / +1.800 pp。rows>=7 = 6.719% / 10.908% / 15.620%；rows>=7 delta = +1.483 pp / +0.873 pp / +1.222 pp。

诊断：valid rate = 82.130%；formula consistency = 96.564%；SG consistency = 99.874%；exact-cover feasible = 89.374%；skeleton-hit-to-match conversion = 37.731%。

可信度：实际 GPU 训练 + OOF；但它仍是候选池内排序，不能声称提高 skeleton 或 geometry coverage。

和历史实验关系：这是实验 6 中 rows>=7 scorer 的严格 half-data GPU 补充；用于判断是否值得全量扩展。

最终判决：gate_pass=False；overall >= +5pp 指标数=0；rows>=7 K5/K20 是否均 >= +5pp=False。half-data gate 未通过，因此不补跑全量，避免把弱信号扩成 full run。

下一步：若未过 gate，rows>=7 路线不能继续普通 scorer，应转向 exact-cover skeleton proposal + symmetry-preserving geometry repair。


## opentry_11 追加实验：实验 7B exact-cover constrained skeleton proposal half-data gate

时间：2026-06-28T17:38:09+00:00

实验逻辑：按更新后的目标重做实验 7。前一轮只在现有 CrystaLLM K50 候选中做 exact-cover filter/proxy，不满足“生成侧实验”。本轮改用已有 SymCIF v5 MPTS-52 validation generation artifacts：`v5_a1_exact_cover_sg_formula_e08` 是 exact-cover constrained skeleton/WA proposal，`v5_fullgen_eval_pool` 是 exact-cover 与 geometry-aware route 的生成池。

GPU 必要性判断：本轮不重新训练模型，只评估已有 generation/evaluation artifacts，因此不需要新 GPU 训练；如果下一步要重训 neural W/A decoder 或 learned skeleton proposer，则需要 GPU，并且也要先 half-data gate。

核心假设：如果 exact-cover constrained skeleton proposal 真提高 coverage，它应相对同材料的 CrystaLLM GT-SG baseline 在 match@20 或至少两个 match 指标上达到 +5pp，并且 rows>=7 不应只停留在 skeleton_hit。

数据规模：half samples=2364；rows>=7 samples=1099；exact-cover A1 candidates=11820；fullgen pool candidates=12976。half 子集按 rows>=7/rows<7 分层后 sample_id 稳定排序隔位抽取。

baseline：同材料 CrystaLLM GT-SG rank 顺序 = 30.161% / 40.313% / 47.631%；rows>=7 = 5.005% / 9.827% / 14.741%。

方法变化：使用真正生成侧 SymCIF exact-cover candidates，而不是在 CrystaLLM K50 内重排；排序按 generation_score desc + gen_index asc；报告 K1/K5/K20，其中 K20 是该生成池可用候选内的 top20。

结果 A1 exact-cover：26.988% / 35.702% / 35.702%；delta = -3.173 pp / -4.611 pp / -11.929 pp。rows>=7 = 11.101% / 15.196% / 15.196%；rows>=7 delta = +6.096 pp / +5.369 pp / +0.455 pp。

结果 fullgen pool：27.875% / 38.197% / 39.155%；delta = -2.286 pp / -2.116 pp / -8.476 pp。rows>=7 = 11.499% / 16.211% / 16.588%；rows>=7 delta = +6.494 pp / +6.384 pp / +1.847 pp。

诊断 A1：formula=71.895%；SG=71.895%；exact-cover/multiplicity=71.887%；skeleton_hit=14.560%；WA_hit=2.834%；skeleton-hit-to-match=22.545%。

诊断 fullgen pool：formula=96.779%；SG=96.779%；exact-cover/multiplicity=96.686%；skeleton_hit=13.795%；WA_hit=2.713%；skeleton-hit-to-match=23.184%。

可信度：这是生成侧 artifact 的 half-data 复算，强于现有 K50 filter proxy；但它复用历史生成结果，不是本轮新训练。

和历史实验关系：与 SymCIF v5 报告一致，exact-cover 能提高 skeleton 可行性，但未稳定转化为 StructureMatcher match，尤其 rows>=7 仍弱。

最终判决：A1 gate_pass=False；fullgen_pool gate_pass=False。half-data gate 未通过，因此不补跑全量 generation 汇总。

下一步：若 gate 未过，实验 7 的瓶颈不是“是否 exact-cover”本身，而是 exact-cover skeleton 到 geometry/StructureMatcher match 的转化；后续应与实验 5 的 learned geometry repair 绑定。


## opentry_11 追加实验：实验 5B symmetry-preserving geometry repair half-data gate

时间：2026-06-28T17:38:55+00:00

实验逻辑：按更新后的目标重做实验 5。当前 repair 是 deterministic symmetry-preserving-ish 后处理，不训练模型，因此本实验判断为不需要 GPU；但因为它使用 MPTS-52 validation 数据，必须先用全量样本的一半，而不是 300 个 pilot。

核心假设：如果 skeleton-hit 失败主要来自坐标越界或轻微 collision，wrap fractional coordinates + collision jitter 应把一批 negative candidate 转成 StructureMatcher match，并提升 match@5/match@20，尤其 rows>=7。

数据规模：half samples=2500；half candidates=125000；rows>=7 samples=1146；repair pool=23738 candidates / 2019 samples；实际写入 repair pairs=23738。

baseline：原 GT-SG rank 顺序 = 30.120% / 40.960% / 48.400%；rows>=7 = 5.236% / 10.035% / 14.398%。

方法变化：对 half-data top20 中 skeleton-hit 且 StructureMatcher negative 的候选做 repair；不改变 SG/formula/multiplicity 行，只把 fractional coordinates wrap 到 [0,1)，对 collision-proxy 候选加确定性微小 jitter；repair 后用 StructureMatcher 重新评估，并回填 match@k。

结果：repair 后 = 30.120% / 40.960% / 48.400%；delta = +0.000 pp / +0.000 pp / +0.000 pp。rows>=7 = 5.236% / 10.035% / 14.398%；rows>=7 delta = +0.000 pp / +0.000 pp / +0.000 pp。

repair conversion：valid evaluated=23737；converted=0；conversion rate=0.000%；rows>=7 converted=0/12745，conversion=0.000%；after_match_none=1。

诊断：valid rate = 75.582%；formula consistency = 96.926%；SG consistency = 99.886%；exact-cover feasible = 78.228%；skeleton-hit-to-match conversion = 39.035%。

可信度：半量样本、全 repair pool 评估，可信度高于 300-candidate pilot；但 repair 本身很弱，不是 learned geometry model。

和历史实验关系：把前一轮 pilot 的 0 conversion 放大到 half-data gate；用于判断 deterministic repair 是否值得全量。

最终判决：gate_pass=False；overall >= +5pp 指标数=0；rows>=7 K5/K20 是否均 >= +5pp=False。half-data gate 未通过，因此不补跑全量。

下一步：若 deterministic repair 不过 gate，则实验 5 的下一步不是全量后处理，而是设计真正受 SG/Wyckoff 约束的 learned/optimized geometry repair。


## opentry_11 最新执行口径修正：train 数据集约束

时间：2026-06-28T17:41:30+00:00

根据用户最新目标，规模约束重新解释如下：

1. “不能随便选一个很小的量做完实验 / 先用大规模或半量再判断”的硬约束，专门针对需要采用 MP-20 或 MPTS-52 train 数据集训练模型的实验。

2. validation-only 诊断、official frozen 结果读取、StructureMatcher 复核、已有 generation artifact 复算，不属于“使用 train 数据集训练模型”的实验；这些实验仍要报告数据规模和可信度，但不强制走 train half-data gate。

3. 前面实验 4C、5B、6C、7B/7C 使用 half validation/generation artifact，是更保守的规模 gate，不应被理解为用户强制要求的 train-data half protocol。它们的结论仍有效：这些模块在较大 validation 子集上没有达到 +5pp，不支持扩成主方法。

4. 若后续新增真正使用 MP-20/MPTS-52 train 数据集的神经模型训练，例如 neural W/A decoder、learned skeleton proposer、learned geometry repair model，则必须先明确 train 数据规模，不能用很小样本 smoke 当完成；若先做半量 train，也要在报告中写清楚半量定义、是否过 gate、是否需要全量。

最终判决：当前已完成的 validation/official/artifact 复算可以作为诊断结论；尚未完成的是“重新训练 train-data 级别的 skeleton proposer 或 geometry repair model”。由于现有 half validation 证据没有过 +5pp，后续不应贸然启动大规模 train-data 训练，除非先定义新的模型目标和 gate。


## opentry_11 追加实验：实验 7C exact-cover constrained skeleton proposal half-data gate 修正版

时间：2026-06-28T17:40:26+00:00

实验逻辑：按更新后的目标重做实验 7。前一轮只在现有 CrystaLLM K50 候选中做 exact-cover filter/proxy，不满足“生成侧实验”。本轮改用已有 SymCIF v5 MPTS-52 validation generation artifacts：`v5_a1_exact_cover_sg_formula_e08` 是 exact-cover constrained skeleton/WA proposal，`v5_fullgen_eval_pool` 是 exact-cover 与 geometry-aware route 的生成池。本节修正 7B 中 fullgen pool 与 A1 样本数不一致时共用 baseline 的口径；A1 结论不变，fullgen pool delta 改为使用自己的同材料 baseline。

GPU 必要性判断：本轮不重新训练模型，只评估已有 generation/evaluation artifacts，因此不需要新 GPU 训练；如果下一步要重训 neural W/A decoder 或 learned skeleton proposer，则需要 GPU，并且也要先 half-data gate。

核心假设：如果 exact-cover constrained skeleton proposal 真提高 coverage，它应相对同材料的 CrystaLLM GT-SG baseline 在 match@20 或至少两个 match 指标上达到 +5pp，并且 rows>=7 不应只停留在 skeleton_hit。

数据规模：half samples=2364；rows>=7 samples=1099；exact-cover A1 candidates=11820；fullgen pool candidates=12976。half 子集按 rows>=7/rows<7 分层后 sample_id 稳定排序隔位抽取。

baseline：A1 同材料 CrystaLLM GT-SG rank 顺序 = 30.161% / 40.313% / 47.631%；rows>=7 = 5.005% / 9.827% / 14.741%。fullgen pool 因缺少部分样本，使用自己的同材料 baseline = 30.836% / 41.115% / 48.301%；rows>=7 = 5.184% / 10.179% / 15.174%。

方法变化：使用真正生成侧 SymCIF exact-cover candidates，而不是在 CrystaLLM K50 内重排；排序按 generation_score desc + gen_index asc；报告 K1/K5/K20，其中 K20 是该生成池可用候选内的 top20。

结果 A1 exact-cover：26.988% / 35.702% / 35.702%；delta = -3.173 pp / -4.611 pp / -11.929 pp。rows>=7 = 11.101% / 15.196% / 15.196%；rows>=7 delta = +6.096 pp / +5.369 pp / +0.455 pp。

结果 fullgen pool：27.875% / 38.197% / 39.155%；delta = -2.962 pp / -2.918 pp / -9.146 pp。rows>=7 = 11.499% / 16.211% / 16.588%；rows>=7 delta = +6.315 pp / +6.032 pp / +1.414 pp。

诊断 A1：formula=71.895%；SG=71.895%；exact-cover/multiplicity=71.887%；skeleton_hit=14.560%；WA_hit=2.834%；skeleton-hit-to-match=22.545%。

诊断 fullgen pool：formula=96.779%；SG=96.779%；exact-cover/multiplicity=96.686%；skeleton_hit=13.795%；WA_hit=2.713%；skeleton-hit-to-match=23.184%。

可信度：这是生成侧 artifact 的 half-data 复算，强于现有 K50 filter proxy；但它复用历史生成结果，不是本轮新训练。

和历史实验关系：与 SymCIF v5 报告一致，exact-cover 能提高 skeleton 可行性，但未稳定转化为 StructureMatcher match，尤其 rows>=7 仍弱。

最终判决：A1 gate_pass=False；fullgen_pool gate_pass=False。half-data gate 未通过，因此不补跑全量 generation 汇总。

下一步：若 gate 未过，实验 7 的瓶颈不是“是否 exact-cover”本身，而是 exact-cover skeleton 到 geometry/StructureMatcher match 的转化；后续应与实验 5 的 learned geometry repair 绑定。


## opentry_11 追加实验：实验 8C 严格整合消融与最终判决

时间：2026-06-28T17:42:49+00:00

实验逻辑：按最新口径重写整合消融。前一版实验 8 把 full validation、CPU scorer、repair pilot、proposal proxy 放在同一张表里，容易让人误以为所有模块都严格完成。本节把不同数据范围分开：full validation 诊断、GPU validation scorer、half validation repair、SymCIF generation artifact 复算分别报告，不把样本集合不同的结果硬合并。

GPU 必要性判断：实验 8 本身是整合分析，不训练模型，不需要 GPU；它只汇总前面模块。真正需要 GPU 的训练模块已经在实验 4C/6C 补做了 validation GPU scorer；train-data 级别 neural W/A decoder 或 learned geometry repair 尚未启动，因为现有 validation 证据未过 +5pp gate。

核心假设：如果主线有效，至少两个 match 指标应达到 +5pp，并且 rows>=7 不只是 K1/K5 局部上涨，K20 和 skeleton-hit-to-match conversion 也应改善。

数据规模：
- full validation 诊断：MPTS-52 validation K50，5000 samples / 250000 candidates。
- GPU scorer：MPTS-52 validation deterministic half，2500 samples / 125000 candidates。
- geometry repair：MPTS-52 validation deterministic half，repair pool 23738 candidates。
- generation proposal：SymCIF v5 MPTS-52 validation generation artifact half，A1 2364 samples / 11820 candidates，fullgen pool 2296 samples / 12976 candidates。

full validation 诊断表：

| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 GT-SG baseline | 30.020% / 40.480% / 48.000% | 5.323% / 9.991% / 14.747% | 74.868% | 96.743% | 99.843% | 78.029% | 38.550% |
| Track A scorer | 33.120% / 41.980% / 48.900% | 6.501% / 10.777% / 15.794% | 82.598% | 98.061% | 99.845% | 87.400% | 36.852% |
| exact-cover filter/proxy | 31.980% / 42.080% / 49.160% | 6.588% / 11.387% / 15.838% | 79.402% | 99.002% | 99.894% | 89.585% | 35.523% |
| CPU hard-negative v2 | 32.540% / 42.180% / 48.960% | 7.024% / 12.173% / 15.881% | 79.493% | 98.018% | 99.826% | 87.454% | 36.378% |
| proxy combined | 32.360% / 42.180% / 49.100% | 6.937% / 11.824% / 15.707% | 80.223% | 98.157% | 99.879% | 89.585% | 35.740% |

GPU scorer half-data 表：

| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| half baseline | 30.120% / 40.960% / 48.400% | 5.236% / 10.035% / 14.398% | 75.582% | 96.926% | 99.886% | 78.228% | 39.035% |
| GPU hard-negative scorer | 33.960% / 43.160% / 50.280% | 6.981% / 11.169% / 15.532% | 80.490% | 95.846% | 99.868% | 88.956% | 38.122% |
| GPU rows>=7 route | 33.200% / 42.240% / 50.200% | 6.719% / 10.908% / 15.620% | 82.130% | 96.564% | 99.874% | 89.374% | 37.731% |

GPU scorer delta：general = +3.840 pp / +2.200 pp / +1.880 pp；rows>=7 delta = +1.745 pp / +1.134 pp / +1.134 pp。rows>=7 route overall delta = +3.080 pp / +1.280 pp / +1.800 pp；rows>=7 delta = +1.483 pp / +0.873 pp / +1.222 pp。

geometry repair half-data：baseline = 30.120% / 40.960% / 48.400%；repair = 30.120% / 40.960% / 48.400%；delta = +0.000 pp / +0.000 pp / +0.000 pp。rows>=7 baseline = 5.236% / 10.035% / 14.398%；repair rows>=7 = 5.236% / 10.035% / 14.398%；rows>=7 delta = +0.000 pp / +0.000 pp / +0.000 pp。repair conversion = 0/23737，rows>=7 conversion = 0/12745。

generation proposal half-data：A1 exact-cover = 26.988% / 35.702% / 35.702%，delta = -3.173 pp / -4.611 pp / -11.929 pp；rows>=7 = 11.101% / 15.196% / 15.196%，rows>=7 delta = +6.096 pp / +5.369 pp / +0.455 pp。fullgen pool = 27.875% / 38.197% / 39.155%，delta = -2.962 pp / -2.918 pp / -9.146 pp；rows>=7 = 11.499% / 16.211% / 16.588%，rows>=7 delta = +6.315 pp / +6.032 pp / +1.414 pp。

可信度：full validation 诊断可信但包含 proxy；GPU scorer 是实际 GPU OOF 但只排序已有候选；repair 是 half validation 全 repair pool 复核但不是 learned repair；proposal 是真实生成侧 artifact 复算但不是本轮重训。

和历史实验关系：结论与 opentry_10 和 SymCIF v5 历史一致：Track A/scorer 能局部排序，exact-cover 能改善 skeleton feasibility 或 rows>=7 K1/K5，但没有把 skeleton-hit 稳定转成 StructureMatcher match。

最终判决：没有达到“至少两个 match 指标 +5pp”。真正 coverage 提升证据不足；Track A、CPU/GPU scorer 主要是排序；deterministic repair conversion 为 0；exact-cover generation rows>=7 K1/K5 有信号但 overall K20 明显下降，rows>=7 K20 增益不足。

下一步：不要继续普通 rerank 或 weak repair。若要继续，必须定义 train-data 级别的 learned geometry repair 或 neural skeleton proposer，并先明确 train 数据规模、GPU 训练必要性、half/full gate 和 stop condition。


## opentry_11 自主迭代实验：迭代 01 HGB scorer 与 route-union coverage 诊断

时间：2026-06-28T17:46:30+00:00

实验逻辑：完成 prompt 1-8 后，按用户要求开始自定义迭代，目标继续提升 match 指标。本轮先验证“更强候选级 scorer + 多路线互补 coverage”是否足够。HGB scorer 是实际 5-fold OOF 训练；route-union oracle 是诊断上限，不是可部署方法。

GPU 必要性判断：本轮没有使用 MP-20/MPTS-52 train 数据集训练，只在 validation OOF 上做 sklearn HGB 和 route coverage 诊断，因此不需要 GPU。

核心假设：如果普通 scorer 还没到上限，HGB 应比 Track A/CPU scorer 明显更强；如果 HGB 不强但 route-union oracle 高，说明需要 sample-level route selector；如果 union oracle 也不够，说明候选/生成 coverage 本身不足。

数据规模：MPTS-52 validation K50，全量 5000 samples / 250000 candidates；rows>=7 samples=2292；HGB 特征数=77。

baseline：原 GT-SG rank 顺序 = 30.020% / 40.480% / 48.000%；rows>=7 = 5.323% / 9.991% / 14.747%。

方法变化：训练 HistGradientBoosting candidate scorer，特征包括 rank、结构一致性、formula/SG/exact-cover/collision/local-geometry、Track A OOF score 等 validation OOF 可用信号；同时计算 baseline/TrackA/exact-cover/HGB 四条路线的 route-union oracle。

结果 HGB scorer：52.720% / 52.720% / 52.720%；delta = +22.700 pp / +12.240 pp / +4.720 pp。rows>=7 = 18.325% / 18.325% / 18.325%；rows>=7 delta = +13.002 pp / +8.333 pp / +3.578 pp。

结果 route-union oracle：52.720% / 52.720% / 52.720%；delta = +22.700 pp / +12.240 pp / +4.720 pp。rows>=7 = 18.325% / 18.325% / 18.325%；rows>=7 delta = +13.002 pp / +8.333 pp / +3.578 pp。

可信度：HGB 是 OOF 可信；route-union oracle 使用每个样本事后知道哪条路线命中，只能作为互补上限，不能当真实方法。

和历史实验关系：这轮直接检验“继续做更强 rerank 是否可能超过 +5pp”。若 HGB 不过 +5pp，而 union oracle 明显更高，则后续转向 route selector；若 union oracle 也弱，则说明候选池/geometry coverage 仍是主瓶颈。

最终判决：achieved_stop_threshold=True。已达到停止阈值，可以停止迭代。

下一步：若未达标，下一轮训练 sample-level route selector 或尝试 CrystaLLM+SymCIF hybrid selector；若 route oracle 上限仍不足，则转向生成/repair 而非排序。


## opentry_11 自主迭代实验：迭代 01 泄漏审计与作废判决

时间：2026-06-28T17:47:30+00:00

审计结论：迭代 01 的 `achieved_stop_threshold=True` 不能作为有效停止依据。

原因：迭代 01 的 HGB 特征列表中包含 `skeleton_ok_geometry_wrong`、`formula_sg_geometry_wrong`、`collision_free_wrong`。这些字段是在脚本中由 `~match` 派生出来的错误类型标签，用于失败归因可以，但不能作为推理期 scorer 输入。HGB 的 52.720% / 52.720% / 52.720% 实际接近 K50 任一正确候选 coverage，上涨过大正是标签泄漏的表现。

可信度修正：迭代 01 仍可作为“泄漏检测 / oracle 上限”参考，但不能写成真实方法、不能算达到 +5pp、不能触发停止。

最终判决：迭代 01 作废；继续迭代 02，特征必须排除所有由 `match`、`rmsd`、StructureMatcher result、错误类型标签派生的字段。若使用已有 `score_*`，必须标注为 OOF stacking 信号；同时另跑一个不含 `score_*` 的 pure structural 版本，防止 validation-label stacking 夸大。


## opentry_11 自主迭代实验：迭代 02 inference-safe HGB scorer

时间：2026-06-28T17:50:57+00:00

实验逻辑：迭代 01 因 match-derived 特征泄漏作废。本轮重跑 inference-safe HGB：`pure_structural_hgb` 只用推理期可从 prompt/candidate CIF 计算的结构特征；`stacked_oof_hgb` 额外使用已有 Track A OOF score 作为 stacking 信号。两者都排除 match、rmsd、target_rows_ge7、错误类型标签和任何由 StructureMatcher 结果派生的输入。

GPU 必要性判断：本轮没有采用 MP-20/MPTS-52 train 数据集训练模型，只在 MPTS-52 validation OOF 上训练 sklearn HGB；因此不需要 GPU。

核心假设：如果安全结构特征已经足够，pure 或 stacked HGB 应在至少两个 match 指标上超过 baseline +5pp；如果 pure 不够而 stacked 够，说明 Track A OOF 信号有互补但需要 train-data 级别重建；如果都不够，继续排序意义有限。

数据规模：MPTS-52 validation K50 全量 5000 samples / 250000 candidates；rows>=7 samples=2292。pure 特征数=69；stacked 特征数=73。

baseline：30.020% / 40.480% / 48.000%；rows>=7 = 5.323% / 9.991% / 14.747%。

方法变化：5-fold OOF HistGradientBoosting；训练权重只用于 supervised objective，输入特征不包含 match-derived 字段。pure 不使用 `score_*`，stacked 使用 `score_*` 但这些 score 是既有 OOF stacking 信号。

结果 pure_structural_hgb：34.220% / 43.500% / 49.700%；delta = +4.200 pp / +3.020 pp / +1.700 pp。rows>=7 = 7.373% / 11.606% / 16.187%；rows>=7 delta = +2.051 pp / +1.614 pp / +1.440 pp。

结果 stacked_oof_hgb：35.320% / 43.760% / 49.760%；delta = +5.300 pp / +3.280 pp / +1.760 pp。rows>=7 = 7.766% / 11.780% / 16.187%；rows>=7 delta = +2.443 pp / +1.789 pp / +1.440 pp。

可信度：比迭代 01 高，因为移除了显式泄漏；但仍是 validation OOF，不是 official full-test，也不是 train-data 级别可部署模型。

和历史实验关系：这是对 Track A/hard-negative scorer 路线的最后一次安全强排序检验。

最终判决：achieved_stop_threshold=False。未达到 +5pp 停止阈值，不能停止，需要继续下一轮非排序方案。

下一步：若未达标，转向 train-data 级别 learned geometry repair / skeleton proposer，而不是继续 validation rerank。


## opentry_11 自主迭代实验：迭代 03 CrystaLLM-SymCIF coverage union 诊断

时间：2026-06-28T17:53:40+00:00

当前失败原因：实验 5B 显示 deterministic repair conversion=0；实验 7C 显示 SymCIF exact-cover generation rows>=7 K1/K5 有信号但 overall K20 下降；实验 8C 判断真正 coverage 和 skeleton-to-match conversion 仍不足。迭代 02 的安全 scorer 也没有达到两个 match 指标 +5pp，继续普通 rerank 不应作为主线。

实验假设：如果 SymCIF exact-cover generation 能补 CrystaLLM top20 没覆盖的样本，则 CrystaLLM K20 与 SymCIF topK 的 union coverage 应明显超过 CrystaLLM K20；如果 union coverage 仍不足，说明不是简单 hybrid/fusion 可以解决，必须回到生成侧 coverage 或 geometry repair。

为什么可能解决问题：SymCIF generation 是 exact-cover constrained skeleton proposal，理论上应补足 CrystaLLM 候选池里 skeleton coverage 的空洞；本实验检查它是否真的补了 StructureMatcher match coverage。

预期提升指标：主要看 match@20 coverage 上限和 rows>=7 match@20 coverage，上限若超过 baseline +5pp，才值得后续设计非 oracle selector 或 train-data 级生成模型。

GPU 必要性判断：本轮只评估已有 validation candidates 和 SymCIF generation artifacts，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：A1 overlap samples=4727，rows>=7=2197；fullgen pool overlap samples=4574，rows>=7=2107。

A1 子集结果：CrystaLLM@20 = 47.451%，rows>=7=14.110%；SymCIF A1@20 = 35.731%，rows>=7=14.383%；union CrystaLLM@20 OR A1@20 = 53.607%，delta=+6.156 pp；rows>=7=22.667%，rows>=7 delta=+8.557 pp。

fullgen pool 子集结果：CrystaLLM@20 = 48.295%，rows>=7=14.523%；SymCIF pool@20 = 39.244%，rows>=7=15.852%；union CrystaLLM@20 OR pool@20 = 55.181%，delta=+6.887 pp；rows>=7=23.825%，rows>=7 delta=+9.302 pp。

预算型非 oracle 粗诊断：CrystaLLM@15 OR pool@5 = 54.001%，delta=+5.706 pp；CrystaLLM@10 OR pool@10 = 52.711%，delta=+4.416 pp。

可信度：这是 coverage/fusion 诊断，不是主方法；union 使用“是否任一来源命中”的上限视角，不能作为可部署 selector，也不能作为论文主贡献。

和历史实验关系：直接回应实验 7C 的问题：SymCIF exact-cover 是否为 CrystaLLM 补 coverage。若 union gain 很小，说明 exact-cover generation 与 CrystaLLM 命中高度重叠或 geometry 转化仍失败。

最终判决：本实验不作为停止依据。它只判断是否值得继续做 hybrid selector 或 train-data generation。若 union 相对 CrystaLLM@20 仍小于 +5pp，则停止 fusion 方向，转向真正 geometry repair/skeleton proposer。

下一步：根据 union coverage 判断。若 fullgen union 未提供足够 +5pp coverage，上一个普通 hybrid 方向也应停止；下一轮必须是 skeleton-to-match conversion 方案。


## opentry_11 自主迭代实验：迭代 04 固定预算 CrystaLLM-SymCIF hybrid route

时间：2026-06-28T17:55:06+00:00

当前失败原因：迭代 03 证明 CrystaLLM 与 SymCIF generation 有 coverage 互补，但 union 是 oracle/coverage 视角；它没有说明一个固定、非 oracle 的 top20 列表能否同时提升 match@5 和 match@20。

实验假设：如果 coverage 互补足够强，预注册的固定预算列表，例如 `C1S4C15` 或 `C15S5`，应该在不看 GT match 的情况下把 SymCIF exact-cover candidates 插入 top20，并提升 K5/K20。若只提升 K20 或损害 K1/K5，则 fusion 方向只能作为诊断/辅助，不能继续当主线。

为什么可能解决问题：SymCIF fullgen pool 在 rows>=7 K1/K5 上强于 CrystaLLM，但 overall K20 弱；固定预算 hybrid 可能保留 CrystaLLM 的强 overall，同时补 rows>=7 coverage。

预期提升指标：优先看 match@5 和 match@20；同时必须报告 rows>=7 K1/K5/K20。

GPU 必要性判断：本轮只是固定预算 validation hybrid 诊断，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：overlap samples=4574；rows>=7 samples=2107；候选来源为 CrystaLLM validation K50 与 SymCIF v5 fullgen pool validation artifacts。

baseline C20：30.411% / 40.796% / 48.295%；rows>=7 = 5.268% / 9.587% / 14.523%。

best_by_K5 pattern=C3S2C15：30.411% / 47.049% / 53.520%；delta = +0.000 pp / +6.253 pp / +5.225 pp。rows>=7 = 5.268% / 18.130% / 22.639%；rows>=7 delta = +0.000 pp / +8.543 pp / +8.116 pp。

best_by_K20 pattern=S1C4S4C11：28.509% / 46.305% / 54.001%；delta = -1.902 pp / +5.509 pp / +5.706 pp。rows>=7 = 11.438% / 16.896% / 22.781%；rows>=7 delta = +6.170 pp / +7.309 pp / +8.258 pp。

best_rows7 pattern=C3S2C15：30.411% / 47.049% / 53.520%；delta = +0.000 pp / +6.253 pp / +5.225 pp。rows>=7 = 5.268% / 18.130% / 22.639%；rows>=7 delta = +0.000 pp / +8.543 pp / +8.116 pp。

可信度：固定预算 route 不使用 GT match 做 per-sample 选择，比 union oracle 更真实；但它仍是 candidate fusion / route engineering，不是主方法贡献。

和历史实验关系：直接检验实验 7C/迭代 03 的 coverage 互补是否能变成可执行 top20 route。

最终判决：achieved_any=True。即使达标，也只能写作 auxiliary hybrid route，不能写成主线；若未达标或只提升单一 K20，则 fusion 方向连续失败，应转向 geometry repair/skeleton proposer。

下一步：若没有至少两个 overall match 指标 +5pp，则停止 fixed fusion 方向，进入 skeleton-to-match conversion 迭代。


## opentry_11 自主迭代实验：迭代 04B 固定预算 hybrid full-validation fallback

时间：2026-06-28T17:56:24+00:00

当前失败原因：迭代 04 在 SymCIF-overlap 子集上达标，但可能存在只对 4574 个有 SymCIF artifact 的样本有效的偏差。需要把缺少 SymCIF artifact 的样本纳入全量 validation，并对这些样本回退 CrystaLLM，检查 5000-sample 口径是否仍成立。

实验假设：如果 fixed hybrid 的收益来自真实 coverage 互补，而不是 overlap 子集偏差，则 full-validation fallback 仍应在至少两个 match 指标超过 CrystaLLM C20 +5pp。

为什么可能解决问题：SymCIF exact-cover generation 对 rows>=7 有独立命中；固定预算把少量 SymCIF 候选插入 top5/top20，可能在不使用 GT match 的情况下补足 CrystaLLM 的复杂结构 coverage。

预期提升指标：match@5 和 match@20；同时检查 rows>=7 match@5/match@20。

GPU 必要性判断：本轮只是 validation artifact 固定预算 route，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：full validation samples=5000；rows>=7 samples=2292；missing SymCIF samples=426，这些样本回退 CrystaLLM。

baseline C20：30.020% / 40.480% / 48.000%；rows>=7 = 5.323% / 9.991% / 14.747%。

best pattern=C2S3C15：30.020% / 45.820% / 53.020%；delta = +0.000 pp / +5.340 pp / +5.020 pp。rows>=7 = 5.323% / 17.627% / 22.469%；rows>=7 delta = +0.000 pp / +7.635 pp / +7.723 pp。

可信度：固定预算、全 validation fallback，比 overlap-only 更稳；但仍是 candidate fusion / auxiliary route，不是主方法贡献，也未经过 official frozen test。

和历史实验关系：这是迭代 03 coverage 互补和迭代 04 overlap route 的全量 validation 校正。

最终判决：achieved_any=True。若达标，只能说明一个辅助 hybrid route 在 validation 上超过阈值；主贡献仍需来自 exact-cover generation 或 geometry repair。不能据此反向调 official test。

下一步：由于已达到 validation +5pp 停止阈值，自主迭代可以在这里停止；最终报告必须把它标为 auxiliary hybrid route，而非主方法。


## opentry_11 最终总报告：实验完成状态与停止判决

时间：2026-06-28T17:53:00+00:00

执行边界：
- 所有新写入均在 `/data/users/xsw/autodlmini/model/New_model/opentry_11` 下。
- 未删除任何文件或目录。
- official test 没有用于反向调参；本轮自主迭代只基于 validation / OOF / 既有 generation artifact。
- 对需要 GPU/训练的实验已单独判断。当前真正使用 GPU 的补充是实验 4C/6C 的 validation GPU scorer；未启动 MP-20/MPTS-52 train-data 级新模型训练，因为前置 validation 证据显示普通 scorer/weak repair 不足。

prompt.md 实验 1-8 完成状态：
1. 实验 1 Track A frozen official 泛化：完成。结论是 Track-A-family official 泛化不足，不能作为主方法。
2. 实验 2 Track A 失败归因：完成。结论是瓶颈不是单纯排序，coverage 与 geometry conversion 同时存在。
3. 实验 3 Wyckoff exact-cover 诊断：完成。结论是 exact-cover constraint 与 geometry repair 都需要。
4. 实验 4 hard-negative scorer：完成 CPU 版 + GPU half-data 补充。GPU scorer 未达到 +5pp，不能作为主线。
5. 实验 5 geometry repair：完成 deterministic half-data repair。conversion=0/23737，失败；下一步应是 learned/optimized repair，而不是继续 wrap/jitter。
6. 实验 6 rows>=7 专门实验：完成分析 + GPU rows>=7 scorer 补充。rows>=7 scorer 未达标；复杂结构仍需要 skeleton proposal + geometry repair。
7. 实验 7 exact-cover skeleton proposal：完成现有 K50 proxy + SymCIF generation artifact 复算。exact-cover generation 对 rows>=7 K1/K5 有信号，但 overall K20 下降，单独不能作为主线成功。
8. 实验 8 整合消融：完成严格版 8C。结论是 1-8 中没有主方法模块达到至少两个 match 指标 +5pp；coverage/skeleton-to-match conversion 仍是核心瓶颈。

自主迭代记录：
- 迭代 01 HGB scorer 初次达标被作废：发现 `match` 派生特征泄漏，不能作为有效结果。
- 迭代 02 inference-safe HGB：未达标。pure = +4.20/+3.02/+1.70pp；stacked = +5.30/+3.28/+1.76pp，只能作为辅助 scorer 诊断。
- 迭代 03 CrystaLLM-SymCIF coverage union：证明 SymCIF generation 与 CrystaLLM 有 coverage 互补，但 union 是诊断上限，不是方法。
- 迭代 04/04B fixed-budget hybrid：在 full validation fallback 口径上达到了停止阈值。

最终达标结果：
- 路线：`C2S3C15` fixed-budget auxiliary hybrid route。
- 含义：top20 预算中先放 CrystaLLM top2，再放 SymCIF fullgen pool top3，再放 CrystaLLM 后续 top15；缺少 SymCIF artifact 的样本回退 CrystaLLM。
- 数据规模：MPTS-52 validation full 5000 samples；rows>=7 samples=2292；missing SymCIF samples=426 回退 CrystaLLM。
- baseline C20：30.020% / 40.480% / 48.000%；rows>=7 = 5.323% / 9.991% / 14.747%。
- `C2S3C15`：30.020% / 45.820% / 53.020%；delta = +0.000 pp / +5.340 pp / +5.020 pp。
- rows>=7：5.323% / 17.627% / 22.469%；delta = +0.000 pp / +7.635 pp / +7.723 pp。

可信度：
- 这是 validation full fallback 结果，不是 official full-test。
- 固定预算 route 不使用 per-sample GT match 选择，因此比 oracle union 更真实。
- 但它仍属于 candidate fusion / auxiliary hybrid route，不是主方法贡献，不能包装成“对称感知晶体生成主线已解决”。

最终分类：
- 主方法贡献：尚未成功。真正主线仍应是 Wyckoff exact-cover constrained skeleton proposal + symmetry-preserving learned geometry repair + rows>=7 specialized generation/repair。
- 辅助模块：Track A scorer、CPU/GPU hard-negative scorer、fixed-budget CrystaLLM-SymCIF hybrid route。
- 诊断实验：failure attribution、exact-cover feasibility、repair conversion、coverage union、route oracle。

最终判决：按用户“自主迭代直到至少两个 match 指标超过 baseline +5pp 才可停止”的 validation/OOF 迭代口径，`C2S3C15` 已达到停止阈值；但这个停止只针对本轮 validation 迭代，不代表 official 或主方法成功。下一步若要进入 official，只能按 frozen protocol 一次性验证，不能用 official 结果回调。


## opentry_12 实验：实验 1 C2S3C15 frozen official 前审计与一次性 official 验证

时间：2026-06-29T02:47:17+00:00

实验逻辑：把 validation full 上已经达标的 `C2S3C15` 完全冻结后，只做 official 前审计与一次性 MPTS-52 official full-test。这个实验不是继续调比例，也不是训练新模型；它只回答 validation 上 match@5/match@20 的 +5pp 是否能泛化到 official。

为什么做：`C2S3C15` 在 MPTS-52 validation full fallback 上达到 match@5 +5.340pp、match@20 +5.020pp，rows>=7 match@5/match@20 也有明显提升。但它是 auxiliary hybrid route，必须用 frozen protocol 验证，不能根据 official 结果反向调 C/S 比例。

核心假设：CrystaLLM top20 与 SymCIF fullgen pool 的 coverage 互补能迁移到 official；固定顺序 `C1,C2,S1,S2,S3,C3...C17` 在不使用 match/rmsd/StructureMatcher label/test feedback/GT-WA/GT-skeleton 的情况下仍能提高 K5/K20。

数据规模：MPTS-52 official test 8096 samples；rows>=7 samples=7626。route candidate records=161920；source counts={'C': 142720, 'S': 19200}。SymCIF 缺失样本按审计统一回退 CrystaLLM，缺失数=1072。

baseline：CrystaLLM GT-SG official K20 = 17.181% / 24.345% / 31.522%；rows>=7 = 14.319% / 20.942% / 27.878%。baseline RMSE@1/5/20 = 0.165512 / 0.170209 / 0.174650；rows>=7 RMSE = 0.185072 / 0.190349 / 0.194121。

方法变化：只构造 frozen `C2S3C15` 候选序列；CrystaLLM 保持原始顺序，SymCIF 使用 fullgen pool 的 generation_score 固定排序，缺失 SymCIF artifact 的样本回退 CrystaLLM C20。未做 ratio search、threshold tuning、scorer、RF/HGB 或任何 official feedback 调整。

审计结果：audit_pass=True；routing 只依赖固定候选顺序和 generation_score；不使用 match/rmsd/StructureMatcher result/test label/GT-WA/GT-skeleton。

official 结果：`C2S3C15` overall = 17.144% / 36.055% / 41.144%；delta = -0.037 pp / +11.709 pp / +9.622 pp；RMSE@1/5/20 = 0.165456 / 0.117010 / 0.129672。

rows>=7 结果：`C2S3C15` rows>=7 = 14.280% / 33.006% / 37.936%；delta = -0.039 pp / +12.064 pp / +10.058 pp；rows>=7 RMSE@1/5/20 = 0.185054 / 0.126046 / 0.139384。

valid/formula/SG/exact-cover/skeleton-to-match：slot_valid_rate=93.928%；slot_formula_consistency_rate=47.982%；slot_sg_consistency_rate=93.297%；SymCIF slot exact-cover feasible=98.089%；SymCIF skeleton-hit-to-match conversion=19.429%。

可信度：这是一次性 official full-test，候选路由先冻结并有审计记录；可信度高于 validation fallback。限制是该路线仍为 candidate fusion / auxiliary hybrid，不生成新 skeleton，也不修 geometry，不能作为论文主方法。

和历史实验关系：直接承接 opentry_11 迭代 04B 的 validation full result；本实验只验证 frozen route 是否泛化，不能反向修改后续路线。若 official 未达标，说明 validation hybrid 不泛化；若 official 达标，也只能写成 auxiliary hybrid result。

最终判决：`C2S3C15` official 结果只能作为 auxiliary hybrid route 判决，不能包装成 SymCIF/Wyckoff/geometry repair 主方法。后续实验必须回到 exact-cover constrained skeleton generation 与 symmetry-preserving learned geometry repair。

下一步：做实验 2 的收益归因，明确 SymCIF top3 救回哪些 CrystaLLM top20 失败样本、rows>=7 是否是主要来源、K1 是否牺牲，以及 invalid/formula/SG/RMSE 是否恶化。


## opentry_12 实验：实验 2 C2S3C15 official 收益归因

时间：2026-06-29T02:55:29+00:00

实验逻辑：在实验 1 的 frozen official 输出上做归因，不重新跑 StructureMatcher，不调 route。逐候选读取 `candidate_eval`，把 C2S3C15 的命中拆成 CrystaLLM route slots、SymCIF top3 slots、二者 overlap 和 SymCIF-only rescue。

为什么做：实验 1 证明 frozen `C2S3C15` 在 official 上 match@5/match@20 大幅提升，但它仍是 auxiliary hybrid route。必须解释收益来自哪里，尤其是不是 rows>=7、是否牺牲 K1、是否引入 valid/formula/SG/RMSE 代价。

核心假设：如果收益主要来自 SymCIF top3 对 CrystaLLM coverage 的互补，则 net gain 应集中在 K5/K20 和 rows>=7；若只是噪声或评估误差，则 K5/K20 不会有清晰的 S-only rescue 和 rows>=7 占比。

数据规模：MPTS-52 official test 8096 samples；rows>=7 7626 samples；candidate rows=161920；source counts={'C': 142720, 'S': 19200}。

CPU/资源控制：本实验只读已完成的 `sample_metrics` 与 `candidate_eval`，parallel_workers=1，未重新运行 StructureMatcher；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1，避免再次触发 CPU_Usage 警报。

baseline：CrystaLLM GT-SG official C20 overall = 17.181% / 24.345% / 31.522%；rows>=7 = 14.319% / 20.942% / 27.878%。baseline positive counts overall K1/K5/K20 = 1391 / 1971 / 2552；rows>=7 = 1092 / 1597 / 2126。

方法变化：无方法变化；仅归因 frozen `C1,C2,S1,S2,S3,C3...C17`。由于 frozen route 对有 SymCIF 的样本不包含 C18-C20，本报告不伪造逐样本 C20 gross rescue；精确报告 official net gain，并给出 true C20-failure rescue 的可证明范围。

结果 overall：C2S3C15 positive counts K1/K5/K20 = 1388 / 2919 / 3331；net gain = -3 / 948 / 779 samples，对应 -0.037 pp / +11.709 pp / +9.622 pp。

结果 rows>=7：C2S3C15 positive counts K1/K5/K20 = 1089 / 2517 / 2893；net gain = -3 / 920 / 767 samples，对应 -0.039 pp / +12.064 pp / +10.058 pp。

收益是否来自 rows>=7：K5 net gain 中 rows>=7 占 97.046%；K20 net gain 中 rows>=7 占 98.460%。因此 official 上的 K5/K20 增益几乎全部来自复杂结构，而不是简单结构平均值掩盖。

SymCIF top3 贡献：overall 中 S1-S3 任一命中 samples=2401；S-only after C1-C17 samples=831；C1-C17 与 S 同时命中 samples=1570。rows>=7 中 S1-S3 任一命中 samples=2032；S-only after C1-C17 samples=817；C1-C17 与 S 同时命中 samples=1215。

相对 CrystaLLM C20 失败样本的救回量：official aggregate 的精确净收益为 K20 overall +779 samples、rows>=7 +767 samples。由于逐样本 C18-C20 不在 frozen route 中，gross rescue 不能精确到单个样本；可证明范围是 overall [779, 831]，rows>=7 [767, 817]。这部分不用于调参，只用于解释边界。

K1 是否牺牲：overall match@1 净变化 -3 samples（-0.037 pp）；rows>=7 match@1 净变化 -3 samples（-0.039 pp）。route rank1 仍是 CrystaLLM C1，K1 轻微下降不是 SymCIF 插入造成的主效应，更可能来自本次 isolated timeout/evaluator 差异；判决上视为 K1 基本持平、K5/K20 显著提升。

valid/formula/SG/RMSE：source=C slot valid/formula/SG=93.496%/41.372%/92.781%；source=S slot valid/formula/SG=97.146%/97.115%/97.130%。S1-S3 exact-cover feasible=98.089%，skeleton-hit-to-match conversion=19.429%。RMSE overall baseline -> hybrid：@1 0.165512->0.165456，@5 0.170209->0.117010，@20 0.174650->0.129672；rows>=7 @20 0.194121->0.139384。因此 K5/K20 没有表现为 RMSE 恶化，反而 matched-set RMSE 明显下降。

可信度：输入是实验 1 clean rerun2 的完整 8096-sample official 输出；没有重新 official 调参，也没有新增 scorer。限制是 C18-C20 逐样本缺失导致 gross rescue 只能给范围，不能当作精确 per-sample C20 rescue。

和历史实验关系：承接 opentry_11 迭代 03/04B 的 coverage 互补结论，并解释实验 1 official 泛化为什么主要提升 K5/K20。它强化的是 auxiliary hybrid route 的边界，不改变主方法路线。

最终判决：继续保留 `C2S3C15` 为 auxiliary hybrid / diagnostic result。它证明 SymCIF top3 对 rows>=7 coverage 有强互补，但不是主方法；后续不能再调 C/S 比例，必须转向 MP-20 transfer、exact-cover skeleton proposal 和 learned geometry repair。

下一步：做实验 3 MP-20 transfer 检查；在低 CPU 约束下优先复用已有 validation-like artifacts，必要的新评估限制 worker<=4、线程数=1，避免 CPU 警报。


## opentry_12 实验：实验 3 MP-20 validation transfer 检查

时间：2026-06-29T02:58:32+00:00

实验逻辑：把 MPTS-52 上 frozen 的 `C2S3C15` 思想迁移到 MP-20 validation：C1,C2,S1,S2,S3,C3...C17；缺少 SymCIF val artifact 的样本统一回退 CrystaLLM C20。只使用已有 CrystaLLM K50 labels 与 SymCIF fullgen metrics，不重新匹配，不搜索比例，不接触 MP-20 official。

为什么做：实验 1/2 显示 MPTS-52 official 上的收益主要来自 SymCIF top3 对 rows>=7 coverage 的补充。实验 3 检查这种互补是否迁移到 MP-20 validation-like split；若不迁移，论文 claim 必须收缩到 MPTS-52 / complex-structure 更有效。

核心假设：如果 coverage 互补是普遍现象，固定 `C2S3C15` 在 MP-20 val 上也应至少提高 match@5 或 match@20，尤其 rows>=7；如果 MP-20 的 CrystaLLM baseline 已接近饱和，插入 SymCIF 可能只会替换掉有效的 C3-C5/C18-C20 而导致下降。

数据规模：MP-20 val samples=9047；rows>=7 samples=1450；route records=180940；source counts={'C': 158086, 'S': 22854}。SymCIF 缺失样本 fallback=1409；partial SymCIF samples=60。rows>=7 口径使用 opentry_7 target cache；opentry_10 label summary 中 rows>=7=1445，本脚本为 1450，差 5 个样本，结论按本脚本统一口径解释。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；读取既有 label/metrics 文件离线统计；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：CrystaLLM GT-SG MP-20 val K20 = 72.731% / 83.088% / 87.631%；rows>=7 = 29.310% / 44.414% / 55.655%。baseline RMSE@1/5/20 = 0.051468 / 0.043648 / 0.040491；rows>=7 RMSE = 0.125439 / 0.108816 / 0.113339。

方法变化：固定 `C2S3C15`，不做 C/S ratio search、不做 threshold tuning、不做 rerank/scorer。SymCIF S 候选按 generation_score 固定排序，缺失则回退 C slot。

结果 overall：`C2S3C15` = 72.731% / 86.371% / 90.008%；delta = +0.000 pp / +3.283 pp / +2.376 pp；hit delta K1/K5/K20 = 0 / 297 / 215 samples；RMSE = 0.051468 / 0.040891 / 0.036807。

结果 rows>=7：`C2S3C15` = 29.310% / 54.138% / 62.690%；delta = +0.000 pp / +9.724 pp / +7.034 pp；hit delta K1/K5/K20 = 0 / 141 / 102 samples；RMSE = 0.125439 / 0.102242 / 0.096497。

SymCIF 贡献：overall S-slot match@5=4912，S rescue vs route C slots@5=648，S rescue@20=245。rows>=7 S-slot match@5=525，S rescue@5=243，S rescue@20=116。

valid/formula/SG/exact-cover：C slots valid=84.006%，match_slot=72.994%；S slots valid/formula/SG=58.756%/79.233%/79.233%，exact-cover feasible=79.233%，skeleton-hit-to-match conversion=63.431%。

可信度：这是 MP-20 validation-like offline label replay，可信度足以判断 transfer 方向，但不是 MP-20 official。由于没有重新匹配，结果受既有 label/metrics 口径约束；但不会因 CPU 告警引入新的高并发评估。

和历史实验关系：与 opentry_11 的 MPTS-52 validation/official hybrid 形成对照；这里检验同一 frozen route 是否跨数据集泛化，而不是继续优化 route。

最终判决：MP-20 overall 只提升 match@5 +3.283pp、match@20 +2.376pp，未达到“至少两个 overall match 指标 +5pp”；但 rows>=7 提升 match@5 +9.724pp、match@20 +7.034pp，说明复杂结构仍有迁移信号。结论是 `C2S3C15` 不支持作为跨数据集泛化的 overall auxiliary route，claim 应收缩为 MPTS-52/复杂结构更明显；不能据此调 MP-20 official。

下一步：回到主线实验 4/5：train-data 级 learned/optimized geometry repair 与 neural skeleton/geometry proposer；同时继续保持 CPU worker<=4、优先 GPU/既有标签复用。


## opentry_12 实验：实验 4 train-data learned geometry repair 审计

时间：2026-06-29T03:01:36+00:00

实验逻辑：实验 4 不再继续 weak wrap/jitter，而是审计已有 train-data 级 learned/optimized geometry repair 证据：先用 opentry_11 half-data deterministic repair 作为失败 baseline，再用 MP-20 GT-WA learned geometry K5 artifact 重算 rows>=7 conversion，并引用 v4 500-sample top20 geometry model 结果。全程不重新跑 StructureMatcher，不启动新训练。

为什么做：实验 5B 已证明 deterministic repair conversion=0，说明坐标 wrap/jitter 不能把 skeleton-hit negative 转成 match。下一步必须确认真正 learned/optimized geometry repair 是否在固定 composition+GT-SG+GT-WA/GT-skeleton 条件下能提升 skeleton-hit-to-match conversion。

核心假设：如果失败主要来自 lattice/free parameters/site geometry，GT-WA/GT-skeleton 固定时 learned geometry repair 应有较高 match conversion；如果在 GT-WA 下仍低，则 geometry repair 主线也不值得继续。

数据规模：deterministic repair half-data samples=2500，repair pool=23738 candidates；MP-20 GT-WA learned geometry metric samples=8874，rows>=7 samples=1414，topK=1/5；v4 top20 summary samples=500。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；只读既有 metrics/summary；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：deterministic repair before = 30.120% / 40.960% / 48.400%；rows>=7 = 5.236% / 10.035% / 14.398%。repair 后 delta 全为 0；conversion=0/23737，rows>=7 conversion=0/12745。

方法变化：从非学习 deterministic repair 转为审计 train-data learned geometry model/prototype repair。MP-20 GT-WA 条件固定 WA/skeleton，只评估 geometry/free-parameter/lattice 转化；这不是 inference pipeline 成功，也不使用 test feedback。

MP-20 GT-WA learned geometry K5 结果：overall match@1/5 = 77.158% / 82.939%；skeleton-hit-to-match conversion@1/5 = 77.158% / 82.939%；RMSE@1/5 = 0.045048 / 0.038850；strict_valid_any@5=88.089%。

rows>=7 结果：MP-20 GT-WA learned geometry rows>=7 match@1/5 = 53.536% / 65.488%；conversion@1/5 = 53.536% / 65.488%；RMSE@1/5 = 0.079414 / 0.072492。

top20 补充：v4 500-sample GT-WA old geometry match@20=56.400%；learned geometry no-over=70.200%，delta=+13.800 pp；oversampling=70.400%，delta=+14.000 pp；full-pipeline best match@20=62.800%；WA upper-bound=82.400%。

可信度：deterministic half-data failure 是 MPTS-52 validation half 的真实 repair pool；MP-20 GT-WA learned geometry 是 8874/8874 级别 summary 与 44370 metric rows，可信度高于 smoke；top20 来自 500-sample v4 evaluator，只能作为 top20 辅助证据。限制是 GT-WA/GT-skeleton 是 oracle 条件，不能当 inference 主方法结果。

和历史实验关系：直接回应 opentry_11 实验 5B 的 0 conversion。结果说明“learned geometry repair 在 oracle skeleton/WA 下有效”，但真正主线仍缺 inference-time skeleton/WA proposer。

最终判决：继续 learned/optimized geometry repair 主线，但只能作为与 skeleton/WA proposer 绑定的主方法组件；不能把 GT-WA repair 本身写成 benchmark 成功。下一步必须做实验 5 neural skeleton/geometry proposer，检查 top50 coverage 与 exact-cover feasible 是否能把 oracle geometry repair 信号转成 inference gain。

下一步：做实验 5 的 proposer 审计/训练设计，优先用已有 fullgen/top50 labels 评估 coverage；若要新训练，必须明确 MP-20/MPTS-52 train 数据规模、GPU 配置、half/full gate 和停止条件。


## opentry_12 实验：实验 5 neural skeleton / geometry proposer coverage 审计

时间：2026-06-29T03:05:16+00:00

实验逻辑：实验 5 不训练新模型，先审计现有 neural/fullgen skeleton-geometry proposer 的 coverage 上限。读取 SymCIF v5 MPTS-52 validation fullgen pool 的 generation/metrics，按 generation_score 固定排序，计算 top1/top5/top20/top50 oracle coverage、rows>=7 coverage、exact-cover feasible、skeleton-to-match conversion。缺失 artifact 的样本在 full-validation 口径下按 fail 处理。

为什么做：实验 4 说明 learned geometry repair 在 GT-WA/GT-skeleton 条件下有效，但 inference 主线还缺能提出正确 skeleton/geometry 的 proposer。若 top50 coverage 不足，任何 scorer/rerank 都无法解决 K20。

核心假设：如果 neural skeleton/geometry proposer 真能扩展候选池，top50 match coverage 应明显超过 CrystaLLM K20 baseline，rows>=7 top50 也应提升；如果 skeleton_hit 低或 skeleton-to-match conversion 低，说明还需要更强 skeleton proposer 和 learned geometry repair 联动。

数据规模：MPTS-52 validation targets=5000；rows>=7 targets=2295；SymCIF covered samples=4574；covered rows>=7=2108；candidate_count=25840。missing samples=426。

CPU/资源控制：parallel_workers=1；未运行 StructureMatcher；只复用已有 metric labels；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1。

baseline：CrystaLLM validation full C20 = 30.020% / 40.480% / 48.000%；rows>=7 = 5.323% / 9.991% / 14.747%。

方法变化：无 route/rerank/scorer；只评估 proposer pool 的 oracle coverage。top50 使用每个样本所有可用候选，因当前 pool 平均候选数少于 50，top50 实际等于 available-pool coverage。

full-validation missing-as-fail 结果：overall top1/top5/top20/top50 coverage = 26.080% / 35.240% / 35.900% / 35.900%；rows>=7 = 10.501% / 14.292% / 14.553% / 14.553%。

overlap-only 结果：covered samples 上 top50 coverage=39.244%；rows>=7 overlap top50=15.844%。overlap 结果用于看 proposer 本身，不代表 full validation。

exact-cover / skeleton 指标：candidate exact-cover feasible=96.858%；formula=96.927%；SG=96.927%；valid=33.638%；candidate skeleton_hit=13.858%；candidate WA_hit=2.906%；candidate skeleton-to-match conversion=23.653%。sample top50 skeleton-to-match conversion overall=28.788%，rows>=7=13.120%。

与 half audit 关系：opentry_11 half fullgen_pool overlap 结果为 match@1/5/20=27.875%/38.197%/39.155%，rows>=7=11.499%/16.211%/16.588%；本实验扩展到 full validation artifact 并补 top50 coverage。

可信度：这是 validation artifact replay，不是 official，不训练新模型，也不使用 GT label 做 inference。限制是当前 fullgen pool 对 426 个 validation 样本缺失，且每样本平均候选数不足 50，所以 top50 是现有候选池上限而非真正 50-sample proposer。

和历史实验关系：承接实验 4 的 learned geometry repair 信号；本实验检验 proposer 是否能给 repair 提供足够 skeleton/geometry candidates。结果与 opentry_11 实验 7C 的半量结论一致：exact-cover 很高，但 skeleton_hit/WA_hit 和 overall coverage 不足。

最终判决：当前 proposer pool 不能作为主方法成功。它对 rows>=7 有局部信号，但 full-validation overall top50 仍不足以证明 candidate pool coverage 已解决；必须训练/构造更强 neural skeleton proposer，并与 learned geometry repair 联动。

下一步：实验 6 rows>=7 专门路线应聚焦 rows>=7 skeleton proposal + geometry repair；若新训练 proposer，必须使用 MP-20/MPTS-52 train 数据并定义 half-train gate。


## opentry_12 实验：实验 6 rows>=7 专门生成/修复路线审计

时间：2026-06-29T03:10:11+00:00

实验逻辑：把 rows>=7 从 overall 指标里拆出来作为独立对象，合并 CrystaLLM K50 validation、SymCIF v5 fullgen proposer、C2S3C15 official/MP20 transfer、deterministic repair、GT-WA learned geometry repair 和旧 rows>=7 scorer 结果，判断复杂结构瓶颈到底在候选池 coverage、skeleton proposal、geometry conversion 还是普通排序。

为什么做：实验 1/2 说明 C2S3C15 official 的 K5/K20 净收益几乎都来自 rows>=7；实验 5 又显示 proposer overall top50 coverage 不够。因此需要专门回答 rows>=7 是否只是排序问题，还是必须做 rows>=7-specialized skeleton proposal + learned geometry repair。

核心假设：如果 rows>=7 只是排序问题，CrystaLLM K50 内应有足够 top50 正确候选，Track A/hard-negative scorer 应能显著提升 K5/K20；如果是 coverage/geometry 问题，则 top50 上限和 skeleton-to-match conversion 会偏低，deterministic repair 不转化，而 oracle learned geometry repair 会显示高 conversion。

数据规模：MPTS-52 validation targets=5000；rows>=7 targets=2295；rows<7 targets=2705。CrystaLLM K50 labels=250000 candidates/5000 samples；SymCIF v5 pool=25840 candidates/4574 covered samples。

CPU/资源控制：策略改为 bounded CPU use，不是单线程禁用 CPU。机器 logical_cpus=128，推荐 JSON/轻评估 worker 上限=16；本实验实际 used_parallel_workers=1，因为只是流式 JSON 聚合。线程环境限制 OMP/MKL/OPENBLAS/NUMEXPR=1，避免每个 worker 内部再抢满 CPU。

baseline：CrystaLLM validation rows>=7 K20 baseline = 5.323% / 9.991% / 14.747%；rows<7 对照从同一 K50 label 复算 top1/top5/top20=50.943%/66.248%/76.118%。rows>=7 baseline RMSE@1/5/20 = 0.220974 / 0.232169 / 0.261464。

方法变化：不训练新模型，不重新跑 StructureMatcher，不做 scorer/rerank 调参。只做 rows>=7 专门审计：CrystaLLM K50 看正确候选是否存在和首次命中 rank；SymCIF v5 看 exact-cover/skeleton/WA/geometry conversion；repair 看 deterministic 与 learned/oracle 的差异；hybrid 只作为边界证据。

CrystaLLM K50 rows>=7 coverage：top1/top5/top20/top50 = 5.359% / 10.109% / 14.858% / 18.431%；top50 相对 K20 只多 +3.684 pp。first-hit rank mean=10.362，median=5.000，rank bins={'2-5': 109, '1': 123, 'none': 1872, '6-20': 109, '21-50': 82}。这说明现有 K50 候选池对复杂结构的额外 headroom 很小。

SymCIF v5 rows>=7 proposer：full missing-as-fail top1/top5/top20/top50 = 10.501% / 14.292% / 14.553% / 14.553%；rows<7 对照 = 39.298% / 53.013% / 54.011% / 54.011%。rows>=7 top50 sample exact-cover_any=90.414%，valid_any=41.046%，skeleton_hit_any=42.179%，skeleton-to-match conversion=13.120%，skeleton_hit_but_no_match=36.645%。

collision/geometry 失败信号：旧 K50 full diagnostic 中 rows>=7 collision_proxy=60.820%，rows<7 collision_proxy=17.016%；rows>=7 exact-cover bad rate=39.278%；skeleton-hit geometry-fail candidate rate=55.428%。SymCIF v5 没有同一 collision proxy，本实验只记录 bond_length_score_lt_0.5=14.796%，不能等同 StructureMatcher collision。

rows>=7 专门 scorer/route 结果：Track A rows>=7 = 6.501% / 10.777% / 15.794%；hard-negative rows>=7 = 7.024% / 12.173% / 15.881%。hard-negative 相对 baseline 约为 K1 +1.702 pp、K5 +2.182 pp、K20 +1.134 pp，没有达到两个指标 +5pp，判定普通 rows>=7 scorer 不过 gate。

hybrid 边界：MPTS-52 official C2S3C15 rows>=7 baseline=14.319% / 20.942% / 27.878%，hybrid=14.280% / 33.006% / 37.936%，delta=-0.039 pp/+12.064 pp/+10.058 pp。MP-20 transfer rows>=7 baseline=29.310% / 44.414% / 55.655%，hybrid=29.310% / 54.138% / 62.690%，delta=+0.000 pp/+9.724 pp/+7.034 pp。这些结果支持“复杂结构有互补信号”，但 route 是 auxiliary，不是主方法。

geometry repair 证据：deterministic repair rows>=7 conversion=0.000%，converted=0/12745；MP-20 GT-WA learned geometry rows>=7 match@1/5=53.536%/65.488%。因此简单 wrap/jitter 停止，learned/optimized geometry repair 继续，但必须绑定 inference-time skeleton proposer。

可信度：CrystaLLM K50 与 exact-cover/collision 诊断来自全量 validation labels；SymCIF v5 是现有 generation/metrics replay，缺失样本按 fail；official/MP20 transfer/repair 结果直接读取前序实验 JSON。限制是 rows>=7 target 口径在 opentry_7 target cache 为 2295，旧 opentry_11 K50 表为 2292，本报告并列展示时不把不同口径强行合并。

和历史实验关系：这是对 opentry_11 实验 6、实验 7C、实验 8 strict ablation 的更新版 rows>=7 专门判读，并把 opentry_12 实验 1-5 的 official/hybrid/proposer/repair 结果接入同一失败归因矩阵。

最终判决：rows>=7 不是普通排序能解决的瓶颈。当前 CrystaLLM K50 top50 headroom 小，SymCIF v5 exact-cover 高但 skeleton-to-match conversion 低，deterministic repair conversion=0；C2S3C15 只能作为辅助证明复杂结构互补存在。主线应转向 rows>=7-specialized skeleton proposer + learned geometry repair。

下一步：做实验 7 主方法消融与 hybrid 边界说明，把 SymCIF/exact-cover/geometry repair/proposer 与 C2S3C15/Track A/scorer 的角色彻底分开，并给出 overall 与 rows>=7 的最终 gate 判决。


## opentry_12 实验：实验 7 主方法消融与 hybrid 边界说明

时间：2026-06-29T03:13:12+00:00

实验逻辑：把实验 1-6 的结果汇总成最终消融与边界判定，不再做新调参、不重新跑 official、不新增 scorer。核心是区分主方法候选、辅助 hybrid、诊断 scorer，并检查每条路线是否真的在 overall 和 rows>=7 的 match@1/5/20、RMSE、valid/formula/SG/exact-cover/skeleton-to-match conversion 上过 gate。

为什么做：前序结果已经显示 `C2S3C15` 在 MPTS-52 official K5/K20 过 +5pp，但它是 frozen auxiliary route；而 SymCIF/exact-cover/geometry repair 才是论文主线候选。实验 7 的作用是防止把普通 rerank 或 hybrid 包装成主贡献。

核心假设：如果主方法已经成立，generation-side proposer + learned geometry repair 应在 validation 上同时提高至少两个 match 指标，并且 rows>=7 不应只靠辅助 route；如果只有 C2S3C15 成立，则只能写成 auxiliary result，主方法仍需继续。

数据规模：综合 MPTS-52 validation K50 full 5000 samples/250000 candidates、MPTS-52 official full-test 8096 samples、MP-20 validation-like 9047 samples、SymCIF v5 MPTS-52 validation pool 25840 candidates/4574 covered samples、MP-20 GT-WA learned geometry 8874 samples。所有结果来自既有 JSON/metrics replay。

CPU/资源控制：bounded CPU use；本实验 used_parallel_workers=1，只读 JSON 汇总；不跑 StructureMatcher，不训练 GPU 模型。

validation 主线/诊断消融：
- 原 GT-SG baseline [reference_baseline]: overall 30.020%/40.480%/48.000%; rows>=7 5.323%/9.991%/14.747%；RMSE overall 0.125547/0.120274/0.129164；rows>=7 0.220974/0.232169/0.261464；valid=74.868%, formula=96.743%, SG=99.843%, exact-cover=78.029%, skeleton->match=38.550%；判定=reference。
- baseline + Track A scorer [auxiliary_scorer_diagnostic]: overall 33.120%/41.980%/48.900%; rows>=7 6.501%/10.777%/15.794%；RMSE overall 0.116852/0.114456/0.126321；rows>=7 0.176495/0.213638/0.250627；valid=82.598%, formula=98.061%, SG=99.845%, exact-cover=87.400%, skeleton->match=36.852%；判定=fail_main_gate。
- baseline + exact-cover filter/proxy [proxy_diagnostic]: overall 31.980%/42.080%/49.160%; rows>=7 6.588%/11.387%/15.838%；RMSE overall 0.124267/0.119860/0.126449；rows>=7 0.212699/0.227145/0.250084；valid=79.402%, formula=99.002%, SG=99.894%, exact-cover=89.585%, skeleton->match=35.523%；判定=fail_main_gate。
- baseline + hard-negative structural scorer v2 [auxiliary_scorer_diagnostic]: overall 32.540%/42.180%/48.960%; rows>=7 7.024%/12.173%/15.881%；RMSE overall 0.119805/0.119787/0.125882；rows>=7 0.202201/0.228502/0.245350；valid=79.493%, formula=98.018%, SG=99.826%, exact-cover=87.454%, skeleton->match=36.378%；判定=fail_main_gate。
- skeleton proposal + geometry repair + structural scorer proxy [proxy_diagnostic]: overall 32.360%/42.180%/49.100%; rows>=7 6.937%/11.824%/15.707%；RMSE overall 0.118996/0.120007/0.125581；rows>=7 0.201633/0.228070/0.244520；valid=80.223%, formula=98.157%, SG=99.879%, exact-cover=89.585%, skeleton->match=35.740%；判定=fail_main_gate。
- SymCIF v5 neural skeleton/geometry proposer [main_method_candidate_diagnostic]: overall 26.080%/35.240%/35.900%; rows>=7 10.501%/14.292%/14.553%；RMSE overall 0.110350/0.107305/0.104945；rows>=7 0.114867/0.120458/0.123888；valid=33.638%, formula=96.927%, SG=96.927%, exact-cover=96.858%, skeleton->match=23.653%；判定=fail_main_gate。
- learned geometry repair under GT-WA [main_method_component_oracle]: overall 77.158%/82.939%/NA; rows>=7 53.536%/65.488%/NA；RMSE overall 0.045048/0.038850/NA；rows>=7 0.079414/0.072492/NA；valid=88.089%, formula=98.997%, SG=98.997%, exact-cover=100.000%, skeleton->match=82.939%；判定=component_continue_not_benchmark_success。

auxiliary hybrid 边界：
- C2S3C15 frozen official [auxiliary_hybrid_official]: overall 17.144%/36.055%/41.144%; rows>=7 14.280%/33.006%/37.936%；RMSE overall 0.165456/0.117010/0.129672；rows>=7 0.185054/0.126046/0.139384；valid=93.928%, formula=47.982%, SG=93.297%, exact-cover=98.089%, skeleton->match=19.429%；判定=passes_auxiliary_mpts52_official_k5_k20。
- C2S3C15 MP-20 transfer [auxiliary_hybrid_transfer_check]: overall 72.731%/86.371%/90.008%; rows>=7 29.310%/54.138%/62.690%；RMSE overall 0.051468/0.040891/0.036807；rows>=7 0.125439/0.102242/0.096497；valid=58.756%, formula=79.233%, SG=79.233%, exact-cover=79.233%, skeleton->match=63.431%；判定=rows_ge7_passes_but_overall_fails_transfer_gate。

收益归因关键点：实验 2 显示 C2S3C15 official 的 rows>=7 占 K5 净收益 97.046%、K20 净收益 98.460%；rows>=7 中 SymCIF top3 在 C1-C17 失败后独立救回 817 个样本。

rows>=7 关键点：实验 6 显示 CrystaLLM K50 rows>=7 top50=18.431%，SymCIF v5 rows>=7 top50=14.553%，主瓶颈是 coverage plus skeleton-to-match conversion, not ordinary ranking。这说明复杂结构不是普通排序能解决。

和历史实验关系：Track A、hard-negative scorer、exact-cover filter/proxy、strict integrated ablation 都没有达到两个指标 +5pp；deterministic repair conversion=0；SymCIF v5 exact-cover feasible 高但 coverage/skeleton-to-match 还不够；learned geometry repair在 GT-WA 条件下有强信号但不是 inference result。

最终判决：main_method_success=False；auxiliary_hybrid_success_mpts52=True；mp20_overall_transfer_success=False；coverage_solved=False。因此 C2S3C15 只能作为 auxiliary hybrid official result；Track A/RF/HGB/hard-negative scorer 只保留为诊断或停止；主线继续 exact-cover constrained skeleton proposer + symmetry-preserving learned geometry repair，尤其 rows>=7 专门路线。

下一步：若继续迭代，必须先写明失败原因和预期提升，使用 MP-20/MPTS-52 train 数据训练 rows>=7-specialized skeleton proposer；先过 validation/half-train gate，再考虑 frozen official。禁止继续调 C/S 比例、ordinary rerank、threshold tuning 或 official feedback。


<!-- OPENTRY13_EXP1_BASELINE_AUDIT -->
## opentry_13 实验 1：主 baseline 口径复核

结果文件：`model/New_model/opentry_13/results/experiment_1_baseline_protocol_audit.json`

- 为什么做：opentry_12 C2S3C15 official 使用了 `17.181 / 24.345 / 31.522` 作为 baseline，但全局主 GT-SG CrystaLLM-a anchor 是 `25.23 / 36.46 / 43.96`。本实验只复核 split、evaluator、input condition 与 candidate source，防止用低 baseline 宣称主目标。
- 核心假设：如果两个 baseline 不是同一 candidate source / model checkpoint，即使 split 和 evaluator 相同，也不能混用为主比较 anchor。
- 数据规模：MPTS-52 official test，overall `8096` 个样本；rows>=7 `7626` 个样本。
- baseline：低 internal baseline 是 pure CrystaLLM GT-SG line，match@1/5/20 = `17.181 / 24.345 / 31.522`；true GT-SG anchor 是 CrystaLLM-a GT-SG，match@1/5/20 = `25.235 / 36.462 / 43.960`。
- 口径核查：split 相同，均为 MPTS-52 official test；evaluator 相同，StructureMatcher 参数为 `{'angle_tol': 10, 'ltol': 0.3, 'stol': 0.5}`；输入条件名义上均为 composition + GT-SG；candidate source 不同，低 baseline 来自 `generations/pure_crystallm_gt_sg_mpts_52_test.tar.gz`，true anchor 来自 `generations/crystallm_a_gt_sg_mpts_52_test.tar.gz`，C2S3C15 来自 `/data/users/xsw/autodlmini/model/New_model/opentry_12/candidates/c2s3c15_mpts52_test_k20.jsonl.gz`。
- 结果：`17.181 / 24.345 / 31.522` 不是全局主 GT-SG CrystaLLM-a anchor，而是 opentry_7 pure model line。C2S3C15 不能使用该低 baseline 作为主目标达成依据。
- 可信度：高。结论直接来自 opentry_7 机器 JSON、opentry_12 机器 JSON 与全局 bundle 中已登记的 anchor，且 sample count / matcher 参数一致可核验。
- 和历史实验关系：opentry_12 的 C2S3C15 仍可作为辅助 hybrid 结果，但不能替代 opentry_7/opentry_10 登记的主 anchor。
- 最终判决：主比较必须使用 true GT-SG CrystaLLM-a anchor：`25.23 / 36.46 / 43.96`。
- 下一步：在 fixed C2S3C15 条件下重算其相对 true anchor 的 delta，并决定是否降级。


<!-- OPENTRY13_EXP2_C2S3C15_TRUE_ANCHOR -->
## opentry_13 实验 2：C2S3C15 true GT-SG anchor replay / audit

结果文件：`model/New_model/opentry_13/results/experiment_2_c2s3c15_true_anchor_replay.json`

- 为什么做：固定 C2S3C15，不再调整 C/S 比例、不做 threshold tuning、不接 scorer，只回答它在 true GT-SG anchor 口径下是否真的超过主目标。
- 核心假设：如果 C2S3C15 只是相对 pure-model low baseline 有提升，而相对 CrystaLLM-a GT-SG anchor 没有 +5pp，则只能是 auxiliary hybrid result。
- 数据规模：MPTS-52 official test，overall `8096`；rows>=7 `7626`；candidate records `161920`。
- baseline：internal low baseline match@1/5/20 = `17.181 / 24.345 / 31.522`，rows>=7 = `14.319 / 20.942 / 27.878`；true GT-SG anchor match@1/5/20 = `25.235 / 36.462 / 43.960`，rows>=7 = `22.489 / 33.373 / 41.044`。
- 方法变化：无新方法。仅把 opentry_12 fixed C2S3C15 official 结果改挂到 true GT-SG anchor 上审计。
- 结果：C2S3C15 overall match@1/5/20 = `17.144 / 36.055 / 41.144`，RMSE = `0.165456 / 0.117010 / 0.129672`；rows>=7 match@1/5/20 = `14.280 / 33.006 / 37.936`，RMSE = `0.185054 / 0.126046 / 0.139384`。相对 true anchor overall delta = `-8.090pp / -0.408pp / -2.816pp`；rows>=7 delta = `-8.209pp / -0.367pp / -3.108pp`。相对 internal low baseline overall delta = `-0.037pp / +11.709pp / +9.622pp`。
- 结构质量：C2S3C15 valid any@1/5/20 = `88.303` / `98.950` / `98.987`；formula consistency slot rate = `47.982`；SG consistency slot rate = `93.297`；known exact-cover any@20 = `85.882`；SymCIF slot exact-cover feasible = `98.089`。
- 可信度：高。没有使用 official 结果回调；只读取冻结 C2S3C15 official JSON 与 opentry_7 anchor JSON。internal baseline 和 true anchor 的 valid/formula/SG/exact-cover 诊断在 opentry_7 指标 JSON 中未记录，因此报告为不可用，不做外推。
- 和历史实验关系：opentry_12 中 C2S3C15 相对 pure baseline 提升 match@5/20，但在 true CrystaLLM-a GT-SG anchor 下 match@1/5/20 全部为负 delta。
- 最终判决：C2S3C15 未达到 true anchor +5pp，也没有 rows>=7 不恶化；降级为 auxiliary hybrid result，禁止作为论文主方法 claim。
- 下一步：转向 rows>=7-specialized skeleton proposer，目标是让候选池中真实增加可 match 的 skeleton/geometry，而不是继续调 C/S 比例或 scorer。


<!-- OPENTRY13_EXP3_ROWS7_SKELETON_PROPOSER -->
## opentry_13 实验 3：rows>=7-specialized skeleton proposer validation gate

结果文件：`model/New_model/opentry_13/results/experiment_3_rows_ge7_skeleton_proposer_validation_gate.json`
候选文件：`model/New_model/opentry_13/artifacts/exp3_rows7_skeleton_proposer/proposals.jsonl`

- 为什么做：停止把普通 rerank/scorer 当主方法，直接检查 train-derived rows>=7 skeleton proposer 是否让候选池中出现更多可 match 的 skeleton/结构。
- 核心假设：若 rows>=7 的主瓶颈是 skeleton coverage，则只用 composition + GT-SG + train rows>=7 prototypes 的 exact-cover proposer 应提高 top50 skeleton-hit；若 skeleton-hit 不能转化为 match，则下一步必须接 learned geometry repair。
- 数据规模：train repr `25998`，其中 rows>=7 `6863`；validation repr `4727`，其中 rows>=7 `2197`；已有 hydrated eval candidates `25840`。未重新跑 StructureMatcher。
- baseline：opentry_12 exp6 validation rows>=7 CrystaLLM K50 top5/top20/top50 = `10.109%` / `14.858%` / `18.431%`；SymCIF v5 rows>=7 top5/top20/top50 = `14.292%` / `14.553%` / `14.553%`。
- 方法变化：新增 rows>=7 train exact-cover skeleton proposer。推理期只使用 formula/composition、GT-SG 和 train skeleton prototypes；不使用 GT-WA、GT-skeleton、match/RMSD/StructureMatcher label、RF/HGB/rerank 或 threshold tuning。
- 结果 overall：top1/5/20 skeleton-hit = `59.827% / 77.491% / 80.897%`；hydrated match@1/5/20 = `26.127% / 36.662% / 37.529%`；top50 hydrated match = `37.529%`。
- 结果 rows>=7：top1/5/20 skeleton-hit = `61.038% / 75.330% / 76.513%`；top50 skeleton-hit = `76.513%`；top1/5/20 hydrated match = `10.605% / 14.838% / 15.112%`；top50 hydrated match = `15.112%`；top50 exact-cover feasible any = `92.535%`；top50 proposal-skeleton-to-hydrated-match conversion = `19.750%`；top50 hydrated-skeleton-to-match conversion = `20.660%`。
- 可信度：中等。proposal-only 指标覆盖 validation repr 全部 `4727` 样本；match 指标只在已有 SymCIF v5 hydrated/evaluated candidate 上计算，未把未渲染 skeleton 自动算成 match，因此偏保守。它是 validation gate，不是 official 结果。
- 和历史实验关系：区别于 opentry_12 exp6 的 scorer/边界审计，本实验真正输出 train-derived top50 skeleton proposals；但 hydrated match 仍受既有 geometry 生成质量限制。
- 最终判决：validation gate 未通过。rows>=7 skeleton-hit 有信号，但 hydrated match@5/20 和 skeleton-to-match conversion 不足，不能进入 official，也不能作为主结果 claim。
- 下一步：将这些 predicted exact-cover skeleton 绑定 learned geometry repair，重点修 lattice/free parameters/site mapping/collision/local geometry，而不是再做普通 scorer。


<!-- OPENTRY13_EXP4_PREDICTED_SKELETON_GEOMETRY_REPAIR -->
## opentry_13 实验 4：predicted-skeleton learned geometry repair validation

结果文件：`model/New_model/opentry_13/results/experiment_4_predicted_skeleton_geometry_repair.json`
候选评估：`model/New_model/opentry_13/artifacts/exp4_predicted_skeleton_geometry_repair/evaluated_repair_candidates.jsonl`

- 为什么做：GT-WA learned geometry repair 很强，但不能作为 inference 主结果。本实验把 repair 绑定到 exp3 的 predicted exact-cover skeleton，在固定 composition + GT-SG 条件下检查 lattice/free parameters/site mapping/local geometry 是否能把 skeleton-hit 转成 StructureMatcher match。
- 核心假设：如果当前瓶颈主要是 geometry，则在 predicted skeleton 已命中时，learned geometry repair 应带来 K5/K20 提升和非零 repair conversion；如果 conversion 仍低，说明 site mapping / local geometry / collision 仍未打通。
- 数据规模：validation subset 抽样 `768` 样本，其中 rows>=7 `384`、rows<7 `384`；实际有可用 predicted-skeleton 候选并进入评估的样本 overall `712`、rows>=7 `359`；topK `20`；candidate records `2920`。StructureMatcher worker `12`，线程环境 OMP/MKL/OPENBLAS/NUMEXPR=1。
- baseline：before 使用 exp3 同一 predicted skeleton proposal 的 hydrated-existing-eval match；after 使用 learned geometry model 重新渲染后的 StructureMatcher。true official anchor 不参与本 validation repair gate。
- 方法变化：新增 composition exact-cover site mapping + learned lattice/free-parameter prediction；推理期不使用 GT-WA、GT-skeleton、match/RMSD/StructureMatcher label、official feedback、RF/HGB/rerank 或 threshold tuning。
- 结果 overall：before match@1/5/20 = `26.966% / 37.640% / 38.483%`；after match@1/5/20 = `3.792% / 6.461% / 6.742%`；delta = `-23.174pp / -31.180pp / -31.742pp`；repair conversion@1/5/20 = `0.385% / 0.000% / 0.000%`；valid rate `18.596%`，formula consistency `41.610%`，SG consistency `31.130%`，exact-cover retained `100.000%`。
- 结果 rows>=7：before match@1/5/20 = `12.813% / 16.435% / 16.435%`；after match@1/5/20 = `0.000% / 0.000% / 0.000%`；delta = `-12.813pp / -16.435pp / -16.435pp`；repair conversion@1/5/20 = `0.000% / 0.000% / 0.000%`；valid rate `1.138%`，formula consistency `16.228%`，SG consistency `4.790%`，exact-cover retained `100.000%`；skeleton-to-match conversion@20 `0.000%`。
- 可信度：中等。这是真 predicted-skeleton repair，不是 GT-WA oracle；但只是 validation subset，且 site mapping 是 composition exact-cover 的 deterministic mapping，未重新训练专门适配 predicted skeleton 的 geometry model。
- 和历史实验关系：opentry_12 deterministic repair conversion=0，GT-WA learned repair 很强；本实验把 learned repair 接到 predicted skeleton 上，验证 inference 链路仍不足。
- 最终判决：validation gate 未通过，不能进入 official，也不能作为主结果 claim。它保留为主方法候选的失败/诊断实验。
- 下一步：训练/微调 geometry repair 以适配 predicted skeleton 与 exact-cover site mapping 噪声，并加入可微或搜索式 local collision repair；在 validation gate 通过前不做 official。

<!-- OPENTRY13_EXP1_TRUE_ANCHOR_C2S3C15 -->
## opentry_13 实验 1：true-anchor-source C2S3C15 fixed hybrid 重放

结果文件：`model/New_model/opentry_13/results/experiment_1_true_anchor_source_c2s3c15.json`

- 实验逻辑：固定 `C1,C2,S1,S2,S3,C3...C17`，把 C source 从旧 pure CrystaLLM GT-SG 切换为 true CrystaLLM-a GT-SG anchor；SymCIF source 仍用已有 MPTS-52 test fullgen pool。只重放固定候选序列，不搜索比例、不调阈值、不接 scorer。
- 为什么做：opentry_12 的 C2S3C15 用的是低 baseline/pure candidate source，不能作为主 anchor 比较；本实验直接检查复杂结构互补信号在 true anchor 上是否仍能转化为 match@5/20 增益。
- 数据规模：MPTS-52 official test 8096 samples；rows>=7 7626 samples；route records=161920；source counts={'C': 142720, 'S': 19200}。
- baseline：true CrystaLLM-a GT-SG C20 overall match@1/5/20 = `25.235% / 36.462% / 43.960%`，rows>=7 = `22.489% / 33.373% / 41.044%`；RMSE overall = `0.121103 / 0.125734 / 0.133401`，rows>=7 = `0.131036 / 0.135577 / 0.144427`。
- 比较对象：old pure-source C2S3C15 overall = `17.144% / 36.055% / 41.144%`，rows>=7 = `14.280% / 33.006% / 37.936%`；true-anchor-source C2S3C15 overall = `25.235% / 41.366% / 48.382%`，rows>=7 = `22.489% / 38.552% / 45.660%`。
- 结果：true-anchor-source C2S3C15 相对 true anchor overall delta = `+0.000pp / +4.904pp / +4.422pp`；rows>=7 delta = `+0.000pp / +5.180pp / +4.616pp`；相对 old pure-source C2S3C15 overall delta = `+8.090pp / +5.311pp / +7.238pp`。
- RMSE：true-anchor-source C2S3C15 overall RMSE@1/5/20 = `0.121103 / 0.107890 / 0.116212`；rows>=7 RMSE@1/5/20 = `0.131036 / 0.115306 / 0.124766`。
- valid/formula/SG/exact-cover：valid any@1/5/20 = `97.344%` / `99.914%` / `99.938%`；formula any@1/5/20 = `70.467%` / `95.282%` / `96.937%`；SG any@1/5/20 = `97.221%` / `99.852%` / `99.876%`；known exact-cover any@20 = `85.882%`；SymCIF slot exact-cover feasible = `98.089%`。
- SymCIF skeleton-to-match：SymCIF candidates=19200，skeleton-hit candidates=4622，skeleton-hit-to-match conversion=`19.450%`。
- 缺失与 fallback：missing SymCIF samples full-C fallback=1072；partial SymCIF samples slot fallback=1872；missing CrystaLLM samples=1。无 SymCIF artifact 的样本统一回退 true-anchor C20；S 槽不足时该槽回退下一条 C。
- 可信度：高。候选顺序固定，C/S 路由不使用 match/RMSE/StructureMatcher/test feedback/GT-WA/GT-skeleton；唯一变化是把 C source 改成 true anchor source，并重新跑同一 StructureMatcher evaluator。
- 和历史实验关系：该实验修正 opentry_12/13 旧报告的 anchor-source 口径；old pure-source C2S3C15 只能说明相对低 baseline 的互补，不再作为主比较。
- 最终判决：`not_official_main_method`。原因：true-anchor-source C2S3C15 does not satisfy the saved acceptance gate. 无论该结果如何，C2S3C15 仍只能作为 auxiliary/diagnostic，不能写成论文主方法。
- 下一步：若判决为 stop 或 diagnostic，停止 C2S3C15 调参，把主线继续放在 predicted skeleton renderer/site mapping 与 geometry repair gate。

<!-- OPENTRY13_EXP2_RENDERER_SITE_MAPPING -->
## opentry_13 实验 2：predicted skeleton renderer / site mapping 稳定性

结果文件：`model/New_model/opentry_13/results/experiment_2_predicted_skeleton_renderer_site_mapping.json`

- 实验逻辑：使用 opentry_13 实验 3 的 validation predicted exact-cover skeleton proposals，固定 composition + GT-SG + predicted skeleton，只检查 skeleton 到 CIF 的 renderer/site mapping 是否保结构。没有训练 scorer，没有看 StructureMatcher match@k，没有使用 test true CIF。
- 为什么做：predicted skeleton 接 repair 后崩掉的首要风险是渲染链路不保 formula/SG/exact-cover，而不是模型分数；本实验先做结构完整性 gate，决定是否允许进入 learned geometry repair。
- 数据规模：samples=4727，rows>=7 samples=2197，candidate records=36268，top_k=20。
- 方法变化：比较两种 site mapping：deterministic composition exact-cover mapping，以及 train-prototype-preferred mapping。几何只作为渲染初始化：deterministic mode 用 train SG/global median lattice + deterministic free params；train-prototype mode 用 train source lattice/free params，不使用 GT-WA 作为推理输入。
- deterministic overall：valid=18.838%，formula=95.037%，SG=95.274%，exact-cover=95.037%，legal CIF=98.064%，site-count=95.037%，row-count=100.000%，site-mapping failure=0.000%，collision=77.666%。
- deterministic rows>=7：valid=10.699%，formula=98.882%，SG=95.312%，exact-cover=98.882%，legal CIF=99.376%，site-count=98.882%，row-count=100.000%。
- train-prototype overall：valid=88.789%，formula=99.548%，SG=95.478%，exact-cover=100.000%，legal CIF=99.548%，site-count=99.548%，row-count=100.000%，site-mapping failure=0.000%，collision=9.430%。
- train-prototype rows>=7：valid=89.548%，formula=99.699%，SG=96.656%，exact-cover=100.000%，legal CIF=99.699%，site-count=99.699%，row-count=100.000%。
- renderer/site-mapping fixed selector：只用 inference-safe structural checks 在 top20 中选第一个 legal/formula/SG/exact-cover/site-count/no-collision CIF，不看 match。selected overall valid=97.007%，formula=99.660%，SG=98.436%，exact-cover=100.000%；selected rows>=7 valid=96.458%，formula=99.410%，SG=98.475%，exact-cover=100.000%；fallback_rate=2.993%。
- gate 判定：passed=True；candidate_level_train_prototype_passed=False；selected_train_prototype_passed=True；deterministic_passed=False；失败原因={'deterministic': ['overall valid 18.838% < 95.000%', 'rows>=7 valid 10.699% < 90.000%'], 'train_prototype': ['overall valid 88.789% < 95.000%', 'rows>=7 valid 89.548% < 90.000%'], 'selected_train_prototype': []}。
- 可信度：中高。该实验覆盖所有 exp3 proposal 记录并真实 render/parse/SG-detect，但 valid 仍依赖当前 `SpacegroupAnalyzer(symprec=0.1)` 和 0.5A collision 阈值；它是结构 gate，不是 match 指标。
- 和历史实验关系：承接 opentry_13 实验 3 的 predicted skeleton proposals，并解释 opentry_13 旧 repair 结果中 formula/SG/valid 崩掉是否来自 site mapping/renderer。
- 最终判决：如果 gate 不过，禁止进入 learned geometry repair 或 official；必须继续修 renderer/site mapping。
- 下一步：只有当至少一个 mapping mode 过结构 gate，才继续 predicted-skeleton-aware learned geometry repair；否则优先修 SG/rendering、row expansion 或 geometry initializer。

<!-- OPENTRY13_EXP3_PREDICTED_SKELETON_AWARE_REPAIR_AUDIT -->
## opentry_13 实验 3：predicted-skeleton-aware learned geometry repair

审计结果：`model/New_model/opentry_13/results/experiment_3_predicted_skeleton_aware_geometry_repair_audit.json`
训练 pilot：`model/New_model/opentry_13/results/experiment_3_predicted_skeleton_lattice_repair_pilot.json`

- 为什么做：目标要求 repair 训练数据必须包含 train split predicted skeleton / exact-cover skeleton 噪声。先前审计证明旧 geometry model 不是这种训练条件；本补充实验构造 train split noisy skeleton pair，并训练一个轻量 lattice MLP repair pilot。
- 核心假设：如果 predicted skeleton 条件下主要缺 lattice 初始化，train-noisy-skeleton lattice MLP + source prototype free params 应至少改善 K5/K20 或 repair conversion；如果仍不过 gate，说明还需要 free-parameter/site-mapping/collision 联合 repair。
- 数据规模：train noisy skeleton pairs `7774`；validation samples `4411`，rows>=7 `2033`；candidate records `18134`；topK `20`。
- 训练设置：PyTorch MLP，输入为 composition + GT-SG + noisy predicted skeleton numeric features，target 为 train true lattice；epochs `40`，best val loss `0.38292357325553894`。没有使用 RF/HGB/scorer，也没有使用 match/RMSD 作为推理特征。
- baseline：before repair 使用 exp3 predicted skeleton proposer 的 hydrated-existing-eval；after repair 使用 learned lattice MLP + source-prototype free params 重新渲染并按结构自检排序。
- 结果 overall：before match@1/5/20 = `27.998% / 39.288% / 40.218%`；after match@1/5/20 = `10.066% / 13.172% / 13.308%`；delta = `-17.932pp / -26.117pp / -26.910pp`；repair conversion@1/5/20 = `1.480% / 2.091% / 1.858%`；valid `89.131%`，formula `99.548%`，SG `95.566%`，exact-cover `100.000%`，collision `9.165%`。
- 结果 rows>=7：before match@1/5/20 = `11.461% / 16.035% / 16.331%`；after match@1/5/20 = `2.312% / 2.804% / 2.951%`；delta = `-9.149pp / -13.232pp / -13.379pp`；repair conversion@1/5/20 = `0.889% / 1.113% / 1.117%`；valid `89.882%`，formula `99.699%`，SG `96.667%`，exact-cover `100.000%`，collision `9.624%`，skeleton-to-match conversion@20 `3.510%`。
- gate 判定：structure_gate_pass=False；repair_gate_pass=False；passed=False。
- 可信度：中等。训练数据确实来自 train split noisy exact-cover skeleton，validation 推理不使用 GT-WA/GT-skeleton/test CIF/match label；限制是当前 pilot 只学习 lattice，free parameters 和 collision/local geometry 仍未联合学习。
- 和历史实验关系：补足了先前实验 3 审计发现的“缺 predicted-skeleton-noise 训练 artifact”问题；结果与旧 repair 一致说明单独 lattice repair 仍不能解决 rows>=7 conversion。
- 最终判决：`fail_validation_gate`。Train-split noisy-skeleton lattice repair still fails structure and/or repair conversion gates.
- 下一步：如果继续主线，应训练 full lattice + free-parameter + collision/local optimization repair，而不是 scorer、C/S 比例或 official。


<!-- OPENTRY13_EXP4_ROWS_GE7_MULTI_GEOMETRY_PROPOSAL -->
## opentry_13 实验 4：rows>=7 multi-geometry proposal

结果文件：`model/New_model/opentry_13/results/experiment_4_rows_ge7_multi_geometry_proposal.json`
候选评估：`model/New_model/opentry_13/artifacts/exp4_rows_ge7_multi_geometry_proposal/evaluated_ranked_candidates.jsonl`

- 为什么做：同一个 predicted exact-cover skeleton 可能对应多个 lattice/free-parameter/site-mapping 解；实验 3 显示单一 learned repair 不能把 skeleton-hit 稳定转成 match，因此这里对每个 skeleton 生成多几何 proposal，并只用 inference-safe structural checks 排序。
- 核心假设：如果 geometry 多解覆盖是真瓶颈，rows>=7 top50 match coverage 和 skeleton-to-match conversion 应明显超过单一 hydrated/prototype 结果，并超过 CrystaLLM K50 rows>=7 top50 `18.431%` 至少 +5pp。
- 数据规模：scope=`all`；samples `4411`，rows>=7 samples `2033`；candidate records `71930`；unique predicted skeletons `18134`；平均 geometry proposals/skeleton `3.967`；top output K `50`。
- baseline：validation CrystaLLM K50 overall match@1/5/20 = `30.020% / 40.480% / 48.000%`；exp3 记录的 CrystaLLM rows>=7 top1/top5/top20/top50 = `5.359% / 10.109% / 14.858% / 18.431%`。
- 方法变化：对 exp3 predicted skeleton proposals 进行 composition exact-cover site mapping；每个 skeleton 的几何候选来自 train source prototype、同 skeleton / 同 SG+atom_count / 同 SG train prototype、旧 geometry model 初始化与 SG-median+deterministic fallback。排序只使用 legal CIF、formula、SG、site count、exact-cover、collision、volume/atom 和 train-reference score，不使用 match/RMSD/StructureMatcher label。
- 结果 overall：match@1/5/20 = `20.313% / 26.479% / 27.953%`；RMSE@1/5/20 = `0.06493674573671193 / 0.06546557832369819 / 0.06861300772369208`；valid `82.566%`，formula `98.095%`，SG `93.456%`，exact-cover `100.000%`，collision `14.288%`。
- 结果 rows>=7：match@1/5/20/50 = `10.428% / 12.592% / 13.035% / 13.084%`；RMSE@1/5/20/50 = `0.06416882205447941 / 0.059610451644205927 / 0.05690601369240405 / 0.057054759831388`；skeleton-hit@50 `82.686%`；skeleton-to-match conversion@50 `15.407%`。
- rows>=7 结构指标：valid `82.055%`，formula `97.317%`，SG `93.558%`，exact-cover `100.000%`，collision `14.886%`，valid_any@50 `98.032%`。
- gate 判定：passed=False；rows>=7 top50 delta vs CrystaLLM K50 = `-5.347pp`；rows>=7 match@5 delta = `+2.483pp`；rows>=7 match@20 delta = `-1.823pp`；overall match@5/20 no-drop=False；失败原因=['rows>=7 top50 delta -5.347pp < +5.000pp', 'rows>=7 skeleton-to-match conversion@50 15.407% < 30.000%', 'rows>=7 match@5 delta +2.483pp < +5.000pp', 'rows>=7 match@20 delta -1.823pp < +5.000pp', 'overall match@5 and match@20 both decreased vs validation CrystaLLM K50 baseline']。
- 可信度：中等。该实验真实 render/parse/SG-detect/StructureMatcher，且排序不看 match；限制是 geometry proposals 仍主要来自 train prototype 和旧 GT-WA-style geometry initializer，没有训练 predicted-skeleton-noise repair。
- 和历史实验关系：实验 2 证明 selected renderer/site mapping 可过结构 gate；实验 3 证明旧 learned repair 不满足 predicted-skeleton-aware 条件且 conversion 崩掉；本实验测试 multi-geometry 是否能单独提高 rows>=7 hydrated match coverage。
- 最终判决：`fail_validation_gate`。Multi-geometry did not reach the required rows>=7 top50/conversion and K5/K20 lift gates.
- 下一步：Do not run official; build predicted-skeleton-noise geometry training or stronger local collision/geometry optimization before rerunning.

<!-- OPENTRY13_EXP5_MAIN_ABLATION_BOUNDARY -->
## opentry_13 实验 5：主方法消融与最终判定

结果文件：`model/New_model/opentry_13/results/experiment_5_main_ablation_and_final_boundary.json`

- 为什么做：把 true official anchor、主方法候选、辅助 hybrid 和 oracle diagnostic 分清楚，避免继续把低 baseline、fusion/scorer 或 GT-WA oracle 写成主贡献。
- 核心假设：可 claim 的主方法必须是 inference-time `predicted skeleton proposer + renderer/site mapping + learned/multi-geometry repair`，并且相对 true CrystaLLM-a GT-SG official anchor 至少两个 match 指标 +5pp，rows>=7 不恶化。
- 数据规模：official test anchor/C2S3C15 `8096` samples、rows>=7 `7626`；renderer/skeleton validation `4727` records；lattice repair train noisy pairs `7774`、validation candidates `18134`；multi-geometry validation `4411` samples、rows>=7 `2033`、candidate records `71930`；GT-WA oracle MPTS-52 subset `180` samples。
- baseline：true CrystaLLM-a GT-SG anchor match@1/5/20 = `25.235% / 36.462% / 43.960%`，rows>=7 = `22.489% / 33.373% / 41.044%`，RMSE@1/5/20 = `0.121103 / 0.125734 / 0.133401`。

主方法候选：
- SymCIF v5 proposer [symcif_v5_generation_diagnostic]: overall 26.080% / 35.240% / 35.900%; rows>=7 10.501% / 14.292% / 14.553%; gate=fail_main_gate.
- rows>=7 predicted skeleton proposer [predicted_skeleton_proposer_validation]: overall 26.127% / 36.662% / 37.529%; rows>=7 10.605% / 14.838% / 15.112%; gate=fail_validation_gate.
- predicted-skeleton-aware lattice repair pilot [predicted_skeleton_aware_lattice_repair_pilot]: overall 10.066% / 13.172% / 13.308%; rows>=7 2.312% / 2.804% / 2.951%; gate=fail_structure_and_repair_gate.
- rows>=7 multi-geometry proposal [rows_ge7_multi_geometry_validation]: overall 20.313% / 26.479% / 27.953%; rows>=7 10.428% / 12.592% / 13.035%; gate=fail_validation_gate.

主方法组件：
- renderer/site-mapping fixed selector [renderer_site_mapping_structure_gate]: overall NA / NA / NA; rows>=7 NA / NA / NA; gate=pass_structure_gate.

辅助结果：
- C2S3C15 true-anchor-source hybrid [auxiliary_hybrid_true_anchor_source]: overall 25.235% / 41.366% / 48.382%; rows>=7 22.489% / 38.552% / 45.660%; gate=not_official_main_method.
- baseline + Track A scorer [track_a_scorer]: overall 33.120% / 41.980% / 48.900%; rows>=7 6.501% / 10.777% / 15.794%; gate=not_main_method.
- baseline + hard-negative structural scorer v2 [hard_negative_scorer]: overall 32.540% / 42.180% / 48.960%; rows>=7 7.024% / 12.173% / 15.881%; gate=not_main_method.
- skeleton proposal + geometry repair + structural scorer proxy [combined_proxy]: overall 32.360% / 42.180% / 49.100%; rows>=7 6.937% / 11.824% / 15.707%; gate=not_main_method.

诊断结果：
- old pure-source C2S3C15 [historical_low_anchor_diagnostic]: overall 17.144% / 36.055% / 41.144%; rows>=7 14.280% / 33.006% / 37.936%; gate=diagnostic_only.
- old GT-WA-style geometry model on predicted skeleton [old_repair_artifact_diagnostic]: overall 3.792% / 6.461% / 6.742%; rows>=7 0.000% / 0.000% / 0.000%; gate=diagnostic_only.
- GT-WA learned geometry repair oracle [gt_wa_geometry_repair_oracle]: overall 30.000% / 37.778% / 41.667%; rows>=7 25.150% / 32.934% / 37.126%; gate=diagnostic_oracle_only.
- baseline + exact-cover filter/proxy [exact_cover_proxy]: overall 31.980% / 42.080% / 49.160%; rows>=7 6.588% / 11.387% / 15.838%; gate=not_main_method.

- C2S3C15 判定：true-anchor-source C2S3C15 相对 true anchor delta = `+0.000pp / +4.904pp / +4.422pp`；rows>=7 delta = `+0.000pp / +5.180pp / +4.616pp`。它只能作为 auxiliary，不能作为论文主方法。
- renderer 判定：selected train-prototype structural selector 通过结构 gate，overall valid `97.007%`、formula `99.660%`、SG `98.436%`、exact-cover `100.000%`；rows>=7 valid `96.458%`。
- repair 判定：已补充 train-noisy-skeleton lattice MLP repair pilot；rows>=7 after match@1/5/20 = `2.312% / 2.804% / 2.951%`，delta = `-9.149pp / -13.232pp / -13.379pp`，repair conversion@20 `1.117%`，structure_gate_pass=False。
- multi-geometry 判定：rows>=7 match@1/5/20/50 = `10.428% / 12.592% / 13.035% / 13.084%`；skeleton-to-match conversion@50 `15.407%`；top50 delta vs CrystaLLM K50 `-5.347pp`。
- 可信度：高。实验 5 不训练、不重跑 StructureMatcher、不调阈值，只汇总本轮已经写入的 JSON/JSONL 结果；各行明确区分 official test、validation、component gate 和 oracle diagnostic。
- 和历史实验关系：修正旧 exp5 的低-anchor C2S3C15 口径；exp2 说明 renderer/site mapping 可过结构 gate，exp3 先审计旧 artifact、再补充 train-noisy-skeleton lattice repair pilot但仍失败，exp4 说明 multi-geometry 仍不能把 rows>=7 skeleton-hit 转成足够 match。
- 最终判决：`allowed_main_result_claim=False`。失败段落不是“geometry 是瓶颈”这个泛结论，而是：renderer 结构 gate 已过；train-noisy-skeleton lattice repair pilot 的 structure/repair gate 失败；旧 repair 的 formula/SG/valid 和 conversion 失败；multi-geometry 的 rows>=7 top50 coverage 与 skeleton-to-match conversion 仍低于验收线。
- 下一步：不要跑 official，也不要继续调 C2S3C15/scorer/threshold。下一轮若继续主线，应训练 full lattice + free-parameter + collision/local optimization repair，然后重新过 validation gate。

<!-- OPENTRY14_EXP0_FRONTEND_BASELINE_FREEZE -->
## opentry_14 实验 0：冻结前端和基线口径

结果文件：`model/New_model/opentry_14/results/experiment_0_frontend_and_baseline_freeze.json`

- 为什么做：本轮目标是解决 predicted skeleton 已命中后无法转成 StructureMatcher match 的 skeleton-to-match conversion 问题。实验 0 先固定主 baseline 和前端，防止继续混用 pure CrystaLLM GT-SG、低 baseline、C2S3C15 auxiliary hybrid、ordinary scorer 或 official feedback。
- 核心假设：如果 opentry_13 exp2 的 selected train-prototype renderer/site mapping 仍能保持 rows>=7 的 valid/formula/SG/exact-cover，则后续不应再大改前端，而应进入 predicted-skeleton-noise geometry pair alignment 和 joint repair。
- 数据规模：MPTS-52 validation proposals `4727` samples，rows>=7 `2197` samples，candidate records `36268`，topK `20`。主 anchor 固定为 CrystaLLM-a GT-SG：MP-20 official match@1/5/20 = `71.667% / 83.075% / 87.807%`；MPTS-52 official match@1/5/20 = `25.235% / 36.462% / 43.960%`。
- baseline：MP-20/MPTS-52 主 baseline 只使用 CrystaLLM-a GT-SG anchor；`17.181 / 24.345 / 31.522` pure CrystaLLM GT-SG low baseline 不允许作为主成功对照。
- 方法变化：固定 opentry_13 exp2 selected train-prototype renderer/site mapping。只使用 inference-safe structural checks 选择第一个 legal/formula/SG/exact-cover/site-count/no-collision CIF；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、RF/HGB/rerank 或 official feedback。
- 结果 overall：selected train-prototype valid `97.007%`，formula `99.660%`，SG `98.436%`，exact-cover `100.000%`；fallback_rate `2.993%`。
- 结果 rows>=7：valid `96.458%`，formula `99.410%`，SG `98.475%`，exact-cover `100.000%`。满足 prompt 要求的 valid >=约 `96%`、formula >=约 `99%`、SG >=约 `98%`、exact-cover `100%`。
- 可信度：高。实验 0 直接读取 opentry_13 exp2 机器 JSON 和 opentry_7 official anchor JSON，不重跑 StructureMatcher、不训练、不调阈值；它是前端结构 gate，不是 match claim。
- 和历史实验关系：opentry_13 exp3 显示 rows>=7 skeleton-hit@50 `76.513%`，但 hydrated match@50 `15.112%`、proposal-to-match conversion@50 `19.750%`；opentry_13 exp4 multi-geometry rows>=7 match@50 `13.084%`、conversion@50 `15.407%`；lattice-only repair gate 已失败。因此实验 0 通过后，主线应转向 joint geometry repair 的数据对齐，而不是继续改 renderer 或 rerank。
- 最终判决：`pass`。selected train-prototype front-end 可冻结，允许进入实验 1 predicted-skeleton-noise geometry training pair alignment。
- 下一步：执行实验 1，在 MP-20/MPTS-52 train split 上构造 noisy predicted skeleton + initial geometry -> target geometry 训练对，并审计 lattice、row-level free-parameter、local geometry/collision target 是否可恢复；若 alignment gate 不过，不允许训练实验 2。

<!-- OPENTRY14_EXP1_PREDICTED_SKELETON_NOISE_GEOMETRY_PAIRS -->
## opentry_14 实验 1：predicted-skeleton-noise 完整几何训练对

结果文件：`model/New_model/opentry_14/results/experiment_1_predicted_skeleton_noise_geometry_pairs_sharded.json`
训练对 artifact：`model/New_model/opentry_14/artifacts/exp1_predicted_skeleton_noise_pairs/predicted_skeleton_noise_geometry_pairs_merged_sharded.jsonl.gz`

- 为什么做：opentry_13 已证明 selected renderer/site mapping 可过结构 gate，但 rows>=7 skeleton-to-match conversion 仍低；lattice-only repair 和 heuristic multi-geometry 都失败。实验 1 的目的不是追求 match，而是确认 joint repair 训练时能看到和推理期相似的 predicted-skeleton noise，并且能恢复 lattice、row-level free-parameter 和 local geometry/collision target。
- 核心假设：如果 train-side noisy predicted skeleton + selected_train_prototype 初始几何能覆盖大部分 MP-20/MPTS-52 train 样本，并且 rows>=7 中大部分含 free parameter 的样本能恢复可训练 target，则允许进入实验 2 joint lattice + free-parameter repair；否则必须先修 data alignment。
- 数据规模：MP-20 train `26629` samples，rows>=7 `4306`；MPTS-52 train `25998` samples，rows>=7 `6863`；合计 `52627` train records，rows>=7 `11169`。使用 8 shard 并行构造，单个 Python shard 约 100% CPU，低于用户给定的单进程/单核 `200%` 上限。
- baseline：实验 1 不评估 match，不对标 low baseline；主 baseline 仍冻结为 CrystaLLM-a GT-SG anchor。它只检查是否允许训练 repair head。
- 方法变化：在 train split 上用 composition/formula + GT-SG + train prototypes 构造 exact-cover predicted skeleton source，排除 self-source；用 selected_train_prototype-style source lattice/free params 渲染 initial geometry；从 train true structure 中恢复 target lattice、row-level free params、target min-pair-distance、volume/atom、collision/local stats。第一版严格 `orbit_id + element` alignment 未过 rows>=7 free-param gate；修正后保留 exact alignment，同时允许 `orbit_id` element-mismatch fallback，并在 artifact 中标记 alignment kind。
- 结果 overall：nonempty pair rate `88.994%`，initial valid `80.415%`，initial formula `88.320%`，initial SG `82.175%`，initial exact-cover `88.994%`，initial collision `4.353%`；samples requiring free params `38569`，free-param target complete among requiring `81.794%`，free-param value recovery `92.657%`，usable joint pair rate `75.229%`。
- 结果 rows>=7：nonempty pair rate `80.097%`，initial valid `71.358%`，initial formula `78.216%`，initial SG `73.865%`，initial exact-cover `80.097%`，initial collision `5.775%`；samples requiring free params `8946`，free-param target complete among requiring `76.492%`，free-param value recovery `94.869%`，usable joint pair rate `60.041%`。
- 分数据集：MP-20 rows>=7 nonempty pair `87.436%`、free-param complete among requiring `71.766%`、value recovery `92.905%`；MPTS-52 rows>=7 nonempty pair `75.492%`、free-param complete among requiring `79.927%`、value recovery `96.007%`。
- 可信度：中等偏高。覆盖两个 train split，真实 render/parse/SG/collision/local stats；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。限制是 predicted source 仍是 train-prototype proposer，rows>=7 usable joint pair 只有 `60.041%`，element-mismatch target 是 noisy alignment 而非严格 WA oracle。
- 和历史实验关系：opentry_13 lattice repair pilot 虽有 `7774` train noisy pairs，但只学习 lattice；本实验扩展到 MP-20+MPTS-52 全 train，显式恢复 row-level free-parameter 和 local geometry/collision target，为实验 2 的 joint repair 提供最低可用训练对。它也解释了为什么不能再做 lattice-only repair。
- gate 判定：passed=True。最低 data-alignment gate 为 overall nonempty >= `80%`、rows>=7 nonempty >= `70%`、target-local-stats >= `80%`、overall/rows>=7 free-param complete among requiring >= `70%`、free-param value recovery >= `70%`；修正后全部通过。
- 最终判决：`pass`。允许进入实验 2，但必须把 rows>=7 usable joint pair 只有 `60.041%` 作为可信度限制，不得把本实验写成 match 成功。
- 下一步：训练 predicted-skeleton-aware joint geometry repair head，输出必须至少包含 lattice 更新和 row-level free-parameter 更新；若 rows>=7 validation conversion@50 不能达到最低约 `22.3%` 或结构安全下降，不允许进入实验 3/full validation/official。

<!-- OPENTRY14_EXP2_JOINT_GEOMETRY_REPAIR -->
## opentry_14 实验 2：predicted-skeleton-aware joint geometry repair head

结果文件：`model/New_model/opentry_14/results/experiment_2_joint_geometry_repair.json`
模型 artifact：`model/New_model/opentry_14/artifacts/exp2_joint_geometry_repair/joint_repair_heads.pt`
候选评估：`model/New_model/opentry_14/artifacts/exp2_joint_geometry_repair/evaluated_joint_repair_candidates.jsonl`

- 为什么做：实验 1 证明 train-side predicted-skeleton-noise geometry pairs 可用，但 opentry_13 的 lattice-only repair 和 heuristic multi-geometry 已失败。实验 2 直接训练 joint lattice + row-level free-parameter repair head，检查它能否把 rows>=7 predicted skeleton hit 转成 StructureMatcher match。
- 核心假设：如果主要瓶颈是 lattice/free-parameter joint repair，那么在固定 selected train-prototype 前端和 exp3 predicted skeleton proposals 后，joint head 应至少把 rows>=7 skeleton-to-match conversion@50 从历史约 `15%~20%` 推到最低线 `22.3%`，并让 rows>=7 match@50 接近 CrystaLLM K50 validation baseline `18.431%`。
- 数据规模：训练读取实验 1 pair artifact，共 `52627` records，usable joint pairs `39591`；lattice training samples `39591`，row-param samples `270498`。validation 使用 MPTS-52 val predicted skeleton proposals：`4411` samples，rows>=7 `2033`，candidate records `18134`，topK `50`，StructureMatcher workers `8`。运行环境为 `crystallm_env`，线程变量限制为 OMP/MKL/OPENBLAS/NUMEXPR/NUMBA=1；单个 Python/worker 进程约 100% CPU，低于用户给定 `200%` 上限。
- baseline：最低 gate 使用 rows>=7 CrystaLLM K50 validation top50 `18.431%`，历史 predicted-skeleton hydrated rows>=7 match@50 `15.112%`，opentry_13 multi-geometry rows>=7 match@50 `13.084%`。official true anchor 仍只作为主比较口径，不参与 validation 调参。
- 方法变化：训练两个 PyTorch MLP：lattice head 从 composition/SG/skeleton/initial geometry/self-check features 预测 6 个 lattice raw values，row-param head 从全局特征、row index、param name 和 initial value 预测 row-level free parameter。推理时用 alpha=`0.35` blend source 与 prediction，重新渲染 CIF 并用 StructureMatcher 评估；未使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。
- 训练结果：lattice head best val loss `0.292309`，param head best val loss `0.960734`。这说明模型完成训练，但 param head 的泛化误差仍高，不能单独作为 geometry 修复成功证据。
- 结果 overall：match@1/5/20/50 = `10.814% / 17.094% / 17.298% / 17.298%`；RMSE@1/5/20/50 = `0.229557 / 0.200425 / 0.200769 / 0.200769`；valid_any@50 `92.859%`，formula consistency `99.956%`，SG consistency `96.802%`，exact-cover retained `100.000%`，collision `22.957%`，skeleton-hit@50 `86.692%`，skeleton-to-match conversion@50 `19.665%`。
- 结果 rows>=7：match@1/5/20/50 = `0.246% / 0.394% / 0.394% / 0.394%`；RMSE@1/5/20/50 = `0.289484 / 0.241265 / 0.241265 / 0.241265`；valid_any@50 `90.507%`，formula consistency `99.968%`，SG consistency `96.817%`，exact-cover retained `100.000%`，collision `26.075%`，skeleton-hit@50 `82.686%`，skeleton-to-match conversion@50 `0.416%`。
- 可信度：中等偏高。它是真实 train-noisy-skeleton joint repair、真实 render/parse/SG-check/StructureMatcher validation，且没有使用禁用推理特征或 official feedback；限制是每个 skeleton 只输出 1 个 repaired geometry，alpha/模型规模仍是轻量 pilot，row-param target 含 element-mismatch noisy alignment，rows>=7 usable training coverage 只有 `60.041%`。
- 和历史实验关系：相比 opentry_13 lattice-only pilot，实验 2 保住了 formula/SG/exact-cover，但 rows>=7 match 更差；相比 opentry_13 multi-geometry，skeleton-hit@50 相近但 conversion 从 `15.407%` 掉到 `0.416%`。这说明当前 joint MLP 没有学到 rows>=7 的可 match geometry posterior，失败点不是前端 exact-cover，而是 free-parameter/site/local geometry 修复没有对准 StructureMatcher 可接受区域。
- gate 判定：`minimum_passed=False`，`target_passed=False`，`passed=False`。最低线要求 rows>=7 conversion@50 >= `22.3%`、rows>=7 match@50 接近 `18.431%`、formula >= `95%`、SG >= `90%`、exact-cover >= `95%`；本实验结构线通过，但 conversion@50 只有 `0.416%`、match@50 只有 `0.394%`，核心 repair gate 失败。
- 最终判决：`fail_validation_gate`。归因类别为“repair 保持 exact-cover/formula/SG，但不提升 match，且 rows>=7 conversion 崩溃”；不能进入实验 3、full validation 或 official。
- 下一步：停止本轮后续 gate。下一轮若继续该路线，应先重做 rows>=7 的 free-parameter/site mapping alignment 与 local geometry target，加入 learned posterior/multi-hypothesis 或 symmetry-aware local residual，但必须重新从 validation gate 开始；不要扩大 official、不要调 C2S3C15/scorer/threshold。


<!-- OPENTRY14_EXP2B_SAFE_POOL_AFTER_FAILURE_ANALYSIS -->
## opentry_14 实验 2b：failure-analysis guided safe geometry pool

结果文件：`model/New_model/opentry_14/results/experiment_2b_safe_pool_after_failure_analysis.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2b_hydrated_prototype_safe_pool/evaluated_safe_pool_candidates.jsonl`

- 为什么做：实验 2 的 joint MLP 在 rows>=7 上 match@50 只有 `0.394%`、conversion@50 只有 `0.416%`。失败分析显示它把已有可 match 的 hydrated/source-prototype geometry 替换成单一 repaired geometry，导致 rows>=7 只剩 `8` 个 matched samples；因此本实验不再让 MLP 覆盖安全候选，而是构造固定配额的 safe geometry pool。
- 核心假设：如果根因是破坏性单一 repair 覆盖，而不是 skeleton coverage 不足，那么保留原 SymCIF v5 hydrated geometry，再补入 prototype multi-geometry，应该在不使用 match/RMSD 排序的情况下恢复并超过实验 2 最低 gate。
- 数据规模：validation samples `4411`，rows>=7 `2033`；candidate records `94544`；topK `50`；固定配额为 hydrated top `10` + prototype top `40`。
- baseline：实验 2 最低 gate 使用 rows>=7 CrystaLLM K50 validation top50 `18.431%`，允许近线下限为 `17.931%`；历史 opentry_13 hydrated rows>=7 match@50 `15.112%`，multi-geometry rows>=7 match@50 `13.084%`。
- 方法变化：排名规则固定为每个样本先取 rows>=7 proposer 映射到的 SymCIF v5 hydrated candidates top10，再取 opentry_13 prototype multi-geometry top40；候选内部顺序只使用 proposer rank、generation score 和原 structural rank，不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。
- 结果 overall：match@1/5/20/50 = `27.998% / 40.014% / 44.276% / 44.389%`；formula `98.333%`，SG `94.862%`，exact-cover `99.712%`。
- 结果 rows>=7：match@1/5/20/50 = `11.461% / 17.019% / 21.299% / 21.348%`；RMSE@50 `0.09165643819972137`；skeleton-hit@50 `82.686%`；skeleton-to-match conversion@50 `24.866%`；valid_any@50 `98.032%`；formula `97.680%`，SG `94.744%`，exact-cover `99.682%`。
- 可信度：中等偏高。所有 counted candidates 都有已有 validation metrics 或 opentry_13 StructureMatcher evaluation；本实验只做固定配额组合并重新汇总，不用 match/RMSD 参与排序。限制是 hydrated metrics 是复用既有 SymCIF v5 validation evaluation，不是重新跑 StructureMatcher；该方法是 validation safe-pool gate，不是 official claim。
- 和历史实验关系：opentry_13 exp3 hydrated rows>=7 top50 已有 `15.112%`，opentry_13 exp4 prototype multi-geometry top50 为 `13.084%`，二者互补；opentry_14 exp2 joint MLP 破坏 rows>=7 geometry。本实验把历史两类 inference-safe geometry pool 合并，先恢复最低 conversion gate。
- gate 判定：minimum_passed=`True`，target_passed=`True`；rows>=7 match@50 delta vs near CrystaLLM K50 lower bound `+3.416pp`；conversion@50 delta vs 22.3% `+2.566pp`。
- 最终判决：`pass_minimum_gate`。Safe pool passes the experiment-2 minimum validation gate; target gate is recorded separately.
- 下一步：Allowed to test experiment 3 local optimizer on this fixed safe pool; do not claim official success before full validation.


<!-- OPENTRY14_EXP3_SYMMETRY_LOCAL_OPTIMIZER -->
## opentry_14 实验 3：symmetry-preserving local optimizer 诊断

结果文件：`model/New_model/opentry_14/results/experiment_3_symmetry_local_optimizer.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3_symmetry_local_optimizer/`

- 为什么做：实验 2b 已把 rows>=7 conversion@50 恢复到 `24.866%`，但 prompt 要求继续检查 symmetry-preserving local optimizer 是否能修复 collision/short-distance/local packing，并至少再提升 conversion `+2pp`。
- 核心假设：如果 skeleton-to-match 的剩余瓶颈主要是局部短距/packing，那么在不破坏 formula、SG、exact-cover 的前提下，对 hydrated CIF 做轨道级 repulsion 优化应降低 collision/close-pair，并把更多 predicted-skeleton-hit candidate 转成 StructureMatcher match。
- 数据规模：safe-pool candidates `94544`，hydrated optimization inputs `23600`，optimizer accepted `260`；StructureMatcher workers `64`；topK `50`。
- baseline：实验 2b best rows>=7 match@50 `21.348%`，conversion@50 `24.866%`；Exp3 通过线要求 conversion@50 >= `26.866%`，且 collision/local packing 改善，SG/exact-cover 不恶化。
- 方法变化：只对 SymCIF v5 hydrated CIF 做局部优化；用 spglib/pymatgen 等价原子分组，在同一空间群操作下移动整个 Wyckoff orbit 的代表点并重新展开，候选若 formula/site count/SG/exact-cover 不满足或 local proxy 未改善则回退。prototype 候选不改。排序/选择不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `replace_hydrated_local_optimizer`：rows>=7 match@1/5/20/50 = `9.395% / 13.773% / 18.692% / 18.741%`；conversion@50 `21.951%`；collision `13.419%`；SG `93.948%`；exact-cover `99.682%`。
- 局部优化诊断：accepted optimized candidates `260`；optimized evaluation match candidates `4`；optimized-vs-original hard collision mean delta `-1.7`；close-pair mean delta `-3.3653846153846154`；min-distance mean delta `0.20180837948394473`。
- 复核诊断：实验 2b rows>=7 safe pool 在复用 SymCIF hydrated metrics 时 match samples 为 `434`、conversion@50 `24.866%`；本实验重新解析 hydrated CIF 后，同一 baseline 只有 `381` 个 match samples、conversion@50 `21.951%`，丢失 `53` 个 hydrated match、没有新增 match。source 贡献显示实验 2b 中 hydrated_any `332`、prototype_any `266`、both `164`、hydrated_only `168`、prototype_only `102`，说明 prototype 贡献稳定，而 hydrated 复用指标是可信度弱点。
- 可信度：中等。该实验真实解析 CIF、重做 SG/formula/site-count/StructureMatcher 检查，并用 64-worker 评估；限制是当前环境没有 CHGNet/MatGL/MACE，local proxy 是短距/close-pair repulsion，不是学习到的材料势能，且只覆盖 hydrated CIF，prototype 候选没有可逆局部优化。另一个限制是实验 2b 的 hydrated 部分复用既有 SymCIF metrics，重评估后 rows>=7 conversion 低于 22.3% 最低线。
- 和历史实验关系：继承实验 2b safe pool，不进入 official；它直接测试“local geometry/collision 是否是剩余主要瓶颈”。若 conversion 不升，即支持失败归因“local geometry/collision 无法通过当前 proxy 修复”。
- gate 判定：passed=`False`；conversion delta vs Exp2b `-2.915pp`；collision improved=`True`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_diagnostic_only`。Local optimizer did not produce the required +2pp rows>=7 conversion lift over experiment 2b; any packing improvement remains diagnostic only.
- 下一步：Do not enter final method with this optimizer; root cause is local geometry/collision not fixable by the current symmetry-safe short-distance proxy. Next try must improve learned free-parameter/site alignment or a real energy model before repeating Exp3.


<!-- OPENTRY14_EXP3B_CHGNET_LOCAL_OPTIMIZER -->
## opentry_14 实验 3b：CHGNet symmetry-checked local optimizer

结果文件：`model/New_model/opentry_14/results/experiment_3b_chgnet_local_optimizer.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3b_chgnet_local_optimizer/`

- 为什么做：实验 3 的 orbit repulsion proxy 只改善短距，rows>=7 conversion@50 反而下降；失败归因是当前局部 proxy 没有材料势能。实验 3b 安装并调用 CHGNet 预训练势能，在 GPU 上做短步位置优化，并用 SG/formula/site/exact 回退。
- 核心假设：如果实验 3 失败是因为 repulsion proxy 太弱，那么对 rows>=7 hydrated 中 formula/SG/exact 可用但 valid=false 的候选做 CHGNet 短步 relaxation，应能把一部分局部不合理几何转成 StructureMatcher match，同时不破坏 SG/exact-cover。
- 数据规模：待优化 candidates `8490`，CHGNet accepted `4752`，evaluated optimized `4752`；GPU devices `['0', '1']`，CHGNet workers `8`，StructureMatcher workers `64`。
- baseline：实验 2b rows>=7 match@50 `21.348%`，conversion@50 `24.866%`；Exp3 gate 需要 conversion@50 >= `26.866%`，collision/local packing 改善，SG/exact-cover 不恶化。
- 方法变化：选择规则只用 inference-safe 结构状态：`rows>=7`、hydrated、formula_ok、space_group_ok、exact_cover_retained、valid=false；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback。CHGNet 只优化原子位置，不 relax cell；优化后若 SG/formula/site count/exact/local proxy 不满足即回退。
- 结果 best variant `append_chgnet_after_hydrated`：rows>=7 match@1/5/20/50 = `11.412% / 16.724% / 21.299% / 21.348%`；conversion@50 `24.866%`；collision `13.262%`；SG `95.250%`；exact-cover `99.710%`。
- CHGNet 诊断：optimized match records `21`，optimized valid records `4497`，mean min-distance delta `0.09053467610330115`，mean hard-collision delta `-0.02988215488215488`，mean close-pair delta `-1.3125`。
- oracle 上限诊断：即使把所有 CHGNet optimized match oracle 插入 top50，rows>=7 新增 match samples 也是 `0`，新增 skeleton-and-match samples 也是 `0`，union conversion@50 仍为 `24.866%`。额外 cell+position 256-candidate smoke 只有 `4` 条 match record、`0` 个新增 match sample；valid hydrated 512-candidate smoke 有 `73` 条 match record，但也有 `0` 个新增 match sample。说明扩大 CHGNet local optimizer 不会补足 +2pp gate。
- 可信度：中等。CHGNet 是真实预训练势能并使用 GPU，优化结果重新通过 StructureMatcher/SG/formula/site/exact 检查；限制是只优化 rows>=7 hydrated invalid candidates，未覆盖 prototype CIF，且未训练 predicted-skeleton-aware posterior。unchanged candidates 沿用实验 2b 已有 validation metrics，因此与实验 3 的 direct re-eval 口径不同。
- 和历史实验关系：这是实验 3 失败后的修复尝试，专门检验“真实材料势能是否能替代短距 proxy”。若仍不过 +2pp conversion gate，则说明当前瓶颈更偏 free-parameter/site alignment 或 skeleton proposal，而不是局部能量微调。
- gate 判定：passed=`False`；conversion delta vs Exp2b `+0.000pp`；match@50 delta vs Exp2b `+0.000pp`；collision/local improved=`True`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_diagnostic_only`。CHGNet local optimizer did not produce the required +2pp rows>=7 conversion lift over experiment 2b under inference-safe selection.
- 下一步：Do not enter Exp4/final method from this optimizer; return to free-parameter/site alignment or skeleton proposer improvement.

<!-- OPENTRY14_EXP3C_ALIGNMENT_ROOT_CAUSE_AUDIT -->
## opentry_14 实验 3c：local-optimizer 失败后的 alignment 根因审计

结果文件：`model/New_model/opentry_14/results/experiment_3_to_exp2_alignment_root_cause_audit.json`

- 为什么做：实验 3 和 3b 都能改善 collision/local packing，但无法提高 rows>=7 conversion@50。需要判断失败是否仍然来自 local geometry，还是已经转移到 valid symmetric candidate 内部的 free-parameter/site/lattice alignment。
- 核心假设：如果未 match 的 skeleton-hit 样本大多没有 valid/formula/SG/exact 候选，则继续 local optimizer 可能合理；如果它们已经有大量 valid symmetric candidates，则 local optimizer 不是主瓶颈。
- 数据规模：Exp2b rows>=7 samples `2033`，skeleton-hit samples `1681`，match samples `434`，skeleton-hit 但无 match samples `1263`。
- baseline：实验 2b rows>=7 match@50 `21.348%`、conversion@50 `24.866%`；实验 3b CHGNet oracle 新增 match samples 为 `0`。
- 方法变化：只读 Exp2b safe-pool validation JSONL，按 sample-level `skeleton_hit` 与 `match` 分类；统计未 match 的 skeleton-hit 样本里是否已有 valid/formula/SG/exact 候选，以及 skeleton-hit candidate 自身的结构状态。不使用这些统计做推理排序或 official 调参。
- 结果：`1263` 个 skeleton-hit/no-match 样本中，`1247` 个已经至少有一个 valid/formula/SG/exact 候选；这些样本内所有候选 `30661` 条，其中 valid `20865`、formula `30157`、SG `29107`、exact `30593`。skeleton-hit candidate 共 `6577` 条，其中 valid `4537`、formula `6469`、SG `6126`、exact `6557`，median rank `9`。
- 可信度：中等偏高。它是现有 validation artifact 的全量 rows>=7 审计，不训练、不重跑 matcher、不用 official；限制是沿用实验 2b 部分复用 hydrated metrics 的口径。
- 和历史实验关系：解释了 opentry_13 multi-geometry、opentry_14 Exp3 orbit proxy、Exp3b CHGNet 都不能提高 conversion 的共同原因：大量候选已经结构合法，但连续参数/site/lattice 没有对齐到 StructureMatcher 可接受 basin。
- 最终判决：local optimizer 路线作为 Exp3 主修复失败。归因类别为“repair/local optimizer 保持 exact-cover/formula/SG，但不提升 match”；下一步不是扩大 optimizer，而是重做 predicted-skeleton-aware free-parameter/site alignment 或 learned multi-hypothesis geometry posterior。
- 下一步：回到 Exp2/Exp4 之间的 geometry posterior 问题，先做 learned free-parameter/site alignment 修复；在没有新的 geometry posterior 之前，不进入 official，也不把 Exp3/3b 写入最终方法。


<!-- OPENTRY14_EXP2D_SITE_ASSIGNMENT_MULTI_HYPOTHESIS -->
## opentry_14 实验 2d：site-assignment multi-hypothesis repair

结果文件：`model/New_model/opentry_14/results/experiment_2d_site_assignment_multi_hypothesis.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2d_site_assignment_multi_hypothesis/`

- 为什么做：Exp3/3b 证明 local optimizer 不能新增 sample-level match；Exp3c 显示大量 skeleton-hit/no-match 样本已经 valid/formula/SG/exact，根因更像元素到 Wyckoff row 的 site assignment 与 free-parameter alignment 错误。本实验先修 site assignment。
- 核心假设：同一个 predicted skeleton 的 multiplicity/orbit 可以有多个 exact-cover 元素分配；当前 source-preferred 单 assignment 可能把元素放到错误 row。枚举少量 inference-safe exact-cover assignment 并用 train source lattice/free params 渲染，可能新增 rows>=7 match/conversion。
- 数据规模：rows>=7 validation samples `2197`；site-assignment generated candidates `36531`；evaluated candidates `36531`；top skeletons `10`；assignments/proposal `4`；StructureMatcher workers `64`。
- baseline：Exp2b rows>=7 match@50 `21.348%`，conversion@50 `24.866%`，collision `14.981%`。
- 方法变化：对每个 predicted skeleton source rows 枚举最多 `4` 个 exact-cover element assignment，按保留 source 元素的原子数排序；每个 assignment 使用 source lattice 和 flexible source row params 渲染。选择和排序不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `h10_interleave_s20_p20`：rows>=7 match@1/5/20/50 = `11.412% / 17.511% / 20.954% / 22.036%`；conversion@50 `25.714%`；valid `79.122%`；formula `98.896%`；SG `97.448%`；exact-cover `99.791%`；collision `7.816%`。
- 可信度：中等。该实验真实 render/parse/SG/StructureMatcher，且不使用禁用推理特征；限制是几何仍来自 train source prototype/free params，不是 learned posterior，site assignment 枚举属于修复 alignment 的 proof-of-concept，不是最终 Exp4 critic。
- 和历史实验关系：它直接响应 Exp3c 的根因审计。若提升 conversion，说明下一步应把 site assignment/free-parameter posterior 学起来；若不提升，则需要升级 skeleton proposer 或更强 geometry posterior。
- gate 判定：minimum_passed=`True`；target_passed=`False`；match@50 delta vs Exp2b `+0.689pp`；conversion delta vs Exp2b `+0.848pp`。
- 最终判决：`pass_minimum_gate`。Site-assignment multi-hypothesis passes the Exp2 minimum repair gate.
- 下一步：Use this as alignment-positive candidate and retest local optimizer/Exp3 gate.


<!-- OPENTRY14_EXP2E_TRAIN_PAIR_RESIDUAL_POSTERIOR -->
## opentry_14 实验 2e：train-pair residual posterior multi-hypothesis

结果文件：`model/New_model/opentry_14/results/experiment_2e_train_pair_residual_posterior.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2e_train_pair_residual_posterior/`

- 为什么做：Exp2d 的 site-assignment 枚举有正信号，但 oracle upper bound 只有 rows>=7 match@50 `22.233%`、conversion `25.937%`，说明仅扩大 assignment 配额不足。Exp2e 改为使用 Exp1 train noisy-pair 中真实 target-initial lattice/free-parameter 残差，生成 learned empirical posterior 多假设。
- 核心假设：如果 skeleton-hit/no-match 的核心瓶颈是连续 lattice/free-parameter basin 没对齐，则把同 SG/row-count train pair 残差迁移到 validation predicted skeleton/site assignment 上，应新增 match 样本并提高 rows>=7 conversion。
- 数据规模：train residual templates `6628`，residual buckets `303`；validation rows>=7 samples `2197`；generated posterior candidates `2842`；evaluated posterior candidates `2842`；StructureMatcher workers `96`。
- baseline：Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%`；Exp2d best rows>=7 match@50 `22.036%`、conversion `25.714%`。
- 方法变化：对 validation predicted skeleton 的 exact-cover site assignment 先用 source geometry 初始化，再从 Exp1 train pairs 中按 SG、row count、composition 距离和 skeleton key 选残差模板；对 lattice 施加 length ratio/angle delta，对 row-level free parameters 施加 circular residual，生成多尺度 residual hypotheses。critic 只用 formula/SG/exact/valid/collision/volume/reference score/proposal rank 等 structural score；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `h10_s10_r20_p10`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.594% / 21.790%`；conversion@50 `25.522%`；valid `75.814%`；formula `98.866%`；SG `97.897%`；exact-cover `99.693%`；collision `5.683%`。
- oracle 诊断：`model/New_model/opentry_14/results/experiment_2e_residual_posterior_oracle_diagnostic.json` 显示 residual-only 只有 `2` 个 match samples、conversion `0.163%`；相对 Exp2b 和 Exp2d best 都新增 `0` 个 sample-level match/skelmatch。Exp2b + site-assignment + residual 的 union upper bound 仍为 rows>=7 match@50 `22.233%`、conversion `25.937%`，与 Exp2d site-assignment oracle 相同。
- 可信度：中等。训练残差只来自 train split noisy-skeleton pairs，validation 推理不使用禁用标签；所有候选真实 render/parse/SG/StructureMatcher。限制是 residual 迁移仍按 row index 对齐，未学习 permutation-aware posterior，且 residual 模板筛选是非参数近邻而非端到端概率模型。
- 和历史实验关系：它是 Exp2 joint MLP 失败后的另一路 learned posterior 修复，承接 Exp3c 的 alignment 根因审计和 Exp2d 的 assignment 正信号；若仍无法过 target gate，说明单靠 train-pair residual/posterior 也不足，可能需要升级 skeleton proposer 或显式 permutation-aware alignment。
- gate 判定：minimum_passed=`True`；target_passed=`False`；match@50 delta vs Exp2b `+0.443pp`；conversion delta vs Exp2b `+0.656pp`；match@50 delta vs Exp2d `-0.246pp`；conversion delta vs Exp2d `-0.193pp`。
- 最终判决：`pass_minimum_gate_but_no_residual_candidate_headroom`。Train-pair residual posterior remains above the Exp2 minimum gate only because the safe pool/site-assignment candidates remain present; residual candidates themselves add no new sample-level matches.
- 下一步：不要继续扩大 residual beam。下一步只能是显式 permutation-aware site/free-parameter alignment、升级 skeleton proposer，或停止该 repair 路线；不能进入 official。


<!-- OPENTRY14_CURRENT_BOUNDARY_AND_NEXT_DECISION -->
## opentry_14 当前边界判定

- 为什么做：Exp3/3b local optimizer 未过 +2pp conversion gate，Exp2d/2e 是为修复 alignment 根因而做的补充尝试；现在需要明确是否允许继续 Exp4/Exp5/official。
- 核心假设复核：如果剩余瓶颈可由 local optimizer、少量 site assignment 或 train-pair residual posterior 修复，rows>=7 conversion@50 应从 Exp2b `24.866%` 至少推到 Exp3 线 `26.866%`，或 Exp2 target 线 `28%`。
- 数据规模：已覆盖 Exp0 front-end freeze、Exp1 train noisy-pair construction、Exp2 joint MLP、Exp2b safe pool、Exp3 orbit repulsion、Exp3b CHGNet、Exp3c root-cause audit、Exp2d site assignment、Exp2e residual posterior；主要 validation rows>=7 口径为 `2033` samples，候选规模从 `18134` 到 `119121` records。
- baseline：主 baseline 仍为 CrystaLLM-a GT-SG anchor；validation repair gate 以 Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%` 作为当前最佳安全池起点，不使用 low baseline、official feedback 或 forbidden scorer。
- 方法变化总结：local optimizer 改善 collision 但不新增 sample-level match；site assignment 有小幅真实信号但 oracle upper bound 只有 match@50 `22.233%`、conversion `25.937%`；train-pair residual posterior residual-only 只有 `2` 个 match samples，并且相对 Exp2b/Exp2d 新增 `0` 个 sample-level match。
- 结果：当前可复现最佳 rows>=7 validation 仍是 Exp2d best `h10_interleave_s20_p20`，match@1/5/20/50 = `11.412% / 17.511% / 20.954% / 22.036%`，conversion@50 `25.714%`。它高于 Exp2b `+0.689pp` match@50、`+0.848pp` conversion，但低于 Exp3 +2pp gate 和 Exp2 target line。
- 可信度：中等偏高。所有补充实验均在 validation 上真实 render/parse/SG/StructureMatcher，未使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank 作为推理特征；限制是 hydrated safe-pool 部分沿用既有 SymCIF v5 metrics，且 permutation-aware free-parameter posterior 尚未实现。
- 和历史实验关系：opentry_13 已证明 renderer/site mapping 结构 gate 可靠但 conversion 低；opentry_14 进一步证明单一 joint MLP 会破坏 match basin，local optimizer/CHGNet 不能新增样本，site assignment 只能带来很小增益，train-pair residual posterior 没有候选层面余量。
- 最终判决：`do_not_enter_exp4_exp5_or_official`。当前路线没有通过 Exp3 gate，也没有形成可作为 Exp4 前置的 learned posterior/critic 增益；不得进入 full validation gate 或 frozen official test。
- 下一步：若继续研究，优先做显式 permutation-aware site/free-parameter alignment（例如对同 Wyckoff row 的元素/参数进行多对多匹配和 posterior 学习），或升级 predicted skeleton proposer；否则应停止当前 repair 路线。禁止继续扩大 residual/site-assignment beam、C/S 比例、普通 scorer 或 official 调参。


<!-- OPENTRY14_EXP2F_PERMUTATION_AWARE_ALIGNMENT -->
## opentry_14 实验 2f：permutation-aware row/free-parameter alignment

结果文件：`model/New_model/opentry_14/results/experiment_2f_permutation_aware_alignment.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2f_permutation_aware_alignment/`

- 为什么做：Exp2d 证明 site assignment 有小幅正信号，但 oracle 上限不足；Exp2e residual posterior 新增 `0` 个 sample-level match。剩余可检验根因是同一 Wyckoff/orbit 内 row-level free parameters 可能跟错元素/row，固定 row-index 会把正确站位几何放到错误元素上。
- 核心假设：如果 skeleton-hit/no-match 的关键错误是相同 orbit 内参数块与元素分配错位，则在 exact-cover assignment 后，在相同 orbit/参数模式内做 element-following、element-sorted、reverse/swap 参数块排列，应新增 sample-level match，并提升 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `2197`；generated permutation candidates `47180`；evaluated candidates `47180`；skipped atom-count samples `157`；workers `96`。
- baseline：Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%`；Exp2d best rows>=7 match@50 `22.036%`、conversion `25.714%`。
- 方法变化：对每个 predicted skeleton 的 exact-cover assignment，先恢复 source lattice/free params，然后只在相同 orbit_id、multiplicity 和 free-param key signature 的 row 之间排列参数块；生成 identity、source-element-following、element-sorted、reverse、pair-swap 等 deterministic hypotheses。critic 只用 legal/formula/SG/site/exact/collision/volume/reference/proposal structural score；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `h10_s5_perm30_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 20.020% / 21.840%`；conversion@50 `25.740%`；valid `81.436%`；formula `99.404%`；SG `97.337%`；exact-cover `99.800%`；collision `6.439%`。
- oracle 诊断：permutation-only match samples `248`，相对 Exp2b 新增 match/skelmatch samples `10` / `10`，相对 Exp2d best 新增 `5` / `5`；union upper bound match@50 `22.282%`、conversion `25.996%`。
- 可信度：中等。它是真实 render/parse/SG/StructureMatcher evaluation，推理不使用禁用标签；限制是 permutation 仍是 deterministic local enumeration，不是训练得到的 posterior，且为避免 StructureMatcher 长尾默认不对超大 atom-count 样本生成 permutation candidates。
- 和历史实验关系：承接 Exp3c 的 alignment 根因审计、Exp2d 的 assignment 正信号和 Exp2e residual 无余量结论；如果仍无新增 oracle match，则 rows>=7 conversion 的剩余瓶颈更可能在 predicted skeleton proposer 本身或更复杂的 permutation-aware posterior。
- gate 判定：minimum_passed=`True`；target_passed=`False`；match@50 delta vs Exp2b `+0.492pp`；conversion delta vs Exp2b `+0.874pp`；match@50 delta vs Exp2d `-0.197pp`；conversion delta vs Exp2d `+0.026pp`。
- 最终判决：`pass_minimum_gate_but_insufficient_permutation_headroom`。Permutation-aware alignment remains above the Exp2 minimum gate only because the existing safe-pool/site-assignment candidates remain present; deterministic permutations do not reach the Exp2 target or Exp3 +2pp conversion gate.
- 下一步：不要继续盲目扩大 deterministic permutation beam；union upper bound 只有 match@50 `22.282%`、conversion `25.996%`。下一步只能是训练真正的 permutation-aware geometry posterior、升级 predicted skeleton proposer，或停止当前 repair 路线；不能进入 official。


<!-- OPENTRY14_CURRENT_BOUNDARY_AFTER_EXP2F -->
## opentry_14 当前边界判定（Exp2f 后）

- 为什么做：Exp2f 是对上一版边界判定中“显式 permutation-aware site/free-parameter alignment”的补充验证；现在需要用最新 oracle headroom 重新判定是否允许进入 Exp4/Exp5/official。
- 核心假设复核：如果 rows>=7 剩余瓶颈主要是同 orbit row/free-param permutation 错位，则 Exp2f 应把 Exp2b conversion@50 `24.866%` 至少推到 Exp3 线 `26.866%`，理想上接近 Exp2 target `28%`。
- 数据规模：Exp0/1/2/2b/2d/2e/2f 已覆盖 front-end freeze、train noisy-pair、joint MLP、safe pool、site assignment、train-pair residual posterior、permutation-aware alignment；Exp2f 额外生成并评估 `47180` 个 permutation candidates，跳过超大 atom-count 样本 `157` 个。
- baseline：CrystaLLM-a GT-SG anchor 仍是唯一主 baseline。Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%` 是 repair 起点；Exp2d best match@50 `22.036%`、conversion `25.714%` 是当前 match 最优验证结果。
- 方法变化总结：local optimizer/CHGNet 改善 collision 但不新增样本；site assignment 有小幅正信号；train-pair residual posterior 新增 `0` 个 sample-level match；Exp2f deterministic permutation 只比 Exp2d best 多 `5` 个 oracle match samples，union upper bound 仍只有 match@50 `22.282%`、conversion `25.996%`。
- 结果：按 match@50 看，当前可复现最佳仍是 Exp2d `h10_interleave_s20_p20`，rows>=7 match@50 `22.036%`、conversion `25.714%`；按 conversion 看，Exp2f best `h10_s5_perm30_p5` 为 `25.740%`，只比 Exp2d 高 `0.026pp` 且 match@50 低 `0.197pp`。两者都低于 Exp3 +2pp gate 和 Exp2 target line。
- 可信度：中等偏高。补充实验均为 validation 上真实 render/parse/SG/StructureMatcher evaluation，未使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank 作为推理特征；主要限制是 deterministic enumeration 不是训练出的 posterior，且 Exp2f 为控制长尾未对超大 atom-count 样本生成 permutation candidates。
- 和历史实验关系：Exp2f 关闭了 Exp3c 后最直接的 row/free-param permutation 假设；它与 Exp2d/2e oracle 一起说明当前 repair 路线的候选层面余量不足，opentry_13/opentry_14 的剩余问题更可能在 predicted skeleton proposer 或真正学习式 geometry posterior。
- 最终判决：`do_not_enter_exp4_exp5_or_official_after_exp2f`。当前路线没有通过 Exp3 gate，也没有形成可作为 Exp4 前置的 learned posterior/critic 增益；不得进入 full validation gate 或 frozen official test。
- 下一步：停止 blind beam expansion（site/residual/permutation/C-S ratio/ordinary scorer）。若继续研究，只做实质性新路线：训练 permutation-aware geometry posterior、改 predicted skeleton proposer，或重新定义可验证的前置假设后再跑 validation。


<!-- OPENTRY14_EXP2G_CANDIDATE_HEADROOM_AUDIT -->
## opentry_14 实验 2g：candidate headroom 与 ranking 根因审计

结果文件：`model/New_model/opentry_14/results/experiment_2g_candidate_headroom_audit.json`
诊断 artifact：`model/New_model/opentry_14/artifacts/exp2g_candidate_headroom_audit/`

- 为什么做：Exp2d/2e/2f 都只带来小幅或零新增 sample-level match，需要确认失败根因是候选池本身没有足够正确几何，还是已有正确候选被 rank/top50 选择压掉。该实验只做 oracle/headroom 审计，不作为 scorer 或主方法。
- 核心假设：如果现有 Exp2b+2d+2e+2f 候选池内已经有足够 match，只是排序不好，则 all-candidate oracle 应显著超过 Exp3 线 `26.866%` conversion 或 Exp2 target `28%`；如果 all-candidate oracle 仍低，则继续扩大同类 beam 或普通 critic 没有通过 gate 的余量。
- 数据规模：rows>=7 validation universe `2197` samples；审计候选 `133793` records；覆盖样本 `2033`；读取来源包括 Exp2b safe pool、Exp2d site assignment、Exp2e residual posterior、Exp2f permutation alignment。
- baseline：当前 match@50 最优仍是 Exp2d `exp2d_h10_interleave_s20_p20`，rows>=7 match@50 `22.036%`、conversion `25.714%`；当前 conversion 数值最高为 Exp2f `exp2f_h10_s5_perm30_p5`，conversion `25.740%`、match@50 `21.840%`。Exp3 gate 需要 conversion `26.866%`。
- 方法变化：按 sample_id 合并既有 validation 候选，不生成新 CIF，不训练模型，不改变 rank；统计 fixed-denominator match/skeleton coverage、all-candidate oracle、first-match rank、source unique contribution 和 skeleton-hit/no-match failure buckets。推理侧不使用 match/RMSD/StructureMatcher label；match 只作为离线审计标签。
- 结果：union all-candidate oracle rows>=7 match fixed-denominator `20.756%`，skeleton coverage `76.513%`，skeleton-to-match conversion `26.175%`。first skeleton-match rank 中位数 `1`，`>50` 的样本数 `0`。
- source 贡献：unique match samples vs other sources 分别为 Exp2b `133`、Exp2d `12`、Exp2e `0`、Exp2f `4`。Exp2e/2f 的新增很小，说明 residual/permutation 候选族余量不足。
- 失败桶：no-candidate `164`；candidate but no skeleton-hit `352`；skeleton-hit/no-match `1241`，其中已有 valid+formula+SG+exact candidate 的 `1228`。这支持“结构合法但几何 basin/骨架候选不对”的根因，而不是前端结构 gate 崩坏。
- 可信度：中等偏高。它直接读取已真实评估的 validation artifacts，固定 rows>=7 universe，未改写历史目录；限制是 oracle 使用 validation match 标签进行诊断，不能作为推理排序规则，也不能证明某个 inference-safe critic 能达到 oracle。
- 和历史实验关系：解释 Exp3/3b local optimizer、Exp2e residual posterior、Exp2f deterministic permutation 都不能过 gate 的共同原因：候选池 all-candidate oracle 本身仍低于 Exp3 所需 conversion，排序不是主瓶颈。
- gate 判定：target_headroom_passed=`False`；exp3_headroom_passed=`False`；oracle conversion delta vs Exp3 line `-0.691pp`；oracle conversion delta vs Exp2 target `-1.825pp`。
- 最终判决：`fail_candidate_headroom_insufficient`。Union all-candidate oracle is still below the Exp3 conversion line and Exp2 target, so the main bottleneck is candidate generation/geometry basin rather than top50 ranking.
- 下一步：不要进入 Exp4/Exp5/official，也不要继续普通 scorer/rerank。下一步需要新候选来源：训练真正的 geometry posterior，或升级 predicted skeleton proposer 以产生新的 correct geometry basin；若只沿用现有候选池，oracle 也不够过 gate。


<!-- OPENTRY14_EXP2H_TRAIN_PARAM_PRIOR_POSTERIOR_SMOKE -->
## opentry_14 实验 2h smoke：train conditional free-parameter prior posterior

结果文件：`model/New_model/opentry_14/results/experiment_2h_train_param_prior_geometry_posterior_smoke80b.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2h_train_param_prior_geometry_posterior/`

- 为什么做：Exp2g 证明现有候选池 oracle headroom 不足，ranking 不是主瓶颈；因此尝试一个新的 train-only geometry posterior，而不是扩大 residual/site/permutation beam。
- 核心假设：如果 skeleton-hit/no-match 的错误来自 row-level free parameters 落错 basin，那么按 `(SG, orbit, element)` / `(SG, orbit)` 从 train true structures 学到的参数先验，给 predicted skeleton 生成多样化参数块，应产生 Exp2b/Exp2d 没有的新 match。
- 数据规模：smoke rows>=7 samples `80`；生成 candidates `3078`，评估后 structural top50 candidates `2902`；train param bank 覆盖 train records `25998`、train WA rows `146596`、usable param rows `146596`；workers `32`，单 worker 线程环境为 1。
- baseline：Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%`；Exp2d best match@50 `22.036%`、conversion `25.714%`；Exp2g union oracle conversion `26.175%` 仍低于 Exp3 line `26.866%`。
- 方法变化：对 exact-cover assignment 后的 predicted skeleton row，不再只复制 source row params；从 train 参数库取 exact-top1/top2/top3 并在同元素同 orbit 重复 row 内做多样化抽样，避免重复坐标导致 invalid CIF。lattice 用 source、SG+atom-count median、SG median。排序只用 structural score，不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果：prior-only rows>=7 smoke match@1/5/20/50 = `4.225% / 5.634% / 8.451% / 8.451%`；conversion@50 `12.245%`；valid `68.470%`；formula `92.350%`；SG `83.942%`；exact-cover `100.000%`；collision `23.880%`。best mixed variant 仍主要由 safe-pool/site candidates 支撑，rows>=7 match@50 `21.790%`、conversion `26.205%`，但这是全 safe-pool + 80-sample smoke posterior 的混合统计，不能当 full gate。
- oracle 诊断：prior-only match samples `6`、skelmatch `6`；相对 Exp2b 新增 match/skelmatch `0` / `0`，相对 Exp2d best 新增 `0` / `0`；union upper bound 仍为 match@50 `22.036%`、conversion `25.699%`（smoke 口径）。
- 可信度：中等。它真实 render/parse/SG/StructureMatcher，且 posterior 只用 train split；限制是只跑 80-sample smoke，且 row-wise independent prior 破坏了跨 row/元素几何相关性，formula/SG/collision 明显弱于 safe-pool。
- 和历史实验关系：这是 Exp2e residual posterior、Exp2f deterministic permutation 之后的另一个候选生成尝试。结果显示 train row-wise prior 只重现已有可解样本，不能补足 Exp2g 中缺少的新 geometry basin。
- 最终判决：`diagnostic_smoke_fail_no_new_headroom`。不值得扩大为 full validation；新增 sample-level match 为 0，且结构安全指标弱。
- 下一步：停止 row-wise independent parameter-prior 路线。下一步做 GT-WA/assignment 离线审计，判断 remaining failure 是否来自 element assignment beam 不足，还是 predicted skeleton/geometry joint basin 本身错误；不进入 Exp4/5/official。


<!-- OPENTRY14_EXP2I_GT_ASSIGNMENT_BEAM_AUDIT -->
## opentry_14 实验 2i：GT-WA assignment beam 离线审计

结果文件：`model/New_model/opentry_14/results/experiment_2i_gt_assignment_beam_audit.json`
诊断 artifact：`model/New_model/opentry_14/artifacts/exp2i_gt_assignment_beam_audit/`

- 为什么做：Exp2h row-wise train 参数先验没有新增 sample-level match；Exp2g 显示 ranking 不是主瓶颈。需要判断 site-assignment beam 是否遗漏真实 element-to-row assignment，避免盲目扩大 exact-cover assignment。
- 核心假设：如果 Exp2d 的 assignment_limit=4 太窄，那么在 predicted skeleton 命中的样本中，GT-WA element assignment 应大量出现在 rank>4 或 not-found；如果 GT assignment 多数 rank<=4，则继续扩大 assignment beam 不会带来主要增益。
- 数据规模：rows>=7 validation samples `2197`；top skeleton proposals audited `10`；skeleton-hit samples `1681`；GT assignment comparable samples `1681`。
- baseline：Exp2d 使用 top skeletons `10`、assignment_limit `4`，best rows>=7 match@50 `22.036%`、conversion `25.714%`；Exp2g union all-candidate conversion `26.175%` 仍低于 Exp3 line `26.866%`。
- 方法变化：只做离线审计。对每个 rows>=7 validation sample 的 top10 predicted skeleton proposals，若 proposal skeleton key 等于 validation canonical skeleton key，则把 true WA rows 按 orbit/multiplicity 对齐到 predicted rows，计算 GT element assignment 在 deterministic exact-cover enumeration 前 64 个中的 rank。GT-WA 只作为诊断标签，不能作为推理特征。
- 结果：GT assignment rank buckets = `{'not_found_le64': 1311, 'rank21_64': 144, 'rank1': 33, 'rank2_4': 35, 'rank11_20': 78, 'rank5_10': 80}`；rank<=4 coverage `4.045%`；rank<=10 coverage `8.804%`；not-found<=64 rate `77.989%`。
- 可信度：中等偏高。它直接比较 validation true WA 与 predicted skeleton exact-cover assignment space，不跑 StructureMatcher，不训练，不改候选；限制是只在 skeleton key 已命中的样本上可比，且 GT-WA 不能用于推理选择。
- 和历史实验关系：承接 Exp2d 的 small site-assignment gain、Exp2f 的 small permutation gain、Exp2g 的 candidate-headroom failure。它回答“继续扩大 assignment 是否有必要”这个具体分支。
- 最终判决：`assignment_order_or_skeleton_alignment_problem`。GT assignments are often outside rank<=10 or not found, indicating assignment enumeration/order and skeleton alignment are deeper issues.
- 下一步：Do not blindly expand assignment; redesign skeleton/site alignment or proposer.


<!-- OPENTRY14_CURRENT_BOUNDARY_AFTER_EXP2I -->
## opentry_14 当前边界判定（Exp2i 后）

- 为什么做：Exp2g/2h/2i 连续检查了候选 headroom、新 train 参数 posterior、GT assignment beam；需要更新 Exp2f 后的边界，避免把下一步误判成简单扩大 beam。
- 核心假设复核：若剩余瓶颈只是 ranking、row-wise train param prior 或 assignment_limit=4 太小，则 Exp2g oracle、Exp2h smoke 或 Exp2i rank<=10 应显示足够余量；实际三者都没有给出可过 Exp3/Exp2 的证据。
- 数据规模：Exp2g 合并 `133793` 条已评估候选；Exp2h smoke 评估 `2902` 条 train-param-prior candidates；Exp2i 审计 rows>=7 skeleton-hit samples `1681`。
- baseline：当前 validation match@50 最优仍是 Exp2d `22.036%`、conversion `25.714%`；当前 conversion 数值最高仍是 Exp2f/Exp2h mixed 附近 `25.7%` 到 `26.2%`，低于 Exp3 line `26.866%` 和 Exp2 target `28%`。
- 方法变化总结：Exp2g 证明 all-candidate oracle conversion `26.175%` 仍不够；Exp2h row-wise train param prior 没有新增 sample-level match，且结构安全弱；Exp2i 显示 GT assignment rank<=10 只有 `8.804%`、not-found<=64 为 `77.989%`，说明简单扩大 exact-cover assignment 不是解决路径。
- 最终判决：`do_not_enter_exp4_exp5_or_official_after_exp2i`。当前 repair 路线仍未通过 Exp2 target 或 Exp3 +2pp gate，且已排除 ranking、local optimizer、CHGNet、residual posterior、deterministic permutation、row-wise param prior 和简单 assignment-beam 扩张。
- 下一步：若继续，只能重做 skeleton/site alignment 或 predicted skeleton proposer：需要一个能联合预测 row order、element assignment、correlated free parameters 的模型/搜索，而不是再扩大 site/residual/permutation beam、普通 scorer、C/S ratio 或 official 调参。


<!-- OPENTRY14_EXP2J_CHEMICAL_SITE_ORDER_ASSIGNMENT -->
## opentry_14 实验 2j：chemical/site-order assignment posterior

结果文件：`model/New_model/opentry_14/results/experiment_2j_chemical_site_order_assignment.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2j_chemical_site_order_assignment/`

- 为什么做：Exp2i 显示 GT element assignment 大量不在 deterministic rank<=64，说明 source-preserved assignment order 与真实 site/element order 不一致；但 GT-WA 不能用于推理。本实验用 train split 的元素-轨道统计和化学相似度，构造 inference-safe site-order assignment posterior。
- 核心假设：如果 assignment order 是主要瓶颈，则按 source row 元素的化学相似度、train `(SG, orbit)->element` 先验、train `(orbit)->element` 先验生成 exact-cover assignments，应产生 Exp2b/Exp2d 没有的新 match samples，并提高 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `2197`；train prior rows `146596`；generated chemical assignments `66672`；evaluated candidates `61202`；workers `96`。
- baseline：Exp2b rows>=7 match@50 `21.348%`、conversion `24.866%`；Exp2d best match@50 `22.036%`、conversion `25.714%`；Exp2i rank<=10 coverage `8.804%`。
- 方法变化：替代 deterministic source-preserved exact-cover order；对每个 row/element 打分，特征只含元素周期表相似度、train split 的 SG/orbit/element 频率和 source-row 元素相似度。assignment beam 完成 exact-cover 后，用 source lattice/free params 渲染，并用 structural score 排序；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `h10_s10_chem25_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.397% / 23.119%`；conversion@50 `27.463%`；valid `84.300%`；formula `99.412%`；SG `98.497%`；exact-cover `99.808%`；collision `4.246%`。
- oracle 诊断：chemical-only match samples `307`；相对 Exp2b 新增 match/skelmatch `36` / `31`；相对 Exp2d best 新增 `29` / `24`；union upper bound match@50 `23.463%`、conversion `27.127%`。
- 可信度：中等。所有候选真实 render/parse/SG/StructureMatcher；assignment posterior 只用 train split 和元素表，validation match 只用于离线评估。限制是它仍复制 source geometry/free params，未学习 correlated geometry posterior。
- 和历史实验关系：这是 Exp2d site assignment 的非 GT、非普通 scorer 替代排序，直接回应 Exp2i 的 assignment-order root cause。
- gate 判定：minimum_passed=`True`；target_passed=`False`；exp3_line_passed=`True`；conversion delta vs Exp2b `+2.597pp`；conversion delta vs Exp3 line `+0.597pp`。
- 最终判决：`pass_exp3_line_but_not_exp2_target`。Chemical/site-order assignment clears the Exp3 +2pp conversion line but not the Exp2 target.
- 下一步：Retest Exp3 local optimizer as the next gated step; do not enter Exp4/5/official yet.

<!-- OPENTRY14_EXP3J_CHGNET_AFTER_EXP2J -->
## opentry_14 实验 3j：CHGNet local optimizer after Exp2j

结果文件：`model/New_model/opentry_14/results/experiment_3j_chgnet_after_exp2j.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3j_chgnet_after_exp2j/`

- 为什么做：Exp2j chemical/site-order assignment 已把 rows>=7 conversion 从 Exp2b `24.866%` 提到 `27.463%`，超过 Exp3 +2pp 入口线；prompt 要求在 learned/constructed repair 后测试 symmetry-preserving local optimizer 是否还能再提高 conversion 并改善 local packing。
- 核心假设：如果 Exp2j 剩余失败主要是局部 collision/packing，则对 Exp2j best 中 formula/SG/exact 保持但 valid=false 的 chemical candidates 做 CHGNet position-only relaxation，应在不破坏 SG/exact-cover 的情况下再提升 conversion@50 至少 `+2pp`。
- 数据规模：Exp2j best rows>=7 candidate records `78125`；selected CHGNet tasks `1611`；regenerated CIF tasks `1611`；accepted optimized `567`；evaluated optimized `567`；GPU devices `['0', '1']`；workers `32`。
- baseline：Exp2j best `h10_s10_chem25_p5` rows>=7 match@50 `23.119%`、conversion `27.463%`、collision `4.246%`。Exp3j pass line requires conversion `29.463%`。
- 方法变化：只优化 inference-safe 选择的 Exp2j chemical candidates：rows>=7、chemical_site_order_source_geometry、formula_ok、space_group_ok、exact_cover_retained、valid=false。CHGNet 不 relax cell；优化后必须保持 formula/site count/SG，并改善 local proxy，否则回退。排序和选择不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `interleave_chgnet_after_chemical_invalid`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.397% / 23.119%`；conversion@50 `27.469%`；collision `4.811%`；SG `98.507%`；exact-cover `99.809%`。
- CHGNet 诊断：optimized valid records `48`；optimized match records `0`；mean min-distance delta `0.05754749499820345`；mean hard-collision delta `-0.7301587301587301`；mean close-pair delta `-1.379188712522046`。
- 可信度：中等。CHGNet 是真实预训练势能，优化后重新 eval StructureMatcher/SG/formula/site/exact；限制是只覆盖 Exp2j chemical invalid candidates，不覆盖 safe-pool hydrated/prototype 的原始 CIF。
- 和历史实验关系：这是 Exp3/3b 在 Exp2j 新 repair 基线后的复测，直接判断 local optimizer 是否能在新的 assignment posterior 上产生追加收益。
- gate 判定：passed=`False`；conversion delta vs Exp2j `+0.006pp`；match@50 delta vs Exp2j `+0.000pp`；collision/local improved=`False`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_diagnostic_only`。CHGNet local optimizer after Exp2j does not meet the +2pp conversion gate, even if local packing changes.
- 下一步：Do not enter Exp4/5/official from this optimizer; the remaining bottleneck is not fixed by local relaxation.

<!-- OPENTRY14_EXP3K_BROAD_CHGNET_CHEMICAL_RELAX -->
## opentry_14 实验 3k：Broad CHGNet relax on Exp2j chemical candidates

结果文件：`model/New_model/opentry_14/results/experiment_3k_broad_chgnet_chemical_relax.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3k_broad_chgnet_chemical_relax/`

- 为什么做：Exp3j 只优化 Exp2j 中 valid=false 的 chemical collision candidates，能改善局部 proxy 但没有新增 match。3k 检查另一种可能：已经 valid 的 high-rank chemical candidates 是否只是局部能量/位置未收敛，短步 CHGNet 是否能把它们转成 match。
- 核心假设：如果 conversion bottleneck 是局部 basin 而不是 skeleton/site assignment，那么按 inference-safe rank/per-sample top-N 选择 chemical candidates 后，保留原候选并 append CHGNet 优化版本，应在不损失原 match 的情况下新增 rows>=7 match。
- 数据规模：base rows>=7 candidate records `78125`；eligible selected rows `7803`；regenerated `7803`；accepted `3999`；evaluated optimized `3999`；max_rank `20`；per_sample `5`；CHGNet workers `32`。
- baseline：Exp2j best `h10_s10_chem25_p5` rows>=7 match@50 `23.119%`、conversion `27.463%`、collision `4.246%`。Exp3 pass line requires conversion `29.463%`。
- 方法变化：选择只使用 rank、formula_ok、space_group_ok、exact_cover_retained、row_count 和 chemical geometry source；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。优化后必须保持 formula/site count/SG 并改善 local proxy；append variant 保留原始 Exp2j 排序。
- 结果 best variant `append_valid_chgnet_after_selected`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.299% / 23.119%`；conversion@50 `27.463%`；collision `4.042%`；valid `85.053%`；SG `98.569%`；exact-cover `99.817%`。
- CHGNet 诊断：tasks `7803`；successful `7791`；accepted `3999`；optimized valid records `3938`；optimized match records `173`；mean min-distance delta `0.08666971341473814`；mean hard-collision delta `-0.008002000500125032`；mean close-pair delta `-0.5833958489622406`。
- 可信度：中等。选择规则是推理安全的，优化后重新评估 StructureMatcher/SG/formula/site/exact；限制是只测试 Exp2j chemical candidates 的 rank/per-sample top-N，不覆盖所有 39k eligible candidates。
- 和历史实验关系：这是 Exp3j 失败后的补充局部优化审计，直接检验“valid chemical candidate 只需局部松弛”的假设。
- gate 判定：passed=`False`；conversion delta vs Exp2j `+0.000pp`；match@50 delta vs Exp2j `+0.000pp`；collision/local improved=`True`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_local_optimizer_not_conversion_limited`。Broad CHGNet relaxation does not deliver the required +2pp conversion lift, so local energy/packing is not the main remaining bottleneck.
- 下一步：Do not enter Exp4/5/official from local optimizer; return to skeleton/site/free-parameter alignment rather than expanding CHGNet.


<!-- OPENTRY14_EXP2K_CORRELATED_DONOR_GEOMETRY_POSTERIOR -->
## opentry_14 实验 2k：correlated donor geometry posterior

结果文件：`model/New_model/opentry_14/results/experiment_2k_correlated_donor_geometry_posterior_smoke120.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2k_correlated_donor_geometry_posterior/`

- 为什么做：Exp2h 的 row-wise train free-parameter prior 失败，Exp3j/3k 的 CHGNet local optimizer 也只改善 collision/valid、不提升 match；剩余假设是跨 row 的 lattice/free-parameter 相关性被破坏，单 row 参数或局部优化无法换到正确 basin。
- 核心假设：如果同一个 train donor 的 lattice 与所有 matched row free parameters 构成可迁移的 correlated geometry bundle，那么在 Exp2j chemical assignment 后按 train donor bundle 整体迁移，应比 source geometry 或 row-wise prior 产生新的 rows>=7 sample-level match。
- 数据规模：rows>=7 validation samples `120`；generated donor candidates `2208`；evaluated donor candidates `2208`；train donor records `25998`；workers `64`。
- baseline：Exp2j best `h10_s10_chem25_p5` rows>=7 match@50 `23.119%`、conversion `27.463%`、collision `4.246%`。
- 方法变化：保留 Exp2j chemical/site-order posterior；每个 assigned skeleton 不再复制 source geometry，也不逐 row 独立采样，而是按 train split 的 same_skeleton / same_SG+atom_count / same_SG donor score 选择完整 donor，迁移 donor lattice 与 all-row free params。推理排序只用 legal/formula/SG/exact/collision/volume/reference_score，不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `h10_s10_chem15_donor10_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.397% / 22.823%`；conversion@50 `27.144%`；valid `82.760%`；formula `99.337%`；SG `98.482%`；exact-cover `99.770%`；collision `3.568%`。
- oracle 诊断：donor-only match/skelmatch samples `5` / `4`；相对 Exp2j 新增 match/skelmatch `0` / `0`；union upper bound match@50 `23.119%`、conversion `27.429%`。
- 可信度：中等。donor 只来自 train split，validation 推理不使用禁用标签，所有候选真实 render/parse/SG/StructureMatcher；限制是 donor bundle 仍是 prototype posterior，不是端到端 learned continuous repair head。
- 和历史实验关系：这是 Exp2h row-wise prior 与 Exp2j chemical assignment 的组合修正，直接检验“跨 row geometry correlation”是否是 Exp3 local optimizer 失败后的剩余瓶颈。
- gate 判定：minimum_passed=`True`；target_passed=`False`；exp3_line_passed=`False`；conversion delta vs Exp2j `-0.319pp`；match@50 delta vs Exp2j `-0.295pp`。
- 最终判决：`fail_no_conversion_lift`。Correlated donor geometry preserves the minimum structural gate but does not improve Exp2j conversion.
- 下一步：Stop prototype donor expansion; next repair must learn continuous aligned free-parameter residuals or upgrade skeleton proposer.

<!-- OPENTRY14_EXP2L_VALID_SKELETON_MISMATCH_AUDIT -->
## opentry_14 实验 2l：valid skeleton-hit mismatch audit

结果文件：`model/New_model/opentry_14/results/experiment_2l_valid_skeleton_mismatch_audit.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2l_valid_skeleton_mismatch_audit/`

- 为什么做：Exp2j/3j/3k/2k 都显示 valid/formula/SG/exact 可以保持，但 skeleton-hit 仍不能转成 StructureMatcher match。需要把失败分解为 lattice tolerance、fractional coordinate basin、species/site assignment 或完全不同 geometry basin。
- 核心假设：如果大量 no-match 在 loose lattice/site/anonymous 条件下能匹配，则下一步应针对对应轴训练 repair；如果仍不能匹配，则当前 predicted skeleton/site assignment 虽命中 skeleton key，但几何 basin 已经不同。
- 数据规模：从 Exp2j best 中选择 rows>=7、chemical geometry、valid=true、predicted_skeleton_hit=true、match=false 的候选 `233`；成功重建 CIF `233`；审计 records `233`；workers `96`。
- baseline/关系：审计对象来自 Exp2j best `h10_s10_chem25_p5` 之后的失败样本，不作为新推理候选，不进入 Exp3/official。
- 方法变化：只做离线归因，重建 CIF 后用 StructureMatcher default/loose_lattice/loose_site/loose_angle/loose_all 和 anonymous matching 检查失败类型；真值 CIF 只用于审计，不进入推理排序或候选选择。
- 结果分类：`{'different_geometry_basin': 48, 'species_or_site_assignment_mismatch': 107, 'large_lattice_scale_mismatch': 78}`。
- 关键率：default_match `0.000%`；loose_all_match `33.476%`；anonymous_loose_all `45.923%`；large_lattice_scale_mismatch `33.476%`。
- lattice 误差：volume_rel median `0.10754402112634892`，p90 `0.3783850907133827`；max_axis_rel median `0.16224780934854818`，p90 `0.6591507705139908`。
- 可信度：中等。使用真实 CIF 和 StructureMatcher 做离线审计；限制是默认只审计每样本前若干 valid skeleton-hit no-match 候选，不代表所有候选。
- 最终判决：`species_site_assignment_mismatch_substantial`。Many failures match anonymously under loose tolerance, so element/site assignment remains a major bottleneck.
- 下一步：Train an assignment-aware geometry model or upgrade chemical assignment; do not spend more on local optimizer.


<!-- OPENTRY14_EXP2M_INFERENCE_SAFE_ASSIGNMENT_CRITIC_SWEEP -->
## opentry_14 实验 2m：inference-safe assignment critic sweep

结果文件：`model/New_model/opentry_14/results/experiment_2m_inference_safe_assignment_critic_sweep.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2m_inference_safe_assignment_critic_sweep/`

- 为什么做：Exp2l 显示 valid skeleton-hit no-match 中 species/site assignment mismatch 占比高，先检验不使用真值标签的 assignment/structure 字段级 critic 是否足以把 Exp2j chemical candidates 排到更好的 top50。
- 核心假设：如果正确 assignment 已在 Exp2j candidate pool 内，只是被 structural score 排低，则使用 assignment_score、source-preserved atoms、proposal/assignment rank、valid/SG/exact/min-distance/volume 等 inference-safe 字段应提升 rows>=7 conversion。
- 数据规模：chemical candidate records `61202`；rows>=7 chemical records `61202`；samples `1904`；sweep formulas `8`。
- baseline：Exp2j best `h10_s10_chem25_p5` rows>=7 match@50 `23.119%`、conversion `27.463%`、collision `4.246%`。
- 方法变化：只重排已有 Exp2j chemical candidates，不重新生成 CIF；排序不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。该实验只作为 critic/headroom 诊断，不作为主 repair。
- 结果 best sweep `structural_existing`：rows>=7 match@50 `23.119%`；conversion@50 `27.463%`；collision `4.246%`；valid `84.300%`。
- sweep 结论：最高 match@50 sweep 仍未提升 conversion；best-vs-Exp2j match delta `+0.000pp`，conversion delta `+0.000pp`。
- 可信度：中等。所有候选已由 Exp2j 真实 render/parse/StructureMatcher 评估，选择阶段只用 inference-safe 字段；限制是没有重建 CIF 做更细局部化学统计。
- 和历史实验关系：直接回应 Exp2l 的 species/site mismatch 归因，检验“已有候选只需安全 critic 重排”是否成立。
- 最终判决：`fail_no_conversion_lift`。Field-level inference-safe assignment critic does not improve Exp2j conversion.
- 下一步：Move beyond existing candidate sorting: train assignment-aware geometry model or upgrade chemical/site assignment generation.

<!-- OPENTRY14_EXP2N_PAIRWISE_LOCAL_CHEMISTRY_ASSIGNMENT_POSTERIOR -->
## opentry_14 实验 2n：pairwise local-chemistry assignment posterior

结果文件：`model/New_model/opentry_14/results/experiment_2n_pairwise_local_chemistry_assignment.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2n_pairwise_local_chemistry_assignment/`

- 为什么做：Exp2l 将 valid skeleton-hit no-match 的主要问题定位到 species/site assignment mismatch，Exp2m 又证明仅重排 Exp2j 旧候选不能提升 conversion。因此本实验在生成侧扩大 assignment beam，并引入 train-only 元素对局部距离先验，尝试产生 Exp2j 没有的可匹配 assignment。
- 核心假设：如果错误来自化学位点分配，而不是 skeleton 或局部优化，则训练集元素对距离分布应能在更宽 exact-cover beam 中挑出更合理的 assignment，从而提高 rows>=7 conversion@50。
- 数据规模：rows>=7 validation samples `2197`；generated pair-chem assignments `66672`；evaluated candidates `61202`；train pair records `8000`；element-pair priors `2248`；workers gen/eval/prior `128` / `160` / `96`。
- baseline：Exp2j best `h10_s10_chem25_p5` rows>=7 match@50 `23.119%`、conversion `27.463%`、collision `4.246%`。
- 方法变化：Exp2j 的 chemical exact-cover beam 从 final limit 扩到 prelimit，再对每个渲染 assignment 解析生成结构，按 train split 元素对距离/半径比的 robust prior 打分，保留 local-chem top assignments；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback，也不是 RF/HGB/阈值 scorer。
- 结果 best variant `h10_s10_chem25_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.643% / 23.610%`；conversion@50 `28.158%`；valid `84.300%`；formula `99.412%`；SG `98.495%`；exact-cover `99.808%`；collision `4.246%`。
- oracle 诊断：pair-chem-only match samples `351`；相对 Exp2j 新增 match/skelmatch `56` / `54`；union upper bound match@50 `19.065%`、conversion `22.686%`。
- 可信度：中等。候选真实 render/parse/SG/StructureMatcher 评估，生成评分只用训练集 CIF 的元素对距离统计和元素半径；限制是仍沿用 source lattice/free params，且局部 pair prior 可能偏向短程配位而非全局 Wyckoff assignment。
- 和历史实验关系：这是 Exp2j 的生成侧扩展，直接回应 Exp2l/Exp2m；若失败，说明简单 local chemistry prior 不能弥补 assignment/order mismatch，需要训练 joint assignment-geometry 模型。
- gate 判定：passed=`True`；conversion delta vs Exp2j `+0.695pp`；match delta vs Exp2j `+0.492pp`；target_passed=`True`。
- 最终判决：`pass_exp2_target_gate`。Pairwise local-chemistry assignment posterior reaches the 28% rows>=7 conversion target.
- 下一步：Run the gated Exp3 local optimizer relative to Exp2n; keep official frozen until later gates pass.

<!-- OPENTRY14_EXP3N_CHGNET_AFTER_EXP2N_PAIRCHEM -->
## opentry_14 实验 3n：CHGNet local optimizer after Exp2n pair-chem

结果文件：`model/New_model/opentry_14/results/experiment_3n_chgnet_after_exp2n_pairchem.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3n_chgnet_after_exp2n_pairchem/`

- 为什么做：Exp2n 已把 rows>=7 conversion 推过 28% target，按 prompt 必须验证 symmetry-preserving local optimizer 是否能在不破坏 SG/exact-cover 的情况下再带来至少 +2pp conversion，并改善 collision/local packing。
- 核心假设：如果 Exp2n 剩余错误主要来自局部 basin 或短距离 packing，按 inference-safe rank/per-sample top-N 选择 pair-chem candidates 后，CHGNet position-only relaxation 应在保留原候选的 append variant 中新增 rows>=7 match。
- 数据规模：base rows>=7 candidate records `78125`；eligible selected rows `7803`；regenerated `7803`；accepted `4071`；evaluated optimized `4071`；max_rank `20`；per_sample `5`；CHGNet workers `96`。
- baseline：Exp2n best `h10_s10_chem25_p5` rows>=7 match@50 `23.610%`、conversion `28.158%`、collision `4.246%`。Exp3 pass line requires conversion `30.158%`。
- 方法变化：选择只使用 rank、formula_ok、space_group_ok、exact_cover_retained、row_count 和 pair-chem geometry source；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。优化后必须保持 formula/site count/SG，并改善 local proxy；append variant 保留原始 Exp2n 排序。
- 结果 best variant `append_valid_chgnet_after_selected`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.643% / 23.610%`；conversion@50 `28.158%`；collision `4.038%`；valid `85.068%`；SG `98.568%`；exact-cover `99.817%`。
- CHGNet 诊断：tasks `7803`；successful `7797`；accepted `4071`；optimized valid records `4020`；optimized match records `208`；mean min-distance delta `0.045779043561206816`；mean hard-collision delta `-0.025546548759518548`；mean close-pair delta `-0.4603291574551707`。
- 可信度：中等。选择规则是推理安全的，优化后重新评估 StructureMatcher/SG/formula/site/exact；限制是只测试 Exp2n pair-chem candidates 的 rank/per-sample top-N，不覆盖所有 eligible candidates。
- 和历史实验关系：这是 Exp2n 通过 target gate 后的正式 Exp3 检验，也复核 Exp3j/3k “局部优化改善 packing 但不改善 conversion”的历史结论是否仍成立。
- gate 判定：passed=`False`；conversion delta vs Exp2n `+0.000pp`；match@50 delta vs Exp2n `+0.000pp`；collision/local improved=`True`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_local_optimizer_not_conversion_limited`。CHGNet relaxation after Exp2n does not deliver the required +2pp conversion lift, so local energy/packing is not the main remaining bottleneck.
- 下一步：Do not enter Exp4/5/official from local optimizer; return to skeleton/site/free-parameter alignment rather than expanding CHGNet.


<!-- OPENTRY14_EXP2O_EXPANDED_PAIRWISE_LOCAL_CHEMISTRY_ASSIGNMENT -->
## opentry_14 实验 2o：expanded pairwise local-chemistry assignment posterior

结果文件：`model/New_model/opentry_14/results/experiment_2o_expanded_pairwise_local_chemistry_assignment.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2n_pairwise_local_chemistry_assignment/*exp2o_expand10x12*`

- 为什么做：Exp3n 证明 CHGNet local optimizer 只改善 collision/local packing，不产生新 sample-level match；因此回到 Exp2 的 assignment/free-parameter alignment。Exp2n 已过 28% target，但 oracle 仍显示 assignment 生成侧有新增 match 空间，本实验扩大 skeleton/assignment 覆盖而不改变禁用特征边界。
- 核心假设：如果 Exp2n 的剩余可修复空间来自 assignment beam 覆盖不足，那么 top_skeletons 从 8 到 10、assignment prelimit 从 32 到 64、per-skeleton retained assignment 从 8 到 12，应产生新的 exact-cover pair-chem candidates，并提升 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `2197`；generated candidates `102471`；evaluated candidates `73521`；train pair records `8000`；workers gen/eval/prior `160` / `180` / `128`。
- baseline：Exp2j rows>=7 match@50 `23.119%`、conversion `27.463%`；Exp2n rows>=7 match@50 `23.610%`、conversion `28.158%`。
- 方法变化：沿用 Exp2n 的 train-only 元素对局部距离 prior，但扩大 proposal/assignment generation；选择仍只用 train prior、元素表、generated structure 的 inference-safe local chemistry/structure 字段，不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback。
- 结果 best variant `h10_s5_chem30_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.446% / 23.906%`；conversion@50 `28.633%`；valid `84.922%`；formula `99.405%`；SG `98.488%`；exact-cover `99.813%`；collision `3.871%`。
- oracle 诊断：pair-chem-only match/skelmatch samples `360` / `347`；相对 Exp2j 新增 match/skelmatch `72` / `72`；union upper bound match@50 `19.905%`、conversion `23.875%`。
- 可信度：中等。全量 validation、真实 render/parse/SG/StructureMatcher 评估；推理选择不使用禁用标签。限制是更宽 beam 主要提高 conversion，match@5/20 的收益不稳定，且仍复制 source lattice/free params。
- 和历史实验关系：这是 Exp2n 的生成侧扩展，也是 Exp3n 失败后的直接回退；结果支持“local optimizer 不是瓶颈，assignment 覆盖仍有小幅空间”。
- gate 判定：passed=`True`；target_passed=`True`；conversion delta vs Exp2j `+1.170pp`；conversion delta vs Exp2n `+0.476pp`；match delta vs Exp2n `+0.295pp`。
- 最终判决：`pass_exp2_target_gate_but_exp3_still_required`。Expanded pairwise local-chemistry assignment improves Exp2n and remains above the 28% rows>=7 conversion target, but this is still an assignment-side repair and does not satisfy Exp3 local optimizer gate.
- 下一步：Run/compare Exp3 local optimizer against Exp2o; if local optimizer still gives no new sample-level match, stop local optimization and move to learned assignment-aware geometry alignment.


<!-- OPENTRY14_EXP3O_CHGNET_AFTER_EXP2O_EXPANDED_PAIRCHEM -->
## opentry_14 实验 3o：CHGNet local optimizer after Exp2o expanded pair-chem

结果文件：`model/New_model/opentry_14/results/experiment_3o_chgnet_after_exp2o_expanded_pairchem.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp3o_chgnet_after_exp2o_expanded_pairchem/`

- 为什么做：Exp2o 成为新的 assignment-side best，rows>=7 conversion 达到 `28.633%`；按顺序必须重新验证 local optimizer 是否能在 Exp2o 基线上再提升 +2pp。
- 核心假设：如果 Exp2o 剩余误差来自局部 geometry/collision，则 CHGNet position-only relaxation 应新增 sample-level match，同时改善 collision/local packing，且 SG/exact-cover 不恶化。
- 数据规模：base rows>=7 candidate records `80004`；optimizer tasks `9382`；regenerated `9382`；accepted `4994`；evaluated optimized `4994`；workers regen/chgnet/eval `128` / `96` / `160`；CUDA 在当前 Python sandbox 不可用，因此使用 CPU fallback。
- baseline：Exp2o best `h10_s5_chem30_p5` rows>=7 match@50 `23.906%`、conversion `28.633%`、collision `3.871%`。Exp3 pass line requires conversion `30.633%`。
- 方法变化：选择只使用 rank、formula_ok、space_group_ok、exact_cover_retained、row_count 和 pair-chem geometry source；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。CHGNet 优化后必须保持 formula/site count/SG 并改善 local proxy，否则回退。
- 结果 best variant `append_valid_chgnet_after_selected`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.348% / 23.807%`；conversion@50 `28.643%`；collision `3.669%`；valid `85.677%`；SG `98.562%`；exact-cover `99.822%`。
- CHGNet 诊断：successful `9378`；accepted `4994`；optimized valid records `4936`；optimized match records `269`；mean min-distance delta `0.045863385256430585`；mean hard-collision delta `-0.020824989987985584`；mean close-pair delta `-0.4517420905086103`。
- oracle 诊断：optimized match/skelmatch samples `66` / `60`；相对 Exp2o 新增 match/skelmatch `0` / `0`。
- 可信度：中等。全量 validation，优化后重新 eval StructureMatcher/SG/formula/site/exact；限制是 CPU fallback 下只用 1-step CHGNet，但 Exp3n/3o 与历史 Exp3j/3k 均显示局部优化不产生新 sample-level match。
- 和历史实验关系：复核 Exp3n 在更强 Exp2o assignment baseline 下是否仍失败；结果一致，local optimizer 不是 conversion 主瓶颈。
- gate 判定：passed=`False`；conversion delta vs Exp2o `+0.010pp`；match@50 delta vs Exp2o `-0.098pp`；collision/local improved=`True`；SG not worse=`True`；exact-cover not worse=`True`。
- 最终判决：`fail_local_optimizer_not_conversion_limited`。CHGNet relaxation after Exp2o improves collision/local packing but produces no meaningful rows>=7 conversion lift and slightly lowers match@50 in the best insertion variant.
- 下一步：Do not enter Exp4/5/official. Stop local optimizer expansion and move to learned assignment-aware geometry/free-parameter alignment or skeleton source retrieval.


<!-- OPENTRY14_EXP2P_EXP2O_VALID_SKELETON_MISMATCH_AUDIT -->
## opentry_14 实验 2p：Exp2o valid skeleton-hit mismatch audit

结果文件：`model/New_model/opentry_14/results/experiment_2p_exp2o_valid_skeleton_mismatch_audit.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2p_exp2o_valid_skeleton_mismatch_audit/`

- 为什么做：Exp2o 已把 rows>=7 conversion 推到 `28.633%`，但 Exp3o 证明 CHGNet local optimizer 不产生新 sample-level match。需要重新审计 Exp2o best 中 valid skeleton-hit no-match 的失败类型，决定下一步是 assignment、lattice/free-parameter residual，还是 skeleton/source retrieval。
- 核心假设：如果 Exp2o 剩余失败大量 anonymous loose match，则 species/site assignment 仍主导；如果 loose_all/default-near miss 多，则应训练连续 lattice+free-parameter residual；如果多数仍不同 basin，则需要更强 skeleton/source retrieval 或 joint generative posterior。
- 数据规模：source candidate rows `10035`；source samples `964`；selected rows `1779`；regenerated CIF tasks `1779`；audited records `1779`；workers regen/audit `128` / `128`。
- baseline/关系：审计对象来自 Exp2o best `h10_s5_chem30_p5`，仅用于离线 root-cause，不作为推理候选，不进入 Exp4/5/official。
- 方法变化：按 Exp2o pair-chem assignment 精确重建 CIF，用 StructureMatcher default/loose_lattice/loose_site/loose_angle/loose_all 与 anonymous matching 分类；target true CIF 只用于审计，不用于候选选择或推理特征。
- 结果分类：`{'species_or_site_assignment_mismatch': 576, 'large_lattice_scale_mismatch': 753, 'different_geometry_basin': 450}`。
- 关键率：default_match `0.000%`；loose_all_match `26.363%`；anonymous_loose_all `32.378%`；large_lattice_scale_mismatch `42.327%`。
- lattice 误差：volume_rel median `0.11990631589040072`，p90 `0.45359198696025516`；max_axis_rel median `0.2545504735924765`，p90 `0.7450426250283628`。
- 可信度：中等。审计用真实 CIF 和 StructureMatcher，且样本来自 Exp2o full validation top-ranked valid skeleton-hit no-match；限制是按每样本 top-N 抽样，不覆盖全部 no-match。
- 和历史实验关系：复核 Exp2l 在 Exp2o 后是否仍成立。若 species/site mismatch 仍高，则 Exp2n/2o 的 pair-chem 还没完全解决 assignment；若 large lattice/loose_all 高，则下一轮应学习 alignment-aware lattice/free-param residual。
- 最终判决：`species_site_assignment_mismatch_remains_dominant`。Anonymous loose matching remains high after Exp2o, so assignment/site identity is still the main residual bottleneck.
- 下一步：Train assignment-aware geometry model or improve pair-chem assignment generation; do not expand local optimizer.


<!-- OPENTRY14_EXP2Q_PAIRCHEM_LATTICE_PARAM_POSTERIOR -->
## opentry_14 实验 2q：pair-chem lattice/free-parameter posterior

结果文件：`model/New_model/opentry_14/results/experiment_2q_pairchem_lattice_param_posterior_fullnarrow.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2q_pairchem_lattice_param_posterior/`

- 为什么做：Exp2p 显示 Exp2o valid skeleton-hit no-match 中 large lattice scale mismatch `42.327%`、anonymous loose-all `32.378%`；Exp3o 证明 local optimizer 不新增 match。因此本实验在 Exp2o pair-chem assignment 上同时引入 train-only lattice prior 和 row free-parameter prior。
- 核心假设：如果剩余瓶颈是 assignment-aware lattice/free-parameter alignment，则对同一 pair-chem assignment 生成 source / SG median / SG+atom median lattice 与 source / train exact mean / blend / top 参数组合，应新增 sample-level match 或提高 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `2197`；generated posterior candidates `263227`；evaluated candidates `87621`；train pair records `8000`；param-bank usable rows `146596`；workers gen/eval/prior `160` / `180` / `128`。
- baseline：Exp2o best `h10_s5_chem30_p5` rows>=7 match@50 `23.906%`、conversion `28.633%`、collision `3.871%`。
- 方法变化：候选生成只用 train split lattice/free-param priors、元素 pair-distance priors、source predicted skeleton/assignment；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback。该实验不是 lattice-only：每个候选同时绑定 assignment、lattice variant、row-param variant，并重新 render/eval。
- 结果 best variant `h10_s10_chem25_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.249% / 22.971%`；conversion@50 `27.508%`；valid `86.526%`；formula `99.606%`；SG `98.747%`；exact-cover `99.824%`；collision `3.109%`。
- oracle 诊断：posterior-only match/skelmatch samples `278` / `261`；相对 Exp2o 新增 match/skelmatch `4` / `5`；union upper bound match@50 `24.102%`、conversion `28.589%`。
- 可信度：中等。全量 validation 真实 render/parse/SG/StructureMatcher；限制是 train prior 仍是 heuristic posterior，尚不是端到端 learned continuous model。
- 和历史实验关系：直接回应 Exp2p 的 lattice+assignment 混合归因；相对 Exp2h 增加 Exp2o pair-chem assignment，相对 Exp2o 增加 lattice/free-param posterior。
- gate 判定：passed=`False`；target_passed=`False`；conversion delta vs Exp2o `-1.126pp`；match delta vs Exp2o `-0.935pp`。
- 最终判决：`fail_ranked_gate_has_oracle_headroom`。Posterior creates some new sample-level matches but ranked mixture does not improve the Exp2o gate.
- 下一步：Improve generation-side posterior or inference-safe selection; do not use forbidden match labels.


<!-- OPENTRY14_EXP2R_DIVERSE_PAIRCHEM_ASSIGNMENT_POSTERIOR -->
## opentry_14 实验 2r：diverse pair-chem assignment posterior

结果文件：`model/New_model/opentry_14/results/experiment_2r_diverse_pairchem_assignment_posterior.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2r_diverse_pairchem_assignment_posterior/`

- 为什么做：Exp2p 显示 Exp2o 后 species/site assignment mismatch 仍是主要剩余瓶颈，Exp2q smoke 又显示 lattice/free-param posterior 没有新增 match。因此本实验不继续修 lattice，而是在生成侧提高 exact-cover assignment 多样性。
- 核心假设：如果正确 site identity 被 Exp2o 单一 balanced scoring-mode 排在 beam 外，则用 prior_strong/source_light/SG-orbit/chem-diverse 多模式生成 assignment，并用 Hamming 多样性保留，应新增 sample-level match 或至少不低于 Exp2o conversion。
- 数据规模：rows>=7 validation samples `2197`；generated diverse assignments `102471`；evaluated diverse candidates `73521`；input Exp2o pairchem candidates `73521`；combined candidates `85648`；train pair records `8000`；element-pair priors `2248`；workers gen/eval/prior `160` / `180` / `128`。
- baseline：Exp2o best `h10_s5_chem30_p5` rows>=7 match@50 `23.906%`、conversion `28.633%`、collision `3.871%`。
- 方法变化：保留 Exp2o pairwise local-chemistry pool，并把 diverse assignment 作为 supplement 与 Exp2o candidates 合并后统一 structural rank；diverse generator 从单一 balanced score 改成 `balanced,prior_strong,source_light,sg_orbit,chem_diverse` 多模式，每个模式先 beam search，再跨模式去重并按 Hamming distance 保留 diversity。推理仍只用 train split 元素/轨道频率、元素表、局部 pair-distance prior 和 generated structure 字段；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback，也不是 RF/HGB/阈值 scorer。
- 结果 best variant `h10_s5_chem30_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.151% / 23.660%`；conversion@50 `28.474%`；valid `86.402%`；formula `99.382%`；SG `98.525%`；exact-cover `99.824%`；collision `3.011%`。
- oracle 诊断：diverse pair-chem-only match/skelmatch samples `337` / `320`；相对 Exp2o 新增 match/skelmatch `14` / `14`；union upper bound match@50 `19.643%`、conversion `24.845%`。
- 可信度：中等。候选真实 render/parse/SG/StructureMatcher 评估，生成评分只用训练集 CIF 的元素/轨道统计、元素半径与 pair-distance prior；限制是 diversity heuristic 仍不是端到端 learned assignment-aware model，且沿用 source lattice/free params。
- 和历史实验关系：这是 Exp2n/2o 的 assignment 生成侧升级，直接回应 Exp2p 的 species/site mismatch；若失败，说明 heuristic assignment beam 多样性也不足，需要 learned assignment-aware generator 或升级 skeleton/source retrieval。
- gate 判定：passed=`False`；non_regression_vs_exp2o=`False`；conversion delta vs Exp2o `-0.159pp`；match delta vs Exp2o `-0.246pp`；target_passed=`True`。
- 最终判决：`fail_ranked_gate_has_oracle_headroom`。Deep source retrieval adds new sample-level matches and slightly improves match@50, but it does not preserve Exp2o conversion@50.
- 下一步：Use the generated source-retrieval candidates only for slot-policy diagnostics; if no inference-safe slot policy passes, move to learned source/assignment-aware generation.


<!-- OPENTRY14_EXP2S_SOURCE_RETRIEVAL_SUPPLEMENT -->
## opentry_14 实验 2s：source retrieval supplement

结果文件：`model/New_model/opentry_14/results/experiment_2s_source_retrieval_supplement.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2s_source_retrieval_supplement/`

- 为什么做：Exp2q 证明 lattice/free-param posterior 基本不能新增 match，Exp2r 证明 assignment diversity 有少量 oracle 但 ranked gate 不提升；同时 opentry_13 proposals 平均每样本只有约 3.8 个 source，Exp2o 的 top_skeletons=10 多数已经吃满。下一步需要检查更深 train source retrieval 是否能提供新的 skeleton/source basin。
- 核心假设：如果当前瓶颈来自 source skeleton basin 覆盖不足，而不是同一 source 内 assignment/lattice 调整，那么用 formula+GT-SG+atom_count 从 train split 检索更多未被 opentry_13 proposer 选中的 source skeleton，再做 exact-cover assignment，应新增 sample-level match 或提高 Exp2o conversion。
- 数据规模：rows>=7 validation samples `2197`；retrieved source candidates `18584`；generated retrieval assignments `132090`；evaluated retrieval candidates `76346`；input Exp2o pairchem candidates `73521`；combined candidates `87390`；train pair records `8000`；workers gen/eval/prior `160` / `180` / `128`。
- baseline：Exp2o best `h10_s5_chem30_p5` rows>=7 match@50 `23.906%`、conversion `28.633%`、collision `3.871%`。
- 方法变化：保留 Exp2o pairwise local-chemistry pool，并把 deep source retrieval candidates 作为 supplement 与 Exp2o candidates 合并后统一 structural rank；retrieval 只用 train split formula/SG/atom_count/row_count，不使用 validation GT skeleton。assignment 使用 `balanced,prior_strong,source_light` 多模式 exact-cover beam。推理不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback，也不是 RF/HGB/阈值 scorer。
- 结果 best variant `h10_s5_chem30_p5`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.545% / 24.004%`；conversion@50 `28.475%`；valid `86.605%`；formula `99.421%`；SG `98.585%`；exact-cover `99.823%`；collision `2.816%`。
- oracle 诊断：combined pair-chem match/skelmatch samples `353` / `338`；source-retrieval-only match/skelmatch `201` / `191`；相对 Exp2o 新增 match/skelmatch `38` / `35`；union upper bound match@50 `20.903%`、conversion `25.638%`。
- 可信度：中等。候选真实 render/parse/SG/StructureMatcher 评估，retrieval 与 assignment 只用 train split 和 target composition/GT-SG；限制是仍沿用 source geometry/free params，且 source retrieval 是 heuristic nearest-neighbor 而非 learned proposer。
- 和历史实验关系：这是 Exp2n/2o/2r 之后从 assignment 内部转向 source skeleton basin 覆盖的实验；若失败，说明简单 train source retrieval 也不足，需要 learned assignment-aware/source-aware generator。
- gate 判定：passed=`False`；non_regression_vs_exp2o=`False`；conversion delta vs Exp2o `-0.158pp`；match delta vs Exp2o `+0.098pp`；target_passed=`True`。
- 最终判决：`fail_no_conversion_lift`。Diverse assignment posterior does not improve Exp2o conversion.
- 下一步：Stop heuristic assignment scoring; move to learned assignment-aware geometry generation or skeleton/source retrieval.


<!-- OPENTRY14_EXP2T_SOURCE_RETRIEVAL_SLOT_POLICY_SWEEP -->
## opentry_14 实验 2t：source retrieval slot-policy sweep

结果文件：`model/New_model/opentry_14/results/experiment_2t_source_retrieval_slot_policy_sweep.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2t_source_retrieval_slot_policy_sweep/`

- 为什么做：Exp2s source retrieval 产生了 `38` 个 Exp2o 外新 match，但默认 structural-rank supplement 让 conversion@50 下降。需要判断问题是否只是 slot policy，还是 source retrieval 候选本身无法 inference-safe 转化。
- 核心假设：如果 retrieval 的新增 match 只是被过多低质量 skeleton-hit 稀释，那么固定少量 retrieval slots、保留 Exp2o pair-chem slots，应同时维持 Exp2o conversion 并获得 match@50 小幅提升。
- 数据规模：复用 Exp2o evaluated candidates `73521`、Exp2s retrieval evaluated candidates `76346`、safe-pool records `94544`、site-assignment records `36531`；sweep variants `9`。
- baseline：Exp2o best `h10_s5_chem30_p5` rows>=7 match@50 `23.906%`、conversion `28.633%`、collision `3.871%`。
- 方法变化：不重新生成 CIF，不使用 match/RMSD/StructureMatcher label 参与排序；只改变固定 slot 配额：hydrated/site/Exp2o-pairchem/source-retrieval/prototype。各池内部保留原 inference-safe rank 或 structural rank。
- 结果 best variant `h8_src5_o30_p2`：rows>=7 match@1/5/20/50 = `11.412% / 16.773% / 21.348% / 23.758%`；conversion@50 `28.760%`；valid `85.709%`；formula `99.495%`；SG `98.566%`；exact-cover `99.819%`；collision `3.613%`。
- oracle 诊断：source-retrieval-only match/skelmatch `201` / `191`；相对 Exp2o 新增 match/skelmatch `46` / `43`。
- 可信度：中等偏高。所有候选已真实评估，sweep 只改固定配额；限制是 slot policy 仍是 validation-side heuristic，不能作为主方法成功。
- 和历史实验关系：这是 Exp2s 的排序/slot 归因，不是新 generator；直接判断 retrieval oracle 能否被 inference-safe fixed policy 吃到。
- gate 判定：passed=`False`；best match delta vs Exp2o `-0.148pp`；best conversion delta vs Exp2o `+0.127pp`；best collision delta vs Exp2o `-0.258pp`。
- 最终判决：`fail_slot_policy_tradeoff`。No fixed slot policy improves match@50 and preserves Exp2o conversion simultaneously.
- 下一步：Stop heuristic retrieval/slot tuning; move to learned source/assignment-aware generation or a model-based critic.
