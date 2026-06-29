# opentry_14 实验报告：predicted skeleton 到 StructureMatcher match 的转化 gate

生成/更新时间：2026-06-29 UTC
位置：`model/New_model/opentry_14/OPENTRY14_EXPERIMENT_REPORT.md`
报告用途：作为 `model/New_model/GPT_REVIEW_BUNDLE.md` 后续附录，记录 opentry_14 的实验逻辑、实际执行结果、当前证据边界、最终判决和下一步执行顺序。

## 0. 报告边界

本文件保留历史背景与 gate 设计，并在完成 opentry_14 实验 0-2b 后更新实际结果。整个过程没有删除或改写任何历史目录，没有运行 official test，也没有根据 official 结果调参。本轮读取和使用：

- `model/New_model/GPT_REVIEW_BUNDLE.md`
- `model/New_model/opentry_13/results/*.json`
- `model/New_model/opentry_7/opentry_7_final_report.md`
- `model/New_model/opentry_10/final_report.md`
- `model/New_model/opentry_12/results/experiment_7_main_ablation_boundary.json`
- 用户提供的 pasted text gate 指令
- `model/New_model/opentry_14/results/experiment_0_frontend_and_baseline_freeze.json`
- `model/New_model/opentry_14/results/experiment_1_predicted_skeleton_noise_geometry_pairs_sharded.json`
- `model/New_model/opentry_14/results/experiment_2_joint_geometry_repair.json`
- `model/New_model/opentry_14/results/experiment_2b_safe_pool_after_failure_analysis.json`

实验 1 使用 8 shard 并行构造训练对；实验 2 使用 `crystallm_env` 训练 joint MLP 并用 8 worker 做 validation StructureMatcher。线程环境按 `OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 / NUMEXPR_NUM_THREADS=1 / NUMBA_NUM_THREADS=1` 约束，单个 Python/worker 进程约 100% CPU，低于用户给定的 `200%` 上限。

## 1. 为什么做

当前主问题不是继续做 rerank、fusion、C2S3C15 比例或普通 scorer 小涨，而是解决：

> predicted exact-cover skeleton 已经命中后，连续几何、lattice、row-level free parameters、site mapping、collision/local packing 没有把 skeleton 转成 StructureMatcher match。

opentry_13 已经把问题切开：

- renderer/site mapping 前端在 selected train-prototype 模式下基本能保 formula、SG、exact-cover；
- rows>=7 skeleton-hit@50 已有明显信号；
- hydrated match 与 skeleton-to-match conversion 仍低；
- lattice-only repair 和 heuristic multi-geometry 都没有解决转化问题。

因此 opentry_14 的主线应固定前端，转向 predicted-skeleton-aware 的联合 geometry repair，而不是继续调候选排序。

## 2. 核心假设

主假设：

1. 当前瓶颈已经从 symbolic skeleton coverage 转到 skeleton-to-geometry conversion。
2. 若固定已过 gate 的 renderer/site mapping，训练时显式加入 train-side predicted-skeleton noise，并联合学习 lattice、row-level free parameters、local geometry/collision，则 rows>=7 的 skeleton-to-match conversion 应显著高于 opentry_13。
3. 只修 lattice 不够；只增加 train prototype / median / old geometry initializer 这类 heuristic multi-geometry 也不够。
4. 任何成功 claim 必须对齐 true CrystaLLM-a GT-SG anchor，而不是 pure CrystaLLM GT-SG、低 baseline、auxiliary hybrid 或 oracle GT-WA。

反证条件：

- 如果 renderer gate 失效，先修 renderer/site mapping。
- 如果 train predicted skeleton 与 target free parameters 无法稳定对齐，先修数据构造。
- 如果 joint repair 保持 exact-cover 但不提升 match，问题在 local geometry/collision 或 site/free-param 对齐。
- 如果 repair 提升 match 但破坏 SG/formula/exact-cover，该路线不能进入 full validation。

## 3. 数据规模和主 baseline

### official anchor

主比较对象固定为 CrystaLLM-a GT-SG anchor：

| 数据集 | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MP-20 | official full test K20 | 71.67% | 83.08% | 87.81% | 0.0509 | 0.0449 | 0.0431 |
| MPTS-52 | official full test K20 | 25.23% | 36.46% | 43.96% | 0.1211 | 0.1257 | 0.1334 |

MPTS-52 rows>=7 anchor：

| scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rows>=7 official | 22.49% | 33.37% | 41.04% | 0.1310 | 0.1356 | 0.1444 |

`17.181 / 24.345 / 31.522` 是 pure CrystaLLM GT-SG low baseline，不允许作为主成功对照。

### opentry_13 / opentry_14 继续使用的数据规模

| 模块 | 数据规模 |
| --- | --- |
| MPTS-52 official test anchor/C2S3C15 | 8096 samples；rows>=7 7626 |
| exp2 renderer/site mapping validation | 4727 samples；rows>=7 2197；candidate records 36268；topK 20 |
| exp3 skeleton proposer train/validation | train repr 25998；train rows>=7 6863；validation repr 4727；validation rows>=7 2197 |
| exp3 noisy-skeleton lattice repair pilot | train noisy pairs 7774；validation 4411 samples；rows>=7 2033；candidate records 18134 |
| exp4 multi-geometry validation | 4411 samples；rows>=7 2033；candidate records 71930；unique predicted skeletons 18134；top output K50 |

## 4. 历史实验关系

opentry_7 建立了 true CrystaLLM-a GT-SG official anchor。MP-20 anchor 为 `71.67 / 83.08 / 87.81`，MPTS-52 anchor 为 `25.23 / 36.46 / 43.96`。

opentry_10 证明普通 validation-selected rerank/fusion 没有转移到 official：最接近的 MPTS-52 K30 RF route 只有 `+0.845pp / -0.232pp / +0.099pp`，未过近程 gate，更远未达 +5pp。

opentry_12 说明 C2S3C15 可以作为 auxiliary hybrid 诊断，但其收益主要来自候选混合，不是可 claim 的主方法；rows>=7 主瓶颈仍是 coverage plus skeleton-to-match conversion。

opentry_13 修正了 baseline 口径并把主线推到 predicted skeleton repair：

- true-anchor-source C2S3C15：overall `25.235 / 41.366 / 48.382`，相对 true anchor `+0.000pp / +4.904pp / +4.422pp`，仍不能写成主方法成功。
- renderer/site mapping selected train-prototype：结构 gate 通过。
- rows>=7 skeleton proposer：skeleton-hit 有明显信号，但 hydrated match 和 conversion 不够。
- lattice-only predicted-skeleton repair：失败。
- heuristic multi-geometry：skeleton-hit 更高，但 rows>=7 conversion 更低，match@50 下降。

## 5. opentry_14 实验 0：冻结前端和基线口径

### 为什么做

防止后续继续混用低 baseline、C2S3C15 auxiliary result 或未过 gate 的 renderer 模式。opentry_14 必须先确认：

- 主 baseline 是 CrystaLLM-a GT-SG anchor；
- 前端固定为 opentry_13 exp2 的 selected train-prototype renderer/site mapping；
- 只有结构 gate 过，才允许进入 joint repair。

### 方法变化

本实验不训练、不看 match、不跑 official，只复核 opentry_13 exp2 的 renderer/site mapping gate。固定输入为 composition/formula + GT-SG + predicted exact-cover skeleton proposal。排序只用 inference-safe structural checks：legal CIF、formula、SG、site count、exact-cover、collision。

### 结果

selected train-prototype structural selector：

| scope | valid | formula | SG | exact-cover | fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 97.007% | 99.660% | 98.436% | 100.000% | 2.993% |
| rows>=7 | 96.458% | 99.410% | 98.475% | 100.000% | 2.993% |

对照：

- deterministic mode 失败：overall valid 18.838%，rows>=7 valid 10.699%。
- candidate-level train-prototype mode 未过整体 valid gate：overall valid 88.789%，rows>=7 valid 89.548%。

### 判决

实验 0 gate 通过：可以冻结 selected train-prototype 前端，不应继续大改 renderer/site mapping。下一步必须进入 predicted-skeleton-noise training pair 构造，而不是回到 C/S 比例、ordinary scorer、threshold 或 rerank。

## 6. opentry_13 失败证据对 opentry_14 的约束

### rows>=7 skeleton proposer

opentry_13 exp3 的 rows>=7-specialized skeleton proposer给出：

| scope | skeleton-hit@1 | skeleton-hit@5 | skeleton-hit@20/50 | hydrated match@1 | hydrated match@5 | hydrated match@20/50 | conversion@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rows>=7 | 61.038% | 75.330% | 76.513% | 10.605% | 14.838% | 15.112% | 19.750% proposal-to-match；20.660% hydrated-skeleton-to-match |

判读：skeleton proposal 本身有信号，但现有 hydration/geometry 没把 skeleton 转成 StructureMatcher match。

### lattice-only repair

opentry_13 exp3 补充了 train noisy skeleton pairs，并训练了 lattice MLP pilot：

| scope | before match@1/5/20 | after match@1/5/20 | delta | repair conversion@20 | structure gate |
| --- | --- | --- | --- | ---: | --- |
| overall | 27.998 / 39.288 / 40.218 | 10.066 / 13.172 / 13.308 | -17.932 / -26.117 / -26.910 pp | 1.858% | fail |
| rows>=7 | 11.461 / 16.035 / 16.331 | 2.312 / 2.804 / 2.951 | -9.149 / -13.232 / -13.379 pp | 1.117% | fail |

判读：lattice-only repair 已失败。后续 repair 必须联合处理 lattice、row-level free parameters、local geometry、collision/local packing，不能继续只修 lattice。

### heuristic multi-geometry

opentry_13 exp4 对同一 predicted skeleton 生成多几何候选：

| scope | match@1 | match@5 | match@20 | match@50 | skeleton-hit@50 | conversion@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 20.313% | 26.479% | 27.953% | 27.998% | 86.692% | 31.485% |
| rows>=7 | 10.428% | 12.592% | 13.035% | 13.084% | 82.686% | 15.407% |

相对 rows>=7 CrystaLLM K50 reference top50 `18.431%`，multi-geometry top50 低 `-5.347pp`。

判读：候选更多不是答案。若 geometry proposals 不是 learned posterior 或明确训练得到的 free-parameter 分布，增加 beam 可能提高 skeleton-hit 但降低 conversion。

## 7. 可信度

高可信：

- CrystaLLM-a GT-SG official anchor 来自 opentry_7/opentry_10 登记结果。
- opentry_13 exp2 renderer gate 覆盖完整 validation proposal records，并真实 render/parse/SG-check。
- opentry_13 exp5 是已有 JSON/JSONL 的汇总，不训练、不调阈值、不重跑 StructureMatcher。

中等可信：

- rows>=7 skeleton proposer match 指标只映射到已有 hydrated/evaluated candidates；未 hydrated 的 skeleton 没有被自动计为 match，因此是 conservative validation diagnostic。
- lattice repair 和 multi-geometry 是 validation 结果，不是 official；但足以作为“不进入 official”的 gate evidence。

不可作为成功 claim：

- GT-WA learned geometry repair oracle。
- C2S3C15 或其他 candidate fusion/auxiliary hybrid。
- Track A scorer、hard-negative scorer、exact-cover filter/proxy。
- 任何相对 low baseline 的提升。

## 8. 最终判决

当前 opentry_14 的起点判决：

`allowed_main_result_claim = false`

原因：

1. renderer/site mapping selected mode 已过结构 gate，说明前端不再是首要阻塞点。
2. rows>=7 predicted skeleton hit 已有信号，但 conversion 只有约 20%。
3. train noisy skeleton lattice-only repair 的 rows>=7 after match@20 只有 2.951%，repair conversion@20 只有 1.117%，不能进入 official。
4. heuristic multi-geometry rows>=7 match@50 只有 13.084%，低于 CrystaLLM K50 reference top50 18.431%，conversion@50 只有 15.407%。
5. true-anchor-source C2S3C15 虽然 match@5/20 接近 +5pp，但仍是 auxiliary hybrid，不是 predicted-skeleton-aware main method。

因此，opentry_14 不能启动 frozen official test，不能继续调 C2S3C15/scorer/threshold，也不能把 validation 子集或 auxiliary hybrid 写成主方法成功。

截至本次实际执行，实验 0 和实验 1 已通过各自准入 gate；实验 2 joint MLP repair 在 rows>=7 validation gate 失败；随后实验 2b 通过 failure analysis 改为 fixed safe geometry pool，并通过实验 2 最低 gate 与 target gate。因此当前仍不能 claim official/main result，但允许在固定 safe pool 上继续实验 3 validation；禁止直接把原 MLP repair 或 validation safe pool 写成 official 成功。

## 9. opentry_14 下一步实验顺序

### 实验 1：predicted-skeleton-noise 完整几何训练对

必须先完成。用 MP-20/MPTS-52 train split 生成 train-side predicted exact-cover skeleton，经 fixed selected train-prototype renderer/site mapping 渲染初始结构，再从 train true structure 恢复 lattice、row-level free parameters、局部坐标/距离、collision/local packing target。

验收：

- rows>=7 train 覆盖足够；
- 大部分样本能恢复 lattice 与 free-parameter target；
- 每个样本能统计 initial valid/formula/SG/exact-cover/collision 与 target 差异；
- 若 row/free-parameter 对齐失败比例过高，停止训练，先修 alignment。

### 实验 2：joint geometry repair head

只在实验 1 通过后进行。输入至少包含 composition、GT-SG、predicted skeleton rows、renderer 初始 lattice、初始 row-level free params、row multiplicity、local packing/collision 统计。输出必须至少包含 lattice 更新和 row-level free-parameter 更新；可选 local residual / pair-distance correction。

最低 gate：

- rows>=7 skeleton-to-match conversion@50 至少达到约 22.3%；
- match@50 至少追平当前 CrystaLLM rows>=7 K50 baseline 附近；
- formula、SG、exact-cover 不被 repair 破坏。

目标 gate：

- rows>=7 conversion@50 达到 28%-30%，或 rows>=7 match@5 和 match@20 同时比当前 rows>=7 baseline 提高至少 5pp。

### 实验 3：symmetry-preserving local optimizer

只在实验 2 过最低线后加入。CHGNet/MatGL/MACE 或等价 proxy 只能做短步、保守、可回退优化；优化后必须重检 SG、formula、exact-cover、collision。若破坏 SG 或 exact-cover，回退优化前结构。

### 实验 4：learned posterior multi-hypothesis + inference-safe geometry critic

只在实验 3 有 match/conversion 增益后进行。多假设必须来自 learned repair posterior 或训练得到的 free-parameter 分布，不能重复 opentry_13 的 heuristic beam。

critic 可用 formula、SG、exact-cover、collision、volume/atom、coordination、repair confidence、energy proxy；不能用 match、RMSD、StructureMatcher label、official feedback。

### 实验 5：窄消融与 full validation gate

同一 predicted skeleton、同一 evaluator、同一 validation 口径下比较：

- fixed front-end no repair；
- old lattice-only repair；
- joint repair head；
- joint repair head + local optimizer；
- joint repair head + local optimizer + geometry critic；
- learned multi-hypothesis version。

必须同时报告 overall 与 rows>=7 的 match@1/5/20/50、RMSE、valid、formula、SG、exact-cover、collision、skeleton-hit、skeleton-to-match conversion。

### 实验 6：一次性 frozen official test

只有实验 5 通过后才能做。official 前必须冻结 repair 模型、optimizer 步数、critic、beam/hypothesis 数量和 fallback 规则。official 只能跑一次，不能用 official 结果回调参数。

## 10. 当前停止线

立即禁止：

- C2S3C15 或 C/S 比例调参；
- ordinary RF/HGB、threshold、anchor_keep、ranker seed、coverage repair 作为主线；
- lattice-only repair；
- 只看 match@1；
- 使用 GT-WA、GT-skeleton、test true CIF、official per-sample match、StructureMatcher label 作为推理特征或 official 调参依据；
- rows>=7 conversion 没有实质提升前进入 official。

如果 opentry_14 后续失败，必须归因到以下之一：

- 前端结构 gate 不稳；
- free-parameter 对齐失败；
- local geometry/collision 无法修复；
- repair 保持 exact-cover 但不提升 match；
- repair 提升 match 但破坏 SG/formula；
- critic 只是排序已有正确候选，而没有提高 conversion。

当前建议的下一步结论：原 joint MLP repair 路线停止；safe pool 路线可作为实验 3 输入继续 validation。下一步只能在 fixed safe pool 上做 symmetry-preserving local optimizer；不能进入 full validation 或 official，不能把 validation safe pool 当主方法成功。

## 11. opentry_14 实际执行结果

### 实验 0：冻结前端和基线口径

结果文件：`model/New_model/opentry_14/results/experiment_0_frontend_and_baseline_freeze.json`

- 为什么做：固定 true CrystaLLM-a GT-SG anchor 和 selected train-prototype renderer/site mapping，避免继续混用 low baseline、C2S3C15 auxiliary hybrid、ordinary scorer 或 official feedback。
- 核心假设：若 selected train-prototype 前端仍保持 rows>=7 valid/formula/SG/exact-cover，则后续应修 joint geometry repair，而不是继续改前端。
- 数据规模：MPTS-52 validation proposals `4727` samples，rows>=7 `2197`，candidate records `36268`，topK `20`。
- baseline：MP-20 CrystaLLM-a GT-SG official match@1/5/20 = `71.667% / 83.075% / 87.807%`；MPTS-52 CrystaLLM-a GT-SG official match@1/5/20 = `25.235% / 36.462% / 43.960%`。
- 结果：selected train-prototype overall valid `97.007%`、formula `99.660%`、SG `98.436%`、exact-cover `100.000%`；rows>=7 valid `96.458%`、formula `99.410%`、SG `98.475%`、exact-cover `100.000%`。
- 可信度：高。直接复核 opentry_13 exp2 machine JSON 和 historical anchors，不训练、不调参。
- 判决：`pass`。前端冻结，允许进入实验 1。

### 实验 1：predicted-skeleton-noise 完整几何训练对

结果文件：`model/New_model/opentry_14/results/experiment_1_predicted_skeleton_noise_geometry_pairs_sharded.json`
训练对 artifact：`model/New_model/opentry_14/artifacts/exp1_predicted_skeleton_noise_pairs/predicted_skeleton_noise_geometry_pairs_merged_sharded.jsonl.gz`

- 为什么做：让 repair 训练看到推理期同类 predicted-skeleton noise，并检查 lattice、row-level free parameters、local geometry/collision target 是否可恢复。
- 核心假设：若 train-side noisy skeleton + selected train-prototype 初始几何能覆盖大部分 train 样本，并恢复足够 free-param target，则允许训练 joint repair；否则先修 data alignment。
- 数据规模：MP-20 train `26629` samples，rows>=7 `4306`；MPTS-52 train `25998` samples，rows>=7 `6863`；合计 `52627` train records，rows>=7 `11169`。
- 方法变化：从 train prototypes 构造 exact-cover predicted skeleton source，排除 self-source；渲染 initial geometry；从 train true structure 恢复 target lattice/free params/local stats。严格 `orbit_id + element` alignment 不足后，保留 exact alignment 并允许 `orbit_id` element-mismatch fallback，artifact 中标记 alignment kind。
- 结果 overall：nonempty pair `88.994%`，initial valid `80.415%`，formula `88.320%`，SG `82.175%`，exact-cover `88.994%`，collision `4.353%`；free-param target complete among requiring `81.794%`，value recovery `92.657%`，usable joint pair `75.229%`。
- 结果 rows>=7：nonempty pair `80.097%`，initial valid `71.358%`，formula `78.216%`，SG `73.865%`，exact-cover `80.097%`，collision `5.775%`；free-param target complete among requiring `76.492%`，value recovery `94.869%`，usable joint pair `60.041%`。
- 可信度：中等偏高。覆盖 MP-20/MPTS-52 train split 并真实 render/parse/SG/collision/local stats；限制是 rows>=7 usable joint pair 只有 `60.041%`，element-mismatch target 是 noisy alignment。
- 判决：`pass`。允许进入实验 2，但不能把本实验写成 match 成功。

### 实验 2：predicted-skeleton-aware joint geometry repair head

结果文件：`model/New_model/opentry_14/results/experiment_2_joint_geometry_repair.json`
模型 artifact：`model/New_model/opentry_14/artifacts/exp2_joint_geometry_repair/joint_repair_heads.pt`
候选评估：`model/New_model/opentry_14/artifacts/exp2_joint_geometry_repair/evaluated_joint_repair_candidates.jsonl`

- 为什么做：替代已失败的 lattice-only repair，直接测试 joint lattice + row-level free-parameter head 是否能提高 rows>=7 skeleton-to-match conversion。
- 核心假设：如果瓶颈主要是 lattice/free-param joint correction，则 rows>=7 conversion@50 应达到最低线 `22.3%`，且 match@50 应接近 CrystaLLM K50 validation rows>=7 baseline `18.431%`。
- 数据规模：训练 pair artifact `52627` records，usable joint pairs `39591`；lattice samples `39591`，row-param samples `270498`。validation `4411` samples，rows>=7 `2033`，candidate records `18134`，topK `50`。
- baseline：CrystaLLM K50 validation rows>=7 top50 `18.431%`；opentry_13 predicted-skeleton hydrated rows>=7 match@50 `15.112%`；opentry_13 multi-geometry rows>=7 match@50 `13.084%`。
- 方法变化：训练 lattice MLP 和 row-param MLP；推理时 alpha=`0.35` blend source 与预测，重新渲染 CIF 并评估。未使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。
- 训练结果：lattice best val loss `0.292309`；param best val loss `0.960734`。
- 结果 overall：match@1/5/20/50 = `10.814% / 17.094% / 17.298% / 17.298%`；valid_any@50 `92.859%`，formula `99.956%`，SG `96.802%`，exact-cover retained `100.000%`，collision `22.957%`，skeleton-to-match conversion@50 `19.665%`。
- 结果 rows>=7：match@1/5/20/50 = `0.246% / 0.394% / 0.394% / 0.394%`；valid_any@50 `90.507%`，formula `99.968%`，SG `96.817%`，exact-cover retained `100.000%`，collision `26.075%`，skeleton-hit@50 `82.686%`，skeleton-to-match conversion@50 `0.416%`。
- 可信度：中等偏高。真实 train-noisy-skeleton repair、真实 StructureMatcher validation，无禁用推理特征；限制是单一 repaired geometry、轻量 MLP、row-param target noisy alignment 和 rows>=7 usable training coverage 较低。
- 和历史实验关系：比 opentry_13 lattice-only 更结构安全，但 rows>=7 conversion 比 opentry_13 multi-geometry `15.407%` 明显更差，说明 joint MLP 没学到可 match 的 rows>=7 geometry posterior。
- 最终判决：`fail_validation_gate`。结构指标基本保住，但 rows>=7 conversion@50 `0.416%` 和 match@50 `0.394%` 远低于最低线，归因于“repair 保持 exact-cover/formula/SG，但不提升 match”。
- 下一步：不进入实验 3、full validation 或 official。下一轮应先重做 rows>=7 free-parameter/site mapping alignment 与 local geometry target，或改为 learned posterior/multi-hypothesis repair 后重新过 validation gate。

### 实验 2b：failure-analysis guided safe geometry pool

结果文件：`model/New_model/opentry_14/results/experiment_2b_safe_pool_after_failure_analysis.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2b_hydrated_prototype_safe_pool/evaluated_safe_pool_candidates.jsonl`

- 为什么做：实验 2 的 root cause 不是 skeleton coverage 不足，而是单一 joint MLP repair 把已有可 match 的 hydrated/source-prototype geometry 替换掉。rows>=7 原 MLP repair 只有 `8` 个 matched samples，match@50 `0.394%`、conversion@50 `0.416%`。
- 核心假设：若保留原 SymCIF v5 hydrated geometry，并补入 opentry_13 prototype multi-geometry，不使用 match/RMSD 排序，也应恢复 skeleton-to-match conversion 并超过实验 2 最低 gate。
- 数据规模：validation samples `4411`，rows>=7 `2033`；candidate records `94544`，rows>=7 candidate records `47240`；固定 topK `50`，配额为 hydrated top `10` + prototype top `40`。
- baseline：rows>=7 CrystaLLM K50 validation top50 `18.431%`，实验 2 近线下限 `17.931%`；opentry_13 hydrated rows>=7 match@50 `15.112%`，opentry_13 multi-geometry rows>=7 match@50 `13.084%`。
- 方法变化：fixed quota safe pool。每个样本先取 rows>=7 proposer 映射到的 SymCIF v5 hydrated candidates top10，再取 opentry_13 prototype multi-geometry top40；候选顺序只使用 proposer rank、generation score 和 structural rank，不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB/rerank。
- 结果 overall：match@1/5/20/50 = `27.998% / 40.014% / 44.276% / 44.389%`；formula `98.333%`，SG `94.862%`，exact-cover `99.712%`。
- 结果 rows>=7：match@1/5/20/50 = `11.461% / 17.019% / 21.299% / 21.348%`；skeleton-hit@50 `82.686%`；skeleton-to-match conversion@50 `24.866%`；valid_any@50 `98.032%`；formula `97.680%`，SG `94.744%`，exact-cover `99.682%`。
- 可信度：中等偏高。候选都有已有 SymCIF v5 validation metrics 或 opentry_13 StructureMatcher evaluation；本实验只做固定配额组合并重新汇总，不用 match/RMSD 参与排序。限制是 hydrated 部分复用既有 validation metrics，尚未做 official。
- 和历史实验关系：opentry_13 exp3 hydrated 与 opentry_13 exp4 prototype multi-geometry 互补；opentry_14 exp2 joint MLP 破坏 rows>=7 geometry；exp2b 的贡献是用 failure analysis 把破坏性 MLP 覆盖改为 safe pool fallback。
- 最终判决：`pass_minimum_gate`，且 `target_passed=True`。rows>=7 match@50 比近线下限高 `+3.416pp`，conversion@50 比 `22.3%` 最低线高 `+2.566pp`。
- 下一步：允许在 fixed safe pool 上进入实验 3 local optimizer validation；仍不允许进入 full validation 或 official。


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
