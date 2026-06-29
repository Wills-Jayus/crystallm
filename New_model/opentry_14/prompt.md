你现在要继续推进 CrystaLLM / SymCIF 晶体生成实验。当前目标不是做 rerank、fusion、C2S3C15 比例调参，也不是继续追求普通 scorer 的小涨，而是专门解决 predicted skeleton 已经命中后无法转成 StructureMatcher match 的问题，也就是 skeleton-to-match conversion 低的问题。

请先读完当前仓库中 New_model 下最新的 opentry_13 结果、相关历史 opentry 报告和已有审计材料，只允许读取历史目录，不要改写历史实验。新实验请放在新的 opentry 目录中。你需要沿用当前已经确认的主比较口径：MP-20 和 MPTS-52 的主 baseline 必须是 CrystaLLM-a GT-SG anchor，而不是 pure CrystaLLM GT-SG，也不是任何低 baseline。任何相对低 baseline 的提升都不能作为主方法成功。

当前已经确认的事实如下：

1. true GT-SG anchor 才是主比较对象。MPTS-52 true anchor 是约 25.23 / 36.46 / 43.96，MP-20 true anchor 是约 71.67 / 83.08 / 87.81。不要再混用低 baseline。
2. opentry_13 exp2 说明 predicted skeleton 到 CIF 的 renderer/site mapping 在 selected_train_prototype 模式下已经基本过结构 gate，尤其 rows>=7 的 valid、formula、SG、exact-cover 已经比较稳定。因此下一轮不要大改前端 renderer，不要把前端和 repair 混在一起调。
3. opentry_13 exp3 说明 rows>=7 skeleton-hit@50 已经有明显信号，但 hydrated match@50 仍低，skeleton-to-match conversion 只有约 20%。
4. opentry_13 exp4 说明多几何候选本身不是答案：skeleton-hit 更高，但 conversion 更低，match@50 反而下降。
5. lattice-only repair 已经失败，不能继续做只修 lattice 的变体。后续 repair 必须同时处理 lattice、row-level free parameters、local geometry、collision/local packing。

本轮实验的核心目标是：固定已过 gate 的 predicted skeleton renderer/site mapping 前端，训练或构造一个 predicted-skeleton-aware 的联合 geometry repair 系统，把 rows>=7 的 skeleton-to-match conversion 明显拉起来。不要再做普通 RF/HGB、threshold、anchor_keep、C/S ratio、candidate fusion 或 official feedback 调参。

你需要完成以下实验，必须按顺序做，不允许跳过前面的 gate。

实验 0：冻结前端和基线口径
目标是防止后续继续混口径。固定 CrystaLLM-a GT-SG 为主 anchor，固定 opentry_13 exp2 selected_train_prototype renderer/site mapping 作为前端。先确认 rows>=7 的前端结构 gate 是否仍接近 exp2 水平：valid 至少约 96%，formula 至少约 99%，SG 至少约 98%，exact-cover 必须保持 100%。这个实验不追求 match 提升，只作为是否允许进入 repair 的准入门槛。如果这个 gate 不过，先修 renderer/site mapping；如果这个 gate 过，不要再继续大改前端。

实验 1：构造 predicted-skeleton-noise 的完整几何训练对
目标是让 repair 模型在训练时看到和推理时相同类型的 skeleton 噪声。请在 MP-20/MPTS-52 train split 上，用当前 rows>=7-specialized skeleton proposer 或最接近的 predicted exact-cover skeleton proposer 生成 train-side predicted skeleton。然后用固定的 selected_train_prototype renderer/site mapping 渲染成初始结构。再从 train true structure 中恢复对应的 lattice、row-level free parameters、局部坐标/距离统计、collision/local packing 指标，形成“noisy predicted skeleton + initial geometry -> target geometry”的训练对。
验收标准不是产出某个文件，而是训练对本身必须可用：rows>=7 train 样本要有足够覆盖；大部分样本必须能恢复可训练的 lattice 与 free-parameter target；必须能统计每个样本的初始 valid/formula/SG/exact-cover/collision 与 target 差异。如果 row/free-parameter 无法对齐的比例太高，先修数据构造，不要直接训练。

实验 2：训练 predicted-skeleton-aware 联合 geometry repair head
目标是替代失败的 lattice-only repair。模型输入至少包括 composition、GT-SG、predicted skeleton rows、renderer 初始 lattice、renderer 初始 row-level free parameters、row multiplicity、局部距离/packing 统计、可选 train prototype 统计。模型输出不能只包含 lattice，必须至少包含 lattice 更新和 row-level free-parameter 更新；如果可控，再加入 local coordinate residual 或 pair-distance correction。
训练目标要同时约束 lattice、free parameters、局部距离、collision、volume/atom 合理性，并强制保持 formula、SG、exact-cover。不要在推理特征中使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton 或 official test 结果。
验收标准分两级。最低通过线：在 rows>=7 validation gate 上，skeleton-to-match conversion@50 至少达到约 22.3%，并且 match@50 至少追平当前 CrystaLLM rows>=7 K50 baseline 附近；结构安全指标不能明显下降，尤其 formula、SG、exact-cover 不能被 repair 破坏。进入下一阶段的目标线：rows>=7 conversion@50 达到 28% 到 30% 区间，或者 rows>=7 match@5 和 match@20 同时比当前 rows>=7 baseline 提高至少 5pp。若最低通过线都达不到，不要进入 full validation，也不要做 official。

实验 3：加入 symmetry-preserving local optimizer
目标是在 learned repair 后处理局部 collision、短距离、packing 和 local environment 问题，但不能破坏对称性。可以使用 CHGNet、MatGL、MACE 或当前环境中可用的等价局部几何/势能 proxy，做短步、保守、可回退的局部优化。优化后必须用 spglib、PyXtal 或当前内部 checker 重新检查 SG、formula、exact-cover、collision。如果优化破坏 SG 或 exact-cover，必须回退到优化前结构。
验收标准：相比实验 2 最佳版本，rows>=7 conversion@50 至少再提高 2pp，collision/local packing 指标明显改善，并且 SG/exact-cover 不恶化。如果只降低 collision 但 match 不提升，则把它记为诊断结果，不进入最终方法。

实验 4：多假设 free-parameter generator + inference-safe geometry critic
目标是替代之前失败的 heuristic multi-geometry。对同一个 predicted skeleton 生成多组几何假设，但这些假设必须来自 learned repair posterior 或明确训练得到的 free-parameter 分布，不能只是简单复制 train prototype、median 或无脑 beam。然后用一个 inference-safe geometry critic 排序。critic 可以使用 formula、SG、exact-cover、collision、volume/atom、local coordination、repair confidence、能量 proxy 等，但不能使用 match、RMSD、StructureMatcher label 或 official feedback。
验收标准：相比实验 3 最佳版本，rows>=7 match@20 或 match@50 至少再提高 2pp，且 conversion 不能下降。如果只是候选更多但 conversion 下降，立即停止该方向，不要扩大 beam。

实验 5：窄消融与 full validation gate
目标是清楚回答每个模块到底有没有用。必须在同一批 predicted skeleton、同一 evaluator、同一 validation 口径下比较以下版本：固定前端无 repair；旧 lattice-only repair；联合 repair head；联合 repair head + local optimizer；联合 repair head + local optimizer + geometry critic；多假设版本。
每个版本都必须报告 overall 与 rows>=7 的 match@1/5/20/50、RMSE、valid、formula、SG、exact-cover、collision、skeleton-hit、skeleton-to-match conversion。
验收标准：只有当 full validation 上至少两个 overall match 指标相对 GT-SG validation baseline 达到 +5pp，并且 rows>=7 match@5 和 match@20 也至少 +5pp，同时 K1 不出现明显下降，才允许进入 official frozen test。如果只在 rows>=7 子集提升但 overall 不达标，只能作为 complex-structure 子集结果，不能作为主方法成功。

实验 6：一次性 frozen official test
只有实验 5 通过后才能做 official。official 前必须冻结所有配置，包括 repair 模型、local optimizer 步数、critic、beam/hypothesis 数量和 fallback 规则。official 只能跑一次，不能根据 official 结果回调参数。
official 成功标准：MP-20 或 MPTS-52 official full-test 上，至少两个 match@1/5/20 指标超过对应 CrystaLLM-a GT-SG anchor +5pp，并且 RMSE 不出现不可解释的大幅恶化。若 official 未达标，不允许继续用 official 结果调参；只能回到 validation 阶段重新提出新假设。

本轮严禁事项：

1. 禁止继续调 C2S3C15 或任何 C/S 比例；它只能作为 auxiliary hybrid 诊断，不是主方法。
2. 禁止普通 RF/HGB、threshold、anchor_keep、ranker seed、coverage repair 作为主线。
3. 禁止只做 lattice repair。
4. 禁止只看 match@1，必须同时看 match@5、match@20 和 rows>=7。
5. 禁止使用 GT-WA、GT-skeleton、test true CIF、official per-sample match、StructureMatcher label 作为推理特征或 official 调参依据。
6. 禁止把 validation 小涨、子集小涨、candidate fusion 小涨写成主方法成功。
7. 禁止在 rows>=7 conversion 没有实质提升前进入 official。

如果实验失败，请不要继续无意义扩大训练或调阈值。失败时必须明确归因到以下类别之一：前端结构 gate 不稳；free-parameter 对齐失败；local geometry/collision 无法修复；repair 保持 exact-cover 但不提升 match；repair 提升 match 但破坏 SG/formula；scorer/critic 只是在排序已有正确候选而不是提高 conversion。最终结论必须说明下一步是继续 repair、重做 data alignment、升级 skeleton proposer，还是停止该路线。
