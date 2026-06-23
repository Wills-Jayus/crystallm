# opentry_6 Final Summary

* 四个阶段是否全部完成：yes.
* 每个阶段核心结果：
  * stage1: match@20=not_run, rows>=7 match@20=not_run, rows>=7 new positives=301
  * stage2: match@20=not_run, rows>=7 match@20=not_run, rows>=7 new positives=283
  * stage3_refined: match@20=not_run, rows>=7 match@20=not_run, rows>=7 new positives=288
  * stage4: match@20=not_run, rows>=7 match@20=not_run, rows>=7 new positives=289
* 哪个阶段最有效：以 rows>=7 match@20 和 new positives 为准，见上方核心结果及 `opentry_6_experiment_log.md` 的 fold reports。
* rows>=7 是否真正提升：按固定顺序、无排序评估结果判断；若 rows>=7 new positives 为 0，则未证明真实提升。
* 是否仍然卡在 continuous geometry：若 GT W/A 条件下 rows>=7 仍低或为 0，则主要仍卡在 continuous geometry。
* 与 opentry_4 / opentry_5 相比是否进步：使用 opentry_5 E8028/E8034/E8036 rows>=7 positives 作为旧基线，new positives 记录在日志。
* 下一轮最应该继续的唯一方向：继续 CrystalFormer-style continuous geometry/refiner，但应扩大真实 canonical rows>=7 覆盖或重建 full train canonical geometry 数据，而不是回到 selector/ranker。
* 明确不要再走的弯路：selector、reranker、ranker、compatibility score、energy rejection、anchor-safe insertion、oracle selection、根据 match/RMSE/validity/logprob 筛选候选。
* GT-W/A 下 geometry 是否可以学会：见 stage1/stage2 GT W/A rows>=7 match@20；若 GT 仍失败，则不能证明。
* predicted/OOF W/A 是否是主要瓶颈：见 stage4 A/B/C 差异；若 A 成功 B/C 失败则 W/A gap 是主瓶颈，否则 geometry 仍是主瓶颈。
* fixed-step refiner 是否有用：见 stage3 raw vs refined；若 refined rows>=7 match/collision 未改善或简单结构变差，则 refiner 暂无用。
* 是否仍然陷入局部：若 stage2 unique rows>=7 仍只有 opentry_5 clean split 的有限数量，则仍受数据覆盖局部限制。
