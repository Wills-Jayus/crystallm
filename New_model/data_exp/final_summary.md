# MPTS-52 从 0 训练数据格式实验总结

## 任务

本次实验从 0 开始训练两个全量 GPT-style crystal language model，用来回答一个问题：我们把预训练数据从原始 CIF 改成 SymCIF-v4 这种结构化格式，是否真的有效？

输出目录：`/data/users/xsw/autodlmini/model/New_model/data_exp`

这里的“全量”指使用完整 MPTS-52 train split、完整 3500-step 训练流程，而不是 smoke test 或小子集训练。模型架构是本仓库此前多次使用的 CrystaLLM-small 同量级配置：8 layer / 8 head / n_embd=512 / 25.31M params。

## 对照设置

| 项目 | 原始 CIF baseline | 修改格式实验 |
| --- | --- | --- |
| 数据集 | MPTS-52 train/val | MPTS-52 train/val |
| 格式 | 原始 CIF 文本 | SymCIF-v4 结构化文本 |
| 训练方式 | 从 0 训练 | 从 0 训练 |
| tokenizer | UTF-8 byte-level, vocab=256 | 同左 |
| 模型 | 8 layer, 8 head, n_embd=512, 25.31M params | 同左 |
| 训练步数 | 3500 | 3500 |
| test split | 未读取 | 未读取 |

## 数据

- 原始 CIF：train 27,380 条，val 5,000 条，train tokens 42,000,708，val tokens 9,203,571。
- SymCIF-v4：train 25,998 条，val 4,727 条，train tokens 19,095,053，val tokens 3,794,718。
- SymCIF-v4 train 中 `rows>=7` 的样本有 6,863 条，unique formula 24,521，unique space group 197，unique Wyckoff assignment 25,739。

## 核心结果

| 模型 | best step | best train loss | best val loss | final train loss | final val loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 CIF baseline | 2500 | 0.1566 | 0.2014 | 0.1466 | 0.2054 |
| SymCIF-v4 修改格式 | 1250 | 0.1829 | 0.3616 | 0.1524 | 0.3931 |

结论很直接：在这组严格对照的从 0 训练实验里，当前 SymCIF-v4 修改格式没有超过原始 CIF baseline。看 validation loss，原始 CIF 明显更好；lower is better。

大白话解释：模型在 SymCIF-v4 上也能把训练集学下来，但验证集 loss 很早就不降了，后面甚至变差。这说明当前格式在这个 tokenizer + decoder-only GPT 设置下泛化不好，可能更容易让模型记格式/记训练样本，而不是学到对新结构更有用的规律。

## 快速生成检查

使用每种格式的前 8 个 validation prompt，每个 prompt 生成 2 个样本，共 16 个样本。

| 模型 | 生成数 | 基本 tag/字段完整 | pymatgen 可解析 |
| --- | ---: | ---: | ---: |
| 原始 CIF baseline | 16 | 16/16 | 10/16 |
| SymCIF-v4 修改格式 | 16 | 16/16 | 0/16 |

这里要注意：SymCIF-v4 不是直接 CIF，所以 pymatgen 解析率不能作为它的最终公平指标。它需要先有稳定的 SymCIF-v4 -> CIF renderer/converter，再做 structure-level benchmark。

## 最终判断

目前证据不支持“只修改预训练数据格式为 SymCIF-v4 就能提升 MPTS-52 从 0 训练效果”这个假设。

本实验能回答的是：在相同模型、相同训练步数、相同 byte-level tokenizer、相同 train/val-only 协议下，原始 CIF baseline 的 held-out token likelihood 明显优于当前 SymCIF-v4 格式。

如果继续推进 SymCIF 路线，下一步不应只是继续用同一个 byte-level GPT 硬训，而应该优先检查：

- SymCIF-v4 是否丢失了部分训练样本或关键信息。
- 是否需要专门 tokenizer，而不是 byte-level tokenizer。
- 是否需要稳定的 SymCIF-v4 -> CIF 渲染器。
- 是否需要用结构级指标做最终评估，而不是只看 token loss。
- 是否需要保存 best checkpoint，而不是只保留 final checkpoint。

## 关键路径

- 总日志：`/data/users/xsw/autodlmini/model/New_model/data_exp/experiment_log.md`
- 数据构建摘要：`/data/users/xsw/autodlmini/model/New_model/data_exp/data/data_build_summary.json`
- 原始 CIF 训练日志：`/data/users/xsw/autodlmini/model/New_model/data_exp/logs/scratch_orig_cif_small.train.log`
- SymCIF-v4 训练日志：`/data/users/xsw/autodlmini/model/New_model/data_exp/logs/scratch_symcif_v4_small.train.log`
- 原始 CIF loss 曲线：`/data/users/xsw/autodlmini/model/New_model/data_exp/eval/orig_cif_loss_curve.csv`
- SymCIF-v4 loss 曲线：`/data/users/xsw/autodlmini/model/New_model/data_exp/eval/symcif_v4_loss_curve.csv`
- 快速生成摘要：`/data/users/xsw/autodlmini/model/New_model/data_exp/eval/training_and_quick_sample_summary.json`
- 原始 CIF final checkpoint：`/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_orig_cif_small/ckpt.pt`
- SymCIF-v4 final checkpoint：`/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_symcif_v4_small/ckpt.pt`

说明：训练脚本设置了 `always_save_checkpoint=True`，每次 eval 都会覆盖同一个 `ckpt.pt`，所以当前只保留 final checkpoint；best step 的指标已经保存在日志和 loss curve CSV 里。
