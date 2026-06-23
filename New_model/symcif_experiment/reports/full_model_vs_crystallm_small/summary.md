# Full Model vs CrystaLLM-small 实验报告

## 1. 训练配置

| 项目 | 值 |
| --- | --- |
| route | `symcif_v2_constrained` / full train set |
| data path | `data/tokens_symcif_v2` |
| model | 12 layers, 12 heads, 768 embedding, 85.26M params |
| tokenizer | CrystaLLM `CIFTokenizer` |
| batch/block | batch_size=64, block_size=1024 |
| lr | 6e-4 cosine decay, min_lr=6e-5, warmup=100 |
| steps | 2000, eval every 250 |
| seed | 1337 |
| dtype/device | bfloat16, train on cuda:0 |
| sampling | n=20, temp_discrete=1.0/topk=10, temp_coord=0.7/topk=5, temp_cell=0.5/topk=5 |
| config | `configs/exp_symcif_v2_full_large.yaml` |
| best checkpoint | `runs/exp_symcif_v2_full_large/ckpt_best.pt` |
| log | `runs/logs/exp_symcif_v2_full_large_20260522_085542.log` |

## 2. 训练曲线

| step | train loss | val loss |
| ---: | ---: | ---: |
| 250 | 1.0502 | 1.0527 |
| 500 | 0.7484 | 0.7540 |
| 750 | 0.6405 | 0.6640 **best** |
| 1000 | 0.5613 | 0.6648 |
| 1250 | 0.4172 | 0.7685 |
| 1500 | 0.2437 | 1.0151 |
| 1750 | 0.1345 | 1.2812 |
| 2000 | 0.0921 | 1.4444 |

验证损失在 step 750 达到最优 0.6640，之后持续过拟合；评测使用 `ckpt_best.pt`。

## 3. 同 evaluator 对比

| model / route | match@1 | match@5 | match@20 | RMSE | readable | formula_ok | SG_ok | valid | artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline same-as-v4 | 16.2% | 33.8% | 44.6% | 0.1164 | 91.2% | 79.0% | 86.1% | 19.7% | eval_runs/baseline_reeval_same_as_v4_20260522 |
| baseline_minprompt same-as-v4 | 22.4% | 38.6% | 47.8% | 0.1188 | 91.5% | 90.9% | 90.7% | 23.0% | eval_runs/baseline_minprompt_reeval_same_as_v4_20260522 |
| v2_constrained small n20 | 24.2% | N/A | 49.2% | 0.1358 | 87.6% | 89.2% | 85.9% | 26.6% | eval_runs/symcif_v2_constrained_eval_t1_topk10_n20 |
| v2_fixed_cell best n5 | 26.2% | 41.8% | N/A | 0.1445 | 92.4% | 94.0% | 90.2% | 28.4% | eval_runs/symcif_v2_fixed_cell_match5_20260521 |
| v2_full3500 constrained n5 | 25.2% | 41.8% | N/A | 0.1481 | 91.0% | 92.6% | 88.6% | 28.8% | eval_runs/symcif_v3_vs_v2full_n5_20260521 |
| v2_full_large constrained n20 | 17.4% | 35.4% | 47.2% | 0.1468 | 87.8% | 86.0% | 81.2% | 23.8% | eval_runs/symcif_v2_full_large_constrained_n20_20260522 |
| v4 current full pipeline | 34.8% | 52.0% | 56.6% | 0.1030 | 69.9% | 69.9% | 69.9% | 0.0% | reports/symcif_v4_full_eval_current |

## 4. 结论

- 同等 evaluator 条件下，baseline match@20=44.6%，baseline_minprompt match@20=47.8%；v4 current match@20=56.6%，优势明确。
- 新的大模型没有超过既有 v2 小模型：match@20=47.2%，低于既有 `symcif_v2_constrained` n20 的 49.2%，也低于 v4 current 的 56.6%。
- 过拟合很明显：train loss 从 0.6405 降到 0.0921，但 val loss 从 0.6640 升到 1.4444。下一轮如果继续 v2 大模型，需要更强正则、早停、数据扩增或更保守训练步数。
- v4 current 已经是本轮最高的 end-to-end 结果，且 WA upper-bound 仍有 82.4% 的 top20 上限；相比继续单纯放大 v2，优先补 v4 geometry/free-params/lattice 更直接。

## 5. 输出

- generation/eval：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/`
- same-condition baseline：`eval_runs/baseline_reeval_same_as_v4_20260522/`
- same-condition baseline_minprompt：`eval_runs/baseline_minprompt_reeval_same_as_v4_20260522/`
- n=1/5/20 聚合：`eval_runs/symcif_v2_full_large_constrained_n20_20260522/summary_with_n5.json`
- 本报告 JSON：`reports/full_model_vs_crystallm_small/summary.json`
