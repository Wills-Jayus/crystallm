# 06 Figures and Tables

## Figure 1: SymCIF framework schematic

### Message

SymCIF 将晶体生成从 CIF file-record order 重组为 symmetry-conditioned Wyckoff assignment selection 和 geometry rendering。

### Suggested panels

- a. Original CIF language-modeling sequence: data block -> cell parameters -> formula -> symmetry operations -> atom_site loop -> autoregressive model -> generated CIF。
- b. SymCIF sequence: composition + oracle SG -> Wyckoff template -> WA selection/reranking -> free parameters/coordinates -> lattice -> CIF reconstruction -> validation。
- c. Diagnostic outputs: WA_hit、skeleton_hit、formula_ok、SG_ok、match、RMSE。

### Existing assets

- `paper_symcif/figures/symcif_pipeline_schematic.svg`
- `paper_symcif/figures/symcif_pipeline_schematic.png`

TODO: 检查图中文字是否过密；必要时重画成 2-panel。

## Figure 2: MP-20 benchmark comparison

### Message

SymCIF hybrid-prior 在 oracle-SG 设置下取得强 MP-20 top-1/top-5 recovery，但 RMSE 仍落后于 CrystaLLM-a reference。

### Suggested panels

- a. match@1 comparison: current order, hybrid-prior structured, hybrid-prior original-adjusted, CrystaLLM-a published reference。
- b. RMSE@1 comparison: current order, hybrid-prior, CrystaLLM-a reference。
- c. match@1/match@5 paired bar for SymCIF variants。
- d. Input condition annotation: composition-only vs composition + oracle SG。

### Required caution

CrystaLLM-a row/panel 必须标注 `composition-only published reference`。SymCIF 必须标注 `composition + oracle ground-truth SG`。

## Figure 3: WA reranking and coverage mechanism

### Message

Hybrid-prior 的增益主要来自提高 top-k Wyckoff assignment coverage。

### Suggested panels

- a. WA_hit@1/@5: current order vs hybrid-prior。
- b. match@1/@5: current order vs hybrid-prior。
- c. coverage-to-match gap: WA_hit@5 vs match@5。
- d. selector ablation: policy rank only、train prior only、hybrid-prior。

TODO: 补 selector ablation 后再定最终 panel d。

## Figure 4: failure audit / geometry bottleneck

### Message

SymCIF 的剩余失败主要来自 geometry/render/validity，而不是单纯缺少 WA candidates。

### Suggested panels

- a. WA_hit@5 vs match@5 gap。
- b. failed_with_WA_hit vs failed_without_WA_hit。
- c. geometry_source success/failure breakdown。
- d. complex subset performance: n_sites>=6、n_sites>=12、num_elements>=4、rare_sg、extraction_hard。

TODO: 补具体数字和最终 failure audit 表。

## Table 1: Main benchmark table with input condition

### Required columns

- Method
- Input condition
- Test scope
- Number of samples
- match@1 ↑
- match@5 or match@20 ↑
- RMSE@1 ↓
- RMSE@5 or RMSE@20 ↓
- Notes

### Draft rows

| Method | Input condition | Test scope | Samples | match@1 | match@5/20 | RMSE@1 | RMSE@5/20 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| SymCIF current order | composition + oracle SG | structured test | 8,893 | 51.29% | 66.11% @5 | 0.0668 | 0.0742 @5 | baseline ordering |
| SymCIF hybrid-prior | composition + oracle SG | structured test | 8,893 | 60.90% | 73.68% @5 | 0.0573 | 0.0615 @5 | main result |
| SymCIF hybrid-prior | composition + oracle SG | original-test adjusted | 9,046 | 59.87% | 72.43% @5 | 0.0573 | 0.0615 @5 | 153 extraction failures counted as failures |
| CrystaLLM-a published | composition only | published MP-20 reference | TODO | 55.85% | 75.14% @20 | 0.0437 | 0.0395 @20 | not same input |

## Table 2: Selector ablation table

### Required columns

- Selector
- Prior source
- Inference-feasible?
- WA_hit@1 ↑
- WA_hit@5 ↑
- match@1 ↑
- match@5 ↑
- RMSE@1 ↓
- RMSE@5 ↓
- Notes

### Draft rows

| Selector | Prior source | Inference-feasible | WA_hit@1 | WA_hit@5 | match@1 | match@5 | RMSE@1 | RMSE@5 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current order | none/current rank | yes | 49.80% | 71.27% | 51.29% | 66.11% | 0.0668 | 0.0742 | baseline |
| policy rank only | TODO | yes | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| train WA prior only | train split | yes | TODO | TODO | TODO | TODO | TODO | TODO | distribution-dependent |
| train skeleton prior only | train split | yes | TODO | TODO | TODO | TODO | TODO | TODO | distribution-dependent |
| hybrid-prior | train split + candidate rank | yes | 59.02% | 81.66% | 60.90% | 73.68% | 0.0573 | 0.0615 | main selector |
| oracle GT-WA diagnostic | test label | no | TODO | TODO | TODO | TODO | TODO | TODO | diagnostic only |

## Supplementary Table S1: dataset and extraction audit

### Required rows

- MP-20 original test samples = 9,046。
- structured test samples = 8,893。
- structured extraction failures = 153。
- full evaluation coverage = 8,893/8,893。
- top-5 candidate records = 44,465。
- eval_timeout@1/@5 = 0。
- render_success@1 = 100.00%。
- render_success@5 = 99.10%。

## Supplementary Table S2: complex subset performance

Suggested subsets:

- overall
- n_sites>=6
- n_sites>=12
- num_elements>=4
- rare_sg
- high_multiplicity_orbit
- extraction_hard

TODO: 从 final reports 中填入 match@1、match@5、failed_with_WA_hit、failed_without_WA_hit、RMSE。

