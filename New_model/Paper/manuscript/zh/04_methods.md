# 04 Methods

## 方法总览

SymCIF 的方法部分应围绕一个 pipeline figure 展开，而不是按代码文件罗列。建议结构：

1. Problem setting and input condition
2. SymCIF representation
3. CIF-to-SymCIF conversion
4. Wyckoff assignment candidate selection and hybrid-prior reranking
5. Geometry rendering and CIF reconstruction
6. Evaluation metrics and protocol

## 4.1 Problem setting and input condition

本文研究 symmetry-conditioned crystal structure generation。给定 chemical composition 和 oracle ground-truth space group，方法需要生成候选晶体结构，并通过 CIF reconstruction 与 StructureMatcher evaluation 评估是否恢复真实结构。

TODO: 明确输入 composition 使用 reduced formula 还是 conventional cell formula。  
TODO: 明确 oracle SG 是编号、符号还是二者都使用。  
RISK: Methods 必须清楚写出 oracle SG，否则 Results 比较会显得不透明。

## 4.2 SymCIF representation

SymCIF 将晶体表示为结构化记录，而不是直接使用 CIF token order。每条记录包含：

- `formula` 和 `formula_counts`：常规晶胞中的元素计数。
- `sg` 和 `sg_symbol`：空间群编号和符号。
- `lattice`：`a`、`b`、`c`、`alpha`、`beta`、`gamma` 和 `volume`。
- `wa_table`：每行对应一个元素占据的 Wyckoff orbit。
- `canonical_skeleton_key`：只包含 orbit ids，用于 skeleton comparison。
- `canonical_wa_key`：包含 orbit ids 和 elements，用于完整 WA comparison。

`wa_table` 每行包含 `element`、`sg`、`letter`、`multiplicity`、`site_symmetry`、`enumeration`、`orbit_id`、`representative_expr`、`free_symbols` 和 `free_params`。诊断字段包括 `source_coord`、`mapped_coord`、`extraction_method`、`extraction_residual`、`expansion_count_after_reextract`、`expansion_ok`、`extraction_success` 和 `fallback_reason`。

## 4.3 CIF-to-SymCIF conversion

转换流程：

1. 使用 CIF parser 读取 benchmark split 中的 CIF。
2. 使用空间群分析器将结构转换为 conventional standard cell。
3. 从 conventional cell 提取整数 `formula_counts`。
4. 识别空间群编号、空间群符号和 symmetrized equivalent sites。
5. 对每个等价位点组识别 Wyckoff symbol、multiplicity、site symmetry 和 element。
6. 从 Wyckoff lookup table 取出对应 `OrbitToken`。
7. 将 source fractional coordinates 反解为 orbit 的 `free_params`。
8. 重新展开 orbit，检查 expansion count、source inclusion、residual 和 formula closure。
9. 写出 structured SymCIF record 和 audit fields。

TODO: 报告 `symprec`、angle tolerance、free-parameter tolerance。  
TODO: 报告 Wyckoff lookup table 来源、版本和许可。  
TODO: 明确 partial occupancy、disorder、non-integer composition 和 mixed-element equivalent groups 的处理策略。  
RISK: 如果 fallback free-parameter extraction 用于训练/评估，需要报告比例和影响。

## 4.4 Hybrid-prior Wyckoff assignment reranking

Hybrid-prior selector 对候选 Wyckoff assignments 进行 train-only、inference-feasible reranking。该 selector 使用训练集中可见的 WA/skeleton 统计规律和已有候选排序信息，提高正确 assignment 进入 top-1/top-5 的概率。

需要写清楚：

- 训练统计只来自 train split。
- 不使用 test label、StructureMatcher outcome 或 oracle GT-WA。
- 推理时只使用输入 composition、oracle SG、候选 WA 和训练先验。
- 该 prior 依赖训练分布，因此需要在 Discussion 和 TODO experiments 中保留泛化边界。

TODO: 补精确 scoring 公式。  
TODO: 补 smoothing、tie-breaking、unseen assignment fallback。  
TODO: 补 selector ablation。

## 4.5 Geometry rendering and CIF reconstruction

给定一个候选 SymCIF record，renderer 根据 `orbit_id` 找回 orbit token，并用 `free_params` 计算代表坐标和 symmetry-expanded fractional coordinates。随后，renderer 将展开后的 atom rows 与 `formula_counts`、`sg`、`sg_symbol` 和 `lattice` 写回 CIF。

重建 CIF 后需要重新解析并检查：

- readable
- formula_ok
- atom_count_ok
- SG_ok
- valid / strict_valid
- render_success

TODO: 明确 lattice 是来自模型生成、候选继承、检索还是其他 geometry source。  
TODO: 明确 geometry variants 的生成方式和排序方式。

## 4.6 Evaluation protocol

指标：

- `match@k`: top-k 候选中任一候选通过 StructureMatcher 即为成功。
- `RMSE@k`: 在 matched samples 上统计结构匹配误差。
- `WA_hit@k`: top-k 中是否包含 ground-truth/equivalent canonical WA。
- `skeleton_hit@k`: top-k 中是否包含 ground-truth/equivalent canonical skeleton。
- `render_success@k`: top-k 候选渲染是否成功。
- `eval_timeout`: StructureMatcher 或 evaluation 是否超时。

MP-20 口径：

- original test samples = 9,046。
- structured test samples = 8,893。
- structured extraction failures = 153。
- structured-only rates 以 8,893 为分母。
- original-test adjusted rates 以 9,046 为分母，并将 153 个 failures 计为失败。

TODO: 补 StructureMatcher 参数。  
TODO: 补 RMSE 定义。  
TODO: 补硬件、运行命令、随机种子或 deterministic 说明。  

