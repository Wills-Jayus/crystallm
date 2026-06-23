# Methods: SymCIF representation

## Representation of symmetry-constrained crystal structures

SymCIF 将一个晶体结构表示为由化学组成、空间群、Wyckoff assignment、连续自由参数和晶格参数组成的结构化记录，而不是直接使用 CIF 文件中的 token 顺序作为生成对象。给定一个输入 CIF，转换脚本首先将其解析为 conventional standard structure，并从该 conventional cell 中提取整型化学组成、空间群编号、空间群符号、Wyckoff 等价位点组和晶格参数。这样得到的 SymCIF 记录保留 CIF 可以重构所需的信息，但将它们组织为更接近晶体学约束传播顺序的字段。

每个 SymCIF 记录包含样本标识、组成、空间群、Wyckoff 表、晶格和若干诊断字段。组成由 `formula` 和 `formula_counts` 表示，其中 `formula_counts` 记录 conventional cell 中每种元素的整数原子数，并作为后续 Wyckoff exact-cover 的目标。空间群由 `sg` 和 `sg_symbol` 表示。晶格由 `lattice` 字段给出，包括 `a`、`b`、`c`、`alpha`、`beta`、`gamma` 和 `volume`。结构的对称性核心由 `wa_table` 表示；其中每一行对应一个 Wyckoff orbit 上的一个元素占据，而不是对应 CIF 中的单个 atom-site token。

`wa_table` 的每一行包含元素、Wyckoff orbit 标识和该 orbit 的几何自由度。具体而言，`element` 表示该等价位点组的元素类型；`sg`、`letter` 和 `multiplicity` 表示空间群、Wyckoff letter 和 multiplicity；`site_symmetry` 和 `enumeration` 用于区分同一空间群内的 site-symmetry 类型和枚举版本；`orbit_id` 是由 setting、space group、multiplicity-letter、enumeration 和 site symmetry 组成的 canonical orbit identifier。`representative_expr` 记录该 orbit 的代表坐标表达式，例如 `(1/3, 2/3, z)`；`free_symbols` 标记表达式中的自由变量；`free_params` 给出从原始坐标反解得到的自由变量数值。对于坐标反解过程，记录中还保存 `source_coord`、`mapped_coord`、`extraction_method`、`extraction_residual`、`expansion_count_after_reextract` 和 `expansion_ok` 等字段，用于审计该行是否能重新展开为正确数量的等价坐标。

为支持候选比较和 reranking，SymCIF 进一步定义了两个 canonical key。`canonical_skeleton_key` 只由排序后的 `orbit_id` 组成，表示结构选择了哪些 Wyckoff orbits，但不包含元素分配；`canonical_wa_key` 在每个 `orbit_id` 后附加对应元素，表示完整的 element-Wyckoff assignment。排序规则优先考虑 multiplicity、Wyckoff letter、enumeration、site symmetry、element 和 orbit id，从而避免 CIF 文件中 atom-site 记录顺序或等价位点枚举顺序造成 key 不稳定。这个设计使得 skeleton selection、element assignment 和 geometry rendering 可以分别诊断。

## Conversion workflow

SymCIF 转换从官方 benchmark split 中的 CIF 文本开始，并保持 train、validation 和 test split 不混合。对于每个样本，脚本首先使用 CIF parser 读取结构；随后用空间群分析器在给定 `symprec` 和 angle tolerance 下检测空间群，并将结构转换到 conventional standard cell。转换后，脚本重新计算 conventional cell 的元素计数，并要求每个元素计数为整数。该计数不是 reduced formula，而是后续 Wyckoff multiplicity 求和必须精确匹配的 cell composition。

在得到 conventional structure 后，脚本从 symmetrized structure 中读取等价位点组及其 Wyckoff symbol。每个等价位点组必须满足三个条件：该组非空；组内所有 site 具有同一元素且 occupancy 为 1；Wyckoff symbol 中声明的 multiplicity 与该组实际 site 数一致。通过这些检查后，脚本根据空间群和 Wyckoff letter 从 Wyckoff lookup table 中取出对应的 `OrbitToken`。`OrbitToken` 封装了该 orbit 的 multiplicity、site symmetry、enumeration、代表坐标表达式、自由变量、固定坐标、对称操作和 canonical orbit id。

接下来，脚本将每个等价位点组的一个或多个源坐标映射回 orbit 的自由参数。对于给定的 source fractional coordinate 和 orbit，free-parameter extractor 会遍历该 orbit 的 symmetry operations，在模 1 的分数坐标空间内求解 `x`、`y`、`z` 等自由变量。常见情形下，脚本使用 direct coordinate solve；对于包含变量线性组合的表达式，则使用 least-squares solve，并在求解后重新展开整个 orbit。只有当重新展开后的坐标集合包含源坐标、展开数量等于 Wyckoff multiplicity、并且残差低于容差时，该反解结果才被接受。如果多个解满足条件，脚本按照 expansion correctness、source inclusion、residual 和参数规范化顺序选择最优解。

若某一 orbit 无法通过上述反解得到自由参数，脚本会使用 fallback 策略，从源坐标中按自由变量名称抽取近似参数，并记录 `extraction_success = false` 和 `fallback_reason`。无论使用直接反解还是 fallback，脚本都会重新展开该 orbit，并检查展开数量是否与声明的 multiplicity 一致。随后，所有 `wa_table` 行按元素累加 multiplicity；只有当该累加结果与 `formula_counts` 完全一致时，该样本才被写入 structured SymCIF 数据集。转换输出还保存 conventional CIF 路径、free-parameter re-extraction 审计行、canonical keys、`n_sites`、`num_elements` 和 `atom_count` 等统计字段。

从 SymCIF 记录重构 CIF 时，renderer 不再依赖原始 CIF 中 atom-site loop 的顺序。它逐行读取 `wa_table`，根据 `orbit_id` 找回 orbit token，并用 `free_params` 计算代表坐标和所有 symmetry-expanded fractional coordinates。随后，renderer 将展开后的 atom rows 与 `formula_counts`、`sg`、`sg_symbol` 和 `lattice` 一起写回标准 CIF 字段。重构后可以通过 parser 和 space-group analyzer 重新验证 `formula_ok`、`atom_count_ok` 和 `sg_ok`。因此，SymCIF representation 同时提供了生成所需的结构字段和验证所需的可追踪审计字段。

## Crystallographic causal order

SymCIF 的字段顺序符合晶体学因果顺序，因为它先确定离散约束，再生成连续几何。首先，`formula_counts` 给定每种元素需要被 Wyckoff multiplicities 覆盖的原子数；`sg` 决定哪些 Wyckoff orbits、site symmetries 和 symmetry operations 是合法的。其次，`canonical_skeleton_key` 和 `canonical_wa_key` 分别描述 Wyckoff orbit skeleton 与元素分配，它们在坐标生成之前决定结构的对称骨架和占据方式。第三，`representative_expr`、`free_symbols` 和 `free_params` 在已选 orbit 的条件下确定连续坐标自由度；同一个自由参数只有在给定空间群和 Wyckoff orbit 后才有明确意义。最后，`lattice` 和 symmetry-expanded coordinates 被序列化回 CIF，并由 validation 检查与目标 composition 和 space group 是否一致。

这种顺序与原始 CIF 文件的记录顺序不同。CIF 文件通常先给出 cell parameters、chemical formula、symmetry operations 和 atom_site loop；这些字段在文件中是线性排列的，但晶体学上彼此强耦合。SymCIF 将这些耦合关系显式化：composition 和 space group 限定 Wyckoff 搜索空间，Wyckoff assignment 限定坐标自由度，free parameters 与 lattice 决定最终几何，CIF reconstruction 只是最后的序列化步骤。因此，模型或搜索器可以分别学习 WA selection、element assignment、geometry rendering 和 validation，而不是在一个长 CIF token 序列中隐式恢复所有关系。

## TODOs before final manuscript submission

- TODO: 明确最终 Methods 中使用的 Wyckoff lookup table 来源、版本和许可；当前代码使用 `wyckoff_lookup_full.json`，`setting_id` 标记为 `crystalformer`。
- TODO: 确认最终实验中固定使用的 `symprec`、angle tolerance 和 free-parameter tolerance，并在 Methods 或 Supplementary 中报告。
- TODO: 说明 test split 中 prepared original CIF 与 CSV 内 CIF 的优先级规则，以及该规则是否用于所有数据集。
- TODO: 补充 structured extraction 的成功率、失败类型和各 split 样本数，尤其是 MP-20 原始 test 中 153 条 extraction failure 的处理。
- TODO: 明确 partial occupancy、disorder、non-integer composition 和 mixed-element equivalent group 的处理策略；当前转换脚本会拒绝这些样本。
- TODO: 澄清同一 space group 和 Wyckoff letter 存在多个 enumeration 时的查表逻辑；代码记录 `enumeration`，但部分路径以 `(sg, letter)` 查询 orbit。
- TODO: 明确 fallback free-parameter extraction 是否进入训练数据、评估数据或仅作为诊断字段，并报告 fallback 比例。
- TODO: 决定论文中是否把 `lattice` 归为 representation 的一部分，还是归为 geometry rendering target；当前 structured record 中保存 lattice，并在 CIF reconstruction 时使用。
- TODO: 如果要声称 causal order 带来性能提升，需要在 Results 中用 ablation 或 coverage/match 诊断支持；Methods 中只应描述设计动机和实现。

## Code evidence used for this draft

- `scripts/build_symcif_v4_benchmark_structured.py`
- `src/symcif_v4/orbit_token.py`
- `src/symcif_v4/orbit_engine.py`
- `src/symcif_v4/free_param_extractor.py`
- `src/symcif_v4/canonicalize.py`
- `src/symcif_v4/render.py`
- `src/symcif_v4/validation.py`
