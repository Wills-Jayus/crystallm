### 物理化学过滤器模块综述

在当前仓库里，“物理化学过滤器”主要由这几部分组成：

- **核心指标函数**：`crystallm/_metrics.py`  
- **清洗 + 验证封装**：`crystallm/cif_cleaning.py`  
- **批量评估脚本**：`bin/evaluate_cifs.py`（原始项目评估用）  

它们一起完成对生成 CIF 的一系列“物理+晶体学”检查：晶胞是否合理、化学计量是否一致、位点多重度是否自洽、键长是否正常、空间群声明是否和真实对称性一致等。你在闭环里看到的 `bond_length_score`、`formula_ok`、`space_group_ok`、`atom_site_multiplicity_ok`、`validation_ok` 等列，都是由这个模块算出来的。

---

### 一、核心指标函数：`crystallm/_metrics.py`

#### 1. 键长合理性评分 `bond_length_reasonableness_score`

**路径**：`crystallm/_metrics.py`  
**签名**：`bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5)`

**作用**：衡量 CIF 中所有键长是否“在合理范围内”，返回 \([0,1]\) 之间的分数。

**实现要点：**

- 用 `pymatgen` 解析 CIF 得到 `Structure`，再用 `CrystalNN` 找每个原子附近的近邻，构造所有“键”。  
- 对每一条键：
  - 根据两端元素的电负性差 \(\Delta \chi\) 决定用 **原子半径** 还是 **离子半径** 估计期望键长：  
    - \(\Delta \chi \ge 1.7\)：偏离子键，用阳离子半径 + 阴离子半径；  
    - \(\Delta \chi < 1.7\)：偏共价键，用原子半径之和。
  - 计算比值 `bond_ratio = 实际键长 / 期望键长`。  
  - 非氢键：只要 `bond_ratio` 落在 \([1 - tolerance, 1 + tolerance]\) 内（默认 ±32%），记为“合理”；  
  - 氢键：用 `h_factor` 特殊处理（氢相关键更宽松），只要不是明显过短就算合理。  
- 对所有键统计“合理键数 / 全部键数”作为 `bond_length_reasonableness_score`。

**直觉解释**：  
- 1.0：几乎所有键长都在预期范围内；  
- 接近 0：大量键过短/过长，结构在几何上高度不合理。

---

#### 2. 空间群一致性 `is_space_group_consistent`

**签名**：`is_space_group_consistent(cif_str) -> bool`

**作用**：检查 CIF 中**声明的空间群**是否和结构解析出来的空间群一致。

**实现要点：**

- 用 `CifParser.from_string` 读出 `_symmetry_space_group_name_H-M`（如 `"P4/mmm"`）；  
- 用 `Structure.from_str` + `SpacegroupAnalyzer(structure, symprec=0.1)` 自动分析对称性，得到检测空间群符号；  
- 比较两者 `strip()` 后是否相等。

**失败典型原因**：

- 声明 `P4/mmm`，但原子分布实际上对应 `Pm-3m`；  
- 或者声明的空间群是非法符号/拼写（如之前的 `P4/<unk>`），导致解析、比较异常。

---

#### 3. 化学计量一致性 `is_formula_consistent`

**签名**：`is_formula_consistent(cif_str) -> bool`

**作用**：检查三处化学式是否**约化后一致**：

- `data_` 行的 formula（通过 `extract_data_formula` 抽取）；  
- `_chemical_formula_sum`；  
- `_chemical_formula_structural`。

**实现要点：**

- 对每一处用 `pymatgen.core.Composition` 解析为配比（Na₂Cl₂、NaCl 等）；  
- 比较它们的 `reduced_formula` 是否完全相同。  

**示例：**

- `data_Na2Cl2` / `_chemical_formula_sum 'Na2 Cl2'` / `_chemical_formula_structural NaCl`  
  → 三者都约化为 NaCl → `formula_ok = True`。  
- `data_Na2Cl2` + `_chemical_formula_sum 'Na2 Ru4'`（Na–Ru 系）  
  → 与 data 行不一致 → `formula_ok = False`。

---

#### 4. 位点多重度一致性 `is_atom_site_multiplicity_consistent`

**签名**：`is_atom_site_multiplicity_consistent(cif_str) -> bool`

**作用**：检查 `_atom_site_type_symbol` 与 `_atom_site_symmetry_multiplicity` 组合起来的**实际原子计数**，是否与 `_chemical_formula_sum` 推出的**期望原子数**一致。

**实现要点：**

- 从 `_chemical_formula_sum` 得到一个元素计数字典 `expected_atoms`（如 `{'Na': 2, 'Cl': 2}`）；  
- 遍历所有 CIF block 中的 `_atom_site_type_symbol` 与 `_atom_site_symmetry_multiplicity`，按照多重度统计 `actual_atoms`；  
- 比较字典是否完全相同。

**典型不一致场景**：

- 化学式写 `Na2Cl2`，但 atom_site 只给了一个 `Na` 位点多重度 1 + 一个 `Cl` 位点多重度 1 → 实际是 NaCl；  
- 或者某个元素多写/少写了一个位点。

---

#### 5. 晶胞“粗合理性” `is_sensible`

**签名**：`is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.)`

**作用**：用非常宽松的阈值，快速筛掉几何严重不合理的晶胞（边长、角度）。

**规则**：

- 所有 `_cell_length_[abc]` 必须在 \([0.5, 1000]\) Å 之间；  
- 所有 `_cell_angle_(alpha|beta|gamma)` 必须在 \([10°, 170°]\) 之间。  

如果有任一超出边界，就返回 False。

---

#### 6. 综合有效性 `is_valid`

**签名**：`is_valid(cif_str, bond_length_acceptability_cutoff=1.0) -> bool`

**作用**：作为“严格模式”的综合判定，仅在以下全部满足时返回 True：

- `is_formula_consistent(cif_str)` 为 True；  
- `is_atom_site_multiplicity_consistent(cif_str)` 为 True；  
- `bond_length_reasonableness_score(cif_str) >= cutoff`（通常为 1.0，即所有键长合理）；  
- `is_space_group_consistent(cif_str)` 为 True。

这个函数在更高层（如 `cif_cleaning.clean_and_validate_cif`）里被用来生成 `strict_valid` 字段，代表**最苛刻的通过条件**。

---

### 二、清洗与验证封装：`crystallm/cif_cleaning.py`

#### 1. `clean_cif(cif_str: str) -> str`

**作用**：对 CIF 做轻量级“清洗”，为下游几何/物性评估准备更干净的输入：

1. **空间群字段兜底修正**（针对当前 Na₂Cl₂ 实验）：  
   - 扫描行中 `_symmetry_space_group_name_H-M`；  
   - 如果右侧空间群字符串包含 `<`、`>` 或 `unk`，认为是坏 token，强制改成：  
     ```text
     _symmetry_space_group_name_H-M   P4/mmm
     ```  

2. **晶胞体积 sanity check**：  
   - 用 `extract_numeric_property` 读出 `_cell_length_a/b/c` 与 `_cell_angle_alpha/beta/gamma`；  
   - 用 `get_unit_cell_volume` 计算体积，如果参数明显非法会抛异常。

3. **空间群对称操作标准化**：  
   - 取 `extract_space_group_symbol(cif_str)`；若不是 `P 1`：  
     - 若符号本身不含 `<`、`>`、`unk` 等非法 token，则用 `replace_symmetry_operators` 按空间群重写对称操作块；  
     - 否则跳过（让后续 `space_group_ok` 报错，而不在这里崩溃）。

4. **去掉原子属性 block**：  
   - 看情况移除 `_atom_type_*` 一类 block，避免某些下游工具被多余信息干扰。

返回值是“清洗后”的 CIF 文本，结构上应与原始 CIF 相同，但更适合做几何/对称性分析。

---

#### 2. `validate_cif(cif_str, bond_length_acceptability_cutoff=1.0) -> CIFValidationResult`

**作用**：基于 `_metrics.py` 的几个函数，对单个 CIF 做一次完整验证，并返回结构化结果：

- `valid`：综合是否通过（formula + multiplicity + bond lengths + space group）；  
- `reasons`：失败原因列表（如 `"composition inconsistent"`, `"unreasonable bond lengths (~30% flagged)"`, `"space group inconsistent"`）；  
- `bond_length_score`：0–1 浮点分数；  
- `formula_ok` / `space_group_ok` / `atom_site_multiplicity_ok`：各子条件布尔值。

**内部逻辑**完全对应前一节的指标定义，只是集中“打包”成一个 dataclass。

---

#### 3. `clean_and_validate_cif(cif_str, bond_length_acceptability_cutoff=1.0) -> (Optional[str], Dict[str, Any])`

**作用**：这是闭环和生成实验实际调用的“物理化学过滤器”入口，它做三件事：

1. **尝试清洗**：  
   - 调 `clean_cif(cif_str)`；  
   - 若抛异常，在 `metadata` 中记录：  
     - `cleaning_error`；  
     - `reasons` 包含 `"cleaning_failed: ..."`；  
   - 然后在**原始 cif** 上继续做验证（尽量不因为清洗失败而丢失信息）。

2. **验证阶段**：  
   - 对 `target_cif = cleaned or cif_str` 调 `validate_cif(...)`；  
   - 如验证本身抛异常，再次在 metadata 中写入：  
     - `valid=False`、`validation_failed: ...`；  
     - 并将各个布尔标记设为 False 或 None。

3. **衍生指标 & 汇总**：  
   在验证成功的情况下，计算并写入更多字段：

   - `bond_lengths_reasonable`：`bond_length_score >= cutoff`；  
   - `strict_valid`：调用 `_metrics.is_valid` 得到“最严格模式”的 True/False；  
   - `formula_ok_relaxed`：通过 `_formula_sum_matches_atom_sites`，只要求 `_chemical_formula_sum` 和 atom_site 计数一致（放松 data 行、一部分字段的约束）；  
   - 最终 `metadata` 中包含：
     - `valid`（等于 `CIFValidationResult.valid`）  
     - `reasons`（清洗错误 + 逐项原因）  
     - `bond_length_score`、`bond_lengths_reasonable`  
     - `formula_ok`、`formula_ok_relaxed`  
     - `space_group_ok`、`atom_site_multiplicity_ok`  
     - `strict_valid`  

如果 `validation.valid` 为 True，函数返回 `(cleaned_cif, metadata)`；否则返回 `(None, metadata)`，表示“未通过过滤”。

在你的各个实验里，`cif_quality.json` / `cif_quality.csv` 就是直接/间接来自这个函数的结果。

---

### 三、批量评估脚本：`bin/evaluate_cifs.py`

**用途**：这是原始项目用于对大量生成 CIF 做离线评估的脚本，与当前闭环中的过滤逻辑高度一致：

- 读入 `.tar.gz` 中的所有 CIF 文本；  
- 用 `CIFTokenizer` + `is_sensible` / `is_atom_site_multiplicity_consistent` / `is_space_group_consistent` / `bond_length_reasonableness_score` / `is_valid` 等指标逐一评估；  
- 输出整体统计（比如 space group 一致率、键长平均分数、valid 比例等）到 stdout 和 CSV。

当前闭环没有直接调用这个脚本，而是复用其中的核心函数，通过 `cif_cleaning` 写出每轮 `cif_quality.*` 文件。

---

### 四、详细示例：几类典型结构在过滤器下的表现

#### 示例 1：理想 Na₂Cl₂ 结构（空间群 P4/mmm）

假设有一个 CIF：

```text
data_Na2Cl2
_symmetry_space_group_name_H-M   P4/mmm
_cell_length_a 3.21
_cell_length_b 3.21
_cell_length_c 11.06
...
_chemical_formula_structural NaCl
_chemical_formula_sum 'Na2 Cl2'
...
loop_
_atom_site_type_symbol
_atom_site_label
_atom_site_symmetry_multiplicity
...
Na Na0 2 0.0000 0.0000 0.1961 1
Cl Cl1 1 0.0000 0.0000 0.5000 1
Cl Cl2 1 0.5000 0.5000 0.0000 1
```

在过滤器中，大致会得到：

- `formula_ok = True`（data 行 / sum / structural 都约化为 NaCl）；  
- `atom_site_multiplicity_ok = True`（Na: 2，Cl: 2 与 `_chemical_formula_sum` 一致）；  
- `bond_length_score ≈ 1.0`（键长在期望范围内）；  
- `space_group_ok = True`（检测到的空间群确实是 P4/mmm）；  
- `valid = True`，`bond_lengths_reasonable = True`，`strict_valid = True`；  
- `cleaned_cif` 返回的是移除了 atom props block、规范化过对称操作的版本。

在 `cif_quality.json` 中，就会看到这一条被标记为 `validation_ok: true` 或等价的字段。

---

#### 示例 2：组成错误的 Na–Ru 结构

你日志里有类似的样本（来自早期实验）：

```text
data_Na2Cl2
_symmetry_space_group_name_H-M P4/mmm
...
_chemical_formula_structural NaRu2
_chemical_formula_sum 'Na2 Ru4'
...
loop_
_atom_site_type_symbol ...
Na 2 个位点
Ru 4 个位点
```

过滤器中的结果大致为：

- `formula_ok = False`：  
  - data 行是 Na2Cl2，而 `_chemical_formula_sum` / structural 是 Na–Ru 系，约化配方不一致；  
- `atom_site_multiplicity_ok = True`（Na 2, Ru 4 与 Na2Ru4 一致，但与 data 行不一致）；  
- `bond_length_score` 可能为 1.0（键长合理）；  
- `space_group_ok = True`（对称性一致）；  
- `valid = False`，`strict_valid = False`；  
- `reasons` 包含 `"composition inconsistent"`，其它项可能为空。

这类结构通常在你的闭环/基线结果中表现为“物性可算（ALIGNN OK），但在物理化学过滤器下不通过”。

---

#### 示例 3：空间群声明有问题 / 含 `<unk>` 的结构

最初的错误样本：

```text
data_Na2Cl2
_symmetry_space_group_name_H-M P4/<unk>
...
```

在**未加入 `clean_cif` 的兜底修正之前**：

- `replace_symmetry_operators` 查不到 `"P4/<unk>"` 的合法表项，抛异常；  
- `clean_and_validate_cif` 捕获为 `cleaning_failed: Bad international symbol P4/<unk>`；  
- 后续验证阶段可能也失败，`reasons` 中只看到清洗相关错误。

在现在的版本里，由于 `clean_cif` 对 `<unk>` 做了强制替换 + 非法空间群跳过对称操作替换：

- `_symmetry_space_group_name_H-M` 行被改成 `P4/mmm`；  
- 如果后续 `SpacegroupAnalyzer` 解析出的空间群确实是 P4/mmm，则 `space_group_ok=True`；  
- 否则，仍会在 `reasons` 中记录 `"space group inconsistent"`。  

这类兜底逻辑就是“物理化学过滤器”中**针对特定任务（Na₂Cl₂ + P4/mmm）的专门修补**，防止 `<unk>` 这种 LLM 级别的噪声直接把整个验证链条打挂。

---

### 五、总结

- 这个“物理化学过滤器”模块本质上是一个 **CIF 净化 + 严格多维度验证** 的组合：  
  - 净化步骤保证晶胞参数、对称操作、原子属性块不会干扰主评估；  
  - 验证步骤从几何（键长）、化学（化学计量、价态）和对称性（空间群、一致性）多角度筛选掉不合理结构。  
- 过滤器输出的布尔标记和分数，被汇总到 `cif_quality.*` 文件和 `evaluator_summary.json` 中，既能给 LLaMA 闭环提供反馈特征，也能让你在后验分析中明确“坏在哪一项”。  
- 结合你现有的实验结果来看：  
  - 很多结构在 ALIGNN 层面“物性可算”，但在本过滤器下失败，主要集中在 **化学计量 / 位点多重度 / 空间群一致性**；  
  - 这也正是后续你要通过 prompt 设计和 LLaMA 优化器逐步修正的方向。  

如果你愿意，我可以再帮你把这些内容整理成一份单独的 `docs/physical_filters.md` 或中文笔记，直接放进仓库里当作模块说明书。
