# SymCIF-v4 Gate 1 Root Cause Summary

本轮只做诊断：没有 jitter、没有 relabel、没有训练 scorer、没有跑 full match@5。

## 1. Gate 1 失败中 row_degeneracy 占多少？

- total samples: 6000
- failed samples: 446 (7.43%)
- row_degeneracy samples: 363 (6.05% of all, 81.39% of failed)
- primary class counts: {'PASS': 5554, 'C.sg_detection_only': 78, 'A.row_degeneracy': 363, 'D.unreadable_cif': 5}

## 2. formula_ok 失败主要来自哪里？

formula/atom_count 失败主要来自 row_degeneracy：OrbitEngine 在写 CIF 前展开的 unique atom count 已经低于 declared multiplicity。

进一步回到 source CIF 后，top failed SG/letter 的原始 atom_site 行通常仍声明 high multiplicity，并且有非特殊坐标；而 structured v3/v4 的 free_params 被写成 0、1/2、1/4 这类特殊值，导致 OrbitEngine 展开塌缩。因此主因不是 CIF read/write，也不是原始结构天然缺原子，而是 source atom_site coordinate 到 canonical representative/free_params 的提取或规范映射错误。

CIF read/write 单独导致的 formula loss 存在但不是主因；partial occupancy/disorder 在可推断 source CIF 中不是主导因素。

## 3. SG_ok 失败主要来自哪里？

SG_ok 失败包含两部分：

- atom_count/formula 已错的样本，spglib SG detection 没有稳定意义；
- formula/atom_count 正确但 SG 错的 detection-only 样本，多与 spglib 对低维/高对称/setting 的识别不稳定有关。

## 4. SG225 24e 全退化是什么原因？

- SG225 24e cases: 154
- expanded_count < declared_multiplicity: 154/154
- source CIF has same-element multiplicity-24 atom_site row: 154/154
- source multiplicity-24 row has non-special coordinate: 154/154
- structured/v4 free_params are special eighth-grid values: 154/154

154/154 全退化不是 tolerance 问题，也不像原始 CIF 真实处在 lower-multiplicity orbit。抽样和全量 SG225 24e 对照显示：source CIF 里 24e 行存在，坐标例如 `(0,0,0.226657)`、`(0,0,0.290858)`；但 structured/v4 free_params 变成 `{x:0,y:0,z:0}` 或等价特殊值，OrbitEngine 按代表式 `(x,0,0)` 展开后自然退化到 4a。

因此更可能是 free_params extraction / representative convention mapping bug：原始 CIF 中沿某个等价轴的自由参数没有被映射到 OrbitEngine 当前 canonical representative 的参数槽。

## 5. 是否应该 relabel degenerate rows 到 lower-multiplicity orbit？

不应该静默 relabel。nearest lower-multiplicity candidate 是“错误 free_params 输入 OrbitEngine 后”的塌缩结果，不代表原始结构真实属于 lower orbit。必须先从原始 CIF 用统一 setting/origin/convention 重新提取 representative/free_params。

## 6. 是否存在 partial occupancy / disorder？

- suspected partial/disorder samples from parsed source rows: 0

可用 source CIF 中没有证据表明 partial occupancy/disorder 是 Gate 1 主因。主要问题仍是 orbit label/free_params/setting consistency。

## 7. 当前是否可以在 clean subset 上推进 streaming WA search？

可以做工程 sanity check，但不能作为全量主结论。subset_A sample_count=5637，formula_ok=99.91%，SG_ok=98.53%。

## 8. 全量数据是否需要重新从原始 CIF 提取 Wyckoff labels/free_params？

需要。尤其 SG225 24e、SG189 3f/3g、SG193 6g、SG216 24f 等系统性失败源，必须用统一 OrbitEngine/CrystalFormer convention 重新提取 label 和 free_params，并记录 degeneracy/occupancy，而不是在现有 free_params 标签上继续训练。

top failed source-CIF 对照统计：

| SG/letter | failed rows | source same mult rows | source non-special coords | structured params special-grid |
| --- | ---: | ---: | ---: | ---: |
| 225/24e | 154 | 154 | 154 | 154 |
| 189/3f | 79 | 79 | 79 | 79 |
| 189/3g | 80 | 80 | 80 | 80 |
| 193/6g | 51 | 51 | 50 | 51 |
| 216/24f | 25 | 25 | 23 | 25 |

这说明 label 本身在 source/spglib 侧多数是可复现的；真正丢失的是 canonical free parameter。

## 9. Tolerance 判断

tolerance sensitivity: {'row_expansion_ok_min': 0.1726457399103139, 'row_expansion_ok_max': 0.1860986547085202, 'formula_ok_min': 0.16367713004484305, 'formula_ok_max': 0.17488789237668162, 'sg_ok_min': 0.12331838565022421, 'sg_ok_max': 0.29596412556053814}

失败不是主要由数值 tolerance 导致；row expansion 和 formula_ok 在 1e-6 到 1e-3 的 unique tolerance、1e-3 到 1e-1 的 spglib symprec 下没有足够大的恢复。
