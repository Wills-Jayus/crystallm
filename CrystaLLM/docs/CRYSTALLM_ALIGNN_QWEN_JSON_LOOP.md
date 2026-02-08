# CrystaLLM × ALIGNN × Qwen 闭环说明

## 1. 组件与职责

- **生成器（CrystaLLM）**：根据 `prompt.txt` 生成 CIF 候选  
  - 使用：`CrystaLLM/bin/sample.py`
- **过滤器（清洗 + 验证）**：对 CIF 做轻量清洗与质量检查，输出结构化失败原因  
  - 使用：`CrystaLLM/crystallm/cif_cleaning.py`
  - 指标实现：`CrystaLLM/crystallm/_metrics.py`
- **打分器（ALIGNN via ZMQ）**：预测 `formation_energy`、`bandgap` 等物性  
  - 服务端：`CrystaLLM/resources/alignn_zmq_server_multi.py`
  - 客户端：`CrystaLLM/bin/alignn_client.py`
- **优化器（Qwen）**：读取上一轮 `prompt` 与 `evaluator_summary`，输出下一轮 `next_prompt_lines`  
  - 客户端：`CrystaLLM/bin/qwen_client.py`

## 2. 每轮闭环数据流

每轮（round_XX）基本发生：

1. 写入 `round_XX/prompt.txt`
2. 生成 `round_XX/cifs/sample_*.cif`
3. 产出 `round_XX/cif_quality.json|csv` 与摘要 `cif_quality_summary.txt`
4. 通过 ALIGNN 得到 `round_XX/scores.csv`
5. 汇总统计与 top 结构：`round_XX/summary.json`
6. 压缩成 LLM 输入：`round_XX/evaluator_summary.json`
7. 调用 LLM 并审计：`round_XX/qwen_output.json`

闭环入口：`CrystaLLM/bin/prompt_optimization_loop.py`

## 3. ZMQ 多物性协议

请求（JSON 字符串）：

```json
{"cif":"<CIF文本>","properties":["formation_energy","bandgap"]}
```

响应（JSON 字符串）：

```json
{"ok":true,"formation_energy":-1.23,"bandgap":3.45}
```

失败时 `ok=false` 并附带 `errors`：

```json
{"ok":false,"formation_energy":null,"bandgap":null,"errors":{"atoms":"...","bandgap":"..."}}
```

## 4. 过滤器开关：成分/计量一致性

闭环脚本支持：

- `--validation-check-composition`（默认）
- `--no-validation-check-composition`

关闭后，`formula_ok` / `atom_site_multiplicity_ok` 不再作为硬门槛，且在元数据里会写为 `null`。
