\# Qwen 模板改造与问题汇总（历史记录：早期 LLaMA 阶段）



\## 背景

\- 之前的闭环要求 LLaMA 直接输出严格 JSON，模型频繁返回乱码/回退，prompt 无法更新。

\- 为了降低约束，改成“固定文本模板”并在本地解析为 JSON：三段头 `ANALYSIS:` / `NEXT\_PROMPT\_LINES:` / `END`。

\- 服务端 `qwen\_local\_api.py`（旧名：`llama\_local\_api.py`）也改为默认返回原始生成（`QWEN\_STRICT\_JSON=false`，兼容 `LLAMA\_STRICT\_JSON`），不再强制 JSON 解析，增加了尝试在 `END` 处停的逻辑。



\## 已做的代码改动（关键点）

\- `bin/qwen\_client.py`（旧名：`bin/llama\_client.py`）：

&nbsp; - System prompt 改为英文模板，禁止回显用户 JSON，只允许三段输出，并给了示例。

&nbsp; - 本地解析 `\_parse\_template` 读取三段，失败则回退，附 `\_raw`。

\- `qwen\_local\_api.py`：

&nbsp; - 默认关闭严格 JSON（通过环境变量 `QWEN\_STRICT\_JSON` 控制，兼容 `LLAMA\_STRICT\_JSON`）；关闭时直接返回模型原始生成。

&nbsp; - 追加 `/root/autodl-tmp/model/CrystaLLM/qwen\_raw\_generation.log` 记录 `generated\_text` 原文。

&nbsp; - 尝试把 `\\nEND` 的 token 加入 `eos\_token\_id`（但目前只是 last token，不是完整 stop）。

&nbsp; - 修复了多次缩进错误导致的 `IndentationError`。



\## 当前启动方式（建议）

```bash

cd /root/autodl-tmp/model

PYTHONPATH=. QWEN\_STRICT\_JSON=false uvicorn qwen\_local\_api:app --host 0.0.0.0 --port 8000

```

确保加载的是仓库内的 `llama\_local\_api.py` 而非安装包，必要时在相同环境下运行：

```bash

python - <<'PY'

import qwen\_local\_api, sys

print("path:", qwen\_local\_api.\_\_file\_\_)

print("sys.path\[:3]:", sys.path\[:3])

PY

```



\## 现象（仍未解决）

\- 单次调用命令（温度 0.1）：

&nbsp; ```bash

&nbsp; cd /root/autodl-tmp/model/CrystaLLM

&nbsp; python bin/qwen\_single\_test.py \\

&nbsp;   --input-json experiments/na2cl2\_json\_loop\_v8\_Large/round\_02/qwen\_output.json \\

&nbsp;   --api-base http://localhost:8000/v1 \\

&nbsp;   --temperature 0.1

&nbsp; ```

&nbsp; 返回的 `content` 依旧是“回显 + 乱码”，例如：

&nbsp; ```

&nbsp; {"last\_prompt\_lines": \["data\_Na2Cl2", "P4/mmm"], ... 1-1<1-1-1-1-1 ...}

&nbsp; ```

&nbsp; 没有出现 `ANALYSIS:` / `NEXT\_PROMPT\_LINES:`，说明模型未遵守模板。

\- `qwen\_raw\_generation.log` 也只记录了同样的乱码段落。



\## 怀疑原因

1\. 模型对模板不服从：即便有示例，仍回显用户 JSON 并生成垃圾 token。

2\. Stop 未生效：仅将 `\\nEND` 的最后一个 token 放进 `eos\_token\_id`，无法匹配完整序列。

3\. 仍可能加载了旧代码路径（但已多次确认/重启）。



\## 建议的下一步（可由接手者实施）

1\. \*\*实现真正的 stopping criteria\*\*：在 `qwen\_local\_api.py` 添加自定义 `StoppingCriteria` 匹配完整 `\\nEND` 序列，而不是只追加单个 eos token。

2\. \*\*进一步降随机性\*\*：调用端试 `temperature=0`、`top\_p=0.6`，配合新的 stop。

3\. \*\*更强模板提示\*\*：增加 few-shot 示例，强调“不回显、不输出 JSON”；必要时加入禁止词（如 `{`, `"`）或让模型先输出纯文本，再由本地脚本转 JSON。

4\. \*\*确认加载路径\*\*：重启前先打印 `qwen\_local\_api.\_\_file\_\_`，确保确实使用仓库文件。



如果仍无效，可考虑：

\- 在服务端过滤掉回显的用户 JSON，再让模型续写模板（风险：模型上下文缺失）。

\- 使用支持 JSON/grammar decoding 的生成器替代当前 HF pipeline。



\## 参考文件与路径

\- 服务端：`/root/autodl-tmp/model/llama\_local\_api.py`

\- 客户端：`/root/autodl-tmp/model/CrystaLLM/bin/llama\_client.py`

\- 原始生成日志：`/root/autodl-tmp/model/CrystaLLM/llama\_raw\_generation.log`

\- 单次调用脚本：`/root/autodl-tmp/model/CrystaLLM/bin/llama\_single\_test.py`

\- 最新实验输出示例：`/root/autodl-tmp/model/CrystaLLM/experiments/na2cl2\_json\_loop\_v8\_Large/round\_02/llama\_output.json`



