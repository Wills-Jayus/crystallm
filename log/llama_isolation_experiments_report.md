\# LLaMA 本地部署隔离实验报告



\## 1. 背景与目的



在完整 CrystaLLM + ALIGNN + LLaMA JSON 闭环中，我们观察到 LLaMA 优化器始终返回回退文案（“LLM 输出不符合要求，使用回退。” / "LLM output invalid, using fallback."），`next\_prompt\_lines` 一直保持不变，说明模型原始输出无法被解析为合法 JSON。本报告在 `sandbox/` 目录下设计了两组\*\*隔离实验\*\*，目的是：



1\) 脱离闭环，直接调用本地 `m3rg-iitd/llamat-3-chat`，验证在当前 API 包装下能否产出合法 JSON；  

2\) 在不要求 JSON 的前提下，用普通问答测试模型本体是否能输出正常、语义正确的自然语言。



---



\## 2. 实验一：JSON 模式单次调用自检（`sandbox/llama\_json\_sanity/`）



\### 2.1 目标



在最小化、固定输入的前提下，通过 `bin/llama\_single\_test.py` 直接调用本地 LLaMA API，检查：



\- 部署好的服务在\*\*严格要求 JSON\*\* 的情况下，是否曾经成功返回过一个可解析的 JSON 对象；

\- 温度从 0 提高到 0.5 是否能改善/恶化这一情况。



\### 2.2 实验设置



目录：`sandbox/llama\_json\_sanity/`



\- \*\*输入 payload\*\*：`payload\_round1.json`

&nbsp; ```json

&nbsp; {

&nbsp;   "input": {

&nbsp;     "last\_prompt\_lines": \[

&nbsp;       "data\_Na2Cl2",

&nbsp;       "\_symmetry\_space\_group\_name\_H-M   P4/mmm"

&nbsp;     ],

&nbsp;     "evaluator\_summary": {

&nbsp;       "round": 0,

&nbsp;       "metrics": {

&nbsp;         "n\_structures": 0,

&nbsp;         "n\_scored\_ok": 0,

&nbsp;         "valid\_physical\_ratio": 0.0,

&nbsp;         "n\_validation\_pass": 0,

&nbsp;         "properties": {}

&nbsp;       },

&nbsp;       "top\_structures": \[]

&nbsp;     }

&nbsp;   }

&nbsp; }

&nbsp; ```

&nbsp; - 结构与闭环中的 `llama\_output.json\["input"]` 一致，只保留最关键字段，用作“最小重现实验”。



\- \*\*调用脚本\*\*：`python bin/llama\_single\_test.py ...`

&nbsp; ```bash

&nbsp; cd /root/autodl-tmp/model/CrystaLLM

&nbsp; for temp in 0.0 0.2 0.5; do

&nbsp;   python bin/llama\_single\_test.py \\

&nbsp;     --input-json sandbox/llama\_json\_sanity/payload\_round1.json \\

&nbsp;     --api-base http://localhost:8000/v1 \\

&nbsp;     --model m3rg-iitd/llamat-3-chat \\

&nbsp;     --temperature $temp \\

&nbsp;     --top-p 0.8 \\

&nbsp;     --max-tokens 256 \\

&nbsp;     > sandbox/llama\_json\_sanity/runs/t${temp}\_response.log \\

&nbsp;     2> sandbox/llama\_json\_sanity/runs/t${temp}\_stderr.log;

&nbsp; done

&nbsp; ```



\- \*\*调用协议\*\*：与闭环完全一致——走 `llama\_local\_api.py` 的 `/v1/chat/completions`，system prompt 使用 `bin/llama\_client.py` 中的 JSON 制约版，期望 LLaMA 直接输出满足协议的 JSON 文本。



\### 2.3 结果概览



每个温度对应的响应都保存在：



\- `runs/t0.0\_response.log`

\- `runs/t0.2\_response.log`

\- `runs/t0.5\_response.log`



三组结果的共同点：



\- `choices\[0].message.content` \*\*始终是服务端构造的回退 JSON\*\*：

&nbsp; ```json

&nbsp; {

&nbsp;   "analysis": "LLM output invalid, using fallback.",

&nbsp;   "next\_prompt\_lines": \["data\_Na2Cl2", "\_symmetry\_space\_group\_name\_H-M   P4/mmm"],

&nbsp;   "\_raw": "..."

&nbsp; }

&nbsp; ```

\- `\_raw` 字段中保存了模型的“原始输出片段”，但均无法解析为\*\*一个干净的 JSON 对象\*\*：

&nbsp; - `t0.0`：`\_raw` 里包含 `ucwords("cement\_composition")` 等看似模板/代码残片；

&nbsp; - `t0.2`：在重复的 payload 片段中混入 `ucwords("prompt\_lines")` 等字符串；

&nbsp; - `t0.5`：`\_raw` 中嵌套多份原始 payload，加上杂质标记如 `JADX PROXY`，最后被截断。



这些响应都在服务端 `llama\_local\_api.py` 的 `\_extract\_json` 阶段就失败了，因此被包装成统一的回退 JSON。客户端 `llama\_single\_test.py` 只是原样打印返回内容，并没有再进行二次回退。



\### 2.4 小结



\- 即使在\*\*极简输入\*\*和多组温度下，本地 `m3rg-iitd/llamat-3-chat` + 当前 FastAPI 包装 \*\*从未成功返回一个合规 JSON 对象\*\*。  

\- 所谓“乱码 JSON 输出”并非闭环逻辑或 `llama\_client.py` 的问题，而是在最前面的 `/v1/chat/completions` 层就已经无法从模型生成中提取 JSON，被迫走回退路径。



---



\## 3. 实验二：自由文本问答能力自检（`sandbox/llama\_freeform\_test/`）



\### 3.1 目标



在\*\*完全不要求 JSON\*\* 的前提下，直接把 LLaMA 当作普通聊天模型使用，验证：



\- 模型本体是否具备正常的自然语言理解与生成能力；

\- 如果有，问题是否单纯出在我们强行要求“只输出 JSON”的协议上。



\### 3.2 实验设置



目录：`sandbox/llama\_freeform\_test/`



\- \*\*脚本\*\*：`run\_freeform.py`

&nbsp; ```python

&nbsp; API\_BASE = "http://localhost:8000/v1"

&nbsp; MODEL = "m3rg-iitd/llamat-3-chat"

&nbsp; SYSTEM\_PROMPT = "You are a helpful assistant. Answer the user's question directly."

&nbsp; USER\_PROMPT = "Please explain what band gap means in solid state physics."



&nbsp; messages = \[

&nbsp;     {"role": "system", "content": SYSTEM\_PROMPT},

&nbsp;     {"role": "user", "content": USER\_PROMPT},

&nbsp; ]



&nbsp; payload = {

&nbsp;     "model": MODEL,

&nbsp;     "messages": messages,

&nbsp;     "temperature": 0.2,

&nbsp;     "top\_p": 0.8,

&nbsp;     "max\_tokens": 256,

&nbsp; }

&nbsp; ```



\- \*\*运行命令\*\*：

&nbsp; ```bash

&nbsp; cd /root/autodl-tmp/model/CrystaLLM/sandbox/llama\_freeform\_test

&nbsp; python run\_freeform.py > run.log 2> run.err

&nbsp; ```



脚本会：



\- 将完整 JSON 响应写入 `freeform\_response.json`；

\- 将 `choices\[0].message.content`（即 API 返回的 content 字段）打印到 `run.log`。



\### 3.3 结果解析



\- `run.log` 内容：

&nbsp; ```text

&nbsp; === Raw response ===

&nbsp; {"analysis": "LLM output invalid, using fallback.", "next\_prompt\_lines": \[], "\_raw": "The band gap in solid state physics is the energy difference between the highest occupied energy level (valence band) and the lowest unoccupied energy level (conduction band). This energy difference is a key property of a material and is used to describe the material's electrical conductivity and optical properties.<|im\_end|>\\n<|im\_start|>answer\\nThe band gap in solid state physics is the energy difference between the highest occupied energy level (valence band) and the lowest unoccupied energy level (conduction band).<|im\_end|> ..."}

&nbsp; ```

\- 打开 `freeform\_response.json` 可以看到：

&nbsp; - `choices\[0].message.content` 是一段 JSON 字符串；

&nbsp; - 其中的 `\_raw` 字段里，模型多次给出了\*\*完全合理的英文解释\*\*：

&nbsp;   > The band gap in solid state physics is the energy difference between the highest occupied energy level (valence band) and the lowest unoccupied energy level (conduction band). This energy difference is a key property of a material and is used to describe the material's electrical conductivity and optical properties.

&nbsp; - 同时夹杂了一些 chat 模板控制标记，如 `<|im\_start|>answer` / `<|im\_end|>`，这与 LLaMA3 系列默认 chat 模板一致。



\### 3.4 小结



\- \*\*模型本体是正常的\*\*：在不要求 JSON 的简单问答场景下，`m3rg-iitd/llamat-3-chat` 能够给出语义正确、结构清晰的物理概念解释。  

\- 问题在于我们当前的本地 API 封装 (`llama\_local\_api.py`) 对所有请求一视同仁地执行 `\_extract\_json`，强行把自由文本当 JSON 解析；一旦失败，就返回回退 JSON，并把原始自然语言塞进 `\_raw`。这就是为什么即便在自由实验中，`analysis` 仍是 fallback，而真正有用的答案藏在 `\_raw` 里。



---



\## 4. 综合结论与后续方向



1\. \*\*JSON 模式实验的关键信息\*\*：

&nbsp;  - 在极简 payload 和不同温度下，本地服务从未返回合规 JSON；

&nbsp;  - 这说明“闭环中 LLaMA 输出乱码”并不是下游解析代码的问题，而是\*\*当前协议 + 模型组合本身不具备稳定 JSON 输出能力\*\*。



2\. \*\*自由文本实验的关键信息\*\*：

&nbsp;  - 模型本体的自然语言能力是正常的，能够在物理问答任务上给出高质量回答；

&nbsp;  - 换句话说，“模型坏了”和“协议要求过于苛刻”这两个可能性里，更大问题出在后者。



3\. \*\*直接改进思路（仅列方向，不在本报告中实现）\*\*：

&nbsp;  - 在保持 JSON 闭环前提下：

&nbsp;    - 收紧 system prompt（加入 JSON few-shot 模板）、进一步降温，观察是否能提高 JSON 合规率；

&nbsp;    - 或者在生成时引入 grammar / constrained decoding，而不仅仅用普通 `pipeline`。

&nbsp;  - 若短期只想要稳定运行闭环：

&nbsp;    - 可以考虑在 LLaMA 层先退回“自然语言 + 模板后处理”的方案，例如让模型输出结构化 bullet points，由我们再转成 JSON，而不是要求模型一步到位输出 JSON。



通过这两组隔离实验，我们已经比较清楚地把问题定位到“当前 JSON 协议 + 本地 API 包装”这一层，而不是 ALIGNN 或 CrystaLLM 本身，也证明了部署好的 LLaMA 模型在自由输出场景下是可用的。今后无论是继续强化 JSON 输出，还是调整协议回到自由文本，本报告都可以作为排查依据与对照基线。



