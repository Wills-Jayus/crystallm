\# LLaMA 输出乱码问题复盘（v7\_Large 闭环）



本文面向第一次接触本项目的同学，解释为什么本地 LLaMA 模型的输出仍然是乱码，并给出直接证据与代码位置。



\## 1. 现象与结论

\- 在 `experiments/na2cl2\_json\_loop\_v7\_Large/round\_01/llama\_output.json` 与 `round\_02/llama\_output.json` 中，`analysis` 字段均为回退文案“LLM 输出不符合要求，使用回退。”，`next\_prompt\_lines` 被原样复制为上一轮 prompt（`data\_Na2Cl2` + `\_symmetry\_space\_group\_name\_H-M   P4/mmm`），说明模型原始回复无法解析成 JSON。

\- `\_raw` 字段保存的原始生成文本是汉字碎片、半截字段名或乱码，缺乏有效的 JSON 结构，导致解析失败并触发回退。

\- 因此闭环的 prompt 从 round\_01 到 round\_03 一直未被 LLaMA 更新，实际上停滞在初始提示。



\## 2. 乱码示例

\- Round 1（`experiments/na2cl2\_json\_loop\_v7\_Large/round\_01/llama\_output.json`）

&nbsp; ```json

&nbsp; "\_raw": "口在《\_副进行对自对应的 \_over在\\n非对的 \_cr\\n选者过长过过过你过，程序的《的提示提示导过过的对\\nCs\\n有需要自\\nCs\\nZ\\nCa\\n优合上上\\n原\\n你\\n的\\n与《\_过\\n的\\n的\\n上\\n上\\n《cr过过都的特《cr\\n中在上的\_过过过的《过进行于过自过优电上\_过过下过在\_的在\_上\_上过对上其自上《\_常《\_中个\_的\\n提示于\_以上\\n通过\\n问题的提示的上\\nD\\ncr过的\_的\_的上\_表《对《\_的《\_的《过优提示的《过每过的进行过对你-ma自即过\_的\_《\_的 2需要在物系统系统的加即下优进行\\n与行\\n提示上《\_《进行《\_推的\_和\_行的上用户提示的的上是\_《c，\_《下》进行上有系统\\n上\_《在系统\\n"

&nbsp; ```

\- Round 2（`experiments/na2cl2\_json\_loop\_v7\_Large/round\_02/llama\_output.json`）

&nbsp; ```json

&nbsp; "\_raw": "\[{\\"\_chemical\_formula\_sum\\": \\"\\", \\"cif\\": {\\"\_chemical\_formula\_sum\\": \\"\\", \\"cif\\": {\\"\_chemical\_formula\_sum\\": \\"\\", \\"cif\\": {\\"\_chemical\_formula\_sum\\": \\"\\", \\"cif\\": {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum）。\\n\\n\[{\\"\_chemical\_formula\_sum : {\\"\_chemical\_formula\_sum 的 \_cement\_chemical\_chemical\_smtv的 \_cement\_chemical\_formula\_sum\_chemical\_formula\_sum 的 \_chemical\_formula\_sum {\\n\_0\\nco的 \_ cation，\_cho的 \_ c\\n在\\n在\\n在\\ncr\\n，\_化学引优合合最长时的 \_cement助的 \_cement的 c\\n的 CIF c\\n手\\n选\\n每的 \_c\\n选\\n中\\n工作\\n的\\n对\\n可以使用的 \_c，\_在起需要\_来自的 \_c过过过过过）来过过过\\n\\n可以在过过的\\n对进行运行更反过过"

&nbsp; ```

\- 在两轮中，客户端解析失败后都回退为固定 JSON，`next\_prompt\_lines` 未被更新。



\## 3. 关键代码位置

\- 本地 LLaMA 服务端：`model/llama\_local\_api.py`

&nbsp; - 使用 HF pipeline 直接生成，没有 OpenAI 的 `response\_format` 约束：

&nbsp;   ```python

&nbsp;   \_text\_gen = pipeline("text-generation", model=\_model, tokenizer=\_tokenizer)

&nbsp;   out = \_text\_gen(

&nbsp;       prompt,

&nbsp;       max\_new\_tokens=req.max\_tokens,

&nbsp;       temperature=req.temperature,

&nbsp;       top\_p=req.top\_p,

&nbsp;       do\_sample=req.temperature > 0,

&nbsp;       return\_full\_text=False,

&nbsp;       eos\_token\_id=\_tokenizer.eos\_token\_id,

&nbsp;       pad\_token\_id=\_tokenizer.eos\_token\_id,

&nbsp;   )\[0]\["generated\_text"]

&nbsp;   ```

&nbsp; - 解析失败时的回退逻辑（将 `\_raw` 填入）：

&nbsp;   ```python

&nbsp;   try:

&nbsp;       parsed = \_extract\_json(content\_raw)

&nbsp;       ...

&nbsp;   except Exception:

&nbsp;       fallback\_lines = user\_payload.get("last\_prompt\_lines") or \[]

&nbsp;       safe\_output = {

&nbsp;           "analysis": "LLM 输出不符合要求，使用回退。",

&nbsp;           "next\_prompt\_lines": fallback\_lines,

&nbsp;           "\_raw": content\_raw,

&nbsp;       }

&nbsp;       content = json.dumps(safe\_output, ensure\_ascii=False)

&nbsp;   ```

\- LLaMA 客户端：`model/CrystaLLM/bin/llama\_client.py`

&nbsp; - system prompt 仅做软约束，仍采用较高随机性（默认 `temperature=0.7, top\_p=0.95`），没有硬性语法限制。

&nbsp; - `\_extract\_first\_json` 解析失败时同样回退，保持上一轮 prompt。



\## 4. 原因归纳

\- \*\*缺乏硬性 JSON 约束\*\*：HF `pipeline` 不支持 OpenAI 的 `response\_format={"type": "json\_object"}`，也未使用语法约束/grammar decoding，模型生成阶段完全依赖提示自律。

\- \*\*采样随机性较高\*\*：实验配置 `llama\_temperature=0.7, llama\_top\_p=0.95`（见 `experiments/na2cl2\_json\_loop\_v7\_Large/experiment\_config.json`），在大模型未对齐 JSON 的情况下更容易跑飞。

\- \*\*模型指令对齐不足\*\*：`m3rg-iitd/llamat-3-chat` 对“仅输出 JSON”遵从度有限，在当前系统提示下仍产出汉字碎片/重复字段名。

\- \*\*无额外停止符或内容过滤\*\*：生成调用只依赖默认 `eos`，未设置自定义 `stop\_sequences` 或输出过滤，乱码直接进入 `\_raw` 并触发回退。



\## 5. 直接后果

\- LLaMA 输出无法被解析，闭环每轮都沿用旧的 prompt，失去“优化提示词”功能。

\- 日志被大量乱码污染，`llama\_output.json` 只能看到回退 JSON，无有效的模型分析信息。



\## 6. 调用链与回退路径细化

\- 闭环存在\*\*双层回退\*\*：  

&nbsp; 1. 服务端 `llama\_local\_api.py` 在解析失败时立即返回 `{"analysis": "LLM 输出不符合要求，使用回退。", ...}`，并把原始生成塞入 `\_raw`。一旦此处回退，客户端拿到的就是“伪 JSON”。  

&nbsp; 2. 客户端 `bin/llama\_client.py` 再尝试解析该 JSON；如果仍失败或字段缺失，就再次构造同样的回退结构。  

\- 客户端在 `\_clean\_raw\_text` 中会剔除控制字符并在 2000 字节处截断，因此 `llama\_output.json` 中的 `\_raw` 可能只是原始乱码的片段，分析时要注意这一点。

\- 由于两个环节的回退提示完全一致，可以通过 `\_raw` 是否仍是乱码来判断问题发生在哪一层：若 `\_raw` 乱码，说明模型输出在服务端就已失控；若 `\_raw` 看似正常而仍被回退，问题才可能出在客户端解析。



\## 7. 可行的改进方向（供参考）

\- 降温/收紧采样（如 `temperature 0.3–0.5`、`top\_p 0.8–0.9`）先验证稳定性。

\- 在生成端增加语法约束或 JSON decoding（如 transformers 的约束解码、logits processor、或通过第三方工具提供 JSON grammar）。

\- 在 system prompt 中列出合法键/空间群白名单，并增加自定义 `stop\_sequences`，减少无关文本。

\- 如果仍不稳，考虑换用在结构化输出上对齐更好的模型。 



