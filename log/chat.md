\# CrystaLLM + ALIGNN + LLaMat‑3 调试日志（2025‑12‑02～03，新服务器闭环实验）



> 用途：记录在新环境上从“单次采样”到“JSON 闭环 + ALIGNN 打分”完整调通过程中的关键问题、修复点和对比实验，作为后续复现实验和排查问题的参考。



---



\## 1. 环境与总体目标



\- 路径与项目：`/root/autodl-tmp/model/CrystaLLM`

\- 三个 Conda 环境：

&nbsp; - `crystallm\_env`：运行 CrystaLLM（`sample.py`、`prompt\_optimization\_loop.py` 等）。

&nbsp; - `alignn-score`：运行 ALIGNN ZMQ 评分服务。

&nbsp; - `llama3-chat`：运行本地 LLaMat‑3‑Chat HTTP 服务（FastAPI + Transformers）。

\- 目标闭环：

&nbsp; 1. 用 CrystaLLM 从 CIF 提示行生成一批候选结构（`sample\_\*.cif`）。

&nbsp; 2. 用 `crystallm.cif\_cleaning` 做 CIF 清洗 + 物理/晶体学规则检查，过滤明显不合理结构。

&nbsp; 3. 用 ALIGNN（ZMQ 服务）为每个 CIF 预测 formation\_energy / bandgap。

&nbsp; 4. 将统计摘要 `evaluator\_summary` + 上一轮 prompt 输入 LLaMat‑3(JSON 优化器)，得到新的 CIF 高层提示行。

&nbsp; 5. 多轮迭代后，选出带隙高且物理/几何合理的结构。



---



\## 2. LLaMat‑3 本地服务与 JSON API



\### 2.1 基础自测脚本 `run\_llamat3\_chat.py`



\- 路径：`model/run\_llamat3\_chat.py`

\- 核心逻辑：

&nbsp; - 使用 `AutoTokenizer` + `AutoModelForCausalLM` 加载 `m3rg-iitd/llamat-3-chat`；

&nbsp; - 用 `tokenizer.apply\_chat\_template` 构造对话 prompt；

&nbsp; - 调 `pipeline("text-generation")` 生成文本并打印。

\- 日志中看到：

&nbsp; - 所有 `model-0000x-of-00005.safetensors` 分片下载完成；

&nbsp; - `Loading checkpoint shards: 100%`；

&nbsp; - 输出里可以看到正确回答 “band gap 是什么” 的自然语言内容。

\- 结论：Transformers + 权重缓存正常，模型能在 GPU 上推理。



\### 2.2 本地 HTTP 服务 `llama\_local\_api.py`



\- 路径：`model/llama\_local\_api.py`

\- 功能：提供 OpenAI 兼容的 `/v1/chat/completions`，供 `bin/llama\_client.py` 与闭环脚本调用。

\- 关键实现：

&nbsp; - 懒加载模型：在第一次请求时加载 tokenizer / model / pipeline。

&nbsp; - `ChatRequest` / `ChatResponse` Pydantic 模型约束入参与返回值。

&nbsp; - 使用 `apply\_chat\_template` 将 messages 转为字符串 prompt。

&nbsp; - 初版只是普通 `text-generation`，返回模型生成的全文。

\- 为了配合“严格 JSON 输出 + 减少跑飞”，做了几处改动：

&nbsp; 1. 在 pipeline 调用中加：

&nbsp;    - `do\_sample=req.temperature > 0`、`return\_full\_text=False`，并指定 `eos\_token\_id` / `pad\_token\_id`，减少乱跑。

&nbsp; 2. 新增 `\_extract\_json(text)`：若 `content\_raw` 不是合法 JSON，尝试从第一个 `{` 起解析 JSON 对象；失败则抛错。

&nbsp; 3. 在 `chat()` 里：

&nbsp;    - 尝试解析模型输出为 JSON，保留 `analysis` / `next\_prompt\_lines`，再 `json.dumps` 回 content；

&nbsp;    - 如缺字段或解析失败，则使用回退 JSON：

&nbsp;      ```json

&nbsp;      {

&nbsp;        "analysis": "LLM 输出不符合要求，使用回退。",

&nbsp;        "next\_prompt\_lines": <从用户 payload 的 last\_prompt\_lines 复制>,

&nbsp;        "\_raw": "<原始字符串>"

&nbsp;      }

&nbsp;      ```



这保证了对外 `choices\[0].message.content` \*\*始终\*\*是合法 JSON 字符串，客户端解析更稳定。



---



\## 3. CrystaLLM 采样脚本与路径相关修复



\### 3.1 `bin/sample.py` 的行为



\- 使用 dataclass `SampleDefaults` 读取命令行 key=value 配置：

&nbsp; - `out\_dir`：模型目录，内部拼成 `ckpt\_path = os.path.join(out\_dir, "ckpt.pt")`。

&nbsp; - `start`：可以是 `FILE:path` 形式，从文件读取 prompt。

&nbsp; - `num\_samples`、`max\_new\_tokens`、`temperature`、`top\_k`、`seed`、`device`、`dtype` 等控制采样。

\- 生成逻辑：

&nbsp; - 用 CIFTokenizer 编码 prompt，调用 GPT 模型生成 token 序列；

&nbsp; - 解码成 CIF 文本，如果 `target=file`，则写 `sample\_\*.cif`。



\### 3.2 `prompt\_optimization\_loop.py` 中的路径 bug



初始版本在运行 `sample.py` 时有两个问题：



1\. \*\*模型目录相对路径问题\*\*  

&nbsp;  - 调用 `sample.py` 时使用 `out\_dir=crystallm\_v1\_small`，但当前工作目录是 `round\_XX/cifs`。

&nbsp;  - 结果 `sample.py` 在 `round\_XX/cifs/crystallm\_v1\_small/ckpt.pt` 查找 ckpt，导致：\\

&nbsp;    `FileNotFoundError: crystallm\_v1\_small/ckpt.pt`。

&nbsp;  - 修复：在 `run\_crystallm\_sampling` 中将 `args.model\_dir` 规范为绝对路径：

&nbsp;    ```python

&nbsp;    model\_dir = Path(args.model\_dir)

&nbsp;    if not model\_dir.is\_absolute():

&nbsp;        model\_dir = (REPO\_ROOT / model\_dir).resolve()

&nbsp;    ...

&nbsp;    f"out\_dir={model\_dir}"

&nbsp;    ```



2\. \*\*prompt.txt 相对路径问题\*\*  

&nbsp;  - `start=FILE:experiments/.../round\_01/prompt.txt`，但在子目录 `cifs` 中运行，导致找不到该文件。

&nbsp;  - 修复：在 `run\_crystallm\_sampling` 里对 `prompt\_path` 和 `cifs\_dir` 使用 `resolve()`，确保传给 `sample.py` 的路径是绝对路径。



修复后，闭环脚本能正常调用 `sample.py`，在每轮 `round\_XX/cifs/` 下生成 8 个 `sample\_\*.cif`。



---



\## 4. LLaMA JSON 闭环逻辑与容错处理



\### 4.1 `bin/llama\_client.py` 的协议与 system prompt



\- 新的 `DEFAULT\_SYSTEM\_PROMPT` 按你的设计收紧：

&nbsp; - 角色：晶体材料提示词优化助手。

&nbsp; - 输入：单个 JSON 对象，包含 `last\_prompt\_lines` 与 `evaluator\_summary`。

&nbsp; - 输出约束（强制）：

&nbsp;   - 只能输出一个合法 JSON 对象；

&nbsp;   - 且 \*\*仅包含\*\* 字段：`analysis`（中文诊断）、`next\_prompt\_lines`（新的 CIF 行列表）；

&nbsp;   - 禁止输出 `edit\_suggestions` 等其它字段，禁止回显输入，禁止 JSON 外任何文字。

&nbsp; - 内部解释了全局优化目标：带隙尽量大、形成能不过高、满足物理约束。



\### 4.2 请求与解析逻辑



\- `\_build\_messages`：

&nbsp; - system：上述 `DEFAULT\_SYSTEM\_PROMPT`。

&nbsp; - user：`json.dumps({"last\_prompt\_lines": \[...], "evaluator\_summary": {...}}, ensure\_ascii=False)`。

\- `\_request\_chat`：

&nbsp; - 向 `f"{api\_base}/chat/completions"` 发送标准 OpenAI 风格请求：

&nbsp;   ```json

&nbsp;   {

&nbsp;     "model": "<llm-name>",

&nbsp;     "messages": \[...],

&nbsp;     "temperature": ...,

&nbsp;     "top\_p": ...,

&nbsp;     "max\_tokens": ...

&nbsp;   }

&nbsp;   ```



\- `optimize\_prompt`：

&nbsp; 1. 从响应中取出 `content = choices\[0].message.content`。

&nbsp; 2. 调 `\_parse\_json\_content(content)` 做“宽容 JSON 解析”：

&nbsp;    - 先尝试 `json.loads(stripped)`；

&nbsp;    - 若失败，从第一个 `{` 起用 `JSONDecoder().raw\_decode` 解析第一个 JSON 对象，忽略后面垃圾；

&nbsp;    - 若仍失败，抛出 `JSONDecodeError`。

&nbsp; 3. 若解析仍失败，则直接构造回退 payload：

&nbsp;    ```python

&nbsp;    parsed = {

&nbsp;      "analysis": "LLM 输出无法解析，使用回退。",

&nbsp;      "next\_prompt\_lines": last\_prompt\_lines\_without\_blanks,

&nbsp;      "\_raw\_content": content,

&nbsp;    }

&nbsp;    ```

&nbsp; 4. 即便解析成功，若 `analysis` 为空或 `next\_prompt\_lines` 不合法，也会回退到“沿用上一轮 prompt”。



这一层配合服务端的 JSON 强制，使闭环在 LLaMA 输出不规范时也不会崩溃，而是保持 prompt 不变往下跑。



---



\## 5. CIF 清洗与验证：`crystallm/cif\_cleaning.py` 与 `\_metrics.py`



\### 5.1 `\_metrics.py` 的核心函数



\- `bond\_length\_reasonableness\_score(cif\_str, tolerance=0.32, h\_factor=2.5)`：

&nbsp; - 用 CrystalNN 寻找近邻，对每一条键：

&nbsp;   - 根据电负性差决定用“原子半径”还是“离子半径”估算期望键长；

&nbsp;   - 计算 `bond\_ratio = actual\_length / expected\_length`；

&nbsp;   - 在 `1±tolerance` 范围内记为合理，否则记为不合理（H 键有特殊处理）；

&nbsp; - 最终返回 “合理键数 / 全部键数” 的 0–1 分数。



\- `is\_space\_group\_consistent(cif\_str)`：

&nbsp; - 用 `Structure.from\_str` + `SpacegroupAnalyzer` 解析结构，得到检测空间群；

&nbsp; - 与 CIF 中 `\_symmetry\_space\_group\_name\_H-M` 的声明空间群比较是否相同。



\- `is\_formula\_consistent(cif\_str)`：

&nbsp; - 用 `extract\_data\_formula` 得到 data 行符号对应的 Composition；

&nbsp; - 比较 `\_chemical\_formula\_sum` 与 `\_chemical\_formula\_structural` 是否与之在约化配比上一致。



\- `is\_atom\_site\_multiplicity\_consistent(cif\_str)`：

&nbsp; - 从 `\_chemical\_formula\_sum` 得到“期望原子计数”；

&nbsp; - 扫描 `\_atom\_site\_type\_symbol` + `\_atom\_site\_symmetry\_multiplicity` 统计实际计数；

&nbsp; - 两者完全相等则返回 True。



\- `is\_sensible(cif\_str, length\_lo=0.5, length\_hi=1000., angle\_lo=10., angle\_hi=170.)`：

&nbsp; - 粗略筛查晶胞长度与角度是否在合理范围。



\- `is\_valid(cif\_str, bond\_length\_acceptability\_cutoff=1.0)`：

&nbsp; - 等价于：

&nbsp;   - `is\_formula\_consistent`、`is\_atom\_site\_multiplicity\_consistent`、`is\_space\_group\_consistent` 全 True；

&nbsp;   - `bond\_length\_reasonableness\_score(cif\_str) >= cutoff`。



\### 5.2 `cif\_cleaning.py` 中的清洗 + 验证组合



\- `clean\_cif(cif\_str)`：

&nbsp; 1. 调用 `extract\_numeric\_property` 提取 `\_cell\_length\_\*` 与 `\_cell\_angle\_\*`；

&nbsp; 2. 调 `get\_unit\_cell\_volume(a, b, c, alpha, beta, gamma)` 做体积 sanity check；

&nbsp; 3. 提取 `\_symmetry\_space\_group\_name\_H-M`，非 `P 1` 时调用 `replace\_symmetry\_operators` 重写对称操作；

&nbsp; 4. 调 `remove\_atom\_props\_block` 去掉 `\_atom\_type\_\*` block。



\- `validate\_cif(cif\_str, bond\_length\_acceptability\_cutoff)`：

&nbsp; - 调用上述 `\_metrics` 几个函数，返回一个 `CIFValidationResult`：

&nbsp;   - `valid`（综合布尔）、`reasons`（字符串列表）、`bond\_length\_score`、`formula\_ok`、`space\_group\_ok`、`atom\_site\_multiplicity\_ok`。



\- `clean\_and\_validate\_cif(cif\_str, bond\_length\_acceptability\_cutoff)`：

&nbsp; - 尝试 `clean\_cif`；如抛异常，则记录 `cleaning\_failed: ...`，并标记 `valid=False`；

&nbsp; - 否则调用 `validate\_cif`，把各项结果写入 `metadata`；

&nbsp; - 若 `validation.valid=True`，返回 `(cleaned\_cif, metadata)`，否则返回 `(None, metadata)`。



\### 5.3 与原始项目评估脚本的对齐程度



\- 原始 `bin/evaluate\_cifs.py`：

&nbsp; - 使用 `is\_sensible` + `is\_atom\_site\_multiplicity\_consistent` + `is\_space\_group\_consistent` + `bond\_length\_reasonableness\_score` + `\_metrics.is\_valid` 做评估。

\- 当前闭环脚本：

&nbsp; - 已使用 `is\_formula\_consistent` / `is\_atom\_site\_multiplicity\_consistent` / `bond\_length\_reasonableness\_score` / `is\_space\_group\_consistent`；

&nbsp; - 未显式调用 `is\_sensible`（几何粗过滤），体积 sanity check 由 `get\_unit\_cell\_volume` 完成；

&nbsp; - 综合有效性通过 `validation\_ok`（与 `\_metrics.is\_valid` 逻辑等价）体现，并拆分出各子项布尔 + `bond\_length\_score` 数值。



---



\## 6. ALIGNN 集成与多处 bug 修复



\### 6.1 ZMQ 服务端：`resources/alignn\_zmq\_server\_multi.py`



\- 接口约定：

&nbsp; - 请求：`{"cif": "<CIF文本>", "properties": \["formation\_energy", "bandgap"]}`；

&nbsp; - 响应：`{"ok": bool, "formation\_energy": float?, "bandgap": float?, "errors": {...}?}`。

\- 核心函数：

&nbsp; - `load\_atoms(cif\_text)`：调用 `Atoms.from\_cif(from\_string=cif\_text, get\_primitive\_atoms=True/False)` 解析为 `jarvis.core.atoms.Atoms`；

&nbsp; - `predict\_property(prop, atoms, args)`：

&nbsp;   - 根据 `PROPERTY\_MODEL\_MAP` 选择 ALIGNN 预训练模型名；

&nbsp;   - 调用 `alignn.pretrained.get\_prediction`；

&nbsp;   - 再用 `extract\_numeric` 从返回值中提取 float。



\### 6.2 发现与修复的错误



1\. \*\*`get\_prediction` 参数重复 atoms\*\*  

&nbsp;  - 原始调用：

&nbsp;    ```python

&nbsp;    prediction = get\_prediction(

&nbsp;        model\_name,

&nbsp;        args.device,

&nbsp;        atoms=atoms,

&nbsp;        cutoff=args.cutoff,

&nbsp;        max\_neighbors=args.max\_neighbors,

&nbsp;    )

&nbsp;    ```

&nbsp;  - 但 `get\_prediction` 签名（在 `alignn-score` 环境里查看）是：\\

&nbsp;    `(model\_name='jv\_formation\_energy\_peratom\_alignn', atoms=None, cutoff=8, max\_neighbors=12)`

&nbsp;  - 于是报错：`get\_prediction() got multiple values for argument 'atoms'`。

&nbsp;  - 修复：改为：

&nbsp;    ```python

&nbsp;    prediction = get\_prediction(

&nbsp;        model\_name,

&nbsp;        atoms,

&nbsp;        cutoff=args.cutoff,

&nbsp;        max\_neighbors=args.max\_neighbors,

&nbsp;    )

&nbsp;    ```



2\. \*\*误传 `device` 关键字参数\*\*  

&nbsp;  - 在进一步调整时曾尝试传 `device=args.device`，但当前 `get\_prediction` 实现并不接受 `device` 参数：\\

&nbsp;    报错：`get\_prediction() got an unexpected keyword argument 'device'`。

&nbsp;  - 修复：\*\*完全移除\*\* `device` 关键字，只依赖 `get\_prediction` 内部根据环境决定设备。



3\. \*\*DGL 未启用 CUDA，引发 GPU device 报错\*\*  

&nbsp;  - 即使我们不显式传 `device`，`get\_prediction` / DGL 内部仍会探测 GPU；由于该环境安装的是 CPU 版 DGL，导致：\\

&nbsp;    `Device API cuda is not enabled. Please install the cuda version of dgl.`

&nbsp;  - 临时解决方案（不安装 GPU 版 DGL 的前提下）：

&nbsp;    - 启动 ALIGNN 服务时禁用 GPU：

&nbsp;      ```bash

&nbsp;      CUDA\_VISIBLE\_DEVICES="" python resources/alignn\_zmq\_server\_multi.py \\

&nbsp;        --host 0.0.0.0 --port 5555 \\

&nbsp;        --device cpu \\

&nbsp;        --properties formation\_energy bandgap

&nbsp;      ```

&nbsp;    - 这样 Torch/DGL 认为“没有 GPU”，整条 ALIGNN pipeline 在 CPU 上跑，评分正常。



修复后，在基线实验中 `scores\_sample\_base.csv` 显示 8 个样本均 `ok=True`，`formation\_energy` 多为负值、`bandgap` 在 ~0–0.48 eV 范围内，说明评分链路已经正常工作（虽然结构本身在我们的 CIF 规则下仍不合格）。



---



\## 7. 基线对比实验：只跑 CrystaLLM + CIF 检查 + ALIGNN



目的：区分“base 模型本身的问题”与“闭环 / LLaMA / 新规则引入的问题”。



\### 7.1 实验步骤



1\. 在 `crystallm\_env` 中，用原始 `sample.py` 从 prompt 采样：



&nbsp;  ```bash

&nbsp;  conda activate crystallm\_env

&nbsp;  cd /root/autodl-tmp/model/CrystaLLM



&nbsp;  python bin/sample.py \\

&nbsp;    out\_dir=crystallm\_v1\_small \\

&nbsp;    start=FILE:prompts/na2cl2\_round0.txt \\

&nbsp;    num\_samples=8 \\

&nbsp;    max\_new\_tokens=3000 \\

&nbsp;    temperature=0.8 \\

&nbsp;    top\_k=10 \\

&nbsp;    seed=1337 \\

&nbsp;    device=cuda \\

&nbsp;    dtype=bfloat16 \\

&nbsp;    target=file

&nbsp;  ```



&nbsp;  得到 8 个 `sample\_\*.cif`（直接在 repo 根目录下）。



2\. 用 `clean\_and\_validate\_cif` 批量检查这 8 个 CIF，写 `cif\_quality\_sample\_base.csv`。  

3\. 在 `alignn-score` + `crystallm\_env` 两个终端配合下，用 `alignn\_client.py` 对这 8 个 CIF 打分，得 `scores\_sample\_base.csv`。



\### 7.2 结果观察



1\. `cif\_quality\_sample\_base.csv`：

&nbsp;  - 所有样本 `valid=False`（或 `validation\_ok=False`），`reasons` 多为：

&nbsp;    - `'composition inconsistent'`；

&nbsp;    - 部分再加 `'atom site multiplicity inconsistent'`；

&nbsp;  - `bond\_length\_score` = 1.0（键长合理性良好）；

&nbsp;  - `space\_group\_ok=True`；

&nbsp;  - 结论：\*\*base 采样结构的主要问题在于化学式/位点 multiplicity 不一致，而不是键长或空间群\*\*。



2\. `scores\_sample\_base.csv`：

&nbsp;  - 所有行 `ok=True`；

&nbsp;  - `formation\_energy` 约在 \[-0.92, 0.41] eV/atom；

&nbsp;  - `bandgap` 多数接近 0，最高 ~0.48 eV。



这给出了一个“CrystaLLM\_v1\_small + ALIGNN”的\*\*干净基线\*\*：

\- 即使不加任何 JSON 闭环或 LLaMA，我们对 base 模型采样的评价已经是：

&nbsp; - “物性可算（ALIGNN OK），但组分/多重性在当前 CIF 规则下无一通过”。



---



\## 8. 闭环实验结果与基线对比



闭环实验输出路径：`experiments/na2cl2\_json\_loop\_final/`。



\### 8.1 Round 1 CIF 质量与评分



1\. `round\_01/cif\_quality.csv`（截取前几行）：



&nbsp;  ```text

&nbsp;  cif\_path,validation\_ok,validation\_reasons,bond\_length\_score,formula\_ok,space\_group\_ok,atom\_site\_multiplicity\_ok,cleaning\_error,cleaned,relative\_path

&nbsp;  .../round\_01/cifs/sample\_1.cif,False,\['cleaning\_failed: Bad international symbol P4/<unk>'],,False,False,False,,False,cifs/sample\_1.cif

&nbsp;  .../sample\_2.cif,False,\['cleaning\_failed: Bad international symbol P4/<unk>'],,False,False,False,,False,cifs/sample\_2.cif

&nbsp;  ...

&nbsp;  ```



&nbsp;  - 全部 `validation\_ok=False`；

&nbsp;  - 原因集中在 `cleaning\_failed: Bad international symbol P4/<unk>`；

&nbsp;  - `bond\_length\_score` 为空（清洗失败后没法继续算）。



2\. `round\_01/scores.csv`（截取前几行）：



&nbsp;  ```text

&nbsp;  cif\_path,ok,formation\_energy,bandgap,round,relative\_path,validation\_ok,validation\_reasons, ...

&nbsp;  ...sample\_1.cif,True,-1.22,0.13,1,cifs/sample\_1.cif,False,\['cleaning\_failed: Bad international symbol P4/<unk>'], ...

&nbsp;  ...sample\_2.cif,True,-0.02,0.96,...

&nbsp;  ...sample\_3.cif,True,-1.02,1.39,...

&nbsp;  ...sample\_4.cif,True,-0.88,2.37,...

&nbsp;  ```



&nbsp;  - ALIGNN 评分层面：所有样本 `ok=True`，formation\_energy/bandgap 有正常数值；

&nbsp;  - 带隙明显高于 base 采样那一批（0.96、1.39、2.37 eV vs 基线多在 <0.5 eV）。



\### 8.2 闭环 vs 基线：结构质量与带隙表现



\- \*\*结构合理性（按我们 CIF 规则）\*\*：

&nbsp; - 基线：`bond\_length\_score=1.0`、`space\_group\_ok=True`，但 `formula\_ok=False` / `atom\_site\_multiplicity\_ok=False` → 失败。

&nbsp; - 闭环 Round 1：直接死在 `cleaning\_failed: Bad international symbol P4/<unk>`，连键长/配比检查都没跑完 → 失败。



\- \*\*GNN 带隙/形成能（ALIGNN）\*\*：

&nbsp; - 基线：带隙大多接近 0，最高 ~0.48 eV；

&nbsp; - 闭环 Round 1：带隙能到 1–2.4 eV；形成能仍在合理负值/小正值范围。



综合来看：

\- \*\*带隙目标方面\*\*，闭环在当前设定下确实能“推动”搜索出带隙更大的结构（从 ALIGNN 角度看）；

\- \*\*物理/晶体学约束方面\*\*，不论是 base 采样还是闭环样本，在当前 prompt + CrystaLLM\_v1\_small 下都普遍不满足严格的配比/空间群/清洗规则，只是坏掉的形式不同：

&nbsp; - 基线：配比与 multiplicity 对不上；

&nbsp; - 闭环 Round1：空间群字段直接生成成 `P4/<unk>` 这种非法符号。



因此，后续需同时从两个方向改进：

1\. 更严格的 prompt 模板约束（特别是 `\_chemical\_formula\_sum`、`\_chemical\_formula\_structural` 和 `\_atom\_site\_\*` 部分）；

2\. 在清洗失败时仍尽量评估各子指标（例如在 `clean\_cif` 失败后用原始 CIF 跑 `validate\_cif`），让 `bond\_length\_score`、`formula\_ok`、`space\_group\_ok` 等数值尽可能可见。



---



\## 9. P4/<unk> 空间群问题分析



\### 9.1 prompt 中的空间群



\- 初始 prompt 文件：`prompts/na2cl2\_round0.txt`：

&nbsp; ```text

&nbsp; data\_Na2Cl2

&nbsp; \_symmetry\_space\_group\_name\_H-M   P4/mmm

&nbsp; ```

\- `P4/mmm` 是标准 H‑M 空间群符号，本身是\*\*完全合法\*\*的，不会导致 `\_metrics.is\_space\_group\_consistent` 报错。



\### 9.2 `P4/<unk>` 的来源



\- 在闭环 Round 1 生成的 CIF 中看到的行是：\\

&nbsp; `\_symmetry\_space\_group\_name\_H-M   P4/<unk>`。

\- 这行并非来自 LLaMA，而是来自 CrystaLLM 的采样：

&nbsp; - Round 1 是从初始 prompt 直接调用 `sample.py` 得到 CIF，此时尚未调用 LLaMA 优化器；

&nbsp; - CrystaLLM 的 tokenizer 内部有 `<unk>` 这样的特殊 token，采样时有概率采到它；

&nbsp; - 解码时，`<unk>` 被原样写成文本，形成 `P4/<unk>`，这在 CIF 语义上当然非法。



\### 9.3 为什么会导致清洗失败



\- `clean\_cif` 中：

&nbsp; - 提取 `space\_group\_symbol = extract\_space\_group\_symbol(cif\_str)` → `"P4/<unk>"`；

&nbsp; - `replace\_symmetry\_operators(cif\_str, space\_group\_symbol)` 会查表，预期只包含合法 H‑M 符号；

&nbsp; - 表里不存在 `"P4/<unk>"`，底层抛出 `ValueError("Bad international symbol P4/<unk>")`；

&nbsp; - 在 `clean\_and\_validate\_cif` 中被捕获成 `cleaning\_failed: Bad international symbol P4/<unk>`，并提前返回。



因此：

\- `P4/mmm` 本身没问题；

\- `P4/<unk>` 是 CrystaLLM 生成过程中的“坏 token”造成的清洗失败，而非 LLaMA 或 prompt 写法造成。



---



\## 10. 已集成与未集成的评估指标（与原始项目对比）



\### 10.1 已集成到闭环的指标



来自 `crystallm/\_metrics.py` + `cif\_cleaning.py`：



\- 键长合理性：`bond\_length\_reasonableness\_score` → `bond\_length\_score` 列；

\- 组成一致性：`is\_formula\_consistent` → `formula\_ok` 列；

\- 位点 multiplicity 一致性：`is\_atom\_site\_multiplicity\_consistent` → `atom\_site\_multiplicity\_ok` 列；

\- 空间群一致性：`is\_space\_group\_consistent` → `space\_group\_ok` 列；

\- 综合有效性：

&nbsp; - `\_metrics.is\_valid` 的逻辑已在 `validate\_cif` 中展开为组合条件；

&nbsp; - 闭环中通过 `validation\_ok`（或基线脚本中的 `valid`）体现。



\### 10.2 仅用于离线评估、尚未接入闭环的指标



位于 `bin/benchmark\_metrics.py` 等脚本中：



\- `smact\_validity`：SMAct 价态、电荷中和、电负性合理性检查；

\- `structure\_validity`：最近邻距离阈值 + 体积阈值的结构合理性；

\- `get\_unconditional\_metrics` / `compute\_cov` / `get\_match\_rate\_and\_rms`：

&nbsp; - 用于“无条件生成”的整体质量评估：COV、Wasserstein 距离、匹配率、RMSD 等。



当前 JSON 闭环不会在每轮在线计算这些指标，它们更适合作为“离线大样本基准评估”使用（对照原始论文/项目）。



如果后续需要，我们可以：

\- 在闭环结束后，对 `experiments/.../round\_XX/cifs/\*.cif` 运行 `bin/evaluate\_cifs.py` 或 `bin/benchmark\_metrics.py`，得到与原始项目完全一致的一整套指标；

\- 或在闭环中增加轻量版本，例如在每轮之后对 Top‑K 结构运行 `smact\_validity` / `structure\_validity`，作为额外的过滤条件或 LLM 输入特征。



---



\## 11. 关键脚本与文件一览



\- 采样与闭环：

&nbsp; - `bin/sample.py`：CrystaLLM 采样脚本（单次生成）。

&nbsp; - `bin/prompt\_optimization\_loop.py`：CrystaLLM + ALIGNN + LLaMA JSON 闭环调度脚本。

&nbsp; - `prompts/na2cl2\_round0.txt`：初始 CIF prompt（Na2Cl2 + P4/mmm）。



\- CIF 清洗与验证：

&nbsp; - `crystallm/\_metrics.py`：键长、空间群、配比、多重性等低层评估函数。

&nbsp; - `crystallm/cif\_cleaning.py`：封装成 `clean\_cif` + `validate\_cif` + `clean\_and\_validate\_cif`。

&nbsp; - `experiments/\*/round\_XX/cif\_quality.csv`：每轮 CIF 质量结果。



\- ALIGNN 集成：

&nbsp; - `resources/alignn\_zmq\_server\_multi.py`：多物性 ZMQ 评分服务（含多次 bug 修复）。

&nbsp; - `bin/alignn\_client.py`：ZMQ 客户端 + `score\_cifs\_via\_alignn` + `summarize\_scores`。

&nbsp; - `experiments/\*/round\_XX/scores.csv`：每轮 ALIGNN 打分。

&nbsp; - `scores\_sample\_base.csv` / `scores\_sample\_base\_summary.json`：base 采样 ALIGNN 基线结果。



\- LLaMA 优化器：

&nbsp; - `model/llama\_local\_api.py`：本地 Transformers + FastAPI 服务，提供 `/v1/chat/completions`。

&nbsp; - `bin/llama\_client.py`：严格 JSON 模式客户端，将 `last\_prompt\_lines` + `evaluator\_summary` 传给 LLaMA，并解析 JSON 输出。

&nbsp; - `experiments/\*/round\_XX/llama\_output.json`：记录每轮 LLaMA 输入输出与原始响应。



\- 原始评估工具：

&nbsp; - `bin/evaluate\_cifs.py`：原始项目用于评估生成 CIF 的脚本。

&nbsp; - `bin/benchmark\_metrics.py`：原始项目用于 mp20/carbon/perovskite 等基准数据集的整体指标计算。

&nbsp; - `docs/CRYSTALLM\_ALIGNN\_LLAMAT3\_JSON\_LOOP.md`：我们先前的集成方案总体设计文档。

&nbsp; - `docs/CHAT\_LOG\_2025-12-01\_CrystaLLM\_ALIGNN\_LLaMat3.md`：旧环境下的第一次长对话摘要。



---



\## 12. 后续改进建议（基于本次调试）



1\. \*\*增强清洗失败时的“尽量评估”能力\*\*：

&nbsp;  - 在 `clean\_and\_validate\_cif` 里，对于 `clean\_cif` 失败的 CIF，仍使用原始 `cif\_str` 尝试跑 `validate\_cif`，至少输出 `bond\_length\_score` / `formula\_ok` / `space\_group\_ok` / `atom\_site\_multiplicity\_ok` 等数值；

&nbsp;  - 避免 `bond\_length\_score` 一直为空，让你能更细致地分析“坏在哪一项”。



2\. \*\*对非法空间群如 `P4/<unk>` 做更温和的处理\*\*：

&nbsp;  - 检测到 `\_symmetry\_space\_group\_name\_H-M` 中含有尖括号或明显非法 token 时：

&nbsp;    - 要么直接退化为 `P 1`，只使用最简单的对称性；

&nbsp;    - 要么跳过 `replace\_symmetry\_operators` 步骤，仅基于 CIF 中已有的坐标做验证。



3\. \*\*在 prompt 和 LLaMA 输出约束中显式列出“合法空间群列表”\*\*：

&nbsp;  - 对 LLaMA system prompt 加一段“空间群必须从以下有限集合中选择：P4/mmm, Pm-3m, P-1, ...”，减少 LLaMA/CrystaLLM 生成怪符号的概率。



4\. \*\*引入 SMAct 检查作为更强的组成合理性指标（可选）\*\*：

&nbsp;  - 在闭环或 base 对比实验中，增加一列 SMAct 评估结果（通过/不通过），辅助判断“化学上是否合理”。



5\. \*\*针对当前任务（Na–Cl 系）精调 prompt\*\*：

&nbsp;  - 在 `prompts/na2cl2\_round0.txt` 中增加 `\_chemical\_formula\_sum 'Na2 Cl2'`，乃至一部分 `\_atom\_site\_\*` 行示例；

&nbsp;  - 并在闭环的 LLaMA system prompt 中强调“不允许改变元素集合和总配比”。



通过这次完整的调通与对比，我们已经确认：

\- 环境与脚本链路（CrystaLLM → CIF 清洗 → ALIGNN → LLaMA JSON → 再采样）总体是通的；

\- 当前主要瓶颈在于：base 模型在该 prompt 下的结构质量有限 + 清洗逻辑对非法空间群非常敏感；

\- 闭环机制已能在 GNN 层面推高带隙，但需要配合更强的 prompt / 规则约束，才能真正提高物理合理性。 





