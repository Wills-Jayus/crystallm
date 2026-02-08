# 环境准备（CrystaLLM 闭环）

本项目的“闭环”通常拆成 2~3 个进程/环境：

1. **CrystaLLM 生成 + CIF 清洗/验证（pymatgen）**  
   - 建议环境：`crystallm_env`  
   - 关键依赖：`pymatgen`, `spglib`（用于空间群/对称性相关诊断与恢复）

2. **ALIGNN ZMQ 评分服务（jarvis + alignn + pyzmq）**  
   - 建议环境：`alignn-score`  
   - 关键依赖：`pyzmq`, `jarvis-tools`, `alignn`
   - 服务脚本：`CrystaLLM/resources/alignn_zmq_server_multi.py`

3. **优化器 LLM（OpenAI 兼容 Chat Completions）**  
   - 你当前主要用：`qwen3_vllm`（vLLM OpenAI server）  
   - 或使用本仓库根目录的 `qwen_local_api.py`（若依赖齐全；兼容旧入口 `llama_local_api.py`）

## 建议的目录位置

- 仓库根：`/root/autodl-tmp/model`
- CrystaLLM 代码：`/root/autodl-tmp/model/CrystaLLM`
- 日志/记录：`/root/autodl-tmp/model/log`

## 采样权重（ckpt.pt）下载与放置（必做）

闭环会调用 `CrystaLLM/bin/sample.py`，它要求 `--model-dir` 指向一个包含 `ckpt.pt` 的目录。

官方权重在 `CrystaLLM/ARTIFACTS.md`（Zenodo）里，示例下载 `crystallm_v1_small`：

```bash
cd /root/autodl-tmp/model/CrystaLLM
python bin/download.py crystallm_v1_small.tar.gz
tar xvf crystallm_v1_small.tar.gz
ls -la crystallm_v1_small/ckpt.pt
```

注意：权重较大且不应提交到 Git；本仓库的 `.gitignore`/钩子默认会阻止提交 `ckpt.pt`。

## 依赖服务启动示例

### 1) 启动 ALIGNN ZMQ 服务（alignn-score 环境）

```bash
conda activate alignn-score
export DGL_DISABLE_GRAPHBOLT=1

# 推荐：用控制脚本启动/关闭（会写 pidfile 与日志）
cd /root/autodl-tmp/model/CrystaLLM
bash bin/alignn_zmq_server_ctl.sh start --host 0.0.0.0 --port 5555 --properties "formation_energy bandgap"

# 查看状态：
# bash bin/alignn_zmq_server_ctl.sh status
#
# 暂停/恢复（进程级别，不退出）：
# bash bin/alignn_zmq_server_ctl.sh pause
# bash bin/alignn_zmq_server_ctl.sh resume
#
# 关闭：
# bash bin/alignn_zmq_server_ctl.sh stop
```

### 2) 启动 Qwen3 vLLM OpenAI API（按你机器上的脚本为准）

```bash
cd /root/autodl-tmp/qwen3_vllm
bash qwen3_vllmctl.sh start
```

确认 API 可用（示例）：

```bash
curl http://127.0.0.1:8000/v1/models
```
