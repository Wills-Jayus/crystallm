# Demo：跑一组 3 轮闭环

## 0. 前置：服务都已启动

- ALIGNN ZMQ：参考 `CrystaLLM/docs/ENV_SETUP.md`（推荐用 `bin/alignn_zmq_server_ctl.sh start|stop` 管理）
- OpenAI 兼容 LLM API（例如 vLLM）：确保 `http://127.0.0.1:8000/v1/chat/completions` 可访问

## 1. 准备初始 prompt

示例（Na2Cl2 + P4/mmm）：

```bash
mkdir -p /root/autodl-tmp/model/CrystaLLM/prompts
cat > /root/autodl-tmp/model/CrystaLLM/prompts/na2cl2_round0.txt <<'EOF'
data_Na2Cl2
_symmetry_space_group_name_H-M   P4/mmm
EOF
```

## 2. 启动闭环

```bash
conda activate crystallm_env

python /root/autodl-tmp/model/CrystaLLM/bin/prompt_optimization_loop.py \
  --model-dir /root/autodl-tmp/model/CrystaLLM/crystallm_v1_small \
  --initial-prompt-file /root/autodl-tmp/model/CrystaLLM/prompts/na2cl2_round0.txt \
  --out-dir /root/autodl-tmp/model/CrystaLLM/experiments/na2cl2_qwen_vX_small \
  --rounds 3 \
  --samples-per-round 8 \
  --temperature 0.8 --top-k 10 --max-new-tokens 2048 \
  --alignn-host 127.0.0.1 --alignn-port 5555 \
  --alignn-properties formation_energy bandgap \
  --score-property bandgap --score-goal max \
  --top-structures 3 --final-top-k 5 \
  --qwen-api-base http://127.0.0.1:8000/v1 \
  --qwen-model Qwen3-30B-A3B-Instruct-2507 \
  --qwen-temperature 0.2 --qwen-max-tokens 256
```

### 2.1 只追 bandgap（关闭成分/计量检查）

```bash
python /root/autodl-tmp/model/CrystaLLM/bin/prompt_optimization_loop.py \
  ... \
  --no-validation-check-composition
```

## 3. 看一轮发生了什么

每轮目录：`experiments/<exp>/round_XX/`

- `prompt.txt`：该轮 prompt
- `cifs/`：生成的 CIF
- `cif_quality.json`：清洗/验证明细
- `scores.csv`：ALIGNN 预测 + 关键验证字段
- `evaluator_summary.json`：喂给 LLM 的压缩摘要
- `qwen_output.json`：LLM 输入/输出审计
