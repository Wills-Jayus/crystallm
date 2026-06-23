# Baseline provenance

- Created at: 2026-06-22T16:35:32+00:00
- Historical GT-SG test provenance was traced through:
  - `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/mp20_test_data_atomtype_gt_sg_k20_20260531`
  - `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20`
  - `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/reports/crystallm_gt_sg_mp20_mpts52_report.json`
  - `/data/users/xsw/autodlmini/model/New_model/opentry_7/configs/crystallm_a_baselines.json`

## Recovered generation protocol

- Task: CrystaLLM CSP generation with original data/atom_type prompt plus GT space group
- Prompt format: data_<Composition.formula> + CrystaLLM atom_type block + _symmetry_space_group_name_H-M <GT>
- Generation script: `bin/generate_cifs_from_prompts_dir.py`
- Postprocess script: `bin/postprocess.py`
- Metrics script: `bin/benchmark_metrics.py`
- Temperature: 0.8
- Top-k: 10
- Max new tokens: 2048
- Candidate budget: 20

## MP-20 generation command

```bash
/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python /data/users/xsw/autodlmini/model/scp_task/CrystaLLM/bin/generate_cifs_from_prompts_dir.py --model-dir /data/users/xsw/autodlmini/model/scp_task/CrystaLLM/crystallm_benchmarkmodel/cif_model_mp_20_b --prompts-dir /data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/mp20_test_data_atomtype_gt_sg_k20_20260531/prompts/data_atomtype_gt_sg --out-dir /data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/crystallm_gt_sg_csp_test_20260531/mp20_test_data_atomtype_gt_sg_k20_20260531/cifs_raw/data_atomtype_gt_sg --num-samples-per-prompt 20 --sample-seed-stride 100000 --seed 1337 --temperature 0.8 --top-k 10 --max-new-tokens 2048 --device cuda:auto --dtype bfloat16 --workers 4 --batch-samples --retry-missing-single-worker
```

## MPTS-52 note

The historical report states that MPTS-52 reused a verified full K20 GT-SG run from `mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20`.

## Tokenizer

The scp_task/CrystaLLM benchmark generator constructs CIFTokenizer() from package code/resources. It does not load a per-run meta.pkl. opentry_7 pure-model tokenization artifacts are still recorded in resource_audit for separate pure-model provenance.
