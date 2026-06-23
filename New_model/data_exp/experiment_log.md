# MPTS-52 Scratch Data-Format Experiment

## Goal

Train full GPT-style crystal language models from scratch on MPTS-52 to answer whether the modified pretraining data format is effective.

## Protocol

- Output root: `/data/users/xsw/autodlmini/model/New_model/data_exp`
- Environment: `crystallm_env`
- Target dataset: MPTS-52
- Comparison:
  - `scratch_orig_cif_small`: original MPTS-52 CIF text, from scratch.
  - `scratch_symcif_v4_small`: modified SymCIF-v4 structured text, from scratch.
- Shared tokenizer: UTF-8 byte-level ids, vocab size 256.
- Shared model: 8 layers, 8 heads, 512 embedding, block 1024.
- Shared schedule: 3500 iterations, batch 32, grad accumulation 4, cosine LR 1e-3 to 1e-4.
- Test split policy: do not read test during training setup or model selection.

## Resource Snapshot

- Date: 2026-06-19
- Conda env: `crystallm_env`
- Python: 3.10.19
- Torch: 2.5.1+cu121
- CUDA available: yes
- GPU: 2 x NVIDIA A800 80GB PCIe; both idle at start.
- Memory: 251GiB total, 227GiB available at start.
- Disk: `/data/users/xsw/autodlmini` 1.8T total, 1.1T available at start.

## Data Build

- Script: `/data/users/xsw/autodlmini/model/New_model/data_exp/scripts/build_mpts52_scratch_data.py`
- Test split read: no.
- Original CIF source: `/data/users/xsw/autodlmini/model/CrystaLLM/resources/benchmarks/mpts_52/{train,val}.csv`
- SymCIF-v4 source: `/data/users/xsw/autodlmini/model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52/{train,val}.jsonl`
- Original CIF samples: train 27,380; val 5,000.
- SymCIF-v4 samples: train 25,998; val 4,727.
- SymCIF-v4 train rows>=7: 6,863 unique structured samples.
- SymCIF-v4 val rows>=7: 2,197 samples.
- Tokenizer: shared UTF-8 byte-level tokenizer, vocab size 256.
- Output summaries:
  - `/data/users/xsw/autodlmini/model/New_model/data_exp/data/data_build_summary.json`
  - `/data/users/xsw/autodlmini/model/New_model/data_exp/data/tokens_mpts52_orig_cif_byte/dataset_summary.json`
  - `/data/users/xsw/autodlmini/model/New_model/data_exp/data/tokens_mpts52_symcif_v4_byte/dataset_summary.json`

## Dryrun

- Original CIF command: `conda run -n crystallm_env python model/CrystaLLM/bin/train.py --config model/New_model/data_exp/configs/scratch_orig_cif_small.yaml out_dir=/data/users/xsw/autodlmini/model/New_model/data_exp/runs/dryrun_orig eval_only=true eval_iters_train=1 eval_iters_val=1 compile=false device=cuda:0`
- Original CIF result: scratch init successful; 25.31M parameters; step 0 train loss 5.3930, val loss 5.4027.
- SymCIF-v4 command: `conda run -n crystallm_env python model/CrystaLLM/bin/train.py --config model/New_model/data_exp/configs/scratch_symcif_v4_small.yaml out_dir=/data/users/xsw/autodlmini/model/New_model/data_exp/runs/dryrun_symcif eval_only=true eval_iters_train=1 eval_iters_val=1 compile=false device=cuda:1`
- SymCIF-v4 result: scratch init successful; 25.31M parameters; step 0 train loss 5.5322, val loss 5.5412.

## Training Launch

- Attempt 1 used `bash -lc` with shell redirection and failed before training because torch could not see CUDA: `RuntimeError: No CUDA GPUs are available`; a direct check inside `bash -lc` returned `torch.cuda.is_available=False` and device count 0.
- Direct `conda run -n crystallm_env ...` sees CUDA normally: `torch.cuda.is_available=True`, device count 2.
- Stable launch uses direct `conda run`, no shell wrapper, with `compile=false` for reliability.
- Final launch uses `run_train_with_log.py` and `python -u` so logs are persisted and unbuffered.
- Original CIF command: `conda run -n crystallm_env python model/New_model/data_exp/scripts/run_train_with_log.py --log model/New_model/data_exp/logs/scratch_orig_cif_small.train.log -- python -u model/CrystaLLM/bin/train.py --config model/New_model/data_exp/configs/scratch_orig_cif_small.yaml compile=false`
- SymCIF-v4 command: `conda run -n crystallm_env python model/New_model/data_exp/scripts/run_train_with_log.py --log model/New_model/data_exp/logs/scratch_symcif_v4_small.train.log -- python -u model/CrystaLLM/bin/train.py --config model/New_model/data_exp/configs/scratch_symcif_v4_small.yaml compile=false`
- Launch status: completed; both training runs exited with `rc=0`.
- Final GPU status after completion: no running GPU processes.

## Training Progress

- Step 250:
  - Original CIF: train loss 0.4151, val loss 0.4592; checkpoint saved at `/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_orig_cif_small/ckpt.pt`.
  - SymCIF-v4: train loss 0.3505, val loss 0.5432; checkpoint saved at `/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_symcif_v4_small/ckpt.pt`.
  - Early token-loss readout: original CIF is ahead on val loss at step 250; continue to full 3500 before drawing conclusions.
- Step 500:
  - Original CIF: train loss 0.2443, val loss 0.2802.
  - SymCIF-v4: train loss 0.2449, val loss 0.4044.
- Step 750:
  - Original CIF: train loss 0.1942, val loss 0.2279.
  - SymCIF-v4: train loss 0.1993, val loss 0.4020.
  - Interim readout: original CIF is still ahead on validation token loss; SymCIF-v4 train loss improves but validation loss plateaus early, suggesting stronger overfit or harder generalization under this simple byte-level full-model setup.
- Step 1000:
  - Original CIF: train loss 0.1845, val loss 0.2168.
  - SymCIF-v4: train loss 0.1880, val loss 0.3730.
- Step 1250:
  - Original CIF: train loss 0.1781, val loss 0.2127.
  - SymCIF-v4: train loss 0.1829, val loss 0.3616.
  - Interim readout: original CIF remains substantially better on held-out byte-token likelihood. SymCIF-v4 is improving but much slower, so generation/render checks are required before interpreting this as final data-format failure.
- Step 1500:
  - Original CIF: train loss 0.1752, val loss 0.2108.
  - SymCIF-v4: train loss 0.1780, val loss 0.3749.
- Step 1750:
  - Original CIF: train loss 0.1673, val loss 0.2063.
  - SymCIF-v4: train loss 0.1747, val loss 0.3742.
- Step 2000:
  - Original CIF: train loss 0.1644, val loss 0.2057.
  - SymCIF-v4: train loss 0.1725, val loss 0.3785.
- Step 2250:
  - Original CIF: train loss 0.1603, val loss 0.2040.
  - SymCIF-v4: train loss 0.1697, val loss 0.3784.
  - Interim readout: original CIF continues to improve slowly; SymCIF-v4 has plateaued around 0.37-0.38 validation loss despite lower train loss, indicating overfit or poor validation generalization for this text format/model/tokenizer combination.
- Step 2500:
  - Original CIF: train loss 0.1566, val loss 0.2014.
  - SymCIF-v4: train loss 0.1615, val loss 0.3795.
- Step 2750:
  - Original CIF: train loss 0.1536, val loss 0.2042.
  - SymCIF-v4: train loss 0.1620, val loss 0.3857.
- Step 3000:
  - Original CIF: train loss 0.1511, val loss 0.2032.
  - SymCIF-v4: train loss 0.1574, val loss 0.3835.
  - Interim readout: original CIF reaches its best validation loss so far around step 2500; SymCIF-v4 validation loss worsens after step 2500, so the current modified format is not winning on likelihood under this controlled scratch setup.
- Step 3250:
  - Original CIF: train loss 0.1495, val loss 0.2039.
  - SymCIF-v4: train loss 0.1528, val loss 0.3865.
- Step 3500:
  - Original CIF: train loss 0.1466, val loss 0.2054.
  - SymCIF-v4: train loss 0.1524, val loss 0.3931.

## Training Result

| run | best step | best train loss | best val loss | final step | final train loss | final val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original CIF scratch baseline | 2500 | 0.1566 | 0.2014 | 3500 | 0.1466 | 0.2054 |
| SymCIF-v4 scratch modified format | 1250 | 0.1829 | 0.3616 | 3500 | 0.1524 | 0.3931 |

- Original CIF is the stronger from-scratch baseline on held-out MPTS-52 validation token likelihood.
- SymCIF-v4 reaches its best validation loss much earlier, then gets worse while train loss keeps improving. This is an overfit/generalization warning for this representation under the same decoder-only GPT and byte-level tokenizer.
- Best validation gap: SymCIF-v4 best val loss 0.3616 vs Original CIF best val loss 0.2014. Lower is better, so the modified format is worse by 0.1602 loss in this controlled setup.
- Final validation gap: SymCIF-v4 final val loss 0.3931 vs Original CIF final val loss 0.2054.

## Quick Generation Sanity Check

- Script: `/data/users/xsw/autodlmini/model/New_model/data_exp/scripts/summarize_and_sample.py`
- Sampling policy: first 8 validation prompts for each format, 2 generations per prompt, 16 generations per model.
- Sampling settings: temperature 0.8, top_k 10, max_new_tokens 1536.
- Test split read: no.

| run | total generations | required/basic tags ok | required/basic tag rate | pymatgen parse ok | pymatgen parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original CIF scratch baseline | 16 | 16 | 100.0% | 10 | 62.5% |
| SymCIF-v4 scratch modified format | 16 | 16 | 100.0% | 0 | 0.0% |

- Original CIF generations preserve required CIF tags in this small sanity check and 10/16 can be parsed by pymatgen.
- SymCIF-v4 generations preserve the expected structured fields in the basic text check, but they are not direct CIFs, so pymatgen parse is not a fair final metric without a renderer/converter and full structural evaluation.

## Final Interpretation

This experiment does not support the claim that the current modified pretraining data format is more effective than original CIF for MPTS-52 when both are trained from scratch under the same GPT architecture, optimizer schedule, tokenizer, and train/validation-only protocol.

In plain terms: changing the text format alone did not help here. The original CIF text is much easier for this model/tokenizer setup to generalize on validation data. SymCIF-v4 learns the training text, but validation loss stops improving early and then worsens, which means the model is memorizing or fitting the format without gaining better held-out generalization.

Important caveat: this is a controlled scratch language-model and quick generation sanity experiment, not a full StructureMatcher benchmark on the MPTS-52 test split. The test split was intentionally not read. A final CSP-style benchmark would require a proper SymCIF-v4-to-CIF renderer and structure-level evaluation.

## Artifacts

- Data summary: `/data/users/xsw/autodlmini/model/New_model/data_exp/data/data_build_summary.json`
- Original CIF loss curve: `/data/users/xsw/autodlmini/model/New_model/data_exp/eval/orig_cif_loss_curve.csv`
- SymCIF-v4 loss curve: `/data/users/xsw/autodlmini/model/New_model/data_exp/eval/symcif_v4_loss_curve.csv`
- Quick sample summary: `/data/users/xsw/autodlmini/model/New_model/data_exp/eval/training_and_quick_sample_summary.json`
- Original CIF training log: `/data/users/xsw/autodlmini/model/New_model/data_exp/logs/scratch_orig_cif_small.train.log`
- SymCIF-v4 training log: `/data/users/xsw/autodlmini/model/New_model/data_exp/logs/scratch_symcif_v4_small.train.log`
- Original CIF final checkpoint: `/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_orig_cif_small/ckpt.pt`
- SymCIF-v4 final checkpoint: `/data/users/xsw/autodlmini/model/New_model/data_exp/runs/scratch_symcif_v4_small/ckpt.pt`

Note: `always_save_checkpoint=True` overwrote `ckpt.pt` at each evaluation point, so only the final checkpoints remain. Best validation steps are recorded in logs/CSVs, but separate best-step checkpoint files were not preserved.
