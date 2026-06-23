# Resource audit

- Created at: 2026-06-22T16:35:32+00:00
- Write root: `/data/users/xsw/autodlmini/model/New_model/opentry_9`
- CPU cores: 128
- CrystaLLM repo used for reproduction: `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM`
- crystallm_env python: `/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python`

## GPU
```
0, NVIDIA A800 80GB PCIe, 81920 MiB, 81153 MiB, 580.65.06
1, NVIDIA A800 80GB PCIe, 81920 MiB, 53767 MiB, 580.65.06
```

## Torch CUDA
```
{"torch": "2.5.1+cu121", "cuda_available": true, "cuda_version": "12.1", "device_count": 2, "devices": ["NVIDIA A800 80GB PCIe", "NVIDIA A800 80GB PCIe"]}
```

## Disk
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb        1.8T  762G  1.1T  43% /data
```

## Official CSV record counts

| split | records | sha256 | path |
|---|---:|---|---|
| mp_20_train | 27136 | `133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mp_20/train.csv` |
| mp_20_val | 9047 | `7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mp_20/val.csv` |
| mp_20_test | 9046 | `dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mp_20/test.csv` |
| mpts_52_train | 27380 | `db33cfc86530bbd4b020405989c5cd12130f447ad7b9258d29b3ff9c8a26d764` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/train.csv` |
| mpts_52_val | 5000 | `b5d9f7a8ff72f3a1f24d6bb97e11ad201be40d10be7b27924234da06af04baba` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/val.csv` |
| mpts_52_test | 8096 | `4d6e76c4726fb12fb2a7e5bb23594f9b82dbd9bf67b7bfd4866e9c74a8f49038` | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/resources/benchmarks/mpts_52/test.csv` |

## Key assets

| asset | exists | size | sha256/path |
|---|---:|---:|---|
| crystallm_repo | True | 4096 | `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM` |
| public_crystallm_repo | True | 4096 | `/data/users/xsw/autodlmini/model/CrystaLLM` |
| python | True | 17501760 | `/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python` |
| mp20_benchmark_ckpt | True | 310722362 | `8f9d7eed7c100ebe25a1e1c3e4d3e820781f18254cc5de91d79372bee2ddde19` |
| mpts52_benchmark_ckpt | True | 310722362 | `dfc454903872afda4dfda9affc28bc274625512e3c56cd3c76995fc12ca0ef5d` |
| opentry7_mp20_pure_ckpt | True | 310722917 | `/data/users/xsw/autodlmini/model/New_model/opentry_7/checkpoints/pure_crystallm_gt_sg_mp_20/ckpt.pt` |
| opentry7_mpts52_pure_ckpt | True | 310722917 | `/data/users/xsw/autodlmini/model/New_model/opentry_7/checkpoints/pure_crystallm_gt_sg_mpts_52/ckpt.pt` |
| opentry7_mp20_meta | True | 7543 | `51e06393bb1e68ab28ad00de68f1c763d1e01461e66010d7e2b2f5c0bb242160` |
| opentry7_mpts52_meta | True | 7543 | `51e06393bb1e68ab28ad00de68f1c763d1e01461e66010d7e2b2f5c0bb242160` |
