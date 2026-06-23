# opentry_6 Experiment Log

Initial resource inventory:

```json
{
  "cpu_count": 128,
  "cuda_available": true,
  "cuda_device_count": 2,
  "cwd": "/data/users/xsw/autodlmini",
  "disk": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sdb        1.8T  746G  1.1T  42% /data\n",
  "executable": "/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python",
  "memory_gb": 251.46,
  "nvidia_smi": "Fri Jun 19 19:55:51 2026       \n+-----------------------------------------------------------------------------------------+\n| NVIDIA-SMI 580.65.06              Driver Version: 580.65.06      CUDA Version: 13.0     |\n+-----------------------------------------+------------------------+----------------------+\n| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |\n| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |\n|                                         |                        |               MIG M. |\n|=========================================+========================+======================|\n|   0  NVIDIA A800 80GB PCIe          Off |   00000000:16:00.0 Off |                    0 |\n| N/A   77C    P0            279W /  300W |   13309MiB /  81920MiB |    100%      Default |\n|                                         |                        |             Disabled |\n+-----------------------------------------+------------------------+----------------------+\n|   1  NVIDIA A800 80GB PCIe          Off |   00000000:B8:00.0 Off |                    0 |\n| N/A   78C    P0            310W /  300W |   13309MiB /  81920MiB |    100%      Default |\n|                                         |                        |             Disabled |\n+-----------------------------------------+------------------------+----------------------+\n\n+-----------------------------------------------------------------------------------------+\n| Processes:                                                                              |\n|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |\n|        ID   ID                                                               Usage      |\n|=========================================================================================|\n|    0   N/A  N/A         3671896      C   python                                13300MiB |\n|    1   N/A  N/A         3671963      C   python                                13300MiB |\n+-----------------------------------------------------------------------------------------+\n",
  "python": "3.10.19",
  "time": "2026-06-19 19:55:51 UTC",
  "torch": "2.5.1+cu121"
}
```

Initial data stats:

```json
{
  "fold_a": {
    "count": 256,
    "row_bins": {
      "10-14": 13,
      "15+": 1,
      "7-9": 7,
      "<7": 235
    },
    "rows_ge7": 21,
    "unique_formula": 251,
    "unique_sg": 43,
    "unique_wyckoff_pattern": 108
  },
  "fold_b": {
    "count": 256,
    "row_bins": {
      "10-14": 12,
      "7-9": 12,
      "<7": 232
    },
    "rows_ge7": 24,
    "unique_formula": 252,
    "unique_sg": 42,
    "unique_wyckoff_pattern": 110
  },
  "opentry5_147_reference": "opentry_5 final reports cite 147 unique rows>=7 in the fold_b tuning branch; opentry_6 uses all clean train rows>=7 available without duplicating records.",
  "scale_note": "K=20/full fold was attempted but generation/rendering was too slow under current CPU/GPU contention; if num_candidates=10 or eval_limit=256 is set, it is the prompt-allowed minimum viable scale.",
  "train": {
    "count": 2029,
    "row_bins": {
      "10-14": 108,
      "7-9": 39,
      "<7": 1882
    },
    "rows_ge7": 147,
    "unique_formula": 1881,
    "unique_sg": 99,
    "unique_wyckoff_pattern": 674
  }
}
```
## Stage 1 / E9001: GT-W/A CrystalFormer-style geometry-only model

* 时间：2026-06-19 20:26:11 UTC
* 是否使用 crystallm_env：yes; executable=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python
* 读取文件：/data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_a_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_b_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/data/oof_wa_predictions_dev.jsonl; /data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl; /data/users/xsw/autodlmini/model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json
* 写入文件：checkpoints/stage1_gtwa_geometry/last.pt; eval/stage1_fold_a; eval/stage1_fold_b
* 是否写入 opentry_6 之外：no
* 是否读取 test：no
* 是否使用排序/筛选/打分：no
* candidate 顺序：candidate 0 deterministic decode; candidate 1-9 fixed seeds [9100, 9101, 9102, 9103, 9104, 9110, 9111, 9112, 9113, 9114]; invalid slots retained
* 数据范围：train=2029 all clean train records; train rows>=7=147; fold_a=256; fold_b=256
* 模型结构：CrystalFormer-style autoregressive orbit-conditioned geometry decoder; formula/SG/global lattice context; per-orbit W/A/element/multiplicity/site-sym/enumeration/free-mask and previous generated params; rows>=64 support
* 训练目标：Gaussian mixture lattice NLL + sin/cos mixture density circular free-parameter NLL + row-pair/separation auxiliary; fixed coordinates masked out
* 关键参数：{"batch_size": 64, "bond_timeout_seconds": 4.0, "coord_components": 4, "device": "cpu", "emb_dim": 128, "epochs_refiner": 6, "epochs_stage1": 6, "epochs_stage2": 8, "eval_batch_size": 128, "eval_limit": 256, "eval_workers": 16, "force": true, "hidden_dim": 256, "lattice_components": 4, "lr": 0.0002, "match_timeout_seconds": 8.0, "max_eval_sites": 300, "max_match_sites": 300, "num_candidates": 10, "num_workers": 2, "parse_timeout_seconds": 4.0, "refiner_steps": 4, "sample_timeout_seconds": 30.0, "seed": 20260619, "sg_timeout_seconds": 4.0, "stage1_info": {"best_val_loss": 4.2881351709365845, "history": [{"coord_circular_mixture_nll": 2.141232840716839, "coord_wrapped_mae": 0.2274117823690176, "epoch": 1, "lattice_nll": 3.5521774291992188, "lattice_normalized_mae": 0.800394469872117, "loss": 5.6942702159285545, "row_pair_loss": 0.016418173996498808, "seconds": 5.311201333999634, "separation_loss": 0.001947863886016421, "val_coord_circular_mixture_nll": 1.970334142446518, "val_coord_wrapped_mae": 0.22031931951642036, "val_lattice_nll": 3.286859691143036, "val_lattice_normalized_mae": 0.7268149554729462, "val_loss": 5.258134126663208, "val_row_pair_loss": 0.017969294916838408, "val_separation_loss": 0.0020854220492765307}, {"coord_circular_mixture_nll": 1.453350406140089, "coord_wrapped_mae": 0.20809312630444765, "epoch": 2, "lattice_nll": 2.157488491386175, "lattice_normalized_mae": 0.5427099280059338, "loss": 3.611748158931732, "row_pair_loss": 0.017334137024590746, "seconds": 12.178443431854248, "separation_loss": 0.0021289707292453386, "val_coord_circular_mixture_nll": 1.6858121156692505, "val_coord_wrapped_mae": 0.22039275243878365, "val_lattice_nll": 2.7415530681610107, "val_lattice_normalized_mae": 0.6604466885328293, "val_loss": 4.428250551223755, "val_row_pair_loss": 0.016901013907045126, "val_separation_loss": 0.0020114680228289217}, {"coord_circular_mixture_nll": 1.0268774777650833, "coord_wrapped_mae": 0.20463798521086574, "epoch": 3, "lattice_nll": 1.5906749591231346, "lattice_normalized_mae": 0.475362298078835, "loss": 2.618369035422802, "row_pair_loss": 0.015515547711402178, "seconds": 18.465647220611572, "separation_loss": 0.0020386908181535546, "val_coord_circular_mixture_nll": 1.645229309797287, "val_coord_wrapped_mae": 0.21965602785348892, "val_lattice_nll": 2.6420928239822388, "val_lattice_normalized_mae": 0.6285079568624496, "val_loss": 4.2881351709365845, "val_row_pair_loss": 0.015486629446968436, "val_separation_loss": 0.0019299883279018104}, {"coord_circular_mixture_nll": 0.7710435511544347, "coord_wrapped_mae": 0.20138244330883026, "epoch": 4, "lattice_nll": 1.3609427399933338, "lattice_normalized_mae": 0.4471025401726365, "loss": 2.1327843964099884, "row_pair_loss": 0.01516827053274028, "seconds": 24.525953769683838, "separation_loss": 0.0019854156707879156, "val_coord_circular_mixture_nll": 1.7649423480033875, "val_coord_wrapped_mae": 0.21583372727036476, "val_lattice_nll": 2.537780523300171, "val_lattice_normalized_mae": 0.6208563148975372, "val_loss": 4.303548574447632, "val_row_pair_loss": 0.015748169273138046, "val_separation_loss": 0.001923384756082669}, {"coord_circular_mixture_nll": 0.5560956192202866, "coord_wrapped_mae": 0.19840734638273716, "epoch": 5, "lattice_nll": 1.1002012509852648, "lattice_normalized_mae": 0.4278291007503867, "loss": 1.6571071203798056, "row_pair_loss": 0.015414555789902806, "seconds": 30.523948907852173, "separation_loss": 0.001976354036742123, "val_coord_circular_mixture_nll": 1.8758670091629028, "val_coord_wrapped_mae": 0.21522996574640274, "val_lattice_nll": 2.504801779985428, "val_lattice_normalized_mae": 0.6032229140400887, "val_loss": 4.381503403186798, "val_row_pair_loss": 0.015922934049740434, "val_separation_loss": 0.001924465614138171}, {"coord_circular_mixture_nll": 0.354425064928364, "coord_wrapped_mae": 0.19628888834267855, "epoch": 6, "lattice_nll": 0.8671585358679295, "lattice_normalized_mae": 0.4079712266102433, "loss": 1.2223528251051903, "row_pair_loss": 0.01461129033123143, "seconds": 36.71785855293274, "separation_loss": 0.001933263301907573, "val_coord_circular_mixture_nll": 2.039201498031616, "val_coord_wrapped_mae": 0.21093661338090897, "val_lattice_nll": 2.4851337671279907, "val_lattice_normalized_mae": 0.5847955867648125, "val_loss": 4.5251476764678955, "val_row_pair_loss": 0.015493119135499, "val_separation_loss": 0.0018927544297184795}], "sampler": {"type": "shuffle"}}, "valid_timeout_seconds": 4.0, "weight_decay": 0.0001}
* readable：NA
* composition exact：NA
* SG/Wyckoff legal：NA
* match@1：38.67%
* match@5：60.16%
* match@20：not_run
* match@50：not_run
* RMSE@1/5/20：0.1766 / 0.1722 / NA
* rows>=7 match@1/5/20/50：0.00% / 2.38% / not_run / not_run
* rows>=7 positive-any：350
* rows>=7 new positives：301
* W/A-hit match-fail：NA
* skeleton-hit match-fail：NA
* collision-like rate：NA
* fold reports：{"stage1_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage1_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}
* extra：{"determinism_check": {"a": {"generation_hash": "5aaa6155ab78ca8796f9c80481a109733bb4e3b5ab640cf68c151209ff2d5f89", "metrics_core_hash": "e5083727d655a695eab54283fce082f3a4989d1cd1864cc86081a370e1dfaca8", "report": {"num_records": 64, "num_rows_ge7": 9, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}, "b": {"generation_hash": "5aaa6155ab78ca8796f9c80481a109733bb4e3b5ab640cf68c151209ff2d5f89", "metrics_core_hash": "e5083727d655a695eab54283fce082f3a4989d1cd1864cc86081a370e1dfaca8", "report": {"num_records": 64, "num_rows_ge7": 9, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}, "generation_hash_identical": true, "metrics_core_hash_identical": true}}
* 结论：GT W/A geometry-only path trained and evaluated on both grouped folds with fixed candidate order.
* 失败原因：If rows>=7 match remains low, failure is continuous lattice/free-parameter geometry rather than W/A prediction.
* 下一步：Train complex-focused full unique rows>=7 variant without duplicating records.

## Stage 2 / E9002: Full unique complex rows>=7 data training without duplicate inflation

* 时间：2026-06-19 20:52:52 UTC
* 是否使用 crystallm_env：yes; executable=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python
* 读取文件：/data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_a_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_b_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/data/oof_wa_predictions_dev.jsonl; /data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl; /data/users/xsw/autodlmini/model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json
* 写入文件：checkpoints/stage2_complex_focused_geometry/last.pt; eval/stage2_fold_a; eval/stage2_fold_b
* 是否写入 opentry_6 之外：no
* 是否读取 test：no
* 是否使用排序/筛选/打分：no
* candidate 顺序：candidate 0 deterministic decode; candidate 1-9 fixed seeds [9100, 9101, 9102, 9103, 9104, 9110, 9111, 9112, 9113, 9114]; invalid slots retained
* 数据范围：unique rows>=7 train count=147; simple records retained=1882; sampler balanced batches without materializing duplicate JSONL records; stats={"count": 2029, "row_bins": {"10-14": 108, "7-9": 39, "<7": 1882}, "rows_ge7": 147, "unique_formula": 1881, "unique_sg": 99, "unique_wyckoff_pattern": 674}
* 模型结构：Same geometry-only CrystalFormer-style decoder as Stage 1, trained with complex-focused weighted sampler.
* 训练目标：Same lattice/coordinate multimodal NLL plus stronger row-pair and collision/separation losses.
* 关键参数：{"batch_size": 64, "bond_timeout_seconds": 4.0, "coord_components": 4, "device": "cpu", "emb_dim": 128, "epochs_refiner": 6, "epochs_stage1": 6, "epochs_stage2": 8, "eval_batch_size": 128, "eval_limit": 256, "eval_workers": 16, "force": true, "hidden_dim": 256, "lattice_components": 4, "lr": 0.0002, "match_timeout_seconds": 8.0, "max_eval_sites": 300, "max_match_sites": 300, "num_candidates": 10, "num_workers": 2, "parse_timeout_seconds": 4.0, "refiner_steps": 4, "sample_timeout_seconds": 30.0, "seed": 20260619, "sg_timeout_seconds": 4.0, "stage2_info": {"best_val_loss": 4.585621774196625, "history": [{"coord_circular_mixture_nll": 1.867164272814989, "coord_wrapped_mae": 0.22263616789132357, "epoch": 1, "lattice_nll": 3.90448147803545, "lattice_normalized_mae": 0.8372565843164921, "loss": 5.775735586881638, "row_pair_loss": 0.026570441492367536, "seconds": 4.321831703186035, "separation_loss": 0.0013034445864832378, "val_coord_circular_mixture_nll": 2.0077649652957916, "val_coord_wrapped_mae": 0.2246336117386818, "val_lattice_nll": 3.708621859550476, "val_lattice_normalized_mae": 0.783942773938179, "val_loss": 5.720357239246368, "val_row_pair_loss": 0.026124833151698112, "val_separation_loss": 0.0006485575722763315}, {"coord_circular_mixture_nll": 1.007747245952487, "coord_wrapped_mae": 0.19946759613230824, "epoch": 2, "lattice_nll": 2.150115165859461, "lattice_normalized_mae": 0.5419540368020535, "loss": 3.1612416803836823, "row_pair_loss": 0.02212781459093094, "seconds": 9.194933891296387, "separation_loss": 0.0007510312207159586, "val_coord_circular_mixture_nll": 1.880136877298355, "val_coord_wrapped_mae": 0.2209761179983616, "val_lattice_nll": 3.25285667181015, "val_lattice_normalized_mae": 0.7458029985427856, "val_loss": 5.137197852134705, "val_row_pair_loss": 0.027716272044926882, "val_separation_loss": 0.000584304565563798}, {"coord_circular_mixture_nll": 0.6325975148938596, "coord_wrapped_mae": 0.19381103804334998, "epoch": 3, "lattice_nll": 1.5840850230306387, "lattice_normalized_mae": 0.49530244898051023, "loss": 2.2199793867766857, "row_pair_loss": 0.02158934104954824, "seconds": 13.809738397598267, "separation_loss": 0.0007303707534447312, "val_coord_circular_mixture_nll": 1.8845009505748749, "val_coord_wrapped_mae": 0.2167668491601944, "val_lattice_nll": 3.0497142672538757, "val_lattice_normalized_mae": 0.7296483516693115, "val_loss": 4.938703119754791, "val_row_pair_loss": 0.029597877524793148, "val_separation_loss": 0.0006031483208062127}, {"coord_circular_mixture_nll": 0.36859419371467084, "coord_wrapped_mae": 0.19025960797443986, "epoch": 4, "lattice_nll": 1.239961614832282, "lattice_normalized_mae": 0.4633810482919216, "loss": 1.6118437852710485, "row_pair_loss": 0.02159341477090493, "seconds": 17.994786024093628, "separation_loss": 0.0006121561727923108, "val_coord_circular_mixture_nll": 1.8431997299194336, "val_coord_wrapped_mae": 0.21341155841946602, "val_lattice_nll": 2.8663942515850067, "val_lattice_normalized_mae": 0.7036041766405106, "val_loss": 4.713663697242737, "val_row_pair_loss": 0.026860168669372797, "val_separation_loss": 0.0005064192591817118}, {"coord_circular_mixture_nll": 0.14023052417906, "coord_wrapped_mae": 0.18371201492846012, "epoch": 5, "lattice_nll": 1.077656589448452, "lattice_normalized_mae": 0.4463547607883811, "loss": 1.2209886405616999, "row_pair_loss": 0.020380948903039098, "seconds": 22.345638275146484, "separation_loss": 0.0005547438231587876, "val_coord_circular_mixture_nll": 1.8446591198444366, "val_coord_wrapped_mae": 0.21327577531337738, "val_lattice_nll": 2.8029850125312805, "val_lattice_normalized_mae": 0.696391299366951, "val_loss": 4.651379108428955, "val_row_pair_loss": 0.024642214179039, "val_separation_loss": 0.00048450048780068755}, {"coord_circular_mixture_nll": -0.03146815224317834, "coord_wrapped_mae": 0.1829170100390911, "epoch": 6, "lattice_nll": 0.9708486860617995, "lattice_normalized_mae": 0.44532649125903845, "loss": 0.942493706708774, "row_pair_loss": 0.020475781086133793, "seconds": 26.78250813484192, "separation_loss": 0.0005225404083830654, "val_coord_circular_mixture_nll": 1.9312079846858978, "val_coord_wrapped_mae": 0.211483646184206, "val_lattice_nll": 2.7570163309574127, "val_lattice_normalized_mae": 0.6847786456346512, "val_loss": 4.692206859588623, "val_row_pair_loss": 0.026278408709913492, "val_separation_loss": 0.0005111193750053644}, {"coord_circular_mixture_nll": -0.2532361716112064, "coord_wrapped_mae": 0.1774445571936667, "epoch": 7, "lattice_nll": 0.6029993335250765, "lattice_normalized_mae": 0.38647200958803296, "loss": 0.35273614968173206, "row_pair_loss": 0.019544461887562647, "seconds": 31.242120027542114, "separation_loss": 0.0005164798976693419, "val_coord_circular_mixture_nll": 1.8544485569000244, "val_coord_wrapped_mae": 0.21230848506093025, "val_lattice_nll": 2.7270694375038147, "val_lattice_normalized_mae": 0.671860858798027, "val_loss": 4.585621774196625, "val_row_pair_loss": 0.02708289446309209, "val_separation_loss": 0.0005171080483705737}, {"coord_circular_mixture_nll": -0.3168626559781842, "coord_wrapped_mae": 0.17834135657176375, "epoch": 8, "lattice_nll": 0.5542679158970714, "lattice_normalized_mae": 0.389869200065732, "loss": 0.24051273573422804, "row_pair_loss": 0.020437909610336646, "seconds": 35.801177740097046, "separation_loss": 0.0005223159669185407, "val_coord_circular_mixture_nll": 1.9767065048217773, "val_coord_wrapped_mae": 0.21517693251371384, "val_lattice_nll": 2.7591455280780792, "val_lattice_normalized_mae": 0.6619439721107483, "val_loss": 4.73978054523468, "val_row_pair_loss": 0.025929066818207502, "val_separation_loss": 0.0004891879725619219}], "sampler": {"complex_weight": 4.0, "num_samples_per_epoch": 2029, "replacement": true, "simple_weight": 1.0, "type": "WeightedRandomSampler"}}, "unique_rows_ge7_train_count": 147, "valid_timeout_seconds": 4.0, "weight_decay": 0.0001}
* readable：NA
* composition exact：NA
* SG/Wyckoff legal：NA
* match@1：35.55%
* match@5：55.66%
* match@20：not_run
* match@50：not_run
* RMSE@1/5/20：0.1598 / 0.1666 / NA
* rows>=7 match@1/5/20/50：0.00% / 0.00% / not_run / not_run
* rows>=7 positive-any：330
* rows>=7 new positives：283
* W/A-hit match-fail：NA
* skeleton-hit match-fail：NA
* collision-like rate：NA
* fold reports：{"stage2_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage2_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}
* extra：{"complex_unique_requirement": ">=1000 not reached"}
* 结论：Complex-focused dataset and sampler executed on both folds.
* 失败原因：If rows>=7 remains 0, likely causes are insufficient true unique rows>=7 coverage in opentry_5 clean train, loss/model underfit, or continuous geometry ambiguity.
* 下一步：Train fixed-step symmetry-space refiner and compare raw vs refined.

## Stage 3 / E9003: Fixed-step symmetry-space learned geometry refiner

* 时间：2026-06-19 21:47:11 UTC
* 是否使用 crystallm_env：yes; executable=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python
* 读取文件：/data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_a_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_b_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/data/oof_wa_predictions_dev.jsonl; /data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl; /data/users/xsw/autodlmini/model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json
* 写入文件：checkpoints/stage3_refiner/last.pt; eval/stage3_refined_fold_a; eval/stage3_refined_fold_b
* 是否写入 opentry_6 之外：no
* 是否读取 test：no
* 是否使用排序/筛选/打分：no
* candidate 顺序：candidate 0 deterministic decode; candidate 1-9 fixed seeds [9100, 9101, 9102, 9103, 9104, 9110, 9111, 9112, 9113, 9114]; invalid slots retained
* 数据范围：synthetic corruption over stage2_train=2029; opentry_4 hard-negative stats={"exists": true, "path": "/data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl", "rows_ge7_wa_hit_match_fail": 590, "total": 11904, "usable_parameter_payload": false, "wa_hit_match_fail": 1061}
* 模型结构：Symmetry parameter-space GRU refiner; formula/SG/W/A fixed; fixed 4 steps; no quality gating.
* 训练目标：Periodic coordinate loss + lattice metric/VPA loss + row-pair distance consistency + collision/short-contact penalty; hard-negative train file used as failure-mode audit because no reusable parameter payload was present.
* 关键参数：{"batch_size": 64, "bond_timeout_seconds": 4.0, "coord_components": 4, "device": "cpu", "emb_dim": 128, "epochs_refiner": 6, "epochs_stage1": 6, "epochs_stage2": 8, "eval_batch_size": 128, "eval_limit": 256, "eval_workers": 16, "force": true, "hidden_dim": 256, "lattice_components": 4, "lr": 0.0002, "match_timeout_seconds": 8.0, "max_eval_sites": 300, "max_match_sites": 300, "num_candidates": 10, "num_workers": 2, "parse_timeout_seconds": 4.0, "refiner_info": {"best_val_loss": 0.08207063563168049, "hard_negative_stats": {"exists": true, "path": "/data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl", "rows_ge7_wa_hit_match_fail": 590, "total": 11904, "usable_parameter_payload": false, "wa_hit_match_fail": 1061}, "history": [{"collision_short_contact_penalty": 0.0015420063500641845, "epoch": 1, "lattice_metric_vpa_loss": 0.06994204747024924, "loss": 0.10923478938639164, "periodic_coordinate_loss": 0.03683236229699105, "row_pair_distance_loss": 0.009224711480783299, "val_collision_short_contact_penalty": 0.0016647927404846996, "val_lattice_metric_vpa_loss": 0.06147792097181082, "val_loss": 0.09243299253284931, "val_periodic_coordinate_loss": 0.02861191937699914, "val_row_pair_distance_loss": 0.008706684922799468}, {"collision_short_contact_penalty": 0.0016145341214723885, "epoch": 2, "lattice_metric_vpa_loss": 0.058882324723526835, "loss": 0.09131994494237006, "periodic_coordinate_loss": 0.030098152812570333, "row_pair_distance_loss": 0.008712059760000557, "val_collision_short_contact_penalty": 0.0016753342933952808, "val_lattice_metric_vpa_loss": 0.0586598040536046, "val_loss": 0.09176703914999962, "val_periodic_coordinate_loss": 0.030734986532479525, "val_row_pair_distance_loss": 0.00881885550916195}, {"collision_short_contact_penalty": 0.0015683765341236722, "epoch": 3, "lattice_metric_vpa_loss": 0.054046836332418025, "loss": 0.08445795555599034, "periodic_coordinate_loss": 0.027992038405500352, "row_pair_distance_loss": 0.00904897291911766, "val_collision_short_contact_penalty": 0.001692144083790481, "val_lattice_metric_vpa_loss": 0.05497030261904001, "val_loss": 0.08585701510310173, "val_periodic_coordinate_loss": 0.028506875969469547, "val_row_pair_distance_loss": 0.008842490147799253}, {"collision_short_contact_penalty": 0.0016192109032999724, "epoch": 4, "lattice_metric_vpa_loss": 0.05028306238818914, "loss": 0.08137019025161862, "periodic_coordinate_loss": 0.028761998983100057, "row_pair_distance_loss": 0.00865283198072575, "val_collision_short_contact_penalty": 0.0016839548479765654, "val_lattice_metric_vpa_loss": 0.05184635799378157, "val_loss": 0.08207063563168049, "val_periodic_coordinate_loss": 0.027967853471636772, "val_row_pair_distance_loss": 0.008352130185812712}, {"collision_short_contact_penalty": 0.0015021701328805648, "epoch": 5, "lattice_metric_vpa_loss": 0.04445637797471136, "loss": 0.07667209825012833, "periodic_coordinate_loss": 0.02970478485804051, "row_pair_distance_loss": 0.009442873793886974, "val_collision_short_contact_penalty": 0.0016484513762407005, "val_lattice_metric_vpa_loss": 0.051561363972723484, "val_loss": 0.084807513281703, "val_periodic_coordinate_loss": 0.03095938917249441, "val_row_pair_distance_loss": 0.008487645885907114}, {"collision_short_contact_penalty": 0.0015395952923427103, "epoch": 6, "lattice_metric_vpa_loss": 0.043391471554059535, "loss": 0.0741370317991823, "periodic_coordinate_loss": 0.028383436787407845, "row_pair_distance_loss": 0.008832656385493465, "val_collision_short_contact_penalty": 0.001672171609243378, "val_lattice_metric_vpa_loss": 0.050720395520329475, "val_loss": 0.08316179551184177, "val_periodic_coordinate_loss": 0.030146329198032618, "val_row_pair_distance_loss": 0.008511410793289542}]}, "refiner_steps": 4, "sample_timeout_seconds": 30.0, "seed": 20260619, "sg_timeout_seconds": 4.0, "valid_timeout_seconds": 4.0, "weight_decay": 0.0001}
* readable：NA
* composition exact：NA
* SG/Wyckoff legal：NA
* match@1：36.91%
* match@5：58.79%
* match@20：not_run
* match@50：not_run
* RMSE@1/5/20：0.1934 / 0.1705 / NA
* rows>=7 match@1/5/20/50：0.00% / 0.00% / not_run / not_run
* rows>=7 positive-any：337
* rows>=7 new positives：288
* W/A-hit match-fail：NA
* skeleton-hit match-fail：NA
* collision-like rate：NA
* fold reports：{"stage3_refined_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage3_refined_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}
* extra：{"raw_reference_reports": {"stage2_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage2_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}}
* 结论：Refiner trained and attached to Stage 2 raw generations on both folds.
* 失败原因：If refined does not improve raw Stage 2, learned correction is underfit or initial generated geometry is outside the synthetic corruption manifold.
* 下一步：Run GT/OOF/exact-cover W/A condition-gap comparison using the same geometry model and fixed-step refiner.

## Stage 4 / E9004: GT / OOF / exact-cover W/A condition-gap full system

* 时间：2026-06-20 00:26:40 UTC
* 是否使用 crystallm_env：yes; executable=/data/users/xsw/autodlmini/miniforge3/envs/crystallm_env/bin/python
* 读取文件：/data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_a_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/checkpoints/minicfjoint_v2_fold_b_gate0_smoke/clean_data; /data/users/xsw/autodlmini/model/New_model/opentry_5/data/oof_wa_predictions_dev.jsonl; /data/users/xsw/autodlmini/model/New_model/opentry_4/cache/hard_negative_dataset_train.jsonl; /data/users/xsw/autodlmini/model/New_model/symcif_experiment/artifacts/wyckoff_lookup_full.json; /data/users/xsw/autodlmini/model/New_model/symcif_experiment/reports/composition_exact_v1
* 写入文件：eval/stage4_A_gtwa_fold_a; eval/stage4_B_oofwa_fold_a; eval/stage4_C_exactcover_fold_a; eval/stage4_A_gtwa_fold_b; eval/stage4_B_oofwa_fold_b; eval/stage4_C_exactcover_fold_b
* 是否写入 opentry_6 之外：no
* 是否读取 test：no
* 是否使用排序/筛选/打分：no
* candidate 顺序：candidate 0 deterministic decode; candidate 1-9 fixed seeds [9100, 9101, 9102, 9103, 9104, 9110, 9111, 9112, 9113, 9114]; invalid slots retained
* 数据范围：same fold_a=256 and fold_b=256 samples for A/B/C; exact-cover matched opentry5 samples=0, fallback to OOF when unavailable.
* 模型结构：Stage 2 geometry generator plus Stage 3 fixed-step refiner; W/A condition varies only by GT, OOF, or exact-cover/fallback.
* 训练目标：No additional training; fixed-order inference only.
* 关键参数：{"batch_size": 64, "bond_timeout_seconds": 4.0, "coord_components": 4, "device": "cpu", "emb_dim": 128, "epochs_refiner": 6, "epochs_stage1": 6, "epochs_stage2": 8, "eval_batch_size": 128, "eval_limit": 256, "eval_workers": 16, "exact_cover_stats": {"files_read": ["/data/users/xsw/autodlmini/model/New_model/symcif_experiment/reports/composition_exact_v1/val_wa_candidates.jsonl", "/data/users/xsw/autodlmini/model/New_model/symcif_experiment/reports/composition_exact_v1/train_wa_candidates.jsonl", "/data/users/xsw/autodlmini/model/New_model/symcif_experiment/reports/composition_exact_v1_trimmed/val_wa_candidates.jsonl", "/data/users/xsw/autodlmini/model/New_model/symcif_experiment/reports/composition_exact_v1_trimmed/train_wa_candidates.jsonl"], "matched_opentry5_samples": 0, "samples_with_candidate": 5500}, "force": true, "hidden_dim": 256, "lattice_components": 4, "lr": 0.0002, "match_timeout_seconds": 8.0, "max_eval_sites": 300, "max_match_sites": 300, "num_candidates": 10, "num_workers": 2, "parse_timeout_seconds": 4.0, "refiner_steps": 4, "sample_timeout_seconds": 30.0, "seed": 20260619, "sg_timeout_seconds": 4.0, "valid_timeout_seconds": 4.0, "weight_decay": 0.0001}
* readable：NA
* composition exact：NA
* SG/Wyckoff legal：NA
* match@1：12.43%
* match@5：20.25%
* match@20：not_run
* match@50：not_run
* RMSE@1/5/20：0.1300 / 0.1883 / NA
* rows>=7 match@1/5/20/50：0.00% / 0.00% / not_run / not_run
* rows>=7 positive-any：338
* rows>=7 new positives：289
* W/A-hit match-fail：NA
* skeleton-hit match-fail：NA
* collision-like rate：NA
* fold reports：{"stage4_A_gtwa_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage4_A_gtwa_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage4_B_oofwa_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage4_B_oofwa_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage4_C_exactcover_fold_a": {"num_records": 256, "num_rows_ge7": 21, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}, "stage4_C_exactcover_fold_b": {"num_records": 256, "num_rows_ge7": 24, "rows_ge7_top20_match": null, "rows_ge7_top20_rmse": null, "top20_match": null}}
* extra：{"A_B_C_keys": ["stage4_A_gtwa_fold_a", "stage4_A_gtwa_fold_b", "stage4_B_oofwa_fold_a", "stage4_B_oofwa_fold_b", "stage4_C_exactcover_fold_a", "stage4_C_exactcover_fold_b"]}
* 结论：A/B/C condition comparison completed. If A succeeds and B/C fail, W/A gap dominates; if A also fails, continuous geometry remains the main bottleneck.
* 失败原因：Exact-cover files did not necessarily share sample_id namespace with opentry_5 folds; missing samples used OOF fallback and are marked per candidate.
* 下一步：Use the condition-gap result to choose either geometry data/model scaling or W/A condition improvement, not ranking-based repair.

