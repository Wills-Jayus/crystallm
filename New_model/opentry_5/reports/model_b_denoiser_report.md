# Model B Denoiser Report

Smoke E8003 completed at 2026-06-19T02:53:05Z.

This is a generative denoiser smoke model, not a scorer, selector, or reranker. It applies the same correction to every input and emits no ordered candidate pool.

```json
{
  "batch_size": 256,
  "candidate_order": "not_applicable_generative_denoiser_no_ranking",
  "created_at": "2026-06-19T02:53:05Z",
  "dev_samples": 1024,
  "device": "cuda:0",
  "environment": "crystallm_env",
  "epochs": 3,
  "experiment_id": "E8003",
  "final_dev": {
    "corrupt_lattice_mse": 0.00025193866167683154,
    "loss": -2.9981138706207275,
    "pred_lattice_mse": 1.0079321327793878e-05,
    "pred_vs_corrupt_mse_ratio": 0.04000704481284771,
    "rows_ge_7_pred_lattice_mse": 1.1352072713373143e-05,
    "rows_ge_7_samples": 459,
    "samples": 1024
  },
  "gate_passed": true,
  "history": [
    {
      "dev": {
        "corrupt_lattice_mse": 0.00025193866167683154,
        "loss": -2.952302932739258,
        "pred_lattice_mse": 0.00026272285322193056,
        "pred_vs_corrupt_mse_ratio": 1.042804829847561,
        "rows_ge_7_pred_lattice_mse": 0.0002576814871078483,
        "rows_ge_7_samples": 459,
        "samples": 1024
      },
      "epoch": 1,
      "train_aux": {
        "huber": 0.006280018365941942,
        "nll": -1.5232718982733786,
        "vpa": 0.00019376476143406762
      },
      "train_loss": -1.5216922333929688
    },
    {
      "dev": {
        "corrupt_lattice_mse": 0.00025193866167683154,
        "loss": -2.976783514022827,
        "pred_lattice_mse": 0.0001273074849450495,
        "pred_vs_corrupt_mse_ratio": 0.5053114281775071,
        "rows_ge_7_pred_lattice_mse": 0.00013405790710760876,
        "rows_ge_7_samples": 459,
        "samples": 1024
      },
      "epoch": 2,
      "train_aux": {
        "huber": 0.004231051221722737,
        "nll": -2.9622844010591507,
        "vpa": 0.0003049698457289196
      },
      "train_loss": -2.9612114131450653
    },
    {
      "dev": {
        "corrupt_lattice_mse": 0.00025193866167683154,
        "loss": -2.9981138706207275,
        "pred_lattice_mse": 1.0079321327793878e-05,
        "pred_vs_corrupt_mse_ratio": 0.04000704481284771,
        "rows_ge_7_pred_lattice_mse": 1.1352072713373143e-05,
        "rows_ge_7_samples": 459,
        "samples": 1024
      },
      "epoch": 3,
      "train_aux": {
        "huber": 0.0014113298238953575,
        "nll": -2.9884467869997025,
        "vpa": 0.00019241444397266605
      },
      "train_loss": -2.988084375858307
    }
  ],
  "parameter_count": 20622,
  "read_test": false,
  "read_val512": false,
  "seed": 8003,
  "train_samples": 4096,
  "train_time_seconds": 0.9828252792358398
}
```

Status: smoke only. Full row/free-param/site-mapping CIF generation and grouped-fold StructureMatcher evaluation are still pending.

## E8007 CIF Recovery Smoke (fold_a)

Fixed-order CIF recovery smoke completed at 2026-06-19T03:09:09Z. This diagnostic uses grouped-dev GT W/A rows and free params to test whether the trained lattice denoiser can connect to OrbitEngine and StructureMatcher. It is not a terminal generation result.

```json
{
  "candidate_order": "generation_index_fixed_seed_no_sorting",
  "checkpoint": "checkpoints/model_b_denoiser_smoke/best.pt",
  "created_at": "2026-06-19T03:09:09Z",
  "device": "cuda:0",
  "diagnostic_limitations": [
    "uses GT W/A rows and GT free params from grouped dev canonical labels",
    "does not prove formula+SG to W/A generation",
    "not a terminal opentry_5 success metric"
  ],
  "experiment_id": "E8007",
  "fold": "fold_a",
  "gate_pass": true,
  "generation_file": "eval/model_b_cif_recovery_smoke/fold_a_generations.jsonl",
  "k": 5,
  "metrics_file": "eval/model_b_cif_recovery_smoke/fold_a_metrics.json",
  "no_ranking": true,
  "parameter_count": 20622,
  "samples": 128,
  "seconds": 13.645779371261597,
  "seed": 8017,
  "summary": {
    "top1": {
      "RMSE@1": 4.700705064819051e-16,
      "atom_count_ok@1": 1.0,
      "composition_exact@1": 1.0,
      "match@1": 0.9375,
      "readable@1": 1.0,
      "rows_ge_7_match@1": 0.8813559322033898,
      "rows_ge_7_samples": 59,
      "samples": 128,
      "sg_ok@1": 0.9609375
    },
    "top5": {
      "RMSE@5": 3.5950750001639204e-16,
      "atom_count_ok@5": 1.0,
      "composition_exact@5": 1.0,
      "match@5": 0.9375,
      "readable@5": 1.0,
      "rows_ge_7_match@5": 0.8813559322033898,
      "rows_ge_7_samples": 59,
      "samples": 128,
      "sg_ok@5": 0.9609375
    }
  },
  "test_access": "none",
  "val512_access": "none"
}
```

## E8008 CIF Recovery Smoke (fold_b)

Fixed-order CIF recovery smoke completed at 2026-06-19T03:12:50Z. This diagnostic uses grouped-dev GT W/A rows and free params to test whether the trained lattice denoiser can connect to OrbitEngine and StructureMatcher. It is not a terminal generation result.

```json
{
  "candidate_order": "generation_index_fixed_seed_no_sorting",
  "checkpoint": "checkpoints/model_b_denoiser_smoke/best.pt",
  "created_at": "2026-06-19T03:12:50Z",
  "device": "cuda:0",
  "diagnostic_limitations": [
    "uses GT W/A rows and GT free params from grouped dev canonical labels",
    "does not prove formula+SG to W/A generation",
    "not a terminal opentry_5 success metric"
  ],
  "experiment_id": "E8008",
  "fold": "fold_b",
  "gate_pass": true,
  "generation_file": "eval/model_b_cif_recovery_smoke/fold_b_generations.jsonl",
  "k": 5,
  "metrics_file": "eval/model_b_cif_recovery_smoke/fold_b_metrics.json",
  "no_ranking": true,
  "parameter_count": 20622,
  "samples": 128,
  "seconds": 41.15805244445801,
  "seed": 8017,
  "summary": {
    "top1": {
      "RMSE@1": 1.7490605363664758e-10,
      "atom_count_ok@1": 1.0,
      "composition_exact@1": 1.0,
      "match@1": 0.9609375,
      "readable@1": 1.0,
      "rows_ge_7_match@1": 1.0,
      "rows_ge_7_samples": 20,
      "samples": 128,
      "sg_ok@1": 0.9921875
    },
    "top5": {
      "RMSE@5": 1.749059410727035e-10,
      "atom_count_ok@5": 1.0,
      "composition_exact@5": 1.0,
      "match@5": 0.9609375,
      "readable@5": 1.0,
      "rows_ge_7_match@5": 1.0,
      "rows_ge_7_samples": 20,
      "samples": 128,
      "sg_ok@5": 0.9921875
    }
  },
  "test_access": "none",
  "val512_access": "none"
}
```
