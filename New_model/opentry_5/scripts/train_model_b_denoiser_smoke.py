#!/usr/bin/env python3
"""Train a small generative lattice denoiser smoke model for Model B."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("WORKDIR", str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache/transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache/xdg"))
os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
os.environ.setdefault("WANDB_DIR", str(ROOT / "logs/wandb"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "cache/cuda"))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


CORRUPTION_TYPES = [
    "free_param_or_site_mapping_noise",
    "collision_or_short_contact_noise",
    "lattice_vpa_noise",
    "inter_row_distance_noise",
]


def lattice_vector(lattice: dict | None) -> list[float] | None:
    if not lattice:
        return None
    vals = [lattice.get(k) for k in ["a", "b", "c", "alpha", "beta", "gamma"]]
    if any(v is None for v in vals):
        return None
    a, b, c, alpha, beta, gamma = [float(v) for v in vals]
    if min(a, b, c) <= 0:
        return None
    return [math.log(a), math.log(b), math.log(c), alpha / 180.0, beta / 180.0, gamma / 180.0]


def vpa_from_lattice(lattice: dict | None, atom_count: float | None) -> float | None:
    if not lattice or not atom_count or atom_count <= 0:
        return None
    volume = lattice.get("volume")
    if volume is None:
        return None
    return math.log(max(float(volume) / float(atom_count), 1e-6))


class DenoiseDataset(Dataset):
    def __init__(self, path: Path, max_rows: int = 0):
        self.rows = []
        for row in read_jsonl(path):
            if row.get("source_type") != "synthetic_train_dev_clean_corruption":
                continue
            target_lat = row.get("target_lattice")
            corrupt_lat = (row.get("corrupted_input") or {}).get("lattice")
            target = lattice_vector(target_lat)
            corrupt = lattice_vector(corrupt_lat)
            if target is None or corrupt is None:
                continue
            atom_count = None
            formula_counts = row.get("formula_counts")
            if isinstance(formula_counts, dict):
                atom_count = sum(float(v) for v in formula_counts.values())
            vpa_t = vpa_from_lattice(target_lat, atom_count)
            vpa_c = vpa_from_lattice(corrupt_lat, atom_count)
            if vpa_t is None:
                vpa_t = 0.0
            if vpa_c is None:
                vpa_c = 0.0
            type_vec = [1.0 if row.get("corruption_type") == t else 0.0 for t in CORRUPTION_TYPES]
            features = (
                corrupt
                + [float(row.get("sg") or 0) / 230.0, float(row.get("row_count") or 0) / 60.0, 1.0 if row.get("rows_ge_7") else 0.0, float(row.get("corruption_strength") or 0.0), vpa_c]
                + type_vec
            )
            self.rows.append(
                {
                    "x": np.asarray(features, dtype=np.float32),
                    "y": np.asarray(target + [vpa_t], dtype=np.float32),
                    "corrupt": np.asarray(corrupt + [vpa_c], dtype=np.float32),
                    "sample_id": row.get("sample_id"),
                    "rows_ge_7": bool(row.get("rows_ge_7")),
                }
            )
            if max_rows and len(self.rows) >= max_rows:
                break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        return torch.from_numpy(row["x"]), torch.from_numpy(row["y"]), torch.from_numpy(row["corrupt"]), row["rows_ge_7"]


class LatticeDenoiser(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 14),
        )

    def forward(self, x):
        out = self.net(x)
        base = torch.cat([x[:, :6], x[:, 10:11]], dim=1)
        mu = base + 0.05 * out[:, :7]
        logvar = out[:, 7:].clamp(-6.0, 3.0)
        return mu, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(mu, logvar, target):
    inv_var = torch.exp(-logvar)
    nll = 0.5 * ((target - mu) ** 2 * inv_var + logvar).mean()
    huber = nn.functional.smooth_l1_loss(mu, target, beta=0.02)
    length_order = nn.functional.relu(-torch.diff(mu[:, :3], dim=1).abs() + 1e-4).mean()
    angle_bounds = (nn.functional.relu(mu[:, 3:6] - 1.0) + nn.functional.relu(-mu[:, 3:6])).mean()
    vpa = nn.functional.smooth_l1_loss(mu[:, 6], target[:, 6], beta=0.02)
    return nll + 0.25 * huber + 0.05 * vpa + 0.01 * length_order + 0.01 * angle_bounds, {
        "nll": float(nll.detach().cpu()),
        "huber": float(huber.detach().cpu()),
        "vpa": float(vpa.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    loss_total = 0.0
    corrupt_mse = 0.0
    pred_mse = 0.0
    rows7_total = 0
    rows7_pred_mse = 0.0
    for x, y, corrupt, rows7 in loader:
        x = x.to(device)
        y = y.to(device)
        corrupt = corrupt.to(device)
        mu, logvar = model(x)
        loss, _ = loss_fn(mu, logvar, y)
        n = x.shape[0]
        total += n
        loss_total += float(loss.detach().cpu()) * n
        pred_err = ((mu[:, :6] - y[:, :6]) ** 2).mean(dim=1)
        corrupt_err = ((corrupt[:, :6] - y[:, :6]) ** 2).mean(dim=1)
        pred_mse += float(pred_err.sum().detach().cpu())
        corrupt_mse += float(corrupt_err.sum().detach().cpu())
        rows7_mask = torch.as_tensor(rows7, device=device).bool()
        if rows7_mask.any():
            rows7_total += int(rows7_mask.sum().detach().cpu())
            rows7_pred_mse += float(pred_err[rows7_mask].sum().detach().cpu())
    return {
        "samples": total,
        "loss": loss_total / max(total, 1),
        "pred_lattice_mse": pred_mse / max(total, 1),
        "corrupt_lattice_mse": corrupt_mse / max(total, 1),
        "pred_vs_corrupt_mse_ratio": (pred_mse / max(total, 1)) / max(corrupt_mse / max(total, 1), 1e-12),
        "rows_ge_7_samples": rows7_total,
        "rows_ge_7_pred_lattice_mse": rows7_pred_mse / max(rows7_total, 1),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float, seed: int) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "seed": seed,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "candidate_order": "not_applicable_generative_denoiser_no_ranking",
        },
        path,
    )


def append_experiment_log(summary: dict) -> None:
    path = ROOT / "reports/opentry_5_experiment_log.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# opentry_5 Experiment Log\n"
    block = f"""

## E8003: Model B lattice denoiser smoke
- Time: {summary['created_at']}
- Core hypothesis: a generative denoiser can learn to reverse lattice/VPA corruption without candidate ranking.
- Difference vs historical failures: trained model objective, not scorer/selector/anchor insertion/direct candidate sorting.
- Model side or data side: model side.
- Contains sorting/filtering: no.
- candidate order: not applicable; denoiser applies the same correction to every input.
- Read files: data/geometry_denoising_train.jsonl, data/geometry_denoising_dev.jsonl.
- Written files: checkpoints/model_b_denoiser_smoke/last.pt, checkpoints/model_b_denoiser_smoke/best.pt, eval/model_b_denoiser_smoke_eval.json.
- Data split: train_core synthetic train, dev_model/dev_gate synthetic dev.
- Data hash: see manifests/opentry_5_manifest.jsonl after refresh.
- Read test: no.
- Read val512: no.
- val512 cumulative use: 0.
- Model: LatticeDenoiser MLP with Gaussian NLL + Huber + VPA auxiliary constraints.
- Parameters: {summary['parameter_count']}
- GPU/CPU: {summary['device']}
- Training time: {summary['train_time_seconds']:.2f}s.
- Inference time: included in smoke eval.
- readable/composition exact/SG-Wyckoff/match/RMSE: not applicable to lattice-only smoke; full CIF generation gate pending.
- grouped dev folds consistent: pending full fold CIF evaluation.
- Conclusion: smoke completed; not a terminal breakthrough.
- Gate: {'pass' if summary['gate_passed'] else 'fail'}.
- Terminate family: no.
- Next: extend Model B to full row/free-param/site-mapping CIF generation.
"""
    write_text(path, existing.rstrip() + block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-max", type=int, default=4096)
    parser.add_argument("--dev-max", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8003)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_ds = DenoiseDataset(ROOT / "data/geometry_denoising_train.jsonl", args.train_max)
    dev_ds = DenoiseDataset(ROOT / "data/geometry_denoising_dev.jsonl", args.dev_max)
    if len(train_ds) == 0 or len(dev_ds) == 0:
        raise RuntimeError("empty denoising dataset")
    input_dim = int(train_ds[0][0].numel())
    model = LatticeDenoiser(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Unit checks: tensor shape, deterministic seed, checkpoint resume load.
    x0, y0, _, _ = train_ds[0]
    mu0, logvar0 = model(x0[None].to(device))
    assert mu0.shape == (1, 7) and logvar0.shape == (1, 7)
    set_seed(args.seed)
    model_a = LatticeDenoiser(input_dim=input_dim).to(device)
    set_seed(args.seed)
    model_b = LatticeDenoiser(input_dim=input_dim).to(device)
    det_a = model_a(x0[None].to(device))[0].detach().cpu()
    det_b = model_b(x0[None].to(device))[0].detach().cpu()
    assert torch.allclose(det_a, det_b), "deterministic seed check failed"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_loss = float("inf")
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_total = 0.0
        aux_total = collections = {"nll": 0.0, "huber": 0.0, "vpa": 0.0}
        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model(x)
            loss, aux = loss_fn(mu, logvar, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = x.shape[0]
            total += n
            loss_total += float(loss.detach().cpu()) * n
            for key in aux_total:
                aux_total[key] += aux[key] * n
        dev_metrics = evaluate(model, dev_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": loss_total / max(total, 1),
            "train_aux": {k: v / max(total, 1) for k, v in aux_total.items()},
            "dev": dev_metrics,
        }
        history.append(epoch_summary)
        save_checkpoint(ROOT / "checkpoints/model_b_denoiser_smoke/last.pt", model, optimizer, epoch, best_loss, args.seed)
        if dev_metrics["loss"] < best_loss:
            best_loss = dev_metrics["loss"]
            save_checkpoint(ROOT / "checkpoints/model_b_denoiser_smoke/best.pt", model, optimizer, epoch, best_loss, args.seed)

    # Resume unit check.
    ckpt = torch.load(ROOT / "checkpoints/model_b_denoiser_smoke/last.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    final_dev = evaluate(model, dev_loader, device)
    elapsed = time.time() - start
    parameter_count = sum(p.numel() for p in model.parameters())
    gate_passed = final_dev["pred_vs_corrupt_mse_ratio"] < 1.0 and final_dev["samples"] > 0
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": "E8003",
        "environment": "crystallm_env",
        "device": str(device),
        "seed": args.seed,
        "train_samples": len(train_ds),
        "dev_samples": len(dev_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parameter_count": parameter_count,
        "history": history,
        "final_dev": final_dev,
        "train_time_seconds": elapsed,
        "gate_passed": gate_passed,
        "candidate_order": "not_applicable_generative_denoiser_no_ranking",
        "read_test": False,
        "read_val512": False,
    }
    write_json(ROOT / "eval/model_b_denoiser_smoke_eval.json", summary)
    append_experiment_log(summary)
    report = f"""# Model B Denoiser Report

Smoke E8003 completed at {summary['created_at']}.

This is a generative denoiser smoke model, not a scorer, selector, or reranker. It applies the same correction to every input and emits no ordered candidate pool.

```json
{json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)}
```

Status: smoke only. Full row/free-param/site-mapping CIF generation and grouped-fold StructureMatcher evaluation are still pending.
"""
    write_text(ROOT / "reports/model_b_denoiser_report.md", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
