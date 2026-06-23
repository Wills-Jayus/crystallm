#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Lattice, Structure


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
for path in (OPENTRY_ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_bond_length_refine_variants as br  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402


DATA_RE = re.compile(r"^data_(\S+)")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return br.read_jsonl(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    br.write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    br.write_jsonl(path, rows)


def grouped(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    return br.grouped(rows)


def load_row_counts(repr_jsonl: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_jsonl(repr_jsonl):
        sample_id = str((row.get("keys") or {}).get("sample_id") or row.get("sample_id") or "")
        if sample_id:
            out[sample_id] = int(row.get("row_count") or len(row.get("assignment_sequence") or row.get("wa_sequence") or []))
    return out


def build_vpa_stats(train_cif_dir: Path, *, max_structures: int, max_sites: int) -> dict[str, float]:
    values: list[float] = []
    for path in sorted(train_cif_dir.glob("*.cif"))[: int(max_structures)]:
        try:
            struct = Structure.from_file(str(path))
        except Exception:
            continue
        if int(struct.num_sites) <= 0 or int(struct.num_sites) > int(max_sites):
            continue
        values.append(float(struct.volume) / float(struct.num_sites))
    if not values:
        return {"median": 18.0, "lo": 8.0, "hi": 36.0, "count": 0}
    return {
        "count": len(values),
        "lo": br.quantile(values, 0.10),
        "median": br.quantile(values, 0.50),
        "hi": br.quantile(values, 0.90),
    }


def lattice_matrix(cell: dict[str, float]) -> np.ndarray:
    lat = Lattice.from_parameters(
        float(cell["length_a"]),
        float(cell["length_b"]),
        float(cell["length_c"]),
        float(cell["angle_alpha"]),
        float(cell["angle_beta"]),
        float(cell["angle_gamma"]),
    )
    return np.asarray(lat.matrix, dtype=np.float32)


def pair_tensors(
    coords: np.ndarray,
    elements: list[str],
    base_matrix: np.ndarray,
    stats: dict[str, dict[str, float]],
    *,
    cutoff: float,
) -> dict[str, torch.Tensor]:
    n = len(coords)
    i_idx: list[int] = []
    j_idx: list[int] = []
    lo_vals: list[float] = []
    med_vals: list[float] = []
    hi_vals: list[float] = []
    active_vals: list[float] = []
    matrix = np.asarray(base_matrix, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            df = coords[i] - coords[j]
            df -= np.round(df)
            init_dist = float(np.linalg.norm(df @ matrix))
            lo, med, hi = br.lookup_pair(stats, elements[i], elements[j])
            # Always repel short contacts, but only attract pairs that are plausible near-neighbors.
            active = 1.0 if init_dist <= min(float(cutoff), max(hi * 1.35, med + 0.75)) else 0.0
            i_idx.append(i)
            j_idx.append(j)
            lo_vals.append(float(lo))
            med_vals.append(float(med))
            hi_vals.append(float(hi))
            active_vals.append(active)
    return {
        "i": torch.tensor(i_idx, dtype=torch.long),
        "j": torch.tensor(j_idx, dtype=torch.long),
        "lo": torch.tensor(lo_vals, dtype=torch.float32),
        "med": torch.tensor(med_vals, dtype=torch.float32),
        "hi": torch.tensor(hi_vals, dtype=torch.float32),
        "active": torch.tensor(active_vals, dtype=torch.float32),
    }


def preset_params(name: str) -> dict[str, float]:
    if name == "repel":
        return {
            "steps": 70,
            "lr": 0.045,
            "max_delta": 0.18,
            "max_log_scale": 0.045,
            "hard_weight": 32.0,
            "lo_weight": 10.0,
            "med_weight": 0.0,
            "hi_weight": 0.0,
            "delta_weight": 0.18,
            "scale_weight": 0.4,
            "vpa_weight": 0.05,
        }
    if name == "pairfield":
        return {
            "steps": 110,
            "lr": 0.032,
            "max_delta": 0.30,
            "max_log_scale": 0.085,
            "hard_weight": 24.0,
            "lo_weight": 7.0,
            "med_weight": 0.65,
            "hi_weight": 0.10,
            "delta_weight": 0.10,
            "scale_weight": 0.25,
            "vpa_weight": 0.15,
        }
    raise SystemExit(f"Unknown preset: {name}")


def optimize_pairfield(
    coords_np: np.ndarray,
    elements: list[str],
    cell: dict[str, float],
    stats: dict[str, dict[str, float]],
    vpa_stats: dict[str, float],
    *,
    preset: str,
    cutoff: float,
    atom_count: int,
) -> tuple[np.ndarray, float, dict[str, float]]:
    params = preset_params(preset)
    device = torch.device("cpu")
    coords0 = torch.tensor(coords_np, dtype=torch.float32, device=device)
    base_matrix = torch.tensor(lattice_matrix(cell), dtype=torch.float32, device=device)
    pair = {key: value.to(device) for key, value in pair_tensors(coords_np, elements, lattice_matrix(cell), stats, cutoff=cutoff).items()}
    raw_delta = torch.zeros_like(coords0, requires_grad=True)
    raw_scale = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw_delta, raw_scale], lr=float(params["lr"]))
    start_loss = None
    last_parts: dict[str, float] = {}
    for step in range(int(params["steps"])):
        optimizer.zero_grad(set_to_none=True)
        delta = float(params["max_delta"]) * torch.tanh(raw_delta)
        coords = (coords0 + delta) % 1.0
        log_scale = float(params["max_log_scale"]) * torch.tanh(raw_scale)
        scale = torch.exp(log_scale)
        mat = base_matrix * scale
        df = coords[pair["i"]] - coords[pair["j"]]
        df = df - torch.round(df.detach())
        dist = torch.linalg.norm(df @ mat, dim=1).clamp_min(1.0e-5)
        hard_lo = torch.clamp(pair["lo"] * 0.72, min=0.55)
        hard_loss = torch.relu(hard_lo - dist).pow(2).mean()
        lo_loss = torch.relu(pair["lo"] - dist).pow(2).mean()
        med_loss = (pair["active"] * ((dist - pair["med"]) / pair["med"].clamp_min(0.8)).pow(2)).sum() / pair["active"].sum().clamp_min(1.0)
        hi_loss = (pair["active"] * torch.relu(dist - pair["hi"]).pow(2)).sum() / pair["active"].sum().clamp_min(1.0)
        delta_loss = delta.pow(2).mean()
        scale_loss = log_scale.pow(2)
        base_volume = float(Lattice(lattice_matrix(cell)).volume)
        vpa = torch.tensor(base_volume / max(1, int(atom_count)), dtype=torch.float32) * scale.pow(3)
        vpa_med = torch.tensor(float(vpa_stats.get("median", 18.0)), dtype=torch.float32)
        vpa_loss = torch.log((vpa / vpa_med).clamp_min(1.0e-5)).pow(2)
        loss = (
            float(params["hard_weight"]) * hard_loss
            + float(params["lo_weight"]) * lo_loss
            + float(params["med_weight"]) * med_loss
            + float(params["hi_weight"]) * hi_loss
            + float(params["delta_weight"]) * delta_loss
            + float(params["scale_weight"]) * scale_loss
            + float(params["vpa_weight"]) * vpa_loss
        )
        if step == 0:
            start_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_delta, raw_scale], 1.5)
        optimizer.step()
        last_parts = {
            "loss": float(loss.detach().cpu()),
            "hard": float(hard_loss.detach().cpu()),
            "lo": float(lo_loss.detach().cpu()),
            "med": float(med_loss.detach().cpu()),
            "hi": float(hi_loss.detach().cpu()),
            "delta": float(delta_loss.detach().cpu()),
            "scale": float(scale.detach().cpu()),
            "vpa": float(vpa.detach().cpu()),
        }
    with torch.no_grad():
        coords_final = ((coords0 + float(params["max_delta"]) * torch.tanh(raw_delta)) % 1.0).detach().cpu().numpy()
        scale_final = float(torch.exp(float(params["max_log_scale"]) * torch.tanh(raw_scale)).detach().cpu())
    last_parts["start_loss"] = float(start_loss if start_loss is not None else last_parts.get("loss", 0.0))
    last_parts["delta_loss"] = float(last_parts["start_loss"] - last_parts.get("loss", 0.0))
    return coords_final, scale_final, last_parts


def rewrite_scaled_cif(
    *,
    lines: list[str],
    atom_indices: list[int],
    coords: np.ndarray,
    cell: dict[str, float],
    scale: float,
    suffix: str,
) -> str:
    lattice = Lattice.from_parameters(
        float(cell["length_a"]) * float(scale),
        float(cell["length_b"]) * float(scale),
        float(cell["length_c"]) * float(scale),
        float(cell["angle_alpha"]),
        float(cell["angle_beta"]),
        float(cell["angle_gamma"]),
    )
    out = list(lines)
    atom_index_set = set(atom_indices)
    coord_iter = iter(coords.tolist())
    renamed = False
    for idx, line in enumerate(lines):
        text = line.strip()
        data_match = DATA_RE.match(text)
        if data_match and not renamed:
            out[idx] = f"data_{data_match.group(1)}_{suffix}"
            renamed = True
            continue
        cell_match = br.CELL_RE.match(text)
        if cell_match:
            key = cell_match.group(1)
            short = key.replace("_cell_", "")
            if short.startswith("length_"):
                out[idx] = f"{key}   {float(cell[short]) * float(scale):.8f}"
            elif short == "volume":
                out[idx] = f"{key}   {float(lattice.volume):.8f}"
            else:
                out[idx] = f"{key}   {float(cell[short]):.8f}"
            continue
        if idx in atom_index_set:
            parts = text.split()
            xyz = next(coord_iter)
            out[idx] = f"  {parts[0]}  {parts[1]}  {parts[2]}  {xyz[0]:.8f}  {xyz[1]:.8f}  {xyz[2]:.8f}  {parts[6]}"
    return "\n".join(out) + "\n"


def make_variant(
    cand: dict[str, Any],
    stats: dict[str, dict[str, float]],
    vpa_stats: dict[str, float],
    *,
    preset: str,
    cutoff: float,
) -> dict[str, Any] | None:
    cell, lines, atom_indices, elements, coords = br.parse_cif(str(cand.get("cif") or ""))
    if len(coords) <= 1:
        return None
    atom_count = int(cand.get("atom_count_after_expansion") or len(coords))
    coords_final, scale, parts = optimize_pairfield(coords, elements, cell, stats, vpa_stats, preset=preset, cutoff=cutoff, atom_count=atom_count)
    suffix = f"pairfield_{preset}_s{scale:.3f}_l{parts.get('loss', 0.0):.3f}".replace(".", "p")
    cif = rewrite_scaled_cif(lines=lines, atom_indices=atom_indices, coords=coords_final, cell=cell, scale=scale, suffix=suffix)
    out = dict(cand)
    out["cif"] = cif
    out["geometry_param_variant_mode"] = f"pairfield_adam_{preset}"
    out["pairfield_preset"] = preset
    out["pairfield_scale"] = float(scale)
    out["pairfield_start_loss"] = float(parts.get("start_loss", 0.0))
    out["pairfield_end_loss"] = float(parts.get("loss", 0.0))
    out["pairfield_delta_loss"] = float(parts.get("delta_loss", 0.0))
    out["pairfield_vpa"] = float(parts.get("vpa", 0.0))
    for key in ("self_min_distance", "self_volume", "self_volume_per_atom", "self_score", "self_parse_error"):
        out.pop(key, None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Differentiable train-stat pair-field coordinate/lattice optimizer for rendered Wyckoff CIFs.")
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--train-cif-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=["repel", "pairfield"], required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=8)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-train-structures", type=int, default=1024)
    parser.add_argument("--max-train-sites", type=int, default=160)
    parser.add_argument("--max-candidate-sites", type=int, default=96)
    parser.add_argument("--pair-cutoff", type=float, default=4.2)
    parser.add_argument("--min-pair-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_stats = br.build_pair_stats(
        args.train_cif_dir,
        max_structures=int(args.max_train_structures),
        max_sites=int(args.max_train_sites),
        cutoff=float(args.pair_cutoff),
        min_count=int(args.min_pair_count),
    )
    vpa_stats = build_vpa_stats(args.train_cif_dir, max_structures=int(args.max_train_structures), max_sites=int(args.max_train_sites))
    row_counts = load_row_counts(args.repr_jsonl)
    groups = grouped(read_jsonl(args.input_rendered_jsonl))
    sample_ids = list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "repr_jsonl": str(args.repr_jsonl),
        "train_cif_dir": str(args.train_cif_dir),
        "preset": str(args.preset),
        "selected_samples": len(sample_ids),
        "min_target_row_count": int(args.min_target_row_count),
        "pair_stats_keys": len(pair_stats),
        "vpa_stats": vpa_stats,
        "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
        "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
        "input_candidates_considered": 0,
        "output_rows": 0,
        "samples_with_output": 0,
        "skipped_target_rows_lt_min": 0,
        "skipped_large_candidate": 0,
        "failed_candidates": 0,
        "note": "Uses only train CIF pair-distance and volume statistics. Inference uses rendered candidate CIFs and predicted W/A metadata only; no val/test StructureMatcher labels or oracle reranking.",
    }
    for sample_pos, sample_id in enumerate(sample_ids, start=1):
        if int(args.min_target_row_count) > 0 and int(row_counts.get(sample_id, 0)) < int(args.min_target_row_count):
            summary["skipped_target_rows_lt_min"] += 1
            continue
        sample_rows: list[dict[str, Any]] = []
        for cand in groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]:
            summary["input_candidates_considered"] += 1
            atom_count = int(cand.get("atom_count_after_expansion") or cand.get("atom_count") or 0)
            if atom_count > int(args.max_candidate_sites):
                summary["skipped_large_candidate"] += 1
                continue
            try:
                variant = make_variant(cand, pair_stats, vpa_stats, preset=str(args.preset), cutoff=float(args.pair_cutoff))
            except Exception:
                summary["failed_candidates"] += 1
                continue
            if variant is None:
                summary["failed_candidates"] += 1
                continue
            variant.update({f"pairfield_{key}": value for key, value in selfscore.cif_self_features(str(variant.get("cif") or "")).items()})
            variant["pairfield_readable_check"] = not bool(variant.get("pairfield_self_parse_error"))
            sample_rows.append(variant)
            if len(sample_rows) >= int(args.max_output_candidates_per_sample):
                break
        if sample_rows:
            summary["samples_with_output"] += 1
        for rank, row in enumerate(sample_rows[: int(args.max_output_candidates_per_sample)], start=1):
            item = dict(row)
            item["sample_id"] = sample_id
            item["rank"] = rank
            out_rows.append(item)
        if int(args.progress_every) > 0 and sample_pos % int(args.progress_every) == 0:
            print(json.dumps({"sample_pos": sample_pos, "out_rows": len(out_rows), "failed": summary["failed_candidates"]}, sort_keys=True), flush=True)
    summary["output_rows"] = len(out_rows)
    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    write_json(out_dir / "pairfield_adam_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
