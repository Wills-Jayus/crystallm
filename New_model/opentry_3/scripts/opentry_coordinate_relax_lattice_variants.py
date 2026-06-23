#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
DATA_RE = re.compile(r"^data_(\S+)")
CELL_RE = re.compile(r"^(_cell_(?:length_[abc]|angle_(?:alpha|beta|gamma)|volume))\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def grouped(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    out: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        out.setdefault(str(row.get("sample_id")), []).append(row)
    for values in out.values():
        values.sort(key=lambda row: int(row.get("rank", row.get("original_rank", 10**9))))
    return out


def load_sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                sample_id = str(json.loads(line).get("sample_id") or "")
            else:
                sample_id = line.strip()
            if sample_id and sample_id not in seen:
                seen.add(sample_id)
                ids.append(sample_id)
    return ids


def parse_cif(cif: str) -> tuple[dict[str, float], list[str], list[int], np.ndarray]:
    cell: dict[str, float] = {}
    lines = str(cif or "").splitlines()
    atom_line_indices: list[int] = []
    coords: list[list[float]] = []
    in_atom_loop = False
    for idx, line in enumerate(lines):
        text = line.strip()
        match = CELL_RE.match(text)
        if match:
            key = match.group(1).replace("_cell_", "")
            cell[key] = float(match.group(2))
            continue
        if text.startswith("loop_"):
            in_atom_loop = False
            continue
        if text.startswith("_atom_site_type_symbol"):
            in_atom_loop = True
            continue
        if not in_atom_loop or text.startswith("_") or not text:
            continue
        parts = text.split()
        if len(parts) < 7:
            continue
        try:
            xyz = [float(parts[3]) % 1.0, float(parts[4]) % 1.0, float(parts[5]) % 1.0]
        except Exception:
            continue
        atom_line_indices.append(idx)
        coords.append(xyz)
    required = ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"]
    if any(key not in cell for key in required) or not coords:
        raise ValueError("missing_cell_or_atom_loop")
    return cell, lines, atom_line_indices, np.asarray(coords, dtype=float)


def scaled_lattice(cell: dict[str, float], scale: float) -> Lattice:
    return Lattice.from_parameters(
        float(cell["length_a"]) * float(scale),
        float(cell["length_b"]) * float(scale),
        float(cell["length_c"]) * float(scale),
        float(cell["angle_alpha"]),
        float(cell["angle_beta"]),
        float(cell["angle_gamma"]),
    )


def relax_fractional(coords: np.ndarray, lattice: Lattice, cutoff: float, steps: int, step_size: float) -> np.ndarray:
    if len(coords) <= 1:
        return coords.copy()
    frac = coords.copy() % 1.0
    matrix = np.asarray(lattice.matrix, dtype=float)
    inv_matrix = np.linalg.inv(matrix)
    for _ in range(int(steps)):
        delta_frac = np.zeros_like(frac)
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                df = frac[i] - frac[j]
                df -= np.round(df)
                cart = df @ matrix
                dist = float(np.linalg.norm(cart))
                if dist <= 1.0e-6 or dist >= float(cutoff):
                    continue
                push_cart = (float(cutoff) - dist) / max(float(cutoff), 1.0e-6) * (cart / dist)
                push_frac = push_cart @ inv_matrix
                delta_frac[i] += push_frac
                delta_frac[j] -= push_frac
        max_norm = float(np.max(np.linalg.norm(delta_frac, axis=1))) if len(delta_frac) else 0.0
        if max_norm <= 1.0e-9:
            break
        if max_norm > 1.0:
            delta_frac /= max_norm
        frac = (frac + float(step_size) * delta_frac) % 1.0
    return frac


def rewrite_cif(cif: str, lines: list[str], atom_indices: list[int], coords: np.ndarray, cell: dict[str, float], scale: float, suffix: str) -> str:
    lattice = scaled_lattice(cell, scale)
    out = list(lines)
    renamed = False
    coord_iter = iter(coords.tolist())
    atom_index_set = set(atom_indices)
    for idx, line in enumerate(lines):
        text = line.strip()
        data_match = DATA_RE.match(text)
        if data_match and not renamed:
            out[idx] = f"data_{data_match.group(1)}_{suffix}"
            renamed = True
            continue
        cell_match = CELL_RE.match(text)
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


def candidate_variants(
    *,
    cif: str,
    scales: list[float],
    cutoffs: list[float],
    steps: int,
    step_size: float,
    max_variants: int,
) -> list[tuple[str, float, float]]:
    cell, lines, atom_indices, coords = parse_cif(cif)
    variants: list[tuple[str, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for scale in scales:
        for cutoff in cutoffs:
            key = (round(float(scale), 4), round(float(cutoff), 4))
            if key in seen:
                continue
            seen.add(key)
            lattice = scaled_lattice(cell, scale)
            relaxed = relax_fractional(coords, lattice, float(cutoff), int(steps), float(step_size))
            suffix = f"coordrelax_s{scale:.3f}_c{cutoff:.2f}".replace(".", "p")
            variants.append((rewrite_cif(cif, lines, atom_indices, relaxed, cell, float(scale), suffix), float(scale), float(cutoff)))
            if len(variants) >= int(max_variants):
                return variants
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description="Create GT-free coupled coordinate-lattice relaxation variants from rendered CIFs.")
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-variants-per-candidate", type=int, default=3)
    parser.add_argument("--trigger-min-distance", type=float, default=1.65)
    parser.add_argument("--min-atom-count", type=int, default=12)
    parser.add_argument("--scales", default="1.00,1.03,1.06")
    parser.add_argument("--cutoffs", default="1.45,1.60,1.75")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.08)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = grouped(read_jsonl(args.input_rendered_jsonl))
    requested = load_sample_ids(args.sample_ids_file)
    sample_ids = requested if requested is not None else list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]
    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    cutoffs = [float(x) for x in str(args.cutoffs).split(",") if x.strip()]

    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "sample_ids_file": None if args.sample_ids_file is None else str(args.sample_ids_file),
        "selected_samples": len(sample_ids),
        "scales": scales,
        "cutoffs": cutoffs,
        "steps": int(args.steps),
        "step_size": float(args.step_size),
        "trigger_min_distance": float(args.trigger_min_distance),
        "min_atom_count": int(args.min_atom_count),
        "input_candidates_considered": 0,
        "triggered_candidates": 0,
        "failed_candidates": 0,
        "output_rows": 0,
        "samples_with_output": 0,
        "note": "Inference uses only candidate CIF/self_min_distance/atom count; no StructureMatcher labels, target GT, row_count, or test data.",
    }

    for sample_id in sample_ids:
        sample_rows: list[dict[str, Any]] = []
        for cand in groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]:
            summary["input_candidates_considered"] += 1
            try:
                min_distance = float(cand.get("self_min_distance"))
            except Exception:
                continue
            atom_count = int(cand.get("atom_count_after_expansion") or cand.get("atom_count") or 0)
            if atom_count < int(args.min_atom_count) or min_distance <= 0.0 or min_distance > float(args.trigger_min_distance):
                continue
            if not cand.get("cif"):
                continue
            summary["triggered_candidates"] += 1
            try:
                variants = candidate_variants(
                    cif=str(cand.get("cif") or ""),
                    scales=scales,
                    cutoffs=cutoffs,
                    steps=int(args.steps),
                    step_size=float(args.step_size),
                    max_variants=int(args.max_variants_per_candidate),
                )
            except Exception:
                summary["failed_candidates"] += 1
                continue
            for variant_idx, (variant_cif, scale, cutoff) in enumerate(variants, start=1):
                row = dict(cand)
                row["cif"] = variant_cif
                row["coord_relax_variant"] = "repel_lattice"
                row["coord_relax_scale"] = float(scale)
                row["coord_relax_cutoff"] = float(cutoff)
                row["coord_relax_steps"] = int(args.steps)
                row["coord_relax_step_size"] = float(args.step_size)
                row["coord_relax_source_min_distance"] = float(min_distance)
                row["geometry_param_variant_mode"] = "coordinate_relax_lattice"
                for key in ("self_min_distance", "self_volume", "self_volume_per_atom", "self_score", "self_parse_error"):
                    row.pop(key, None)
                sample_rows.append(row)
                if len(sample_rows) >= int(args.max_output_candidates_per_sample):
                    break
            if len(sample_rows) >= int(args.max_output_candidates_per_sample):
                break
        if sample_rows:
            summary["samples_with_output"] += 1
        for rank, row in enumerate(sample_rows[: int(args.max_output_candidates_per_sample)], start=1):
            row["sample_id"] = sample_id
            row["rank"] = rank
            out_rows.append(row)

    summary["output_rows"] = len(out_rows)
    write_jsonl(out_dir / "rendered_topk_coord_relax.jsonl", out_rows)
    write_json(out_dir / "coord_relax_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
