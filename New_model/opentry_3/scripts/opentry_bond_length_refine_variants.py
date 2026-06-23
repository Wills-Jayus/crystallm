#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element


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


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def fallback_distance(a: str, b: str) -> tuple[float, float, float]:
    def radius(sym: str) -> float:
        try:
            el = Element(str(sym))
            value = getattr(el, "atomic_radius", None) or getattr(el, "atomic_radius_calculated", None)
            return float(value or 1.4)
        except Exception:
            return 1.4

    median = max(0.9, min(3.2, radius(a) + radius(b)))
    return median * 0.78, median, median * 1.25


def build_pair_stats(
    train_cif_dir: Path,
    *,
    max_structures: int,
    max_sites: int,
    cutoff: float,
    min_count: int,
) -> dict[str, dict[str, float]]:
    raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    paths = sorted(train_cif_dir.glob("*.cif"))[: int(max_structures)]
    for path in paths:
        try:
            struct = Structure.from_file(str(path))
        except Exception:
            continue
        if int(struct.num_sites) > int(max_sites):
            continue
        try:
            neighbors = struct.get_all_neighbors(float(cutoff))
        except Exception:
            continue
        for i, neighs in enumerate(neighbors):
            elem_i = str(struct[i].specie.symbol)
            for nn in neighs:
                try:
                    j = int(nn.index)
                    if j <= i:
                        continue
                    dist = float(nn.nn_distance)
                    if 0.45 <= dist <= float(cutoff):
                        raw[pair_key(elem_i, str(struct[j].specie.symbol))].append(dist)
                except Exception:
                    continue
    stats: dict[str, dict[str, float]] = {}
    for key, values in raw.items():
        if len(values) < int(min_count):
            continue
        stats["|".join(key)] = {
            "count": len(values),
            "lo": quantile(values, 0.18),
            "med": quantile(values, 0.50),
            "hi": quantile(values, 0.82),
        }
    return stats


def lookup_pair(stats: dict[str, dict[str, float]], a: str, b: str) -> tuple[float, float, float]:
    key = "|".join(pair_key(a, b))
    row = stats.get(key)
    if row:
        return float(row["lo"]), float(row["med"]), float(row["hi"])
    return fallback_distance(a, b)


def parse_cif(cif: str) -> tuple[dict[str, float], list[str], list[int], list[str], np.ndarray]:
    cell: dict[str, float] = {}
    lines = str(cif or "").splitlines()
    atom_line_indices: list[int] = []
    elements: list[str] = []
    coords: list[list[float]] = []
    in_atom_loop = False
    for idx, line in enumerate(lines):
        text = line.strip()
        match = CELL_RE.match(text)
        if match:
            cell[match.group(1).replace("_cell_", "")] = float(match.group(2))
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
        elements.append(str(parts[0]))
        atom_line_indices.append(idx)
        coords.append(xyz)
    required = ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"]
    if any(key not in cell for key in required) or not coords:
        raise ValueError("missing_cell_or_atom_loop")
    return cell, lines, atom_line_indices, elements, np.asarray(coords, dtype=float)


def make_lattice(cell: dict[str, float], scale: float) -> Lattice:
    return Lattice.from_parameters(
        float(cell["length_a"]) * float(scale),
        float(cell["length_b"]) * float(scale),
        float(cell["length_c"]) * float(scale),
        float(cell["angle_alpha"]),
        float(cell["angle_beta"]),
        float(cell["angle_gamma"]),
    )


def bond_energy(coords: np.ndarray, elements: list[str], lattice: Lattice, stats: dict[str, dict[str, float]], cutoff: float) -> float:
    if len(coords) <= 1:
        return 0.0
    matrix = np.asarray(lattice.matrix, dtype=float)
    total = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            df = coords[i] - coords[j]
            df -= np.round(df)
            dist = float(np.linalg.norm(df @ matrix))
            if dist > float(cutoff):
                continue
            lo, med, hi = lookup_pair(stats, elements[i], elements[j])
            hard_lo = max(0.55, 0.72 * lo)
            if dist < hard_lo:
                total += 20.0 * ((hard_lo - dist) / max(hard_lo, 1.0e-6)) ** 2
            elif dist < lo:
                total += 3.0 * ((lo - dist) / max(lo, 1.0e-6)) ** 2
            elif dist > hi:
                # Only weakly pull overly long near-neighbor contacts; do not collapse distant atoms.
                total += 0.12 * ((min(dist, float(cutoff)) - hi) / max(med, 1.0e-6)) ** 2
    return float(total)


def refine_coords(
    coords: np.ndarray,
    elements: list[str],
    lattice: Lattice,
    stats: dict[str, dict[str, float]],
    *,
    cutoff: float,
    steps: int,
    step_size: float,
) -> np.ndarray:
    frac = coords.copy() % 1.0
    matrix = np.asarray(lattice.matrix, dtype=float)
    inv_matrix = np.linalg.inv(matrix)
    for _ in range(int(steps)):
        forces_cart = np.zeros_like(frac)
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                df = frac[i] - frac[j]
                df -= np.round(df)
                cart = df @ matrix
                dist = float(np.linalg.norm(cart))
                if dist <= 1.0e-6 or dist > float(cutoff):
                    continue
                lo, _med, hi = lookup_pair(stats, elements[i], elements[j])
                hard_lo = max(0.55, 0.72 * lo)
                direction = cart / dist
                mag = 0.0
                if dist < hard_lo:
                    mag = 2.5 * (hard_lo - dist) / max(hard_lo, 1.0e-6)
                elif dist < lo:
                    mag = 0.75 * (lo - dist) / max(lo, 1.0e-6)
                elif dist > hi:
                    mag = -0.05 * (min(dist, float(cutoff)) - hi) / max(hi, 1.0e-6)
                if mag == 0.0:
                    continue
                forces_cart[i] += mag * direction
                forces_cart[j] -= mag * direction
        forces_frac = forces_cart @ inv_matrix
        max_norm = float(np.max(np.linalg.norm(forces_frac, axis=1))) if len(forces_frac) else 0.0
        if max_norm <= 1.0e-9:
            break
        if max_norm > 1.0:
            forces_frac /= max_norm
        frac = (frac + float(step_size) * forces_frac) % 1.0
    return frac


def rewrite_cif(
    cif: str,
    lines: list[str],
    atom_indices: list[int],
    coords: np.ndarray,
    cell: dict[str, float],
    scale: float,
    suffix: str,
) -> str:
    lattice = make_lattice(cell, scale)
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


def make_variants(
    cand: dict[str, Any],
    stats: dict[str, dict[str, float]],
    *,
    scales: list[float],
    cutoff: float,
    steps: int,
    step_size: float,
    keep_only_improved: bool,
) -> list[dict[str, Any]]:
    cell, lines, atom_indices, elements, coords = parse_cif(str(cand.get("cif") or ""))
    out: list[dict[str, Any]] = []
    for scale in scales:
        lattice = make_lattice(cell, float(scale))
        start_energy = bond_energy(coords, elements, lattice, stats, cutoff)
        refined = refine_coords(coords, elements, lattice, stats, cutoff=cutoff, steps=steps, step_size=step_size)
        end_energy = bond_energy(refined, elements, lattice, stats, cutoff)
        if keep_only_improved and end_energy > start_energy - 1.0e-8:
            continue
        row = dict(cand)
        suffix = f"bondref_s{scale:.3f}_e{end_energy:.3f}".replace(".", "p")
        row["cif"] = rewrite_cif(str(cand.get("cif") or ""), lines, atom_indices, refined, cell, float(scale), suffix)
        row["geometry_param_variant_mode"] = "bond_length_refine"
        row["bond_refine_scale"] = float(scale)
        row["bond_refine_start_energy"] = float(start_energy)
        row["bond_refine_end_energy"] = float(end_energy)
        row["bond_refine_delta_energy"] = float(start_energy - end_energy)
        for key in ("self_min_distance", "self_volume", "self_volume_per_atom", "self_score", "self_parse_error"):
            row.pop(key, None)
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Create GT-free bond-length-aware coordinate/lattice refinement variants.")
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--train-cif-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=20)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-train-structures", type=int, default=1024)
    parser.add_argument("--max-train-sites", type=int, default=160)
    parser.add_argument("--max-candidate-sites", type=int, default=96)
    parser.add_argument("--pair-cutoff", type=float, default=4.0)
    parser.add_argument("--min-pair-count", type=int, default=8)
    parser.add_argument("--scales", default="0.985,1.000,1.025")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.055)
    parser.add_argument("--keep-only-improved", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = build_pair_stats(
        args.train_cif_dir,
        max_structures=int(args.max_train_structures),
        max_sites=int(args.max_train_sites),
        cutoff=float(args.pair_cutoff),
        min_count=int(args.min_pair_count),
    )
    scales = [float(x) for x in str(args.scales).split(",") if x.strip()]
    groups = grouped(read_jsonl(args.input_rendered_jsonl))
    sample_ids = list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "train_cif_dir": str(args.train_cif_dir),
        "selected_samples": len(sample_ids),
        "pair_stats_keys": len(stats),
        "max_train_structures": int(args.max_train_structures),
        "max_train_sites": int(args.max_train_sites),
        "max_candidate_sites": int(args.max_candidate_sites),
        "pair_cutoff": float(args.pair_cutoff),
        "scales": scales,
        "steps": int(args.steps),
        "step_size": float(args.step_size),
        "keep_only_improved": bool(args.keep_only_improved),
        "input_candidates_considered": 0,
        "skipped_large_candidate": 0,
        "failed_candidates": 0,
        "output_rows": 0,
        "samples_with_output": 0,
        "note": "Uses only train CIF pair-distance statistics and rendered candidate CIFs at inference; no test data, target GT, W/A labels, StructureMatcher labels, row_count labels, or oracle ranking.",
    }
    for sample_id in sample_ids:
        sample_rows: list[dict[str, Any]] = []
        for cand in groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]:
            summary["input_candidates_considered"] += 1
            try:
                atom_count = int(cand.get("atom_count_after_expansion") or 0)
                if atom_count > int(args.max_candidate_sites):
                    summary["skipped_large_candidate"] += 1
                    continue
                variants = make_variants(
                    cand,
                    stats,
                    scales=scales,
                    cutoff=float(args.pair_cutoff),
                    steps=int(args.steps),
                    step_size=float(args.step_size),
                    keep_only_improved=bool(args.keep_only_improved),
                )
            except Exception:
                summary["failed_candidates"] += 1
                continue
            for row in variants:
                sample_rows.append(row)
                if len(sample_rows) >= int(args.max_output_candidates_per_sample):
                    break
            if len(sample_rows) >= int(args.max_output_candidates_per_sample):
                break
        if sample_rows:
            summary["samples_with_output"] += 1
        for rank, row in enumerate(sample_rows[: int(args.max_output_candidates_per_sample)], start=1):
            item = dict(row)
            item["sample_id"] = sample_id
            item["rank"] = rank
            out_rows.append(item)
    summary["output_rows"] = len(out_rows)
    write_jsonl(out_dir / "rendered_topk_bond_refine.jsonl", out_rows)
    write_json(out_dir / "bond_refine_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
