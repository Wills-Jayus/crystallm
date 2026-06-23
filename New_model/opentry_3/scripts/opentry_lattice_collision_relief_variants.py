#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
CELL_FLOAT_RE = re.compile(r"^(_cell_(?:length_[abc]|volume))\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
DATA_RE = re.compile(r"^data_(\S+)")


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


def grouped_in_input_order(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(str(row.get("sample_id")), []).append(row)
    for values in groups.values():
        values.sort(key=lambda item: int(item.get("rank", item.get("original_rank", 10**9))))
    return groups


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


def scale_cif_lengths(cif: str, scale: float, suffix: str) -> str:
    out: list[str] = []
    volume_scale = float(scale) ** 3
    renamed = False
    for line in str(cif or "").splitlines():
        match_data = DATA_RE.match(line.strip())
        if match_data and not renamed:
            out.append(f"data_{match_data.group(1)}_{suffix}")
            renamed = True
            continue
        match = CELL_FLOAT_RE.match(line.strip())
        if match is None:
            out.append(line)
            continue
        key, value_text = match.groups()
        value = float(value_text)
        factor = volume_scale if key == "_cell_volume" else float(scale)
        out.append(f"{key}   {value * factor:.8f}")
    return "\n".join(out) + "\n"


def scale_schedule(min_distance: float, target_min_distance: float, max_scale: float, extra_scales: list[float]) -> list[float]:
    scales: list[float] = []
    if min_distance > 0.0 and min_distance < target_min_distance:
        scales.append(target_min_distance / max(0.2, min_distance))
    scales.extend(extra_scales)
    clean: list[float] = []
    for scale in scales:
        value = max(1.0, min(float(max_scale), float(scale)))
        if value <= 1.0001:
            continue
        if all(abs(value - old) > 0.004 for old in clean):
            clean.append(value)
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description="Create GT-free lattice collision-relief variants from rendered CIF candidates.")
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    parser.add_argument("--target-min-distance", type=float, default=1.75)
    parser.add_argument("--trigger-min-distance", type=float, default=1.65)
    parser.add_argument("--max-scale", type=float, default=1.16)
    parser.add_argument("--extra-scales", default="1.04,1.08,1.12")
    parser.add_argument("--keep-original", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_scales = [float(x) for x in str(args.extra_scales).split(",") if x.strip()]
    groups = grouped_in_input_order(read_jsonl(args.input_rendered_jsonl))
    requested_sample_ids = load_sample_ids(args.sample_ids_file)
    sample_ids = requested_sample_ids if requested_sample_ids is not None else list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]

    out_rows: list[dict[str, Any]] = []
    summary = {
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "selected_samples": len(sample_ids),
        "sample_ids_file": None if args.sample_ids_file is None else str(args.sample_ids_file),
        "target_min_distance": float(args.target_min_distance),
        "trigger_min_distance": float(args.trigger_min_distance),
        "max_scale": float(args.max_scale),
        "extra_scales": extra_scales,
        "keep_original": bool(args.keep_original),
        "input_candidates_considered": 0,
        "triggered_candidates": 0,
        "output_rows": 0,
        "samples_with_output": 0,
        "note": "Uses only rendered candidate self_min_distance and CIF cell fields at inference; no StructureMatcher labels, target GT, row_count, or test data.",
    }
    for sample_id in sample_ids:
        candidates = groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        sample_rows: list[dict[str, Any]] = []
        for cand in candidates:
            summary["input_candidates_considered"] += 1
            if bool(args.keep_original):
                base = dict(cand)
                base["relief_variant"] = "original"
                sample_rows.append(base)
            try:
                min_distance = float(cand.get("self_min_distance"))
            except Exception:
                continue
            if min_distance <= 0.0 or min_distance > float(args.trigger_min_distance):
                continue
            if not cand.get("cif"):
                continue
            summary["triggered_candidates"] += 1
            for variant_idx, scale in enumerate(
                scale_schedule(min_distance, float(args.target_min_distance), float(args.max_scale), extra_scales),
                start=1,
            ):
                row = dict(cand)
                row["cif"] = scale_cif_lengths(str(cand.get("cif") or ""), float(scale), f"relief{variant_idx}_s{scale:.3f}".replace(".", "p"))
                row["relief_variant"] = "collision_lattice_scale"
                row["relief_scale"] = float(scale)
                row["relief_source_min_distance"] = float(min_distance)
                row["geometry_param_variant_mode"] = "collision_relief_lattice"
                row.pop("self_min_distance", None)
                row.pop("self_volume", None)
                row.pop("self_volume_per_atom", None)
                row.pop("self_score", None)
                row.pop("self_parse_error", None)
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
    write_jsonl(out_dir / "rendered_topk_collision_relief.jsonl", out_rows)
    write_json(out_dir / "collision_relief_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
