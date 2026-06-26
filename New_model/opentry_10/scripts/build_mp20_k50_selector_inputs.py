#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cif_scalar(cif: str, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s+(.+?)\s*$", re.MULTILINE)
    match = pattern.search(cif)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def atom_site_rows(cif: str) -> int:
    lines = cif.splitlines()
    in_atom_loop = False
    headers_seen = False
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            if in_atom_loop and headers_seen:
                break
            in_atom_loop = False
            headers_seen = False
            continue
        if stripped.startswith("_atom_site_"):
            in_atom_loop = True
            headers_seen = True
            continue
        if in_atom_loop and headers_seen:
            if not stripped or stripped.startswith("_") or stripped.startswith("#"):
                break
            count += 1
    return count


def build_row(record: dict[str, Any]) -> dict[str, Any]:
    cif = str(record.get("generated_text") or "")
    cfg = record.get("generation_config") or {}
    rank = int(record["rank"])
    return {
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "sample_id": record.get("sample_id"),
        "material_id": record.get("material_id"),
        "rank": rank,
        "gen_index": record.get("gen_index"),
        "rank_source": "anchor_k20" if rank <= 20 else "expanded_k50",
        "generation_config_name": cfg.get("name"),
        "generation_config_rank_start": cfg.get("rank_start"),
        "temperature": cfg.get("temperature"),
        "top_k": cfg.get("top_k"),
        "seed": cfg.get("seed"),
        "logprob_available": bool(record.get("logprob_available")),
        "normalized_token_logprob": record.get("normalized_token_logprob"),
        "cif_num_chars": len(cif),
        "cif_num_lines": len(cif.splitlines()),
        "atom_site_rows": atom_site_rows(cif),
        "declared_sg_symbol": cif_scalar(cif, "_symmetry_space_group_name_H-M"),
        "declared_sg_number": as_int(cif_scalar(cif, "_symmetry_Int_Tables_number")),
        "formula_structural": cif_scalar(cif, "_chemical_formula_structural"),
        "formula_sum": cif_scalar(cif, "_chemical_formula_sum"),
        "cell_length_a": as_float(cif_scalar(cif, "_cell_length_a")),
        "cell_length_b": as_float(cif_scalar(cif, "_cell_length_b")),
        "cell_length_c": as_float(cif_scalar(cif, "_cell_length_c")),
        "cell_angle_alpha": as_float(cif_scalar(cif, "_cell_angle_alpha")),
        "cell_angle_beta": as_float(cif_scalar(cif, "_cell_angle_beta")),
        "cell_angle_gamma": as_float(cif_scalar(cif, "_cell_angle_gamma")),
        "cell_volume": as_float(cif_scalar(cif, "_cell_volume")),
        "cell_formula_units_Z": as_int(cif_scalar(cif, "_cell_formula_units_Z")),
        "match_label": None,
        "rmsd_label": None,
        "label_status": "pending",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MP-20 K50 selector input features from the validation K100 JSONL.")
    parser.add_argument("--input", default=str(ROOT / "candidates/crystallm_gt_sg_mp20_val_k100.jsonl"))
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--out", default=str(ROOT / "features/mp20_val_k50_candidate_features.jsonl"))
    parser.add_argument("--summary", default=str(ROOT / "metrics/mp20_val_k50_selector_input_summary.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mp20_k50_selector_inputs.md"))
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = under_root(Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    sample_ids: set[str] = set()
    material_ids: set[str] = set()
    rank_counts: Counter[int] = Counter()
    config_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    rows_by_sample: defaultdict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total_rows += 1
            record = json.loads(line)
            rank = int(record.get("rank", 0))
            if rank < 1 or rank > int(args.max_rank):
                continue
            row = build_row(record)
            fout.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            kept_rows += 1
            sample_id = str(row["sample_id"])
            material_id = str(row["material_id"])
            sample_ids.add(sample_id)
            material_ids.add(material_id)
            rows_by_sample[sample_id] += 1
            rank_counts[int(row["rank"])] += 1
            config_counts[str(row.get("generation_config_name"))] += 1
            source_counts[str(row.get("rank_source"))] += 1
            for key, value in row.items():
                if value is None and key not in {"normalized_token_logprob", "match_label", "rmsd_label"}:
                    missing_fields[key] += 1

    incomplete_samples = sorted(sample for sample, count in rows_by_sample.items() if count != int(args.max_rank))
    expected_rows = len(sample_ids) * int(args.max_rank)
    summary = {
        "created_at": now_iso(),
        "input": str(input_path.resolve()),
        "output": str(out_path.resolve()),
        "total_input_rows": total_rows,
        "kept_rows": kept_rows,
        "max_rank": int(args.max_rank),
        "sample_count": len(sample_ids),
        "material_count": len(material_ids),
        "expected_rows": expected_rows,
        "coverage_complete": kept_rows == expected_rows and not incomplete_samples,
        "incomplete_sample_count": len(incomplete_samples),
        "first_incomplete_samples": incomplete_samples[:20],
        "rank_counts": {str(k): rank_counts[k] for k in sorted(rank_counts)},
        "generation_config_counts": dict(sorted(config_counts.items())),
        "rank_source_counts": dict(sorted(source_counts.items())),
        "missing_field_counts": dict(sorted(missing_fields.items())),
        "label_status": "pending",
        "note": "This is a leakage-safe feature/input table only. StructureMatcher labels are intentionally not computed here.",
    }
    write_json(Path(args.summary), summary)

    lines = [
        "# MP-20 K50 Selector Inputs",
        "",
        f"Created: {summary['created_at']}",
        "",
        f"- Input rows scanned: {total_rows}",
        f"- K<={int(args.max_rank)} rows kept: {kept_rows}",
        f"- Samples/materials: {len(sample_ids)} / {len(material_ids)}",
        f"- Coverage complete: {summary['coverage_complete']}",
        f"- Output: `{out_path}`",
        f"- Summary: `{Path(args.summary)}`",
        "",
        "Labels are pending. This artifact is intended to feed a separate timeout-safe labeling step and reranker training.",
    ]
    write_text(Path(args.report), "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
