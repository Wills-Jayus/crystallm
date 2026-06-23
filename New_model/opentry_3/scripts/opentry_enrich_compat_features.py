#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
CELL_RE = re.compile(r"^_cell_(length_[abc]|angle_alpha|angle_beta|angle_gamma|volume)\s+([0-9eE+\-.]+)", re.MULTILINE)
EL_RE = re.compile(r":([A-Z][a-z]?)")


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_repr(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row["keys"]["sample_id"])
        out[sample_id] = row
    return out


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def formula_features(target: dict[str, int], source: dict[str, int]) -> dict[str, float]:
    elements = sorted(set(target) | set(source))
    total_t = max(1, sum(int(v) for v in target.values()))
    total_s = max(1, sum(int(v) for v in source.values()))
    l1_frac = 0.0
    dot = 0.0
    norm_t = 0.0
    norm_s = 0.0
    common = 0
    for el in elements:
        tv = float(target.get(el, 0)) / float(total_t)
        sv = float(source.get(el, 0)) / float(total_s)
        l1_frac += abs(tv - sv)
        dot += tv * sv
        norm_t += tv * tv
        norm_s += sv * sv
        common += int(el in target and el in source)
    cosine = 0.0 if norm_t <= 0 or norm_s <= 0 else dot / math.sqrt(norm_t * norm_s)
    union = max(1, len(elements))
    return {
        "src_formula_l1_frac": l1_frac,
        "src_formula_cosine": cosine,
        "src_formula_element_jaccard": common / union,
        "src_formula_element_overlap": common,
        "src_atom_count_ratio": total_s / float(total_t),
        "src_atom_count_absdiff": abs(total_s - total_t),
    }


def parse_cell(cif: str) -> dict[str, float]:
    values = {key: safe_float(value) for key, value in CELL_RE.findall(str(cif or ""))}
    a = values.get("length_a", 0.0)
    b = values.get("length_b", 0.0)
    c = values.get("length_c", 0.0)
    lengths = [x for x in (a, b, c) if x > 0]
    angles = [values.get("angle_alpha", 90.0), values.get("angle_beta", 90.0), values.get("angle_gamma", 90.0)]
    max_len = max(lengths) if lengths else 0.0
    min_len = min(lengths) if lengths else 0.0
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "cell_a": a,
        "cell_b": b,
        "cell_c": c,
        "cell_alpha": angles[0],
        "cell_beta": angles[1],
        "cell_gamma": angles[2],
        "cell_volume_header": values.get("volume", 0.0),
        "cell_len_ratio_max_min": 0.0 if min_len <= 0 else max_len / min_len,
        "cell_len_cv": 0.0 if mean_len <= 0 else math.sqrt(sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) / mean_len,
        "cell_angle_absdev_90_mean": sum(abs(x - 90.0) for x in angles) / 3.0,
        "cell_angle_absdev_90_max": max(abs(x - 90.0) for x in angles),
    }


def candidate_elements(wa_key: str) -> set[str]:
    return set(EL_RE.findall(str(wa_key or "")))


def enrich_row(row: dict[str, Any], target_repr: dict[str, Any] | None, source_repr: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(row)
    target_formula = {str(k): int(v) for k, v in (target_repr or {}).get("formula_counts", {}).items()}
    source_formula = {str(k): int(v) for k, v in (source_repr or {}).get("formula_counts", {}).items()}
    out.update(parse_cell(str(row.get("cif") or "")))
    out["src_found"] = source_repr is not None
    if source_repr is not None:
        src_sg = int(source_repr.get("sg", 0))
        src_row_count = int(source_repr.get("row_count", 0))
        src_atom_count = int(source_repr.get("atom_count", 0))
        out.update(
            {
                "src_sg": src_sg,
                "src_same_sg": int(src_sg == int(row.get("sg", 0))),
                "src_crystal_system": crystal_system(src_sg),
                "src_atom_count": src_atom_count,
                "src_row_count": src_row_count,
                "src_rows_ge_7": int(src_row_count >= 7),
                "src_atom_ge_12": int(src_atom_count >= 12),
                "src_complex_flag": int(bool(source_repr.get("complex_flag"))),
                "src_num_elements": int(source_repr.get("num_elements", len(source_formula))),
                "src_candidate_row_delta": src_row_count - int(row.get("candidate_row_count", 0) or 0),
                "src_candidate_row_absdiff": abs(src_row_count - int(row.get("candidate_row_count", 0) or 0)),
                "src_skeleton_eq_candidate": int(str(source_repr.get("canonical_skeleton_key", "")) == str(row.get("canonical_skeleton_key", ""))),
                "src_wa_eq_candidate": int(str(source_repr.get("canonical_wa_key", "")) == str(row.get("canonical_wa_key", ""))),
            }
        )
        out.update(formula_features(target_formula, source_formula))
        src_elements = set(source_formula)
        cand_elements = candidate_elements(str(row.get("canonical_wa_key") or ""))
        union = max(1, len(src_elements | cand_elements))
        out["src_candidate_element_jaccard"] = len(src_elements & cand_elements) / union
    else:
        out.update(
            {
                "src_sg": 0,
                "src_same_sg": 0,
                "src_crystal_system": "missing",
                "src_atom_count": 0,
                "src_row_count": 0,
                "src_rows_ge_7": 0,
                "src_atom_ge_12": 0,
                "src_complex_flag": 0,
                "src_num_elements": 0,
                "src_candidate_row_delta": 0,
                "src_candidate_row_absdiff": 0,
                "src_skeleton_eq_candidate": 0,
                "src_wa_eq_candidate": 0,
                "src_candidate_element_jaccard": 0.0,
            }
        )
        out.update(formula_features(target_formula, {}))
    out["candidate_rows_ge_7"] = int(int(row.get("candidate_row_count", 0) or 0) >= 7)
    out["candidate_atom_density_proxy"] = safe_float(row.get("atom_count_after_expansion", row.get("atom_count", 0))) / max(
        1.0, safe_float(row.get("self_volume", 0.0), default=0.0)
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Add GT-free source-pair and rendered-CIF context features to compatibility rows.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--target-repr-jsonl", type=Path, required=True)
    parser.add_argument("--source-train-repr-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "train.jsonl")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    out_jsonl = ensure_under_opentry(args.out_jsonl)
    summary_json = ensure_under_opentry(args.summary_json)
    rows = read_jsonl(args.features_jsonl)
    target_repr = load_repr(args.target_repr_jsonl)
    source_repr = load_repr(args.source_train_repr_jsonl)
    enriched: list[dict[str, Any]] = []
    missing_source = 0
    for row in rows:
        sample_id = str(row.get("sample_id"))
        source_id = str(row.get("source_sample_id") or "")
        src = source_repr.get(source_id)
        missing_source += int(src is None)
        enriched.append(enrich_row(row, target_repr.get(sample_id), src))
    write_jsonl(out_jsonl, enriched)
    summary = {
        "features_jsonl": str(args.features_jsonl),
        "target_repr_jsonl": str(args.target_repr_jsonl),
        "source_train_repr_jsonl": str(args.source_train_repr_jsonl),
        "rows": len(enriched),
        "missing_source_rows": missing_source,
        "source_found_rate": 1.0 - missing_source / max(1, len(enriched)),
        "added_feature_groups": [
            "source_formula_similarity",
            "source_row_atom_sg_context",
            "source_candidate_skeleton_wa_relation",
            "rendered_cell_shape",
        ],
        "no_leakage_note": "Uses target formula/SG from split representation and train-source metadata only; target row_count/keys remain labels/reporting and should be blocked by trainers.",
    }
    write_json(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
