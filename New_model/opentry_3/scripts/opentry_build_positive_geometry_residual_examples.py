#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_render_wyckoff_cifs_e07e08 as rb  # noqa: E402
import opentry_train_source_residual_geometry as train_resid  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_wa_key(wa_key: str) -> list[tuple[str, str]]:
    chunks = str(wa_key or "").split("|setting=")
    out: list[tuple[str, str]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk if idx == 0 else "setting=" + chunk
        if ":" not in text:
            continue
        orbit_id, element = text.rsplit(":", 1)
        out.append((orbit_id, element))
    return out


def rows_from_wa_key(engine: OrbitEngine, wa_key: str, params_by_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, (orbit_id, element) in enumerate(split_wa_key(wa_key)):
        orbit = engine.get_orbit_by_id(str(orbit_id))
        raw_params = dict(params_by_row.get(str(row_idx)) or params_by_row.get(int(row_idx), {}) or {})
        free_params = {str(k): float(v) % 1.0 for k, v in raw_params.items()}
        rows.append(
            {
                "element": str(element),
                "orbit_id": orbit.canonical_orbit_id,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
                "free_params": free_params,
            }
        )
    return v2.canonical_rows({"wa_table": rows})


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def build_examples(
    *,
    split: str,
    features_path: Path,
    rendered_path: Path,
    repr_path: Path,
    train_records: list[dict[str, Any]],
    engine: OrbitEngine,
    min_target_row_count: int,
    max_examples: int,
    require_wa_hit: bool,
    source_pool_k: int,
    vpa_strength: float,
    complex_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rendered_by_key = {
        (str(row.get("sample_id") or ""), int(row.get("rank") or 0)): row
        for row in read_jsonl(rendered_path)
    }
    repr_by_id = {str(row["keys"]["sample_id"]): row for row in read_jsonl(repr_path)}
    selector = train_resid.build_selector(train_records)
    examples: list[dict[str, Any]] = []
    stats = {
        "split": split,
        "feature_rows": 0,
        "positive_rows": 0,
        "kept_examples": 0,
        "skipped_no_render": 0,
        "skipped_no_metadata": 0,
        "skipped_no_repr": 0,
        "skipped_no_source": 0,
        "rows_ge_7_examples": 0,
        "atoms_ge_12_examples": 0,
    }
    for feature in read_jsonl(features_path):
        stats["feature_rows"] += 1
        if str(feature.get("split") or split) != split:
            raise SystemExit(f"Expected split={split}, got {feature.get('split')!r}")
        if not bool(feature.get("label_match")):
            continue
        if require_wa_hit and not bool(feature.get("candidate_wa_hit")):
            continue
        stats["positive_rows"] += 1
        target_row_count = int(feature.get("target_row_count") or feature.get("candidate_row_count") or 0)
        if int(min_target_row_count) > 0 and target_row_count < int(min_target_row_count):
            continue
        sample_id = str(feature.get("sample_id") or "")
        rendered = rendered_by_key.get((sample_id, int(feature.get("rank") or 0)))
        if rendered is None:
            stats["skipped_no_render"] += 1
            continue
        lattice = dict(rendered.get("geometry_lattice") or {})
        params = dict(rendered.get("geometry_params") or {})
        if not lattice or not params:
            stats["skipped_no_metadata"] += 1
            continue
        repr_row = repr_by_id.get(sample_id)
        if repr_row is None:
            stats["skipped_no_repr"] += 1
            continue
        wa_key = str(feature.get("canonical_wa_key") or rendered.get("canonical_wa_key") or "")
        rows = rows_from_wa_key(engine, wa_key, params)
        record = {
            "sample_id": sample_id,
            "material_id": feature.get("material_id") or repr_row["keys"].get("material_id"),
            "sg": int(feature.get("sg") or repr_row["sg"]),
            "formula_counts": dict(repr_row["formula_counts"]),
            "wa_table": rows,
            "lattice": {key: float(lattice[key]) for key in ("a", "b", "c", "alpha", "beta", "gamma")},
            "n_sites": len(rows),
            "num_elements": len(dict(repr_row["formula_counts"])),
        }
        exclude = sample_id if split == "train" else None
        source = train_resid.select_source(
            selector,
            record,
            source_pool_k=int(source_pool_k),
            exclude_sample_id=exclude,
        )
        if source is None:
            stats["skipped_no_source"] += 1
            continue
        base_params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
        base_lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
        sample_weight = train_resid.complex_sample_weight(record, float(complex_weight))
        if int(target_row_count) >= 7:
            stats["rows_ge_7_examples"] += 1
        if rb.gb.atom_count(record) >= 12:
            stats["atoms_ge_12_examples"] += 1
        examples.append(
            {
                "record": record,
                "base_params": base_params,
                "base_lattice": base_lattice,
                "source_sample_id": str(source.get("sample_id")),
                "source_distance": float(rb.gb.row_condition_distance(record, source)),
                "align_cost": float(align_cost),
                "sample_weight": float(sample_weight),
                "label_source": "StructureMatcher-positive rendered candidate from train/val only",
                "source_feature_rank": int(feature.get("rank") or 0),
                "source_candidate_uid": str(feature.get("candidate_uid") or ""),
            }
        )
        if int(max_examples) > 0 and len(examples) >= int(max_examples):
            break
    stats["kept_examples"] = len(examples)
    return examples, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached source-residual examples from StructureMatcher-positive rendered candidate geometry.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--train-repr-jsonl", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--val-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--val-repr-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-val-examples", type=int, default=0)
    parser.add_argument("--require-wa-hit", action="store_true")
    parser.add_argument("--source-pool-k", type=int, default=24)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--complex-weight", type=float, default=6.0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_records = rb.load_records(args.data_root / "train.jsonl")
    engine = OrbitEngine(args.lookup_json)
    train_examples, train_stats = build_examples(
        split="train",
        features_path=args.train_features,
        rendered_path=args.train_rendered_jsonl,
        repr_path=args.train_repr_jsonl,
        train_records=train_records,
        engine=engine,
        min_target_row_count=int(args.min_target_row_count),
        max_examples=int(args.max_train_examples),
        require_wa_hit=bool(args.require_wa_hit),
        source_pool_k=int(args.source_pool_k),
        vpa_strength=float(args.vpa_strength),
        complex_weight=float(args.complex_weight),
    )
    val_examples, val_stats = build_examples(
        split="val",
        features_path=args.val_features,
        rendered_path=args.val_rendered_jsonl,
        repr_path=args.val_repr_jsonl,
        train_records=train_records,
        engine=engine,
        min_target_row_count=int(args.min_target_row_count),
        max_examples=int(args.max_val_examples),
        require_wa_hit=bool(args.require_wa_hit),
        source_pool_k=int(args.source_pool_k),
        vpa_strength=float(args.vpa_strength),
        complex_weight=1.0,
    )
    write_jsonl(out_dir / "train_examples.jsonl", train_examples)
    write_jsonl(out_dir / "val_examples.jsonl", val_examples)
    summary = {
        "config": {
            "min_target_row_count": int(args.min_target_row_count),
            "require_wa_hit": bool(args.require_wa_hit),
            "source_pool_k": int(args.source_pool_k),
            "vpa_strength": float(args.vpa_strength),
            "complex_weight": float(args.complex_weight),
        },
        "train": train_stats,
        "val": val_stats,
        "note": "Labels and target geometry come only from train/val StructureMatcher-positive rendered candidates; no test records used.",
    }
    write_json(out_dir / "positive_geometry_residual_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
