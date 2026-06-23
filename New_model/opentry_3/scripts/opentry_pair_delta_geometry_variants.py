#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_positive_geometry_residual_examples as pos_geom  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


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
        out.setdefault(str(row.get("sample_id") or ""), []).append(row)
    for values in out.values():
        values.sort(key=lambda item: int(item.get("rank", item.get("original_rank", 10**9))))
    return out


def load_repr(path: Path, max_records: int = 0) -> OrderedDict[str, dict[str, Any]]:
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in read_jsonl(path):
        sample_id = str(row["keys"]["sample_id"])
        out[sample_id] = row
        if int(max_records) > 0 and len(out) >= int(max_records):
            break
    return out


def param_signature(row: dict[str, Any], sg: int | None = None) -> tuple[Any, ...]:
    wa_key = str(row.get("canonical_wa_key") or "")
    row_count = len(pos_geom.split_wa_key(wa_key))
    params = dict(row.get("geometry_params") or {})
    sequence: list[str] = []
    for idx in range(row_count):
        values = params.get(str(idx)) or params.get(idx) or {}
        sequence.append("".join(sorted(str(key) for key in dict(values).keys())))
    return (int(sg if sg is not None else row.get("sg") or 0), int(row_count), tuple(sequence))


def lattice_vector(lattice: dict[str, Any]) -> list[float]:
    return [
        math.log(max(1.0e-6, float(lattice["a"]))),
        math.log(max(1.0e-6, float(lattice["b"]))),
        math.log(max(1.0e-6, float(lattice["c"]))),
        float(lattice["alpha"]) / v2.ANGLE_SCALE,
        float(lattice["beta"]) / v2.ANGLE_SCALE,
        float(lattice["gamma"]) / v2.ANGLE_SCALE,
    ]


def wrapped_delta(pos_value: float, neg_value: float) -> float:
    delta = (float(pos_value) - float(neg_value)) % 1.0
    if delta > 0.5:
        delta -= 1.0
    return float(delta)


def param_delta(positive: dict[str, Any], negative: dict[str, Any]) -> dict[str, dict[str, float]]:
    pos_params = dict(positive.get("geometry_params") or {})
    neg_params = dict(negative.get("geometry_params") or {})
    out: dict[str, dict[str, float]] = {}
    for row_key, neg_row in neg_params.items():
        pos_row = dict(pos_params.get(str(row_key)) or pos_params.get(int(row_key), {}) or {})
        neg_values = dict(neg_row or {})
        deltas: dict[str, float] = {}
        for sym, neg_val in neg_values.items():
            if sym in pos_row:
                deltas[str(sym)] = wrapped_delta(float(pos_row[sym]), float(neg_val))
        out[str(row_key)] = deltas
    return out


def delta_norm(delta_params: dict[str, dict[str, float]], delta_lattice: list[float]) -> float:
    coord = sum(abs(float(v)) for row in delta_params.values() for v in row.values())
    latt = sum(abs(float(v)) for v in delta_lattice)
    return float(coord + latt)


def apply_delta_params(base_params: dict[str, Any], delta_params: dict[str, dict[str, float]], alpha: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row_key, row_values in dict(base_params or {}).items():
        new_row: dict[str, float] = {}
        row_delta = delta_params.get(str(row_key), {})
        for sym, value in dict(row_values or {}).items():
            new_row[str(sym)] = (float(value) + float(alpha) * float(row_delta.get(str(sym), 0.0))) % 1.0
        out[str(row_key)] = new_row
    return out


def apply_delta_lattice(base_lattice: dict[str, Any], delta_lattice: list[float], sg: int, alpha: float, max_log_scale: float) -> dict[str, float]:
    base = lattice_vector(base_lattice)
    values = []
    for idx, value in enumerate(base):
        raw = float(value) + float(alpha) * float(delta_lattice[idx])
        if idx < 3:
            lo = float(value) - float(max_log_scale)
            hi = float(value) + float(max_log_scale)
            raw = min(hi, max(lo, raw))
        values.append(raw)
    return v2.lattice_from_target(values, int(sg))


def cif_digest(cif: str) -> str:
    return hashlib.sha1(str(cif).encode("utf-8")).hexdigest()


def build_delta_bank(
    *,
    train_features: list[dict[str, Any]],
    min_target_row_count: int,
    max_pairs_per_signature: int,
) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], dict[str, Any]]:
    by_sample_sig: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in train_features:
        if int(row.get("target_row_count") or 0) < int(min_target_row_count):
            continue
        if not row.get("geometry_lattice") or not row.get("geometry_params"):
            continue
        by_sample_sig[(str(row.get("sample_id")), param_signature(row))].append(row)
    bank: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    pair_groups = 0
    raw_pairs = 0
    for (_sample_id, sig), rows in by_sample_sig.items():
        positives = [row for row in rows if bool(row.get("label_match"))]
        negatives = [row for row in rows if not bool(row.get("label_match"))]
        if not positives or not negatives:
            continue
        pair_groups += 1
        for pos in positives:
            pos_lat = lattice_vector(dict(pos["geometry_lattice"]))
            for neg in negatives:
                neg_lat = lattice_vector(dict(neg["geometry_lattice"]))
                d_lat = [float(a - b) for a, b in zip(pos_lat, neg_lat)]
                d_params = param_delta(pos, neg)
                raw_pairs += 1
                bank[sig].append(
                    {
                        "delta_lattice": d_lat,
                        "delta_params": d_params,
                        "delta_norm": delta_norm(d_params, d_lat),
                        "positive_rmsd": None if pos.get("label_rmsd") is None else float(pos["label_rmsd"]),
                        "train_positive_uid": str(pos.get("candidate_uid") or ""),
                        "train_negative_uid": str(neg.get("candidate_uid") or ""),
                        "train_sample_id": str(pos.get("sample_id") or ""),
                    }
                )
    kept_pairs = 0
    for sig, rows in list(bank.items()):
        rows.sort(key=lambda item: (float("inf") if item["positive_rmsd"] is None else float(item["positive_rmsd"]), float(item["delta_norm"])))
        bank[sig] = rows[: max(1, int(max_pairs_per_signature))]
        kept_pairs += len(bank[sig])
    return dict(bank), {
        "sample_signature_groups_with_pairs": pair_groups,
        "raw_pairs": raw_pairs,
        "kept_pairs": kept_pairs,
        "signatures": len(bank),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pair-delta geometry variants from train negative-to-positive candidate pairs.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--max-pairs-per-signature", type=int, default=6)
    parser.add_argument("--max-deltas-per-candidate", type=int, default=3)
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--max-log-scale", type=float, default=0.18)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_features = read_jsonl(args.train_features)
    bank, bank_summary = build_delta_bank(
        train_features=train_features,
        min_target_row_count=int(args.min_target_row_count),
        max_pairs_per_signature=int(args.max_pairs_per_signature),
    )
    repr_rows = load_repr(args.repr_jsonl, max_records=int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))
    engine = OrbitEngine(args.lookup_json)
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "samples_with_input": 0,
        "samples_with_output": 0,
        "input_rows_seen": 0,
        "input_rows_signature_covered": 0,
        "rows_ge_7_input_seen": 0,
        "rows_ge_7_signature_covered": 0,
        **bank_summary,
    }
    for sample_id, repr_row in repr_rows.items():
        candidates = rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        if candidates:
            stats["samples_with_input"] += 1
        generated: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        sg = int(repr_row["sg"])
        record_base = {
            "sample_id": sample_id,
            "formula_counts": dict(repr_row["formula_counts"]),
            "sg": sg,
            "sg_symbol": str(repr_row.get("sg_symbol") or ""),
            "atom_count": int(repr_row["atom_count"]),
        }
        for cand in candidates:
            stats["input_rows_seen"] += 1
            if int(repr_row["row_count"]) >= 7:
                stats["rows_ge_7_input_seen"] += 1
            if not cand.get("geometry_lattice") or not cand.get("geometry_params"):
                continue
            sig = param_signature(cand, sg=sg)
            deltas = bank.get(sig) or []
            if not deltas:
                continue
            stats["input_rows_signature_covered"] += 1
            if int(repr_row["row_count"]) >= 7:
                stats["rows_ge_7_signature_covered"] += 1
            base_params = dict(cand["geometry_params"])
            base_lattice = dict(cand["geometry_lattice"])
            wa_key = str(cand.get("canonical_wa_key") or "")
            for delta_idx, delta in enumerate(deltas[: int(args.max_deltas_per_candidate)]):
                for alpha in alphas:
                    params = apply_delta_params(base_params, dict(delta["delta_params"]), alpha)
                    lattice = apply_delta_lattice(base_lattice, list(delta["delta_lattice"]), sg, alpha, float(args.max_log_scale))
                    rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
                    render = v2.render_candidate(
                        engine,
                        record_base,
                        {"rows": rows, "params": params, "lattice": lattice},
                        len(generated) + 1,
                        f"pair_delta_a{alpha:g}_d{delta_idx}",
                    )
                    if not render.get("ok"):
                        continue
                    cif = str(render.get("cif") or "")
                    digest = cif_digest(cif)
                    if digest in seen_cifs:
                        continue
                    seen_cifs.add(digest)
                    metric = validate_cif(cif, record_base["formula_counts"], sg)
                    generated.append(
                        {
                            **{key: value for key, value in cand.items() if key not in {"rank", "cif"}},
                            "rank": len(generated) + 1,
                            "cif": cif,
                            "readable": bool(metric.get("readable")),
                            "formula_ok": bool(metric.get("formula_ok")),
                            "composition_exact": bool(metric.get("composition_exact")),
                            "atom_count_ok": bool(render.get("atom_count_ok")),
                            "detected_sg": metric.get("detected_sg"),
                            "sg_ok": bool(metric.get("sg_ok")),
                            "geometry_lattice": lattice,
                            "geometry_params": params,
                            "geometry_source": "pair_delta_signature",
                            "geometry_param_variant_mode": "pair_delta_signature",
                            "geometry_lattice_mode": "pair_delta_signature",
                            "pair_delta_alpha": float(alpha),
                            "pair_delta_index": int(delta_idx),
                            "pair_delta_signature": repr(sig),
                            "pair_delta_train_positive_uid": delta["train_positive_uid"],
                            "pair_delta_train_negative_uid": delta["train_negative_uid"],
                            "pair_delta_train_sample_id": delta["train_sample_id"],
                            "pair_delta_norm": float(delta["delta_norm"]),
                            "source_rank": int(cand.get("rank", 0)),
                            "original_rank": int(cand.get("rank", 0)),
                        }
                    )
                    if len(generated) >= int(args.max_output_candidates_per_sample):
                        break
                if len(generated) >= int(args.max_output_candidates_per_sample):
                    break
            if len(generated) >= int(args.max_output_candidates_per_sample):
                break
        if generated:
            stats["samples_with_output"] += 1
        out_rows.extend(generated[: int(args.max_output_candidates_per_sample)])

    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    summary = {
        "config": {
            "train_features": str(args.train_features),
            "input_rendered_jsonl": str(args.input_rendered_jsonl),
            "repr_jsonl": str(args.repr_jsonl),
            "max_records": int(args.max_records),
            "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "min_target_row_count": int(args.min_target_row_count),
            "max_pairs_per_signature": int(args.max_pairs_per_signature),
            "max_deltas_per_candidate": int(args.max_deltas_per_candidate),
            "alphas": alphas,
            "max_log_scale": float(args.max_log_scale),
        },
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Delta bank uses only train StructureMatcher labels. Inference uses GT-free candidate metadata plus formula/GT-SG from repr.",
    }
    write_json(out_dir / "pair_delta_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
