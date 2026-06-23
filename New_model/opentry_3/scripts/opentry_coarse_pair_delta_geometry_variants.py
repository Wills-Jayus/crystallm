#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()

for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(path))

import opentry_pair_delta_geometry_variants as exact_delta  # noqa: E402
import opentry_build_geometry_compat_features as compat_features  # noqa: E402
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


def wa_row_count(row: dict[str, Any]) -> int:
    return len(pos_geom.split_wa_key(str(row.get("canonical_wa_key") or "")))


def row_bucket(row_count: int) -> str:
    n = int(row_count)
    if n <= 4:
        return "r1_4"
    if n <= 6:
        return "r5_6"
    if n <= 8:
        return "r7_8"
    if n <= 10:
        return "r9_10"
    return "r11_plus"


def free_symbols(row: dict[str, Any]) -> tuple[str, ...]:
    params = dict(row.get("geometry_params") or {})
    symbols: list[str] = []
    for values in params.values():
        symbols.extend(str(key) for key in dict(values or {}).keys())
    return tuple(sorted(symbols))


def candidate_keys(row: dict[str, Any], sg: int) -> list[tuple[Any, ...]]:
    row_count = wa_row_count(row)
    system = compat_features.crystal_system(int(sg))
    bucket = row_bucket(row_count)
    symbols = free_symbols(row)
    return [
        ("sg_row_symbols", int(sg), row_count, symbols),
        ("sg_row", int(sg), row_count),
        ("system_row_symbols", system, row_count, symbols),
        ("system_row", system, row_count),
        ("sg_bucket_symbols", int(sg), bucket, symbols),
        ("sg_bucket", int(sg), bucket),
        ("system_bucket_symbols", system, bucket, symbols),
        ("system_bucket", system, bucket),
        ("row", row_count),
        ("bucket", bucket),
    ]


def pair_id(pos: dict[str, Any], neg: dict[str, Any]) -> str:
    return f"{pos.get('candidate_uid') or pos.get('rank')}<<{neg.get('candidate_uid') or neg.get('rank')}"


def build_delta_bank(
    *,
    train_features: list[dict[str, Any]],
    min_target_row_count: int,
    negative_mode: str,
    max_positive_rmsd: float,
    max_pairs_per_key: int,
    max_row_count_delta: int,
    lattice_only: bool,
) -> tuple[dict[tuple[Any, ...], list[dict[str, Any]]], dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_rows = 0
    for row in train_features:
        if int(row.get("target_row_count") or 0) < int(min_target_row_count):
            continue
        if not row.get("geometry_lattice") or not row.get("geometry_params"):
            continue
        by_sample[str(row.get("sample_id") or "")].append(row)
        used_rows += 1

    bank: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    raw_pairs = 0
    kept_pair_ids: set[str] = set()
    positive_rows = 0
    negative_rows = 0
    for sample_id, rows in by_sample.items():  # noqa: B007
        positives = [row for row in rows if bool(row.get("label_match"))]
        if max_positive_rmsd > 0:
            positives = [
                row
                for row in positives
                if row.get("label_rmsd") is not None and float(row["label_rmsd"]) <= float(max_positive_rmsd)
            ]
        negatives = [row for row in rows if not bool(row.get("label_match"))]
        if negative_mode == "wa_hit":
            negatives = [row for row in negatives if bool(row.get("candidate_wa_hit"))]
        elif negative_mode == "skeleton_hit":
            negatives = [row for row in negatives if bool(row.get("candidate_skeleton_hit"))]
        if not positives or not negatives:
            continue
        positive_rows += len(positives)
        negative_rows += len(negatives)
        for pos in positives:
            pos_row_count = wa_row_count(pos)
            pos_lat = exact_delta.lattice_vector(dict(pos["geometry_lattice"]))
            for neg in negatives:
                neg_row_count = wa_row_count(neg)
                if int(max_row_count_delta) >= 0 and abs(pos_row_count - neg_row_count) > int(max_row_count_delta):
                    continue
                neg_lat = exact_delta.lattice_vector(dict(neg["geometry_lattice"]))
                d_lat = [float(a - b) for a, b in zip(pos_lat, neg_lat)]
                d_params = {} if lattice_only else exact_delta.param_delta(pos, neg)
                raw_pairs += 1
                item = {
                    "delta_lattice": d_lat,
                    "delta_params": d_params,
                    "delta_norm": exact_delta.delta_norm(d_params, d_lat),
                    "positive_rmsd": None if pos.get("label_rmsd") is None else float(pos["label_rmsd"]),
                    "train_positive_uid": str(pos.get("candidate_uid") or ""),
                    "train_negative_uid": str(neg.get("candidate_uid") or ""),
                    "train_sample_id": str(pos.get("sample_id") or ""),
                    "train_positive_rank": int(pos.get("rank", 0)),
                    "train_negative_rank": int(neg.get("rank", 0)),
                    "train_positive_row_count": int(pos_row_count),
                    "train_negative_row_count": int(neg_row_count),
                    "pair_id": pair_id(pos, neg),
                }
                kept_pair_ids.add(str(item["pair_id"]))
                sg = int(neg.get("sg") or pos.get("sg") or 0)
                for key in candidate_keys(neg, sg):
                    bank[key].append(item)

    for key, items in list(bank.items()):
        items.sort(
            key=lambda item: (
                float("inf") if item["positive_rmsd"] is None else float(item["positive_rmsd"]),
                float(item["delta_norm"]),
                str(item["pair_id"]),
            )
        )
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            pid = str(item["pair_id"])
            if pid in seen:
                continue
            seen.add(pid)
            deduped.append(item)
            if len(deduped) >= int(max_pairs_per_key):
                break
        bank[key] = deduped

    return dict(bank), {
        "train_rows_seen": used_rows,
        "train_samples_with_rows": len(by_sample),
        "positive_rows_used": positive_rows,
        "negative_rows_used": negative_rows,
        "raw_pairs": raw_pairs,
        "unique_pair_ids": len(kept_pair_ids),
        "keys": len(bank),
        "key_type_counts": dict(
            sorted(
                {
                    str(key[0]): sum(1 for candidate_key in bank if str(candidate_key[0]) == str(key[0]))
                    for key in bank
                }.items()
            )
        ),
    }


def select_deltas(
    *,
    bank: dict[tuple[Any, ...], list[dict[str, Any]]],
    row: dict[str, Any],
    sg: int,
    max_deltas: int,
    key_type_stats: dict[str, int],
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    out: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen: set[str] = set()
    for key in candidate_keys(row, sg):
        items = bank.get(key) or []
        if items:
            key_type_stats[str(key[0])] += 1
        for item in items:
            pid = str(item["pair_id"])
            if pid in seen:
                continue
            seen.add(pid)
            out.append((key, item))
            if len(out) >= int(max_deltas):
                return out
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate coarse-key negative-to-positive geometry delta variants.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--negative-mode", choices=["all", "wa_hit", "skeleton_hit"], default="all")
    parser.add_argument("--max-positive-rmsd", type=float, default=0.0)
    parser.add_argument("--max-row-count-delta", type=int, default=3)
    parser.add_argument("--max-pairs-per-key", type=int, default=64)
    parser.add_argument("--max-deltas-per-candidate", type=int, default=4)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--max-log-scale", type=float, default=0.12)
    parser.add_argument("--lattice-only", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_features = read_jsonl(args.train_features)
    bank, bank_summary = build_delta_bank(
        train_features=train_features,
        min_target_row_count=int(args.min_target_row_count),
        negative_mode=str(args.negative_mode),
        max_positive_rmsd=float(args.max_positive_rmsd),
        max_pairs_per_key=int(args.max_pairs_per_key),
        max_row_count_delta=int(args.max_row_count_delta),
        lattice_only=bool(args.lattice_only),
    )
    repr_rows = load_repr(args.repr_jsonl, max_records=int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))
    engine = OrbitEngine(args.lookup_json)
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    out_rows: list[dict[str, Any]] = []
    key_type_stats: dict[str, int] = defaultdict(int)
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "samples_with_input": 0,
        "samples_with_output": 0,
        "input_rows_seen": 0,
        "input_rows_with_delta": 0,
        "rows_ge_7_input_seen": 0,
        "rows_ge_7_input_with_delta": 0,
        "skipped_target_rows_lt_min": 0,
        **bank_summary,
    }
    for sample_id, repr_row in repr_rows.items():
        target_row_count = int(repr_row["row_count"])
        if target_row_count < int(args.min_target_row_count):
            stats["skipped_target_rows_lt_min"] += 1
            continue
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
            stats["rows_ge_7_input_seen"] += 1
            if not cand.get("geometry_lattice") or not cand.get("geometry_params"):
                continue
            selected = select_deltas(
                bank=bank,
                row=cand,
                sg=sg,
                max_deltas=int(args.max_deltas_per_candidate),
                key_type_stats=key_type_stats,
            )
            if not selected:
                continue
            stats["input_rows_with_delta"] += 1
            stats["rows_ge_7_input_with_delta"] += 1
            base_params = dict(cand["geometry_params"])
            base_lattice = dict(cand["geometry_lattice"])
            wa_key = str(cand.get("canonical_wa_key") or "")
            for delta_idx, (matched_key, delta) in enumerate(selected):
                for alpha in alphas:
                    params = exact_delta.apply_delta_params(base_params, dict(delta["delta_params"]), alpha)
                    lattice = exact_delta.apply_delta_lattice(
                        base_lattice,
                        list(delta["delta_lattice"]),
                        sg,
                        alpha,
                        float(args.max_log_scale),
                    )
                    rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
                    render = v2.render_candidate(
                        engine,
                        record_base,
                        {"rows": rows, "params": params, "lattice": lattice},
                        len(generated) + 1,
                        f"coarse_pair_delta_a{alpha:g}_d{delta_idx}",
                    )
                    if not render.get("ok"):
                        continue
                    cif = str(render.get("cif") or "")
                    digest = exact_delta.cif_digest(cif)
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
                            "geometry_source": "coarse_pair_delta",
                            "geometry_param_variant_mode": "coarse_pair_delta_lattice_only" if args.lattice_only else "coarse_pair_delta_full",
                            "geometry_lattice_mode": "coarse_pair_delta",
                            "coarse_pair_delta_alpha": float(alpha),
                            "coarse_pair_delta_index": int(delta_idx),
                            "coarse_pair_delta_key": repr(matched_key),
                            "coarse_pair_delta_key_type": str(matched_key[0]),
                            "coarse_pair_delta_train_positive_uid": delta["train_positive_uid"],
                            "coarse_pair_delta_train_negative_uid": delta["train_negative_uid"],
                            "coarse_pair_delta_train_sample_id": delta["train_sample_id"],
                            "coarse_pair_delta_norm": float(delta["delta_norm"]),
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
            "negative_mode": str(args.negative_mode),
            "max_positive_rmsd": float(args.max_positive_rmsd),
            "max_row_count_delta": int(args.max_row_count_delta),
            "max_pairs_per_key": int(args.max_pairs_per_key),
            "max_deltas_per_candidate": int(args.max_deltas_per_candidate),
            "alphas": alphas,
            "max_log_scale": float(args.max_log_scale),
            "lattice_only": bool(args.lattice_only),
        },
        "stats": {**stats, "matched_key_type_counts": dict(sorted(key_type_stats.items()))},
        "rendered_rows": len(out_rows),
        "note": "Bank uses only train StructureMatcher labels. Val inference uses GT-free candidate geometry metadata plus formula/GT-SG from representation.",
    }
    write_json(out_dir / "coarse_pair_delta_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
