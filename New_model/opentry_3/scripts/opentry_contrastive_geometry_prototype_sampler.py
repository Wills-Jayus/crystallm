#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import joblib


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_geometry_compat_features as compat_features  # noqa: E402
import opentry_build_positive_geometry_residual_examples as pos_geom  # noqa: E402
import opentry_compat_guided_source_free_sampler as guided_sf  # noqa: E402
import opentry_pair_delta_geometry_variants as delta_utils  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import opentry_source_free_prior_geometry_sampler as sf_prior  # noqa: E402
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


def atom_bucket(n: int) -> str:
    n = int(n)
    if n < 12:
        return "lt12"
    if n < 24:
        return "a12_23"
    return "a24_plus"


def actual_row_count(row_or_wa: dict[str, Any] | str) -> int:
    wa_key = str(row_or_wa if isinstance(row_or_wa, str) else row_or_wa.get("canonical_wa_key") or "")
    return len(pos_geom.split_wa_key(wa_key))


def row_signature(engine: OrbitEngine, wa_key: str) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for orbit_id, element in pos_geom.split_wa_key(str(wa_key)):
        orbit = engine.get_orbit_by_id(str(orbit_id))
        sig.append(
            {
                "orbit_id": str(orbit_id),
                "element": str(element),
                "letter": str(orbit.letter),
                "multiplicity": int(orbit.multiplicity),
                "site_symmetry": str(orbit.site_symmetry),
                "free_symbols": tuple(str(x) for x in orbit.free_symbols),
            }
        )
    return sig


def context_keys(row: dict[str, Any], *, sg: int, atom_count: int, row_count: int) -> list[tuple[Any, ...]]:
    system = compat_features.crystal_system(int(sg))
    rb = sf_prior.row_bucket(int(row_count))
    ab = atom_bucket(int(atom_count))
    elem_count = int(row.get("formula_element_count") or 0)
    return [
        ("sg_row_atom", int(sg), rb, ab, elem_count),
        ("sg_row", int(sg), rb),
        ("system_row_atom", system, rb, ab),
        ("system_row", system, rb),
        ("row_atom", rb, ab),
        ("row", rb),
        ("global",),
    ]


def lattice_volume(lattice: dict[str, Any]) -> float:
    return sf_prior.lattice_volume(lattice)


def clean_lattice(lattice: dict[str, Any]) -> dict[str, float]:
    return {
        "a": float(lattice["a"]),
        "b": float(lattice["b"]),
        "c": float(lattice["c"]),
        "alpha": float(lattice["alpha"]),
        "beta": float(lattice["beta"]),
        "gamma": float(lattice["gamma"]),
    }


def clean_params(params: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row_key, row_values in dict(params or {}).items():
        out[str(row_key)] = {str(k): float(v) % 1.0 for k, v in dict(row_values or {}).items()}
    return out


def prototype_score(pos: int, total: int) -> float:
    return (float(pos) + 1.0) / (float(total) + 4.0)


class PrototypeBank:
    def __init__(self) -> None:
        self.stats: dict[tuple[Any, ...], dict[str, int]] = defaultdict(lambda: {"pos": 0, "total": 0})
        self.positives: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    def add(self, row: dict[str, Any]) -> None:
        if not row.get("geometry_lattice") or not row.get("geometry_params"):
            return
        sg = int(row.get("sg") or 0)
        atom_count = int(row.get("atom_count") or 0)
        rc = actual_row_count(row)
        for key in context_keys(row, sg=sg, atom_count=atom_count, row_count=rc):
            self.stats[key]["total"] += 1
            if bool(row.get("label_match")):
                self.stats[key]["pos"] += 1
                self.positives[key].append(row)

    def select(self, target_row: dict[str, Any], *, sg: int, atom_count: int, row_count: int, min_pos: int, min_score: float, limit: int) -> list[tuple[dict[str, Any], tuple[Any, ...], float]]:
        selected: list[tuple[dict[str, Any], tuple[Any, ...], float]] = []
        seen: set[str] = set()
        for key in context_keys(target_row, sg=sg, atom_count=atom_count, row_count=row_count):
            stat = self.stats.get(key) or {"pos": 0, "total": 0}
            score = prototype_score(int(stat["pos"]), int(stat["total"]))
            if int(stat["pos"]) < int(min_pos) or score < float(min_score):
                continue
            values = self.positives.get(key) or []
            for row in sorted(values, key=lambda item: (int(item.get("rank", 9999)), str(item.get("sample_id")))):
                uid = str(row.get("candidate_uid") or f"{row.get('sample_id')}::{row.get('rank')}")
                if uid in seen:
                    continue
                seen.add(uid)
                selected.append((row, key, score))
                if len(selected) >= int(limit):
                    return selected
        if not selected:
            values = self.positives.get(("global",), [])
            for row in values[: int(limit)]:
                selected.append((row, ("global_fallback",), 0.0))
        return selected[: int(limit)]

    def summary(self) -> dict[str, Any]:
        nonempty = {key: stat for key, stat in self.stats.items() if stat["total"] > 0}
        return {
            "keys": len(nonempty),
            "keys_with_positive": sum(int(stat["pos"] > 0) for stat in nonempty.values()),
            "rows_total": sum(int(stat["total"]) for stat in nonempty.values()),
            "rows_positive": sum(int(stat["pos"]) for stat in nonempty.values()),
        }


def build_bank(train_features: list[dict[str, Any]]) -> PrototypeBank:
    bank = PrototypeBank()
    for row in train_features:
        bank.add(row)
    return bank


def map_params(engine: OrbitEngine, target_wa_key: str, prototype_wa_key: str, prototype_params: dict[str, Any]) -> dict[str, dict[str, float]]:
    target_sig = row_signature(engine, target_wa_key)
    proto_sig = row_signature(engine, prototype_wa_key)
    proto_params = clean_params(prototype_params)
    used: set[int] = set()
    out: dict[str, dict[str, float]] = {}
    for target_idx, target in enumerate(target_sig):
        target_symbols = tuple(target["free_symbols"])
        best_idx: int | None = None
        for idx, proto in enumerate(proto_sig):
            if idx in used:
                continue
            if tuple(proto["free_symbols"]) == target_symbols and proto["site_symmetry"] == target["site_symmetry"]:
                best_idx = idx
                break
        if best_idx is None:
            for idx, proto in enumerate(proto_sig):
                if idx in used:
                    continue
                if tuple(proto["free_symbols"]) == target_symbols:
                    best_idx = idx
                    break
        row_values: dict[str, float] = {}
        if best_idx is not None:
            used.add(best_idx)
            source_values = proto_params.get(str(best_idx), {})
            for symbol in target_symbols:
                if str(symbol) in source_values:
                    row_values[str(symbol)] = float(source_values[str(symbol)]) % 1.0
        for symbol in target_symbols:
            row_values.setdefault(str(symbol), 0.5)
        out[str(target_idx)] = row_values
    return out


def scale_prototype_lattice(proto_lattice: dict[str, Any], *, sg: int, atom_count: int, proto_atom_count: int, jitter_token: str, jitter: float) -> dict[str, float]:
    vec = delta_utils.lattice_vector(clean_lattice(proto_lattice))
    proto_vpa = lattice_volume(proto_lattice) / max(1, int(proto_atom_count))
    return sf_prior.scale_lattice(vec, int(sg), int(atom_count), proto_vpa, jitter_token, float(jitter))


def make_rendered(
    *,
    engine: OrbitEngine,
    sample_id: str,
    repr_row: dict[str, Any],
    source_row: dict[str, Any],
    proto_row: dict[str, Any],
    proto_key: tuple[Any, ...],
    proto_score_value: float,
    variant_idx: int,
    lattice_jitter: float,
) -> dict[str, Any] | None:
    wa_key = str(source_row.get("canonical_wa_key") or "")
    if not wa_key:
        return None
    sg = int(repr_row["sg"])
    record_base = {
        "sample_id": sample_id,
        "formula_counts": dict(repr_row["formula_counts"]),
        "sg": sg,
        "sg_symbol": str(repr_row.get("sg_symbol") or ""),
        "atom_count": int(repr_row["atom_count"]),
    }
    params = map_params(
        engine,
        wa_key,
        str(proto_row.get("canonical_wa_key") or ""),
        dict(proto_row.get("geometry_params") or {}),
    )
    lattice = scale_prototype_lattice(
        dict(proto_row.get("geometry_lattice") or {}),
        sg=sg,
        atom_count=int(repr_row["atom_count"]),
        proto_atom_count=int(proto_row.get("atom_count") or 1),
        jitter_token=f"{sample_id}:{source_row.get('rank')}:{variant_idx}:{proto_row.get('candidate_uid')}",
        jitter=float(lattice_jitter),
    )
    rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
    render = v2.render_candidate(
        engine,
        record_base,
        {"rows": rows, "params": params, "lattice": lattice},
        int(variant_idx) + 1,
        f"contrastive_proto_{variant_idx}",
    )
    if not render.get("ok"):
        return None
    cif = str(render.get("cif") or "")
    metric = validate_cif(cif, record_base["formula_counts"], sg)
    self_features = selfscore.cif_self_features(cif)
    item = {
        **guided_sf.safe_copy_source_fields(source_row),
        **compat_features.parse_wa_metadata(wa_key),
        **self_features,
        "sample_id": sample_id,
        "rank": 0,
        "cif": cif,
        "readable": bool(metric.get("readable")),
        "formula_ok": bool(metric.get("formula_ok")),
        "composition_exact": bool(metric.get("composition_exact")),
        "atom_count_ok": bool(render.get("atom_count_ok")),
        "detected_sg": metric.get("detected_sg"),
        "sg_ok": bool(metric.get("sg_ok")),
        "sg": sg,
        "crystal_system": compat_features.crystal_system(sg),
        "atom_count": int(repr_row["atom_count"]),
        "formula_element_count": len(dict(repr_row["formula_counts"])),
        "canonical_wa_key": wa_key,
        "canonical_skeleton_key": str(source_row.get("canonical_skeleton_key") or ""),
        "geometry_lattice": lattice,
        "geometry_params": params,
        "geometry_source": "contrastive_train_geometry_prototype",
        "geometry_param_variant_mode": "contrastive_prototype",
        "geometry_lattice_mode": "contrastive_prototype_scaled",
        "prototype_key": repr(proto_key),
        "prototype_score": float(proto_score_value),
        "prototype_sample_id": str(proto_row.get("sample_id") or ""),
        "prototype_candidate_uid": str(proto_row.get("candidate_uid") or ""),
        "prototype_label_match": bool(proto_row.get("label_match")),
        "source_rank": int(source_row.get("rank", 0)),
        "original_rank": int(source_row.get("rank", 0)),
    }
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Contrastive train-positive/negative geometry prototype sampler.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=12)
    parser.add_argument("--prototypes-per-wa", type=int, default=4)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-per-wa", type=int, default=5)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--min-prototype-positives", type=int, default=2)
    parser.add_argument("--min-prototype-score", type=float, default=0.05)
    parser.add_argument("--lattice-jitter", type=float, default=0.02)
    parser.add_argument("--model-kind", choices=["gbdt", "gbdt_strict"], default="gbdt")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    model_out = ensure_under_opentry(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    train_features = read_jsonl(args.train_features)
    bank = build_bank(train_features)
    model, model_summary = guided_sf.train_compat_model(
        train_features,
        model_kind=str(args.model_kind),
        positive_weight=5.0,
        candidate_row7_weight=2.0,
        atom12_weight=1.25,
    )
    joblib.dump(model, model_out / "compat_model.joblib")

    engine = OrbitEngine(args.lookup_json)
    repr_rows = load_repr(args.repr_jsonl, max_records=int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))
    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "skipped_target_rows_lt_min": 0,
        "samples_with_input": 0,
        "samples_with_output": 0,
        "input_wa_candidates_seen": 0,
        "prototype_rows_considered": 0,
        "render_attempts": 0,
        "render_ok": 0,
        "proposal_rows_scored": 0,
        "deduped_cifs": 0,
        **{f"bank_{k}": v for k, v in bank.summary().items()},
    }
    for sample_id, repr_row in repr_rows.items():
        if int(repr_row["row_count"]) < int(args.min_target_row_count):
            stats["skipped_target_rows_lt_min"] += 1
            continue
        input_rows = rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        if input_rows:
            stats["samples_with_input"] += 1
        proposals: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        seen_wa: set[str] = set()
        for source_row in input_rows:
            wa_key = str(source_row.get("canonical_wa_key") or "")
            if not wa_key or wa_key in seen_wa:
                continue
            seen_wa.add(wa_key)
            stats["input_wa_candidates_seen"] += 1
            prototypes = bank.select(
                source_row,
                sg=int(repr_row["sg"]),
                atom_count=int(repr_row["atom_count"]),
                row_count=actual_row_count(source_row),
                min_pos=int(args.min_prototype_positives),
                min_score=float(args.min_prototype_score),
                limit=int(args.prototypes_per_wa),
            )
            stats["prototype_rows_considered"] += len(prototypes)
            for proto_idx, (proto_row, proto_key, proto_score_value) in enumerate(prototypes):
                stats["render_attempts"] += 1
                rendered = make_rendered(
                    engine=engine,
                    sample_id=sample_id,
                    repr_row=repr_row,
                    source_row=source_row,
                    proto_row=proto_row,
                    proto_key=proto_key,
                    proto_score_value=float(proto_score_value),
                    variant_idx=proto_idx,
                    lattice_jitter=float(args.lattice_jitter),
                )
                if rendered is None:
                    continue
                digest = delta_utils.cif_digest(str(rendered.get("cif") or ""))
                if digest in seen_cifs:
                    stats["deduped_cifs"] += 1
                    continue
                seen_cifs.add(digest)
                proposals.append(rendered)
                stats["render_ok"] += 1
        scored = guided_sf.score_rows(model, proposals)
        stats["proposal_rows_scored"] += len(scored)
        selected = guided_sf.select_diverse(
            scored,
            top_k=int(args.max_output_candidates_per_sample),
            max_per_wa=int(args.max_per_wa),
        )
        if selected:
            stats["samples_with_output"] += 1
        out_rows.extend(selected)

    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    summary = {
        "config": {
            "train_features": str(args.train_features),
            "input_rendered_jsonl": str(args.input_rendered_jsonl),
            "repr_jsonl": str(args.repr_jsonl),
            "max_records": int(args.max_records),
            "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
            "prototypes_per_wa": int(args.prototypes_per_wa),
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "max_per_wa": int(args.max_per_wa),
            "min_target_row_count": int(args.min_target_row_count),
            "min_prototype_positives": int(args.min_prototype_positives),
            "min_prototype_score": float(args.min_prototype_score),
            "lattice_jitter": float(args.lattice_jitter),
            "model_kind": str(args.model_kind),
        },
        "model": model_summary,
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Prototype keys are fitted from train positive/negative labels only. Inference uses predicted W/A, formula, GT-SG, prototype context keys, and GT-free CIF features for compatibility scoring.",
    }
    write_json(out_dir / "contrastive_geometry_prototype_summary.json", summary)
    write_json(model_out / "contrastive_geometry_prototype_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
