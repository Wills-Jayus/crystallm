#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict
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
import opentry_pair_delta_geometry_variants as delta_utils  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import opentry_compat_guided_source_free_sampler as guided_sf  # noqa: E402
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


def stable_u01(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return (value + 0.5) / float(1 << 64)


def centered(*parts: Any) -> float:
    return 2.0 * stable_u01(*parts) - 1.0


def clean_lattice(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "a": float(raw["a"]),
        "b": float(raw["b"]),
        "c": float(raw["c"]),
        "alpha": float(raw["alpha"]),
        "beta": float(raw["beta"]),
        "gamma": float(raw["gamma"]),
    }


def clean_params(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row_key, row_values in dict(raw or {}).items():
        out[str(row_key)] = {str(k): float(v) % 1.0 for k, v in dict(row_values or {}).items()}
    return out


def perturb_lattice(lattice: dict[str, float], sg: int, token: str, scale: float) -> dict[str, float]:
    vec = delta_utils.lattice_vector(lattice)
    out: list[float] = []
    for idx, value in enumerate(vec):
        if idx < 3:
            delta = centered(token, "lat", idx) * float(scale)
        else:
            delta = centered(token, "ang", idx) * float(scale) * 0.20
        out.append(float(value) + float(delta))
    return v2.lattice_from_target(out, int(sg))


def perturb_params(params: dict[str, dict[str, float]], token: str, scale: float) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row_key, row_values in params.items():
        new_row: dict[str, float] = {}
        for symbol, value in row_values.items():
            delta = centered(token, "param", row_key, symbol) * float(scale)
            new_row[str(symbol)] = (float(value) + delta) % 1.0
        out[str(row_key)] = new_row
    return out


def safe_source_fields(source_row: dict[str, Any]) -> dict[str, Any]:
    blocked_prefixes = ("label_",)
    blocked_exact = {"rank", "cif", "readable", "formula_ok", "atom_count_ok", "composition_exact", "detected_sg", "sg_ok"}
    out: dict[str, Any] = {}
    for key, value in source_row.items():
        if key in blocked_exact or any(str(key).startswith(prefix) for prefix in blocked_prefixes):
            continue
        out[key] = value
    return out


def make_proposal(
    *,
    engine: OrbitEngine,
    sample_id: str,
    repr_row: dict[str, Any],
    source_row: dict[str, Any],
    lattice: dict[str, float],
    params: dict[str, dict[str, float]],
    search_mode: str,
    step: int,
    neighbor: int,
) -> dict[str, Any] | None:
    sg = int(repr_row["sg"])
    wa_key = str(source_row.get("canonical_wa_key") or "")
    if not wa_key:
        return None
    record_base = {
        "sample_id": sample_id,
        "formula_counts": dict(repr_row["formula_counts"]),
        "sg": sg,
        "sg_symbol": str(repr_row.get("sg_symbol") or ""),
        "atom_count": int(repr_row["atom_count"]),
    }
    rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
    render = v2.render_candidate(
        engine,
        record_base,
        {"rows": rows, "params": params, "lattice": lattice},
        int(neighbor) + 1,
        f"compat_local_{search_mode}_s{step}_n{neighbor}",
    )
    if not render.get("ok"):
        return None
    cif = str(render.get("cif") or "")
    metric = validate_cif(cif, record_base["formula_counts"], sg)
    self_features = selfscore.cif_self_features(cif)
    return {
        **safe_source_fields(source_row),
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
        "geometry_source": "compat_local_geometry_search",
        "geometry_param_variant_mode": f"compat_local_{search_mode}",
        "geometry_lattice_mode": f"compat_local_{search_mode}",
        "local_search_mode": str(search_mode),
        "local_search_step": int(step),
        "local_search_neighbor": int(neighbor),
        "source_rank": int(source_row.get("rank", 0)),
        "original_rank": int(source_row.get("rank", 0)),
    }


def score_proposals(model: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = guided_sf.score_rows(model, rows)
    return sorted(
        scored,
        key=lambda row: (
            -float(row.get("compat_score", 0.0)),
            int(row.get("original_rank") or row.get("rank") or 10**9),
            int(row.get("local_search_step") or 0),
            int(row.get("local_search_neighbor") or 0),
        ),
    )


def search_from_source(
    *,
    model: Any,
    engine: OrbitEngine,
    sample_id: str,
    repr_row: dict[str, Any],
    source_row: dict[str, Any],
    search_mode: str,
    iterations: int,
    beam_size: int,
    neighbors_per_state: int,
    lattice_scale: float,
    param_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_lattice = clean_lattice(dict(source_row.get("geometry_lattice") or {}))
    base_params = clean_params(dict(source_row.get("geometry_params") or {}))
    beam: list[dict[str, Any]] = [{"lattice": base_lattice, "params": base_params, "compat_score": 0.0}]
    collected: list[dict[str, Any]] = []
    stats = {"attempts": 0, "ok": 0, "deduped": 0}
    seen_cifs: set[str] = set()
    for step in range(int(iterations)):
        step_scale = 1.0 / math.sqrt(step + 1.0)
        proposals: list[dict[str, Any]] = []
        for state_idx, state in enumerate(beam):
            for neighbor in range(int(neighbors_per_state)):
                token = f"{sample_id}:{source_row.get('rank')}:{search_mode}:{step}:{state_idx}:{neighbor}"
                lattice = dict(state["lattice"])
                params = clean_params(dict(state["params"]))
                if search_mode in {"joint", "lattice_only"}:
                    lattice = perturb_lattice(lattice, int(repr_row["sg"]), token, float(lattice_scale) * step_scale)
                if search_mode in {"joint", "param_only"}:
                    params = perturb_params(params, token, float(param_scale) * step_scale)
                stats["attempts"] += 1
                proposal = make_proposal(
                    engine=engine,
                    sample_id=sample_id,
                    repr_row=repr_row,
                    source_row=source_row,
                    lattice=lattice,
                    params=params,
                    search_mode=search_mode,
                    step=step + 1,
                    neighbor=neighbor,
                )
                if proposal is None:
                    continue
                digest = delta_utils.cif_digest(str(proposal.get("cif") or ""))
                if digest in seen_cifs:
                    stats["deduped"] += 1
                    continue
                seen_cifs.add(digest)
                proposals.append(proposal)
                stats["ok"] += 1
        scored = score_proposals(model, proposals)
        collected.extend(scored)
        beam = [
            {
                "lattice": dict(row.get("geometry_lattice") or {}),
                "params": clean_params(dict(row.get("geometry_params") or {})),
                "compat_score": float(row.get("compat_score") or 0.0),
            }
            for row in scored[: int(beam_size)]
        ]
        if not beam:
            break
    return collected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Train-label compatibility-guided local geometry search around rendered candidates.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=12)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-per-wa", type=int, default=5)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--search-mode", choices=["joint", "param_only", "lattice_only"], default="joint")
    parser.add_argument("--model-kind", choices=["gbdt", "gbdt_strict"], default="gbdt")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--neighbors-per-state", type=int, default=8)
    parser.add_argument("--lattice-scale", type=float, default=0.055)
    parser.add_argument("--param-scale", type=float, default=0.045)
    parser.add_argument("--positive-weight", type=float, default=5.0)
    parser.add_argument("--candidate-row7-weight", type=float, default=2.0)
    parser.add_argument("--atom12-weight", type=float, default=1.25)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    model_out = ensure_under_opentry(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    train_features = read_jsonl(args.train_features)
    model, model_summary = guided_sf.train_compat_model(
        train_features,
        model_kind=str(args.model_kind),
        positive_weight=float(args.positive_weight),
        candidate_row7_weight=float(args.candidate_row7_weight),
        atom12_weight=float(args.atom12_weight),
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
        "source_candidates_seen": 0,
        "search_attempts": 0,
        "search_ok": 0,
        "search_deduped": 0,
        "proposal_rows_scored": 0,
    }
    for sample_id, repr_row in repr_rows.items():
        if int(repr_row["row_count"]) < int(args.min_target_row_count):
            stats["skipped_target_rows_lt_min"] += 1
            continue
        input_rows = rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        if input_rows:
            stats["samples_with_input"] += 1
        sample_proposals: list[dict[str, Any]] = []
        for source_row in input_rows:
            if not source_row.get("geometry_lattice") or not source_row.get("geometry_params"):
                continue
            stats["source_candidates_seen"] += 1
            proposals, search_stats = search_from_source(
                model=model,
                engine=engine,
                sample_id=sample_id,
                repr_row=repr_row,
                source_row=source_row,
                search_mode=str(args.search_mode),
                iterations=int(args.iterations),
                beam_size=int(args.beam_size),
                neighbors_per_state=int(args.neighbors_per_state),
                lattice_scale=float(args.lattice_scale),
                param_scale=float(args.param_scale),
            )
            stats["search_attempts"] += int(search_stats["attempts"])
            stats["search_ok"] += int(search_stats["ok"])
            stats["search_deduped"] += int(search_stats["deduped"])
            sample_proposals.extend(proposals)
        stats["proposal_rows_scored"] += len(sample_proposals)
        selected = guided_sf.select_diverse(
            sample_proposals,
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
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "max_per_wa": int(args.max_per_wa),
            "min_target_row_count": int(args.min_target_row_count),
            "search_mode": str(args.search_mode),
            "model_kind": str(args.model_kind),
            "iterations": int(args.iterations),
            "beam_size": int(args.beam_size),
            "neighbors_per_state": int(args.neighbors_per_state),
            "lattice_scale": float(args.lattice_scale),
            "param_scale": float(args.param_scale),
        },
        "model": model_summary,
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Compatibility model is trained on train labels only. Local search mutates lattice/free params around rendered predicted W/A candidates, scores generated CIFs with GT-free features, and writes only top generated candidates.",
    }
    write_json(out_dir / "compat_local_geometry_search_summary.json", summary)
    write_json(model_out / "compat_local_geometry_search_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
