#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_geometry_compat_features as compat_features  # noqa: E402
import opentry_build_positive_geometry_residual_examples as pos_geom  # noqa: E402
import opentry_pair_delta_geometry_variants as delta_utils  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import opentry_source_free_prior_geometry_sampler as sf_prior  # noqa: E402
import opentry_train_geometry_compat_selector as compat_selector  # noqa: E402
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


def train_compat_model(
    train_rows: list[dict[str, Any]],
    *,
    model_kind: str,
    positive_weight: float,
    candidate_row7_weight: float,
    atom12_weight: float,
) -> tuple[Pipeline, dict[str, Any]]:
    x_train = [compat_selector.feature_dict(row) for row in train_rows]
    y_train = [int(bool(row.get("label_match"))) for row in train_rows]
    weights: list[float] = []
    for row, y in zip(train_rows, y_train):
        weight = float(positive_weight) if y else 1.0
        if int(row.get("candidate_row_count", 0)) >= 7:
            weight *= float(candidate_row7_weight)
        if int(row.get("atom_count", 0)) >= 12:
            weight *= float(atom12_weight)
        weights.append(weight)

    if str(model_kind) == "gbdt_strict":
        estimator = HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=11,
            min_samples_leaf=24,
            l2_regularization=0.12,
            random_state=20260615,
        )
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=16,
            l2_regularization=0.05,
            random_state=20260615,
        )
    model: Pipeline = Pipeline([("vec", DictVectorizer(sparse=False)), ("clf", estimator)])
    model.fit(x_train, y_train, clf__sample_weight=weights)
    summary = {
        "train_rows": len(train_rows),
        "train_positive_rows": int(sum(y_train)),
        "train_positive_rate": float(sum(y_train) / max(1, len(y_train))),
        "model_kind": str(model_kind),
        "positive_weight": float(positive_weight),
        "candidate_row7_weight": float(candidate_row7_weight),
        "atom12_weight": float(atom12_weight),
        "feature_policy": {
            "blocked": sorted(compat_selector.BLOCKED_FEATURE_KEYS),
            "note": "Training labels are train-only. Inference features use compat_selector.feature_dict, which blocks GT labels, target keys, sample ids, row_count labels, and match/rms fields.",
        },
    }
    return model, summary


def score_rows(model: Pipeline, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = [compat_selector.feature_dict(row) for row in rows]
    scores = model.predict_proba(features)[:, 1]
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["compat_score"] = float(score)
        out.append(item)
    return out


def select_diverse(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    max_per_wa: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get("compat_score", 0.0)),
            int(row.get("original_rank") or row.get("source_rank") or 10**9),
            int(row.get("source_free_sample_index") or 0),
        ),
    )
    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    selected_source_ids: set[int] = set()
    for row in ordered:
        wa_key = str(row.get("canonical_wa_key") or "")
        if int(max_per_wa) > 0 and counts.get(wa_key, 0) >= int(max_per_wa):
            continue
        counts[wa_key] = counts.get(wa_key, 0) + 1
        item = dict(row)
        item["rank"] = len(selected) + 1
        selected.append(item)
        selected_source_ids.add(id(row))
        if len(selected) >= int(top_k):
            break
    if len(selected) < int(top_k):
        for row in ordered:
            if id(row) in selected_source_ids:
                continue
            item = dict(row)
            item["rank"] = len(selected) + 1
            selected.append(item)
            selected_source_ids.add(id(row))
            if len(selected) >= int(top_k):
                break
    return selected


def safe_copy_source_fields(source_row: dict[str, Any]) -> dict[str, Any]:
    blocked_prefixes = ("label_",)
    blocked_exact = {"rank", "cif", "readable", "formula_ok", "atom_count_ok", "composition_exact", "detected_sg", "sg_ok"}
    out: dict[str, Any] = {}
    for key, value in source_row.items():
        if key in blocked_exact or any(str(key).startswith(prefix) for prefix in blocked_prefixes):
            continue
        out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility-guided source-free train-prior geometry sampler.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=20)
    parser.add_argument("--proposal-samples-per-wa", type=int, default=16)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-per-wa", type=int, default=4)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--mode", choices=["uniform", "orbit_prior"], default="orbit_prior")
    parser.add_argument("--model-kind", choices=["gbdt", "gbdt_strict"], default="gbdt")
    parser.add_argument("--positive-only-prior", action="store_true")
    parser.add_argument("--positive-weight", type=float, default=5.0)
    parser.add_argument("--candidate-row7-weight", type=float, default=2.0)
    parser.add_argument("--atom12-weight", type=float, default=1.25)
    parser.add_argument("--lattice-jitter", type=float, default=0.10)
    parser.add_argument("--param-jitter", type=float, default=0.06)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    model_out = ensure_under_opentry(args.model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    train_features = read_jsonl(args.train_features)
    model, model_summary = train_compat_model(
        train_features,
        model_kind=str(args.model_kind),
        positive_weight=float(args.positive_weight),
        candidate_row7_weight=float(args.candidate_row7_weight),
        atom12_weight=float(args.atom12_weight),
    )
    joblib.dump(model, model_out / "compat_model.joblib")

    engine = OrbitEngine(args.lookup_json)
    prior = sf_prior.build_prior(train_features, positive_only=bool(args.positive_only_prior))
    repr_rows = load_repr(args.repr_jsonl, max_records=int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))

    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "skipped_target_rows_lt_min": 0,
        "samples_with_input": 0,
        "samples_with_proposals": 0,
        "samples_with_output": 0,
        "input_wa_candidates_seen": 0,
        "render_attempts": 0,
        "render_ok": 0,
        "deduped_cifs": 0,
        "proposal_rows_scored": 0,
        **{f"prior_{k}": v for k, v in prior.summary.items()},
    }

    for sample_id, repr_row in repr_rows.items():
        target_row_count = int(repr_row["row_count"])
        if target_row_count < int(args.min_target_row_count):
            stats["skipped_target_rows_lt_min"] += 1
            continue
        input_rows = rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        if input_rows:
            stats["samples_with_input"] += 1
        sg = int(repr_row["sg"])
        record_base = {
            "sample_id": sample_id,
            "formula_counts": dict(repr_row["formula_counts"]),
            "sg": sg,
            "sg_symbol": str(repr_row.get("sg_symbol") or ""),
            "atom_count": int(repr_row["atom_count"]),
        }
        proposals: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        seen_wa: set[str] = set()
        for source_row in input_rows:
            wa_key = str(source_row.get("canonical_wa_key") or "")
            if not wa_key or wa_key in seen_wa:
                continue
            seen_wa.add(wa_key)
            stats["input_wa_candidates_seen"] += 1
            row_count = sf_prior.candidate_row_count(source_row)
            for sample_idx in range(int(args.proposal_samples_per_wa)):
                token = f"{sample_id}:{wa_key}:{sample_idx}:{args.mode}:{args.model_kind}"
                lat_vec, vpa, lat_key = prior.select_lattice_base(sg, row_count, token)
                lattice = sf_prior.scale_lattice(
                    lat_vec,
                    sg,
                    int(repr_row["atom_count"]),
                    vpa,
                    token,
                    float(args.lattice_jitter),
                )
                params, param_sources = sf_prior.params_from_prior(
                    engine,
                    prior,
                    wa_key,
                    str(args.mode),
                    token,
                    float(args.param_jitter),
                )
                rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
                stats["render_attempts"] += 1
                render = v2.render_candidate(
                    engine,
                    record_base,
                    {"rows": rows, "params": params, "lattice": lattice},
                    len(proposals) + 1,
                    f"compat_source_free_{args.mode}_{sample_idx}",
                )
                if not render.get("ok"):
                    continue
                cif = str(render.get("cif") or "")
                digest = delta_utils.cif_digest(cif)
                if digest in seen_cifs:
                    stats["deduped_cifs"] += 1
                    continue
                seen_cifs.add(digest)
                metric = validate_cif(cif, record_base["formula_counts"], sg)
                self_features = selfscore.cif_self_features(cif)
                stats["render_ok"] += 1
                proposal = {
                    **safe_copy_source_fields(source_row),
                    **compat_features.parse_wa_metadata(wa_key),
                    **self_features,
                    "sample_id": sample_id,
                    "rank": len(proposals) + 1,
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
                    "geometry_source": "compat_guided_source_free_train_prior",
                    "geometry_param_variant_mode": f"compat_source_free_{args.mode}_{args.model_kind}",
                    "geometry_lattice_mode": "compat_source_free_train_prior_scaled",
                    "source_free_mode": str(args.mode),
                    "source_free_sample_index": int(sample_idx),
                    "source_free_lattice_key": repr(lat_key),
                    "source_free_param_source_keys": param_sources[:20],
                    "source_rank": int(source_row.get("rank", 0)),
                    "original_rank": int(source_row.get("rank", 0)),
                }
                proposals.append(proposal)
        if proposals:
            stats["samples_with_proposals"] += 1
        scored = score_rows(model, proposals)
        stats["proposal_rows_scored"] += len(scored)
        selected = select_diverse(
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
            "proposal_samples_per_wa": int(args.proposal_samples_per_wa),
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "max_per_wa": int(args.max_per_wa),
            "min_target_row_count": int(args.min_target_row_count),
            "mode": str(args.mode),
            "model_kind": str(args.model_kind),
            "positive_only_prior": bool(args.positive_only_prior),
            "lattice_jitter": float(args.lattice_jitter),
            "param_jitter": float(args.param_jitter),
        },
        "model": model_summary,
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Train-label compatibility model scores source-free train-prior proposals before top-k persistence. Inference uses predicted W/A candidates, formula, GT-SG, rendered CIF self features, and GT-free candidate metadata only.",
    }
    write_json(out_dir / "compat_guided_source_free_summary.json", summary)
    write_json(model_out / "compat_guided_source_free_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
