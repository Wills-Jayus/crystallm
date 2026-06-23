#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import signal
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SCRIPT_DIR, SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_enrich_source_pair_features as sourcepair  # noqa: E402
import opentry_render_wyckoff_cifs_e07e08 as renderer  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
import run_mp20_geometry_breakthrough as gb  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402


LABEL_KEYS = {
    "label_match",
    "label_rmsd",
    "label_error",
    "candidate_wa_hit",
    "candidate_skeleton_hit",
    "target_wa_key",
    "target_skeleton_key",
    "target_row_count",
    "target_complex_flag",
}

ID_KEYS = {
    "sample_id",
    "material_id",
    "candidate_uid",
    "split",
    "cif",
    "cif_path",
    "source_sample_id",
    "canonical_wa_key",
    "canonical_skeleton_key",
    "raw_dp_wa_key",
    "raw_dp_skeleton_key",
    "error",
    "self_parse_error",
}

POST_RENDER_KEYS = {
    "readable",
    "formula_ok",
    "atom_count_ok",
    "composition_exact",
    "sg_ok",
    "detected_sg",
    "atom_count_after_expansion",
    "self_min_distance",
    "self_parsed_sites",
    "self_score",
    "self_train_volume_prior_score",
    "self_volume",
    "self_volume_per_atom",
    "cell_a",
    "cell_b",
    "cell_c",
    "cell_alpha",
    "cell_beta",
    "cell_gamma",
    "cell_volume_header",
    "cell_len_cv",
    "cell_len_ratio_max_min",
    "cell_angle_absdev_90_mean",
    "cell_angle_absdev_90_max",
}

CATEGORICAL_KEYS = {
    "crystal_system",
    "geometry_lattice_mode",
    "geometry_mode",
    "geometry_param_variant_mode",
    "geometry_source",
}


class RecordTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float | None):
    if seconds is None or float(seconds) <= 0:
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise RecordTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        signal.signal(signal.SIGALRM, old)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def blocked_feature_key(key: str) -> bool:
    if key in LABEL_KEYS or key in ID_KEYS or key in POST_RENDER_KEYS:
        return True
    if key.startswith("label_") or key.startswith("target_") or key.startswith("self_") or key.startswith("cell_"):
        return True
    if key.endswith("_key") or key.endswith("_uid"):
        return True
    return False


def feature_dict(row: dict[str, Any]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for key, value in row.items():
        key_s = str(key)
        if blocked_feature_key(key_s):
            continue
        if value is None:
            out[f"{key_s}__missing"] = 1.0
            continue
        if isinstance(value, bool):
            out[key_s] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            value_f = float(value)
            if math.isfinite(value_f):
                out[key_s] = value_f
            else:
                out[f"{key_s}__missing"] = 1.0
        elif isinstance(value, str) and key_s in CATEGORICAL_KEYS:
            out[key_s] = value
    return out


def sample_weight(row: dict[str, Any], y: int) -> float:
    weight = 5.0 if y else 1.0
    if int(row.get("atom_count", 0) or 0) >= 12:
        weight *= 1.35
    if int(row.get("target_row_count", 0) or 0) >= 7:
        weight *= 2.5
    if bool(row.get("target_complex_flag")):
        weight *= 1.15
    return float(weight)


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("sample_id") or ""), []).append(row)
    for items in out.values():
        items.sort(key=lambda r: int(r.get("rank", 10**9) or 10**9))
    return out


def label_metrics(rows: list[dict[str, Any]], scores: np.ndarray, budgets: tuple[int, ...] = (1, 5, 20, 50)) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["_score"] = float(score)
        scored.append(item)
    per_sample: list[dict[str, Any]] = []
    for sample_id, items in grouped(scored).items():
        ordered = sorted(items, key=lambda r: (-float(r.get("_score", 0.0)), int(r.get("rank", 10**9) or 10**9)))
        sample = {"sample_id": sample_id, "rows_ge_7": int(items[0].get("target_row_count", 0) or 0) >= 7, "atom_ge_12": int(items[0].get("atom_count", 0) or 0) >= 12}
        for k in budgets:
            top = ordered[:k]
            sample[f"match@{k}"] = any(bool(r.get("label_match")) for r in top)
        per_sample.append(sample)

    def summarize(items: list[dict[str, Any]]) -> dict[str, float]:
        return {f"hit@{k}": sum(bool(row[f"match@{k}"]) for row in items) / max(1, len(items)) for k in budgets}

    rows7 = [row for row in per_sample if bool(row["rows_ge_7"])]
    atom12 = [row for row in per_sample if bool(row["atom_ge_12"])]
    return {"full": summarize(per_sample), "rows_ge_7": summarize(rows7), "atom_ge_12": summarize(atom12), "samples": len(per_sample)}


def train_model(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_features)
    val_rows = read_jsonl(args.val_features)
    y_train = [int(bool(row.get("label_match"))) for row in train_rows]
    y_val = [int(bool(row.get("label_match"))) for row in val_rows]
    x_train = [feature_dict(row) for row in train_rows]
    weights = [sample_weight(row, y) for row, y in zip(train_rows, y_train)]
    model = Pipeline(
        [
            ("vec", DictVectorizer(sparse=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=int(args.max_iter),
                    learning_rate=float(args.learning_rate),
                    max_leaf_nodes=int(args.max_leaf_nodes),
                    min_samples_leaf=int(args.min_samples_leaf),
                    l2_regularization=float(args.l2_regularization),
                    random_state=int(args.seed),
                ),
            ),
        ]
    )
    model.fit(x_train, y_train, clf__sample_weight=weights)
    train_scores = np.asarray(model.predict_proba(x_train)[:, 1], dtype=float)
    val_features = [feature_dict(row) for row in val_rows]
    val_scores = np.asarray(model.predict_proba(val_features)[:, 1], dtype=float)
    joblib.dump(model, out_dir / "source_proposal_model.joblib")
    summary = {
        "train_features": str(args.train_features),
        "val_features": str(args.val_features),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_positive_rate": sum(y_train) / max(1, len(y_train)),
        "val_positive_rate": sum(y_val) / max(1, len(y_val)),
        "candidate_score_metrics": {
            "train_roc_auc": float(roc_auc_score(y_train, train_scores)) if len(set(y_train)) > 1 else None,
            "train_average_precision": float(average_precision_score(y_train, train_scores)) if len(set(y_train)) > 1 else None,
            "val_roc_auc": float(roc_auc_score(y_val, val_scores)) if len(set(y_val)) > 1 else None,
            "val_average_precision": float(average_precision_score(y_val, val_scores)) if len(set(y_val)) > 1 else None,
        },
        "val_label_metrics": label_metrics(val_rows, val_scores),
        "feature_policy": {
            "blocked_label_keys": sorted(LABEL_KEYS),
            "blocked_post_render_keys": sorted(POST_RENDER_KEYS),
            "blocked_id_keys": sorted(ID_KEYS),
            "note": "Source proposal uses only pre-render GT-free target/source/W-A metadata. StructureMatcher labels are used only on train/val for training and selection.",
        },
    }
    write_json(out_dir / "source_proposal_training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def candidate_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    orbit_ids = [str(row.get("orbit_id") or row.get("canonical_orbit_id") or "") for row in rows]
    elements = [str(row.get("element") or "") for row in rows if row.get("element")]
    multiplicities = [int(row.get("multiplicity") or 0) for row in rows]
    return {
        "candidate_row_count": len(rows),
        "candidate_unique_orbit_count": len(set(orbit_ids)),
        "candidate_duplicate_orbit_count": max(0, len(orbit_ids) - len(set(orbit_ids))),
        "candidate_unique_element_count": len(set(elements)),
        "candidate_max_multiplicity": max(multiplicities) if multiplicities else 0,
        "candidate_mean_multiplicity": float(sum(multiplicities) / len(multiplicities)) if multiplicities else 0.0,
    }


def proposal_feature_row(
    *,
    target_record: dict[str, Any],
    pred_record: dict[str, Any],
    candidate: dict[str, Any],
    wa_index: int,
    source: dict[str, Any],
    source_rank: int,
    selector: renderer.OpentryGeometrySelector,
    lattice_mode: str,
) -> dict[str, Any]:
    rows = v2.canonical_rows(pred_record)
    skel, wa = renderer.record_keys(pred_record)
    source_skel, source_wa = renderer.record_keys(source)
    align_cost = selector.source_alignment_cost(pred_record, source, chemical=True)
    geometry_distance = gb.row_condition_distance(pred_record, source) + 0.1 * float(align_cost)
    out: dict[str, Any] = {
        "sample_id": str(target_record.get("sample_id") or ""),
        "rank": int(source_rank) + 1,
        "original_rank": int(wa_index) + 1,
        "candidate_score": candidate.get("score"),
        "geometry_rank": int(source_rank),
        "geometry_distance": float(geometry_distance),
        "geometry_lattice_mode": lattice_mode,
        "geometry_mode": "e08",
        "geometry_param_variant_mode": "none",
        "geometry_source": "e08_row_aligned_chem_sourceproposal_quality",
        "sg": int(pred_record.get("sg") or 0),
        "atom_count": int(sum(int(v) for v in dict(pred_record.get("formula_counts") or {}).values())),
        "formula_element_count": len(dict(pred_record.get("formula_counts") or {})),
        "crystal_system": gb.crystal_system(int(pred_record.get("sg") or 0)),
        "canonical_skeleton_key": skel,
        "canonical_wa_key": wa,
        "source_sample_id": str(source.get("sample_id") or ""),
    }
    out.update(candidate_meta(rows))
    enriched = sourcepair.enrich_row(out, pred_record, source)
    enriched["source_pair_same_canonical_skeleton"] = skel == source_skel
    enriched["source_pair_same_canonical_wa"] = wa == source_wa
    return enriched


def build_cache(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(args.model)
    train_records = renderer.load_records(args.data_root / "train.jsonl")
    split_records = renderer.load_records(args.data_root / f"{args.split}.jsonl")
    pred_by_id = {str(row["sample_id"]): row for row in renderer.read_jsonl(args.predictions)}
    selected_records = [record for record in split_records if str(record["sample_id"]) in pred_by_id]
    if int(args.start_index) > 0:
        selected_records = selected_records[int(args.start_index) :]
    if int(args.max_records) > 0:
        selected_records = selected_records[: int(args.max_records)]
    sg_symbols = v2.sg_symbols_from_splits({"train": train_records, args.split: selected_records})
    engine = OrbitEngine(str(args.lookup_json), sg_symbols)
    selector = renderer.OpentryGeometrySelector(train_records, gb.GeometryIndex(train_records, []), None)
    entries: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    timeout_records = 0
    failed_records = 0
    for rec_idx, target_record in enumerate(selected_records, start=1):
        entry_start = len(entries)
        scored_start = len(scored_rows)
        try:
            with time_limit(float(args.record_timeout_seconds)):
                prediction = pred_by_id.get(str(target_record["sample_id"]))
                if prediction is None:
                    continue
                candidates = list(prediction.get("ranked_wa_candidates") or [])[: int(args.max_wa)]
                for wa_index, candidate in enumerate(candidates):
                    rows = renderer.normalize_rows(engine, list(candidate.get("rows") or []))
                    pred_record = renderer.pseudo_record(target_record, rows)
                    if str(args.base_source_strategy) == "cheap_knn":
                        raw_sources = selector.base_index.candidates(pred_record, "row_conditioned_knn", max(int(args.source_pool_k) * 3, int(args.source_pool_k)))
                        seen_source_ids: set[str] = set()
                        sources = []
                        for source in raw_sources:
                            sid = str(source.get("sample_id") or "")
                            if sid in seen_source_ids:
                                continue
                            seen_source_ids.add(sid)
                            sources.append(source)
                            if len(sources) >= int(args.source_pool_k):
                                break
                    else:
                        sources = selector.row_aligned_chem_quality_candidates(pred_record, int(args.source_pool_k))
                    feature_rows = [
                        proposal_feature_row(
                            target_record=target_record,
                            pred_record=pred_record,
                            candidate=candidate,
                            wa_index=wa_index,
                            source=source,
                            source_rank=source_rank,
                            selector=selector,
                            lattice_mode=str(args.geometry_lattice_mode),
                        )
                        for source_rank, source in enumerate(sources)
                    ]
                    if not feature_rows:
                        continue
                    scores = np.asarray(model.predict_proba([feature_dict(row) for row in feature_rows])[:, 1], dtype=float)
                    ordered = sorted(zip(feature_rows, scores), key=lambda item: (-float(item[1]), int(item[0].get("rank", 10**9))))
                    top = ordered[: int(args.sources_per_wa)]
                    entries.append(
                        {
                            "sample_id": str(target_record["sample_id"]),
                            "wa_index": int(wa_index),
                            "canonical_wa_key": str(feature_rows[0]["canonical_wa_key"]),
                            "canonical_skeleton_key": str(feature_rows[0]["canonical_skeleton_key"]),
                            "source_sample_ids": [str(row.get("source_sample_id") or "") for row, _score in top],
                            "source_scores": [float(_score) for _row, _score in top],
                        }
                    )
                    for row, score in ordered[: min(len(ordered), int(args.write_scored_per_wa))]:
                        item = dict(row)
                        item["source_proposal_score"] = float(score)
                        scored_rows.append(item)
        except RecordTimeout:
            del entries[entry_start:]
            del scored_rows[scored_start:]
            timeout_records += 1
        except Exception:
            del entries[entry_start:]
            del scored_rows[scored_start:]
            failed_records += 1
        if int(args.progress_every) > 0 and (rec_idx % int(args.progress_every) == 0 or rec_idx == len(selected_records)):
            print(json.dumps({"completed_records": rec_idx, "selected_records": len(selected_records), "entries": len(entries), "timeout_records": timeout_records, "failed_records": failed_records}, sort_keys=True), flush=True)
    payload = {
        "metadata": {
            "split": str(args.split),
            "predictions": str(args.predictions),
            "model": str(args.model),
            "start_index": int(args.start_index),
            "max_records": int(args.max_records),
            "selected_records": len(selected_records),
            "max_wa": int(args.max_wa),
            "source_pool_k": int(args.source_pool_k),
            "sources_per_wa": int(args.sources_per_wa),
            "record_timeout_seconds": float(args.record_timeout_seconds),
            "feature_policy": "pre-render GT-free source/W-A features scored by train-label source proposal model",
        },
        "entries": entries,
    }
    write_json(out_dir / "source_proposal_cache.json", payload)
    write_jsonl(out_dir / "source_proposal_scored_rows.jsonl", scored_rows)
    summary = {
        "entries": len(entries),
        "scored_rows": len(scored_rows),
        "selected_records": len(selected_records),
        "samples_with_entries": len({entry["sample_id"] for entry in entries}),
        "mean_entries_per_sample": len(entries) / max(1, len({entry["sample_id"] for entry in entries})),
        "source_pool_k": int(args.source_pool_k),
        "sources_per_wa": int(args.sources_per_wa),
        "timeout_records": int(timeout_records),
        "failed_records": int(failed_records),
    }
    write_json(out_dir / "source_proposal_cache_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and apply a pre-render source/free-param proposal model.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-features", type=Path, required=True)
    p_train.add_argument("--val-features", type=Path, required=True)
    p_train.add_argument("--out-dir", type=Path, required=True)
    p_train.add_argument("--max-iter", type=int, default=220)
    p_train.add_argument("--learning-rate", type=float, default=0.035)
    p_train.add_argument("--max-leaf-nodes", type=int, default=23)
    p_train.add_argument("--min-samples-leaf", type=int, default=12)
    p_train.add_argument("--l2-regularization", type=float, default=0.08)
    p_train.add_argument("--seed", type=int, default=20260615)
    p_cache = sub.add_parser("build-cache")
    p_cache.add_argument("--model", type=Path, required=True)
    p_cache.add_argument("--data-root", type=Path, default=renderer.SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    p_cache.add_argument("--lookup-json", type=Path, default=renderer.SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    p_cache.add_argument("--predictions", type=Path, required=True)
    p_cache.add_argument("--split", default="val")
    p_cache.add_argument("--start-index", type=int, default=0)
    p_cache.add_argument("--max-records", type=int, default=128)
    p_cache.add_argument("--max-wa", type=int, default=20)
    p_cache.add_argument("--source-pool-k", type=int, default=64)
    p_cache.add_argument("--base-source-strategy", choices=["cheap_knn", "row_aligned_chem_quality"], default="cheap_knn")
    p_cache.add_argument("--sources-per-wa", type=int, default=12)
    p_cache.add_argument("--geometry-lattice-mode", default="source_vpa_calibrated_soft")
    p_cache.add_argument("--write-scored-per-wa", type=int, default=8)
    p_cache.add_argument("--record-timeout-seconds", type=float, default=0.0)
    p_cache.add_argument("--progress-every", type=int, default=16)
    p_cache.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        return train_model(args)
    if args.command == "build-cache":
        return build_cache(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
