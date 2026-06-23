#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_source_mode_examples as source_mode_data  # noqa: E402
import opentry_render_source_residual_geometry as source_render  # noqa: E402
import opentry_render_wyckoff_cifs_e07e08 as rb  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import opentry_train_source_residual_geometry as train_resid  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


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


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def select_records(records: list[dict[str, Any]], start: int, max_records: int) -> list[dict[str, Any]]:
    selected = records[max(0, int(start)) :]
    if int(max_records) > 0:
        selected = selected[: int(max_records)]
    return selected


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def render_health_score(row: dict[str, Any], *, geometry_distance_weight: float, original_rank_weight: float) -> float:
    return selfscore.self_score(
        row,
        geometry_distance_weight=float(geometry_distance_weight),
        original_rank_weight=float(original_rank_weight),
        train_volume_priors={},
        train_volume_prior_weight=0.0,
        score_profile="standard",
    )


def candidate_payloads(
    *,
    selector: rb.OpentryGeometrySelector,
    record: dict[str, Any],
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
) -> list[dict[str, Any]]:
    query_k = int(source_pool_k) + (1 if exclude_self else 0)
    sources = train_resid.fast_chem_quality_candidates(selector, record, query_k)
    if exclude_self:
        sample_id = str(record.get("sample_id"))
        sources = [source for source in sources if str(source.get("sample_id")) != sample_id]
    out: list[dict[str, Any]] = []
    for source_rank, source in enumerate(sources[: int(source_pool_k)]):
        params, align_cost = selector.source_aligned_params(record, source, 0, chemical=True)
        lattice = selector.vpa_calibrated_lattice(source, record, strength=float(vpa_strength))
        features = source_mode_data.numeric_source_features(
            selector=selector,
            record=record,
            source=source,
            source_rank=source_rank,
            align_cost=align_cost,
            source_pool_k=int(source_pool_k),
            base_lattice=lattice,
        )
        out.append(
            {
                "source": source,
                "source_sample_id": str(source.get("sample_id")),
                "source_rank": int(source_rank),
                "base_params": params,
                "base_lattice": lattice,
                "align_cost": float(align_cost),
                "source_distance": float(rb.gb.row_condition_distance(record, source)),
                "features": features,
            }
        )
    return out


def examples_for_residual_model(record: dict[str, Any], rows: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in payloads:
        model_record = dict(record)
        model_record["lattice"] = dict(payload["base_lattice"])
        model_record["wa_table"] = [dict(row) for row in rows]
        for row in model_record["wa_table"]:
            row.pop("free_params", None)
        out.append(
            {
                "record": model_record,
                "base_params": payload["base_params"],
                "base_lattice": payload["base_lattice"],
                "source_sample_id": payload["source_sample_id"],
                "source_rank": int(payload["source_rank"]),
                "source_distance": float(payload["source_distance"]),
                "align_cost": float(payload["align_cost"]),
                "sample_weight": 1.0,
            }
        )
    return out


@torch.no_grad()
def predict_geometry(
    *,
    model: train_resid.SourceResidualGeometryNet | None,
    vocabs: dict[str, dict[str, int]] | None,
    lattice_mean: torch.Tensor | None,
    lattice_std: torch.Tensor | None,
    device: torch.device,
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> list[tuple[dict[str, float], dict[int, dict[str, float]]]]:
    if model is None or vocabs is None or lattice_mean is None or lattice_std is None:
        return [(dict(payload["base_lattice"]), dict(payload["base_params"])) for payload in payloads]
    examples = examples_for_residual_model(record, rows, payloads)
    batch = train_resid.collate_residual(examples, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu())
    batch = train_resid.move_batch(batch, device)
    lattice_pred, coord_pred, _aux = model(batch)
    out: list[tuple[dict[str, float], dict[int, dict[str, float]]]] = []
    for idx, _payload in enumerate(payloads):
        lattice = source_render.decode_lattice(lattice_pred[idx], lattice_mean, lattice_std, int(record["sg"]))
        params = source_render.decode_params(coord_pred[idx], rows)
        out.append((lattice, params))
    return out


def build_group(
    *,
    engine: OrbitEngine,
    selector: rb.OpentryGeometrySelector,
    model: train_resid.SourceResidualGeometryNet | None,
    vocabs: dict[str, dict[str, int]] | None,
    lattice_mean: torch.Tensor | None,
    lattice_std: torch.Tensor | None,
    device: torch.device,
    record: dict[str, Any],
    split: str,
    source_pool_k: int,
    exclude_self: bool,
    vpa_strength: float,
    geometry_distance_weight: float,
    original_rank_weight: float,
) -> dict[str, Any] | None:
    payloads = candidate_payloads(
        selector=selector,
        record=record,
        source_pool_k=int(source_pool_k),
        exclude_self=bool(exclude_self),
        vpa_strength=float(vpa_strength),
    )
    if not payloads:
        return None
    rows = rb.v2.canonical_rows(record)
    rendered_geometries = predict_geometry(
        model=model,
        vocabs=vocabs,
        lattice_mean=lattice_mean,
        lattice_std=lattice_std,
        device=device,
        record=record,
        rows=rows,
        payloads=payloads,
    )
    candidates: list[dict[str, Any]] = []
    for idx, (payload, (lattice, params)) in enumerate(zip(payloads, rendered_geometries)):
        rendered = rb.v2.render_candidate(
            engine,
            record,
            {"rows": rows, "params": params, "lattice": lattice},
            idx + 1,
            "source_render_quality_label",
        )
        cif = str(rendered.get("cif") or "")
        metric = validate_cif(cif, record["formula_counts"], int(record["sg"])) if cif else {
            "readable": False,
            "formula_ok": False,
            "atom_count_ok": False,
            "composition_exact": False,
            "sg_ok": False,
        }
        label_row = {
            **metric,
            **selfscore.cif_self_features(cif),
            "rank": idx + 1,
            "original_rank": idx + 1,
            "geometry_rank": idx,
            "geometry_distance": float(payload["source_distance"]),
            "atom_count_after_expansion": metric.get("atom_count_after_expansion"),
            "detected_sg": metric.get("detected_sg"),
        }
        health_score = render_health_score(
            label_row,
            geometry_distance_weight=float(geometry_distance_weight),
            original_rank_weight=float(original_rank_weight),
        )
        candidates.append(
            {
                "source_sample_id": payload["source_sample_id"],
                "source_rank": int(payload["source_rank"]),
                "features": payload["features"],
                "combined_error": float(-health_score),
                "render_health_score": float(health_score),
                "lattice_error": float(-health_score),
                "coord_error": 0.0,
                "readable": bool(metric.get("readable")),
                "formula_ok": bool(metric.get("formula_ok")),
                "atom_count_ok": bool(metric.get("atom_count_ok")),
                "composition_exact": bool(metric.get("composition_exact")),
                "sg_ok": bool(metric.get("sg_ok")),
                "self_min_distance": label_row.get("self_min_distance"),
                "self_volume_per_atom": label_row.get("self_volume_per_atom"),
                "source_distance": float(payload["source_distance"]),
                "align_cost": float(payload["align_cost"]),
            }
        )
    best_index = min(range(len(candidates)), key=lambda idx: (float(candidates[idx]["combined_error"]), int(candidates[idx]["source_rank"])))
    heuristic_error = float(candidates[0]["combined_error"])
    best_error = float(candidates[best_index]["combined_error"])
    return {
        "split": str(split),
        "sample_id": str(record.get("sample_id")),
        "material_id": str(record.get("material_id")),
        "sg": int(record["sg"]),
        "atom_count": int(rb.gb.atom_count(record)),
        "num_elements": int(record.get("num_elements", len(record.get("formula_counts") or {}))),
        "row_count_label": int(len(record.get("wa_table") or [])),
        "complex_flags": {
            "rows_ge_7": int(len(record.get("wa_table") or [])) >= 7,
            "atoms_ge_12": int(rb.gb.atom_count(record)) >= 12,
            "num_elements_ge_4": int(record.get("num_elements", len(record.get("formula_counts") or {}))) >= 4,
        },
        "canonical_skeleton_key": str(record.get("canonical_skeleton_key")),
        "canonical_wa_key": str(record.get("canonical_wa_key")),
        "source_pool_k": int(source_pool_k),
        "vpa_strength": float(vpa_strength),
        "label_kind": "GT-free rendered CIF health score; no StructureMatcher",
        "candidates": candidates,
        "best_source_index": int(best_index),
        "best_source_rank": int(candidates[best_index]["source_rank"]),
        "heuristic_error": heuristic_error,
        "best_error": best_error,
        "oracle_gain": float(heuristic_error - best_error),
        "heuristic_is_best": int(best_index) == 0,
    }


def summarize(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"groups": 0}

    def sub(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"groups": 0}
        all_cands = [cand for group in items for cand in group.get("candidates") or []]
        return {
            "groups": len(items),
            "candidate_count_mean": float(mean([len(group["candidates"]) for group in items])),
            "heuristic_is_best_rate": float(mean([1.0 if group["heuristic_is_best"] else 0.0 for group in items])),
            "best_source_rank_mean": float(mean([float(group["best_source_rank"]) for group in items])),
            "heuristic_health_score_mean": float(mean([-float(group["heuristic_error"]) for group in items])),
            "best_health_score_mean": float(mean([-float(group["best_error"]) for group in items])),
            "oracle_health_gain_mean": float(mean([float(group["oracle_gain"]) for group in items])),
            "oracle_gain_positive_rate": float(mean([1.0 if float(group["oracle_gain"]) > 1e-8 else 0.0 for group in items])),
            "candidate_readable_rate": float(mean([1.0 if cand.get("readable") else 0.0 for cand in all_cands])) if all_cands else None,
            "candidate_composition_exact_rate": float(mean([1.0 if cand.get("composition_exact") else 0.0 for cand in all_cands])) if all_cands else None,
            "candidate_sg_ok_rate": float(mean([1.0 if cand.get("sg_ok") else 0.0 for cand in all_cands])) if all_cands else None,
        }

    rows_ge7 = [group for group in groups if group["complex_flags"]["rows_ge_7"]]
    atoms_ge12 = [group for group in groups if group["complex_flags"]["atoms_ge_12"]]
    return {"full": sub(groups), "rows_ge_7": sub(rows_ge7), "atoms_ge_12": sub(atoms_ge12)}


def load_residual_runtime(ckpt_path: Path | None, device: torch.device) -> tuple[
    train_resid.SourceResidualGeometryNet | None,
    dict[str, dict[str, int]] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, Any] | None,
]:
    if ckpt_path is None:
        return None, None, None, None, None
    ckpt = torch.load(ensure_under_opentry(ckpt_path), map_location="cpu")
    vocabs = ckpt["vocabs"]
    config = dict(ckpt.get("config") or {})
    model = train_resid.SourceResidualGeometryNet(
        {name: len(vocab) for name, vocab in vocabs.items()},
        hidden_dim=int(config.get("hidden_dim", 256)),
        emb_dim=int(config.get("emb_dim", 64)),
        lattice_delta_scale=float(config.get("lattice_delta_scale", 0.75)),
        coord_delta_scale=float(config.get("coord_delta_scale", 0.35)),
        enable_delta_gate=bool(config.get("enable_delta_gate", False)),
        gate_bias_init=float(config.get("gate_bias_init", -1.0)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return (
        model,
        vocabs,
        torch.tensor(ckpt["lattice_mean"], dtype=torch.float32),
        torch.tensor(ckpt["lattice_std"], dtype=torch.float32),
        {
            "path": str(ensure_under_opentry(ckpt_path)),
            "best_epoch": ckpt.get("best_epoch"),
            "config": config,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-mode labels from GT-free rendered CIF health scores.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--geometry-ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selector-max-train-records", type=int, default=0)
    parser.add_argument("--start-train-index", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=1024)
    parser.add_argument("--start-val-index", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=256)
    parser.add_argument("--source-pool-k", type=int, default=8)
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--geometry-distance-weight", type=float, default=5.0)
    parser.add_argument("--original-rank-weight", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = out_dir / "train_source_mode_examples.jsonl"
    val_out = out_dir / "val_source_mode_examples.jsonl"
    if not args.overwrite and (train_out.exists() or val_out.exists()):
        raise SystemExit(f"Refusing to overwrite existing examples in {out_dir}; pass --overwrite")

    all_train = rb.load_records(args.data_root / "train.jsonl")
    all_val = rb.load_records(args.data_root / "val.jsonl")
    selector_records = all_train[: int(args.selector_max_train_records)] if int(args.selector_max_train_records) > 0 else all_train
    selector = train_resid.build_selector(selector_records)
    train_targets = select_records(all_train, int(args.start_train_index), int(args.max_train_records))
    val_targets = select_records(all_val, int(args.start_val_index), int(args.max_val_records))
    sg_symbols = rb.v2.sg_symbols_from_splits({"train": all_train, "val": all_val})
    engine = OrbitEngine(str(args.lookup_json), sg_symbols)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, vocabs, lattice_mean, lattice_std, residual_runtime = load_residual_runtime(args.geometry_ckpt, device)

    split_groups: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split, targets, exclude_self in (("train", train_targets, True), ("val", val_targets, False)):
        groups = split_groups[split]
        for idx, record in enumerate(targets, start=1):
            group = build_group(
                engine=engine,
                selector=selector,
                model=model,
                vocabs=vocabs,
                lattice_mean=lattice_mean,
                lattice_std=lattice_std,
                device=device,
                record=record,
                split=split,
                source_pool_k=int(args.source_pool_k),
                exclude_self=bool(exclude_self),
                vpa_strength=float(args.vpa_strength),
                geometry_distance_weight=float(args.geometry_distance_weight),
                original_rank_weight=float(args.original_rank_weight),
            )
            if group is not None:
                groups.append(group)
            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                print(json.dumps({"split": split, "seen": idx, "groups": len(groups)}, sort_keys=True), flush=True)

    write_jsonl(train_out, split_groups["train"])
    write_jsonl(val_out, split_groups["val"])
    summary = {
        "config": jsonable_args(args),
        "residual_runtime": residual_runtime,
        "selector_train_records": len(selector_records),
        "all_train_records": len(all_train),
        "all_val_records": len(all_val),
        "train": summarize(split_groups["train"]),
        "val": summarize(split_groups["val"]),
        "outputs": {"train": str(train_out), "val": str(val_out)},
        "leakage_guard": {
            "selector_split": "train",
            "train_targets": "train",
            "val_targets": "val",
            "test_records_used": 0,
            "structurematcher_labels_used": False,
            "label_kind": "GT-free rendered CIF health score from formula/GT-SG validity, min-distance and source distance",
            "gt_wa_usage": "train/val GT W/A only to render source-quality labels on the same split; not used for test or final selection",
        },
    }
    write_json(out_dir / "source_render_quality_examples_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
