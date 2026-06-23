#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_render_wyckoff_cifs_e07e08 as rb  # noqa: E402
import opentry_build_source_mode_examples as source_mode_data  # noqa: E402
import opentry_train_source_residual_geometry as train_resid  # noqa: E402
import opentry_train_source_mode_selector as source_mode_train  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


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


def decode_lattice(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, sg: int) -> dict[str, float]:
    values = (raw.detach().cpu() * std.cpu() + mean.cpu()).tolist()
    return rb.v2.lattice_from_target([float(x) for x in values], int(sg))


def decode_params(coord_pred: torch.Tensor, rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    coord_index = {"x": 0, "y": 1, "z": 2}
    values = coord_pred.detach().cpu()
    params: dict[int, dict[str, float]] = {}
    for row_idx, row in enumerate(rows):
        row_params: dict[str, float] = {}
        for sym in row.get("free_symbols") or []:
            if str(sym) in coord_index:
                row_params[str(sym)] = float(values[row_idx, coord_index[str(sym)]]) % 1.0
        params[int(row_idx)] = row_params
    return params


def load_source_mode_runtime(ckpt_path: Path | None, device: torch.device) -> dict[str, Any] | None:
    if ckpt_path is None:
        return None
    ckpt = torch.load(ensure_under_opentry(ckpt_path), map_location="cpu")
    feature_names = [str(x) for x in ckpt["feature_names"]]
    config = dict(ckpt.get("config") or {})
    model = source_mode_train.SourceModeScorer(
        len(feature_names),
        hidden_dim=int(config.get("hidden_dim", 96)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "feature_names": feature_names,
        "feature_mean": torch.tensor(ckpt["feature_mean"], dtype=torch.float32, device=device),
        "feature_std": torch.tensor(ckpt["feature_std"], dtype=torch.float32, device=device),
        "path": str(ensure_under_opentry(ckpt_path)),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_selected_error": ckpt.get("best_val_selected_error"),
    }


@torch.no_grad()
def select_sources_with_mode_model(
    *,
    selector: rb.OpentryGeometrySelector,
    pred_record: dict[str, Any],
    source_mode_runtime: dict[str, Any],
    source_pool_k: int,
    vpa_strength: float,
    min_margin: float,
    top_n: int,
    include_rank0: bool,
    exclude_sample_id: str | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    sources = train_resid.fast_chem_quality_candidates(selector, pred_record, int(source_pool_k))
    if exclude_sample_id is not None:
        sources = [source for source in sources if str(source.get("sample_id")) != str(exclude_sample_id)]
    if not sources:
        return []
    feature_names = list(source_mode_runtime["feature_names"])
    rows: list[list[float]] = []
    payloads: list[dict[str, Any]] = []
    for source_rank, source in enumerate(sources[: int(source_pool_k)]):
        params, align_cost = selector.source_aligned_params(pred_record, source, 0, chemical=True)
        base_lattice = selector.vpa_calibrated_lattice(source, pred_record, strength=float(vpa_strength))
        features = source_mode_data.numeric_source_features(
            selector=selector,
            record=pred_record,
            source=source,
            source_rank=source_rank,
            align_cost=align_cost,
            source_pool_k=int(source_pool_k),
            base_lattice=base_lattice,
        )
        rows.append(source_mode_train.vector_for_candidate({"features": features}, feature_names))
        payloads.append(
            {
                "source": source,
                "source_rank": int(source_rank),
                "base_params": params,
                "base_lattice": base_lattice,
                "align_cost": float(align_cost),
                "features": features,
            }
        )
    if not payloads:
        return []
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    x = (x - source_mode_runtime["feature_mean"]) / source_mode_runtime["feature_std"].clamp_min(1e-4)
    scores = source_mode_runtime["model"](x)
    score_values = [float(v) for v in scores.detach().cpu().tolist()]
    order = sorted(range(len(score_values)), key=lambda idx: score_values[idx], reverse=True)
    raw_best_idx = int(order[0])
    raw_best_score = float(scores[raw_best_idx].detach().cpu())
    rank0_score = float(scores[0].detach().cpu())
    margin = float(raw_best_score - rank0_score)
    overrode_to_rank0 = bool(raw_best_idx != 0 and margin < float(min_margin))
    if raw_best_idx != 0 and margin < float(min_margin):
        selected_indices = [0]
    else:
        selected_indices = list(order[: max(1, int(top_n))])
        if include_rank0 and 0 not in selected_indices:
            selected_indices = [0] + selected_indices
            selected_indices = selected_indices[: max(1, int(top_n))]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for choice_rank, selected_idx in enumerate(selected_indices):
        source_id = str(payloads[selected_idx]["source"].get("sample_id"))
        if source_id in seen:
            continue
        seen.add(source_id)
        payload = dict(payloads[selected_idx])
        payload["source_mode_score"] = float(scores[selected_idx].detach().cpu())
        payload["source_mode_scores"] = score_values
        payload["source_mode_raw_best_idx"] = int(raw_best_idx)
        payload["source_mode_raw_best_score"] = float(raw_best_score)
        payload["source_mode_rank0_score"] = float(rank0_score)
        payload["source_mode_margin"] = float(margin)
        payload["source_mode_min_margin"] = float(min_margin)
        payload["source_mode_overrode_to_rank0"] = bool(overrode_to_rank0)
        payload["source_mode_choice_rank"] = int(choice_rank)
        out.append(payload)
    return out


def select_sources_from_fast_pool(
    *,
    selector: rb.OpentryGeometrySelector,
    pred_record: dict[str, Any],
    source_pool_k: int,
    vpa_strength: float,
    top_n: int,
    exclude_sample_id: str | None,
) -> list[dict[str, Any]]:
    sources = train_resid.fast_chem_quality_candidates(selector, pred_record, int(source_pool_k))
    if exclude_sample_id is not None:
        sources = [source for source in sources if str(source.get("sample_id")) != str(exclude_sample_id)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_rank, source in enumerate(sources[: int(source_pool_k)]):
        source_id = str(source.get("sample_id"))
        if source_id in seen:
            continue
        seen.add(source_id)
        params, align_cost = selector.source_aligned_params(pred_record, source, 0, chemical=True)
        base_lattice = selector.vpa_calibrated_lattice(source, pred_record, strength=float(vpa_strength))
        out.append(
            {
                "source": source,
                "source_rank": int(source_rank),
                "source_choice_rank": len(out),
                "base_params": params,
                "base_lattice": base_lattice,
                "align_cost": float(align_cost),
            }
        )
        if len(out) >= max(1, int(top_n)):
            break
    return out


@torch.no_grad()
def select_source_with_mode_model(
    *,
    selector: rb.OpentryGeometrySelector,
    pred_record: dict[str, Any],
    source_mode_runtime: dict[str, Any],
    source_pool_k: int,
    vpa_strength: float,
    min_margin: float,
    exclude_sample_id: str | None,
    device: torch.device,
) -> dict[str, Any] | None:
    payloads = select_sources_with_mode_model(
        selector=selector,
        pred_record=pred_record,
        source_mode_runtime=source_mode_runtime,
        source_pool_k=int(source_pool_k),
        vpa_strength=float(vpa_strength),
        min_margin=float(min_margin),
        top_n=1,
        include_rank0=False,
        exclude_sample_id=exclude_sample_id,
        device=device,
    )
    return None if not payloads else payloads[0]


def make_inference_example(
    *,
    selector: rb.OpentryGeometrySelector,
    target_record: dict[str, Any],
    rows: list[dict[str, Any]],
    source_pool_k: int,
    vpa_strength: float,
    source_mode_runtime: dict[str, Any] | None = None,
    source_mode_pool_k: int = 0,
    source_mode_min_margin: float = 0.0,
    exclude_self_source: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any] | None:
    pred_record = rb.pseudo_record(target_record, rows)
    exclude_sample_id = str(target_record.get("sample_id")) if bool(exclude_self_source) else None
    source_mode_payload: dict[str, Any] | None = None
    if source_mode_runtime is not None:
        source_mode_payload = select_source_with_mode_model(
            selector=selector,
            pred_record=pred_record,
            source_mode_runtime=source_mode_runtime,
            source_pool_k=int(source_mode_pool_k or source_pool_k),
            vpa_strength=float(vpa_strength),
            min_margin=float(source_mode_min_margin),
            exclude_sample_id=exclude_sample_id,
            device=device or torch.device("cpu"),
        )
    if source_mode_payload is None:
        source = train_resid.select_source(
            selector,
            pred_record,
            source_pool_k=int(source_pool_k),
            exclude_sample_id=exclude_sample_id,
        )
        if source is None:
            return None
        base_params, align_cost = selector.source_aligned_params(pred_record, source, 0, chemical=True)
        base_lattice = selector.vpa_calibrated_lattice(source, pred_record, strength=float(vpa_strength))
        source_rank = 0
        source_mode_score = None
    else:
        source = dict(source_mode_payload["source"])
        base_params = source_mode_payload["base_params"]
        align_cost = float(source_mode_payload["align_cost"])
        base_lattice = dict(source_mode_payload["base_lattice"])
        source_rank = int(source_mode_payload["source_rank"])
        source_mode_score = source_mode_payload.get("source_mode_score")
    source_mode_details = dict(source_mode_payload or {})
    # Avoid carrying validation/test lattice into the model input batch. These fields
    # are unused by the forward pass, but keeping them source-derived makes the
    # artifact unambiguous under leakage audits.
    model_record = dict(pred_record)
    model_record["lattice"] = dict(base_lattice)
    model_record["wa_table"] = [dict(row) for row in rows]
    for row in model_record["wa_table"]:
        row.pop("free_params", None)
    return {
        "record": model_record,
        "base_params": base_params,
        "base_lattice": base_lattice,
        "source_sample_id": str(source.get("sample_id")),
        "source_rank": int(source_rank),
        "source_mode_score": source_mode_score,
        "source_mode_used": source_mode_runtime is not None and source_mode_payload is not None,
        "source_mode_margin": source_mode_details.get("source_mode_margin"),
        "source_mode_min_margin": source_mode_details.get("source_mode_min_margin"),
        "source_mode_raw_best_idx": source_mode_details.get("source_mode_raw_best_idx"),
        "source_mode_overrode_to_rank0": source_mode_details.get("source_mode_overrode_to_rank0"),
        "source_distance": float(rb.gb.row_condition_distance(pred_record, source)),
        "align_cost": float(align_cost),
        "sample_weight": 1.0,
        "render_record": pred_record,
        "rows": rows,
    }


def make_inference_examples(
    *,
    selector: rb.OpentryGeometrySelector,
    target_record: dict[str, Any],
    rows: list[dict[str, Any]],
    source_pool_k: int,
    vpa_strength: float,
    source_mode_runtime: dict[str, Any] | None = None,
    source_mode_pool_k: int = 0,
    source_mode_min_margin: float = 0.0,
    source_mode_mixture_size: int = 1,
    source_mode_include_rank0: bool = False,
    source_expand_k: int = 1,
    exclude_self_source: bool = False,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    exclude_sample_id = str(target_record.get("sample_id")) if bool(exclude_self_source) else None
    if source_mode_runtime is None or int(source_mode_mixture_size) <= 1:
        if source_mode_runtime is None and int(source_expand_k) > 1:
            pred_record = rb.pseudo_record(target_record, rows)
            payloads = select_sources_from_fast_pool(
                selector=selector,
                pred_record=pred_record,
                source_pool_k=int(source_pool_k),
                vpa_strength=float(vpa_strength),
                top_n=int(source_expand_k),
                exclude_sample_id=exclude_sample_id,
            )
            examples: list[dict[str, Any]] = []
            for payload in payloads:
                source = dict(payload["source"])
                base_params = payload["base_params"]
                align_cost = float(payload["align_cost"])
                base_lattice = dict(payload["base_lattice"])
                model_record = dict(pred_record)
                model_record["lattice"] = dict(base_lattice)
                model_record["wa_table"] = [dict(row) for row in rows]
                for row in model_record["wa_table"]:
                    row.pop("free_params", None)
                examples.append(
                    {
                        "record": model_record,
                        "base_params": base_params,
                        "base_lattice": base_lattice,
                        "source_sample_id": str(source.get("sample_id")),
                        "source_rank": int(payload["source_rank"]),
                        "source_choice_rank": int(payload["source_choice_rank"]),
                        "source_mode_score": None,
                        "source_mode_used": False,
                        "source_mode_margin": None,
                        "source_mode_min_margin": None,
                        "source_mode_raw_best_idx": None,
                        "source_mode_overrode_to_rank0": None,
                        "source_mode_choice_rank": None,
                        "source_distance": float(rb.gb.row_condition_distance(pred_record, source)),
                        "align_cost": float(align_cost),
                        "sample_weight": 1.0,
                        "render_record": pred_record,
                        "rows": rows,
                    }
                )
            return examples
        example = make_inference_example(
            selector=selector,
            target_record=target_record,
            rows=rows,
            source_pool_k=int(source_pool_k),
            vpa_strength=float(vpa_strength),
            source_mode_runtime=source_mode_runtime,
            source_mode_pool_k=int(source_mode_pool_k),
            source_mode_min_margin=float(source_mode_min_margin),
            exclude_self_source=bool(exclude_self_source),
            device=device,
        )
        if example is not None:
            example["source_choice_rank"] = 0
        return [] if example is None else [example]

    pred_record = rb.pseudo_record(target_record, rows)
    payloads = select_sources_with_mode_model(
        selector=selector,
        pred_record=pred_record,
        source_mode_runtime=source_mode_runtime,
        source_pool_k=int(source_mode_pool_k or source_pool_k),
        vpa_strength=float(vpa_strength),
        min_margin=float(source_mode_min_margin),
        top_n=int(source_mode_mixture_size),
        include_rank0=bool(source_mode_include_rank0),
        exclude_sample_id=exclude_sample_id,
        device=device or torch.device("cpu"),
    )
    examples: list[dict[str, Any]] = []
    for payload in payloads:
        source = dict(payload["source"])
        base_params = payload["base_params"]
        align_cost = float(payload["align_cost"])
        base_lattice = dict(payload["base_lattice"])
        source_mode_details = dict(payload)
        model_record = dict(pred_record)
        model_record["lattice"] = dict(base_lattice)
        model_record["wa_table"] = [dict(row) for row in rows]
        for row in model_record["wa_table"]:
            row.pop("free_params", None)
        examples.append(
            {
                "record": model_record,
                "base_params": base_params,
                "base_lattice": base_lattice,
                "source_sample_id": str(source.get("sample_id")),
                "source_rank": int(payload["source_rank"]),
                "source_mode_score": payload.get("source_mode_score"),
                "source_mode_used": True,
                "source_mode_margin": source_mode_details.get("source_mode_margin"),
                "source_mode_min_margin": source_mode_details.get("source_mode_min_margin"),
                "source_mode_raw_best_idx": source_mode_details.get("source_mode_raw_best_idx"),
                "source_mode_overrode_to_rank0": source_mode_details.get("source_mode_overrode_to_rank0"),
                "source_mode_choice_rank": source_mode_details.get("source_mode_choice_rank"),
                "source_choice_rank": int(payload.get("source_mode_choice_rank") or 0),
                "source_distance": float(rb.gb.row_condition_distance(pred_record, source)),
                "align_cost": float(align_cost),
                "sample_weight": 1.0,
                "render_record": pred_record,
                "rows": rows,
            }
        )
    return examples


@torch.no_grad()
def render_for_record(
    *,
    engine: OrbitEngine,
    selector: rb.OpentryGeometrySelector,
    model: train_resid.SourceResidualGeometryNet,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    target_record: dict[str, Any],
    prediction: dict[str, Any],
    top_k: int,
    source_pool_k: int,
    vpa_strength: float,
    source_mode_runtime: dict[str, Any] | None,
    source_mode_pool_k: int,
    source_mode_min_margin: float,
    source_mode_mixture_size: int,
    source_mode_include_rank0: bool,
    source_expand_k: int,
    exclude_self_source: bool,
    device: torch.device,
) -> list[dict[str, Any]]:
    candidates = list(prediction.get("ranked_wa_candidates") or [])
    examples: list[dict[str, Any]] = []
    seen_wa: set[str] = set()
    wa_count = 0
    for candidate in candidates:
        if wa_count >= int(top_k):
            break
        rows = rb.normalize_rows(engine, list(candidate.get("rows") or []))
        skel, wa = rb.v2.canonical_keys_from_rows(rows)
        if wa in seen_wa:
            continue
        new_examples = make_inference_examples(
            selector=selector,
            target_record=target_record,
            rows=rows,
            source_pool_k=int(source_pool_k),
            vpa_strength=float(vpa_strength),
            source_mode_runtime=source_mode_runtime,
            source_mode_pool_k=int(source_mode_pool_k),
            source_mode_min_margin=float(source_mode_min_margin),
            source_mode_mixture_size=int(source_mode_mixture_size),
            source_mode_include_rank0=bool(source_mode_include_rank0),
            source_expand_k=int(source_expand_k),
            exclude_self_source=bool(exclude_self_source),
            device=device,
        )
        if not new_examples:
            continue
        seen_wa.add(wa)
        wa_count += 1
        for example in new_examples:
            example["candidate_score"] = candidate.get("score")
            example["canonical_skeleton_key"] = skel
            example["canonical_wa_key"] = wa
            example["source_mode_mixture_size"] = int(source_mode_mixture_size)
            example["wa_rank"] = int(wa_count)
            examples.append(example)
    if not examples:
        return []

    batch = train_resid.collate_residual(examples, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu())
    batch = train_resid.move_batch(batch, device)
    lattice_pred, coord_pred, _aux = model(batch)
    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for idx, example in enumerate(examples):
        rows = list(example["rows"])
        lattice = decode_lattice(lattice_pred[idx], lattice_mean, lattice_std, int(example["render_record"]["sg"]))
        params = decode_params(coord_pred[idx], rows)
        rendered = rb.v2.render_candidate(
            engine,
            example["render_record"],
            {"rows": rows, "params": params, "lattice": lattice},
            len(out) + 1,
            "source_residual_geometry",
        )
        cif = str(rendered.get("cif") or "")
        if not cif:
            continue
        cif_hash = hashlib.sha1(cif.encode("utf-8", errors="ignore")).hexdigest()
        if cif_hash in seen_hashes:
            continue
        seen_hashes.add(cif_hash)
        metric = validate_cif(cif, example["render_record"]["formula_counts"], int(example["render_record"]["sg"]))
        out.append(
            {
                "sample_id": target_record["sample_id"],
                "rank": len(out) + 1,
                "geometry_mode": "source_residual",
                "geometry_source": "source_mode_selector" if example.get("source_mode_used") else "row_aligned_chem_quality",
                "geometry_lattice_mode": "source_residual",
                "source_sample_id": example["source_sample_id"],
                "source_rank": int(example.get("source_rank", 0)),
                "source_mode_score": example.get("source_mode_score"),
                "source_mode_margin": example.get("source_mode_margin"),
                "source_mode_min_margin": example.get("source_mode_min_margin"),
                "source_mode_raw_best_idx": example.get("source_mode_raw_best_idx"),
                "source_mode_overrode_to_rank0": example.get("source_mode_overrode_to_rank0"),
                "source_mode_choice_rank": example.get("source_mode_choice_rank"),
                "source_choice_rank": example.get("source_choice_rank"),
                "source_mode_mixture_size": example.get("source_mode_mixture_size"),
                "wa_rank": example.get("wa_rank"),
                "geometry_distance": float(example["source_distance"]),
                "align_cost": float(example["align_cost"]),
                "canonical_wa_key": example["canonical_wa_key"],
                "canonical_skeleton_key": example["canonical_skeleton_key"],
                "candidate_score": example.get("candidate_score"),
                "cif": cif,
                **metric,
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(row.get("readable")) for row in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(row.get("formula_ok")) for row in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(row.get("atom_count_ok")) for row in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(row.get("sg_ok")) for row in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(row.get("composition_exact")) for row in rows) / max(1, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render opentry_3 W/A predictions with a source-conditioned residual geometry model.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--geometry-ckpt", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-row-count", type=int, default=0)
    parser.add_argument("--source-pool-k", type=int, default=24)
    parser.add_argument("--source-mode-ckpt", type=Path, default=None)
    parser.add_argument("--source-mode-pool-k", type=int, default=8)
    parser.add_argument("--source-mode-min-margin", type=float, default=0.0)
    parser.add_argument("--source-mode-mixture-size", type=int, default=1)
    parser.add_argument("--source-mode-include-rank0", action="store_true")
    parser.add_argument("--source-expand-k", type=int, default=1)
    parser.add_argument("--exclude-self-source", action="store_true")
    parser.add_argument("--vpa-strength", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-cif-files", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--resume-partial", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = out_dir / "rendered_cifs"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    train_records = rb.load_records(args.data_root / "train.jsonl")
    split_records = rb.load_records(args.data_root / f"{args.split}.jsonl")
    pred_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    split_records = [record for record in split_records if str(record["sample_id"]) in pred_by_id]
    if int(args.min_row_count) > 0:
        split_records = [record for record in split_records if len(record.get("wa_table") or []) >= int(args.min_row_count)]
    total_filtered_records = len(split_records)
    if int(args.start_index) > 0:
        split_records = split_records[int(args.start_index) :]
    if int(args.max_records) > 0:
        split_records = split_records[: int(args.max_records)]
    selector = train_resid.build_selector(train_records)
    sg_symbols = rb.v2.sg_symbols_from_splits({"train": train_records, "eval": split_records})
    engine = OrbitEngine(str(args.lookup_json), sg_symbols)

    ckpt = torch.load(args.geometry_ckpt, map_location="cpu")
    vocabs = ckpt["vocabs"]
    config = ckpt.get("config") or {}
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
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    lattice_mean = torch.tensor(ckpt["lattice_mean"], dtype=torch.float32)
    lattice_std = torch.tensor(ckpt["lattice_std"], dtype=torch.float32)
    source_mode_runtime = load_source_mode_runtime(args.source_mode_ckpt, device)

    partial_jsonl = out_dir / "rendered_topk.partial.jsonl"
    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if bool(args.resume_partial) and partial_jsonl.exists():
        rows = read_jsonl(partial_jsonl)
        completed_ids = {str(row["sample_id"]) for row in rows}
    partial_mode = "a" if bool(args.resume_partial) and partial_jsonl.exists() else "w"
    start_time = time.time()
    with partial_jsonl.open(partial_mode, encoding="utf-8") as partial_f:
        for local_idx, record in enumerate(split_records, start=1):
            sample_id = str(record["sample_id"])
            if sample_id in completed_ids:
                continue
            rendered = render_for_record(
                engine=engine,
                selector=selector,
                model=model,
                vocabs=vocabs,
                lattice_mean=lattice_mean,
                lattice_std=lattice_std,
                target_record=record,
                prediction=pred_by_id[sample_id],
                top_k=int(args.top_k),
                source_pool_k=int(args.source_pool_k),
                vpa_strength=float(args.vpa_strength),
                source_mode_runtime=source_mode_runtime,
                source_mode_pool_k=int(args.source_mode_pool_k),
                source_mode_min_margin=float(args.source_mode_min_margin),
                source_mode_mixture_size=int(args.source_mode_mixture_size),
                source_mode_include_rank0=bool(args.source_mode_include_rank0),
                source_expand_k=int(args.source_expand_k),
                exclude_self_source=bool(args.exclude_self_source),
                device=device,
            )
            for row in rendered:
                partial_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            partial_f.flush()
            rows.extend(rendered)
            completed_ids.add(sample_id)
            if int(args.progress_every) > 0 and (local_idx % int(args.progress_every) == 0 or local_idx == len(split_records)):
                progress = {
                    "completed_records": len(completed_ids),
                    "current_batch_index": int(local_idx),
                    "selected_records": len(split_records),
                    "rendered_rows": len(rows),
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }
                write_json(out_dir / "render_progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)

    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["rank"])))
    with (out_dir / "rendered_topk.jsonl").open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if idx < int(args.write_cif_files):
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sample_id"]))
                path = rendered_dir / f"{safe_id}_rank{int(row['rank'])}.cif"
                path.write_text(str(row["cif"]), encoding="utf-8")
                row = dict(row)
                row["cif_path"] = str(path)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "split": str(args.split),
        "top_k": int(args.top_k),
        "start_index": int(args.start_index),
        "max_records": int(args.max_records),
        "min_row_count": int(args.min_row_count),
        "geometry_ckpt": str(args.geometry_ckpt),
        "source_pool_k": int(args.source_pool_k),
        "source_mode_ckpt": None if args.source_mode_ckpt is None else str(ensure_under_opentry(args.source_mode_ckpt)),
        "source_mode_pool_k": int(args.source_mode_pool_k),
        "source_mode_min_margin": float(args.source_mode_min_margin),
        "source_mode_mixture_size": int(args.source_mode_mixture_size),
        "source_mode_include_rank0": bool(args.source_mode_include_rank0),
        "source_expand_k": int(args.source_expand_k),
        "exclude_self_source": bool(args.exclude_self_source),
        "source_mode_best_epoch": None if source_mode_runtime is None else source_mode_runtime.get("best_epoch"),
        "source_mode_best_val_selected_error": None if source_mode_runtime is None else source_mode_runtime.get("best_val_selected_error"),
        "vpa_strength": float(args.vpa_strength),
        "total_filtered_records": int(total_filtered_records),
        "partial_jsonl": str(partial_jsonl),
        "resume_partial": bool(args.resume_partial),
        "samples_with_prediction_rows": len(split_records),
        "samples_with_rendered_candidates": len({str(row["sample_id"]) for row in rows}),
        "rendered_rows": len(rows),
        "overall_rows": summarize(rows),
        "rank1": summarize([row for row in rows if int(row["rank"]) == 1]),
        "rank_le_5": summarize([row for row in rows if int(row["rank"]) <= 5]),
        "rank_le_20": summarize([row for row in rows if int(row["rank"]) <= 20]),
    }
    write_json(out_dir / "render_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
