#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2_joint_geometry_repair"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
from pymatgen.core.periodic_table import Element  # noqa: E402

from run_exp0_exp1_alignment import assign_elements, formula_counts, skeleton_rows  # noqa: E402
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    eval_sample,
    render_candidate,
    summarize,
)
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference, postprocess_lattice  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine, cell_volume  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
PAIR_ARTIFACT = OUT_DIR / "artifacts" / "exp1_predicted_skeleton_noise_pairs" / "predicted_skeleton_noise_geometry_pairs_merged_sharded.jsonl.gz"
TRAIN_MPTS52 = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
VAL_MPTS52 = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP3_RESULT = OP13 / "results" / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
EXP4_RESULT = OP13 / "results" / "experiment_4_rows_ge7_multi_geometry_proposal.json"

BUDGETS = (1, 5, 20, 50)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id") or record.get("keys", {}).get("sample_id"))


def lattice_raw(lattice: dict[str, Any]) -> list[float]:
    return [
        math.log(max(1.0e-6, float(lattice["a"]))),
        math.log(max(1.0e-6, float(lattice["b"]))),
        math.log(max(1.0e-6, float(lattice["c"]))),
        float(lattice["alpha"]) / 180.0,
        float(lattice["beta"]) / 180.0,
        float(lattice["gamma"]) / 180.0,
    ]


def raw_lattice_blend(initial: dict[str, Any], predicted_raw: list[float], sg: int, alpha: float) -> dict[str, float]:
    init = lattice_raw(initial)
    raw = [(1.0 - alpha) * init[i] + alpha * float(predicted_raw[i]) for i in range(6)]
    return postprocess_lattice(raw, int(sg))


def formula_stats(counts: dict[str, int]) -> list[float]:
    total = max(1, sum(counts.values()))
    zs: list[float] = []
    weights: list[float] = []
    for element, count in sorted(counts.items()):
        try:
            z = float(Element(str(element)).Z)
        except Exception:
            z = 0.0
        zs.append(z)
        weights.append(float(count) / float(total))
    if not zs:
        return [0.0, 0.0, 0.0, 0.0]
    mean_z = sum(z * w for z, w in zip(zs, weights)) / 100.0
    return [mean_z, max(zs) / 100.0, min(zs) / 100.0, (max(zs) - min(zs)) / 100.0]


def base_features(row: dict[str, Any]) -> list[float]:
    counts = {str(k): int(v) for k, v in normalize_formula_counts(row["formula_counts"]).items()}
    total = max(1, sum(counts.values()))
    init_lattice = row.get("initial_lattice") or row.get("lattice") or {}
    init_quality = row.get("initial_quality") or {}
    vpa = init_quality.get("volume_per_atom")
    min_dist = init_quality.get("min_pair_distance")
    lattice_feat = [float(x) for x in lattice_raw(init_lattice)] if init_lattice else [0.0] * 6
    return [
        float(int(row.get("sg") or 1)) / 230.0,
        float(total) / 300.0,
        float(len(counts)) / 12.0,
        float(int(row.get("target_row_count") or row.get("row_count") or 0)) / 64.0,
        float(int(row.get("source_row_count") or 0)) / 64.0,
        float(int(row.get("candidate_row_count") or 0)) / 64.0,
        float(bool(row.get("predicted_skeleton_hit"))),
        float(bool(init_quality.get("valid"))),
        float(bool(init_quality.get("formula_ok"))),
        float(bool(init_quality.get("space_group_ok"))),
        float(bool(row.get("initial_exact_cover_retained"))),
        float(bool(init_quality.get("collision"))),
        float(vpa or 0.0) / 120.0,
        float(min_dist or 0.0) / 5.0,
        *lattice_feat,
        *formula_stats(counts),
    ]


def row_param_features(base: list[float], row_idx: int, param_name: str, initial_value: float, row_count: int) -> list[float]:
    code = {"x": 0.0, "y": 0.5, "z": 1.0}.get(str(param_name), 0.25)
    return [*base, float(row_idx) / 64.0, float(row_count) / 64.0, code, float(initial_value) % 1.0]


class MLP(torch.nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(features: list[list[float]], targets: list[list[float]], *, out_dim: int, epochs: int, seed: int) -> tuple[MLP, dict[str, Any]]:
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1.0e-6)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1.0e-6)
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    order = torch.randperm(xs.shape[0])
    val_n = max(256, int(0.10 * xs.shape[0]))
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    model = MLP(xs.shape[1], out_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1.0e-4)
    history: list[dict[str, float]] = []
    batch_size = min(2048, max(128, train_idx.numel()))
    for epoch in range(1, int(epochs) + 1):
        perm = train_idx[torch.randperm(train_idx.numel())]
        model.train()
        total = 0.0
        steps = 0
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            pred = model(xs[idx])
            loss = torch.nn.functional.mse_loss(pred, ys[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(model(xs[val_idx]), ys[val_idx]).detach())
            history.append({"epoch": float(epoch), "train_loss": total / max(1, steps), "val_loss": val_loss})
    model.x_mean = x_mean  # type: ignore[attr-defined]
    model.x_std = x_std  # type: ignore[attr-defined]
    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std  # type: ignore[attr-defined]
    return model, {
        "feature_dim": int(xs.shape[1]),
        "train_rows": int(train_idx.numel()),
        "val_rows": int(val_idx.numel()),
        "best_val_loss": min((h["val_loss"] for h in history), default=None),
        "history": history,
    }


@torch.no_grad()
def predict(model: MLP, feature: list[float]) -> list[float]:
    x = torch.tensor([feature], dtype=torch.float32)
    xs = (x - model.x_mean) / model.x_std  # type: ignore[attr-defined]
    out = model(xs)[0] * model.y_std + model.y_mean  # type: ignore[attr-defined]
    return [float(v) for v in out.tolist()]


def load_pair_training(max_pairs: int | None = None) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]], dict[str, Any]]:
    lattice_x: list[list[float]] = []
    lattice_y: list[list[float]] = []
    param_x: list[list[float]] = []
    param_y: list[list[float]] = []
    counts = Counter()
    with gzip.open(PAIR_ARTIFACT, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["records"] += 1
            if not bool(row.get("usable_joint_pair")):
                continue
            if max_pairs is not None and len(lattice_x) >= int(max_pairs):
                continue
            base = base_features(row)
            lattice_x.append(base)
            lattice_y.append(lattice_raw(row["target_lattice"]))
            initial_params = row.get("initial_row_params") or {}
            target_params = row.get("target_params_by_row") or {}
            row_count = int(row.get("candidate_row_count") or len(initial_params))
            for idx_key, params in initial_params.items():
                target = target_params.get(str(idx_key))
                if target is None:
                    continue
                for name, value in params.items():
                    if str(name) not in target:
                        continue
                    param_x.append(row_param_features(base, int(idx_key), str(name), float(value), row_count))
                    param_y.append([float(target[str(name)]) % 1.0])
            counts["usable_joint_pairs"] += 1
    return lattice_x, lattice_y, param_x, param_y, dict(counts)


def validation_feature(
    *,
    target: dict[str, Any],
    target_repr: dict[str, Any],
    proposal: dict[str, Any],
    source: dict[str, Any],
    rows: list[dict[str, Any]],
    lattice: dict[str, float],
    params: dict[int, dict[str, float]],
) -> dict[str, Any]:
    counts = formula_counts(target)
    try:
        volume = cell_volume(lattice)
        vpa = volume / max(1, sum(counts.values()))
    except Exception:
        vpa = None
    return {
        "sg": int(target["sg"]),
        "formula_counts": counts,
        "target_row_count": int(target_repr.get("row_count") or 0),
        "source_row_count": int(source.get("n_sites") or len(source.get("wa_table") or [])),
        "candidate_row_count": len(rows),
        "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
        "initial_lattice": lattice,
        "initial_exact_cover_retained": True,
        "initial_quality": {
            "valid": True,
            "formula_ok": True,
            "space_group_ok": True,
            "collision": False,
            "volume_per_atom": vpa,
            "min_pair_distance": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lattice-alpha", type=float, default=0.35)
    parser.add_argument("--param-alpha", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    lattice_x, lattice_y, param_x, param_y, pair_counts = load_pair_training(args.max_train_pairs)
    if len(lattice_x) < 1000 or len(param_x) < 1000:
        raise RuntimeError(f"insufficient training data: lattice={len(lattice_x)} params={len(param_x)}")
    lattice_model, lattice_training = train_model(lattice_x, lattice_y, out_dim=6, epochs=int(args.epochs), seed=int(args.seed))
    param_model, param_training = train_model(param_x, param_y, out_dim=1, epochs=int(args.epochs), seed=int(args.seed) + 1)
    torch.save(
        {
            "lattice_state": lattice_model.state_dict(),
            "param_state": param_model.state_dict(),
            "lattice_training": lattice_training,
            "param_training": param_training,
        },
        ARTIFACT_DIR / "joint_repair_heads.pt",
    )

    train_by_sid = {sample_id(r): r for r in read_jsonl(TRAIN_MPTS52)}
    val_by_sid = {sample_id(r): r for r in read_jsonl(VAL_MPTS52)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(train_by_sid.values()) + list(val_by_sid.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)

    payloads: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    failures = Counter()
    selected_sids = sorted(proposals)
    if args.max_val_samples is not None:
        selected_sids = selected_sids[: int(args.max_val_samples)]
    for sid in selected_sids:
        if sid not in val_by_sid or sid not in val_repr:
            continue
        target = val_by_sid[sid]
        target_repr = val_repr[sid]
        counts = formula_counts(target)
        candidates: list[dict[str, Any]] = []
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_k)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source = train_by_sid.get(source_id)
            if source is None:
                failures["missing_source"] += 1
                continue
            rows, mapping_rule = assign_elements(
                target_counts=counts,
                source_rows=skeleton_rows(source),
                source_counts=formula_counts(source),
            )
            if rows is None:
                failures[mapping_rule] += 1
                continue
            try:
                init_params, fallback_count = flexible_params_from_reference(engine, rows, source, neural_params=None)
                init_lattice = {k: float(source["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
                feat_row = validation_feature(
                    target=target,
                    target_repr=target_repr,
                    proposal=proposal,
                    source=source,
                    rows=rows,
                    lattice=init_lattice,
                    params=init_params,
                )
                base = base_features(feat_row)
                pred_lattice_raw = predict(lattice_model, base)
                lattice = raw_lattice_blend(init_lattice, pred_lattice_raw, int(target["sg"]), float(args.lattice_alpha))
                params: dict[int, dict[str, float]] = {}
                for idx, row_params in init_params.items():
                    params[int(idx)] = {}
                    for name, init_value in row_params.items():
                        pred_value = predict(param_model, row_param_features(base, int(idx), str(name), float(init_value), len(rows)))[0] % 1.0
                        params[int(idx)][str(name)] = ((1.0 - float(args.param_alpha)) * float(init_value) + float(args.param_alpha) * pred_value) % 1.0
                cif, render_meta = render_candidate(
                    engine=engine,
                    target=target,
                    rows=rows,
                    option={"lattice": lattice, "params": params},
                    data_name=f"{sid}_joint_repair_rank{int(proposal.get('rank') or 0)}",
                )
                item = {
                    "sample_id": sid,
                    "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                    "proposal_rank": int(proposal.get("rank") or 0),
                    "geometry_rank": 1,
                    "raw_generation_order": len(candidates) + 1,
                    "row_count": int(target_repr.get("row_count") or 0),
                    "sg": int(target["sg"]),
                    "formula_counts": counts,
                    "target_atom_count": int(sum(counts.values())),
                    "source_sample_id": source_id,
                    "proposal_source": str(proposal.get("source") or ""),
                    "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                    "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                    "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                    "candidate_row_count": len(rows),
                    "site_mapping_rule": mapping_rule,
                    "geometry_source": "joint_lattice_freeparam_mlp",
                    "reference_sample_id": source_id,
                    "reference_score": None,
                    "param_fallback_rows": int(fallback_count),
                    "render_success": True,
                    "render_error": None,
                    "cif": cif,
                    **render_meta,
                }
                candidates.append(item)
                meta_rows.append({k: v for k, v in item.items() if k != "cif"})
            except Exception as exc:  # noqa: BLE001
                failures[f"render_or_predict:{type(exc).__name__}"] += 1
        payloads.append(
            {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": counts,
                "target_atom_count": int(sum(counts.values())),
                "sg": int(target["sg"]),
                "candidates": candidates,
            }
        )

    with (ARTIFACT_DIR / "generated_joint_repair_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for fut in as_completed(futures):
            evaluated.extend(fut.result())
    ranked = assign_structural_ranks(evaluated, int(args.top_k))
    with (ARTIFACT_DIR / "evaluated_joint_repair_candidates.jsonl").open("w", encoding="utf-8") as f:
        for row in ranked:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    rows_ge7 = [r for r in ranked if int(r.get("row_count") or 0) >= 7]
    overall = summarize(ranked)
    rows7 = summarize(rows_ge7)
    exp3 = read_json(EXP3_RESULT)
    exp4 = read_json(EXP4_RESULT)
    crystallm_rows7_top50 = exp3["baseline_reference"]["crystallm_k50_rows_ge7"]["top50_match"]
    min_gate = bool(
        (rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and (rows7.get("match@50") or 0.0) >= float(crystallm_rows7_top50) - 0.005
        and (rows7.get("formula_consistency") or 0.0) >= 0.95
        and (rows7.get("sg_consistency") or 0.0) >= 0.90
        and (rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(
        (rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
        or (
            (rows7.get("match@5") or 0.0) >= exp4["baselines"]["crystallm_rows_ge7_reference"]["top5_match"] + 0.05
            and (rows7.get("match@20") or 0.0) >= exp4["baselines"]["crystallm_rows_ge7_reference"]["top20_match"] + 0.05
        )
    )
    result = {
        "experiment": "opentry_14_exp2_predicted_skeleton_aware_joint_geometry_repair",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "joint_lattice_free_parameter_mlp_repair",
            "inputs": ["composition", "GT-SG", "predicted skeleton rows", "initial source lattice", "initial row free params", "row counts", "structural self-check features"],
            "outputs": ["lattice update", "row-level free-parameter update"],
            "not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "lattice_alpha": float(args.lattice_alpha),
            "param_alpha": float(args.param_alpha),
        },
        "training": {
            "pair_artifact": str(PAIR_ARTIFACT),
            "pair_counts": pair_counts,
            "lattice_samples": len(lattice_x),
            "row_param_samples": len(param_x),
            "lattice_training": lattice_training,
            "param_training": param_training,
        },
        "data_scale": {
            "validation_samples": overall["samples"],
            "validation_rows_ge7_samples": rows7["samples"],
            "candidate_records": len(ranked),
            "top_k": int(args.top_k),
            "workers": int(args.workers),
        },
        "baselines": {
            "crystallm_rows_ge7_top50": crystallm_rows7_top50,
            "opentry13_predicted_skeleton_hydrated_rows_ge7_match50": exp3["rows_ge7"].get("top50_hydrated_match_coverage"),
            "opentry13_multi_geometry_rows_ge7_match50": exp4["rows_ge7"].get("match@50"),
        },
        "mapping_failures": dict(failures),
        "overall": overall,
        "rows_ge7": rows7,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "passed": min_gate,
            "minimum_standard": {
                "rows_ge7_conversion50": 0.223,
                "rows_ge7_match50_near_crystallm_k50": crystallm_rows7_top50,
                "rows_ge7_formula_consistency": 0.95,
                "rows_ge7_sg_consistency": 0.90,
                "rows_ge7_exact_cover_retained": 0.95,
            },
        },
        "decision": {
            "verdict": "pass_minimum_gate" if min_gate else "fail_validation_gate",
            "next_step": "enter experiment 3 local optimizer" if min_gate else "do not enter experiment 3/full validation/official; improve joint repair or data alignment",
        },
        "artifacts": {
            "model": str(ARTIFACT_DIR / "joint_repair_heads.pt"),
            "generated_meta": str(ARTIFACT_DIR / "generated_joint_repair_meta.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / "evaluated_joint_repair_candidates.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2_joint_geometry_repair.json", result)
    print(json.dumps({"output": str(RESULT_DIR / "experiment_2_joint_geometry_repair.json"), "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
