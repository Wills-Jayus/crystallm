#!/usr/bin/env python3
"""Evaluate Model B lattice denoiser as a fixed-order CIF recovery smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYMCIF_ROOT = ROOT.parents[0] / "symcif_experiment"
for path in (ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("WORKDIR", str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache/transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache/xdg"))
os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
os.environ.setdefault("WANDB_DIR", str(ROOT / "logs/wandb"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "cache/cuda"))

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from symcif_v4.orbit_engine import OrbitEngine, cell_volume
from train_model_b_denoiser_smoke import CORRUPTION_TYPES, LatticeDenoiser, lattice_vector


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def formula_counts_from_formula(formula: str | None) -> dict[str, int]:
    if not formula:
        return {}
    comp = Composition(formula)
    return {str(el): int(round(amount)) for el, amount in comp.get_el_amt_dict().items()}


def all_canonical_dev_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name in ("dev_model", "dev_gate"):
        path = ROOT / "data/canonical_dev" / f"{name}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                rows[row["sample_id"]] = row
    return rows


def vpa_from_lattice(lattice: dict[str, Any] | None, atom_count: float | None) -> float:
    if not lattice or not atom_count or atom_count <= 0:
        return 0.0
    try:
        volume = cell_volume(lattice)
    except Exception:
        volume = lattice.get("volume", 0.0)
    return math.log(max(float(volume) / max(float(atom_count), 1.0), 1e-6))


def model_input(example: dict[str, Any], target: dict[str, Any], corrupt_lattice: dict[str, Any]) -> torch.Tensor | None:
    corrupt = lattice_vector(corrupt_lattice)
    if corrupt is None:
        return None
    formula_counts = target.get("formula_counts") or formula_counts_from_formula(target.get("formula"))
    atom_count = float(sum(float(v) for v in formula_counts.values()) or target.get("atom_count") or 1.0)
    vpa_c = vpa_from_lattice(corrupt_lattice, atom_count)
    type_vec = [1.0 if example.get("corruption_type") == t else 0.0 for t in CORRUPTION_TYPES]
    features = (
        corrupt
        + [
            float(target.get("sg") or example.get("sg") or 0) / 230.0,
            float(target.get("row_count") or example.get("row_count") or 0) / 60.0,
            1.0 if (target.get("rows_ge_7") or example.get("rows_ge_7")) else 0.0,
            float(example.get("corruption_strength") or 0.0),
            vpa_c,
        ]
        + type_vec
    )
    return torch.tensor(features, dtype=torch.float32)


def vector_to_lattice(vec: np.ndarray, sg: int) -> dict[str, float]:
    a = float(math.exp(float(vec[0])))
    b = float(math.exp(float(vec[1])))
    c = float(math.exp(float(vec[2])))
    alpha = float(vec[3] * 180.0)
    beta = float(vec[4] * 180.0)
    gamma = float(vec[5] * 180.0)
    if 195 <= sg <= 230:
        b = c = a
        alpha = beta = gamma = 90.0
    elif 75 <= sg <= 142:
        b = a
        alpha = beta = gamma = 90.0
    elif 168 <= sg <= 194 or 143 <= sg <= 167:
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif 16 <= sg <= 74:
        alpha = beta = gamma = 90.0
    elif 3 <= sg <= 15:
        alpha = gamma = 90.0
    return {
        "a": max(a, 0.5),
        "b": max(b, 0.5),
        "c": max(c, 0.5),
        "alpha": min(max(alpha, 30.0), 150.0),
        "beta": min(max(beta, 30.0), 150.0),
        "gamma": min(max(gamma, 30.0), 150.0),
    }


def perturb_vector(vec: np.ndarray, generation_index: int, seed: int) -> np.ndarray:
    if generation_index == 0:
        return vec.copy()
    rng = np.random.default_rng(seed + generation_index * 1009)
    out = vec.copy()
    out[:3] += rng.normal(0.0, 0.0025, size=3)
    out[3:6] += rng.normal(0.0, 0.0008, size=3)
    return out


def render_target_rows(engine: OrbitEngine, target: dict[str, Any], lattice: dict[str, float], generation_index: int) -> tuple[bool, str, str | None]:
    rows = target.get("wa_table") or []
    params = {idx: dict(row.get("free_params") or {}) for idx, row in enumerate(rows)}
    formula_counts = {str(k): int(round(float(v))) for k, v in (target.get("formula_counts") or {}).items()}
    try:
        cif = engine.render_cif_from_wa_table(
            rows,
            lattice=lattice,
            free_params_by_row=params,
            formula_counts=formula_counts,
            sg=int(target["sg"]),
            sg_symbol=str(target.get("sg_symbol") or ""),
            data_name=f"{target['sample_id']}_model_b_cif_smoke_g{generation_index}",
        )
        return True, cif, None
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"


def evaluate_cif(cif: str, target: dict[str, Any], matcher: StructureMatcher) -> dict[str, Any]:
    out: dict[str, Any] = {
        "readable": False,
        "composition_exact": False,
        "sg_ok": False,
        "atom_count_ok": False,
        "match": False,
        "rmsd": None,
        "parse_error": None,
    }
    try:
        pred = Structure.from_str(cif, fmt="cif")
        ref_path = ROOT / target["canonical_cif_path"]
        ref = Structure.from_file(str(ref_path))
    except Exception as exc:
        out["parse_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["readable"] = True
    target_counts = target.get("formula_counts") or {}
    out["composition_exact"] = Composition({k: float(v) for k, v in target_counts.items()}).fractional_composition.almost_equals(pred.composition.fractional_composition)
    out["atom_count_ok"] = len(pred) == int(sum(float(v) for v in target_counts.values()))
    try:
        detected = SpacegroupAnalyzer(pred, symprec=0.1).get_space_group_number()
        out["detected_sg"] = int(detected)
        out["sg_ok"] = int(detected) == int(target["sg"])
    except Exception as exc:
        out["sg_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["match"] = bool(matcher.fit(pred, ref))
        if out["match"]:
            rms = matcher.get_rms_dist(pred, ref)
            if rms is not None:
                out["rmsd"] = float(rms[0] if isinstance(rms, (tuple, list)) else rms)
    except Exception as exc:
        out["match_error"] = f"{type(exc).__name__}: {exc}"
    return out


def summarize(rows: list[dict[str, Any]], sample_ids: list[str], ks: list[int]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = {sid: [] for sid in sample_ids}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for vals in by_sample.values():
        vals.sort(key=lambda r: int(r["generation_index"]))
    out: dict[str, Any] = {}
    for k in ks:
        subset_name = f"top{k}"
        match = readable = comp = sg_ok = atom = 0
        rmsds: list[float] = []
        rows7_match = rows7_total = 0
        for sid in sample_ids:
            vals = [r for r in by_sample.get(sid, []) if int(r["generation_index"]) < k]
            if not vals:
                continue
            is_rows7 = bool(vals[0].get("rows_ge_7"))
            rows7_total += int(is_rows7)
            readable += int(any(v.get("readable") for v in vals))
            comp += int(any(v.get("composition_exact") for v in vals))
            sg_ok += int(any(v.get("sg_ok") for v in vals))
            atom += int(any(v.get("atom_count_ok") for v in vals))
            matched = [v for v in vals if v.get("match")]
            if matched:
                match += 1
                if is_rows7:
                    rows7_match += 1
                sample_rms = [float(v["rmsd"]) for v in matched if v.get("rmsd") is not None]
                if sample_rms:
                    rmsds.append(min(sample_rms))
        n = max(1, len(sample_ids))
        out[subset_name] = {
            "samples": len(sample_ids),
            f"match@{k}": match / n,
            f"readable@{k}": readable / n,
            f"composition_exact@{k}": comp / n,
            f"sg_ok@{k}": sg_ok / n,
            f"atom_count_ok@{k}": atom / n,
            f"RMSE@{k}": (sum(rmsds) / len(rmsds)) if rmsds else None,
            f"rows_ge_7_match@{k}": rows7_match / max(1, rows7_total),
            "rows_ge_7_samples": rows7_total,
        }
    return out


def append_experiment_log(report: dict[str, Any]) -> None:
    path = ROOT / "reports/opentry_5_experiment_log.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# opentry_5 Experiment Log\n"
    fold = report["fold"]
    exp_id = report["experiment_id"]
    block = f"""

## {exp_id}: Model B fixed-order CIF recovery smoke ({fold})
- Time: {report['created_at']}
- Core hypothesis: the trained Model B lattice denoiser can be connected to OrbitEngine to emit parseable CIFs from synthetic geometry corruption without any candidate ranking.
- Difference vs historical failures: same correction is applied to every input; candidates are fixed generation_index/seed perturbations, not score-sorted or selected.
- Model side or data side: model side.
- Contains sorting/filtering: no.
- candidate order: generation_index 0..{report['k'] - 1}; invalid slots are retained.
- Read files: checkpoints/model_b_denoiser_smoke/best.pt, data/geometry_denoising_dev.jsonl, data/canonical_dev/dev_model.jsonl, data/canonical_dev/dev_gate.jsonl.
- Written files: {report['generation_file']}, {report['metrics_file']}.
- Data split: {fold}.
- Read test: no.
- Read val512: no.
- val512 cumulative use: 0.
- Model: Model B lattice denoiser plus GT-W/A geometry recovery diagnostic.
- Parameters: {report['parameter_count']}
- GPU/CPU: {report['device']}.
- Training time: already reported in E8003.
- Inference time: {report['seconds']:.2f}s.
- readable: top1 {report['summary']['top1']['readable@1']:.4f}.
- composition exact: top1 {report['summary']['top1']['composition_exact@1']:.4f}.
- SG/Wyckoff legal: top1 SG {report['summary']['top1']['sg_ok@1']:.4f}; W/A rows are GT diagnostic rows.
- match@1: {report['summary']['top1']['match@1']:.4f}.
- match@5: {report['summary'].get('top5', {}).get('match@5')}.
- RMSE@1/5: {report['summary']['top1'].get('RMSE@1')} / {report['summary'].get('top5', {}).get('RMSE@5')}.
- rows>=7 match: top1 {report['summary']['top1']['rows_ge_7_match@1']:.4f}.
- grouped dev folds consistent: pending paired run if only one fold; diagnostic only.
- Conclusion: CIF smoke completed; not terminal because it uses GT W/A/free params for diagnostic geometry recovery and only evaluates {report['samples']} samples.
- Gate: smoke {'pass' if report['gate_pass'] else 'fail'}.
- Terminate family: no.
- Next: replace GT W/A/free params with OOF/predicted W/A and full free-param generator.
"""
    write_text(path, existing.rstrip() + block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_a", "fold_b", "all"], default="fold_a")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8017)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ckpt_path = ROOT / "checkpoints/model_b_denoiser_smoke/best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = int(ckpt["model"]["net.0.weight"].shape[1])
    model = LatticeDenoiser(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    parameter_count = sum(p.numel() for p in model.parameters())

    target_by_id = all_canonical_dev_rows()
    examples = []
    for row in read_jsonl(ROOT / "data/geometry_denoising_dev.jsonl"):
        target = target_by_id.get(row["sample_id"])
        if not target or not target.get("canonical_cif_path"):
            continue
        if args.fold != "all" and target.get("grouped_dev_fold") != args.fold:
            continue
        if row.get("source_type") != "synthetic_train_dev_clean_corruption":
            continue
        examples.append(row)
        if args.limit and len(examples) >= args.limit:
            break

    sg_symbols = {int(row["sg"]): str(row.get("sg_symbol") or f"SG{int(row['sg'])}") for row in target_by_id.values() if row.get("sg")}
    engine = OrbitEngine(SYMCIF_ROOT / "artifacts/wyckoff_lookup_full.json", sg_symbols)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True, attempt_supercell=False)
    started = time.time()
    out_rows: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for example in examples:
            target = target_by_id[example["sample_id"]]
            sample_ids.append(example["sample_id"])
            corrupt_lattice = (example.get("corrupted_input") or {}).get("lattice") or target.get("lattice")
            x = model_input(example, target, corrupt_lattice)
            if x is None:
                for generation_index in range(args.k):
                    out_rows.append(
                        {
                            "sample_id": example["sample_id"],
                            "generation_index": generation_index,
                            "rows_ge_7": bool(target.get("rows_ge_7")),
                            "render_ok": False,
                            "error": "missing_model_input",
                            "readable": False,
                            "composition_exact": False,
                            "sg_ok": False,
                            "atom_count_ok": False,
                            "match": False,
                            "rmsd": None,
                        }
                    )
                continue
            mu, _logvar = model(x[None].to(device))
            base_vec = mu[0].detach().cpu().numpy()
            for generation_index in range(args.k):
                pred_vec = perturb_vector(base_vec, generation_index, args.seed)
                lattice = vector_to_lattice(pred_vec, int(target["sg"]))
                render_ok, cif, render_error = render_target_rows(engine, target, lattice, generation_index)
                row = {
                    "sample_id": example["sample_id"],
                    "generation_index": generation_index,
                    "seed": args.seed + generation_index * 1009,
                    "fold": target.get("grouped_dev_fold"),
                    "rows_ge_7": bool(target.get("rows_ge_7")),
                    "row_count": target.get("row_count"),
                    "sg": target.get("sg"),
                    "formula": target.get("formula"),
                    "render_ok": render_ok,
                    "render_error": render_error,
                    "predicted_lattice": lattice,
                    "candidate_order": "generation_index_fixed_seed_no_sorting",
                    "uses_gt_wa_rows": True,
                    "uses_gt_free_params": True,
                    "test_access": "none",
                }
                if render_ok:
                    row.update(evaluate_cif(cif, target, matcher))
                    row["cif"] = cif
                else:
                    row.update({"readable": False, "composition_exact": False, "sg_ok": False, "atom_count_ok": False, "match": False, "rmsd": None})
                    row["cif"] = ""
                out_rows.append(row)

    summary = summarize(out_rows, sample_ids, ks=[1, min(5, args.k)])
    gate_pass = bool(summary["top1"]["readable@1"] >= 0.95 and summary["top1"]["composition_exact@1"] >= 0.95 and summary["top1"]["atom_count_ok@1"] >= 0.95)
    out_dir = ROOT / "eval/model_b_cif_recovery_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_label = args.fold
    generation_file = out_dir / f"{fold_label}_generations.jsonl"
    metrics_file = out_dir / f"{fold_label}_metrics.json"
    write_jsonl(generation_file, out_rows)
    exp_id = "E8007" if args.fold == "fold_a" else ("E8008" if args.fold == "fold_b" else "E8009")
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_id": exp_id,
        "fold": fold_label,
        "samples": len(sample_ids),
        "k": args.k,
        "device": str(device),
        "seed": args.seed,
        "parameter_count": parameter_count,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "generation_file": str(generation_file.relative_to(ROOT)),
        "metrics_file": str(metrics_file.relative_to(ROOT)),
        "summary": summary,
        "gate_pass": gate_pass,
        "seconds": time.time() - started,
        "no_ranking": True,
        "candidate_order": "generation_index_fixed_seed_no_sorting",
        "diagnostic_limitations": [
            "uses GT W/A rows and GT free params from grouped dev canonical labels",
            "does not prove formula+SG to W/A generation",
            "not a terminal opentry_5 success metric",
        ],
        "test_access": "none",
        "val512_access": "none",
    }
    write_json(metrics_file, report)
    append_experiment_log(report)
    model_b_report = ROOT / "reports/model_b_denoiser_report.md"
    existing = model_b_report.read_text(encoding="utf-8") if model_b_report.exists() else "# Model B Denoiser Report\n"
    block = f"""

## {exp_id} CIF Recovery Smoke ({fold_label})

Fixed-order CIF recovery smoke completed at {report['created_at']}. This diagnostic uses grouped-dev GT W/A rows and free params to test whether the trained lattice denoiser can connect to OrbitEngine and StructureMatcher. It is not a terminal generation result.

```json
{json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    write_text(model_b_report, existing.rstrip() + block)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
