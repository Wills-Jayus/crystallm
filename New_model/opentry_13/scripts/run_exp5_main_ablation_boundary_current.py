#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

EXP1 = RESULT_DIR / "experiment_1_true_anchor_source_c2s3c15.json"
EXP2_RENDERER = RESULT_DIR / "experiment_2_predicted_skeleton_renderer_site_mapping.json"
EXP3_REPAIR_AUDIT = RESULT_DIR / "experiment_3_predicted_skeleton_aware_geometry_repair_audit.json"
EXP3_LATTICE_PILOT = RESULT_DIR / "experiment_3_predicted_skeleton_lattice_repair_pilot.json"
EXP3_SKELETON = RESULT_DIR / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
EXP4_MULTIGEOM = RESULT_DIR / "experiment_4_rows_ge7_multi_geometry_proposal.json"
OLD_PRED_REPAIR = RESULT_DIR / "experiment_4_predicted_skeleton_geometry_repair.json"
OP12_BOUNDARY = NEW_MODEL / "opentry_12" / "results" / "experiment_7_main_ablation_boundary.json"
GTWA_MPTS52_MET = (
    NEW_MODEL
    / "symcif_experiment"
    / "reports"
    / "symcif_v4_geometry_model_gtwa"
    / "no_oversampling"
    / "metrics"
    / "baseline_per_generation_metrics.jsonl"
)
MPTS52_TEST_TARGETS = NEW_MODEL / "opentry_7" / "cache" / "mpts_52_test_targets.jsonl"

BUDGETS = (1, 5, 20)
ALL_BUDGETS = (1, 5, 20, 50)
MARKER = "<!-- OPENTRY13_EXP5_MAIN_ABLATION_BOUNDARY -->"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_report_at_end(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    while marker in text:
        start = text.index(marker)
        next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
        if next_marker == -1:
            text = text[:start].rstrip()
        else:
            text = text[:start].rstrip() + text[next_marker:]
    text = text.rstrip() + "\n\n" + marker + "\n" + body.rstrip() + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def triplet(d: dict[str, Any] | None, prefix: str = "match") -> str:
    if not d:
        return "NA / NA / NA"
    return " / ".join(pct(d.get(f"{prefix}@{k}")) for k in BUDGETS)


def rmse_triplet(d: dict[str, Any] | None, prefix: str = "RMSE") -> str:
    if not d:
        return "NA / NA / NA"
    return " / ".join("NA" if d.get(f"{prefix}@{k}") is None else f"{float(d[f'{prefix}@{k}']):.6f}" for k in BUDGETS)


def match_dict(metrics: dict[str, Any], *, prefix: str = "match") -> dict[str, float | None]:
    return {f"match@{k}": None if metrics.get(f"{prefix}@{k}") is None else float(metrics.get(f"{prefix}@{k}")) for k in BUDGETS}


def rmse_dict(metrics: dict[str, Any], *, prefix: str = "RMSE") -> dict[str, float | None]:
    return {f"RMSE@{k}": None if metrics.get(f"{prefix}@{k}") is None else float(metrics.get(f"{prefix}@{k}")) for k in BUDGETS}


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        key = f"match@{k}"
        if a.get(key) is None or b.get(key) is None:
            out[key] = None
        else:
            out[key] = float(a[key]) - float(b[key])
    return out


def row(
    *,
    name: str,
    role: str,
    category: str,
    scope: str,
    source: Path | str,
    overall_match: dict[str, Any] | None,
    rows_ge7_match: dict[str, Any] | None,
    overall_rmse: dict[str, Any] | None = None,
    rows_ge7_rmse: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    conversion: dict[str, Any] | None = None,
    baseline_name: str | None = None,
    delta_vs_baseline: dict[str, Any] | None = None,
    rows_ge7_delta_vs_baseline: dict[str, Any] | None = None,
    gate: str = "not_evaluated",
    boundary: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "baseline_name": baseline_name,
        "boundary": boundary,
        "category": category,
        "conversion": conversion or {},
        "delta_vs_baseline": delta_vs_baseline,
        "gate": gate,
        "name": name,
        "notes": notes or [],
        "overall_match": overall_match or {f"match@{k}": None for k in BUDGETS},
        "overall_rmse": overall_rmse or {f"RMSE@{k}": None for k in BUDGETS},
        "quality": quality or {},
        "role": role,
        "rows_ge7_delta_vs_baseline": rows_ge7_delta_vs_baseline,
        "rows_ge7_match": rows_ge7_match or {f"match@{k}": None for k in BUDGETS},
        "rows_ge7_rmse": rows_ge7_rmse or {f"RMSE@{k}": None for k in BUDGETS},
        "scope": scope,
        "source": str(source),
    }


def load_gtwa_mpts52_oracle() -> dict[str, Any]:
    rows_ge7: dict[str, bool] = {}
    with MPTS52_TEST_TARGETS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                rows_ge7[str(record["sample_id"])] = int(record.get("row_count") or 0) >= 7
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with GTWA_MPTS52_MET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                sid = str(record["sample_id"])
                if sid.startswith("mpts_52_test_orig__"):
                    groups[sid].append(record)
    samples = sorted(groups)
    rows7 = [sid for sid in samples if rows_ge7.get(sid, False)]
    overall_match: dict[str, float | None] = {}
    rows_ge7_match: dict[str, float | None] = {}
    overall_rmse: dict[str, float | None] = {}
    rows_ge7_rmse: dict[str, float | None] = {}
    quality: dict[str, Any] = {
        "samples": len(samples),
        "rows_ge7_samples": len(rows7),
        "condition": "GT-WA/GT-skeleton oracle subset; diagnostic only",
    }
    for k in BUDGETS:
        all_hits: list[bool] = []
        rows7_hits: list[bool] = []
        all_rms: list[float] = []
        rows7_rms: list[float] = []
        valid_any: list[bool] = []
        formula_any: list[bool] = []
        sg_any: list[bool] = []
        exact_any: list[bool] = []
        for sid in samples:
            arr = sorted(groups[sid], key=lambda r: int(r.get("rank") or int(r.get("gen_index") or 0) + 1))[:k]
            hit = any(bool(r.get("match_ok")) for r in arr)
            all_hits.append(hit)
            rms_vals = [float(r["rms_dist"]) for r in arr if bool(r.get("match_ok")) and r.get("rms_dist") is not None]
            if rms_vals:
                all_rms.append(min(rms_vals))
            if rows_ge7.get(sid, False):
                rows7_hits.append(hit)
                if rms_vals:
                    rows7_rms.append(min(rms_vals))
            valid_any.append(any(bool(r.get("valid")) for r in arr))
            formula_any.append(any(bool(r.get("formula_ok")) for r in arr))
            sg_any.append(any(bool(r.get("space_group_ok")) for r in arr))
            exact_any.append(any(bool(r.get("multiplicity_ok")) for r in arr))
        overall_match[f"match@{k}"] = sum(all_hits) / max(1, len(all_hits))
        rows_ge7_match[f"match@{k}"] = sum(rows7_hits) / max(1, len(rows7_hits))
        overall_rmse[f"RMSE@{k}"] = sum(all_rms) / len(all_rms) if all_rms else None
        rows_ge7_rmse[f"RMSE@{k}"] = sum(rows7_rms) / len(rows7_rms) if rows7_rms else None
        quality[f"valid_any@{k}"] = sum(valid_any) / max(1, len(valid_any))
        quality[f"formula_ok_any@{k}"] = sum(formula_any) / max(1, len(formula_any))
        quality[f"sg_ok_any@{k}"] = sum(sg_any) / max(1, len(sg_any))
        quality[f"exact_cover_any@{k}"] = sum(exact_any) / max(1, len(exact_any))
    return {
        "overall_match": overall_match,
        "rows_ge7_match": rows_ge7_match,
        "overall_rmse": overall_rmse,
        "rows_ge7_rmse": rows_ge7_rmse,
        "quality": quality,
    }


def find_boundary_row(boundary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in boundary.get("validation_ablation_rows", []):
        if item.get("name") == name:
            return item
    return None


def normalize_boundary_match(block: dict[str, Any] | None, rows_key: bool = False) -> dict[str, float | None]:
    if not block:
        return {f"match@{k}": None for k in BUDGETS}
    source = block.get("rows_ge7_match" if rows_key else "overall_match") or {}
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        value = source.get(f"match@{k}")
        if value is None:
            value = source.get(f"rows>=7_match@{k}") if rows_key else None
        out[f"match@{k}"] = None if value is None else float(value)
    return out


def normalize_boundary_rmse(block: dict[str, Any] | None, rows_key: bool = False) -> dict[str, float | None]:
    if not block:
        return {f"RMSE@{k}": None for k in BUDGETS}
    source = block.get("rows_ge7_rmse" if rows_key else "overall_rmse") or {}
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        value = source.get(f"RMSE@{k}")
        if value is None:
            value = source.get(f"rows>=7_RMSE@{k}") if rows_key else None
        if value is None:
            value = source.get(f"rmsd@{k}") if not rows_key else source.get(f"rows>=7_rmsd@{k}")
        out[f"RMSE@{k}"] = None if value is None else float(value)
    return out


def quality_from_match_metrics(metrics: dict[str, Any], key_style: str = "direct") -> dict[str, Any]:
    if key_style == "candidate_quality":
        return {
            "valid_any@1": metrics.get("valid_any@1"),
            "valid_any@5": metrics.get("valid_any@5"),
            "valid_any@20": metrics.get("valid_any@20"),
            "formula_ok_any@1": metrics.get("formula_ok_any@1"),
            "formula_ok_any@5": metrics.get("formula_ok_any@5"),
            "formula_ok_any@20": metrics.get("formula_ok_any@20"),
            "sg_ok_any@1": metrics.get("space_group_ok_any@1"),
            "sg_ok_any@5": metrics.get("space_group_ok_any@5"),
            "sg_ok_any@20": metrics.get("space_group_ok_any@20"),
            "exact_cover_any@20": metrics.get("known_exact_cover_any@20"),
            "skeleton_to_match_conversion": metrics.get("symcif_skeleton_hit_to_match_conversion"),
        }
    return {
        "valid_rate": metrics.get("valid_rate"),
        "formula_consistency": metrics.get("formula_consistency"),
        "sg_consistency": metrics.get("sg_consistency"),
        "exact_cover_retained": metrics.get("exact_cover_retained"),
        "collision_rate": metrics.get("collision_rate"),
    }


def main() -> int:
    exp1 = read_json(EXP1)
    exp2 = read_json(EXP2_RENDERER)
    exp3_audit = read_json(EXP3_REPAIR_AUDIT)
    exp3_lattice = read_json(EXP3_LATTICE_PILOT)
    exp3_skel = read_json(EXP3_SKELETON)
    exp4 = read_json(EXP4_MULTIGEOM)
    old_repair = read_json(OLD_PRED_REPAIR)
    boundary = read_json(OP12_BOUNDARY)

    anchor = exp1["systems"]["true_crystallm_a_gt_sg_C20_anchor"]
    c2s = exp1["systems"]["true_anchor_source_C2S3C15"]
    old_c2s = exp1["systems"]["old_pure_source_C2S3C15"]
    anchor_match = match_dict(anchor["all"])
    anchor_rows = match_dict(anchor["rows_ge7"])

    rows: list[dict[str, Any]] = []
    rows.append(
        row(
            name="true CrystaLLM-a GT-SG anchor",
            role="official_true_anchor",
            category="reference",
            scope="MPTS-52 official test K20",
            source=anchor.get("candidate_source"),
            overall_match=anchor_match,
            rows_ge7_match=anchor_rows,
            overall_rmse=rmse_dict(anchor["all"]),
            rows_ge7_rmse=rmse_dict(anchor["rows_ge7"]),
            gate="reference",
            boundary="Only true official anchor for main success comparison.",
        )
    )
    c2s_match = match_dict(c2s["all"])
    c2s_rows = match_dict(c2s["rows_ge7"])
    rows.append(
        row(
            name="C2S3C15 true-anchor-source hybrid",
            role="auxiliary_hybrid_true_anchor_source",
            category="auxiliary",
            scope="MPTS-52 official test K20",
            source=EXP1,
            overall_match=c2s_match,
            rows_ge7_match=c2s_rows,
            overall_rmse=rmse_dict(c2s["all"]),
            rows_ge7_rmse=rmse_dict(c2s["rows_ge7"]),
            baseline_name="true CrystaLLM-a GT-SG anchor",
            delta_vs_baseline=delta(c2s_match, anchor_match),
            rows_ge7_delta_vs_baseline=delta(c2s_rows, anchor_rows),
            quality=quality_from_match_metrics(c2s.get("candidate_quality") or {}, "candidate_quality"),
            gate=exp1["acceptance_gate"]["verdict"],
            boundary="Auxiliary only; no scorer/tuning, but match@5 and match@20 are not both >= +5pp vs true anchor.",
        )
    )
    rows.append(
        row(
            name="old pure-source C2S3C15",
            role="historical_low_anchor_diagnostic",
            category="diagnostic",
            scope="MPTS-52 official test K20",
            source=EXP1,
            overall_match=match_dict(old_c2s["all"]),
            rows_ge7_match=match_dict(old_c2s["rows_ge7"]),
            overall_rmse=rmse_dict(old_c2s["all"]),
            rows_ge7_rmse=rmse_dict(old_c2s["rows_ge7"]),
            baseline_name="true CrystaLLM-a GT-SG anchor",
            delta_vs_baseline=delta(match_dict(old_c2s["all"]), anchor_match),
            rows_ge7_delta_vs_baseline=delta(match_dict(old_c2s["rows_ge7"]), anchor_rows),
            gate="diagnostic_only",
            boundary="Old pure-source baseline is not the main anchor.",
        )
    )

    symcif_row = find_boundary_row(boundary, "SymCIF v5 neural skeleton/geometry proposer")
    symcif_rows_ref = exp3_skel["baseline_reference"].get("symcif_v5_rows_ge7") or {}
    rows.append(
        row(
            name="SymCIF v5 proposer",
            role="symcif_v5_generation_diagnostic",
            category="main_method_candidate",
            scope="MPTS-52 validation",
            source=OP12_BOUNDARY,
            overall_match=normalize_boundary_match(symcif_row, rows_key=False),
            rows_ge7_match={
                "match@1": symcif_rows_ref.get("top1_match_coverage"),
                "match@5": symcif_rows_ref.get("top5_match_coverage"),
                "match@20": symcif_rows_ref.get("top20_match_coverage"),
            },
            overall_rmse=normalize_boundary_rmse(symcif_row, rows_key=False),
            rows_ge7_rmse={
                "RMSE@1": symcif_rows_ref.get("top1_RMSE"),
                "RMSE@5": symcif_rows_ref.get("top5_RMSE"),
                "RMSE@20": symcif_rows_ref.get("top20_RMSE"),
            },
            quality={
                "rows_ge7_valid_any@20": symcif_rows_ref.get("top20_valid_any"),
                "rows_ge7_formula_ok_any@20": symcif_rows_ref.get("top20_formula_ok_any"),
                "rows_ge7_sg_ok_any@20": symcif_rows_ref.get("top20_sg_ok_any"),
                "rows_ge7_exact_cover_any@20": symcif_rows_ref.get("top20_exact_cover_feasible_any"),
                "rows_ge7_skeleton_hit@20": symcif_rows_ref.get("top20_skeleton_hit_any"),
                "rows_ge7_skeleton_to_match_conversion@20": symcif_rows_ref.get("top20_sample_skeleton_to_match_conversion"),
            },
            gate="fail_main_gate",
            boundary="Validation/generation diagnostic; rows>=7 hydrated match and conversion remain low.",
        )
    )

    skel_overall = {f"match@{k}": exp3_skel["overall"].get(f"top{k}_hydrated_match_coverage") for k in BUDGETS}
    skel_rows = {f"match@{k}": exp3_skel["rows_ge7"].get(f"top{k}_hydrated_match_coverage") for k in BUDGETS}
    rows.append(
        row(
            name="rows>=7 predicted skeleton proposer",
            role="predicted_skeleton_proposer_validation",
            category="main_method_candidate",
            scope="MPTS-52 validation; hydrated existing SymCIF candidates",
            source=EXP3_SKELETON,
            overall_match=skel_overall,
            rows_ge7_match=skel_rows,
            overall_rmse={f"RMSE@{k}": exp3_skel["overall"].get(f"top{k}_hydrated_RMSE") for k in BUDGETS},
            rows_ge7_rmse={f"RMSE@{k}": exp3_skel["rows_ge7"].get(f"top{k}_hydrated_RMSE") for k in BUDGETS},
            quality={
                "overall_valid_any@20": exp3_skel["overall"].get("top20_hydrated_valid_any"),
                "overall_formula_ok_any@20": exp3_skel["overall"].get("top20_hydrated_formula_ok_any"),
                "overall_sg_ok_any@20": exp3_skel["overall"].get("top20_hydrated_sg_ok_any"),
                "overall_exact_cover_any@20": exp3_skel["overall"].get("top20_hydrated_exact_cover_any"),
                "rows_ge7_valid_any@20": exp3_skel["rows_ge7"].get("top20_hydrated_valid_any"),
                "rows_ge7_skeleton_hit@20": exp3_skel["rows_ge7"].get("top20_skeleton_hit_coverage"),
            },
            conversion={
                "overall_skeleton_to_match@20": exp3_skel["overall"].get("top20_proposal_skeleton_to_hydrated_match_conversion"),
                "rows_ge7_skeleton_to_match@20": exp3_skel["rows_ge7"].get("top20_proposal_skeleton_to_hydrated_match_conversion"),
            },
            gate="fail_validation_gate",
            boundary="Skeleton-hit exists, but hydrated match/conversion fails rows>=7 gate.",
        )
    )

    selected = exp2["modes"]["train_prototype"]["selected_by_safe_checks"]
    rows.append(
        row(
            name="renderer/site-mapping fixed selector",
            role="renderer_site_mapping_structure_gate",
            category="main_method_component",
            scope="MPTS-52 validation; structure gate only, no StructureMatcher by design",
            source=EXP2_RENDERER,
            overall_match=None,
            rows_ge7_match=None,
            quality={
                "overall_valid": selected["overall"].get("valid_rate"),
                "overall_formula": selected["overall"].get("formula_consistency"),
                "overall_sg": selected["overall"].get("sg_consistency"),
                "overall_exact_cover": selected["overall"].get("exact_cover_retained"),
                "rows_ge7_valid": selected["rows_ge7"].get("valid_rate"),
                "rows_ge7_formula": selected["rows_ge7"].get("formula_consistency"),
                "rows_ge7_sg": selected["rows_ge7"].get("sg_consistency"),
                "rows_ge7_exact_cover": selected["rows_ge7"].get("exact_cover_retained"),
                "fallback_rate": selected["selection"].get("fallback_rate"),
            },
            gate="pass_structure_gate" if exp2["gate"].get("selected_train_prototype_passed") else "fail_structure_gate",
            boundary="Component gate only; match@k intentionally not measured in exp2.",
            notes=["Ranking/selection used only inference-safe structural checks, not StructureMatcher."],
        )
    )

    repair_all = exp3_lattice["overall"]
    repair_rows = exp3_lattice["rows_ge7"]
    rows.append(
        row(
            name="predicted-skeleton-aware lattice repair pilot",
            role="predicted_skeleton_aware_lattice_repair_pilot",
            category="main_method_candidate",
            scope="MPTS-52 validation; train-noisy-skeleton lattice MLP repair",
            source=EXP3_LATTICE_PILOT,
            overall_match={f"match@{k}": repair_all.get(f"after_match@{k}") for k in BUDGETS},
            rows_ge7_match={f"match@{k}": repair_rows.get(f"after_match@{k}") for k in BUDGETS},
            overall_rmse=rmse_dict(repair_all),
            rows_ge7_rmse=rmse_dict(repair_rows),
            quality={
                "overall_valid": repair_all.get("valid_rate"),
                "overall_formula": repair_all.get("formula_consistency"),
                "overall_sg": repair_all.get("sg_consistency"),
                "overall_exact_cover": repair_all.get("exact_cover_retained"),
                "rows_ge7_valid": repair_rows.get("valid_rate"),
                "rows_ge7_formula": repair_rows.get("formula_consistency"),
                "rows_ge7_sg": repair_rows.get("sg_consistency"),
                "rows_ge7_exact_cover": repair_rows.get("exact_cover_retained"),
                "train_noisy_skeleton_pairs": exp3_lattice["data_scale"].get("train_pairs"),
                "best_val_loss": exp3_lattice["training"].get("best_val_loss"),
            },
            conversion={
                "overall_repair_conversion@20": repair_all.get("repair_conversion@20"),
                "rows_ge7_repair_conversion@20": repair_rows.get("repair_conversion@20"),
                "rows_ge7_skeleton_to_match@20": repair_rows.get("skeleton_to_match_conversion@20"),
            },
            gate="fail_structure_and_repair_gate",
            boundary="Uses train split noisy skeleton pairs, but lattice-only repair fails structure and repair conversion gates.",
        )
    )

    old_diag = exp3_audit["diagnostic_old_repair"]
    rows.append(
        row(
            name="old GT-WA-style geometry model on predicted skeleton",
            role="old_repair_artifact_diagnostic",
            category="diagnostic",
            scope="MPTS-52 validation subset; old artifact audit",
            source=EXP3_REPAIR_AUDIT,
            overall_match={f"match@{k}": old_diag["overall"].get(f"after_match@{k}") for k in BUDGETS},
            rows_ge7_match={f"match@{k}": old_diag["rows_ge7"].get(f"after_match@{k}") for k in BUDGETS},
            quality={
                "overall_valid": old_diag["overall"].get("valid_rate"),
                "rows_ge7_valid": old_diag["rows_ge7"].get("valid_rate"),
                "artifact_training_data_root": exp3_audit["training_data_audit"].get("geometry_model_data_root"),
            },
            conversion={
                "rows_ge7_repair_conversion@20": old_diag["rows_ge7"].get("repair_conversion@20"),
                "rows_ge7_skeleton_to_match@20": old_diag["rows_ge7"].get("skeleton_to_match_conversion@20"),
            },
            gate="diagnostic_only",
            boundary="Old model was not trained on predicted-skeleton noise and fails badly.",
        )
    )

    mult_all = exp4["overall"]
    mult_rows = exp4["rows_ge7"]
    rows.append(
        row(
            name="rows>=7 multi-geometry proposal",
            role="rows_ge7_multi_geometry_validation",
            category="main_method_candidate",
            scope="MPTS-52 validation; top50 structural-ranked multi-geometry",
            source=EXP4_MULTIGEOM,
            overall_match=match_dict(mult_all),
            rows_ge7_match=match_dict(mult_rows),
            overall_rmse=rmse_dict(mult_all),
            rows_ge7_rmse=rmse_dict(mult_rows),
            quality={
                "overall_valid": mult_all.get("valid_rate"),
                "overall_formula": mult_all.get("formula_consistency"),
                "overall_sg": mult_all.get("sg_consistency"),
                "overall_exact_cover": mult_all.get("exact_cover_retained"),
                "overall_collision": mult_all.get("collision_rate"),
                "rows_ge7_valid": mult_rows.get("valid_rate"),
                "rows_ge7_formula": mult_rows.get("formula_consistency"),
                "rows_ge7_sg": mult_rows.get("sg_consistency"),
                "rows_ge7_exact_cover": mult_rows.get("exact_cover_retained"),
                "rows_ge7_collision": mult_rows.get("collision_rate"),
                "mean_geometry_proposals_per_skeleton": exp4["data_scale"].get("mean_geometry_proposals_per_skeleton"),
                "rows_ge7_match@50": mult_rows.get("match@50"),
            },
            conversion={
                "overall_skeleton_hit@20": mult_all.get("skeleton_hit_coverage@20"),
                "overall_skeleton_to_match@20": mult_all.get("skeleton_to_match_conversion@20"),
                "rows_ge7_skeleton_hit@20": mult_rows.get("skeleton_hit_coverage@20"),
                "rows_ge7_skeleton_to_match@20": mult_rows.get("skeleton_to_match_conversion@20"),
                "rows_ge7_skeleton_hit@50": mult_rows.get("skeleton_hit_coverage@50"),
                "rows_ge7_skeleton_to_match@50": mult_rows.get("skeleton_to_match_conversion@50"),
            },
            gate="fail_validation_gate",
            boundary="Multi-geometry improves structural validity but rows>=7 top50 and conversion remain below gate.",
        )
    )

    gtwa = load_gtwa_mpts52_oracle()
    rows.append(
        row(
            name="GT-WA learned geometry repair oracle",
            role="gt_wa_geometry_repair_oracle",
            category="diagnostic",
            scope="MPTS-52 test subset from mixed GT-WA oracle evaluation",
            source=GTWA_MPTS52_MET,
            overall_match=gtwa["overall_match"],
            rows_ge7_match=gtwa["rows_ge7_match"],
            overall_rmse=gtwa["overall_rmse"],
            rows_ge7_rmse=gtwa["rows_ge7_rmse"],
            quality=gtwa["quality"],
            gate="diagnostic_oracle_only",
            boundary="Uses GT-WA/GT-skeleton; proves component signal only and cannot be an inference result.",
        )
    )

    track_names = [
        ("baseline + Track A scorer", "track_a_scorer"),
        ("baseline + hard-negative structural scorer v2", "hard_negative_scorer"),
        ("baseline + exact-cover filter/proxy", "exact_cover_proxy"),
        ("skeleton proposal + geometry repair + structural scorer proxy", "combined_proxy"),
    ]
    for name, role in track_names:
        item = find_boundary_row(boundary, name)
        if item is None:
            continue
        rows.append(
            row(
                name=name,
                role=role,
                category="auxiliary" if "scorer" in name else "diagnostic",
                scope=str(item.get("scope") or "MPTS-52 validation"),
                source=OP12_BOUNDARY,
                overall_match=normalize_boundary_match(item, rows_key=False),
                rows_ge7_match=normalize_boundary_match(item, rows_key=True),
                overall_rmse=normalize_boundary_rmse(item, rows_key=False),
                rows_ge7_rmse=normalize_boundary_rmse(item, rows_key=True),
                quality=item.get("quality") or {},
                gate="not_main_method",
                boundary="Scorer/proxy/ordinary rerank result; forbidden as main method in this goal.",
            )
        )

    c2s_delta = c2s["delta_vs_true_anchor_all"]
    c2s_rows_delta = c2s["delta_vs_true_anchor_rows_ge7"]
    final_judgment = {
        "allowed_main_result_claim": False,
        "official_anchor": "true CrystaLLM-a GT-SG anchor",
        "c2s3c15_auxiliary_only": True,
        "c2s3c15_delta_vs_true_anchor": c2s_delta,
        "c2s3c15_rows_ge7_delta_vs_true_anchor": c2s_rows_delta,
        "renderer_site_mapping_gate_passed": bool(exp2["gate"].get("selected_train_prototype_passed")),
        "predicted_skeleton_noise_repair_training_artifact_found": True,
        "predicted_skeleton_lattice_repair_gate_passed": bool(exp3_lattice["gates"].get("passed")),
        "multi_geometry_gate_passed": bool(exp4["gate"].get("passed")),
        "main_failure_stage": [
            "lattice-only predicted-skeleton-aware repair fails structure/match conversion",
            "old GT-WA-style repair fails structure/match conversion",
            "multi-geometry rows>=7 top50 and skeleton-to-match conversion fail gate",
        ],
        "reason": (
            "Renderer/site-mapping selected mode passes structure gate, and a train-noisy-skeleton lattice repair pilot was trained, but it fails validation "
            "with rows>=7 after match@20 2.951% and repair conversion@20 1.117%. The old repair collapses formula/SG/valid and conversion, "
            "and multi-geometry reaches rows>=7 top50 13.084% with conversion@50 15.407%, "
            "below the required 23.431% top50 and 30% conversion. No inference-time main method satisfies +5pp on at least two true-anchor metrics with rows>=7 non-degradation."
        ),
    }

    result = {
        "experiment": "opentry_13_exp5_main_ablation_and_final_boundary_current",
        "time": now_iso(),
        "rows": rows,
        "final_judgment": final_judgment,
        "data_sources": {
            "exp1_true_anchor_c2s3c15": str(EXP1),
            "exp2_renderer_site_mapping": str(EXP2_RENDERER),
            "exp3_repair_audit": str(EXP3_REPAIR_AUDIT),
            "exp3_lattice_repair_pilot": str(EXP3_LATTICE_PILOT),
            "exp3_skeleton_proposer": str(EXP3_SKELETON),
            "exp4_multi_geometry": str(EXP4_MULTIGEOM),
            "old_predicted_skeleton_repair": str(OLD_PRED_REPAIR),
            "gtwa_oracle": str(GTWA_MPTS52_MET),
            "legacy_boundary_rows": str(OP12_BOUNDARY),
        },
        "cpu_policy": {
            "training": False,
            "structurematcher_rerun": False,
            "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
    }
    out_path = RESULT_DIR / "experiment_5_main_ablation_and_final_boundary.json"
    write_json(out_path, result)

    def lines_for(category: str) -> str:
        selected = [r for r in rows if r["category"] == category]
        return "\n".join(
            f"- {r['name']} [{r['role']}]: overall {triplet(r['overall_match'])}; rows>=7 {triplet(r['rows_ge7_match'])}; gate={r['gate']}."
            for r in selected
        )

    report = f"""## opentry_13 实验 5：主方法消融与最终判定

结果文件：`model/New_model/opentry_13/results/experiment_5_main_ablation_and_final_boundary.json`

- 为什么做：把 true official anchor、主方法候选、辅助 hybrid 和 oracle diagnostic 分清楚，避免继续把低 baseline、fusion/scorer 或 GT-WA oracle 写成主贡献。
- 核心假设：可 claim 的主方法必须是 inference-time `predicted skeleton proposer + renderer/site mapping + learned/multi-geometry repair`，并且相对 true CrystaLLM-a GT-SG official anchor 至少两个 match 指标 +5pp，rows>=7 不恶化。
- 数据规模：official test anchor/C2S3C15 `8096` samples、rows>=7 `7626`；renderer/skeleton validation `4727` records；lattice repair train noisy pairs `{exp3_lattice['data_scale']['train_pairs']}`、validation candidates `{exp3_lattice['data_scale']['candidate_records']}`；multi-geometry validation `4411` samples、rows>=7 `2033`、candidate records `71930`；GT-WA oracle MPTS-52 subset `{gtwa['quality']['samples']}` samples。
- baseline：true CrystaLLM-a GT-SG anchor match@1/5/20 = `{triplet(anchor_match)}`，rows>=7 = `{triplet(anchor_rows)}`，RMSE@1/5/20 = `{rmse_triplet(rmse_dict(anchor['all']))}`。

主方法候选：
{lines_for('main_method_candidate')}

主方法组件：
{lines_for('main_method_component')}

辅助结果：
{lines_for('auxiliary')}

诊断结果：
{lines_for('diagnostic')}

- C2S3C15 判定：true-anchor-source C2S3C15 相对 true anchor delta = `{pp(c2s_delta.get('match@1'))} / {pp(c2s_delta.get('match@5'))} / {pp(c2s_delta.get('match@20'))}`；rows>=7 delta = `{pp(c2s_rows_delta.get('match@1'))} / {pp(c2s_rows_delta.get('match@5'))} / {pp(c2s_rows_delta.get('match@20'))}`。它只能作为 auxiliary，不能作为论文主方法。
- renderer 判定：selected train-prototype structural selector 通过结构 gate，overall valid `{pct(selected['overall'].get('valid_rate'))}`、formula `{pct(selected['overall'].get('formula_consistency'))}`、SG `{pct(selected['overall'].get('sg_consistency'))}`、exact-cover `{pct(selected['overall'].get('exact_cover_retained'))}`；rows>=7 valid `{pct(selected['rows_ge7'].get('valid_rate'))}`。
- repair 判定：已补充 train-noisy-skeleton lattice MLP repair pilot；rows>=7 after match@1/5/20 = `{triplet({f'match@{k}': repair_rows.get(f'after_match@{k}') for k in BUDGETS})}`，delta = `{pp(repair_rows.get('delta_match@1'))} / {pp(repair_rows.get('delta_match@5'))} / {pp(repair_rows.get('delta_match@20'))}`，repair conversion@20 `{pct(repair_rows.get('repair_conversion@20'))}`，structure_gate_pass={exp3_lattice['gates'].get('structure_gate_pass')}。
- multi-geometry 判定：rows>=7 match@1/5/20/50 = `{pct(mult_rows.get('match@1'))} / {pct(mult_rows.get('match@5'))} / {pct(mult_rows.get('match@20'))} / {pct(mult_rows.get('match@50'))}`；skeleton-to-match conversion@50 `{pct(mult_rows.get('skeleton_to_match_conversion@50'))}`；top50 delta vs CrystaLLM K50 `{pp(exp4['gate'].get('rows_ge7_top50_delta_vs_crystallm_k50'))}`。
- 可信度：高。实验 5 不训练、不重跑 StructureMatcher、不调阈值，只汇总本轮已经写入的 JSON/JSONL 结果；各行明确区分 official test、validation、component gate 和 oracle diagnostic。
- 和历史实验关系：修正旧 exp5 的低-anchor C2S3C15 口径；exp2 说明 renderer/site mapping 可过结构 gate，exp3 先审计旧 artifact、再补充 train-noisy-skeleton lattice repair pilot但仍失败，exp4 说明 multi-geometry 仍不能把 rows>=7 skeleton-hit 转成足够 match。
- 最终判决：`allowed_main_result_claim=False`。失败段落不是“geometry 是瓶颈”这个泛结论，而是：renderer 结构 gate 已过；train-noisy-skeleton lattice repair pilot 的 structure/repair gate 失败；旧 repair 的 formula/SG/valid 和 conversion 失败；multi-geometry 的 rows>=7 top50 coverage 与 skeleton-to-match conversion 仍低于验收线。
- 下一步：不要跑 official，也不要继续调 C2S3C15/scorer/threshold。下一轮若继续主线，应训练 full lattice + free-parameter + collision/local optimization repair，然后重新过 validation gate。
"""
    append_report_at_end(MARKER, report)
    print(json.dumps({"output": str(out_path), "allowed_main_result_claim": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
