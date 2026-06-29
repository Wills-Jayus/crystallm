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

EXP2 = RESULT_DIR / "experiment_2_c2s3c15_true_anchor_replay.json"
EXP3 = RESULT_DIR / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json"
EXP4 = RESULT_DIR / "experiment_4_predicted_skeleton_geometry_repair.json"
OP11_STRICT = NEW_MODEL / "opentry_11" / "results" / "experiment_8_strict_integrated_ablation.json"
OP12_BOUNDARY = NEW_MODEL / "opentry_12" / "results" / "experiment_7_main_ablation_boundary.json"
OP12_REPAIR = NEW_MODEL / "opentry_12" / "results" / "experiment_4_learned_geometry_repair_audit.json"
CRYSTALLM_VAL_LABELS = NEW_MODEL / "opentry_10" / "labels" / "mpts52_val_k50_candidate_labels.jsonl"
MPTS52_VAL_TARGETS = NEW_MODEL / "opentry_7" / "cache" / "mpts_52_val_targets.jsonl"
MPTS52_TEST_TARGETS = NEW_MODEL / "opentry_7" / "cache" / "mpts_52_test_targets.jsonl"
SYMCIF_VAL_GEN = ROOT / "runs" / "symcif_v5_multidataset_wa_decoder" / "mpts52" / "val" / "generations" / "v5_fullgen_eval_pool.jsonl"
SYMCIF_VAL_MET = ROOT / "runs" / "symcif_v5_multidataset_wa_decoder" / "mpts52" / "val" / "metrics" / "v5_fullgen_eval_pool_metrics.jsonl"
GTWA_MPTS52_MET = (
    NEW_MODEL
    / "symcif_experiment"
    / "reports"
    / "symcif_v4_geometry_model_gtwa"
    / "no_oversampling"
    / "metrics"
    / "baseline_per_generation_metrics.jsonl"
)

BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    if marker in text:
        start = text.index(marker)
        next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
        replacement = marker + "\n" + body.rstrip() + "\n"
        if next_marker == -1:
            new_text = text[:start].rstrip() + "\n\n" + replacement
        else:
            new_text = text[:start].rstrip() + "\n\n" + replacement + text[next_marker:]
        REPORT_PATH.write_text(new_text, encoding="utf-8")
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(marker)
        f.write("\n")
        f.write(body.rstrip())
        f.write("\n")


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f}pp"


def triplet_text(m: dict[str, Any] | None) -> str:
    if not m:
        return "NA / NA / NA"
    return " / ".join(pct(m.get(f"match@{k}")) for k in BUDGETS)


def delta_text(d: dict[str, Any] | None) -> str:
    if not d:
        return "NA / NA / NA"
    return " / ".join(pp(d.get(f"match@{k}")) for k in BUDGETS)


def normalize_match(metrics: dict[str, Any], *, prefix: str = "match@") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        key = f"{prefix}{k}"
        value = metrics.get(key)
        out[f"match@{k}"] = None if value is None else float(value)
    return out


def normalize_rows_match(metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        value = metrics.get(f"rows>=7_match@{k}")
        if value is None:
            value = metrics.get(f"rows_ge7_match@{k}")
        if value is None:
            value = metrics.get(f"match@{k}")
        out[f"match@{k}"] = None if value is None else float(value)
    return out


def delta(a: dict[str, float | None], b: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in BUDGETS:
        key = f"match@{k}"
        out[key] = None if a.get(key) is None or b.get(key) is None else float(a[key]) - float(b[key])
    return out


def quality_from_validation_row(row: dict[str, Any]) -> dict[str, Any]:
    q = dict(row.get("quality") or {})
    return {
        "valid_rate": q.get("valid_rate"),
        "formula_consistency": q.get("formula_consistency"),
        "sg_consistency": q.get("sg_consistency"),
        "exact_cover_feasible_rate": q.get("exact_cover_feasible_rate"),
        "skeleton_hit_to_match_conversion": q.get("skeleton_hit_to_match_conversion"),
    }


def make_row(
    *,
    name: str,
    category: str,
    role: str,
    scope: str,
    source: str,
    overall_match: dict[str, float | None],
    rows_ge7_match: dict[str, float | None],
    baseline_name: str | None,
    delta_vs_baseline: dict[str, float | None] | None,
    rows_ge7_delta_vs_baseline: dict[str, float | None] | None,
    gate: str,
    boundary: str,
    quality: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "baseline_name": baseline_name,
        "boundary": boundary,
        "category": category,
        "delta_vs_baseline": delta_vs_baseline,
        "gate": gate,
        "name": name,
        "notes": notes or [],
        "overall_match": overall_match,
        "quality": quality or {},
        "role": role,
        "rows_ge7_delta_vs_baseline": rows_ge7_delta_vs_baseline,
        "rows_ge7_match": rows_ge7_match,
        "scope": scope,
        "source": source,
    }


def load_rows_ge7(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[material_id_from_sample_id(str(row["sample_id"]))] = int(row.get("row_count") or 0) >= 7
    return out


def material_id_from_sample_id(sample_id: str) -> str:
    return str(sample_id).split("__")[-1]


def summarize_hits(
    sample_hits: dict[str, dict[int, bool]],
    rows_ge7: dict[str, bool],
    *,
    sample_universe: set[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    samples = sorted(sample_universe or set(sample_hits))
    rows7 = [sid for sid in samples if rows_ge7.get(sid, False)]
    out: dict[str, dict[str, float | int]] = {}
    for k in BUDGETS:
        hits = [bool(sample_hits.get(sid, {}).get(k, False)) for sid in samples]
        hits7 = [bool(sample_hits.get(sid, {}).get(k, False)) for sid in rows7]
        out[f"match@{k}"] = {
            "samples": len(samples),
            "rows_ge7_samples": len(rows7),
            "overall": float(sum(hits) / max(1, len(hits))),
            "rows_ge7": float(sum(hits7) / max(1, len(hits7))),
        }
    return out


def load_crystallm_val_hits() -> dict[str, dict[int, bool]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with CRYSTALLM_VAL_LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                groups[material_id_from_sample_id(str(row["sample_id"]))].append(row)
    out: dict[str, dict[int, bool]] = {}
    for sid, rows in groups.items():
        rows.sort(key=lambda r: int(r.get("rank") or 10**9))
        out[sid] = {k: any(bool(r.get("match")) for r in rows[:k]) for k in BUDGETS}
    return out


def load_symcif_val_hits() -> dict[str, dict[int, bool]]:
    groups: dict[str, list[tuple[float, int, bool]]] = defaultdict(list)
    with SYMCIF_VAL_GEN.open("r", encoding="utf-8") as gf, SYMCIF_VAL_MET.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            if not gen_line.strip() or not met_line.strip():
                continue
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            score = gen.get("generation_score")
            groups[material_id_from_sample_id(str(gen["sample_id"]))].append(
                (
                    float(score) if score is not None else -1.0e30,
                    int(gen.get("gen_index") or 0),
                    bool(met.get("match_ok")),
                )
            )
    out: dict[str, dict[int, bool]] = {}
    for sid, rows in groups.items():
        rows.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        out[sid] = {k: any(match for _, _, match in rows[:k]) for k in BUDGETS}
    return out


def build_oracle_union_diagnostic(rows_ge7: dict[str, bool]) -> dict[str, Any]:
    c_hits = load_crystallm_val_hits()
    s_hits = load_symcif_val_hits()
    universe = set(rows_ge7)
    union_hits: dict[str, dict[int, bool]] = {}
    for sid in universe:
        union_hits[sid] = {k: bool(c_hits.get(sid, {}).get(k, False) or s_hits.get(sid, {}).get(k, False)) for k in BUDGETS}
    c_summary = summarize_hits(c_hits, rows_ge7, sample_universe=universe)
    s_summary = summarize_hits(s_hits, rows_ge7, sample_universe=universe)
    union_summary = summarize_hits(union_hits, rows_ge7, sample_universe=universe)
    return {
        "candidate_budget_note": "diagnostic oracle coverage: C@K OR S@K can use up to 2K candidates and is not a ranked topK method",
        "crystallm_summary": c_summary,
        "samples": len(universe),
        "symcif_summary": s_summary,
        "union_summary": union_summary,
    }


def union_summary_to_matches(summary: dict[str, Any], field: str) -> dict[str, float | None]:
    return {f"match@{k}": float(summary["union_summary"][f"match@{k}"][field]) for k in BUDGETS}


def load_gtwa_mpts52_oracle() -> dict[str, Any]:
    rows_ge7: dict[str, bool] = {}
    with MPTS52_TEST_TARGETS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows_ge7[str(row["sample_id"])] = int(row.get("row_count") or 0) >= 7
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with GTWA_MPTS52_MET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = str(row["sample_id"])
                if sid.startswith("mpts_52_test_orig__"):
                    groups[sid].append(row)
    samples = sorted(groups)
    rows7 = [sid for sid in samples if rows_ge7.get(sid, False)]
    overall_match: dict[str, float | None] = {}
    rows_ge7_match: dict[str, float | None] = {}
    quality: dict[str, Any] = {
        "condition": "GT-WA/GT-skeleton oracle on the MPTS-52 subset of a mixed 500-sample evaluation; diagnostic only",
        "rows_ge7_samples": len(rows7),
        "samples": len(samples),
    }
    for k in BUDGETS:
        all_hits: list[bool] = []
        rows7_hits: list[bool] = []
        valid_any: list[bool] = []
        formula_any: list[bool] = []
        sg_any: list[bool] = []
        exact_any: list[bool] = []
        for sid in samples:
            arr = sorted(groups[sid], key=lambda r: int(r.get("rank") or int(r.get("gen_index") or 0) + 1))[:k]
            hit = any(bool(r.get("match_ok")) for r in arr)
            all_hits.append(hit)
            if rows_ge7.get(sid, False):
                rows7_hits.append(hit)
            valid_any.append(any(bool(r.get("valid")) for r in arr))
            formula_any.append(any(bool(r.get("formula_ok")) for r in arr))
            sg_any.append(any(bool(r.get("space_group_ok")) for r in arr))
            exact_any.append(any(bool(r.get("multiplicity_ok")) for r in arr))
        overall_match[f"match@{k}"] = float(sum(all_hits) / max(1, len(all_hits)))
        rows_ge7_match[f"match@{k}"] = float(sum(rows7_hits) / max(1, len(rows7_hits)))
        quality[f"valid_any@{k}"] = float(sum(valid_any) / max(1, len(valid_any)))
        quality[f"formula_ok_any@{k}"] = float(sum(formula_any) / max(1, len(formula_any)))
        quality[f"sg_ok_any@{k}"] = float(sum(sg_any) / max(1, len(sg_any)))
        quality[f"exact_cover_any@{k}"] = float(sum(exact_any) / max(1, len(exact_any)))
    return {"overall_match": overall_match, "quality": quality, "rows_ge7_match": rows_ge7_match}


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    exp2 = read_json(EXP2)
    exp3 = read_json(EXP3)
    exp4 = read_json(EXP4)
    strict = read_json(OP11_STRICT)
    boundary = read_json(OP12_BOUNDARY)
    repair = read_json(OP12_REPAIR)

    true_anchor = exp2["systems"]["true_gt_sg_anchor_crystallm_a"]
    internal_low = exp2["systems"]["internal_low_baseline_pure_model"]
    c2s = exp2["systems"]["c2s3c15"]
    official_anchor_match = normalize_match(true_anchor["all_match"])
    official_anchor_rows = normalize_match(true_anchor["rows_ge7_match"])
    validation_baseline = strict["full_validation"]["baseline"]["metrics"]
    validation_baseline_match = normalize_match(validation_baseline)
    validation_baseline_rows = normalize_rows_match(validation_baseline)

    rows: list[dict[str, Any]] = []
    rows.append(
        make_row(
            name="true GT-SG CrystaLLM-a anchor",
            category="reference",
            role="official_true_anchor",
            scope="MPTS-52 official test K20 full",
            source=str(true_anchor["source"]),
            overall_match=official_anchor_match,
            rows_ge7_match=official_anchor_rows,
            baseline_name=None,
            delta_vs_baseline=None,
            rows_ge7_delta_vs_baseline=None,
            gate="reference",
            boundary="Only official anchor allowed for main claim.",
            quality=true_anchor.get("quality_diagnostics"),
        )
    )
    rows.append(
        make_row(
            name="internal low pure-model baseline",
            category="reference",
            role="official_low_internal_baseline_not_main_anchor",
            scope="MPTS-52 official test K20 full",
            source=str(internal_low["source"]),
            overall_match=normalize_match(internal_low["all_match"]),
            rows_ge7_match=normalize_match(internal_low["rows_ge7_match"]),
            baseline_name="true GT-SG CrystaLLM-a anchor",
            delta_vs_baseline=delta(normalize_match(internal_low["all_match"]), official_anchor_match),
            rows_ge7_delta_vs_baseline=delta(normalize_match(internal_low["rows_ge7_match"]), official_anchor_rows),
            gate="not_a_main_anchor",
            boundary="Different candidate source from CrystaLLM-a; cannot be used to claim main success.",
            quality=internal_low.get("quality_diagnostics"),
        )
    )
    rows.append(
        make_row(
            name="C2S3C15 frozen official",
            category="auxiliary",
            role="auxiliary_hybrid_official",
            scope="MPTS-52 official test K20 full",
            source=str(c2s["source"]),
            overall_match=normalize_match(c2s["all_match"]),
            rows_ge7_match=normalize_match(c2s["rows_ge7_match"]),
            baseline_name="true GT-SG CrystaLLM-a anchor",
            delta_vs_baseline=exp2["deltas"]["c2s3c15_vs_true_gt_sg_anchor_all"],
            rows_ge7_delta_vs_baseline=exp2["deltas"]["c2s3c15_vs_true_gt_sg_anchor_rows_ge7"],
            gate="fail_true_anchor_main_gate",
            boundary="Auxiliary hybrid only; fixed C2S3C15 is below true official anchor.",
            quality=c2s.get("quality_diagnostics"),
            notes=["No C/S retuning, no threshold tuning, no scorer."],
        )
    )
    rows.append(
        make_row(
            name="validation GT-SG baseline",
            category="reference",
            role="validation_reference_baseline",
            scope="MPTS-52 validation K50 full",
            source=str(OP11_STRICT),
            overall_match=validation_baseline_match,
            rows_ge7_match=validation_baseline_rows,
            baseline_name=None,
            delta_vs_baseline=None,
            rows_ge7_delta_vs_baseline=None,
            gate="reference",
            boundary="Validation reference only; not interchangeable with official test anchor.",
            quality=strict["full_validation"]["baseline"].get("diagnostics"),
        )
    )

    exp3_overall = {
        f"match@{k}": exp3["overall"].get(f"top{k}_hydrated_match_coverage") for k in BUDGETS
    }
    exp3_rows = {
        f"match@{k}": exp3["rows_ge7"].get(f"top{k}_hydrated_match_coverage") for k in BUDGETS
    }
    rows.append(
        make_row(
            name="rows>=7 train-derived skeleton proposer",
            category="main_method_candidate",
            role="skeleton_proposer_validation_gate",
            scope="MPTS-52 validation; hydrated existing SymCIF candidates",
            source=str(EXP3),
            overall_match=exp3_overall,
            rows_ge7_match=exp3_rows,
            baseline_name="validation GT-SG baseline",
            delta_vs_baseline=delta(exp3_overall, validation_baseline_match),
            rows_ge7_delta_vs_baseline=delta(exp3_rows, validation_baseline_rows),
            gate="fail_validation_gate",
            boundary="Main-route proposer candidate, but match conversion is insufficient.",
            quality={
                "overall_top20_skeleton_hit_coverage": exp3["overall"].get("top20_skeleton_hit_coverage"),
                "rows_ge7_top20_skeleton_hit_coverage": exp3["rows_ge7"].get("top20_skeleton_hit_coverage"),
                "rows_ge7_top50_exact_cover_feasible_any": exp3["rows_ge7"].get("top50_exact_cover_feasible_any"),
                "rows_ge7_top50_proposal_skeleton_to_hydrated_match_conversion": exp3["rows_ge7"].get(
                    "top50_proposal_skeleton_to_hydrated_match_conversion"
                ),
            },
        )
    )

    exp4_overall = {f"match@{k}": exp4["overall"].get(f"after_match@{k}") for k in BUDGETS}
    exp4_rows = {f"match@{k}": exp4["rows_ge7"].get(f"after_match@{k}") for k in BUDGETS}
    rows.append(
        make_row(
            name="predicted-skeleton learned geometry repair",
            category="main_method_candidate",
            role="skeleton_proposer_plus_learned_repair_validation",
            scope="MPTS-52 validation subset; predicted skeleton top20",
            source=str(EXP4),
            overall_match=exp4_overall,
            rows_ge7_match=exp4_rows,
            baseline_name="pre-repair same predicted skeleton subset",
            delta_vs_baseline={f"match@{k}": exp4["overall"].get(f"delta_match@{k}") for k in BUDGETS},
            rows_ge7_delta_vs_baseline={f"match@{k}": exp4["rows_ge7"].get(f"delta_match@{k}") for k in BUDGETS},
            gate="fail_validation_gate",
            boundary="True predicted-skeleton repair, but formula/SG validity and conversion fail.",
            quality={
                "overall_repair_conversion@5": exp4["overall"].get("repair_conversion@5"),
                "overall_repair_conversion@20": exp4["overall"].get("repair_conversion@20"),
                "rows_ge7_repair_conversion@5": exp4["rows_ge7"].get("repair_conversion@5"),
                "rows_ge7_repair_conversion@20": exp4["rows_ge7"].get("repair_conversion@20"),
                "overall_formula_consistency_rate": exp4["overall"].get("formula_consistency_rate"),
                "rows_ge7_formula_consistency_rate": exp4["rows_ge7"].get("formula_consistency_rate"),
                "overall_sg_consistency_rate": exp4["overall"].get("sg_consistency_rate"),
                "rows_ge7_sg_consistency_rate": exp4["rows_ge7"].get("sg_consistency_rate"),
                "exact_cover_retained_rate": exp4["overall"].get("exact_cover_retained_rate"),
            },
        )
    )

    by_name = {row["name"]: row for row in boundary["validation_ablation_rows"]}
    for source_name, category, role, gate, boundary_text in [
        ("SymCIF v5 neural skeleton/geometry proposer", "main_method_candidate", "symcif_v5_generation_diagnostic", "fail_main_gate", "Generation-side main candidate diagnostic; validation only."),
        ("baseline + Track A scorer", "auxiliary", "track_a_scorer", "fail_main_gate", "Auxiliary scorer/rerank, not a main method."),
        ("baseline + hard-negative structural scorer v2", "auxiliary", "hard_negative_scorer", "fail_main_gate", "Auxiliary scorer trained from validation labels; not a main method."),
        ("baseline + exact-cover filter/proxy", "diagnostic", "exact_cover_proxy", "fail_main_gate", "Diagnostic proxy only; uses candidate filtering, not a proposer success."),
        ("skeleton proposal + geometry repair + structural scorer proxy", "diagnostic", "combined_proxy", "fail_main_gate", "Proxy/scorer diagnostic; not an inference-time main pipeline."),
    ]:
        br = by_name[source_name]
        om = normalize_match(br["overall_match"])
        rm = normalize_rows_match(br["rows_ge7_match"])
        rows.append(
            make_row(
                name=source_name,
                category=category,
                role=role,
                scope=str(br.get("scope") or "MPTS-52 validation"),
                source=str(OP12_BOUNDARY),
                overall_match=om,
                rows_ge7_match=rm,
                baseline_name="validation GT-SG baseline",
                delta_vs_baseline=delta(om, validation_baseline_match),
                rows_ge7_delta_vs_baseline=delta(rm, validation_baseline_rows),
                gate=gate,
                boundary=boundary_text,
                quality=quality_from_validation_row(br),
            )
        )

    rows_ge7_val = load_rows_ge7(MPTS52_VAL_TARGETS)
    oracle = build_oracle_union_diagnostic(rows_ge7_val)
    oracle_overall = union_summary_to_matches(oracle, "overall")
    oracle_rows = union_summary_to_matches(oracle, "rows_ge7")
    rows.append(
        make_row(
            name="oracle union coverage C@K OR SymCIF@K",
            category="diagnostic",
            role="oracle_union_coverage_upper_bound",
            scope="MPTS-52 validation full; up to 2K candidates per K",
            source=str(RESULT_DIR / "experiment_5_main_ablation_and_final_boundary.json"),
            overall_match=oracle_overall,
            rows_ge7_match=oracle_rows,
            baseline_name="validation GT-SG baseline",
            delta_vs_baseline=delta(oracle_overall, validation_baseline_match),
            rows_ge7_delta_vs_baseline=delta(oracle_rows, validation_baseline_rows),
            gate="diagnostic_upper_bound_not_ranked_method",
            boundary="Oracle coverage/fusion upper bound, not a ranked topK inference method.",
            quality={
                "candidate_budget_note": oracle["candidate_budget_note"],
                "samples": oracle["samples"],
            },
        )
    )

    gtwa = load_gtwa_mpts52_oracle()
    rows.append(
        make_row(
            name="GT-WA learned geometry repair oracle",
            category="diagnostic",
            role="gt_wa_geometry_repair_oracle",
            scope="MPTS-52 test subset from mixed GT-WA/GT-skeleton oracle top20",
            source=str(GTWA_MPTS52_MET),
            overall_match=gtwa["overall_match"],
            rows_ge7_match=gtwa["rows_ge7_match"],
            baseline_name=None,
            delta_vs_baseline=None,
            rows_ge7_delta_vs_baseline=None,
            gate="diagnostic_oracle_only",
            boundary="Uses GT-WA/GT-skeleton; component signal only, never an inference result.",
            quality=gtwa["quality"],
            notes=[
                "opentry_12 MP-20 GT-WA K5 also showed strong geometry signal, but GT-WA is forbidden at inference.",
                f"MP-20 GT-WA K1/K5 all={triplet_text({'match@1': repair['mp20_gtwa_learned_geometry_k5']['all']['match@1'], 'match@5': repair['mp20_gtwa_learned_geometry_k5']['all']['match@5'], 'match@20': None})}.",
            ],
        )
    )

    official_c2s_delta = exp2["deltas"]["c2s3c15_vs_true_gt_sg_anchor_all"]
    official_c2s_rows_delta = exp2["deltas"]["c2s3c15_vs_true_gt_sg_anchor_rows_ge7"]
    official_plus5_count = sum(1 for k in BUDGETS if float(official_c2s_delta[f"match@{k}"]) >= 0.05)
    official_rows_not_degraded = all(float(official_c2s_rows_delta[f"match@{k}"]) >= -1.0e-12 for k in BUDGETS)
    main_candidate_gates = {
        "exp3_skeleton_proposer_validation_gate_pass": bool(exp3["decision"]["validation_gate_pass"]),
        "exp4_predicted_skeleton_repair_validation_gate_pass": bool(exp4["decision"]["validation_gate_pass"]),
    }
    final_judgment = {
        "allowed_main_result_claim": False,
        "official_true_anchor_for_claim": "true GT-SG CrystaLLM-a anchor",
        "official_c2s3c15_plus5_metric_count_vs_true_anchor": official_plus5_count,
        "official_c2s3c15_rows_ge7_not_degraded": official_rows_not_degraded,
        "main_candidate_validation_gates": main_candidate_gates,
        "reason": (
            "No inference-time main candidate exceeds the true GT-SG CrystaLLM-a official anchor by +5pp on at least two match metrics. "
            "C2S3C15 is below the true anchor; skeleton proposer and predicted-skeleton repair fail validation; oracle/GT-WA results are diagnostic only."
        ),
        "result_classification": {
            "main_method_candidates": "failed_validation_or_diagnostic_only",
            "auxiliary_results": "C2S3C15/Track A/hard-negative only",
            "diagnostic_results": "oracle union/GT-WA repair/exact-cover proxy only",
        },
    }

    result = {
        "cpu_policy": {
            "parallel_workers": 1,
            "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "structurematcher_rerun": False,
            "training": False,
        },
        "data_sources": {
            "c2s3c15_true_anchor_replay": str(EXP2),
            "gtwa_mpts52_oracle_metrics": str(GTWA_MPTS52_MET),
            "oracle_union_crystallm_labels": str(CRYSTALLM_VAL_LABELS),
            "oracle_union_symcif_metrics": str(SYMCIF_VAL_MET),
            "predicted_skeleton_repair": str(EXP4),
            "rows7_skeleton_proposer": str(EXP3),
            "validation_strict_ablation": str(OP11_STRICT),
        },
        "experiment": "opentry_13_exp5_main_ablation_and_final_boundary",
        "final_judgment": final_judgment,
        "oracle_union_diagnostic": oracle,
        "rows": rows,
        "time": now_iso(),
    }

    out_path = RESULT_DIR / "experiment_5_main_ablation_and_final_boundary.json"
    write_json(out_path, result)

    marker = "<!-- OPENTRY13_EXP5_MAIN_ABLATION_BOUNDARY -->"
    main_rows = [r for r in rows if r["category"] == "main_method_candidate"]
    aux_rows = [r for r in rows if r["category"] == "auxiliary"]
    diag_rows = [r for r in rows if r["category"] == "diagnostic"]

    def row_lines(selected: list[dict[str, Any]]) -> str:
        lines = []
        for row in selected:
            lines.append(
                f"- {row['name']} [{row['role']}]: overall {triplet_text(row['overall_match'])}; "
                f"rows>=7 {triplet_text(row['rows_ge7_match'])}; gate={row['gate']}."
            )
        return "\n".join(lines)

    report = f"""## opentry_13 实验 5：主方法消融与最终边界判定

结果文件：`model/New_model/opentry_13/results/experiment_5_main_ablation_and_final_boundary.json`

- 为什么做：把主方法候选、辅助结果和诊断结果拆开，统一报告 overall 与 rows>=7 的 match@1/5/20，避免把 C2S3C15、Track A、hard-negative scorer、oracle union 或 GT-WA oracle 包装成论文主贡献。
- 核心假设：真正可 claim 的主方法必须是 inference-time `skeleton proposer + learned geometry repair`，并且在 true GT-SG CrystaLLM-a official anchor 上至少两个 match 指标 +5pp，rows>=7 不恶化；否则只能辅助或诊断。
- 数据规模：official test `8096` 样本、rows>=7 `7626`；validation strict ablation `5000` 样本；exp3 validation repr `4727`；exp4 predicted-skeleton repair subset 有候选样本 `712`；oracle union validation `5000` 样本；GT-WA oracle MPTS-52 test subset `{gtwa['quality']['samples']}` 样本。
- baseline：主 official anchor 是 true GT-SG CrystaLLM-a，match@1/5/20 = `{triplet_text(official_anchor_match)}`，rows>=7 = `{triplet_text(official_anchor_rows)}`。低 pure-model baseline 只保留为历史参考，不允许作为主 anchor。
- 方法变化：本实验不新增方法、不调 C/S 比例、不做 threshold tuning、不训练 scorer；只读取既有 JSON/JSONL 做 replay/audit。oracle union 是 `C@K OR SymCIF@K` coverage upper-bound，最多 2K 候选，不是 ranked topK 方法。

主方法候选：
{row_lines(main_rows)}

辅助结果：
{row_lines(aux_rows)}

诊断结果：
{row_lines(diag_rows)}

- 关键 official 判定：C2S3C15 相对 true anchor delta = `{delta_text(official_c2s_delta)}`；rows>=7 delta = `{delta_text(official_c2s_rows_delta)}`。它没有任何一个 true-anchor 指标达到 +5pp，rows>=7 也全部下降，因此只能是 auxiliary hybrid result。
- 可信度：高。没有重新跑 official、没有使用 official 反馈回调、没有训练新 scorer；GT-WA 与 oracle union 明确标为 diagnostic。限制是 validation/official/500-sample oracle 的 split 和样本规模不同，表内已用 scope 区分，不做跨 split 主 claim。
- 和历史实验关系：exp1/2 已证明低 baseline 不能当主 anchor；exp3 说明 skeleton-hit 有信号但 hydrated match/conversion 不够；exp4 说明 predicted-skeleton repair 链路失败；opentry_11/12 的 Track A、hard-negative、exact-cover proxy 继续只作为辅助/诊断。
- 最终判决：`allowed_main_result_claim=False`。当前没有任何 inference-time 主方法满足 true GT-SG anchor 上至少两个 match 指标 +5pp 且 rows>=7 不恶化。C2S3C15 降级 auxiliary；Track A/hard-negative 停止作为主线；oracle union、GT-WA repair、exact-cover proxy 只作为诊断。
- 下一步：继续主线只能做 train-data 设计的 rows>=7 skeleton proposer 与适配 predicted skeleton/site-mapping 噪声的 learned/local geometry repair；先过 validation gate，再考虑 official。禁止继续调 C2S3C15 比例、普通 rerank、threshold tuning 或 official 结果回调。
"""
    append_or_replace_report(marker, report)
    print(json.dumps({"output": str(out_path), "final_judgment": final_judgment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
