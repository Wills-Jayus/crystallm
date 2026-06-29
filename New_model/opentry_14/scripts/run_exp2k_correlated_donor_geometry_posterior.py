#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2k_correlated_donor_geometry_posterior"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import formula_counts, read_jsonl, sample_id, source_skeleton_rows  # noqa: E402
from run_exp2j_chemical_site_order_assignment import build_train_priors, chemical_exact_cover_assignments, pct, pp  # noqa: E402
from run_exp2f_permutation_aware_alignment import by_sid_ranked, sample_sets, split_safe_pool  # noqa: E402
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    build_reference_indexes,
    eval_sample,
    ranked_references,
    render_candidate,
    summarize,
)
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
EXP2D_RESULT = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
EXP2D_SITE_EVAL = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl"
EXP2J_RESULT = RESULT_DIR / "experiment_2j_chemical_site_order_assignment.json"
EXP2J_CHEM_EVAL = OUT_DIR / "artifacts" / "exp2j_chemical_site_order_assignment" / "evaluated_chemical_assignment_candidates.jsonl"
BUDGETS = (1, 5, 20, 50)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def append_or_replace(path: Path, marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with path.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n\n"):
                f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def lattice_from_record(record: dict[str, Any]) -> dict[str, float]:
    return {k: float(record["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}


def row_signature(rows: list[dict[str, Any]], include_element: bool) -> tuple[tuple[str, ...], ...]:
    items: list[tuple[str, ...]] = []
    for row in rows:
        if include_element:
            items.append((str(row.get("orbit_id")), str(row.get("element")), str(int(row.get("multiplicity") or 0))))
        else:
            items.append((str(row.get("orbit_id")), str(int(row.get("multiplicity") or 0))))
    return tuple(sorted(items))


def donor_pool(
    *,
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    skeleton_key: str,
    source_record: dict[str, Any] | None,
    indexes: dict[str, Any],
    limit: int,
    include_source: bool,
) -> list[tuple[str, dict[str, Any], float]]:
    refs = ranked_references(
        target=target,
        rows=rows,
        skeleton_key=skeleton_key,
        source_record=source_record if include_source else None,
        indexes=indexes,
        limit=max(1, int(limit)),
    )
    if not include_source and source_record is not None:
        refs = [(tier, ref, score) for tier, ref, score in refs if str(ref.get("sample_id")) != str(source_record.get("sample_id"))]
    return refs[: max(1, int(limit))]


def build_variant_rows(
    *,
    variant: str,
    hydrated: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    siteassign: dict[str, list[dict[str, Any]]],
    chemassign: dict[str, list[dict[str, Any]]],
    donor: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_donor25_p5": (10, 10, 0, 25, 5),
        "h10_s10_chem15_donor10_p5": (10, 10, 15, 10, 5),
        "donor25_h10_s10_p5": (10, 10, 0, 25, 5),
        "h10_interleave_s10_chem10_donor15_p5": (10, 10, 10, 15, 5),
    }
    hq, sq, cq, dq, pq = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign) | set(chemassign) | set(donor)):
        hyd = hydrated.get(sid, [])
        site = siteassign.get(sid, [])
        chem = chemassign.get(sid, [])
        don = donor.get(sid, [])
        proto = prototype.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "donor25_h10_s10_p5":
            selected.extend(don[:dq])
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq])
        elif variant == "h10_interleave_s10_chem10_donor15_p5":
            for i in range(max(hq, sq, cq, dq)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
                if i < cq and i < len(chem):
                    selected.append(chem[i])
                if i < dq and i < len(don):
                    selected.append(don[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(chem[:cq])
            selected.extend(don[:dq])
            selected.extend(proto[:pq])
        for rank, row in enumerate(selected[:top_k], start=1):
            item = dict(row)
            item["rank"] = rank
            item["selection_variant"] = variant
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 10**9)))
    return out


def report_body(result: dict[str, Any]) -> str:
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    exp2j = result["baseline_exp2j"]["rows_ge7"]
    oracle = result["oracle_diagnostic"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2k：correlated donor geometry posterior

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2k_correlated_donor_geometry_posterior/`

- 为什么做：Exp2h 的 row-wise train free-parameter prior 失败，Exp3j/3k 的 CHGNet local optimizer 也只改善 collision/valid、不提升 match；剩余假设是跨 row 的 lattice/free-parameter 相关性被破坏，单 row 参数或局部优化无法换到正确 basin。
- 核心假设：如果同一个 train donor 的 lattice 与所有 matched row free parameters 构成可迁移的 correlated geometry bundle，那么在 Exp2j chemical assignment 后按 train donor bundle 整体迁移，应比 source geometry 或 row-wise prior 产生新的 rows>=7 sample-level match。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；generated donor candidates `{result['data_scale']['generated_donor_candidates']}`；evaluated donor candidates `{result['data_scale']['evaluated_donor_candidates']}`；train donor records `{result['reference_index']['train_records']}`；workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2j best `{result['baseline_exp2j']['best_variant']}` rows>=7 match@50 `{pct(exp2j.get('match@50'))}`、conversion `{pct(exp2j.get('skeleton_to_match_conversion@50'))}`、collision `{pct(exp2j.get('collision_rate'))}`。
- 方法变化：保留 Exp2j chemical/site-order posterior；每个 assigned skeleton 不再复制 source geometry，也不逐 row 独立采样，而是按 train split 的 same_skeleton / same_SG+atom_count / same_SG donor score 选择完整 donor，迁移 donor lattice 与 all-row free params。推理排序只用 legal/formula/SG/exact/collision/volume/reference_score，不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：donor-only match/skelmatch samples `{oracle['donor_match_samples']}` / `{oracle['donor_skelmatch_samples']}`；相对 Exp2j 新增 match/skelmatch `{oracle['new_match_samples_vs_exp2j']}` / `{oracle['new_skelmatch_samples_vs_exp2j']}`；union upper bound match@50 `{pct(oracle['union_match50_upper_bound'])}`、conversion `{pct(oracle['union_conversion50_upper_bound'])}`。
- 可信度：中等。donor 只来自 train split，validation 推理不使用禁用标签，所有候选真实 render/parse/SG/StructureMatcher；限制是 donor bundle 仍是 prototype posterior，不是端到端 learned continuous repair head。
- 和历史实验关系：这是 Exp2h row-wise prior 与 Exp2j chemical assignment 的组合修正，直接检验“跨 row geometry correlation”是否是 Exp3 local optimizer 失败后的剩余瓶颈。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；exp3_line_passed=`{gate['exp3_line_passed']}`；conversion delta vs Exp2j `{pp(gate['rows_ge7_conversion50_delta_vs_exp2j'])}`；match@50 delta vs Exp2j `{pp(gate['rows_ge7_match50_delta_vs_exp2j'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=6)
    parser.add_argument("--assignment-limit", type=int, default=4)
    parser.add_argument("--assignment-beam-width", type=int, default=128)
    parser.add_argument("--donor-limit", type=int, default=4)
    parser.add_argument("--include-source-donor", action="store_true")
    parser.add_argument("--max-target-atoms", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    train_struct_list = list(train_struct.values())
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_struct_list + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    priors = build_train_priors(train_struct_list)
    indexes = build_reference_indexes(train_struct_list)

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    exp2j = read_json(EXP2J_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))
    chem_rows = list(read_jsonl_iter(EXP2J_CHEM_EVAL))

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    donor_tiers: Counter[str] = Counter()
    fallback_rows: Counter[str] = Counter()
    donor_exact_sig = donor_orbit_sig = 0

    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        target_atom_count = int(sum(counts.values()))
        candidates: list[dict[str, Any]] = []
        if target_atom_count > int(args.max_target_atoms):
            failures["skipped_target_atom_count_gt_limit"] += 1
            payloads.append({"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates})
            continue
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_struct = train_struct.get(source_id)
            if source_repr is None or source_struct is None:
                failures["missing_source"] += 1
                continue
            original_rows = source_skeleton_rows(engine, source_repr)
            if not original_rows:
                failures["empty_source_rows"] += 1
                continue
            assignments = chemical_exact_cover_assignments(
                original_rows,
                counts,
                int(target["sg"]),
                priors,
                limit=int(args.assignment_limit),
                beam_width=int(args.assignment_beam_width),
            )
            if not assignments:
                failures["no_chemical_exact_cover_assignment"] += 1
                continue
            skeleton_key = str(proposal.get("skeleton_key") or "")
            for assignment_rank, (assignment_score, assignment, assignment_meta) in enumerate(assignments, start=1):
                assigned_rows: list[dict[str, Any]] = []
                for row, element in zip(original_rows, assignment):
                    item = dict(row)
                    item["source_element"] = str(row.get("element"))
                    item["element"] = str(element)
                    assigned_rows.append(item)
                exact_sig = row_signature(assigned_rows, include_element=True)
                orbit_sig = row_signature(assigned_rows, include_element=False)
                refs = donor_pool(
                    target=target,
                    rows=assigned_rows,
                    skeleton_key=skeleton_key,
                    source_record=source_struct,
                    indexes=indexes,
                    limit=int(args.donor_limit),
                    include_source=bool(args.include_source_donor),
                )
                if not refs:
                    failures["no_donor_refs"] += 1
                    continue
                for donor_rank, (tier, ref, ref_score) in enumerate(refs, start=1):
                    donor_tiers[tier] += 1
                    ref_exact_sig = row_signature(ref.get("wa_table") or [], include_element=True)
                    ref_orbit_sig = row_signature(ref.get("wa_table") or [], include_element=False)
                    donor_exact_sig += int(ref_exact_sig == exact_sig)
                    donor_orbit_sig += int(ref_orbit_sig == orbit_sig)
                    try:
                        params, fallback_count = flexible_params_from_reference(engine, assigned_rows, ref, neural_params=None)
                        option = {"lattice": lattice_from_record(ref), "params": params}
                        cif, render_meta = render_candidate(
                            engine=engine,
                            target=target,
                            rows=assigned_rows,
                            option=option,
                            data_name=f"{sid}_exp2k_p{int(proposal.get('rank') or 0)}_a{assignment_rank}_d{donor_rank}",
                        )
                        fallback_rows[str(fallback_count)] += 1
                        row = {
                            "sample_id": sid,
                            "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                            "proposal_rank": int(proposal.get("rank") or 0),
                            "assignment_rank": int(assignment_rank),
                            "assignment_score": float(assignment_score),
                            "assignment_source_preserved_atoms": int(assignment_meta.get("source_preserved_atoms") or 0),
                            "donor_rank": int(donor_rank),
                            "geometry_rank": len(candidates) + 1,
                            "raw_generation_order": len(candidates) + 1,
                            "row_count": int(target_repr.get("row_count") or 0),
                            "sg": int(target["sg"]),
                            "formula_counts": counts,
                            "target_atom_count": target_atom_count,
                            "source_sample_id": source_id,
                            "proposal_source": str(proposal.get("source") or ""),
                            "predicted_skeleton_key": skeleton_key,
                            "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                            "predicted_skeleton_hit": skeleton_key == str(target_repr.get("canonical_skeleton_key") or ""),
                            "candidate_row_count": len(assigned_rows),
                            "site_mapping_rule": "chemical_site_order_plus_correlated_train_donor_geometry",
                            "geometry_source": f"correlated_donor_{tier}",
                            "reference_sample_id": str(ref["sample_id"]),
                            "reference_score": float(ref_score + assignment_score),
                            "donor_reference_score": float(ref_score),
                            "param_fallback_rows": int(fallback_count),
                            "donor_exact_row_signature_match": bool(ref_exact_sig == exact_sig),
                            "donor_orbit_row_signature_match": bool(ref_orbit_sig == orbit_sig),
                            "cif": cif,
                            **render_meta,
                        }
                        candidates.append(row)
                        generation_meta.append({k: v for k, v in row.items() if k != "cif"})
                    except Exception as exc:  # noqa: BLE001
                        failures[f"render_failed:{type(exc).__name__}"] += 1
        payloads.append({"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": int(target["sg"]), "candidates": candidates})
        if si % 200 == 0:
            print(f"[exp2k-donor] rendered {si}/{len(selected_sids)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    result_filename = f"experiment_2k_correlated_donor_geometry_posterior{suffix}.json"
    write_jsonl(ARTIFACT_DIR / f"generated_correlated_donor_meta{suffix}.jsonl", generation_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2k-donor] evaluated {i}/{len(futures)}", flush=True)
    ranked_donor = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_correlated_donor_candidates{suffix}.jsonl", ranked_donor)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    chem_by_sid = by_sid_ranked(assign_structural_ranks(chem_rows, int(args.top_k)))
    donor_by_sid = by_sid_ranked(ranked_donor)
    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_donor25_p5", "h10_s10_chem15_donor10_p5", "donor25_h10_s10_p5", "h10_interleave_s10_chem10_donor15_p5"):
        rows = build_variant_rows(
            variant=variant,
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by_sid,
            chemassign=chem_by_sid,
            donor=donor_by_sid,
            rows_lt7=rows_lt7,
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates{suffix}.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[variant] = {"overall": summarize(rows), "rows_ge7": summarize(rows7), "rows_lt7": summarize(rowslt7)}

    best_variant = max(
        variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    exp2j_best = exp2j["best_variant"]
    exp2j_rows7 = exp2j["variants"][exp2j_best]["rows_ge7"]
    exp2b_rows7 = read_json(EXP2B_RESULT)["rows_ge7"]
    exp3_line = float(exp2j_rows7.get("skeleton_to_match_conversion@50") or 0.0) + 0.02
    min_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and float(best_rows7.get("match@50") or 0.0) >= 0.17931372549019609
        and float(best_rows7.get("formula_consistency") or 0.0) >= 0.95
        and float(best_rows7.get("sg_consistency") or 0.0) >= 0.90
        and float(best_rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28)
    exp3_line_passed = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= exp3_line)

    exp2j_best_path = Path(exp2j["artifacts"]["selected_variants"][exp2j_best])
    exp2j_sets = sample_sets(list(read_jsonl_iter(exp2j_best_path)))
    donor_sets = sample_sets(ranked_donor)
    all_rows7_samples = exp2j_sets["samples"] | donor_sets["samples"]
    union_match = exp2j_sets["match"] | donor_sets["match"]
    union_skel = exp2j_sets["skeleton"] | donor_sets["skeleton"]
    union_skelmatch = exp2j_sets["skelmatch"] | donor_sets["skelmatch"]
    oracle = {
        "donor_match_samples": len(donor_sets["match"]),
        "donor_skelmatch_samples": len(donor_sets["skelmatch"]),
        "new_match_samples_vs_exp2j": len(donor_sets["match"] - exp2j_sets["match"]),
        "new_skelmatch_samples_vs_exp2j": len(donor_sets["skelmatch"] - exp2j_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(all_rows7_samples)),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    conversion_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2j_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(exp2j_rows7.get("match@50") or 0.0)
    if target_gate:
        verdict = "pass_exp2_target_gate"
        reason = "Correlated donor geometry reaches the Exp2 target conversion range."
        next_step = "Proceed to Exp3 local optimizer relative to Exp2k; no official until later gates pass."
    elif exp3_line_passed:
        verdict = "pass_exp3_line_but_not_exp2_target"
        reason = "Correlated donor geometry clears the +2pp conversion line over Exp2j but not the 28% target."
        next_step = "Retest Exp3 local optimizer relative to Exp2k; do not enter Exp4/5/official yet."
    elif conversion_delta > 0.002 or match_delta > 0.002:
        verdict = "partial_gain_below_gate"
        reason = "Correlated donor geometry produces a small validation lift but not the required gate."
        next_step = "Analyze donor oracle and consider a learned donor/geometry predictor; do not enter Exp3/official from this variant."
    elif min_gate:
        verdict = "fail_no_conversion_lift"
        reason = "Correlated donor geometry preserves the minimum structural gate but does not improve Exp2j conversion."
        next_step = "Stop prototype donor expansion; next repair must learn continuous aligned free-parameter residuals or upgrade skeleton proposer."
    else:
        verdict = "fail_structural_gate"
        reason = "Correlated donor geometry does not maintain the minimum validation/structure gate."
        next_step = "Do not expand this donor posterior; diagnose structural breakage and return to data alignment."

    result = {
        "experiment": "opentry_14_exp2k_correlated_donor_geometry_posterior",
        "result_filename": result_filename,
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "chemical_assignment_plus_correlated_train_donor_geometry_bundle",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "assignment_beam_width": int(args.assignment_beam_width),
            "donor_limit": int(args.donor_limit),
            "include_source_donor": bool(args.include_source_donor),
            "max_target_atoms": int(args.max_target_atoms),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "single-thread worker processes; keep individual Python workers under the user 200% CPU cap",
        },
        "train_priors": priors["data_scale"],
        "reference_index": {
            "train_records": len(train_struct_list),
            "by_sg_keys": len(indexes.get("by_sg", {})),
            "by_sg_atom_count_keys": len(indexes.get("by_sg_atom_count", {})),
            "by_skeleton_keys": len(indexes.get("by_skeleton", {})),
        },
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "input_exp2j_chemical_records": len(chem_rows),
            "generated_donor_candidates": len(generation_meta),
            "evaluated_donor_candidates": len(ranked_donor),
            "skipped_target_atom_count_gt_limit": int(failures.get("skipped_target_atom_count_gt_limit", 0)),
            "selected_variant_paths": selected_paths,
        },
        "donor_diagnostics": {
            "donor_tiers": dict(donor_tiers),
            "fallback_rows_histogram": dict(fallback_rows),
            "donor_exact_row_signature_match_count": donor_exact_sig,
            "donor_orbit_row_signature_match_count": donor_orbit_sig,
        },
        "baseline_exp2b": {"result_path": str(EXP2B_RESULT), "rows_ge7": exp2b_rows7},
        "baseline_exp2j": {"result_path": str(EXP2J_RESULT), "best_variant": exp2j_best, "rows_ge7": exp2j_rows7, "overall": exp2j["variants"][exp2j_best]["overall"]},
        "mapping_failures": dict(failures),
        "donor_only": {"rows_ge7": summarize([r for r in ranked_donor if int(r.get("row_count") or 0) >= 7]) if ranked_donor else {}},
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "exp3_line_passed": exp3_line_passed,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2j": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2j": conversion_delta,
            "rows_ge7_conversion50_delta_vs_exp3_line": float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - exp3_line,
            "minimum_standard": {"rows_ge7_conversion50": 0.223, "rows_ge7_match50_allowed_lower_bound": 0.17931372549019609, "target_rows_ge7_conversion50": 0.28, "exp3_required_conversion50": exp3_line},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "generated_meta": str(ARTIFACT_DIR / f"generated_correlated_donor_meta{suffix}.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / f"evaluated_correlated_donor_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result_filename, result)
    if not args.skip_report:
        body = report_body(result)
        marker = "<!-- OPENTRY14_EXP2K_CORRELATED_DONOR_GEOMETRY_POSTERIOR -->"
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result_filename), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
