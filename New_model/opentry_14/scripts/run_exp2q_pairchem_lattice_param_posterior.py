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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2q_pairchem_lattice_param_posterior"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import formula_counts, read_jsonl, sample_id, source_skeleton_rows  # noqa: E402
from run_exp2h_train_param_prior_geometry_posterior import build_lattice_index, build_param_bank, params_from_pool  # noqa: E402
from run_exp2j_chemical_site_order_assignment import (  # noqa: E402
    EXP2B_EVAL,
    EXP2B_RESULT,
    EXP2D_RESULT,
    EXP2D_SITE_EVAL,
    EXP3_PROPOSALS,
    LOOKUP,
    STRUCTURED_TRAIN,
    STRUCTURED_VAL,
    TRAIN_REPR,
    VAL_REPR,
    append_or_replace,
    build_train_priors,
    build_variant_rows,
    chemical_exact_cover_assignments,
    pct,
    pp,
    read_json,
    read_jsonl_iter,
    ratio,
    write_json,
    write_jsonl,
)
from run_exp2n_pairwise_local_chemistry_assignment_posterior import build_pair_priors, score_rendered_cif  # noqa: E402
from run_exp2f_permutation_aware_alignment import by_sid_ranked, sample_sets, split_safe_pool  # noqa: E402
from run_exp4_rows_ge7_multi_geometry_proposal import assign_structural_ranks, eval_sample, render_candidate, summarize  # noqa: E402
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


EXP2O_RESULT = RESULT_DIR / "experiment_2o_expanded_pairwise_local_chemistry_assignment.json"
G: dict[str, Any] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def lattice_from_record(record: dict[str, Any]) -> dict[str, float]:
    return {k: float(record["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}


def render_sid(sid: str) -> dict[str, Any]:
    engine: OrbitEngine = G["engine"]
    args = G["args"]
    target = G["val"][sid]
    target_repr = G["val_repr"][sid]
    counts = {str(k): int(v) for k, v in formula_counts(target).items()}
    target_atom_count = int(sum(counts.values()))
    sg = int(target["sg"])
    candidates: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    if target_atom_count > int(args["max_target_atoms"]):
        failures["skipped_target_atom_count_gt_limit"] += 1
        return {"payload": {"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": sg, "candidates": candidates}, "meta": meta, "failures": dict(failures)}

    sg_atom_lattice = G["lattice_index"]["sg_atom_median"].get((sg, target_atom_count))
    sg_lattice = G["lattice_index"]["sg_median"].get(sg) or G["lattice_index"]["global_median"]
    lattice_variants = [("source", None), ("sg_median", sg_lattice)]
    if sg_atom_lattice is not None:
        lattice_variants.append(("sg_atom_median", sg_atom_lattice))
    param_variants = ["source", "blend_source_exact_mean", "exact_circular_mean", "exact_top1"][: int(args["param_variants"])]

    for proposal in G["proposals"][sid].get("proposals", [])[: int(args["top_skeletons"])]:
        source_id = str(proposal.get("source_sample_id") or "")
        source_repr = G["train_repr"].get(source_id)
        source_struct = G["train_struct"].get(source_id)
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
            sg,
            G["train_priors"],
            limit=int(args["assignment_prelimit"]),
            beam_width=int(args["assignment_beam_width"]),
        )
        source_lattice = lattice_from_record(source_struct)
        for assignment_rank, (assignment_score, assignment, assignment_meta) in enumerate(assignments, start=1):
            assigned_rows: list[dict[str, Any]] = []
            for row, element in zip(original_rows, assignment):
                item = dict(row)
                item["source_element"] = str(row.get("element"))
                item["element"] = str(element)
                assigned_rows.append(item)
            try:
                source_params, source_fallback = flexible_params_from_reference(engine, assigned_rows, source_struct, neural_params=None)
            except Exception:
                source_params, source_fallback = {}, len(assigned_rows)
            rendered_for_assignment: list[dict[str, Any]] = []
            for p_rank, p_variant in enumerate(param_variants, start=1):
                if p_variant == "source":
                    params = source_params
                    fallback_count = source_fallback
                    pstats = {"source_rows": len(assigned_rows)}
                else:
                    params, fallback_count, pstats = params_from_pool(
                        engine=engine,
                        rows=assigned_rows,
                        sg=sg,
                        target_counts=counts,
                        bank=G["param_bank"],
                        variant=p_variant,
                        source_params=source_params,
                    )
                for l_rank, (l_variant, lattice_prior) in enumerate(lattice_variants, start=1):
                    lattice = source_lattice if l_variant == "source" else lattice_prior
                    try:
                        cif, render_meta = render_candidate(
                            engine=engine,
                            target=target,
                            rows=assigned_rows,
                            option={"lattice": lattice, "params": params},
                            data_name=f"{sid}_exp2q_p{int(proposal.get('rank') or 0)}_a{assignment_rank}_pv{p_rank}_lv{l_rank}",
                        )
                        local = score_rendered_cif(cif, G["pair_priors"], float(args["pair_cutoff"]))
                        combined = (
                            float(args["assignment_score_weight"]) * float(assignment_score)
                            + float(args["local_pair_weight"]) * float(local["local_pair_score"])
                            - 0.03 * float(fallback_count)
                            - 0.02 * float(l_rank - 1)
                        )
                        row = {
                            "sample_id": sid,
                            "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                            "proposal_rank": int(proposal.get("rank") or 0),
                            "assignment_rank": int(assignment_rank),
                            "assignment_score": float(assignment_score),
                            "assignment_source_preserved_atoms": int(assignment_meta.get("source_preserved_atoms") or 0),
                            "posterior_param_rank": int(p_rank),
                            "posterior_lattice_rank": int(l_rank),
                            "posterior_param_variant": p_variant,
                            "posterior_lattice_variant": l_variant,
                            "local_pair_assignment_score": float(combined),
                            "row_count": int(target_repr.get("row_count") or 0),
                            "sg": sg,
                            "formula_counts": counts,
                            "target_atom_count": target_atom_count,
                            "source_sample_id": source_id,
                            "proposal_source": str(proposal.get("source") or ""),
                            "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                            "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                            "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                            "candidate_row_count": len(assigned_rows),
                            "site_mapping_rule": "pairchem_exact_cover_plus_train_lattice_param_posterior",
                            "geometry_source": "pairchem_lattice_param_posterior",
                            "reference_sample_id": source_id,
                            "reference_score": float(combined),
                            "param_fallback_rows": int(fallback_count),
                            "param_source_stats": pstats,
                            "cif": cif,
                            **local,
                            **render_meta,
                        }
                        rendered_for_assignment.append(row)
                    except Exception as exc:  # noqa: BLE001
                        failures[f"render_failed:{type(exc).__name__}"] += 1
            rendered_for_assignment.sort(key=lambda r: (-float(r.get("local_pair_assignment_score") or -1.0e9), int(r.get("posterior_param_rank") or 999), int(r.get("posterior_lattice_rank") or 999)))
            for row in rendered_for_assignment[: int(args["variant_limit_per_assignment"])]:
                item = dict(row)
                item["geometry_rank"] = len(candidates) + 1
                item["raw_generation_order"] = len(candidates) + 1
                candidates.append(item)
                meta.append({k: v for k, v in item.items() if k != "cif"})
    return {"payload": {"sample_id": sid, "target_cif_path": str(target["source_path"]), "formula_counts": counts, "target_atom_count": target_atom_count, "sg": sg, "candidates": candidates}, "meta": meta, "failures": dict(failures)}


def report_body(result: dict[str, Any]) -> str:
    best = result["variants"][result["best_variant"]]["rows_ge7"]
    base = result["baseline_exp2o"]["rows_ge7"]
    gate = result["gate"]
    oracle = result["oracle_diagnostic"]
    return f"""## opentry_14 实验 2q：pair-chem lattice/free-parameter posterior

结果文件：`model/New_model/opentry_14/results/{result['result_filename']}`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2q_pairchem_lattice_param_posterior/`

- 为什么做：Exp2p 显示 Exp2o valid skeleton-hit no-match 中 large lattice scale mismatch `42.327%`、anonymous loose-all `32.378%`；Exp3o 证明 local optimizer 不新增 match。因此本实验在 Exp2o pair-chem assignment 上同时引入 train-only lattice prior 和 row free-parameter prior。
- 核心假设：如果剩余瓶颈是 assignment-aware lattice/free-parameter alignment，则对同一 pair-chem assignment 生成 source / SG median / SG+atom median lattice 与 source / train exact mean / blend / top 参数组合，应新增 sample-level match 或提高 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；generated posterior candidates `{result['data_scale']['generated_candidates']}`；evaluated candidates `{result['data_scale']['evaluated_candidates']}`；train pair records `{result['pair_priors']['train_records_requested']}`；param-bank usable rows `{result['param_bank']['usable_param_rows']}`；workers gen/eval/prior `{result['cpu_policy']['generation_workers']}` / `{result['cpu_policy']['eval_workers']}` / `{result['cpu_policy']['prior_workers']}`。
- baseline：Exp2o best `{result['baseline_exp2o']['best_variant']}` rows>=7 match@50 `{pct(base.get('match@50'))}`、conversion `{pct(base.get('skeleton_to_match_conversion@50'))}`、collision `{pct(base.get('collision_rate'))}`。
- 方法变化：候选生成只用 train split lattice/free-param priors、元素 pair-distance priors、source predicted skeleton/assignment；不使用 match/RMSD/StructureMatcher label、GT-WA、GT-skeleton、test true CIF 或 official feedback。该实验不是 lattice-only：每个候选同时绑定 assignment、lattice variant、row-param variant，并重新 render/eval。
- 结果 best variant `{result['best_variant']}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：posterior-only match/skelmatch samples `{oracle['posterior_match_samples']}` / `{oracle['posterior_skelmatch_samples']}`；相对 Exp2o 新增 match/skelmatch `{oracle['new_match_samples_vs_exp2o']}` / `{oracle['new_skelmatch_samples_vs_exp2o']}`；union upper bound match@50 `{pct(oracle['union_match50_upper_bound'])}`、conversion `{pct(oracle['union_conversion50_upper_bound'])}`。
- 可信度：中等。全量 validation 真实 render/parse/SG/StructureMatcher；限制是 train prior 仍是 heuristic posterior，尚不是端到端 learned continuous model。
- 和历史实验关系：直接回应 Exp2p 的 lattice+assignment 混合归因；相对 Exp2h 增加 Exp2o pair-chem assignment，相对 Exp2o 增加 lattice/free-param posterior。
- gate 判定：passed=`{gate['passed']}`；target_passed=`{gate['target_passed']}`；conversion delta vs Exp2o `{pp(gate['rows_ge7_conversion50_delta_vs_exp2o'])}`；match delta vs Exp2o `{pp(gate['rows_ge7_match50_delta_vs_exp2o'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=10)
    parser.add_argument("--assignment-prelimit", type=int, default=64)
    parser.add_argument("--assignment-beam-width", type=int, default=512)
    parser.add_argument("--variant-limit-per-assignment", type=int, default=3)
    parser.add_argument("--param-variants", type=int, default=4)
    parser.add_argument("--max-target-atoms", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--generation-workers", type=int, default=128)
    parser.add_argument("--eval-workers", type=int, default=160)
    parser.add_argument("--prior-workers", type=int, default=96)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--max-train-structures", type=int, default=8000)
    parser.add_argument("--train-pair-max-sites", type=int, default=120)
    parser.add_argument("--max-pairs-per-structure", type=int, default=3000)
    parser.add_argument("--max-values-per-pair", type=int, default=6000)
    parser.add_argument("--min-pair-count", type=int, default=8)
    parser.add_argument("--min-category-count", type=int, default=25)
    parser.add_argument("--pair-cutoff", type=float, default=4.2)
    parser.add_argument("--assignment-score-weight", type=float, default=0.02)
    parser.add_argument("--local-pair-weight", type=float, default=4.0)
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    train_struct_list = list(train_struct.values())
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in train_struct_list + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    pair_priors = build_pair_priors(train_struct_list, args)
    train_priors = build_train_priors(train_struct_list)
    param_bank = build_param_bank(engine, train_struct_list)
    lattice_index = build_lattice_index(train_struct_list)

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    G.update(
        {
            "args": {
                "top_skeletons": int(args.top_skeletons),
                "assignment_prelimit": int(args.assignment_prelimit),
                "assignment_beam_width": int(args.assignment_beam_width),
                "variant_limit_per_assignment": int(args.variant_limit_per_assignment),
                "param_variants": int(args.param_variants),
                "max_target_atoms": int(args.max_target_atoms),
                "pair_cutoff": float(args.pair_cutoff),
                "assignment_score_weight": float(args.assignment_score_weight),
                "local_pair_weight": float(args.local_pair_weight),
            },
            "engine": engine,
            "pair_priors": pair_priors,
            "train_priors": train_priors,
            "param_bank": param_bank,
            "lattice_index": lattice_index,
            "train_repr": train_repr,
            "train_struct": train_struct,
            "val": val,
            "val_repr": val_repr,
            "proposals": proposals,
        }
    )

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.generation_workers))) as pool:
        futures = [pool.submit(render_sid, sid) for sid in selected_sids]
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            payloads.append(res["payload"])
            generation_meta.extend(res["meta"])
            failures.update(res["failures"])
            if i % 200 == 0:
                print(f"[exp2q] rendered {i}/{len(futures)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    result_filename = f"experiment_2q_pairchem_lattice_param_posterior{suffix}.json"
    write_jsonl(ARTIFACT_DIR / f"generated_pairchem_lattice_param_meta{suffix}.jsonl", generation_meta)
    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.eval_workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2q] evaluated {i}/{len(futures)}", flush=True)
    ranked = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_pairchem_lattice_param_candidates{suffix}.jsonl", ranked)

    exp2o = read_json(EXP2O_RESULT)
    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))
    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    post_by_sid = by_sid_ranked(ranked)

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_chem25_p5", "h10_s5_chem30_p5", "chem30_h10_s5_p5", "h10_interleave_s10_chem25_p5"):
        rows = build_variant_rows(variant=variant, hydrated=hydrated, prototype=prototype, siteassign=site_by_sid, chemassign=post_by_sid, rows_lt7=rows_lt7, top_k=int(args.top_k))
        path = ARTIFACT_DIR / f"selected_{variant}_candidates{suffix}.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        variants[variant] = {
            "overall": summarize(rows),
            "rows_ge7": summarize([r for r in rows if int(r.get("row_count") or 0) >= 7]),
            "rows_lt7": summarize([r for r in rows if int(r.get("row_count") or 0) < 7]),
        }

    best_variant = max(
        variants,
        key=lambda n: (
            float(variants[n]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[n]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[n]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    exp2o_rows7 = exp2o["variants"][exp2o["best_variant"]]["rows_ge7"]
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2o_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(exp2o_rows7.get("match@50") or 0.0)
    target_gate = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
    passed = bool(target_gate and match_delta >= 0.0 and conv_delta >= 0.0)

    base_path = Path(exp2o["artifacts"]["selected_variants"][exp2o["best_variant"]])
    base_sets = sample_sets([r for r in read_jsonl_iter(base_path) if int(r.get("row_count") or 0) >= 7])
    post_sets = sample_sets([r for r in ranked if int(r.get("row_count") or 0) >= 7])
    union_match = base_sets["match"] | post_sets["match"]
    union_skel = base_sets["skeleton"] | post_sets["skeleton"]
    union_skelmatch = base_sets["skelmatch"] | post_sets["skelmatch"]
    oracle = {
        "posterior_match_samples": len(post_sets["match"]),
        "posterior_skelmatch_samples": len(post_sets["skelmatch"]),
        "new_match_samples_vs_exp2o": len(post_sets["match"] - base_sets["match"]),
        "new_skelmatch_samples_vs_exp2o": len(post_sets["skelmatch"] - base_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(base_sets["samples"] | post_sets["samples"])),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    if passed and conv_delta >= 0.02:
        verdict = "pass_plus2pp_gate"
        reason = "Joint lattice/free-parameter posterior improves Exp2o by at least +2pp conversion without match loss."
        next_step = "Retest Exp3 local optimizer only after this new Exp2 best; no official/full validation yet."
    elif passed:
        verdict = "pass_target_but_small_lift"
        reason = "Joint lattice/free-parameter posterior remains above target and does not hurt Exp2o, but lift is below +2pp."
        next_step = "Use as diagnostic; inspect oracle before expanding posterior."
    elif oracle["new_match_samples_vs_exp2o"] > 0:
        verdict = "fail_ranked_gate_has_oracle_headroom"
        reason = "Posterior creates some new sample-level matches but ranked mixture does not improve the Exp2o gate."
        next_step = "Improve generation-side posterior or inference-safe selection; do not use forbidden match labels."
    else:
        verdict = "fail_no_new_match"
        reason = "Train lattice/free-parameter posterior does not create new sample-level match beyond Exp2o."
        next_step = "Stop this heuristic posterior; move to learned assignment-aware model or skeleton/source retrieval."

    result = {
        "experiment": "opentry_14_exp2q_pairchem_lattice_param_posterior",
        "result_filename": result_filename,
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "pairchem_assignment_with_train_lattice_and_row_param_posterior",
            "top_skeletons": int(args.top_skeletons),
            "assignment_prelimit": int(args.assignment_prelimit),
            "assignment_beam_width": int(args.assignment_beam_width),
            "variant_limit_per_assignment": int(args.variant_limit_per_assignment),
            "param_variants": int(args.param_variants),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
        },
        "cpu_policy": {
            "generation_workers": int(args.generation_workers),
            "eval_workers": int(args.eval_workers),
            "prior_workers": int(args.prior_workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "pair_priors": pair_priors["data_scale"],
        "param_bank": param_bank["data_scale"],
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "generated_candidates": len(generation_meta),
            "evaluated_candidates": len(ranked),
            "mapping_failures": dict(failures),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2o": {"result_path": str(EXP2O_RESULT), "best_variant": exp2o["best_variant"], "rows_ge7": exp2o_rows7},
        "baseline_exp2b": {"result_path": str(EXP2B_RESULT), "rows_ge7": exp2b["rows_ge7"]},
        "baseline_exp2d": {"result_path": str(EXP2D_RESULT), "best_variant": exp2d["best_variant"], "rows_ge7": exp2d["variants"][exp2d["best_variant"]]["rows_ge7"]},
        "posterior_only": {"rows_ge7": summarize([r for r in ranked if int(r.get("row_count") or 0) >= 7]) if ranked else {}},
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "passed": passed,
            "target_passed": target_gate,
            "rows_ge7_match50_delta_vs_exp2o": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2o": conv_delta,
            "minimum_standard": {"target_rows_ge7_conversion50": 0.28, "match_not_worse_vs_exp2o": True, "conversion_not_worse_vs_exp2o": True},
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "generated_meta": str(ARTIFACT_DIR / f"generated_pairchem_lattice_param_meta{suffix}.jsonl"),
            "evaluated_candidates": str(ARTIFACT_DIR / f"evaluated_pairchem_lattice_param_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / result_filename, result)
    if not args.skip_report:
        marker = "<!-- OPENTRY14_EXP2Q_PAIRCHEM_LATTICE_PARAM_POSTERIOR -->"
        body = report_body(result)
        append_or_replace(REPORT_PATH, marker, body)
        append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / result_filename), "best_variant": best_variant, "gate": result["gate"], "decision": result["decision"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
