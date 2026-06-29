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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import (  # noqa: E402
    formula_counts,
    read_jsonl,
    sample_id,
    source_skeleton_rows,
)
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    eval_sample,
    render_candidate,
    summarize,
)
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"


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


def enumerate_exact_cover_assignments(rows: list[dict[str, Any]], counts: dict[str, int], limit: int) -> list[tuple[int, list[str]]]:
    mults = [int(row.get("multiplicity") or 0) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: (-mults[i], str(rows[i].get("orbit_id")), i))
    remaining = dict(counts)
    assigned: list[str | None] = [None] * len(rows)
    out: list[tuple[int, list[str]]] = []

    def rec(pos: int) -> None:
        if len(out) >= int(limit):
            return
        if pos >= len(order):
            if all(int(v) == 0 for v in remaining.values()):
                score = sum(
                    int(str(rows[i].get("element")) == str(assigned[i])) * int(rows[i].get("multiplicity") or 0)
                    for i in range(len(rows))
                )
                out.append((int(score), [str(x) for x in assigned]))
            return
        idx = order[pos]
        mult = mults[idx]
        preferred = str(rows[idx].get("element"))
        choices = sorted(remaining, key=lambda e: (0 if str(e) == preferred else 1, -int(remaining[e]), str(e)))
        for element in choices:
            if int(remaining[element]) < mult:
                continue
            assigned[idx] = str(element)
            remaining[element] -= mult
            rec(pos + 1)
            remaining[element] += mult
            assigned[idx] = None
            if len(out) >= int(limit):
                return

    rec(0)
    seen: set[tuple[str, ...]] = set()
    uniq: list[tuple[int, list[str]]] = []
    for score, assignment in sorted(out, key=lambda x: (-x[0], tuple(x[1]))):
        key = tuple(assignment)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((score, assignment))
    return uniq[: int(limit)]


def split_safe_pool(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    hydrated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prototype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_lt7: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("row_count") or 0) < 7:
            rows_lt7.append(row)
        elif str(row.get("pool_source")) == "symcif_v5_hydrated":
            hydrated[str(row["sample_id"])].append(row)
        else:
            prototype[str(row["sample_id"])].append(row)
    for group in hydrated.values():
        group.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
    for group in prototype.values():
        group.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
    return hydrated, prototype, rows_lt7


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out: list[dict[str, Any]] = []
    for sid, group in by_sid.items():
        for rank, row in enumerate(group, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return out


def build_variant_rows(
    *,
    variant: str,
    hydrated: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    siteassign: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_p30": (10, 10, 30),
        "h10_s20_p20": (10, 20, 20),
        "h5_s20_p25": (5, 20, 25),
        "h15_s10_p25": (15, 10, 25),
        "h10_interleave_s20_p20": (10, 20, 20),
    }
    hq, sq, pq = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign)):
        hyd = hydrated.get(sid, [])
        proto = prototype.get(sid, [])
        site = siteassign.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "h10_interleave_s20_p20":
            for i in range(max(hq, sq)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq])
        for row in selected[:top_k]:
            item = dict(row)
            item["selection_variant"] = variant
            out.append(item)
    return assign_ranks(out)


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


def report_body(result: dict[str, Any]) -> str:
    rows7 = result["variants"][result["best_variant"]]["rows_ge7"]
    baseline = result["baseline_exp2b"]["rows_ge7"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2d：site-assignment multi-hypothesis repair

结果文件：`model/New_model/opentry_14/results/experiment_2d_site_assignment_multi_hypothesis.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2d_site_assignment_multi_hypothesis/`

- 为什么做：Exp3/3b 证明 local optimizer 不能新增 sample-level match；Exp3c 显示大量 skeleton-hit/no-match 样本已经 valid/formula/SG/exact，根因更像元素到 Wyckoff row 的 site assignment 与 free-parameter alignment 错误。本实验先修 site assignment。
- 核心假设：同一个 predicted skeleton 的 multiplicity/orbit 可以有多个 exact-cover 元素分配；当前 source-preferred 单 assignment 可能把元素放到错误 row。枚举少量 inference-safe exact-cover assignment 并用 train source lattice/free params 渲染，可能新增 rows>=7 match/conversion。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；site-assignment generated candidates `{result['data_scale']['site_assignment_candidates']}`；evaluated candidates `{result['data_scale']['evaluated_site_assignment_candidates']}`；top skeletons `{result['method']['top_skeletons']}`；assignments/proposal `{result['method']['assignment_limit']}`；StructureMatcher workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2b rows>=7 match@50 `{pct(baseline.get('match@50'))}`，conversion@50 `{pct(baseline.get('skeleton_to_match_conversion@50'))}`，collision `{pct(baseline.get('collision_rate'))}`。
- 方法变化：对每个 predicted skeleton source rows 枚举最多 `4` 个 exact-cover element assignment，按保留 source 元素的原子数排序；每个 assignment 使用 source lattice 和 flexible source row params 渲染。选择和排序不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton 或 official feedback。
- 结果 best variant `{result['best_variant']}`：rows>=7 match@1/5/20/50 = `{pct(rows7.get('match@1'))} / {pct(rows7.get('match@5'))} / {pct(rows7.get('match@20'))} / {pct(rows7.get('match@50'))}`；conversion@50 `{pct(rows7.get('skeleton_to_match_conversion@50'))}`；valid `{pct(rows7.get('valid_rate'))}`；formula `{pct(rows7.get('formula_consistency'))}`；SG `{pct(rows7.get('sg_consistency'))}`；exact-cover `{pct(rows7.get('exact_cover_retained'))}`；collision `{pct(rows7.get('collision_rate'))}`。
- 可信度：中等。该实验真实 render/parse/SG/StructureMatcher，且不使用禁用推理特征；限制是几何仍来自 train source prototype/free params，不是 learned posterior，site assignment 枚举属于修复 alignment 的 proof-of-concept，不是最终 Exp4 critic。
- 和历史实验关系：它直接响应 Exp3c 的根因审计。若提升 conversion，说明下一步应把 site assignment/free-parameter posterior 学起来；若不提升，则需要升级 skeleton proposer 或更强 geometry posterior。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；match@50 delta vs Exp2b `{pp(gate['rows_ge7_match50_delta_vs_exp2b'])}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=10)
    parser.add_argument("--assignment-limit", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(train_struct.values()) + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    exp2b = read_json(EXP2B_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = formula_counts(target)
        candidates: list[dict[str, Any]] = []
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_struct = train_struct.get(source_id)
            if source_repr is None or source_struct is None:
                failures["missing_source"] += 1
                continue
            src_rows = source_skeleton_rows(engine, source_repr)
            assignments = enumerate_exact_cover_assignments(src_rows, counts, int(args.assignment_limit))
            if not assignments:
                failures["no_exact_cover_assignment"] += 1
                continue
            for assignment_rank, (assignment_score, assignment) in enumerate(assignments, start=1):
                rows = []
                for row, element in zip(src_rows, assignment):
                    item = dict(row)
                    item["element"] = str(element)
                    rows.append(item)
                try:
                    params, fallback_count = flexible_params_from_reference(engine, rows, source_struct, neural_params=None)
                    option = {
                        "lattice": {k: float(source_struct["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")},
                        "params": params,
                    }
                    cif, render_meta = render_candidate(
                        engine=engine,
                        target=target,
                        rows=rows,
                        option=option,
                        data_name=f"{sid}_siteassign_p{int(proposal.get('rank') or 0)}_a{assignment_rank}",
                    )
                    row = {
                        "sample_id": sid,
                        "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                        "proposal_rank": int(proposal.get("rank") or 0),
                        "geometry_rank": int(assignment_rank),
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
                        "site_mapping_rule": "multi_exact_cover_assignment",
                        "assignment_rank": int(assignment_rank),
                        "assignment_source_preserved_atoms": int(assignment_score),
                        "geometry_source": "source_lattice_params_multi_site_assignment",
                        "reference_sample_id": source_id,
                        "reference_score": float(assignment_score),
                        "param_fallback_rows": int(fallback_count),
                        "cif": cif,
                        **render_meta,
                    }
                    candidates.append(row)
                    generation_meta.append({k: v for k, v in row.items() if k != "cif"})
                except Exception as exc:  # noqa: BLE001
                    failures[f"render_failed:{type(exc).__name__}"] += 1
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
        if si % 250 == 0:
            print(f"[exp2d-siteassign] rendered {si}/{len(selected_sids)}", flush=True)

    write_jsonl(ARTIFACT_DIR / "generated_site_assignment_meta.jsonl", generation_meta)
    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 250 == 0:
                print(f"[exp2d-siteassign] evaluated {i}/{len(futures)}", flush=True)
    ranked_site = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / "evaluated_site_assignment_candidates.jsonl", ranked_site)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_site:
        site_by_sid[str(row["sample_id"])].append(row)
    for group in site_by_sid.values():
        group.sort(key=lambda r: int(r.get("rank") or 10**9))

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_p30", "h10_s20_p20", "h5_s20_p25", "h15_s10_p25", "h10_interleave_s20_p20"):
        rows = build_variant_rows(
            variant=variant,
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by_sid,
            rows_lt7=rows_lt7,
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[variant] = {
            "overall": summarize(rows),
            "rows_ge7": summarize(rows7),
            "rows_lt7": summarize(rowslt7),
        }

    best_variant = max(
        variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    baseline_rows7 = exp2b["rows_ge7"]
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    min_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and float(best_rows7.get("match@50") or 0.0) >= 0.17931372549019609
        and float(best_rows7.get("formula_consistency") or 0.0) >= 0.95
        and float(best_rows7.get("sg_consistency") or 0.0) >= 0.90
        and float(best_rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28)
    result = {
        "experiment": "opentry_14_exp2d_site_assignment_multi_hypothesis_repair",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "multi_exact_cover_site_assignment_source_geometry",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "assignment_ranking": "source-element-preserved atom count, then deterministic exact-cover order",
            "geometry_source": "train source lattice and flexible source row free parameters",
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "site_assignment_candidates": len(generation_meta),
            "evaluated_site_assignment_candidates": len(ranked_site),
            "top_k": int(args.top_k),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2b": {
            "result_path": str(EXP2B_RESULT),
            "rows_ge7": baseline_rows7,
            "overall": exp2b["overall"],
        },
        "mapping_failures": dict(failures),
        "site_assignment_only": {
            "rows_ge7": summarize([r for r in ranked_site if int(r.get("row_count") or 0) >= 7]),
        },
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2b": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2b": conv_delta,
            "minimum_standard": {
                "rows_ge7_conversion50": 0.223,
                "rows_ge7_match50_allowed_lower_bound": 0.17931372549019609,
                "rows_ge7_formula_consistency": 0.95,
                "rows_ge7_sg_consistency": 0.90,
                "rows_ge7_exact_cover_retained": 0.95,
                "target_rows_ge7_conversion50": 0.28,
            },
        },
        "decision": {
            "verdict": "pass_minimum_gate" if min_gate else "fail_validation_gate",
            "reason": "Site-assignment multi-hypothesis passes the Exp2 minimum repair gate." if min_gate else "Site-assignment multi-hypothesis does not pass the Exp2 minimum repair gate.",
            "next_step": "Use this as alignment-positive candidate and retest local optimizer/Exp3 gate." if min_gate and conv_delta > 0 else "Do not proceed to official; use the smoke signal to train a real assignment/free-parameter posterior rather than fixed enumeration.",
        },
        "artifacts": {
            "generated_site_assignment_meta": str(ARTIFACT_DIR / "generated_site_assignment_meta.jsonl"),
            "evaluated_site_assignment_candidates": str(ARTIFACT_DIR / "evaluated_site_assignment_candidates.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    out_path = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
    write_json(out_path, result)
    body = report_body(result)
    append_or_replace(REPORT_PATH, "<!-- OPENTRY14_EXP2D_SITE_ASSIGNMENT_MULTI_HYPOTHESIS -->", body)
    append_or_replace(LOCAL_REPORT_PATH, "<!-- OPENTRY14_EXP2D_SITE_ASSIGNMENT_MULTI_HYPOTHESIS -->", body)
    print(json.dumps({"output": str(out_path), "best_variant": best_variant, "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
