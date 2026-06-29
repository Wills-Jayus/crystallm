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
from itertools import combinations
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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2f_permutation_aware_alignment"
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
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
EXP2D_RESULT = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
EXP2D_SITE_EVAL = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"

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


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


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


def param_signature(row: dict[str, Any], params: dict[str, float]) -> tuple[str, tuple[str, ...], int]:
    return (
        str(row.get("orbit_id") or ""),
        tuple(sorted(str(k) for k in params.keys())),
        int(row.get("multiplicity") or 0),
    )


def normalize_mapping(mapping: dict[int, int], n: int) -> tuple[int, ...]:
    return tuple(int(mapping.get(i, i)) for i in range(n))


def grouped_indices(rows: list[dict[str, Any]], source_params: dict[int, dict[str, float]]) -> dict[tuple[str, tuple[str, ...], int], list[int]]:
    groups: dict[tuple[str, tuple[str, ...], int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        params = source_params.get(idx) or {}
        groups[param_signature(row, params)].append(idx)
    return groups


def generate_param_mappings(
    *,
    original_rows: list[dict[str, Any]],
    assigned_rows: list[dict[str, Any]],
    source_params: dict[int, dict[str, float]],
    max_mappings: int,
) -> list[tuple[str, dict[int, int]]]:
    n = len(assigned_rows)
    identity = {i: i for i in range(n)}
    out: list[tuple[str, dict[int, int]]] = [("identity_source_position", dict(identity))]
    seen = {normalize_mapping(identity, n)}
    groups = grouped_indices(assigned_rows, source_params)

    # If an element moved during exact-cover assignment, try carrying the source-row
    # parameter block with the same source element inside the same orbit/signature.
    follow = dict(identity)
    for indices in groups.values():
        unused = set(indices)
        for target_idx in indices:
            target_element = str(assigned_rows[target_idx].get("element"))
            choices = [src_idx for src_idx in sorted(unused) if str(original_rows[src_idx].get("element")) == target_element]
            if choices:
                src_idx = choices[0]
                follow[target_idx] = src_idx
                unused.remove(src_idx)
        for target_idx in indices:
            if target_idx not in follow or follow[target_idx] in set(indices) - unused:
                continue
        for target_idx in indices:
            if follow.get(target_idx, target_idx) not in indices:
                follow[target_idx] = target_idx
        for target_idx in indices:
            if follow.get(target_idx, target_idx) in unused:
                unused.discard(follow[target_idx])
        for target_idx in indices:
            if target_idx in follow and follow[target_idx] != target_idx:
                continue
            if target_idx in unused:
                follow[target_idx] = target_idx
                unused.discard(target_idx)
            elif unused:
                follow[target_idx] = sorted(unused)[0]
                unused.remove(follow[target_idx])
    key = normalize_mapping(follow, n)
    if key not in seen:
        seen.add(key)
        out.append(("source_element_following", dict(follow)))

    for sig, indices in sorted(groups.items(), key=lambda item: (str(item[0]), item[1])):
        if len(out) >= max_mappings:
            break
        if len(indices) < 2:
            continue

        rev = dict(identity)
        for target_idx, src_idx in zip(indices, reversed(indices)):
            rev[target_idx] = src_idx
        key = normalize_mapping(rev, n)
        if key not in seen:
            seen.add(key)
            out.append((f"reverse_group_{sig[0]}", rev))
            if len(out) >= max_mappings:
                break

        sorted_targets = sorted(indices, key=lambda i: (str(assigned_rows[i].get("element")), i))
        sorted_sources = sorted(indices, key=lambda i: (str(original_rows[i].get("element")), i))
        sorted_map = dict(identity)
        for target_idx, src_idx in zip(sorted_targets, sorted_sources):
            sorted_map[target_idx] = src_idx
        key = normalize_mapping(sorted_map, n)
        if key not in seen:
            seen.add(key)
            out.append((f"element_sorted_group_{sig[0]}", sorted_map))
            if len(out) >= max_mappings:
                break

        for i, j in combinations(indices, 2):
            if len(out) >= max_mappings:
                break
            swap = dict(identity)
            swap[i] = j
            swap[j] = i
            key = normalize_mapping(swap, n)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"swap_rows_{i}_{j}", swap))

    return out[: max(1, int(max_mappings))]


def apply_param_mapping(source_params: dict[int, dict[str, float]], mapping: dict[int, int]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for target_idx in sorted(source_params):
        src_idx = int(mapping.get(target_idx, target_idx))
        block = source_params.get(src_idx) or source_params.get(target_idx) or {}
        out[int(target_idx)] = {str(k): float(v) for k, v in block.items()}
    return out


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


def by_sid_ranked(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["sample_id"])].append(row)
    for group in out.values():
        group.sort(key=lambda r: int(r.get("rank") or r.get("geometry_rank") or 10**9))
    return out


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
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
    permuted: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_perm20_p10": (10, 10, 20, 10),
        "h10_s5_perm30_p5": (10, 5, 30, 5),
        "h5_s15_perm25_p5": (5, 15, 25, 5),
        "h10_interleave_s10_perm20_p10": (10, 10, 20, 10),
        "perm30_h10_s5_p5": (10, 5, 30, 5),
    }
    hq, sq, pq_perm, pq_proto = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign) | set(permuted)):
        hyd = hydrated.get(sid, [])
        site = siteassign.get(sid, [])
        perm = permuted.get(sid, [])
        proto = prototype.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "h10_interleave_s10_perm20_p10":
            for i in range(max(hq, sq, pq_perm)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
                if i < pq_perm and i < len(perm):
                    selected.append(perm[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        elif variant == "perm30_h10_s5_p5":
            selected.extend(perm[:pq_perm])
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq_proto])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(perm[:pq_perm])
            selected.extend(proto[:pq_proto])
        for row in selected[:top_k]:
            item = dict(row)
            item["selection_variant"] = variant
            out.append(item)
    return assign_ranks(out)


def sample_sets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("row_count") or 0) >= 7:
            by_sid[str(row["sample_id"])].append(row)
    samples = set(by_sid)
    match = set()
    skel = set()
    skmatch = set()
    for sid, group in by_sid.items():
        has_match = any(bool(r.get("match")) for r in group)
        has_skel = any(bool(r.get("predicted_skeleton_hit")) for r in group)
        if has_match:
            match.add(sid)
        if has_skel:
            skel.add(sid)
        if has_match and has_skel:
            skmatch.add(sid)
    return {"samples": samples, "match": match, "skeleton": skel, "skelmatch": skmatch}


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
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    exp2b = result["baseline_exp2b"]["rows_ge7"]
    exp2d = result["baseline_exp2d"]["rows_ge7"]
    gate = result["gate"]
    oracle = result["oracle_diagnostic"]
    return f"""## opentry_14 实验 2f：permutation-aware row/free-parameter alignment

结果文件：`model/New_model/opentry_14/results/experiment_2f_permutation_aware_alignment.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2f_permutation_aware_alignment/`

- 为什么做：Exp2d 证明 site assignment 有小幅正信号，但 oracle 上限不足；Exp2e residual posterior 新增 `0` 个 sample-level match。剩余可检验根因是同一 Wyckoff/orbit 内 row-level free parameters 可能跟错元素/row，固定 row-index 会把正确站位几何放到错误元素上。
- 核心假设：如果 skeleton-hit/no-match 的关键错误是相同 orbit 内参数块与元素分配错位，则在 exact-cover assignment 后，在相同 orbit/参数模式内做 element-following、element-sorted、reverse/swap 参数块排列，应新增 sample-level match，并提升 rows>=7 conversion。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；generated permutation candidates `{result['data_scale']['generated_permutation_candidates']}`；evaluated candidates `{result['data_scale']['evaluated_permutation_candidates']}`；skipped atom-count samples `{result['data_scale']['skipped_target_atom_count_gt_limit']}`；workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2b rows>=7 match@50 `{pct(exp2b.get('match@50'))}`、conversion `{pct(exp2b.get('skeleton_to_match_conversion@50'))}`；Exp2d best rows>=7 match@50 `{pct(exp2d.get('match@50'))}`、conversion `{pct(exp2d.get('skeleton_to_match_conversion@50'))}`。
- 方法变化：对每个 predicted skeleton 的 exact-cover assignment，先恢复 source lattice/free params，然后只在相同 orbit_id、multiplicity 和 free-param key signature 的 row 之间排列参数块；生成 identity、source-element-following、element-sorted、reverse、pair-swap 等 deterministic hypotheses。critic 只用 legal/formula/SG/site/exact/collision/volume/reference/proposal structural score；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- oracle 诊断：permutation-only match samples `{oracle['permutation_match_samples']}`，相对 Exp2b 新增 match/skelmatch samples `{oracle['new_match_samples_vs_exp2b']}` / `{oracle['new_skelmatch_samples_vs_exp2b']}`，相对 Exp2d best 新增 `{oracle['new_match_samples_vs_exp2d_best']}` / `{oracle['new_skelmatch_samples_vs_exp2d_best']}`；union upper bound match@50 `{pct(oracle['union_match50_upper_bound'])}`、conversion `{pct(oracle['union_conversion50_upper_bound'])}`。
- 可信度：中等。它是真实 render/parse/SG/StructureMatcher evaluation，推理不使用禁用标签；限制是 permutation 仍是 deterministic local enumeration，不是训练得到的 posterior，且为避免 StructureMatcher 长尾默认不对超大 atom-count 样本生成 permutation candidates。
- 和历史实验关系：承接 Exp3c 的 alignment 根因审计、Exp2d 的 assignment 正信号和 Exp2e residual 无余量结论；如果仍无新增 oracle match，则 rows>=7 conversion 的剩余瓶颈更可能在 predicted skeleton proposer 本身或更复杂的 permutation-aware posterior。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；match@50 delta vs Exp2b `{pp(gate['rows_ge7_match50_delta_vs_exp2b'])}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；match@50 delta vs Exp2d `{pp(gate['rows_ge7_match50_delta_vs_exp2d'])}`；conversion delta vs Exp2d `{pp(gate['rows_ge7_conversion50_delta_vs_exp2d'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=6)
    parser.add_argument("--assignment-limit", type=int, default=2)
    parser.add_argument("--max-permutations", type=int, default=6)
    parser.add_argument("--max-target-atoms", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
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
    exp2d = read_json(EXP2D_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        target_atom_count = int(sum(counts.values()))
        candidates: list[dict[str, Any]] = []
        if target_atom_count > int(args.max_target_atoms):
            failures["skipped_target_atom_count_gt_limit"] += 1
            payloads.append(
                {
                    "sample_id": sid,
                    "target_cif_path": str(target["source_path"]),
                    "formula_counts": counts,
                    "target_atom_count": target_atom_count,
                    "sg": int(target["sg"]),
                    "candidates": candidates,
                }
            )
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
            assignments = enumerate_exact_cover_assignments(original_rows, counts, int(args.assignment_limit))
            if not assignments:
                failures["no_exact_cover_assignment"] += 1
                continue
            for assignment_rank, (assignment_score, assignment) in enumerate(assignments, start=1):
                assigned_rows: list[dict[str, Any]] = []
                for row, element in zip(original_rows, assignment):
                    item = dict(row)
                    item["source_element"] = str(row.get("element"))
                    item["element"] = str(element)
                    assigned_rows.append(item)
                try:
                    source_params, fallback_count = flexible_params_from_reference(engine, assigned_rows, source_struct, neural_params=None)
                except Exception as exc:  # noqa: BLE001
                    failures[f"param_recovery_failed:{type(exc).__name__}"] += 1
                    continue
                mappings = generate_param_mappings(
                    original_rows=original_rows,
                    assigned_rows=assigned_rows,
                    source_params=source_params,
                    max_mappings=int(args.max_permutations),
                )
                source_lattice = {k: float(source_struct["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}
                for perm_rank, (variant_name, mapping) in enumerate(mappings, start=1):
                    if variant_name == "identity_source_position":
                        # Exp2d already evaluates the identity assignment; keep it only in
                        # smoke if max_permutations is one, otherwise spend budget on true permutations.
                        if int(args.max_permutations) > 1:
                            continue
                    try:
                        params = apply_param_mapping(source_params, mapping)
                        option = {"lattice": source_lattice, "params": params}
                        cif, render_meta = render_candidate(
                            engine=engine,
                            target=target,
                            rows=assigned_rows,
                            option=option,
                            data_name=f"{sid}_perm_p{int(proposal.get('rank') or 0)}_a{assignment_rank}_m{perm_rank}",
                        )
                        row = {
                            "sample_id": sid,
                            "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                            "proposal_rank": int(proposal.get("rank") or 0),
                            "assignment_rank": int(assignment_rank),
                            "assignment_source_preserved_atoms": int(assignment_score),
                            "permutation_rank": int(perm_rank),
                            "geometry_rank": len(candidates) + 1,
                            "raw_generation_order": len(candidates) + 1,
                            "row_count": int(target_repr.get("row_count") or 0),
                            "sg": int(target["sg"]),
                            "formula_counts": counts,
                            "target_atom_count": target_atom_count,
                            "source_sample_id": source_id,
                            "proposal_source": str(proposal.get("source") or ""),
                            "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                            "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                            "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                            "candidate_row_count": len(assigned_rows),
                            "site_mapping_rule": "exact_cover_assignment_plus_orbit_param_permutation",
                            "geometry_source": "permutation_aware_row_param_alignment",
                            "alignment_variant": variant_name,
                            "reference_sample_id": source_id,
                            "reference_score": float(assignment_score + max(0, 100 - perm_rank)),
                            "param_fallback_rows": int(fallback_count),
                            "permutation_mapping": {str(k): int(v) for k, v in mapping.items()},
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
                "target_atom_count": target_atom_count,
                "sg": int(target["sg"]),
                "candidates": candidates,
            }
        )
        if si % 200 == 0:
            print(f"[exp2f-permutation] rendered {si}/{len(selected_sids)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    write_jsonl(ARTIFACT_DIR / f"generated_permutation_alignment_meta{suffix}.jsonl", generation_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2f-permutation] evaluated {i}/{len(futures)}", flush=True)
    ranked_perm = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_permutation_alignment_candidates{suffix}.jsonl", ranked_perm)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    perm_by_sid = by_sid_ranked(ranked_perm)

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_perm20_p10", "h10_s5_perm30_p5", "h5_s15_perm25_p5", "h10_interleave_s10_perm20_p10", "perm30_h10_s5_p5"):
        rows = build_variant_rows(
            variant=variant,
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by_sid,
            permuted=perm_by_sid,
            rows_lt7=rows_lt7,
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates{suffix}.jsonl"
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
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    baseline_rows7 = exp2b["rows_ge7"]
    exp2d_rows7 = exp2d["variants"][exp2d["best_variant"]]["rows_ge7"]

    min_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and float(best_rows7.get("match@50") or 0.0) >= 0.17931372549019609
        and float(best_rows7.get("formula_consistency") or 0.0) >= 0.95
        and float(best_rows7.get("sg_consistency") or 0.0) >= 0.90
        and float(best_rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
        or (
            float(best_rows7.get("match@5") or 0.0) >= float(baseline_rows7.get("match@5") or 0.0) + 0.05
            and float(best_rows7.get("match@20") or 0.0) >= float(baseline_rows7.get("match@20") or 0.0) + 0.05
        )
    )
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    conv_delta_2d = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2d_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta_2d = float(best_rows7.get("match@50") or 0.0) - float(exp2d_rows7.get("match@50") or 0.0)

    base_sets = sample_sets([r for r in safe_rows if int(r.get("row_count") or 0) >= 7])
    site_best_path = Path(exp2d["artifacts"]["selected_variants"][exp2d["best_variant"]])
    site_best_sets = sample_sets(list(read_jsonl_iter(site_best_path)))
    perm_sets = sample_sets(ranked_perm)
    union_match = base_sets["match"] | perm_sets["match"] | site_best_sets["match"]
    union_skel = base_sets["skeleton"] | perm_sets["skeleton"] | site_best_sets["skeleton"]
    union_skelmatch = base_sets["skelmatch"] | perm_sets["skelmatch"] | site_best_sets["skelmatch"]
    all_rows7_samples = base_sets["samples"] | site_best_sets["samples"] | perm_sets["samples"]
    oracle = {
        "permutation_match_samples": len(perm_sets["match"]),
        "permutation_skelmatch_samples": len(perm_sets["skelmatch"]),
        "new_match_samples_vs_exp2b": len(perm_sets["match"] - base_sets["match"]),
        "new_skelmatch_samples_vs_exp2b": len(perm_sets["skelmatch"] - base_sets["skelmatch"]),
        "new_match_samples_vs_exp2d_best": len(perm_sets["match"] - site_best_sets["match"]),
        "new_skelmatch_samples_vs_exp2d_best": len(perm_sets["skelmatch"] - site_best_sets["skelmatch"]),
        "union_match50_upper_bound": ratio(len(union_match), len(all_rows7_samples)),
        "union_conversion50_upper_bound": ratio(len(union_skelmatch), len(union_skel)),
    }

    result = {
        "experiment": "opentry_14_exp2f_permutation_aware_row_free_parameter_alignment",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "orbit_signature_param_block_permutation_after_exact_cover_assignment",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "max_permutations": int(args.max_permutations),
            "max_target_atoms": int(args.max_target_atoms),
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "permutation_scope": "same orbit_id, multiplicity, and free-parameter-key signature only",
            "critic": "inference-safe structural score from legal/formula/SG/site/exact/collision/volume/reference/proposal features",
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "many single-thread worker processes; target each Python worker near 100% CPU and below user 200/300% per-core cap",
        },
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "generated_permutation_candidates": len(generation_meta),
            "evaluated_permutation_candidates": len(ranked_perm),
            "skipped_target_atom_count_gt_limit": int(failures.get("skipped_target_atom_count_gt_limit", 0)),
            "top_k": int(args.top_k),
            "selected_variant_paths": selected_paths,
        },
        "baseline_exp2b": {
            "result_path": str(EXP2B_RESULT),
            "rows_ge7": baseline_rows7,
            "overall": exp2b["overall"],
        },
        "baseline_exp2d": {
            "result_path": str(EXP2D_RESULT),
            "best_variant": exp2d["best_variant"],
            "rows_ge7": exp2d_rows7,
            "overall": exp2d["variants"][exp2d["best_variant"]]["overall"],
        },
        "mapping_failures": dict(failures),
        "permutation_only": {
            "rows_ge7": summarize([r for r in ranked_perm if int(r.get("row_count") or 0) >= 7]) if ranked_perm else {},
        },
        "oracle_diagnostic": oracle,
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2b": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2b": conv_delta,
            "rows_ge7_match50_delta_vs_exp2d": match_delta_2d,
            "rows_ge7_conversion50_delta_vs_exp2d": conv_delta_2d,
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
            "verdict": "pass_target_gate" if target_gate else ("pass_minimum_gate" if min_gate else "fail_validation_gate"),
            "reason": (
                "Permutation-aware alignment reaches the target repair gate."
                if target_gate
                else (
                    "Permutation-aware alignment remains above the Exp2 minimum gate but does not reach the target line."
                    if min_gate
                    else "Permutation-aware alignment does not pass the Exp2 minimum repair gate."
                )
            ),
            "next_step": (
                "If target gate passes, retest Exp3 local optimizer relative to this candidate; otherwise inspect oracle headroom and do not expand permutations blindly."
            ),
        },
        "artifacts": {
            "generated_permutation_alignment_meta": str(ARTIFACT_DIR / f"generated_permutation_alignment_meta{suffix}.jsonl"),
            "evaluated_permutation_alignment_candidates": str(ARTIFACT_DIR / f"evaluated_permutation_alignment_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }
    out_path = RESULT_DIR / f"experiment_2f_permutation_aware_alignment{suffix}.json"
    write_json(out_path, result)
    if not suffix:
        body = report_body(result)
        append_or_replace(REPORT_PATH, "<!-- OPENTRY14_EXP2F_PERMUTATION_AWARE_ALIGNMENT -->", body)
        append_or_replace(LOCAL_REPORT_PATH, "<!-- OPENTRY14_EXP2F_PERMUTATION_AWARE_ALIGNMENT -->", body)
    print(json.dumps({"output": str(out_path), "best_variant": best_variant, "gate": result["gate"], "oracle": oracle}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
