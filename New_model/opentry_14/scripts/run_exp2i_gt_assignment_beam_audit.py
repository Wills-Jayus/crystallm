#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
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
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2i_gt_assignment_beam_audit"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import formula_counts, read_jsonl, sample_id, source_skeleton_rows  # noqa: E402
from run_exp2f_permutation_aware_alignment import enumerate_exact_cover_assignments  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP2G_RESULT = RESULT_DIR / "experiment_2g_candidate_headroom_audit.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


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


def target_assignment_for_rows(candidate_rows: list[dict[str, Any]], target_wa: list[dict[str, Any]]) -> list[str] | None:
    pools: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in target_wa:
        pools[(str(row.get("orbit_id") or ""), int(row.get("multiplicity") or row.get("declared_multiplicity") or 0))].append(row)
    assignment: list[str] = []
    for row in candidate_rows:
        key = (str(row.get("orbit_id") or ""), int(row.get("multiplicity") or 0))
        pool = pools.get(key) or []
        if not pool:
            return None
        target_row = pool.pop(0)
        assignment.append(str(target_row.get("element") or ""))
    return assignment


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


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found_le64"
    if rank == 1:
        return "rank1"
    if rank <= 4:
        return "rank2_4"
    if rank <= 10:
        return "rank5_10"
    if rank <= 20:
        return "rank11_20"
    if rank <= 64:
        return "rank21_64"
    return "not_found_le64"


def report_body(result: dict[str, Any]) -> str:
    audit = result["assignment_rank_audit"]
    gate = result["decision_metrics"]
    return f"""## opentry_14 实验 2i：GT-WA assignment beam 离线审计

结果文件：`model/New_model/opentry_14/results/experiment_2i_gt_assignment_beam_audit.json`
诊断 artifact：`model/New_model/opentry_14/artifacts/exp2i_gt_assignment_beam_audit/`

- 为什么做：Exp2h row-wise train 参数先验没有新增 sample-level match；Exp2g 显示 ranking 不是主瓶颈。需要判断 site-assignment beam 是否遗漏真实 element-to-row assignment，避免盲目扩大 exact-cover assignment。
- 核心假设：如果 Exp2d 的 assignment_limit=4 太窄，那么在 predicted skeleton 命中的样本中，GT-WA element assignment 应大量出现在 rank>4 或 not-found；如果 GT assignment 多数 rank<=4，则继续扩大 assignment beam 不会带来主要增益。
- 数据规模：rows>=7 validation samples `{result['data_scale']['rows_ge7_samples']}`；top skeleton proposals audited `{result['data_scale']['top_skeletons_audited']}`；skeleton-hit samples `{audit['skeleton_hit_samples']}`；GT assignment comparable samples `{audit['gt_assignment_comparable_samples']}`。
- baseline：Exp2d 使用 top skeletons `10`、assignment_limit `4`，best rows>=7 match@50 `22.036%`、conversion `25.714%`；Exp2g union all-candidate conversion `26.175%` 仍低于 Exp3 line `26.866%`。
- 方法变化：只做离线审计。对每个 rows>=7 validation sample 的 top10 predicted skeleton proposals，若 proposal skeleton key 等于 validation canonical skeleton key，则把 true WA rows 按 orbit/multiplicity 对齐到 predicted rows，计算 GT element assignment 在 deterministic exact-cover enumeration 前 64 个中的 rank。GT-WA 只作为诊断标签，不能作为推理特征。
- 结果：GT assignment rank buckets = `{audit['rank_buckets']}`；rank<=4 coverage `{pct(audit['rank_le4_rate'])}`；rank<=10 coverage `{pct(audit['rank_le10_rate'])}`；not-found<=64 rate `{pct(audit['not_found_le64_rate'])}`。
- 可信度：中等偏高。它直接比较 validation true WA 与 predicted skeleton exact-cover assignment space，不跑 StructureMatcher，不训练，不改候选；限制是只在 skeleton key 已命中的样本上可比，且 GT-WA 不能用于推理选择。
- 和历史实验关系：承接 Exp2d 的 small site-assignment gain、Exp2f 的 small permutation gain、Exp2g 的 candidate-headroom failure。它回答“继续扩大 assignment 是否有必要”这个具体分支。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    val_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(train_struct.values()) + list(val_struct.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    exp2g = read_json(EXP2G_RESULT)

    rows7_sids = [sid for sid in sorted(proposals) if sid in val_repr and sid in val_struct and int(val_repr[sid].get("row_count") or 0) >= 7]
    records: list[dict[str, Any]] = []
    sample_best: dict[str, dict[str, Any]] = {}
    failures = Counter()
    for sid in rows7_sids:
        target_repr = val_repr[sid]
        target_struct = val_struct[sid]
        target_key = str(target_repr.get("canonical_skeleton_key") or "")
        target_wa = list(target_struct.get("wa_table") or [])
        counts = {str(k): int(v) for k, v in formula_counts(target_struct).items()}
        best_rank = None
        best_proposal_rank = None
        comparable = False
        for proposal in proposals[sid].get("proposals", [])[:10]:
            if str(proposal.get("skeleton_key") or "") != target_key:
                continue
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            if source_repr is None:
                failures["missing_source_repr"] += 1
                continue
            candidate_rows = source_skeleton_rows(engine, source_repr)
            if not candidate_rows:
                failures["empty_source_rows"] += 1
                continue
            gt_assignment = target_assignment_for_rows(candidate_rows, target_wa)
            if gt_assignment is None:
                failures["gt_assignment_not_comparable"] += 1
                continue
            comparable = True
            assignments = enumerate_exact_cover_assignments(candidate_rows, counts, 64)
            rank = None
            for idx, (_, assignment) in enumerate(assignments, start=1):
                if list(assignment) == list(gt_assignment):
                    rank = idx
                    break
            rec = {
                "sample_id": sid,
                "proposal_rank": int(proposal.get("rank") or 0),
                "source_sample_id": source_id,
                "gt_assignment_rank_le64": rank,
                "rank_bucket": rank_bucket(rank),
                "candidate_rows": len(candidate_rows),
                "assignment_count_le64": len(assignments),
                "target_formula_counts": counts,
            }
            records.append(rec)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_proposal_rank = int(proposal.get("rank") or 0)
        if comparable:
            sample_best[sid] = {"sample_id": sid, "best_gt_assignment_rank_le64": best_rank, "best_proposal_rank": best_proposal_rank, "rank_bucket": rank_bucket(best_rank)}

    write_jsonl(ARTIFACT_DIR / "per_proposal_gt_assignment_rank.jsonl", records)
    write_jsonl(ARTIFACT_DIR / "per_sample_best_gt_assignment_rank.jsonl", list(sample_best.values()))

    buckets = Counter(row["rank_bucket"] for row in sample_best.values())
    n_comp = len(sample_best)
    rank_le4 = sum(1 for row in sample_best.values() if row["best_gt_assignment_rank_le64"] is not None and int(row["best_gt_assignment_rank_le64"]) <= 4)
    rank_le10 = sum(1 for row in sample_best.values() if row["best_gt_assignment_rank_le64"] is not None and int(row["best_gt_assignment_rank_le64"]) <= 10)
    rank_le20 = sum(1 for row in sample_best.values() if row["best_gt_assignment_rank_le64"] is not None and int(row["best_gt_assignment_rank_le64"]) <= 20)
    not_found = sum(1 for row in sample_best.values() if row["best_gt_assignment_rank_le64"] is None)
    skeleton_hit_samples = len({sid for sid in rows7_sids if any(str(p.get("skeleton_key") or "") == str(val_repr[sid].get("canonical_skeleton_key") or "") for p in proposals[sid].get("proposals", [])[:10])})

    if ratio(rank_le4, n_comp) is not None and float(ratio(rank_le4, n_comp) or 0.0) >= 0.85:
        verdict = "assignment_beam_not_main_bottleneck"
        reason = "Most comparable skeleton-hit samples already have the GT element assignment inside the current rank<=4 beam, so expanding assignment alone is unlikely to supply the missing geometry headroom."
        next_step = "Do not run a high-assignment beam as the next main experiment; focus on joint skeleton/geometry basin or learned correlated free-parameter model."
    elif float(ratio(rank_le10, n_comp) or 0.0) >= 0.85:
        verdict = "limited_assignment_expansion_may_be_diagnostic"
        reason = "A substantial fraction of GT assignments fall outside rank<=4 but inside rank<=10, so a controlled assignment-limit=10 run has a root-cause basis."
        next_step = "Run a separate non-overwriting Exp2j assignment-limit=10 diagnostic with fixed structural ranking; do not enter official."
    else:
        verdict = "assignment_order_or_skeleton_alignment_problem"
        reason = "GT assignments are often outside rank<=10 or not found, indicating assignment enumeration/order and skeleton alignment are deeper issues."
        next_step = "Do not blindly expand assignment; redesign skeleton/site alignment or proposer."

    result = {
        "experiment": "opentry_14_exp2i_gt_assignment_beam_audit",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "data_scale": {
            "rows_ge7_samples": len(rows7_sids),
            "top_skeletons_audited": 10,
            "per_proposal_records": len(records),
            "per_sample_comparable_records": len(sample_best),
        },
        "method": {
            "purpose": "offline GT-WA diagnostic for exact-cover assignment beam adequacy",
            "not_used_for_inference": ["GT-WA", "GT-skeleton", "match", "RMSD", "StructureMatcher label", "official feedback"],
            "assignment_enumeration_limit": 64,
        },
        "baseline_exp2g": {
            "union_conversion_any": exp2g["union_all_candidates"]["skeleton_to_match_conversion_any"],
            "union_match_any": exp2g["union_all_candidates"]["match_rate_any_fixed_denominator"],
        },
        "assignment_rank_audit": {
            "skeleton_hit_samples": skeleton_hit_samples,
            "gt_assignment_comparable_samples": n_comp,
            "rank_buckets": dict(buckets),
            "rank_le4_samples": rank_le4,
            "rank_le10_samples": rank_le10,
            "rank_le20_samples": rank_le20,
            "not_found_le64_samples": not_found,
            "rank_le4_rate": ratio(rank_le4, n_comp),
            "rank_le10_rate": ratio(rank_le10, n_comp),
            "rank_le20_rate": ratio(rank_le20, n_comp),
            "not_found_le64_rate": ratio(not_found, n_comp),
        },
        "failures": dict(failures),
        "decision_metrics": {
            "rank_le4_rate": ratio(rank_le4, n_comp),
            "rank_le10_rate": ratio(rank_le10, n_comp),
        },
        "decision": {"verdict": verdict, "reason": reason, "next_step": next_step},
        "artifacts": {
            "per_proposal_gt_assignment_rank": str(ARTIFACT_DIR / "per_proposal_gt_assignment_rank.jsonl"),
            "per_sample_best_gt_assignment_rank": str(ARTIFACT_DIR / "per_sample_best_gt_assignment_rank.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(RESULT_DIR / "experiment_2i_gt_assignment_beam_audit.json", result)
    body = report_body(result)
    marker = "<!-- OPENTRY14_EXP2I_GT_ASSIGNMENT_BEAM_AUDIT -->"
    append_or_replace(REPORT_PATH, marker, body)
    append_or_replace(LOCAL_REPORT_PATH, marker, body)
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2i_gt_assignment_beam_audit.json"), "audit": result["assignment_rank_audit"], "decision": result["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
