#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TOP_KS = (1, 5, 20, 100, 200, 700)
TARGET_SGS = (2, 65, 71, 127)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def q(values: list[int | float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p50": None, "p90": None, "max": None}
    vals = sorted(float(v) for v in values)
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p50": float(statistics.median(vals)),
        "p90": vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
        "max": vals[-1],
    }


def hit_at(candidates: list[dict[str, Any]], key: str, field: str, k: int) -> bool:
    return any(str(c.get(field)) == str(key) for c in candidates[: int(k)])


def source_rows(path: Path, split: str, source_name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("split", split)) != split:
            continue
        row = dict(row)
        row["source_name"] = source_name
        out[str(row["sample_id"])] = row
    return out


def summary_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def normalize_score(score: Any, rank: int) -> float:
    try:
        value = float(score)
    except Exception:
        value = 0.0
    return value / max(1.0, abs(value), float(rank))


def tag_candidate(candidate: dict[str, Any], source: str, rank: int) -> dict[str, Any]:
    c = dict(candidate)
    c.setdefault("source_labels", [])
    if source not in c["source_labels"]:
        c["source_labels"].append(source)
    c[f"{source}_rank"] = int(rank)
    c[f"{source}_score"] = None if candidate.get("score") is None else float(candidate.get("score", 0.0))
    return c


def merge_one(
    policy_candidates: list[dict[str, Any]],
    old_candidates: list[dict[str, Any]],
    *,
    fallback_enabled: bool,
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def add(candidate: dict[str, Any], source: str, rank: int) -> None:
        key = str(candidate.get("canonical_wa_key"))
        if key in merged:
            current = merged[key]
            if source not in current["source_labels"]:
                current["source_labels"].append(source)
            current[f"{source}_rank"] = min(int(rank), int(current.get(f"{source}_rank", rank)))
            current[f"{source}_score"] = None if candidate.get("score") is None else float(candidate.get("score", 0.0))
            return
        merged[key] = tag_candidate(candidate, source, rank)

    for rank, candidate in enumerate(policy_candidates, start=1):
        add(candidate, "policy", rank)
    for rank, candidate in enumerate(old_candidates, start=1):
        add(candidate, "old", rank)

    ranked: list[dict[str, Any]] = []
    for candidate in merged.values():
        policy_rank = int(candidate.get("policy_rank", 1_000_000))
        old_rank = int(candidate.get("old_rank", 1_000_000))
        if policy_rank < 1_000_000:
            score = 2.0 + 1.0 / float(policy_rank + 2)
            score += 0.02 * normalize_score(candidate.get("policy_score"), policy_rank)
        else:
            score = 1.0 + 1.0 / float(old_rank + 2)
            score += 0.02 * normalize_score(candidate.get("old_score"), old_rank)
        candidate["hybrid_score"] = float(score)
        candidate["hybrid_rank_reason"] = "policy_first_with_fallback_pool" if fallback_enabled else "policy_first_union"
        ranked.append(candidate)
    ranked.sort(
        key=lambda c: (
            0 if int(c.get("policy_rank", 1_000_000)) < 1_000_000 else 1,
            int(c.get("policy_rank", 1_000_000)),
            int(c.get("old_rank", 1_000_000)),
            str(c.get("canonical_wa_key")),
        )
    )
    return ranked[: int(top_k)]


def candidates_for(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    return [dict(c) for c in row.get("candidates", [])]


def fallback_flags(record: dict[str, Any], policy_summary: dict[str, Any] | None) -> dict[str, bool]:
    timeout = bool(policy_summary.get("timeout")) if policy_summary else False
    candidate_nonempty = bool(policy_summary.get("candidate_nonempty")) if policy_summary else bool(policy_summary)
    return {
        "timeout": timeout,
        "candidate_empty": not candidate_nonempty,
        "sg2": int(record["sg"]) == 2,
        "n_sites_ge6": int(record["n_sites"]) >= 6,
        "num_elements_ge4": int(record["num_elements"]) >= 4,
    }


def summarize_candidate_set(
    *,
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_name: str,
    elapsed_s: float | None = None,
    timeout: bool = False,
    fallback_enabled: bool = False,
    flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    gt_skeleton = str(record["canonical_skeleton_key"])
    gt_wa = str(record["canonical_wa_key"])
    out: dict[str, Any] = {
        "split": record["split"],
        "sample_id": record["sample_id"],
        "source": source_name,
        "formula": record["formula"],
        "formula_counts": record["formula_counts"],
        "sg": int(record["sg"]),
        "n_sites": int(record["n_sites"]),
        "num_elements": int(record["num_elements"]),
        "total_atoms": int(record["atom_count"]),
        "candidate_count": len(candidates),
        "candidate_nonempty": bool(candidates),
        "timeout": bool(timeout),
        "elapsed_s": elapsed_s,
        "fallback_enabled": bool(fallback_enabled),
        "fallback_reasons": sorted(k for k, v in (flags or {}).items() if v),
        "gt_skeleton_key": gt_skeleton,
        "gt_wa_key": gt_wa,
    }
    for k in TOP_KS:
        out[f"gt_skeleton_in_top{k}"] = hit_at(candidates, gt_skeleton, "canonical_skeleton_key", k)
        out[f"gt_wa_in_top{k}"] = hit_at(candidates, gt_wa, "canonical_wa_key", k)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "samples": len(rows),
        "candidate_nonempty_rate": sum(bool(r["candidate_nonempty"]) for r in rows) / max(1, len(rows)),
        "timeout_samples": sum(bool(r["timeout"]) for r in rows),
        "fallback_samples": sum(bool(r.get("fallback_enabled")) for r in rows),
        "candidate_count": q([r["candidate_count"] for r in rows]),
        "elapsed_s": q([r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]),
    }
    for k in TOP_KS:
        out[f"gt_skeleton_in_top{k}"] = sum(bool(r[f"gt_skeleton_in_top{k}"]) for r in rows) / max(1, len(rows))
        out[f"gt_wa_in_top{k}"] = sum(bool(r[f"gt_wa_in_top{k}"]) for r in rows) / max(1, len(rows))
    out["acceptance"] = {
        "wa_top700_ge_95": out["gt_wa_in_top700"] >= 0.95,
        "wa_top200_ge_95": out["gt_wa_in_top200"] >= 0.95,
        "wa_top100_ge_92": out["gt_wa_in_top100"] >= 0.92,
        "wa_top20_gt_83": out["gt_wa_in_top20"] > 0.83,
        "skeleton_top200_ge_97": out["gt_skeleton_in_top200"] >= 0.97,
        "timeout_le_1": out["timeout_samples"] <= 1,
    }
    return out


def group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        s = summarize(items)
        item: dict[str, Any] = {key: value, "samples": len(items)}
        for name in (
            "candidate_nonempty_rate",
            "gt_skeleton_in_top20",
            "gt_skeleton_in_top100",
            "gt_skeleton_in_top200",
            "gt_skeleton_in_top700",
            "gt_wa_in_top20",
            "gt_wa_in_top100",
            "gt_wa_in_top200",
            "gt_wa_in_top700",
            "timeout_samples",
            "fallback_samples",
        ):
            item[name] = s[name]
        item["candidate_count_mean"] = s["candidate_count"]["mean"]
        item["elapsed_s_p50"] = s["elapsed_s"]["p50"]
        item["elapsed_s_p90"] = s["elapsed_s"]["p90"]
        item["elapsed_s_max"] = s["elapsed_s"]["max"]
        out.append(item)
    return out


def target_subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_sites_ge6": summarize([r for r in rows if int(r["n_sites"]) >= 6]),
        "num_elements_ge4": summarize([r for r in rows if int(r["num_elements"]) >= 4]),
    }
    for sg in TARGET_SGS:
        out[f"sg_{sg}"] = summarize([r for r in rows if int(r["sg"]) == sg])
    return out


def source_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source"])].append(row)
    return {source: summarize(items) for source, items in sorted(by_source.items())}


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    union = summary["sources"]["union"]
    rerank = summary.get("reranker")
    lines = [
        "# SymCIF-v4 Hybrid WA Search Report",
        "",
        "## Scope",
        "",
        "- Dataset: `data/structured_symcif_v4_reextracted`.",
        "- No full match@5, no coords/lattice loss, no post-hoc repair.",
        "- Hybrid candidates are a deduplicated union of policy search and old frequency search with source labels retained.",
        "",
        "## Test Metrics",
        "",
        "| Source | WA@20 | WA@100 | WA@200 | WA@700 | Skeleton@700 | Nonempty | Timeout | Runtime p50/p90/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for source, metrics in summary["sources"].items():
        elapsed = metrics["elapsed_s"]
        lines.append(
            f"| {source} | {metrics['gt_wa_in_top20']:.4f} | {metrics['gt_wa_in_top100']:.4f} | "
            f"{metrics['gt_wa_in_top200']:.4f} | {metrics['gt_wa_in_top700']:.4f} | "
            f"{metrics['gt_skeleton_in_top700']:.4f} | "
            f"{metrics['candidate_nonempty_rate']:.4f} | {metrics['timeout_samples']} | "
            f"{elapsed['p50']}/{elapsed['p90']}/{elapsed['max']} |"
        )
    if rerank:
        lines += [
            "",
            "## Reranker",
            "",
            f"- WA@1/5/20/100/200: {rerank['wa_top1']:.4f} / {rerank['wa_top5']:.4f} / {rerank['wa_top20']:.4f} / {rerank['wa_top100']:.4f} / {rerank['wa_top200']:.4f}",
            f"- Skeleton@200: {rerank['skeleton_top200']:.4f}",
        ]
    lines += [
        "",
        "## Acceptance Snapshot",
        "",
        f"- hybrid union WA@200 >= 95%: {union['acceptance']['wa_top200_ge_95']} ({union['gt_wa_in_top200']:.4f})",
        f"- hybrid union WA@700 >= 95%: {union['acceptance']['wa_top700_ge_95']} ({union['gt_wa_in_top700']:.4f})",
        f"- hybrid union WA@100 >= 92%: {union['acceptance']['wa_top100_ge_92']} ({union['gt_wa_in_top100']:.4f})",
        f"- hybrid union WA@20 > 83%: {union['acceptance']['wa_top20_gt_83']} ({union['gt_wa_in_top20']:.4f})",
        f"- hybrid union skeleton@200 >= 97%: {union['acceptance']['skeleton_top200_ge_97']} ({union['gt_skeleton_in_top200']:.4f})",
        f"- hybrid union timeout <= 1: {union['acceptance']['timeout_le_1']} ({union['timeout_samples']})",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hybrid SymCIF-v4 WA search candidates and metrics from policy and old search outputs.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--policy-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_policy_search")
    parser.add_argument("--old-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=700)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    if args.max_records is not None:
        records = records[: int(args.max_records)]
    policy_path = args.policy_dir / f"{args.split}_policy_candidates.jsonl"
    old_path = args.old_dir / f"{args.split}_streaming_candidates.jsonl"
    policy = source_rows(policy_path, args.split, "policy")
    old = source_rows(old_path, args.split, "old")
    policy_sample = summary_rows(args.policy_dir / f"{args.split}_policy_per_sample.jsonl")
    old_sample = summary_rows(args.old_dir / f"{args.split}_streaming_per_sample.jsonl")

    hybrid_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    all_source_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        p_row = policy.get(sid)
        o_row = old.get(sid)
        p_candidates = candidates_for(p_row)
        o_candidates = candidates_for(o_row)
        flags = fallback_flags(record, policy_sample.get(sid))
        fallback_enabled = any(flags.values())
        union_candidates = merge_one(p_candidates, o_candidates, fallback_enabled=fallback_enabled, top_k=args.top_k)
        hybrid_rows.append(
            {
                "split": args.split,
                "sample_id": sid,
                "formula": record["formula"],
                "formula_counts": record["formula_counts"],
                "sg": int(record["sg"]),
                "n_sites": int(record["n_sites"]),
                "num_elements": int(record["num_elements"]),
                "gt_skeleton_key": record["canonical_skeleton_key"],
                "gt_wa_key": record["canonical_wa_key"],
                "fallback_enabled": fallback_enabled,
                "fallback_reasons": sorted(k for k, v in flags.items() if v),
                "candidates": union_candidates,
            }
        )
        p_stats = policy_sample.get(sid, {})
        o_stats = old_sample.get(sid, {})
        elapsed = None
        if p_stats or o_stats:
            elapsed = float(p_stats.get("elapsed_s") or 0.0) + (float(o_stats.get("elapsed_s") or 0.0) if fallback_enabled else 0.0)
        per_sample.append(
            summarize_candidate_set(
                record=record,
                candidates=union_candidates,
                source_name="union",
                elapsed_s=elapsed,
                timeout=False,
                fallback_enabled=fallback_enabled,
                flags=flags,
            )
        )
        all_source_rows.append(
            summarize_candidate_set(
                record=record,
                candidates=p_candidates,
                source_name="policy_only",
                elapsed_s=None if not p_stats else float(p_stats.get("elapsed_s") or 0.0),
                timeout=bool(p_stats.get("timeout")) if p_stats else False,
                flags=flags,
            )
        )
        all_source_rows.append(
            summarize_candidate_set(
                record=record,
                candidates=o_candidates,
                source_name="old_only",
                elapsed_s=None if not o_stats else float(o_stats.get("elapsed_s") or 0.0),
                timeout=bool(o_stats.get("timeout")) if o_stats else False,
                flags=flags,
            )
        )
        all_source_rows.append(
            summarize_candidate_set(
                record=record,
                candidates=o_candidates if fallback_enabled else [],
                source_name="fallback",
                elapsed_s=None if not o_stats else float(o_stats.get("elapsed_s") or 0.0),
                timeout=bool(o_stats.get("timeout")) if o_stats else False,
                fallback_enabled=fallback_enabled,
                flags=flags,
            )
        )
        all_source_rows.append(per_sample[-1])

    write_jsonl(args.out_dir / f"{args.split}_hybrid_candidates.jsonl", hybrid_rows)
    write_jsonl(args.out_dir / f"{args.split}_hybrid_per_sample.jsonl", per_sample)
    if args.split == "test":
        write_jsonl(args.out_dir / "test_hybrid_candidates.jsonl", hybrid_rows)
        write_csv(args.out_dir / "hybrid_breakdown_per_sg.csv", group_rows(per_sample, "sg"))
        write_csv(args.out_dir / "hybrid_breakdown_per_nsites.csv", group_rows(per_sample, "n_sites"))
        write_csv(args.out_dir / "hybrid_breakdown_per_num_elements.csv", group_rows(per_sample, "num_elements"))
    write_csv(args.out_dir / f"hybrid_breakdown_per_sg_{args.split}.csv", group_rows(per_sample, "sg"))
    write_csv(args.out_dir / f"hybrid_breakdown_per_nsites_{args.split}.csv", group_rows(per_sample, "n_sites"))
    write_csv(args.out_dir / f"hybrid_breakdown_per_num_elements_{args.split}.csv", group_rows(per_sample, "num_elements"))

    split_summary = {
        "split": args.split,
        "config": {
            "data_root": str(args.data_root),
            "policy_dir": str(args.policy_dir),
            "old_dir": str(args.old_dir),
            "top_k": args.top_k,
            "max_records": args.max_records,
        },
        "sources": source_metrics(all_source_rows),
        "target_subsets": target_subset_summary(per_sample),
        "timeout_samples": [
            {"sample_id": r["sample_id"], "sg": r["sg"], "n_sites": r["n_sites"], "elapsed_s": r["elapsed_s"]}
            for r in all_source_rows
            if r["source"] == "policy_only" and r["timeout"]
        ],
    }
    (args.out_dir / f"hybrid_search_summary_{args.split}.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.split == "test":
        (args.out_dir / "hybrid_search_summary.json").write_text(
            json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_summary_md(args.out_dir / "summary.md", split_summary)
    print(json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
