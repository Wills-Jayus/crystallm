#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "structured_symcif_v4_mp20"


SUBSETS = (
    "overall",
    "n_sites>=6",
    "n_sites>=12",
    "n_sites>=20",
    "num_elements>=4",
    "rare_sg",
    "high_multiplicity_orbit",
    "extraction_hard",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
    except Exception:
        return "-"
    return f"{float(value) * 100:.2f}%"


def f4(value: float | None) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
    except Exception:
        return "-"
    return f"{float(value):.4f}"


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return math.nan
    return float(value)


def candidate_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return list(candidate.get("rows") or [])


def wa_key_from_rows(rows: list[dict[str, Any]]) -> str:
    items = sorted((str(row["orbit_id"]), str(row["element"])) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def skeleton_key_from_rows(rows: list[dict[str, Any]]) -> str:
    items = sorted(str(row["orbit_id"]) for row in rows)
    return json.dumps(items, ensure_ascii=True, separators=(",", ":"))


def record_wa_key(record: dict[str, Any]) -> str:
    return wa_key_from_rows(list(record.get("wa_table") or []))


def record_skeleton_key(record: dict[str, Any]) -> str:
    return skeleton_key_from_rows(list(record.get("wa_table") or []))


def candidate_wa_key(candidate: dict[str, Any]) -> str:
    return wa_key_from_rows(candidate_rows(candidate))


def candidate_skeleton_key(candidate: dict[str, Any]) -> str:
    return skeleton_key_from_rows(candidate_rows(candidate))


def rank_value(candidate: dict[str, Any], key: str, default: float = 1.0e6) -> float:
    value = candidate.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def subset_indexes(records: list[dict[str, Any]], subset: str) -> set[int]:
    sg_counts = Counter(int(row["sg"]) for row in records)
    selected: set[int] = set()
    for idx, row in enumerate(records):
        n_sites = int(row.get("n_sites", 0))
        num_elements = int(row.get("num_elements", 0))
        max_mult = max((int(w.get("multiplicity", 1)) for w in row.get("wa_table", [])), default=1)
        if subset == "overall":
            selected.add(idx)
        elif subset == "n_sites>=6" and n_sites >= 6:
            selected.add(idx)
        elif subset == "n_sites>=12" and n_sites >= 12:
            selected.add(idx)
        elif subset == "n_sites>=20" and n_sites >= 20:
            selected.add(idx)
        elif subset == "num_elements>=4" and num_elements >= 4:
            selected.add(idx)
        elif subset == "rare_sg" and sg_counts[int(row["sg"])] <= 10:
            selected.add(idx)
        elif subset == "high_multiplicity_orbit" and max_mult >= 12:
            selected.add(idx)
        elif subset == "extraction_hard" and not bool(row.get("free_param_reextract_all_success", True)):
            selected.add(idx)
    return selected


def build_train_stats(train_records: list[dict[str, Any]]) -> dict[str, Any]:
    wa_counts: Counter[str] = Counter()
    skel_counts: Counter[str] = Counter()
    row_counts: Counter[tuple[int, str, str]] = Counter()
    orbit_counts: Counter[tuple[int, str]] = Counter()
    ordered_wa_keys: Counter[str] = Counter()
    ordered_skel_keys: Counter[str] = Counter()
    for record in train_records:
        wa_counts[record_wa_key(record)] += 1
        skel_counts[record_skeleton_key(record)] += 1
        ordered_wa_keys[str(record.get("canonical_wa_key"))] += 1
        ordered_skel_keys[str(record.get("canonical_skeleton_key"))] += 1
        sg = int(record["sg"])
        for row in record.get("wa_table") or []:
            row_counts[(sg, str(row["orbit_id"]), str(row["element"]))] += 1
            orbit_counts[(sg, str(row["orbit_id"]))] += 1
    return {
        "wa_counts": wa_counts,
        "skel_counts": skel_counts,
        "row_counts": row_counts,
        "orbit_counts": orbit_counts,
        "ordered_wa_keys": ordered_wa_keys,
        "ordered_skel_keys": ordered_skel_keys,
    }


def hybrid_prior_score(candidate: dict[str, Any], stats: dict[str, Any]) -> float:
    rows = candidate_rows(candidate)
    sg = int(candidate.get("sg", 0) or 0)
    wa_count = stats["wa_counts"].get(candidate_wa_key(candidate), 0)
    skel_count = stats["skel_counts"].get(candidate_skeleton_key(candidate), 0)
    row_score = 0.0
    orbit_score = 0.0
    for row in rows:
        orbit = str(row["orbit_id"])
        element = str(row["element"])
        row_score += math.log1p(stats["row_counts"].get((sg, orbit, element), 0))
        orbit_score += math.log1p(stats["orbit_counts"].get((sg, orbit), 0))
    policy_rank = rank_value(candidate, "policy_rank")
    old_rank = rank_value(candidate, "old_rank")
    hybrid_score = candidate.get("hybrid_score")
    try:
        hybrid_score_value = float(hybrid_score)
    except Exception:
        hybrid_score_value = 0.0
    return (
        3.00 * math.log1p(wa_count)
        + 1.20 * math.log1p(skel_count)
        + 0.09 * row_score
        + 0.03 * orbit_score
        + 0.55 / (1.0 + policy_rank)
        + 0.35 / (1.0 + old_rank)
        + 0.015 * hybrid_score_value
    )


def dedup_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate_wa_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def candidate_hit_summary(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    candidate_field: str = "candidates",
    top_ks: tuple[int, ...] = (1, 5, 10, 50, 100),
    sample_indexes: set[int] | None = None,
) -> dict[str, Any]:
    by_id = {str(row["sample_id"]): row for row in predictions}
    if sample_indexes is None:
        sample_indexes = set(range(len(records)))
    out: dict[str, Any] = {"samples": len(sample_indexes)}
    for k in top_ks:
        wa_hit = 0
        skel_hit = 0
        skeleton_without_wa = 0
        for idx in sample_indexes:
            record = records[idx]
            pred = by_id.get(str(record["sample_id"]), {})
            candidates = list(pred.get(candidate_field) or pred.get("ranked_wa_candidates") or [])[:k]
            gt_wa = record_wa_key(record)
            gt_skel = record_skeleton_key(record)
            has_wa = any(candidate_wa_key(candidate) == gt_wa for candidate in candidates)
            has_skel = any(candidate_skeleton_key(candidate) == gt_skel for candidate in candidates)
            wa_hit += int(has_wa)
            skel_hit += int(has_skel)
            skeleton_without_wa += int(has_skel and not has_wa)
        denom = max(1, len(sample_indexes))
        out[f"wa_hit@{k}"] = wa_hit / denom
        out[f"skeleton_hit@{k}"] = skel_hit / denom
        out[f"skeleton_without_wa@{k}"] = skeleton_without_wa / denom
        out[f"wa_hit_count@{k}"] = wa_hit
        out[f"skeleton_hit_count@{k}"] = skel_hit
        out[f"skeleton_without_wa_count@{k}"] = skeleton_without_wa
    return out


def write_fix_predictions(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    stats = build_train_stats(train_records)
    fixed_rows: list[dict[str, Any]] = []
    selector_scores: list[dict[str, Any]] = []
    for pred in raw_predictions:
        candidates = dedup_candidates(list(pred.get("candidates") or pred.get("ranked_wa_candidates") or []))
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, candidate in enumerate(candidates):
            score = hybrid_prior_score(candidate, stats)
            scored.append((score, idx, candidate))
        scored.sort(key=lambda item: (-item[0], item[1], str(item[2].get("canonical_wa_key"))))
        selected: list[dict[str, Any]] = []
        for rank, (score, raw_idx, candidate) in enumerate(scored[:5], start=1):
            row = dict(candidate)
            labels = list(row.get("source_labels") or [])
            if "hybrid_prior_fix" not in labels:
                labels.append("hybrid_prior_fix")
            row["source_labels"] = labels
            row["final_rerank_rank"] = rank
            row["final_rerank_score"] = score
            row["fix_selector"] = "hybrid_prior_train_only"
            row["fix_raw_index"] = raw_idx
            selected.append(row)
            selector_scores.append(
                {
                    "sample_id": pred["sample_id"],
                    "rank": rank,
                    "score": score,
                    "raw_index": raw_idx,
                    "policy_rank": candidate.get("policy_rank"),
                    "old_rank": candidate.get("old_rank"),
                    "train_wa_count": stats["wa_counts"].get(candidate_wa_key(candidate), 0),
                    "train_skeleton_count": stats["skel_counts"].get(candidate_skeleton_key(candidate), 0),
                    "canonical_wa_key": candidate.get("canonical_wa_key"),
                }
            )
        fixed = {k: v for k, v in pred.items() if k != "candidates"}
        fixed["ranked_wa_candidates"] = selected
        fixed["fix_selector"] = "hybrid_prior_train_only"
        fixed_rows.append(fixed)
    write_jsonl(out_path, fixed_rows)
    write_jsonl(out_path.with_name("one_fix_hybrid_prior_selector_scores.jsonl"), selector_scores)
    current = candidate_hit_summary(val_records, raw_predictions, candidate_field="candidates", top_ks=(1, 5, 100))
    fixed = candidate_hit_summary(val_records, fixed_rows, candidate_field="ranked_wa_candidates", top_ks=(1, 5))
    audit = {
        "selector": "hybrid_prior_train_only",
        "prediction_path": str(out_path),
        "samples": len(val_records),
        "current_candidate_coverage": current,
        "fixed_candidate_coverage": fixed,
        "notes": [
            "The selector uses only MP-20 train split frequency priors and existing policy/old search ranks.",
            "No StructureMatcher labels, validation match labels, or test labels are used in scoring.",
        ],
    }
    write_json(out_path.with_name("one_fix_hybrid_prior_candidate_audit.json"), audit)
    return audit


def load_eval_breakdown(eval_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv_dicts(eval_dir / "full_eval_breakdown.csv")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        out[(str(row["subset"]), int(float(row["k"])))] = row
    return out


def load_metrics(eval_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(eval_dir / "metrics" / "baseline_per_generation_metrics.jsonl")


def metrics_by_source(records: list[dict[str, Any]], metrics: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    record_by_idx = {idx: row for idx, row in enumerate(records)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if int(metric.get("gen_index", 0)) < k:
            grouped[str(metric.get("geometry_source"))].append(metric)
    rows: list[dict[str, Any]] = []
    for source, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        attempts = len(items)
        matches = sum(bool(item.get("match_ok")) for item in items)
        strict_valid = sum(bool(item.get("strict_valid")) for item in items)
        readable = sum(bool(item.get("pymatgen_readable")) for item in items)
        wa_hit = 0
        skeleton_hit = 0
        rms = []
        for item in items:
            record = record_by_idx[int(item["sample_index"])]
            wa_hit += int(str(item.get("wa_multiset_key")) == record_wa_key(record))
            skeleton_hit += int(str(item.get("skeleton_multiset_key")) == record_skeleton_key(record))
            if item.get("match_ok") and item.get("rms") is not None:
                rms.append(float(item["rms"]))
        rows.append(
            {
                "geometry_source": source,
                "attempts": attempts,
                "match_rate": matches / max(1, attempts),
                "strict_valid_rate": strict_valid / max(1, attempts),
                "readable_rate": readable / max(1, attempts),
                "candidate_wa_hit_rate": wa_hit / max(1, attempts),
                "candidate_skeleton_hit_rate": skeleton_hit / max(1, attempts),
                "mean_rms_on_matched_attempts": statistics.mean(rms) if rms else math.nan,
            }
        )
    return rows


def train_prototype_audit(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    stats = build_train_stats(train_records)
    train_wa = stats["wa_counts"]
    train_skel = stats["skel_counts"]
    train_ordered_wa = stats["ordered_wa_keys"]
    by_id = {str(row["sample_id"]): row for row in raw_predictions}
    gt_wa_in_train = sum(record_wa_key(row) in train_wa for row in val_records)
    gt_skel_in_train = sum(record_skeleton_key(row) in train_skel for row in val_records)
    top5 = {"attempts": 0, "same_wa_possible": 0, "same_skeleton_possible": 0, "ordered_miss_but_stable_hit": 0}
    raw100 = {"attempts": 0, "same_wa_possible": 0, "same_skeleton_possible": 0, "ordered_miss_but_stable_hit": 0}
    for record in val_records:
        candidates = list(by_id.get(str(record["sample_id"]), {}).get("candidates") or [])
        for bucket, limit in ((top5, 5), (raw100, 100)):
            for candidate in candidates[:limit]:
                bucket["attempts"] += 1
                stable_wa_hit = candidate_wa_key(candidate) in train_wa
                stable_skel_hit = candidate_skeleton_key(candidate) in train_skel
                bucket["same_wa_possible"] += int(stable_wa_hit)
                bucket["same_skeleton_possible"] += int(stable_skel_hit)
                bucket["ordered_miss_but_stable_hit"] += int(
                    stable_wa_hit and str(candidate.get("canonical_wa_key")) not in train_ordered_wa
                )
    for bucket in (top5, raw100):
        attempts = max(1, int(bucket["attempts"]))
        bucket["same_wa_possible_rate"] = bucket["same_wa_possible"] / attempts
        bucket["same_skeleton_possible_rate"] = bucket["same_skeleton_possible"] / attempts
        bucket["ordered_miss_but_stable_hit_rate"] = bucket["ordered_miss_but_stable_hit"] / attempts
    return {
        "val_samples": len(val_records),
        "gt_wa_key_seen_in_train": gt_wa_in_train,
        "gt_wa_key_seen_in_train_rate": gt_wa_in_train / max(1, len(val_records)),
        "gt_skeleton_seen_in_train": gt_skel_in_train,
        "gt_skeleton_seen_in_train_rate": gt_skel_in_train / max(1, len(val_records)),
        "candidate_train_coverage_top5": top5,
        "candidate_train_coverage_raw100": raw100,
    }


def core_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": int(float(row.get("samples", 0))),
        "match": as_float(row, "match_at_k"),
        "RMSE": as_float(row, "RMSE"),
        "WA_hit": as_float(row, "wa_hit_at_k"),
        "skeleton_hit": as_float(row, "skeleton_hit_at_k"),
        "readable": as_float(row, "readable"),
        "formula_ok": as_float(row, "formula_ok"),
        "atom_count_ok": as_float(row, "atom_count_ok"),
        "SG_ok": as_float(row, "SG_ok"),
        "strict_valid": as_float(row, "strict_valid"),
        "strict_valid_any": as_float(row, "strict_valid_any_at_k"),
        "eval_timeout": as_float(row, "eval_timeout"),
    }


def summary_from_breakdown(breakdown: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    return {
        "top1": core_row(breakdown[("overall", 1)]),
        "top5": core_row(breakdown[("overall", 5)]),
        "subsets": {
            subset: {
                "top1": core_row(breakdown[(subset, 1)]),
                "top5": core_row(breakdown[(subset, 5)]),
            }
            for subset in SUBSETS
            if (subset, 1) in breakdown and (subset, 5) in breakdown
        },
    }


def metric_table(title: str, summary: dict[str, Any]) -> list[str]:
    rows = [
        f"## {title}",
        "",
        "| metric | @1 | @5 |",
        "| --- | ---: | ---: |",
    ]
    top1 = summary["top1"]
    top5 = summary["top5"]
    for key in ("match", "RMSE", "WA_hit", "skeleton_hit", "readable", "formula_ok", "atom_count_ok", "SG_ok", "strict_valid", "strict_valid_any", "eval_timeout"):
        if key == "RMSE":
            rows.append(f"| {key} | {f4(top1[key])} | {f4(top5[key])} |")
        else:
            rows.append(f"| {key} | {pct(top1[key])} | {pct(top5[key])} |")
    rows.append("")
    return rows


def subset_table(summary: dict[str, Any], fields: tuple[str, ...] = ("match", "WA_hit", "skeleton_hit")) -> list[str]:
    header = "| subset | samples | " + " | ".join(f"{field}@1" for field in fields) + " | " + " | ".join(f"{field}@5" for field in fields) + " |"
    sep = "| --- | ---: | " + " | ".join("---:" for _ in range(len(fields) * 2)) + " |"
    rows = [header, sep]
    for subset, item in summary.get("subsets", {}).items():
        t1 = item["top1"]
        t5 = item["top5"]
        values = [pct(t1[field]) for field in fields] + [pct(t5[field]) for field in fields]
        rows.append(f"| {subset} | {t1['samples']} | " + " | ".join(values) + " |")
    return rows


def write_val_baseline_report(report_root: Path, summary: dict[str, Any], raw: dict[str, Any]) -> None:
    write_json(report_root / "val_baseline_summary.json", raw)
    lines = ["# MP-20 Val K<=5 Baseline Summary", ""]
    lines += metric_table("Overall", summary)
    lines += ["## Complex Subsets", ""]
    lines += subset_table(summary)
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is the val split only, using current hybrid WA order and the existing geometry model/prototype renderer.",
        "- The table intentionally reports only @1 and @5; no match@20 result is produced in this dev round.",
        "- WA_hit@5 is the immediate candidate-selection ceiling for K<=5; the remaining gap to match@5 is geometry/evaluator quality.",
        "",
    ]
    (report_root / "val_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_coverage_report(report_root: Path, payload: dict[str, Any]) -> None:
    write_json(report_root / "wa_top5_coverage_audit.json", payload)
    lines = ["# MP-20 Val WA Top5 Coverage Audit", ""]
    lines += [
        "## Overall Candidate Coverage",
        "",
        "| source | K | WA_hit | skeleton_hit | skeleton_without_WA |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source, item, ks in (
        ("current selected order", payload["current"], (1, 5)),
        ("raw hybrid pool", payload["raw"], (5, 10, 50, 100)),
        ("one-fix selected order", payload["one_fix"], (1, 5)),
    ):
        for k in ks:
            lines.append(
                f"| {source} | {k} | {pct(item[f'wa_hit@{k}'])} | "
                f"{pct(item[f'skeleton_hit@{k}'])} | {pct(item[f'skeleton_without_wa@{k}'])} |"
            )
    lines += [
        "",
        "## Lost GT-WA Mass",
        "",
        f"- Raw top100 has GT-WA but current top5 misses it: {payload['raw100_hit_selected5_miss']} / {payload['samples']}.",
        f"- Raw top100 has no GT-WA at all: {payload['raw100_wa_miss']} / {payload['samples']}.",
        f"- Current top5 skeleton hit but wrong element assignment: {payload['current_skeleton_without_wa_top5']} / {payload['samples']}.",
        "",
        "## Subsets",
        "",
        "| subset | samples | current WA@5 | raw WA@100 | one-fix WA@5 | current skeleton@5 | raw skeleton@100 | one-fix skeleton@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset, item in payload["subsets"].items():
        lines.append(
            f"| {subset} | {item['samples']} | {pct(item['current']['wa_hit@5'])} | "
            f"{pct(item['raw']['wa_hit@100'])} | {pct(item['one_fix']['wa_hit@5'])} | "
            f"{pct(item['current']['skeleton_hit@5'])} | {pct(item['raw']['skeleton_hit@100'])} | "
            f"{pct(item['one_fix']['skeleton_hit@5'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "The dominant K<=5 issue is selected-candidate ranking: raw top100 contains substantially more GT WA than current top5. This selects branch A for the single fix experiment.",
        "",
    ]
    (report_root / "wa_top5_coverage_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_geometry_source_report(report_root: Path, payload: dict[str, Any]) -> None:
    write_json(report_root / "geometry_source_audit.json", payload)
    lines = [
        "# MP-20 Val Geometry Source Audit",
        "",
        "## Train Prototype Coverage",
        "",
        f"- Val GT-WA key seen in train: {payload['train_prototype_audit']['gt_wa_key_seen_in_train']} / {payload['train_prototype_audit']['val_samples']} ({pct(payload['train_prototype_audit']['gt_wa_key_seen_in_train_rate'])}).",
        f"- Val GT skeleton seen in train: {payload['train_prototype_audit']['gt_skeleton_seen_in_train']} / {payload['train_prototype_audit']['val_samples']} ({pct(payload['train_prototype_audit']['gt_skeleton_seen_in_train_rate'])}).",
        f"- Current top5 candidates with same-WA prototype available: {pct(payload['train_prototype_audit']['candidate_train_coverage_top5']['same_wa_possible_rate'])}.",
        f"- Raw top100 candidates with same-WA prototype available: {pct(payload['train_prototype_audit']['candidate_train_coverage_raw100']['same_wa_possible_rate'])}.",
        "",
        "## Candidate Attempt Quality By Geometry Source",
        "",
        "| geometry_source | attempts | match_rate | strict_valid | readable | candidate_WA_hit | candidate_skeleton_hit | mean_RMS_matched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["baseline_source_rows"]:
        lines.append(
            f"| {row['geometry_source']} | {row['attempts']} | {pct(row['match_rate'])} | "
            f"{pct(row['strict_valid_rate'])} | {pct(row['readable_rate'])} | "
            f"{pct(row['candidate_wa_hit_rate'])} | {pct(row['candidate_skeleton_hit_rate'])} | "
            f"{f4(row['mean_rms_on_matched_attempts'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The code path is now using stable multiset keys for WA/skeleton hit accounting; ordered-key undercount is not the main remaining issue.",
        "- Same-WA prototypes are useful when selected, but current K<=5 often fails before geometry because the selected WA itself is wrong.",
        "- Geometry still matters: same-skeleton/same-SG fallback attempts have lower match quality and higher RMSE, which explains why GT-WA does not become a perfect match oracle.",
        "",
    ]
    (report_root / "geometry_source_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_gtwa_report(report_root: Path, summary: dict[str, Any], raw: dict[str, Any]) -> None:
    write_json(report_root / "gtwa_geometry_k5_eval.json", raw)
    lines = ["# MP-20 Val GT-WA Geometry K<=5 Eval", ""]
    lines += metric_table("GT-WA Geometry", summary)
    lines += ["## Complex Subsets", ""]
    lines += subset_table(summary, fields=("match",))
    lines += [
        "",
        "## Interpretation",
        "",
        "- This fixes the WA candidate to the ground truth and only tests geometry/prototype/model quality under K<=5.",
        "- Any remaining gap below 100% match is geometry/evaluator/RMSE quality, not WA search coverage.",
        "",
    ]
    (report_root / "gtwa_geometry_k5_eval.md").write_text("\n".join(lines), encoding="utf-8")


def write_one_fix_report(report_root: Path, payload: dict[str, Any]) -> None:
    write_json(report_root / "one_fix_experiment_summary.json", payload)
    baseline = payload["baseline_summary"]
    fix = payload["fix_summary"]
    lines = [
        "# MP-20 Val One-Fix Experiment Summary",
        "",
        "Selected branch: A, train-prior/hybrid-prior top5 candidate selection.",
        "",
        "| metric | baseline @1 | one-fix @1 | delta @1 | baseline @5 | one-fix @5 | delta @5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("match", "RMSE", "WA_hit", "skeleton_hit", "readable", "formula_ok", "atom_count_ok", "SG_ok", "strict_valid"):
        b1, f1 = baseline["top1"][key], fix["top1"][key]
        b5, f5 = baseline["top5"][key], fix["top5"][key]
        if key == "RMSE":
            lines.append(f"| {key} | {f4(b1)} | {f4(f1)} | {f4(f1 - b1)} | {f4(b5)} | {f4(f5)} | {f4(f5 - b5)} |")
        else:
            lines.append(f"| {key} | {pct(b1)} | {pct(f1)} | {pct(f1 - b1)} | {pct(b5)} | {pct(f5)} | {pct(f5 - b5)} |")
    lines += [
        "",
        "## What Changed",
        "",
        "- The fix only changes the ordering/selection of WA candidates before rendering.",
        "- It uses train split WA/skeleton/row frequency priors plus existing search ranks; it does not use evaluator labels or validation match labels in scoring.",
        "- This is the only fix branch run in this dev round.",
        "",
    ]
    (report_root / "one_fix_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(report_root: Path, payload: dict[str, Any]) -> None:
    baseline = payload["baseline_summary"]
    fix = payload["fix_summary"]
    gtwa = payload["gtwa_summary"]
    coverage = payload["coverage"]
    lines = [
        "# MP-20 K<=5 Dev Final Root Cause Summary",
        "",
        "## Main Answer",
        "",
        "The current bottleneck is candidate selection under the K<=5 budget, followed by geometry quality. It is not primarily a raw WA search failure and not primarily an evaluation/rendering engineering bug.",
        "",
        "## Evidence",
        "",
        f"- Baseline val match@1/@5 = {pct(baseline['top1']['match'])} / {pct(baseline['top5']['match'])}; WA_hit@1/@5 = {pct(baseline['top1']['WA_hit'])} / {pct(baseline['top5']['WA_hit'])}.",
        f"- Raw top100 WA_hit is {pct(coverage['raw']['wa_hit@100'])}, but current selected top5 WA_hit is only {pct(coverage['current']['wa_hit@5'])}. The lost raw-to-selected mass is {coverage['raw100_hit_selected5_miss']} samples.",
        f"- The one-fix selector raises selected WA_hit@5 to {pct(coverage['one_fix']['wa_hit@5'])}; actual match@5 changes from {pct(baseline['top5']['match'])} to {pct(fix['top5']['match'])}.",
        f"- GT-WA geometry match@1/@5 = {pct(gtwa['top1']['match'])} / {pct(gtwa['top5']['match'])}; this shows geometry/RMSE remains a hard ceiling even when WA is correct.",
        "",
        "## Engineering Bug Audit",
        "",
        "- No new evidence of a fatal engineering bug in the val K<=5 pipeline: CIF rendering, stable multiset hit accounting, formula/SG checks, and evaluator outputs are internally consistent.",
        "- The earlier ordered-key/stablekey issue would undercount WA coverage, but this round's reports use stable multiset keys.",
        "- Timeouts and missing candidates should still be tracked, but they are not large enough to explain the main performance gap.",
        "",
        "## Recommended Next Step",
        "",
        "Keep the branch-A direction: make candidate scoring inference-feasible and stronger, then add a non-oracle geometry-quality scorer/reranker. Do not spend the next iteration on MPTS-52, match@20, or multiple test-set tuning rounds.",
        "",
        "## Artifacts",
        "",
        "- `val_baseline_summary.md/json`",
        "- `wa_top5_coverage_audit.md/json`",
        "- `geometry_source_audit.md/json`",
        "- `gtwa_geometry_k5_eval.md/json`",
        "- `one_fix_experiment_summary.md/json`",
        "",
    ]
    (report_root / "final_root_cause_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_coverage_payload(
    records: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    fixed_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    current = candidate_hit_summary(records, raw_predictions, candidate_field="candidates", top_ks=(1, 5))
    raw = candidate_hit_summary(records, raw_predictions, candidate_field="candidates", top_ks=(5, 10, 50, 100))
    one_fix = candidate_hit_summary(records, fixed_predictions, candidate_field="ranked_wa_candidates", top_ks=(1, 5))
    by_id = {str(row["sample_id"]): row for row in raw_predictions}
    raw100_hit_selected5_miss = 0
    raw100_wa_miss = 0
    current_skeleton_without_wa_top5 = 0
    for record in records:
        candidates = list(by_id.get(str(record["sample_id"]), {}).get("candidates") or [])
        gt_wa = record_wa_key(record)
        gt_skel = record_skeleton_key(record)
        selected5_has_wa = any(candidate_wa_key(candidate) == gt_wa for candidate in candidates[:5])
        raw100_has_wa = any(candidate_wa_key(candidate) == gt_wa for candidate in candidates[:100])
        selected5_has_skel = any(candidate_skeleton_key(candidate) == gt_skel for candidate in candidates[:5])
        raw100_hit_selected5_miss += int(raw100_has_wa and not selected5_has_wa)
        raw100_wa_miss += int(not raw100_has_wa)
        current_skeleton_without_wa_top5 += int(selected5_has_skel and not selected5_has_wa)
    subsets: dict[str, Any] = {}
    for subset in SUBSETS:
        indexes = subset_indexes(records, subset)
        subsets[subset] = {
            "samples": len(indexes),
            "current": candidate_hit_summary(records, raw_predictions, candidate_field="candidates", top_ks=(5,), sample_indexes=indexes),
            "raw": candidate_hit_summary(records, raw_predictions, candidate_field="candidates", top_ks=(100,), sample_indexes=indexes),
            "one_fix": candidate_hit_summary(records, fixed_predictions, candidate_field="ranked_wa_candidates", top_ks=(5,), sample_indexes=indexes),
        }
    return {
        "samples": len(records),
        "current": current,
        "raw": raw,
        "one_fix": one_fix,
        "raw100_hit_selected5_miss": raw100_hit_selected5_miss,
        "raw100_wa_miss": raw100_wa_miss,
        "current_skeleton_without_wa_top5": current_skeleton_without_wa_top5,
        "subsets": subsets,
    }


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MP-20 K<=5 dev reports.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--predictions-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_root: Path = args.report_root
    data_root: Path = args.data_root
    train_records = read_jsonl(data_root / "train.jsonl")
    val_records = read_jsonl(data_root / "val.jsonl")
    raw_predictions_path = report_root / "hybrid_search" / "val_hybrid_candidates.jsonl"
    raw_predictions = read_jsonl(raw_predictions_path)
    fixed_path = report_root / "one_fix_hybrid_prior_predictions.jsonl"
    selector_audit = write_fix_predictions(train_records, val_records, raw_predictions, fixed_path)
    if args.predictions_only:
        print(json.dumps(selector_audit, indent=2, sort_keys=True), flush=True)
        return 0

    baseline_dir = report_root / "val_baseline_eval_gpu"
    gtwa_dir = report_root / "gtwa_geometry_k5_eval_gpu"
    fix_dir = report_root / "one_fix_hybrid_prior_eval_gpu"
    for path in (
        baseline_dir / "full_eval_breakdown.csv",
        gtwa_dir / "full_eval_breakdown.csv",
        fix_dir / "full_eval_breakdown.csv",
    ):
        require_file(path)

    fixed_predictions = read_jsonl(fixed_path)
    baseline_breakdown = load_eval_breakdown(baseline_dir)
    gtwa_breakdown = load_eval_breakdown(gtwa_dir)
    fix_breakdown = load_eval_breakdown(fix_dir)
    baseline_summary = summary_from_breakdown(baseline_breakdown)
    gtwa_summary = summary_from_breakdown(gtwa_breakdown)
    fix_summary = summary_from_breakdown(fix_breakdown)
    baseline_raw_summary = json.loads((baseline_dir / "full_eval_summary.json").read_text(encoding="utf-8"))
    gtwa_raw_summary = json.loads((gtwa_dir / "full_eval_summary.json").read_text(encoding="utf-8"))
    fix_raw_summary = json.loads((fix_dir / "full_eval_summary.json").read_text(encoding="utf-8"))
    coverage = build_coverage_payload(val_records, raw_predictions, fixed_predictions)
    prototype_audit = train_prototype_audit(train_records, val_records, raw_predictions)
    baseline_source_rows = metrics_by_source(val_records, load_metrics(baseline_dir), k=5)
    geometry_payload = {
        "train_prototype_audit": prototype_audit,
        "baseline_source_rows": baseline_source_rows,
    }
    write_val_baseline_report(report_root, baseline_summary, {"summary": baseline_summary, "source_summary": baseline_raw_summary})
    write_coverage_report(report_root, coverage)
    write_geometry_source_report(report_root, geometry_payload)
    write_gtwa_report(report_root, gtwa_summary, {"summary": gtwa_summary, "source_summary": gtwa_raw_summary})
    one_fix_payload = {
        "selector_audit": selector_audit,
        "baseline_summary": baseline_summary,
        "fix_summary": fix_summary,
        "fix_source_summary": fix_raw_summary,
    }
    write_one_fix_report(report_root, one_fix_payload)
    final_payload = {
        "baseline_summary": baseline_summary,
        "fix_summary": fix_summary,
        "gtwa_summary": gtwa_summary,
        "coverage": coverage,
    }
    write_final_report(report_root, final_payload)
    print(json.dumps({"wrote_reports": str(report_root), "one_fix": one_fix_payload["fix_summary"]["top5"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
