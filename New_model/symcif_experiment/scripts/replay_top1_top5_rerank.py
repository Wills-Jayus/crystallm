#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


PUBLISHED_CRYSTALLM_A_MP20 = {
    "match1": 0.5585,
    "rmse1": 0.0437,
    "rmse20": 0.0395,
}

OLD_WA20_GEOM1_FIRST1000 = {
    "label": "B nonreuse first1000 WA20xgeom1 structured-only",
    "samples": 1000,
    "match1": 0.4940,
    "rmse1": 0.0833,
    "match5": 0.6610,
    "rmse5": 0.0965,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def fmt_float(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def record_wa_multiset_key(record: dict[str, Any]) -> str:
    if record.get("gt_wa_multiset_key"):
        return str(record["gt_wa_multiset_key"])
    if record.get("wa_table"):
        items = sorted((str(row["orbit_id"]), str(row["element"])) for row in record["wa_table"])
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    return ""


def record_skeleton_multiset_key(record: dict[str, Any]) -> str:
    if record.get("gt_skeleton_multiset_key"):
        return str(record["gt_skeleton_multiset_key"])
    if record.get("wa_table"):
        items = sorted(str(row["orbit_id"]) for row in record["wa_table"])
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    return ""


def benchmark_subset_indexes(records: list[dict[str, Any]], subset: str) -> set[int]:
    sg_counts = Counter(int(row["sg"]) for row in records)
    out: set[int] = set()
    for idx, row in enumerate(records):
        n_sites = int(row.get("n_sites", 0))
        num_elements = int(row.get("num_elements", 0))
        max_mult = max((int(w.get("multiplicity", 1)) for w in row.get("wa_table", [])), default=1)
        if subset == "overall":
            out.add(idx)
        elif subset == "n_sites>=6" and n_sites >= 6:
            out.add(idx)
        elif subset == "n_sites>=12" and n_sites >= 12:
            out.add(idx)
        elif subset == "num_elements>=4" and num_elements >= 4:
            out.add(idx)
        elif subset == "rare_sg" and sg_counts[int(row["sg"])] <= 10:
            out.add(idx)
        elif subset == "high_multiplicity_orbit" and max_mult >= 12:
            out.add(idx)
        elif subset == "extraction_hard" and not bool(row.get("free_param_reextract_all_success", True)):
            out.add(idx)
    return out


def source_bucket(value: Any) -> str:
    text = str(value or "")
    if not text or text == "None" or "missing_candidate" in text:
        return "missing_candidate"
    if "same_wa" in text:
        return "same_wa"
    if "same_skeleton" in text:
        return "same_skeleton"
    if "same_sg" in text:
        return "same_sg"
    if "model_fallback" in text or "global" in text:
        return "model_fallback"
    return text


def failure_category(row: dict[str, Any]) -> str:
    error = str(row.get("error") or "")
    if row.get("match"):
        return "matched"
    if (not row.get("render_success")) or error == "missing_candidate":
        return "missing_candidate"
    if row.get("eval_timeout") or row.get("parse_timeout") or row.get("sg_timeout") or row.get("matcher_timeout"):
        return "timeout"
    if not row.get("readable"):
        return "invalid_cif"
    if not row.get("formula_ok"):
        return "formula_mismatch"
    if not row.get("atom_count_ok"):
        return "atom_count_mismatch"
    if not row.get("SG_ok"):
        return "SG_mismatch"
    return "matcher_no_match"


def primary_failure(rows: list[dict[str, Any]]) -> str:
    if any(row.get("match") for row in rows):
        return "matched"
    cats = [failure_category(row) for row in rows] or ["missing_candidate"]
    for name in (
        "missing_candidate",
        "timeout",
        "invalid_cif",
        "formula_mismatch",
        "atom_count_mismatch",
        "SG_mismatch",
        "matcher_no_match",
    ):
        if name in cats:
            return name
    return cats[0]


def to_candidate_row(metric: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    wa_idx = metric.get("wa_candidate_index")
    geom_idx = metric.get("geometry_variant_index")
    row = {
        "sample_id": metric["sample_id"],
        "sample_index": int(metric["sample_index"]),
        "candidate_rank": int(metric.get("rank") or int(metric.get("gen_index", 0)) + 1),
        "gen_index": int(metric.get("gen_index", 0)),
        "wa_rank": (int(wa_idx) + 1) if wa_idx is not None else None,
        "wa_candidate_index": int(wa_idx) if wa_idx is not None else None,
        "geom_variant_id": int(geom_idx) if geom_idx is not None else None,
        "geometry_source": metric.get("geometry_source"),
        "geometry_source_bucket": source_bucket(metric.get("geometry_source")),
        "render_success": bool(metric.get("render_success")),
        "readable": bool(metric.get("pymatgen_readable") or metric.get("readable")),
        "formula_ok": bool(metric.get("formula_ok")),
        "atom_count_ok": bool(metric.get("atom_count_ok")),
        "SG_ok": bool(metric.get("space_group_ok") or metric.get("SG_ok")),
        "valid": bool(metric.get("valid")),
        "strict_valid": bool(metric.get("strict_valid")),
        "eval_timeout": bool(metric.get("eval_timeout")),
        "parse_timeout": bool(metric.get("parse_timeout")),
        "sg_timeout": bool(metric.get("sg_timeout")),
        "matcher_timeout": bool(metric.get("matcher_timeout")),
        "match": bool(metric.get("match_ok")),
        "rms": metric.get("rms"),
        "bond_length_score": metric.get("bond_length_score"),
        "reference_score": metric.get("reference_score"),
        "selection_score": metric.get("selection_score"),
        "hybrid_score": metric.get("hybrid_score"),
        "policy_rank": metric.get("policy_rank"),
        "old_rank": metric.get("old_rank"),
        "WA_hit": str(metric.get("wa_multiset_key")) == record_wa_multiset_key(record),
        "skeleton_hit": str(metric.get("skeleton_multiset_key")) == record_skeleton_multiset_key(record),
        "wa_multiset_key": metric.get("wa_multiset_key"),
        "skeleton_multiset_key": metric.get("skeleton_multiset_key"),
        "failure_reason": None,
        "n_sites": int(record.get("n_sites", 0)),
        "num_elements": int(record.get("num_elements", 0)),
        "sg": int(record.get("sg", 0)),
    }
    row["failure_reason"] = failure_category(row)
    return row


def summarize_selection(
    selected_by_sample: dict[int, list[dict[str, Any]]],
    sample_indexes: set[int],
    *,
    k: int,
    denominator_samples: int,
) -> dict[str, Any]:
    attempts = int(denominator_samples) * int(k)
    rows = [row for idx in sample_indexes for row in selected_by_sample.get(idx, [])[:k]]
    out: dict[str, Any] = {
        "k": int(k),
        "samples": int(denominator_samples),
        "structured_samples": len(sample_indexes),
        "num_attempts": attempts,
    }
    fields = [
        "render_success",
        "readable",
        "formula_ok",
        "atom_count_ok",
        "SG_ok",
        "valid",
        "strict_valid",
        "eval_timeout",
    ]
    for field in fields:
        out[field] = sum(bool(row.get(field)) for row in rows) / attempts if attempts else math.nan
    any_counts = Counter()
    match_count = 0
    rms_values: list[float] = []
    wa_hit = 0
    skeleton_hit = 0
    for idx in sample_indexes:
        sample_rows = selected_by_sample.get(idx, [])[:k]
        if any(row.get("match") for row in sample_rows):
            match_count += 1
        sample_rms = [float(row["rms"]) for row in sample_rows if row.get("match") and row.get("rms") is not None]
        if sample_rms:
            rms_values.append(min(sample_rms))
        for field in fields:
            if any(row.get(field) for row in sample_rows):
                any_counts[field] += 1
        if any(row.get("WA_hit") for row in sample_rows):
            wa_hit += 1
        if any(row.get("skeleton_hit") for row in sample_rows):
            skeleton_hit += 1
    out["match_at_k"] = match_count / max(1, denominator_samples)
    out["RMSE"] = float(statistics.mean(rms_values)) if rms_values else math.nan
    out["matched_samples_for_RMSE"] = len(rms_values)
    for field in fields:
        out[f"{field}_any_at_k"] = any_counts[field] / max(1, denominator_samples)
    out["WA_hit_at_k"] = wa_hit / max(1, denominator_samples)
    out["skeleton_hit_at_k"] = skeleton_hit / max(1, denominator_samples)
    return out


def by_current_rank(candidate_by_sample: dict[int, list[dict[str, Any]]], k: int) -> dict[int, list[dict[str, Any]]]:
    return {idx: sorted(rows, key=lambda r: int(r["candidate_rank"]))[:k] for idx, rows in candidate_by_sample.items()}


def by_budget(candidate_by_sample: dict[int, list[dict[str, Any]]], wa_count: int, geom_count: int, k: int) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for idx, rows in candidate_by_sample.items():
        filtered = [
            row
            for row in rows
            if row.get("wa_candidate_index") is not None
            and int(row["wa_candidate_index"]) < wa_count
            and row.get("geom_variant_id") is not None
            and int(row["geom_variant_id"]) < geom_count
        ]
        out[idx] = sorted(filtered, key=lambda r: int(r["candidate_rank"]))[:k]
    return out


def scorer_current(row: dict[str, Any]) -> tuple[Any, ...]:
    return (int(row["candidate_rank"]),)


def scorer_validity_first(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row.get("strict_valid"))),
        -int(bool(row.get("readable"))),
        -int(bool(row.get("formula_ok"))),
        -int(bool(row.get("SG_ok"))),
        -int(bool(row.get("atom_count_ok"))),
        -float(row.get("hybrid_score") or 0.0),
        int(row.get("wa_candidate_index") if row.get("wa_candidate_index") is not None else 999),
        int(row["candidate_rank"]),
    )


def source_priority(row: dict[str, Any]) -> int:
    return {
        "same_wa": 0,
        "same_skeleton": 1,
        "same_sg": 2,
        "model_fallback": 3,
        "missing_candidate": 9,
    }.get(str(row.get("geometry_source_bucket")), 5)


def scorer_geometry_quality(row: dict[str, Any]) -> tuple[Any, ...]:
    bond = row.get("bond_length_score")
    return (
        source_priority(row),
        -float(bond if bond is not None else -1.0),
        -float(row.get("reference_score") or 0.0),
        int(row.get("geom_variant_id") if row.get("geom_variant_id") is not None else 999),
        int(row.get("wa_candidate_index") if row.get("wa_candidate_index") is not None else 999),
        int(row["candidate_rank"]),
    )


def scorer_hybrid(row: dict[str, Any]) -> tuple[Any, ...]:
    bond = row.get("bond_length_score")
    return (
        -int(bool(row.get("strict_valid"))),
        source_priority(row),
        int(row.get("wa_candidate_index") if row.get("wa_candidate_index") is not None else 999),
        -float(row.get("hybrid_score") or 0.0),
        -float(bond if bond is not None else -1.0),
        int(row.get("geom_variant_id") if row.get("geom_variant_id") is not None else 999),
        int(row["candidate_rank"]),
    )


def by_scorer(candidate_by_sample: dict[int, list[dict[str, Any]]], scorer: Callable[[dict[str, Any]], tuple[Any, ...]], k: int) -> dict[int, list[dict[str, Any]]]:
    return {idx: sorted(rows, key=scorer)[:k] for idx, rows in candidate_by_sample.items()}


def failure_breakdown(selected: dict[int, list[dict[str, Any]]], sample_indexes: set[int], k: int) -> dict[str, Any]:
    sample_failures = Counter()
    candidate_failures = Counter()
    diag = Counter()
    geometry = defaultdict(Counter)
    for idx in sample_indexes:
        rows = selected.get(idx, [])[:k]
        matched = any(row.get("match") for row in rows)
        wa_hit = any(row.get("WA_hit") for row in rows)
        skeleton_hit = any(row.get("skeleton_hit") for row in rows)
        if matched:
            diag["matched"] += 1
        else:
            diag["failed"] += 1
            sample_failures[primary_failure(rows)] += 1
            diag["failed_with_WA_hit" if wa_hit else "failed_without_WA_hit"] += 1
            if skeleton_hit:
                diag["failed_with_skeleton_hit"] += 1
        for row in rows:
            candidate_failures[failure_category(row)] += 1
            bucket = str(row.get("geometry_source_bucket"))
            geometry[bucket]["attempts"] += 1
            if row.get("match"):
                geometry[bucket]["matched"] += 1
            if row.get("strict_valid"):
                geometry[bucket]["strict_valid"] += 1
    return {
        "k": int(k),
        "sample_failure_counts": dict(sample_failures),
        "candidate_failure_counts": dict(candidate_failures),
        "sample_diagnostics": dict(diag),
        "geometry_source_breakdown": {k: dict(v) for k, v in sorted(geometry.items())},
    }


def subset_breakdowns(
    records: list[dict[str, Any]],
    selected: dict[int, list[dict[str, Any]]],
    names: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        indexes = benchmark_subset_indexes(records, name)
        out[name] = {
            "samples": len(indexes),
            "top1": summarize_selection(selected, indexes, k=1, denominator_samples=len(indexes)),
            "top5": summarize_selection(selected, indexes, k=5, denominator_samples=len(indexes)),
            "failure_top5": failure_breakdown(selected, indexes, 5),
        }
    return out


def table_rows_for_metrics(metrics: dict[str, dict[str, Any]], labels: list[str]) -> list[str]:
    lines = ["| scope | k | match | RMSE | readable | formula_ok | atom_count_ok | SG_ok | strict_valid | eval_timeout | WA_hit | skeleton_hit |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope in labels:
        for top in ("top1", "top5"):
            m = metrics[scope][top]
            lines.append(
                "| "
                + " | ".join(
                    [
                        scope,
                        str(m["k"]),
                        pct(m["match_at_k"]),
                        fmt_float(m["RMSE"]),
                        pct(m["readable"]),
                        pct(m["formula_ok"]),
                        pct(m["atom_count_ok"]),
                        pct(m["SG_ok"]),
                        pct(m["strict_valid"]),
                        pct(m["eval_timeout"]),
                        pct(m["WA_hit_at_k"]),
                        pct(m["skeleton_hit_at_k"]),
                    ]
                )
                + " |"
            )
    return lines


def build_markdown(summary: dict[str, Any]) -> str:
    current = summary["current_order"]
    budget = summary["budget_replay"]["structured_only"]
    rerank = summary["rerank_replay"]["structured_only"]
    old = summary["references"]["old_wa20_geom1_first1000"]
    crystal = summary["references"]["published_crystallm_a_mp20"]
    cur_full = current["full_test"]
    cur_struct = current["structured_only"]
    delta_crystal = cur_full["top1"]["match_at_k"] - crystal["match1"]
    old_delta_m1 = cur_struct["top1"]["match_at_k"] - old["match1"]
    old_delta_m5 = cur_struct["top5"]["match_at_k"] - old["match5"]

    lines: list[str] = [
        "# MP-20 Top1/Top5 Rerank Replay Summary",
        "",
        "本报告只做 MP-20 top1/top5 replay；没有重新 render，没有重新跑 StructureMatcher，没有跑 MPTS-52，也没有生成新的 match@20 主表。",
        "",
        "## Current Ordering",
        "",
        *table_rows_for_metrics(current, ["structured_only", "full_test"]),
        "",
        "## Top5 Budget Replay",
        "",
        "| strategy | match@1 | match@5 | RMSE@1 | RMSE@5 | strict_valid@1 | strict_valid@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in budget.items():
        lines.append(
            f"| {name} | {pct(metrics['top1']['match_at_k'])} | {pct(metrics['top5']['match_at_k'])} | "
            f"{fmt_float(metrics['top1']['RMSE'])} | {fmt_float(metrics['top5']['RMSE'])} | "
            f"{pct(metrics['top1']['strict_valid'])} | {pct(metrics['top5']['strict_valid'])} |"
        )
    lines += [
        "",
        "## Lightweight Rerank Replay",
        "",
        "| scorer | type | match@1 | match@5 | RMSE@1 | RMSE@5 | strict_valid@1 | strict_valid@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in rerank.items():
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {item['type']} | {pct(metrics['top1']['match_at_k'])} | {pct(metrics['top5']['match_at_k'])} | "
            f"{fmt_float(metrics['top1']['RMSE'])} | {fmt_float(metrics['top5']['RMSE'])} | "
            f"{pct(metrics['top1']['strict_valid'])} | {pct(metrics['top5']['strict_valid'])} |"
        )
    fb1 = summary["error_breakdown"]["current_order_top1"]["sample_failure_counts"]
    fb5 = summary["error_breakdown"]["current_order_top5"]["sample_failure_counts"]
    diag5 = summary["error_breakdown"]["current_order_top5"]["sample_diagnostics"]
    lines += [
        "",
        "## Error Breakdown",
        "",
        f"- top1 failed reason breakdown: `{json.dumps(fb1, ensure_ascii=False, sort_keys=True)}`",
        f"- top5 failed reason breakdown: `{json.dumps(fb5, ensure_ascii=False, sort_keys=True)}`",
        f"- top5 failed_with_WA_hit / failed_without_WA_hit / failed_with_skeleton_hit: "
        f"{diag5.get('failed_with_WA_hit', 0)} / {diag5.get('failed_without_WA_hit', 0)} / {diag5.get('failed_with_skeleton_hit', 0)}",
        f"- top5 candidate failure counts: `{json.dumps(summary['error_breakdown']['current_order_top5']['candidate_failure_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "### Geometry Source @5",
        "",
        "| source | attempts | matched | strict_valid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, row in summary["error_breakdown"]["current_order_top5"]["geometry_source_breakdown"].items():
        lines.append(
            f"| {source} | {int(row.get('attempts', 0))} | {int(row.get('matched', 0))} | {int(row.get('strict_valid', 0))} |"
        )
    lines += [
        "",
        "### Complex Subsets @5",
        "",
        "| subset | samples | match@1 | match@5 | failed | failed_with_WA_hit | failed_without_WA_hit | failed_with_skeleton_hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset, item in summary["error_breakdown"]["complex_subset_breakdown_top5"].items():
        diag = item["failure_top5"]["sample_diagnostics"]
        lines.append(
            f"| {subset} | {int(item['samples'])} | {pct(item['top1']['match_at_k'])} | {pct(item['top5']['match_at_k'])} | "
            f"{int(diag.get('failed', 0))} | {int(diag.get('failed_with_WA_hit', 0))} | "
            f"{int(diag.get('failed_without_WA_hit', 0))} | {int(diag.get('failed_with_skeleton_hit', 0))} |"
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"1. current top1/top5 相比旧 WA20xgeom1 first1000：structured match@1 高 {old_delta_m1 * 100:.2f} pp，match@5 高 {old_delta_m5 * 100:.2f} pp；match@1 只是小幅提升，match@5 基本持平，但 RMSE@1/@5 明显更低。",
        "2. top5 预算更应该给 WA diversity。current top5 最高，因为它保留 WA5 优先，同时在少数样本不足 5 个 WA 时补后续 geometry variant；纯 `WA5 x geom1` 次之。`WA3 x geom2`、`WA2 x geom3`、`WA1 x geom4` 随 WA 数减少明显下降，说明 K<=5 下 WA 覆盖比 geometry diversity 更关键。",
        "3. 简单 rerank 可以显著提升 diagnostic match@1，但主要依赖 strict_valid/readable/bond 等 evaluator-derived fields，属于 oracle/diagnostic，不是真实 inference 结果。非 oracle 的 current order 仍是可信 baseline。",
        f"4. current full-test match@1 = {pct(cur_full['top1']['match_at_k'])}，距离 CrystaLLM a MP-20 55.85% 还差 {abs(delta_crystal) * 100:.2f} pp。",
        f"5. current RMSE@1/@5 = {fmt_float(cur_full['top1']['RMSE'])}/{fmt_float(cur_full['top5']['RMSE'])}，仍高于 CrystaLLM a RMSE@1 = {fmt_float(crystal['rmse1'])}；结构质量仍偏弱。",
        "6. 下一步 K<=5 小实验建议保持 `WA5` 覆盖，做 inference-feasible rerank/score 改进：优先改 WA scorer + 不依赖 evaluator label 的 geometry-quality predictor；不要把预算改成 `WA3/WA2/WA1` 去换更多 geometry variants。",
        "7. 尚未达到扩大到 MPTS-52 或重新考虑 match@20 的条件。full-test match@1 仍低于 CrystaLLM a，top5 rerank 的有效提升目前主要是 diagnostic。",
        "",
        "## Artifacts",
        "",
        f"- candidate-level table: `{summary['artifacts']['candidate_level_table']}`",
        f"- summary json: `{summary['artifacts']['summary_json']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("reports/symcif_v4_table3_fixed_full_rerun_stablekey_hybrid/mp20/wa5_geom4_stablekey_hybrid"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/structured_symcif_v4_mp20"))
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir if args.run_dir.is_absolute() else project / args.run_dir
    data_root = args.data_root if args.data_root.is_absolute() else project / args.data_root
    records = read_jsonl(data_root / "test.jsonl")
    top20_records = read_jsonl(run_dir / "top20_predictions.jsonl")
    if len(records) != len(top20_records):
        raise RuntimeError(f"record/top20 length mismatch: {len(records)} != {len(top20_records)}")

    enriched_records: list[dict[str, Any]] = []
    for record, top in zip(records, top20_records):
        item = dict(record)
        item["gt_wa_multiset_key"] = top.get("gt_wa_multiset_key")
        item["gt_skeleton_multiset_key"] = top.get("gt_skeleton_multiset_key")
        enriched_records.append(item)

    candidate_rows: list[dict[str, Any]] = []
    candidate_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in iter_jsonl(run_dir / "metrics.jsonl"):
        sample_index = int(metric["sample_index"])
        row = to_candidate_row(metric, enriched_records[sample_index])
        candidate_rows.append(row)
        candidate_by_sample[sample_index].append(row)
    for rows in candidate_by_sample.values():
        rows.sort(key=lambda r: int(r["candidate_rank"]))

    structured_indexes = set(range(len(enriched_records)))
    audit = json.loads((run_dir / "artifact_audit.json").read_text(encoding="utf-8"))
    full_denominator = int(audit.get("expected_full_test_samples") or 9046)

    current_selected = by_current_rank(candidate_by_sample, 5)
    current = {
        "structured_only": {
            "top1": summarize_selection(current_selected, structured_indexes, k=1, denominator_samples=len(structured_indexes)),
            "top5": summarize_selection(current_selected, structured_indexes, k=5, denominator_samples=len(structured_indexes)),
        },
        "full_test": {
            "top1": summarize_selection(current_selected, structured_indexes, k=1, denominator_samples=full_denominator),
            "top5": summarize_selection(current_selected, structured_indexes, k=5, denominator_samples=full_denominator),
        },
    }
    for top in ("top1", "top5"):
        current["full_test"][top]["structured_failures_counted_as_failure"] = full_denominator - len(structured_indexes)

    budget_defs = {
        "WA5_x_geom1": (5, 1),
        "WA3_x_geom2_select_top5": (3, 2),
        "WA2_x_geom3_select_top5": (2, 3),
        "WA1_x_geom4": (1, 4),
        "current_WA5_x_geom4_current_top5": (5, 4),
    }
    budget_summary = {"structured_only": {}, "full_test": {}}
    for name, (wa_count, geom_count) in budget_defs.items():
        selected = current_selected if name == "current_WA5_x_geom4_current_top5" else by_budget(candidate_by_sample, wa_count, geom_count, 5)
        budget_summary["structured_only"][name] = {
            "top1": summarize_selection(selected, structured_indexes, k=1, denominator_samples=len(structured_indexes)),
            "top5": summarize_selection(selected, structured_indexes, k=5, denominator_samples=len(structured_indexes)),
        }
        budget_summary["full_test"][name] = {
            "top1": summarize_selection(selected, structured_indexes, k=1, denominator_samples=full_denominator),
            "top5": summarize_selection(selected, structured_indexes, k=5, denominator_samples=full_denominator),
        }

    scorers: dict[str, tuple[str, str, Callable[[dict[str, Any]], tuple[Any, ...]]]] = {
        "current_order": ("baseline", "no evaluator-derived fields", scorer_current),
        "validity_first": ("oracle/diagnostic", "uses strict_valid/readable/formula_ok/SG_ok", scorer_validity_first),
        "geometry_quality": ("oracle/diagnostic", "uses bond_length_score plus prototype source/reference score", scorer_geometry_quality),
        "hybrid": ("oracle/diagnostic", "uses strict_valid, bond proxy, WA rank/score", scorer_hybrid),
    }
    rerank_summary = {"structured_only": {}, "full_test": {}}
    for name, (typ, fields, scorer) in scorers.items():
        selected = by_scorer(candidate_by_sample, scorer, 5)
        item_struct = {
            "type": typ,
            "fields": fields,
            "metrics": {
                "top1": summarize_selection(selected, structured_indexes, k=1, denominator_samples=len(structured_indexes)),
                "top5": summarize_selection(selected, structured_indexes, k=5, denominator_samples=len(structured_indexes)),
            },
        }
        item_full = {
            "type": typ,
            "fields": fields,
            "metrics": {
                "top1": summarize_selection(selected, structured_indexes, k=1, denominator_samples=full_denominator),
                "top5": summarize_selection(selected, structured_indexes, k=5, denominator_samples=full_denominator),
            },
        }
        rerank_summary["structured_only"][name] = item_struct
        rerank_summary["full_test"][name] = item_full

    subset_names = [
        "overall",
        "n_sites>=6",
        "n_sites>=12",
        "num_elements>=4",
        "rare_sg",
        "high_multiplicity_orbit",
        "extraction_hard",
    ]
    error_breakdown = {
        "current_order_top1": failure_breakdown(current_selected, structured_indexes, 1),
        "current_order_top5": failure_breakdown(current_selected, structured_indexes, 5),
        "complex_subset_breakdown_top5": subset_breakdowns(enriched_records, current_selected, subset_names),
    }

    table_path = run_dir / "candidate_level_table_top1_top5_replay.jsonl"
    summary_json_path = run_dir / "top1_top5_rerank_replay_summary.json"
    summary_md_path = run_dir / "top1_top5_rerank_replay_summary.md"
    write_jsonl(table_path, candidate_rows)

    summary = {
        "scope": {
            "dataset": "MP-20",
            "structured_samples": len(structured_indexes),
            "full_test_samples": full_denominator,
            "non_structured_failures": full_denominator - len(structured_indexes),
            "no_new_render": True,
            "no_new_structure_matcher": True,
            "no_match20_main_table": True,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "candidate_level_table": str(table_path),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
        "current_order": current,
        "budget_replay": budget_summary,
        "rerank_replay": rerank_summary,
        "error_breakdown": error_breakdown,
        "references": {
            "old_wa20_geom1_first1000": OLD_WA20_GEOM1_FIRST1000,
            "published_crystallm_a_mp20": PUBLISHED_CRYSTALLM_A_MP20,
        },
    }
    write_json(summary_json_path, summary)
    summary_md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(f"wrote {summary_json_path}")
    print(f"wrote {summary_md_path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
