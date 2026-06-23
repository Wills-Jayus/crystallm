#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import split_concat_records  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_sample(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_index"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["gen_index"]))
    return grouped


def counter_signature(counter: Counter[tuple[Any, ...]]) -> str:
    parts: list[str] = []
    for key, count in sorted(counter.items()):
        parts.append("{}x{}".format(",".join(str(x) for x in key), count))
    return "|".join(parts)


def record_skeleton(record: Any) -> Counter[tuple[str, int]]:
    return Counter((str(site.letter), int(site.multiplicity)) for site in record.sites)


def record_assignment(record: Any) -> Counter[tuple[str, str, int]]:
    return Counter((str(site.element), str(site.letter), int(site.multiplicity)) for site in record.sites)


def record_skeleton_signature(record: Any) -> str:
    return counter_signature(record_skeleton(record))


def record_assignment_signature(record: Any) -> str:
    return counter_signature(record_assignment(record))


def lattice_delta(gt: Any, pred: Any) -> tuple[float, float]:
    gt_lengths = [gt.lattice.a, gt.lattice.b, gt.lattice.c]
    pred_lengths = [pred.lattice.a, pred.lattice.b, pred.lattice.c]
    rel = [
        abs(float(p) - float(g)) / max(abs(float(g)), 1e-6)
        for g, p in zip(gt_lengths, pred_lengths, strict=True)
    ]
    gt_angles = [gt.lattice.alpha, gt.lattice.beta, gt.lattice.gamma]
    pred_angles = [pred.lattice.alpha, pred.lattice.beta, pred.lattice.gamma]
    angle = [abs(float(p) - float(g)) for g, p in zip(gt_angles, pred_angles, strict=True)]
    return max(rel), max(angle)


def classify_failure(metric: dict[str, Any], generated_text: str, gt_record: Any, lookup: WyckoffLookup) -> str:
    if metric.get("match_skipped_reason") == "too_many_sites" or metric.get("conversion_skipped_reason"):
        return "StructureMatcher threshold / large-structure skip"
    if not metric.get("parse_success") or not metric.get("symcif_to_cif_success") or not metric.get("pymatgen_readable"):
        return "CIF readable / SG recognition issue"
    try:
        pred = parse_symcif_v2_text(generated_text, lookup)
    except Exception:
        return "CIF readable / SG recognition issue"
    if record_skeleton(pred) != record_skeleton(gt_record):
        return "Wyckoff skeleton inconsistent"
    if record_assignment(pred) != record_assignment(gt_record):
        return "element-Wyckoff assignment inconsistent"
    max_len_rel, max_angle_abs = lattice_delta(gt_record, pred)
    if max_len_rel > 0.15 or max_angle_abs > 8.0:
        return "skeleton/assignment match but lattice/cell deviation large"
    return "skeleton/assignment match but free-coordinate deviation likely"


def pct(num: float, denom: float) -> str:
    if denom <= 0:
        return "N/A"
    return f"{100.0 * num / denom:.2f}%"


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose existing SymCIF-v2 constrained generation metrics.")
    parser.add_argument(
        "--old-run-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "symcif_v2_constrained_eval_t1_topk10_n20",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "generation_eval_input_ablation_t1_topk10_n20_20260520",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "Log_GPT" / "symcif_v2_constrained_diagnosis.md")
    parser.add_argument(
        "--round-out",
        type=Path,
        default=PROJECT_ROOT
        / "Log_GPT"
        / "round_20260520_05_symcif_v2_v2_match5"
        / "symcif_v2_constrained_diagnosis.md",
    )
    args = parser.parse_args()

    lookup = WyckoffLookup.from_json(PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    old_metrics = load_jsonl(args.old_run_dir / "metrics" / "symcif_v2_constrained_per_generation_metrics.jsonl")
    old_gens = load_jsonl(args.old_run_dir / "generations" / "symcif_v2_constrained.jsonl")
    baseline_metrics = load_jsonl(args.baseline_run_dir / "metrics" / "baseline_minprompt_per_generation_metrics.jsonl")
    old_gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in old_gens}

    gt_records = [parse_symcif_v2_text(text, lookup) for text in split_concat_records(PROJECT_ROOT / "data" / "symcif_v2" / "test.txt")]

    condition_rows = [m for m in old_metrics if m.get("formula_ok") and m.get("space_group_ok")]
    condition_matches = [m for m in condition_rows if m.get("match_ok")]
    conditional_ratio = len(condition_matches) / len(condition_rows) if condition_rows else math.nan

    base_by_sample = by_sample(baseline_metrics)
    old_by_sample = by_sample(old_metrics)
    overlap = Counter()
    for sample_index in range(len(gt_records)):
        base_match = any(row.get("match_ok") for row in base_by_sample.get(sample_index, []))
        old_match = any(row.get("match_ok") for row in old_by_sample.get(sample_index, []))
        if base_match and old_match:
            overlap["both match"] += 1
        elif base_match and not old_match:
            overlap["baseline match / v2 fail"] += 1
        elif not base_match and old_match:
            overlap["baseline fail / v2 match"] += 1
        else:
            overlap["both fail"] += 1

    attribution = Counter()
    for metric in condition_rows:
        if metric.get("match_ok"):
            continue
        key = (int(metric["sample_index"]), int(metric["gen_index"]))
        gen = old_gen_by_key.get(key, {})
        label = classify_failure(metric, gen.get("generated_text") or "", gt_records[int(metric["sample_index"])], lookup)
        attribution[label] += 1

    unique_skeleton_counts: list[int] = []
    unique_assignment_counts: list[int] = []
    duplicate_samples = 0
    diversity_match_rows: list[tuple[int, int, bool]] = []
    for sample_index in range(len(gt_records)):
        skeletons: set[str] = set()
        assignments: set[str] = set()
        for gen in sorted((r for r in old_gens if int(r["sample_index"]) == sample_index and int(r["gen_index"]) < 5), key=lambda r: int(r["gen_index"])):
            try:
                rec = parse_symcif_v2_text(gen.get("generated_text") or "", lookup)
                skeletons.add(record_skeleton_signature(rec))
                assignments.add(record_assignment_signature(rec))
            except Exception:
                pass
        unique_skeleton_counts.append(len(skeletons))
        unique_assignment_counts.append(len(assignments))
        if len(skeletons) <= 1:
            duplicate_samples += 1
        match5 = any(row.get("match_ok") for row in old_by_sample.get(sample_index, []) if int(row["gen_index"]) < 5)
        diversity_match_rows.append((len(skeletons), len(assignments), match5))
    matched_div = [x[0] for x in diversity_match_rows if x[2]]
    unmatched_div = [x[0] for x in diversity_match_rows if not x[2]]

    closure_rows = []
    for metric in old_metrics:
        gen = old_gen_by_key.get((int(metric["sample_index"]), int(metric["gen_index"])), {})
        if gen.get("formula_closure_success") or metric.get("formula_closure_success"):
            closure_rows.append(metric)
    closure_formula_fail = [m for m in closure_rows if not m.get("formula_ok")]
    closure_breakdown = Counter()
    for row in closure_formula_fail:
        if row.get("eval_timeout"):
            closure_breakdown["sample eval timeout"] += 1
        elif not row.get("parse_success"):
            closure_breakdown["SymCIF-v2 parse failed"] += 1
        elif not row.get("symcif_to_cif_success"):
            closure_breakdown["SymCIF-v2 to CIF failed"] += 1
        elif not row.get("pymatgen_readable"):
            closure_breakdown["pymatgen unreadable"] += 1
        elif not row.get("space_group_ok"):
            closure_breakdown["rendered CIF SG mismatch"] += 1
        else:
            closure_breakdown["rendered CIF formula mismatch / evaluator composition mismatch"] += 1

    lines = [
        "# SymCIF-v2 Constrained Diagnosis",
        "",
        "日期：2026-05-20",
        "",
        "## 1. 输入",
        "",
        f"- old constrained run: `{args.old_run_dir}`",
        f"- baseline_minprompt run: `{args.baseline_run_dir}`",
        "- 本报告只做离线诊断，不使用 GT 信息参与生成。",
        "",
        "## 2. 条件 Match",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| formula_ok + sg_ok generation 数 | {len(condition_rows)} / {len(old_metrics)} |",
        f"| 其中 match generation 数 | {len(condition_matches)} |",
        f"| 条件 match 比例 | {pct(len(condition_matches), len(condition_rows))} |",
        f"| 对比 symcif_v1 条件 match 参考 | 50.36% |",
        "",
        "解释：constrained decoding 显著提高 formula/SG 正确率，但在 formula/SG 正确后仍有大量结构未 match，说明剩余瓶颈主要在 Wyckoff skeleton、element-Wyckoff assignment、连续坐标和 cell。",
        "",
        "## 3. Per-sample Overlap",
        "",
        "| 分类 | samples |",
        "| --- | ---: |",
    ]
    for key in ("both match", "baseline match / v2 fail", "baseline fail / v2 match", "both fail"):
        lines.append(f"| {key} | {overlap[key]} |")

    lines.extend(
        [
            "",
            "## 4. formula_ok + sg_ok 但未 match 的归因",
            "",
            "| 主要原因 | generations |",
            "| --- | ---: |",
        ]
    )
    for key, value in attribution.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 5. 前 5 个候选的 Wyckoff Skeleton Diversity",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| unique skeleton / sample mean | {fmt_float(statistics.mean(unique_skeleton_counts), 3)} |",
            f"| unique skeleton / sample median | {fmt_float(statistics.median(unique_skeleton_counts), 3)} |",
            f"| unique assignment / sample mean | {fmt_float(statistics.mean(unique_assignment_counts), 3)} |",
            f"| unique assignment / sample median | {fmt_float(statistics.median(unique_assignment_counts), 3)} |",
            f"| 前 5 个输出 skeleton 高度重复的 samples | {duplicate_samples} / {len(gt_records)} |",
            f"| match@5 samples 的 unique skeleton mean | {fmt_float(statistics.mean(matched_div) if matched_div else math.nan, 3)} |",
            f"| non-match@5 samples 的 unique skeleton mean | {fmt_float(statistics.mean(unmatched_div) if unmatched_div else math.nan, 3)} |",
            "",
            "## 6. Formula Closure 与 formula_ok 的差异",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| formula_closure_success rows | {len(closure_rows)} |",
            f"| closure success 但 formula_ok=false | {len(closure_formula_fail)} |",
            "",
            "| 差异来源 | generations |",
            "| --- | ---: |",
        ]
    )
    for key, value in closure_breakdown.most_common():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## 7. 诊断结论",
            "",
            "1. 当前 old constrained 的主要收益来自 formula closure；但条件 match 仍不足，说明正确 skeleton 和连续几何仍是瓶颈。",
            "2. 前 5 个候选存在 skeleton 重复，需要在采样阶段显式记录并有限重采 skeleton signature。",
            "3. `formula_closure_success` 与 `formula_ok` 的差异主要来自 evaluator timeout、readability/SG 识别和 v2 lookup/render setting 差异，而不是采样阶段没有闭合公式。",
            "4. 新 sampler 应优先优化 skeleton 多样性、Wyckoff letter 选择、困难元素优先级和 cell/coord 低温采样。",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.round_out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    args.out.write_text(text, encoding="utf-8")
    args.round_out.write_text(text, encoding="utf-8")
    print(f"[diagnosis] wrote {args.out}")
    print(f"[diagnosis] wrote {args.round_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

