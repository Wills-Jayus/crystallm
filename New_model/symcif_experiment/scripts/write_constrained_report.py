#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


STAGE_ORDER = [
    ("schema_invalid", "schema parse / column length"),
    ("sg_missing", "space-group missing"),
    ("sg_invalid", "space-group invalid"),
    ("sg_not_target", "space-group equals prompt target"),
    ("wyckoff_invalid", "Wyckoff letter legal in SG"),
    ("coord_invalid", "free/fixed coordinate mask valid"),
    ("formula_not_closed", "formula closure"),
    ("cell_missing", "cell parameters present"),
    ("cell_invalid", "cell parameters valid"),
    ("render_cif_failed", "rendered CIF readable"),
]


def read_csv_summary(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[(row["mode"], int(float(row["n"])))] = row
    return rows


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def fmt_pct(value: Any) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "N/A"
    return f"{100 * x:.2f}%"


def fmt_num(value: Any, digits: int = 4) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_failed_cases(path: Path) -> tuple[Counter[str], Counter[str]]:
    stage_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    if not path.exists():
        return stage_counts, reason_counts
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = str(row.get("stage") or "unknown")
            reason = str(row.get("reason") or "unknown")
            stage_counts[stage] += 1
            reason_counts[f"{stage}: {reason}"] += 1
    return stage_counts, reason_counts


def row_for(summary: dict[tuple[str, int], dict[str, Any]], mode: str, n: int) -> dict[str, Any]:
    return summary.get((mode, n), {"mode": mode, "n": n})


def update_repair_report(repair_path: Path, constrained_summary: dict[tuple[str, int], dict[str, Any]]) -> None:
    if not repair_path.exists():
        return
    report = load_json(repair_path)
    n1 = row_for(constrained_summary, "symcif_v1_constrained", 1)
    n20 = row_for(constrained_summary, "symcif_v1_constrained", 20)
    report["final_valid"] = as_float(n20.get("valid"))
    report["match_at_1"] = as_float(n1.get("match_rate_n1"))
    report["match_at_20"] = as_float(n20.get("match_rate_n20"))
    report["rmse"] = as_float(n20.get("RMSE"))
    repair_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def comparison_lines(reference: dict[tuple[str, int], dict[str, Any]], constrained: dict[tuple[str, int], dict[str, Any]]) -> list[str]:
    rows = [
        ("baseline", 1, row_for(reference, "baseline", 1)),
        ("baseline", 20, row_for(reference, "baseline", 20)),
        ("symcif_v1(raw)", 1, row_for(reference, "symcif_v1", 1)),
        ("symcif_v1(raw)", 20, row_for(reference, "symcif_v1", 20)),
        ("symcif_v1_constrained", 1, row_for(constrained, "symcif_v1_constrained", 1)),
        ("symcif_v1_constrained", 20, row_for(constrained, "symcif_v1_constrained", 20)),
    ]
    lines = [
        "| mode | n | parse | to CIF | readable | formula_ok | sg_ok | valid | match | RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, n, row in rows:
        match_key = "match_rate_n1" if n == 1 else "match_rate_n20"
        lines.append(
            "| {mode} | {n} | {parse} | {conv} | {readable} | {formula} | {sg} | {valid} | {match} | {rmse} |".format(
                mode=name,
                n=n,
                parse=fmt_pct(row.get("parse_success")),
                conv=fmt_pct(row.get("symcif_to_cif_success")),
                readable=fmt_pct(row.get("pymatgen_readable")),
                formula=fmt_pct(row.get("formula_ok")),
                sg=fmt_pct(row.get("space_group_ok")),
                valid=fmt_pct(row.get("valid")),
                match=fmt_pct(row.get(match_key)),
                rmse=fmt_num(row.get("RMSE")),
            )
        )
    return lines


def stage_lines(total: int, stage_counts: Counter[str], successes: int) -> list[str]:
    lines = [
        "| gate | failed here | cumulative pass | cumulative pass rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    remaining = total
    accounted: set[str] = set()
    for stage, label in STAGE_ORDER:
        failed = stage_counts.get(stage, 0)
        accounted.add(stage)
        remaining -= failed
        lines.append(f"| {label} | {failed} | {remaining} | {fmt_pct(remaining / total if total else math.nan)} |")
    other_failed = sum(count for stage, count in stage_counts.items() if stage not in accounted)
    if other_failed:
        remaining -= other_failed
        lines.append(f"| other | {other_failed} | {remaining} | {fmt_pct(remaining / total if total else math.nan)} |")
    if successes != remaining:
        lines.append(f"| final accounting check | N/A | {successes} | {fmt_pct(successes / total if total else math.nan)} |")
    return lines


def verdict(reference: dict[tuple[str, int], dict[str, Any]], constrained: dict[tuple[str, int], dict[str, Any]]) -> list[str]:
    raw = row_for(reference, "symcif_v1", 20)
    cons = row_for(constrained, "symcif_v1_constrained", 20)
    raw_match = as_float(raw.get("match_rate_n20"))
    cons_match = as_float(cons.get("match_rate_n20"))
    raw_valid = as_float(raw.get("valid"))
    cons_valid = as_float(cons.get("valid"))
    raw_formula = as_float(raw.get("formula_ok"))
    cons_formula = as_float(cons.get("formula_ok"))
    lines = []
    if not math.isnan(cons_match) and not math.isnan(raw_match):
        delta = cons_match - raw_match
        direction = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
        lines.append(f"- constrained 相对 raw SymCIF 的 n=20 match rate {direction} {100 * delta:.2f} 个百分点。")
    if not math.isnan(cons_valid) and not math.isnan(raw_valid):
        delta = cons_valid - raw_valid
        direction = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
        lines.append(f"- constrained 相对 raw SymCIF 的 valid rate {direction} {100 * delta:.2f} 个百分点。")
    if not math.isnan(cons_formula) and not math.isnan(raw_formula):
        delta = cons_formula - raw_formula
        direction = "提升" if delta > 0 else "下降" if delta < 0 else "持平"
        lines.append(
            f"- formula_ok {direction} {100 * delta:.2f} 个百分点；这里的分母仍是全部 10000 次尝试，"
            "所以大量保守拒绝会压低总成功率。"
        )
    lines.append("- 该约束器只修正 Wyckoff 派生字段、固定坐标和晶系晶格约束；不增删元素/位点，也不替换生成元素，因此 match 的提升上限受 raw generation 的位点/成分正确率限制。")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the constrained SymCIF repair/eval report.")
    parser.add_argument(
        "--constrained-dir",
        type=Path,
        default=Path("eval_runs/generation_eval_t1_topk10_n20_20260520_constrained"),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path("eval_runs/generation_eval_t1_topk10_n20_20260520_fixedgt/summary.csv"),
    )
    parser.add_argument("--update-repair-report", action="store_true")
    args = parser.parse_args()

    constrained_summary = read_csv_summary(args.constrained_dir / "summary.csv")
    reference_summary = read_csv_summary(args.reference_summary)
    repair_path = args.constrained_dir / "repair_report.json"
    failed_path = args.constrained_dir / "failed_repair_cases.jsonl"
    repair = load_json(repair_path)
    stage_counts, reason_counts = read_failed_cases(failed_path)
    if args.update_repair_report:
        update_repair_report(repair_path, constrained_summary)
        repair = load_json(repair_path)

    total = int(repair.get("total_generations") or (repair.get("successes", 0) + sum(stage_counts.values())))
    successes = int(repair.get("successes") or 0)
    report_path = args.constrained_dir / "generation_eval_report_constrained.md"
    lines = [
        "# SymCIF constrained repair/eval report",
        "",
        "## 1. Scope",
        "",
        "- 本实验只对既有 `symcif_v1` 生成文本做保守约束修复/拒绝，没有重新训练，也没有重新生成。",
        "- 修复过程使用的信息：生成文本本身、prompt 中的公式和空间群、Wyckoff lookup 表、CIF/Structure 可读性规则。",
        "- 修复过程不使用 GT 原子坐标、GT 晶胞参数或 GT 匹配结果；prompt 公式/空间群属于生成条件。",
        "- 约束策略只允许规范化 schema、空间群、Wyckoff multiplicity/site symmetry/enumeration、FIXED/free coordinate mask、晶系晶格角/轴约束。",
        "- 遇到成分不闭合、空间群不等于 prompt、非法 Wyckoff、free coordinate 缺失或 CIF 不可读时直接拒绝，不做增删位点或元素替换。",
        "",
        "## 2. Repair Gates",
        "",
    ]
    lines.extend(stage_lines(total, stage_counts, successes))
    lines.extend(
        [
            "",
            "## 3. Evaluation Comparison",
            "",
        ]
    )
    lines.extend(comparison_lines(reference_summary, constrained_summary))
    lines.extend(
        [
            "",
            "## 4. Main Findings",
            "",
        ]
    )
    lines.extend(verdict(reference_summary, constrained_summary))
    lines.extend(
        [
            "",
            "## 5. Top Failure Reasons",
            "",
            "| rank | reason | count |",
            "| ---: | --- | ---: |",
        ]
    )
    for rank, (reason, count) in enumerate(reason_counts.most_common(20), start=1):
        lines.append(f"| {rank} | `{reason}` | {count} |")
    lines.extend(
        [
            "",
            "## 6. Output Files",
            "",
            f"- repaired/rejected generations: `{args.constrained_dir / 'generations_repaired' / 'symcif_v1_constrained.jsonl'}`",
            f"- repair report JSON: `{repair_path}`",
            f"- failed repair cases: `{failed_path}`",
            f"- standard CIFs for repaired generations: `{args.constrained_dir / 'standard_cifs'}`",
            f"- evaluator summary: `{args.constrained_dir / 'summary.csv'}`",
            f"- evaluator metrics: `{args.constrained_dir / 'metrics'}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
