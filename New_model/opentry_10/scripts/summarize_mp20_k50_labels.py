#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "labels/mp20_val_k50_candidate_labels.jsonl"
DEFAULT_TRUE_TAR = ROOT / "generations/crystallm_gt_sg_val_anchor_symprec0p1/mp20_val_data_atomtype_gt_sg_symprec0p1_k100/tars/true.tar.gz"


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def atom_site_rows(cif: str) -> int:
    in_atom_loop = False
    headers_seen = False
    count = 0
    for line in cif.splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            if in_atom_loop and headers_seen:
                break
            in_atom_loop = False
            headers_seen = False
            continue
        if stripped.startswith("_atom_site_"):
            in_atom_loop = True
            headers_seen = True
            continue
        if in_atom_loop and headers_seen:
            if not stripped or stripped.startswith("_") or stripped.startswith("#"):
                break
            count += 1
    return count


def read_target_rows(true_tar: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with tarfile.open(true_tar, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            cif = handle.read().decode("utf-8", errors="replace")
            material_id = Path(member.name).stem
            rows[material_id] = atom_site_rows(cif)
    return rows


def summarize(labels_path: Path, target_rows: dict[str, int], max_rank: int, budgets: list[int]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    rows_by_material: Counter[str] = Counter()
    parse_known = parse_ok = 0
    valid_known = valid_ok = 0
    sensible_known = sensible_ok = 0

    hits_by_material: dict[str, dict[int, bool]] = defaultdict(lambda: {k: False for k in budgets})
    rms_by_material: dict[str, dict[int, list[float]]] = defaultdict(lambda: {k: [] for k in budgets})

    label_rows = 0
    first_material = None
    last_material = None
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            label_rows += 1
            material_id = str(row.get("material_id"))
            rank = int(row.get("rank", 0))
            status = str(row.get("label_status"))
            label_counts[status] += 1
            rank_counts[rank] += 1
            rows_by_material[material_id] += 1
            if first_material is None:
                first_material = material_id
            last_material = material_id

            value = row.get("parse_ok")
            if value is not None:
                parse_known += 1
                parse_ok += int(bool(value))
            value = row.get("valid")
            if value is not None:
                valid_known += 1
                valid_ok += int(bool(value))
            value = row.get("sensible")
            if value is not None:
                sensible_known += 1
                sensible_ok += int(bool(value))

            if bool(row.get("match")):
                rms = row.get("rmsd")
                for budget in budgets:
                    if rank <= budget:
                        hits_by_material[material_id][budget] = True
                        if rms is not None:
                            rms_by_material[material_id][budget].append(float(rms))

    expected_materials = len(target_rows)
    expected_rows = expected_materials * int(max_rank)
    covered_materials = set(rows_by_material)
    complete_materials = sorted(mid for mid, count in rows_by_material.items() if count == int(max_rank))
    incomplete_materials = sorted(mid for mid, count in rows_by_material.items() if count != int(max_rank))
    rows_ge7_materials = {mid for mid, rows in target_rows.items() if rows >= 7}
    complete_rows_ge7 = [mid for mid in complete_materials if mid in rows_ge7_materials]

    def rate(num: int, den: int) -> float | None:
        return None if den == 0 else float(num / den)

    def metric_for(materials: list[str], budget: int) -> dict[str, Any]:
        hit_mids = [mid for mid in materials if hits_by_material[mid][budget]]
        rms_values = []
        for mid in hit_mids:
            values = rms_by_material[mid][budget]
            if values:
                rms_values.append(min(values))
        return {
            "materials": len(materials),
            "hits": len(hit_mids),
            "match_rate": rate(len(hit_mids), len(materials)),
            "rms_dist": None if not rms_values else float(sum(rms_values) / len(rms_values)),
        }

    complete_metrics = {f"k{budget}": metric_for(complete_materials, budget) for budget in budgets}
    rows_ge7_metrics = {f"k{budget}": metric_for(complete_rows_ge7, budget) for budget in budgets}
    expected_missing_fail = {}
    all_expected_ids = sorted(target_rows)
    for budget in budgets:
        hit_mids = [mid for mid in all_expected_ids if hits_by_material[mid][budget]]
        expected_missing_fail[f"k{budget}"] = {
            "materials": expected_materials,
            "hits": len(hit_mids),
            "match_rate": rate(len(hit_mids), expected_materials),
        }

    k20_hits = {mid for mid in complete_materials if hits_by_material[mid][20]}
    k50_hits = {mid for mid in complete_materials if hits_by_material[mid][50]}
    rescue_20_to_50 = sorted(k50_hits - k20_hits)

    return {
        "created_at": now_iso(),
        "labels": str(labels_path.resolve()),
        "label_rows": label_rows,
        "expected_rows": expected_rows,
        "max_rank": int(max_rank),
        "coverage_complete": label_rows == expected_rows and len(complete_materials) == expected_materials,
        "target_materials": expected_materials,
        "covered_materials": len(covered_materials),
        "complete_materials": len(complete_materials),
        "incomplete_materials": len(incomplete_materials),
        "first_incomplete_materials": incomplete_materials[:20],
        "first_material_seen": first_material,
        "last_material_seen": last_material,
        "target_rows_ge7_materials": len(rows_ge7_materials),
        "complete_rows_ge7_materials": len(complete_rows_ge7),
        "label_status_counts": dict(sorted(label_counts.items())),
        "rank_counts": {str(k): rank_counts[k] for k in sorted(rank_counts)},
        "candidate_parse_rate_known": rate(parse_ok, parse_known),
        "candidate_valid_rate_known": rate(valid_ok, valid_known),
        "candidate_sensible_rate_known": rate(sensible_ok, sensible_known),
        "metrics_over_complete_materials": complete_metrics,
        "rows_ge7_metrics_over_complete_materials": rows_ge7_metrics,
        "metrics_over_expected_materials_missing_as_fail": expected_missing_fail,
        "k50_rescues_over_k20_complete_materials": len(rescue_20_to_50),
        "first_k50_rescue_materials": rescue_20_to_50[:50],
    }


def format_rate(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.3f}%"


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MP-20 K50 Label Summary",
        "",
        f"Created: {summary['created_at']}",
        "",
        f"- Label rows: {summary['label_rows']} / {summary['expected_rows']}",
        f"- Complete materials: {summary['complete_materials']} / {summary['target_materials']}",
        f"- Coverage complete: {summary['coverage_complete']}",
        f"- Target rows>=7 materials: {summary['target_rows_ge7_materials']}",
        f"- Complete rows>=7 materials: {summary['complete_rows_ge7_materials']}",
        f"- K50 rescues over K20 among complete materials: {summary['k50_rescues_over_k20_complete_materials']}",
        "",
        "## Complete-Material Metrics",
        "",
        "| budget | match_rate | RMSE | hits/materials | rows>=7 match_rate | rows>=7 hits/materials |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    complete = summary["metrics_over_complete_materials"]
    rows7 = summary["rows_ge7_metrics_over_complete_materials"]
    for key in ["k1", "k5", "k20", "k50"]:
        metric = complete[key]
        bucket = rows7[key]
        rms = metric["rms_dist"]
        lines.append(
            f"| {key.upper()} | {format_rate(metric['match_rate'])} | "
            f"{'NA' if rms is None else f'{rms:.10f}'} | "
            f"{metric['hits']}/{metric['materials']} | "
            f"{format_rate(bucket['match_rate'])} | {bucket['hits']}/{bucket['materials']} |"
        )
    lines.extend(
        [
            "",
            "If coverage is incomplete, these are progress diagnostics over complete materials only. Formal selector gates require coverage_complete=true.",
            "",
            "## Label Status Counts",
            "",
            "```json",
            json.dumps(summary["label_status_counts"], indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MP-20 K50 per-candidate labels, including target rows>=7 buckets.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--true-tar", default=str(DEFAULT_TRUE_TAR))
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "metrics/mp20_val_k50_candidate_label_metrics.json"))
    parser.add_argument("--report", default=str(ROOT / "reports/mp20_k50_label_summary.md"))
    args = parser.parse_args()

    labels = Path(args.labels)
    target_rows = read_target_rows(Path(args.true_tar))
    summary = summarize(labels, target_rows, int(args.max_rank), budgets=[1, 5, 20, 50])
    summary["true_tar"] = str(Path(args.true_tar).resolve())
    summary["note"] = "Formal selector gates require coverage_complete=true. Use --allow-partial only for progress diagnostics."
    if not summary["coverage_complete"] and not args.allow_partial:
        raise SystemExit(
            f"label coverage incomplete: {summary['label_rows']} / {summary['expected_rows']} rows; "
            "rerun with --allow-partial for a diagnostic snapshot"
        )

    write_json(Path(args.out), summary)
    write_text(Path(args.report), make_report(summary))


if __name__ == "__main__":
    main()
