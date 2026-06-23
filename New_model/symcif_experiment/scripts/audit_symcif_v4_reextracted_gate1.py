#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.formula import total_atoms
from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.validation import validate_cif


TOP_FAILED = ["225|24e", "189|3f", "189|3g", "193|6g", "216|16e", "216|24f"]

_ENGINE: OrbitEngine | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def init_worker(lookup_json: str, structured_root: str) -> None:
    global _ENGINE
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, structured_root)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(r["readable"]) for r in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(r["formula_ok"]) for r in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(r["atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(r["sg_ok"]) for r in rows) / max(1, len(rows)),
        "row_expansion_ok": sum(bool(r["row_expansion_ok"]) for r in rows) / max(1, len(rows)),
        "expanded_atom_count_ok": sum(bool(r["expanded_atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(r["composition_exact"]) for r in rows) / max(1, len(rows)),
    }


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        item = {key: value, **summarize(items)}
        item["failed"] = sum(
            not (r["readable"] and r["formula_ok"] and r["atom_count_ok"] and r["sg_ok"] and r["row_expansion_ok"])
            for r in items
        )
        out.append(item)
    return out


def process_record(payload: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    split, input_index, row = payload
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    engine = _ENGINE
    wa_table = list(row["wa_table"])
    free_params = {idx: dict(w.get("free_params") or {}) for idx, w in enumerate(wa_table)}
    cif = engine.render_cif_from_wa_table(
        wa_table,
        lattice=row["lattice"],
        free_params_by_row=free_params,
        formula_counts=row["formula_counts"],
        sg=int(row["sg"]),
        sg_symbol=str(row.get("sg_symbol") or f"SG{int(row['sg'])}"),
        data_name=str(row["sample_id"]),
    )
    metric = validate_cif(cif, row["formula_counts"], int(row["sg"]))
    expanded_count = 0
    degenerate_rows: list[dict[str, Any]] = []
    orbit_keys: list[str] = []
    for idx, w in enumerate(wa_table):
        orbit = engine.get_orbit_by_id(str(w["orbit_id"]))
        key = f"{orbit.sg}|{orbit.multiplicity}{orbit.letter}"
        orbit_keys.append(key)
        expanded = engine.expand_orbit(orbit, free_params[idx])
        expanded_count += len(expanded)
        if len(expanded) != int(orbit.multiplicity):
            degenerate_rows.append(
                {
                    "row_index": idx,
                    "sg_letter": key,
                    "element": w.get("element"),
                    "expanded_count": len(expanded),
                    "declared_multiplicity": int(orbit.multiplicity),
                    "free_params": free_params[idx],
                    "source_coord": w.get("source_coord"),
                    "extraction_method": w.get("extraction_method"),
                }
            )
    row_expansion_ok = not degenerate_rows
    total = int(total_atoms(row["formula_counts"]))
    item = {
        "split": split,
        "input_index": input_index,
        "sample_id": row["sample_id"],
        "sg": int(row["sg"]),
        "sg_symbol": row.get("sg_symbol"),
        "n_sites": int(row["n_sites"]),
        "num_elements": int(row["num_elements"]),
        "total_atoms": total,
        "expanded_atom_count": int(expanded_count),
        "expanded_atom_count_ok": int(expanded_count) == total,
        "row_expansion_ok": row_expansion_ok,
        "degenerate_orbit_rows": degenerate_rows,
        "orbit_keys": orbit_keys,
        "formula": row.get("formula"),
        **metric,
    }
    return item


def old_row_breakdown(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[str(row["sg_letter"])] = {"rows": int(row["rows"]), "failed": int(row["failed"])}
    return out


def write_top_failed_report(out_dir: Path, sg_letter_rows: list[dict[str, Any]], old_rows: dict[str, dict[str, int]]) -> None:
    after = {str(row["sg_letter"]): row for row in sg_letter_rows}
    def esc(text: str) -> str:
        return str(text).replace("|", "\\|")

    lines = ["# Top Failed SG/Letter After Reextract", ""]
    lines.append("| SG/letter | old failed/rows | new degenerate/occurrences | new expansion_ok | failed sample occurrence rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key in TOP_FAILED:
        old = old_rows.get(key, {})
        new = after.get(key, {})
        old_text = f"{old.get('failed', 'NA')}/{old.get('rows', 'NA')}"
        occ = int(new.get("occurrences", 0))
        deg = int(new.get("degenerate_occurrences", 0))
        ok = 1.0 - (deg / max(1, occ))
        fail_rate = float(new.get("failed_sample_occurrence_rate", 0.0))
        lines.append(f"| {esc(key)} | {old_text} | {deg}/{occ} | {ok:.4f} | {fail_rate:.4f} |")
    lines.append("")
    lines.append("Interpretation: this table checks whether the previously systematic row collapse remains after source-coordinate free_param re-extraction.")
    (out_dir / "top_failed_sg_letter_after_reextract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_cause_after_reextract(out_dir: Path, summary: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    overall = summary["overall"]
    if summary["gate1"]["passed"]:
        lines = [
            "# Root Cause After Reextract",
            "",
            "Gate 1 passed after source-coordinate free_param re-extraction. No additional failure-root-cause split is required for this round.",
        ]
        (out_dir / "root_cause_after_reextract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    classes: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        if row.get("degenerate_orbit_rows"):
            cls = "A.extraction_unsupported_or_true_degeneracy"
        elif not row.get("readable"):
            cls = "C.CIF_read_write"
        elif row.get("formula_ok") and row.get("atom_count_ok") and not row.get("sg_ok"):
            cls = "D.spglib_detection_only"
        elif not row.get("formula_ok") or not row.get("atom_count_ok"):
            cls = "B.source_label_or_free_param_convention_mismatch"
        else:
            cls = "F.unknown"
        classes[cls] += 1
        if len(examples[cls]) < 5:
            examples[cls].append(
                {
                    "sample_id": row.get("sample_id"),
                    "sg": row.get("sg"),
                    "readable": row.get("readable"),
                    "formula_ok": row.get("formula_ok"),
                    "atom_count_ok": row.get("atom_count_ok"),
                    "sg_ok": row.get("sg_ok"),
                    "degenerate_orbit_rows": row.get("degenerate_orbit_rows"),
                    "error": row.get("error"),
                }
            )
    lines = [
        "# Root Cause After Reextract",
        "",
        f"- Gate 1 passed: {summary['gate1']['passed']}",
        f"- readable/formula_ok/atom_count_ok/SG_ok/row_expansion_ok: {overall['readable']:.4f} / {overall['formula_ok']:.4f} / {overall['atom_count_ok']:.4f} / {overall['sg_ok']:.4f} / {overall['row_expansion_ok']:.4f}",
        "",
        "## Remaining Failure Classes",
        "",
    ]
    for cls, count in classes.most_common():
        lines.append(f"- {cls}: {count}")
    lines.append("")
    lines.append("## Examples")
    lines.append("")
    for cls, items in examples.items():
        lines.append(f"### {cls}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(items, indent=2, ensure_ascii=False, sort_keys=True))
        lines.append("```")
        lines.append("")
    (out_dir / "root_cause_after_reextract.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Gate 1 on structured_symcif_v4_reextracted.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_reextract")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--max-failures", type=int, default=3000)
    parser.add_argument("--old-row-summary", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_debug" / "row_expansion_summary.csv")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, int, dict[str, Any]]] = []
    for split in args.splits:
        rows = read_jsonl(args.structured_root / f"{split}.jsonl")
        payloads.extend((split, i, row) for i, row in enumerate(rows))

    if args.workers <= 1:
        init_worker(str(args.lookup_json), str(args.structured_root))
        metrics = [process_record(payload) for payload in payloads]
    else:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(str(args.lookup_json), str(args.structured_root))) as pool:
            metrics = list(pool.imap_unordered(process_record, payloads, chunksize=8))

    metrics.sort(key=lambda r: (str(r["split"]), int(r["input_index"])))
    failures = [
        row
        for row in metrics
        if not (row["readable"] and row["formula_ok"] and row["atom_count_ok"] and row["sg_ok"] and row["row_expansion_ok"])
    ]
    summary = {"overall": summarize(metrics), "splits": {}}
    for split in args.splits:
        summary["splits"][split] = summarize([row for row in metrics if row["split"] == split])
    summary["gate1"] = {
        "passed": bool(
            summary["overall"]["readable"] >= 0.99
            and summary["overall"]["formula_ok"] >= 0.99
            and summary["overall"]["atom_count_ok"] >= 0.99
            and summary["overall"]["sg_ok"] >= 0.98
            and summary["overall"]["row_expansion_ok"] >= 0.99
        ),
        "criteria": {
            "readable": 0.99,
            "formula_ok": 0.99,
            "atom_count_ok": 0.99,
            "sg_ok": 0.98,
            "row_expansion_ok": 0.99,
        },
    }
    summary["failed_samples"] = len(failures)

    orbit_occurrence: Counter[str] = Counter()
    orbit_degenerate: Counter[str] = Counter()
    orbit_in_failed_sample: Counter[str] = Counter()
    for row in metrics:
        for key in row["orbit_keys"]:
            orbit_occurrence[key] += 1
            if not (row["readable"] and row["formula_ok"] and row["atom_count_ok"] and row["sg_ok"] and row["row_expansion_ok"]):
                orbit_in_failed_sample[key] += 1
        for deg_row in row["degenerate_orbit_rows"]:
            orbit_degenerate[str(deg_row["sg_letter"])] += 1
    sg_letter_rows: list[dict[str, Any]] = []
    for key, total in orbit_occurrence.most_common():
        sg_letter_rows.append(
            {
                "sg_letter": key,
                "occurrences": int(total),
                "degenerate_occurrences": int(orbit_degenerate.get(key, 0)),
                "degenerate_rate": orbit_degenerate.get(key, 0) / max(1, total),
                "failed_sample_occurrences": int(orbit_in_failed_sample.get(key, 0)),
                "failed_sample_occurrence_rate": orbit_in_failed_sample.get(key, 0) / max(1, total),
            }
        )
    summary["top_failed_sg_letters"] = sorted(
        sg_letter_rows,
        key=lambda row: (float(row["failed_sample_occurrence_rate"]), int(row["failed_sample_occurrences"])),
        reverse=True,
    )[:50]

    (args.out_dir / "gate1_roundtrip_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "gate1_failures.jsonl", failures[: args.max_failures])
    write_csv(args.out_dir / "gate1_breakdown_per_sg.csv", group_summary(metrics, "sg"))
    write_csv(args.out_dir / "gate1_breakdown_per_nsites.csv", group_summary(metrics, "n_sites"))
    write_csv(args.out_dir / "gate1_breakdown_per_sg_letter.csv", sg_letter_rows)
    write_top_failed_report(args.out_dir, sg_letter_rows, old_row_breakdown(args.old_row_summary))
    write_root_cause_after_reextract(args.out_dir, summary, failures)
    print(json.dumps({
        "gate1_passed": summary["gate1"]["passed"],
        "failed_samples": len(failures),
        "overall": summary["overall"],
        "out_dir": str(args.out_dir),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
