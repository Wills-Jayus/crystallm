#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.canonicalize import canonical_skeleton_key, canonical_wa_key, wa_table_from_structured
from symcif_v4.formula import total_atoms
from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.validation import validate_cif
from train_skeleton_template_ranker import read_jsonl


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(r["readable"]) for r in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(r["formula_ok"]) for r in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(r["sg_ok"]) for r in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(r["atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(r["composition_exact"]) for r in rows) / max(1, len(rows)),
    }


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        item = {key: value, **summarize(items)}
        item["failed"] = sum(not (r["formula_ok"] and r["sg_ok"] and r["readable"] and r["atom_count_ok"]) for r in items)
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GT WA + GT coords/lattice through OrbitEngine and validate CIF roundtrip.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_orbit_engine")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-failures", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.structured_root)
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fail_letters: Counter[str] = Counter()
    orbit_occurrence: Counter[str] = Counter()
    orbit_degenerate: Counter[str] = Counter()
    orbit_in_failed_sample: Counter[str] = Counter()
    for split in args.splits:
        rows = read_jsonl(args.structured_root / f"{split}.jsonl")
        for i, row in enumerate(rows):
            wa_table, free_params = wa_table_from_structured(row, engine)
            cif = engine.render_cif_from_wa_table(
                wa_table,
                lattice=row["lattice"],
                free_params_by_row=free_params,
                formula_counts=row["formula_counts"],
                sg=int(row["sg"]),
                sg_symbol=str(row["sg_symbol"]),
                data_name=str(row["sample_id"]),
            )
            metric = validate_cif(cif, row["formula_counts"], int(row["sg"]))
            expanded_count = 0
            row_orbit_keys: list[str] = []
            degenerate_rows: list[dict[str, Any]] = []
            for j, w in enumerate(wa_table):
                orbit = engine.get_orbit_by_id(w["orbit_id"])
                key = f"{orbit.sg}|{orbit.multiplicity}{orbit.letter}"
                row_orbit_keys.append(key)
                orbit_occurrence[key] += 1
                expanded = engine.expand_orbit(orbit, free_params[j])
                expanded_count += len(expanded)
                if len(expanded) != int(orbit.multiplicity):
                    orbit_degenerate[key] += 1
                    degenerate_rows.append(
                        {
                            "sg_letter": key,
                            "expanded_count": len(expanded),
                            "multiplicity": int(orbit.multiplicity),
                            "params": free_params[j],
                            "representative_expr": list(orbit.representative_expr),
                        }
                    )
            item = {
                "split": split,
                "sample_id": row["sample_id"],
                "sg": int(row["sg"]),
                "sg_symbol": row["sg_symbol"],
                "n_sites": int(row["n_sites"]),
                "num_elements": int(row["num_elements"]),
                "total_atoms": int(total_atoms(row["formula_counts"])),
                "expanded_atom_count": int(expanded_count),
                "expanded_atom_count_ok": int(expanded_count) == int(total_atoms(row["formula_counts"])),
                "degenerate_orbit_rows": degenerate_rows,
                "canonical_skeleton_key": canonical_skeleton_key(wa_table),
                "canonical_wa_key": canonical_wa_key(wa_table),
                **metric,
            }
            metrics.append(item)
            ok = bool(item["readable"] and item["formula_ok"] and item["sg_ok"] and item["atom_count_ok"] and item["expanded_atom_count_ok"])
            if not ok:
                for key in row_orbit_keys:
                    fail_letters[key] += 1
                    orbit_in_failed_sample[key] += 1
                if len(failures) < args.max_failures:
                    failures.append({**item, "formula": row["formula"], "wa_letters": row["skeleton_template_key"], "error": item.get("error")})
        print(f"[roundtrip] {split} done", flush=True)
    summary = {"overall": summarize(metrics), "splits": {}}
    for split in args.splits:
        summary["splits"][split] = summarize([r for r in metrics if r["split"] == split])
    summary["gate1"] = {
        "passed": bool(
            summary["overall"]["readable"] >= 0.99
            and summary["overall"]["formula_ok"] >= 0.99
            and summary["overall"]["sg_ok"] >= 0.98
            and summary["overall"]["atom_count_ok"] >= 0.99
        ),
        "criteria": {"readable": 0.99, "formula_ok": 0.99, "sg_ok": 0.98, "atom_count_ok": 0.99},
    }
    summary["top_failed_sg_letters"] = [{"sg_letter": k, "count": v} for k, v in fail_letters.most_common(50)]
    (args.out_dir / "gt_roundtrip_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "gt_roundtrip_failures.jsonl").open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_csv(args.out_dir / "gt_roundtrip_breakdown_per_sg.csv", group_summary(metrics, "sg"))
    write_csv(args.out_dir / "gt_roundtrip_breakdown_per_nsites.csv", group_summary(metrics, "n_sites"))
    sg_letter_rows = []
    for key, total in orbit_occurrence.most_common():
        sg_letter_rows.append(
            {
                "sg_letter": key,
                "occurrences": total,
                "degenerate_occurrences": orbit_degenerate.get(key, 0),
                "degenerate_rate": orbit_degenerate.get(key, 0) / max(1, total),
                "failed_sample_occurrences": orbit_in_failed_sample.get(key, 0),
                "failed_sample_occurrence_rate": orbit_in_failed_sample.get(key, 0) / max(1, total),
            }
        )
    write_csv(args.out_dir / "gt_roundtrip_breakdown_per_sg_letter.csv", sg_letter_rows)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
