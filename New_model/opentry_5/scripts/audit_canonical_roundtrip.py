#!/usr/bin/env python3
"""Audit canonical CIF copies against their source CIFs with StructureMatcher."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def canonical_files() -> list[tuple[str, Path]]:
    return [
        ("train_core", ROOT / "data/canonical_train/train_core.jsonl"),
        ("dev_model", ROOT / "data/canonical_dev/dev_model.jsonl"),
        ("dev_gate", ROOT / "data/canonical_dev/dev_gate.jsonl"),
        ("fold_a", ROOT / "data/canonical_dev/fold_a.jsonl"),
        ("fold_b", ROOT / "data/canonical_dev/fold_b.jsonl"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=1024)
    parser.add_argument("--split", default="all", choices=["all", "train_core", "dev_model", "dev_gate", "fold_a", "fold_b"])
    args = parser.parse_args()

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True, attempt_supercell=False)
    stats = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_records": args.max_records,
        "split_filter": args.split,
        "structure_matcher": {
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5.0,
            "primitive_cell": True,
            "scale": True,
            "attempt_supercell": False,
        },
        "records_checked": 0,
        "records_considered": 0,
        "skipped_missing_cif": 0,
        "parse_failures": 0,
        "composition_mismatches": 0,
        "structurematcher_failures": 0,
        "successes": 0,
        "by_split": {},
        "examples": [],
        "test_access": "none",
    }

    for split_name, path in canonical_files():
        if args.split != "all" and split_name != args.split:
            continue
        split_stats = stats["by_split"].setdefault(
            split_name,
            {
                "checked": 0,
                "considered": 0,
                "skipped_missing_cif": 0,
                "parse_failures": 0,
                "composition_mismatches": 0,
                "structurematcher_failures": 0,
                "successes": 0,
            },
        )
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if args.max_records and stats["records_considered"] >= args.max_records:
                break
            stats["records_considered"] += 1
            split_stats["considered"] += 1
            sample_id = row.get("sample_id")
            canonical_rel = row.get("canonical_cif_path")
            source_path = row.get("source_path")
            if not canonical_rel or not source_path:
                stats["skipped_missing_cif"] += 1
                split_stats["skipped_missing_cif"] += 1
                if len(stats["examples"]) < 20:
                    stats["examples"].append(
                        {
                            "sample_id": sample_id,
                            "split": split_name,
                            "error": "skipped_missing_cif",
                            "reason": row.get("canonical_cif_excluded_reason"),
                        }
                    )
                continue
            stats["records_checked"] += 1
            split_stats["checked"] += 1
            try:
                canonical = Structure.from_file(str(ROOT / canonical_rel))
                source = Structure.from_file(str(source_path))
            except Exception as exc:
                stats["parse_failures"] += 1
                split_stats["parse_failures"] += 1
                if len(stats["examples"]) < 20:
                    stats["examples"].append({"sample_id": sample_id, "split": split_name, "error": f"parse: {exc}"})
                continue
            if canonical.composition.reduced_formula != source.composition.reduced_formula:
                stats["composition_mismatches"] += 1
                split_stats["composition_mismatches"] += 1
                if len(stats["examples"]) < 20:
                    stats["examples"].append(
                        {
                            "sample_id": sample_id,
                            "split": split_name,
                            "error": "composition_mismatch",
                            "canonical": canonical.composition.reduced_formula,
                            "source": source.composition.reduced_formula,
                        }
                    )
                continue
            if not matcher.fit(canonical, source):
                stats["structurematcher_failures"] += 1
                split_stats["structurematcher_failures"] += 1
                if len(stats["examples"]) < 20:
                    stats["examples"].append({"sample_id": sample_id, "split": split_name, "error": "structurematcher_failure"})
                continue
            stats["successes"] += 1
            split_stats["successes"] += 1
        if args.max_records and stats["records_considered"] >= args.max_records:
            break

    denom = max(stats["records_checked"], 1)
    stats["success_rate"] = stats["successes"] / denom
    stats["pass"] = stats["records_checked"] > 0 and stats["success_rate"] >= 0.995 and stats["parse_failures"] == 0
    write_json(ROOT / "eval/canonical_roundtrip_audit.json", stats)

    report_path = ROOT / "reports/canonical_representation_report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Canonical Representation Report\n"
    audit_block = [
        "",
        "## Round-Trip Audit",
        "",
        f"Audit time: {stats['created_at']}",
        f"Records checked: {stats['records_checked']}",
        f"Records considered: {stats['records_considered']}",
        f"Skipped missing/invalid CIF: {stats['skipped_missing_cif']}",
        f"Success rate: {stats['success_rate']:.4f}",
        f"Parse failures: {stats['parse_failures']}",
        f"Composition mismatches: {stats['composition_mismatches']}",
        f"StructureMatcher failures: {stats['structurematcher_failures']}",
        f"Pass: {stats['pass']}",
        "",
        "This audit reads canonical CIF copies and source CIFs only from train/dev data; no test files are accessed.",
    ]
    if "## Round-Trip Audit" in existing:
        existing = existing.split("## Round-Trip Audit")[0].rstrip() + "\n"
    write_text(report_path, existing.rstrip() + "\n" + "\n".join(audit_block) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
