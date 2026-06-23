#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.parse import parse_symcif_text  # noqa: E402
from symcif_v3.parse import parse_symcif_v3_text  # noqa: E402
from symcif_v3.render import render_symcif_v3  # noqa: E402


def split_concat_records(path: Path) -> list[str]:
    records: list[str] = []
    cur: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("data_") and cur:
            records.append("\n".join(cur).rstrip() + "\n")
            cur = [line]
        else:
            cur.append(line)
    if cur:
        records.append("\n".join(cur).rstrip() + "\n")
    return records


def write_split(path: Path, records: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record.rstrip() for record in records).rstrip() + "\n", encoding="utf-8")


def report_dict(counts: Counter[str], failures: Counter[str]) -> dict[str, Any]:
    total = max(1, counts["total"])
    keys = [
        "parse_success",
        "render_success",
        "v3_parse_success",
        "signature_consistent",
        "lattice_consistent",
        "retained",
    ]
    out: dict[str, Any] = {"total": counts["total"]}
    for key in keys:
        out[key] = int(counts[key])
        out[f"{key}_rate"] = float(counts[key] / total)
    out["top_failure_reasons"] = dict(failures.most_common(20))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clean SymCIF-v3 staged corpus from the clean SymCIF-v1 corpus.")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v3")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--strict-clean", action="store_true", default=True)
    args = parser.parse_args()

    lookup = WyckoffLookup.from_json(args.lookup_json)
    corpora: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    failed_cases: list[dict[str, Any]] = []

    for split in ("train", "val", "test"):
        for index, source_text in enumerate(split_concat_records(args.source_dir / f"{split}.txt")):
            counts["total"] += 1
            sample_id = "unknown"
            try:
                record = parse_symcif_text(source_text, lookup)
                sample_id = record.sample_id
                counts["parse_success"] += 1
                text = render_symcif_v3(record)
                counts["render_success"] += 1
                parsed = parse_symcif_v3_text(text, lookup)
                counts["v3_parse_success"] += 1
                src_sig = sorted((s.index, s.element, int(s.multiplicity), s.letter, tuple(round(x, 4) for x in s.representative_coord)) for s in record.sites)
                dst_sig = sorted((s.index, s.element, int(s.multiplicity), s.letter, tuple(round(x, 4) for x in s.representative_coord)) for s in parsed.sites)
                signature_ok = (
                    record.cell_formula == parsed.cell_formula
                    and int(record.sg_number) == int(parsed.sg_number)
                    and src_sig == dst_sig
                )
                lattice_ok = all(
                    abs(float(getattr(record.lattice, key)) - float(getattr(parsed.lattice, key))) < 5e-4
                    for key in ("a", "b", "c", "alpha", "beta", "gamma", "volume")
                )
                if signature_ok:
                    counts["signature_consistent"] += 1
                if lattice_ok:
                    counts["lattice_consistent"] += 1
                quality_ok = signature_ok and lattice_ok
                if not quality_ok:
                    failed_checks = []
                    if not signature_ok:
                        failed_checks.append("signature_consistent")
                    if not lattice_ok:
                        failed_checks.append("lattice_consistent")
                    reason = ",".join(failed_checks)
                    failures[f"quality_gate:{reason}"] += 1
                    failed_cases.append(
                        {
                            "split": split,
                            "index": index,
                            "sample_id": sample_id,
                            "stage": "quality_gate",
                            "reason": reason,
                        }
                    )
                    if args.strict_clean:
                        continue
                corpora[split].append(text)
                counts["retained"] += 1
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"
                failures[f"convert:{reason}"] += 1
                failed_cases.append(
                    {
                        "split": split,
                        "index": index,
                        "sample_id": sample_id,
                        "stage": "convert",
                        "reason": reason,
                    }
                )

    for split, records in corpora.items():
        write_split(args.out_dir / f"{split}.txt", records)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = report_dict(counts, failures)
    report["source_dir"] = str(args.source_dir)
    report["out_dir"] = str(args.out_dir)
    report["retained_sizes"] = {split: len(records) for split, records in corpora.items()}
    report["symprec"] = args.symprec
    report["angle_tolerance"] = args.angle_tolerance
    source_report = args.reports_dir / "conversion_report_symcif_v1.json"
    if source_report.exists():
        report["source_clean_report"] = str(source_report)
        report["source_clean_report_summary"] = json.loads(source_report.read_text(encoding="utf-8"))
    (args.reports_dir / "symcif_v3_conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.reports_dir / "symcif_v3_failed_cases.jsonl").open("w", encoding="utf-8") as f:
        for item in failed_cases:
            f.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    with tarfile.open(args.out_dir / "symcif_v3.tar.gz", "w:gz") as tar:
        for split in ("train", "val", "test"):
            tar.add(args.out_dir / f"{split}.txt", arcname=f"symcif_v3/{split}.txt")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
