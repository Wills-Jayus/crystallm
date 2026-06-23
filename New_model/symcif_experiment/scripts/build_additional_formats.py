#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from symcif.pipeline import FormatName, Report, convert_one, load_or_build_lookup, write_json  # noqa: E402


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT.parents[2] / path


def write_split(path: Path, samples: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(s.strip() for s in samples) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build additional corpora on the existing fixed train/val/test split.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "split_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["baseline_minprompt", "symcif_v1_atomprops"],
        choices=["baseline_minprompt", "symcif_v1_atomprops"],
    )
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--roundtrip", action="store_true", default=True)
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup.json")
    parser.add_argument(
        "--source-wyckoff-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "source_tables" / "crystalformer_wyckoff_list.csv",
    )
    parser.add_argument(
        "--wyformer-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "source_tables" / "wyformer_wyckoffs_enumerated_by_ss.json",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty manifest: {args.manifest}")

    lookup = load_or_build_lookup(
        args.source_wyckoff_csv,
        args.lookup_json,
        wyformer_json=args.wyformer_json,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )

    corpora: dict[str, dict[str, list[str]]] = {
        fmt: {"train": [], "val": [], "test": []} for fmt in args.formats
    }
    reports = {fmt: Report() for fmt in args.formats}
    failures: list[dict[str, str]] = []

    for i, row in enumerate(rows, start=1):
        split = row["split"]
        source_path = resolve_path(row["source_path"])
        for fmt in args.formats:
            result = convert_one(
                source_path,
                fmt,  # type: ignore[arg-type]
                lookup,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                roundtrip=args.roundtrip and fmt.startswith("symcif"),
            )
            reports[fmt].counts["total"] += 1
            if result.ok and result.text:
                corpora[fmt][split].append(result.text)
                if fmt.startswith("baseline"):
                    reports[fmt].counts["parse_success"] += 1
                    reports[fmt].counts["render_success"] += 1
                else:
                    reports[fmt].add_success(result.validation, has_enum="UNKNOWN" not in result.text)
            else:
                reports[fmt].add_failure(result.stage or "convert", result.reason or "unknown")
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "source_path": str(source_path),
                        "split": split,
                        "format": fmt,
                        "stage": result.stage or "convert",
                        "reason": result.reason or "unknown",
                        "traceback": result.traceback_text or "",
                    }
                )
        if i % 500 == 0:
            print(f"processed {i}/{len(rows)} manifest rows", flush=True)

    for fmt, by_split in corpora.items():
        for split, samples in by_split.items():
            write_split(args.out_dir / fmt / f"{split}.txt", samples)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for fmt, report in reports.items():
        write_json(args.reports_dir / f"conversion_report_{fmt}.json", report.as_dict())
    with (args.reports_dir / "failed_cases_additional_formats.jsonl").open("w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    write_json(
        args.reports_dir / "additional_formats_build_summary.json",
        {
            "manifest": str(args.manifest),
            "formats": args.formats,
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "failures": len(failures),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
