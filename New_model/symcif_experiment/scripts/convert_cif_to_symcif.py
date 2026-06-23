#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from symcif.pipeline import FormatName, Report, convert_one, load_or_build_lookup, write_json

warnings.filterwarnings("ignore")


FORMAT_DIR = {
    "baseline": ("format_A", ".cif"),
    "cf_like": ("format_B", ".ciflike"),
    "symcif_v1": ("format_C", ".symcif"),
}
ROUNDTRIP_DIR = {
    "cf_like": "roundtrip_B",
    "symcif_v1": "roundtrip_C",
}


def iter_cifs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.cif") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CIF files to baseline / CF-like / SymCIF-v1 text.")
    parser.add_argument("input", type=Path, help="CIF file or directory containing CIFs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--format", choices=["baseline", "cf_like", "symcif_v1", "all"], default="all")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--roundtrip", action="store_true", help="Write and validate round-trip CIFs.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rebuild-lookup", action="store_true")
    parser.add_argument(
        "--source-wyckoff-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts/source_tables/crystalformer_wyckoff_list.csv",
    )
    parser.add_argument(
        "--wyformer-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts/source_tables/wyformer_wyckoffs_enumerated_by_ss.json",
    )
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts/wyckoff_lookup.json")
    args = parser.parse_args()

    formats: list[FormatName] = ["baseline", "cf_like", "symcif_v1"] if args.format == "all" else [args.format]
    cifs = iter_cifs(args.input)
    if args.limit:
        cifs = cifs[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_or_build_lookup(
        args.source_wyckoff_csv,
        args.lookup_json,
        wyformer_json=args.wyformer_json,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        rebuild=args.rebuild_lookup,
    )

    all_failures = []
    for fmt in formats:
        report = Report()
        report.counts["total"] = len(cifs)
        text_dir_name, suffix = FORMAT_DIR[fmt]
        text_dir = args.out_dir / text_dir_name
        text_dir.mkdir(parents=True, exist_ok=True)
        rt_dir = args.out_dir / ROUNDTRIP_DIR[fmt] if fmt in ROUNDTRIP_DIR else None
        if rt_dir:
            rt_dir.mkdir(parents=True, exist_ok=True)

        for idx, cif_path in enumerate(cifs, start=1):
            result = convert_one(
                cif_path,
                fmt,
                lookup,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                roundtrip=args.roundtrip and fmt != "baseline",
            )
            if result.ok and result.text is not None:
                (text_dir / f"{result.sample_id}{suffix}").write_text(result.text, encoding="utf-8")
                if rt_dir and result.roundtrip_cif:
                    (rt_dir / f"{result.sample_id}.cif").write_text(result.roundtrip_cif, encoding="utf-8")
                has_enum = fmt != "symcif_v1" or "UNKNOWN" not in result.text
                if fmt == "baseline":
                    report.counts["parse_success"] += 1
                    report.counts["render_success"] += 1
                else:
                    report.add_success(result.validation, has_enum=has_enum)
            else:
                report.add_failure(result.stage or "convert", result.reason or "unknown")
                failure = {
                    "cif_path": str(cif_path),
                    "format": fmt,
                    "stage": result.stage,
                    "reason": result.reason,
                    "traceback": result.traceback_text,
                }
                all_failures.append(failure)
            if idx % 250 == 0:
                print(f"[{fmt}] processed {idx}/{len(cifs)}", flush=True)

        report_name = f"conversion_report_{fmt}.json"
        write_json(args.out_dir / report_name, report.as_dict())

    if all_failures:
        with open(args.out_dir / "failed_cases.jsonl", "w", encoding="utf-8") as f:
            for item in all_failures:
                f.write(json.dumps(item, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
