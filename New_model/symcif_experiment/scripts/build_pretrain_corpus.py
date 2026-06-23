#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from symcif.pipeline import FormatName, Report, convert_one, load_or_build_lookup, write_json

warnings.filterwarnings("ignore")


FORMATS: list[FormatName] = ["baseline", "cf_like", "symcif_v1"]


def iter_cifs(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.cif") if p.is_file())


def write_split(path: Path, samples: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(s.strip() for s in samples) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fair A/B/C pretraining corpora from a shared CIF split.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--roundtrip", action="store_true", default=True)
    parser.add_argument("--max-candidates", type=int, default=0)
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

    target = args.train_size + args.val_size + args.test_size
    cifs = iter_cifs(args.input_root)
    rng = random.Random(args.seed)
    rng.shuffle(cifs)
    if args.max_candidates:
        cifs = cifs[: args.max_candidates]

    lookup = load_or_build_lookup(
        args.source_wyckoff_csv,
        args.lookup_json,
        wyformer_json=args.wyformer_json,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        rebuild=args.rebuild_lookup,
    )

    raw_reports = {fmt: Report() for fmt in FORMATS}
    retained_reports = {fmt: Report() for fmt in FORMATS}
    failures = []
    accepted: list[dict[str, object]] = []
    corpora = {fmt: [] for fmt in FORMATS}

    for idx, cif_path in enumerate(cifs, start=1):
        outputs = {
            fmt: convert_one(
                cif_path,
                fmt,
                lookup,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                roundtrip=args.roundtrip and fmt != "baseline",
            )
            for fmt in FORMATS
        }
        all_ok = True
        for fmt, result in outputs.items():
            raw_reports[fmt].counts["total"] += 1
            if result.ok and result.text:
                if fmt == "baseline":
                    raw_reports[fmt].counts["parse_success"] += 1
                    raw_reports[fmt].counts["render_success"] += 1
                else:
                    has_enum = fmt != "symcif_v1" or "UNKNOWN" not in result.text
                    raw_reports[fmt].add_success(result.validation, has_enum=has_enum)
            else:
                all_ok = False
                raw_reports[fmt].add_failure(result.stage or "convert", result.reason or "unknown")
                failures.append(
                    {
                        "cif_path": str(cif_path),
                        "format": fmt,
                        "stage": result.stage,
                        "reason": result.reason,
                        "traceback": result.traceback_text,
                    }
                )
        if all_ok:
            # Fairness gate: all non-baseline formats must pass the minimum round-trip checks.
            quality_ok = True
            for fmt in ("cf_like", "symcif_v1"):
                val = outputs[fmt].validation
                quality_ok = quality_ok and all(
                    bool(val.get(k))
                    for k in (
                        "pymatgen_readable",
                        "formula_consistent",
                        "space_group_consistent",
                        "multiplicity_consistent",
                    )
                )
            if quality_ok:
                split = "train" if len(accepted) < args.train_size else "val" if len(accepted) < args.train_size + args.val_size else "test"
                accepted.append(
                    {
                        "sample_id": outputs["baseline"].sample_id,
                        "source_path": str(cif_path),
                        "split": split,
                    }
                )
                for fmt in FORMATS:
                    corpora[fmt].append(outputs[fmt].text or "")
                    retained_reports[fmt].counts["total"] += 1
                    if fmt == "baseline":
                        retained_reports[fmt].counts["parse_success"] += 1
                        retained_reports[fmt].counts["render_success"] += 1
                    else:
                        has_enum = fmt != "symcif_v1" or "UNKNOWN" not in (outputs[fmt].text or "")
                        retained_reports[fmt].add_success(outputs[fmt].validation, has_enum=has_enum)
            else:
                for fmt in ("cf_like", "symcif_v1"):
                    val = outputs[fmt].validation
                    failed_checks = [
                        key
                        for key in (
                            "pymatgen_readable",
                            "formula_consistent",
                            "space_group_consistent",
                            "multiplicity_consistent",
                        )
                        if not bool(val.get(key))
                    ]
                    if failed_checks:
                        failures.append(
                            {
                                "cif_path": str(cif_path),
                                "format": fmt,
                                "stage": "quality_gate",
                                "reason": ",".join(failed_checks),
                                "traceback": "",
                            }
                        )
        if idx % 250 == 0:
            print(f"processed {idx}/{len(cifs)}; accepted {len(accepted)}/{target}", flush=True)
        if len(accepted) >= target:
            break

    if len(accepted) < target:
        raise SystemExit(f"only accepted {len(accepted)} samples, target was {target}")

    split_ranges = {
        "train": (0, args.train_size),
        "val": (args.train_size, args.train_size + args.val_size),
        "test": (args.train_size + args.val_size, target),
    }
    for fmt in FORMATS:
        for split, (lo, hi) in split_ranges.items():
            write_split(args.out_dir / fmt / f"{split}.txt", corpora[fmt][lo:hi])

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for fmt, report in retained_reports.items():
        write_json(args.reports_dir / f"conversion_report_{fmt}.json", report.as_dict())
    for fmt, report in raw_reports.items():
        write_json(args.reports_dir / f"conversion_report_{fmt}_raw.json", report.as_dict())

    with open(args.reports_dir / "failed_cases.jsonl", "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    with open(args.out_dir / "split_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_path", "split"])
        writer.writeheader()
        writer.writerows(accepted)

    split_counts = Counter(row["split"] for row in accepted)
    write_json(
        args.reports_dir / "corpus_build_summary.json",
        {
            "input_root": str(args.input_root),
            "target": {"train": args.train_size, "val": args.val_size, "test": args.test_size},
            "accepted": dict(split_counts),
            "processed_candidates": sum(report.counts["total"] for report in raw_reports.values()) // len(FORMATS),
            "seed": args.seed,
            "symprec": args.symprec,
            "angle_tolerance": args.angle_tolerance,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
