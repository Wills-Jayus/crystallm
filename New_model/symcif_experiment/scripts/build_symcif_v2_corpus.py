#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTODL_ROOT = PROJECT_ROOT.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from symcif.extract import extract_record_from_cif  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.parse import parse_symcif_text  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.render import render_symcif_v2  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402
from symcif_v2.validate import validate_roundtrip_v2  # noqa: E402

warnings.filterwarnings("ignore")


def resolve_source_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return AUTODL_ROOT / path


def write_split(path: Path, samples: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sample.strip() for sample in samples) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def report_dict(counts: Counter[str], failures: Counter[str]) -> dict[str, Any]:
    total = int(counts["total"])
    out: dict[str, Any] = {
        "total": total,
        "conversion_success": int(counts["conversion_success"]),
        "parse_success": int(counts["parse_success"]),
        "render_success": int(counts["render_success"]),
        "roundtrip_cif_success": int(counts["roundtrip_cif_success"]),
        "pymatgen_readable": int(counts["pymatgen_readable"]),
        "formula_consistent": int(counts["formula_consistent"]),
        "space_group_consistent": int(counts["space_group_consistent"]),
        "multiplicity_consistent": int(counts["multiplicity_consistent"]),
        "retained": int(counts["retained"]),
        "top_failure_reasons": dict(failures.most_common(50)),
    }
    for key in (
        "conversion_success",
        "parse_success",
        "render_success",
        "roundtrip_cif_success",
        "pymatgen_readable",
        "formula_consistent",
        "space_group_consistent",
        "multiplicity_consistent",
        "retained",
    ):
        out[f"{key}_rate"] = out[key] / total if total else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SymCIF-v2 corpus from the existing fair split manifest.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "split_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v2")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument(
        "--source-mode",
        choices=["cif", "symcif-v1"],
        default="cif",
        help=(
            "cif extracts directly from manifest source CIFs; symcif-v1 re-renders "
            "the already validated fair SymCIF-v1 split into v2 while preserving "
            "the same sample order and source paths."
        ),
    )
    parser.add_argument("--symcif-v1-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v1")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
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
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup.json")
    parser.add_argument("--strict-target-sizes", action="store_true", default=True)
    parser.add_argument(
        "--require-roundtrip-quality",
        action="store_true",
        help="Drop samples that fail the strict v2 round-trip quality gates instead of retaining the fair split.",
    )
    args = parser.parse_args()

    lookup = WyckoffLookup.from_crystalformer_csv(
        args.source_wyckoff_csv,
        wyformer_json=args.wyformer_json,
        infer_site_symmetry=False,
    )
    manifest = read_manifest(args.manifest)
    target_sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    corpora: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    retained_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    failed_cases: list[dict[str, Any]] = []

    if args.source_mode == "symcif-v1":
        manifest_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
        for row in manifest:
            if row.get("split") in manifest_by_split:
                manifest_by_split[row["split"]].append(row)
        iter_rows: list[tuple[int, dict[str, str], str | None]] = []
        global_idx = 0
        for split in ("train", "val", "test"):
            records = split_concat_records(args.symcif_v1_dir / f"{split}.txt")
            rows = manifest_by_split[split]
            if len(records) < target_sizes[split]:
                raise SystemExit(f"{args.symcif_v1_dir / f'{split}.txt'} has only {len(records)} records")
            if len(rows) < target_sizes[split]:
                raise SystemExit(f"manifest has only {len(rows)} rows for split {split}")
            for record_text, row in zip(records[: target_sizes[split]], rows[: target_sizes[split]]):
                global_idx += 1
                iter_rows.append((global_idx, row, record_text))
    else:
        iter_rows = [(idx, row, None) for idx, row in enumerate(manifest, start=1)]

    for idx, row, source_text in iter_rows:
        split = row["split"]
        if split not in corpora:
            continue
        if len(corpora[split]) >= target_sizes[split]:
            continue
        counts["total"] += 1
        sample_id = row["sample_id"]
        source_path = resolve_source_path(row["source_path"])
        try:
            if source_text is None:
                record = extract_record_from_cif(
                    source_path,
                    lookup,
                    symprec=args.symprec,
                    angle_tolerance=args.angle_tolerance,
                    standardize=True,
                )
            else:
                record = parse_symcif_text(source_text, lookup, source_path=source_path)
                if record.sample_id != sample_id:
                    raise ValueError(f"sample_id mismatch: corpus={record.sample_id} manifest={sample_id}")
            text = render_symcif_v2(record)
            counts["conversion_success"] += 1
            parsed = parse_symcif_v2_text(text, lookup, source_path=source_path)
            counts["parse_success"] += 1
            rt_cif = render_standard_cif_v2(parsed, symprec=args.symprec, lookup=lookup)
            counts["render_success"] += 1
            validation = validate_roundtrip_v2(record, rt_cif, symprec=args.symprec, angle_tolerance=args.angle_tolerance)
            counts["roundtrip_cif_success"] += 1
            for key in (
                "pymatgen_readable",
                "formula_consistent",
                "space_group_consistent",
                "multiplicity_consistent",
            ):
                if validation.get(key):
                    counts[key] += 1
            quality_ok = all(
                bool(validation.get(key))
                for key in (
                    "pymatgen_readable",
                    "formula_consistent",
                    "space_group_consistent",
                    "multiplicity_consistent",
                )
            )
            if not quality_ok:
                failed_checks = [
                    key
                    for key in (
                        "pymatgen_readable",
                        "formula_consistent",
                        "space_group_consistent",
                        "multiplicity_consistent",
                    )
                    if not bool(validation.get(key))
                ]
                reason = ",".join(failed_checks)
                failures[f"quality_gate:{reason}"] += 1
                failed_cases.append(
                    {
                        "sample_id": sample_id,
                        "source_path": str(source_path),
                        "split": split,
                        "stage": "quality_gate",
                        "reason": reason,
                        "validation": validation,
                    }
                )
                if args.require_roundtrip_quality:
                    continue
            corpora[split].append(text)
            retained_rows.append({"sample_id": sample_id, "source_path": row["source_path"], "split": split})
            counts["retained"] += 1
        except Exception as exc:  # noqa: BLE001
            stage = "convert"
            reason = f"{type(exc).__name__}: {exc}"
            failures[f"{stage}:{reason}"] += 1
            failed_cases.append(
                {
                    "sample_id": sample_id,
                    "source_path": str(source_path),
                    "split": split,
                    "stage": stage,
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                }
            )
        if idx % 250 == 0:
            sizes = {key: len(value) for key, value in corpora.items()}
            print(f"processed manifest rows={idx}; retained={sizes}", flush=True)
        if all(len(corpora[key]) >= target_sizes[key] for key in target_sizes):
            break

    sizes = {key: len(value) for key, value in corpora.items()}
    if args.strict_target_sizes and sizes != target_sizes:
        raise SystemExit(f"SymCIF-v2 retained sizes {sizes} do not match required target {target_sizes}")

    for split, samples in corpora.items():
        write_split(args.out_dir / f"{split}.txt", samples)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = report_dict(counts, failures)
    report["target_sizes"] = target_sizes
    report["retained_sizes"] = sizes
    report["symprec"] = args.symprec
    report["angle_tolerance"] = args.angle_tolerance
    report["manifest"] = str(args.manifest)
    report["out_dir"] = str(args.out_dir)
    (args.reports_dir / "symcif_v2_conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.reports_dir / "symcif_v2_failed_cases.jsonl").open("w", encoding="utf-8") as f:
        for item in failed_cases:
            f.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    with (args.out_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_path", "split"])
        writer.writeheader()
        writer.writerows(retained_rows)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
