#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM")
OUT_ROOT = ROOT / "cache/official_benchmark_cifs_symprec0p1"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_log(text: str) -> None:
    path = under_root(ROOT / "experiment_log.md")
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def rewrite_data_token(cif_text: str, desired: str) -> str:
    lines = cif_text.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if not replaced and line.lstrip().lower().startswith("data_"):
            out.append(f"data_{desired}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.insert(0, f"data_{desired}")
    return "\n".join(out).rstrip() + "\n"


def formula_sum_token(cif_text: str, fallback: str) -> str:
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_text)
    if match:
        val = match.group(2) if match.group(2) is not None else match.group(3)
        return str(val).replace(" ", "").strip().strip("'").strip('"')
    return fallback.replace(" ", "")


def process_dataset(dataset: str, split: str, *, symprec: float, angle_tolerance: float, overwrite: bool) -> dict[str, Any]:
    csv_path = CRYSTALLM / f"resources/benchmarks/{dataset}/{split}.csv"
    out_dir = under_root(OUT_ROOT / dataset / split / "cifs")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    n_p1 = 0
    n_sg_match_csv = 0
    n_sg_with_csv = 0

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for csv_row, row in enumerate(reader):
            mid = str(row.get("material_id", "")).strip()
            cif_text = str(row.get("cif", "") or "")
            if not mid or not cif_text.strip():
                errors.append({"csv_row": csv_row, "material_id": mid, "error": "missing material_id or cif"})
                continue
            out_path = out_dir / f"{mid}.cif"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {out_path}")
            desired = formula_sum_token(cif_text, fallback=mid)
            detected_symbol = None
            detected_number = None
            status = "symmetrized"
            try:
                struct = Structure.from_str(cif_text, fmt="cif")
                sga = SpacegroupAnalyzer(struct, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
                detected_symbol = str(sga.get_space_group_symbol())
                detected_number = int(sga.get_space_group_number())
                conventional = sga.get_conventional_standard_structure()
                cif_out = str(CifWriter(conventional, symprec=float(symprec)))
                cif_out = rewrite_data_token(cif_out, desired)
            except Exception as exc:  # noqa: BLE001
                status = "fallback_original_p1"
                detected_symbol = "P1"
                detected_number = 1
                cif_out = rewrite_data_token(cif_text, desired)
                errors.append(
                    {
                        "csv_row": csv_row,
                        "material_id": mid,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            out_path.write_text(cif_out, encoding="utf-8")
            if detected_number == 1:
                n_p1 += 1
            csv_sg = row.get("spacegroup.number")
            csv_sg_int = None
            if csv_sg not in (None, ""):
                try:
                    csv_sg_int = int(float(str(csv_sg)))
                    n_sg_with_csv += 1
                    if detected_number == csv_sg_int:
                        n_sg_match_csv += 1
                except ValueError:
                    pass
            manifest_rows.append(
                {
                    "material_id": mid,
                    "csv_row": csv_row,
                    "path": str(out_path),
                    "status": status,
                    "detected_space_group_symbol": detected_symbol,
                    "detected_space_group_number": detected_number,
                    "csv_spacegroup_number": csv_sg_int,
                    "formula_sum_token": desired,
                }
            )

    manifest_path = OUT_ROOT / dataset / split / "manifest.tsv"
    manifest_path = under_root(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(manifest_rows[0].keys()) if manifest_rows else [
        "material_id",
        "csv_row",
        "path",
        "status",
        "detected_space_group_symbol",
        "detected_space_group_number",
        "csv_spacegroup_number",
        "formula_sum_token",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "dataset": dataset,
        "split": split,
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "records": len(manifest_rows),
        "errors": len(errors),
        "p1_detected": n_p1,
        "non_p1_detected": len(manifest_rows) - n_p1,
        "csv_sg_available": n_sg_with_csv,
        "csv_sg_match": n_sg_match_csv,
        "csv_sg_match_rate": None if n_sg_with_csv == 0 else n_sg_match_csv / n_sg_with_csv,
        "first_errors": errors[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = [
        process_dataset("mp_20", args.split, symprec=args.symprec, angle_tolerance=args.angle_tolerance, overwrite=args.overwrite),
        process_dataset("mpts_52", args.split, symprec=args.symprec, angle_tolerance=args.angle_tolerance, overwrite=args.overwrite),
    ]
    payload = {
        "created_at": now_iso(),
        "symprec": args.symprec,
        "angle_tolerance": args.angle_tolerance,
        "out_root": str(OUT_ROOT),
        "datasets": results,
        "note": (
            "These validation CIFs are symmetrized/conventionalized from official CSV CIFs for GT-SG prompt construction. "
            "They replace the earlier direct CSV extraction cache whose CIFs expose P1 in _symmetry_space_group_name_H-M."
        ),
    }
    write_json(ROOT / "metrics/validation_gt_cifs_symprec_audit.json", payload)

    lines = [
        "# Validation GT CIF Symprec Audit",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Output root: `{OUT_ROOT}`",
        f"- symprec: {args.symprec}",
        f"- angle_tolerance: {args.angle_tolerance}",
        "",
        "| dataset | split | records | non-P1 detected | P1 detected | errors | CSV SG match rate | manifest |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        rate = result["csv_sg_match_rate"]
        lines.append(
            f"| {result['dataset']} | {result['split']} | {result['records']} | {result['non_p1_detected']} | "
            f"{result['p1_detected']} | {result['errors']} | {'NA' if rate is None else f'{rate:.4f}'} | `{result['manifest']}` |"
        )
    lines.extend(
        [
            "",
            "The earlier `cache/official_benchmark_cifs` direct extraction keeps `_symmetry_space_group_name_H-M P 1` for validation rows and is not valid for GT-SG prompt construction.",
        ]
    )
    write_text(ROOT / "reports/validation_gt_cifs_symprec_audit.md", "\n".join(lines))
    append_log(
        f"## {now_iso().replace('T', ' ').replace('+00:00', ' UTC')} validation GT-SG CIF cache repair\n"
        "- Built symprec=0.1 validation CIF cache under cache/official_benchmark_cifs_symprec0p1.\n"
        "- Reason: direct CSV extraction exposes P1 in validation prompts and invalidates GT-SG anchor reproduction.\n"
        "- Artifact: reports/validation_gt_cifs_symprec_audit.md and metrics/validation_gt_cifs_symprec_audit.json."
    )


if __name__ == "__main__":
    main()
