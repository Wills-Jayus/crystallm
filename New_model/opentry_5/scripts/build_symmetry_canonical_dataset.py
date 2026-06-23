#!/usr/bin/env python3
"""Build grouped splits and a lossless canonical CIF dataset for opentry_5."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
MPTS52_STRUCTURED = WORKSPACE / "model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52/train.jsonl"
MP20_STRUCTURED = WORKSPACE / "model/New_model/symcif_experiment/data/structured_symcif_v4_mp20/train.jsonl"


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def ensure_dir(path: Path) -> None:
    under_root(path).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crystal_system(sg: int | None) -> str:
    if sg is None:
        return "unknown"
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    if 195 <= sg <= 230:
        return "cubic"
    return "unknown"


FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9.]+)?")


def parse_formula_counts(formula: str | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not formula:
        return counts
    for element, raw_count in FORMULA_RE.findall(formula.replace(" ", "")):
        count = float(raw_count) if raw_count else 1.0
        counts[element] = counts.get(element, 0) + int(round(count))
    return counts


def anonymized_reduced_formula(counts: dict[str, int]) -> str:
    vals = [abs(int(v)) for v in counts.values() if v]
    if not vals:
        return "unknown"
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    reduced = sorted(v // max(g, 1) for v in vals)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(f"{labels[i]}{v}" for i, v in enumerate(reduced))


def normalize_wa_table(record: dict) -> list[dict]:
    rows = record.get("wa_table") or record.get("wa_sequence") or record.get("skeleton_sequence") or []
    out = []
    for i, row in enumerate(rows):
        mult = row.get("multiplicity", row.get("declared_multiplicity"))
        try:
            mult = int(mult)
        except Exception:
            mult = 0
        free_symbols = row.get("free_symbols") or row.get("free_param_names") or []
        out.append(
            {
                "site_order": i,
                "element": row.get("element"),
                "multiplicity": mult,
                "letter": row.get("letter") or str(row.get("sg_letter", "")).split("|")[-1],
                "site_symmetry": row.get("site_symmetry"),
                "orbit_id": row.get("orbit_id") or row.get("site_key"),
                "free_symbols": sorted(str(x) for x in free_symbols),
                "free_params": row.get("free_params") or row.get("new_free_params") or {},
                "representative_expr": row.get("representative_expr"),
                "source_coord": row.get("source_coord"),
            }
        )
    return out


def compact_record(record: dict, dataset: str, input_path: Path, line_no: int) -> dict:
    keys = record.get("keys") or {}
    sample_id = record.get("sample_id") or record.get("id") or keys.get("sample_id")
    if not sample_id:
        raise ValueError(f"missing sample_id in {input_path}:{line_no}")
    formula_counts = record.get("formula_counts") or parse_formula_counts(record.get("formula") or record.get("pretty_formula"))
    wa_rows = normalize_wa_table(record)
    row_count = int(record.get("row_count") or record.get("n_sites") or len(wa_rows) or 0)
    sg = record.get("sg")
    sg = int(sg) if sg is not None else None
    mult_pattern = "|".join(str(r["multiplicity"]) for r in wa_rows) or "none"
    letter_seq = "|".join(str(r["letter"]) for r in wa_rows) or "none"
    free_pattern = "|".join(",".join(r["free_symbols"]) if r["free_symbols"] else "-" for r in wa_rows) or "none"
    anon_formula = anonymized_reduced_formula(formula_counts)
    system = crystal_system(sg)
    proto_material = "|".join(
        [
            f"sg={sg}",
            f"formula={anon_formula}",
            f"elements={record.get('num_elements') or len(formula_counts)}",
            f"rows={row_count}",
            f"mult={mult_pattern}",
            f"letters={letter_seq}",
            f"free={free_pattern}",
            f"system={system}",
        ]
    )
    proto_hash = stable_hash(proto_material, 20)
    group_key = f"{dataset}|{proto_material}|proto={proto_hash}"
    source_path = record.get("source_path") or keys.get("source_path")
    return {
        "sample_id": sample_id,
        "material_id": record.get("material_id") or keys.get("material_id"),
        "dataset": dataset,
        "source_split": "train",
        "source_jsonl": str(input_path),
        "source_jsonl_line": line_no,
        "source_path": source_path,
        "formula": record.get("formula") or record.get("pretty_formula"),
        "formula_counts": formula_counts,
        "anonymized_reduced_formula": anon_formula,
        "num_elements": record.get("num_elements") or len(formula_counts),
        "atom_count": record.get("atom_count"),
        "sg": sg,
        "sg_symbol": record.get("sg_symbol"),
        "crystal_system": system,
        "row_count": row_count,
        "rows_ge_7": row_count >= 7,
        "wyckoff_multiplicity_pattern": mult_pattern,
        "anonymized_wyckoff_letter_sequence": letter_seq,
        "free_symbol_pattern": free_pattern,
        "prototype_fingerprint": proto_hash,
        "group_key": group_key,
        "group_key_hash": stable_hash(group_key, 20),
        "canonical_skeleton_key": record.get("canonical_skeleton_key"),
        "canonical_wa_key": record.get("canonical_wa_key"),
        "legacy_skeleton_template_key": record.get("legacy_skeleton_template_key"),
        "lattice": record.get("lattice"),
        "wa_table": wa_rows,
        "quality_flags": {
            "row_expansion_all_ok": record.get("row_expansion_all_ok")
            or (record.get("quality_flags") or {}).get("row_expansion_all_ok"),
            "free_param_reextract_all_success": record.get("free_param_reextract_all_success")
            or (record.get("quality_flags") or {}).get("free_param_reextract_all_success"),
        },
    }


def load_records(max_per_dataset: int | None = None) -> list[dict]:
    records: list[dict] = []
    for dataset, path in [("mpts52", MPTS52_STRUCTURED), ("mp20", MP20_STRUCTURED)]:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if max_per_dataset and line_no > max_per_dataset:
                    break
                records.append(compact_record(json.loads(line), dataset, path, line_no))
    return records


def assign_splits(records: list[dict]) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        by_group[rec["group_key"]].append(rec)
    splits = {"train_core": [], "dev_model": [], "dev_gate": [], "fold_a": [], "fold_b": []}
    for group_key in sorted(by_group):
        bucket = int(stable_hash(group_key, 8), 16) % 1000
        if bucket < 800:
            split = "train_core"
        elif bucket < 900:
            split = "dev_model"
        else:
            split = "dev_gate"
        fold = "fold_a" if (int(stable_hash(group_key + "|fold", 8), 16) % 2 == 0) else "fold_b"
        for rec in by_group[group_key]:
            rec["opentry5_split"] = split
            rec["grouped_dev_fold"] = None if split == "train_core" else fold
            splits[split].append(rec)
            if split != "train_core":
                splits[fold].append(rec)
    for rows in splits.values():
        rows.sort(key=lambda r: (r["dataset"], r["sample_id"]))
    return splits


def split_projection(rec: dict) -> dict:
    keep = [
        "sample_id",
        "material_id",
        "dataset",
        "source_split",
        "source_path",
        "formula",
        "formula_counts",
        "anonymized_reduced_formula",
        "num_elements",
        "atom_count",
        "sg",
        "sg_symbol",
        "crystal_system",
        "row_count",
        "rows_ge_7",
        "wyckoff_multiplicity_pattern",
        "anonymized_wyckoff_letter_sequence",
        "free_symbol_pattern",
        "prototype_fingerprint",
        "group_key",
        "group_key_hash",
        "canonical_skeleton_key",
        "canonical_wa_key",
        "opentry5_split",
        "grouped_dev_fold",
    ]
    return {k: rec.get(k) for k in keep}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def copy_cif_to_canonical(rec: dict) -> str | None:
    source = rec.get("source_path")
    if not source:
        rec["canonical_cif_excluded_reason"] = "missing_source_path"
        return None
    source_path = Path(source)
    if not source_path.exists():
        rec["canonical_cif_excluded_reason"] = "source_path_not_found"
        return None
    try:
        head = source_path.read_text(encoding="utf-8", errors="ignore")[:512]
    except Exception as exc:
        rec["canonical_cif_excluded_reason"] = f"source_read_error:{exc}"
        return None
    if head.strip() == "None" or "data_" not in head:
        rec["canonical_cif_excluded_reason"] = "source_cif_invalid_or_empty"
        return None
    bucket = "canonical_train" if rec["opentry5_split"] == "train_core" else "canonical_dev"
    out_dir = ROOT / "data" / bucket / "cifs" / rec["dataset"]
    ensure_dir(out_dir)
    out_path = out_dir / f"{rec['sample_id']}.cif"
    if not out_path.exists() or out_path.stat().st_size != source_path.stat().st_size:
        shutil.copy2(source_path, out_path)
    rec["canonical_cif_excluded_reason"] = None
    return str(out_path.relative_to(ROOT))


def canonical_projection(rec: dict) -> dict:
    row = dict(rec)
    row["canonical_cif_path"] = copy_cif_to_canonical(rec)
    row["canonical_cif_excluded_reason"] = rec.get("canonical_cif_excluded_reason")
    row["canonical_representation_version"] = "opentry5_symmetry_canonical_v1_lossless_structured_symcif_v4"
    row["candidate_order_policy"] = "not_applicable_training_target"
    row["test_access"] = "none"
    return row


def count_stats(rows: list[dict]) -> dict:
    by_dataset = collections.Counter(r["dataset"] for r in rows)
    rows_ge_7 = sum(1 for r in rows if r.get("rows_ge_7"))
    groups = {r["group_key"] for r in rows}
    return {
        "records": len(rows),
        "groups": len(groups),
        "rows_ge_7": rows_ge_7,
        "datasets": dict(sorted(by_dataset.items())),
    }


def sample_hash(rows: list[dict]) -> str:
    material = "\n".join(f"{r['dataset']}::{r['sample_id']}" for r in rows)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_reports(splits: dict[str, list[dict]], records: list[dict], args: argparse.Namespace) -> None:
    all_groups = {r["group_key"]: r.get("opentry5_split") for r in records}
    split_groups = {name: {r["group_key"] for r in rows} for name, rows in splits.items()}
    train_dev_overlap = sorted(split_groups["train_core"] & (split_groups["dev_model"] | split_groups["dev_gate"]))
    fold_overlap_with_train = sorted((split_groups["fold_a"] | split_groups["fold_b"]) & split_groups["train_core"])
    leakage = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_policy": "deterministic SHA256 bucket on prototype-aware group_key; all identical group_keys stay in one split",
        "group_key_fields": [
            "dataset",
            "space_group",
            "anonymized_reduced_formula",
            "element_count",
            "row_count",
            "wyckoff_multiplicity_pattern",
            "anonymized_wyckoff_letter_sequence",
            "free_symbol_pattern",
            "crystal_system",
            "prototype_fingerprint",
        ],
        "stats": {name: count_stats(rows) for name, rows in splits.items()},
        "train_dev_group_overlap_count": len(train_dev_overlap),
        "fold_train_group_overlap_count": len(fold_overlap_with_train),
        "max_per_dataset": args.max_per_dataset or "all",
        "total_groups": len(all_groups),
        "total_records": len(records),
    }
    write_json(ROOT / "eval/grouped_split_leakage_audit.json", leakage)
    lines = [
        "# Grouped Split Leakage Audit",
        "",
        f"Created: {leakage['created_at']}",
        "",
        "Split policy: deterministic SHA256 bucket on a prototype-aware group key. The key includes SG, anonymized reduced formula, element count, Wyckoff multiplicity and letter sequence, free-symbol pattern, row count, crystal system, and a prototype fingerprint. Identical keys are never split across train/dev.",
        "",
        "| split | records | groups | rows>=7 | datasets |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ["train_core", "dev_model", "dev_gate", "fold_a", "fold_b"]:
        st = leakage["stats"][name]
        lines.append(f"| {name} | {st['records']} | {st['groups']} | {st['rows_ge_7']} | {json.dumps(st['datasets'], sort_keys=True)} |")
    lines += [
        "",
        f"Train/dev group overlap count: {len(train_dev_overlap)}.",
        f"Grouped fold/train overlap count: {len(fold_overlap_with_train)}.",
        "",
        "Status: pass for exact prototype-key leakage. Similarity beyond the explicit key is not used to move samples back into train; future stricter clustering must update the manifest before any model comparison.",
    ]
    write_text(ROOT / "reports/grouped_split_leakage_audit.md", "\n".join(lines) + "\n")

    canonical_lines = [
        "# Canonical Representation Report",
        "",
        f"Created: {leakage['created_at']}",
        "",
        "Representation: `opentry5_symmetry_canonical_v1_lossless_structured_symcif_v4`.",
        "",
        "The dataset preserves the structured SymCIF-v4 fields already extracted with symmetry-aware Wyckoff rows, lattice parameters, free-parameter metadata, source coordinates, and a frozen copy of the source CIF under `data/canonical_train` or `data/canonical_dev`. This first representation is lossless by design; destructive origin/axis equivalence reduction is deferred until it can be proven by round-trip audit.",
        "",
        "Key fields: formula, GT-SG, crystal system, canonical skeleton key, canonical W/A key, row order, multiplicity, Wyckoff letter, free-symbol mask, free params, lattice, source coordinate, and canonical CIF path.",
        "",
        "| split | canonical records | CIF copies |",
        "|---|---:|---:|",
    ]
    for name in ["train_core", "dev_model", "dev_gate", "fold_a", "fold_b"]:
        rows = splits[name]
        copied = sum(1 for r in rows if r.get("canonical_cif_path"))
        canonical_lines.append(f"| {name} | {len(rows)} | {copied} |")
    canonical_lines += [
        "",
        "Round-trip audit is executed by `scripts/audit_canonical_roundtrip.py` and writes `eval/canonical_roundtrip_audit.json`.",
    ]
    write_text(ROOT / "reports/canonical_representation_report.md", "\n".join(canonical_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-dataset", type=int, default=0, help="debug only; 0 means all records")
    args = parser.parse_args()
    for rel in [
        "data",
        "data/canonical_train",
        "data/canonical_dev",
        "eval",
        "reports",
        "manifests",
        "configs",
        "logs",
        "tmp",
    ]:
        ensure_dir(ROOT / rel)

    records = load_records(args.max_per_dataset or None)
    splits = assign_splits(records)

    for name, rows in splits.items():
        write_jsonl(ROOT / "data" / f"grouped_split_{name}.jsonl", [split_projection(r) for r in rows])

    canonical_by_split = {name: [canonical_projection(r) for r in rows] for name, rows in splits.items()}
    write_jsonl(ROOT / "data/canonical_train/train_core.jsonl", canonical_by_split["train_core"])
    write_jsonl(ROOT / "data/canonical_dev/dev_model.jsonl", canonical_by_split["dev_model"])
    write_jsonl(ROOT / "data/canonical_dev/dev_gate.jsonl", canonical_by_split["dev_gate"])
    write_jsonl(ROOT / "data/canonical_dev/fold_a.jsonl", canonical_by_split["fold_a"])
    write_jsonl(ROOT / "data/canonical_dev/fold_b.jsonl", canonical_by_split["fold_b"])

    # Update in-memory rows with canonical paths for report counts.
    by_id = {r["sample_id"]: r for rows in canonical_by_split.values() for r in rows}
    for rows in splits.values():
        for r in rows:
            r["canonical_cif_path"] = by_id.get(r["sample_id"], {}).get("canonical_cif_path")

    build_reports(splits, records, args)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workdir": str(ROOT),
        "environment": "crystallm_env",
        "inputs": {
            "mpts52_structured_train": str(MPTS52_STRUCTURED),
            "mp20_structured_train": str(MP20_STRUCTURED),
        },
        "split_files": {
            name: str((ROOT / "data" / f"grouped_split_{name}.jsonl").relative_to(ROOT))
            for name in ["train_core", "dev_model", "dev_gate", "fold_a", "fold_b"]
        },
        "canonical_files": {
            "train_core": "data/canonical_train/train_core.jsonl",
            "dev_model": "data/canonical_dev/dev_model.jsonl",
            "dev_gate": "data/canonical_dev/dev_gate.jsonl",
            "fold_a": "data/canonical_dev/fold_a.jsonl",
            "fold_b": "data/canonical_dev/fold_b.jsonl",
        },
        "split_stats": {name: count_stats(rows) for name, rows in splits.items()},
        "sample_hashes": {name: sample_hash(rows) for name, rows in splits.items()},
        "test_access": "none",
    }
    write_json(ROOT / "manifests/canonical_sample_manifest.json", manifest)

    data_hashes = {
        "created_at": manifest["created_at"],
        "input_hashes": {
            str(MPTS52_STRUCTURED): sha256_file(MPTS52_STRUCTURED),
            str(MP20_STRUCTURED): sha256_file(MP20_STRUCTURED),
        },
        "generated_hashes": {
            str((ROOT / "data" / f"grouped_split_{name}.jsonl").relative_to(ROOT)): sha256_file(ROOT / "data" / f"grouped_split_{name}.jsonl")
            for name in ["train_core", "dev_model", "dev_gate", "fold_a", "fold_b"]
        },
        "canonical_jsonl_hashes": {
            "data/canonical_train/train_core.jsonl": sha256_file(ROOT / "data/canonical_train/train_core.jsonl"),
            "data/canonical_dev/dev_model.jsonl": sha256_file(ROOT / "data/canonical_dev/dev_model.jsonl"),
            "data/canonical_dev/dev_gate.jsonl": sha256_file(ROOT / "data/canonical_dev/dev_gate.jsonl"),
            "data/canonical_dev/fold_a.jsonl": sha256_file(ROOT / "data/canonical_dev/fold_a.jsonl"),
            "data/canonical_dev/fold_b.jsonl": sha256_file(ROOT / "data/canonical_dev/fold_b.jsonl"),
        },
        "test_access": "none",
    }
    write_json(ROOT / "manifests/canonical_data_hashes.json", data_hashes)

    yaml = """# opentry_5 canonical evaluation protocol
workdir: /data/users/xsw/autodlmini/model/New_model/opentry_5
environment: crystallm_env
test_access: none
candidate_order: generation_index_then_seed
candidate_count:
  primary_k: 20
  extended_k: 50
invalid_handling:
  keep_invalid_slot: true
  do_not_shift_later_candidates_forward: true
structure_matcher:
  ltol: 0.2
  stol: 0.3
  angle_tol: 5.0
  primitive_cell: true
  scale: true
  attempt_supercell: false
rmse:
  definition: best_structurematcher_rmsd_among_first_k_positive_candidates
rows_ge_7:
  definition: row_count >= 7 from canonical structured train/dev metadata
composition_exact:
  definition: reduced composition equality after parse; invalid candidates count as false
sg_wyckoff_legal:
  definition: parsed CIF detected SG and rendered Wyckoff rows satisfy GT-SG legal orbits
splits:
  train_core: data/grouped_split_train_core.jsonl
  dev_model: data/grouped_split_dev_model.jsonl
  dev_gate: data/grouped_split_dev_gate.jsonl
  fold_a: data/grouped_split_fold_a.jsonl
  fold_b: data/grouped_split_fold_b.jsonl
no_ranking:
  forbidden: [log_probability_sort, confidence_sort, energy_sort, collision_sort, validity_sort, self_score_sort, oracle_selection, invalid_drop_and_shift]
"""
    write_text(ROOT / "configs/canonical_evaluation.yaml", yaml)

    print(json.dumps({"status": "ok", "records": len(records), "split_stats": manifest["split_stats"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
