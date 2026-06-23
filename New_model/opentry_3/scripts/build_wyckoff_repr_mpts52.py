#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise SystemExit(f"refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_site_key(site: dict[str, Any]) -> str:
    return (
        f"setting={site.get('setting_id') or 'crystalformer'}"
        f"|sg={int(site.get('sg'))}"
        f"|{int(site.get('multiplicity'))}{site.get('letter')}"
        f"|enum={site.get('enumeration')}"
        f"|sym={site.get('site_symmetry') or 'UNKNOWN'}"
    )


def canonical_sort_key(site: dict[str, Any]) -> tuple[Any, ...]:
    enum = site.get("enumeration")
    enum_key = "" if enum is None else str(enum)
    return (
        int(site.get("multiplicity", 0)),
        str(site.get("letter", "")),
        enum_key,
        str(site.get("site_symmetry") or ""),
        str(site.get("element") or ""),
        str(site.get("orbit_id") or row_site_key(site)),
    )


def canonical_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in row.get("wa_table", []):
        orbit_id = str(raw.get("orbit_id") or row_site_key(raw))
        item = {
            "element": str(raw["element"]),
            "orbit_id": orbit_id,
            "site_key": row_site_key(raw),
            "letter": str(raw["letter"]),
            "multiplicity": int(raw["multiplicity"]),
            "enumeration": raw.get("enumeration"),
            "site_symmetry": str(raw.get("site_symmetry") or "UNKNOWN"),
            "setting_id": str(raw.get("setting_id") or "crystalformer"),
            "free_symbols": [str(x) for x in raw.get("free_symbols", [])],
            "representative_expr": [str(x) for x in raw.get("representative_expr", [])],
            "free_param_names": sorted(str(x) for x in (raw.get("free_params") or {}).keys()),
            "expansion_ok": bool(raw.get("expansion_ok", False)),
            "expansion_count": int(raw.get("expansion_count_after_reextract", raw.get("multiplicity", 0))),
            "extraction_success": bool(raw.get("extraction_success", False)),
        }
        out.append(item)
    return sorted(out, key=canonical_sort_key)


def skeleton_key(rows: list[dict[str, Any]]) -> str:
    return "|".join(str(r["orbit_id"]) for r in rows)


def assignment_key(rows: list[dict[str, Any]]) -> str:
    return "|".join(f"{r['orbit_id']}:{r['element']}" for r in rows)


def short_skeleton_key(rows: list[dict[str, Any]]) -> str:
    return "|".join(f"{r['multiplicity']}{r['letter']}" for r in rows)


def short_assignment_key(rows: list[dict[str, Any]]) -> str:
    return "|".join(f"{r['multiplicity']}{r['letter']}:{r['element']}" for r in rows)


def complex_reasons(row_count: int, atom_count: int, num_elements: int, sg: int) -> list[str]:
    reasons: list[str] = []
    if row_count >= 7:
        reasons.append("rows>=7")
    if atom_count >= 12:
        reasons.append("atom_count>=12")
    if num_elements >= 4:
        reasons.append("num_elements>=4")
    if sg in {2, 14, 65, 71, 127, 166, 194, 225, 227}:
        reasons.append("known_hard_or_common_sg")
    return reasons


def make_record(row: dict[str, Any], split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = canonical_rows(row)
    formula_counts = {str(k): int(v) for k, v in row["formula_counts"].items()}
    row_count = len(rows)
    atom_count = int(row.get("atom_count") or sum(formula_counts.values()))
    num_elements = int(row.get("num_elements") or len(formula_counts))
    sg = int(row["sg"])
    skey = skeleton_key(rows)
    akey = assignment_key(rows)
    reasons = complex_reasons(row_count, atom_count, num_elements, sg)
    record = {
        "dataset": "mpts52",
        "split": split,
        "keys": {
            "id": row.get("id"),
            "sample_id": row.get("sample_id"),
            "material_id": row.get("material_id"),
            "input_index": row.get("input_index"),
            "source_path": row.get("source_path"),
        },
        "formula": row.get("formula"),
        "formula_counts": formula_counts,
        "sg": sg,
        "sg_symbol": row.get("sg_symbol"),
        "atom_count": atom_count,
        "num_elements": num_elements,
        "row_count": row_count,
        "complex_flag": bool(reasons),
        "complex_reasons": reasons,
        "canonical_skeleton_key": skey,
        "canonical_wa_key": akey,
        "short_skeleton_key": short_skeleton_key(rows),
        "short_wa_key": short_assignment_key(rows),
        "source_canonical_skeleton_key": row.get("canonical_skeleton_key"),
        "source_canonical_wa_key": row.get("canonical_wa_key"),
        "source_keys_match": {
            "skeleton": row.get("canonical_skeleton_key") == skey,
            "wa": row.get("canonical_wa_key") == akey,
        },
        "skeleton_sequence": [
            {
                "orbit_id": r["orbit_id"],
                "site_key": r["site_key"],
                "letter": r["letter"],
                "multiplicity": r["multiplicity"],
                "enumeration": r["enumeration"],
                "site_symmetry": r["site_symmetry"],
                "free_symbols": r["free_symbols"],
                "representative_expr": r["representative_expr"],
            }
            for r in rows
        ],
        "assignment_sequence": [
            {
                "element": r["element"],
                "orbit_id": r["orbit_id"],
                "multiplicity": r["multiplicity"],
                "letter": r["letter"],
            }
            for r in rows
        ],
        "wa_sequence": [
            {
                "element": r["element"],
                "orbit_id": r["orbit_id"],
                "site_key": r["site_key"],
                "multiplicity": r["multiplicity"],
                "letter": r["letter"],
                "enumeration": r["enumeration"],
                "site_symmetry": r["site_symmetry"],
                "free_param_names": r["free_param_names"],
            }
            for r in rows
        ],
        "element_assignment": [
            {
                "element": r["element"],
                "multiplicity": r["multiplicity"],
                "letter": r["letter"],
                "orbit_id": r["orbit_id"],
            }
            for r in rows
        ],
        "multiplicities": [int(r["multiplicity"]) for r in rows],
        "quality_flags": {
            "row_expansion_all_ok": bool(row.get("row_expansion_all_ok", False)),
            "free_param_reextract_all_success": bool(row.get("free_param_reextract_all_success", False)),
            "all_rows_expansion_ok": all(bool(r["expansion_ok"]) for r in rows),
            "expanded_atom_count": sum(int(r["expansion_count"]) for r in rows),
            "expanded_atom_count_matches_formula": sum(int(r["expansion_count"]) for r in rows) == atom_count,
        },
    }
    audit = {
        "id": row.get("id"),
        "split": split,
        "row_count": row_count,
        "atom_count": atom_count,
        "sg": sg,
        "wa_rows": len(row.get("wa_table", [])),
        "has_formula_counts": isinstance(row.get("formula_counts"), dict),
        "has_canonical_skeleton_key": bool(row.get("canonical_skeleton_key")),
        "has_canonical_wa_key": bool(row.get("canonical_wa_key")),
        "skeleton_key_changed_by_row_canonicalization": row.get("canonical_skeleton_key") != skey,
        "wa_key_changed_by_row_canonicalization": row.get("canonical_wa_key") != akey,
        "expanded_atom_count_matches_formula": record["quality_flags"]["expanded_atom_count_matches_formula"],
        "row_expansion_all_ok": record["quality_flags"]["row_expansion_all_ok"],
        "free_param_reextract_all_success": record["quality_flags"]["free_param_reextract_all_success"],
    }
    return record, audit


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "min": int(ordered[0]),
        "mean": float(mean(ordered)),
        "median": float(median(ordered)),
        "p90": int(ordered[p90_idx]),
        "max": int(ordered[-1]),
    }


def split_summary(records: list[dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts = [int(r["row_count"]) for r in records]
    atom_counts = [int(r["atom_count"]) for r in records]
    return {
        "records": len(records),
        "unique_sg": len({int(r["sg"]) for r in records}),
        "unique_formula": len({str(r["formula"]) for r in records}),
        "unique_skeleton": len({str(r["canonical_skeleton_key"]) for r in records}),
        "unique_wa": len({str(r["canonical_wa_key"]) for r in records}),
        "complex_records": sum(int(bool(r["complex_flag"])) for r in records),
        "rows_ge_7": sum(int(int(r["row_count"]) >= 7) for r in records),
        "atoms_ge_12": sum(int(int(r["atom_count"]) >= 12) for r in records),
        "row_count": quantiles(row_counts),
        "atom_count": quantiles(atom_counts),
        "schema_audit": {
            "skeleton_key_changed_by_row_canonicalization": sum(
                int(bool(a["skeleton_key_changed_by_row_canonicalization"])) for a in audits
            ),
            "wa_key_changed_by_row_canonicalization": sum(
                int(bool(a["wa_key_changed_by_row_canonicalization"])) for a in audits
            ),
            "expanded_atom_count_mismatch": sum(
                int(not bool(a["expanded_atom_count_matches_formula"])) for a in audits
            ),
            "row_expansion_not_all_ok": sum(int(not bool(a["row_expansion_all_ok"])) for a in audits),
            "free_param_reextract_not_all_success": sum(
                int(not bool(a["free_param_reextract_all_success"])) for a in audits
            ),
        },
    }


def train_priors(records: list[dict[str, Any]]) -> dict[str, Any]:
    sg_counts: Counter[str] = Counter()
    element_counts: Counter[str] = Counter()
    orbit_counts: Counter[str] = Counter()
    element_orbit_counts: Counter[str] = Counter()
    sg_orbit_counts: Counter[str] = Counter()
    skeleton_counts: Counter[str] = Counter()
    wa_counts: Counter[str] = Counter()
    repeat_limits: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        sg = str(int(record["sg"]))
        sg_counts[sg] += 1
        skeleton_counts[str(record["canonical_skeleton_key"])] += 1
        wa_counts[str(record["canonical_wa_key"])] += 1
        per_record_repeat: Counter[str] = Counter()
        for element, count in record["formula_counts"].items():
            element_counts[str(element)] += int(count)
        for site in record["wa_sequence"]:
            orbit = str(site["orbit_id"])
            element = str(site["element"])
            orbit_counts[orbit] += 1
            element_orbit_counts[f"{element}@{orbit}"] += 1
            sg_orbit_counts[f"{sg}@{orbit}"] += 1
            per_record_repeat[orbit] += 1
        for orbit, count in per_record_repeat.items():
            repeat_limits[sg][orbit] = max(int(repeat_limits[sg][orbit]), int(count))
    return {
        "source_split": "train",
        "sg_counts": dict(sorted(sg_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "element_atom_counts": dict(sorted(element_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "orbit_row_counts": dict(sorted(orbit_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "element_orbit_counts_top5000": dict(element_orbit_counts.most_common(5000)),
        "sg_orbit_counts_top5000": dict(sg_orbit_counts.most_common(5000)),
        "skeleton_counts_top2000": dict(skeleton_counts.most_common(2000)),
        "wa_counts_top2000": dict(wa_counts.most_common(2000)),
        "repeat_limits_by_sg": {
            sg: dict(sorted(counter.items(), key=lambda kv: kv[0])) for sg, counter in sorted(repeat_limits.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical MPTS-52 Wyckoff representation for opentry_3.")
    parser.add_argument(
        "--structured-root",
        type=Path,
        default=Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment/data/structured_symcif_v4_mpts52"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52",
    )
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict[str, Any] = {}
    train_records: list[dict[str, Any]] = []
    for split in [x.strip() for x in args.splits.split(",") if x.strip()]:
        source_path = args.structured_root / f"{split}.jsonl"
        source_rows = read_jsonl(source_path)
        records: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for row in source_rows:
            record, audit = make_record(row, split)
            records.append(record)
            audits.append(audit)
        write_jsonl(out_dir / f"{split}.jsonl", records)
        write_jsonl(out_dir / f"schema_audit_{split}.jsonl", audits)
        all_summaries[split] = split_summary(records, audits)
        if split == "train":
            train_records = records

    if train_records:
        write_json(out_dir / "train_priors.json", train_priors(train_records))
    write_json(
        out_dir / "build_summary.json",
        {
            "structured_root": str(args.structured_root),
            "out_dir": str(out_dir),
            "splits": all_summaries,
            "notes": [
                "Only train split is used for priors.",
                "row_count is label/analysis only and must not be a test-time input feature.",
                "test split is intentionally not built in E01.",
            ],
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "splits": all_summaries}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
