#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id(row: dict[str, Any]) -> str:
    keys = row.get("keys") or {}
    return str(keys.get("sample_id") or keys.get("id") or row.get("sample_id") or "")


def split_key_entries(key: str) -> list[str]:
    text = str(key or "")
    if not text:
        return []
    text = re.sub(r"\|(?=setting=crystalformer)", "\n", text)
    return [part for part in text.splitlines() if part]


def split_wa_entries(key: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for entry in split_key_entries(key):
        if ":" in entry:
            orbit, element = entry.rsplit(":", 1)
        else:
            orbit, element = entry, ""
        out.append((orbit, element))
    return out


def load_repr_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = sample_id(row)
        if sid:
            out[sid] = row
    return out


def formula_counts(row: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in (row.get("formula_counts") or {}).items()}


def formula_l1(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    total = max(1, sum(a.values()))
    return sum(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in keys) / float(total)


def element_jaccard(a: dict[str, int] | set[str], b: dict[str, int] | set[str]) -> float:
    aset = set(a)
    bset = set(b)
    if not aset and not bset:
        return 1.0
    return len(aset & bset) / max(1, len(aset | bset))


def multiset_overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())


def free_param_stats(row: dict[str, Any]) -> tuple[int, int]:
    rows = row.get("wa_sequence") or row.get("skeleton_sequence") or []
    row_count = 0
    total = 0
    for item in rows:
        names = item.get("free_param_names") or item.get("free_symbols") or []
        if names:
            row_count += 1
            total += len(names)
    return row_count, total


def enrich_row(row: dict[str, Any], target_repr: dict[str, Any] | None, source_repr: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(row)
    target_formula = formula_counts(target_repr or {})
    source_formula = formula_counts(source_repr or {})
    candidate_pairs = split_wa_entries(str(row.get("canonical_wa_key") or ""))
    candidate_orbits = [orbit for orbit, _ in candidate_pairs]
    candidate_elements = [element for _, element in candidate_pairs if element]
    source_pairs = split_wa_entries(str((source_repr or {}).get("canonical_wa_key") or ""))
    source_orbits = [orbit for orbit, _ in source_pairs]
    source_elements = [element for _, element in source_pairs if element]
    cand_orbit_counter = Counter(candidate_orbits)
    source_orbit_counter = Counter(source_orbits)
    cand_pair_counter = Counter(candidate_pairs)
    source_pair_counter = Counter(source_pairs)
    source_free_rows, source_free_total = free_param_stats(source_repr or {})
    cand_rows = len(candidate_pairs)
    source_rows = int((source_repr or {}).get("row_count") or 0)
    source_atom_count = int((source_repr or {}).get("atom_count") or 0)
    target_atom_count = int((target_repr or {}).get("atom_count") or row.get("atom_count") or 0)
    common_orbits = multiset_overlap(cand_orbit_counter, source_orbit_counter)
    common_pairs = multiset_overlap(cand_pair_counter, source_pair_counter)
    source_sg = int((source_repr or {}).get("sg") or 0)
    target_sg = int(row.get("sg") or (target_repr or {}).get("sg") or 0)
    item.update(
        {
            "source_pair_has_source": source_repr is not None,
            "source_pair_source_sg": source_sg,
            "source_pair_source_sg_same": source_sg == target_sg if source_sg else False,
            "source_pair_source_atom_count": source_atom_count,
            "source_pair_source_formula_element_count": len(source_formula),
            "source_pair_source_row_count": source_rows,
            "source_pair_source_complex_flag": bool((source_repr or {}).get("complex_flag")),
            "source_pair_candidate_row_count": cand_rows,
            "source_pair_candidate_unique_orbit_count": len(set(candidate_orbits)),
            "source_pair_candidate_duplicate_orbit_count": max(0, cand_rows - len(set(candidate_orbits))),
            "source_pair_candidate_unique_element_count": len(set(candidate_elements)),
            "source_pair_source_free_param_rows": source_free_rows,
            "source_pair_source_free_param_total": source_free_total,
            "source_pair_formula_l1": formula_l1(target_formula, source_formula) if source_formula else 4.0,
            "source_pair_formula_element_jaccard": element_jaccard(target_formula, source_formula) if source_formula else 0.0,
            "source_pair_candidate_source_element_jaccard": element_jaccard(set(candidate_elements), set(source_elements)),
            "source_pair_atom_count_ratio_log": math.log((source_atom_count + 1.0) / (target_atom_count + 1.0)),
            "source_pair_row_count_delta": float(source_rows - cand_rows),
            "source_pair_abs_row_count_delta": float(abs(source_rows - cand_rows)),
            "source_pair_common_orbit_count": common_orbits,
            "source_pair_common_orbit_frac_candidate": common_orbits / max(1, cand_rows),
            "source_pair_common_orbit_frac_source": common_orbits / max(1, source_rows),
            "source_pair_common_wa_pair_count": common_pairs,
            "source_pair_common_wa_frac_candidate": common_pairs / max(1, cand_rows),
            "source_pair_common_wa_frac_source": common_pairs / max(1, source_rows),
            "source_pair_same_canonical_skeleton": str(row.get("canonical_skeleton_key") or "")
            == str((source_repr or {}).get("canonical_skeleton_key") or ""),
            "source_pair_same_canonical_wa": str(row.get("canonical_wa_key") or "")
            == str((source_repr or {}).get("canonical_wa_key") or ""),
        }
    )
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Add GT-free source-pair context features to rendered candidate feature rows.")
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument("--target-repr-jsonl", type=Path, required=True)
    parser.add_argument("--source-train-repr-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "train.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_map = load_repr_map(args.target_repr_jsonl)
    source_map = load_repr_map(args.source_train_repr_jsonl)
    rows = read_jsonl(args.features_jsonl)
    enriched: list[dict[str, Any]] = []
    missing_target = 0
    missing_source = 0
    for row in rows:
        target = target_map.get(str(row.get("sample_id")))
        source = source_map.get(str(row.get("source_sample_id")))
        if target is None:
            missing_target += 1
        if source is None:
            missing_source += 1
        enriched.append(enrich_row(row, target, source))
    write_jsonl(out_dir / f"{args.split}_candidate_features_sourcepair.jsonl", enriched)
    summary = {
        "split": str(args.split),
        "features_jsonl": str(args.features_jsonl),
        "target_repr_jsonl": str(args.target_repr_jsonl),
        "source_train_repr_jsonl": str(args.source_train_repr_jsonl),
        "rows": len(rows),
        "missing_target_rows": missing_target,
        "missing_source_rows": missing_source,
        "feature_note": "All source_pair_* fields are GT-free at inference; target formula/SG are inputs, source metadata comes from train source records, and target row_count/labels are not added as features.",
    }
    write_json(out_dir / f"{args.split}_sourcepair_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
