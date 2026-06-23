#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pymatgen.core import Composition  # type: ignore  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif_v3.parse import parse_symcif_v3_text  # noqa: E402


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


def composition_counts(formula: str) -> dict[str, int]:
    comp = Composition(formula)
    out: dict[str, int] = {}
    for el, amt in comp.as_dict().items():
        rounded = int(round(float(amt)))
        if not math.isclose(float(amt), rounded, abs_tol=1e-6):
            raise ValueError(f"non-integer formula amount: {formula} -> {el}={amt}")
        out[str(el)] = rounded
    return dict(sorted(out.items()))


def canonical_site_key(site: Any) -> tuple[int, str, int, str, int]:
    enum = -1 if site.enumeration is None else int(site.enumeration)
    return (int(site.multiplicity), str(site.letter), enum, str(site.site_symmetry or ""), int(site.index))


def canonical_sites(record: Any) -> list[Any]:
    return sorted(record.sites, key=canonical_site_key)


def skeleton_key_from_sites(sites: list[Any]) -> str:
    return "|".join(f"{int(site.multiplicity)}{site.letter}" for site in sites)


def template_uid(sg_number: int, skeleton_key: str) -> str:
    return f"sg={int(sg_number)}|{skeleton_key}"


@lru_cache(maxsize=250_000)
def subset_remainders(remaining: tuple[int, ...], target: int) -> tuple[tuple[int, ...], ...]:
    out: set[tuple[int, ...]] = set()
    n = len(remaining)

    def rec(pos: int, total: int, chosen: list[int]) -> None:
        if total == target:
            chosen_set = set(chosen)
            out.add(tuple(remaining[i] for i in range(n) if i not in chosen_set))
            return
        if total > target or pos >= n:
            return
        prev: int | None = None
        for i in range(pos, n):
            val = remaining[i]
            if prev == val:
                continue
            prev = val
            chosen.append(i)
            rec(i + 1, total + val, chosen)
            chosen.pop()

    rec(0, 0, [])
    return tuple(sorted(out))


@lru_cache(maxsize=250_000)
def multiplicities_compatible(multiplicities: tuple[int, ...], counts: tuple[int, ...]) -> bool:
    remaining = tuple(sorted(int(v) for v in multiplicities))
    targets = tuple(sorted((int(c) for c in counts if int(c) > 0), reverse=True))
    if sum(remaining) != sum(targets):
        return False
    if not targets:
        return not remaining
    if len(targets) > len(remaining):
        return False

    @lru_cache(maxsize=None)
    def rec(rem: tuple[int, ...], idx: int) -> bool:
        if idx == len(targets):
            return not rem
        target = targets[idx]
        for next_rem in subset_remainders(rem, target):
            if rec(next_rem, idx + 1):
                return True
        return False

    return rec(remaining, 0)


def is_formula_compatible(multiplicities: list[int], formula_counts: dict[str, int]) -> bool:
    return multiplicities_compatible(
        tuple(sorted(int(v) for v in multiplicities)),
        tuple(sorted((int(v) for v in formula_counts.values()), reverse=True)),
    )


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p90": None, "max": None}
    sorted_values = sorted(values)
    p90_idx = min(len(sorted_values) - 1, int(round(0.9 * (len(sorted_values) - 1))))
    return {
        "min": int(sorted_values[0]),
        "median": float(statistics.median(sorted_values)),
        "mean": float(statistics.mean(sorted_values)),
        "p90": int(sorted_values[p90_idx]),
        "max": int(sorted_values[-1]),
    }


def json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not json serializable: {type(obj)!r}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    import pandas as pd  # type: ignore

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        for key in ("formula_counts", "assignment", "free_coords", "lattice", "multiplicities"):
            if key in flat:
                flat[key] = json.dumps(flat[key], ensure_ascii=False, sort_keys=True, default=json_default)
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_parquet(path, index=False)
    return True


def make_rows(corpus_dir: Path, lookup: WyckoffLookup) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = corpus_dir / f"{split}.txt"
        for idx, text in enumerate(split_concat_records(path)):
            rec = parse_symcif_v3_text(text, lookup, source_path=path)
            sites = canonical_sites(rec)
            skey = skeleton_key_from_sites(sites)
            counts = composition_counts(rec.cell_formula)
            multiplicities = [int(site.multiplicity) for site in sites]
            assignment = [
                {
                    "site_order": i,
                    "multiplicity": int(site.multiplicity),
                    "letter": str(site.letter),
                    "element": str(site.element),
                }
                for i, site in enumerate(sites)
            ]
            free_coords = [
                {
                    "site_order": i,
                    "multiplicity": int(site.multiplicity),
                    "letter": str(site.letter),
                    "free_mask": [bool(v) for v in site.free_mask],
                    "x": float(site.representative_coord[0]),
                    "y": float(site.representative_coord[1]),
                    "z": float(site.representative_coord[2]),
                }
                for i, site in enumerate(sites)
            ]
            lattice = {
                "a": float(rec.lattice.a),
                "b": float(rec.lattice.b),
                "c": float(rec.lattice.c),
                "alpha": float(rec.lattice.alpha),
                "beta": float(rec.lattice.beta),
                "gamma": float(rec.lattice.gamma),
                "volume": float(rec.lattice.volume),
            }
            rows.append(
                {
                    "split": split,
                    "split_index": idx,
                    "sample_id": rec.sample_id,
                    "formula": rec.cell_formula,
                    "formula_counts": counts,
                    "atom_count": int(sum(counts.values())),
                    "num_elements": int(len(counts)),
                    "sg": int(rec.sg_number),
                    "sg_symbol": rec.sg_symbol,
                    "z": int(rec.z),
                    "n_sites": int(len(sites)),
                    "skeleton_template_key": skey,
                    "skeleton_template_uid": template_uid(rec.sg_number, skey),
                    "multiplicities": multiplicities,
                    "assignment": assignment,
                    "free_coords": free_coords,
                    "lattice": lattice,
                    "gt_formula_compatible": bool(is_formula_compatible(multiplicities, counts)),
                }
            )
    return rows


def build_template_catalog(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_uid: dict[str, dict[str, Any]] = {}
    freq: Counter[str] = Counter()
    split_freq: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    for row in rows:
        uid = row["skeleton_template_uid"]
        freq[uid] += 1
        split_freq[row["split"]][uid] += 1
        if uid not in by_uid:
            by_uid[uid] = {
                "skeleton_template_uid": uid,
                "sg": int(row["sg"]),
                "skeleton_template_key": row["skeleton_template_key"],
                "multiplicities": list(row["multiplicities"]),
                "n_sites": int(row["n_sites"]),
                "atom_count": int(sum(row["multiplicities"])),
            }
    ordered = sorted(by_uid.values(), key=lambda item: (item["sg"], item["skeleton_template_key"]))
    uid_to_id = {item["skeleton_template_uid"]: f"T{i:06d}" for i, item in enumerate(ordered)}
    catalog: list[dict[str, Any]] = []
    for item in ordered:
        uid = item["skeleton_template_uid"]
        out = dict(item)
        out["skeleton_template_id"] = uid_to_id[uid]
        out["frequency"] = int(freq[uid])
        out["train_frequency"] = int(split_freq["train"][uid])
        out["val_frequency"] = int(split_freq["val"][uid])
        out["test_frequency"] = int(split_freq["test"][uid])
        catalog.append(out)
    return catalog, uid_to_id


def compute_stats(rows: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    train_templates = {item["skeleton_template_uid"]: item for item in catalog if item["train_frequency"] > 0}
    train_by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_by_sg: dict[int, set[str]] = defaultdict(set)
    for item in catalog:
        all_by_sg[int(item["sg"])].add(item["skeleton_template_uid"])
        if item["train_frequency"] > 0:
            train_by_sg[int(item["sg"])].append(item)

    compatible_train_counts: list[int] = []
    gt_seen_train = 0
    gt_seen_train_and_compatible = 0
    has_any_compatible = 0
    test_rows = [row for row in rows if row["split"] == "test"]
    for row in test_rows:
        counts = row["formula_counts"]
        compatible = [
            item
            for item in train_by_sg[int(row["sg"])]
            if is_formula_compatible(list(item["multiplicities"]), counts)
        ]
        compatible_train_counts.append(len(compatible))
        if compatible:
            has_any_compatible += 1
        if row["skeleton_template_uid"] in train_templates:
            gt_seen_train += 1
            item = train_templates[row["skeleton_template_uid"]]
            if is_formula_compatible(list(item["multiplicities"]), counts):
                gt_seen_train_and_compatible += 1

    per_sg_template_count = {
        str(sg): len(uids)
        for sg, uids in sorted(all_by_sg.items(), key=lambda kv: kv[0])
    }
    per_sg_train_template_count = {
        str(sg): len(items)
        for sg, items in sorted(train_by_sg.items(), key=lambda kv: kv[0])
    }
    freq_rows = sorted(catalog, key=lambda item: (-int(item["frequency"]), int(item["sg"]), item["skeleton_template_key"]))
    return {
        "num_records": len(rows),
        "split_records": dict(Counter(row["split"] for row in rows)),
        "unique_templates": len(catalog),
        "unique_templates_by_split": {
            split: len({row["skeleton_template_uid"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "unique_train_templates": len(train_templates),
        "template_frequency_top20": [
            {
                "skeleton_template_id": item["skeleton_template_id"],
                "skeleton_template_uid": item["skeleton_template_uid"],
                "frequency": int(item["frequency"]),
                "train_frequency": int(item["train_frequency"]),
                "val_frequency": int(item["val_frequency"]),
                "test_frequency": int(item["test_frequency"]),
            }
            for item in freq_rows[:20]
        ],
        "per_sg_template_count": per_sg_template_count,
        "per_sg_train_template_count": per_sg_train_template_count,
        "gt_formula_compatible_rate": sum(bool(row["gt_formula_compatible"]) for row in rows) / max(1, len(rows)),
        "per_formula_atom_count_compatibility": {
            "test_samples": len(test_rows),
            "compatible_train_templates_same_sg": quantiles(compatible_train_counts),
            "has_any_formula_compatible_train_template_rate": has_any_compatible / max(1, len(test_rows)),
            "zero_compatible_train_template_count": int(sum(v == 0 for v in compatible_train_counts)),
        },
        "gt_template_retrieval_upper_bound": {
            "definition": "A ranker trained only on train templates cannot retrieve test GT templates unseen in train.",
            "test_samples": len(test_rows),
            "gt_template_seen_in_train": gt_seen_train,
            "gt_template_seen_in_train_rate": gt_seen_train / max(1, len(test_rows)),
            "gt_template_seen_in_train_and_formula_compatible": gt_seen_train_and_compatible,
            "gt_template_seen_in_train_and_formula_compatible_rate": gt_seen_train_and_compatible / max(1, len(test_rows)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export structured SymCIF-v3 skeleton dataset.")
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "data" / "symcif_v3")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "structured_symcif_v3")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    lookup = WyckoffLookup.from_json(args.lookup_json)

    rows = make_rows(args.corpus_dir, lookup)
    catalog, uid_to_id = build_template_catalog(rows)
    for row in rows:
        row["skeleton_template_id"] = uid_to_id[row["skeleton_template_uid"]]

    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
        write_parquet(args.out_dir / f"{split}.parquet", split_rows)
    write_jsonl(args.out_dir / "all.jsonl", rows)
    write_jsonl(args.out_dir / "template_catalog.jsonl", catalog)
    write_parquet(args.out_dir / "all.parquet", rows)
    write_parquet(args.out_dir / "template_catalog.parquet", catalog)

    stats = compute_stats(rows, catalog)
    (args.out_dir / "template_id_map.json").write_text(
        json.dumps(uid_to_id, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.reports_dir / "structured_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
