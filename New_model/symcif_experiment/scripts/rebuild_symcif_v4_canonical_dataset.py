#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.canonicalize import canonical_skeleton_key, canonical_wa_key, wa_table_from_structured
from symcif_v4.orbit_engine import OrbitEngine
from train_skeleton_template_ranker import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild structured SymCIF-v4 records using OrbitToken canonical ids.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_orbit_engine")
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.structured_root)
    stats: dict[str, Any] = {"splits": {}, "failed_map_examples": []}
    template_counter: Counter[str] = Counter()
    wa_counter: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        rows = read_jsonl(args.structured_root / f"{split}.jsonl")
        failures = 0
        with (args.out_root / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                try:
                    wa_table, _free_params = wa_table_from_structured(row, engine)
                    skey = canonical_skeleton_key(wa_table)
                    wkey = canonical_wa_key(wa_table)
                    out = {
                        "id": row["sample_id"],
                        "sample_id": row["sample_id"],
                        "split": split,
                        "formula": row["formula"],
                        "formula_counts": row["formula_counts"],
                        "sg": int(row["sg"]),
                        "sg_symbol": row["sg_symbol"],
                        "wa_table": wa_table,
                        "lattice": row["lattice"],
                        "canonical_wa_key": wkey,
                        "canonical_skeleton_key": skey,
                        "legacy_skeleton_template_key": row.get("skeleton_template_key"),
                        "n_sites": int(row["n_sites"]),
                        "num_elements": int(row["num_elements"]),
                        "atom_count": int(row["atom_count"]),
                    }
                    template_counter[skey] += 1
                    wa_counter[wkey] += 1
                    f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    if len(stats["failed_map_examples"]) < 50:
                        stats["failed_map_examples"].append(
                            {"split": split, "sample_id": row.get("sample_id"), "error": f"{type(exc).__name__}:{exc}"}
                        )
        stats["splits"][split] = {"records": len(rows), "map_failures": failures, "written": len(rows) - failures}
    stats["unique_canonical_skeletons"] = len(template_counter)
    stats["unique_canonical_wa"] = len(wa_counter)
    stats["top_canonical_skeletons"] = [{"key": k, "count": v} for k, v in template_counter.most_common(25)]
    stats["top_canonical_wa"] = [{"key": k, "count": v} for k, v in wa_counter.most_common(25)]
    (args.out_dir / "canonical_dataset_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

