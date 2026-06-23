#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_records is not None and len(rows) >= int(max_records):
                    break
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


def hit_at(keys: list[str], target: str, k: int) -> bool:
    return str(target) in [str(x) for x in keys[: int(k)]]


def summarize(rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    denom = max(1, len(rows))
    out: dict[str, Any] = {
        "samples": len(rows),
        "candidate_nonempty_rate": sum(int(row["candidate_count"] > 0) for row in rows) / denom,
        "unique_wa_mean@50": sum(int(row.get("unique_wa@50", 0)) for row in rows) / denom,
        "unique_skeleton_mean@50": sum(int(row.get("unique_skeleton@50", 0)) for row in rows) / denom,
        "composition_exact_any@50": sum(int(row.get("composition_exact_any@50", False)) for row in rows) / denom,
    }
    for budget in budgets:
        out[f"skeleton@{budget}"] = sum(int(row[f"skeleton_hit@{budget}"]) for row in rows) / denom
        out[f"W/A@{budget}"] = sum(int(row[f"wa_hit@{budget}"]) for row in rows) / denom
        out[f"unique_wa_mean@{budget}"] = sum(int(row[f"unique_wa@{budget}"]) for row in rows) / denom
        out[f"unique_skeleton_mean@{budget}"] = sum(int(row[f"unique_skeleton@{budget}"]) for row in rows) / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate canonical skeleton/W-A prediction recall without CIF rendering.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "val.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="1,5,20,50")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    repr_rows = read_jsonl(args.repr_jsonl, max_records)
    pred_rows = read_jsonl(args.predictions, max_records)
    pred_by_id = {str(row["sample_id"]): row for row in pred_rows}
    per_sample: list[dict[str, Any]] = []
    for idx, record in enumerate(repr_rows):
        sample_id = str(record["keys"]["sample_id"])
        cands = list(pred_by_id.get(sample_id, {}).get("ranked_wa_candidates") or [])
        wa_keys = [str(cand.get("canonical_wa_key") or "") for cand in cands]
        skel_keys = [str(cand.get("canonical_skeleton_key") or "") for cand in cands]
        row: dict[str, Any] = {
            "index": idx,
            "sample_id": sample_id,
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "complex_flag": bool(record.get("complex_flag")),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
            "target_wa_key": str(record["canonical_wa_key"]),
            "candidate_count": len(cands),
            "composition_exact_any@50": any(bool(cand.get("composition_exact")) for cand in cands[:50]),
        }
        for budget in budgets:
            row[f"skeleton_hit@{budget}"] = hit_at(skel_keys, str(record["canonical_skeleton_key"]), budget)
            row[f"wa_hit@{budget}"] = hit_at(wa_keys, str(record["canonical_wa_key"]), budget)
            row[f"unique_skeleton@{budget}"] = len(set(skel_keys[: int(budget)]))
            row[f"unique_wa@{budget}"] = len(set(wa_keys[: int(budget)]))
        per_sample.append(row)

    subsets = {
        "full": per_sample,
        "rows_ge_7": [row for row in per_sample if int(row["row_count"]) >= 7],
        "atoms_ge_12": [row for row in per_sample if int(row["atom_count"]) >= 12],
        "complex_flag": [row for row in per_sample if bool(row["complex_flag"])],
    }
    summary = {
        "predictions": str(args.predictions),
        "repr_jsonl": str(args.repr_jsonl),
        "budgets": budgets,
        "subsets": {name: summarize(rows, budgets) for name, rows in subsets.items()},
    }
    write_jsonl(out_dir / "wa_per_sample.jsonl", per_sample)
    write_json(out_dir / "wa_summary.json", summary)
    lines = ["subset\tbudget\tskeleton\tW/A\tunique_skeleton_mean\tunique_wa_mean"]
    for subset_name, payload in summary["subsets"].items():
        for budget in budgets:
            lines.append(
                "\t".join(
                    [
                        subset_name,
                        str(budget),
                        str(payload[f"skeleton@{budget}"]),
                        str(payload[f"W/A@{budget}"]),
                        str(payload[f"unique_skeleton_mean@{budget}"]),
                        str(payload[f"unique_wa_mean@{budget}"]),
                    ]
                )
            )
    (out_dir / "wa_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["subsets"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
