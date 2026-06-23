#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
CRYSTALLM_BIN = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/bin").resolve()
if str(CRYSTALLM_BIN) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_BIN))

import benchmark_metrics as bm  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
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


def grouped_rendered(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["sample_id"])].append(row)
    for items in by_id.values():
        items.sort(key=lambda row: int(row["rank"]))
    return dict(by_id)


def hit_at(keys: list[str], target: str, k: int) -> bool:
    return str(target) in [str(x) for x in keys[: int(k)]]


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize_subset(rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(rows)}
    denom = max(1, len(rows))
    out["candidate_nonempty_rate"] = sum(int(row["candidate_count"] > 0) for row in rows) / denom
    out["unique_wa_mean@50"] = sum(int(row.get("unique_wa@50", 0)) for row in rows) / denom
    out["composition_exact_any@50"] = sum(int(row.get("composition_exact_any@50", False)) for row in rows) / denom
    out["readable_any@50"] = sum(int(row.get("readable_any@50", False)) for row in rows) / denom
    for k in budgets:
        matches = [row[f"match@{k}"] for row in rows]
        rms_vals = [row[f"rmsd@{k}"] for row in rows if row[f"rmsd@{k}"] is not None]
        out[f"match@{k}"] = sum(int(v) for v in matches) / denom
        out[f"RMSE@{k}"] = None if not rms_vals else float(sum(float(v) for v in rms_vals) / len(rms_vals))
        out[f"skeleton@{k}"] = sum(int(row[f"skeleton_hit@{k}"]) for row in rows) / denom
        out[f"W/A@{k}"] = sum(int(row[f"wa_hit@{k}"]) for row in rows) / denom
        out[f"unique_wa_mean@{k}"] = sum(int(row[f"unique_wa@{k}"]) for row in rows) / denom
        out[f"composition_exact_all_candidates@{k}"] = sum(int(row[f"composition_exact_all@{k}"]) for row in rows) / denom
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rendered opentry_3 Wyckoff CIFs on validation StructureMatcher budgets.")
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--true-cif-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="1,5,20,50")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--hard-timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    repr_rows = read_jsonl(args.repr_jsonl, max_records=max_records)
    rendered = grouped_rendered(read_jsonl(args.rendered_jsonl))

    ids = [str(row["keys"]["sample_id"]) for row in repr_rows]
    id_to_true: dict[str, str] = {}
    id_to_gen: dict[str, list[str]] = {}
    for sample_id in ids:
        true_path = args.true_cif_dir / f"{sample_id}.cif"
        id_to_true[sample_id] = true_path.read_text(encoding="utf-8")
        id_to_gen[sample_id] = [str(row["cif"]) for row in rendered.get(sample_id, [])]

    metrics_by_budget: dict[int, dict[str, Any]] = {}
    rms_by_budget: dict[int, dict[str, float | None]] = {}
    for budget in budgets:
        metrics = bm.get_match_rate_and_rms_robust_mp(
            id_to_gen,
            id_to_true,
            n_gens=int(budget),
            length_lo=0.5,
            length_hi=1000.0,
            angle_lo=10.0,
            angle_hi=170.0,
            ltol=0.3,
            stol=0.5,
            angle_tol=10.0,
            max_sites=int(args.max_sites),
            rmsd_timeout_s=float(args.rmsd_timeout_seconds),
            workers=int(args.workers),
            hard_timeout_s=float(args.hard_timeout_seconds),
        )
        metrics_by_budget[budget] = metrics
        rms_by_budget[budget] = dict(getattr(bm, "_BENCH_LAST_RMS_BY_ID", {}))

    per_sample: list[dict[str, Any]] = []
    for record in repr_rows:
        sample_id = str(record["keys"]["sample_id"])
        cands = rendered.get(sample_id, [])
        wa_keys = [str(row.get("canonical_wa_key") or "") for row in cands]
        skel_keys = [str(row.get("canonical_skeleton_key") or "") for row in cands]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "complex_flag": bool(record.get("complex_flag")),
            "candidate_count": len(cands),
            "target_wa_key": str(record["canonical_wa_key"]),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
            "readable_any@50": any(bool(x.get("readable")) for x in cands[:50]),
            "composition_exact_any@50": any(bool(x.get("composition_exact")) for x in cands[:50]),
        }
        for budget in budgets:
            row[f"wa_hit@{budget}"] = hit_at(wa_keys, str(record["canonical_wa_key"]), budget)
            row[f"skeleton_hit@{budget}"] = hit_at(skel_keys, str(record["canonical_skeleton_key"]), budget)
            row[f"unique_wa@{budget}"] = len(set(wa_keys[: int(budget)]))
            row[f"composition_exact_all@{budget}"] = bool(cands[: int(budget)]) and all(bool(x.get("composition_exact")) for x in cands[: int(budget)])
            rms = rms_by_budget[budget].get(sample_id)
            row[f"rmsd@{budget}"] = None if rms is None else float(rms)
            row[f"match@{budget}"] = rms is not None
        per_sample.append(row)

    subsets = {
        "full": per_sample,
        "rows_ge_7": [row for row in per_sample if int(row["row_count"]) >= 7],
        "atoms_ge_12": [row for row in per_sample if int(row["atom_count"]) >= 12],
        "complex_flag": [row for row in per_sample if bool(row["complex_flag"])],
    }
    summary = {
        "rendered_jsonl": str(args.rendered_jsonl),
        "repr_jsonl": str(args.repr_jsonl),
        "true_cif_dir": str(args.true_cif_dir),
        "budgets": budgets,
        "matcher": {
            "ltol": 0.3,
            "stol": 0.5,
            "angle_tol": 10.0,
            "max_sites": int(args.max_sites),
            "rmsd_timeout_seconds": float(args.rmsd_timeout_seconds),
            "hard_timeout_seconds": float(args.hard_timeout_seconds),
            "workers": int(args.workers),
        },
        "budget_raw_metrics": metrics_by_budget,
        "subsets": {name: summarize_subset(rows, budgets) for name, rows in subsets.items()},
    }
    write_jsonl(out_dir / "per_sample_metrics.jsonl", per_sample)
    write_json(out_dir / "summary_metrics.json", summary)
    lines = ["subset\tbudget\tmatch\tRMSE\tskeleton\tW/A\tunique_wa_mean\tcomposition_exact_all"]
    for subset_name, payload in summary["subsets"].items():
        for budget in budgets:
            lines.append(
                "\t".join(
                    [
                        subset_name,
                        str(budget),
                        str(payload.get(f"match@{budget}")),
                        str(payload.get(f"RMSE@{budget}")),
                        str(payload.get(f"skeleton@{budget}")),
                        str(payload.get(f"W/A@{budget}")),
                        str(payload.get(f"unique_wa_mean@{budget}")),
                        str(payload.get(f"composition_exact_all_candidates@{budget}")),
                    ]
                )
            )
    (out_dir / "summary_metrics.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["subsets"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
