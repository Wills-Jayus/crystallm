#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def formula_frac(record: dict[str, Any]) -> dict[str, float]:
    counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
    total = max(1, sum(counts.values()))
    return {k: float(v) / total for k, v in counts.items()}


def formula_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def skeleton_from_record(record: dict[str, Any], *, score: float, source: str, source_sample_id: str | None = None) -> dict[str, Any]:
    return {
        "orbits": [str(row["orbit_id"]) for row in record["skeleton_sequence"]],
        "skeleton_key": str(record["canonical_skeleton_key"]),
        "score": float(score),
        "source": source,
        "source_sample_id": source_sample_id,
        "composition_exact": True,
    }


def build_train_index(train_records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_sg_atom: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        by_sg_atom[(int(record["sg"]), int(record["atom_count"]))].append(record)
    return dict(by_sg_atom)


def prior_skeletons(target: dict[str, Any], pool: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    target_frac = formula_frac(target)
    scored: list[tuple[float, dict[str, Any]]] = []
    for idx, source in enumerate(pool):
        # Uses formula+SG+atom count only. row_count is not used in scoring.
        dist = formula_l1(target_frac, formula_frac(source))
        dist += 0.01 * abs(len(target["formula_counts"]) - len(source["formula_counts"]))
        score = -dist - 1e-5 * idx
        scored.append((score, source))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, source in scored:
        key = str(source["canonical_skeleton_key"])
        if key in seen:
            continue
        seen.add(key)
        out.append(skeleton_from_record(source, score=score, source="train_sg_atom_formula_prior", source_sample_id=str(source["keys"].get("sample_id"))))
        if len(out) >= int(top_n):
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment model skeleton candidates with train-only same-SG/atom-count skeleton priors.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--model-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--prior-top-n", type=int, default=50)
    parser.add_argument("--out-top-n", type=int, default=50)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    train_records = read_jsonl(args.data_dir / "train.jsonl")
    eval_records = read_jsonl(args.data_dir / f"{args.split}.jsonl", int(args.max_records))
    model_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.model_candidates)}
    train_index = build_train_index(train_records)
    out_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for idx, record in enumerate(eval_records):
        sample_id = str(record["keys"]["sample_id"])
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cand in model_by_id.get(sample_id, {}).get("skeleton_candidates") or []:
            key = str(cand.get("skeleton_key") or "")
            if key and key not in seen:
                item = dict(cand)
                item["source"] = str(item.get("source") or "model_beam")
                merged.append(item)
                seen.add(key)
        for cand in prior_skeletons(record, train_index.get((int(record["sg"]), int(record["atom_count"])), []), int(args.prior_top_n)):
            key = str(cand.get("skeleton_key") or "")
            if key and key not in seen:
                merged.append(cand)
                seen.add(key)
        merged = merged[: int(args.out_top_n)]
        keys = [str(c["skeleton_key"]) for c in merged]
        row = {
            "index": idx,
            "sample_id": sample_id,
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "atom_count": int(record["atom_count"]),
            "row_count": int(record["row_count"]),
            "candidate_count": len(merged),
            "unique_skeleton": len(set(keys)),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
        }
        for k in (1, 5, 20, 50):
            row[f"skeleton_hit@{k}"] = str(record["canonical_skeleton_key"]) in keys[:k]
        per_sample.append(row)
        out_rows.append({"sample_id": sample_id, "skeleton_candidates": merged})

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        denom = max(1, len(items))
        return {
            "samples": len(items),
            "candidate_nonempty_rate": sum(int(x["candidate_count"] > 0) for x in items) / denom,
            "unique_skeleton_mean": sum(int(x["unique_skeleton"]) for x in items) / denom,
            **{f"skeleton@{k}": sum(int(x[f"skeleton_hit@{k}"]) for x in items) / denom for k in (1, 5, 20, 50)},
        }

    summary = {
        "records": len(eval_records),
        "split": args.split,
        "model_candidates": str(args.model_candidates),
        "prior_source": "train_same_sg_atom_count_formula_distance",
        "uses_row_count_input": False,
        "prior_top_n": int(args.prior_top_n),
        "out_top_n": int(args.out_top_n),
        "full": summarize(per_sample),
        "rows_ge_7": summarize([x for x in per_sample if int(x["row_count"]) >= 7]),
        "atoms_ge_12": summarize([x for x in per_sample if int(x["atom_count"]) >= 12]),
    }
    write_jsonl(out_dir / "skeleton_candidates.jsonl", out_rows)
    write_jsonl(out_dir / "skeleton_per_sample.jsonl", per_sample)
    write_json(out_dir / "skeleton_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
