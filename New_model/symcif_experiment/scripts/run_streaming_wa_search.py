#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine
from symcif_v4.wa_search import build_search_priors, streaming_exact_cover_search

TOP_KS = (1, 5, 20, 100, 200, 700)


_ENGINE: OrbitEngine | None = None
_PRIORS: dict[str, Counter[str]] | None = None
_ARGS: dict[str, Any] = {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def init_worker(lookup_json: str, data_root: str, priors: dict[str, Counter[str]], args_dict: dict[str, Any]) -> None:
    global _ENGINE, _PRIORS, _ARGS
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, data_root)
    _PRIORS = priors
    _ARGS = args_dict


def q(values: list[int | float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    vals = sorted(float(v) for v in values)
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
        "max": vals[-1],
    }


def hit_at(candidates: list[dict[str, Any]], key: str, field: str, k: int) -> bool:
    return any(str(c.get(field)) == str(key) for c in candidates[:k])


def process_record(payload: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, record = payload
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    engine = _ENGINE
    args = _ARGS
    raw_top_k = int(args["top_k"]) * int(args.get("candidate_multiplier", 1))
    candidates, stats = streaming_exact_cover_search(
        int(record["sg"]),
        {str(k): int(v) for k, v in record["formula_counts"].items()},
        engine.get_orbits(int(record["sg"])),
        priors=_PRIORS,
        beam_size=int(args["beam_size"]),
        top_k=raw_top_k,
        max_expanded_states=int(args["max_expanded_states"]),
        timeout_s=float(args["timeout_per_sample"]),
    )
    candidates = rerank_and_trim_candidates(candidates, _PRIORS or {}, int(args["top_k"]))
    gt_skeleton = str(record["canonical_skeleton_key"])
    gt_wa = str(record["canonical_wa_key"])
    summary: dict[str, Any] = {
        "input_index": index,
        "split": record["split"],
        "sample_id": record["sample_id"],
        "formula": record["formula"],
        "formula_counts": record["formula_counts"],
        "sg": int(record["sg"]),
        "n_sites": int(record["n_sites"]),
        "num_elements": int(record["num_elements"]),
        "total_atoms": int(record["atom_count"]),
        "candidate_count": len(candidates),
        "candidate_nonempty": bool(candidates),
        "expanded_states": int(stats.expanded_states),
        "generated_states": int(stats.generated_states),
        "complete_states": int(stats.complete_states),
        "timeout": bool(stats.timeout),
        "truncated": bool(stats.truncated),
        "elapsed_s": float(stats.elapsed_s),
        "reason": stats.reason,
        "gt_skeleton_key": gt_skeleton,
        "gt_wa_key": gt_wa,
    }
    for k in TOP_KS:
        summary[f"gt_skeleton_in_top{k}"] = hit_at(candidates, gt_skeleton, "canonical_skeleton_key", k)
        summary[f"gt_wa_in_top{k}"] = hit_at(candidates, gt_wa, "canonical_wa_key", k)
    return {
        "input_index": index,
        "candidate_row": {
            "split": record["split"],
            "sample_id": record["sample_id"],
            "formula": record["formula"],
            "formula_counts": record["formula_counts"],
            "sg": int(record["sg"]),
            "n_sites": int(record["n_sites"]),
            "num_elements": int(record["num_elements"]),
            "gt_skeleton_key": gt_skeleton,
            "gt_wa_key": gt_wa,
            "candidates": candidates,
        },
        "summary": summary,
    }


def rerank_and_trim_candidates(candidates: list[dict[str, Any]], priors: dict[str, Counter[str]], top_k: int) -> list[dict[str, Any]]:
    if len(candidates) <= int(top_k):
        return sorted(candidates, key=lambda c: (-float(c.get("score", 0.0)), str(c.get("canonical_wa_key"))))
    skeleton_counts = priors.get("skeleton_counts", Counter())
    wa_counts = priors.get("wa_counts", Counter())
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        c = dict(candidate)
        base = float(c.get("score", 0.0))
        prior_score = 8.0 * (wa_counts.get(str(c["canonical_wa_key"]), 0) > 0)
        prior_score += 3.0 * math.log1p(skeleton_counts.get(str(c["canonical_skeleton_key"]), 0))
        c["search_score"] = base
        c["score"] = base + prior_score
        scored.append(c)
    scored.sort(key=lambda c: (-float(c.get("score", 0.0)), str(c.get("canonical_wa_key"))))
    return scored[: int(top_k)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "samples": len(rows),
        "candidate_nonempty_rate": sum(bool(r["candidate_nonempty"]) for r in rows) / max(1, len(rows)),
        "timeout_samples": sum(bool(r["timeout"]) for r in rows),
        "truncated_samples": sum(bool(r["truncated"]) for r in rows),
        "candidate_count": q([r["candidate_count"] for r in rows]),
        "expanded_states": q([r["expanded_states"] for r in rows]),
        "elapsed_s": q([r["elapsed_s"] for r in rows]),
    }
    for k in TOP_KS:
        out[f"gt_skeleton_in_top{k}"] = sum(bool(r[f"gt_skeleton_in_top{k}"]) for r in rows) / max(1, len(rows))
        out[f"gt_wa_in_top{k}"] = sum(bool(r[f"gt_wa_in_top{k}"]) for r in rows) / max(1, len(rows))
    out["gate2_search"] = {
        "skeleton_top200_pass": out["gt_skeleton_in_top200"] >= 0.99,
        "wa_top200_pass": out["gt_wa_in_top200"] >= 0.90,
        "timeout_pass": out["timeout_samples"] == 0,
    }
    return out


def group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        item: dict[str, Any] = {key: value, "samples": len(items)}
        s = summarize(items)
        for name in (
            "candidate_nonempty_rate",
            "gt_skeleton_in_top20",
            "gt_skeleton_in_top100",
            "gt_skeleton_in_top200",
            "gt_skeleton_in_top700",
            "gt_wa_in_top20",
            "gt_wa_in_top100",
            "gt_wa_in_top200",
            "gt_wa_in_top700",
            "timeout_samples",
            "truncated_samples",
        ):
            item[name] = s[name]
        item["candidate_count_mean"] = s["candidate_count"]["mean"]
        item["expanded_states_median"] = s["expanded_states"]["median"]
        out.append(item)
    return out


def merge_summary(out_dir: Path, split: str, summary: dict[str, Any]) -> None:
    path = out_dir / "search_summary.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = {"splits": {}}
    raw.setdefault("splits", {})[split] = summary
    path.write_text(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming composition-exact SymCIF-v4 WA search.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--beam-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--max-expanded-states", type=int, default=200_000)
    parser.add_argument("--timeout-per-sample", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.data_root / "train.jsonl")
    priors = build_search_priors(train_records)
    rows = read_jsonl(args.data_root / f"{args.split}.jsonl")
    if args.max_records is not None:
        rows = rows[: int(args.max_records)]
    payloads = list(enumerate(rows))
    args_dict = {
        "beam_size": args.beam_size,
        "top_k": args.top_k,
        "candidate_multiplier": args.candidate_multiplier,
        "max_expanded_states": args.max_expanded_states,
        "timeout_per_sample": args.timeout_per_sample,
    }
    if args.workers <= 1:
        init_worker(str(args.lookup_json), str(args.data_root), priors, args_dict)
        results = [process_record(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(str(args.lookup_json), str(args.data_root), priors, args_dict),
        ) as pool:
            results = list(pool.imap_unordered(process_record, payloads, chunksize=8))
    results.sort(key=lambda r: int(r["input_index"]))
    candidate_rows = [r["candidate_row"] for r in results]
    summary_rows = [r["summary"] for r in results]
    write_jsonl(args.out_dir / f"{args.split}_streaming_candidates.jsonl", candidate_rows)
    write_jsonl(args.out_dir / f"{args.split}_streaming_per_sample.jsonl", summary_rows)
    split_summary = {
        **summarize(summary_rows),
        "config": {
            "data_root": str(args.data_root),
            "split": args.split,
            "beam_size": args.beam_size,
            "top_k": args.top_k,
            "candidate_multiplier": args.candidate_multiplier,
            "max_expanded_states": args.max_expanded_states,
            "timeout_per_sample": args.timeout_per_sample,
            "workers": args.workers,
        },
    }
    (args.out_dir / f"search_summary_{args.split}.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    merge_summary(args.out_dir, args.split, split_summary)
    write_csv(args.out_dir / f"search_breakdown_per_sg_{args.split}.csv", group_rows(summary_rows, "sg"))
    write_csv(args.out_dir / f"search_breakdown_per_nsites_{args.split}.csv", group_rows(summary_rows, "n_sites"))
    write_csv(args.out_dir / f"search_breakdown_per_num_elements_{args.split}.csv", group_rows(summary_rows, "num_elements"))
    if args.split == "test":
        write_csv(args.out_dir / "search_breakdown_per_sg.csv", group_rows(summary_rows, "sg"))
        write_csv(args.out_dir / "search_breakdown_per_nsites.csv", group_rows(summary_rows, "n_sites"))
        write_csv(args.out_dir / "search_breakdown_per_num_elements.csv", group_rows(summary_rows, "num_elements"))
    print(json.dumps(split_summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
