#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.exact_cover import enumerate_skeleton_candidates, enumerate_wa_tables_for_skeleton
from symcif_v4.formula import normalize_formula_counts, total_atoms
from symcif_v4.wa_table import gt_skeleton_key, gt_wa_key
from symcif_v4.wyckoff_table import load_lookup, observed_repeat_limits, wyckoff_tokens_for_sg
from train_skeleton_template_ranker import read_jsonl


def compact_skeleton(skel: Any) -> dict[str, Any]:
    return {
        "sg": skel.sg,
        "sites": [site.short_key for site in skel.sites],
        "multiplicities": skel.multiplicities,
        "skeleton_key": skel.skeleton_key,
        "total_atoms": skel.total_atoms,
        "truncated": skel.truncated,
    }


def compact_wa(wa: Any) -> dict[str, Any]:
    return {
        "sg": wa.sg,
        "formula_counts": wa.formula_counts,
        "rows": [
            {
                "element": element,
                "site": token.short_key,
                "letter": token.letter,
                "multiplicity": token.multiplicity,
                "site_symmetry": token.site_symmetry,
                "enumeration": token.enumeration,
                "canonical_key": token.canonical_key,
                "free_mask": list(token.free_mask),
            }
            for element, token in wa.rows
        ],
        "skeleton_key": wa.skeleton_key,
        "assignment_key": wa.assignment_key,
        "wa_key": wa.wa_key,
        "source": wa.source,
        "score": wa.score,
    }


def q(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    vals = sorted(float(v) for v in values)
    return {
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]),
        "max": float(vals[-1]),
    }


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


def summarize_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        out.append(
            {
                key: value,
                "samples": len(items),
                "gt_skeleton_coverage": sum(bool(x["gt_skeleton_in_candidates"]) for x in items) / len(items),
                "gt_wa_coverage": sum(bool(x["gt_wa_in_candidates"]) for x in items) / len(items),
                "candidate_count_mean": q([x["skeleton_candidate_count"] for x in items])["mean"],
                "wa_candidate_count_mean": q([x["wa_candidate_count"] for x in items])["mean"],
                "zero_candidate_samples": sum(int(x["skeleton_candidate_count"] == 0) for x in items),
                "zero_wa_candidate_samples": sum(int(x["wa_candidate_count"] == 0) for x in items),
                "truncated_samples": sum(int(bool(x["skeleton_truncated"])) for x in items),
                "truncated_wa_samples": sum(int(bool(x["wa_truncated"])) for x in items),
                "timeout_samples": sum(int(bool(x["skeleton_timeout"])) for x in items),
                "wa_timeout_samples": sum(int(bool(x.get("wa_timeout"))) for x in items),
            }
        )
    return out


def load_all_rows(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        p = root / f"{split}.jsonl"
        if p.exists():
            out.extend(read_jsonl(p))
    return out


def merge_summary(out_dir: Path, split: str, split_summary: dict[str, Any]) -> None:
    path = out_dir / "candidate_generation_summary.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = {"splits": {}}
    raw.setdefault("splits", {})[split] = split_summary
    (out_dir / f"candidate_generation_summary_{split}.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build composition-exact skeleton and WA candidates.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    parser.add_argument("--max-skeleton-candidates", type=int, default=20000)
    parser.add_argument("--max-wa-candidates", type=int, default=50000)
    parser.add_argument("--timeout-per-sample", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_lookup(args.lookup_json)
    all_rows = load_all_rows(args.structured_root)
    fixed_limits = observed_repeat_limits(all_rows, lookup)
    rows = read_jsonl(args.structured_root / f"{args.split}.jsonl")

    skel_lines: list[dict[str, Any]] = []
    wa_lines: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    debug_failures: list[dict[str, Any]] = []
    token_cache: dict[int, list[Any]] = {}
    for i, row in enumerate(rows):
        if i and i % args.progress_every == 0:
            print(f"[{args.split}] {i}/{len(rows)} samples done", flush=True)
        counts = normalize_formula_counts(row["formula_counts"])
        sg = int(row["sg"])
        token_cache.setdefault(sg, wyckoff_tokens_for_sg(lookup, sg, fixed_repeat_limits=fixed_limits))
        skels, sstats = enumerate_skeleton_candidates(
            counts,
            sg,
            token_cache[sg],
            max_candidates=args.max_skeleton_candidates,
            timeout_s=args.timeout_per_sample,
        )
        gt_skel = gt_skeleton_key(row)
        gt_wa = gt_wa_key(row)
        wa_candidates: list[dict[str, Any]] = []
        wa_count = 0
        wa_truncated = False
        wa_timeout = False
        gt_wa_found = False
        wa_start = time.monotonic()
        for skel in skels:
            if wa_count >= args.max_wa_candidates:
                wa_truncated = True
                break
            wa_elapsed = time.monotonic() - wa_start
            if wa_elapsed > args.timeout_per_sample:
                wa_timeout = True
                break
            remaining_cap = args.max_wa_candidates - wa_count
            was, wstats = enumerate_wa_tables_for_skeleton(
                counts,
                skel,
                max_assignments=remaining_cap,
                timeout_s=max(0.01, args.timeout_per_sample - wa_elapsed),
            )
            if wstats.truncated:
                wa_truncated = True
            if wstats.timeout:
                wa_timeout = True
            for wa in was:
                item = compact_wa(wa)
                wa_candidates.append(item)
                if item["wa_key"] == gt_wa:
                    gt_wa_found = True
            wa_count += len(was)
            if wa_timeout or (wa_truncated and wa_count >= args.max_wa_candidates):
                break
        gt_skel_found = any(s.skeleton_key == gt_skel for s in skels)
        sample_summary = {
            "split": args.split,
            "sample_id": row["sample_id"],
            "formula": row["formula"],
            "sg": sg,
            "n_sites": int(row["n_sites"]),
            "total_atoms": int(total_atoms(counts)),
            "num_elements": int(len(counts)),
            "full_formula_count_ok": True,
            "no_reduced_formula_used": True,
            "gt_skeleton_key": gt_skel,
            "gt_wa_key": gt_wa,
            "gt_skeleton_in_candidates": bool(gt_skel_found),
            "gt_wa_in_candidates": bool(gt_wa_found),
            "skeleton_candidate_count": len(skels),
            "wa_candidate_count": len(wa_candidates),
            "skeleton_truncated": bool(sstats.truncated),
            "skeleton_timeout": bool(sstats.timeout),
            "wa_truncated": bool(wa_truncated),
            "wa_timeout": bool(wa_timeout),
            "visited_states": int(sstats.visited_states),
            "elapsed_s": float(sstats.elapsed_s),
            "zero_candidate_reason": None,
        }
        if not skels:
            sample_summary["zero_candidate_reason"] = sstats.reason or "no_exact_cover_skeleton"
            debug_failures.append(sample_summary)
        elif not wa_candidates:
            sample_summary["zero_candidate_reason"] = "no_exact_cover_assignment"
            debug_failures.append(sample_summary)
        per_sample.append(sample_summary)
        skel_lines.append(
            {
                **{k: sample_summary[k] for k in ("split", "sample_id", "formula", "sg", "n_sites", "total_atoms", "num_elements", "gt_skeleton_key", "gt_skeleton_in_candidates")},
                "skeleton_candidates": [compact_skeleton(s) for s in skels],
            }
        )
        wa_lines.append(
            {
                **{k: sample_summary[k] for k in ("split", "sample_id", "formula", "sg", "n_sites", "total_atoms", "num_elements", "gt_wa_key", "gt_wa_in_candidates")},
                "wa_candidates": wa_candidates,
            }
        )

    write_jsonl(args.out_dir / f"{args.split}_skeleton_candidates.jsonl", skel_lines)
    write_jsonl(args.out_dir / f"{args.split}_wa_candidates.jsonl", wa_lines)
    write_jsonl(args.out_dir / f"{args.split}_candidate_per_sample.jsonl", per_sample)
    write_jsonl(args.out_dir / f"{args.split}_zero_candidate_debug.jsonl", debug_failures)

    summary = {
        "split": args.split,
        "samples": len(rows),
        "full_formula_count_ok": 1.0,
        "no_reduced_formula_used": True,
        "gt_skeleton_coverage": sum(bool(x["gt_skeleton_in_candidates"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_coverage": sum(bool(x["gt_wa_in_candidates"]) for x in per_sample) / max(1, len(per_sample)),
        "candidate_count": q([x["skeleton_candidate_count"] for x in per_sample]),
        "wa_candidate_count": q([x["wa_candidate_count"] for x in per_sample]),
        "zero_candidate_samples": sum(int(x["skeleton_candidate_count"] == 0) for x in per_sample),
        "zero_wa_candidate_samples": sum(int(x["wa_candidate_count"] == 0) for x in per_sample),
        "truncated_samples": sum(int(bool(x["skeleton_truncated"])) for x in per_sample),
        "timeout_samples": sum(int(bool(x["skeleton_timeout"])) for x in per_sample),
        "truncated_wa_samples": sum(int(bool(x["wa_truncated"])) for x in per_sample),
        "wa_timeout_samples": sum(int(bool(x.get("wa_timeout"))) for x in per_sample),
    }
    merge_summary(args.out_dir, args.split, summary)
    per_sg = summarize_group(per_sample, "sg")
    per_nsites = summarize_group(per_sample, "n_sites")
    per_total_atoms = summarize_group(per_sample, "total_atoms")
    per_num_elements = summarize_group(per_sample, "num_elements")
    write_csv(args.out_dir / f"candidate_generation_per_sg_{args.split}.csv", per_sg)
    write_csv(args.out_dir / f"candidate_generation_per_nsites_{args.split}.csv", per_nsites)
    write_csv(args.out_dir / f"candidate_generation_per_total_atoms_{args.split}.csv", per_total_atoms)
    write_csv(args.out_dir / f"candidate_generation_per_num_elements_{args.split}.csv", per_num_elements)
    if args.split == "test":
        write_csv(args.out_dir / "candidate_generation_per_sg.csv", per_sg)
        write_csv(args.out_dir / "candidate_generation_per_nsites.csv", per_nsites)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
