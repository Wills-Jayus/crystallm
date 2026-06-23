#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment")
OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2")
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_composition_exact_candidates import (  # noqa: E402
    compact_skeleton,
    compact_wa,
    merge_summary,
    q,
    summarize_group,
    write_csv,
    write_jsonl,
)
from symcif_v4.exact_cover import enumerate_skeleton_candidates, enumerate_wa_tables_for_skeleton  # noqa: E402
from symcif_v4.formula import normalize_formula_counts, total_atoms  # noqa: E402
from symcif_v4.wa_table import gt_skeleton_key, gt_wa_key  # noqa: E402
from symcif_v4.wyckoff_table import load_lookup, observed_repeat_limits, wyckoff_tokens_for_sg  # noqa: E402
from train_skeleton_template_ranker import read_jsonl  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    root = OPENTRY_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside opentry_2: {resolved}")
    return resolved


def parse_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"train", "val", "test"}
    bad = [split for split in splits if split not in allowed]
    if bad:
        raise ValueError(f"Invalid split(s): {bad}")
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def load_rows_for_splits(root: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        path = root / f"{split}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows


def rows_for_repeat_limits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if "assignment" in row:
            normalized.append(row)
            continue
        assignment = [{"letter": site.get("letter")} for site in row.get("wa_table", []) if site.get("letter")]
        normalized.append({"sg": row["sg"], "assignment": assignment})
    return normalized


def slice_rows(rows: list[dict[str, Any]], start: int, max_records: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start-index must be non-negative")
    end = None if max_records is None or max_records <= 0 else start + max_records
    return rows[start:end]


def hit_at(candidates: list[dict[str, Any]], target: str, field: str, k: int) -> bool:
    return any(str(candidate.get(field)) == target for candidate in candidates[:k])


def target_skeleton_key(row: dict[str, Any]) -> str:
    if row.get("canonical_skeleton_key"):
        return str(row["canonical_skeleton_key"])
    return gt_skeleton_key(row)


def target_wa_key(row: dict[str, Any]) -> str:
    if row.get("canonical_wa_key"):
        return str(row["canonical_wa_key"])
    return gt_wa_key(row)


def orbit_id_from_token_key(key: str) -> str:
    if key.startswith("setting="):
        return key
    return f"setting=crystalformer|{key}"


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("multiplicity", 0) or 0),
        str(row.get("letter", "")),
        str(row.get("enumeration", "")),
        str(row.get("site_symmetry", "")),
        str(row.get("element", "")),
        orbit_id_from_token_key(str(row.get("canonical_key", ""))),
    )


def skeleton_canonical_key_from_skel(skel: Any) -> str:
    rows = [
        {
            "multiplicity": token.multiplicity,
            "letter": token.letter,
            "enumeration": token.enumeration,
            "site_symmetry": token.site_symmetry,
            "canonical_key": token.canonical_key,
        }
        for token in skel.sites
    ]
    return "|".join(orbit_id_from_token_key(str(row["canonical_key"])) for row in sorted(rows, key=row_sort_key))


def add_candidate_canonical_keys(candidate: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(candidate.get("rows", []), key=row_sort_key)
    skeleton_key = "|".join(orbit_id_from_token_key(str(row["canonical_key"])) for row in rows)
    wa_key = "|".join(
        f"{orbit_id_from_token_key(str(row['canonical_key']))}:{row['element']}"
        for row in rows
    )
    candidate["canonical_skeleton_key"] = skeleton_key
    candidate["canonical_wa_key"] = wa_key
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Opentry-only composition-exact subset candidate audit.")
    parser.add_argument(
        "--structured-root",
        type=Path,
        default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52",
    )
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--repeat-limit-splits",
        default="train",
        help="Comma-separated splits used only to estimate observed Wyckoff repeat limits. Use train for clean val/test audits.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0, help="0 means all rows after start-index.")
    parser.add_argument("--max-skeleton-candidates", type=int, default=50000)
    parser.add_argument("--max-wa-candidates", type=int, default=200000)
    parser.add_argument("--timeout-per-sample", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repeat_limit_splits = parse_splits(args.repeat_limit_splits)
    lookup = load_lookup(args.lookup_json)
    repeat_rows = rows_for_repeat_limits(load_rows_for_splits(args.structured_root, repeat_limit_splits))
    fixed_limits = observed_repeat_limits(repeat_rows, lookup)
    all_split_rows = read_jsonl(args.structured_root / f"{args.split}.jsonl")
    rows = slice_rows(all_split_rows, args.start_index, args.max_records or None)

    skel_lines: list[dict[str, Any]] = []
    wa_lines: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    debug_failures: list[dict[str, Any]] = []
    token_cache: dict[int, list[Any]] = {}

    for local_i, row in enumerate(rows):
        if local_i and local_i % args.progress_every == 0:
            print(f"[{args.split}] {local_i}/{len(rows)} samples done", flush=True)
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
        gt_skel = target_skeleton_key(row)
        gt_wa = target_wa_key(row)
        skel_candidates = []
        for skel in skels:
            item = compact_skeleton(skel)
            item["canonical_skeleton_key"] = skeleton_canonical_key_from_skel(skel)
            skel_candidates.append(item)
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
            was, wstats = enumerate_wa_tables_for_skeleton(
                counts,
                skel,
                max_assignments=args.max_wa_candidates - wa_count,
                timeout_s=max(0.01, args.timeout_per_sample - wa_elapsed),
            )
            wa_truncated = wa_truncated or bool(wstats.truncated)
            wa_timeout = wa_timeout or bool(wstats.timeout)
            for wa in was:
                item = add_candidate_canonical_keys(compact_wa(wa))
                wa_candidates.append(item)
                if item["canonical_wa_key"] == gt_wa:
                    gt_wa_found = True
            wa_count += len(was)
            if wa_timeout or (wa_truncated and wa_count >= args.max_wa_candidates):
                break

        gt_skel_found = any(candidate["canonical_skeleton_key"] == gt_skel for candidate in skel_candidates)
        sample_summary = {
            "split": args.split,
            "global_index": args.start_index + local_i,
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
            "gt_skeleton_top1": hit_at(skel_candidates, gt_skel, "canonical_skeleton_key", 1),
            "gt_skeleton_top5": hit_at(skel_candidates, gt_skel, "canonical_skeleton_key", 5),
            "gt_skeleton_top20": hit_at(skel_candidates, gt_skel, "canonical_skeleton_key", 20),
            "gt_wa_top1": hit_at(wa_candidates, gt_wa, "canonical_wa_key", 1),
            "gt_wa_top5": hit_at(wa_candidates, gt_wa, "canonical_wa_key", 5),
            "gt_wa_top20": hit_at(wa_candidates, gt_wa, "canonical_wa_key", 20),
            "gt_wa_top100": hit_at(wa_candidates, gt_wa, "canonical_wa_key", 100),
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
                **{
                    k: sample_summary[k]
                    for k in (
                        "split",
                        "global_index",
                        "sample_id",
                        "formula",
                        "sg",
                        "n_sites",
                        "total_atoms",
                        "num_elements",
                        "gt_skeleton_key",
                        "gt_skeleton_in_candidates",
                    )
                },
                "skeleton_candidates": skel_candidates,
            }
        )
        wa_lines.append(
            {
                **{
                    k: sample_summary[k]
                    for k in (
                        "split",
                        "global_index",
                        "sample_id",
                        "formula",
                        "sg",
                        "n_sites",
                        "total_atoms",
                        "num_elements",
                        "gt_wa_key",
                        "gt_wa_in_candidates",
                    )
                },
                "wa_candidates": wa_candidates,
            }
        )

    write_jsonl(out_dir / f"{args.split}_skeleton_candidates.jsonl", skel_lines)
    write_jsonl(out_dir / f"{args.split}_wa_candidates.jsonl", wa_lines)
    write_jsonl(out_dir / f"{args.split}_candidate_per_sample.jsonl", per_sample)
    write_jsonl(out_dir / f"{args.split}_zero_candidate_debug.jsonl", debug_failures)

    summary = {
        "split": args.split,
        "source_records": len(all_split_rows),
        "start_index": args.start_index,
        "samples": len(rows),
        "repeat_limit_splits": list(repeat_limit_splits),
        "full_formula_count_ok": 1.0,
        "no_reduced_formula_used": True,
        "gt_skeleton_coverage": sum(bool(x["gt_skeleton_in_candidates"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_coverage": sum(bool(x["gt_wa_in_candidates"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_skeleton_top1": sum(bool(x["gt_skeleton_top1"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_skeleton_top5": sum(bool(x["gt_skeleton_top5"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_skeleton_top20": sum(bool(x["gt_skeleton_top20"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_top1": sum(bool(x["gt_wa_top1"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_top5": sum(bool(x["gt_wa_top5"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_top20": sum(bool(x["gt_wa_top20"]) for x in per_sample) / max(1, len(per_sample)),
        "gt_wa_top100": sum(bool(x["gt_wa_top100"]) for x in per_sample) / max(1, len(per_sample)),
        "candidate_count": q([x["skeleton_candidate_count"] for x in per_sample]),
        "wa_candidate_count": q([x["wa_candidate_count"] for x in per_sample]),
        "zero_candidate_samples": sum(int(x["skeleton_candidate_count"] == 0) for x in per_sample),
        "zero_wa_candidate_samples": sum(int(x["wa_candidate_count"] == 0) for x in per_sample),
        "truncated_samples": sum(int(bool(x["skeleton_truncated"])) for x in per_sample),
        "timeout_samples": sum(int(bool(x["skeleton_timeout"])) for x in per_sample),
        "truncated_wa_samples": sum(int(bool(x["wa_truncated"])) for x in per_sample),
        "wa_timeout_samples": sum(int(bool(x.get("wa_timeout"))) for x in per_sample),
    }
    merge_summary(out_dir, f"{args.split}_subset", summary)
    write_csv(out_dir / f"candidate_generation_per_sg_{args.split}.csv", summarize_group(per_sample, "sg"))
    write_csv(out_dir / f"candidate_generation_per_nsites_{args.split}.csv", summarize_group(per_sample, "n_sites"))
    write_csv(out_dir / f"candidate_generation_per_total_atoms_{args.split}.csv", summarize_group(per_sample, "total_atoms"))
    write_csv(out_dir / f"candidate_generation_per_num_elements_{args.split}.csv", summarize_group(per_sample, "num_elements"))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
