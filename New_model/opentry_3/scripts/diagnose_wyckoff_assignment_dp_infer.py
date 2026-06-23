#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_wyckoff_assignment_dp import (  # noqa: E402
    RecordAssignmentScorer,
    assert_same_vocab,
    build_fixed_orbit_set,
    has_duplicate_fixed_orbits,
    load_assignment_scorer,
    read_json,
    read_jsonl,
    top_assignments_dp,
)
from train_wyckoff_sequence_models import ensure_under_opentry, vocab_from_jsonable, write_json, write_jsonl  # noqa: E402


def strip_record_for_infer(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": record.get("keys") or {},
        "sg": int(record["sg"]),
        "atom_count": int(record["atom_count"]),
        "formula_counts": dict(record.get("formula_counts") or {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference-only W/A assignment DP over predicted skeletons without target labels.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--split", default="test")
    parser.add_argument("--train-repr", type=Path, default=None)
    parser.add_argument("--vocab-json", type=Path, default=OPENTRY_ROOT / "runs" / "e05_wyckoff_seq_prior_8k_val32" / "vocab.json")
    parser.add_argument("--priors-json", type=Path, default=OPENTRY_ROOT / "runs" / "e05_wyckoff_seq_prior_8k_val32" / "decoder_priors_train_only.json")
    parser.add_argument("--predicted-candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--state-beam", type=int, default=100)
    parser.add_argument("--max-active-paths", type=int, default=0)
    parser.add_argument("--max-skeletons", type=int, default=5)
    parser.add_argument("--per-skeleton", type=int, default=50)
    parser.add_argument("--allow-duplicate-fixed-skeletons", action="store_true")
    parser.add_argument("--assignment-model-ckpt", type=Path, default=None)
    parser.add_argument("--assignment-device", default="cpu")
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--assignment-model-weight", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()

    start = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = int(args.max_records) if int(args.max_records) > 0 else None
    all_records_raw = read_jsonl(args.data_dir / f"{args.split}.jsonl")
    start_index = max(0, int(args.start_index))
    end_index = None if max_records is None else start_index + int(max_records)
    records_raw = all_records_raw[start_index:end_index]
    records = [strip_record_for_infer(row) for row in records_raw]
    vocab = vocab_from_jsonable(read_json(args.vocab_json))
    priors = read_json(args.priors_json)
    candidate_rows = read_jsonl(args.predicted_candidates)
    by_sample = {str(row.get("sample_id")): row for row in candidate_rows}
    assignment_device = torch.device(args.assignment_device)
    assignment_model = None
    assignment_ckpt_summary = None
    if args.assignment_model_ckpt is not None and float(args.assignment_model_weight) != 0.0:
        assignment_model, assignment_vocab, assignment_ckpt = load_assignment_scorer(args.assignment_model_ckpt, assignment_device)
        assert_same_vocab(vocab, assignment_vocab)
        assignment_ckpt_summary = {
            "path": str(args.assignment_model_ckpt),
            "epoch": assignment_ckpt.get("epoch"),
            "score": assignment_ckpt.get("score"),
            "val": assignment_ckpt.get("val"),
        }
    fixed_orbits = None
    if not args.allow_duplicate_fixed_skeletons:
        train_repr = args.train_repr if args.train_repr is not None else args.data_dir / "train.jsonl"
        fixed_orbits = build_fixed_orbit_set(read_jsonl(train_repr))

    per_sample_path = out_dir / "predicted_skeleton_assignment_per_sample.jsonl"
    candidates_path = out_dir / "predicted_skeleton_assignment_candidates.jsonl"
    per_sample_partial = out_dir / "predicted_skeleton_assignment_per_sample.partial.jsonl"
    candidates_partial = out_dir / "predicted_skeleton_assignment_candidates.partial.jsonl"
    progress_path = out_dir / "assignment_dp_progress.json"
    per_sample_partial.unlink(missing_ok=True)
    candidates_partial.unlink(missing_ok=True)
    progress_every = max(1, int(args.progress_every))
    flush_every = max(1, int(args.flush_every))
    nonempty_count = 0
    unique_sum = 0
    with per_sample_partial.open("w", encoding="utf-8") as per_handle, candidates_partial.open("w", encoding="utf-8") as cand_handle:
        for local_idx, record in enumerate(records):
            idx = start_index + local_idx
            keys_meta = record.get("keys") or {}
            sample_id = str(keys_meta.get("sample_id"))
            source = by_sample.get(sample_id, {})
            all_cands: dict[str, dict[str, Any]] = {}
            stats_rows: list[dict[str, Any]] = []
            used_skeletons = 0
            skipped_duplicate_fixed = 0
            elements = sorted(str(k) for k in record["formula_counts"])
            scorer = (
                RecordAssignmentScorer(assignment_model, vocab, record, elements, assignment_device)
                if assignment_model is not None and float(args.assignment_model_weight) != 0.0
                else None
            )
            for skel in list(source.get("skeleton_candidates") or []):
                skel = dict(skel)
                orbit_ids = [str(x) for x in skel.get("orbits") or []]
                if fixed_orbits is not None and has_duplicate_fixed_orbits(orbit_ids, fixed_orbits):
                    skipped_duplicate_fixed += 1
                    continue
                if used_skeletons >= int(args.max_skeletons):
                    break
                used_skeletons += 1
                skel["source"] = "predicted_skeleton_dp_prior"
                if scorer is not None:
                    skel["source"] = "predicted_skeleton_dp_neural_assignment"
                cands, stats = top_assignments_dp(
                    record,
                    skel,
                    vocab,
                    priors,
                    top_k=int(args.per_skeleton),
                    state_beam=int(args.state_beam),
                    max_active_paths=int(args.max_active_paths),
                    fixed_orbits=fixed_orbits,
                    assignment_scorer=scorer,
                    prior_weight=float(args.prior_weight),
                    assignment_model_weight=float(args.assignment_model_weight),
                )
                stats_rows.append({"skeleton_key": skel.get("skeleton_key"), **stats})
                for cand in cands:
                    all_cands.setdefault(str(cand["wa_key"]), cand)
            ranked = sorted(all_cands.values(), key=lambda item: item["score"] / max(1, len(item.get("pairs") or [])), reverse=True)[: int(args.top_k)]
            unique_count = len({str(c.get("wa_key")) for c in ranked})
            nonempty_count += int(bool(ranked))
            unique_sum += unique_count
            per_row = {
                "index": idx,
                "sample_id": sample_id,
                "material_id": keys_meta.get("material_id"),
                "sg": int(record["sg"]),
                "atom_count": int(record["atom_count"]),
                "pred_skeleton_count": len(source.get("skeleton_candidates") or []),
                "pred_skeleton_used": used_skeletons,
                "pred_skeleton_skipped_duplicate_fixed": skipped_duplicate_fixed,
                "pred_dp_candidate_count": len(ranked),
                "pred_dp_unique": unique_count,
                "pred_dp_eligible": bool(ranked),
                "pred_dp_reason": None if ranked else "no_ranked_assignment",
            }
            cand_row = {"sample_id": sample_id, "candidates": ranked[: min(int(args.top_k), 100)], "stats": stats_rows}
            per_handle.write(json.dumps(per_row, ensure_ascii=False, sort_keys=True) + "\n")
            cand_handle.write(json.dumps(cand_row, ensure_ascii=False, sort_keys=True) + "\n")
            done = idx + 1
            if done % flush_every == 0:
                per_handle.flush()
                cand_handle.flush()
            if done % progress_every == 0 or done == len(records):
                progress = {
                    "records_done": done,
                    "records_total": len(records),
                    "elapsed_s": time.time() - start,
                    "candidate_nonempty_rate_so_far": nonempty_count / max(1, done),
                    "unique_mean_so_far": unique_sum / max(1, done),
                }
                write_json(progress_path, progress)
                print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
    per_sample_partial.replace(per_sample_path)
    candidates_partial.replace(candidates_path)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": len(records),
        "start_index": start_index,
        "records_total_full": len(all_records_raw),
        "elapsed_s": time.time() - start,
        "candidate_nonempty_rate": nonempty_count / max(1, len(records)),
        "unique_mean": unique_sum / max(1, len(records)),
        "fixed_orbit_constraint": {
            "enabled": fixed_orbits is not None,
            "fixed_orbits": 0 if fixed_orbits is None else len(fixed_orbits),
        },
        "assignment_scoring": {
            "prior_weight": float(args.prior_weight),
            "assignment_model_weight": float(args.assignment_model_weight),
            "assignment_model": assignment_ckpt_summary,
        },
        "label_policy": "Inference-only: no target W/A, target skeleton, row_count label, hit metrics, or StructureMatcher labels are emitted.",
    }
    write_json(out_dir / "assignment_dp_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
