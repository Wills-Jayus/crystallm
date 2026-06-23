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

from train_count_aware_skeleton_model import CountAwareSkeletonNet, beam_skeleton_count_aware  # noqa: E402
from train_wyckoff_sequence_models import ensure_under_opentry, read_jsonl, vocab_from_jsonable, write_json, write_jsonl  # noqa: E402


def strip_record_for_infer(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": record.get("keys") or {},
        "sg": int(record["sg"]),
        "atom_count": int(record["atom_count"]),
        "formula_counts": dict(record.get("formula_counts") or {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference-only skeleton candidate generation without target labels.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--ckpt", type=Path, default=OPENTRY_ROOT / "runs" / "e35_count_aware_skeleton_fulltrain_val128" / "best.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--branch", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--count-scale", type=float, default=64.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()

    start = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = int(args.max_records) if int(args.max_records) > 0 else None
    records_raw = read_jsonl(args.data_dir / f"{args.split}.jsonl", max_records)
    records = [strip_record_for_infer(row) for row in records_raw]
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    vocab = vocab_from_jsonable(ckpt["vocab"])
    config = ckpt.get("config") or {}
    hidden = int(config.get("hidden", 256))
    count_scale = float(config.get("count_scale", args.count_scale))
    model = CountAwareSkeletonNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements) * 2,
        num_orbit_tokens=len(vocab.orbits) + 2,
        hidden=hidden,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    nonempty = 0
    unique_sum = 0
    final_path = out_dir / "skeleton_candidates.jsonl"
    partial_path = out_dir / "skeleton_candidates.partial.jsonl"
    progress_path = out_dir / "skeleton_progress.json"
    partial_path.unlink(missing_ok=True)
    progress_every = max(1, int(args.progress_every))
    flush_every = max(1, int(args.flush_every))
    with partial_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            candidates = beam_skeleton_count_aware(
                model,
                record,
                vocab,
                device,
                beam_size=int(args.beam_size),
                branch=int(args.branch),
                max_steps=int(args.max_steps),
                count_scale=float(count_scale),
            )
            keys = [str(x.get("skeleton_key") or "") for x in candidates]
            nonempty += int(bool(candidates))
            unique_sum += len(set(keys))
            keys_meta = record.get("keys") or {}
            row = {
                "index": idx,
                "sample_id": keys_meta.get("sample_id"),
                "material_id": keys_meta.get("material_id"),
                "sg": int(record["sg"]),
                "atom_count": int(record["atom_count"]),
                "candidate_count": len(candidates),
                "unique_skeleton": len(set(keys)),
                "skeleton_candidates": candidates,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            done = idx + 1
            if done % flush_every == 0:
                handle.flush()
            if done % progress_every == 0 or done == len(records):
                progress = {
                    "records_done": done,
                    "records_total": len(records),
                    "elapsed_s": time.time() - start,
                    "candidate_nonempty_rate_so_far": nonempty / max(1, done),
                    "unique_skeleton_mean_so_far": unique_sum / max(1, done),
                }
                write_json(progress_path, progress)
                print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
    partial_path.replace(final_path)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": len(records),
        "elapsed_s": time.time() - start,
        "candidate_nonempty_rate": nonempty / max(1, len(records)),
        "unique_skeleton_mean": unique_sum / max(1, len(records)),
        "label_policy": "Inference-only: no target skeleton/W-A keys, row_count labels, hit metrics, or StructureMatcher labels are emitted.",
    }
    write_json(out_dir / "skeleton_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
