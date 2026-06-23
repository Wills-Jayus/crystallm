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

from train_wyckoff_sequence_models import (  # noqa: E402
    WyckoffModels,
    beam_skeleton,
    ensure_under_opentry,
    hit_at,
    read_jsonl,
    vocab_from_jsonable,
    write_json,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate validation skeleton candidates from an opentry_3 Wyckoff model.")
    parser.add_argument("--data-dir", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52")
    parser.add_argument("--ckpt", type=Path, default=OPENTRY_ROOT / "runs" / "e05_wyckoff_seq_prior_8k_val32" / "best.pt")
    parser.add_argument("--out-dir", type=Path, default=OPENTRY_ROOT / "reports" / "e08_skeleton_candidates_e05_val128")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    start = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.data_dir / f"{args.split}.jsonl", args.max_records)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    vocab = vocab_from_jsonable(ckpt["vocab"])
    hidden = int(ckpt.get("config", {}).get("hidden", 128))
    model = WyckoffModels(vocab, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        candidates = beam_skeleton(
            model,
            record,
            vocab,
            device,
            beam_size=args.beam_size,
            branch=args.branch,
            max_steps=args.max_steps,
        )
        keys = [str(x["skeleton_key"]) for x in candidates]
        row = {
            "index": idx,
            "sample_id": record["keys"].get("sample_id"),
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "row_count": int(record["row_count"]),
            "atom_count": int(record["atom_count"]),
            "target_skeleton_key": str(record["canonical_skeleton_key"]),
            "candidate_count": len(candidates),
            "unique_skeleton": len(set(keys)),
        }
        for k in (1, 5, 20, 50):
            row[f"skeleton_hit@{k}"] = hit_at(keys, str(record["canonical_skeleton_key"]), k)
        per_sample.append(row)
        rows.append({"sample_id": row["sample_id"], "skeleton_candidates": candidates})

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        denom = max(1, len(items))
        return {
            "samples": len(items),
            "candidate_nonempty_rate": sum(int(x["candidate_count"] > 0) for x in items) / denom,
            "unique_skeleton_mean": sum(int(x["unique_skeleton"]) for x in items) / denom,
            **{
                f"skeleton@{k}": sum(int(x[f"skeleton_hit@{k}"]) for x in items) / denom
                for k in (1, 5, 20, 50)
            },
        }

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "records": len(records),
        "elapsed_s": time.time() - start,
        "full": summarize(per_sample),
        "rows_ge_7": summarize([x for x in per_sample if int(x["row_count"]) >= 7]),
        "atoms_ge_12": summarize([x for x in per_sample if int(x["atom_count"]) >= 12]),
    }
    write_jsonl(out_dir / "skeleton_candidates.jsonl", rows)
    write_jsonl(out_dir / "skeleton_per_sample.jsonl", per_sample)
    write_json(out_dir / "skeleton_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
