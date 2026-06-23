#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import run_mp20_minicfjoint_v2 as v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", type=Path, default=v2.REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_train.jsonl")
    parser.add_argument("--lookup-json", type=Path, default=v2.PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = v2.read_jsonl(args.clean_train, limit=max(1, int(args.limit)))
    if not records:
        raise RuntimeError(f"no records loaded from {args.clean_train}")

    sg_symbols = v2.sg_symbols_from_splits({"train": records, "val": []})
    engine = v2.OrbitEngine(args.lookup_json, sg_symbols)
    vocab = v2.build_v2_vocab(records, engine)
    lattice_mean, lattice_std = v2.lattice_stats(records)
    batch = v2.collate_v2(
        records,
        vocab=vocab,
        engine=engine,
        lattice_mean=torch.tensor(lattice_mean, dtype=torch.float32),
        lattice_std=torch.tensor(lattice_std, dtype=torch.float32),
    )

    orbit_targets = batch["target_orbit_ids"]
    orbit_valid = orbit_targets >= 0
    orbit_is_legal = batch["orbit_legal"].gather(-1, orbit_targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    if not bool((orbit_is_legal | ~orbit_valid).all()):
        raise AssertionError("at least one target orbit is outside its legal mask")

    element_targets = batch["target_element_ids"]
    element_valid = element_targets >= 0
    element_is_legal = batch["element_legal"].gather(-1, element_targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    if not bool((element_is_legal | ~element_valid).all()):
        raise AssertionError("at least one target element is outside its legal mask")

    model = v2.MiniCFJointV2Net(vocab, emb_dim=32, hidden_dim=64)
    orbit_logits, element_logits, coords, lattice = model(batch)
    loss, parts = v2.loss_v2(orbit_logits, element_logits, coords, lattice, batch)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("loss is not finite")
    if not all(math.isfinite(float(value)) for value in parts.values()):
        raise AssertionError(f"non-finite loss part: {parts}")

    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(records),
                "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
                "orbit_vocab": len(vocab.id_to_orbit),
                "element_vocab": len(vocab.id_to_element),
                "loss": float(loss.detach().cpu()),
                "loss_parts": parts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
