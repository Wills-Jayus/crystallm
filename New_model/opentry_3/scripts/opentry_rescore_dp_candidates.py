#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SCRIPT_DIR = OPENTRY_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_wyckoff_sequence_models import (  # noqa: E402
    AssignmentNet,
    Vocab,
    ensure_under_opentry,
    formula_vec,
    prior_assignment_logprob,
    remaining_vec,
    seq_key_from_pairs,
    vocab_from_jsonable,
    write_json,
    write_jsonl,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_assignment_scorer(path: Path, device: torch.device) -> tuple[AssignmentNet, Vocab, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "vocab" not in ckpt:
        raise SystemExit(f"invalid assignment checkpoint: {path}")
    vocab = vocab_from_jsonable(ckpt["vocab"])
    hidden = int((ckpt.get("config") or {}).get("hidden", 256))
    model = AssignmentNet(
        num_sgs=len(vocab.sgs),
        formula_dim=len(vocab.elements),
        num_orbits=len(vocab.orbits),
        num_elements=len(vocab.elements),
        hidden=hidden,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab, ckpt


def assert_same_vocab(dp_vocab: Vocab, model_vocab: Vocab) -> None:
    problems: list[str] = []
    if dp_vocab.elements != model_vocab.elements:
        problems.append("elements")
    if dp_vocab.sgs != model_vocab.sgs:
        problems.append("sgs")
    if dp_vocab.orbits != model_vocab.orbits:
        problems.append("orbits")
    if problems:
        raise SystemExit(f"assignment model vocab does not match DP vocab: {', '.join(problems)}")


class SequenceScorer:
    def __init__(self, model: AssignmentNet, vocab: Vocab, record: dict[str, Any], device: torch.device) -> None:
        self.model = model
        self.vocab = vocab
        self.record = record
        self.device = device
        self.total = int(record["atom_count"])
        self.sg = int(record["sg"])
        self.sg_id = torch.tensor([vocab.sg_to_id.get(self.sg, 0)], dtype=torch.long, device=device)
        self.formula = formula_vec(record, vocab).unsqueeze(0).to(device)
        self.cache: dict[tuple[tuple[tuple[str, int], ...], str, int], torch.Tensor] = {}

    @torch.no_grad()
    def model_log_probs(self, remaining: dict[str, int], orbit_id: str, step: int) -> torch.Tensor | None:
        orbit_token = self.vocab.orbit_to_id.get(str(orbit_id))
        info = self.vocab.orbit_info.get(str(orbit_id))
        if orbit_token is None or info is None:
            return None
        key = (tuple(sorted((str(k), int(v)) for k, v in remaining.items())), str(orbit_id), int(step))
        if key not in self.cache:
            orbit = torch.tensor([int(orbit_token)], dtype=torch.long, device=self.device)
            rvec = remaining_vec(remaining, self.total, self.vocab).unsqueeze(0).to(self.device)
            numeric = torch.tensor(
                [
                    [
                        float(sum(int(v) for v in remaining.values())) / max(1.0, float(self.total)),
                        float(step) / 64.0,
                        float(info.multiplicity) / 64.0,
                    ]
                ],
                dtype=torch.float32,
                device=self.device,
            )
            logits = self.model(self.sg_id, self.formula, rvec, orbit, numeric).squeeze(0)
            self.cache[key] = F.log_softmax(logits, dim=-1).detach().cpu()
        return self.cache[key]


def ordered_pairs(pairs: list[Any], vocab: Vocab) -> list[tuple[str, str]]:
    cleaned: list[tuple[str, str, int]] = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        element = str(pair[0])
        orbit_id = str(pair[1])
        if orbit_id not in vocab.orbit_info:
            continue
        cleaned.append((element, orbit_id, idx))
    cleaned.sort(key=lambda item: (vocab.orbit_info[item[1]].sort_key, item[2]))
    return [(element, orbit_id) for element, orbit_id, _ in cleaned]


def score_candidate(
    record: dict[str, Any],
    cand: dict[str, Any],
    vocab: Vocab,
    priors: dict[str, Any],
    scorer: SequenceScorer,
    *,
    prior_weight: float,
    model_weight: float,
    source_weight: float,
) -> dict[str, Any] | None:
    pairs = ordered_pairs(list(cand.get("pairs") or []), vocab)
    if not pairs:
        return None
    remaining = {str(k): int(v) for k, v in record["formula_counts"].items()}
    sequence_score = 0.0
    prior_score = 0.0
    model_score = 0.0
    for step, (element, orbit_id) in enumerate(pairs):
        info = vocab.orbit_info.get(orbit_id)
        element_id = vocab.element_to_id.get(element)
        if info is None or element_id is None:
            return None
        mult = int(info.multiplicity)
        if int(remaining.get(element, 0)) < mult:
            return None
        p_score = prior_assignment_logprob(element, orbit_id, int(record["sg"]), priors)
        probs = scorer.model_log_probs(remaining, orbit_id, step)
        m_score = float(probs[element_id]) if probs is not None else 0.0
        prior_score += float(p_score)
        model_score += float(m_score)
        sequence_score += float(prior_weight) * float(p_score) + float(model_weight) * float(m_score)
        remaining[element] = int(remaining[element]) - mult
    if any(int(v) != 0 for v in remaining.values()):
        return None
    raw_score = float(cand.get("score") or 0.0)
    total_score = sequence_score + float(source_weight) * raw_score
    out = dict(cand)
    out["pairs"] = pairs
    out["wa_key"] = seq_key_from_pairs(pairs)
    out["score"] = float(total_score)
    out["rescore"] = {
        "prior_score": float(prior_score),
        "model_score": float(model_score),
        "source_score": raw_score,
        "sequence_score": float(sequence_score),
        "score_per_row": float(total_score) / max(1, len(pairs)),
    }
    out["source"] = "rescored_dp_assignment_model"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore exact DP W/A candidates with the train-only neural assignment scorer.")
    parser.add_argument("--repr-jsonl", type=Path, default=OPENTRY_ROOT / "data" / "wyckoff_repr_mpts52" / "val.jsonl")
    parser.add_argument("--vocab-json", type=Path, default=OPENTRY_ROOT / "runs" / "e63_assignment_scorer_fulltrain" / "vocab.json")
    parser.add_argument("--priors-json", type=Path, default=OPENTRY_ROOT / "runs" / "e63_assignment_scorer_fulltrain" / "decoder_priors_train_only.json")
    parser.add_argument("--assignment-model-ckpt", type=Path, default=OPENTRY_ROOT / "runs" / "e63_assignment_scorer_fulltrain" / "best.pt")
    parser.add_argument("--candidate-files", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--model-weight", type=float, default=1.0)
    parser.add_argument("--source-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    start = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    records = read_jsonl(args.repr_jsonl, max_records=max_records)
    vocab = vocab_from_jsonable(read_json(args.vocab_json))
    priors = read_json(args.priors_json)
    device = torch.device(args.device)
    model, model_vocab, ckpt = load_assignment_scorer(args.assignment_model_ckpt, device)
    assert_same_vocab(vocab, model_vocab)

    source_by_sample: list[dict[str, dict[str, Any]]] = []
    for path in args.candidate_files:
        rows = read_jsonl(path, max_records=max_records)
        source_by_sample.append({str(row.get("sample_id")): row for row in rows})

    out_rows: list[dict[str, Any]] = []
    summary = {
        "config": {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "records": len(records),
        "assignment_model": {
            "path": str(args.assignment_model_ckpt),
            "epoch": ckpt.get("epoch"),
            "score": ckpt.get("score"),
            "val": ckpt.get("val"),
        },
        "samples_with_candidates": 0,
        "input_candidates": 0,
        "valid_scored_candidates": 0,
        "elapsed_s": None,
    }
    for record in records:
        sample_id = str(record["keys"]["sample_id"])
        scorer = SequenceScorer(model, vocab, record, device)
        scored: dict[str, dict[str, Any]] = {}
        for source_idx, by_sample in enumerate(source_by_sample):
            row = by_sample.get(sample_id) or {}
            for cand in row.get("candidates") or []:
                summary["input_candidates"] += 1
                rescored = score_candidate(
                    record,
                    cand,
                    vocab,
                    priors,
                    scorer,
                    prior_weight=float(args.prior_weight),
                    model_weight=float(args.model_weight),
                    source_weight=float(args.source_weight),
                )
                if rescored is None:
                    continue
                summary["valid_scored_candidates"] += 1
                key = str(rescored.get("wa_key") or "")
                old = scored.get(key)
                if old is None or float(rescored["score"]) > float(old["score"]):
                    rescored["rescore"]["source_file_index"] = source_idx
                    scored[key] = rescored
        ranked = sorted(scored.values(), key=lambda item: float(item["score"]) / max(1, len(item.get("pairs") or [])), reverse=True)
        if ranked:
            summary["samples_with_candidates"] += 1
        out_rows.append({"sample_id": sample_id, "candidates": ranked[: int(args.top_k)], "stats": {"input_sources": len(args.candidate_files), "unique_scored": len(scored)}})
    summary["candidate_nonempty_rate"] = float(summary["samples_with_candidates"]) / max(1, len(records))
    summary["elapsed_s"] = time.time() - start
    write_jsonl(out_dir / "rescored_dp_candidates.jsonl", out_rows)
    write_json(out_dir / "rescore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
