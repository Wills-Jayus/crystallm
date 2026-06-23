#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # type: ignore  # noqa: E402
import torch  # type: ignore  # noqa: E402
import torch.nn as nn  # type: ignore  # noqa: E402
import torch.nn.functional as F  # type: ignore  # noqa: E402

from symcif_v4.scorer import BaselineScorer, rank_candidates_baseline
from symcif_v4.wa_table import gt_skeleton_key, gt_wa_key
from train_skeleton_template_ranker import ELEMENTS, ELEMENT_TO_IDX, read_jsonl


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_TO_IDX = {ch: i for i, ch in enumerate(LETTERS)}


def read_candidate_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def formula_vector(counts: dict[str, Any]) -> np.ndarray:
    vec = np.zeros(len(ELEMENTS) * 2 + 2, dtype=np.float32)
    total = max(1.0, float(sum(float(v) for v in counts.values())))
    for element, raw in counts.items():
        idx = ELEMENT_TO_IDX.get(str(element))
        if idx is None:
            continue
        value = float(raw)
        vec[idx] = value / total
        vec[len(ELEMENTS) + idx] = 1.0
    vec[-2] = np.log1p(total) / np.log1p(300.0)
    vec[-1] = len(counts) / 12.0
    return vec


def row_tensor(candidate: dict[str, Any]) -> torch.Tensor:
    rows = candidate["rows"]
    feats = np.zeros((max(1, len(rows)), 9), dtype=np.float32)
    for i, row in enumerate(rows):
        feats[i, 0] = ELEMENT_TO_IDX.get(str(row["element"]), 0)
        feats[i, 1] = LETTER_TO_IDX.get(str(row["letter"]), 0)
        feats[i, 2] = float(row["multiplicity"])
        enum = row.get("enumeration")
        feats[i, 3] = -1.0 if enum in {None, "None"} else float(enum)
        free = row.get("free_mask") or [False, False, False]
        feats[i, 4] = float(bool(free[0]))
        feats[i, 5] = float(bool(free[1]))
        feats[i, 6] = float(bool(free[2]))
        feats[i, 7] = i / max(1.0, len(rows) - 1.0)
        feats[i, 8] = len(rows) / 64.0
    return torch.from_numpy(feats)


class WATableDeepSetScorer(nn.Module):
    def __init__(self, formula_dim: int, sg_dim: int = 48, hidden: int = 256):
        super().__init__()
        self.sg_emb = nn.Embedding(231, sg_dim)
        self.element_emb = nn.Embedding(len(ELEMENTS), 48)
        self.letter_emb = nn.Embedding(len(LETTERS), 24)
        self.row_net = nn.Sequential(nn.Linear(48 + 24 + 7, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Sequential(
            nn.Linear(hidden + formula_dim + sg_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def score_candidates(self, formula_vec: torch.Tensor, sg: int, row_tensors: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        formula_vec = formula_vec.to(device)
        sg_emb = self.sg_emb(torch.tensor([int(sg)], dtype=torch.long, device=device))[0]
        scores: list[torch.Tensor] = []
        for rows in row_tensors:
            rows = rows.to(device)
            elem = self.element_emb(rows[:, 0].long().clamp(0, len(ELEMENTS) - 1))
            letter = self.letter_emb(rows[:, 1].long().clamp(0, len(LETTERS) - 1))
            numeric = torch.cat([rows[:, 2:3] / 192.0, rows[:, 3:4] / 16.0, rows[:, 4:]], dim=-1)
            row_repr = self.row_net(torch.cat([elem, letter, numeric], dim=-1)).mean(dim=0)
            scores.append(self.out(torch.cat([row_repr, formula_vec, sg_emb], dim=-1)).squeeze())
        return torch.stack(scores) if scores else torch.empty(0, device=device)


def sample_training_set(line: dict[str, Any], negatives: int, rng: random.Random) -> tuple[list[dict[str, Any]], int] | None:
    candidates = line["wa_candidates"]
    if not candidates:
        return None
    target = line["gt_wa_key"]
    pos_idx = next((i for i, c in enumerate(candidates) if c["wa_key"] == target), None)
    if pos_idx is None:
        return None
    neg_indices = [i for i in range(len(candidates)) if i != pos_idx]
    rng.shuffle(neg_indices)
    selected_indices = [pos_idx, *neg_indices[:negatives]]
    rng.shuffle(selected_indices)
    selected = [candidates[i] for i in selected_indices]
    label = selected_indices.index(pos_idx)
    return selected, label


def evaluate_ranked(
    structured_rows: list[dict[str, Any]],
    candidate_lines: list[dict[str, Any]],
    ranked_by_sample: dict[str, dict[str, list[dict[str, Any]]]],
    out_dir: Path,
) -> dict[str, Any]:
    row_by_id = {row["sample_id"]: row for row in structured_rows}
    ks = (1, 5, 20)
    summary: dict[str, Any] = {}
    prediction_lines: list[dict[str, Any]] = []
    for mode in ("baseline", "neural"):
        skeleton_hits = {k: 0 for k in ks}
        wa_hits = {k: 0 for k in ks}
        nonempty = 0
        for line in candidate_lines:
            sid = line["sample_id"]
            row = row_by_id[sid]
            target_skel = gt_skeleton_key(row)
            target_wa = gt_wa_key(row)
            ranked = ranked_by_sample[sid][mode]
            if ranked:
                nonempty += 1
            for k in ks:
                top = ranked[:k]
                if any(c["skeleton_key"] == target_skel for c in top):
                    skeleton_hits[k] += 1
                if any(c["wa_key"] == target_wa for c in top):
                    wa_hits[k] += 1
            if mode == "neural":
                prediction_lines.append(
                    {
                        "sample_id": sid,
                        "formula": row["formula"],
                        "sg": int(row["sg"]),
                        "gt_skeleton_key": target_skel,
                        "gt_wa_key": target_wa,
                        "ranked_wa_candidates": ranked[:20],
                    }
                )
        metrics: dict[str, Any] = {"samples": len(candidate_lines), "candidate_nonempty_rate": nonempty / max(1, len(candidate_lines))}
        for k in ks:
            metrics[f"skeleton_top{k}"] = skeleton_hits[k] / max(1, len(candidate_lines))
            metrics[f"wa_top{k}"] = wa_hits[k] / max(1, len(candidate_lines))
        summary[mode] = metrics
    with (out_dir / "test_wa_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in prediction_lines:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate WA-table scorer.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "wa_table_scorer_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negatives-per-sample", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    train_rows = read_jsonl(args.structured_root / "train.jsonl")
    test_rows = read_jsonl(args.structured_root / "test.jsonl")
    train_lines = read_candidate_jsonl(args.candidate_dir / "train_wa_candidates.jsonl")
    test_lines = read_candidate_jsonl(args.candidate_dir / "test_wa_candidates.jsonl")
    baseline = BaselineScorer.from_rows(train_rows)

    train_items = [sample_training_set(line, args.negatives_per_sample, rng) for line in train_lines]
    train_items = [item for item in train_items if item is not None]
    model = WATableDeepSetScorer(formula_dim=len(ELEMENTS) * 2 + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_items)
        total_loss = 0.0
        total = 0
        for selected, label in train_items:
            formula_counts = selected[0]["formula_counts"]
            sg = int(selected[0]["sg"])
            formula = torch.from_numpy(formula_vector(formula_counts)).float()
            rows = [row_tensor(c) for c in selected]
            optimizer.zero_grad(set_to_none=True)
            logits = model.score_candidates(formula, sg, rows)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            item = {"epoch": epoch, "loss": total_loss / max(1, total), "training_samples": total}
            history.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)

    row_by_id = {row["sample_id"]: row for row in test_rows}
    ranked_by_sample: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model.eval()
    with torch.no_grad():
        for i, line in enumerate(test_lines):
            if i and i % 50 == 0:
                print(f"[eval scorer] {i}/{len(test_lines)} samples", flush=True)
            sid = line["sample_id"]
            sample = row_by_id[sid]
            candidates = line["wa_candidates"]
            base_ranked = rank_candidates_baseline(baseline, sample, candidates)[:200]
            neural_ranked: list[dict[str, Any]] = []
            if candidates:
                scores = model.score_candidates(
                    torch.from_numpy(formula_vector(sample["formula_counts"])).float(),
                    int(sample["sg"]),
                    [row_tensor(c) for c in candidates],
                ).detach().cpu().numpy()
                order = np.argsort(-scores, kind="mergesort")
                for idx in order[:200]:
                    c = dict(candidates[int(idx)])
                    c["score"] = float(scores[int(idx)])
                    c["rank_source"] = "neural"
                    neural_ranked.append(c)
            ranked_by_sample[sid] = {"baseline": base_ranked, "neural": neural_ranked}

    summary = {
        "train": {"device": str(device), "epochs": args.epochs, "training_samples_with_gt_in_candidates": len(train_items)},
        "eval": evaluate_ranked(test_rows, test_lines, ranked_by_sample, args.out_dir),
    }
    (args.out_dir / "wa_scorer_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "wa_scorer_training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save({"model_state": model.state_dict(), "args": vars(args), "summary": summary}, args.run_dir / "ckpt.pt")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
