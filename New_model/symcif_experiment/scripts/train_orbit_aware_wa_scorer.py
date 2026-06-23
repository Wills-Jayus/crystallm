#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.wa_search import build_search_priors


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


class CandidateScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["multiplicity"]),
        str(row["letter"]),
        str(row.get("enumeration")),
        str(row.get("site_symmetry")),
        str(row["element"]),
        str(row["orbit_id"]),
    )


def candidate_aliases(candidate: dict[str, Any]) -> dict[str, Any]:
    c = dict(candidate)
    c["skeleton_key"] = c.get("canonical_skeleton_key")
    c["wa_key"] = c.get("canonical_wa_key")
    return c


def load_candidate_rows(candidate_dir: Path, split: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (candidate_dir / f"{split}_streaming_candidates.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= int(max_rows):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def train_frequency_maps(train_records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    priors = build_search_priors(train_records)
    element_counts: Counter[str] = Counter()
    site_sym_counts: Counter[str] = Counter()
    letter_counts: Counter[str] = Counter()
    free_dof_counts: Counter[str] = Counter()
    for record in train_records:
        for row in record["wa_table"]:
            element_counts[str(row["element"])] += 1
            site_sym_counts[str(row.get("site_symmetry"))] += 1
            letter_counts[f"{int(record['sg'])}|{row.get('letter')}"] += 1
            free_dof_counts[str(len(row.get("free_symbols") or []))] += 1
    priors["element_counts"] = element_counts
    priors["site_sym_counts"] = site_sym_counts
    priors["letter_counts"] = letter_counts
    priors["free_dof_counts"] = free_dof_counts
    return priors


def feature_vector(record: dict[str, Any], candidate: dict[str, Any], priors: dict[str, Counter[str]]) -> list[float]:
    sg = int(record["sg"])
    rows = candidate.get("rows") or []
    action_counts = priors.get("action_counts", Counter())
    orbit_counts = priors.get("orbit_counts", Counter())
    element_mult_counts = priors.get("element_mult_counts", Counter())
    skeleton_counts = priors.get("skeleton_counts", Counter())
    wa_counts = priors.get("wa_counts", Counter())
    element_counts = priors.get("element_counts", Counter())
    site_sym_counts = priors.get("site_sym_counts", Counter())
    letter_counts = priors.get("letter_counts", Counter())
    free_dof_counts = priors.get("free_dof_counts", Counter())

    row_scores = []
    orbit_scores = []
    elem_mult_scores = []
    element_scores = []
    sym_scores = []
    letter_scores = []
    free_dof_scores = []
    multiplicities = []
    free_rows = 0
    fixed_rows = 0
    total_free_dof = 0
    for row in rows:
        element = str(row["element"])
        orbit_id = str(row["orbit_id"])
        mult = int(row["multiplicity"])
        free_dof = len(row.get("free_symbols") or [])
        row_scores.append(math.log1p(action_counts.get(f"{sg}|{element}|{orbit_id}", 0)))
        orbit_scores.append(math.log1p(orbit_counts.get(f"{sg}|{orbit_id}", 0)))
        elem_mult_scores.append(math.log1p(element_mult_counts.get(f"{sg}|{element}|{mult}", 0)))
        element_scores.append(math.log1p(element_counts.get(element, 0)))
        sym_scores.append(math.log1p(site_sym_counts.get(str(row.get("site_symmetry")), 0)))
        letter_scores.append(math.log1p(letter_counts.get(f"{sg}|{row.get('letter')}", 0)))
        free_dof_scores.append(math.log1p(free_dof_counts.get(str(free_dof), 0)))
        multiplicities.append(float(mult))
        total_free_dof += free_dof
        if free_dof:
            free_rows += 1
        else:
            fixed_rows += 1

    def s(vals: list[float]) -> float:
        return float(sum(vals))

    def mean(vals: list[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    target_counts = [int(v) for v in record["formula_counts"].values()]
    return [
        float(candidate.get("score", 0.0)),
        float(candidate.get("search_score", candidate.get("score", 0.0))),
        math.log1p(skeleton_counts.get(str(candidate.get("canonical_skeleton_key")), 0)),
        math.log1p(wa_counts.get(str(candidate.get("canonical_wa_key")), 0)),
        s(row_scores),
        mean(row_scores),
        s(orbit_scores),
        mean(orbit_scores),
        s(elem_mult_scores),
        mean(elem_mult_scores),
        s(element_scores),
        s(sym_scores),
        s(letter_scores),
        s(free_dof_scores),
        float(len(rows)),
        float(fixed_rows),
        float(free_rows),
        float(total_free_dof),
        mean(multiplicities),
        max(multiplicities) if multiplicities else 0.0,
        float(record["sg"]),
        float(record["atom_count"]),
        float(record["num_elements"]),
        float(max(target_counts) if target_counts else 0),
        float(min(target_counts) if target_counts else 0),
    ]


def build_samples(
    data_root: Path,
    candidate_dir: Path,
    split: str,
    priors: dict[str, Counter[str]],
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = {row["sample_id"]: row for row in read_jsonl(data_root / f"{split}.jsonl")}
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for i, cand_row in enumerate(load_candidate_rows(candidate_dir, split, max_rows=max_rows)):
        if i and i % 250 == 0:
            print(f"[features] {split} {i} candidate sets processed", flush=True)
        record = records[cand_row["sample_id"]]
        candidates = [candidate_aliases(c) for c in cand_row.get("candidates", [])]
        target = str(record["canonical_wa_key"])
        labels = [i for i, c in enumerate(candidates) if str(c.get("canonical_wa_key")) == target]
        if not candidates or not labels:
            skipped.append({"sample_id": record["sample_id"], "split": split, "candidate_count": len(candidates), "target_in_candidates": bool(labels)})
            continue
        feats = [feature_vector(record, c, priors) for c in candidates]
        samples.append({"record": record, "candidates": candidates, "features": feats, "target_index": labels[0]})
    return samples, skipped


def rank_split(
    model: CandidateScorer,
    data_root: Path,
    candidate_dir: Path,
    split: str,
    priors: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    records = {row["sample_id"]: row for row in read_jsonl(data_root / f"{split}.jsonl")}
    out: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for cand_row in load_candidate_rows(candidate_dir, split):
            record = records[cand_row["sample_id"]]
            candidates = [candidate_aliases(c) for c in cand_row.get("candidates", [])]
            if candidates:
                feats = torch.tensor([feature_vector(record, c, priors) for c in candidates], dtype=torch.float32)
                scores = model(feats).cpu().tolist()
                for c, score in zip(candidates, scores):
                    c["model_score"] = float(score)
                candidates.sort(key=lambda c: (-float(c.get("model_score", 0.0)), -float(c.get("score", 0.0)), str(c.get("canonical_wa_key"))))
            out.append(
                {
                    "split": split,
                    "sample_id": record["sample_id"],
                    "formula": record["formula"],
                    "formula_counts": record["formula_counts"],
                    "sg": int(record["sg"]),
                    "n_sites": int(record["n_sites"]),
                    "num_elements": int(record["num_elements"]),
                    "gt_skeleton_key": record["canonical_skeleton_key"],
                    "gt_wa_key": record["canonical_wa_key"],
                    "ranked_wa_candidates": candidates,
                }
            )
    return out


def topk_metrics(predictions: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {"samples": len(predictions)}
    for k in (1, 5, 20, 100, 200):
        metrics[f"skeleton_top{k}"] = sum(
            any(str(c.get("canonical_skeleton_key")) == str(p["gt_skeleton_key"]) for c in p["ranked_wa_candidates"][:k])
            for p in predictions
        ) / max(1, len(predictions))
        metrics[f"wa_top{k}"] = sum(
            any(str(c.get("canonical_wa_key")) == str(p["gt_wa_key"]) for c in p["ranked_wa_candidates"][:k])
            for p in predictions
        ) / max(1, len(predictions))
    metrics["candidate_nonempty"] = sum(bool(p["ranked_wa_candidates"]) for p in predictions) / max(1, len(predictions))
    return metrics


def group_metrics(predictions: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        groups[pred[key]].append(pred)
    rows: list[dict[str, Any]] = []
    for value, items in sorted(groups.items(), key=lambda kv: kv[0]):
        m = topk_metrics(items)
        rows.append({key: value, **m})
    return rows


def train_model(
    model: CandidateScorer,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        count = 0
        correct = 0
        for sample in train_samples:
            x = torch.tensor(sample["features"], dtype=torch.float32)
            target = torch.tensor([int(sample["target_index"])], dtype=torch.long)
            logits = model(x).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            count += 1
            correct += int(int(torch.argmax(logits, dim=1).item()) == int(sample["target_index"]))
        val_loss = 0.0
        val_count = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for sample in val_samples:
                x = torch.tensor(sample["features"], dtype=torch.float32)
                target = torch.tensor([int(sample["target_index"])], dtype=torch.long)
                logits = model(x).unsqueeze(0)
                val_loss += float(F.cross_entropy(logits, target))
                val_count += 1
                val_correct += int(int(torch.argmax(logits, dim=1).item()) == int(sample["target_index"]))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, count),
            "train_candidate_set_top1": correct / max(1, count),
            "val_loss": val_loss / max(1, val_count),
            "val_candidate_set_top1": val_correct / max(1, val_count),
            "train_samples_with_positive": count,
            "val_samples_with_positive": val_count,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Train orbit-aware WA ranker on streaming candidates.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "orbit_aware_wa_scorer_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_streaming_wa")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.data_root / "train.jsonl")
    priors = train_frequency_maps(train_records)
    train_samples, train_skipped = build_samples(args.data_root, args.candidate_dir, "train", priors, max_rows=args.max_train_samples)
    val_samples, val_skipped = build_samples(args.data_root, args.candidate_dir, "val", priors, max_rows=args.max_val_samples)
    if not train_samples:
        raise RuntimeError("no train samples with GT WA in candidate set")
    in_dim = len(train_samples[0]["features"][0])
    model = CandidateScorer(in_dim=in_dim, hidden_dim=args.hidden_dim)
    history = train_model(model, train_samples, val_samples, epochs=args.epochs, lr=args.lr, seed=args.seed)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "feature_schema": [
                "candidate_score", "search_score", "skeleton_seen_log", "wa_seen_log", "sum_action_log", "mean_action_log",
                "sum_orbit_log", "mean_orbit_log", "sum_element_mult_log", "mean_element_mult_log", "sum_element_log",
                "sum_site_sym_log", "sum_letter_log", "sum_free_dof_log", "n_rows", "fixed_rows", "free_rows",
                "total_free_dof", "mean_multiplicity", "max_multiplicity", "sg", "atom_count", "num_elements",
                "max_element_count", "min_element_count",
            ],
            "history": history,
        },
        args.run_dir / "ckpt.pt",
    )
    predictions = rank_split(model, args.data_root, args.candidate_dir, "test", priors)
    write_jsonl(args.out_dir / "test_ranked_wa_predictions.jsonl", predictions)
    metrics = topk_metrics(predictions)
    complex_nsites = topk_metrics([p for p in predictions if int(p["n_sites"]) >= 6])
    complex_elements = topk_metrics([p for p in predictions if int(p["num_elements"]) >= 4])
    summary = {
        "train_samples": len(train_samples),
        "train_skipped_no_positive": len(train_skipped),
        "val_samples": len(val_samples),
        "val_skipped_no_positive": len(val_skipped),
        "test": metrics,
        "complex_nsites_ge6": complex_nsites,
        "complex_num_elements_ge4": complex_elements,
        "gate2_ranker": {
            "wa_top20_gt_76p8": float(metrics.get("wa_top20", 0.0)) > 0.768,
            "wa_top100_ge_90": float(metrics.get("wa_top100", 0.0)) >= 0.90,
            "skeleton_top20_gt_80p6": float(metrics.get("skeleton_top20", 0.0)) > 0.806,
        },
        "history": history,
    }
    (args.out_dir / "wa_scorer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "wa_scorer_breakdown_per_sg.csv", group_metrics(predictions, "sg"))
    write_csv(args.out_dir / "wa_scorer_breakdown_per_nsites.csv", group_metrics(predictions, "n_sites"))
    write_csv(args.out_dir / "wa_scorer_breakdown_per_num_elements.csv", group_metrics(predictions, "num_elements"))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
