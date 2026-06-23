#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TOP_KS = (1, 5, 20, 100, 200)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def split_candidates_path(candidate_dir: Path, split: str) -> Path:
    path = candidate_dir / f"{split}_hybrid_candidates.jsonl"
    if path.exists():
        return path
    if split == "test":
        fallback = candidate_dir / "test_hybrid_candidates.jsonl"
        if fallback.exists():
            return fallback
    return path


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


class RerankVocab:
    def __init__(self) -> None:
        self.element: dict[str, int] = {}
        self.orbit: dict[str, int] = {}
        self.sg: dict[str, int] = {}
        self.letter: dict[str, int] = {}
        self.site_sym: dict[str, int] = {}
        self.source: dict[str, int] = {"policy": 1, "old": 2}

    @staticmethod
    def _add(mapping: dict[str, int], value: Any) -> None:
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1

    @classmethod
    def from_rows(cls, records: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> "RerankVocab":
        vocab = cls()
        for record in records:
            cls._add(vocab.sg, int(record["sg"]))
            for element in record["formula_counts"]:
                cls._add(vocab.element, element)
            for row in record.get("wa_table", []):
                cls._add(vocab.element, row.get("element"))
                cls._add(vocab.orbit, row.get("orbit_id"))
                cls._add(vocab.letter, row.get("letter"))
                cls._add(vocab.site_sym, row.get("site_symmetry"))
        for cand_row in candidate_rows:
            cls._add(vocab.sg, int(cand_row["sg"]))
            for candidate in cand_row.get("candidates", []):
                for source in candidate.get("source_labels", []):
                    cls._add(vocab.source, source)
                for row in candidate.get("rows", []):
                    cls._add(vocab.element, row.get("element"))
                    cls._add(vocab.orbit, row.get("orbit_id"))
                    cls._add(vocab.letter, row.get("letter"))
                    cls._add(vocab.site_sym, row.get("site_symmetry"))
        return vocab

    def sizes(self) -> dict[str, int]:
        return {
            "element": max(self.element.values(), default=0),
            "orbit": max(self.orbit.values(), default=0),
            "sg": max(self.sg.values(), default=0),
            "letter": max(self.letter.values(), default=0),
            "site_sym": max(self.site_sym.values(), default=0),
            "source": max(self.source.values(), default=0),
        }

    def to_jsonable(self) -> dict[str, dict[str, int]]:
        return {
            "element": self.element,
            "orbit": self.orbit,
            "sg": self.sg,
            "letter": self.letter,
            "site_sym": self.site_sym,
            "source": self.source,
        }

    @classmethod
    def from_jsonable(cls, raw: dict[str, Any]) -> "RerankVocab":
        vocab = cls()
        vocab.element = {str(k): int(v) for k, v in raw["element"].items()}
        vocab.orbit = {str(k): int(v) for k, v in raw["orbit"].items()}
        vocab.sg = {str(k): int(v) for k, v in raw["sg"].items()}
        vocab.letter = {str(k): int(v) for k, v in raw["letter"].items()}
        vocab.site_sym = {str(k): int(v) for k, v in raw["site_sym"].items()}
        vocab.source = {str(k): int(v) for k, v in raw["source"].items()}
        return vocab


class OrbitAwareListwiseRanker(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        *,
        emb_dim: int = 64,
        hidden_dim: int = 160,
        row_layers: int = 2,
        cand_layers: int = 2,
        heads: int = 4,
        row_numeric_dim: int = 8,
        cand_numeric_dim: int = 12,
    ) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 4)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 4)
        self.source_emb = nn.Embedding(vocab_sizes["source"] + 1, emb_dim // 4)
        row_in = emb_dim * 2 + emb_dim // 2 + emb_dim // 4 + emb_dim // 4 + row_numeric_dim
        self.row_in = nn.Sequential(nn.LayerNorm(row_in), nn.Linear(row_in, hidden_dim), nn.GELU())
        row_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=row_layers)
        cand_in = hidden_dim + emb_dim // 4 + cand_numeric_dim
        self.cand_in = nn.Sequential(nn.LayerNorm(cand_in), nn.Linear(cand_in, hidden_dim), nn.GELU())
        cand_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cand_encoder = nn.TransformerEncoder(cand_layer, num_layers=cand_layers)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cands, rows = batch["element_id"].shape
        row_x = torch.cat(
            [
                self.element_emb(batch["element_id"]),
                self.orbit_emb(batch["orbit_id"]),
                self.sg_emb(batch["sg_id"]),
                self.letter_emb(batch["letter_id"]),
                self.site_sym_emb(batch["site_sym_id"]),
                batch["row_numeric"],
            ],
            dim=-1,
        )
        row_h = self.row_in(row_x.reshape(cands * rows, -1)).reshape(cands, rows, -1)
        row_mask = ~batch["row_mask"].bool()
        encoded_rows = self.row_encoder(row_h, src_key_padding_mask=row_mask)
        denom = batch["row_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_rows = (encoded_rows * batch["row_mask"].unsqueeze(-1)).sum(dim=1) / denom
        source_emb = self.source_emb(batch["source_id"])
        source_denom = batch["source_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
        source_vec = (source_emb * batch["source_mask"].unsqueeze(-1)).sum(dim=1) / source_denom
        cand_x = torch.cat([pooled_rows, source_vec, batch["cand_numeric"]], dim=-1)
        cand_h = self.cand_in(cand_x).unsqueeze(0)
        cand_encoded = self.cand_encoder(cand_h).squeeze(0)
        return self.scorer(cand_encoded).squeeze(-1)


def multiplicity_pattern(candidate: dict[str, Any]) -> tuple[int, ...]:
    return tuple(sorted(safe_int(row.get("multiplicity")) for row in candidate.get("rows", [])))


def site_sym_pattern(candidate: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(row.get("site_symmetry")) for row in candidate.get("rows", [])))


def classify_negative(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    if str(candidate.get("canonical_skeleton_key")) == str(record["canonical_skeleton_key"]):
        return "same_skeleton_wrong_assignment"
    gt_mult = tuple(sorted(safe_int(row.get("multiplicity")) for row in record.get("wa_table", [])))
    gt_sym = tuple(sorted(str(row.get("site_symmetry")) for row in record.get("wa_table", [])))
    if multiplicity_pattern(candidate) == gt_mult and site_sym_pattern(candidate) != gt_sym:
        return "same_multiplicity_pattern_wrong_site_symmetry"
    if len(candidate.get("rows", [])) == int(record["n_sites"]):
        return "same_sg_same_nsites_wrong_orbit_token"
    if candidate.get("policy_rank") is not None:
        return "policy_high_score_wrong_candidate"
    if candidate.get("old_rank") is not None:
        return "old_search_high_score_wrong_candidate"
    return "other_hard_negative"


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        -safe_float(candidate.get("hybrid_score")),
        safe_int(candidate.get("policy_rank"), 1_000_000),
        safe_int(candidate.get("old_rank"), 1_000_000),
        str(candidate.get("canonical_wa_key")),
    )


def select_training_candidates(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int | None, dict[str, int]]:
    target = str(record["canonical_wa_key"])
    positive_indexes = [i for i, c in enumerate(candidates) if str(c.get("canonical_wa_key")) == target]
    if not positive_indexes:
        return [], None, {}
    positive = dict(candidates[positive_indexes[0]])
    positive["training_label"] = "positive_gt_wa"
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if str(candidate.get("canonical_wa_key")) == target:
            continue
        c = dict(candidate)
        label = classify_negative(record, c)
        c["training_label"] = label
        buckets[label].append(c)
    for items in buckets.values():
        items.sort(key=candidate_sort_key)
    selected: list[dict[str, Any]] = [positive]
    quotas = {
        "same_skeleton_wrong_assignment": max(12, max_candidates // 8),
        "same_multiplicity_pattern_wrong_site_symmetry": max(12, max_candidates // 8),
        "same_sg_same_nsites_wrong_orbit_token": max(16, max_candidates // 6),
        "policy_high_score_wrong_candidate": max(24, max_candidates // 5),
        "old_search_high_score_wrong_candidate": max(24, max_candidates // 5),
        "other_hard_negative": max(16, max_candidates // 8),
    }
    seen = {str(positive.get("canonical_wa_key"))}
    for label, quota in quotas.items():
        for candidate in buckets.get(label, [])[:quota]:
            key = str(candidate.get("canonical_wa_key"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= max_candidates:
                break
        if len(selected) >= max_candidates:
            break
    if len(selected) < max_candidates:
        rest = [dict(c) for c in candidates if str(c.get("canonical_wa_key")) not in seen]
        rest.sort(key=candidate_sort_key)
        head = rest[: max(0, max_candidates - len(selected))]
        selected.extend(head)
        seen.update(str(c.get("canonical_wa_key")) for c in head)
    if len(selected) > 1:
        tail = selected[1:]
        rng.shuffle(tail)
        selected = [selected[0], *tail]
    target_index = next(i for i, c in enumerate(selected) if str(c.get("canonical_wa_key")) == target)
    counts = {label: len(items) for label, items in buckets.items()}
    return selected, target_index, counts


def build_samples(
    data_root: Path,
    candidate_dir: Path,
    split: str,
    *,
    max_candidates: int,
    seed: int,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = {str(row["sample_id"]): row for row in read_jsonl(data_root / f"{split}.jsonl")}
    candidate_rows = read_jsonl(split_candidates_path(candidate_dir, split))
    if max_rows is not None:
        candidate_rows = candidate_rows[: int(max_rows)]
    rng = random.Random(seed + hash(split) % 10_000)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cand_row in candidate_rows:
        record = records[str(cand_row["sample_id"])]
        candidates = [dict(c) for c in cand_row.get("candidates", [])]
        selected, target_index, negative_counts = select_training_candidates(
            record,
            candidates,
            max_candidates=max_candidates,
            rng=rng,
        )
        if target_index is None:
            skipped.append(
                {
                    "split": split,
                    "sample_id": cand_row["sample_id"],
                    "candidate_count": len(candidates),
                    "reason": "gt_wa_not_in_candidates",
                }
            )
            continue
        samples.append(
            {
                "split": split,
                "record": record,
                "candidates": selected,
                "target_index": int(target_index),
                "negative_counts": negative_counts,
            }
        )
    return samples, skipped, candidate_rows


def make_batch(record: dict[str, Any], candidates: list[dict[str, Any]], vocab: RerankVocab, device: torch.device) -> dict[str, torch.Tensor]:
    cands = len(candidates)
    max_rows = max(1, max(len(c.get("rows", [])) for c in candidates))
    max_sources = max(1, max(len(c.get("source_labels", [])) for c in candidates))
    element_id = torch.zeros((cands, max_rows), dtype=torch.long)
    orbit_id = torch.zeros((cands, max_rows), dtype=torch.long)
    sg_id = torch.zeros((cands, max_rows), dtype=torch.long)
    letter_id = torch.zeros((cands, max_rows), dtype=torch.long)
    site_sym_id = torch.zeros((cands, max_rows), dtype=torch.long)
    row_mask = torch.zeros((cands, max_rows), dtype=torch.float32)
    row_numeric = torch.zeros((cands, max_rows, 8), dtype=torch.float32)
    source_id = torch.zeros((cands, max_sources), dtype=torch.long)
    source_mask = torch.zeros((cands, max_sources), dtype=torch.float32)
    cand_numeric = torch.zeros((cands, 12), dtype=torch.float32)
    total_atoms = max(1, int(record["atom_count"]))
    max_count = max([safe_int(v) for v in record["formula_counts"].values()] or [0])
    min_count = min([safe_int(v) for v in record["formula_counts"].values()] or [0])
    for i, candidate in enumerate(candidates):
        rows = candidate.get("rows", [])
        for j, row in enumerate(rows):
            element = str(row.get("element"))
            orbit = str(row.get("orbit_id"))
            element_id[i, j] = vocab.element.get(element, 0)
            orbit_id[i, j] = vocab.orbit.get(orbit, 0)
            sg_id[i, j] = vocab.sg.get(str(int(record["sg"])), 0)
            letter_id[i, j] = vocab.letter.get(str(row.get("letter")), 0)
            site_sym_id[i, j] = vocab.site_sym.get(str(row.get("site_symmetry")), 0)
            mult = safe_int(row.get("multiplicity"))
            enumeration = safe_int(row.get("enumeration"), -1)
            row_numeric[i, j] = torch.tensor(
                [
                    float(int(record["sg"])) / 230.0,
                    float(mult) / float(total_atoms),
                    float(j + 1) / 64.0,
                    float(len(rows)) / 64.0,
                    float(enumeration + 1) / 32.0,
                    1.0 if mult == 1 else 0.0,
                    safe_float(candidate.get("policy_rank"), 1_000_000.0) / 1000.0,
                    safe_float(candidate.get("old_rank"), 1_000_000.0) / 1000.0,
                ],
                dtype=torch.float32,
            )
            row_mask[i, j] = 1.0
        labels = candidate.get("source_labels", [])
        for j, source in enumerate(labels):
            source_id[i, j] = vocab.source.get(str(source), 0)
            source_mask[i, j] = 1.0
        if not labels:
            source_mask[i, 0] = 1.0
        cand_numeric[i] = torch.tensor(
            [
                float(int(record["sg"])) / 230.0,
                float(total_atoms) / 256.0,
                float(int(record["n_sites"])) / 64.0,
                float(int(record["num_elements"])) / 10.0,
                float(max_count) / float(total_atoms),
                float(min_count) / float(total_atoms),
                math.log1p(max(0.0, safe_float(candidate.get("hybrid_score")))),
                1.0 / max(1.0, safe_float(candidate.get("policy_rank"), 1_000_000.0)),
                1.0 / max(1.0, safe_float(candidate.get("old_rank"), 1_000_000.0)),
                1.0 if "policy" in labels else 0.0,
                1.0 if "old" in labels else 0.0,
                1.0 if len(labels) > 1 else 0.0,
            ],
            dtype=torch.float32,
        )
    return {
        "element_id": element_id.to(device, non_blocking=True),
        "orbit_id": orbit_id.to(device, non_blocking=True),
        "sg_id": sg_id.to(device, non_blocking=True),
        "letter_id": letter_id.to(device, non_blocking=True),
        "site_sym_id": site_sym_id.to(device, non_blocking=True),
        "row_mask": row_mask.to(device, non_blocking=True),
        "row_numeric": row_numeric.to(device, non_blocking=True),
        "source_id": source_id.to(device, non_blocking=True),
        "source_mask": source_mask.to(device, non_blocking=True),
        "cand_numeric": cand_numeric.to(device, non_blocking=True),
    }


@torch.no_grad()
def evaluate_samples(
    model: OrbitAwareListwiseRanker,
    samples: list[dict[str, Any]],
    vocab: RerankVocab,
    device: torch.device,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    model.eval()
    rows = samples if max_eval_samples is None else samples[: int(max_eval_samples)]
    loss_sum = 0.0
    hits = {k: 0 for k in TOP_KS}
    mrr = 0.0
    for sample in rows:
        logits = model(make_batch(sample["record"], sample["candidates"], vocab, device))
        target = torch.tensor([int(sample["target_index"])], dtype=torch.long, device=device)
        loss_sum += float(F.cross_entropy(logits.unsqueeze(0), target).detach().cpu())
        order = torch.argsort(logits, descending=True).detach().cpu().tolist()
        rank = order.index(int(sample["target_index"])) + 1
        mrr += 1.0 / float(rank)
        for k in TOP_KS:
            hits[k] += int(rank <= k)
    denom = max(1, len(rows))
    out = {"samples": len(rows), "loss": loss_sum / denom, "mrr": mrr / denom}
    for k in TOP_KS:
        out[f"wa_top{k}"] = hits[k] / denom
    return out


def topk_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"samples": len(predictions)}
    for k in TOP_KS:
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
        rows.append({key: value, **topk_metrics(items)})
    return rows


def jsonable_config(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


@torch.no_grad()
def rank_split(
    model: OrbitAwareListwiseRanker,
    data_root: Path,
    candidate_dir: Path,
    split: str,
    vocab: RerankVocab,
    device: torch.device,
) -> list[dict[str, Any]]:
    records = {str(row["sample_id"]): row for row in read_jsonl(data_root / f"{split}.jsonl")}
    candidate_rows = read_jsonl(split_candidates_path(candidate_dir, split))
    model.eval()
    out: list[dict[str, Any]] = []
    for cand_row in candidate_rows:
        record = records[str(cand_row["sample_id"])]
        candidates = [dict(c) for c in cand_row.get("candidates", [])]
        if candidates:
            logits = model(make_batch(record, candidates, vocab, device)).detach().cpu().tolist()
            for candidate, score in zip(candidates, logits):
                candidate["reranker_score"] = float(score)
            candidates.sort(
                key=lambda c: (
                    -safe_float(c.get("reranker_score")),
                    -safe_float(c.get("hybrid_score")),
                    safe_int(c.get("policy_rank"), 1_000_000),
                    safe_int(c.get("old_rank"), 1_000_000),
                    str(c.get("canonical_wa_key")),
                )
            )
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


def train_model(
    model: OrbitAwareListwiseRanker,
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    vocab: RerankVocab,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    eval_max_samples: int | None,
    grad_accum_steps: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        rng.shuffle(train_samples)
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        top1 = 0
        for i, sample in enumerate(train_samples, start=1):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(make_batch(sample["record"], sample["candidates"], vocab, device))
                target = torch.tensor([int(sample["target_index"])], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits.unsqueeze(0), target) / max(1, int(grad_accum_steps))
            scaler.scale(loss).backward()
            if i % int(grad_accum_steps) == 0 or i == len(train_samples):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * max(1, int(grad_accum_steps))
            top1 += int(int(torch.argmax(logits.detach()).item()) == int(sample["target_index"]))
            if i % 250 == 0:
                print(f"[reranker] epoch={epoch} samples={i}/{len(train_samples)} loss={total_loss / i:.4f}", flush=True)
        train_eval = evaluate_samples(model, train_samples, vocab, device, max_eval_samples=eval_max_samples)
        val_eval = evaluate_samples(model, val_samples, vocab, device, max_eval_samples=eval_max_samples)
        row = {
            "epoch": epoch,
            "train_loss_online": total_loss / max(1, len(train_samples)),
            "train_top1_online": top1 / max(1, len(train_samples)),
            "train_eval": train_eval,
            "val_eval": val_eval,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    return history


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an orbit-aware listwise Transformer reranker for hybrid SymCIF-v4 WA candidates.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_reextracted")
    parser.add_argument("--candidate-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs" / "orbit_aware_listwise_reranker_v1")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_hybrid_search")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--row-layers", type=int, default=2)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-train-candidates", type=int, default=320)
    parser.add_argument("--max-val-candidates", type=int, default=500)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_samples, train_skipped, train_candidate_rows = build_samples(
        args.data_root,
        args.candidate_dir,
        "train",
        max_candidates=args.max_train_candidates,
        seed=args.seed,
        max_rows=args.max_train_rows,
    )
    val_samples, val_skipped, val_candidate_rows = build_samples(
        args.data_root,
        args.candidate_dir,
        "val",
        max_candidates=args.max_val_candidates,
        seed=args.seed,
        max_rows=args.max_val_rows,
    )
    if not train_samples:
        raise RuntimeError("no train samples with GT WA in hybrid candidate set")
    train_records = read_jsonl(args.data_root / "train.jsonl")
    vocab = RerankVocab.from_rows(train_records, train_candidate_rows + val_candidate_rows)
    model = OrbitAwareListwiseRanker(
        vocab.sizes(),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        row_layers=args.row_layers,
        cand_layers=args.candidate_layers,
        heads=args.heads,
    ).to(device)
    print(
        json.dumps(
            {
                "device": str(device),
                "torch_cuda_available": torch.cuda.is_available(),
                "train_samples_with_positive": len(train_samples),
                "train_skipped_no_positive": len(train_skipped),
                "val_samples_with_positive": len(val_samples),
                "val_skipped_no_positive": len(val_skipped),
                "vocab_sizes": vocab.sizes(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    history = train_model(
        model,
        train_samples,
        val_samples,
        vocab,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_max_samples=args.eval_max_samples,
        grad_accum_steps=args.grad_accum_steps,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab.to_jsonable(),
            "vocab_sizes": vocab.sizes(),
            "config": vars(args),
            "history": history,
        },
        args.run_dir / "ckpt.pt",
    )
    predictions = rank_split(model, args.data_root, args.candidate_dir, "test", vocab, device)
    write_jsonl(args.out_dir / "test_reranked_predictions.jsonl", predictions)
    metrics = topk_metrics(predictions)
    summary = {
        "config": {**jsonable_config(args), "device": str(device)},
        "train_samples": len(train_samples),
        "train_skipped_no_positive": len(train_skipped),
        "val_samples": len(val_samples),
        "val_skipped_no_positive": len(val_skipped),
        "test": metrics,
        "complex_nsites_ge6": topk_metrics([p for p in predictions if int(p["n_sites"]) >= 6]),
        "complex_num_elements_ge4": topk_metrics([p for p in predictions if int(p["num_elements"]) >= 4]),
        "sg_breakouts": {str(sg): topk_metrics([p for p in predictions if int(p["sg"]) == sg]) for sg in (2, 65, 71, 127)},
        "history": history,
        "train_skipped_examples": train_skipped[:50],
        "val_skipped_examples": val_skipped[:50],
        "acceptance": {
            "wa_top200_ge_95": float(metrics.get("wa_top200", 0.0)) >= 0.95,
            "wa_top100_ge_92": float(metrics.get("wa_top100", 0.0)) >= 0.92,
            "wa_top20_gt_83": float(metrics.get("wa_top20", 0.0)) > 0.83,
            "skeleton_top200_ge_97": float(metrics.get("skeleton_top200", 0.0)) >= 0.97,
        },
    }
    (args.out_dir / "reranker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.out_dir / "reranker_breakdown_per_sg.csv", group_metrics(predictions, "sg"))
    write_csv(args.out_dir / "reranker_breakdown_per_nsites.csv", group_metrics(predictions, "n_sites"))
    write_csv(args.out_dir / "reranker_breakdown_per_num_elements.csv", group_metrics(predictions, "num_elements"))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
