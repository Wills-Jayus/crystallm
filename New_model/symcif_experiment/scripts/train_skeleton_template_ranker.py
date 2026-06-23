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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # type: ignore  # noqa: E402
import torch  # type: ignore  # noqa: E402
import torch.nn as nn  # type: ignore  # noqa: E402
import torch.nn.functional as F  # type: ignore  # noqa: E402
from pymatgen.core import Element  # type: ignore  # noqa: E402

from export_symcif_v3_structured import is_formula_compatible  # noqa: E402


ELEMENTS = [Element.from_Z(z).symbol for z in range(1, 119)]
ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENTS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_features(rows: list[dict[str, Any]]) -> np.ndarray:
    feats = np.zeros((len(rows), len(ELEMENTS) * 3 + 2), dtype=np.float32)
    for i, row in enumerate(rows):
        counts = {str(k): float(v) for k, v in row["formula_counts"].items()}
        atom_count = max(1.0, float(row["atom_count"]))
        denom = max(1.0, math.log1p(atom_count))
        for el, count in counts.items():
            idx = ELEMENT_TO_IDX.get(el)
            if idx is None:
                continue
            feats[i, idx] = count / atom_count
            feats[i, len(ELEMENTS) + idx] = math.log1p(count) / denom
            feats[i, len(ELEMENTS) * 2 + idx] = 1.0
        feats[i, len(ELEMENTS) * 3] = math.log1p(atom_count) / math.log1p(300.0)
        feats[i, len(ELEMENTS) * 3 + 1] = float(row["num_elements"]) / 12.0
    return feats


class RankerDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], class_to_idx: dict[str, int]):
        self.rows = [row for row in rows if row["skeleton_template_uid"] in class_to_idx]
        self.features = make_features(self.rows)
        self.sg = np.array([int(row["sg"]) for row in self.rows], dtype=np.int64)
        self.labels = np.array([class_to_idx[row["skeleton_template_uid"]] for row in self.rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.sg[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class SkeletonTemplateRanker(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, sg_emb_dim: int = 64, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.sg_emb = nn.Embedding(231, sg_emb_dim)
        self.formula_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + sg_emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
        x = self.formula_encoder(features)
        emb = self.sg_emb(sg.clamp(min=0, max=230))
        return self.head(torch.cat([x, emb], dim=-1))


def masked_logits_by_sg(logits: torch.Tensor, sg: torch.Tensor, class_sg: torch.Tensor) -> torch.Tensor:
    mask = class_sg.unsqueeze(0).to(logits.device) == sg.unsqueeze(1)
    return logits.masked_fill(~mask, -1e9)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    class_sg: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for features, sg, labels in loader:
        features = features.to(device)
        sg = sg.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = masked_logits_by_sg(model(features, sg), sg, class_sg)
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(features.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs
    return total_loss / max(1, total)


def topk_from_scores(scores: np.ndarray, candidate_indices: list[int], k: int) -> list[int]:
    if not candidate_indices:
        return []
    cand = np.array(candidate_indices, dtype=np.int64)
    cand_scores = scores[cand]
    order = np.argsort(-cand_scores, kind="mergesort")[: min(k, len(cand))]
    return [int(cand[i]) for i in order]


def compatible_indices(row: dict[str, Any], train_templates: list[dict[str, Any]], sg_to_indices: dict[int, list[int]]) -> list[int]:
    out: list[int] = []
    for idx in sg_to_indices.get(int(row["sg"]), []):
        item = train_templates[idx]
        if is_formula_compatible([int(v) for v in item["multiplicities"]], row["formula_counts"]):
            out.append(idx)
    return out


def eval_predictions(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    class_to_idx: dict[str, int],
    idx_to_uid: list[str],
    train_templates: list[dict[str, Any]],
    sg_to_indices: dict[int, list[int]],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ks = (1, 5, 20)
    metrics: dict[str, Any] = {
        "mode": mode,
        "samples": len(rows),
        "seen_target_samples": 0,
        "unseen_target_samples": 0,
    }
    hits = {k: 0 for k in ks}
    hits_seen = {k: 0 for k in ks}
    compatible_any = {k: 0 for k in ks}
    compatible_fraction_sum = {k: 0.0 for k in ks}
    compatible_mask_empty = 0
    predictions: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        uid = row["skeleton_template_uid"]
        target_idx = class_to_idx.get(uid)
        if target_idx is None:
            metrics["unseen_target_samples"] += 1
        else:
            metrics["seen_target_samples"] += 1
        if mode.endswith("+formula"):
            candidates = compatible_indices(row, train_templates, sg_to_indices)
            if not candidates:
                compatible_mask_empty += 1
        else:
            candidates = sg_to_indices.get(int(row["sg"]), [])
        ranked20 = topk_from_scores(scores[i], candidates, 20)
        pred = {
            "sample_id": row["sample_id"],
            "formula": row["formula"],
            "sg": int(row["sg"]),
            "n_sites": int(row["n_sites"]),
            "target_template_id": row["skeleton_template_id"],
            "target_template_uid": uid,
            "target_seen_train": target_idx is not None,
            "top20_template_ids": [train_templates[j]["skeleton_template_id"] for j in ranked20],
            "top20_template_uids": [idx_to_uid[j] for j in ranked20],
        }
        predictions.append(pred)

        for k in ks:
            ranked = ranked20[:k]
            if target_idx is not None and target_idx in ranked:
                hits[k] += 1
                hits_seen[k] += 1
            if ranked:
                compat_flags = [
                    is_formula_compatible([int(v) for v in train_templates[j]["multiplicities"]], row["formula_counts"])
                    for j in ranked
                ]
                if any(compat_flags):
                    compatible_any[k] += 1
                compatible_fraction_sum[k] += sum(compat_flags) / len(compat_flags)

    for k in ks:
        metrics[f"skeleton_top{k}"] = hits[k] / max(1, len(rows))
        metrics[f"skeleton_top{k}_seen_only"] = hits_seen[k] / max(1, metrics["seen_target_samples"])
        metrics[f"formula_compatible_any_top{k}"] = compatible_any[k] / max(1, len(rows))
        metrics[f"formula_compatible_fraction_top{k}"] = compatible_fraction_sum[k] / max(1, len(rows))
    metrics["compatible_mask_empty_samples"] = compatible_mask_empty
    metrics["seen_target_rate"] = metrics["seen_target_samples"] / max(1, len(rows))
    return metrics, predictions


def frequency_scores(rows: list[dict[str, Any]], train_templates: list[dict[str, Any]], *, by_formula: bool = False) -> np.ndarray:
    scores = np.zeros((len(rows), len(train_templates)), dtype=np.float32)
    for j, item in enumerate(train_templates):
        prior = math.log1p(float(item["train_frequency"]))
        for i, row in enumerate(rows):
            if int(row["sg"]) != int(item["sg"]):
                scores[i, j] = -1e9
                continue
            if by_formula and not is_formula_compatible([int(v) for v in item["multiplicities"]], row["formula_counts"]):
                scores[i, j] = -1e9
                continue
            scores[i, j] = prior
    return scores


@torch.no_grad()
def model_scores(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    feats = make_features(rows)
    sgs = np.array([int(row["sg"]) for row in rows], dtype=np.int64)
    outputs: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        features = torch.from_numpy(feats[start:end]).to(device)
        sg = torch.from_numpy(sgs[start:end]).to(device)
        logits = model(features, sg)
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def aggregate_breakdown(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_to_idx: dict[str, int],
    group_key: str,
    ks: tuple[int, ...] = (1, 5, 20),
) -> list[dict[str, Any]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[group_key]].append(i)
    out: list[dict[str, Any]] = []
    for key, indices in sorted(groups.items(), key=lambda kv: kv[0]):
        item: dict[str, Any] = {group_key: key, "samples": len(indices)}
        seen = 0
        for k in ks:
            hit = 0
            for i in indices:
                target_idx = class_to_idx.get(rows[i]["skeleton_template_uid"])
                if target_idx is not None:
                    seen += int(k == ks[0])
                    if rows[i]["skeleton_template_uid"] in predictions[i]["top20_template_uids"][:k]:
                        hit += 1
            item[f"top{k}"] = hit / max(1, len(indices))
        item["seen_target_rate"] = seen / max(1, len(indices))
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train first-pass SymCIF skeleton template ranker.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "skeleton_template_ranker_v1")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "skeleton_template_ranker_v1")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.data_dir / "train.jsonl")
    val_rows = read_jsonl(args.data_dir / "val.jsonl")
    test_rows = read_jsonl(args.data_dir / "test.jsonl")
    catalog = read_jsonl(args.data_dir / "template_catalog.jsonl")
    train_templates = [item for item in catalog if int(item["train_frequency"]) > 0]
    train_templates.sort(key=lambda item: item["skeleton_template_uid"])
    idx_to_uid = [item["skeleton_template_uid"] for item in train_templates]
    class_to_idx = {uid: i for i, uid in enumerate(idx_to_uid)}
    for i, item in enumerate(train_templates):
        item["class_idx"] = i

    sg_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, item in enumerate(train_templates):
        sg_to_indices[int(item["sg"])].append(i)
    for sg, indices in sg_to_indices.items():
        indices.sort(key=lambda j: (-int(train_templates[j]["train_frequency"]), train_templates[j]["skeleton_template_uid"]))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = RankerDataset(train_rows, class_to_idx)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda" and args.num_workers > 0,
    )
    class_sg = torch.tensor([int(item["sg"]) for item in train_templates], dtype=torch.long, device=device)
    class_counts = torch.tensor([max(1, int(item["train_frequency"])) for item in train_templates], dtype=torch.float32)
    class_weights = (class_counts.float().mean() / class_counts.float()).sqrt()
    class_weights = torch.clamp(class_weights, 0.25, 4.0)

    model = SkeletonTemplateRanker(feature_dim=len(ELEMENTS) * 3 + 2, num_classes=len(train_templates)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, Any]] = []
    best_val_top5 = -1.0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, class_sg, class_weights, device)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_scores = model_scores(model, val_rows, device, args.batch_size)
            val_metrics, _ = eval_predictions(
                val_rows,
                val_scores,
                class_to_idx,
                idx_to_uid,
                train_templates,
                sg_to_indices,
                mode="mlp_sg",
            )
            row = {"epoch": epoch, "loss": loss, **{k: v for k, v in val_metrics.items() if k.startswith("skeleton_top")}}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if float(val_metrics["skeleton_top5"]) > best_val_top5:
                best_val_top5 = float(val_metrics["skeleton_top5"])
                best_state = {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_scores = model_scores(model, test_rows, device, args.batch_size)
    val_scores = model_scores(model, val_rows, device, args.batch_size)
    summaries: dict[str, Any] = {
        "train": {
            "samples": len(train_rows),
            "train_templates": len(train_templates),
            "device": str(device),
            "best_epoch": None if best_state is None else int(best_state["epoch"]),
            "best_val_metrics": None if best_state is None else best_state["val_metrics"],
        },
        "metrics": {},
    }

    all_predictions: dict[str, list[dict[str, Any]]] = {}
    for split_name, rows, scores in (("val", val_rows, val_scores), ("test", test_rows, test_scores)):
        for mode_name, mode_scores in (
            ("frequency_sg", frequency_scores(rows, train_templates, by_formula=False)),
            ("frequency_sg+formula", frequency_scores(rows, train_templates, by_formula=True)),
            ("mlp_sg", scores),
            ("mlp_sg+formula", scores),
        ):
            metrics, preds = eval_predictions(
                rows,
                mode_scores,
                class_to_idx,
                idx_to_uid,
                train_templates,
                sg_to_indices,
                mode=mode_name,
            )
            summaries["metrics"][f"{split_name}:{mode_name}"] = metrics
            if split_name == "test" and mode_name == "mlp_sg+formula":
                all_predictions["test_mlp_sg_formula"] = preds

    predictions = all_predictions["test_mlp_sg_formula"]
    write_csv(args.reports_dir / "per_sg_breakdown.csv", aggregate_breakdown(test_rows, predictions, class_to_idx, "sg"))
    write_csv(args.reports_dir / "per_nsites_breakdown.csv", aggregate_breakdown(test_rows, predictions, class_to_idx, "n_sites"))
    with (args.reports_dir / "test_predictions_mlp_sg_formula.jsonl").open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
    (args.reports_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.reports_dir / "eval_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_uid": idx_to_uid,
            "train_templates": train_templates,
            "element_order": ELEMENTS,
            "args": vars(args),
            "summary": summaries,
        },
        args.out_dir / "ckpt.pt",
    )
    print(json.dumps(summaries["metrics"]["test:mlp_sg+formula"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
