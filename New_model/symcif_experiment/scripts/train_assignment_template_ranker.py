#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # type: ignore  # noqa: E402
import torch  # type: ignore  # noqa: E402
import torch.nn as nn  # type: ignore  # noqa: E402
import torch.nn.functional as F  # type: ignore  # noqa: E402

from train_skeleton_template_ranker import ELEMENTS, ELEMENT_TO_IDX, make_features, read_jsonl, topk_from_scores  # noqa: E402


def assignment_key(row: dict[str, Any]) -> str:
    parts = []
    for site in row["assignment"]:
        parts.append(f"{int(site['multiplicity'])}{site['letter']}:{site['element']}")
    return "|".join(parts)


def assignment_uid(row: dict[str, Any]) -> str:
    return f"{row['skeleton_template_uid']}|assign={assignment_key(row)}"


def assignment_counts_from_key(key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not key:
        return {}
    for part in key.split("|"):
        left, element = part.split(":", 1)
        mult = int("".join(ch for ch in left if ch.isdigit()))
        counts[element] += mult
    return dict(sorted(counts.items()))


def counts_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return {str(k): int(v) for k, v in a.items()} == {str(k): int(v) for k, v in b.items()}


class AssignmentDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], assignment_to_idx: dict[str, int], skeleton_to_idx: dict[str, int]):
        self.rows = [
            row
            for row in rows
            if assignment_uid(row) in assignment_to_idx and row["skeleton_template_uid"] in skeleton_to_idx
        ]
        self.features = make_features(self.rows)
        self.sg = np.array([int(row["sg"]) for row in self.rows], dtype=np.int64)
        self.skeleton_idx = np.array([skeleton_to_idx[row["skeleton_template_uid"]] for row in self.rows], dtype=np.int64)
        self.labels = np.array([assignment_to_idx[assignment_uid(row)] for row in self.rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.sg[idx], dtype=torch.long),
            torch.tensor(self.skeleton_idx[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class AssignmentRanker(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_skeletons: int,
        num_assignments: int,
        sg_emb_dim: int = 48,
        skeleton_emb_dim: int = 96,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sg_emb = nn.Embedding(231, sg_emb_dim)
        self.skeleton_emb = nn.Embedding(num_skeletons, skeleton_emb_dim)
        self.formula_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + sg_emb_dim + skeleton_emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_assignments),
        )

    def forward(self, features: torch.Tensor, sg: torch.Tensor, skeleton_idx: torch.Tensor) -> torch.Tensor:
        x = self.formula_encoder(features)
        sg_emb = self.sg_emb(sg.clamp(min=0, max=230))
        skel_emb = self.skeleton_emb(skeleton_idx)
        return self.head(torch.cat([x, sg_emb, skel_emb], dim=-1))


def mask_logits_by_skeleton(logits: torch.Tensor, skeleton_idx: torch.Tensor, assignment_skeleton_idx: torch.Tensor) -> torch.Tensor:
    mask = assignment_skeleton_idx.unsqueeze(0).to(logits.device) == skeleton_idx.unsqueeze(1)
    return logits.masked_fill(~mask, -1e9)


def train_one_epoch(
    model: AssignmentRanker,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    assignment_skeleton_idx: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for features, sg, skeleton_idx, labels in loader:
        features = features.to(device)
        sg = sg.to(device)
        skeleton_idx = skeleton_idx.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = mask_logits_by_skeleton(model(features, sg, skeleton_idx), skeleton_idx, assignment_skeleton_idx)
        loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(features.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs
    return total_loss / max(1, total)


@torch.no_grad()
def model_scores(
    model: AssignmentRanker,
    rows: list[dict[str, Any]],
    skeleton_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    usable = [row for row in rows if row["skeleton_template_uid"] in skeleton_to_idx]
    if len(usable) != len(rows):
        raise ValueError("model_scores requires all rows to have known skeleton ids")
    feats = make_features(rows)
    sgs = np.array([int(row["sg"]) for row in rows], dtype=np.int64)
    skeletons = np.array([skeleton_to_idx[row["skeleton_template_uid"]] for row in rows], dtype=np.int64)
    outputs: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        out = model(
            torch.from_numpy(feats[start:end]).to(device),
            torch.from_numpy(sgs[start:end]).to(device),
            torch.from_numpy(skeletons[start:end]).to(device),
        )
        outputs.append(out.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def candidate_indices(
    row: dict[str, Any],
    assignment_items: list[dict[str, Any]],
    skeleton_to_assignment_indices: dict[str, list[int]],
    *,
    formula_mask: bool,
) -> list[int]:
    indices = skeleton_to_assignment_indices.get(row["skeleton_template_uid"], [])
    if not formula_mask:
        return indices
    return [
        idx
        for idx in indices
        if counts_match(assignment_items[idx]["formula_counts"], row["formula_counts"])
    ]


def evaluate_assignment(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    assignment_to_idx: dict[str, int],
    assignment_items: list[dict[str, Any]],
    skeleton_to_assignment_indices: dict[str, list[int]],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ks = (1, 5, 20)
    hits = {k: 0 for k in ks}
    hits_seen = {k: 0 for k in ks}
    seen = 0
    empty = 0
    preds: list[dict[str, Any]] = []
    formula_mask = mode.endswith("+formula")
    for i, row in enumerate(rows):
        target_uid = assignment_uid(row)
        target_idx = assignment_to_idx.get(target_uid)
        if target_idx is not None:
            seen += 1
        candidates = candidate_indices(row, assignment_items, skeleton_to_assignment_indices, formula_mask=formula_mask)
        if not candidates:
            empty += 1
        ranked20 = topk_from_scores(scores[i], candidates, 20)
        preds.append(
            {
                "sample_id": row["sample_id"],
                "formula": row["formula"],
                "sg": int(row["sg"]),
                "skeleton_template_id": row["skeleton_template_id"],
                "skeleton_template_uid": row["skeleton_template_uid"],
                "target_assignment_uid": target_uid,
                "target_seen_train": target_idx is not None,
                "top20_assignment_uids": [assignment_items[j]["assignment_uid"] for j in ranked20],
            }
        )
        for k in ks:
            ranked = ranked20[:k]
            if target_idx is not None and target_idx in ranked:
                hits[k] += 1
                hits_seen[k] += 1
    metrics: dict[str, Any] = {
        "mode": mode,
        "samples": len(rows),
        "seen_target_samples": seen,
        "unseen_target_samples": len(rows) - seen,
        "seen_target_rate": seen / max(1, len(rows)),
        "empty_candidate_samples": empty,
    }
    for k in ks:
        metrics[f"assignment_top{k}"] = hits[k] / max(1, len(rows))
        metrics[f"assignment_top{k}_seen_only"] = hits_seen[k] / max(1, seen)
    return metrics, preds


def frequency_scores(rows: list[dict[str, Any]], assignment_items: list[dict[str, Any]]) -> np.ndarray:
    scores = np.zeros((len(rows), len(assignment_items)), dtype=np.float32)
    for j, item in enumerate(assignment_items):
        score = math.log1p(float(item["train_frequency"]))
        for i, row in enumerate(rows):
            scores[i, j] = score if row["skeleton_template_uid"] == item["skeleton_template_uid"] else -1e9
    return scores


def joint_pair_recall(
    test_rows: list[dict[str, Any]],
    skeleton_predictions_path: Path,
    assignment_scores: np.ndarray,
    assignment_to_idx: dict[str, int],
    assignment_items: list[dict[str, Any]],
    skeleton_to_assignment_indices: dict[str, list[int]],
    *,
    skeleton_k: int = 5,
    assignment_k: int = 5,
) -> dict[str, Any]:
    skel_preds = [json.loads(line) for line in skeleton_predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pred_by_id = {row["sample_id"]: row for row in skel_preds}
    hit = 0
    reachable = 0
    for i, row in enumerate(test_rows):
        target_assignment = assignment_uid(row)
        if target_assignment in assignment_to_idx:
            reachable += 1
        sp = pred_by_id.get(row["sample_id"])
        if not sp:
            continue
        candidate_pairs: set[str] = set()
        for skeleton_uid in sp["top20_template_uids"][:skeleton_k]:
            pseudo = dict(row)
            pseudo["skeleton_template_uid"] = skeleton_uid
            indices = candidate_indices(pseudo, assignment_items, skeleton_to_assignment_indices, formula_mask=True)
            ranked = topk_from_scores(assignment_scores[i], indices, assignment_k)
            candidate_pairs.update(assignment_items[j]["assignment_uid"] for j in ranked)
        if target_assignment in candidate_pairs:
            hit += 1
    return {
        "samples": len(test_rows),
        "reachable_seen_assignment_samples": reachable,
        "skeleton_k": skeleton_k,
        "assignment_k": assignment_k,
        "joint_pair_recall": hit / max(1, len(test_rows)),
        "joint_pair_recall_seen_only": hit / max(1, reachable),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train first-pass SymCIF assignment ranker.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--skeleton-predictions", type=Path, default=PROJECT_ROOT / "reports" / "skeleton_template_ranker_v1" / "test_predictions_mlp_sg_formula.jsonl")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "assignment_template_ranker_v1")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "assignment_template_ranker_v1")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
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

    skeleton_uids = sorted({row["skeleton_template_uid"] for row in train_rows})
    skeleton_to_idx = {uid: i for i, uid in enumerate(skeleton_uids)}
    assignment_counter: Counter[str] = Counter(assignment_uid(row) for row in train_rows)
    assignment_items: list[dict[str, Any]] = []
    for uid in sorted(assignment_counter):
        skel_uid, key = uid.split("|assign=", 1)
        assignment_items.append(
            {
                "assignment_uid": uid,
                "skeleton_template_uid": skel_uid,
                "assignment_key": key,
                "formula_counts": assignment_counts_from_key(key),
                "train_frequency": int(assignment_counter[uid]),
            }
        )
    assignment_to_idx = {item["assignment_uid"]: i for i, item in enumerate(assignment_items)}
    skeleton_to_assignment_indices: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(assignment_items):
        skeleton_to_assignment_indices[item["skeleton_template_uid"]].append(i)
    for indices in skeleton_to_assignment_indices.values():
        indices.sort(key=lambda j: (-assignment_items[j]["train_frequency"], assignment_items[j]["assignment_uid"]))

    assignment_skeleton_idx = torch.tensor(
        [skeleton_to_idx[item["skeleton_template_uid"]] for item in assignment_items],
        dtype=torch.long,
    )
    counts = torch.tensor([max(1, int(item["train_frequency"])) for item in assignment_items], dtype=torch.float32)
    class_weights = torch.clamp((counts.mean() / counts).sqrt(), 0.25, 4.0)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = AssignmentDataset(train_rows, assignment_to_idx, skeleton_to_idx)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = AssignmentRanker(
        feature_dim=len(ELEMENTS) * 3 + 2,
        num_skeletons=len(skeleton_to_idx),
        num_assignments=len(assignment_items),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_top5 = -1.0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, assignment_skeleton_idx.to(device), class_weights, device)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_known = [row for row in val_rows if row["skeleton_template_uid"] in skeleton_to_idx]
            scores = model_scores(model, val_known, skeleton_to_idx, device, args.batch_size)
            metrics, _ = evaluate_assignment(
                val_known,
                scores,
                assignment_to_idx,
                assignment_items,
                skeleton_to_assignment_indices,
                mode="mlp_oracle_skeleton+formula",
            )
            row = {"epoch": epoch, "loss": loss, **{k: v for k, v in metrics.items() if k.startswith("assignment_top")}}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if float(metrics["assignment_top5"]) > best_top5:
                best_top5 = float(metrics["assignment_top5"])
                best_state = {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_metrics": metrics,
                }
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    summaries: dict[str, Any] = {
        "train": {
            "samples": len(train_rows),
            "train_skeletons": len(skeleton_to_idx),
            "train_assignments": len(assignment_items),
            "device": str(device),
            "best_epoch": None if best_state is None else int(best_state["epoch"]),
            "best_val_metrics": None if best_state is None else best_state["val_metrics"],
        },
        "metrics": {},
    }
    final_predictions: list[dict[str, Any]] = []
    for split_name, rows in (("val", val_rows), ("test", test_rows)):
        known = [row for row in rows if row["skeleton_template_uid"] in skeleton_to_idx]
        scores = model_scores(model, known, skeleton_to_idx, device, args.batch_size)
        for mode_name, mode_scores in (
            ("frequency_oracle_skeleton", frequency_scores(known, assignment_items)),
            ("frequency_oracle_skeleton+formula", frequency_scores(known, assignment_items)),
            ("mlp_oracle_skeleton", scores),
            ("mlp_oracle_skeleton+formula", scores),
        ):
            metrics, preds = evaluate_assignment(
                known,
                mode_scores,
                assignment_to_idx,
                assignment_items,
                skeleton_to_assignment_indices,
                mode=mode_name,
            )
            metrics["known_skeleton_samples"] = len(known)
            metrics["all_split_samples"] = len(rows)
            summaries["metrics"][f"{split_name}:{mode_name}"] = metrics
            if split_name == "test" and mode_name == "mlp_oracle_skeleton+formula":
                final_predictions = preds
                if args.skeleton_predictions.exists():
                    summaries["metrics"]["test:joint_skeleton5_assignment5"] = joint_pair_recall(
                        known,
                        args.skeleton_predictions,
                        scores,
                        assignment_to_idx,
                        assignment_items,
                        skeleton_to_assignment_indices,
                        skeleton_k=5,
                        assignment_k=5,
                    )

    with (args.reports_dir / "test_predictions_mlp_oracle_skeleton_formula.jsonl").open("w", encoding="utf-8") as f:
        for pred in final_predictions:
            f.write(json.dumps(pred, ensure_ascii=False, sort_keys=True) + "\n")
    (args.reports_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.reports_dir / "eval_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "assignment_to_idx": assignment_to_idx,
            "assignment_items": assignment_items,
            "skeleton_to_idx": skeleton_to_idx,
            "element_order": ELEMENTS,
            "args": vars(args),
            "summary": summaries,
        },
        args.out_dir / "ckpt.pt",
    )
    print(json.dumps(summaries["metrics"].get("test:mlp_oracle_skeleton+formula", {}), indent=2, sort_keys=True))
    print(json.dumps(summaries["metrics"].get("test:joint_skeleton5_assignment5", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
