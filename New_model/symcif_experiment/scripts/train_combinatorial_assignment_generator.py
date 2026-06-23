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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "scripts",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # type: ignore  # noqa: E402
import torch  # type: ignore  # noqa: E402
import torch.nn as nn  # type: ignore  # noqa: E402
import torch.nn.functional as F  # type: ignore  # noqa: E402

from export_symcif_v3_structured import is_formula_compatible  # noqa: E402
from train_assignment_template_ranker import assignment_key, assignment_uid, counts_match  # noqa: E402
from train_skeleton_template_ranker import ELEMENTS, ELEMENT_TO_IDX, read_jsonl  # noqa: E402


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTER_TO_IDX = {ch: i for i, ch in enumerate(LETTERS)}
UNK_SKELETON = "<UNK>"


def formula_counts_vector(counts: dict[str, Any]) -> np.ndarray:
    vec = np.zeros(len(ELEMENTS) * 3, dtype=np.float32)
    total = max(1.0, float(sum(float(v) for v in counts.values())))
    denom = max(1.0, math.log1p(total))
    for el, raw in counts.items():
        idx = ELEMENT_TO_IDX.get(str(el))
        if idx is None:
            continue
        value = float(raw)
        vec[idx] = value / total
        vec[len(ELEMENTS) + idx] = math.log1p(value) / denom
        vec[len(ELEMENTS) * 2 + idx] = 1.0 if value > 0 else 0.0
    return vec


def step_feature(
    formula_counts: dict[str, Any],
    remaining_counts: dict[str, int],
    site: dict[str, Any],
    site_order: int,
    n_sites: int,
    remaining_sites: int,
) -> np.ndarray:
    formula_vec = formula_counts_vector(formula_counts)
    remaining_vec = formula_counts_vector(remaining_counts)
    letter_vec = np.zeros(len(LETTERS), dtype=np.float32)
    letter = str(site["letter"])
    if letter in LETTER_TO_IDX:
        letter_vec[LETTER_TO_IDX[letter]] = 1.0
    mult = float(site["multiplicity"])
    scalars = np.array(
        [
            math.log1p(mult) / math.log1p(192.0),
            mult / 192.0,
            site_order / max(1.0, n_sites - 1.0),
            n_sites / 64.0,
            remaining_sites / 64.0,
            sum(remaining_counts.values()) / 300.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([formula_vec, remaining_vec, letter_vec, scalars]).astype(np.float32)


def subtract_assignment(remaining: dict[str, int], element: str, multiplicity: int) -> dict[str, int]:
    out = dict(remaining)
    out[element] = int(out.get(element, 0)) - int(multiplicity)
    if out[element] == 0:
        out.pop(element)
    return out


def legal_elements(remaining_counts: dict[str, int], multiplicity: int, remaining_multiplicities: list[int]) -> list[str]:
    legal: list[str] = []
    for element, count in sorted(remaining_counts.items()):
        if int(count) < int(multiplicity):
            continue
        new_remaining = subtract_assignment(remaining_counts, element, int(multiplicity))
        if is_formula_compatible(list(remaining_multiplicities), new_remaining):
            legal.append(element)
    return legal


def parse_skeleton_key(skeleton_template_uid: str) -> list[dict[str, Any]]:
    try:
        key = skeleton_template_uid.split("|", 1)[1]
    except IndexError:
        key = skeleton_template_uid
    sites: list[dict[str, Any]] = []
    for i, part in enumerate(key.split("|")):
        digits = "".join(ch for ch in part if ch.isdigit())
        letter = part[len(digits) :]
        sites.append({"site_order": i, "multiplicity": int(digits), "letter": letter})
    return sites


def row_sites(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "site_order": int(site.get("site_order", i)),
            "multiplicity": int(site["multiplicity"]),
            "letter": str(site["letter"]),
            "element": str(site.get("element", "")),
        }
        for i, site in enumerate(row["assignment"])
    ]


def skeleton_assignment_key(sites: list[dict[str, Any]], elements: list[str]) -> str:
    return "|".join(
        f"{int(site['multiplicity'])}{site['letter']}:{element}"
        for site, element in zip(sites, elements)
    )


class StepDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], skeleton_to_idx: dict[str, int]):
        examples: list[tuple[np.ndarray, int, int, int, list[int], int]] = []
        for row in rows:
            sites = row_sites(row)
            remaining = {str(k): int(v) for k, v in row["formula_counts"].items()}
            skeleton_idx = skeleton_to_idx.get(row["skeleton_template_uid"], skeleton_to_idx[UNK_SKELETON])
            for i, site in enumerate(sites):
                mult = int(site["multiplicity"])
                remaining_mults = [int(s["multiplicity"]) for s in sites[i + 1 :]]
                legal = legal_elements(remaining, mult, remaining_mults)
                if not legal:
                    raise ValueError(f"no legal elements for {row['sample_id']} site={i}")
                label = ELEMENT_TO_IDX[site["element"]]
                feature = step_feature(
                    row["formula_counts"],
                    remaining,
                    site,
                    i,
                    len(sites),
                    len(sites) - i,
                )
                legal_ids = [ELEMENT_TO_IDX[el] for el in legal]
                examples.append((feature, int(row["sg"]), skeleton_idx, label, legal_ids, mult))
                remaining = subtract_assignment(remaining, site["element"], mult)
            if remaining:
                raise ValueError(f"row did not close formula: {row['sample_id']} {remaining}")
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature, sg, skeleton_idx, label, legal_ids, _ = self.examples[idx]
        mask = torch.zeros(len(ELEMENTS), dtype=torch.bool)
        mask[legal_ids] = True
        return (
            torch.from_numpy(feature),
            torch.tensor(sg, dtype=torch.long),
            torch.tensor(skeleton_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            mask,
        )


class CombinatorialAssignmentModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_skeletons: int,
        sg_emb_dim: int = 48,
        skeleton_emb_dim: int = 96,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sg_emb = nn.Embedding(231, sg_emb_dim)
        self.skeleton_emb = nn.Embedding(num_skeletons, skeleton_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + sg_emb_dim + skeleton_emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(ELEMENTS)),
        )

    def forward(self, features: torch.Tensor, sg: torch.Tensor, skeleton_idx: torch.Tensor) -> torch.Tensor:
        sg_emb = self.sg_emb(sg.clamp(min=0, max=230))
        skel_emb = self.skeleton_emb(skeleton_idx)
        return self.net(torch.cat([features, sg_emb, skel_emb], dim=-1))


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~legal_mask.to(logits.device), -1e9)
    return F.cross_entropy(masked, labels)


def train_epoch(
    model: CombinatorialAssignmentModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for features, sg, skeleton_idx, labels, legal_mask in loader:
        features = features.to(device)
        sg = sg.to(device)
        skeleton_idx = skeleton_idx.to(device)
        labels = labels.to(device)
        legal_mask = legal_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_ce_loss(model(features, sg, skeleton_idx), labels, legal_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(features.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs
    return total_loss / max(1, total)


@torch.no_grad()
def teacher_forced_metrics(
    model: CombinatorialAssignmentModel,
    rows: list[dict[str, Any]],
    skeleton_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    row_exact = 0
    for row in rows:
        sites = row_sites(row)
        remaining = {str(k): int(v) for k, v in row["formula_counts"].items()}
        skeleton_idx = skeleton_to_idx.get(row["skeleton_template_uid"], skeleton_to_idx[UNK_SKELETON])
        row_ok = True
        for i, site in enumerate(sites):
            mult = int(site["multiplicity"])
            remaining_mults = [int(s["multiplicity"]) for s in sites[i + 1 :]]
            legal = legal_elements(remaining, mult, remaining_mults)
            feature = step_feature(row["formula_counts"], remaining, site, i, len(sites), len(sites) - i)
            logits = model(
                torch.from_numpy(feature).unsqueeze(0).to(device),
                torch.tensor([int(row["sg"])], dtype=torch.long, device=device),
                torch.tensor([skeleton_idx], dtype=torch.long, device=device),
            )[0]
            mask = torch.zeros(len(ELEMENTS), dtype=torch.bool, device=device)
            mask[[ELEMENT_TO_IDX[el] for el in legal]] = True
            pred = int(logits.masked_fill(~mask, -1e9).argmax().item())
            target = ELEMENT_TO_IDX[str(site["element"])]
            if pred == target:
                correct += 1
            else:
                row_ok = False
            total += 1
            remaining = subtract_assignment(remaining, str(site["element"]), mult)
        if row_ok:
            row_exact += 1
    return {
        "site_accuracy": correct / max(1, total),
        "row_exact_teacher_forced": row_exact / max(1, len(rows)),
        "sites": total,
        "samples": len(rows),
    }


@torch.no_grad()
def generate_assignment_beam(
    model: CombinatorialAssignmentModel,
    *,
    formula_counts: dict[str, Any],
    sg: int,
    skeleton_uid: str,
    sites: list[dict[str, Any]],
    skeleton_to_idx: dict[str, int],
    device: torch.device,
    beam_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    initial_remaining = {str(k): int(v) for k, v in formula_counts.items()}
    if not is_formula_compatible([int(site["multiplicity"]) for site in sites], initial_remaining):
        return []
    skeleton_idx = skeleton_to_idx.get(skeleton_uid, skeleton_to_idx[UNK_SKELETON])
    beams: list[tuple[float, list[str], dict[str, int]]] = [(0.0, [], initial_remaining)]
    for i, site in enumerate(sites):
        mult = int(site["multiplicity"])
        remaining_mults = [int(s["multiplicity"]) for s in sites[i + 1 :]]
        candidates: list[tuple[float, list[str], dict[str, int]]] = []
        for score, elems, remaining in beams:
            legal = legal_elements(remaining, mult, remaining_mults)
            if not legal:
                continue
            feature = step_feature(formula_counts, remaining, site, i, len(sites), len(sites) - i)
            logits = model(
                torch.from_numpy(feature).unsqueeze(0).to(device),
                torch.tensor([int(sg)], dtype=torch.long, device=device),
                torch.tensor([skeleton_idx], dtype=torch.long, device=device),
            )[0]
            legal_ids = [ELEMENT_TO_IDX[el] for el in legal]
            masked = torch.full_like(logits, -1e9)
            masked[legal_ids] = logits[legal_ids]
            log_probs = F.log_softmax(masked, dim=-1)
            top_count = min(len(legal_ids), beam_size)
            values, indices = torch.topk(log_probs, k=top_count)
            for value, idx in zip(values.tolist(), indices.tolist()):
                element = ELEMENTS[int(idx)]
                candidates.append((score + float(value), [*elems, element], subtract_assignment(remaining, element, mult)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_size]
        if not beams:
            return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, elems, remaining in beams:
        if remaining:
            continue
        key = skeleton_assignment_key(sites, elems)
        if key in seen:
            continue
        seen.add(key)
        out.append({"assignment_key": key, "elements": elems, "logprob": score})
    return out


def evaluate_oracle_assignment(
    model: CombinatorialAssignmentModel,
    rows: list[dict[str, Any]],
    skeleton_to_idx: dict[str, int],
    device: torch.device,
    beam_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ks = (1, 5, 20)
    hits = {k: 0 for k in ks}
    closed = 0
    top1_site_correct = 0
    total_sites = 0
    preds: list[dict[str, Any]] = []
    for row in rows:
        sites = row_sites(row)
        beams = generate_assignment_beam(
            model,
            formula_counts=row["formula_counts"],
            sg=int(row["sg"]),
            skeleton_uid=row["skeleton_template_uid"],
            sites=sites,
            skeleton_to_idx=skeleton_to_idx,
            device=device,
            beam_size=beam_size,
        )
        if beams:
            closed += 1
        target = assignment_key(row)
        keys = [beam["assignment_key"] for beam in beams]
        for k in ks:
            if target in keys[:k]:
                hits[k] += 1
        if beams:
            pred_elements = beams[0]["elements"]
            for site, pred in zip(sites, pred_elements):
                top1_site_correct += int(str(site["element"]) == str(pred))
                total_sites += 1
        else:
            total_sites += len(sites)
        preds.append(
            {
                "sample_id": row["sample_id"],
                "formula": row["formula"],
                "sg": int(row["sg"]),
                "skeleton_template_uid": row["skeleton_template_uid"],
                "target_assignment_key": target,
                "top20_assignment_keys": keys[:20],
                "beam_closed": bool(beams),
            }
        )
    metrics = {
        "samples": len(rows),
        "beam_size": beam_size,
        "composition_closure_rate": closed / max(1, len(rows)),
        "top1_site_accuracy": top1_site_correct / max(1, total_sites),
    }
    for k in ks:
        metrics[f"assignment_top{k}"] = hits[k] / max(1, len(rows))
    return metrics, preds


def load_template_catalog(path: Path) -> dict[str, dict[str, Any]]:
    return {row["skeleton_template_uid"]: row for row in read_jsonl(path)}


def evaluate_joint(
    model: CombinatorialAssignmentModel,
    rows: list[dict[str, Any]],
    skeleton_predictions_path: Path,
    template_catalog: dict[str, dict[str, Any]],
    skeleton_to_idx: dict[str, int],
    device: torch.device,
    beam_size: int,
    skeleton_k: int,
) -> dict[str, Any]:
    pred_rows = [json.loads(line) for line in skeleton_predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pred_by_id = {row["sample_id"]: row for row in pred_rows}
    hits = {1: 0, 5: 0, 20: 0}
    closed_samples = 0
    total_pairs_mean = 0.0
    for row in rows:
        target_pair = f"{row['skeleton_template_uid']}|assign={assignment_key(row)}"
        skel_pred = pred_by_id.get(row["sample_id"])
        candidates: list[str] = []
        if skel_pred:
            for skeleton_uid in skel_pred["top20_template_uids"][:skeleton_k]:
                item = template_catalog.get(skeleton_uid)
                if not item:
                    continue
                sites = parse_skeleton_key(skeleton_uid)
                beams = generate_assignment_beam(
                    model,
                    formula_counts=row["formula_counts"],
                    sg=int(row["sg"]),
                    skeleton_uid=skeleton_uid,
                    sites=sites,
                    skeleton_to_idx=skeleton_to_idx,
                    device=device,
                    beam_size=beam_size,
                )
                for beam in beams:
                    candidates.append(f"{skeleton_uid}|assign={beam['assignment_key']}")
        if candidates:
            closed_samples += 1
        total_pairs_mean += len(candidates)
        for k in hits:
            if target_pair in candidates[:k]:
                hits[k] += 1
    out = {
        "samples": len(rows),
        "skeleton_k": skeleton_k,
        "assignment_beam_size": beam_size,
        "joint_candidate_nonempty_rate": closed_samples / max(1, len(rows)),
        "joint_candidates_mean": total_pairs_mean / max(1, len(rows)),
    }
    for k, value in hits.items():
        out[f"joint_pair_top{k}"] = value / max(1, len(rows))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def nsites_breakdown(
    rows: list[dict[str, Any]],
    preds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_id = {row["sample_id"]: row for row in preds}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["n_sites"])].append(row)
    out: list[dict[str, Any]] = []
    for nsites, group in sorted(groups.items()):
        item: dict[str, Any] = {"n_sites": nsites, "samples": len(group)}
        for k in (1, 5, 20):
            hit = 0
            closed = 0
            for row in group:
                pred = pred_by_id[row["sample_id"]]
                closed += int(bool(pred["beam_closed"]))
                hit += int(assignment_key(row) in pred["top20_assignment_keys"][:k])
            item[f"top{k}"] = hit / max(1, len(group))
            item["closure_rate"] = closed / max(1, len(group))
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train combinatorial site-level assignment generator.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--skeleton-predictions", type=Path, default=PROJECT_ROOT / "reports" / "skeleton_template_ranker_v1" / "test_predictions_mlp_sg_formula.jsonl")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "assignment_combinatorial_v1")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports" / "assignment_combinatorial_v1")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--beam-size", type=int, default=20)
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
    template_catalog = load_template_catalog(args.data_dir / "template_catalog.jsonl")
    skeleton_uids = sorted({row["skeleton_template_uid"] for row in train_rows})
    skeleton_to_idx = {uid: i for i, uid in enumerate(skeleton_uids)}
    skeleton_to_idx[UNK_SKELETON] = len(skeleton_to_idx)

    sample_site = row_sites(train_rows[0])[0]
    feature_dim = int(
        step_feature(
            train_rows[0]["formula_counts"],
            {str(k): int(v) for k, v in train_rows[0]["formula_counts"].items()},
            sample_site,
            0,
            int(train_rows[0]["n_sites"]),
            int(train_rows[0]["n_sites"]),
        ).shape[0]
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = StepDataset(train_rows, skeleton_to_idx)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = CombinatorialAssignmentModel(feature_dim=feature_dim, num_skeletons=len(skeleton_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            best_state = {"epoch": epoch, "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_tf = teacher_forced_metrics(model, val_rows, skeleton_to_idx, device)
            row = {"epoch": epoch, "loss": loss, **val_tf}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    val_tf = teacher_forced_metrics(model, val_rows, skeleton_to_idx, device)
    test_tf = teacher_forced_metrics(model, test_rows, skeleton_to_idx, device)
    val_oracle, val_preds = evaluate_oracle_assignment(model, val_rows, skeleton_to_idx, device, args.beam_size)
    test_oracle, test_preds = evaluate_oracle_assignment(model, test_rows, skeleton_to_idx, device, args.beam_size)
    joint = evaluate_joint(
        model,
        test_rows,
        args.skeleton_predictions,
        template_catalog,
        skeleton_to_idx,
        device,
        args.beam_size,
        skeleton_k=5,
    )
    summary = {
        "train": {
            "samples": len(train_rows),
            "step_examples": len(train_ds),
            "train_skeletons": len(skeleton_uids),
            "device": str(device),
            "best_epoch": None if best_state is None else int(best_state["epoch"]),
            "best_loss": best_loss,
        },
        "validation": {
            "teacher_forced": val_tf,
            "oracle_skeleton_beam": val_oracle,
        },
        "test": {
            "teacher_forced": test_tf,
            "oracle_skeleton_beam": test_oracle,
            "joint_skeleton5_assignment": joint,
        },
    }
    write_jsonl(args.reports_dir / "test_predictions_oracle_skeleton_beam.jsonl", test_preds)
    write_jsonl(args.reports_dir / "val_predictions_oracle_skeleton_beam.jsonl", val_preds)
    write_csv(args.reports_dir / "per_nsites_breakdown.csv", nsites_breakdown(test_rows, test_preds))
    (args.reports_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.reports_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "skeleton_to_idx": skeleton_to_idx,
            "element_order": ELEMENTS,
            "letters": LETTERS,
            "args": vars(args),
            "summary": summary,
        },
        args.out_dir / "ckpt.pt",
    )
    print(json.dumps(summary["test"], indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
