#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_positive_rowseq_geometry as pg  # noqa: E402
import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402
import opentry_train_geometry_compat_selector as compat  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return pg.read_jsonl(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    pg.write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    pg.write_jsonl(path, rows)


def lattice_target_from_lattice(lattice: dict[str, Any]) -> list[float]:
    return [
        math.log(max(0.1, float(lattice.get("a", lattice.get("length_a", 1.0))))),
        math.log(max(0.1, float(lattice.get("b", lattice.get("length_b", 1.0))))),
        math.log(max(0.1, float(lattice.get("c", lattice.get("length_c", 1.0))))),
        float(lattice.get("alpha", lattice.get("angle_alpha", 90.0))) / pg.ANGLE_SCALE,
        float(lattice.get("beta", lattice.get("angle_beta", 90.0))) / pg.ANGLE_SCALE,
        float(lattice.get("gamma", lattice.get("angle_gamma", 90.0))) / pg.ANGLE_SCALE,
    ]


def coord_arrays_from_params(rows: list[dict[str, Any]], params: dict[str, Any] | dict[int, Any]) -> tuple[list[list[float]], list[list[float]]]:
    values: list[list[float]] = []
    masks: list[list[float]] = []
    for idx, row in enumerate(rows):
        raw = params.get(idx, params.get(str(idx), {})) if isinstance(params, dict) else {}
        symbols = {str(sym) for sym in row.get("free_symbols") or []}
        values.append([float((raw or {}).get(sym, 0.0)) % 1.0 for sym in ("x", "y", "z")])
        masks.append([1.0 if sym in symbols and sym in (raw or {}) else 0.0 for sym in ("x", "y", "z")])
    return values, masks


def example_from_feature(
    feature: dict[str, Any],
    structured_by_id: dict[str, dict[str, Any]],
    engine: OrbitEngine,
    *,
    max_rows: int,
) -> dict[str, Any] | None:
    base = structured_by_id.get(str(feature.get("sample_id") or ""))
    if base is None:
        return None
    try:
        rows = pg.rows_from_wa_key(engine, str(feature.get("canonical_wa_key") or ""))
        if len(rows) > int(max_rows):
            return None
        record = pg.make_pseudo_record(base, rows)
        if isinstance(feature.get("geometry_lattice"), dict):
            lattice_target = lattice_target_from_lattice(dict(feature["geometry_lattice"]))
        else:
            cell, _coords = pg.parse_rendered_cif_targets(str(feature.get("cif") or ""))
            lattice_target = pg.lattice_target_from_cell(cell)
        if isinstance(feature.get("geometry_params"), dict):
            coord_values, coord_mask = coord_arrays_from_params(rows, dict(feature["geometry_params"]))
        else:
            _cell, coords = pg.parse_rendered_cif_targets(str(feature.get("cif") or ""))
            coord_values = []
            coord_mask = []
            for row_idx, row in enumerate(rows):
                raw = coords.get(row_idx, {})
                symbols = {str(sym) for sym in row.get("free_symbols") or []}
                coord_values.append([float(raw.get(sym, 0.0)) % 1.0 for sym in ("x", "y", "z")])
                coord_mask.append([1.0 if sym in symbols and sym in raw else 0.0 for sym in ("x", "y", "z")])
    except Exception:
        return None
    label = int(bool(feature.get("label_match")))
    row_count = len(rows)
    atom_count = int(record["atom_count"])
    return {
        "sample_id": str(record["sample_id"]),
        "record": record,
        "lattice_target": lattice_target,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "label": label,
        "feature": feature,
        "sample_weight": (float(label) * 7.0 + 1.0) * (4.0 if row_count >= 7 else 1.0) * (1.5 if atom_count >= 12 else 1.0),
    }


def build_labeled_examples(
    *,
    features_path: Path,
    structured_by_id: dict[str, dict[str, Any]],
    engine: OrbitEngine,
    max_examples: int,
    min_target_row_count: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    stats = {
        "feature_rows": 0,
        "used_examples": 0,
        "positive_examples": 0,
        "rows_ge_7_examples": 0,
        "rows_ge_7_positive_examples": 0,
        "skipped_target_row_count": 0,
        "skipped_parse": 0,
    }
    for feature in read_jsonl(features_path):
        stats["feature_rows"] += 1
        if int(feature.get("target_row_count", 0) or 0) < int(min_target_row_count):
            stats["skipped_target_row_count"] += 1
            continue
        ex = example_from_feature(feature, structured_by_id, engine, max_rows=int(max_rows))
        if ex is None:
            stats["skipped_parse"] += 1
            continue
        row_count = len(ex["record"]["wa_table"])
        label = int(ex["label"])
        stats["used_examples"] += 1
        stats["positive_examples"] += label
        stats["rows_ge_7_examples"] += int(row_count >= 7)
        stats["rows_ge_7_positive_examples"] += int(row_count >= 7 and label)
        examples.append(ex)
        if int(max_examples) > 0 and len(examples) >= int(max_examples):
            break
    return examples, stats


class EnergyDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_energy(
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    max_rows: int,
) -> dict[str, torch.Tensor]:
    batch = pg.collate_positive(examples, vocabs=vocabs, lattice_mean=lattice_mean, lattice_std=lattice_std, max_rows_limit=max_rows)
    batch["label"] = torch.tensor([float(ex.get("label", 0.0)) for ex in examples], dtype=torch.float32)
    return batch


class RowSeqGeometryEnergyNet(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], hidden_dim: int = 160, emb_dim: int = 40, layers: int = 2, heads: int = 4, max_rows: int = 96) -> None:
        super().__init__()
        self.max_rows = int(max_rows)
        self.element_emb = nn.Embedding(vocab_sizes["element"] + 1, emb_dim)
        self.sg_emb = nn.Embedding(vocab_sizes["sg"] + 1, emb_dim)
        self.orbit_emb = nn.Embedding(vocab_sizes["orbit"] + 1, emb_dim)
        self.site_sym_emb = nn.Embedding(vocab_sizes["site_sym"] + 1, emb_dim // 2)
        self.letter_emb = nn.Embedding(vocab_sizes["letter"] + 1, emb_dim // 2)
        row_in = emb_dim * 2 + emb_dim // 2 + emb_dim // 2 + 6 + 3 + 3
        ctx_in = emb_dim * 2 + 5 + 6
        self.row_proj = nn.Sequential(nn.LayerNorm(row_in), nn.Linear(row_in, hidden_dim), nn.GELU())
        self.ctx_proj = nn.Sequential(nn.LayerNorm(ctx_in), nn.Linear(ctx_in, hidden_dim), nn.GELU())
        self.pos_emb = nn.Embedding(max_rows + 1, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        formula_emb = self.element_emb(batch["formula_element_ids"])
        formula_ctx = (formula_emb * batch["formula_weights"].unsqueeze(-1)).sum(dim=1)
        sg_ctx = self.sg_emb(batch["sg_ids"])
        ctx = self.ctx_proj(torch.cat([formula_ctx, sg_ctx, batch["numeric"], batch["lattice_values"]], dim=-1))
        row_input = torch.cat(
            [
                self.element_emb(batch["row_element_ids"]),
                self.orbit_emb(batch["row_orbit_ids"]),
                self.site_sym_emb(batch["row_site_sym_ids"]),
                self.letter_emb(batch["row_letter_ids"]),
                batch["row_numeric"],
                batch["coord_values"],
                batch["coord_mask"],
            ],
            dim=-1,
        )
        row_h = self.row_proj(row_input) + ctx.unsqueeze(1)
        positions = torch.arange(row_h.shape[1], device=row_h.device).clamp_max(self.max_rows)
        row_h = row_h + self.pos_emb(positions).unsqueeze(0)
        cls = self.cls.expand(row_h.shape[0], -1, -1) + ctx.unsqueeze(1)
        tokens = torch.cat([cls, row_h], dim=1)
        padding = torch.cat([torch.zeros((row_h.shape[0], 1), dtype=torch.bool, device=row_h.device), batch["row_mask"] <= 0.0], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return self.head(encoded[:, 0]).squeeze(-1)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def score_examples(
    model: RowSeqGeometryEnergyNet,
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_rows: int,
) -> list[float]:
    loader = DataLoader(
        EnergyDataset(examples),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda xs: collate_energy(xs, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu(), max_rows=int(max_rows)),
        num_workers=0,
    )
    scores: list[float] = []
    model.eval()
    for raw in loader:
        batch = move_batch(raw, device)
        logits = model(batch)
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return [float(x) for x in scores]


def ranked_metrics_from_examples(
    model: RowSeqGeometryEnergyNet,
    examples: list[dict[str, Any]],
    *,
    vocabs: dict[str, dict[str, int]],
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_rows: int,
    top_k: int,
) -> dict[str, Any]:
    scores = score_examples(model, examples, vocabs=vocabs, lattice_mean=lattice_mean, lattice_std=lattice_std, device=device, batch_size=batch_size, max_rows=max_rows)
    scored_rows: list[dict[str, Any]] = []
    for ex, score in zip(examples, scores):
        row = dict(ex["feature"])
        row["rowseq_energy_score"] = float(score)
        row["compat_score"] = float(score)
        scored_rows.append(row)
    ranked = []
    for _sample_id, items in sorted(compat.grouped(scored_rows).items()):
        chosen = sorted(items, key=lambda row: (-float(row.get("rowseq_energy_score", 0.0)), int(row.get("rank", 10**9))))[: int(top_k)]
        for rank, row in enumerate(chosen, start=1):
            item = dict(row)
            item["rank"] = rank
            ranked.append(item)
    baseline = compat.baseline_by_current_rank([dict(ex["feature"]) for ex in examples], top_k=int(top_k))
    return {
        "scored_rows": scored_rows,
        "ranked_rows": ranked,
        "baseline": {key: value for key, value in compat.metrics_from_ranked(baseline).items() if key != "per_sample"},
        "energy_ranked": {key: value for key, value in compat.metrics_from_ranked(ranked).items() if key != "per_sample"},
    }


def train_energy(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_by_id = pg.load_structured_by_id(args.data_root, "train")
    val_by_id = pg.load_structured_by_id(args.data_root, "val")
    engine = OrbitEngine(args.lookup_json)
    train_examples, train_stats = build_labeled_examples(
        features_path=args.train_features,
        structured_by_id=train_by_id,
        engine=engine,
        max_examples=int(args.max_train_examples),
        min_target_row_count=int(args.min_train_target_row_count),
        max_rows=int(args.max_rows),
    )
    val_examples, val_stats = build_labeled_examples(
        features_path=args.val_features,
        structured_by_id=val_by_id,
        engine=engine,
        max_examples=int(args.max_val_examples),
        min_target_row_count=int(args.min_val_target_row_count),
        max_rows=int(args.max_rows),
    )
    if not train_examples or not val_examples:
        raise SystemExit("No train/val labeled examples.")
    vocabs = pg.build_vocabs(train_examples)
    lattice_mean, lattice_std = pg.lattice_stats(train_examples)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RowSeqGeometryEnergyNet(
        {key: len(value) for key, value in vocabs.items()},
        hidden_dim=int(args.hidden_dim),
        emb_dim=int(args.emb_dim),
        layers=int(args.layers),
        heads=int(args.heads),
        max_rows=int(args.max_rows),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    weights = torch.tensor([float(ex.get("sample_weight", 1.0)) for ex in train_examples], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_examples), replacement=True)
    loader = DataLoader(
        EnergyDataset(train_examples),
        batch_size=int(args.batch_size),
        sampler=sampler,
        collate_fn=lambda xs: collate_energy(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t, max_rows=int(args.max_rows)),
        num_workers=0,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_score = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    print(json.dumps({"train_examples": len(train_examples), "val_examples": len(val_examples), "device": str(device)}, sort_keys=True), flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        total_w = 0.0
        for raw in loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            base_loss = loss_fn(logits, batch["label"])
            weights_t = batch["sample_weight"].clamp_min(0.05)
            loss = (base_loss * weights_t).sum() / weights_t.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * float(weights_t.sum().detach().cpu())
            total_w += float(weights_t.sum().detach().cpu())
        metrics = ranked_metrics_from_examples(
            model,
            val_examples,
            vocabs=vocabs,
            lattice_mean=mean_t,
            lattice_std=std_t,
            device=device,
            batch_size=int(args.batch_size),
            max_rows=int(args.max_rows),
            top_k=int(args.top_k),
        )
        rows7 = metrics["energy_ranked"]["rows_ge_7"]
        score = float(rows7.get("match@20", 0.0)) + 0.5 * float(rows7.get("match@5", 0.0))
        row = {
            "epoch": epoch,
            "train_loss": total / max(1.0, total_w),
            "val_rows_ge_7_match_at_5": rows7.get("match@5"),
            "val_rows_ge_7_match_at_20": rows7.get("match@20"),
            "val_full_match_at_5": metrics["energy_ranked"]["full"].get("match@5"),
            "val_full_match_at_20": metrics["energy_ranked"]["full"].get("match@20"),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocabs": vocabs,
                    "lattice_mean": lattice_mean,
                    "lattice_std": lattice_std,
                    "config": vars(args),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                out_dir / "energy_best.pt",
            )
            write_jsonl(out_dir / "val_scored_candidates_best.jsonl", metrics["scored_rows"])
            write_jsonl(out_dir / "val_reranked_features_best.jsonl", metrics["ranked_rows"])
            best_metrics = {k: v for k, v in metrics.items() if k not in {"scored_rows", "ranked_rows"}}
            write_json(out_dir / "val_energy_ranked_metrics_best.json", best_metrics)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": vars(args),
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        out_dir / "energy_last.pt",
    )
    write_json(
        out_dir / "energy_training_summary.json",
        {
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "history": history,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "note": "Model inputs use predicted W/A geometry metadata and rendered-candidate geometry only; label_match/rmsd and target keys are labels/diagnostics only. No test labels are read.",
        },
    )
    return 0


def load_energy(path: Path, device: torch.device) -> tuple[RowSeqGeometryEnergyNet, dict[str, dict[str, int]], torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    vocabs = ckpt["vocabs"]
    config = ckpt.get("config") or {}
    model = RowSeqGeometryEnergyNet(
        {key: len(value) for key, value in vocabs.items()},
        hidden_dim=int(config.get("hidden_dim", 160)),
        emb_dim=int(config.get("emb_dim", 40)),
        layers=int(config.get("layers", 2)),
        heads=int(config.get("heads", 4)),
        max_rows=int(config.get("max_rows", 96)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocabs, torch.tensor(ckpt["lattice_mean"], dtype=torch.float32, device=device), torch.tensor(ckpt["lattice_std"], dtype=torch.float32, device=device), config


def example_from_proposal(record: dict[str, Any], rows: list[dict[str, Any]], lattice: dict[str, Any], params: dict[int, dict[str, float]]) -> dict[str, Any]:
    coord_values, coord_mask = coord_arrays_from_params(rows, params)
    return {
        "sample_id": str(record["sample_id"]),
        "record": record,
        "lattice_target": lattice_target_from_lattice(lattice),
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "label": 0,
        "sample_weight": 1.0,
        "feature": {"sample_id": str(record["sample_id"])},
    }


def sample_render(args: argparse.Namespace) -> int:
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    energy_model, energy_vocabs, energy_mean, energy_std, energy_config = load_energy(args.energy_ckpt, device)
    gen_model, gen_vocabs, gen_mean, gen_std, gen_config = pg.load_ckpt(args.geometry_ckpt, device)
    structured_by_id = pg.load_structured_by_id(args.data_root, args.split)
    engine = OrbitEngine(args.lookup_json)
    groups = pg.grouped_rendered(read_jsonl(args.input_rendered_jsonl))
    sample_ids = list(groups)
    if int(args.max_records) > 0:
        sample_ids = sample_ids[: int(args.max_records)]
    max_rows = min(int(energy_config.get("max_rows", args.max_rows)), int(gen_config.get("max_rows", args.max_rows)))
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "energy_ckpt": str(args.energy_ckpt),
        "geometry_ckpt": str(args.geometry_ckpt),
        "input_rendered_jsonl": str(args.input_rendered_jsonl),
        "selected_samples": len(sample_ids),
        "min_target_row_count": int(args.min_target_row_count),
        "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
        "proposals_per_wa": int(args.proposals_per_wa),
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "proposal_count": 0,
        "render_attempts": 0,
        "render_ok": 0,
        "samples_with_output": 0,
        "skipped_target_rows_lt_min": 0,
        "skipped_missing_record": 0,
        "skipped_too_many_rows": 0,
        "deduped_wa": 0,
        "note": "Inference uses predicted W/A plus train-label energy scoring of sampled geometry. It does not use val/test StructureMatcher labels during generation.",
    }
    for sample_id in sample_ids:
        base = structured_by_id.get(sample_id)
        if base is None:
            summary["skipped_missing_record"] += 1
            continue
        target_row_count = int(base.get("n_sites", len(base.get("wa_table") or [])))
        if int(args.min_target_row_count) > 0 and target_row_count < int(args.min_target_row_count):
            summary["skipped_target_rows_lt_min"] += 1
            continue
        proposals: list[dict[str, Any]] = []
        seen_wa: set[str] = set()
        for cand in groups.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]:
            wa_key = str(cand.get("canonical_wa_key") or "")
            if not wa_key or wa_key in seen_wa:
                summary["deduped_wa"] += int(bool(wa_key))
                continue
            seen_wa.add(wa_key)
            try:
                rows = pg.rows_from_wa_key(engine, wa_key)
                if len(rows) > max_rows:
                    summary["skipped_too_many_rows"] += 1
                    continue
                record = pg.make_pseudo_record(base, rows)
            except Exception:
                continue
            for variant in range(int(args.proposals_per_wa)):
                try:
                    params, lattice = pg.predict_one(
                        gen_model,
                        gen_vocabs,
                        gen_mean,
                        gen_std,
                        record,
                        device=device,
                        max_rows=max_rows,
                        variant=variant,
                        temperature=float(args.temperature),
                        seed_text=f"{sample_id}|{wa_key}|energy|{variant}|{args.seed}",
                    )
                    proposals.append(
                        {
                            "candidate": cand,
                            "record": record,
                            "rows": rows,
                            "params": params,
                            "lattice": lattice,
                            "proposal_variant": int(variant),
                            "original_rank": cand.get("rank"),
                        }
                    )
                except Exception:
                    continue
        summary["proposal_count"] += len(proposals)
        if not proposals:
            continue
        energy_examples = [example_from_proposal(prop["record"], prop["rows"], prop["lattice"], prop["params"]) for prop in proposals]
        scores = score_examples(
            energy_model,
            energy_examples,
            vocabs=energy_vocabs,
            lattice_mean=energy_mean,
            lattice_std=energy_std,
            device=device,
            batch_size=int(args.score_batch_size),
            max_rows=max_rows,
        )
        for prop, score in zip(proposals, scores):
            prop["energy_score"] = float(score)
        selected = sorted(proposals, key=lambda item: (-float(item["energy_score"]), int(item.get("original_rank") or 10**9), int(item["proposal_variant"])))[: int(args.top_k)]
        sample_rows: list[dict[str, Any]] = []
        for prop in selected:
            summary["render_attempts"] += 1
            try:
                rendered = v2.render_candidate(engine, prop["record"], {"rows": prop["rows"], "params": prop["params"], "lattice": prop["lattice"]}, len(sample_rows) + 1, f"energy_v{prop['proposal_variant']}")
                cif = str(rendered.get("cif") or "")
                metric = validate_cif(cif, prop["record"]["formula_counts"], int(prop["record"]["sg"])) if cif else {
                    "readable": False,
                    "formula_ok": False,
                    "sg_ok": False,
                    "atom_count_ok": False,
                    "composition_exact": False,
                    "atom_count_after_expansion": None,
                    "detected_sg": None,
                    "error": rendered.get("error") or "empty_cif",
                }
                metric.update(selfscore.cif_self_features(cif))
                skel, wa = v2.canonical_keys_from_rows(prop["rows"])
                row = {
                    "sample_id": sample_id,
                    "rank": len(sample_rows) + 1,
                    "original_rank": prop.get("original_rank"),
                    "geometry_mode": "contrastive_rowseq_energy",
                    "geometry_variant": int(prop["proposal_variant"]),
                    "geometry_source": "train_label_energy_over_rowseq_samples",
                    "geometry_lattice_mode": "rowseq_positive_nll_energy_filtered",
                    "rowseq_energy_score": float(prop["energy_score"]),
                    "canonical_skeleton_key": skel,
                    "canonical_wa_key": wa,
                    "cif": cif,
                    **metric,
                }
                sample_rows.append(row)
                summary["render_ok"] += int(bool(cif))
            except Exception as exc:  # noqa: BLE001
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "rank": len(sample_rows) + 1,
                        "original_rank": prop.get("original_rank"),
                        "geometry_mode": "contrastive_rowseq_energy",
                        "geometry_variant": int(prop["proposal_variant"]),
                        "geometry_source": "train_label_energy_over_rowseq_samples",
                        "canonical_skeleton_key": prop["candidate"].get("canonical_skeleton_key"),
                        "canonical_wa_key": prop["candidate"].get("canonical_wa_key"),
                        "cif": "",
                        "readable": False,
                        "formula_ok": False,
                        "sg_ok": False,
                        "atom_count_ok": False,
                        "composition_exact": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if sample_rows:
            summary["samples_with_output"] += 1
        out_rows.extend(sample_rows)
    summary["rendered_rows"] = len(out_rows)
    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    write_json(out_dir / "contrastive_rowseq_energy_render_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Negative-aware row-sequence geometry energy sampler.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-energy")
    p_train.add_argument("--train-features", type=Path, required=True)
    p_train.add_argument("--val-features", type=Path, required=True)
    p_train.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    p_train.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    p_train.add_argument("--out-dir", type=Path, required=True)
    p_train.add_argument("--max-train-examples", type=int, default=0)
    p_train.add_argument("--max-val-examples", type=int, default=0)
    p_train.add_argument("--min-train-target-row-count", type=int, default=7)
    p_train.add_argument("--min-val-target-row-count", type=int, default=7)
    p_train.add_argument("--max-rows", type=int, default=96)
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--batch-size", type=int, default=96)
    p_train.add_argument("--hidden-dim", type=int, default=160)
    p_train.add_argument("--emb-dim", type=int, default=40)
    p_train.add_argument("--layers", type=int, default=2)
    p_train.add_argument("--heads", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=7e-4)
    p_train.add_argument("--weight-decay", type=float, default=2e-4)
    p_train.add_argument("--top-k", type=int, default=50)
    p_train.add_argument("--seed", type=int, default=20260615)
    p_train.add_argument("--device", default="cpu")

    p_render = sub.add_parser("sample-render")
    p_render.add_argument("--energy-ckpt", type=Path, required=True)
    p_render.add_argument("--geometry-ckpt", type=Path, required=True)
    p_render.add_argument("--input-rendered-jsonl", type=Path, required=True)
    p_render.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    p_render.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    p_render.add_argument("--split", default="val")
    p_render.add_argument("--out-dir", type=Path, required=True)
    p_render.add_argument("--max-records", type=int, default=64)
    p_render.add_argument("--min-target-row-count", type=int, default=7)
    p_render.add_argument("--max-input-candidates-per-sample", type=int, default=8)
    p_render.add_argument("--proposals-per-wa", type=int, default=6)
    p_render.add_argument("--top-k", type=int, default=50)
    p_render.add_argument("--temperature", type=float, default=0.45)
    p_render.add_argument("--score-batch-size", type=int, default=256)
    p_render.add_argument("--max-rows", type=int, default=96)
    p_render.add_argument("--seed", type=int, default=20260615)
    p_render.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "train-energy":
        return train_energy(args)
    if args.command == "sample-render":
        return sample_render(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
