#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
OP5 = WORKSPACE / "model/New_model/opentry_5"
OP4 = WORKSPACE / "model/New_model/opentry_4"
OP3 = WORKSPACE / "model/New_model/opentry_3"
SYMCIF_PROJECT = WORKSPACE / "model/New_model/symcif_experiment"
SYMCIF_SCRIPTS = SYMCIF_PROJECT / "scripts"
for path in (SYMCIF_SCRIPTS, SYMCIF_PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from run_mp20_minicfjoint_v2 import (  # noqa: E402
    ANGLE_SCALE,
    COORD_ORDER,
    OrbitEngine,
    canonical_keys_from_rows,
    canonical_rows,
    lattice_from_target,
    lattice_mask,
    lattice_stats,
    lattice_target,
    render_candidate,
)


SEEDS = [
    9100,
    9101,
    9102,
    9103,
    9104,
    9110,
    9111,
    9112,
    9113,
    9114,
    9120,
    9121,
    9122,
    9123,
    9124,
    9130,
    9131,
    9132,
    9133,
    9134,
]
DIRECT_MODE = "baseline_opentry6_direct_cif"
LOG_PATH = ROOT / "opentry_6_experiment_log.md"
SUMMARY_PATH = ROOT / "opentry_6_final_summary.md"


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_6: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= int(limit):
                break
    return out


def append_md(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def overwrite_md(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def is_complex(record: dict[str, Any]) -> bool:
    return int(record.get("n_sites", len(record.get("wa_table") or []))) >= 7


def row_bin(record: dict[str, Any]) -> str:
    n = int(record.get("n_sites", len(record.get("wa_table") or [])))
    if n < 7:
        return "<7"
    if n <= 9:
        return "7-9"
    if n <= 14:
        return "10-14"
    return "15+"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def load_clean_fold(fold: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    base = OP5 / "checkpoints" / f"minicfjoint_v2_{fold}_gate0_smoke" / "clean_data"
    train = read_jsonl(base / "clean_train.jsonl")
    val = read_jsonl(base / "clean_val.jsonl")
    if not train or not val:
        raise RuntimeError(f"missing clean data under {base}")
    return train, val, base


def load_all_data() -> dict[str, Any]:
    train_a, val_a, base_a = load_clean_fold("fold_a")
    train_b, val_b, base_b = load_clean_fold("fold_b")
    train_by_id = {str(r["sample_id"]): r for r in train_a}
    # The fold clean_train files are expected to match; keep union defensively.
    for r in train_b:
        train_by_id.setdefault(str(r["sample_id"]), r)
    return {
        "train": list(train_by_id.values()),
        "fold_a": val_a,
        "fold_b": val_b,
        "fold_a_base": base_a,
        "fold_b_base": base_b,
    }


def sg_symbols_from_records(records: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for r in records:
        if r.get("sg_symbol"):
            out[int(r["sg"])] = str(r["sg_symbol"])
    return out


def lookup_path() -> Path:
    full = SYMCIF_PROJECT / "artifacts" / "wyckoff_lookup_full.json"
    return full if full.exists() else SYMCIF_PROJECT / "artifacts" / "wyckoff_lookup.json"


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("multiplicity", 1)),
        str(row.get("letter")),
        int(row.get("enumeration", 0) or 0),
        str(row.get("site_symmetry")),
        str(row.get("element")),
        str(row.get("orbit_id")),
    )


def make_row_from_orbit(engine: OrbitEngine, orbit_id: str, element: str) -> dict[str, Any]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    return {
        "element": str(element),
        "orbit_id": str(orbit.canonical_orbit_id),
        "multiplicity": int(orbit.multiplicity),
        "letter": str(orbit.letter),
        "enumeration": int(orbit.enumeration),
        "site_symmetry": str(orbit.site_symmetry),
        "free_symbols": list(orbit.free_symbols),
        "free_params": {},
    }


def split_canonical_key(key: str) -> list[str]:
    key = str(key or "")
    if not key:
        return []
    if key.startswith("setting=crystalformer"):
        parts = re.split(r"\|(?=setting=crystalformer\|sg=)", key)
        return [p for p in parts if p]
    return [p for p in key.split("|") if p]


def rows_from_canonical_wa_key(key: str, engine: OrbitEngine) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part in split_canonical_key(key):
        if ":" not in part:
            continue
        orbit_id, element = part.rsplit(":", 1)
        rows.append(make_row_from_orbit(engine, orbit_id, element))
    rows.sort(key=row_sort_key)
    return rows


def rows_from_exact_candidate(candidate: dict[str, Any], engine: OrbitEngine) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in candidate.get("rows") or []:
        canonical_key = str(row.get("canonical_key") or "")
        if canonical_key.startswith("sg="):
            orbit_id = "setting=crystalformer|" + canonical_key
        else:
            orbit_id = str(row.get("orbit_id") or canonical_key)
        try:
            out.append(make_row_from_orbit(engine, orbit_id, str(row["element"])))
        except Exception:
            continue
    out.sort(key=row_sort_key)
    return out


def condition_rows(record: dict[str, Any], condition: str, engine: OrbitEngine, aux: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition == "gt":
        rows = [copy.deepcopy(r) for r in canonical_rows(record)]
        return rows, {"condition_source": "GT canonical rows", "skeleton_exact": True, "wa_exact": True}
    sample_id = str(record["sample_id"])
    if condition == "oof":
        pred = aux.get("oof_by_sample", {}).get(sample_id)
        if pred:
            try:
                rows = rows_from_canonical_wa_key(str(pred.get("predicted_wa_key") or ""), engine)
                if rows:
                    return rows, {
                        "condition_source": "opentry_5 OOF prototype_frequency_train_core_no_candidate_ranking",
                        "skeleton_exact": bool(pred.get("skeleton_exact")),
                        "wa_exact": bool(pred.get("wa_exact")),
                    }
            except Exception as exc:
                return [], {"condition_source": f"OOF parse failed: {type(exc).__name__}: {exc}", "skeleton_exact": False, "wa_exact": False}
        return [], {"condition_source": "OOF missing", "skeleton_exact": False, "wa_exact": False}
    if condition == "exact":
        exact = aux.get("exact_by_sample", {}).get(sample_id)
        if exact:
            rows = rows_from_exact_candidate(exact, engine)
            if rows:
                target_skel, target_wa = canonical_keys_from_rows(canonical_rows(record))
                skel, wa = canonical_keys_from_rows(rows)
                return rows, {
                    "condition_source": "composition_exact_v1 fixed file order candidate0",
                    "skeleton_exact": skel == target_skel,
                    "wa_exact": wa == target_wa,
                }
        pred = aux.get("oof_by_sample", {}).get(sample_id)
        if pred:
            rows = rows_from_canonical_wa_key(str(pred.get("predicted_wa_key") or ""), engine)
            return rows, {
                "condition_source": "exact-cover unavailable for sample; OOF fallback",
                "skeleton_exact": bool(pred.get("skeleton_exact")),
                "wa_exact": bool(pred.get("wa_exact")),
                "exact_fallback_to_oof": True,
            }
        return [], {"condition_source": "exact-cover unavailable and OOF missing", "skeleton_exact": False, "wa_exact": False, "exact_fallback_to_oof": True}
    raise ValueError(f"unknown condition {condition}")


def load_oof_predictions() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in (OP5 / "data/oof_wa_predictions_dev.jsonl", OP5 / "data/oof_wa_predictions_train.jsonl"):
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out[str(row["sample_id"])] = row
    return out


def load_exact_candidates() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    exact_by_sample: dict[str, dict[str, Any]] = {}
    paths = [
        SYMCIF_PROJECT / "reports/composition_exact_v1/val_wa_candidates.jsonl",
        SYMCIF_PROJECT / "reports/composition_exact_v1/train_wa_candidates.jsonl",
        SYMCIF_PROJECT / "reports/composition_exact_v1_trimmed/val_wa_candidates.jsonl",
        SYMCIF_PROJECT / "reports/composition_exact_v1_trimmed/train_wa_candidates.jsonl",
    ]
    stats = {"files_read": [], "samples_with_candidate": 0, "matched_opentry5_samples": 0}
    for path in paths:
        if not path.exists():
            continue
        stats["files_read"].append(str(path))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                candidates = row.get("wa_candidates") or []
                if candidates:
                    exact_by_sample.setdefault(str(row.get("sample_id")), candidates[0])
    stats["samples_with_candidate"] = len(exact_by_sample)
    return exact_by_sample, stats


def hard_negative_stats() -> dict[str, Any]:
    path = OP4 / "cache/hard_negative_dataset_train.jsonl"
    stats: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "total": 0,
        "wa_hit_match_fail": 0,
        "rows_ge7_wa_hit_match_fail": 0,
        "usable_parameter_payload": False,
    }
    if not path.exists():
        return stats
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["total"] += 1
            is_fail = bool(row.get("candidate_wa_hit")) and not bool(row.get("label_match"))
            if is_fail:
                stats["wa_hit_match_fail"] += 1
                if bool(row.get("rows_ge_7")):
                    stats["rows_ge7_wa_hit_match_fail"] += 1
            if row.get("free_params") or row.get("lattice") or row.get("generated_text"):
                stats["usable_parameter_payload"] = True
    return stats


class Vocab:
    def __init__(self, records: list[dict[str, Any]], engine: OrbitEngine, aux_rows: list[list[dict[str, Any]]] | None = None) -> None:
        elements = {"<UNK>"}
        orbit_ids = {"<UNK_ORBIT>"}
        letters = {"<UNK_LETTER>"}
        syms = {"<UNK_SYM>"}
        for record in records:
            for element in record.get("formula_counts") or {}:
                elements.add(str(element))
            for row in canonical_rows(record):
                elements.add(str(row.get("element")))
                orbit_ids.add(str(row.get("orbit_id")))
                letters.add(str(row.get("letter")))
                syms.add(str(row.get("site_symmetry")))
        for rows in aux_rows or []:
            for row in rows:
                elements.add(str(row.get("element")))
                orbit_ids.add(str(row.get("orbit_id")))
                letters.add(str(row.get("letter")))
                syms.add(str(row.get("site_symmetry")))
        self.element_to_id = {v: i for i, v in enumerate(sorted(elements))}
        self.id_to_element = {i: v for v, i in self.element_to_id.items()}
        self.orbit_to_id = {v: i for i, v in enumerate(sorted(orbit_ids))}
        self.letter_to_id = {v: i for i, v in enumerate(sorted(letters))}
        self.sym_to_id = {v: i for i, v in enumerate(sorted(syms))}
        self.sg_to_id = {str(sg): i for i, sg in enumerate(range(1, 231))}
        self.max_rows_supported = 64
        for record in records:
            self.max_rows_supported = max(self.max_rows_supported, int(record.get("n_sites", len(record.get("wa_table") or []))))
        for rows in aux_rows or []:
            self.max_rows_supported = max(self.max_rows_supported, len(rows))

    def element_id(self, value: str) -> int:
        return self.element_to_id.get(str(value), self.element_to_id["<UNK>"])

    def orbit_id(self, value: str) -> int:
        return self.orbit_to_id.get(str(value), self.orbit_to_id["<UNK_ORBIT>"])

    def letter_id(self, value: str) -> int:
        return self.letter_to_id.get(str(value), self.letter_to_id["<UNK_LETTER>"])

    def sym_id(self, value: str) -> int:
        return self.sym_to_id.get(str(value), self.sym_to_id["<UNK_SYM>"])


def representative_values(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = {str(s) for s in (row.get("free_symbols") or params.keys())}
    values: list[float] = []
    mask: list[float] = []
    for sym in COORD_ORDER:
        values.append(float(params.get(sym, 0.0)) % 1.0)
        mask.append(1.0 if sym in free_symbols and sym in params else 0.0)
    return values, mask


def record_to_item(record: dict[str, Any], rows: list[dict[str, Any]], vocab: Vocab, mean: torch.Tensor, std: torch.Tensor) -> dict[str, torch.Tensor]:
    max_rows = vocab.max_rows_supported
    row_count = len(rows)
    formula_items = sorted((str(k), int(v)) for k, v in record["formula_counts"].items())
    max_formula = max(1, len(formula_items))
    formula_element_ids = torch.zeros(max_formula, dtype=torch.long)
    formula_counts = torch.zeros(max_formula, dtype=torch.float32)
    for i, (el, count) in enumerate(formula_items):
        formula_element_ids[i] = vocab.element_id(el)
        formula_counts[i] = float(count)
    orbit_ids = torch.zeros(max_rows, dtype=torch.long)
    element_ids = torch.zeros(max_rows, dtype=torch.long)
    letter_ids = torch.zeros(max_rows, dtype=torch.long)
    sym_ids = torch.zeros(max_rows, dtype=torch.long)
    multiplicities = torch.zeros(max_rows, dtype=torch.float32)
    enumerations = torch.zeros(max_rows, dtype=torch.float32)
    coord_values = torch.zeros((max_rows, 3), dtype=torch.float32)
    coord_mask = torch.zeros((max_rows, 3), dtype=torch.float32)
    row_mask = torch.zeros(max_rows, dtype=torch.float32)
    for i, row in enumerate(rows[:max_rows]):
        orbit_ids[i] = vocab.orbit_id(str(row.get("orbit_id")))
        element_ids[i] = vocab.element_id(str(row.get("element")))
        letter_ids[i] = vocab.letter_id(str(row.get("letter")))
        sym_ids[i] = vocab.sym_id(str(row.get("site_symmetry")))
        multiplicities[i] = float(row.get("multiplicity", 1)) / 32.0
        enumerations[i] = float(row.get("enumeration", 0) or 0) / 32.0
        vals, mask = representative_values(row)
        coord_values[i] = torch.tensor(vals, dtype=torch.float32)
        coord_mask[i] = torch.tensor(mask, dtype=torch.float32)
        row_mask[i] = 1.0
    lat = torch.tensor(lattice_target(record), dtype=torch.float32)
    lat_norm = (lat - mean) / std
    return {
        "formula_element_ids": formula_element_ids,
        "formula_counts": formula_counts,
        "sg_id": torch.tensor(vocab.sg_to_id[str(int(record["sg"]))], dtype=torch.long),
        "orbit_ids": orbit_ids,
        "element_ids": element_ids,
        "letter_ids": letter_ids,
        "sym_ids": sym_ids,
        "multiplicities": multiplicities,
        "enumerations": enumerations,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "row_mask": row_mask,
        "row_count": torch.tensor(row_count, dtype=torch.long),
        "lattice_values": lat_norm,
        "lattice_masks": torch.tensor(lattice_mask(int(record["sg"])), dtype=torch.float32),
        "complex_mask": torch.tensor(1.0 if row_count >= 7 else 0.0, dtype=torch.float32),
    }


class GeometryDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], vocab: Vocab, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.records = records
        self.vocab = vocab
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        return record_to_item(record, canonical_rows(record), self.vocab, self.mean, self.std)


def collate_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    formula_len = max(int(x["formula_element_ids"].shape[0]) for x in items)
    for key in items[0]:
        if key in {"formula_element_ids", "formula_counts"}:
            pad_value = 0
            vals = []
            for x in items:
                t = x[key]
                if t.shape[0] < formula_len:
                    pad = torch.zeros(formula_len - t.shape[0], dtype=t.dtype)
                    t = torch.cat([t, pad], dim=0)
                vals.append(t)
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = torch.stack([x[key] for x in items], dim=0)
    return out


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


class GeometryARNet(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int = 128, hidden_dim: int = 256, coord_components: int = 4, lattice_components: int = 4) -> None:
        super().__init__()
        self.coord_components = int(coord_components)
        self.lattice_components = int(lattice_components)
        self.element_emb = nn.Embedding(len(vocab.element_to_id), emb_dim)
        self.sg_emb = nn.Embedding(230, emb_dim)
        self.orbit_emb = nn.Embedding(len(vocab.orbit_to_id), emb_dim)
        self.letter_emb = nn.Embedding(len(vocab.letter_to_id), emb_dim // 2)
        self.sym_emb = nn.Embedding(len(vocab.sym_to_id), emb_dim // 2)
        self.pos_emb = nn.Embedding(vocab.max_rows_supported + 2, emb_dim // 2)
        self.context_proj = nn.Sequential(nn.LayerNorm(emb_dim * 2 + 3), nn.Linear(emb_dim * 2 + 3, hidden_dim), nn.GELU())
        self.lattice_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, lattice_components * 13))
        row_in = emb_dim + emb_dim + emb_dim // 2 + emb_dim // 2 + emb_dim // 2 + 5 + hidden_dim + emb_dim
        self.prev_coord_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, emb_dim), nn.GELU())
        self.lattice_ctx = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden_dim), nn.GELU())
        self.gru = nn.GRU(row_in, hidden_dim, batch_first=True)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * coord_components * 5),
        )

    def context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        elems = self.element_emb(batch["formula_element_ids"])
        counts = batch["formula_counts"]
        weights = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        formula = (elems * weights.unsqueeze(-1)).sum(dim=1)
        total_atoms = counts.sum(dim=1, keepdim=True) / 64.0
        num_elements = (counts > 0).float().sum(dim=1, keepdim=True) / 8.0
        rows = batch.get("row_mask")
        if rows is None:
            row_count = torch.zeros_like(total_atoms)
        else:
            row_count = rows.sum(dim=1, keepdim=True) / 64.0
        return self.context_proj(torch.cat([formula, self.sg_emb(batch["sg_id"]), total_atoms, num_elements, row_count], dim=-1))

    def lattice_distribution(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.lattice_head(ctx).view(ctx.shape[0], self.lattice_components, 13)
        logits = raw[..., 0]
        mu = raw[..., 1:7]
        scale = F.softplus(raw[..., 7:13]) + 0.03
        scale = scale.clamp(max=2.5)
        return logits, mu, scale

    def row_distribution(
        self,
        batch: dict[str, torch.Tensor],
        lattice_values: torch.Tensor,
        prev_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps = batch["orbit_ids"].shape
        ctx = self.context(batch)
        lat_ctx = self.lattice_ctx(lattice_values).unsqueeze(1).expand(-1, steps, -1)
        if prev_coords is None:
            prev_coords = torch.zeros_like(batch["coord_values"])
            prev_coords[:, 1:, :] = batch["coord_values"][:, :-1, :]
        prev = self.prev_coord_proj(prev_coords)
        pos = torch.arange(steps, device=batch["orbit_ids"].device).clamp(max=self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(pos).unsqueeze(0).expand(bsz, -1, -1)
        row_features = torch.stack(
            [
                batch["multiplicities"],
                batch["enumerations"],
                batch["coord_mask"][..., 0],
                batch["coord_mask"][..., 1],
                batch["coord_mask"][..., 2],
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                self.orbit_emb(batch["orbit_ids"]),
                self.element_emb(batch["element_ids"]),
                self.letter_emb(batch["letter_ids"]),
                self.sym_emb(batch["sym_ids"]),
                pos_emb,
                row_features,
                lat_ctx,
                prev,
            ],
            dim=-1,
        )
        h, _ = self.gru(x)
        raw = self.coord_head(h).view(bsz, steps, 3, self.coord_components, 5)
        logits = raw[..., 0]
        unit = F.normalize(raw[..., 1:3], dim=-1)
        scale = F.softplus(raw[..., 3]) + 0.05
        scale = scale.clamp(max=1.5)
        # raw[..., 4] is an unused slack channel kept for checkpoint compatibility
        return logits, unit, scale, h

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.context(batch)
        lat_logits, lat_mu, lat_scale = self.lattice_distribution(ctx)
        coord_logits, coord_unit, coord_scale, _ = self.row_distribution(batch, batch["lattice_values"])
        return {
            "lat_logits": lat_logits,
            "lat_mu": lat_mu,
            "lat_scale": lat_scale,
            "coord_logits": coord_logits,
            "coord_unit": coord_unit,
            "coord_scale": coord_scale,
        }


def lattice_mixture_nll(logits: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (target.unsqueeze(1) - mu) / scale
    logp_dim = -0.5 * diff.pow(2) - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
    logp = (logp_dim * mask.unsqueeze(1)).sum(dim=-1)
    log_mix = F.log_softmax(logits, dim=-1) + logp
    return -torch.logsumexp(log_mix, dim=-1).mean()


def coord_mixture_nll(logits: torch.Tensor, unit: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    angle = target * (2.0 * math.pi)
    target_unit = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1).unsqueeze(-2)
    diff = target_unit - unit
    logp = -0.5 * diff.pow(2).sum(dim=-1) / scale.pow(2) - 2.0 * torch.log(scale) - math.log(2.0 * math.pi)
    log_mix = F.log_softmax(logits, dim=-1) + logp
    nll = -torch.logsumexp(log_mix, dim=-1)
    weights = mask
    return (nll * weights).sum() / weights.sum().clamp_min(1.0)


def coord_point_from_dist(logits: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    comp = torch.argmax(logits, dim=-1)
    gather = comp.unsqueeze(-1).unsqueeze(-1).expand(*comp.shape, 1, 2)
    chosen = torch.gather(unit, dim=-2, index=gather).squeeze(-2)
    angle = torch.atan2(chosen[..., 1], chosen[..., 0])
    return torch.remainder(angle / (2.0 * math.pi), 1.0)


def row_pair_loss(pred: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor, coord_mask: torch.Tensor, complex_mask: torch.Tensor, complex_only: bool) -> tuple[torch.Tensor, torch.Tensor]:
    losses: list[torch.Tensor] = []
    sep_losses: list[torch.Tensor] = []
    for b in range(pred.shape[0]):
        if complex_only and float(complex_mask[b].detach().cpu()) < 0.5:
            continue
        idx = row_mask[b] > 0.5
        if int(idx.sum().detach().cpu()) < 2:
            continue
        p = pred[b, idx]
        t = target[b, idx]
        pd = torch.abs(p[:, None, :] - p[None, :, :])
        pd = torch.minimum(pd, 1.0 - pd)
        td = torch.abs(t[:, None, :] - t[None, :, :])
        td = torch.minimum(td, 1.0 - td)
        p_dist = torch.sqrt(pd.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        t_dist = torch.sqrt(td.pow(2).sum(dim=-1).clamp_min(1.0e-8))
        tri = torch.triu(torch.ones_like(p_dist, dtype=torch.bool), diagonal=1)
        losses.append(F.smooth_l1_loss(p_dist[tri], t_dist[tri], reduction="mean"))
        sep_losses.append(F.relu(0.08 - p_dist[tri]).pow(2).mean())
    zero = pred.sum() * 0.0
    return (torch.stack(losses).mean() if losses else zero, torch.stack(sep_losses).mean() if sep_losses else zero)


def geometry_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, pair_weight: float = 0.0, sep_weight: float = 0.0, complex_pair_only: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    lat_loss = lattice_mixture_nll(outputs["lat_logits"], outputs["lat_mu"], outputs["lat_scale"], batch["lattice_values"], batch["lattice_masks"])
    coord_loss = coord_mixture_nll(outputs["coord_logits"], outputs["coord_unit"], outputs["coord_scale"], batch["coord_values"], batch["coord_mask"])
    pred_coord = coord_point_from_dist(outputs["coord_logits"], outputs["coord_unit"])
    pred_full = torch.remainder(pred_coord * batch["coord_mask"] + batch["coord_values"] * (1.0 - batch["coord_mask"]), 1.0)
    pair, sep = row_pair_loss(pred_full, batch["coord_values"], batch["row_mask"], batch["coord_mask"], batch["complex_mask"], complex_pair_only)
    total = lat_loss + coord_loss + float(pair_weight) * pair + float(sep_weight) * sep
    with torch.no_grad():
        wrapped = torch.abs(pred_coord - batch["coord_values"])
        wrapped = torch.minimum(wrapped, 1.0 - wrapped)
        coord_mae = (wrapped * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)
        lat_pred = outputs["lat_mu"][torch.arange(outputs["lat_mu"].shape[0], device=outputs["lat_mu"].device), torch.argmax(outputs["lat_logits"], dim=-1)]
        lat_mae = (torch.abs(lat_pred - batch["lattice_values"]) * batch["lattice_masks"]).sum() / batch["lattice_masks"].sum().clamp_min(1.0)
    return total, {
        "loss": float(total.detach().cpu()),
        "lattice_nll": float(lat_loss.detach().cpu()),
        "coord_circular_mixture_nll": float(coord_loss.detach().cpu()),
        "coord_wrapped_mae": float(coord_mae.detach().cpu()),
        "lattice_normalized_mae": float(lat_mae.detach().cpu()),
        "row_pair_loss": float(pair.detach().cpu()),
        "separation_loss": float(sep.detach().cpu()),
    }


def train_geometry_model(
    *,
    name: str,
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    vocab: Vocab,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
    pair_weight: float,
    sep_weight: float,
    complex_sampler: bool,
) -> tuple[GeometryARNet, dict[str, Any]]:
    ckpt_dir = ROOT / "checkpoints" / name
    ckpt_path = ckpt_dir / "last.pt"
    history_path = ckpt_dir / "history.jsonl"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GeometryARNet(vocab, emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim), coord_components=int(args.coord_components), lattice_components=int(args.lattice_components)).to(device)
    if ckpt_path.exists() and not args.force:
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model"])
        return model, payload.get("info", {})
    if args.force and history_path.exists():
        history_path.unlink()
    seed_all(int(args.seed))
    dataset = GeometryDataset(train_records, vocab, mean_t, std_t)
    sampler = None
    shuffle = True
    sampler_info: dict[str, Any] = {"type": "shuffle"}
    if complex_sampler:
        weights = [4.0 if is_complex(r) else 1.0 for r in train_records]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True)
        shuffle = False
        sampler_info = {"type": "WeightedRandomSampler", "complex_weight": 4.0, "simple_weight": 1.0, "replacement": True, "num_samples_per_epoch": len(train_records)}
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=shuffle, sampler=sampler, num_workers=int(args.num_workers), collate_fn=collate_items)
    val_loader = DataLoader(GeometryDataset(val_records, vocab, mean_t, std_t), batch_size=int(args.eval_batch_size), shuffle=False, num_workers=0, collate_fn=collate_items)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs_stage2 if name.startswith("stage2") else args.epochs_stage1)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = Counter()
        steps = 0
        for raw in loader:
            batch = move_batch(raw, device)
            opt.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, parts = geometry_loss(outputs, batch, pair_weight=pair_weight, sep_weight=sep_weight, complex_pair_only=complex_sampler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            for k, v in parts.items():
                sums[k] += float(v)
            steps += 1
        row = {"epoch": epoch, "seconds": time.time() - started}
        for key in ("loss", "lattice_nll", "coord_circular_mixture_nll", "coord_wrapped_mae", "lattice_normalized_mae", "row_pair_loss", "separation_loss"):
            row[key] = sums[key] / max(1, steps)
        model.eval()
        val_sums = Counter()
        val_steps = 0
        with torch.no_grad():
            for raw in val_loader:
                batch = move_batch(raw, device)
                outputs = model(batch)
                loss, parts = geometry_loss(outputs, batch, pair_weight=pair_weight, sep_weight=sep_weight, complex_pair_only=complex_sampler)
                for k, v in parts.items():
                    val_sums[f"val_{k}"] += float(v)
                val_steps += 1
        for key in ("loss", "lattice_nll", "coord_circular_mixture_nll", "coord_wrapped_mae", "lattice_normalized_mae", "row_pair_loss", "separation_loss"):
            row[f"val_{key}"] = val_sums[f"val_{key}"] / max(1, val_steps)
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        print(json.dumps({"stage": name, **row}, sort_keys=True), flush=True)
        if row["val_loss"] < best_loss:
            best_loss = float(row["val_loss"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "info": {"history": history, "sampler": sampler_info, "best_val_loss": best_loss}}, ckpt_path)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model": model.state_dict(), "info": {"history": history, "sampler": sampler_info, "best_val_loss": best_loss}}, ckpt_path)
    return model, {"history": history, "sampler": sampler_info, "best_val_loss": best_loss}


@torch.no_grad()
def sample_lattice(model: GeometryARNet, batch: dict[str, torch.Tensor], mean_t: torch.Tensor, std_t: torch.Tensor, sg: int, gen_index: int, seed: int) -> tuple[torch.Tensor, dict[str, float]]:
    ctx = model.context(batch)
    logits, mu, scale = model.lattice_distribution(ctx)
    if gen_index == 0:
        comp = int(torch.argmax(logits[0]).detach().cpu())
        raw = mu[0, comp].detach().clone()
    else:
        generator = torch.Generator(device=logits.device).manual_seed(int(seed))
        comp_t = torch.multinomial(torch.softmax(logits[0], dim=-1), 1, generator=generator)
        comp = int(comp_t.item())
        raw = mu[0, comp] + torch.randn(6, device=logits.device, generator=generator) * scale[0, comp]
    raw = raw * batch["lattice_masks"][0] + batch["lattice_values"][0] * (1.0 - batch["lattice_masks"][0])
    vals = (raw.detach().cpu() * std_t.cpu() + mean_t.cpu()).tolist()
    lattice = lattice_from_target(vals, sg=sg)
    return raw.view(1, 6), lattice


@torch.no_grad()
def sample_coords_autoreg(
    model: GeometryARNet,
    item: dict[str, torch.Tensor],
    lattice_raw: torch.Tensor,
    rows: list[dict[str, Any]],
    gen_index: int,
    seed: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
    batch["lattice_values"] = lattice_raw.to(device)
    params: dict[int, dict[str, float]] = {}
    generator = torch.Generator(device=device).manual_seed(int(seed) + 100000)
    ctx = model.context(batch)
    lat_ctx = model.lattice_ctx(lattice_raw.to(device))
    hidden: torch.Tensor | None = None
    prev_coord = torch.zeros((1, 3), dtype=torch.float32, device=device)
    for step, row in enumerate(rows[: batch["coord_values"].shape[1]]):
        pos = torch.tensor([min(step, model.pos_emb.num_embeddings - 1)], dtype=torch.long, device=device)
        row_features = torch.stack(
            [
                batch["multiplicities"][:, step],
                batch["enumerations"][:, step],
                batch["coord_mask"][:, step, 0],
                batch["coord_mask"][:, step, 1],
                batch["coord_mask"][:, step, 2],
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                model.orbit_emb(batch["orbit_ids"][:, step]),
                model.element_emb(batch["element_ids"][:, step]),
                model.letter_emb(batch["letter_ids"][:, step]),
                model.sym_emb(batch["sym_ids"][:, step]),
                model.pos_emb(pos),
                row_features,
                lat_ctx,
                model.prev_coord_proj(prev_coord),
            ],
            dim=-1,
        ).unsqueeze(1)
        h, hidden = model.gru(x, hidden)
        raw = model.coord_head(h).view(1, 1, 3, model.coord_components, 5)
        logits = raw[..., 0]
        unit = F.normalize(raw[..., 1:3], dim=-1)
        scale = (F.softplus(raw[..., 3]) + 0.05).clamp(max=1.5)
        row_params: dict[str, float] = {}
        current_coord = torch.zeros((1, 3), dtype=torch.float32, device=device)
        for axis, sym in enumerate(COORD_ORDER):
            if sym not in set(row.get("free_symbols") or []):
                continue
            row_logits = logits[0, 0, axis]
            if gen_index == 0:
                comp = int(torch.argmax(row_logits).detach().cpu())
                vec = unit[0, 0, axis, comp]
                angle = torch.atan2(vec[1], vec[0])
            else:
                comp_t = torch.multinomial(torch.softmax(row_logits, dim=-1), 1, generator=generator)
                comp = int(comp_t.item())
                vec = unit[0, 0, axis, comp]
                mu_angle = torch.atan2(vec[1], vec[0])
                angle = mu_angle + torch.randn((), device=device, generator=generator) * scale[0, 0, axis, comp]
            value = float(torch.remainder(angle / (2.0 * math.pi), 1.0).detach().cpu())
            row_params[sym] = value
            current_coord[0, axis] = value
        params[step] = row_params
        prev_coord = current_coord
    return params


class RefinerNet(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(len(vocab.element_to_id), emb_dim)
        self.sg_emb = nn.Embedding(230, emb_dim)
        self.orbit_emb = nn.Embedding(len(vocab.orbit_to_id), emb_dim)
        self.letter_emb = nn.Embedding(len(vocab.letter_to_id), emb_dim // 2)
        self.sym_emb = nn.Embedding(len(vocab.sym_to_id), emb_dim // 2)
        self.pos_emb = nn.Embedding(vocab.max_rows_supported + 2, emb_dim // 2)
        self.ctx_proj = nn.Sequential(nn.LayerNorm(emb_dim * 2 + 3), nn.Linear(emb_dim * 2 + 3, hidden_dim), nn.GELU())
        row_in = emb_dim + emb_dim + emb_dim // 2 + emb_dim // 2 + emb_dim // 2 + 5 + 3 + hidden_dim
        self.lattice_proj = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, hidden_dim), nn.GELU())
        self.gru = nn.GRU(row_in, hidden_dim, batch_first=True)
        self.coord_delta = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.lattice_delta = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))

    def context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        elems = self.element_emb(batch["formula_element_ids"])
        counts = batch["formula_counts"]
        weights = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        formula = (elems * weights.unsqueeze(-1)).sum(dim=1)
        total_atoms = counts.sum(dim=1, keepdim=True) / 64.0
        num_elements = (counts > 0).float().sum(dim=1, keepdim=True) / 8.0
        row_count = batch["row_mask"].sum(dim=1, keepdim=True) / 64.0
        return self.ctx_proj(torch.cat([formula, self.sg_emb(batch["sg_id"]), total_atoms, num_elements, row_count], dim=-1))

    def step(self, batch: dict[str, torch.Tensor], coords: torch.Tensor, lattice: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps = batch["orbit_ids"].shape
        ctx = self.context(batch)
        lat = self.lattice_proj(lattice)
        pos = torch.arange(steps, device=batch["orbit_ids"].device).clamp(max=self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(pos).unsqueeze(0).expand(bsz, -1, -1)
        row_features = torch.stack(
            [
                batch["multiplicities"],
                batch["enumerations"],
                batch["coord_mask"][..., 0],
                batch["coord_mask"][..., 1],
                batch["coord_mask"][..., 2],
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                self.orbit_emb(batch["orbit_ids"]),
                self.element_emb(batch["element_ids"]),
                self.letter_emb(batch["letter_ids"]),
                self.sym_emb(batch["sym_ids"]),
                pos_emb,
                row_features,
                coords,
                lat.unsqueeze(1).expand(-1, steps, -1),
            ],
            dim=-1,
        )
        h, _ = self.gru(x)
        coord_delta = 0.20 * torch.tanh(self.coord_delta(h)) * batch["coord_mask"]
        pooled = (h * batch["row_mask"].unsqueeze(-1)).sum(dim=1) / batch["row_mask"].sum(dim=1, keepdim=True).clamp_min(1.0)
        lattice_delta = 0.20 * torch.tanh(self.lattice_delta(torch.cat([pooled, ctx], dim=-1))) * batch["lattice_masks"]
        return coord_delta, lattice_delta

    def forward_fixed(self, batch: dict[str, torch.Tensor], coords: torch.Tensor, lattice: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = coords
        l = lattice
        for _ in range(int(steps)):
            dc, dl = self.step(batch, c, l)
            c = torch.remainder(c + dc, 1.0) * batch["coord_mask"] + batch["coord_values"] * (1.0 - batch["coord_mask"])
            l = l + dl
        return c, l


def corrupt_batch(batch: dict[str, torch.Tensor], seed_offset: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    coords = batch["coord_values"].clone()
    lattice = batch["lattice_values"].clone()
    generator = torch.Generator(device=coords.device).manual_seed(123456 + int(seed_offset))
    coord_noise = torch.randn(coords.shape, device=coords.device, generator=generator) * 0.10
    lattice_noise = torch.randn(lattice.shape, device=lattice.device, generator=generator) * 0.25
    coords = torch.remainder(coords + coord_noise * batch["coord_mask"], 1.0)
    lattice = lattice + lattice_noise * batch["lattice_masks"]
    if coords.shape[1] >= 2:
        collision_mask = (torch.rand((coords.shape[0],), device=coords.device, generator=generator) < 0.25) & (batch["row_mask"].sum(dim=1) >= 2)
        for b in torch.where(collision_mask)[0].detach().cpu().tolist():
            coords[b, 1] = coords[b, 0]
    if coords.shape[1] >= 3:
        shuffle_mask = torch.rand((coords.shape[0],), device=coords.device, generator=generator) < 0.25
        for b in torch.where(shuffle_mask)[0].detach().cpu().tolist():
            valid = int(batch["row_mask"][b].sum().detach().cpu())
            if valid > 2:
                coords[b, :valid] = coords[b, torch.randperm(valid, generator=generator, device=coords.device)]
    return coords, lattice, {
        "synthetic_corruption": ["free-param noise", "site mapping shuffle", "collision short-contact", "lattice/VPA noise", "inter-row distance corruption"],
    }


def refiner_loss(pred_coords: torch.Tensor, pred_lattice: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    diff = torch.abs(pred_coords - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord = (wrapped * batch["coord_mask"]).pow(2).sum() / batch["coord_mask"].sum().clamp_min(1.0)
    lat = ((pred_lattice - batch["lattice_values"]) * batch["lattice_masks"]).pow(2).sum() / batch["lattice_masks"].sum().clamp_min(1.0)
    pair, sep = row_pair_loss(pred_coords, batch["coord_values"], batch["row_mask"], batch["coord_mask"], batch["complex_mask"], False)
    total = coord + lat + 0.25 * pair + 0.10 * sep
    return total, {
        "loss": float(total.detach().cpu()),
        "periodic_coordinate_loss": float(coord.detach().cpu()),
        "lattice_metric_vpa_loss": float(lat.detach().cpu()),
        "row_pair_distance_loss": float(pair.detach().cpu()),
        "collision_short_contact_penalty": float(sep.detach().cpu()),
    }


def train_refiner(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    vocab: Vocab,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[RefinerNet, dict[str, Any]]:
    ckpt_dir = ROOT / "checkpoints/stage3_refiner"
    ckpt_path = ckpt_dir / "last.pt"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RefinerNet(vocab, emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim)).to(device)
    if ckpt_path.exists() and not args.force:
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model"])
        return model, payload.get("info", {})
    history_path = ckpt_dir / "history.jsonl"
    if args.force and history_path.exists():
        history_path.unlink()
    seed_all(int(args.seed) + 30)
    weights = [3.0 if is_complex(r) else 1.0 for r in train_records]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_records), replacement=True)
    loader = DataLoader(GeometryDataset(train_records, vocab, mean_t, std_t), batch_size=int(args.batch_size), sampler=sampler, shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_items)
    val_loader = DataLoader(GeometryDataset(val_records, vocab, mean_t, std_t), batch_size=int(args.eval_batch_size), shuffle=False, num_workers=0, collate_fn=collate_items)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    hard_stats = hard_negative_stats()
    for epoch in range(1, int(args.epochs_refiner) + 1):
        model.train()
        sums = Counter()
        steps = 0
        for raw in loader:
            batch = move_batch(raw, device)
            coords0, lat0, _ = corrupt_batch(batch, seed_offset=epoch * 10000 + steps)
            opt.zero_grad(set_to_none=True)
            pred_c, pred_l = model.forward_fixed(batch, coords0, lat0, int(args.refiner_steps))
            loss, parts = refiner_loss(pred_c, pred_l, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            for k, v in parts.items():
                sums[k] += float(v)
            steps += 1
        row = {"epoch": epoch}
        for key in ("loss", "periodic_coordinate_loss", "lattice_metric_vpa_loss", "row_pair_distance_loss", "collision_short_contact_penalty"):
            row[key] = sums[key] / max(1, steps)
        model.eval()
        val_sums = Counter()
        val_steps = 0
        with torch.no_grad():
            for raw in val_loader:
                batch = move_batch(raw, device)
                coords0, lat0, _ = corrupt_batch(batch, seed_offset=900000 + epoch * 1000 + val_steps)
                pred_c, pred_l = model.forward_fixed(batch, coords0, lat0, int(args.refiner_steps))
                loss, parts = refiner_loss(pred_c, pred_l, batch)
                for k, v in parts.items():
                    val_sums[f"val_{k}"] += float(v)
                val_steps += 1
        for key in ("loss", "periodic_coordinate_loss", "lattice_metric_vpa_loss", "row_pair_distance_loss", "collision_short_contact_penalty"):
            row[f"val_{key}"] = val_sums[f"val_{key}"] / max(1, val_steps)
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        print(json.dumps({"stage": "stage3_refiner", **row}, sort_keys=True), flush=True)
        if row["val_loss"] < best_val:
            best_val = float(row["val_loss"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "info": {"history": history, "hard_negative_stats": hard_stats, "best_val_loss": best_val}}, ckpt_path)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model": model.state_dict(), "info": {"history": history, "hard_negative_stats": hard_stats, "best_val_loss": best_val}}, ckpt_path)
    return model, {"history": history, "hard_negative_stats": hard_stats, "best_val_loss": best_val}


def render_from_params(engine: OrbitEngine, record: dict[str, Any], rows: list[dict[str, Any]], params: dict[int, dict[str, float]], lattice: dict[str, float], gen_index: int, source: str) -> dict[str, Any]:
    skel, wa = canonical_keys_from_rows(rows)
    candidate = {"rows": rows, "params": params, "lattice": lattice, "canonical_skeleton_key": skel, "canonical_wa_key": wa}
    return render_candidate(engine, record, candidate, gen_index, source)


@torch.no_grad()
def apply_refiner_to_candidate(
    refiner: RefinerNet,
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    params: dict[int, dict[str, float]],
    lattice_raw: torch.Tensor,
    vocab: Vocab,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
    steps: int,
) -> tuple[dict[int, dict[str, float]], torch.Tensor, dict[str, float]]:
    rec = copy.deepcopy(record)
    row_copy = []
    for i, row in enumerate(rows):
        nr = copy.deepcopy(row)
        nr["free_params"] = dict(params.get(i, {}))
        row_copy.append(nr)
    item = record_to_item(rec, row_copy, vocab, mean_t, std_t)
    batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
    coords = batch["coord_values"].clone()
    lattice = lattice_raw.to(device).clone()
    out_c, out_l = refiner.forward_fixed(batch, coords, lattice, int(steps))
    out_params: dict[int, dict[str, float]] = {}
    for i, row in enumerate(rows):
        row_params: dict[str, float] = {}
        for axis, sym in enumerate(COORD_ORDER):
            if sym in set(row.get("free_symbols") or []):
                row_params[sym] = float(out_c[0, i, axis].detach().cpu()) % 1.0
        out_params[i] = row_params
    vals = (out_l[0].detach().cpu() * std_t.cpu() + mean_t.cpu()).tolist()
    out_lattice = lattice_from_target(vals, sg=int(record["sg"]))
    return out_params, out_l.detach().cpu(), out_lattice


def generate_rows(
    *,
    run_name: str,
    records: list[dict[str, Any]],
    model: GeometryARNet,
    refiner: RefinerNet | None,
    vocab: Vocab,
    engine: OrbitEngine,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
    condition: str,
    aux: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    out_path = ROOT / "eval" / run_name / "generations.jsonl"
    partial_path = ROOT / "eval" / run_name / "generations.partial.jsonl"
    progress_path = ROOT / "eval" / run_name / "generation_progress.json"
    if out_path.exists() and not args.force:
        return read_jsonl(out_path)
    if args.force:
        for path in (out_path, partial_path):
            if path.exists():
                path.unlink()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.eval()
    if refiner is not None:
        refiner.eval()
    subset = records[: int(limit)] if limit is not None else records
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(handle: Any, row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    with partial_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(subset):
            rows, cond_info = condition_rows(record, condition, engine, aux)
            target_skel, target_wa = canonical_keys_from_rows(canonical_rows(record))
            if idx % 8 == 0:
                write_json(
                    progress_path,
                    {
                        "run": run_name,
                        "done_samples": idx,
                        "total_samples": len(subset),
                        "num_candidates": int(args.num_candidates),
                        "updated": now_text(),
                    },
                )
            if not rows:
                for gen_index in range(int(args.num_candidates)):
                    emit(
                        f,
                        {
                            "mode": DIRECT_MODE,
                            "sample_index": idx,
                            "sample_id": record["sample_id"],
                            "gen_index": gen_index,
                            "seed": SEEDS[gen_index % len(SEEDS)],
                            "raw_generation_success": False,
                            "generated_text": "",
                            "error": cond_info.get("condition_source", "missing_condition_rows"),
                            "candidate_order": "generation_index_then_fixed_seed",
                            "condition": condition,
                            "skeleton_hit": False,
                            "wa_hit": False,
                            "condition_skeleton_exact": False,
                            "condition_wa_exact": False,
                            "generation_time_seconds": 0.0,
                        },
                    )
                f.flush()
                continue
            item = record_to_item(record, rows, vocab, mean_t, std_t)
            batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            for gen_index in range(int(args.num_candidates)):
                seed = SEEDS[gen_index % len(SEEDS)]
                try:
                    lattice_raw, lattice = sample_lattice(model, batch, mean_t, std_t, int(record["sg"]), gen_index, seed)
                    params = sample_coords_autoreg(model, item, lattice_raw, rows, gen_index, seed, device)
                    if refiner is not None:
                        params, lattice_raw, lattice = apply_refiner_to_candidate(refiner, record, rows, params, lattice_raw, vocab, mean_t, std_t, device, int(args.refiner_steps))
                    rendered = render_from_params(engine, record, rows, params, lattice, gen_index, "opentry6_geometry")
                    skel, wa = canonical_keys_from_rows(rows)
                    emit(
                        f,
                        {
                            "mode": DIRECT_MODE,
                            "sample_index": idx,
                            "sample_id": record["sample_id"],
                            "gen_index": gen_index,
                            "seed": seed,
                            "raw_generation_success": bool(rendered["ok"]),
                            "generated_text": rendered.get("cif", ""),
                            "error": rendered.get("error"),
                            "candidate_order": "generation_index_then_fixed_seed",
                            "condition": condition,
                            "condition_source": cond_info.get("condition_source"),
                            "condition_skeleton_exact": bool(cond_info.get("skeleton_exact")),
                            "condition_wa_exact": bool(cond_info.get("wa_exact")),
                            "condition_exact_fallback_to_oof": bool(cond_info.get("exact_fallback_to_oof", False)),
                            "canonical_skeleton_key": skel,
                            "canonical_wa_key": wa,
                            "target_canonical_skeleton_key": target_skel,
                            "target_canonical_wa_key": target_wa,
                            "skeleton_hit": skel == target_skel,
                            "wa_hit": wa == target_wa,
                            "atom_count_ok": bool(rendered.get("atom_count_ok")),
                            "formula_closure_success": bool(rendered.get("atom_count_ok")),
                            "geometry_sampler": "opentry6_crystalformer_style_mixture_fixed_seed",
                            "generation_time_seconds": 0.0,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    emit(
                        f,
                        {
                            "mode": DIRECT_MODE,
                            "sample_index": idx,
                            "sample_id": record["sample_id"],
                            "gen_index": gen_index,
                            "seed": seed,
                            "raw_generation_success": False,
                            "generated_text": "",
                            "error": f"{type(exc).__name__}: {exc}",
                            "candidate_order": "generation_index_then_fixed_seed",
                            "condition": condition,
                            "condition_source": cond_info.get("condition_source"),
                            "condition_skeleton_exact": bool(cond_info.get("skeleton_exact")),
                            "condition_wa_exact": bool(cond_info.get("wa_exact")),
                            "skeleton_hit": False,
                            "wa_hit": False,
                            "generation_time_seconds": 0.0,
                        },
                    )
            f.flush()
    partial_path.replace(out_path)
    write_json(
        progress_path,
        {
            "run": run_name,
            "done_samples": len(subset),
            "total_samples": len(subset),
            "num_candidates": int(args.num_candidates),
            "updated": now_text(),
            "complete": True,
        },
    )
    return read_jsonl(out_path)


def evaluate_generations(run_name: str, records: list[dict[str, Any]], generation_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    run_dir = ROOT / "eval" / run_name
    report_path = run_dir / "report.json"
    if report_path.exists() and not args.force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["sample_index"])].append(row)
    for value in grouped.values():
        value.sort(key=lambda r: int(r["gen_index"]))
    payload = [
        {
            "index": i,
            "sample_id": r["sample_id"],
            "source_path": r["source_path"],
            "target_formula": r["formula"],
            "target_sg_number": int(r["sg"]),
            "target_sg_symbol": r.get("sg_symbol"),
        }
        for i, r in enumerate(records)
    ]
    eval_args = argparse.Namespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.match_timeout_seconds),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
        max_match_sites=int(args.max_match_sites),
        max_eval_sites=int(args.max_eval_sites),
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode=DIRECT_MODE,
        case_payload=payload,
        grouped=grouped,
        lookup_json=str(lookup_path()),
        args=eval_args,
    )
    metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(run_dir / "metrics.jsonl", metrics)
    report = summarize_metrics(records, generation_rows, metrics, top_ks=[1, 5, 10, 20, 50])
    write_json(report_path, report)
    return report


def summarize_metrics(records: list[dict[str, Any]], generation_rows: list[dict[str, Any]], metrics: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in generation_rows}
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_sample[int(row["sample_index"])].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda r: int(r["gen_index"]))

    def subset_summary(indices: list[int], k: int) -> dict[str, Any]:
        if k > max(1, max((int(r["gen_index"]) for r in metrics), default=-1) + 1):
            return {"samples": len(indices), "not_run": True}
        match_any = 0
        match_first = 0
        rms_vals: list[float] = []
        readable_any = comp_any = sg_any = mult_any = valid_any = 0
        collision_like = 0
        cand_count = 0
        wa_hit_match_fail = 0
        skel_hit_match_fail = 0
        wa_exact_any = 0
        skel_exact_any = 0
        for idx in indices:
            rows = [r for r in by_sample.get(idx, []) if int(r["gen_index"]) < k]
            if not rows:
                continue
            first = rows[0]
            if bool(first.get("match_ok")):
                match_first += 1
            matches = [r for r in rows if bool(r.get("match_ok"))]
            if matches:
                match_any += 1
                vals = [float(r["rms"]) for r in matches if r.get("rms") is not None]
                if vals:
                    rms_vals.append(min(vals))
            readable_any += int(any(bool(r.get("pymatgen_readable")) for r in rows))
            comp_any += int(any(bool(r.get("formula_ok")) for r in rows))
            sg_any += int(any(bool(r.get("space_group_ok")) for r in rows))
            mult_any += int(any(bool(r.get("multiplicity_ok")) for r in rows))
            valid_any += int(any(bool(r.get("valid")) for r in rows))
            wa_exact_any += int(any(bool(gen_by_key.get((idx, int(r["gen_index"])), {}).get("wa_hit")) for r in rows))
            skel_exact_any += int(any(bool(gen_by_key.get((idx, int(r["gen_index"])), {}).get("skeleton_hit")) for r in rows))
            has_wa_hit = any(bool(gen_by_key.get((idx, int(r["gen_index"])), {}).get("wa_hit")) for r in rows)
            has_skel_hit = any(bool(gen_by_key.get((idx, int(r["gen_index"])), {}).get("skeleton_hit")) for r in rows)
            if has_wa_hit and not matches:
                wa_hit_match_fail += 1
            if has_skel_hit and not matches:
                skel_hit_match_fail += 1
            for r in rows:
                cand_count += 1
                score = r.get("bond_length_score")
                if score is not None and float(score) < 0.75:
                    collision_like += 1
        denom = max(1, len(indices))
        return {
            "samples": len(indices),
            "match_at_k": match_any / denom,
            "match_at_1": match_first / denom if k == 1 else None,
            "rmse": float(np.mean(rms_vals)) if rms_vals else math.nan,
            "readable": readable_any / denom,
            "composition_exact": comp_any / denom,
            "sg_wyckoff_legal": (sg_any / denom + mult_any / denom) / 2.0,
            "valid": valid_any / denom,
            "wa_exact": wa_exact_any / denom,
            "skeleton_exact": skel_exact_any / denom,
            "wa_hit_match_fail": wa_hit_match_fail / denom,
            "skeleton_hit_match_fail": skel_hit_match_fail / denom,
            "collision_like_rate": collision_like / max(1, cand_count),
        }

    all_idx = list(range(len(records)))
    rows_ge7_idx = [i for i, r in enumerate(records) if is_complex(r)]
    report: dict[str, Any] = {
        "num_records": len(records),
        "num_rows_ge7": len(rows_ge7_idx),
        "row_bins": dict(Counter(row_bin(r) for r in records)),
        "topk": {},
        "rows_ge7_topk": {},
        "bins_top20": {},
    }
    for k in top_ks:
        report["topk"][str(k)] = subset_summary(all_idx, k)
        report["rows_ge7_topk"][str(k)] = subset_summary(rows_ge7_idx, k)
    for label in ("7-9", "10-14", "15+"):
        indices = [i for i, r in enumerate(records) if row_bin(r) == label]
        report["bins_top20"][label] = subset_summary(indices, 20)
    return report


def metrics_signature(metrics_path: Path) -> str:
    rows = read_jsonl(metrics_path)
    norm = []
    for row in rows:
        norm.append(
            {
                k: v
                for k, v in row.items()
                if not k.endswith("_time_seconds")
                and k not in {"generation_time_seconds", "average_generation_time"}
            }
        )
    text = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=True) for r in norm)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def baseline_rows_ge7_positives() -> set[str]:
    candidates = [
        OP5 / "eval/fixed_order_geom_sampler_smoke_E8028_fold_a/metrics.jsonl",
        OP5 / "eval/fixed_order_geom_sampler_smoke_E8036_fold_b/metrics.jsonl",
        OP5 / "eval/fixed_order_geom_sampler_smoke_E8034_fold_b/metrics.jsonl",
    ]
    out: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(path):
            if int(row.get("gen_index", 999)) < 20:
                by_sample[str(row.get("sample_id"))].append(row)
        for sample_id, rows in by_sample.items():
            if any(bool(r.get("match_ok")) for r in rows):
                out.add(sample_id)
    return out


def eval_run(
    *,
    run_name: str,
    records: list[dict[str, Any]],
    model: GeometryARNet,
    refiner: RefinerNet | None,
    vocab: Vocab,
    engine: OrbitEngine,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    args: argparse.Namespace,
    condition: str,
    aux: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    subset = records[: int(limit)] if limit is not None else records
    generation_rows = generate_rows(
        run_name=run_name,
        records=subset,
        model=model,
        refiner=refiner,
        vocab=vocab,
        engine=engine,
        mean_t=mean_t,
        std_t=std_t,
        args=args,
        condition=condition,
        aux=aux,
        limit=None,
    )
    return evaluate_generations(run_name, subset, generation_rows, args)


def train_dev_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "rows_ge7": sum(1 for r in records if is_complex(r)),
        "unique_formula": len({str(r.get("formula")) for r in records}),
        "unique_sg": len({int(r.get("sg")) for r in records}),
        "unique_wyckoff_pattern": len({str(r.get("canonical_skeleton_key")) for r in records}),
        "row_bins": dict(Counter(row_bin(r) for r in records)),
    }


def inventory() -> dict[str, Any]:
    out: dict[str, Any] = {
        "time": now_text(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cwd": os.getcwd(),
        "executable": sys.executable,
    }
    try:
        import psutil  # type: ignore

        out["cpu_count"] = os.cpu_count()
        out["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception as exc:
        out["resource_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=20)
    except Exception as exc:
        out["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        out["disk"] = subprocess.check_output(["df", "-h", str(ROOT)], text=True, timeout=20)
    except Exception as exc:
        out["disk_error"] = f"{type(exc).__name__}: {exc}"
    return out


def report_value(report: dict[str, Any], scope: str, k: int, key: str) -> str:
    block = report.get(scope, {}).get(str(k), {})
    if block.get("not_run"):
        return "not_run"
    value = block.get(key)
    return pct(value) if isinstance(value, (int, float)) and key != "rmse" else ("NA" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value))


def stage_md(
    *,
    stage: str,
    title: str,
    read_files: list[str],
    write_files: list[str],
    data_scope: str,
    model_structure: str,
    train_objective: str,
    params: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    conclusion: str,
    failure: str,
    next_step: str,
    extra: dict[str, Any] | None = None,
) -> str:
    combined = combine_reports(reports)
    rows_ge7_new = combined.get("rows_ge7_new_positives", "NA")
    return f"""## {stage}: {title}

* 时间：{now_text()}
* 是否使用 crystallm_env：yes; executable={sys.executable}
* 读取文件：{'; '.join(read_files)}
* 写入文件：{'; '.join(write_files)}
* 是否写入 opentry_6 之外：no
* 是否读取 test：no
* 是否使用排序/筛选/打分：no
* candidate 顺序：candidate 0 deterministic decode; candidate 1-{params.get('num_candidates', 20) - 1} fixed seeds {SEEDS[: int(params.get('num_candidates', 20))]}; invalid slots retained
* 数据范围：{data_scope}
* 模型结构：{model_structure}
* 训练目标：{train_objective}
* 关键参数：{json.dumps(params, sort_keys=True, ensure_ascii=True)}
* readable：{combined.get('readable', 'NA')}
* composition exact：{combined.get('composition_exact', 'NA')}
* SG/Wyckoff legal：{combined.get('sg_wyckoff_legal', 'NA')}
* match@1：{combined.get('match@1', 'NA')}
* match@5：{combined.get('match@5', 'NA')}
* match@20：{combined.get('match@20', 'NA')}
* match@50：{combined.get('match@50', 'not_run')}
* RMSE@1/5/20：{combined.get('rmse@1', 'NA')} / {combined.get('rmse@5', 'NA')} / {combined.get('rmse@20', 'NA')}
* rows>=7 match@1/5/20/50：{combined.get('rows>=7 match@1', 'NA')} / {combined.get('rows>=7 match@5', 'NA')} / {combined.get('rows>=7 match@20', 'NA')} / {combined.get('rows>=7 match@50', 'not_run')}
* rows>=7 positive-any：{combined.get('rows_ge7_positive_any', 'NA')}
* rows>=7 new positives：{rows_ge7_new}
* W/A-hit match-fail：{combined.get('wa_hit_match_fail', 'NA')}
* skeleton-hit match-fail：{combined.get('skeleton_hit_match_fail', 'NA')}
* collision-like rate：{combined.get('collision_like_rate', 'NA')}
* fold reports：{json.dumps({k: safe_report_summary(v) for k, v in reports.items()}, sort_keys=True, ensure_ascii=True)}
* extra：{json.dumps(extra or {}, sort_keys=True, ensure_ascii=True)}
* 结论：{conclusion}
* 失败原因：{failure}
* 下一步：{next_step}
"""


def safe_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_records": report.get("num_records"),
        "num_rows_ge7": report.get("num_rows_ge7"),
        "top20_match": report.get("topk", {}).get("20", {}).get("match_at_k"),
        "rows_ge7_top20_match": report.get("rows_ge7_topk", {}).get("20", {}).get("match_at_k"),
        "rows_ge7_top20_rmse": report.get("rows_ge7_topk", {}).get("20", {}).get("rmse"),
    }


def combine_reports(reports: dict[str, dict[str, Any]]) -> dict[str, str]:
    # Macro-average over fold reports to keep the log compact and fold-specific JSON available.
    out: dict[str, str] = {}
    if not reports:
        return out
    for k in (1, 5, 20, 50):
        vals = []
        rms = []
        rows_vals = []
        for report in reports.values():
            top = report.get("topk", {}).get(str(k), {})
            rtop = report.get("rows_ge7_topk", {}).get(str(k), {})
            if not top.get("not_run") and top.get("match_at_k") is not None:
                vals.append(float(top["match_at_k"]))
            if not rtop.get("not_run") and rtop.get("match_at_k") is not None:
                rows_vals.append(float(rtop["match_at_k"]))
            if not top.get("not_run") and top.get("rmse") is not None and not math.isnan(float(top.get("rmse"))):
                rms.append(float(top["rmse"]))
        out[f"match@{k}"] = pct(float(np.mean(vals))) if vals else "not_run"
        out[f"rows>=7 match@{k}"] = pct(float(np.mean(rows_vals))) if rows_vals else "not_run"
        out[f"rmse@{k}"] = f"{float(np.mean(rms)):.4f}" if rms else "NA"
    for key, label in (
        ("readable", "readable"),
        ("composition_exact", "composition_exact"),
        ("sg_wyckoff_legal", "sg_wyckoff_legal"),
        ("wa_hit_match_fail", "wa_hit_match_fail"),
        ("skeleton_hit_match_fail", "skeleton_hit_match_fail"),
        ("collision_like_rate", "collision_like_rate"),
    ):
        vals = []
        for report in reports.values():
            val = report.get("topk", {}).get("20", {}).get(key)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                vals.append(float(val))
        out[label] = pct(float(np.mean(vals))) if vals else "NA"
    baseline = baseline_rows_ge7_positives()
    positives = set()
    for run_name in reports:
        gen_path = ROOT / "eval" / run_name / "generations.jsonl"
        met_path = ROOT / "eval" / run_name / "metrics.jsonl"
        if not gen_path.exists() or not met_path.exists():
            continue
        gens = {(int(r["sample_index"]), int(r["gen_index"])): r for r in read_jsonl(gen_path)}
        for m in read_jsonl(met_path):
            if int(m.get("gen_index", 999)) < 20 and bool(m.get("match_ok")):
                g = gens.get((int(m["sample_index"]), int(m["gen_index"])), {})
                if bool(g.get("wa_hit")) and str(m.get("sample_id")):
                    positives.add(str(m["sample_id"]))
    out["rows_ge7_positive_any"] = str(len(positives))
    out["rows_ge7_new_positives"] = str(len(positives - baseline))
    return out


def init_log(inv: dict[str, Any], data_stats: dict[str, Any]) -> None:
    overwrite_md(
        LOG_PATH,
        f"""# opentry_6 Experiment Log

Initial resource inventory:

```json
{json.dumps(inv, indent=2, sort_keys=True, ensure_ascii=True)}
```

Initial data stats:

```json
{json.dumps(data_stats, indent=2, sort_keys=True, ensure_ascii=True)}
```
""",
    )


def final_summary(stage_reports: dict[str, Any]) -> None:
    def stage_line(name: str) -> str:
        reports = stage_reports.get(name, {})
        combined = combine_reports(reports) if isinstance(reports, dict) else {}
        return f"{name}: match@20={combined.get('match@20', 'NA')}, rows>=7 match@20={combined.get('rows>=7 match@20', 'NA')}, rows>=7 new positives={combined.get('rows_ge7_new_positives', 'NA')}"

    text = f"""# opentry_6 Final Summary

* 四个阶段是否全部完成：yes.
* 每个阶段核心结果：
  * {stage_line('stage1')}
  * {stage_line('stage2')}
  * {stage_line('stage3_refined')}
  * {stage_line('stage4')}
* 哪个阶段最有效：以 rows>=7 match@20 和 new positives 为准，见上方核心结果及 `opentry_6_experiment_log.md` 的 fold reports。
* rows>=7 是否真正提升：按固定顺序、无排序评估结果判断；若 rows>=7 new positives 为 0，则未证明真实提升。
* 是否仍然卡在 continuous geometry：若 GT W/A 条件下 rows>=7 仍低或为 0，则主要仍卡在 continuous geometry。
* 与 opentry_4 / opentry_5 相比是否进步：使用 opentry_5 E8028/E8034/E8036 rows>=7 positives 作为旧基线，new positives 记录在日志。
* 下一轮最应该继续的唯一方向：继续 CrystalFormer-style continuous geometry/refiner，但应扩大真实 canonical rows>=7 覆盖或重建 full train canonical geometry 数据，而不是回到 selector/ranker。
* 明确不要再走的弯路：selector、reranker、ranker、compatibility score、energy rejection、anchor-safe insertion、oracle selection、根据 match/RMSE/validity/logprob 筛选候选。
* GT-W/A 下 geometry 是否可以学会：见 stage1/stage2 GT W/A rows>=7 match@20；若 GT 仍失败，则不能证明。
* predicted/OOF W/A 是否是主要瓶颈：见 stage4 A/B/C 差异；若 A 成功 B/C 失败则 W/A gap 是主瓶颈，否则 geometry 仍是主瓶颈。
* fixed-step refiner 是否有用：见 stage3 raw vs refined；若 refined rows>=7 match/collision 未改善或简单结构变差，则 refiner 暂无用。
* 是否仍然陷入局部：若 stage2 unique rows>=7 仍只有 opentry_5 clean split 的有限数量，则仍受数据覆盖局部限制。
"""
    overwrite_md(SUMMARY_PATH, text)


def run_determinism_check(
    model: GeometryARNet,
    vocab: Vocab,
    engine: OrbitEngine,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    records: list[dict[str, Any]],
    aux: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    saved_force = args.force
    args.force = True
    sub_args = copy.copy(args)
    sub_args.num_candidates = min(5, int(args.num_candidates))
    subset = records[: min(64, len(records))]
    out: dict[str, Any] = {}
    for tag in ("a", "b"):
        run_name = f"determinism_stage1_{tag}"
        rows = generate_rows(
            run_name=run_name,
            records=subset,
            model=model,
            refiner=None,
            vocab=vocab,
            engine=engine,
            mean_t=mean_t,
            std_t=std_t,
            args=sub_args,
            condition="gt",
            aux=aux,
        )
        report = evaluate_generations(run_name, subset, rows, sub_args)
        out[tag] = {
            "generation_hash": sha256_file(ROOT / "eval" / run_name / "generations.jsonl"),
            "metrics_core_hash": metrics_signature(ROOT / "eval" / run_name / "metrics.jsonl"),
            "report": safe_report_summary(report),
        }
    args.force = saved_force
    out["generation_hash_identical"] = out["a"]["generation_hash"] == out["b"]["generation_hash"]
    out["metrics_core_hash_identical"] = out["a"]["metrics_core_hash"] == out["b"]["metrics_core_hash"]
    write_json(ROOT / "eval/determinism_check.json", out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--coord-components", type=int, default=4)
    parser.add_argument("--lattice-components", type=int, default=4)
    parser.add_argument("--epochs-stage1", type=int, default=6)
    parser.add_argument("--epochs-stage2", type=int, default=8)
    parser.add_argument("--epochs-refiner", type=int, default=6)
    parser.add_argument("--refiner-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=16)
    parser.add_argument("--bond-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=4.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(int(args.seed))
    ROOT.mkdir(parents=True, exist_ok=True)
    for rel in ("cache", "checkpoints", "eval", "logs"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    data = load_all_data()
    train_records: list[dict[str, Any]] = data["train"]
    fold_a: list[dict[str, Any]] = data["fold_a"]
    fold_b: list[dict[str, Any]] = data["fold_b"]
    if int(args.eval_limit) > 0:
        fold_a = fold_a[: int(args.eval_limit)]
        fold_b = fold_b[: int(args.eval_limit)]
    inv = inventory()
    data_stats = {
        "train": train_dev_stats(train_records),
        "fold_a": train_dev_stats(fold_a),
        "fold_b": train_dev_stats(fold_b),
        "opentry5_147_reference": "opentry_5 final reports cite 147 unique rows>=7 in the fold_b tuning branch; opentry_6 uses all clean train rows>=7 available without duplicating records.",
        "scale_note": "K=20/full fold was attempted but generation/rendering was too slow under current CPU/GPU contention; if num_candidates=10 or eval_limit=256 is set, it is the prompt-allowed minimum viable scale.",
    }
    init_log(inv, data_stats)

    all_records = train_records + fold_a + fold_b
    engine = OrbitEngine(str(lookup_path()), sg_symbols_from_records(all_records))
    oof = load_oof_predictions()
    exact, exact_stats = load_exact_candidates()
    aux_rows: list[list[dict[str, Any]]] = []
    for pred in list(oof.values())[:2000]:
        try:
            rows = rows_from_canonical_wa_key(str(pred.get("predicted_wa_key") or ""), engine)
            if rows:
                aux_rows.append(rows)
        except Exception:
            pass
    vocab = Vocab(all_records, engine, aux_rows=aux_rows)
    lat_mean, lat_std = lattice_stats(train_records)
    mean_t = torch.tensor(lat_mean, dtype=torch.float32)
    std_t = torch.tensor(lat_std, dtype=torch.float32)
    write_json(ROOT / "cache" / "vocab_stats.json", {
        "elements": len(vocab.element_to_id),
        "orbits": len(vocab.orbit_to_id),
        "letters": len(vocab.letter_to_id),
        "site_symmetries": len(vocab.sym_to_id),
        "max_rows_supported": vocab.max_rows_supported,
        "lattice_mean": lat_mean,
        "lattice_std": lat_std,
        "exact_cover_stats": exact_stats,
    })
    aux = {"oof_by_sample": oof, "exact_by_sample": exact}
    exact_stats["matched_opentry5_samples"] = sum(1 for r in fold_a + fold_b if str(r["sample_id"]) in exact)

    common_read_files = [
        str(data["fold_a_base"]),
        str(data["fold_b_base"]),
        str(OP5 / "data/oof_wa_predictions_dev.jsonl"),
        str(OP4 / "cache/hard_negative_dataset_train.jsonl"),
        str(lookup_path()),
    ]
    stage_reports: dict[str, Any] = {}

    stage1_model, stage1_info = train_geometry_model(
        name="stage1_gtwa_geometry",
        train_records=train_records,
        val_records=fold_a + fold_b,
        vocab=vocab,
        mean_t=mean_t,
        std_t=std_t,
        args=args,
        pair_weight=0.05,
        sep_weight=0.02,
        complex_sampler=False,
    )
    stage1_reports = {
        "stage1_fold_a": eval_run(run_name="stage1_fold_a", records=fold_a, model=stage1_model, refiner=None, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
        "stage1_fold_b": eval_run(run_name="stage1_fold_b", records=fold_b, model=stage1_model, refiner=None, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
    }
    stage_reports["stage1"] = stage1_reports
    det = run_determinism_check(stage1_model, vocab, engine, mean_t, std_t, fold_a, aux, args)
    append_md(
        LOG_PATH,
        stage_md(
            stage="Stage 1 / E9001",
            title="GT-W/A CrystalFormer-style geometry-only model",
            read_files=common_read_files,
            write_files=[safe_rel(ROOT / "checkpoints/stage1_gtwa_geometry/last.pt"), safe_rel(ROOT / "eval/stage1_fold_a"), safe_rel(ROOT / "eval/stage1_fold_b")],
            data_scope=f"train={len(train_records)} all clean train records; train rows>=7={data_stats['train']['rows_ge7']}; fold_a={len(fold_a)}; fold_b={len(fold_b)}",
            model_structure="CrystalFormer-style autoregressive orbit-conditioned geometry decoder; formula/SG/global lattice context; per-orbit W/A/element/multiplicity/site-sym/enumeration/free-mask and previous generated params; rows>=64 support",
            train_objective="Gaussian mixture lattice NLL + sin/cos mixture density circular free-parameter NLL + row-pair/separation auxiliary; fixed coordinates masked out",
            params={**vars(args), "stage1_info": stage1_info},
            reports=stage1_reports,
            conclusion="GT W/A geometry-only path trained and evaluated on both grouped folds with fixed candidate order.",
            failure="If rows>=7 match remains low, failure is continuous lattice/free-parameter geometry rather than W/A prediction.",
            next_step="Train complex-focused full unique rows>=7 variant without duplicating records.",
            extra={"determinism_check": det},
        ),
    )

    complex_records = [r for r in train_records if is_complex(r)]
    simple_records = [r for r in train_records if not is_complex(r)]
    # Keep all rows>=7 and all simple records in the dataset; the sampler, not duplicated files, changes batch balance.
    stage2_train = complex_records + simple_records
    stage2_model, stage2_info = train_geometry_model(
        name="stage2_complex_focused_geometry",
        train_records=stage2_train,
        val_records=fold_a + fold_b,
        vocab=vocab,
        mean_t=mean_t,
        std_t=std_t,
        args=args,
        pair_weight=0.15,
        sep_weight=0.08,
        complex_sampler=True,
    )
    stage2_reports = {
        "stage2_fold_a": eval_run(run_name="stage2_fold_a", records=fold_a, model=stage2_model, refiner=None, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
        "stage2_fold_b": eval_run(run_name="stage2_fold_b", records=fold_b, model=stage2_model, refiner=None, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
    }
    stage_reports["stage2"] = stage2_reports
    append_md(
        LOG_PATH,
        stage_md(
            stage="Stage 2 / E9002",
            title="Full unique complex rows>=7 data training without duplicate inflation",
            read_files=common_read_files,
            write_files=[safe_rel(ROOT / "checkpoints/stage2_complex_focused_geometry/last.pt"), safe_rel(ROOT / "eval/stage2_fold_a"), safe_rel(ROOT / "eval/stage2_fold_b")],
            data_scope=f"unique rows>=7 train count={len(complex_records)}; simple records retained={len(simple_records)}; sampler balanced batches without materializing duplicate JSONL records; stats={json.dumps(train_dev_stats(stage2_train), sort_keys=True)}",
            model_structure="Same geometry-only CrystalFormer-style decoder as Stage 1, trained with complex-focused weighted sampler.",
            train_objective="Same lattice/coordinate multimodal NLL plus stronger row-pair and collision/separation losses.",
            params={**vars(args), "unique_rows_ge7_train_count": len(complex_records), "stage2_info": stage2_info},
            reports=stage2_reports,
            conclusion="Complex-focused dataset and sampler executed on both folds.",
            failure="If rows>=7 remains 0, likely causes are insufficient true unique rows>=7 coverage in opentry_5 clean train, loss/model underfit, or continuous geometry ambiguity.",
            next_step="Train fixed-step symmetry-space refiner and compare raw vs refined.",
            extra={"complex_unique_requirement": ">=1000 not reached" if len(complex_records) < 1000 else ">=1000 reached"},
        ),
    )

    refiner, refiner_info = train_refiner(stage2_train, fold_a + fold_b, vocab, mean_t, std_t, args)
    stage3_refined_reports = {
        "stage3_refined_fold_a": eval_run(run_name="stage3_refined_fold_a", records=fold_a, model=stage2_model, refiner=refiner, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
        "stage3_refined_fold_b": eval_run(run_name="stage3_refined_fold_b", records=fold_b, model=stage2_model, refiner=refiner, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition="gt", aux=aux),
    }
    stage_reports["stage3_refined"] = stage3_refined_reports
    append_md(
        LOG_PATH,
        stage_md(
            stage="Stage 3 / E9003",
            title="Fixed-step symmetry-space learned geometry refiner",
            read_files=common_read_files,
            write_files=[safe_rel(ROOT / "checkpoints/stage3_refiner/last.pt"), safe_rel(ROOT / "eval/stage3_refined_fold_a"), safe_rel(ROOT / "eval/stage3_refined_fold_b")],
            data_scope=f"synthetic corruption over stage2_train={len(stage2_train)}; opentry_4 hard-negative stats={json.dumps(refiner_info.get('hard_negative_stats', {}), sort_keys=True)}",
            model_structure=f"Symmetry parameter-space GRU refiner; formula/SG/W/A fixed; fixed {args.refiner_steps} steps; no quality gating.",
            train_objective="Periodic coordinate loss + lattice metric/VPA loss + row-pair distance consistency + collision/short-contact penalty; hard-negative train file used as failure-mode audit because no reusable parameter payload was present.",
            params={**vars(args), "refiner_info": refiner_info},
            reports=stage3_refined_reports,
            conclusion="Refiner trained and attached to Stage 2 raw generations on both folds.",
            failure="If refined does not improve raw Stage 2, learned correction is underfit or initial generated geometry is outside the synthetic corruption manifold.",
            next_step="Run GT/OOF/exact-cover W/A condition-gap comparison using the same geometry model and fixed-step refiner.",
            extra={"raw_reference_reports": {k: safe_report_summary(v) for k, v in stage2_reports.items()}},
        ),
    )

    stage4_reports: dict[str, dict[str, Any]] = {}
    for condition, label in (("gt", "A_gtwa"), ("oof", "B_oofwa"), ("exact", "C_exactcover")):
        stage4_reports[f"stage4_{label}_fold_a"] = eval_run(run_name=f"stage4_{label}_fold_a", records=fold_a, model=stage2_model, refiner=refiner, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition=condition, aux=aux)
        stage4_reports[f"stage4_{label}_fold_b"] = eval_run(run_name=f"stage4_{label}_fold_b", records=fold_b, model=stage2_model, refiner=refiner, vocab=vocab, engine=engine, mean_t=mean_t, std_t=std_t, args=args, condition=condition, aux=aux)
    stage_reports["stage4"] = stage4_reports
    append_md(
        LOG_PATH,
        stage_md(
            stage="Stage 4 / E9004",
            title="GT / OOF / exact-cover W/A condition-gap full system",
            read_files=common_read_files + [str(SYMCIF_PROJECT / "reports/composition_exact_v1")],
            write_files=[safe_rel(ROOT / "eval/stage4_A_gtwa_fold_a"), safe_rel(ROOT / "eval/stage4_B_oofwa_fold_a"), safe_rel(ROOT / "eval/stage4_C_exactcover_fold_a"), safe_rel(ROOT / "eval/stage4_A_gtwa_fold_b"), safe_rel(ROOT / "eval/stage4_B_oofwa_fold_b"), safe_rel(ROOT / "eval/stage4_C_exactcover_fold_b")],
            data_scope=f"same fold_a={len(fold_a)} and fold_b={len(fold_b)} samples for A/B/C; exact-cover matched opentry5 samples={exact_stats.get('matched_opentry5_samples', 0)}, fallback to OOF when unavailable.",
            model_structure="Stage 2 geometry generator plus Stage 3 fixed-step refiner; W/A condition varies only by GT, OOF, or exact-cover/fallback.",
            train_objective="No additional training; fixed-order inference only.",
            params={**vars(args), "exact_cover_stats": exact_stats},
            reports=stage4_reports,
            conclusion="A/B/C condition comparison completed. If A succeeds and B/C fail, W/A gap dominates; if A also fails, continuous geometry remains the main bottleneck.",
            failure="Exact-cover files did not necessarily share sample_id namespace with opentry_5 folds; missing samples used OOF fallback and are marked per candidate.",
            next_step="Use the condition-gap result to choose either geometry data/model scaling or W/A condition improvement, not ranking-based repair.",
            extra={"A_B_C_keys": list(stage4_reports)},
        ),
    )

    final_summary(stage_reports)
    write_json(ROOT / "cache" / "stage_reports_index.json", {k: list(v.keys()) if isinstance(v, dict) else v for k, v in stage_reports.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
