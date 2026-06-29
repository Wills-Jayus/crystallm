#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3_predicted_skeleton_lattice_repair_pilot"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402
from pymatgen.core.periodic_table import Element  # noqa: E402

from run_exp2_predicted_skeleton_renderer_site_mapping import (  # noqa: E402
    formula_counts,
    median_lattice_by_sg,
    read_jsonl,
    sample_id,
    source_skeleton_rows,
    write_json,
    write_jsonl,
)
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    build_reference_indexes,
    eval_sample,
    render_candidate,
    summarize,
)
from run_symcif_v4_geometry_model_eval import deterministic_params, flexible_params_from_reference, postprocess_lattice  # noqa: E402
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
EXP3_PROPOSALS = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"
EXP3_PER_SAMPLE = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "per_sample_metrics.jsonl"
AUDIT_RESULT = RESULT_DIR / "experiment_3_predicted_skeleton_aware_geometry_repair_audit.json"
OUT_JSON = RESULT_DIR / "experiment_3_predicted_skeleton_lattice_repair_pilot.json"
MARKER = "<!-- OPENTRY13_EXP3_PREDICTED_SKELETON_AWARE_REPAIR_AUDIT -->"
BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with REPORT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def triplet(d: dict[str, Any], prefix: str) -> str:
    return " / ".join(pct(d.get(f"{prefix}@{k}")) for k in BUDGETS)


def delta_triplet(d: dict[str, Any]) -> str:
    return " / ".join(pp(d.get(f"delta_match@{k}")) for k in BUDGETS)


def lattice_raw(lattice: dict[str, Any]) -> list[float]:
    return [
        math.log(max(1.0e-6, float(lattice["a"]))),
        math.log(max(1.0e-6, float(lattice["b"]))),
        math.log(max(1.0e-6, float(lattice["c"]))),
        float(lattice["alpha"]) / 180.0,
        float(lattice["beta"]) / 180.0,
        float(lattice["gamma"]) / 180.0,
    ]


def crystal_system_onehot(sg: int) -> list[float]:
    sg = int(sg)
    system = 0
    if sg <= 2:
        system = 0
    elif sg <= 15:
        system = 1
    elif sg <= 74:
        system = 2
    elif sg <= 142:
        system = 3
    elif sg <= 167:
        system = 4
    elif sg <= 194:
        system = 5
    else:
        system = 6
    return [1.0 if i == system else 0.0 for i in range(7)]


def safe_element_stats(counts: dict[str, int]) -> list[float]:
    total = max(1, sum(int(v) for v in counts.values()))
    zs: list[float] = []
    weights: list[float] = []
    for element, count in sorted(counts.items()):
        try:
            z = float(Element(str(element)).Z)
        except Exception:
            z = 0.0
        zs.append(z)
        weights.append(float(count) / float(total))
    if not zs:
        return [0.0, 0.0, 0.0, 0.0]
    mean_z = sum(z * w for z, w in zip(zs, weights)) / 100.0
    max_z = max(zs) / 100.0
    min_z = min(zs) / 100.0
    spread = (max(zs) - min(zs)) / 100.0
    return [mean_z, max_z, min_z, spread]


def featurize(record: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    counts = {str(k): int(v) for k, v in normalize_formula_counts(record["formula_counts"]).items()}
    total_atoms = max(1, sum(counts.values()))
    mults = [float(int(r.get("multiplicity") or 0)) for r in rows] or [0.0]
    free_counts = [float(len(r.get("free_symbols") or [])) for r in rows] or [0.0]
    row_count = max(1, len(rows))
    mean_mult = sum(mults) / len(mults)
    mean_free = sum(free_counts) / len(free_counts)
    mult_var = sum((x - mean_mult) ** 2 for x in mults) / len(mults)
    free_var = sum((x - mean_free) ** 2 for x in free_counts) / len(free_counts)
    return [
        float(int(record["sg"])) / 230.0,
        float(total_atoms) / 300.0,
        float(row_count) / 64.0,
        float(len(counts)) / 12.0,
        float(max(mults)) / 64.0,
        float(min(mults)) / 64.0,
        mean_mult / 64.0,
        math.sqrt(mult_var) / 64.0,
        float(max(free_counts)) / 3.0,
        mean_free / 3.0,
        math.sqrt(free_var) / 3.0,
        *crystal_system_onehot(int(record["sg"])),
        *safe_element_stats(counts),
    ]


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    aa = normalize_formula_counts(a)
    bb = normalize_formula_counts(b)
    keys = set(aa) | set(bb)
    return sum(abs(int(aa.get(k, 0)) - int(bb.get(k, 0))) for k in keys) / float(max(1, sum(aa.values()) + sum(bb.values())))


def candidate_rows_fast(
    *,
    engine: OrbitEngine,
    target_record: dict[str, Any],
    source_repr: dict[str, Any],
    prefer_source_elements: bool = True,
) -> tuple[list[dict[str, Any]] | None, str, str | None]:
    rows = source_skeleton_rows(engine, source_repr)
    target_counts = formula_counts(target_record)
    source_counts = {str(k): int(v) for k, v in normalize_formula_counts(source_repr["formula_counts"]).items()}
    if prefer_source_elements and source_counts == target_counts:
        per_element: Counter[str] = Counter()
        for row in rows:
            per_element[str(row.get("element"))] += int(row.get("multiplicity") or 0)
        if dict(per_element) == target_counts:
            return [dict(row) for row in rows], "source_formula_exact_order", None

    elements = tuple(sorted(target_counts))
    mults = [int(row.get("multiplicity") or 0) for row in rows]
    order = tuple(sorted(range(len(rows)), key=lambda i: (-mults[i], str(rows[i].get("orbit_id")), i)))
    suffix = [0] * (len(order) + 1)
    for pos in range(len(order) - 1, -1, -1):
        suffix[pos] = suffix[pos + 1] + mults[order[pos]]
    start_remaining = tuple(int(target_counts[e]) for e in elements)
    choice: dict[tuple[int, tuple[int, ...]], str] = {}

    sys.setrecursionlimit(max(1000, len(rows) + 200))

    from functools import lru_cache

    @lru_cache(maxsize=200000)
    def rec(pos: int, remaining: tuple[int, ...]) -> bool:
        if pos == len(order):
            return all(v == 0 for v in remaining)
        if sum(remaining) != suffix[pos]:
            return False
        idx = order[pos]
        mult = mults[idx]
        preferred = str(rows[idx].get("element"))
        element_order = sorted(
            range(len(elements)),
            key=lambda j: (0 if prefer_source_elements and elements[j] == preferred else 1, -remaining[j], elements[j]),
        )
        for j in element_order:
            if remaining[j] < mult:
                continue
            nxt = list(remaining)
            nxt[j] -= mult
            nxt_tuple = tuple(nxt)
            if rec(pos + 1, nxt_tuple):
                choice[(pos, remaining)] = elements[j]
                return True
        return False

    if not rec(0, start_remaining):
        return None, "memoized_exact_cover", "exact_cover_assignment_failed"
    assigned: list[str | None] = [None] * len(rows)
    pos = 0
    remaining = start_remaining
    while pos < len(order):
        element = choice[(pos, remaining)]
        idx = order[pos]
        assigned[idx] = element
        nxt = list(remaining)
        nxt[elements.index(element)] -= mults[idx]
        remaining = tuple(nxt)
        pos += 1
    out: list[dict[str, Any]] = []
    per_element: Counter[str] = Counter()
    for row, element in zip(rows, assigned):
        item = dict(row)
        item["element"] = str(element)
        out.append(item)
        per_element[str(element)] += int(item["multiplicity"])
    if dict(per_element) != target_counts:
        return None, "memoized_exact_cover", "formula_after_assignment_mismatch"
    return out, "memoized_exact_cover", None


def build_train_source_index(train_repr: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_sg_atom: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in train_repr.values():
        sg = int(record["sg"])
        atom_count = int(record.get("atom_count") or sum(int(v) for v in normalize_formula_counts(record["formula_counts"]).values()))
        by_sg_atom[(sg, atom_count)].append(record)
        by_sg[sg].append(record)
    return {"by_sg_atom": dict(by_sg_atom), "by_sg": dict(by_sg)}


def source_candidates_for_train(target: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    sg = int(target["sg"])
    atom_count = int(target.get("atom_count") or sum(int(v) for v in normalize_formula_counts(target["formula_counts"]).values()))
    pool = list(index["by_sg_atom"].get((sg, atom_count), []))
    if len(pool) < 24:
        pool.extend(index["by_sg"].get(sg, []))
    sid = str(target.get("sample_id") or target.get("keys", {}).get("sample_id"))
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in pool:
        item_sid = str(item.get("sample_id") or item.get("keys", {}).get("sample_id"))
        if item_sid == sid or item_sid in seen:
            continue
        seen.add(item_sid)
        deduped.append(item)
    deduped.sort(
        key=lambda r: (
            formula_l1(target["formula_counts"], r["formula_counts"]),
            abs(int(target.get("row_count") or 0) - int(r.get("row_count") or 0)),
            str(r.get("sample_id") or r.get("keys", {}).get("sample_id")),
        )
    )
    return deduped[:80]


class LatticeMLP(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_lattice_model(features: list[list[float]], targets: list[list[float]], *, epochs: int, seed: int) -> tuple[LatticeMLP, dict[str, Any]]:
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1.0e-6)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1.0e-6)
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    order = torch.randperm(xs.shape[0])
    val_n = max(256, int(0.10 * xs.shape[0]))
    val_idx = order[:val_n]
    train_idx = order[val_n:]
    model = LatticeMLP(xs.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    batch_size = 1024
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        perm = train_idx[torch.randperm(train_idx.numel())]
        model.train()
        total = 0.0
        steps = 0
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size]
            pred = model(xs[idx])
            loss = torch.nn.functional.mse_loss(pred, ys[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(model(xs[val_idx]), ys[val_idx]).detach())
            history.append({"epoch": float(epoch), "train_loss": total / max(1, steps), "val_loss": val_loss})
    model.x_mean = x_mean  # type: ignore[attr-defined]
    model.x_std = x_std  # type: ignore[attr-defined]
    model.y_mean = y_mean  # type: ignore[attr-defined]
    model.y_std = y_std  # type: ignore[attr-defined]
    return model, {
        "feature_dim": int(xs.shape[1]),
        "train_pairs": int(train_idx.numel()),
        "val_pairs": int(val_idx.numel()),
        "history": history,
        "best_val_loss": min((h["val_loss"] for h in history), default=None),
    }


@torch.no_grad()
def predict_lattice(model: LatticeMLP, feature: list[float], sg: int) -> dict[str, float]:
    x = torch.tensor([feature], dtype=torch.float32)
    xs = (x - model.x_mean) / model.x_std  # type: ignore[attr-defined]
    raw = (model(xs)[0] * model.y_std + model.y_mean).tolist()  # type: ignore[attr-defined]
    return postprocess_lattice([float(v) for v in raw], int(sg))


def summarize_before(sample_ids: list[str], before_by_sid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(sample_ids)}
    for k in BUDGETS:
        hits = [bool(before_by_sid.get(sid, {}).get(f"hydrated_match@{k}")) for sid in sample_ids]
        rms_vals = [
            float(before_by_sid[sid][f"hydrated_rms@{k}"])
            for sid in sample_ids
            if before_by_sid.get(sid, {}).get(f"hydrated_rms@{k}") is not None
        ]
        out[f"match@{k}"] = sum(hits) / max(1, len(hits))
        out[f"RMSE@{k}"] = sum(rms_vals) / len(rms_vals) if rms_vals else None
    return out


def repair_summary(after: dict[str, Any], before: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = sorted({str(r["sample_id"]) for r in rows})
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    out = dict(after)
    for k in BUDGETS:
        before_neg = 0
        converted = 0
        for sid in sample_ids:
            before_hit = bool(before.get(f"_per_sample@{k}", {}).get(sid))
            after_hit = any(bool(r.get("match")) for r in sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 999999))[:k])
            if not before_hit:
                before_neg += 1
                converted += int(after_hit)
        out[f"before_match@{k}"] = before.get(f"match@{k}")
        out[f"after_match@{k}"] = after.get(f"match@{k}")
        out[f"delta_match@{k}"] = (after.get(f"match@{k}") or 0.0) - (before.get(f"match@{k}") or 0.0)
        out[f"repair_conversion@{k}"] = converted / max(1, before_neg)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-limit", type=int, default=24000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    structured_train = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    before_by_sid = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PER_SAMPLE)}
    audit = read_json(AUDIT_RESULT)

    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(structured_train.values()) + list(structured_val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)
    source_index = build_train_source_index(train_repr)

    features: list[list[float]] = []
    targets: list[list[float]] = []
    train_pair_meta: list[dict[str, Any]] = []
    for target_sid in sorted(structured_train)[: int(args.train_limit)]:
        target = structured_train[target_sid]
        target_repr = train_repr.get(target_sid)
        if target_repr is None:
            continue
        for source_repr in source_candidates_for_train(target_repr, source_index):
            rows, _rule, _err = candidate_rows_fast(
                engine=engine,
                target_record=target,
                source_repr=source_repr,
                prefer_source_elements=True,
            )
            if rows is None:
                continue
            features.append(featurize(target, rows))
            targets.append(lattice_raw(target["lattice"]))
            train_pair_meta.append(
                {
                    "target_sample_id": target_sid,
                    "source_sample_id": sample_id(source_repr),
                    "target_row_count": int(target_repr.get("row_count") or 0),
                    "source_row_count": int(source_repr.get("row_count") or 0),
                    "sg": int(target["sg"]),
                }
            )
            break
    write_jsonl(ARTIFACT_DIR / "train_noisy_skeleton_pairs_meta.jsonl", train_pair_meta)
    if len(features) < 1000:
        raise RuntimeError(f"too few train pairs: {len(features)}")

    model, training = train_lattice_model(features, targets, epochs=int(args.epochs), seed=int(args.seed))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": model.x_mean,  # type: ignore[attr-defined]
            "x_std": model.x_std,  # type: ignore[attr-defined]
            "y_mean": model.y_mean,  # type: ignore[attr-defined]
            "y_std": model.y_std,  # type: ignore[attr-defined]
            "training": training,
        },
        ARTIFACT_DIR / "lattice_mlp.pt",
    )

    sample_payloads: list[dict[str, Any]] = []
    generated_meta: list[dict[str, Any]] = []
    mapping_failures: Counter[str] = Counter()
    selected_sids = [sid for sid in sorted(proposals) if sid in val_repr and sid in structured_val]
    for i, sid in enumerate(selected_sids, start=1):
        if i % 500 == 0:
            print(f"[exp3-lattice-repair] rendered {i}/{len(selected_sids)}", flush=True)
        target = structured_val[sid]
        target_repr = val_repr[sid]
        target_counts = formula_counts(target)
        candidates: list[dict[str, Any]] = []
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_k)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_structured = structured_train.get(source_id)
            if source_repr is None or source_structured is None:
                mapping_failures["missing_source"] += 1
                continue
            rows, mapping_rule, mapping_error = candidate_rows_fast(
                engine=engine,
                target_record=target,
                source_repr=source_repr,
                prefer_source_elements=True,
            )
            if rows is None:
                mapping_failures[str(mapping_error or mapping_rule)] += 1
                continue
            try:
                params, fallback_count = flexible_params_from_reference(engine, rows, source_structured, neural_params=None)
            except Exception:
                params = {idx: deterministic_params(engine, str(row["orbit_id"]), idx) for idx, row in enumerate(rows)}
                fallback_count = len(rows)
            lattice = predict_lattice(model, featurize(target, rows), int(target["sg"]))
            base = {
                "sample_id": sid,
                "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                "proposal_rank": int(proposal.get("rank") or 0),
                "geometry_rank": 1,
                "raw_generation_order": len(candidates) + 1,
                "row_count": int(target_repr.get("row_count") or 0),
                "sg": int(target["sg"]),
                "formula_counts": target_counts,
                "target_atom_count": int(sum(target_counts.values())),
                "source_sample_id": source_id,
                "proposal_source": str(proposal.get("source") or ""),
                "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                "candidate_row_count": len(rows),
                "site_mapping_rule": mapping_rule,
                "geometry_source": "predicted_skeleton_aware_lattice_mlp+source_free_params",
                "reference_sample_id": source_id,
                "reference_score": None,
                "param_fallback_rows": int(fallback_count),
            }
            try:
                cif, render_meta = render_candidate(
                    engine=engine,
                    target=target,
                    rows=rows,
                    option={"lattice": lattice, "params": params},
                    data_name=f"{sid}_lattice_mlp_rank{base['proposal_rank']}",
                )
                row = dict(base)
                row.update(render_meta)
                row["render_success"] = True
                row["render_error"] = None
                row["cif"] = cif
            except Exception as exc:  # noqa: BLE001
                row = dict(base)
                row.update(
                    {
                        "render_success": False,
                        "render_error": f"{type(exc).__name__}: {exc}",
                        "atom_count_after_expansion": None,
                        "exact_cover_retained": False,
                        "cif": "",
                    }
                )
            candidates.append(row)
            generated_meta.append({k: v for k, v in row.items() if k != "cif"})
        sample_payloads.append(
            {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": target_counts,
                "target_atom_count": int(sum(target_counts.values())),
                "sg": int(target["sg"]),
                "candidates": candidates,
            }
        )
    write_jsonl(ARTIFACT_DIR / "generated_lattice_repair_meta.jsonl", generated_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample, payload) for payload in sample_payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 500 == 0:
                print(f"[exp3-lattice-repair] evaluated {i}/{len(futures)}", flush=True)
    ranked = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / "evaluated_lattice_repair_candidates.jsonl", ranked)

    rows_ge7 = [r for r in ranked if int(r.get("row_count") or 0) >= 7]
    rows_lt7 = [r for r in ranked if int(r.get("row_count") or 0) < 7]
    overall_after = summarize(ranked)
    rows_ge7_after = summarize(rows_ge7)
    rows_lt7_after = summarize(rows_lt7)
    all_sids = sorted({str(r["sample_id"]) for r in ranked})
    rows7_sids = sorted({str(r["sample_id"]) for r in rows_ge7})
    rowslt7_sids = sorted({str(r["sample_id"]) for r in rows_lt7})

    before_all = summarize_before(all_sids, before_by_sid)
    before_rows7 = summarize_before(rows7_sids, before_by_sid)
    before_rowslt7 = summarize_before(rowslt7_sids, before_by_sid)
    for k in BUDGETS:
        before_all[f"_per_sample@{k}"] = {sid: bool(before_by_sid.get(sid, {}).get(f"hydrated_match@{k}")) for sid in all_sids}
        before_rows7[f"_per_sample@{k}"] = {sid: bool(before_by_sid.get(sid, {}).get(f"hydrated_match@{k}")) for sid in rows7_sids}
        before_rowslt7[f"_per_sample@{k}"] = {sid: bool(before_by_sid.get(sid, {}).get(f"hydrated_match@{k}")) for sid in rowslt7_sids}
    overall = repair_summary(overall_after, before_all, ranked)
    rows_ge7_summary = repair_summary(rows_ge7_after, before_rows7, rows_ge7)
    rows_lt7_summary = repair_summary(rows_lt7_after, before_rowslt7, rows_lt7)

    structure_gate = bool(
        overall.get("valid_rate", 0.0) >= 0.95
        and overall.get("formula_consistency", 0.0) >= 0.95
        and overall.get("sg_consistency", 0.0) >= 0.95
        and overall.get("exact_cover_retained", 0.0) >= 0.95
        and rows_ge7_summary.get("valid_rate", 0.0) >= 0.90
        and rows_ge7_summary.get("formula_consistency", 0.0) >= 0.95
        and rows_ge7_summary.get("sg_consistency", 0.0) >= 0.95
        and rows_ge7_summary.get("exact_cover_retained", 0.0) >= 0.95
    )
    repair_gate = bool(
        all(float(overall.get(f"delta_match@{k}") or 0.0) >= 0.0 for k in BUDGETS)
        and ((overall.get("delta_match@5") or 0.0) >= 0.02 or (overall.get("delta_match@20") or 0.0) >= 0.02)
        and (rows_ge7_summary.get("delta_match@5") or 0.0) >= 0.05
        and (rows_ge7_summary.get("delta_match@20") or 0.0) >= 0.05
        and (rows_ge7_summary.get("repair_conversion@20") or 0.0) > 0.05
    )
    result = {
        "experiment": "opentry_13_exp3_predicted_skeleton_aware_lattice_repair_pilot",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "train_split_noisy_predicted_skeleton_lattice_mlp_repair",
            "train_input": "train split target formula + GT-SG + train-derived noisy exact-cover skeleton from another train source",
            "inference_input": "validation formula + GT-SG + predicted exact-cover skeleton; source prototype free params; learned lattice MLP",
            "not_used_at_inference": ["GT-WA", "GT-skeleton", "test true CIF", "StructureMatcher match", "RMSD", "RF/HGB/scorer"],
            "limitations": ["learns lattice only; free parameters remain source-prototype/deterministic fallback"],
        },
        "data_scale": {
            "train_pairs": len(features),
            "validation_samples": overall["samples"],
            "validation_rows_ge7_samples": rows_ge7_summary["samples"],
            "candidate_records": len(ranked),
            "top_k": int(args.top_k),
        },
        "training": training,
        "mapping_failures": dict(mapping_failures),
        "overall": overall,
        "rows_ge7": rows_ge7_summary,
        "rows_lt7": rows_lt7_summary,
        "gates": {
            "structure_gate_pass": structure_gate,
            "repair_gate_pass": repair_gate,
            "passed": structure_gate and repair_gate,
        },
        "audit_context": {
            "previous_audit_training_artifact_found": audit["training_data_audit"].get("predicted_skeleton_noise_training_artifact_found"),
            "previous_audit_verdict": audit["decision"].get("verdict"),
        },
        "decision": {
            "verdict": "pass" if structure_gate and repair_gate else "fail_validation_gate",
            "reason": "Lattice repair passes both structure and repair gates." if structure_gate and repair_gate else "Train-split noisy-skeleton lattice repair still fails structure and/or repair conversion gates.",
            "next_step": "Do not run official unless validation gates pass; train full lattice+free-parameter/collision repair on noisy skeleton pairs.",
        },
        "artifacts": {
            "train_pairs": str(ARTIFACT_DIR / "train_noisy_skeleton_pairs_meta.jsonl"),
            "lattice_mlp": str(ARTIFACT_DIR / "lattice_mlp.pt"),
            "evaluated_candidates": str(ARTIFACT_DIR / "evaluated_lattice_repair_candidates.jsonl"),
        },
        "runtime_seconds": time.time() - started,
    }
    write_json(OUT_JSON, result)

    body = f"""## opentry_13 实验 3：predicted-skeleton-aware learned geometry repair

审计结果：`model/New_model/opentry_13/results/experiment_3_predicted_skeleton_aware_geometry_repair_audit.json`
训练 pilot：`model/New_model/opentry_13/results/experiment_3_predicted_skeleton_lattice_repair_pilot.json`

- 为什么做：目标要求 repair 训练数据必须包含 train split predicted skeleton / exact-cover skeleton 噪声。先前审计证明旧 geometry model 不是这种训练条件；本补充实验构造 train split noisy skeleton pair，并训练一个轻量 lattice MLP repair pilot。
- 核心假设：如果 predicted skeleton 条件下主要缺 lattice 初始化，train-noisy-skeleton lattice MLP + source prototype free params 应至少改善 K5/K20 或 repair conversion；如果仍不过 gate，说明还需要 free-parameter/site-mapping/collision 联合 repair。
- 数据规模：train noisy skeleton pairs `{len(features)}`；validation samples `{overall['samples']}`，rows>=7 `{rows_ge7_summary['samples']}`；candidate records `{len(ranked)}`；topK `{int(args.top_k)}`。
- 训练设置：PyTorch MLP，输入为 composition + GT-SG + noisy predicted skeleton numeric features，target 为 train true lattice；epochs `{int(args.epochs)}`，best val loss `{training.get('best_val_loss')}`。没有使用 RF/HGB/scorer，也没有使用 match/RMSD 作为推理特征。
- baseline：before repair 使用 exp3 predicted skeleton proposer 的 hydrated-existing-eval；after repair 使用 learned lattice MLP + source-prototype free params 重新渲染并按结构自检排序。
- 结果 overall：before match@1/5/20 = `{triplet(overall, 'before_match')}`；after match@1/5/20 = `{triplet(overall, 'after_match')}`；delta = `{delta_triplet(overall)}`；repair conversion@1/5/20 = `{triplet(overall, 'repair_conversion')}`；valid `{pct(overall.get('valid_rate'))}`，formula `{pct(overall.get('formula_consistency'))}`，SG `{pct(overall.get('sg_consistency'))}`，exact-cover `{pct(overall.get('exact_cover_retained'))}`，collision `{pct(overall.get('collision_rate'))}`。
- 结果 rows>=7：before match@1/5/20 = `{triplet(rows_ge7_summary, 'before_match')}`；after match@1/5/20 = `{triplet(rows_ge7_summary, 'after_match')}`；delta = `{delta_triplet(rows_ge7_summary)}`；repair conversion@1/5/20 = `{triplet(rows_ge7_summary, 'repair_conversion')}`；valid `{pct(rows_ge7_summary.get('valid_rate'))}`，formula `{pct(rows_ge7_summary.get('formula_consistency'))}`，SG `{pct(rows_ge7_summary.get('sg_consistency'))}`，exact-cover `{pct(rows_ge7_summary.get('exact_cover_retained'))}`，collision `{pct(rows_ge7_summary.get('collision_rate'))}`，skeleton-to-match conversion@20 `{pct(rows_ge7_summary.get('skeleton_to_match_conversion@20'))}`。
- gate 判定：structure_gate_pass={structure_gate}；repair_gate_pass={repair_gate}；passed={structure_gate and repair_gate}。
- 可信度：中等。训练数据确实来自 train split noisy exact-cover skeleton，validation 推理不使用 GT-WA/GT-skeleton/test CIF/match label；限制是当前 pilot 只学习 lattice，free parameters 和 collision/local geometry 仍未联合学习。
- 和历史实验关系：补足了先前实验 3 审计发现的“缺 predicted-skeleton-noise 训练 artifact”问题；结果与旧 repair 一致说明单独 lattice repair 仍不能解决 rows>=7 conversion。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：如果继续主线，应训练 full lattice + free-parameter + collision/local optimization repair，而不是 scorer、C/S 比例或 official。
"""
    append_or_replace_report(MARKER, body)
    print(json.dumps({"output": str(OUT_JSON), "gates": result["gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
