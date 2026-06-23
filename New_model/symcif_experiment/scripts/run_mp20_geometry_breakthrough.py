#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine, cell_volume  # noqa: E402

import run_mp20_minicfjoint_v2 as v2  # noqa: E402


PREVIOUS_BASELINE = {
    "match@1": 0.5573448546739984,
    "match@5": 0.5887666928515318,
    "RMSE@1": 0.1739349169633023,
    "RMSE@5": 0.168429924024472,
}


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "e00_prev_deterministic",
        "mode": "baseline_geometry_breakthrough_e00_prev_deterministic",
        "family": "baseline",
        "description": "Previous deterministic Mini-CFJoint-v2 GT-WA geometry head with rank-0 plus wrapped-coordinate jitter.",
    },
    {
        "id": "e01_volume_scaled_sg_knn",
        "mode": "baseline_geometry_breakthrough_e01_volume_scaled_sg_knn",
        "family": "A_target_normalization",
        "strategy": "sg_formula_knn",
        "lattice": "volume_scaled_source",
        "rank0_baseline": False,
        "description": "Same-SG formula/W/A kNN; copy train free params and scale source lattice volume by target atom count.",
    },
    {
        "id": "e02_sg_row_knn_direct",
        "mode": "baseline_geometry_breakthrough_e02_sg_row_knn_direct",
        "family": "C_codebook_retrieval",
        "strategy": "sg_row_formula_knn",
        "lattice": "source_direct",
        "rank0_baseline": False,
        "description": "Same-SG and row-count kNN retrieval; direct train lattice and row-position free-param transfer.",
    },
    {
        "id": "e03_signature_codebook_scaled",
        "mode": "baseline_geometry_breakthrough_e03_signature_codebook_scaled",
        "family": "C_codebook_retrieval",
        "strategy": "signature_codebook",
        "lattice": "volume_scaled_source",
        "rank0_baseline": False,
        "description": "Skeleton/signature codebook when available, otherwise SG-row fallback; volume-scaled lattice.",
    },
    {
        "id": "e04_quantile_modes",
        "mode": "baseline_geometry_breakthrough_e04_quantile_modes",
        "family": "B_multimodal",
        "strategy": "quantile_modes",
        "lattice": "quantile_stats",
        "rank0_baseline": False,
        "description": "Distributional nonparametric quantile head: per-SG/atom-bucket lattice quantiles and per-orbit free-param quantiles.",
    },
    {
        "id": "e05_baseline_plus_quantile",
        "mode": "baseline_geometry_breakthrough_e05_baseline_plus_quantile",
        "family": "B_multimodal",
        "strategy": "quantile_modes",
        "lattice": "quantile_stats",
        "rank0_baseline": True,
        "description": "Rank-0 deterministic geometry plus learned train-set quantile modes for K=5 diversity.",
    },
    {
        "id": "e06_complex_moe_knn",
        "mode": "baseline_geometry_breakthrough_e06_complex_moe_knn",
        "family": "D_complex_curriculum_moe",
        "strategy": "complex_moe",
        "lattice": "volume_scaled_source",
        "rank0_baseline": True,
        "description": "Complexity-routed mixture: keep deterministic rank-0 and use retrieval/codebook candidates for atom_count>=6/12.",
    },
    {
        "id": "e07_row_conditioned_knn",
        "mode": "baseline_geometry_breakthrough_e07_row_conditioned_knn",
        "family": "E_stronger_conditioning",
        "strategy": "row_conditioned_knn",
        "lattice": "volume_scaled_source",
        "rank0_baseline": False,
        "description": "Row-conditioned retrieval using element, orbit, multiplicity, free-symbol, SG, atom-count, and composition features.",
    },
    {
        "id": "e08_baseline_plus_row_knn",
        "mode": "baseline_geometry_breakthrough_e08_baseline_plus_row_knn",
        "family": "F_candidate_generation",
        "strategy": "row_conditioned_knn",
        "lattice": "volume_scaled_source",
        "rank0_baseline": True,
        "description": "Rank-0 deterministic geometry plus row-conditioned kNN candidates as learned geometry-mode candidates.",
    },
]


def selected_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = str(getattr(args, "experiment_ids", "") or "").strip()
    if not raw:
        return EXPERIMENTS
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    return [exp for exp in EXPERIMENTS if str(exp["id"]) in wanted or str(exp["mode"]) in wanted]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def num(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def atom_count(record: dict[str, Any]) -> int:
    return int(record.get("atom_count") or sum(int(v) for v in record["formula_counts"].values()))


def row_count(record: dict[str, Any]) -> int:
    return len(record.get("wa_table") or [])


def atom_bucket(n: int) -> str:
    n = int(n)
    if n < 6:
        return "lt6"
    if n < 12:
        return "ge6_lt12"
    return "ge12"


def row_bucket(n: int) -> str:
    n = int(n)
    if n <= 3:
        return "rows_1_3"
    if n <= 6:
        return "rows_4_6"
    return "rows_ge7"


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    return "cubic"


def formula_frac(record: dict[str, Any]) -> dict[str, float]:
    counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
    total = max(1, sum(counts.values()))
    return {k: float(v) / float(total) for k, v in counts.items()}


def formula_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def mult_signature(record: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(row.get("multiplicity", 1)) for row in v2.canonical_rows(record))


def free_signature(record: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for row in v2.canonical_rows(record):
        syms = "".join(sorted(str(s) for s in row.get("free_symbols") or []))
        out.append(f"{int(row.get('multiplicity', 1))}:{syms}")
    return tuple(out)


def row_condition_distance(target: dict[str, Any], source: dict[str, Any]) -> float:
    t_rows = v2.canonical_rows(target)
    s_rows = v2.canonical_rows(source)
    dist = 0.0
    dist += 0.03 * abs(atom_count(target) - atom_count(source))
    dist += 0.20 * abs(len(t_rows) - len(s_rows))
    dist += 1.0 * formula_l1(formula_frac(target), formula_frac(source))
    if int(target["sg"]) != int(source["sg"]):
        dist += 4.0
    for idx in range(max(len(t_rows), len(s_rows))):
        if idx >= len(t_rows) or idx >= len(s_rows):
            dist += 0.7
            continue
        tr = t_rows[idx]
        sr = s_rows[idx]
        if str(tr.get("orbit_id")) != str(sr.get("orbit_id")):
            dist += 0.35
        if str(tr.get("element")) != str(sr.get("element")):
            dist += 0.10
        if int(tr.get("multiplicity", 1)) != int(sr.get("multiplicity", 1)):
            dist += 0.20
        if set(tr.get("free_symbols") or []) != set(sr.get("free_symbols") or []):
            dist += 0.20
    return dist


def source_distance(target: dict[str, Any], source: dict[str, Any], strategy: str) -> float:
    base = formula_l1(formula_frac(target), formula_frac(source))
    base += 0.025 * abs(atom_count(target) - atom_count(source))
    base += 0.075 * abs(row_count(target) - row_count(source))
    if strategy in {"row_conditioned_knn", "complex_moe"}:
        base += row_condition_distance(target, source)
    elif strategy in {"signature_codebook"}:
        base += 0.25 * (mult_signature(target) != mult_signature(source))
        base += 0.25 * (free_signature(target) != free_signature(source))
    return float(base)


def cheap_source_distance(target: dict[str, Any], source: dict[str, Any]) -> float:
    return (
        formula_l1(formula_frac(target), formula_frac(source))
        + 0.025 * abs(atom_count(target) - atom_count(source))
        + 0.075 * abs(row_count(target) - row_count(source))
        + (0.0 if int(target["sg"]) == int(source["sg"]) else 4.0)
    )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


class GeometryIndex:
    def __init__(self, train_records: list[dict[str, Any]], val_records: list[dict[str, Any]]) -> None:
        self.train_records = train_records
        self.val_records = val_records
        self.by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.by_sg_row: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        self.by_signature: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
        self.by_crystal_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.by_sg_atom_bucket: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for record in train_records:
            sg = int(record["sg"])
            self.by_sg[sg].append(record)
            self.by_sg_row[(sg, row_count(record))].append(record)
            self.by_signature[(sg, free_signature(record))].append(record)
            self.by_crystal_bucket[(crystal_system(sg), atom_bucket(atom_count(record)))].append(record)
            self.by_sg_atom_bucket[(sg, atom_bucket(atom_count(record)))].append(record)
        self.free_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.global_free_stats: dict[str, list[float]] = defaultdict(list)
        self.lattice_stats: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        self.sg_lattice_stats: dict[int, list[dict[str, float]]] = defaultdict(list)
        for record in train_records:
            sg = int(record["sg"])
            self.sg_lattice_stats[sg].append(record["lattice"])
            self.lattice_stats[(str(sg), atom_bucket(atom_count(record)))].append(record["lattice"])
            self.lattice_stats[(crystal_system(sg), atom_bucket(atom_count(record)))].append(record["lattice"])
            for row in v2.canonical_rows(record):
                oid = str(row["orbit_id"])
                for sym, value in dict(row.get("free_params") or {}).items():
                    self.free_stats[(oid, str(sym))].append(float(value) % 1.0)
                    self.global_free_stats[str(sym)].append(float(value) % 1.0)

    def candidates(self, target: dict[str, Any], strategy: str, k: int) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        pools: list[list[dict[str, Any]]] = []
        if strategy == "sg_formula_knn":
            pools = [self.by_sg.get(sg, []), self.by_crystal_bucket.get((crystal_system(sg), atom_bucket(atom_count(target))), []), self.train_records]
        elif strategy == "sg_row_formula_knn":
            pools = [self.by_sg_row.get((sg, row_count(target)), []), self.by_sg.get(sg, []), self.train_records]
        elif strategy == "signature_codebook":
            pools = [self.by_signature.get((sg, free_signature(target)), []), self.by_sg_row.get((sg, row_count(target)), []), self.by_sg.get(sg, []), self.train_records]
        elif strategy == "row_conditioned_knn":
            pools = [self.by_sg_row.get((sg, row_count(target)), []), self.by_sg.get(sg, []), self.by_crystal_bucket.get((crystal_system(sg), atom_bucket(atom_count(target))), []), self.train_records]
        elif strategy == "complex_moe":
            if atom_count(target) >= 12:
                pools = [self.by_signature.get((sg, free_signature(target)), []), self.by_sg_row.get((sg, row_count(target)), []), self.by_crystal_bucket.get((crystal_system(sg), "ge12"), []), self.by_sg.get(sg, [])]
            elif atom_count(target) >= 6:
                pools = [self.by_sg_row.get((sg, row_count(target)), []), self.by_sg_atom_bucket.get((sg, "ge6_lt12"), []), self.by_sg.get(sg, [])]
            else:
                pools = [self.by_sg.get(sg, [])]
        else:
            pools = [self.by_sg.get(sg, []), self.train_records]
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []
        max_pool = max(k * 40, 200)
        for source_pool in pools:
            reduced_pool = source_pool
            if len(reduced_pool) > max_pool:
                reduced_pool = sorted(reduced_pool, key=lambda rec: cheap_source_distance(target, rec))[:max_pool]
            for rec in reduced_pool:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= max(k * 12, 32):
                break
        scored = sorted(pool, key=lambda rec: source_distance(target, rec, strategy))
        return scored[: max(k, 1)]

    def free_value(self, orbit_id: str, symbol: str, q: float) -> float:
        values = self.free_stats.get((str(orbit_id), str(symbol))) or self.global_free_stats.get(str(symbol)) or [0.0]
        return quantile(values, q) % 1.0

    def lattice_quantile(self, record: dict[str, Any], q: float) -> dict[str, float]:
        sg = int(record["sg"])
        pools = [
            self.lattice_stats.get((str(sg), atom_bucket(atom_count(record))), []),
            self.sg_lattice_stats.get(sg, []),
            self.lattice_stats.get((crystal_system(sg), atom_bucket(atom_count(record))), []),
        ]
        pool: list[dict[str, float]] = []
        for item in pools:
            if item:
                pool = item
                break
        if not pool:
            pool = [r["lattice"] for r in self.train_records]
        return {
            key: quantile([float(lat[key]) for lat in pool], q)
            for key in ("a", "b", "c", "alpha", "beta", "gamma")
        }


def scale_lattice_to_target(source: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    lattice = {k: float(v) for k, v in source["lattice"].items()}
    src_atoms = max(1, atom_count(source))
    tgt_atoms = max(1, atom_count(target))
    factor = (float(tgt_atoms) / float(src_atoms)) ** (1.0 / 3.0)
    out = dict(lattice)
    for key in ("a", "b", "c"):
        out[key] = max(0.5, float(out[key]) * factor)
    return out


def source_lattice(source: dict[str, Any], target: dict[str, Any], mode: str) -> dict[str, float]:
    if mode == "volume_scaled_source":
        return scale_lattice_to_target(source, target)
    return {k: float(v) for k, v in source["lattice"].items()}


def params_from_source(index: GeometryIndex, target: dict[str, Any], source: dict[str, Any], rank: int) -> dict[int, dict[str, float]]:
    params: dict[int, dict[str, float]] = {}
    t_rows = v2.canonical_rows(target)
    s_rows = v2.canonical_rows(source)
    q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9]
    q = q_by_rank[min(rank, len(q_by_rank) - 1)]
    for row_idx, t_row in enumerate(t_rows):
        row_params: dict[str, float] = {}
        source_params = dict(s_rows[row_idx].get("free_params") or {}) if row_idx < len(s_rows) else {}
        for symbol in t_row.get("free_symbols") or []:
            sym = str(symbol)
            if sym in source_params:
                row_params[sym] = float(source_params[sym]) % 1.0
            else:
                row_params[sym] = index.free_value(str(t_row["orbit_id"]), sym, q)
        params[row_idx] = row_params
    return params


def params_from_quantiles(index: GeometryIndex, target: dict[str, Any], rank: int) -> dict[int, dict[str, float]]:
    q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9]
    q = q_by_rank[min(rank, len(q_by_rank) - 1)]
    params: dict[int, dict[str, float]] = {}
    for row_idx, row in enumerate(v2.canonical_rows(target)):
        row_params: dict[str, float] = {}
        for symbol in row.get("free_symbols") or []:
            row_params[str(symbol)] = index.free_value(str(row["orbit_id"]), str(symbol), q)
        params[row_idx] = row_params
    return params


def candidate_hash(rendered: dict[str, Any]) -> str:
    return hashlib.sha1(str(rendered.get("cif") or "").encode("utf-8", errors="ignore")).hexdigest()


def make_generation_line(
    *,
    mode: str,
    record: dict[str, Any],
    sample_index: int,
    rank: int,
    rendered: dict[str, Any],
    source: str,
    source_sample_id: str | None = None,
    lattice: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "sample_index": sample_index,
        "sample_id": record["sample_id"],
        "gen_index": rank,
        "seed": rank,
        "raw_generation_success": bool(rendered["ok"]),
        "generated_text": rendered["cif"],
        "error": rendered.get("error"),
        "atom_count_ok": bool(rendered.get("atom_count_ok")),
        "formula_closure_success": True,
        "geometry_source": source,
        "source_sample_id": source_sample_id,
        "lattice_volume": cell_volume(lattice) if lattice else None,
    }


def render_retrieval_candidate(
    *,
    engine: OrbitEngine,
    index: GeometryIndex,
    record: dict[str, Any],
    source: dict[str, Any],
    lattice_mode: str,
    rank: int,
    source_name: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    lattice = source_lattice(source, record, lattice_mode)
    candidate = {
        "rows": v2.canonical_rows(record),
        "params": params_from_source(index, record, source, rank),
        "lattice": lattice,
    }
    return v2.render_candidate(engine, record, candidate, rank, source_name), lattice


def render_quantile_candidate(
    *,
    engine: OrbitEngine,
    index: GeometryIndex,
    record: dict[str, Any],
    rank: int,
    source_name: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9]
    q = q_by_rank[min(rank, len(q_by_rank) - 1)]
    lattice = index.lattice_quantile(record, q)
    candidate = {
        "rows": v2.canonical_rows(record),
        "params": params_from_quantiles(index, record, rank),
        "lattice": lattice,
    }
    return v2.render_candidate(engine, record, candidate, rank, source_name), lattice


def load_baseline_rank0(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("gen_index", -1)) == 0:
                out[int(row["sample_index"])] = row
    return out


def load_baseline_top5(path: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("gen_index", 999)) < 5:
                out[int(row["sample_index"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    return out


def copy_baseline_line(row: dict[str, Any], mode: str, sample_index: int, rank: int) -> dict[str, Any]:
    out = dict(row)
    out["mode"] = mode
    out["sample_index"] = sample_index
    out["gen_index"] = rank
    out["seed"] = rank
    out["geometry_source"] = "previous_minicfjoint_v2_rank0"
    return out


def generate_for_experiment(
    *,
    exp: dict[str, Any],
    engine: OrbitEngine,
    index: GeometryIndex,
    val_records: list[dict[str, Any]],
    baseline_rank0: dict[int, dict[str, Any]],
    baseline_top5: dict[int, list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    mode = str(exp["mode"])
    if exp["id"] == "e00_prev_deterministic":
        rows: list[dict[str, Any]] = []
        for sample_index, record in enumerate(val_records):
            for rank, old in enumerate(baseline_top5.get(sample_index, [])[:top_k]):
                rows.append(copy_baseline_line(old, mode, sample_index, rank))
        return rows

    rows = []
    for sample_index, record in enumerate(val_records):
        rank = 0
        seen_hashes: set[str] = set()
        if exp.get("rank0_baseline") and sample_index in baseline_rank0:
            line = copy_baseline_line(baseline_rank0[sample_index], mode, sample_index, rank)
            rows.append(line)
            seen_hashes.add(hashlib.sha1(str(line.get("generated_text") or "").encode("utf-8", errors="ignore")).hexdigest())
            rank += 1

        strategy = str(exp.get("strategy") or "")
        if strategy == "quantile_modes":
            source_count = top_k - rank
            for _ in range(source_count):
                rendered, lattice = render_quantile_candidate(engine=engine, index=index, record=record, rank=rank, source_name=exp["id"])
                h = candidate_hash(rendered)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    rows.append(
                        make_generation_line(
                            mode=mode,
                            record=record,
                            sample_index=sample_index,
                            rank=rank,
                            rendered=rendered,
                            source=str(exp["id"]),
                            lattice=lattice,
                        )
                    )
                    rank += 1
                if rank >= top_k:
                    break
        else:
            sources = index.candidates(record, strategy, top_k * 3)
            for source in sources:
                rendered, lattice = render_retrieval_candidate(
                    engine=engine,
                    index=index,
                    record=record,
                    source=source,
                    lattice_mode=str(exp.get("lattice") or "source_direct"),
                    rank=rank,
                    source_name=exp["id"],
                )
                h = candidate_hash(rendered)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                rows.append(
                    make_generation_line(
                        mode=mode,
                        record=record,
                        sample_index=sample_index,
                        rank=rank,
                        rendered=rendered,
                        source=str(exp["id"]),
                        source_sample_id=str(source["sample_id"]),
                        lattice=lattice,
                    )
                )
                rank += 1
                if rank >= top_k:
                    break

        while rank < top_k:
            rendered, lattice = render_quantile_candidate(engine=engine, index=index, record=record, rank=rank, source_name=f"{exp['id']}_fallback")
            rows.append(
                make_generation_line(
                    mode=mode,
                    record=record,
                    sample_index=sample_index,
                    rank=rank,
                    rendered=rendered,
                    source=f"{exp['id']}_quantile_fallback",
                    lattice=lattice,
                )
            )
            rank += 1
    return rows


def summarize_topk(metrics: list[dict[str, Any]], records: list[dict[str, Any]], k: int, subset: str) -> dict[str, Any]:
    if subset == "all":
        indices = set(range(len(records)))
    elif subset == "atom_lt6":
        indices = {i for i, r in enumerate(records) if atom_count(r) < 6}
    elif subset == "atom_ge6":
        indices = {i for i, r in enumerate(records) if atom_count(r) >= 6}
    elif subset == "atom_ge12":
        indices = {i for i, r in enumerate(records) if atom_count(r) >= 12}
    elif subset.startswith("rows_"):
        indices = {i for i, r in enumerate(records) if row_bucket(row_count(r)) == subset}
    elif subset.startswith("crystal_"):
        system = subset.removeprefix("crystal_")
        indices = {i for i, r in enumerate(records) if crystal_system(int(r["sg"])) == system}
    else:
        indices = set(range(len(records)))
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        idx = int(metric["sample_index"])
        if idx in indices and int(metric.get("gen_index", 0)) < k:
            by_sample[idx].append(metric)
    match = 0
    rms_values: list[float] = []
    readable = formula_ok = atom_ok = sg_ok = strict = 0
    for idx in indices:
        rows = by_sample.get(idx, [])
        matched = [m for m in rows if m.get("match_ok") and m.get("rms") is not None]
        if matched:
            match += 1
            rms_values.append(min(float(m["rms"]) for m in matched))
        readable += int(any(m.get("pymatgen_readable") for m in rows))
        formula_ok += int(any(m.get("formula_ok") for m in rows))
        atom_ok += int(any(m.get("formula_ok") and not m.get("early_match_skip_reason") == "atom_count_mismatch" for m in rows))
        sg_ok += int(any(m.get("space_group_ok") for m in rows))
        strict += int(any(m.get("pymatgen_readable") and m.get("formula_ok") and m.get("space_group_ok") for m in rows))
    denom = max(1, len(indices))
    return {
        "samples": len(indices),
        f"match@{k}": match / denom,
        f"RMSE@{k}": float(sum(rms_values) / len(rms_values)) if rms_values else math.nan,
        f"matched_samples_for_RMSE@{k}": len(rms_values),
        f"readable@{k}": readable / denom,
        f"formula_ok@{k}": formula_ok / denom,
        f"atom_count_ok_proxy@{k}": atom_ok / denom,
        f"SG_ok@{k}": sg_ok / denom,
        f"strict_valid_proxy@{k}": strict / denom,
    }


def diversity_stats(generation_rows: list[dict[str, Any]], records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        if int(row.get("gen_index", 0)) < k:
            by_sample[int(row["sample_index"])].append(row)
    unique_counts: list[int] = []
    volume_stds: list[float] = []
    for idx in range(len(records)):
        rows = by_sample.get(idx, [])
        hashes = {hashlib.sha1(str(r.get("generated_text") or "").encode("utf-8", errors="ignore")).hexdigest() for r in rows}
        unique_counts.append(len(hashes))
        volumes = [float(r["lattice_volume"]) for r in rows if r.get("lattice_volume") is not None]
        if len(volumes) >= 2:
            volume_stds.append(float(statistics.pstdev(volumes)))
    return {
        f"unique_candidate_mean@{k}": float(sum(unique_counts) / max(1, len(unique_counts))),
        f"unique_candidate_min@{k}": min(unique_counts) if unique_counts else 0,
        f"lattice_volume_std_mean@{k}": float(sum(volume_stds) / len(volume_stds)) if volume_stds else 0.0,
    }


def aggregate_experiment(
    *,
    exp: dict[str, Any],
    records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    subsets = ["all", "atom_lt6", "atom_ge6", "atom_ge12", "rows_1_3", "rows_4_6", "rows_ge7"]
    systems = sorted({crystal_system(int(r["sg"])) for r in records})
    subsets.extend(f"crystal_{s}" for s in systems)
    out: dict[str, Any] = {
        "id": exp["id"],
        "mode": exp["mode"],
        "family": exp.get("family"),
        "description": exp.get("description"),
        "summary": {},
        "diversity": diversity_stats(generation_rows, records, 5),
    }
    for subset in subsets:
        out["summary"][subset] = {
            "top1": summarize_topk(metrics, records, 1, subset),
            "top5": summarize_topk(metrics, records, 5, subset),
        }
    return out


def write_audit_and_plan(args: argparse.Namespace, train_records: list[dict[str, Any]], val_records: list[dict[str, Any]]) -> None:
    report_dir = args.report_dir
    old = json.loads((REPO_ROOT / "reports" / "symcif_v4_mp20_minicfjoint_v2" / "03_gtwa_geometry_report.json").read_text(encoding="utf-8"))
    audit = {
        "scope": {"dataset": "MP-20", "splits": ["clean_train", "clean_val"], "test_used": False, "mpts52_used": False, "match20_primary": False},
        "previous_gtwa_geometry": old.get("summary"),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "val_atom_subsets": {
            "atom_lt6": sum(1 for r in val_records if atom_count(r) < 6),
            "atom_ge6": sum(1 for r in val_records if atom_count(r) >= 6),
            "atom_ge12": sum(1 for r in val_records if atom_count(r) >= 12),
        },
        "experiments": selected_experiments(args),
    }
    write_json(report_dir / "00_audit.json", audit)
    write_md(
        report_dir / "00_audit.md",
        "\n".join(
            [
                "# MP-20 Geometry Breakthrough Audit",
                "",
                "Scope: MP-20 clean_train/clean_val only. MP-20 test, MPTS-52, CrystaLLM baseline, source-CIF repair, and StructureMatcher-label training are not used.",
                "",
                f"- clean_train records: {len(train_records)}",
                f"- clean_val records: {len(val_records)}",
                f"- previous GT-WA match@1/@5: {pct(PREVIOUS_BASELINE['match@1'])} / {pct(PREVIOUS_BASELINE['match@5'])}",
                f"- previous GT-WA RMSE@1/@5: {num(PREVIOUS_BASELINE['RMSE@1'])} / {num(PREVIOUS_BASELINE['RMSE@5'])}",
                f"- val atom_count < 6: {audit['val_atom_subsets']['atom_lt6']}",
                f"- val atom_count >= 6: {audit['val_atom_subsets']['atom_ge6']}",
                f"- val atom_count >= 12: {audit['val_atom_subsets']['atom_ge12']}",
            ]
        ),
    )
    lines = [
        "# Geometry Breakthrough Experiment Plan",
        "",
        "The campaign tests source-level geometry modeling strategies for P(X, L | formula, GT_SG, GT_W, GT_A).",
        "",
        "Command:",
        "",
        "```bash",
        "conda run -n crystallm_env python model/New_model/symcif_experiment/scripts/run_mp20_geometry_breakthrough.py --stage all --eval-workers 32",
        "```",
        "",
        "| id | family | description |",
        "|---|---|---|",
    ]
    for exp in selected_experiments(args):
        lines.append(f"| {exp['id']} | {exp.get('family','')} | {exp.get('description','')} |")
    write_md(report_dir / "01_experiment_plan.md", "\n".join(lines))


def generate_all(args: argparse.Namespace) -> None:
    train_records = [v2.geometry_training_record(r) for r in read_jsonl(args.clean_train, limit=args.train_limit)]
    val_records = [v2.geometry_training_record(r) for r in read_jsonl(args.clean_val, limit=args.val_limit)]
    sg_symbols = v2.sg_symbols_from_splits({"train": train_records, "val": val_records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    index = GeometryIndex(train_records, val_records)
    baseline_rank0 = load_baseline_rank0(args.previous_generations)
    baseline_top5 = load_baseline_top5(args.previous_generations)
    write_audit_and_plan(args, train_records, val_records)
    for exp in selected_experiments(args):
        out_path = args.run_dir / "generations" / f"{exp['mode']}.jsonl"
        if out_path.exists() and args.skip_existing:
            continue
        started = time.time()
        rows = generate_for_experiment(
            exp=exp,
            engine=engine,
            index=index,
            val_records=val_records,
            baseline_rank0=baseline_rank0,
            baseline_top5=baseline_top5,
            top_k=args.top_k,
        )
        write_jsonl(out_path, rows)
        print(json.dumps({"stage": "generated", "experiment": exp["id"], "rows": len(rows), "seconds": time.time() - started}, sort_keys=True), flush=True)


def evaluate_all(args: argparse.Namespace) -> None:
    val_records = [v2.geometry_training_record(r) for r in read_jsonl(args.clean_val, limit=args.val_limit)]
    case_payload = v2.case_payload_from_clean_records(val_records)
    eval_args = argparse.Namespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.rmsd_timeout_seconds),
        max_match_sites=int(args.max_sites),
        max_eval_sites=int(args.max_sites),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
    )
    for exp in selected_experiments(args):
        gen_path = args.run_dir / "generations" / f"{exp['mode']}.jsonl"
        metrics_path = args.run_dir / "metrics" / f"{exp['mode']}_metrics.jsonl"
        if metrics_path.exists() and args.skip_existing:
            continue
        generation_rows = read_jsonl(gen_path)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in generation_rows:
            grouped[int(row["sample_index"])].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda r: int(r["gen_index"]))
        started = time.time()
        metrics = evaluate_mode_with_hard_timeouts(
            mode=str(exp["mode"]),
            case_payload=case_payload,
            grouped=grouped,
            lookup_json=str(args.lookup_json),
            args=eval_args,
        )
        metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        write_jsonl(metrics_path, metrics)
        print(json.dumps({"stage": "evaluated", "experiment": exp["id"], "metrics": len(metrics), "seconds": time.time() - started}, sort_keys=True), flush=True)


def canonical_cif_hash(text: str) -> str:
    lines = [line for line in str(text or "").splitlines() if not line.startswith("data_")]
    return hashlib.sha1("\n".join(lines).encode("utf-8", errors="ignore")).hexdigest()


def synthesize_all(args: argparse.Namespace) -> None:
    """Reuse metrics for identical rendered CIFs across fixed candidate mixtures."""
    experiments = selected_experiments(args)
    by_mode = {str(exp["mode"]): exp for exp in EXPERIMENTS}
    source_index: dict[tuple[int, str], dict[str, Any]] = {}
    source_count = 0
    for metrics_path in sorted((args.run_dir / "metrics").glob("*_metrics.jsonl")):
        mode = metrics_path.name.removesuffix("_metrics.jsonl")
        if mode not in by_mode:
            continue
        gen_path = args.run_dir / "generations" / f"{mode}.jsonl"
        if not gen_path.exists():
            continue
        metrics = {
            (int(row["sample_index"]), int(row["gen_index"])): row
            for row in read_jsonl(metrics_path)
        }
        for gen in read_jsonl(gen_path):
            key = (int(gen["sample_index"]), int(gen["gen_index"]))
            metric = metrics.get(key)
            if not metric:
                continue
            source_index[(int(gen["sample_index"]), canonical_cif_hash(str(gen.get("generated_text") or "")))] = metric
            source_count += 1

    for exp in experiments:
        mode = str(exp["mode"])
        metrics_path = args.run_dir / "metrics" / f"{mode}_metrics.jsonl"
        if metrics_path.exists() and args.skip_existing:
            continue
        gen_path = args.run_dir / "generations" / f"{mode}.jsonl"
        if not gen_path.exists():
            continue
        rows: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for gen in read_jsonl(gen_path):
            sample_index = int(gen["sample_index"])
            gen_index = int(gen["gen_index"])
            text = str(gen.get("generated_text") or "")
            key = (sample_index, canonical_cif_hash(text))
            source = source_index.get(key)
            if source is None:
                missing.append({"sample_index": sample_index, "gen_index": gen_index})
                continue
            row = dict(source)
            row["mode"] = mode
            row["gen_index"] = gen_index
            row["seed"] = gen.get("seed", row.get("seed"))
            row["generated_sha1"] = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            row["evaluation_synthesized"] = True
            row["evaluation_synthesized_from_mode"] = source.get("mode")
            row["evaluation_synthesized_from_gen_index"] = source.get("gen_index")
            rows.append(row)
        if missing:
            write_json(args.run_dir / "metrics" / f"{mode}_synthesis_missing.json", {"missing": missing[:100], "missing_count": len(missing), "source_metrics_indexed": source_count})
            print(json.dumps({"stage": "synthesis_skipped", "experiment": exp["id"], "missing": len(missing), "source_metrics_indexed": source_count}, sort_keys=True), flush=True)
            continue
        rows.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        write_jsonl(metrics_path, rows)
        print(json.dumps({"stage": "synthesized", "experiment": exp["id"], "metrics": len(rows), "source_metrics_indexed": source_count}, sort_keys=True), flush=True)


def write_reports(args: argparse.Namespace) -> None:
    val_records = [v2.geometry_training_record(r) for r in read_jsonl(args.clean_val, limit=args.val_limit)]
    experiment_summaries: list[dict[str, Any]] = []
    for exp in selected_experiments(args):
        gen_path = args.run_dir / "generations" / f"{exp['mode']}.jsonl"
        metrics_path = args.run_dir / "metrics" / f"{exp['mode']}_metrics.jsonl"
        if not gen_path.exists() or not metrics_path.exists():
            continue
        generation_rows = read_jsonl(gen_path)
        metrics = read_jsonl(metrics_path)
        experiment_summaries.append(aggregate_experiment(exp=exp, records=val_records, metrics=metrics, generation_rows=generation_rows))
    write_json(args.report_dir / "experiments_table.json", {"experiments": experiment_summaries, "previous_baseline": PREVIOUS_BASELINE})

    def key_metric(row: dict[str, Any], subset: str, metric: str) -> float:
        top = "top5" if metric in {"match@5", "RMSE@5"} else "top1"
        return float(row["summary"][subset][top].get(metric, math.nan))

    best_match5 = max(experiment_summaries, key=lambda r: key_metric(r, "all", "match@5")) if experiment_summaries else None
    best_ge6 = max(experiment_summaries, key=lambda r: key_metric(r, "atom_ge6", "match@5")) if experiment_summaries else None
    best_ge12 = max(experiment_summaries, key=lambda r: key_metric(r, "atom_ge12", "match@5")) if experiment_summaries else None
    finite_rmse = [r for r in experiment_summaries if math.isfinite(key_metric(r, "all", "RMSE@5"))]
    best_rmse = min(finite_rmse, key=lambda r: key_metric(r, "all", "RMSE@5")) if finite_rmse else None

    lines = [
        "# Geometry Breakthrough Experiments Table",
        "",
        "| id | family | match@1 | match@5 | RMSE@1 | RMSE@5 | atom>=6 match@5 | atom>=12 match@5 | unique@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiment_summaries:
        all_top1 = row["summary"]["all"]["top1"]
        all_top5 = row["summary"]["all"]["top5"]
        ge6 = row["summary"]["atom_ge6"]["top5"]
        ge12 = row["summary"]["atom_ge12"]["top5"]
        lines.append(
            "| {id} | {family} | {m1} | {m5} | {r1} | {r5} | {g6} | {g12} | {uniq} |".format(
                id=row["id"],
                family=row["family"],
                m1=pct(all_top1["match@1"]),
                m5=pct(all_top5["match@5"]),
                r1=num(all_top1["RMSE@1"]),
                r5=num(all_top5["RMSE@5"]),
                g6=pct(ge6["match@5"]),
                g12=pct(ge12["match@5"]),
                uniq=num(row["diversity"].get("unique_candidate_mean@5")),
            )
        )
    write_md(args.report_dir / "experiments_table.md", "\n".join(lines))

    failure_lines = [
        "# Geometry Breakthrough Failure Analysis",
        "",
        "The previous deterministic head fails mainly on complex atom-count subsets. This campaign tests whether source-level retrieval/codebook and nonparametric distributional candidates can add meaningful K=5 coverage.",
        "",
        "Key failure modes to inspect:",
        "",
        "- KNN/codebook candidates can be structurally valid but may map free parameters by row position rather than a learned row correspondence.",
        "- Quantile candidates are diverse but weakly conditioned on chemistry and W/A interactions.",
        "- Baseline-plus-candidate mixtures protect match@1 but need complementary candidates to improve match@5.",
        "- If retrieval improves complex subsets but worsens RMSE, the next step should be a learned residual model on top of retrieved modes.",
    ]
    write_md(args.report_dir / "failure_analysis.md", "\n".join(failure_lines))

    final = [
        "# MP-20 Geometry Breakthrough Final Report",
        "",
        "Scope: MP-20 clean_train/clean_val only. MP-20 test was not used.",
        "",
        "## Previous Bottleneck",
        "",
        f"- Previous GT-WA match@1/@5: {pct(PREVIOUS_BASELINE['match@1'])} / {pct(PREVIOUS_BASELINE['match@5'])}",
        f"- Previous GT-WA RMSE@1/@5: {num(PREVIOUS_BASELINE['RMSE@1'])} / {num(PREVIOUS_BASELINE['RMSE@5'])}",
        "- The old deterministic head was good on small structures but weak on complex atom-count subsets.",
        "",
        "## Strategy Families Tried",
        "",
        "- Family A: target/normalization redesign through volume-scaled retrieval.",
        "- Family B: nonparametric multimodal quantile/codebook candidates.",
        "- Family C: source-level geometry codebook/retrieval priors.",
        "- Family D: complex-subset mixture-of-experts routing.",
        "- Family E: row-conditioned retrieval using explicit W/A features.",
        "- Family F: baseline-plus-learned-candidate internal K=5 selection.",
        "",
        "## Best Results",
        "",
    ]
    if best_match5:
        final += [
            f"- Best overall match@5: `{best_match5['id']}` = {pct(key_metric(best_match5, 'all', 'match@5'))}",
            f"- Best overall match@1 for that run: {pct(key_metric(best_match5, 'all', 'match@1'))}",
        ]
    if best_ge6:
        final.append(f"- Best atom_count>=6 match@5: `{best_ge6['id']}` = {pct(key_metric(best_ge6, 'atom_ge6', 'match@5'))}")
    if best_ge12:
        final.append(f"- Best atom_count>=12 match@5: `{best_ge12['id']}` = {pct(key_metric(best_ge12, 'atom_ge12', 'match@5'))}")
    if best_rmse:
        final.append(f"- Best RMSE@5: `{best_rmse['id']}` = {num(key_metric(best_rmse, 'all', 'RMSE@5'))}")
    final += [
        "",
        "## Comparison to Previous GT-WA Geometry",
        "",
    ]
    if best_match5:
        improvement = key_metric(best_match5, "all", "match@5") - PREVIOUS_BASELINE["match@5"]
        final.append(f"- Best match@5 delta vs previous: {100.0 * improvement:+.2f} pp.")
    if best_rmse:
        improvement = key_metric(best_rmse, "all", "RMSE@5") - PREVIOUS_BASELINE["RMSE@5"]
        final.append(f"- Best RMSE@5 delta vs previous: {improvement:+.4f}.")
    final += [
        "",
        "## Full Generation Decision",
        "",
        "This campaign is a GT-WA geometry campaign. Full Mini-CFJoint generation should only be run diagnostically after selecting a frozen geometry branch; no MP-20 test command was run.",
        "",
        "## Recommended Next Experiment",
        "",
        "If a retrieval/codebook branch improves K=5 coverage, train a neural residual model conditioned on the selected source geometry mode. If no branch improves coverage, move to a true mixture-density or latent-code transformer geometry head rather than more deterministic regression.",
    ]
    write_md(args.report_dir / "final_report.md", "\n".join(final))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "evaluate", "synthesize", "report", "all"], default="all")
    parser.add_argument("--clean-train", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_train.jsonl")
    parser.add_argument("--clean-val", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_val.jsonl")
    parser.add_argument("--previous-generations", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "gtwa_geometry" / "gtwa_geometry_generations.jsonl")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports" / "symcif_v4_mp20_geometry_breakthrough")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_geometry_breakthrough")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-workers", type=int, default=32)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-sites", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--experiment-ids", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "generations").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    if args.stage in {"generate", "all"}:
        generate_all(args)
    if args.stage in {"evaluate", "all"}:
        evaluate_all(args)
    if args.stage in {"synthesize", "all"}:
        synthesize_all(args)
    if args.stage in {"report", "all"}:
        write_reports(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
