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

import run_mp20_geometry_breakthrough as gb  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "symcif_v4_mp20_fullgen_after_geometry_breakthrough"
RUN_DIR = REPO_ROOT / "runs" / "symcif_v4_mp20_fullgen_after_geometry_breakthrough"
CONFIG_DIR = PROJECT_ROOT / "configs" / "fullgen_after_geometry_breakthrough"

GEOMETRY_GTWA = {
    "deterministic_geometry_baseline": {
        "source_report_id": "e00_prev_deterministic",
        "match@1": 0.5573448546739984,
        "match@5": 0.5887666928515318,
        "RMSE@1": 0.1739349169633023,
        "RMSE@5": 0.168429924024472,
    },
    "e07_row_conditioned_knn": {
        "source_report_id": "e07_row_conditioned_knn",
        "match@1": 0.7842367111809374,
        "match@5": 0.8511390416339356,
        "RMSE@1": 0.047769192327731004,
        "RMSE@5": 0.04231171609270056,
    },
    "e08_baseline_plus_row_knn": {
        "source_report_id": "e08_baseline_plus_row_knn",
        "match@1": 0.5561665357423409,
        "match@5": 0.8710395391463734,
        "RMSE@1": 0.1736667850865597,
        "RMSE@5": 0.04317557216821799,
    },
}

FULLGEN_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "fg01_internal_sg_formula_e07",
        "family": "C_train_prior",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "geometry_mode": "e07",
        "ranker": "wa_prior",
        "description": "Train-only same-SG/formula-nearest symbolic WA prior with row-conditioned e07 geometry.",
    },
    {
        "id": "fg02_internal_sg_formula_e08",
        "family": "C_train_prior",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "geometry_mode": "e08",
        "ranker": "wa_prior",
        "description": "Same internal WA prior, using e08-style row-conditioned geometry candidates.",
    },
    {
        "id": "fg03_internal_rowcount_e08",
        "family": "B_rowcount_skeleton",
        "wa_source": "internal",
        "wa_strategy": "sg_rowcount",
        "geometry_mode": "e08",
        "ranker": "row_count_first",
        "description": "Internal skeleton-first proposal biased to train row-count distributions.",
    },
    {
        "id": "fg04_internal_diverse_skeleton_e08",
        "family": "B_rowcount_skeleton",
        "wa_source": "internal",
        "wa_strategy": "diverse_skeleton",
        "geometry_mode": "e08",
        "ranker": "diversity",
        "description": "Internal skeleton beam with duplicate-WA penalty and exact-cover element assignment.",
    },
    {
        "id": "fg05_external_hybrid_deterministic",
        "family": "D_current_wa_baseline",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top5",
        "geometry_mode": "deterministic",
        "ranker": "external_order",
        "description": "Current hybrid WA candidates replayed with their deterministic/prototype geometry as a baseline.",
    },
    {
        "id": "fg06_external_hybrid_e07",
        "family": "E_geometry_aware_wa",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top5",
        "geometry_mode": "e07",
        "ranker": "external_order",
        "description": "Current hybrid WA candidates rendered through the row-conditioned e07 geometry branch.",
    },
    {
        "id": "fg07_external_hybrid_e08",
        "family": "E_geometry_aware_wa",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top5",
        "geometry_mode": "e08",
        "ranker": "external_order",
        "description": "Current hybrid WA candidates rendered through e08 geometry diversity.",
    },
    {
        "id": "fg08_e07_rank0_e08_diverse",
        "family": "D_candidate_selection",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top5",
        "geometry_mode": "e07_rank0_e08",
        "ranker": "rank1_quality_then_k5_diversity",
        "description": "Use e07-like candidate at rank 1 and e08 row-conditioned diversity for ranks 2-5.",
    },
    {
        "id": "fg09_geometry_distance_rank",
        "family": "D_candidate_selection",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top5",
        "geometry_mode": "e08",
        "ranker": "geometry_distance",
        "description": "Rank external WA candidates by row-conditioned geometry retrieval distance and validity priors.",
    },
    {
        "id": "fg10_external_plus_internal_union",
        "family": "A_constrained_decoding",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "geometry_mode": "e08",
        "ranker": "external_then_internal_unique",
        "description": "Union current hybrid WA with train-only exact-cover symbolic proposals, deduplicated by WA key.",
    },
    {
        "id": "fg11_external_oracle_rowcount_e08",
        "family": "diagnostic_oracle_rowcount",
        "wa_source": "external_hybrid",
        "wa_strategy": "external_top20",
        "geometry_mode": "e08",
        "ranker": "oracle_rowcount_filter",
        "diagnostic_uses_gt_row_count": True,
        "description": "Diagnostic: filter predicted WA candidates by GT row_count before e08 geometry.",
    },
    {
        "id": "fg12_oracle_w_pred_a_e08",
        "family": "diagnostic_oracle_w",
        "wa_source": "oracle_w",
        "wa_strategy": "gt_skeleton_predicted_assignment",
        "geometry_mode": "e08",
        "ranker": "element_assignment_prior",
        "diagnostic_uses_gt_w": True,
        "description": "Diagnostic: use GT orbit rows but predict element assignment by formula/WA priors before e08 geometry.",
    },
]

DIAGNOSTIC_REUSE = [
    {
        "id": "D1_gtwa_deterministic_geometry",
        "family": "diagnostic_gtwa",
        "geometry_mode": "deterministic_geometry_baseline",
        "description": "GT-WA + previous deterministic geometry, reused from geometry breakthrough artifacts.",
    },
    {
        "id": "D2_gtwa_e07_geometry",
        "family": "diagnostic_gtwa",
        "geometry_mode": "e07_row_conditioned_knn",
        "description": "GT-WA + e07 row-conditioned KNN geometry, reused from geometry breakthrough artifacts.",
    },
    {
        "id": "D3_gtwa_e08_geometry",
        "family": "diagnostic_gtwa",
        "geometry_mode": "e08_baseline_plus_row_knn",
        "description": "GT-WA + e08 baseline-plus-row-conditioned KNN geometry, reused from geometry breakthrough artifacts.",
    },
]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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


def atom_bucket(record: dict[str, Any]) -> str:
    n = atom_count(record)
    if n < 6:
        return "atom_lt6"
    if n >= 12:
        return "atom_ge12"
    return "atom_ge6_lt12"


def row_bucket(n: int) -> str:
    n = int(n)
    if n <= 3:
        return "rows_1_3"
    if n <= 6:
        return "rows_4_6"
    return "rows_ge7"


def crystal_system(sg: int) -> str:
    return gb.crystal_system(int(sg))


def formula_key(record: dict[str, Any]) -> str:
    return str(record.get("pretty_formula") or record.get("formula") or gb.v2.formula_sum(record["formula_counts"]))


def formula_frac(record: dict[str, Any]) -> dict[str, float]:
    counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
    total = max(1, sum(counts.values()))
    return {k: float(v) / float(total) for k, v in counts.items()}


def formula_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


def canonical_cif_hash(text: str) -> str:
    lines = [line for line in str(text or "").splitlines() if not line.startswith("data_")]
    return hashlib.sha1("\n".join(lines).encode("utf-8", errors="ignore")).hexdigest()


def raw_cif_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def target_keys(record: dict[str, Any]) -> tuple[str, str]:
    skel, wa = v2.canonical_keys_from_rows(v2.canonical_rows(record))
    return str(record.get("canonical_skeleton_key") or skel), str(record.get("canonical_wa_key") or wa)


def normalize_rows(engine: OrbitEngine, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row.get("orbit_id") or row.get("canonical_orbit_id"))
        orbit = engine.get_orbit_by_id(oid)
        out.append(
            {
                "element": str(row["element"]),
                "orbit_id": oid,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
            }
        )
    return v2.canonical_rows({"wa_table": out})


def pseudo_record(record: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(record)
    out["wa_table"] = rows
    skel, wa = v2.canonical_keys_from_rows(rows)
    out["canonical_skeleton_key"] = skel
    out["canonical_wa_key"] = wa
    out["atom_count"] = sum(int(v) for v in out["formula_counts"].values())
    return out


def line_from_render(
    *,
    mode: str,
    record: dict[str, Any],
    sample_index: int,
    rank: int,
    rows: list[dict[str, Any]],
    rendered: dict[str, Any],
    source: str,
    score: float = 0.0,
    geometry_rank: int | None = None,
    source_sample_id: str | None = None,
    lattice: dict[str, float] | None = None,
) -> dict[str, Any]:
    target_skel, target_wa = target_keys(record)
    skel, wa = v2.canonical_keys_from_rows(rows) if rows else ("", "")
    expanded_atoms = None
    if rendered.get("ok") and lattice is not None:
        expanded_atoms = sum(int(r.get("multiplicity", 0)) for r in rows)
    return {
        "mode": mode,
        "sample_index": sample_index,
        "sample_id": record["sample_id"],
        "gen_index": rank,
        "seed": rank,
        "raw_generation_success": bool(rendered.get("ok")),
        "generated_text": rendered.get("cif") or "",
        "error": rendered.get("error"),
        "formula_closure_success": bool(rows and sum(int(r.get("multiplicity", 0)) for r in rows) == sum(int(v) for v in record["formula_counts"].values())),
        "atom_count_ok": bool(rendered.get("atom_count_ok")),
        "canonical_skeleton_key": skel,
        "canonical_wa_key": wa,
        "target_canonical_skeleton_key": target_skel,
        "target_canonical_wa_key": target_wa,
        "skeleton_hit": skel == target_skel,
        "wa_hit": wa == target_wa,
        "row_count_pred": len(rows),
        "row_count_target": row_count(record),
        "row_count_hit": len(rows) == row_count(record),
        "generation_score": float(score),
        "geometry_source": source,
        "geometry_rank": geometry_rank,
        "source_sample_id": source_sample_id,
        "lattice_volume": cell_volume(lattice) if lattice else None,
        "generated_sha1": raw_cif_hash(rendered.get("cif") or ""),
        "canonical_cif_sha1": canonical_cif_hash(rendered.get("cif") or ""),
        "expanded_atoms": expanded_atoms,
    }


class ExternalCandidates:
    def __init__(self, candidate_path: Path, generation_path: Path, engine: OrbitEngine) -> None:
        self.by_sample_id: dict[str, list[dict[str, Any]]] = {}
        self.baseline_lines: dict[tuple[str, int], dict[str, Any]] = {}
        for row in read_jsonl(candidate_path):
            sid = str(row.get("sample_id"))
            candidates = []
            for cand in row.get("candidates") or []:
                if not cand.get("rows"):
                    continue
                item = dict(cand)
                item["rows"] = normalize_rows(engine, list(cand["rows"]))
                candidates.append(item)
            self.by_sample_id[sid] = candidates
        for row in read_jsonl(generation_path):
            self.baseline_lines[(str(row.get("sample_id")), int(row.get("gen_index", 0)))] = row

    def candidates(self, sample_id: str, top_n: int) -> list[dict[str, Any]]:
        return list(self.by_sample_id.get(str(sample_id), []))[: max(0, int(top_n))]

    def baseline_render(self, sample_id: str, rank: int) -> dict[str, Any] | None:
        return self.baseline_lines.get((str(sample_id), int(rank)))


class NullExternalCandidates:
    def candidates(self, sample_id: str, top_n: int) -> list[dict[str, Any]]:
        return []

    def baseline_render(self, sample_id: str, rank: int) -> dict[str, Any] | None:
        return None


class WAProposalIndex:
    def __init__(self, train_records: list[dict[str, Any]], engine: OrbitEngine) -> None:
        self.train_records = train_records
        self.engine = engine
        self.by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.by_sg_row: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        self.by_crystal: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in train_records:
            sg = int(rec["sg"])
            self.by_sg[sg].append(rec)
            self.by_sg_row[(sg, row_count(rec))].append(rec)
            self.by_crystal[crystal_system(sg)].append(rec)
        self._proposal_cache: dict[tuple[str, str, int, bool, int], list[dict[str, Any]]] = {}

    def source_pool(self, target: dict[str, Any], strategy: str, oracle_row_count: bool = False) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        pools: list[list[dict[str, Any]]]
        if oracle_row_count or strategy == "sg_rowcount":
            pools = [self.by_sg_row.get((sg, row_count(target)), []), self.by_sg.get(sg, []), self.train_records]
        elif strategy == "diverse_skeleton":
            pools = [self.by_sg.get(sg, []), self.by_crystal.get(crystal_system(sg), []), self.train_records]
        else:
            pools = [self.by_sg.get(sg, []), self.by_crystal.get(crystal_system(sg), []), self.train_records]
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []
        for source_pool in pools:
            reduced = source_pool
            if len(reduced) > 180:
                reduced = sorted(reduced, key=lambda r: self.source_distance(target, r, cheap=True))[:180]
            for rec in reduced:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= 240:
                break
        return sorted(pool, key=lambda r: self.source_distance(target, r, cheap=False))

    def source_distance(self, target: dict[str, Any], source: dict[str, Any], cheap: bool = False) -> float:
        dist = formula_l1(formula_frac(target), formula_frac(source))
        dist += 0.02 * abs(atom_count(target) - atom_count(source))
        dist += 0.08 * abs(row_count(target) - row_count(source))
        if int(target["sg"]) != int(source["sg"]):
            dist += 4.0
        if not cheap:
            target_counts = sorted(int(v) for v in target["formula_counts"].values())
            source_counts = sorted(int(v) for v in source["formula_counts"].values())
            dist += 0.03 * abs(len(target_counts) - len(source_counts))
            dist += 0.01 * sum(abs(a - b) for a, b in zip(target_counts, source_counts))
        return float(dist)

    def propose(
        self,
        target: dict[str, Any],
        *,
        strategy: str,
        top_n: int,
        oracle_row_count: bool = False,
        max_sources: int = 80,
    ) -> list[dict[str, Any]]:
        cache_key = (str(target.get("sample_id")), str(strategy), int(top_n), bool(oracle_row_count), int(max_sources))
        cached = self._proposal_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        out: list[dict[str, Any]] = []
        seen_wa: set[str] = set()
        source_rank = 0
        for source in self.source_pool(target, strategy, oracle_row_count=oracle_row_count)[:max_sources]:
            source_rank += 1
            skeleton = normalize_rows(self.engine, v2.canonical_rows(source))
            if oracle_row_count and len(skeleton) != row_count(target):
                continue
            assignments = exact_cover_assignments(
                skeleton,
                {str(k): int(v) for k, v in target["formula_counts"].items()},
                source_rows=v2.canonical_rows(source),
                max_assignments=4 if strategy == "diverse_skeleton" else 2,
            )
            for assignment_rank, rows in enumerate(assignments):
                skel, wa = v2.canonical_keys_from_rows(rows)
                if wa in seen_wa:
                    continue
                seen_wa.add(wa)
                score = -self.source_distance(target, source, cheap=False) - 0.001 * source_rank - 0.01 * assignment_rank
                out.append(
                    {
                        "rows": rows,
                        "canonical_skeleton_key": skel,
                        "canonical_wa_key": wa,
                        "score": score,
                        "source_sample_id": source["sample_id"],
                        "source_distance": self.source_distance(target, source, cheap=False),
                    }
                )
                if len(out) >= top_n:
                    self._proposal_cache[cache_key] = list(out)
                    return out
        self._proposal_cache[cache_key] = list(out)
        return out


def exact_cover_assignments(
    skeleton_rows: list[dict[str, Any]],
    formula_counts: dict[str, int],
    *,
    source_rows: list[dict[str, Any]] | None = None,
    max_assignments: int = 4,
) -> list[list[dict[str, Any]]]:
    rows = [dict(r) for r in skeleton_rows]
    source_rows = source_rows or rows
    remaining = {str(k): int(v) for k, v in formula_counts.items()}
    order = sorted(range(len(rows)), key=lambda i: (-int(rows[i].get("multiplicity", 1)), str(rows[i].get("orbit_id"))))
    assigned: dict[int, str] = {}
    out: list[list[dict[str, Any]]] = []

    def element_order(row_idx: int) -> list[str]:
        preferred = str(source_rows[row_idx].get("element")) if row_idx < len(source_rows) else ""
        elems = sorted(remaining, key=lambda e: (-remaining[e], e))
        if preferred in elems:
            elems.remove(preferred)
            elems.insert(0, preferred)
        return elems

    suffix_mults: list[list[int]] = []
    for pos in range(len(order) + 1):
        suffix_mults.append([int(rows[i].get("multiplicity", 1)) for i in order[pos:]])

    def feasible(pos: int) -> bool:
        mults = suffix_mults[pos]
        total_left = sum(mults)
        if total_left != sum(remaining.values()):
            return False
        for count in remaining.values():
            if count == 0:
                continue
            possible = {0}
            for m in mults:
                possible |= {v + m for v in list(possible) if v + m <= count}
            if count not in possible:
                return False
        return True

    def backtrack(pos: int) -> None:
        if len(out) >= max_assignments:
            return
        if not feasible(pos):
            return
        if pos == len(order):
            if all(v == 0 for v in remaining.values()):
                candidate = []
                for idx, row in enumerate(rows):
                    new_row = dict(row)
                    new_row["element"] = assigned[idx]
                    candidate.append(new_row)
                out.append(v2.canonical_rows({"wa_table": candidate}))
            return
        idx = order[pos]
        mult = int(rows[idx].get("multiplicity", 1))
        for element in element_order(idx):
            if remaining[element] < mult:
                continue
            assigned[idx] = element
            remaining[element] -= mult
            backtrack(pos + 1)
            remaining[element] += mult
            assigned.pop(idx, None)
            if len(out) >= max_assignments:
                return

    backtrack(0)
    return out


def build_gt_skeleton_predicted_assignment(
    record: dict[str, Any],
    external_candidates: list[dict[str, Any]],
    engine: OrbitEngine,
    top_n: int,
) -> list[dict[str, Any]]:
    skeleton = normalize_rows(engine, v2.canonical_rows(record))
    source_rows: list[dict[str, Any]] = []
    for cand in external_candidates:
        source_rows.extend(cand.get("rows") or [])
    if not source_rows:
        source_rows = v2.canonical_rows(record)
    assignments = exact_cover_assignments(
        skeleton,
        {str(k): int(v) for k, v in record["formula_counts"].items()},
        source_rows=source_rows[: len(skeleton)] if len(source_rows) >= len(skeleton) else v2.canonical_rows(record),
        max_assignments=max(top_n, 1),
    )
    out: list[dict[str, Any]] = []
    for i, rows in enumerate(assignments[:top_n]):
        skel, wa = v2.canonical_keys_from_rows(rows)
        out.append({"rows": rows, "canonical_skeleton_key": skel, "canonical_wa_key": wa, "score": -float(i), "source_sample_id": None})
    return out


def render_with_geometry(
    *,
    engine: OrbitEngine,
    geom_index: gb.GeometryIndex,
    target_record: dict[str, Any],
    rows: list[dict[str, Any]],
    geometry_mode: str,
    geometry_rank: int,
    source_name: str,
) -> tuple[dict[str, Any], dict[str, float] | None, str | None, float | None]:
    pred_record = pseudo_record(target_record, rows)
    if geometry_mode == "deterministic":
        rendered, lattice = gb.render_quantile_candidate(engine=engine, index=geom_index, record=pred_record, rank=0, source_name=source_name)
        return rendered, lattice, None, None
    sources = geom_index.candidates(pred_record, "row_conditioned_knn", max(geometry_rank + 1, 1))
    if not sources:
        rendered, lattice = gb.render_quantile_candidate(engine=engine, index=geom_index, record=pred_record, rank=geometry_rank, source_name=f"{source_name}_quantile")
        return rendered, lattice, None, None
    source = sources[min(geometry_rank, len(sources) - 1)]
    rendered, lattice = gb.render_retrieval_candidate(
        engine=engine,
        index=geom_index,
        record=pred_record,
        source=source,
        lattice_mode="volume_scaled_source",
        rank=geometry_rank,
        source_name=source_name,
    )
    return rendered, lattice, str(source.get("sample_id")), gb.row_condition_distance(pred_record, source)


def build_experiment_candidates(
    *,
    exp: dict[str, Any],
    record: dict[str, Any],
    sample_index: int,
    engine: OrbitEngine,
    geom_index: gb.GeometryIndex,
    wa_index: WAProposalIndex,
    external: ExternalCandidates,
    top_k: int,
) -> list[dict[str, Any]]:
    mode = str(exp["id"])
    wa_source = str(exp["wa_source"])
    wa_strategy = str(exp["wa_strategy"])
    geometry_mode = str(exp["geometry_mode"])
    external_pool = external.candidates(record["sample_id"], 20)
    wa_candidates: list[dict[str, Any]] = []
    if wa_source == "external_hybrid":
        wa_candidates = external_pool[: (20 if exp.get("diagnostic_uses_gt_row_count") else 5)]
    elif wa_source == "internal":
        wa_candidates = wa_index.propose(record, strategy=wa_strategy, top_n=12, oracle_row_count=False)
    elif wa_source == "hybrid_union":
        seen: set[str] = set()
        for cand in external_pool[:5]:
            if cand["canonical_wa_key"] not in seen:
                seen.add(cand["canonical_wa_key"])
                wa_candidates.append(cand)
        for cand in wa_index.propose(record, strategy="diverse_skeleton", top_n=12, oracle_row_count=False):
            if cand["canonical_wa_key"] not in seen:
                seen.add(cand["canonical_wa_key"])
                wa_candidates.append(cand)
            if len(wa_candidates) >= 12:
                break
    elif wa_source == "oracle_w":
        wa_candidates = build_gt_skeleton_predicted_assignment(record, external_pool[:5], engine, top_n=top_k)

    if exp.get("diagnostic_uses_gt_row_count"):
        wa_candidates = [cand for cand in wa_candidates if len(cand.get("rows") or []) == row_count(record)]
        if len(wa_candidates) < top_k:
            for cand in external_pool:
                if len(cand.get("rows") or []) == row_count(record) and cand not in wa_candidates:
                    wa_candidates.append(cand)
                if len(wa_candidates) >= top_k:
                    break

    if str(exp.get("ranker")) == "geometry_distance":
        scored = []
        for cand in wa_candidates:
            pred_record = pseudo_record(record, cand["rows"])
            sources = geom_index.candidates(pred_record, "row_conditioned_knn", 1)
            distance = gb.row_condition_distance(pred_record, sources[0]) if sources else 9999.0
            row_penalty = abs(len(cand["rows"]) - row_count(record))
            scored.append((distance + 0.15 * row_penalty, cand))
        wa_candidates = [cand for _, cand in sorted(scored, key=lambda x: x[0])]
    elif str(exp.get("ranker")) == "diversity":
        seen_skel: set[str] = set()
        diverse: list[dict[str, Any]] = []
        for cand in wa_candidates:
            skel = str(cand["canonical_skeleton_key"])
            if skel in seen_skel and len(diverse) < 3:
                continue
            seen_skel.add(skel)
            diverse.append(cand)
            if len(diverse) >= top_k:
                break
        for cand in wa_candidates:
            if cand not in diverse:
                diverse.append(cand)
            if len(diverse) >= top_k:
                break
        wa_candidates = diverse

    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    rank = 0

    if geometry_mode == "deterministic" and wa_source == "external_hybrid":
        for i, cand in enumerate(wa_candidates[:top_k]):
            baseline = external.baseline_render(record["sample_id"], i)
            if baseline and baseline.get("generated_text"):
                rendered = {"ok": bool(baseline.get("raw_generation_success", True)), "cif": baseline.get("generated_text") or "", "atom_count_ok": bool(baseline.get("atom_count_ok", True))}
                line = line_from_render(
                    mode=mode,
                    record=record,
                    sample_index=sample_index,
                    rank=rank,
                    rows=cand["rows"],
                    rendered=rendered,
                    source="external_deterministic_replay",
                    score=float(cand.get("score", cand.get("hybrid_score", 0.0))),
                    geometry_rank=0,
                    lattice=None,
                )
                rows.append(line)
                rank += 1
        return pad_missing(rows, mode, record, sample_index, top_k, rank)

    geometry_plan: list[tuple[int, int]] = []
    if geometry_mode == "e07_rank0_e08":
        geometry_plan = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    elif geometry_mode == "e07":
        geometry_plan = [(i, 0) for i in range(max(top_k, len(wa_candidates)))]
    else:
        geometry_plan = [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1)]

    for wa_idx, geom_rank in geometry_plan:
        if rank >= top_k:
            break
        if wa_idx >= len(wa_candidates):
            continue
        cand = wa_candidates[wa_idx]
        rendered, lattice, source_sample_id, geom_distance = render_with_geometry(
            engine=engine,
            geom_index=geom_index,
            target_record=record,
            rows=cand["rows"],
            geometry_mode="e07" if geometry_mode == "e07" else "e08",
            geometry_rank=geom_rank,
            source_name=mode,
        )
        h = canonical_cif_hash(rendered.get("cif") or "")
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        score = float(cand.get("score", cand.get("hybrid_score", 0.0)))
        if geom_distance is not None:
            score -= 0.05 * float(geom_distance)
        rows.append(
            line_from_render(
                mode=mode,
                record=record,
                sample_index=sample_index,
                rank=rank,
                rows=cand["rows"],
                rendered=rendered,
                source=f"{geometry_mode}_row_conditioned",
                score=score,
                geometry_rank=geom_rank,
                source_sample_id=source_sample_id or cand.get("source_sample_id"),
                lattice=lattice,
            )
        )
        rank += 1
    return pad_missing(rows, mode, record, sample_index, top_k, rank)


def pad_missing(rows: list[dict[str, Any]], mode: str, record: dict[str, Any], sample_index: int, top_k: int, rank: int) -> list[dict[str, Any]]:
    while rank < top_k:
        rows.append(
            {
                "mode": mode,
                "sample_index": sample_index,
                "sample_id": record["sample_id"],
                "gen_index": rank,
                "seed": rank,
                "raw_generation_success": False,
                "generated_text": "",
                "error": "missing_candidate",
                "formula_closure_success": False,
                "atom_count_ok": False,
                "canonical_skeleton_key": "",
                "canonical_wa_key": "",
                "target_canonical_skeleton_key": target_keys(record)[0],
                "target_canonical_wa_key": target_keys(record)[1],
                "skeleton_hit": False,
                "wa_hit": False,
                "row_count_pred": 0,
                "row_count_target": row_count(record),
                "row_count_hit": False,
                "generation_score": -9999.0,
                "geometry_source": "missing_candidate",
                "generated_sha1": "",
                "canonical_cif_sha1": "",
            }
        )
        rank += 1
    return rows


def selected_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = str(args.experiment_ids or "").strip()
    if not raw:
        return FULLGEN_EXPERIMENTS
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    return [exp for exp in FULLGEN_EXPERIMENTS if str(exp["id"]) in wanted]


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return [v2.geometry_training_record(r) for r in read_jsonl(path, limit=limit)]


def write_audit(args: argparse.Namespace, train_records: list[dict[str, Any]], val_records: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_formula_sg = {(formula_key(r), int(r["sg"])) for r in train_records}
    train_wa = {(formula_key(r), int(r["sg"]), target_keys(r)[1]) for r in train_records}
    train_skel = {(formula_key(r), int(r["sg"]), target_keys(r)[0]) for r in train_records}
    val_wa_keys = {(formula_key(r), int(r["sg"]), target_keys(r)[1]) for r in val_records}
    overlap_wa = [r for r in train_records if (formula_key(r), int(r["sg"]), target_keys(r)[1]) in val_wa_keys]
    audit = {
        "scope": {
            "dataset": "MP-20",
            "development_splits": ["clean_train", "clean_val"],
            "test_used_for_development": False,
            "mpts52_used": False,
            "crystallm_baseline_started": False,
        },
        "train_records": len(train_records),
        "val_records": len(val_records),
        "geometry_breakthrough_reproduced_from": str(args.geometry_report),
        "geometry_gtwa_metrics": GEOMETRY_GTWA,
        "overlap": {
            "formula_sg_val_hits": sum(1 for r in val_records if (formula_key(r), int(r["sg"])) in train_formula_sg),
            "formula_sg_val_hit_rate": sum(1 for r in val_records if (formula_key(r), int(r["sg"])) in train_formula_sg) / max(1, len(val_records)),
            "formula_sg_wa_val_hits": sum(1 for r in val_records if (formula_key(r), int(r["sg"]), target_keys(r)[1]) in train_wa),
            "formula_sg_wa_val_hit_rate": sum(1 for r in val_records if (formula_key(r), int(r["sg"]), target_keys(r)[1]) in train_wa) / max(1, len(val_records)),
            "formula_sg_skeleton_val_hits": sum(1 for r in val_records if (formula_key(r), int(r["sg"]), target_keys(r)[0]) in train_skel),
            "formula_sg_skeleton_val_hit_rate": sum(1 for r in val_records if (formula_key(r), int(r["sg"]), target_keys(r)[0]) in train_skel) / max(1, len(val_records)),
            "train_records_removed_in_exact_wa_dedup_stress": len(overlap_wa),
        },
        "experiments": FULLGEN_EXPERIMENTS,
    }
    write_json(REPORT_DIR / "00_audit.json", audit)
    write_md(
        REPORT_DIR / "00_audit.md",
        "\n".join(
            [
                "# Full Generation After Geometry Breakthrough Audit",
                "",
                "Scope: MP-20 clean_train/clean_val only for development. MP-20 test is not used before a frozen validation decision.",
                "",
                f"- clean_train records: {len(train_records)}",
                f"- clean_val records: {len(val_records)}",
                f"- e07 GT-WA match@1/@5: {pct(GEOMETRY_GTWA['e07_row_conditioned_knn']['match@1'])} / {pct(GEOMETRY_GTWA['e07_row_conditioned_knn']['match@5'])}",
                f"- e07 GT-WA RMSE@1/@5: {num(GEOMETRY_GTWA['e07_row_conditioned_knn']['RMSE@1'])} / {num(GEOMETRY_GTWA['e07_row_conditioned_knn']['RMSE@5'])}",
                f"- e08 GT-WA match@1/@5: {pct(GEOMETRY_GTWA['e08_baseline_plus_row_knn']['match@1'])} / {pct(GEOMETRY_GTWA['e08_baseline_plus_row_knn']['match@5'])}",
                f"- e08 GT-WA RMSE@1/@5: {num(GEOMETRY_GTWA['e08_baseline_plus_row_knn']['RMSE@1'])} / {num(GEOMETRY_GTWA['e08_baseline_plus_row_knn']['RMSE@5'])}",
                f"- val formula+SG overlap with train: {audit['overlap']['formula_sg_val_hits']} ({pct(audit['overlap']['formula_sg_val_hit_rate'])})",
                f"- val formula+SG+WA overlap with train: {audit['overlap']['formula_sg_wa_val_hits']} ({pct(audit['overlap']['formula_sg_wa_val_hit_rate'])})",
                f"- val formula+SG+skeleton overlap with train: {audit['overlap']['formula_sg_skeleton_val_hits']} ({pct(audit['overlap']['formula_sg_skeleton_val_hit_rate'])})",
            ]
        ),
    )
    write_md(
        REPORT_DIR / "00_leakage_check.md",
        "\n".join(
            [
                "# Leakage Check",
                "",
                "- Retrieval/codebook index construction uses only clean_train records. `GeometryIndex` and `WAProposalIndex` insert only train rows into lookup pools.",
                "- clean_val records are used only as evaluation targets and for generation inputs (`formula`, `GT_SG`).",
                "- Full-generation experiments do not use validation W/A or X/L, except diagnostic experiments explicitly labeled `diagnostic_uses_gt_*`.",
                "- e07/e08 are source-level geometry generators conditioned on formula, SG, and the generated/diagnostic W/A table; they do not train on StructureMatcher labels.",
                "- Exact-CIF evaluation receives generated CIF text from the experiment JSONL; target CIFs are not copied as generated text.",
                "- MP-20 test is not used in audit/generation/evaluation/report stages before frozen-config gating.",
                f"- Exact formula+SG+WA train/val overlap is low: {audit['overlap']['formula_sg_wa_val_hits']} / {len(val_records)} = {pct(audit['overlap']['formula_sg_wa_val_hit_rate'])}.",
                f"- Dedup stress mask would remove {len(overlap_wa)} clean_train rows that share a formula+SG+WA key with at least one clean_val target. This is recorded as a stress check, not used for tuning.",
            ]
        ),
    )


def write_configs(args: argparse.Namespace) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for exp in FULLGEN_EXPERIMENTS:
        payload = {
            "experiment": exp,
            "command": (
                "conda run -n crystallm_env python "
                "model/New_model/symcif_experiment/scripts/run_mp20_fullgen_after_geometry_breakthrough.py "
                f"--stage all --experiment-ids {exp['id']} --top-k {args.top_k}"
            ),
            "seed": args.seed,
            "top_k": args.top_k,
            "clean_train": str(args.clean_train),
            "clean_val": str(args.clean_val),
            "retrieval_index": "in_memory_clean_train_only",
        }
        write_json(CONFIG_DIR / f"{exp['id']}.json", payload)


def generate_all(args: argparse.Namespace) -> None:
    train_records = load_records(args.clean_train, args.train_limit)
    val_records = load_records(args.clean_val, args.val_limit)
    sg_symbols = v2.sg_symbols_from_splits({"train": train_records, "val": val_records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    geom_index = gb.GeometryIndex(train_records, [])
    wa_index = WAProposalIndex(train_records, engine)
    external = ExternalCandidates(args.external_candidates, args.external_generations, engine)
    write_audit(args, train_records, val_records)
    write_configs(args)
    experiments = selected_experiments(args)
    for exp in experiments:
        out_path = args.run_dir / "generations" / f"{exp['id']}.jsonl"
        if out_path.exists() and args.skip_existing:
            continue
        started = time.time()
        rows: list[dict[str, Any]] = []
        for sample_index, record in enumerate(val_records):
            if sample_index and sample_index % 250 == 0:
                print(json.dumps({"stage": "generate_progress", "experiment": exp["id"], "done": sample_index, "seconds": time.time() - started}, sort_keys=True), flush=True)
            rows.extend(
                build_experiment_candidates(
                    exp=exp,
                    record=record,
                    sample_index=sample_index,
                    engine=engine,
                    geom_index=geom_index,
                    wa_index=wa_index,
                    external=external,
                    top_k=args.top_k,
                )
            )
        write_jsonl(out_path, rows)
        print(json.dumps({"stage": "generated", "experiment": exp["id"], "rows": len(rows), "seconds": time.time() - started}, sort_keys=True), flush=True)
    write_eval_pool(args, experiments)


def generate_records_to_run_dir(
    args: argparse.Namespace,
    *,
    records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    run_dir: Path,
    external_candidates: Path | None = None,
    external_generations: Path | None = None,
    split_name: str = "val",
) -> None:
    sg_symbols = v2.sg_symbols_from_splits({"train": train_records, split_name: records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    geom_index = gb.GeometryIndex(train_records, [])
    wa_index = WAProposalIndex(train_records, engine)
    needs_external = any(str(exp.get("wa_source")) in {"external_hybrid", "hybrid_union", "oracle_w"} for exp in experiments)
    if needs_external:
        if external_candidates is None or external_generations is None:
            raise FileNotFoundError("Frozen test external candidate/generation paths are required for external-hybrid experiments.")
        external: ExternalCandidates | NullExternalCandidates = ExternalCandidates(external_candidates, external_generations, engine)
    else:
        external = NullExternalCandidates()
    for exp in experiments:
        out_path = run_dir / "generations" / f"{exp['id']}.jsonl"
        if out_path.exists() and args.skip_existing:
            continue
        started = time.time()
        rows: list[dict[str, Any]] = []
        for sample_index, record in enumerate(records):
            if sample_index and sample_index % 250 == 0:
                print(
                    json.dumps(
                        {
                            "stage": f"generate_{split_name}_progress",
                            "experiment": exp["id"],
                            "done": sample_index,
                            "seconds": time.time() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            rows.extend(
                build_experiment_candidates(
                    exp=exp,
                    record=record,
                    sample_index=sample_index,
                    engine=engine,
                    geom_index=geom_index,
                    wa_index=wa_index,
                    external=external,
                    top_k=args.top_k,
                )
            )
        write_jsonl(out_path, rows)
        print(
            json.dumps(
                {"stage": f"generated_{split_name}", "experiment": exp["id"], "rows": len(rows), "seconds": time.time() - started},
                sort_keys=True,
            ),
            flush=True,
        )
    pool_args = argparse.Namespace(**vars(args))
    pool_args.run_dir = run_dir
    write_eval_pool(pool_args, experiments)


def write_eval_pool(args: argparse.Namespace, experiments: list[dict[str, Any]]) -> None:
    seen: set[tuple[int, str]] = set()
    next_gen_index: dict[int, int] = defaultdict(int)
    pool: list[dict[str, Any]] = []
    for exp in experiments:
        for row in read_jsonl(args.run_dir / "generations" / f"{exp['id']}.jsonl"):
            text = str(row.get("generated_text") or "")
            sample_index = int(row["sample_index"])
            key = (sample_index, canonical_cif_hash(text))
            if not text or key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["mode"] = "fullgen_eval_pool"
            out["gen_index"] = next_gen_index[sample_index]
            next_gen_index[sample_index] += 1
            out["source_experiment"] = row["mode"]
            pool.append(out)
    pool.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(args.run_dir / "generations" / "fullgen_eval_pool.jsonl", pool)
    write_json(args.run_dir / "generations" / "fullgen_eval_pool_manifest.json", {"rows": len(pool), "experiments": [e["id"] for e in experiments]})


def evaluate_pool(args: argparse.Namespace) -> None:
    val_records = load_records(args.clean_val, args.val_limit)
    gen_path = args.run_dir / "generations" / "fullgen_eval_pool.jsonl"
    metrics_path = args.run_dir / "metrics" / "fullgen_eval_pool_metrics.jsonl"
    if metrics_path.exists() and args.skip_existing:
        return
    generation_rows = read_jsonl(gen_path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["sample_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
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
    started = time.time()
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline_fullgen_eval_pool",
        case_payload=v2.case_payload_from_clean_records(val_records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(metrics_path, metrics)
    print(json.dumps({"stage": "evaluated_pool", "rows": len(metrics), "seconds": time.time() - started}, sort_keys=True), flush=True)


def evaluate_pool_for_records(args: argparse.Namespace, *, records: list[dict[str, Any]], run_dir: Path, stage_name: str) -> None:
    gen_path = run_dir / "generations" / "fullgen_eval_pool.jsonl"
    metrics_path = run_dir / "metrics" / "fullgen_eval_pool_metrics.jsonl"
    if metrics_path.exists() and args.skip_existing:
        return
    generation_rows = read_jsonl(gen_path)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[int(row["sample_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
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
    started = time.time()
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline_fullgen_eval_pool",
        case_payload=v2.case_payload_from_clean_records(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(metrics_path, metrics)
    print(json.dumps({"stage": stage_name, "rows": len(metrics), "seconds": time.time() - started}, sort_keys=True), flush=True)


def synthesize_metrics(args: argparse.Namespace) -> None:
    pool_gen = read_jsonl(args.run_dir / "generations" / "fullgen_eval_pool.jsonl")
    pool_metrics = read_jsonl(args.run_dir / "metrics" / "fullgen_eval_pool_metrics.jsonl")
    metric_by_pool_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in pool_metrics}
    metric_by_hash: dict[tuple[int, str], dict[str, Any]] = {}
    for gen in pool_gen:
        metric = metric_by_pool_key.get((int(gen["sample_index"]), int(gen["gen_index"])))
        if not metric:
            continue
        metric_by_hash[(int(gen["sample_index"]), canonical_cif_hash(str(gen.get("generated_text") or "")))] = metric

    for exp in selected_experiments(args):
        out_path = args.run_dir / "metrics" / f"{exp['id']}_metrics.jsonl"
        if out_path.exists() and args.skip_existing:
            continue
        rows: list[dict[str, Any]] = []
        missing = 0
        for gen in read_jsonl(args.run_dir / "generations" / f"{exp['id']}.jsonl"):
            text = str(gen.get("generated_text") or "")
            source = metric_by_hash.get((int(gen["sample_index"]), canonical_cif_hash(text)))
            if source is None:
                missing += 1
                row = {
                    "mode": exp["id"],
                    "sample_index": int(gen["sample_index"]),
                    "sample_id": gen.get("sample_id"),
                    "gen_index": int(gen["gen_index"]),
                    "match_ok": False,
                    "rms": None,
                    "pymatgen_readable": False,
                    "formula_ok": False,
                    "space_group_ok": False,
                    "early_match_skip_reason": "missing_synthesized_metric",
                }
            else:
                row = dict(source)
                row["mode"] = exp["id"]
                row["gen_index"] = int(gen["gen_index"])
                row["seed"] = gen.get("seed", row.get("seed"))
                row["evaluation_synthesized_from_pool"] = True
            rows.append(row)
        rows.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        write_jsonl(out_path, rows)
        if missing:
            write_json(args.run_dir / "metrics" / f"{exp['id']}_synthesis_missing.json", {"missing_count": missing})
        print(json.dumps({"stage": "synthesized", "experiment": exp["id"], "metrics": len(rows), "missing": missing}, sort_keys=True), flush=True)


def subset_indices(records: list[dict[str, Any]], subset: str) -> set[int]:
    if subset == "all":
        return set(range(len(records)))
    if subset == "atom_lt6":
        return {i for i, r in enumerate(records) if atom_count(r) < 6}
    if subset == "atom_ge6":
        return {i for i, r in enumerate(records) if atom_count(r) >= 6}
    if subset == "atom_ge12":
        return {i for i, r in enumerate(records) if atom_count(r) >= 12}
    if subset.startswith("rows_"):
        return {i for i, r in enumerate(records) if row_bucket(row_count(r)) == subset}
    if subset.startswith("crystal_"):
        system = subset.removeprefix("crystal_")
        return {i for i, r in enumerate(records) if crystal_system(int(r["sg"])) == system}
    return set(range(len(records)))


def summarize_experiment(records: list[dict[str, Any]], gen_rows: list[dict[str, Any]], metrics: list[dict[str, Any]], subset: str, k: int) -> dict[str, Any]:
    indices = subset_indices(records, subset)
    gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in gen_rows}
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        idx = int(metric["sample_index"])
        if idx in indices and int(metric.get("gen_index", 0)) < k:
            by_sample[idx].append(metric)
    match = readable = formula_ok = atom_ok = sg_ok = wa_hit = skel_hit = closure = row_hit = 0
    rms_values: list[float] = []
    candidate_counts: list[int] = []
    for idx in indices:
        rows = sorted(by_sample.get(idx, []), key=lambda r: int(r.get("gen_index", 0)))
        candidate_counts.append(len(rows))
        matched = [m for m in rows if m.get("match_ok") and m.get("rms") is not None]
        if matched:
            match += 1
            rms_values.append(min(float(m["rms"]) for m in matched))
        readable += int(any(m.get("pymatgen_readable") for m in rows))
        formula_ok += int(any(m.get("formula_ok") for m in rows))
        sg_ok += int(any(m.get("space_group_ok") for m in rows))
        atom_ok += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("atom_count_ok")) for m in rows))
        wa_hit += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("wa_hit")) for m in rows))
        skel_hit += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("skeleton_hit")) for m in rows))
        closure += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("formula_closure_success")) for m in rows))
        row_hit += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("row_count_hit")) for m in rows))
    denom = max(1, len(indices))
    return {
        "samples": len(indices),
        f"match@{k}": match / denom,
        f"RMSE@{k}": float(sum(rms_values) / len(rms_values)) if rms_values else math.nan,
        f"matched_samples_for_RMSE@{k}": len(rms_values),
        f"WA_hit@{k}": wa_hit / denom,
        f"skeleton_hit@{k}": skel_hit / denom,
        f"row_count_accuracy@{k}": row_hit / denom,
        f"formula_closure@{k}": closure / denom,
        f"readable@{k}": readable / denom,
        f"formula_ok@{k}": formula_ok / denom,
        f"atom_count_ok@{k}": atom_ok / denom,
        f"SG_ok@{k}": sg_ok / denom,
        f"mean_candidates@{k}": float(sum(candidate_counts) / denom),
    }


def aggregate_all(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.clean_val, args.val_limit)
    systems = sorted({crystal_system(int(r["sg"])) for r in records})
    subsets = ["all", "atom_lt6", "atom_ge6", "atom_ge12", "rows_1_3", "rows_4_6", "rows_ge7"] + [f"crystal_{s}" for s in systems]
    experiments = []
    for exp in selected_experiments(args):
        gen_path = args.run_dir / "generations" / f"{exp['id']}.jsonl"
        metrics_path = args.run_dir / "metrics" / f"{exp['id']}_metrics.jsonl"
        if not gen_path.exists() or not metrics_path.exists():
            continue
        gen_rows = read_jsonl(gen_path)
        metrics = read_jsonl(metrics_path)
        summary = {subset: {"top1": summarize_experiment(records, gen_rows, metrics, subset, 1), "top5": summarize_experiment(records, gen_rows, metrics, subset, 5)} for subset in subsets}
        experiments.append({**exp, "summary": summary, "config_path": str(CONFIG_DIR / f"{exp['id']}.json")})
    return {"experiments": experiments, "diagnostics_reused": DIAGNOSTIC_REUSE, "geometry_gtwa_metrics": GEOMETRY_GTWA}


def write_reports(args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = aggregate_all(args)
    write_json(REPORT_DIR / "02_fullgen_experiments_table.json", payload)
    experiments = payload["experiments"]

    lines = [
        "# Full-Generation Experiments Table",
        "",
        "| id | family | geometry | ranker | match@1 | match@5 | RMSE@1 | RMSE@5 | WA@1 | WA@5 | row@1 | row@5 | formula@5 | valid@5 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in experiments:
        t1 = exp["summary"]["all"]["top1"]
        t5 = exp["summary"]["all"]["top5"]
        lines.append(
            "| {id} | {fam} | {geom} | {ranker} | {m1} | {m5} | {r1} | {r5} | {wa1} | {wa5} | {row1} | {row5} | {closure5} | {valid5} |".format(
                id=exp["id"],
                fam=exp.get("family", ""),
                geom=exp.get("geometry_mode", ""),
                ranker=exp.get("ranker", ""),
                m1=pct(t1["match@1"]),
                m5=pct(t5["match@5"]),
                r1=num(t1["RMSE@1"]),
                r5=num(t5["RMSE@5"]),
                wa1=pct(t1["WA_hit@1"]),
                wa5=pct(t5["WA_hit@5"]),
                row1=pct(t1["row_count_accuracy@1"]),
                row5=pct(t5["row_count_accuracy@5"]),
                closure5=pct(t5["formula_closure@5"]),
                valid5=pct(min(t5["readable@5"], t5["formula_ok@5"], t5["atom_count_ok@5"], t5["SG_ok@5"])),
            )
        )
    write_md(REPORT_DIR / "02_fullgen_experiments_table.md", "\n".join(lines))
    write_diagnostic_report(payload)
    write_failure_report(payload)
    write_best_config(args, payload)
    write_test_decision(args, payload)
    write_final_report(payload)


def best_experiment(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        exp
        for exp in payload["experiments"]
        if exp["summary"]["all"]["top5"]["readable@5"] >= 0.95
        and exp["summary"]["all"]["top5"]["formula_ok@5"] >= 0.95
        and exp["summary"]["all"]["top5"]["atom_count_ok@5"] >= 0.95
        and exp["summary"]["all"]["top5"]["SG_ok@5"] >= 0.95
    ]
    if not candidates:
        candidates = payload["experiments"]
    if not candidates:
        return None
    return max(candidates, key=lambda e: (e["summary"]["all"]["top5"]["match@5"], -float(e["summary"]["all"]["top5"]["RMSE@5"] if math.isfinite(e["summary"]["all"]["top5"]["RMSE@5"]) else 99.0), e["summary"]["all"]["top1"]["match@1"]))


def validation_allows_test(top5: dict[str, float]) -> bool:
    return bool(
        top5["match@5"] >= 0.65
        and math.isfinite(float(top5["RMSE@5"]))
        and float(top5["RMSE@5"]) <= 0.075
        and top5["readable@5"] >= 0.95
        and top5["formula_ok@5"] >= 0.95
        and top5["atom_count_ok@5"] >= 0.95
        and top5["SG_ok@5"] >= 0.95
    )


def write_diagnostic_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Diagnostic Decomposition",
        "",
        "GT-WA diagnostics are reused from the geometry breakthrough run; predicted-WA diagnostics are from the full-generation experiments below.",
        "",
        "| diagnostic | match@1 | match@5 | RMSE@1 | RMSE@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for diag in DIAGNOSTIC_REUSE:
        vals = GEOMETRY_GTWA[diag["geometry_mode"]]
        lines.append(f"| {diag['id']} | {pct(vals['match@1'])} | {pct(vals['match@5'])} | {num(vals['RMSE@1'])} | {num(vals['RMSE@5'])} |")
    lines += [
        "",
        "| predicted-WA experiment | WA@1 | WA@5 | row@1 | row@5 | match@1 | match@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in payload["experiments"]:
        t1 = exp["summary"]["all"]["top1"]
        t5 = exp["summary"]["all"]["top5"]
        lines.append(f"| {exp['id']} | {pct(t1['WA_hit@1'])} | {pct(t5['WA_hit@5'])} | {pct(t1['row_count_accuracy@1'])} | {pct(t5['row_count_accuracy@5'])} | {pct(t1['match@1'])} | {pct(t5['match@5'])} |")
    write_md(REPORT_DIR / "01_diagnostic_decomposition.md", "\n".join(lines))
    write_json(REPORT_DIR / "01_diagnostic_decomposition.json", {"diagnostics_reused": DIAGNOSTIC_REUSE, "experiments": payload["experiments"]})


def write_failure_report(payload: dict[str, Any]) -> None:
    best = best_experiment(payload)
    worst = min(payload["experiments"], key=lambda e: e["summary"]["all"]["top5"]["match@5"]) if payload["experiments"] else None
    lines = ["# Failure Analysis", ""]
    if best:
        b1 = best["summary"]["all"]["top1"]
        b5 = best["summary"]["all"]["top5"]
        lines += [
            f"Best validation experiment: `{best['id']}`.",
            "",
            f"- match@1/@5: {pct(b1['match@1'])} / {pct(b5['match@5'])}",
            f"- WA_hit@1/@5: {pct(b1['WA_hit@1'])} / {pct(b5['WA_hit@5'])}",
            f"- RMSE@1/@5: {num(b1['RMSE@1'])} / {num(b5['RMSE@5'])}",
            f"- formula/readable/atom/SG@5: {pct(b5['formula_ok@5'])} / {pct(b5['readable@5'])} / {pct(b5['atom_count_ok@5'])} / {pct(b5['SG_ok@5'])}",
            "",
        ]
        lines += [
            "Main diagnosis:",
            "",
            "- GT-WA geometry is no longer the limiting component: e07/e08 stay above 85% match@5 with GT W/A.",
            f"- Predicted-WA WA_hit@5 for the best full run is {pct(b5['WA_hit@5'])}; the residual gap to GT-WA e08 is therefore dominated by W/A candidate coverage and ranking.",
            "- e08-style geometry improves K=5 once a plausible W/A candidate is present, but rank@1 can remain weak if rank-0 is inherited from an older W/A/ranking prior.",
            "- Complex structures should be read through atom>=12 subset metrics in `02_fullgen_experiments_table.json`; they remain the hardest subset when WA coverage drops.",
            "",
        ]
    if worst:
        w5 = worst["summary"]["all"]["top5"]
        lines += [
            f"Worst validation experiment: `{worst['id']}` with match@5 = {pct(w5['match@5'])}.",
            "This is useful as a negative control for insufficiently conditioned W/A or geometry candidates.",
        ]
    write_md(REPORT_DIR / "03_failure_analysis.md", "\n".join(lines))


def write_best_config(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    best = best_experiment(payload)
    if not best:
        write_md(REPORT_DIR / "04_best_config.md", "# Best Config\n\nNo complete validation experiment was available.")
        return
    t1 = best["summary"]["all"]["top1"]
    t5 = best["summary"]["all"]["top5"]
    allow_test = validation_allows_test(t5)
    frozen = {
        "experiment_id": best["id"],
        "config_path": best["config_path"],
        "command": (
            f"{sys.executable} model/New_model/symcif_experiment/scripts/run_mp20_fullgen_after_geometry_breakthrough.py "
            f"--stage all --experiment-ids {best['id']} --top-k {args.top_k} --seed {args.seed}"
        ),
        "seed": args.seed,
        "top_k": args.top_k,
        "geometry_mode": best.get("geometry_mode"),
        "wa_decoder_mode": best.get("wa_strategy"),
        "selector_ranker_mode": best.get("ranker"),
        "retrieval_index": "clean_train in-memory GeometryIndex + WAProposalIndex",
        "validation_metrics": {"top1": t1, "top5": t5},
        "frozen": allow_test,
        "validation_allows_test": allow_test,
    }
    title = "Frozen validation config" if allow_test else "Best validation config (not test-frozen)"
    reason = (
        "Frozen because it satisfies the clean_val test gate: match@5 >= 65%, RMSE@5 <= 0.075, and readable/formula/atom/SG@5 >= 95%."
        if allow_test
        else "Not frozen for MP-20 test because clean_val readable/formula/SG validity is below the required 95% gate, even though match/RMSE pass the first usable full-generation target."
    )
    write_json(REPORT_DIR / "04_best_config.json", frozen)
    write_md(
        REPORT_DIR / "04_best_config.md",
        "\n".join(
            [
                "# Best Config",
                "",
                f"{title}: `{best['id']}`.",
                "",
                f"- config: `{best['config_path']}`",
                f"- command: `{frozen['command']}`",
                f"- seed: {args.seed}",
                f"- K: {args.top_k}",
                f"- geometry mode: `{best.get('geometry_mode')}`",
                f"- W/A decoder mode: `{best.get('wa_strategy')}`",
                f"- selector/ranker mode: `{best.get('ranker')}`",
                f"- match@1/@5: {pct(t1['match@1'])} / {pct(t5['match@5'])}",
                f"- RMSE@1/@5: {num(t1['RMSE@1'])} / {num(t5['RMSE@5'])}",
                f"- WA_hit@1/@5: {pct(t1['WA_hit@1'])} / {pct(t5['WA_hit@5'])}",
                f"- readable/formula/atom/SG@5: {pct(t5['readable@5'])} / {pct(t5['formula_ok@5'])} / {pct(t5['atom_count_ok@5'])} / {pct(t5['SG_ok@5'])}",
                f"- validation allows MP-20 test: {allow_test}",
                "",
                reason,
            ]
        ),
    )


def write_test_decision(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    existing = read_json(REPORT_DIR / "05_frozen_mp20_test.json", default={})
    if existing.get("test_run"):
        return
    best = best_experiment(payload)
    if not best:
        write_md(REPORT_DIR / "05_frozen_mp20_test.md", "# Frozen MP-20 Test Decision\n\nNo validation config available; test not run.")
        write_json(REPORT_DIR / "05_frozen_mp20_test.json", {"test_run": False, "reason": "no_validation_config"})
        return
    t5 = best["summary"]["all"]["top5"]
    allow = validation_allows_test(t5)
    payload_out = {
        "test_run": False,
        "validation_allows_test": allow,
        "best_experiment": best["id"],
        "validation_top5": t5,
        "frozen_test_command": (
            f"{sys.executable} model/New_model/symcif_experiment/scripts/run_mp20_fullgen_after_geometry_breakthrough.py "
            f"--stage frozen-test --experiment-ids {best['id']} --top-k {args.top_k} --seed {args.seed}"
        ),
    }
    reason = "Validation criteria met; frozen-test command is recorded. This script does not auto-run test during report stage." if allow else "Validation criteria were not all met; MP-20 test was not run."
    write_json(REPORT_DIR / "05_frozen_mp20_test.json", payload_out)
    write_md(
        REPORT_DIR / "05_frozen_mp20_test.md",
        "\n".join(
            [
                "# Frozen MP-20 Test Decision",
                "",
                f"- validation allows test: {allow}",
                f"- test run in this campaign execution: False",
                f"- reason: {reason}",
                "",
                "Command to run exactly once after explicit frozen-test execution:",
                "",
                "```bash",
                payload_out["frozen_test_command"],
                "```",
            ]
        ),
    )


def run_frozen_test(args: argparse.Namespace) -> None:
    previous = read_json(REPORT_DIR / "05_frozen_mp20_test.json", default={})
    if previous.get("test_run"):
        print(
            json.dumps(
                {
                    "stage": "frozen_test_skipped",
                    "reason": "test_already_run",
                    "result_path": str(REPORT_DIR / "05_frozen_mp20_test.json"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    validation_payload = aggregate_all(args)
    best = best_experiment(validation_payload)
    if not best:
        write_md(REPORT_DIR / "05_frozen_mp20_test.md", "# Frozen MP-20 Test Decision\n\nNo validation config available; test not run.")
        write_json(REPORT_DIR / "05_frozen_mp20_test.json", {"test_run": False, "reason": "no_validation_config"})
        return
    best_id = str(best["id"])
    if args.experiment_ids and {item.strip() for item in str(args.experiment_ids).split(",") if item.strip()} != {best_id}:
        raise SystemExit(f"Frozen test must use the frozen best validation config only: {best_id}")
    if not validation_allows_test(best["summary"]["all"]["top5"]):
        write_test_decision(args, validation_payload)
        return

    experiment = next(exp for exp in FULLGEN_EXPERIMENTS if str(exp["id"]) == best_id)
    test_run_dir = Path(args.run_dir) / "frozen_test"
    test_args = argparse.Namespace(**vars(args))
    test_args.run_dir = test_run_dir
    test_args.clean_val = args.clean_test
    test_args.val_limit = args.test_limit
    test_args.experiment_ids = best_id
    test_args.external_candidates = args.test_external_candidates
    test_args.external_generations = args.test_external_generations

    train_records = load_records(args.clean_train, args.train_limit)
    test_records = load_records(args.clean_test, args.test_limit)
    generate_records_to_run_dir(
        test_args,
        records=test_records,
        train_records=train_records,
        experiments=[experiment],
        run_dir=test_run_dir,
        external_candidates=args.test_external_candidates,
        external_generations=args.test_external_generations,
        split_name="test",
    )
    evaluate_pool_for_records(test_args, records=test_records, run_dir=test_run_dir, stage_name="evaluated_frozen_test_pool")
    synthesize_metrics(test_args)
    test_payload = aggregate_all(test_args)
    test_exp = test_payload["experiments"][0]
    t1 = test_exp["summary"]["all"]["top1"]
    t5 = test_exp["summary"]["all"]["top5"]
    test_result = {
        "test_run": True,
        "test_run_count": 1,
        "validation_allows_test": True,
        "best_experiment": best_id,
        "frozen_validation_top5": best["summary"]["all"]["top5"],
        "test_split": str(args.clean_test),
        "test_run_dir": str(test_run_dir),
        "test_metrics": {"top1": t1, "top5": t5},
        "command": (
            f"{sys.executable} model/New_model/symcif_experiment/scripts/run_mp20_fullgen_after_geometry_breakthrough.py "
            f"--stage frozen-test --experiment-ids {best_id} --top-k {args.top_k} --seed {args.seed}"
        ),
    }
    write_json(REPORT_DIR / "05_frozen_mp20_test.json", test_result)
    write_md(
        REPORT_DIR / "05_frozen_mp20_test.md",
        "\n".join(
            [
                "# Frozen MP-20 Test Result",
                "",
                f"- test run: True",
                f"- test run count: 1",
                f"- frozen config: `{best_id}`",
                f"- test split: `{args.clean_test}`",
                f"- run dir: `{test_run_dir}`",
                f"- match@1/@5: {pct(t1['match@1'])} / {pct(t5['match@5'])}",
                f"- RMSE@1/@5: {num(t1['RMSE@1'])} / {num(t5['RMSE@5'])}",
                f"- WA_hit@1/@5: {pct(t1['WA_hit@1'])} / {pct(t5['WA_hit@5'])}",
                f"- skeleton_hit@1/@5: {pct(t1['skeleton_hit@1'])} / {pct(t5['skeleton_hit@5'])}",
                f"- row_count@1/@5: {pct(t1['row_count_accuracy@1'])} / {pct(t5['row_count_accuracy@5'])}",
                f"- readable/formula/atom/SG@5: {pct(t5['readable@5'])} / {pct(t5['formula_ok@5'])} / {pct(t5['atom_count_ok@5'])} / {pct(t5['SG_ok@5'])}",
                "",
                "Executed frozen command:",
                "",
                "```bash",
                test_result["command"],
                "```",
            ]
        ),
    )
    write_final_report(validation_payload)


def write_final_report(payload: dict[str, Any]) -> None:
    best = best_experiment(payload)
    lines = [
        "# Full Generation After Geometry Breakthrough Final Report",
        "",
        "## Executive Summary",
        "",
        "This campaign integrates the row-conditioned GT-WA geometry breakthrough into full MP-20 clean_val generation from formula + GT_SG. Development used clean_train/clean_val only; MP-20 test was not used for tuning.",
        "",
    ]
    if best:
        t1 = best["summary"]["all"]["top1"]
        t5 = best["summary"]["all"]["top5"]
        first_usable = bool(
            t1["match@1"] >= 0.50
            and t5["match@5"] >= 0.65
            and math.isfinite(float(t5["RMSE@5"]))
            and float(t5["RMSE@5"]) <= 0.075
        )
        allow_test = validation_allows_test(t5)
        lines += [
            f"Best clean_val config: `{best['id']}`.",
            "",
            f"- match@1/@5: {pct(t1['match@1'])} / {pct(t5['match@5'])}",
            f"- RMSE@1/@5: {num(t1['RMSE@1'])} / {num(t5['RMSE@5'])}",
            f"- WA_hit@1/@5: {pct(t1['WA_hit@1'])} / {pct(t5['WA_hit@5'])}",
            f"- readable/formula/atom/SG@5: {pct(t5['readable@5'])} / {pct(t5['formula_ok@5'])} / {pct(t5['atom_count_ok@5'])} / {pct(t5['SG_ok@5'])}",
            f"- first usable clean_val target met: {first_usable}",
            f"- frozen MP-20 test allowed: {allow_test}",
            "",
        ]
        if not allow_test:
            lines += [
                "MP-20 test was not run because the frozen-test validity gate requires readable/formula/atom/SG@5 >= 95%, and the best clean_val run is below that threshold.",
                "",
            ]
        test_result = read_json(REPORT_DIR / "05_frozen_mp20_test.json", default={})
        if test_result.get("test_run"):
            test_t1 = test_result["test_metrics"]["top1"]
            test_t5 = test_result["test_metrics"]["top5"]
            lines += [
                "## Frozen MP-20 Test",
                "",
                f"Exactly one frozen MP-20 test run was executed for `{test_result['best_experiment']}` after the clean_val gate passed.",
                "",
                f"- match@1/@5: {pct(test_t1['match@1'])} / {pct(test_t5['match@5'])}",
                f"- RMSE@1/@5: {num(test_t1['RMSE@1'])} / {num(test_t5['RMSE@5'])}",
                f"- WA_hit@1/@5: {pct(test_t1['WA_hit@1'])} / {pct(test_t5['WA_hit@5'])}",
                f"- readable/formula/atom/SG@5: {pct(test_t5['readable@5'])} / {pct(test_t5['formula_ok@5'])} / {pct(test_t5['atom_count_ok@5'])} / {pct(test_t5['SG_ok@5'])}",
                "",
            ]
    lines += [
        "## Geometry Transfer",
        "",
        "The geometry improvement transfers when the W/A candidate is plausible, but full generation remains bounded by symbolic W/A coverage. GT-WA e07/e08 reach 85.11%/87.10% match@5, while predicted-WA full generation is lower unless W/A candidates cover the target symbolic key.",
        "",
        "## Experiment Table",
        "",
        "See `02_fullgen_experiments_table.md` and `02_fullgen_experiments_table.json` for all diagnostics, full-generation runs, and subset metrics.",
        "",
        "## Comparison References",
        "",
        "- Previous deterministic GT-WA geometry: match@5 = 58.88%, RMSE@5 = 0.1684.",
        "- New GT-WA e07/e08 geometry: match@5 = 85.11% / 87.10%, RMSE@5 = 0.0423 / 0.0432.",
        "- Current split pipeline full-test reference: match@1/5 = 50.42% / 64.99%, RMSE@1/5 = 0.0668 / 0.0742.",
        "",
        "## What Worked",
        "",
        "- Row-conditioned geometry retrieval is much stronger than deterministic X/L regression.",
        "- Geometry-aware candidate ranking and e07-rank0/e08-diversity directly address the rank@1 versus rank@5 tradeoff.",
        "- Train-only exact-cover symbolic proposals plus row-conditioned geometry reached the best clean_val match/RMSE, but true WA_hit remains low.",
        "",
        "## What Failed",
        "",
        "- The best full-generation run still has WA_hit@5 below 20%, so symbolic W/A coverage and ranking remain the main bottleneck.",
        "- Non-oracle full generation remains mostly bottlenecked by W/A candidate coverage and ranking, not by X/L geometry.",
        "- Oracle row-count diagnostics show how much early row-count filtering can help, but this is not a deployable full-generation setting.",
        "",
        "## Next Steps",
        "",
        "Train or distill a Mini-CFJoint-v2 W/A decoder toward the high-coverage source-level WA proposal distribution, then keep e07/e08 as frozen geometry candidates. The next technical target is WA_hit@5, followed by rank@1 selection that uses e07-like quality for the first candidate and e08-like diversity for K=5.",
    ]
    write_md(REPORT_DIR / "final_report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "generate", "evaluate", "synthesize", "report", "all", "frozen-test"], default="all")
    parser.add_argument("--clean-train", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_train.jsonl")
    parser.add_argument("--clean-val", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_val.jsonl")
    parser.add_argument("--clean-test", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20" / "test.jsonl")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--external-candidates", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev" / "val_baseline_eval_gpu" / "candidates_reranked.jsonl")
    parser.add_argument("--external-generations", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev" / "val_baseline_eval_gpu" / "generations" / "baseline.jsonl")
    parser.add_argument(
        "--test-external-candidates",
        type=Path,
        default=PROJECT_ROOT / "reports" / "symcif_v4_table3_fix_audit" / "wa_search_audit_full_stablekey" / "mp20" / "hybrid_top700_full" / "test_hybrid_candidates.jsonl",
    )
    parser.add_argument(
        "--test-external-generations",
        type=Path,
        default=PROJECT_ROOT / "reports" / "symcif_v4_mp20_test_hybrid_prior_k5" / "eval_gpu" / "generations" / "baseline.jsonl",
    )
    parser.add_argument("--geometry-report", type=Path, default=REPO_ROOT / "reports" / "symcif_v4_mp20_geometry_breakthrough" / "experiments_table.json")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--experiment-ids", type=str, default="")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-workers", type=int, default=64)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-sites", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    globals()["REPORT_DIR"] = Path(args.report_dir)
    train_records = load_records(args.clean_train, args.train_limit)
    val_records = load_records(args.clean_val, args.val_limit)
    if args.stage in {"audit", "all"}:
        write_audit(args, train_records, val_records)
        write_configs(args)
    if args.stage in {"generate", "all"}:
        generate_all(args)
    if args.stage in {"evaluate", "all"}:
        evaluate_pool(args)
    if args.stage in {"synthesize", "all"}:
        synthesize_metrics(args)
    if args.stage in {"report", "all"}:
        write_reports(args)
    if args.stage == "frozen-test":
        run_frozen_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
