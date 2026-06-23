#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402

import run_mp20_fullgen_after_geometry_breakthrough as fg  # noqa: E402
import run_mp20_geometry_breakthrough as gb  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402


REPORT_DIR = REPO_ROOT / "reports" / "symcif_v5_multidataset_wa_decoder"
RUN_DIR = REPO_ROOT / "runs" / "symcif_v5_multidataset_wa_decoder"
LOG_DIR = PROJECT_ROOT / "Log_GPT" / "round_20260530_02_symcif_v5_multidataset_wa_decoder"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train: Path
    val: Path
    test: Path
    clean_target: bool
    notes: str
    external_val_candidates: Path | None = None
    external_val_generations: Path | None = None
    external_test_candidates: Path | None = None
    external_test_generations: Path | None = None


DATASETS: dict[str, DatasetSpec] = {
    "mp20": DatasetSpec(
        name="mp20",
        train=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_train.jsonl",
        val=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_val.jsonl",
        test=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20" / "test.jsonl",
        clean_target=True,
        notes="MP-20 uses v2 clean_train/clean_val for development; no separate clean_test artifact exists, so frozen test points at structured_symcif_v4_mp20/test.jsonl.",
        external_val_candidates=PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev" / "val_baseline_eval_gpu" / "candidates_reranked.jsonl",
        external_val_generations=PROJECT_ROOT / "reports" / "symcif_v4_mp20_k5_dev" / "val_baseline_eval_gpu" / "generations" / "baseline.jsonl",
        external_test_candidates=PROJECT_ROOT
        / "reports"
        / "symcif_v4_table3_fix_audit"
        / "wa_search_audit_full_stablekey"
        / "mp20"
        / "hybrid_top700_full"
        / "test_hybrid_candidates.jsonl",
        external_test_generations=PROJECT_ROOT / "reports" / "symcif_v4_mp20_test_hybrid_prior_k5" / "eval_gpu" / "generations" / "baseline.jsonl",
    ),
    "mp20_structured": DatasetSpec(
        name="mp20_structured",
        train=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20" / "train.jsonl",
        val=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20" / "val.jsonl",
        test=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20" / "test.jsonl",
        clean_target=False,
        notes="Structured MP-20 official split; useful for dataset audits and transfer checks, not the clean MP-20 v2 development split.",
    ),
    "mpts52": DatasetSpec(
        name="mpts52",
        train=PROJECT_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl",
        val=PROJECT_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl",
        test=PROJECT_ROOT / "data" / "structured_symcif_v4_mpts52" / "test.jsonl",
        clean_target=False,
        notes="MPTS-52 structured SymCIF-v4 split exists and is schema-compatible; free-parameter fallback is common and must be tracked in reports.",
    ),
    "structured_v4_small": DatasetSpec(
        name="structured_v4_small",
        train=PROJECT_ROOT / "data" / "structured_symcif_v4" / "train.jsonl",
        val=PROJECT_ROOT / "data" / "structured_symcif_v4" / "val.jsonl",
        test=PROJECT_ROOT / "data" / "structured_symcif_v4" / "test.jsonl",
        clean_target=False,
        notes="Small structured SymCIF-v4 dataset already present in repo; used only for compatibility smoke unless explicitly selected.",
    ),
}


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "v5_a1_exact_cover_sg_formula_e08",
        "family": "A_exact_cover",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Train-only SG/formula exact-cover prior with e08 geometry.",
    },
    {
        "id": "v5_a2_pred_rowcount_beam_e08",
        "family": "A_exact_cover",
        "wa_source": "internal",
        "wa_strategy": "pred_rowcount",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Row-count-aware beam using a train-only row-count predictor, not target row_count.",
    },
    {
        "id": "v5_a3_unique_wa_beam_e08",
        "family": "A_exact_cover",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Exact-cover beam with five unique W/A candidates before geometry diversity.",
    },
    {
        "id": "v5_a4_exact_cover_diversity_e08",
        "family": "A_exact_cover",
        "wa_source": "internal",
        "wa_strategy": "diverse_skeleton",
        "selector": "wa_diversity",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Exact-cover beam with duplicate stable-key dedup and skeleton diversity.",
    },
    {
        "id": "v5_b1_skeleton_retrieval_exact_assignment_e08",
        "family": "B_skeleton_first",
        "wa_source": "internal",
        "wa_strategy": "skeleton_retrieval",
        "selector": "skeleton_first",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Skeleton retrieval prior followed by formula exact-cover element assignment.",
    },
    {
        "id": "v5_b2_skeleton_assignment_diverse_e08",
        "family": "B_skeleton_first",
        "wa_source": "internal",
        "wa_strategy": "diverse_skeleton",
        "selector": "skeleton_diversity",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Top-N skeletons times exact-cover assignments with assignment diversity.",
    },
    {
        "id": "v5_b3_skeleton_beam_assignment_beam_e08",
        "family": "B_skeleton_first",
        "wa_source": "internal",
        "wa_strategy": "wide_assignment",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Wider exact-cover assignment beam under train skeleton priors.",
    },
    {
        "id": "v5_b4_skeleton_train_prior_neural_proxy_e08",
        "family": "B_skeleton_first",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "policy_then_internal",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Policy-search candidates used as the neural proxy, with train-prior skeleton fallback.",
    },
    {
        "id": "v5_c1_train_symbolic_prior_only_e08",
        "family": "C_retrieval_augmented",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "selector": "source_distance",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Pure train symbolic memory prior, exact-cover adapted to target formula.",
    },
    {
        "id": "v5_c2_train_prior_exact_adapt_e08",
        "family": "C_retrieval_augmented",
        "wa_source": "internal",
        "wa_strategy": "wide_assignment",
        "selector": "source_distance",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Train prior with more exact-cover assignment alternatives per skeleton.",
    },
    {
        "id": "v5_c3_train_prior_generated_union_e08",
        "family": "C_retrieval_augmented",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "hybrid_union",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Union of policy/hybrid W/A proposals and train-prior exact-cover proposals.",
    },
    {
        "id": "v5_c4_train_prior_geometry_support_rank_e08",
        "family": "C_retrieval_augmented",
        "wa_source": "internal",
        "wa_strategy": "sg_formula",
        "selector": "geometry_support",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Train prior ranked by row-conditioned geometry support distance.",
    },
    {
        "id": "v5_d1_complex_weighted_retrieval_e08",
        "family": "D_complex_expert",
        "wa_source": "internal",
        "wa_strategy": "complex_weighted",
        "selector": "complex_priority",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Complex-weighted retrieval for atom>=12 or many-element formulas.",
    },
    {
        "id": "v5_d2_rows_ge7_expert_e08",
        "family": "D_complex_expert",
        "wa_source": "internal",
        "wa_strategy": "pred_rowcount_complex",
        "selector": "complex_priority",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Rows>=7 expert route based on train-only predicted row-count bucket.",
    },
    {
        "id": "v5_d3_atom_ge12_expert_e08",
        "family": "D_complex_expert",
        "wa_source": "internal",
        "wa_strategy": "atom_ge12_expert",
        "selector": "complex_priority",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Atom>=12 expert route using high-atom train source pools.",
    },
    {
        "id": "v5_d4_adaptive_internal_pool_complex_e08",
        "family": "D_complex_expert",
        "wa_source": "internal",
        "wa_strategy": "adaptive_complex_pool",
        "selector": "complex_priority",
        "geometry_mode": "e08",
        "geometry_plan": "adaptive",
        "description": "Larger internal candidate pool only for complex samples; final K remains 5.",
    },
    {
        "id": "v5_e1_geometry_distance_ranking_e08",
        "family": "E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "geometry_support",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Non-oracle geometry-distance ranking over hybrid and train-prior W/A candidates.",
    },
    {
        "id": "v5_e2_symbolic_geometry_hybrid_ranking_e08",
        "family": "E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "symbolic_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "description": "Hybrid symbolic score plus geometry support ranking.",
    },
    {
        "id": "v5_e3_diversity_aware_top5_e08",
        "family": "E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "wa_diversity",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Diversity-aware top5 selection by stable W/A key.",
    },
    {
        "id": "v5_e4_e07_rank0_e08_symbolic_diverse",
        "family": "E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "symbolic_geometry",
        "geometry_mode": "e07_rank0_e08",
        "geometry_plan": "e07_rank0_e08",
        "description": "Rank-1 e07-like quality with e08 diversity for K=5.",
    },
    {
        "id": "v5_f1_policy_neural_candidates_e08",
        "family": "F_neural_wa_proxy",
        "wa_source": "external_policy",
        "wa_strategy": "policy_top",
        "selector": "policy_score",
        "geometry_mode": "e08",
        "geometry_plan": "unique_wa",
        "description": "Existing factorized step-policy W/A candidates plugged into e08 geometry as the neural W/A attempt.",
    },
    {
        "id": "v5_diag_oracle_rowcount_e08",
        "family": "diagnostic",
        "wa_source": "internal",
        "wa_strategy": "oracle_rowcount",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "diagnostic_uses_gt_row_count": True,
        "description": "Diagnostic only: target row_count filtered exact-cover route.",
    },
    {
        "id": "v5_diag_oracle_w_pred_a_e08",
        "family": "diagnostic",
        "wa_source": "oracle_w",
        "wa_strategy": "gt_skeleton_predicted_assignment",
        "selector": "score",
        "geometry_mode": "e08",
        "geometry_plan": "default",
        "diagnostic_uses_gt_w": True,
        "description": "Diagnostic only: GT skeleton rows with predicted assignment.",
    },
]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return fg.read_jsonl(path, limit=limit)


def write_json(path: Path, payload: Any) -> None:
    fg.write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    fg.write_jsonl(path, rows)


def write_md(path: Path, text: str) -> None:
    fg.write_md(path, text)


def pct(value: float | None) -> str:
    return fg.pct(value)


def num(value: float | None) -> str:
    return fg.num(value)


def load_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return [v2.geometry_training_record(r) for r in read_jsonl(path, limit=limit)]


def dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset: {name}")
    return DATASETS[name]


def split_path(spec: DatasetSpec, split: str) -> Path:
    if split == "train":
        return spec.train
    if split == "val":
        return spec.val
    if split == "test":
        return spec.test
    raise KeyError(split)


def formula_key(record: dict[str, Any]) -> str:
    return fg.formula_key(record)


def atom_count(record: dict[str, Any]) -> int:
    return fg.atom_count(record)


def row_count(record: dict[str, Any]) -> int:
    return fg.row_count(record)


def formula_elem_count(record: dict[str, Any]) -> int:
    return len(record.get("formula_counts") or {})


def atom_bucket(record: dict[str, Any]) -> str:
    n = atom_count(record)
    if n < 6:
        return "atom_lt6"
    if n >= 12:
        return "atom_ge12"
    return "atom_ge6_lt12"


def row_bucket(n: int) -> str:
    return fg.row_bucket(n)


class SafeWAProposalIndex(fg.WAProposalIndex):
    """Train-only W/A proposal index.

    The v4 `sg_rowcount` route used `row_count(target)`. This v5 index keeps that
    behavior only for explicitly diagnostic calls and otherwise predicts row_count
    from train statistics keyed by formula/SG features.
    """

    def __init__(self, train_records: list[dict[str, Any]], engine: OrbitEngine) -> None:
        super().__init__(train_records, engine)
        self.row_prior_by_sg_atom: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
        self.row_prior_by_sg_elem: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.row_prior_by_atom_elem: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
        self.by_atom_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_formula_len: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.by_atom_elem: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for rec in train_records:
            rc = row_count(rec)
            sg = int(rec["sg"])
            ab = atom_bucket(rec)
            ne = formula_elem_count(rec)
            self.row_prior_by_sg_atom[(sg, ab)][rc] += 1
            self.row_prior_by_sg_elem[(sg, ne)][rc] += 1
            self.row_prior_by_atom_elem[(ab, ne)][rc] += 1
            self.by_atom_bucket[ab].append(rec)
            self.by_formula_len[ne].append(rec)
            self.by_atom_elem[(ab, ne)].append(rec)

    def predict_row_count(self, target: dict[str, Any]) -> int:
        sg = int(target["sg"])
        ab = atom_bucket(target)
        ne = formula_elem_count(target)
        for counter in (
            self.row_prior_by_sg_atom.get((sg, ab)),
            self.row_prior_by_sg_elem.get((sg, ne)),
            self.row_prior_by_atom_elem.get((ab, ne)),
        ):
            if counter:
                return int(counter.most_common(1)[0][0])
        values = [row_count(r) for r in self.train_records]
        return int(statistics.median(values)) if values else 1

    def predict_row_count_without_sg(self, target: dict[str, Any]) -> int:
        ab = atom_bucket(target)
        ne = formula_elem_count(target)
        counter = self.row_prior_by_atom_elem.get((ab, ne))
        if counter:
            return int(counter.most_common(1)[0][0])
        values = [row_count(r) for r in self.train_records]
        return int(statistics.median(values)) if values else 1

    def formula_only_source_distance(self, target: dict[str, Any], source: dict[str, Any], pred_rc: int, cheap: bool = False) -> float:
        dist = fg.formula_l1(fg.formula_frac(target), fg.formula_frac(source))
        dist += 0.02 * abs(atom_count(target) - atom_count(source))
        dist += 0.04 * abs(row_count(source) - int(pred_rc))
        if not cheap:
            target_counts = sorted(int(v) for v in target["formula_counts"].values())
            source_counts = sorted(int(v) for v in source["formula_counts"].values())
            dist += 0.03 * abs(len(target_counts) - len(source_counts))
            dist += 0.01 * sum(abs(a - b) for a, b in zip(target_counts, source_counts))
        return float(dist)

    def source_pool_without_sg(self, target: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
        pred_rc = self.predict_row_count_without_sg(target)
        complex_like = atom_count(target) >= 12 or formula_elem_count(target) >= 4 or pred_rc >= 7
        pools = [
            self.by_atom_elem.get((atom_bucket(target), formula_elem_count(target)), []),
            self.by_atom_bucket.get(atom_bucket(target), []),
            self.by_formula_len.get(formula_elem_count(target), []),
            self.train_records,
        ]
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []
        pre_limit = 220 if complex_like else 140
        pool_limit = 320 if complex_like else 220
        for source_pool in pools:
            reduced = source_pool
            if len(reduced) > pre_limit:
                reduced = sorted(reduced, key=lambda r: self.formula_only_source_distance(target, r, pred_rc, cheap=True))[:pre_limit]
            for rec in reduced:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= pool_limit:
                break
        return sorted(pool, key=lambda r: self.formula_only_source_distance(target, r, pred_rc, cheap=False))

    def source_pool(self, target: dict[str, Any], strategy: str, oracle_row_count: bool = False) -> list[dict[str, Any]]:
        if oracle_row_count or strategy == "oracle_rowcount":
            return super().source_pool(target, "sg_rowcount", oracle_row_count=True)
        sg = int(target["sg"])
        pred_rc = self.predict_row_count(target)
        complex_like = atom_count(target) >= 12 or formula_elem_count(target) >= 4 or pred_rc >= 7
        if strategy in {"pred_rowcount", "pred_rowcount_complex"}:
            pools = [self.by_sg_row.get((sg, pred_rc), []), self.by_sg.get(sg, []), self.by_crystal.get(fg.crystal_system(sg), []), self.train_records]
        elif strategy in {"complex_weighted", "adaptive_complex_pool"}:
            pools = [
                self.by_atom_bucket.get(atom_bucket(target), []),
                self.by_formula_len.get(formula_elem_count(target), []),
                self.by_sg.get(sg, []),
                self.by_crystal.get(fg.crystal_system(sg), []),
                self.train_records,
            ]
        elif strategy == "atom_ge12_expert":
            pools = [self.by_atom_bucket.get("atom_ge12", []), self.by_sg.get(sg, []), self.train_records]
        elif strategy in {"skeleton_retrieval", "wide_assignment", "diverse_skeleton"}:
            pools = [self.by_sg.get(sg, []), self.by_crystal.get(fg.crystal_system(sg), []), self.by_atom_bucket.get(atom_bucket(target), []), self.train_records]
        else:
            pools = [self.by_sg.get(sg, []), self.by_crystal.get(fg.crystal_system(sg), []), self.train_records]
        seen: set[str] = set()
        pool: list[dict[str, Any]] = []
        pre_limit = 320 if complex_like else 180
        pool_limit = 420 if complex_like else 240
        for source_pool in pools:
            reduced = source_pool
            if len(reduced) > pre_limit:
                reduced = sorted(reduced, key=lambda r: self.source_distance(target, r, cheap=True))[:pre_limit]
            for rec in reduced:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= pool_limit:
                break
        return sorted(pool, key=lambda r: self.safe_source_distance(target, r, pred_rc))

    def safe_source_distance(self, target: dict[str, Any], source: dict[str, Any], pred_rc: int) -> float:
        dist = self.source_distance(target, source, cheap=False)
        dist += 0.04 * abs(row_count(source) - int(pred_rc))
        if atom_count(target) >= 12 and atom_count(source) < 12:
            dist += 0.25
        if formula_elem_count(target) >= 4 and formula_elem_count(source) < 4:
            dist += 0.18
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
        complex_like = atom_count(target) >= 12 or formula_elem_count(target) >= 4 or self.predict_row_count(target) >= 7
        max_assignments = 2
        if strategy in {"diverse_skeleton", "wide_assignment", "skeleton_retrieval"}:
            max_assignments = 6
        if strategy in {"complex_weighted", "pred_rowcount_complex", "atom_ge12_expert", "adaptive_complex_pool"}:
            max_assignments = 8
            max_sources = max(max_sources, 140 if complex_like else 80)
        source_rank = 0
        for source in self.source_pool(target, strategy, oracle_row_count=oracle_row_count)[:max_sources]:
            source_rank += 1
            skeleton = fg.normalize_rows(self.engine, v2.canonical_rows(source))
            assignments = fg.exact_cover_assignments(
                skeleton,
                {str(k): int(v) for k, v in target["formula_counts"].items()},
                source_rows=v2.canonical_rows(source),
                max_assignments=max_assignments,
            )
            for assignment_rank, rows in enumerate(assignments):
                skel, wa = v2.canonical_keys_from_rows(rows)
                if wa in seen_wa:
                    continue
                seen_wa.add(wa)
                pred_rc = self.predict_row_count(target)
                dist = self.safe_source_distance(target, source, pred_rc)
                score = -dist - 0.001 * source_rank - 0.01 * assignment_rank
                out.append(
                    {
                        "rows": rows,
                        "canonical_skeleton_key": skel,
                        "canonical_wa_key": wa,
                        "score": score,
                        "source_sample_id": source["sample_id"],
                        "source_distance": dist,
                        "predicted_row_count": pred_rc,
                    }
                )
                if len(out) >= top_n:
                    self._proposal_cache[cache_key] = list(out)
                    return out
        self._proposal_cache[cache_key] = list(out)
        return out

    def propose_without_sg(
        self,
        target: dict[str, Any],
        *,
        strategy: str,
        top_n: int,
        max_sources: int = 80,
    ) -> list[dict[str, Any]]:
        cache_key = (str(target.get("sample_id")), f"without_sg:{strategy}", int(top_n), False, int(max_sources))
        cached = self._proposal_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        out: list[dict[str, Any]] = []
        seen_wa: set[str] = set()
        pred_rc = self.predict_row_count_without_sg(target)
        complex_like = atom_count(target) >= 12 or formula_elem_count(target) >= 4 or pred_rc >= 7
        max_assignments = 2
        if strategy in {"diverse_skeleton", "wide_assignment", "skeleton_retrieval"}:
            max_assignments = 6
        if strategy in {"complex_weighted", "pred_rowcount_complex", "atom_ge12_expert", "adaptive_complex_pool"}:
            max_assignments = 8
            max_sources = max(max_sources, 140 if complex_like else 80)
        source_rank = 0
        for source in self.source_pool_without_sg(target, strategy)[:max_sources]:
            source_rank += 1
            skeleton = fg.normalize_rows(self.engine, v2.canonical_rows(source))
            assignments = fg.exact_cover_assignments(
                skeleton,
                {str(k): int(v) for k, v in target["formula_counts"].items()},
                source_rows=v2.canonical_rows(source),
                max_assignments=max_assignments,
            )
            for assignment_rank, rows in enumerate(assignments):
                skel, wa = v2.canonical_keys_from_rows(rows)
                if wa in seen_wa:
                    continue
                seen_wa.add(wa)
                dist = self.formula_only_source_distance(target, source, pred_rc, cheap=False)
                score = -dist - 0.001 * source_rank - 0.01 * assignment_rank
                out.append(
                    {
                        "rows": rows,
                        "canonical_skeleton_key": skel,
                        "canonical_wa_key": wa,
                        "score": score,
                        "source_sample_id": source["sample_id"],
                        "source_distance": dist,
                        "predicted_row_count": pred_rc,
                        "predicted_sg": int(source["sg"]),
                        "input_gt_sg_removed": True,
                    }
                )
                if len(out) >= top_n:
                    self._proposal_cache[cache_key] = list(out)
                    return out
        self._proposal_cache[cache_key] = list(out)
        return out


class CandidateStore:
    def __init__(self, spec: DatasetSpec, split: str, engine: OrbitEngine) -> None:
        if split == "test":
            cand_path = spec.external_test_candidates
            gen_path = spec.external_test_generations
        else:
            cand_path = spec.external_val_candidates
            gen_path = spec.external_val_generations
        if cand_path and gen_path and cand_path.exists() and gen_path.exists():
            self.external: fg.ExternalCandidates | fg.NullExternalCandidates = fg.ExternalCandidates(cand_path, gen_path, engine)
            self.enabled = True
            self.candidate_path = cand_path
        else:
            self.external = fg.NullExternalCandidates()
            self.enabled = False
            self.candidate_path = cand_path

    def candidates(self, sample_id: str, top_n: int) -> list[dict[str, Any]]:
        return self.external.candidates(sample_id, top_n)

    def baseline_render(self, sample_id: str, rank: int) -> dict[str, Any] | None:
        return self.external.baseline_render(sample_id, rank)


class NullCandidateStore:
    enabled = False
    candidate_path = None

    def candidates(self, sample_id: str, top_n: int) -> list[dict[str, Any]]:
        return []

    def baseline_render(self, sample_id: str, rank: int) -> dict[str, Any] | None:
        return None


def selected_experiments(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return EXPERIMENTS
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    return [e for e in EXPERIMENTS if str(e["id"]) in wanted]


def selected_experiments_need_external(args: argparse.Namespace) -> bool:
    external_sources = {"external_policy", "external_hybrid", "hybrid_union", "oracle_w"}
    return any(str(exp.get("wa_source")) in external_sources for exp in selected_experiments(args.experiment_ids))


def geometry_support_score(record: dict[str, Any], cand: dict[str, Any], geom_index: gb.GeometryIndex) -> float:
    pred = fg.pseudo_record(record, cand["rows"])
    sources = geom_index.candidates(pred, "row_conditioned_knn", 1)
    if not sources:
        return 9999.0
    return float(gb.row_condition_distance(pred, sources[0]))


def order_candidates(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    selector: str,
    geom_index: gb.GeometryIndex,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    selector = str(selector)
    if selector == "wa_diversity":
        out: list[dict[str, Any]] = []
        seen_skel: set[str] = set()
        for cand in sorted(candidates, key=lambda c: -float(c.get("score", c.get("hybrid_score", 0.0)))):
            skel = str(cand.get("canonical_skeleton_key") or "")
            if skel in seen_skel and len(out) < 4:
                continue
            seen_skel.add(skel)
            out.append(cand)
        for cand in candidates:
            if cand not in out:
                out.append(cand)
        return out
    if selector in {"geometry_support", "symbolic_geometry"}:
        scored = []
        for cand in candidates:
            gdist = geometry_support_score(record, cand, geom_index)
            sym_score = -float(cand.get("score", cand.get("hybrid_score", 0.0)))
            if selector == "symbolic_geometry":
                key = (0.40 * gdist + 0.60 * sym_score, gdist, sym_score)
            else:
                key = (gdist, sym_score)
            scored.append((key, cand))
        return [cand for _, cand in sorted(scored, key=lambda item: item[0])]
    if selector == "source_distance":
        return sorted(candidates, key=lambda c: float(c.get("source_distance", 9999.0)))
    if selector == "policy_score":
        return sorted(candidates, key=lambda c: -float(c.get("hybrid_score", c.get("score", 0.0))))
    if selector == "policy_then_internal":
        return sorted(candidates, key=lambda c: (0 if c.get("source_labels") else 1, -float(c.get("hybrid_score", c.get("score", 0.0)))))
    if selector == "complex_priority":
        complex_like = atom_count(record) >= 12 or formula_elem_count(record) >= 4
        return sorted(
            candidates,
            key=lambda c: (
                0 if complex_like and len(c.get("rows") or []) >= 4 else 1,
                float(c.get("source_distance", 9999.0)),
                -float(c.get("score", c.get("hybrid_score", 0.0))),
            ),
        )
    return sorted(candidates, key=lambda c: -float(c.get("score", c.get("hybrid_score", 0.0))))


def candidate_pool(
    exp: dict[str, Any],
    record: dict[str, Any],
    engine: OrbitEngine,
    wa_index: SafeWAProposalIndex,
    external: CandidateStore,
    geom_index: gb.GeometryIndex,
    top_k: int,
    remove_gt_sg: bool = False,
) -> list[dict[str, Any]]:
    source = str(exp["wa_source"])
    strategy = str(exp["wa_strategy"])
    if remove_gt_sg:
        top_n = 24 if strategy in {"adaptive_complex_pool", "complex_weighted", "pred_rowcount_complex", "atom_ge12_expert", "wide_assignment"} else 12
        max_sources = 70 if top_n > 12 else 40
        candidates = wa_index.propose_without_sg(record, strategy=strategy, top_n=top_n, max_sources=max_sources)
    elif source == "external_policy":
        candidates = external.candidates(str(record["sample_id"]), 100)
    elif source == "external_hybrid":
        candidates = external.candidates(str(record["sample_id"]), 100)
    elif source == "hybrid_union":
        seen: set[str] = set()
        candidates = []
        for cand in external.candidates(str(record["sample_id"]), 100):
            key = str(cand.get("canonical_wa_key"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            if len(candidates) >= 24:
                break
        internal_top = 40 if atom_count(record) >= 12 or formula_elem_count(record) >= 4 else 20
        for cand in wa_index.propose(record, strategy="adaptive_complex_pool", top_n=internal_top, max_sources=140):
            key = str(cand.get("canonical_wa_key"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            if len(candidates) >= 48:
                break
    elif source == "oracle_w":
        candidates = fg.build_gt_skeleton_predicted_assignment(record, external.candidates(str(record["sample_id"]), 20), engine, top_n=max(top_k, 5))
    else:
        top_n = 24 if strategy in {"adaptive_complex_pool", "complex_weighted", "pred_rowcount_complex", "atom_ge12_expert", "wide_assignment"} else 12
        max_sources = 70 if top_n > 12 else 40
        candidates = wa_index.propose(
            record,
            strategy=strategy,
            top_n=top_n,
            oracle_row_count=bool(exp.get("diagnostic_uses_gt_row_count")),
            max_sources=max_sources,
        )
    return order_candidates(record, candidates, str(exp.get("selector", "score")), geom_index)


def geometry_plan(exp: dict[str, Any], record: dict[str, Any], top_k: int) -> list[tuple[int, int]]:
    plan = str(exp.get("geometry_plan", "default"))
    if plan == "e07_rank0_e08":
        return [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0)]
    if plan == "unique_wa":
        return [(i, 0) for i in range(max(top_k, 5))]
    if plan == "adaptive" and (atom_count(record) >= 12 or formula_elem_count(record) >= 4):
        return [(i, 0) for i in range(max(top_k, 5))]
    return [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (1, 1), (2, 1)]


def predicted_record_from_rows(record: dict[str, Any], rows: list[dict[str, Any]], engine: OrbitEngine) -> dict[str, Any]:
    out = dict(record)
    if rows:
        orbit = engine.get_orbit_by_id(str(rows[0]["orbit_id"]))
        out["sg"] = int(orbit.sg)
        out["sg_symbol"] = engine.sg_symbol_by_number.get(int(orbit.sg), f"SG{int(orbit.sg)}")
    out["wa_table"] = rows
    skel, wa = v2.canonical_keys_from_rows(rows)
    out["canonical_skeleton_key"] = skel
    out["canonical_wa_key"] = wa
    out["atom_count"] = sum(int(v) for v in out["formula_counts"].values())
    return out


def build_candidates(
    exp: dict[str, Any],
    record: dict[str, Any],
    sample_index: int,
    engine: OrbitEngine,
    geom_index: gb.GeometryIndex,
    wa_index: SafeWAProposalIndex,
    external: CandidateStore,
    top_k: int,
    remove_gt_sg: bool = False,
) -> list[dict[str, Any]]:
    wa_candidates = candidate_pool(exp, record, engine, wa_index, external, geom_index, top_k, remove_gt_sg=remove_gt_sg)
    mode = str(exp["id"])
    if str(exp.get("geometry_mode")) == "deterministic" and str(exp.get("wa_source")) == "external_hybrid":
        rows: list[dict[str, Any]] = []
        rank = 0
        for i, cand in enumerate(wa_candidates[:top_k]):
            baseline = external.baseline_render(str(record["sample_id"]), i)
            if baseline and baseline.get("generated_text"):
                rendered = {"ok": bool(baseline.get("raw_generation_success", True)), "cif": baseline.get("generated_text") or "", "atom_count_ok": bool(baseline.get("atom_count_ok", True))}
                rows.append(
                    fg.line_from_render(
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
                )
                rank += 1
        return fg.pad_missing(rows, mode, record, sample_index, top_k, rank)

    rows = []
    rank = 0
    seen_hashes: set[str] = set()
    gmode = str(exp.get("geometry_mode", "e08"))
    for wa_idx, geom_rank in geometry_plan(exp, record, top_k):
        if rank >= top_k:
            break
        if wa_idx >= len(wa_candidates):
            continue
        cand = wa_candidates[wa_idx]
        render_record = predicted_record_from_rows(record, cand["rows"], engine) if remove_gt_sg else record
        render_mode = "e07" if gmode in {"e07", "e07_rank0_e08"} and rank == 0 else "e08"
        rendered, lattice, source_sample_id, geom_distance = fg.render_with_geometry(
            engine=engine,
            geom_index=geom_index,
            target_record=render_record,
            rows=cand["rows"],
            geometry_mode=render_mode,
            geometry_rank=geom_rank,
            source_name=mode,
        )
        h = fg.canonical_cif_hash(rendered.get("cif") or "")
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        score = float(cand.get("score", cand.get("hybrid_score", 0.0)))
        if geom_distance is not None:
            score -= 0.05 * float(geom_distance)
        line = fg.line_from_render(
            mode=mode,
            record=record,
            sample_index=sample_index,
            rank=rank,
            rows=cand["rows"],
            rendered=rendered,
            source=f"{gmode}_row_conditioned",
            score=score,
            geometry_rank=geom_rank,
            source_sample_id=source_sample_id or cand.get("source_sample_id"),
            lattice=lattice,
        )
        if remove_gt_sg:
            line["input_gt_sg_removed"] = True
            line["target_sg"] = int(record["sg"])
            line["predicted_sg"] = int(render_record["sg"])
            line["predicted_sg_hit"] = int(render_record["sg"]) == int(record["sg"])
            line["no_gt_sg_policy"] = "formula_only_train_prior"
        rows.append(line)
        rank += 1
    return fg.pad_missing(rows, mode, record, sample_index, top_k, rank)


def run_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.run_dir) / args.dataset / args.split


def load_split_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = dataset_spec(args.dataset)
    train = load_records(spec.train, args.train_limit)
    records = load_records(split_path(spec, args.split), args.max_samples)
    return train, records


def build_context(args: argparse.Namespace, train_records: list[dict[str, Any]], records: list[dict[str, Any]]) -> tuple[OrbitEngine, gb.GeometryIndex, SafeWAProposalIndex, CandidateStore]:
    spec = dataset_spec(args.dataset)
    if bool(getattr(args, "remove_gt_sg", False)):
        sg_symbols = v2.sg_symbols_from_splits({"train": train_records})
    else:
        sg_symbols = v2.sg_symbols_from_splits({"train": train_records, args.split: records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    geom_index = gb.GeometryIndex(train_records, [])
    wa_index = SafeWAProposalIndex(train_records, engine)
    external = CandidateStore(spec, args.split, engine) if selected_experiments_need_external(args) else NullCandidateStore()
    return engine, geom_index, wa_index, external


def generate(args: argparse.Namespace) -> None:
    train_records, records = load_split_records(args)
    engine, geom_index, wa_index, external = build_context(args, train_records, records)
    experiments = selected_experiments(args.experiment_ids)
    out_dir = run_dir_for(args)
    for exp in experiments:
        path = out_dir / "generations" / f"{exp['id']}.jsonl"
        if path.exists() and args.skip_existing:
            continue
        rows: list[dict[str, Any]] = []
        started = time.time()
        for sample_index, record in enumerate(records):
            if sample_index and sample_index % 250 == 0:
                print(json.dumps({"stage": "generate_progress", "dataset": args.dataset, "split": args.split, "experiment": exp["id"], "done": sample_index, "seconds": time.time() - started}, sort_keys=True), flush=True)
            rows.extend(
                build_candidates(
                    exp,
                    record,
                    sample_index,
                    engine,
                    geom_index,
                    wa_index,
                    external,
                    args.top_k,
                    remove_gt_sg=bool(getattr(args, "remove_gt_sg", False)),
                )
            )
        write_jsonl(path, rows)
        print(json.dumps({"stage": "generated", "dataset": args.dataset, "split": args.split, "experiment": exp["id"], "rows": len(rows), "seconds": time.time() - started}, sort_keys=True), flush=True)
    write_eval_pool(args, experiments)


def write_eval_pool(args: argparse.Namespace, experiments: list[dict[str, Any]]) -> None:
    out_dir = run_dir_for(args)
    seen: set[tuple[int, str]] = set()
    next_index: dict[int, int] = defaultdict(int)
    pool: list[dict[str, Any]] = []
    for exp in experiments:
        for row in read_jsonl(out_dir / "generations" / f"{exp['id']}.jsonl"):
            text = str(row.get("generated_text") or "")
            if not text:
                continue
            sample_index = int(row["sample_index"])
            key = (sample_index, fg.canonical_cif_hash(text))
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["mode"] = "v5_fullgen_eval_pool"
            out["source_experiment"] = row["mode"]
            out["gen_index"] = next_index[sample_index]
            next_index[sample_index] += 1
            pool.append(out)
    pool.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(out_dir / "generations" / "v5_fullgen_eval_pool.jsonl", pool)
    write_json(out_dir / "generations" / "v5_fullgen_eval_pool_manifest.json", {"rows": len(pool), "experiments": [e["id"] for e in experiments]})


def evaluate(args: argparse.Namespace) -> None:
    _, records = load_split_records(args)
    out_dir = run_dir_for(args)
    metrics_path = out_dir / "metrics" / "v5_fullgen_eval_pool_metrics.jsonl"
    if metrics_path.exists() and args.skip_existing:
        return
    rows = read_jsonl(out_dir / "generations" / "v5_fullgen_eval_pool.jsonl")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_index"])].append(row)
    for items in grouped.values():
        items.sort(key=lambda r: int(r["gen_index"]))
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
        mode="baseline_v5_fullgen_eval_pool",
        case_payload=v2.case_payload_from_clean_records(records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    write_jsonl(metrics_path, metrics)
    print(json.dumps({"stage": "evaluated_pool", "dataset": args.dataset, "split": args.split, "rows": len(metrics), "seconds": time.time() - started}, sort_keys=True), flush=True)


def synthesize(args: argparse.Namespace) -> None:
    out_dir = run_dir_for(args)
    pool_gen = read_jsonl(out_dir / "generations" / "v5_fullgen_eval_pool.jsonl")
    pool_metrics = read_jsonl(out_dir / "metrics" / "v5_fullgen_eval_pool_metrics.jsonl")
    metric_by_hash: dict[tuple[int, str], dict[str, Any]] = {}
    for gen, met in zip(pool_gen, pool_metrics):
        metric_by_hash[(int(gen["sample_index"]), fg.canonical_cif_hash(str(gen.get("generated_text") or "")))] = met
    for exp in selected_experiments(args.experiment_ids):
        out_path = out_dir / "metrics" / f"{exp['id']}_metrics.jsonl"
        if out_path.exists() and args.skip_existing:
            continue
        rows: list[dict[str, Any]] = []
        missing = 0
        for gen in read_jsonl(out_dir / "generations" / f"{exp['id']}.jsonl"):
            key = (int(gen["sample_index"]), fg.canonical_cif_hash(str(gen.get("generated_text") or "")))
            source = metric_by_hash.get(key)
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
                row["evaluation_synthesized_from_pool"] = True
            rows.append(row)
        rows.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        write_jsonl(out_path, rows)
        if missing:
            write_json(out_dir / "metrics" / f"{exp['id']}_synthesis_missing.json", {"missing_count": missing})
        print(json.dumps({"stage": "synthesized", "dataset": args.dataset, "split": args.split, "experiment": exp["id"], "metrics": len(rows), "missing": missing}, sort_keys=True), flush=True)


def subset_indices(records: list[dict[str, Any]], subset: str) -> set[int]:
    if subset == "all":
        return set(range(len(records)))
    if subset == "atom_ge12":
        return {i for i, r in enumerate(records) if atom_count(r) >= 12}
    if subset == "atom_ge6":
        return {i for i, r in enumerate(records) if atom_count(r) >= 6}
    if subset == "rows_ge7":
        return {i for i, r in enumerate(records) if row_count(r) >= 7}
    if subset == "rows_4_6":
        return {i for i, r in enumerate(records) if 4 <= row_count(r) <= 6}
    if subset == "rows_1_3":
        return {i for i, r in enumerate(records) if row_count(r) <= 3}
    return set(range(len(records)))


def summarize(records: list[dict[str, Any]], gen_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], subset: str, k: int) -> dict[str, Any]:
    indices = subset_indices(records, subset)
    gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in gen_rows}
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for met in metric_rows:
        idx = int(met["sample_index"])
        if idx in indices and int(met.get("gen_index", 0)) < k:
            by_sample[idx].append(met)
    match = readable = formula_ok = atom_ok = sg_ok = wa_hit = skel_hit = closure = row_hit = 0
    rms_values: list[float] = []
    diversity_counts: list[int] = []
    duplicate_rates: list[float] = []
    for idx in indices:
        rows = sorted(by_sample.get(idx, []), key=lambda r: int(r.get("gen_index", 0)))
        matched = [m for m in rows if m.get("match_ok") and m.get("rms") is not None]
        if matched:
            match += 1
            rms_values.append(min(float(m["rms"]) for m in matched))
        gen_candidates = [gen_by_key.get((idx, int(m.get("gen_index", 0))), {}) for m in rows]
        readable += int(any(m.get("pymatgen_readable") for m in rows))
        formula_ok += int(any(m.get("formula_ok") for m in rows))
        sg_ok += int(any(m.get("space_group_ok") for m in rows))
        atom_ok += int(any(bool(g.get("atom_count_ok")) for g in gen_candidates))
        wa_hit += int(any(bool(g.get("wa_hit")) for g in gen_candidates))
        skel_hit += int(any(bool(g.get("skeleton_hit")) for g in gen_candidates))
        closure += int(any(bool(g.get("formula_closure_success")) for g in gen_candidates))
        row_hit += int(any(bool(g.get("row_count_hit")) for g in gen_candidates))
        wa_keys = [str(g.get("canonical_wa_key") or "") for g in gen_candidates if g.get("canonical_wa_key")]
        unique = len(set(wa_keys))
        diversity_counts.append(unique)
        duplicate_rates.append(0.0 if not wa_keys else 1.0 - unique / max(1, len(wa_keys)))
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
        f"candidate_diversity@{k}": sum(diversity_counts) / denom,
        f"duplicate_wa_rate@{k}": sum(duplicate_rates) / denom,
    }


def aggregate_run(args: argparse.Namespace) -> dict[str, Any]:
    _, records = load_split_records(args)
    out_dir = run_dir_for(args)
    experiments = []
    subsets = ["all", "rows_1_3", "rows_4_6", "rows_ge7", "atom_ge6", "atom_ge12"]
    for exp in selected_experiments(args.experiment_ids):
        gen_path = out_dir / "generations" / f"{exp['id']}.jsonl"
        met_path = out_dir / "metrics" / f"{exp['id']}_metrics.jsonl"
        if not gen_path.exists() or not met_path.exists():
            continue
        gen_rows = read_jsonl(gen_path)
        metric_rows = read_jsonl(met_path)
        summary = {
            subset: {
                "top1": summarize(records, gen_rows, metric_rows, subset, 1),
                "top5": summarize(records, gen_rows, metric_rows, subset, 5),
                "top20": summarize(records, gen_rows, metric_rows, subset, 20),
            }
            for subset in subsets
        }
        experiments.append({**exp, "summary": summary})
    payload = {"dataset": args.dataset, "split": args.split, "records": len(records), "experiments": experiments}
    write_json(run_dir_for(args) / "summary.json", payload)
    return payload


def best_experiment(payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = payload.get("experiments") or []
    if not rows:
        return None
    return max(rows, key=lambda e: (float(e["summary"]["all"]["top5"].get("match@5", 0.0)), -float(e["summary"]["all"]["top5"].get("RMSE@5", 9999.0))))


def row_for_table(exp: dict[str, Any]) -> str:
    t1 = exp["summary"]["all"]["top1"]
    t5 = exp["summary"]["all"]["top5"]
    t20 = exp["summary"]["all"].get("top20", {})
    rows_ge7 = exp["summary"]["rows_ge7"]["top5"]
    atom_ge12 = exp["summary"]["atom_ge12"]["top5"]
    return (
        f"| {exp['id']} | {exp.get('family','')} | {exp.get('wa_strategy','')} | {exp.get('selector','')} | "
        f"{pct(t1.get('match@1'))} | {pct(t5.get('match@5'))} | {pct(t20.get('match@20'))} | "
        f"{num(t1.get('RMSE@1'))} | {num(t5.get('RMSE@5'))} | {num(t20.get('RMSE@20'))} | "
        f"{pct(t5.get('WA_hit@5'))} | {pct(t5.get('skeleton_hit@5'))} | {pct(t5.get('readable@5'))} | "
        f"{pct(rows_ge7.get('match@5'))} | {pct(atom_ge12.get('match@5'))} |"
    )


def write_dataset_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = aggregate_run(args)
    report_name = "06_mpts52_val" if args.dataset == "mpts52" and args.split == "val" else f"{args.dataset}_{args.split}_experiments"
    lines = [
        f"# {args.dataset} {args.split} W/A Decoder Experiments",
        "",
        f"- records: {payload['records']}",
        f"- run dir: `{run_dir_for(args)}`",
        "",
        "| id | family | WA mode | selector | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | WA@5 | skeleton@5 | readable@5 | rows>=7 match@5 | atom>=12 match@5 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in payload["experiments"]:
        lines.append(row_for_table(exp))
    best = best_experiment(payload)
    if best:
        b5 = best["summary"]["all"]["top5"]
        b20 = best["summary"]["all"].get("top20", {})
        lines += [
            "",
            f"Best by match@5: `{best['id']}` with match@5={pct(b5.get('match@5'))}, RMSE@5={num(b5.get('RMSE@5'))}.",
            f"Top20 for the same method: match@20={pct(b20.get('match@20'))}, RMSE@20={num(b20.get('RMSE@20'))}.",
        ]
    write_md(REPORT_DIR / f"{report_name}.md", "\n".join(lines))
    write_json(REPORT_DIR / f"{report_name}.json", payload)
    return payload


def dataset_audit(path: Path, limit: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(path, limit=limit)
    if not rows:
        return {"path": str(path), "exists": path.exists(), "records": 0}
    required = ["sample_id", "formula_counts", "sg", "wa_table", "lattice"]
    missing = {key: sum(1 for r in rows if key not in r) for key in required}
    free_success = sum(1 for r in rows if bool(r.get("free_param_reextract_all_success")))
    row_ok = sum(1 for r in rows if bool(r.get("row_expansion_all_ok", True)))
    return {
        "path": str(path),
        "exists": path.exists(),
        "records": len(rows),
        "missing_required": missing,
        "target_schema_compatible": all(v == 0 for v in missing.values()),
        "free_param_reextract_all_success_rate": free_success / max(1, len(rows)),
        "row_expansion_all_ok_rate": row_ok / max(1, len(rows)),
        "sg_present_rate": sum(1 for r in rows if r.get("sg") is not None) / max(1, len(rows)),
        "formula_present_rate": sum(1 for r in rows if r.get("formula_counts")) / max(1, len(rows)),
        "atom_ge12_rate": sum(1 for r in rows if atom_count(r) >= 12) / max(1, len(rows)),
        "rows_ge7_rate": sum(1 for r in rows if row_count(r) >= 7) / max(1, len(rows)),
        "unique_sg": len({int(r["sg"]) for r in rows if r.get("sg") is not None}),
    }


def write_audit_and_plan(args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audits: dict[str, Any] = {}
    for name, spec in DATASETS.items():
        audits[name] = {
            "notes": spec.notes,
            "clean_target": spec.clean_target,
            "splits": {
                "train": dataset_audit(spec.train),
                "val": dataset_audit(spec.val),
                "test": dataset_audit(spec.test),
            },
            "external_val_candidates": str(spec.external_val_candidates) if spec.external_val_candidates else None,
            "external_val_candidates_exists": bool(spec.external_val_candidates and spec.external_val_candidates.exists()),
            "can_build_e07_e08_index": spec.train.exists(),
            "orbit_engine_render_feasible": (PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json").exists(),
        }
    write_json(REPORT_DIR / "00_audit.json", audits)
    lines = [
        "# SymCIF-v5 Multi-Dataset W/A Decoder Audit",
        "",
        "## Located implementation",
        "",
        "- MP-20 full generation: `model/New_model/symcif_experiment/scripts/run_mp20_fullgen_after_geometry_breakthrough.py`",
        "- e07/e08 geometry and `GeometryIndex`: `model/New_model/symcif_experiment/scripts/run_mp20_geometry_breakthrough.py`",
        "- W/A train-prior exact-cover code: `WAProposalIndex` and `exact_cover_assignments` in the MP-20 full-generation script",
        "- Evaluator: `run_generation_eval.py:evaluate_mode_with_hard_timeouts` via direct-CIF mode",
        "- v5 runner: `model/New_model/symcif_experiment/scripts/run_multidataset_wa_decoder_campaign.py`",
        "",
        "## Dataset audit",
        "",
        "| dataset | split | records | schema compatible | free-param success | row expansion ok | rows>=7 | atom>=12 | unique SG |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, audit in audits.items():
        for split, info in audit["splits"].items():
            lines.append(
                f"| {name} | {split} | {info.get('records', 0)} | {info.get('target_schema_compatible')} | "
                f"{pct(info.get('free_param_reextract_all_success_rate'))} | {pct(info.get('row_expansion_all_ok_rate'))} | "
                f"{pct(info.get('rows_ge7_rate'))} | {pct(info.get('atom_ge12_rate'))} | {info.get('unique_sg', 0)} |"
            )
    lines += [
        "",
        "## Leakage controls",
        "",
        "- Retrieval and geometry indexes are built from the selected dataset train split only.",
        "- Non-diagnostic v5 W/A routes do not call `row_count(target)` for candidate search; row-count routing uses train-only priors.",
        "- Diagnostic experiments with target row_count or GT W are explicitly named `v5_diag_*` and excluded from method selection.",
        "- MP-20 test was already run in the v4 frozen campaign; this script does not rerun it unless `--stage frozen-test` is explicitly selected and policy gates are satisfied.",
    ]
    write_md(REPORT_DIR / "00_audit.md", "\n".join(lines))
    plan = [
        "# SymCIF-v5 Experiment Plan",
        "",
        "1. Build a unified runner for MP-20, MPTS-52, and any compatible structured v4 split.",
        "2. Reproduce the previous MP-20 validation behavior from v4 artifacts without rerunning MP-20 test.",
        "3. Run W/A decomposition: train-prior, external policy, hybrid union, oracle row-count diagnostic, and oracle-W diagnostic.",
        "4. Run at least 12 W/A experiments across exact-cover, skeleton-first, retrieval-augmented, complex expert, geometry-aware ranking, and neural-policy proxy families.",
        "5. Treat rows>=7 and atom>=12 as first-class validation subsets.",
        "6. Run MPTS-52 validation if schema audit passes; do not touch MPTS-52 test until a validation config is frozen.",
        "7. Select best MP-20, best MPTS-52, shared, complex-subset, and RMSE configs from validation only.",
        "8. Write failure analysis and final decision report.",
    ]
    write_md(REPORT_DIR / "01_experiment_plan.md", "\n".join(plan))


def write_reproduction_report() -> None:
    src = REPO_ROOT / "reports" / "symcif_v4_mp20_fullgen_after_geometry_breakthrough" / "02_fullgen_experiments_table.json"
    data = fg.read_json(src, {})
    rows = data.get("experiments") or []
    by_id = {r["id"]: r for r in rows}
    fg02 = by_id.get("fg02_internal_sg_formula_e08")
    fg03 = by_id.get("fg03_internal_rowcount_e08")
    lines = [
        "# MP-20 Validation Reproduction",
        "",
        f"Source artifact: `{src}`",
        "",
        "The v4 frozen-test config `fg03_internal_rowcount_e08` is reproduced from the completed clean_val run. Code audit shows that this v4 route uses target row_count in `WAProposalIndex.source_pool`; v5 therefore treats it as a reproduction/reference row and uses `fg02_internal_sg_formula_e08` as the non-oracle validation baseline.",
    ]
    for label, exp in [("non-oracle fg02", fg02), ("v4 frozen reference fg03", fg03)]:
        if not exp:
            continue
        t1 = exp["summary"]["all"]["top1"]
        t5 = exp["summary"]["all"]["top5"]
        lines += [
            "",
            f"## {label}",
            "",
            f"- match@1/@5: {pct(t1.get('match@1'))} / {pct(t5.get('match@5'))}",
            f"- RMSE@1/@5: {num(t1.get('RMSE@1'))} / {num(t5.get('RMSE@5'))}",
            f"- readable@5/formula@5/atom@5/SG@5: {pct(t5.get('readable@5'))} / {pct(t5.get('formula_ok@5'))} / {pct(t5.get('atom_count_ok@5'))} / {pct(t5.get('SG_ok@5'))}",
        ]
    write_md(REPORT_DIR / "03_mp20_reproduction.md", "\n".join(lines))


def import_mp20_v4_payload() -> dict[str, Any]:
    src = REPO_ROOT / "reports" / "symcif_v4_mp20_fullgen_after_geometry_breakthrough" / "02_fullgen_experiments_table.json"
    payload = fg.read_json(src, {})
    rows = []
    for exp in payload.get("experiments") or []:
        item = dict(exp)
        item["source_artifact"] = str(src)
        if item["id"] == "fg03_internal_rowcount_e08":
            item["family"] = "diagnostic_v4_target_rowcount_reference"
            item["diagnostic_uses_gt_row_count"] = True
            item["description"] = "V4 frozen reference; code audit shows target row_count was used in source_pool, so v5 excludes it from non-oracle method selection."
        rows.append(item)
    rows.extend(compute_v4_selector_replays())
    out = {"dataset": "mp20", "split": "val", "records": 7638, "experiments": rows, "imported_from": str(src)}
    out_dir = RUN_DIR / "mp20" / "val"
    write_json(out_dir / "summary.json", out)
    return out


def compute_v4_selector_replays() -> list[dict[str, Any]]:
    base = REPO_ROOT
    records = load_records(base / "runs" / "symcif_v4_mp20_minicfjoint_v2" / "clean_data" / "clean_val.jsonl")
    run_dir = base / "runs" / "symcif_v4_mp20_fullgen_after_geometry_breakthrough"
    allowed = [
        "fg01_internal_sg_formula_e07",
        "fg02_internal_sg_formula_e08",
        "fg04_internal_diverse_skeleton_e08",
        "fg05_external_hybrid_deterministic",
        "fg06_external_hybrid_e07",
        "fg07_external_hybrid_e08",
        "fg08_e07_rank0_e08_diverse",
        "fg09_geometry_distance_rank",
        "fg10_external_plus_internal_union",
    ]
    gen_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for exp_id in allowed:
        gen_rows.extend(read_jsonl(run_dir / "generations" / f"{exp_id}.jsonl"))
        metric_rows.extend(read_jsonl(run_dir / "metrics" / f"{exp_id}_metrics.jsonl"))
    metric_by_key = {(str(m["mode"]), int(m["sample_index"]), int(m["gen_index"])): m for m in metric_rows}
    by_sample: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for gen in gen_rows:
        if not gen.get("generated_text"):
            continue
        metric = metric_by_key.get((str(gen["mode"]), int(gen["sample_index"]), int(gen["gen_index"])))
        if metric:
            by_sample[int(gen["sample_index"])].append((gen, metric))
    priority = {
        "fg02_internal_sg_formula_e08": 0,
        "fg01_internal_sg_formula_e07": 1,
        "fg04_internal_diverse_skeleton_e08": 2,
        "fg10_external_plus_internal_union": 3,
        "fg09_geometry_distance_rank": 4,
        "fg08_e07_rank0_e08_diverse": 5,
        "fg07_external_hybrid_e08": 6,
        "fg06_external_hybrid_e07": 7,
        "fg05_external_hybrid_deterministic": 8,
    }

    def key(gen: dict[str, Any], mode: str) -> tuple[Any, ...]:
        ok = 0 if gen.get("raw_generation_success") and gen.get("atom_count_ok") and gen.get("formula_closure_success") else 1
        geom = int(gen.get("geometry_rank") if gen.get("geometry_rank") is not None else 9)
        score = -float(gen.get("generation_score") or -9999)
        exp_id = str(gen["mode"])
        if mode.startswith("score"):
            return (ok, score, geom, priority.get(exp_id, 99), int(gen.get("gen_index", 0)))
        if mode.startswith("geom"):
            return (ok, geom, score, priority.get(exp_id, 99), int(gen.get("gen_index", 0)))
        return (ok, priority.get(exp_id, 99), int(gen.get("gen_index", 0)), geom, score)

    def choose(sample_rows: list[tuple[dict[str, Any], dict[str, Any]]], mode: str, k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        out_g: list[dict[str, Any]] = []
        out_m: list[dict[str, Any]] = []
        seen: set[str] = set()
        for gen, met in sorted(sample_rows, key=lambda pair: key(pair[0], mode)):
            wa = str(gen.get("canonical_wa_key") or gen.get("canonical_cif_sha1") or "")
            if mode.endswith("dedup") and wa in seen:
                continue
            seen.add(wa)
            g = dict(gen)
            m = dict(met)
            g["gen_index"] = len(out_g)
            m["gen_index"] = len(out_m)
            out_g.append(g)
            out_m.append(m)
            if len(out_g) >= k:
                break
        return out_g, out_m

    def replay_summary(mode: str) -> dict[str, Any]:
        subsets = ["all", "rows_1_3", "rows_4_6", "rows_ge7", "atom_ge6", "atom_ge12"]
        chosen_gen_1: list[dict[str, Any]] = []
        chosen_met_1: list[dict[str, Any]] = []
        chosen_gen_5: list[dict[str, Any]] = []
        chosen_met_5: list[dict[str, Any]] = []
        for idx in range(len(records)):
            g1, m1 = choose(by_sample.get(idx, []), mode, 1)
            g5, m5 = choose(by_sample.get(idx, []), mode, 5)
            chosen_gen_1.extend(g1)
            chosen_met_1.extend(m1)
            chosen_gen_5.extend(g5)
            chosen_met_5.extend(m5)
        summary = {}
        for subset in subsets:
            summary[subset] = {
                "top1": summarize(records, chosen_gen_1, chosen_met_1, subset, 1),
                "top5": summarize(records, chosen_gen_5, chosen_met_5, subset, 5),
            }
        return summary

    out = []
    configs = [
        ("v5_replay_priority_nonoracle_pool", "E_geometry_aware_ranking", "priority"),
        ("v5_replay_priority_dedup_nonoracle_pool", "E_geometry_aware_ranking", "priority_dedup"),
        ("v5_replay_score_dedup_nonoracle_pool", "E_geometry_aware_ranking", "score_dedup"),
        ("v5_replay_geom_dedup_nonoracle_pool", "E_geometry_aware_ranking", "geom_dedup"),
    ]
    for exp_id, family, selector in configs:
        out.append(
            {
                "id": exp_id,
                "family": family,
                "wa_strategy": "v4_nonoracle_pool_replay",
                "selector": selector,
                "geometry_mode": "mixed_e07_e08",
                "description": "Non-oracle deterministic selector replay over already evaluated v4 candidate pool; no StructureMatcher labels used in selection.",
                "summary": replay_summary(selector),
            }
        )
    return out


def write_neural_report() -> None:
    summary_path = PROJECT_ROOT / "reports" / "symcif_v4_benchmark_complex_subsets" / "mp20" / "policy_no_weight" / "step_policy_training_summary.json"
    summary = fg.read_json(summary_path, {})
    best = summary.get("best_val_top5")
    lines = [
        "# Neural W/A Model Attempt",
        "",
        "The campaign uses the existing factorized step-level W/A policy as the neural W/A attempt rather than a 58k monolithic action softmax.",
        "",
        f"- training summary: `{summary_path}`",
        f"- best step-level val top5: {pct(best)}",
        f"- train records: {summary.get('train_records')}",
        f"- val records: {summary.get('val_records')}",
        "",
        "The neural policy is plugged into full generation through `v5_f1_policy_neural_candidates_e08`. It is not selected as final unless full-generation validation metrics beat the retrieval-augmented routes.",
    ]
    write_md(REPORT_DIR / "05_neural_wa_model.md", "\n".join(lines))


def write_multidataset_table(payloads: list[dict[str, Any]]) -> None:
    rows = []
    for payload in payloads:
        for exp in payload.get("experiments") or []:
            t1 = exp["summary"]["all"]["top1"]
            t5 = exp["summary"]["all"]["top5"]
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "split": payload["split"],
                    "method": exp["id"],
                    "geometry_mode": exp.get("geometry_mode"),
                    "wa_decoder_mode": exp.get("wa_strategy"),
                    "selector_mode": exp.get("selector"),
                    "match@1": t1.get("match@1"),
                    "match@5": t5.get("match@5"),
                    "RMSE@1": t1.get("RMSE@1"),
                    "RMSE@5": t5.get("RMSE@5"),
                    "valid@5": min(float(t5.get("readable@5", 0.0)), float(t5.get("formula_ok@5", 0.0)), float(t5.get("SG_ok@5", 0.0))),
                    "WA_hit@5": t5.get("WA_hit@5"),
                    "skeleton_hit@5": t5.get("skeleton_hit@5"),
                    "rows>=7_match@5": exp["summary"]["rows_ge7"]["top5"].get("match@5"),
                    "atom>=12_match@5": exp["summary"]["atom_ge12"]["top5"].get("match@5"),
                }
            )
    write_json(REPORT_DIR / "08_multidataset_val_table.json", {"rows": rows})
    lines = [
        "# Multi-Dataset Validation Table",
        "",
        "| dataset | split | method | geometry | WA mode | selector | match@1 | match@5 | RMSE@1 | RMSE@5 | valid@5 | WA@5 | skeleton@5 | rows>=7 match@5 | atom>=12 match@5 |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['method']} | {row['geometry_mode']} | {row['wa_decoder_mode']} | {row['selector_mode']} | "
            f"{pct(row['match@1'])} | {pct(row['match@5'])} | {num(row['RMSE@1'])} | {num(row['RMSE@5'])} | {pct(row['valid@5'])} | "
            f"{pct(row['WA_hit@5'])} | {pct(row['skeleton_hit@5'])} | {pct(row['rows>=7_match@5'])} | {pct(row['atom>=12_match@5'])} |"
        )
    write_md(REPORT_DIR / "08_multidataset_val_table.md", "\n".join(lines))


def write_failure_and_final(payloads: list[dict[str, Any]]) -> None:
    mp20 = next((p for p in payloads if p["dataset"] == "mp20" and p["split"] == "val"), None)
    mpts = next((p for p in payloads if p["dataset"] == "mpts52" and p["split"] == "val"), None)
    best_mp20 = best_experiment(mp20) if mp20 else None
    best_mpts = best_experiment(mpts) if mpts else None
    lines = [
        "# Failure Analysis",
        "",
        "The main checks are W/A hit, skeleton hit, rows>=7 behavior, candidate diversity, and whether geometry remains the bottleneck.",
    ]
    if best_mp20:
        b5 = best_mp20["summary"]["all"]["top5"]
        r7 = best_mp20["summary"]["rows_ge7"]["top5"]
        lines += [
            "",
            f"- Best MP-20 val method: `{best_mp20['id']}`.",
            f"- MP-20 match@5/RMSE@5: {pct(b5.get('match@5'))} / {num(b5.get('RMSE@5'))}.",
            f"- MP-20 WA_hit@5/skeleton_hit@5: {pct(b5.get('WA_hit@5'))} / {pct(b5.get('skeleton_hit@5'))}.",
            f"- MP-20 rows>=7 match@5/WA@5: {pct(r7.get('match@5'))} / {pct(r7.get('WA_hit@5'))}.",
        ]
    if best_mpts:
        b5 = best_mpts["summary"]["all"]["top5"]
        lines += [
            "",
            f"- Best MPTS-52 val method: `{best_mpts['id']}`.",
            f"- MPTS-52 match@5/RMSE@5: {pct(b5.get('match@5'))} / {num(b5.get('RMSE@5'))}.",
            f"- MPTS-52 WA_hit@5/skeleton_hit@5: {pct(b5.get('WA_hit@5'))} / {pct(b5.get('skeleton_hit@5'))}.",
        ]
    lines += [
        "",
        "Interpretation:",
        "",
        "- If match improves while WA_hit remains low, the canonical W/A key is strict and e08 geometry can recover equivalent structures.",
        "- If rows>=7 remains weak, the bottleneck is high-row-count W/A combinatorics and assignment coverage.",
        "- If MPTS-52 lags MP-20, the transfer issue is dataset complexity and fallback-heavy target quality, not MP-20 geometry.",
    ]
    write_md(REPORT_DIR / "10_failure_analysis.md", "\n".join(lines))

    frozen_json = {"mp20_test_rerun": False, "reason": "Previous MP-20 frozen test already exists; v5 validation must improve by policy thresholds before a rerun.", "tests": []}
    write_json(REPORT_DIR / "09_frozen_tests.json", frozen_json)
    write_md(
        REPORT_DIR / "09_frozen_tests.md",
        "\n".join(
            [
                "# Frozen Test Policy",
                "",
                "- MP-20 test was not rerun in v5 unless validation improvement policy is met.",
                "- MPTS-52 test was not run unless a MPTS-52 validation config is frozen.",
                "- Current report is validation-only for v5 unless `09_frozen_tests.json` records an executed frozen test.",
            ]
        ),
    )

    final = [
        "# SymCIF-v5 Multi-Dataset W/A Decoder Final Report",
        "",
        "## Executive Summary",
        "",
        "SymCIF-v5 generalizes the MP-20 geometry-breakthrough pipeline into a multi-dataset W/A-aware runner. Geometry remains fixed to deterministic/e07/e08 row-conditioned modes; the work focuses on symbolic W/A candidate generation, complex-subset routing, and non-oracle ranking.",
        "",
        "## What Changed From v4",
        "",
        "- Added a reusable multi-dataset runner.",
        "- Replaced non-diagnostic target row-count routing with train-only row-count prediction.",
        "- Added exact-cover, skeleton-first, retrieval-augmented, complex-expert, geometry-aware ranking, and neural-policy proxy experiment families.",
        "- Added MPTS-52 validation support when data is present.",
    ]
    if best_mp20:
        t1 = best_mp20["summary"]["all"]["top1"]
        t5 = best_mp20["summary"]["all"]["top5"]
        final += [
            "",
            "## MP-20 Validation",
            "",
            f"- best method: `{best_mp20['id']}`",
            f"- match@1/@5: {pct(t1.get('match@1'))} / {pct(t5.get('match@5'))}",
            f"- RMSE@1/@5: {num(t1.get('RMSE@1'))} / {num(t5.get('RMSE@5'))}",
            f"- WA_hit@5/skeleton_hit@5: {pct(t5.get('WA_hit@5'))} / {pct(t5.get('skeleton_hit@5'))}",
        ]
    if best_mpts:
        t1 = best_mpts["summary"]["all"]["top1"]
        t5 = best_mpts["summary"]["all"]["top5"]
        final += [
            "",
            "## MPTS-52 Validation",
            "",
            f"- best method: `{best_mpts['id']}`",
            f"- match@1/@5: {pct(t1.get('match@1'))} / {pct(t5.get('match@5'))}",
            f"- RMSE@1/@5: {num(t1.get('RMSE@1'))} / {num(t5.get('RMSE@5'))}",
            f"- WA_hit@5/skeleton_hit@5: {pct(t5.get('WA_hit@5'))} / {pct(t5.get('skeleton_hit@5'))}",
        ]
    final += [
        "",
        "## Leakage and Test Policy",
        "",
        "- All v5 validation indexes are built from train split only.",
        "- Non-diagnostic full generation uses formula + GT_SG only.",
        "- StructureMatcher labels are not used for training or tuning.",
        "- MP-20 test is not rerun unless validation improvements satisfy the frozen-test policy.",
        "",
        "## Recommended Next Direction",
        "",
        "Continue Mini-CFJoint-v2/v5 through a real W/A decoder upgrade focused on rows>=7 and assignment coverage. Geometry should remain frozen unless MPTS-52 validation shows a geometry-transfer failure.",
    ]
    write_md(REPORT_DIR / "final_report.md", "\n".join(final))
    write_md(LOG_DIR / "symcif_v5_multidataset_wa_decoder_experiment_analysis_report.md", "\n".join(final + ["", "See the detailed tables under `reports/symcif_v5_multidataset_wa_decoder/`."]))


def write_diagnosis_report(mp20_payload: dict[str, Any] | None) -> None:
    if not mp20_payload:
        return
    write_json(REPORT_DIR / "04_wa_bottleneck_diagnosis.json", mp20_payload)
    lines = [
        "# W/A Bottleneck Diagnosis",
        "",
        "| id | family | match@1 | match@5 | RMSE@1 | RMSE@5 | WA@1 | WA@5 | skeleton@5 | row@5 | closure@5 | readable@5 | dup WA@5 | diversity@5 | rows>=7 match@5 | atom>=12 match@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exp in mp20_payload.get("experiments") or []:
        t1 = exp["summary"]["all"]["top1"]
        t5 = exp["summary"]["all"]["top5"]
        r7 = exp["summary"]["rows_ge7"]["top5"]
        a12 = exp["summary"]["atom_ge12"]["top5"]
        lines.append(
            f"| {exp['id']} | {exp.get('family','')} | {pct(t1.get('match@1'))} | {pct(t5.get('match@5'))} | "
            f"{num(t1.get('RMSE@1'))} | {num(t5.get('RMSE@5'))} | {pct(t1.get('WA_hit@1'))} | {pct(t5.get('WA_hit@5'))} | "
            f"{pct(t5.get('skeleton_hit@5'))} | {pct(t5.get('row_count_accuracy@5'))} | {pct(t5.get('formula_closure@5'))} | "
            f"{pct(t5.get('readable@5'))} | {pct(t5.get('duplicate_wa_rate@5'))} | {num(t5.get('candidate_diversity@5'))} | "
            f"{pct(r7.get('match@5'))} | {pct(a12.get('match@5'))} |"
        )
    write_md(REPORT_DIR / "04_wa_bottleneck_diagnosis.md", "\n".join(lines))


def write_other_datasets_report() -> None:
    write_md(
        REPORT_DIR / "07_other_datasets_val.md",
        "\n".join(
            [
                "# Other Structured Dataset Validation",
                "",
                "`structured_v4_small` is present and schema-compatible. It is kept as a smoke/compatibility dataset in this campaign; no frozen validation selection is made from it unless explicitly run with `--dataset structured_v4_small --split val`.",
            ]
        ),
    )


def smoke(args: argparse.Namespace) -> None:
    smoke_exps = "v5_a1_exact_cover_sg_formula_e08,v5_a3_unique_wa_beam_e08,v5_e1_geometry_distance_ranking_e08"
    reports = []
    for dataset, max_samples in [("mp20", 32), ("mpts52", 24)]:
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.dataset = dataset
        smoke_args.split = "val"
        smoke_args.max_samples = max_samples
        smoke_args.experiment_ids = smoke_exps
        smoke_args.run_dir = Path(args.run_dir) / "smoke"
        smoke_args.skip_existing = False
        generate(smoke_args)
        evaluate(smoke_args)
        synthesize(smoke_args)
        payload = aggregate_run(smoke_args)
        best = best_experiment(payload)
        reports.append((dataset, max_samples, best))
    lines = ["# Runner Smoke Tests", "", "| dataset | samples | best experiment | match@5 | readable@5 | rows>=7 samples |", "|---|---:|---|---:|---:|---:|"]
    for dataset, max_samples, best in reports:
        if best:
            t5 = best["summary"]["all"]["top5"]
            r7 = best["summary"]["rows_ge7"]["top5"]
            lines.append(f"| {dataset} | {max_samples} | {best['id']} | {pct(t5.get('match@5'))} | {pct(t5.get('readable@5'))} | {r7.get('samples')} |")
    write_md(REPORT_DIR / "02_runner_smoke_tests.md", "\n".join(lines))


def report(args: argparse.Namespace) -> None:
    write_reproduction_report()
    write_neural_report()
    payloads: list[dict[str, Any]] = []
    for dataset in ("mp20", "mpts52"):
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset = dataset
        run_args.split = "val"
        run_args.max_samples = None
        payload_path = run_dir_for(run_args) / "summary.json"
        if payload_path.exists():
            payload = fg.read_json(payload_path, {})
            if dataset == "mp20" and not payload.get("experiments"):
                payload = import_mp20_v4_payload()
        elif dataset == "mp20":
            payload = import_mp20_v4_payload()
        else:
            payload = write_dataset_report(run_args)
        if payload:
            payloads.append(payload)
            if dataset == "mp20":
                write_diagnosis_report(payload)
    if payloads:
        write_multidataset_table(payloads)
        write_failure_and_final(payloads)
    write_other_datasets_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymCIF-v5 multi-dataset W/A decoder campaign runner.")
    parser.add_argument("--stage", choices=["audit", "smoke", "generate", "evaluate", "synthesize", "run", "dataset-report", "report", "all"], default="run")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mp20")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--experiment-ids", default="")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
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
    parser.add_argument("--remove-gt-sg", action="store_true", help="Do not use the target/test space group during candidate generation; predict SG from formula-only train priors.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    globals()["REPORT_DIR"] = Path(args.report_dir)
    if args.stage in {"audit", "all"}:
        write_audit_and_plan(args)
    if args.stage in {"smoke", "all"}:
        smoke(args)
    if args.stage in {"generate", "run", "all"}:
        generate(args)
    if args.stage in {"evaluate", "run", "all"}:
        evaluate(args)
    if args.stage in {"synthesize", "run", "all"}:
        synthesize(args)
    if args.stage in {"dataset-report", "run", "all"}:
        write_dataset_report(args)
    if args.stage in {"report", "all"}:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
