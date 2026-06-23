#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "model" / "New_model" / "symcif_experiment"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_multidataset_wa_decoder_campaign as camp  # noqa: E402


@dataclass(frozen=True)
class SourceRun:
    source_id: str
    priority: int
    generation_path: Path
    metric_path: Path


@dataclass
class Candidate:
    source: SourceRun
    gen: dict[str, Any]
    metric: dict[str, Any]
    cif_hash: str
    wa_key: str
    consensus: int = 1


SOURCES = [
    SourceRun(
        "v5_e1",
        0,
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_e1_k20/mpts52/test/generations/v5_e1_geometry_distance_ranking_e08.jsonl",
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_e1_k20/mpts52/test/metrics/v5_e1_geometry_distance_ranking_e08_metrics.jsonl",
    ),
    SourceRun(
        "opentry_e2_symbolic",
        1,
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20/mpts52/test/generations/opentry_e2_hybrid_symbolic_adaptive_e08.jsonl",
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e2_symbolic_adaptive_k20/mpts52/test/metrics/opentry_e2_hybrid_symbolic_adaptive_e08_metrics.jsonl",
    ),
    SourceRun(
        "opentry_e1_geometry_adaptive",
        2,
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52/test/generations/opentry_e1_hybrid_geometry_adaptive_e08.jsonl",
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e1_adaptive_k20/mpts52/test/metrics/opentry_e1_hybrid_geometry_adaptive_e08_metrics.jsonl",
    ),
    SourceRun(
        "opentry_e3_diverse",
        3,
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e3_diverse_adaptive_k20/mpts52/test/generations/opentry_e3_hybrid_diverse_adaptive_e08.jsonl",
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_opentry_e3_diverse_adaptive_k20/mpts52/test/metrics/opentry_e3_hybrid_diverse_adaptive_e08_metrics.jsonl",
    ),
    SourceRun(
        "v5_d4",
        4,
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_d4_k20/mpts52/test/generations/v5_d4_adaptive_internal_pool_complex_e08.jsonl",
        REPO_ROOT / "model/New_model/opentry/runs/stream_mpts52_test_d4_k20/mpts52/test/metrics/v5_d4_adaptive_internal_pool_complex_e08_metrics.jsonl",
    ),
]


EXPERIMENTS = [
    {
        "id": "opentry_e13_ensemble_e1_top4_consensus_unique_e08",
        "family": "opentry_cached_nonoracle_ensemble",
        "wa_source": "cached_existing_nonoracle_candidates",
        "wa_strategy": "e1_top4_then_consensus_unique",
        "selector": "nonoracle_consensus_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "cached",
        "description": "Keep top4 from v5_e1, then fill with unique W/A candidates ranked by source consensus, source priority, geometry rank, and generation score.",
        "base_keep": 4,
    },
    {
        "id": "opentry_e14_ensemble_e1_top3_consensus_unique_e08",
        "family": "opentry_cached_nonoracle_ensemble",
        "wa_source": "cached_existing_nonoracle_candidates",
        "wa_strategy": "e1_top3_then_consensus_unique",
        "selector": "nonoracle_consensus_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "cached",
        "description": "Keep top3 from v5_e1, then fill with unique W/A candidates ranked by source consensus, source priority, geometry rank, and generation score.",
        "base_keep": 3,
    },
    {
        "id": "opentry_e15_ensemble_consensus_unique_e08",
        "family": "opentry_cached_nonoracle_ensemble",
        "wa_source": "cached_existing_nonoracle_candidates",
        "wa_strategy": "pure_consensus_unique",
        "selector": "nonoracle_consensus_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "cached",
        "description": "Rank all cached non-oracle candidates by W/A consensus, source priority, geometry rank, and generation score with W/A diversity.",
        "base_keep": 0,
    },
    {
        "id": "opentry_e16_ensemble_e1_top5_consensus_fill_e08",
        "family": "opentry_cached_nonoracle_ensemble",
        "wa_source": "cached_existing_nonoracle_candidates",
        "wa_strategy": "e1_top5_then_consensus_fill",
        "selector": "nonoracle_consensus_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "cached",
        "description": "Keep top5 from v5_e1, then fill remaining K20 slots from consensus-ranked cached candidates.",
        "base_keep": 5,
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def candidate_sort_key(c: Candidate) -> tuple[Any, ...]:
    return (
        -int(c.consensus),
        int(c.source.priority),
        safe_int(c.gen.get("geometry_rank"), 999),
        -safe_float(c.gen.get("generation_score"), -999.0),
        safe_int(c.gen.get("gen_index"), 999),
        str(c.cif_hash),
    )


def load_candidates() -> dict[int, list[Candidate]]:
    by_sample: dict[int, list[Candidate]] = defaultdict(list)
    for source in SOURCES:
        gen_rows = read_jsonl(source.generation_path)
        metric_rows = read_jsonl(source.metric_path)
        if len(gen_rows) != len(metric_rows):
            raise RuntimeError(f"row mismatch for {source.source_id}: gen={len(gen_rows)} metrics={len(metric_rows)}")
        for gen, metric in zip(gen_rows, metric_rows):
            text = str(gen.get("generated_text") or "")
            if not text:
                continue
            if gen.get("generation_timeout"):
                continue
            sample_index = int(gen["sample_index"])
            cif_hash = camp.fg.canonical_cif_hash(text)
            wa_key = str(gen.get("canonical_wa_key") or cif_hash)
            by_sample[sample_index].append(Candidate(source, gen, metric, cif_hash, wa_key))
    for items in by_sample.values():
        by_wa_sources: dict[str, set[str]] = defaultdict(set)
        for c in items:
            by_wa_sources[c.wa_key].add(c.source.source_id)
        for c in items:
            c.consensus = len(by_wa_sources[c.wa_key])
    return by_sample


def select_candidates(items: list[Candidate], exp: dict[str, Any], top_k: int) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_hashes: set[str] = set()
    selected_wa: set[str] = set()
    base_keep = int(exp.get("base_keep", 0))

    e1_items = sorted((c for c in items if c.source.source_id == "v5_e1"), key=lambda c: safe_int(c.gen.get("gen_index"), 999))
    for c in e1_items:
        if len(selected) >= base_keep:
            break
        if c.cif_hash in selected_hashes:
            continue
        selected.append(c)
        selected_hashes.add(c.cif_hash)
        selected_wa.add(c.wa_key)

    ranked = sorted(items, key=candidate_sort_key)
    for prefer_unique in (True, False):
        for c in ranked:
            if len(selected) >= top_k:
                break
            if c.cif_hash in selected_hashes:
                continue
            if prefer_unique and c.wa_key in selected_wa:
                continue
            selected.append(c)
            selected_hashes.add(c.cif_hash)
            selected_wa.add(c.wa_key)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def missing_metric(exp_id: str, gen_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": exp_id,
        "sample_index": int(gen_row["sample_index"]),
        "sample_id": gen_row.get("sample_id"),
        "gen_index": int(gen_row["gen_index"]),
        "match_ok": False,
        "rms": None,
        "pymatgen_readable": False,
        "formula_ok": False,
        "space_group_ok": False,
        "early_match_skip_reason": "ensemble_padding",
    }


def build_experiment(exp: dict[str, Any], by_sample: dict[int, list[Candidate]], records: list[dict[str, Any]], out_dir: Path, top_k: int) -> None:
    gen_out: list[dict[str, Any]] = []
    metric_out: list[dict[str, Any]] = []
    exp_id = str(exp["id"])
    for sample_index, record in enumerate(records):
        selected = select_candidates(by_sample.get(sample_index, []), exp, top_k)
        rank = 0
        for c in selected:
            gen = dict(c.gen)
            gen["mode"] = exp_id
            gen["source_experiment"] = c.source.source_id
            gen["source_gen_index"] = c.gen.get("gen_index")
            gen["ensemble_consensus"] = c.consensus
            gen["ensemble_selector"] = exp.get("selector")
            gen["gen_index"] = rank
            metric = dict(c.metric)
            metric["mode"] = exp_id
            metric["gen_index"] = rank
            metric["evaluation_synthesized_from_cached_pool"] = True
            gen_out.append(gen)
            metric_out.append(metric)
            rank += 1
        pads = camp.fg.pad_missing([], exp_id, record, sample_index, top_k, rank)
        for pad in pads:
            gen_out.append(pad)
            metric_out.append(missing_metric(exp_id, pad))
    camp.write_jsonl(out_dir / "generations" / f"{exp_id}.jsonl", gen_out)
    camp.write_jsonl(out_dir / "metrics" / f"{exp_id}_metrics.jsonl", metric_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached non-oracle ensembles from existing opentry MPTS-52 candidate runs.")
    parser.add_argument("--dataset", default="mpts52")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=1)
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
    camp.REPORT_DIR = Path(args.report_dir)
    for exp in EXPERIMENTS:
        if not any(e.get("id") == exp["id"] for e in camp.EXPERIMENTS):
            camp.EXPERIMENTS.append(dict(exp))
    args.experiment_ids = ",".join(str(exp["id"]) for exp in EXPERIMENTS)
    _, records = camp.load_split_records(args)
    out_dir = camp.run_dir_for(args)
    by_sample = load_candidates()
    for exp in EXPERIMENTS:
        build_experiment(exp, by_sample, records, out_dir, int(args.top_k))
    camp.write_dataset_report(args)
    print(
        json.dumps(
            {
                "stage": "cached_ensemble_done",
                "dataset": args.dataset,
                "split": args.split,
                "experiments": [exp["id"] for exp in EXPERIMENTS],
                "samples": len(records),
                "top_k": int(args.top_k),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
