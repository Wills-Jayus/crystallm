#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "model" / "New_model" / "symcif_experiment"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_multidataset_wa_decoder_campaign as camp  # noqa: E402


CUSTOM_EXPERIMENTS: dict[str, dict] = {
    "opentry_e1_hybrid_geometry_adaptive_e08": {
        "id": "opentry_e1_hybrid_geometry_adaptive_e08",
        "family": "opentry_E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "geometry_support",
        "geometry_mode": "e08",
        "geometry_plan": "adaptive",
        "description": "Opentry non-oracle hybrid W/A candidates ranked by geometry support, with adaptive rank-0 W/A spread for complex samples.",
    },
    "opentry_e2_hybrid_symbolic_adaptive_e08": {
        "id": "opentry_e2_hybrid_symbolic_adaptive_e08",
        "family": "opentry_E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "symbolic_geometry",
        "geometry_mode": "e08",
        "geometry_plan": "adaptive",
        "description": "Opentry non-oracle hybrid W/A candidates ranked by symbolic plus geometry support, with adaptive rank-0 W/A spread for complex samples.",
    },
    "opentry_e3_hybrid_diverse_adaptive_e08": {
        "id": "opentry_e3_hybrid_diverse_adaptive_e08",
        "family": "opentry_E_geometry_aware_ranking",
        "wa_source": "hybrid_union",
        "wa_strategy": "external_plus_internal",
        "selector": "wa_diversity",
        "geometry_mode": "e08",
        "geometry_plan": "adaptive",
        "description": "Opentry non-oracle hybrid W/A candidates with skeleton diversity and adaptive rank-0 W/A spread for complex samples.",
    },
}


def install_custom_experiment(experiment_id: str) -> None:
    if experiment_id not in CUSTOM_EXPERIMENTS:
        return
    if any(str(exp.get("id")) == experiment_id for exp in camp.EXPERIMENTS):
        return
    camp.EXPERIMENTS.append(dict(CUSTOM_EXPERIMENTS[experiment_id]))


class GenerationTimeout(RuntimeError):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise GenerationTimeout(f"generation timeout signal={signum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Opentry streaming wrapper for SymCIF-v5 experiments.")
    parser.add_argument("--dataset", choices=sorted(camp.DATASETS), default="mpts52")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=56)
    parser.add_argument("--generation-timeout-seconds", type=int, default=180)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-sites", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def camp_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        stage="run",
        dataset=args.dataset,
        split=args.split,
        experiment_ids=args.experiment_id,
        lookup_json=args.lookup_json,
        run_dir=args.run_dir,
        report_dir=args.report_dir,
        train_limit=args.train_limit,
        max_samples=args.max_samples,
        top_k=args.top_k,
        seed=1337,
        eval_workers=args.eval_workers,
        bond_timeout_seconds=args.bond_timeout_seconds,
        valid_timeout_seconds=args.valid_timeout_seconds,
        rmsd_timeout_seconds=args.rmsd_timeout_seconds,
        parse_timeout_seconds=args.parse_timeout_seconds,
        sg_timeout_seconds=args.sg_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_sites=args.max_sites,
        skip_existing=args.skip_existing,
        remove_gt_sg=False,
    )


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def existing_completed_samples(path: Path, top_k: int) -> set[int]:
    if not path.exists():
        return set()
    counts: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["sample_index"])
            counts[idx] = counts.get(idx, 0) + 1
    return {idx for idx, count in counts.items() if count >= top_k}


def streaming_generate(args: argparse.Namespace) -> None:
    install_custom_experiment(args.experiment_id)
    cargs = camp_args(args)
    train_records, records = camp.load_split_records(cargs)
    engine, geom_index, wa_index, external = camp.build_context(cargs, train_records, records)
    experiments = camp.selected_experiments(args.experiment_id)
    if len(experiments) != 1:
        raise SystemExit(f"Expected one experiment for streaming wrapper, got {len(experiments)}")
    exp = experiments[0]
    out_dir = camp.run_dir_for(cargs)
    path = out_dir / "generations" / f"{exp['id']}.jsonl"
    completed = existing_completed_samples(path, args.top_k) if args.skip_existing else set()
    if path.exists() and not args.skip_existing:
        path.unlink()
        completed = set()

    started = time.time()
    timeout_count = 0
    signal.signal(signal.SIGALRM, _timeout_handler)
    for sample_index, record in enumerate(records):
        if sample_index in completed:
            continue
        if sample_index and sample_index % 250 == 0:
            print(
                json.dumps(
                    {
                        "stage": "stream_generate_progress",
                        "dataset": args.dataset,
                        "split": args.split,
                        "experiment": exp["id"],
                        "done": sample_index,
                        "timeouts": timeout_count,
                        "seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        try:
            signal.alarm(int(args.generation_timeout_seconds))
            rows = camp.build_candidates(
                exp,
                record,
                sample_index,
                engine,
                geom_index,
                wa_index,
                external,
                args.top_k,
                remove_gt_sg=False,
            )
            signal.alarm(0)
        except GenerationTimeout:
            signal.alarm(0)
            timeout_count += 1
            rows = camp.fg.pad_missing([], exp["id"], record, sample_index, args.top_k, 0)
            for row in rows:
                row["generation_timeout"] = True
                row["generation_timeout_seconds"] = int(args.generation_timeout_seconds)
        append_jsonl(path, rows)
    print(
        json.dumps(
            {
                "stage": "stream_generated",
                "dataset": args.dataset,
                "split": args.split,
                "experiment": exp["id"],
                "rows": sum(1 for _ in path.open("r", encoding="utf-8")),
                "timeouts": timeout_count,
                "seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    camp.write_eval_pool(cargs, experiments)


def main() -> int:
    args = parse_args()
    install_custom_experiment(args.experiment_id)
    camp.REPORT_DIR = Path(args.report_dir)
    streaming_generate(args)
    cargs = camp_args(args)
    camp.evaluate(cargs)
    camp.synthesize(cargs)
    camp.write_dataset_report(cargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
