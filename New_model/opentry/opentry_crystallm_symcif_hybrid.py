#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
class HybridSpec:
    exp_id: str
    sequence: tuple[str, ...]
    description: str


K5_SPECS: tuple[HybridSpec, ...] = (
    HybridSpec(
        "opentry_e18_sym1_crystallm4_gt_sg_hybrid_k5",
        ("sym0", "crys1", "crys2", "crys3", "crys4"),
        "SymCIF best top1, then CrystaLLM GT-SG ranks 1-4.",
    ),
    HybridSpec(
        "opentry_e19_sym2_crystallm3_gt_sg_hybrid_k5",
        ("sym0", "sym1", "crys1", "crys2", "crys3"),
        "SymCIF best top2, then CrystaLLM GT-SG ranks 1-3.",
    ),
    HybridSpec(
        "opentry_e20_sym3_crystallm2_gt_sg_hybrid_k5",
        ("sym0", "sym1", "sym2", "crys1", "crys2"),
        "SymCIF best top3, then CrystaLLM GT-SG ranks 1-2.",
    ),
    HybridSpec(
        "opentry_e21_crystallm1_sym1_crystallm3_gt_sg_hybrid_k5",
        ("crys1", "sym0", "crys2", "crys3", "crys4"),
        "CrystaLLM GT-SG top1 first, then SymCIF top1, then CrystaLLM ranks 2-4.",
    ),
    HybridSpec(
        "opentry_e22_sym1_crystallm2_sym2_gt_sg_hybrid_k5",
        ("sym0", "crys1", "crys2", "sym1", "sym2"),
        "SymCIF top1, CrystaLLM GT-SG top2, then SymCIF ranks 2-3.",
    ),
)


K20_SPECS: tuple[HybridSpec, ...] = (
    HybridSpec(
        "opentry_e23_sym1_crystallm19_gt_sg_hybrid_k20",
        tuple(["sym0"] + [f"crys{i}" for i in range(1, 20)]),
        "SymCIF best top1, then CrystaLLM GT-SG ranks 1-19.",
    ),
    HybridSpec(
        "opentry_e24_sym2_crystallm18_gt_sg_hybrid_k20",
        tuple(["sym0", "sym1"] + [f"crys{i}" for i in range(1, 19)]),
        "SymCIF best top2, then CrystaLLM GT-SG ranks 1-18.",
    ),
    HybridSpec(
        "opentry_e25_sym1_crystallm9_sym10_gt_sg_hybrid_k20",
        tuple(["sym0"] + [f"crys{i}" for i in range(1, 10)] + [f"sym{i}" for i in range(1, 11)]),
        "SymCIF top1, CrystaLLM GT-SG ranks 1-9, then SymCIF ranks 2-11.",
    ),
)


CRYSTALLM_REF5_SPECS: tuple[HybridSpec, ...] = (
    HybridSpec(
        "opentry_ref_crystallm5_gt_sg_k5",
        tuple(f"crys{i}" for i in range(1, 6)),
        "CrystaLLM GT-SG ranks 1-5 only; same evaluator reference.",
    ),
)


CRYSTALLM_REF20_SPECS: tuple[HybridSpec, ...] = (
    HybridSpec(
        "opentry_ref_crystallm20_gt_sg_k20",
        tuple(f"crys{i}" for i in range(1, 21)),
        "CrystaLLM GT-SG ranks 1-20 only; same evaluator reference.",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def material_id(record: dict[str, Any]) -> str:
    mid = record.get("material_id")
    if mid:
        return str(mid)
    sid = str(record.get("sample_id", ""))
    return sid.rsplit("__", 1)[-1] if "__" in sid else sid


def load_symcif_by_sample(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        idx = int(row["sample_index"])
        by_sample.setdefault(idx, []).append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda r: int(r.get("gen_index", 0)))
    return by_sample


def crystallm_row(
    *,
    exp_id: str,
    record: dict[str, Any],
    sample_index: int,
    gen_index: int,
    rank: int,
    gen_dir: Path,
) -> dict[str, Any]:
    mid = material_id(record)
    cif_path = gen_dir / f"{mid}__{rank}.cif"
    text = cif_path.read_text(encoding="utf-8") if cif_path.exists() else ""
    target_skel, target_wa = camp.fg.target_keys(record)
    return {
        "mode": exp_id,
        "sample_index": sample_index,
        "sample_id": record["sample_id"],
        "material_id": mid,
        "gen_index": gen_index,
        "seed": gen_index,
        "raw_generation_success": bool(text),
        "generated_text": text,
        "error": None if text else f"missing_crystallm_rank_{rank}",
        "formula_closure_success": False,
        "atom_count_ok": False,
        "canonical_skeleton_key": "",
        "canonical_wa_key": "",
        "target_canonical_skeleton_key": target_skel,
        "target_canonical_wa_key": target_wa,
        "skeleton_hit": False,
        "wa_hit": False,
        "row_count_pred": 0,
        "row_count_target": camp.row_count(record),
        "row_count_hit": False,
        "generation_score": -float(rank),
        "geometry_source": "crystallm_gt_sg_basemodel",
        "geometry_rank": None,
        "source_sample_id": None,
        "crystallm_rank": int(rank),
        "crystallm_model": "cif_model_mpts_52_b",
        "crystallm_prompt_group": "g0_data_atomtype_sg",
        "hybrid_source": "crystallm_gt_sg",
        "generated_sha1": camp.fg.raw_cif_hash(text),
        "canonical_cif_sha1": camp.fg.canonical_cif_hash(text),
    }


def symcif_row(exp_id: str, source: dict[str, Any], gen_index: int) -> dict[str, Any]:
    row = dict(source)
    row["mode"] = exp_id
    row["gen_index"] = int(gen_index)
    row["source_experiment"] = source.get("mode")
    row["source_gen_index"] = source.get("gen_index")
    row["hybrid_source"] = "symcif"
    return row


def build_rows(
    specs: tuple[HybridSpec, ...],
    records: list[dict[str, Any]],
    sym_by_sample: dict[int, list[dict[str, Any]]],
    crystallm_gen_dir: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for spec in specs:
        experiments.append(
            {
                "id": spec.exp_id,
                "family": "opentry_crystallm_symcif_hybrid",
                "wa_source": "symcif_plus_crystallm_gt_sg_predictions",
                "wa_strategy": "+".join(spec.sequence),
                "selector": "fixed_nonoracle_source_rank",
                "geometry_mode": "mixed_cached_predictions",
                "geometry_plan": "cached",
                "description": spec.description,
            }
        )
        rows: list[dict[str, Any]] = []
        for sample_index, record in enumerate(records):
            gen_index = 0
            for token in spec.sequence:
                if token.startswith("sym"):
                    rank = int(token[3:])
                    source_rows = sym_by_sample.get(sample_index, [])
                    if rank < len(source_rows):
                        rows.append(symcif_row(spec.exp_id, source_rows[rank], gen_index))
                    else:
                        rows.append(crystallm_row(exp_id=spec.exp_id, record=record, sample_index=sample_index, gen_index=gen_index, rank=9999, gen_dir=crystallm_gen_dir))
                        rows[-1]["raw_generation_success"] = False
                        rows[-1]["generated_text"] = ""
                        rows[-1]["error"] = f"missing_symcif_rank_{rank}"
                        rows[-1]["hybrid_source"] = "missing_symcif"
                elif token.startswith("crys"):
                    rank = int(token[4:])
                    rows.append(
                        crystallm_row(
                            exp_id=spec.exp_id,
                            record=record,
                            sample_index=sample_index,
                            gen_index=gen_index,
                            rank=rank,
                            gen_dir=crystallm_gen_dir,
                        )
                    )
                else:
                    raise ValueError(f"unknown sequence token: {token}")
                gen_index += 1
        camp.write_jsonl(out_dir / "generations" / f"{spec.exp_id}.jsonl", rows)
    return experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed SymCIF + CrystaLLM GT-SG hybrid candidate orderings.")
    parser.add_argument("--dataset", default="mpts52")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["k5", "k20", "ref5", "ref20"], default="k5")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--symcif-generation", type=Path, default=REPO_ROOT / "model/New_model/opentry/runs/cached_ensemble_mpts52_test_e13_e16_k20/mpts52/test/generations/opentry_e13_ensemble_e1_top4_consensus_unique_e08.jsonl")
    parser.add_argument("--crystallm-gen-dir", type=Path, default=REPO_ROOT / "model/scp_task/CrystaLLM/reproduce/mpts52_gt_prompt_module_ablation_suite7_20260305/mpts52_test_gt_suite7_k1_k20/cifs/g0_data_atomtype_sg")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--eval-workers", type=int, default=56)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camp.REPORT_DIR = Path(args.report_dir)
    if args.mode == "k5":
        specs = K5_SPECS
    elif args.mode == "k20":
        specs = K20_SPECS + CRYSTALLM_REF20_SPECS
    elif args.mode == "ref5":
        specs = CRYSTALLM_REF5_SPECS
    elif args.mode == "ref20":
        specs = CRYSTALLM_REF20_SPECS
    else:
        raise ValueError(f"unsupported mode: {args.mode}")
    args.experiment_ids = ",".join(spec.exp_id for spec in specs)
    args.top_k = 5 if args.mode in {"k5", "ref5"} else 20
    args.remove_gt_sg = False

    _, records = camp.load_split_records(args)
    out_dir = camp.run_dir_for(args)
    sym_by_sample = load_symcif_by_sample(args.symcif_generation)
    experiments = build_rows(specs, records, sym_by_sample, args.crystallm_gen_dir, out_dir)
    for exp in experiments:
        if not any(e.get("id") == exp["id"] for e in camp.EXPERIMENTS):
            camp.EXPERIMENTS.append(exp)
    camp.write_eval_pool(args, experiments)
    camp.evaluate(args)
    camp.synthesize(args)
    camp.write_dataset_report(args)
    print(
        json.dumps(
            {
                "stage": "crystallm_symcif_hybrid_done",
                "mode": args.mode,
                "dataset": args.dataset,
                "split": args.split,
                "experiments": [spec.exp_id for spec in specs],
                "samples": len(records),
                "top_k": args.top_k,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
