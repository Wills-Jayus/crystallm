#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from opentry8_tools import (
    ROOT,
    append_log,
    build_target_cache,
    candidate_text,
    dataset_info,
    load_candidates,
    target_aliases,
    write_json,
    write_jsonl,
)


def dataset_key(dataset: str) -> str:
    prefix = dataset_info(dataset)["prefix"]
    if prefix == "mp_20":
        return "mp20"
    if prefix == "mpts_52":
        return "mpts52"
    raise ValueError(dataset)


def find_group(candidates: dict[str, list[dict[str, Any]]], target: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    for alias in target_aliases(target):
        if alias in candidates:
            return alias, candidates[alias]
    return None, []


def failure_placeholder(sample_id: str, material_id: str, slot: int) -> str:
    return (
        f"# opentry_8 explicit failure placeholder\n"
        f"# sample_id: {sample_id}\n"
        f"# material_id: {material_id}\n"
        f"# slot: {slot}\n"
        f"# This is intentionally invalid CIF text so the official evaluator counts the slot as a failure.\n"
    )


def add_from_group(
    chosen: list[dict[str, Any]],
    seen_texts: set[str],
    target: dict[str, Any],
    group: list[dict[str, Any]],
    source_kind: str,
    source_alias: str | None,
    budget: int,
) -> int:
    added = 0
    for cand in group:
        if len(chosen) >= budget:
            break
        text = candidate_text(cand)
        if not text.strip() or text in seen_texts:
            continue
        seen_texts.add(text)
        chosen.append(
            {
                "system": "strategy_fusion",
                "dataset": target["dataset"],
                "split": target["split"],
                "sample_id": target["sample_id"],
                "material_id": target["material_id"],
                "generated_text": text,
                "source_kind": source_kind,
                "source_path": cand.get("source_path"),
                "source_candidate_alias": source_alias,
                "source_gen_index": cand.get("gen_index"),
                "source_rank": cand.get("rank"),
                "placeholder_failure": False,
            }
        )
        added += 1
    return added


def build_dataset(dataset: str, config_path: Path, out_dir: Path, budget: int = 20) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    key = dataset_key(dataset)
    prefix = dataset_info(dataset)["prefix"]
    targets = build_target_cache(dataset, "test", refresh=False, fast_row_count=True)

    primary_path = Path(config["primary_source"][key])
    primary = load_candidates(primary_path)
    supplemental_paths = [Path(p) for p in config.get("supplemental_sources", {}).get(key, [])]
    supplementals = [(path, load_candidates(path)) for path in supplemental_paths]

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "system": "strategy_fusion",
        "dataset": prefix,
        "split": "test",
        "budget": budget,
        "config": str(config_path),
        "primary_source": str(primary_path),
        "supplemental_sources": [str(p) for p in supplemental_paths],
        "official_samples": len(targets),
        "candidate_rows": 0,
        "samples_with_20_slots": 0,
        "samples_with_primary_candidate": 0,
        "samples_with_supplemental_candidate": 0,
        "samples_requiring_supplemental": 0,
        "samples_requiring_placeholder": 0,
        "primary_slots": 0,
        "supplemental_slots": 0,
        "placeholder_slots": 0,
        "empty_primary_groups": 0,
        "empty_supplemental_groups": 0,
    }

    for target in targets:
        chosen: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        primary_alias, primary_group = find_group(primary, target)
        primary_added = add_from_group(chosen, seen_texts, target, primary_group, "primary_crystallm_a_gt_sg", primary_alias, budget)
        if primary_added:
            manifest["samples_with_primary_candidate"] += 1
            manifest["primary_slots"] += primary_added
        else:
            manifest["empty_primary_groups"] += 1

        used_supplemental_for_sample = False
        if len(chosen) < budget:
            for source_path, source_candidates in supplementals:
                source_alias, source_group = find_group(source_candidates, target)
                before = len(chosen)
                added = add_from_group(
                    chosen,
                    seen_texts,
                    target,
                    source_group,
                    f"supplemental:{source_path.name}",
                    source_alias,
                    budget,
                )
                if added:
                    used_supplemental_for_sample = True
                    manifest["supplemental_slots"] += added
                elif not source_group:
                    manifest["empty_supplemental_groups"] += 1
                if len(chosen) >= budget:
                    break
                if len(chosen) == before:
                    continue
        if used_supplemental_for_sample:
            manifest["samples_with_supplemental_candidate"] += 1
            if primary_added < budget:
                manifest["samples_requiring_supplemental"] += 1

        if len(chosen) < budget:
            manifest["samples_requiring_placeholder"] += 1
        while len(chosen) < budget:
            slot = len(chosen)
            chosen.append(
                {
                    "system": "strategy_fusion",
                    "dataset": target["dataset"],
                    "split": target["split"],
                    "sample_id": target["sample_id"],
                    "material_id": target["material_id"],
                    "generated_text": failure_placeholder(target["sample_id"], target["material_id"], slot),
                    "source_kind": "explicit_failure_placeholder",
                    "source_path": None,
                    "source_candidate_alias": None,
                    "source_gen_index": None,
                    "source_rank": None,
                    "placeholder_failure": True,
                }
            )
            manifest["placeholder_slots"] += 1

        for gen_index, row in enumerate(chosen[:budget]):
            row["gen_index"] = gen_index
            row["rank"] = gen_index + 1
            rows.append(row)
        if len(chosen) >= budget:
            manifest["samples_with_20_slots"] += 1

    manifest["candidate_rows"] = len(rows)
    out_path = out_dir / f"strategy_fusion_{prefix}_test_k{budget}_andidates.jsonl"
    # Keep the typo-free filename as the public artifact required by the prompt.
    out_path = out_dir / f"strategy_fusion_{prefix}_test_k{budget}_candidates.jsonl"
    write_jsonl(out_path, rows)
    write_json(out_path.with_suffix(out_path.suffix + ".manifest.json"), manifest)

    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} build strategy_fusion {dataset}/test\n"
        f"- config: `{config_path}`\n"
        f"- output: `{out_path}`\n"
        f"- official samples: {manifest['official_samples']}; candidate rows: {manifest['candidate_rows']}\n"
        f"- samples with 20 slots: {manifest['samples_with_20_slots']}\n"
        f"- primary slots: {manifest['primary_slots']}; supplemental slots: {manifest['supplemental_slots']}; "
        f"placeholder slots: {manifest['placeholder_slots']}"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build opentry_8 coverage-repaired strategy/fusion candidates.")
    parser.add_argument("--dataset", choices=["mp20", "mpts52", "all"], default="all")
    parser.add_argument("--config", default=str(ROOT / "frozen_strategy/config.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "generations"))
    parser.add_argument("--budget", type=int, default=20)
    args = parser.parse_args()

    datasets = ["mp20", "mpts52"] if args.dataset == "all" else [args.dataset]
    manifests = [build_dataset(ds, Path(args.config), Path(args.out_dir), budget=args.budget) for ds in datasets]
    write_json(ROOT / "frozen_strategy/strategy_fusion_build_manifest.json", manifests)


if __name__ == "__main__":
    main()
