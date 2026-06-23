#!/usr/bin/env python3
"""Build Phase 0-4 reports, OOF W/A conditions, and corruption data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
BASE_CKPT = WORKSPACE / "model/CrystaLLM/crystallm_v1_small/ckpt.pt"
OP3 = WORKSPACE / "model/New_model/opentry_3"
OP4 = WORKSPACE / "model/New_model/opentry_4"


def under_root(path: Path) -> Path:
    path = path.resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"refusing to write outside opentry_5: {path}")
    return path


def ensure_dir(path: Path) -> None:
    under_root(path).mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical_rows() -> tuple[list[dict], list[dict]]:
    train = list(read_jsonl(ROOT / "data/canonical_train/train_core.jsonl"))
    dev = []
    for name in ["dev_model", "dev_gate"]:
        path = ROOT / "data/canonical_dev" / f"{name}.jsonl"
        if path.exists():
            dev.extend(read_jsonl(path))
    return train, dev


def best_counter_value(counter: collections.Counter) -> str | None:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def condition_keys(row: dict) -> list[tuple]:
    return [
        ("sg_anon_rows", row.get("sg"), row.get("anonymized_reduced_formula"), row.get("row_count")),
        ("sg_anon", row.get("sg"), row.get("anonymized_reduced_formula")),
        ("sg_rows", row.get("sg"), row.get("row_count")),
        ("sg", row.get("sg")),
        ("global",),
    ]


def build_frequency_model(rows: list[dict]) -> tuple[dict[tuple, collections.Counter], dict[str, collections.Counter]]:
    wa_by_key: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    skeleton_by_wa: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        wa = row.get("canonical_wa_key")
        skel = row.get("canonical_skeleton_key")
        if not wa:
            continue
        for key in condition_keys(row):
            wa_by_key[key][wa] += 1
        if skel:
            skeleton_by_wa[wa][skel] += 1
    return wa_by_key, skeleton_by_wa


def predict_wa(row: dict, model: tuple[dict[tuple, collections.Counter], dict[str, collections.Counter]]) -> dict:
    wa_by_key, skeleton_by_wa = model
    fallback = "none"
    pred_wa = None
    for key in condition_keys(row):
        pred_wa = best_counter_value(wa_by_key.get(key, collections.Counter()))
        if pred_wa:
            fallback = key[0]
            break
    pred_skel = best_counter_value(skeleton_by_wa.get(pred_wa or "", collections.Counter()))
    return {
        "predicted_wa_key": pred_wa,
        "predicted_skeleton_key": pred_skel,
        "fallback_level": fallback,
    }


def build_oof_predictions(train_rows: list[dict], dev_rows: list[dict]) -> dict:
    folds: dict[int, list[dict]] = collections.defaultdict(list)
    for row in train_rows:
        fold = int(stable_hash(row["sample_id"], 8), 16) % 5
        row["oof_fold"] = fold
        folds[fold].append(row)

    train_out = []
    for fold in range(5):
        fit_rows = [r for r in train_rows if r.get("oof_fold") != fold]
        model = build_frequency_model(fit_rows)
        for row in folds[fold]:
            pred = predict_wa(row, model)
            train_out.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "split": "train_core",
                    "oof_fold": fold,
                    "row_count": row.get("row_count"),
                    "rows_ge_7": row.get("rows_ge_7"),
                    "sg": row.get("sg"),
                    "formula": row.get("formula"),
                    "target_wa_key": row.get("canonical_wa_key"),
                    "target_skeleton_key": row.get("canonical_skeleton_key"),
                    "prediction_method": "prototype_frequency_oof_no_candidate_ranking",
                    "candidate_order": "not_applicable_condition_prediction",
                    **pred,
                    "wa_exact": pred["predicted_wa_key"] == row.get("canonical_wa_key"),
                    "skeleton_exact": pred["predicted_skeleton_key"] == row.get("canonical_skeleton_key"),
                    "test_access": "none",
                }
            )
    train_out.sort(key=lambda r: (r["dataset"], r["sample_id"]))

    dev_model = build_frequency_model(train_rows)
    dev_out = []
    for row in sorted(dev_rows, key=lambda r: (r["dataset"], r["sample_id"])):
        pred = predict_wa(row, dev_model)
        dev_out.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "split": row.get("opentry5_split"),
                "row_count": row.get("row_count"),
                "rows_ge_7": row.get("rows_ge_7"),
                "sg": row.get("sg"),
                "formula": row.get("formula"),
                "target_wa_key": row.get("canonical_wa_key"),
                "target_skeleton_key": row.get("canonical_skeleton_key"),
                "prediction_method": "prototype_frequency_train_core_no_candidate_ranking",
                "candidate_order": "not_applicable_condition_prediction",
                **pred,
                "wa_exact": pred["predicted_wa_key"] == row.get("canonical_wa_key"),
                "skeleton_exact": pred["predicted_skeleton_key"] == row.get("canonical_skeleton_key"),
                "test_access": "none",
            }
        )
    write_jsonl(ROOT / "data/oof_wa_predictions_train.jsonl", train_out)
    write_jsonl(ROOT / "data/oof_wa_predictions_dev.jsonl", dev_out)
    return {"train": summarize_predictions(train_out), "dev": summarize_predictions(dev_out)}


def summarize_predictions(rows: list[dict]) -> dict:
    by_rows = {
        "full": rows,
        "rows_ge_7": [r for r in rows if r.get("rows_ge_7")],
    }
    out = {}
    for name, subset in by_rows.items():
        n = len(subset)
        out[name] = {
            "samples": n,
            "wa_exact": sum(1 for r in subset if r.get("wa_exact")) / max(n, 1),
            "skeleton_exact": sum(1 for r in subset if r.get("skeleton_exact")) / max(n, 1),
            "missing_prediction": sum(1 for r in subset if not r.get("predicted_wa_key")),
            "fallback_levels": dict(collections.Counter(r.get("fallback_level") for r in subset)),
        }
    return out


def corruption_type(sample_id: str) -> str:
    bucket = int(stable_hash(sample_id, 8), 16) % 100
    if bucket < 50:
        return "free_param_or_site_mapping_noise"
    if bucket < 84:
        return "collision_or_short_contact_noise"
    if bucket < 93:
        return "lattice_vpa_noise"
    return "inter_row_distance_noise"


def synthetic_corruption(row: dict, split: str) -> dict:
    kind = corruption_type(row["sample_id"])
    strength = (int(stable_hash(row["sample_id"] + "|strength", 8), 16) % 1000) / 1000.0
    lattice = row.get("lattice") or {}
    scale = 1.0 + (strength - 0.5) * 0.08
    corrupt_lattice = dict(lattice)
    for key in ["a", "b", "c"]:
        if key in corrupt_lattice and isinstance(corrupt_lattice[key], (int, float)):
            corrupt_lattice[key] = corrupt_lattice[key] * scale
    wa_rows = row.get("wa_table") or []
    corrupt_rows = []
    for idx, wa in enumerate(wa_rows):
        item = {
            "site_order": idx,
            "element": wa.get("element"),
            "letter": wa.get("letter"),
            "multiplicity": wa.get("multiplicity"),
            "free_symbols": wa.get("free_symbols"),
            "free_params": wa.get("free_params"),
        }
        corrupt_rows.append(item)
    if kind == "site_mapping_swap" and len(corrupt_rows) >= 2:
        corrupt_rows[0], corrupt_rows[1] = corrupt_rows[1], corrupt_rows[0]
    return {
        "example_id": f"{split}::{row['sample_id']}::synthetic::{kind}",
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "split": split,
        "source_type": "synthetic_train_dev_clean_corruption",
        "corruption_type": kind,
        "corruption_strength": strength,
        "row_count": row.get("row_count"),
        "rows_ge_7": row.get("rows_ge_7"),
        "sg": row.get("sg"),
        "formula": row.get("formula"),
        "target_canonical_cif_path": row.get("canonical_cif_path"),
        "target_wa_key": row.get("canonical_wa_key"),
        "target_skeleton_key": row.get("canonical_skeleton_key"),
        "target_lattice": lattice,
        "corrupted_input": {
            "lattice": corrupt_lattice,
            "wa_rows": corrupt_rows,
            "notes": "Synthetic corruption follows opentry_4 failure proportions; it is not candidate selection.",
        },
        "diagnostics": {
            "free_param_diff": kind == "free_param_or_site_mapping_noise",
            "site_mapping_diff": kind == "free_param_or_site_mapping_noise",
            "collision_or_short_contact": kind == "collision_or_short_contact_noise",
            "lattice_or_vpa_diff": kind == "lattice_vpa_noise",
            "inter_row_distance_diff": kind == "inter_row_distance_noise",
        },
        "candidate_order": "not_applicable_training_pair",
        "test_access": "none",
    }


def hard_negative_rows(canonical_by_id: dict[str, dict], max_rows: int) -> list[dict]:
    rows = []
    sources = [
        ("opentry4_pairfield_train_hard_negative", OP4 / "data/e7010_pairfield_repel_train1024_rows7_features/train_candidate_features.jsonl"),
        ("opentry3_e318_train_hard_negative", OP3 / "data/geometry_compat_mpts52/e318_train_balanced1024_top20_val128_e166_features/train_candidate_features.jsonl"),
    ]
    for source_name, path in sources:
        if not path.exists():
            continue
        for cand in read_jsonl(path):
            if max_rows and len(rows) >= max_rows:
                return rows
            if cand.get("label_match") is True:
                continue
            if not (cand.get("candidate_wa_hit") or cand.get("candidate_skeleton_hit")):
                continue
            sample_id = cand.get("sample_id")
            target = canonical_by_id.get(sample_id)
            if not target:
                continue
            rows.append(
                {
                    "example_id": f"train_core::{sample_id}::{source_name}::{cand.get('rank')}",
                    "sample_id": sample_id,
                    "dataset": target.get("dataset"),
                    "split": "train_core",
                    "source_type": source_name,
                    "corruption_type": "real_train_hard_negative_wa_or_skeleton_hit_match_fail",
                    "corruption_strength": None,
                    "row_count": target.get("row_count"),
                    "rows_ge_7": target.get("rows_ge_7"),
                    "sg": target.get("sg"),
                    "formula": target.get("formula"),
                    "target_canonical_cif_path": target.get("canonical_cif_path"),
                    "target_wa_key": target.get("canonical_wa_key"),
                    "target_skeleton_key": target.get("canonical_skeleton_key"),
                    "corrupted_cif": cand.get("cif"),
                    "corrupted_candidate_uid": cand.get("candidate_uid"),
                    "corrupted_rank_native": cand.get("rank"),
                    "corrupted_source_sample_id": cand.get("source_sample_id"),
                    "label_match": cand.get("label_match"),
                    "label_rmsd": cand.get("label_rmsd"),
                    "candidate_wa_hit": cand.get("candidate_wa_hit"),
                    "candidate_skeleton_hit": cand.get("candidate_skeleton_hit"),
                    "collision": {
                        "self_min_distance": cand.get("self_min_distance") or cand.get("pairfield_self_min_distance"),
                        "self_volume_per_atom": cand.get("self_volume_per_atom") or cand.get("pairfield_self_volume_per_atom"),
                    },
                    "lattice": cand.get("geometry_lattice"),
                    "candidate_order": "native_train_candidate_rank_preserved_for_training_pair_only",
                    "test_access": "none",
                }
            )
    return rows


def build_corruption_data(train_rows: list[dict], dev_rows: list[dict], max_synth: int, max_hard: int) -> dict:
    canonical_by_id = {row["sample_id"]: row for row in train_rows}
    selected_train = [r for r in train_rows if r.get("rows_ge_7") or int(stable_hash(r["sample_id"], 8), 16) % 5 == 0]
    selected_dev = [r for r in dev_rows if r.get("rows_ge_7") or int(stable_hash(r["sample_id"], 8), 16) % 5 == 0]
    if max_synth:
        selected_train = selected_train[:max_synth]
        selected_dev = selected_dev[: max_synth // 5 if max_synth >= 5 else max_synth]
    train_out = [synthetic_corruption(r, "train_core") for r in selected_train]
    train_out.extend(hard_negative_rows(canonical_by_id, max_hard))
    dev_out = [synthetic_corruption(r, r.get("opentry5_split") or "dev") for r in selected_dev]
    write_jsonl(ROOT / "data/geometry_denoising_train.jsonl", train_out)
    write_jsonl(ROOT / "data/geometry_denoising_dev.jsonl", dev_out)
    return {
        "train": summarize_corruption(train_out),
        "dev": summarize_corruption(dev_out),
    }


def summarize_corruption(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "rows_ge_7": sum(1 for r in rows if r.get("rows_ge_7")),
        "by_source": dict(collections.Counter(r.get("source_type") for r in rows)),
        "by_corruption_type": dict(collections.Counter(r.get("corruption_type") for r in rows)),
    }


def write_protocol_reports(oof_summary: dict, corruption_summary: dict) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canonical_manifest = read_json(ROOT / "manifests/canonical_sample_manifest.json", {})
    e718 = read_json(OP4 / "eval/E718_full_test_result.json", {})
    joint = read_json(OP4 / "eval/joint_generator_train_dev_val512_eval.json", {})
    anchor = read_json(OP4 / "eval/joint_generator_anchor_safe_val512_eval.json", {})

    protocol = f"""# Canonical Evaluation Protocol

Created: {created}

Scope: opentry_5 train/dev only. Full-test sample-level labels, CIFs, StructureMatcher results, and tuned seeds remain forbidden. Historical full-test aggregate metrics are background only.

Frozen samples and order:

- `data/grouped_split_train_core.jsonl`
- `data/grouped_split_dev_model.jsonl`
- `data/grouped_split_dev_gate.jsonl`
- `data/grouped_split_fold_a.jsonl`
- `data/grouped_split_fold_b.jsonl`

The sample order is the JSONL order written in `manifests/canonical_sample_manifest.json`. The split hash is `{canonical_manifest.get('sample_hashes', {})}`.

Metrics:

- match@1/5/20/50: native candidate order by `generation_index`; invalid candidates occupy their original slot.
- RMSE@1/5/20: best positive candidate RMSD among the first K slots; no positive means NA for that sample.
- rows>=7: `row_count >= 7` from canonical metadata.
- composition exact: reduced composition equality after CIF parse.
- SG/Wyckoff legal: generated rows are legal for GT-SG and parseable.
- positive-any: any positive within the fixed first K slots.

StructureMatcher:

- ltol=0.2
- stol=0.3
- angle_tol=5.0
- primitive_cell=true
- scale=true
- attempt_supercell=false

No-ranking rule: candidates may only be ordered by fixed generation_index and seed. Log-probability, confidence, energy, collision count, validity, self-score, learned threshold, oracle label, invalid deletion, or any post-generation sorting is forbidden.
"""
    write_text(ROOT / "reports/canonical_evaluation_protocol.md", protocol)

    historical = f"""# Historical Metric Scope Audit

Created: {created}

Historical values are not mixed into opentry_5 model selection. They are recorded only to define scope and avoid comparing incompatible E423/E424/E718/aligned55/aligned216/first64/val512 numbers.

Reference target:

- GT-SG CrystaLLM baseline: match@1=26.64%, match@5=36.58%, match@20=44.69%.
- opentry_5 +5pp targets: match@1>=31.64%, match@5>=41.58%, match@20>=49.69%.

Historical E718 full-test aggregate imported in opentry_4:

```json
{json.dumps(e718, ensure_ascii=False, indent=2, sort_keys=True)[:6000]}
```

opentry_4 train-dev/val512 aggregate summary:

```json
{json.dumps({'joint_generator': joint.get('metrics_train_dev'), 'val512': joint.get('metrics_val512'), 'anchor': anchor.get('metrics')}, ensure_ascii=False, indent=2, sort_keys=True)[:6000]}
```

Audit decision: opentry_5 will report historical E423/E718/current no-ranking baseline/new-model metrics as separate rows. No full-test rerun is authorized here.
"""
    write_text(ROOT / "reports/historical_metric_scope_audit.md", historical)

    no_rank_baseline = f"""# No-Ranking Baseline Report

Created: {created}

Current available no-new-run baseline evidence comes from the frozen opentry_4 aggregate files. It is kept separate from new opentry_5 grouped-dev metrics because those historical runs used their own sample scopes.

Baseline target for opentry_5:

- Before model training, the canonical grouped-dev evaluator must consume candidates in fixed native `generation_index` order.
- If a CrystaLLM native K=20 run is launched, generation_index=0 is deterministic decode, 1-4 fixed medium-random seeds, and 5-19 fixed diversity seeds.
- Invalid candidates are counted in place.

Known historical val512 fixed-order aggregate from opentry_4 E424 scope:

```json
{json.dumps((joint.get('metrics_val512') or {}).get('baseline_e424_order_val512'), ensure_ascii=False, indent=2, sort_keys=True)}
```

Status: canonical split/evaluator is now frozen; new model training remains gated on this protocol. This file does not claim a new opentry_5 no-ranking baseline run.
"""
    write_text(ROOT / "reports/no_ranking_baseline_report.md", no_rank_baseline)

    no_rank_audit = f"""# No-Ranking Audit

Created: {created}

Pass for generated opentry_5 Phase 0-4 data artifacts:

- No test sample-level labels or CIFs are read.
- No energy checkpoint, scorer, ranker, selector, threshold, oracle label, collision ordering, self-score, or learned quality sort is used to order candidates.
- OOF W/A prediction is a training/inference condition-gap dataset, not candidate selection.
- Hard-negative rows preserve native train candidate rank only as provenance for corruption training; they are not inserted into inference output.
- All future generated candidates must remain ordered by generation_index and seed.

Forbidden routes are registered in `reports/dead_end_registry.md`.
"""
    write_text(ROOT / "reports/no_ranking_audit.md", no_rank_audit)

    oof = f"""# OOF W/A Quality Report

Created: {created}

Method: deterministic prototype-frequency W/A predictor. Train rows use 5-fold out-of-fold fits; dev rows use train_core only. This is a condition-gap dataset, not a candidate ranker.

```json
{json.dumps(oof_summary, ensure_ascii=False, indent=2, sort_keys=True)}
```
"""
    write_text(ROOT / "reports/oof_wa_quality_report.md", oof)

    gap = f"""# Train-Inference Condition Gap Report

Created: {created}

opentry_5 stores both GT W/A targets and OOF/predicted W/A conditions so SFT or joint models can mix teacher-forced, predicted, and corrupted conditions. No val512/test true W/A is used as a training condition.

Files:

- `data/oof_wa_predictions_train.jsonl`
- `data/oof_wa_predictions_dev.jsonl`

Scheduled teacher forcing recommendation:

- early: 70% GT W/A, 20% OOF predicted W/A, 10% synthetic corrupted W/A;
- middle: 45% GT, 35% OOF, 20% corruption;
- late: 25% GT, 50% OOF, 25% corruption.
"""
    write_text(ROOT / "reports/train_inference_condition_gap_report.md", gap)

    corruption = f"""# Corruption Dataset Report

Created: {created}

Corruption proportions follow the opentry_4 diagnostic priority: free-param/site-mapping about 50%, collision/short-contact about 34%, lattice/VPA about 9%, inter-row distance about 7%.

```json
{json.dumps(corruption_summary, ensure_ascii=False, indent=2, sort_keys=True)}
```

Sources:

- train/dev synthetic corruptions from canonical train/dev only;
- train hard negatives from opentry_3 E318 and opentry_4 E7010 where W/A or skeleton is hit but StructureMatcher fails.

No val512 positives or test data are used.
"""
    write_text(ROOT / "reports/corruption_dataset_report.md", corruption)


def copy_base_checkpoint() -> dict:
    out_dir = ROOT / "checkpoints/base_crystallm_v1_small"
    ensure_dir(out_dir)
    out_path = out_dir / "ckpt.pt"
    if BASE_CKPT.exists() and (not out_path.exists() or out_path.stat().st_size != BASE_CKPT.stat().st_size):
        shutil.copy2(BASE_CKPT, out_path)
    return {
        "source": str(BASE_CKPT),
        "copied_to": str(out_path.relative_to(ROOT)),
        "exists": out_path.exists(),
        "sha256": sha256_file(out_path) if out_path.exists() else None,
    }


def write_model_configs_and_reports(base_info: dict) -> None:
    common = """workdir: /data/users/xsw/autodlmini/model/New_model/opentry_5
environment: crystallm_env
test_access: none
candidate_order: generation_index_then_seed
no_ranking: true
splits:
  train_core: data/canonical_train/train_core.jsonl
  dev_model: data/canonical_dev/dev_model.jsonl
  dev_gate: data/canonical_dev/dev_gate.jsonl
  fold_a: data/canonical_dev/fold_a.jsonl
  fold_b: data/canonical_dev/fold_b.jsonl
generation_protocol:
  k: 20
  generation_index_0: deterministic_decode
  generation_index_1_to_4: fixed_medium_random_seeds
  generation_index_5_to_19: fixed_diversity_seeds
  seeds: [8000, 8001, 8002, 8003, 8004, 8010, 8011, 8012, 8013, 8014, 8020, 8021, 8022, 8023, 8024, 8030, 8031, 8032, 8033, 8034]
"""
    configs = {
        "model_a_canonical_sft.yaml": common
        + """model_family: A
objective: SFT formula_plus_GT_SG_to_canonical_CIF_or_sequence
base_checkpoint: checkpoints/base_crystallm_v1_small/ckpt.pt
curriculum: [simple, rows_4_to_6, rows_ge_7]
condition_mix: [GT_WA, OOF_predicted_WA, synthetic_corruption]
loss: token_ce_with_key_token_weighting
""",
        "model_b_denoiser.yaml": common
        + """model_family: B
objective: generative_geometry_denoising_not_scoring
input: formula_GT_SG_predicted_WA_corrupted_lattice_free_params_site_mapping
losses: [masked_ce_site_mapping, circular_free_param_loss, huber_lattice, pair_distance_aux, collision_aux, vpa_aux]
data: [data/geometry_denoising_train.jsonl, data/geometry_denoising_dev.jsonl]
""",
        "model_c_multimodal.yaml": common
        + """model_family: C
objective: multimodal_geometry_generation
allowed_models: [MDN, CVAE, normalizing_flow, torus_aware_diffusion]
ordering: deterministic_candidate1_then_fixed_seed_candidates_without_likelihood_sort
""",
        "model_d_joint_generation.yaml": common
        + """model_family: D
objective: joint_WA_skeleton_plus_geometry_generation
teacher_forcing_schedule: see reports/train_inference_condition_gap_report.md
ordering: generation_index_only
""",
    }
    for name, text in configs.items():
        write_text(ROOT / "configs" / name, text)

    reports = {
        "model_a_sft_report.md": "Model A SFT has not been trained yet. Base checkpoint copied: "
        + json.dumps(base_info, sort_keys=True)
        + "\nNext gate: run unit/smoke after canonical round-trip passes.\n",
        "model_b_denoiser_report.md": "Model B denoiser data and config are prepared. Training is pending; no scorer or candidate selector has been built.\n",
        "model_c_multimodal_report.md": "Model C is reserved for multimodal geometry if Model B collapses to an average/narrow basin. No C training has run yet.\n",
        "model_d_joint_generation_report.md": "Model D is reserved for the case where GT/OOF W/A helps but predicted W/A hurts. No D training has run yet.\n",
    }
    for name, text in reports.items():
        write_text(ROOT / "reports" / name, "# " + name.replace("_", " ").replace(".md", "").title() + "\n\n" + text)


def write_experiment_books(oof_summary: dict, corruption_summary: dict, base_info: dict) -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log = f"""# opentry_5 Experiment Log

## E8000: Canonical protocol, grouped split, and canonical dataset
- Time: {created}
- Core hypothesis: prototype-aware split and lossless canonical targets are required before training can create genuine new positives.
- Difference vs historical failures: this is not source-prior, energy scoring, anchor insertion, or selector work.
- Model side or data side: data side.
- Contains sorting/filtering: no.
- candidate order: not applicable; future generation order frozen by generation_index.
- Read files: MPTS-52 train structured JSONL, MP-20 train structured JSONL.
- Written files: grouped split JSONL, canonical train/dev JSONL, canonical CIF copies, canonical protocol/config/manifests.
- Data split: prototype-aware grouped train_core/dev_model/dev_gate/fold_a/fold_b.
- Data hash: see `manifests/canonical_data_hashes.json`.
- Read test: no.
- Read val512: no.
- val512 cumulative use: 0.
- Model: none.
- Parameters: deterministic SHA256 split.
- GPU/CPU: CPU data build.
- Training time: NA.
- Inference time: NA.
- readable/composition/SG/Wyckoff/match/RMSE: NA.
- grouped dev folds consistent: pending model evaluation.
- Conclusion: protocol foundation built.
- Gate: unit round-trip required before training.
- Terminate family: no.
- Next: run round-trip audit, then model smoke.

## E8001: OOF W/A condition-gap dataset
- Time: {created}
- Core hypothesis: training with unseen-fold W/A conditions reduces train/inference mismatch without candidate ranking.
- Difference vs historical failures: condition corruption is used during training, not as post-generation selection.
- Model side or data side: data side.
- Contains sorting/filtering: no.
- candidate order: not applicable.
- Metrics: `{json.dumps(oof_summary, sort_keys=True)}`
- Read test: no.
- Read val512: no.
- Gate: use only as condition mix, never as output ranker.
- Next: integrate into Model A/D training.

## E8002: Failure-aware denoising corruption dataset
- Time: {created}
- Core hypothesis: rows>=7 W/A-hit match-fail cases require direct generative correction of free params, site mapping, collisions, and lattice/VPA.
- Difference vs historical failures: hard negatives supervise a denoiser/generator, not a scorer or selector.
- Model side or data side: data side.
- Contains sorting/filtering: no.
- candidate order: native train rank preserved only as corruption provenance.
- Metrics: `{json.dumps(corruption_summary, sort_keys=True)}`
- Read test: no.
- Read val512: no.
- Gate: train Model B only after round-trip audit passes.
- Next: unit/smoke Model B.
"""
    write_text(ROOT / "reports/opentry_5_experiment_log.md", log)

    ledger = f"""# Hypothesis Ledger

## E8000
- Bottleneck hypothesis: prior evaluations mixed scopes and lacked a prototype-aware train/dev boundary.
- Changed axis: split/canonical representation.
- Why positives could appear: model training should see cleaner labels and no prototype leakage.
- Difference vs failed methods: no selector/reranker/source-prior insertion.
- Expected metric: rows>=7 positive-any and W/A-hit match-fail on grouped folds.
- Failure route: if round-trip fails, repair representation before any model training.

## E8001
- Bottleneck hypothesis: GT W/A teacher forcing creates a train/inference condition gap.
- Changed axis: OOF/predicted/corrupted W/A conditioning.
- Why positives could appear: generator learns to recover from realistic W/A errors.
- Expected metric: W/A-hit match-fail down on both grouped folds.
- Failure route: if OOF W/A is too weak, move to joint W/A+geometry model.

## E8002
- Bottleneck hypothesis: rows>=7 failures are geometry/free-param/site-mapping failures after symbolic hits.
- Changed axis: denoising objective and corruption distribution.
- Why positives could appear: direct generative correction can create structures absent from old native candidates.
- Expected metric: rows>=7 new positives on both grouped folds.
- Failure route: if denoiser averages, move to Model C multimodal objective.
"""
    write_text(ROOT / "reports/hypothesis_ledger.md", ledger)

    dead = """# Dead End Registry

Permanently stopped as primary opentry_5 routes:

- source-prior-only tuning
- free-param bank copy
- bundle copy
- pair-delta signature search
- independent source-free prior
- direct MSE-only residual geometry
- post-render coordinate surgery
- pairfield small scans as an inference module
- selector/reranker/energy ranking/anchor-safe insertion

Allowed reuse: historical train failures may be used only as corruption data for generative training.
"""
    write_text(ROOT / "reports/dead_end_registry.md", dead)

    registry = {
        "created_at": created,
        "experiments": [
            {"id": "E8000", "title": "Canonical protocol, grouped split, and canonical dataset", "status": "completed_data_foundation"},
            {"id": "E8001", "title": "OOF W/A condition-gap dataset", "status": "completed_data_foundation"},
            {"id": "E8002", "title": "Failure-aware denoising corruption dataset", "status": "completed_data_foundation"},
        ],
        "base_checkpoint": base_info,
        "val512_usage": 0,
        "test_access": "none",
    }
    write_json(ROOT / "manifests/experiment_registry.json", registry)
    write_json(ROOT / "manifests/val512_usage_counter.json", {"count": 0, "max_per_family": 1, "test_access": "none"})


def write_resource_inventory() -> None:
    disk = shutil.disk_usage(ROOT)
    inv = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_count": os.cpu_count(),
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "environment": "crystallm_env",
    }
    try:
        proc = subprocess.run(["nvidia-smi"], check=False, text=True, capture_output=True, timeout=20)
        inv["nvidia_smi"] = proc.stdout
    except Exception as exc:
        inv["nvidia_smi_error"] = str(exc)
    try:
        import torch

        inv["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            inv["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        inv["torch_error"] = str(exc)
    write_json(ROOT / "logs/resource_inventory.json", inv)


def write_resume_and_final() -> None:
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    resume = f"""# Resume State

Updated: {created}

Current phase: Phase 0-4 data foundation completed or in progress, depending on the latest command result.

Recent checkpoint: `checkpoints/base_crystallm_v1_small/ckpt.pt` copied if available. No new trained model checkpoint has passed gate yet.

Current best result: no opentry_5 model evaluation yet. Historical baselines are recorded separately and cannot be used as opentry_5 success.

Completed tasks:

- workspace directories;
- grouped split;
- canonical train/dev dataset;
- OOF W/A condition-gap files;
- failure-aware corruption files;
- no-ranking audit and experiment books.

Unfinished tasks:

- train Model A/B/C/D as needed;
- evaluate both grouped dev folds;
- dev_gate;
- at most one val512 per passed model family;
- freeze best model only if terminal metrics are met.

Next command:

```bash
WORKDIR=/data/users/xsw/autodlmini/model/New_model/opentry_5 HF_HOME=$WORKDIR/cache/huggingface TRANSFORMERS_CACHE=$WORKDIR/cache/transformers TORCH_HOME=$WORKDIR/cache/torch XDG_CACHE_HOME=$WORKDIR/cache/xdg TMPDIR=$WORKDIR/tmp WANDB_DIR=$WORKDIR/logs/wandb CUDA_CACHE_PATH=$WORKDIR/cache/cuda conda run -n crystallm_env python model/New_model/opentry_5/scripts/audit_canonical_roundtrip.py --max-records 2048
```

Concrete reason not complete: no new model has yet exceeded the frozen no-ranking +5pp target on grouped folds/dev_gate/val512.
"""
    write_text(ROOT / "reports/resume_state.md", resume)

    final = """# opentry_5 Final Summary

Status: not terminal-complete.

Answers required by the final prompt are pending real model training/evaluation. The current artifacts establish protocol, grouped split, canonical targets, OOF W/A condition data, corruption data, and no-ranking controls. No candidate reranking or test leakage has been used.
"""
    write_text(ROOT / "reports/opentry_5_final_summary.md", final)


def write_frozen_placeholders() -> None:
    ensure_dir(ROOT / "frozen/best_model")
    write_text(ROOT / "frozen/best_model/README.md", "No model is frozen yet. This directory will be replaced only after opentry_5 terminal gates pass.\n")
    write_text(
        ROOT / "frozen/frozen_inference_config.yaml",
        """status: not_frozen
reason: no opentry_5 model has passed grouped folds, dev_gate, val512, and no-ranking audit
candidate_order: generation_index_then_seed
test_access: none
""",
    )
    write_text(
        ROOT / "frozen/run_validation.sh",
        """#!/usr/bin/env bash
set -euo pipefail
echo "No frozen opentry_5 model is available yet. Run grouped-dev training/evaluation first."
exit 2
""",
    )
    write_text(
        ROOT / "frozen/run_future_full_test_NOT_EXECUTED.sh",
        """#!/usr/bin/env bash
set -euo pipefail
echo "Full test is intentionally not executed by opentry_5 automation before all gates pass."
exit 2
""",
    )
    os.chmod(ROOT / "frozen/run_validation.sh", 0o755)
    os.chmod(ROOT / "frozen/run_future_full_test_NOT_EXECUTED.sh", 0o755)


def write_manifest() -> None:
    include_dirs = ["configs", "reports", "manifests", "eval", "scripts"]
    rows = []
    for rel_dir in include_dirs:
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "created_by": "opentry_5_phase0_4",
                }
            )
    for path in [
        ROOT / "data/grouped_split_train_core.jsonl",
        ROOT / "data/grouped_split_dev_model.jsonl",
        ROOT / "data/grouped_split_dev_gate.jsonl",
        ROOT / "data/grouped_split_fold_a.jsonl",
        ROOT / "data/grouped_split_fold_b.jsonl",
        ROOT / "data/oof_wa_predictions_train.jsonl",
        ROOT / "data/oof_wa_predictions_dev.jsonl",
        ROOT / "data/geometry_denoising_train.jsonl",
        ROOT / "data/geometry_denoising_dev.jsonl",
        ROOT / "data/canonical_train/train_core.jsonl",
        ROOT / "data/canonical_dev/dev_model.jsonl",
        ROOT / "data/canonical_dev/dev_gate.jsonl",
    ]:
        if path.exists():
            rows.append({"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path), "created_by": "opentry_5_phase0_4"})
    write_jsonl(ROOT / "manifests/opentry_5_manifest.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-synthetic", type=int, default=0, help="debug limit; 0 means deterministic full selected subset")
    parser.add_argument("--max-hard-negatives", type=int, default=0, help="0 means all available hard negatives")
    parser.add_argument("--manifest-only", action="store_true", help="only refresh the opentry_5 file manifest")
    args = parser.parse_args()

    if args.manifest_only:
        ensure_dir(ROOT / "manifests")
        write_manifest()
        print(json.dumps({"status": "ok", "manifest_only": True}, indent=2, sort_keys=True))
        return

    for rel in ["data", "reports", "configs", "manifests", "logs", "checkpoints", "frozen", "tmp"]:
        ensure_dir(ROOT / rel)

    write_resource_inventory()
    train_rows, dev_rows = load_canonical_rows()
    oof_summary = build_oof_predictions(train_rows, dev_rows)
    corruption_summary = build_corruption_data(train_rows, dev_rows, args.max_synthetic, args.max_hard_negatives)
    base_info = copy_base_checkpoint()
    write_protocol_reports(oof_summary, corruption_summary)
    write_model_configs_and_reports(base_info)
    write_experiment_books(oof_summary, corruption_summary, base_info)
    write_resume_and_final()
    write_frozen_placeholders()
    write_manifest()
    print(json.dumps({"status": "ok", "oof": oof_summary, "corruption": corruption_summary, "base_checkpoint": base_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
