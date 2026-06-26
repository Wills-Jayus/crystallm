#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import benchmark_metrics_opentry10 as bm
from build_mp20_k50_selector_inputs import build_row
from opentry8_tools import build_target_cache
from run_mp20_k20_rerank_oof import add_derived_features, load_table, make_preprocessor, model_specs


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "cif", "generated_cif", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def failure_placeholder(sample_id: str, material_id: str, rank: int) -> str:
    return (
        "# opentry_10 explicit failure placeholder\n"
        f"# sample_id: {sample_id}\n"
        f"# material_id: {material_id}\n"
        f"# rank: {rank}\n"
        "# This intentionally invalid CIF occupies a required official-test slot and must count as failure.\n"
    )


def test_feature_row(record: dict[str, Any], *, fast_candidate_diagnostics: bool) -> dict[str, Any]:
    rank0 = record.get("source_rank", record.get("rank", record.get("gen_index", 0)))
    rank = int(rank0) + 1 if int(rank0) == 0 or record.get("source_rank") is not None else int(rank0)
    cfg = {
        "count": 20,
        "name": "anchor_k20",
        "policy": "historical CrystaLLM-a GT-SG anchor order",
        "rank_start": 1,
        "seed": 1337,
        "temperature": 0.8,
        "top_k": 10,
    }
    normalized = {
        "dataset": record.get("dataset", "mpts_52"),
        "split": "test",
        "sample_id": record.get("sample_id"),
        "material_id": record.get("material_id") or (record.get("metadata") or {}).get("material_id"),
        "rank": rank,
        "gen_index": rank - 1,
        "generated_text": candidate_text(record),
        "generation_config": cfg,
        "logprob_available": False,
        "normalized_token_logprob": None,
    }
    row = build_row(normalized)
    cif = normalized["generated_text"]
    diagnostics = (record.get("metadata") or {}).get("diagnostics") or {}
    readable = diagnostics.get("readable")
    if fast_candidate_diagnostics:
        row["parse_ok"] = bool(readable) if readable is not None else bool(cif.strip())
        row["sensible"] = False
        valid_cif = diagnostics.get("valid_CIF")
        row["valid"] = bool(valid_cif) if valid_cif is not None else False
    else:
        try:
            row["sensible"] = bool(bm.is_sensible(cif, 0.5, 2.0, 50.0, 130.0))
        except Exception:
            row["sensible"] = False
        try:
            pred = bm.Structure.from_str(bm._normalize_cif_symmops_to_declared_sg(cif), fmt="cif")
            row["parse_ok"] = True
        except Exception:
            row["parse_ok"] = bool(readable) if readable is not None else False
            pred = None
        if pred is not None:
            try:
                row["valid"] = bool(bm.is_valid(pred))
            except Exception:
                row["valid"] = False
        else:
            row["valid"] = False
    row["cif"] = cif
    row["candidate_id"] = record.get("candidate_id")
    return row


def train_ranker(features: Path, labels: Path, true_tar: Path, model_out: Path) -> tuple[Pipeline, dict[str, Any]]:
    df, input_summary = load_table(features, labels, true_tar, max_rank=20, allow_partial=False)
    spec = model_specs(["hgb"], [2])[0]
    preprocessor, numeric, categorical = make_preprocessor(df)
    pipe = Pipeline([("features", preprocessor), ("model", spec.estimator)])
    x = df[numeric + categorical].copy()
    y = df["match"].astype(int).to_numpy()
    pipe.fit(x, y)
    model_out = under_root(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "model": "hgb",
            "seed": 2,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "input_summary": input_summary,
        },
        model_out,
    )
    return pipe, {"numeric_features": numeric, "categorical_features": categorical, "input_summary": input_summary}


def build_test_outputs(
    test_candidates: Path,
    pipe: Pipeline,
    feature_info: dict[str, Any],
    out_jsonl: Path,
    out_tar: Path,
    targets: list[dict[str, Any]],
    *,
    fast_candidate_diagnostics: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with test_candidates.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(test_feature_row(json.loads(line), fast_candidate_diagnostics=fast_candidate_diagnostics))
    df = add_derived_features(pd.DataFrame(rows))
    feature_cols = feature_info["numeric_features"] + feature_info["categorical_features"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = None
    scores = pipe.predict_proba(df[feature_cols])[:, 1]
    df["rerank_score"] = scores

    selected_rows: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = {}
    for sample_id, group in df.groupby("sample_id", sort=False):
        ordered = group.sort_values(["rerank_score", "rank"], ascending=[False, True]).head(20)
        if len(ordered) != 20:
            raise RuntimeError(f"sample {sample_id} has {len(ordered)} selected slots, expected 20")
        material_id = str(ordered["material_id"].iloc[0])
        sample_counts[str(sample_id)] = int(len(ordered))
        for new_rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            selected_rows.append(
                {
                    "system": "mpts52_rerank_only_hgb_seed2",
                    "dataset": "mpts_52",
                    "split": "test",
                    "sample_id": str(sample_id),
                    "material_id": material_id,
                    "rank": int(new_rank),
                    "gen_index": int(new_rank - 1),
                    "generated_text": str(row["cif"]),
                    "source_candidate_id": row.get("candidate_id"),
                    "source_rank": int(row["rank"]),
                    "rerank_score": float(row["rerank_score"]),
                    "ranker": "hgb_seed2",
                }
            )
    target_by_sample = {str(row["sample_id"]): row for row in targets}
    missing_samples = sorted(set(target_by_sample) - set(sample_counts))
    for sample_id in missing_samples:
        target = target_by_sample[sample_id]
        material_id = str(target["material_id"])
        for new_rank in range(1, 21):
            selected_rows.append(
                {
                    "system": "mpts52_rerank_only_hgb_seed2",
                    "dataset": "mpts_52",
                    "split": "test",
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "rank": int(new_rank),
                    "gen_index": int(new_rank - 1),
                    "generated_text": failure_placeholder(sample_id, material_id, new_rank),
                    "source_candidate_id": None,
                    "source_rank": None,
                    "rerank_score": None,
                    "ranker": "explicit_failure_placeholder",
                    "placeholder_failure": True,
                }
            )
        sample_counts[sample_id] = 20

    out_jsonl = under_root(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    out_tar = under_root(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for row in selected_rows:
            name = f"{row['material_id']}__{row['rank']}.cif"
            data = row["generated_text"].encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return {
        "test_candidates": str(test_candidates.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "out_tar": str(out_tar.resolve()),
        "samples": int(len(sample_counts)),
        "candidate_rows": int(len(selected_rows)),
        "samples_with_20_slots": int(sum(1 for v in sample_counts.values() if v == 20)),
        "min_slots": int(min(sample_counts.values())),
        "max_slots": int(max(sample_counts.values())),
        "missing_candidate_samples_filled_with_placeholders": missing_samples,
        "placeholder_slots": int(20 * len(missing_samples)),
    }


def build_true_tar(out_tar: Path) -> dict[str, Any]:
    targets = build_target_cache("mpts52", "test", refresh=False, fast_row_count=True)
    out_tar = under_root(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w:gz") as tar:
        for row in targets:
            data = str(row["cif"]).encode("utf-8")
            info = tarfile.TarInfo(name=f"{row['material_id']}.cif")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return {"true_tar": str(out_tar.resolve()), "target_samples": len(targets)}


def update_frozen_registry(entry: dict[str, Any]) -> None:
    registry_path = ROOT / "state/frozen_registry.json"
    if registry_path.exists():
        registry = read_json(registry_path)
    else:
        registry = {"created_at": now_iso(), "frozen_systems": []}
    systems = [s for s in registry.get("frozen_systems", []) if s.get("id") != entry["id"]]
    systems.append(entry)
    registry["frozen_systems"] = systems
    registry["updated_at"] = now_iso()
    write_json(registry_path, registry)


def make_freeze_report(entry: dict[str, Any], validation: dict[str, Any]) -> str:
    d = validation["deltas"]
    ci = validation["bootstrap"]
    return "\n".join(
        [
            "# Pre-Test Freeze Declaration",
            "",
            f"Created: {entry['created_at']}",
            "",
            "This declaration freezes the primary official-test line before any new per-sample test evaluation is read.",
            "",
            "## Primary Strategy",
            "",
            "- ID: mpts52_rerank_only_hgb_seed2",
            "- Dataset: MPTS-52",
            "- Candidate source: CrystaLLM-a GT-SG official test anchor K20",
            "- Final K budget: 20",
            "- Strategy: rerank-only; no supplemental candidates and no test target labels.",
            "- Ranker: HistGradientBoostingClassifier seed=2, trained on full MPTS-52 validation K20 candidate labels after OOF model selection.",
            "",
            "## Validation Evidence",
            "",
            f"- match@1 delta: {100*d['match@1']:.3f} pp; bootstrap CI95 [{100*ci['match@1_delta_ci95'][0]:.3f}, {100*ci['match@1_delta_ci95'][1]:.3f}] pp",
            f"- match@5 delta: {100*d['match@5']:.3f} pp; bootstrap CI95 [{100*ci['match@5_delta_ci95'][0]:.3f}, {100*ci['match@5_delta_ci95'][1]:.3f}] pp",
            f"- match@20 delta: {100*d['match@20']:.3f} pp",
            f"- rows>=7 match@1 delta: {100*d['rows>=7_match@1']:.3f} pp",
            f"- rows>=7 match@5 delta: {100*d['rows>=7_match@5']:.3f} pp",
            f"- rows>=7 match@20 delta: {100*d['rows>=7_match@20']:.3f} pp",
            "",
            "Rows>=7@5 is a validation risk, but the formal primary gain is match@1 and the corresponding rows>=7@1 change is small with CI crossing zero.",
            "",
            "## Frozen Artifacts",
            "",
            f"- Config: `{entry['config']}`",
            f"- Ranker: `{entry['ranker_path']}`",
            f"- Test candidates JSONL: `{entry['test_candidates_jsonl']}`",
            f"- Test generated tar: `{entry['test_generated_tar']}`",
            f"- Test true tar prepared for post-freeze evaluation: `{entry['test_true_tar']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and build MPTS-52 rerank-only HGB seed2 official-test candidates.")
    parser.add_argument("--validation-features", default=str(ROOT / "features/mpts52_val_k50_candidate_features.jsonl"))
    parser.add_argument("--validation-labels", default=str(ROOT / "labels/mpts52_val_k50_candidate_labels.jsonl"))
    parser.add_argument("--validation-true-tar", default=str(ROOT / "generations/crystallm_gt_sg_val_anchor_symprec0p1/mpts52_val_data_atomtype_gt_sg_symprec0p1_k100/tars/true.tar.gz"))
    parser.add_argument("--test-candidates", default=str(ROOT / "candidates/unified_mpts_52_test_crystallm_a_gt_sg_anchor.jsonl"))
    parser.add_argument("--bootstrap-metrics", default=str(ROOT / "metrics/mpts52_rerank_hgb_seed2_bootstrap.json"))
    parser.add_argument("--fast-candidate-diagnostics", action="store_true", default=True)
    args = parser.parse_args()

    frozen_dir = ROOT / "frozen_strategy/mpts52_rerank_only_hgb_seed2"
    model_path = frozen_dir / "ranker_hgb_seed2.joblib"
    config_path = frozen_dir / "config.json"
    out_jsonl = ROOT / "generations/mpts52_rerank_only_hgb_seed2_test_k20_candidates.jsonl"
    out_tar = ROOT / "generations/mpts52_rerank_only_hgb_seed2_test_k20/tars/generated_data_atomtype_gt_sg.tar.gz"
    true_tar = ROOT / "generations/mpts52_official_test/tars/true.tar.gz"

    pipe, feature_info = train_ranker(Path(args.validation_features), Path(args.validation_labels), Path(args.validation_true_tar), model_path)
    targets = build_target_cache("mpts52", "test", refresh=False, fast_row_count=True)
    test_manifest = build_test_outputs(
        Path(args.test_candidates),
        pipe,
        feature_info,
        out_jsonl,
        out_tar,
        targets,
        fast_candidate_diagnostics=bool(args.fast_candidate_diagnostics),
    )
    true_manifest = build_true_tar(true_tar)
    validation = read_json(Path(args.bootstrap_metrics))
    config = {
        "id": "mpts52_rerank_only_hgb_seed2",
        "created_at": now_iso(),
        "dataset": "mpts_52",
        "strategy": "rerank_only",
        "ranker": "HistGradientBoostingClassifier",
        "seed": 2,
        "candidate_source": str(Path(args.test_candidates).resolve()),
        "source_quota": {"anchor_k20": 20, "supplemental": 0},
        "selector_threshold": None,
        "k_budget": 20,
        "validation_metrics": validation,
        "fast_candidate_diagnostics": bool(args.fast_candidate_diagnostics),
        "feature_info": feature_info,
    }
    write_json(config_path, config)
    entry = {
        "id": "mpts52_rerank_only_hgb_seed2",
        "designation": "primary",
        "created_at": now_iso(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_path(config_path),
        "ranker_path": str(model_path.resolve()),
        "ranker_sha256": sha256_path(model_path),
        "candidate_source": str(Path(args.test_candidates).resolve()),
        "candidate_source_sha256": sha256_path(Path(args.test_candidates)),
        "test_candidates_jsonl": test_manifest["out_jsonl"],
        "test_candidates_jsonl_sha256": sha256_path(Path(test_manifest["out_jsonl"])),
        "test_generated_tar": test_manifest["out_tar"],
        "test_generated_tar_sha256": sha256_path(Path(test_manifest["out_tar"])),
        "test_true_tar": true_manifest["true_tar"],
        "test_true_tar_sha256": sha256_path(Path(true_manifest["true_tar"])),
        "test_manifest": test_manifest,
        "true_manifest": true_manifest,
        "validation_metrics": validation,
    }
    update_frozen_registry(entry)
    write_json(frozen_dir / "freeze_manifest.json", entry)
    write_text(ROOT / "reports/pre_test_freeze_declaration.md", make_freeze_report(entry, validation))


if __name__ == "__main__":
    main()
