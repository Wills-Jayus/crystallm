#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from build_mp20_k50_selector_inputs import build_row
from run_mp20_k20_rerank_oof import add_derived_features, load_table, make_preprocessor, model_specs


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ID = "mp20_k50_hgb_mean_seed012_margin_route"
THRESHOLD = 0.20066793518000028
SEEDS = [0, 1, 2]


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "cif", "generated_cif", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def generation_config(rank: int) -> dict[str, Any]:
    if rank <= 20:
        return {
            "count": 20,
            "name": "anchor_k20",
            "policy": "historical CrystaLLM-a GT-SG anchor order",
            "rank_start": 1,
            "seed": 1337,
            "temperature": 0.8,
            "top_k": 10,
        }
    if rank <= 30:
        return {"count": 10, "name": "temp070_top10", "rank_start": 21, "seed": 7337, "temperature": 0.7, "top_k": 10}
    if rank <= 40:
        return {"count": 10, "name": "temp085_top10", "rank_start": 31, "seed": 8537, "temperature": 0.85, "top_k": 10}
    return {"count": 10, "name": "temp100_top10", "rank_start": 41, "seed": 10037, "temperature": 1.0, "top_k": 10}


def feature_row(record: dict[str, Any], *, fast_candidate_diagnostics: bool) -> dict[str, Any]:
    rank = int(record["rank"])
    cif = candidate_text(record)
    normalized = {
        "dataset": "mp_20",
        "split": "test",
        "sample_id": record["sample_id"],
        "material_id": record["material_id"],
        "rank": rank,
        "gen_index": rank - 1,
        "generated_text": cif,
        "generation_config": generation_config(rank),
        "logprob_available": False,
        "normalized_token_logprob": None,
    }
    row = build_row(normalized)
    diagnostics = (record.get("metadata") or {}).get("diagnostics") or {}
    readable = diagnostics.get("readable")
    if fast_candidate_diagnostics:
        row["parse_ok"] = bool(readable) if readable is not None else bool(cif.strip())
        row["sensible"] = False
        valid_cif = diagnostics.get("valid_CIF")
        row["valid"] = bool(valid_cif) if valid_cif is not None else False
    else:
        row["parse_ok"] = bool(cif.strip())
        row["sensible"] = False
        row["valid"] = False
    row["cif"] = cif
    row["candidate_id"] = record.get("candidate_id")
    return row


def failure_placeholder(sample_id: str, material_id: str, rank: int) -> str:
    return (
        "# opentry_10 explicit failure placeholder\n"
        f"# sample_id: {sample_id}\n"
        f"# material_id: {material_id}\n"
        f"# rank: {rank}\n"
        "# This intentionally invalid CIF occupies a required official-test slot and must count as failure.\n"
    )


def anchor_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    sample_by_material: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            src = json.loads(line)
            if src.get("source_rank") is not None:
                rank = int(src["source_rank"]) + 1
            else:
                rank = int(src.get("rank", 0))
            material_id = str(src.get("material_id") or (src.get("metadata") or {}).get("material_id"))
            sample_id = str(src["sample_id"])
            sample_by_material.setdefault(material_id, sample_id)
            rows.append(
                {
                    "candidate_id": src.get("candidate_id"),
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "rank": rank,
                    "cif": candidate_text(src),
                    "metadata": src.get("metadata") or {},
                    "source": "anchor_k20",
                }
            )
    return rows, sample_by_material


def supplemental_records(post_dir: Path, sample_by_material: dict[str, str], *, max_rank: int, system_id: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"^(?P<material>.+)__(?P<rank>[0-9]+)\.cif$")
    rows: list[dict[str, Any]] = []
    for path in sorted(post_dir.glob("*.cif")):
        match = pattern.match(path.name)
        if not match:
            continue
        rank = int(match.group("rank"))
        if rank < 21 or rank > int(max_rank):
            continue
        material_id = match.group("material")
        sample_id = sample_by_material.get(material_id)
        if sample_id is None:
            raise RuntimeError(f"supplemental candidate has no anchor sample_id mapping: {path.name}")
        rows.append(
            {
                "candidate_id": f"{system_id}:supplemental:{sample_id}:{rank}",
                "sample_id": sample_id,
                "material_id": material_id,
                "rank": rank,
                "cif": path.read_text(encoding="utf-8", errors="replace"),
                "metadata": {},
                "source": "supplemental_k50",
            }
        )
    return rows


def train_rankers(features: Path, labels: Path, true_tar: Path, model_out: Path) -> tuple[list[Pipeline], dict[str, Any]]:
    df, input_summary = load_table(features, labels, true_tar, max_rank=50, allow_partial=False)
    preprocessor, numeric, categorical = make_preprocessor(df)
    x = df[numeric + categorical].copy()
    y = df["match"].astype(int).to_numpy()
    pipes: list[Pipeline] = []
    model_records: list[dict[str, Any]] = []
    for spec in model_specs(["hgb"], SEEDS):
        pipe = Pipeline([("features", preprocessor), ("model", spec.estimator)])
        pipe.fit(x, y)
        pipes.append(pipe)
        model_records.append({"model": spec.name, "seed": int(spec.seed)})
    model_out = under_root(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipelines": pipes,
        "models": model_records,
        "ensemble": "mean_predict_proba",
        "numeric_features": numeric,
        "categorical_features": categorical,
        "input_summary": input_summary,
    }
    joblib.dump(payload, model_out)
    return pipes, {"numeric_features": numeric, "categorical_features": categorical, "input_summary": input_summary, "models": model_records}


def target_records_from_true_tar(true_tar: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    with tarfile.open(true_tar, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            material_id = Path(member.name).stem
            targets.append({"material_id": material_id, "sample_id": f"mp_20_test_orig__{material_id}"})
    return targets


def score_candidates(rows: list[dict[str, Any]], pipes: list[Pipeline], feature_info: dict[str, Any], *, fast_candidate_diagnostics: bool) -> pd.DataFrame:
    feature_rows = [feature_row(row, fast_candidate_diagnostics=fast_candidate_diagnostics) for row in rows]
    df = add_derived_features(pd.DataFrame(feature_rows))
    feature_cols = feature_info["numeric_features"] + feature_info["categorical_features"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = None
    scores = [pipe.predict_proba(df[feature_cols])[:, 1] for pipe in pipes]
    df["score"] = np.mean(np.vstack(scores), axis=0)
    return df


def select_outputs(
    rows: list[dict[str, Any]],
    pipes: list[Pipeline],
    feature_info: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    fast_candidate_diagnostics: bool,
    system_id: str,
    expected_k: int,
    route_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    df = score_candidates(rows, pipes, feature_info, fast_candidate_diagnostics=fast_candidate_diagnostics)
    selected_rows: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = {}
    routed_samples = 0
    baseline_samples = 0
    supplemental_slots = 0
    for sample_id, group in df.groupby("sample_id", sort=False):
        if len(group) != int(expected_k):
            raise RuntimeError(f"sample {sample_id} has {len(group)} K{expected_k} candidates, expected {expected_k}")
        rank1_score = float(group.loc[group["rank"].astype(int) == 1, "score"].iloc[0])
        best_score = float(group["score"].max())
        route_value = float(best_score - rank1_score)
        should_route = route_value >= float(route_threshold)
        if should_route:
            ordered = group.sort_values(["score", "rank"], ascending=[False, True]).head(20)
            routed_samples += 1
        else:
            ordered = group[group["rank"].astype(int) <= 20].sort_values("rank").head(20)
            baseline_samples += 1
        material_id = str(ordered["material_id"].iloc[0])
        sample_counts[str(sample_id)] = int(len(ordered))
        for new_rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            source_rank = int(row["rank"])
            supplemental_slots += int(source_rank > 20)
            selected_rows.append(
                {
                    "system": system_id,
                    "dataset": "mp_20",
                    "split": "test",
                    "sample_id": str(sample_id),
                    "material_id": material_id,
                    "rank": int(new_rank),
                    "gen_index": int(new_rank - 1),
                    "generated_text": str(row["cif"]),
                    "source_candidate_id": row.get("candidate_id"),
                    "source_rank": source_rank,
                    "hgb_mean_score": float(row["score"]),
                    "ranker": "hgb_mean_seed012",
                    "route_margin": float(best_score - rank1_score),
                    "route_value": float(route_value),
                    "route_rule": "best_minus_rank1_ge",
                    "route_threshold": float(route_threshold),
                    "route_decision": "hgb_mean_sort" if should_route else "baseline_k20",
                }
            )

    target_by_sample = {str(row["sample_id"]): row for row in targets}
    missing_samples = sorted(set(target_by_sample) - set(sample_counts))
    for sample_id in missing_samples:
        material_id = str(target_by_sample[sample_id]["material_id"])
        for new_rank in range(1, 21):
            selected_rows.append(
                {
                    "system": system_id,
                    "dataset": "mp_20",
                    "split": "test",
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "rank": int(new_rank),
                    "gen_index": int(new_rank - 1),
                    "generated_text": failure_placeholder(sample_id, material_id, new_rank),
                    "source_candidate_id": None,
                    "source_rank": None,
                    "hgb_mean_score": None,
                    "ranker": "explicit_failure_placeholder",
                    "route_decision": "placeholder_failure",
                    "placeholder_failure": True,
                }
            )
        sample_counts[sample_id] = 20

    manifest = {
        "samples": int(len(sample_counts)),
        "candidate_rows": int(len(selected_rows)),
        "samples_with_20_slots": int(sum(1 for v in sample_counts.values() if v == 20)),
        "min_slots": int(min(sample_counts.values())),
        "max_slots": int(max(sample_counts.values())),
        "routed_samples": int(routed_samples),
        "baseline_samples": int(baseline_samples),
        "supplemental_slots": int(supplemental_slots),
        "missing_candidate_samples_filled_with_placeholders": missing_samples,
        "placeholder_slots": int(20 * len(missing_samples)),
    }
    return selected_rows, manifest, df


def write_tar(path: Path, rows: list[dict[str, Any]]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for row in rows:
            name = f"{row['material_id']}__{row['rank']}.cif"
            data = row["generated_text"].encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen MP-20 K50 HGB-ensemble margin-route official-test outputs.")
    parser.add_argument("--validation-features", default=str(ROOT / "features/mp20_val_k50_candidate_features.jsonl"))
    parser.add_argument("--validation-labels", default=str(ROOT / "labels/mp20_val_k50_candidate_labels.jsonl"))
    parser.add_argument("--validation-true-tar", default=str(ROOT / "generations/crystallm_gt_sg_val_anchor_symprec0p1/mp20_val_data_atomtype_gt_sg_symprec0p1_k100/tars/true.tar.gz"))
    parser.add_argument("--anchor-k20", default=str(ROOT / "candidates/unified_mp_20_test_crystallm_a_gt_sg_anchor.jsonl"))
    parser.add_argument("--supplemental-post-dir", default=str(ROOT / "generations/mp20_test_gt_sg_k50/cifs_post/data_atomtype_gt_sg"))
    parser.add_argument("--test-true-tar", default=str(ROOT / "generations/mp20_official_test/tars/true.tar.gz"))
    parser.add_argument("--system-id", default=SYSTEM_ID)
    parser.add_argument("--max-rank", type=int, default=50)
    parser.add_argument("--route-threshold", type=float, default=THRESHOLD)
    parser.add_argument("--route-threshold-source", default="MP-20 validation OOF HGB mean seed012 best_minus_rank1_ge_q0.9")
    parser.add_argument("--fast-candidate-diagnostics", action="store_true", default=True)
    args = parser.parse_args()

    system_id = str(args.system_id)
    frozen_dir = ROOT / f"frozen_strategy/{system_id}"
    model_path = frozen_dir / "ranker_hgb_mean_seed012.joblib"
    config_path = frozen_dir / "config.json"
    manifest_path = frozen_dir / "freeze_manifest.json"
    out_jsonl = ROOT / f"generations/{system_id}_test_k20_candidates.jsonl"
    scored_jsonl = ROOT / f"generations/{system_id}_test_k{int(args.max_rank)}_scored_candidates.jsonl"
    out_tar = ROOT / f"generations/{system_id}_test_k20/tars/generated_data_atomtype_gt_sg.tar.gz"
    true_tar = Path(args.test_true_tar)

    pipes, feature_info = train_rankers(Path(args.validation_features), Path(args.validation_labels), Path(args.validation_true_tar), model_path)
    anchors, sample_by_material = anchor_records(Path(args.anchor_k20))
    supplemental = supplemental_records(Path(args.supplemental_post_dir), sample_by_material, max_rank=int(args.max_rank), system_id=system_id)
    targets = target_records_from_true_tar(true_tar)
    selected_rows, selection_manifest, scored_df = select_outputs(
        anchors + supplemental,
        pipes,
        feature_info,
        targets,
        fast_candidate_diagnostics=bool(args.fast_candidate_diagnostics),
        system_id=system_id,
        expected_k=int(args.max_rank),
        route_threshold=float(args.route_threshold),
    )
    write_jsonl(out_jsonl, selected_rows)
    write_tar(out_tar, selected_rows)
    scored_cols = [
        "sample_id",
        "material_id",
        "rank",
        "candidate_id",
        "score",
        "rank_source",
        "generation_config_name",
        "temperature",
        "top_k",
        "seed",
    ]
    write_jsonl(scored_jsonl, scored_df[[c for c in scored_cols if c in scored_df.columns]].to_dict(orient="records"))

    config = read_json(config_path) if config_path.exists() else {}
    config.update(
        {
            "id": system_id,
            "dataset": "mp_20",
            "status": "frozen_test_outputs_built_pending_official_eval",
            "created_at_utc": config.get("created_at_utc", now_iso()),
            "built_at_utc": now_iso(),
            "ranker": {
                "model_family": "hist_gradient_boosting_classifier",
                "seeds": SEEDS,
                "ensemble": "mean_predict_proba",
                "training_features": str(Path(args.validation_features).resolve()),
                "training_labels": str(Path(args.validation_labels).resolve()),
                "ranker_path": str(model_path.resolve()),
                "feature_info": feature_info,
            },
            "route": {
                "strategy": "unconstrained_score_sort_for_routed_samples_else_baseline_k20",
                "rule": "best_minus_rank1_ge",
                "threshold": float(args.route_threshold),
                "threshold_source": str(args.route_threshold_source),
                "max_rank": int(args.max_rank),
            },
            "test_outputs": {
                "selected_candidates_jsonl": str(out_jsonl.resolve()),
                "scored_k50_jsonl": str(scored_jsonl.resolve()),
                "generated_tar": str(out_tar.resolve()),
                "true_tar": str(true_tar.resolve()),
                "selection_manifest": selection_manifest,
            },
            "test_feedback_policy": "Route and threshold are frozen from validation-only OOF evidence; do not alter them using official test results.",
        }
    )
    write_json(config_path, config)
    entry = {
        "id": system_id,
        "designation": "secondary_candidate",
        "created_at": now_iso(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_path(config_path),
        "ranker_path": str(model_path.resolve()),
        "ranker_sha256": sha256_path(model_path),
        "anchor_k20": str(Path(args.anchor_k20).resolve()),
        "supplemental_post_dir": str(Path(args.supplemental_post_dir).resolve()),
        "test_candidates_jsonl": str(out_jsonl.resolve()),
        "test_candidates_jsonl_sha256": sha256_path(out_jsonl),
        "test_scored_k50_jsonl": str(scored_jsonl.resolve()),
        "test_scored_k50_jsonl_sha256": sha256_path(scored_jsonl),
        "test_generated_tar": str(out_tar.resolve()),
        "test_generated_tar_sha256": sha256_path(out_tar),
        "test_true_tar": str(true_tar.resolve()),
        "test_true_tar_sha256": sha256_path(true_tar),
        "test_manifest": selection_manifest,
        "true_manifest": {"true_tar": str(true_tar.resolve()), "target_samples": len(targets)},
        "status": "pending_official_eval",
    }
    update_frozen_registry(entry)
    write_json(manifest_path, entry)
    print(json.dumps({"system_id": system_id, "selection_manifest": selection_manifest, "generated_tar": str(out_tar)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
