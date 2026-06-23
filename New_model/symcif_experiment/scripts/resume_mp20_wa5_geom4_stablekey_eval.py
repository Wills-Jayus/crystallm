#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import _sample_eval_process, timeout_metrics_for_sample  # noqa: E402
from run_symcif_v4_full_pipeline_eval import (  # noqa: E402
    case_payload,
    failed_cases,
    read_jsonl,
    record_skeleton_multiset_key,
    record_wa_multiset_key,
    write_jsonl,
)
from run_symcif_v4_geometry_model_eval import (  # noqa: E402
    benchmark_subset_records,
    enrich_metrics,
    make_summary,
    timeout_attempt_summary,
    timeout_breakdown_rows,
    write_csv,
)


TOP_KS = (1, 5, 20)
FULL_TEST_DENOMINATOR = 9046
STRUCTURED_TEST_DENOMINATOR = 8893
OLD_WA20_GEOM1 = {
    "match20": 0.6891937478916002,
    "rmse20": 0.08478709616005559,
    "readable20": 0.4879961767682447,
    "strict_valid20": 0.18494321376363432,
    "strict_valid_any20": 0.8097379961767682,
}
PUBLISHED_CRYSTALLM_A_MP20 = {
    "match1": 0.5585,
    "match20": 0.7514,
    "rmse1": 0.0437,
    "rmse20": 0.0395,
}


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_artifact_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT.parents[2] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_meta_by_sample(top20_path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in read_jsonl_iter(top20_path):
        out[int(row["sample_index"])] = row
    return out


def load_generation_groups(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_iter(path):
        grouped[int(row["sample_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    return dict(grouped)


def sample_metric_shard(metrics_shard_dir: Path, sample_index: int) -> Path:
    return metrics_shard_dir / f"sample_{sample_index:05d}.jsonl"


def load_completed_sample_indexes(metrics_shard_dir: Path, top_k: int) -> set[int]:
    done: set[int] = set()
    if not metrics_shard_dir.exists():
        return done
    for path in metrics_shard_dir.glob("sample_*.jsonl"):
        try:
            rows = list(read_jsonl_iter(path))
            if len(rows) >= top_k:
                sample_indexes = {int(row["sample_index"]) for row in rows}
                if len(sample_indexes) == 1:
                    done.add(next(iter(sample_indexes)))
        except Exception:
            continue
    return done


def write_metric_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, rows)
    os.replace(tmp, path)


def load_all_metric_shards(metrics_shard_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_shard_dir.glob("sample_*.jsonl")):
        rows.extend(read_jsonl_iter(path))
    rows.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    return rows


def prediction_meta_by_key(meta_by_sample: dict[int, dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for sample_index, row in meta_by_sample.items():
        sample_id = str(row.get("sample_id"))
        for pred in row.get("predictions", []):
            rank = int(pred.get("rank", 0) or 0)
            if rank <= 0:
                continue
            item = dict(pred)
            item["sample_index"] = int(sample_index)
            item["sample_id"] = sample_id
            item["gen_index"] = rank - 1
            out[(int(sample_index), rank - 1)] = item
    return out


def source_bucket(value: Any) -> str:
    text = str(value or "")
    if not text or text == "None" or "missing_candidate" in text:
        return "missing_candidate"
    if "same_wa" in text:
        return "same_wa"
    if "same_skeleton" in text:
        return "same_skeleton"
    if "same_sg" in text:
        return "same_sg"
    if "model_fallback" in text or "global" in text:
        return "model_fallback"
    return text


def metric_failure_category(metric: dict[str, Any], *, duplicate: bool = False) -> str:
    error = str(metric.get("error") or "")
    if metric.get("match_ok"):
        return "matched"
    if (not metric.get("render_success")) or error == "missing_candidate":
        return "missing_candidate"
    if duplicate:
        return "duplicate_candidate"
    if metric.get("eval_timeout") or metric.get("parse_timeout") or metric.get("sg_timeout") or metric.get("matcher_timeout"):
        return "timeout"
    if (not metric.get("pymatgen_readable")) and ("Invalid cif file" in error or "no structures" in error):
        return "invalid_cif"
    if not metric.get("parse_success"):
        return "parse_fail"
    if not metric.get("pymatgen_readable"):
        return "invalid_cif"
    if not metric.get("formula_ok"):
        return "formula_mismatch"
    if not metric.get("atom_count_ok"):
        return "atom_count_mismatch"
    if not metric.get("space_group_ok"):
        return "SG_mismatch"
    return "matcher_no_match"


def sample_primary_failure(rows: list[dict[str, Any]]) -> str:
    if any(row.get("match_ok") for row in rows):
        return "matched"
    if not rows:
        return "missing_candidate"
    categories = [metric_failure_category(row) for row in rows]
    for name in (
        "missing_candidate",
        "timeout",
        "invalid_cif",
        "parse_fail",
        "formula_mismatch",
        "atom_count_mismatch",
        "SG_mismatch",
        "duplicate_candidate",
        "matcher_no_match",
    ):
        if name in categories:
            return name
    return "matcher_no_match"


def duplicate_flags(rows: list[dict[str, Any]]) -> dict[tuple[int, int], bool]:
    by_sample: dict[int, dict[str, int]] = defaultdict(dict)
    flags: dict[tuple[int, int], bool] = {}
    for row in sorted(rows, key=lambda r: (int(r["sample_index"]), int(r["gen_index"]))):
        sample_index = int(row["sample_index"])
        gen_index = int(row["gen_index"])
        sha1 = str(row.get("generated_sha1") or "")
        if not sha1:
            flags[(sample_index, gen_index)] = False
            continue
        seen = by_sample[sample_index]
        flags[(sample_index, gen_index)] = sha1 in seen
        seen.setdefault(sha1, gen_index)
    return flags


def summarize_metrics(
    records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    sample_indexes: set[int],
    k: int,
    denominator_samples: int | None = None,
) -> dict[str, Any]:
    denominator_samples = int(denominator_samples or len(sample_indexes))
    subset = [m for m in metrics if int(m["sample_index"]) in sample_indexes and int(m["gen_index"]) < int(k)]
    attempts = denominator_samples * int(k)
    out: dict[str, Any] = {
        "samples": int(denominator_samples),
        "structured_samples": len(sample_indexes),
        "k": int(k),
        "num_attempts": attempts,
    }
    fields = {
        "render_success": "render_success",
        "readable": "pymatgen_readable",
        "formula_ok": "formula_ok",
        "atom_count_ok": "atom_count_ok",
        "SG_ok": "space_group_ok",
        "valid": "valid",
        "strict_valid": "strict_valid",
        "eval_timeout": "eval_timeout",
        "parse_timeout": "parse_timeout",
        "sg_timeout": "sg_timeout",
        "matcher_timeout": "matcher_timeout",
    }
    for out_name, field in fields.items():
        out[out_name] = sum(1 for row in subset if row.get(field)) / attempts if attempts else math.nan
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        by_sample[int(row["sample_index"])].append(row)
    match_count = 0
    best_rms: list[float] = []
    any_counts = {name: 0 for name in fields}
    wa_hit = 0
    skeleton_hit = 0
    for sample_index in sample_indexes:
        rows = sorted(by_sample.get(sample_index, []), key=lambda r: int(r["gen_index"]))
        if any(row.get("match_ok") for row in rows):
            match_count += 1
        rms_values = [float(row["rms"]) for row in rows if row.get("match_ok") and row.get("rms") is not None]
        if rms_values:
            best_rms.append(min(rms_values))
        for out_name, field in fields.items():
            if any(row.get(field) for row in rows):
                any_counts[out_name] += 1
        record = records[sample_index]
        gt_wa = record_wa_multiset_key(record)
        gt_skeleton = record_skeleton_multiset_key(record)
        if any(str(row.get("wa_multiset_key")) == gt_wa for row in rows):
            wa_hit += 1
        if any(str(row.get("skeleton_multiset_key")) == gt_skeleton for row in rows):
            skeleton_hit += 1
    out["match_at_k"] = match_count / max(1, denominator_samples)
    out["RMSE"] = float(statistics.mean(best_rms)) if best_rms else math.nan
    out["matched_samples_for_RMSE"] = len(best_rms)
    for name, count in any_counts.items():
        out[f"{name}_any_at_k"] = count / max(1, denominator_samples)
    out["wa_hit_at_k"] = wa_hit / max(1, denominator_samples)
    out["skeleton_hit_at_k"] = skeleton_hit / max(1, denominator_samples)
    return out


def subset_summary(records: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"structured_only": {}, "full_test": {}}
    overall = set(range(len(records)))
    for k in TOP_KS:
        out["structured_only"][f"top{k}"] = summarize_metrics(records, metrics, overall, k, len(records))
        out["full_test"][f"top{k}"] = summarize_metrics(records, metrics, overall, k, FULL_TEST_DENOMINATOR)
        out["full_test"][f"top{k}"]["structured_failures_counted_as_failure"] = FULL_TEST_DENOMINATOR - len(records)
    return out


def breakdown_summary(records: list[dict[str, Any]], metrics: list[dict[str, Any]], k: int = 20) -> dict[str, Any]:
    subset_names = [
        "overall",
        "n_sites>=6",
        "n_sites>=12",
        "n_sites>=20",
        "num_elements>=4",
        "rare_sg",
        "high_multiplicity_orbit",
        "extraction_hard",
    ]
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        if int(row.get("gen_index", 0)) < k:
            by_sample[int(row["sample_index"])].append(row)
    dup_flags = duplicate_flags(metrics)
    candidate_by_subset: dict[str, Any] = {}
    sample_by_subset: dict[str, Any] = {}
    sample_diag_by_subset: dict[str, Any] = {}
    geometry_by_subset: dict[str, Any] = {}
    for subset in subset_names:
        indexes = benchmark_subset_records(records, subset)
        candidate_counts: Counter[str] = Counter()
        geometry_counts: Counter[str] = Counter()
        sample_counts: Counter[str] = Counter()
        sample_diag: Counter[str] = Counter()
        for sample_index in indexes:
            rows = sorted(by_sample.get(sample_index, []), key=lambda r: int(r.get("gen_index", 0)))
            record = records[sample_index]
            gt_wa = record_wa_multiset_key(record)
            gt_skeleton = record_skeleton_multiset_key(record)
            matched = any(row.get("match_ok") for row in rows)
            wa_hit = any(str(row.get("wa_multiset_key")) == gt_wa for row in rows)
            skeleton_hit = any(str(row.get("skeleton_multiset_key")) == gt_skeleton for row in rows)
            sample_counts[sample_primary_failure(rows)] += 1
            sample_diag["matched" if matched else "failed"] += 1
            if matched and wa_hit:
                sample_diag["matched_with_wa_hit"] += 1
            if matched and not wa_hit:
                sample_diag["matched_without_wa_hit"] += 1
            if (not matched) and wa_hit:
                sample_diag["failed_with_wa_hit"] += 1
            if (not matched) and not wa_hit:
                sample_diag["failed_without_wa_hit"] += 1
            if (not matched) and skeleton_hit:
                sample_diag["failed_with_skeleton_hit"] += 1
            for row in rows:
                key = (int(row["sample_index"]), int(row["gen_index"]))
                candidate_counts[metric_failure_category(row, duplicate=dup_flags.get(key, False))] += 1
                geometry_counts[source_bucket(row.get("geometry_source"))] += 1
        candidate_by_subset[subset] = dict(candidate_counts)
        sample_by_subset[subset] = dict(sample_counts)
        sample_diag_by_subset[subset] = dict(sample_diag)
        geometry_by_subset[subset] = dict(geometry_counts)
    return {
        "k": k,
        "candidate_failure_counts": candidate_by_subset,
        "sample_primary_failure_counts": sample_by_subset,
        "sample_diagnostics": sample_diag_by_subset,
        "geometry_source_counts": geometry_by_subset,
    }


def audit_artifacts(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = args.out_dir
    top20_path = out_dir / "top20_predictions.jsonl"
    baseline_path = out_dir / "generations" / "baseline.jsonl"
    selected_path = out_dir / "selected_top20.jsonl"
    generated_dir = out_dir / "generated_cifs"
    metrics_paths = {
        "metrics_jsonl": out_dir / "metrics.jsonl",
        "baseline_per_generation_metrics": out_dir / "metrics" / "baseline_per_generation_metrics.jsonl",
    }
    top20_rows = 0
    samples_with_20_predictions = 0
    samples_without_20_predictions: list[dict[str, Any]] = []
    missing_cifs: list[dict[str, Any]] = []
    empty_cifs: list[dict[str, Any]] = []
    missing_candidate_placeholders: list[dict[str, Any]] = []
    render_failure_no_cif: list[dict[str, Any]] = []
    duplicate_cifs: list[dict[str, Any]] = []
    render_success = 0
    selected_candidates = 0
    samples_with_20_rendered_candidates = 0
    sample_ids_in_top20: set[str] = set()
    if top20_path.exists():
        for row in read_jsonl_iter(top20_path):
            top20_rows += 1
            sample_id = str(row.get("sample_id"))
            sample_ids_in_top20.add(sample_id)
            preds = list(row.get("predictions") or [])
            selected_candidates += len(preds)
            if len(preds) == args.top_k:
                samples_with_20_predictions += 1
            else:
                samples_without_20_predictions.append(
                    {"sample_index": row.get("sample_index"), "sample_id": sample_id, "prediction_count": len(preds)}
                )
            if sum(1 for pred in preds if pred.get("render_success")) == args.top_k:
                samples_with_20_rendered_candidates += 1
            seen_sha: dict[str, int] = {}
            for pred in preds:
                if pred.get("render_success"):
                    render_success += 1
                cif_path_raw = pred.get("cif_path")
                if not cif_path_raw:
                    if not pred.get("render_success"):
                        item = {"sample_id": sample_id, "rank": pred.get("rank"), "error": pred.get("error")}
                        if str(pred.get("error") or "") == "missing_candidate":
                            missing_candidate_placeholders.append(item)
                        else:
                            render_failure_no_cif.append(item)
                        continue
                    missing_cifs.append({"sample_id": sample_id, "rank": pred.get("rank"), "reason": "missing_path"})
                    continue
                cif_path = resolve_artifact_path(cif_path_raw)
                if cif_path is None:
                    missing_cifs.append({"sample_id": sample_id, "rank": pred.get("rank"), "reason": "missing_path"})
                    continue
                if not cif_path.exists():
                    missing_cifs.append({"sample_id": sample_id, "rank": pred.get("rank"), "path": str(cif_path)})
                    continue
                try:
                    data = cif_path.read_bytes()
                except Exception as exc:
                    missing_cifs.append({"sample_id": sample_id, "rank": pred.get("rank"), "path": str(cif_path), "error": str(exc)})
                    continue
                if len(data) == 0 or not data.strip():
                    empty_cifs.append({"sample_id": sample_id, "rank": pred.get("rank"), "path": str(cif_path)})
                sha = hashlib.sha1(data).hexdigest()
                if sha in seen_sha:
                    duplicate_cifs.append(
                        {
                            "sample_id": sample_id,
                            "rank": pred.get("rank"),
                            "duplicates_rank": seen_sha[sha],
                            "path": str(cif_path),
                        }
                    )
                else:
                    seen_sha[sha] = int(pred.get("rank") or 0)
    generated_dirs = sum(1 for path in generated_dir.iterdir() if path.is_dir()) if generated_dir.exists() else 0
    metric_counts = {name: line_count(path) for name, path in metrics_paths.items()}
    shard_dir = out_dir / "metrics_shards"
    completed_shards = len(load_completed_sample_indexes(shard_dir, args.top_k))
    expected_sample_ids = {str(row["sample_id"]) for row in records}
    missing_top20_samples = sorted(expected_sample_ids - sample_ids_in_top20)
    audit = {
        "out_dir": str(out_dir),
        "data_root": str(args.data_root),
        "expected_structured_samples": len(records),
        "expected_full_test_samples": FULL_TEST_DENOMINATOR,
        "generated_cifs_dirs": generated_dirs,
        "baseline_jsonl_lines": line_count(baseline_path),
        "top20_predictions_lines": top20_rows,
        "selected_top20_lines": line_count(selected_path),
        "selected_candidates": selected_candidates,
        "render_success_predictions": render_success,
        "samples_with_20_selected_candidates": samples_with_20_predictions,
        "samples_with_20_rendered_candidates": samples_with_20_rendered_candidates,
        "samples_without_20_selected_candidates_count": len(samples_without_20_predictions),
        "samples_without_20_selected_candidates_examples": samples_without_20_predictions[:20],
        "missing_candidate_placeholders_count": len(missing_candidate_placeholders),
        "missing_candidate_placeholders_examples": missing_candidate_placeholders[:20],
        "render_failure_no_cif_count": len(render_failure_no_cif),
        "render_failure_no_cif_examples": render_failure_no_cif[:20],
        "missing_top20_samples_count": len(missing_top20_samples),
        "missing_top20_samples_examples": missing_top20_samples[:20],
        "missing_cifs_count": len(missing_cifs),
        "missing_cifs_examples": missing_cifs[:20],
        "empty_cifs_count": len(empty_cifs),
        "empty_cifs_examples": empty_cifs[:20],
        "duplicate_cifs_count": len(duplicate_cifs),
        "duplicate_cifs_examples": duplicate_cifs[:20],
        "metric_line_counts": metric_counts,
        "completed_metric_shards": completed_shards,
        "can_resume_eval": (
            generated_dirs == len(records)
            and line_count(baseline_path) == len(records) * args.top_k
            and top20_rows == len(records)
            and len(missing_cifs) == 0
            and len(empty_cifs) == 0
            and len(missing_top20_samples) == 0
        ),
    }
    json_dump(out_dir / "artifact_audit.json", audit)
    lines = [
        "# Artifact Audit",
        "",
        f"- output dir: `{out_dir}`",
        f"- generated_cifs sample dirs: {generated_dirs}",
        f"- baseline.jsonl lines: {audit['baseline_jsonl_lines']}",
        f"- top20_predictions.jsonl lines: {top20_rows}",
        f"- selected_top20.jsonl lines: {audit['selected_top20_lines']}",
        f"- samples with 20 selected candidates: {samples_with_20_predictions}/{len(records)}",
        f"- samples with 20 rendered CIF candidates: {samples_with_20_rendered_candidates}/{len(records)}",
        f"- missing_candidate placeholders: {len(missing_candidate_placeholders)}",
        f"- render failures without CIF: {len(render_failure_no_cif)}",
        f"- missing CIFs: {len(missing_cifs)}",
        f"- empty CIFs: {len(empty_cifs)}",
        f"- duplicate CIFs within sample top20: {len(duplicate_cifs)}",
        f"- existing metric lines: {metric_counts}",
        f"- completed per-sample metric shards: {completed_shards}/{len(records)}",
        f"- can resume eval directly: {audit['can_resume_eval']}",
        "",
        "Resume decision: reuse existing render artifacts; evaluate only missing metric shards.",
    ]
    (out_dir / "artifact_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def run_resume_eval(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    grouped: dict[int, list[dict[str, Any]]],
) -> None:
    shard_dir = args.out_dir / "metrics_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = load_completed_sample_indexes(shard_dir, args.top_k)
    payload = case_payload(records)
    pending = [case for case in payload if int(case["index"]) not in done]
    if not pending:
        return
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    active: list[dict[str, Any]] = []
    sample_timeout = float(args.sample_timeout_seconds or 0)

    def start_case(case: dict[str, Any]) -> None:
        rows = grouped.get(int(case["index"]), [])
        out_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_sample_eval_process,
            args=(
                out_queue,
                "baseline",
                case,
                rows,
                str(args.lookup_json),
                args.bond_timeout_seconds,
                args.valid_timeout_seconds,
                args.match_timeout_seconds,
                args.max_match_sites,
                args.max_eval_sites,
                args.parse_timeout_seconds,
                args.sg_timeout_seconds,
            ),
        )
        proc.start()
        active.append({"case": case, "rows": rows, "queue": out_queue, "proc": proc, "started": time.monotonic()})

    with tqdm(total=len(pending), desc="evaluating missing samples") as pbar:
        while pending or active:
            while pending and len(active) < args.eval_workers:
                start_case(pending.pop(0))
            now = time.monotonic()
            still_active: list[dict[str, Any]] = []
            for task in active:
                proc = task["proc"]
                timed_out = bool(sample_timeout > 0 and (now - float(task["started"])) > sample_timeout)
                if timed_out and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                    reason = f"sample_eval_timeout>{sample_timeout:g}s"
                    metrics = timeout_metrics_for_sample("baseline", task["case"], task["rows"], reason)
                    task["queue"].close()
                    write_metric_shard(sample_metric_shard(shard_dir, int(task["case"]["index"])), metrics)
                    pbar.update(1)
                    continue
                if proc.is_alive():
                    still_active.append(task)
                    continue
                proc.join()
                result = None
                try:
                    result = task["queue"].get_nowait()
                except Exception:
                    pass
                task["queue"].close()
                if result and result.get("ok"):
                    metrics = result["metrics"]
                else:
                    reason = result.get("error") if isinstance(result, dict) else f"sample_eval_failed_exitcode={proc.exitcode}"
                    metrics = timeout_metrics_for_sample("baseline", task["case"], task["rows"], str(reason))
                write_metric_shard(sample_metric_shard(shard_dir, int(task["case"]["index"])), metrics)
                pbar.update(1)
            active = still_active
            if active:
                time.sleep(0.05)


def write_metric_outputs(args: argparse.Namespace, records: list[dict[str, Any]], meta_by_sample: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = load_all_metric_shards(args.out_dir / "metrics_shards")
    metrics = enrich_metrics(metrics, prediction_meta_by_key(meta_by_sample))
    metrics.sort(key=lambda row: (int(row["sample_index"]), int(row["gen_index"])))
    write_jsonl(args.out_dir / "metrics.jsonl", metrics)
    write_jsonl(args.out_dir / "metrics" / "baseline_per_generation_metrics.jsonl", metrics)
    failed = failed_cases(records, metrics, args.top_k)
    write_jsonl(args.out_dir / "failed_cases.jsonl", failed)
    write_jsonl(args.out_dir / "failed_eval_cases.jsonl", failed)
    return metrics


def write_structured_outputs(args: argparse.Namespace, records: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary_stub_args = SimpleNamespace(
        data_root=args.data_root,
        predictions=args.out_dir / "top20_predictions.jsonl",
        top_k=args.top_k,
        full_wa_candidates=5,
        full_max_variants_per_wa=4,
        full_selection_mode="round_robin",
        bond_timeout_seconds=args.bond_timeout_seconds,
        parse_timeout_seconds=args.parse_timeout_seconds,
        sg_timeout_seconds=args.sg_timeout_seconds,
        valid_timeout_seconds=args.valid_timeout_seconds,
        match_timeout_seconds=args.match_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        max_match_sites=args.max_match_sites,
        max_eval_sites=args.max_eval_sites,
        eval_workers=args.eval_workers,
        include_neural=True,
        include_prototypes=True,
    )
    model_stub = SimpleNamespace(checkpoint=str(args.checkpoint), training_config={})
    base_summary = make_summary(mode="full", records=records, metrics=metrics, out_dir=args.out_dir, args=summary_stub_args, model=model_stub)
    score_summary = subset_summary(records, metrics)
    breakdown = breakdown_summary(records, metrics, k=args.top_k)
    timeout_rows: list[dict[str, Any]] = []
    for k in TOP_KS:
        timeout_rows.extend(timeout_breakdown_rows(records, metrics, k))
    write_csv(args.out_dir / "timeout_breakdown.csv", timeout_rows)
    timeout_summary = [timeout_attempt_summary(records, metrics, k) for k in TOP_KS]
    base_summary["score_denominators"] = {
        "structured_only": len(records),
        "full_test": FULL_TEST_DENOMINATOR,
        "structured_failures_counted_as_failure": FULL_TEST_DENOMINATOR - len(records),
        "structured_success_rate": len(records) / FULL_TEST_DENOMINATOR,
    }
    base_summary["structured_only"] = score_summary["structured_only"]
    base_summary["full_test"] = score_summary["full_test"]
    base_summary["error_breakdown"] = breakdown
    base_summary["throughput_timeouts"] = timeout_summary
    base_summary["artifacts"]["metrics_jsonl_flat"] = str(args.out_dir / "metrics.jsonl")
    base_summary["artifacts"]["failed_eval_cases"] = str(args.out_dir / "failed_eval_cases.jsonl")
    json_dump(args.out_dir / "full_eval_summary.json", base_summary)
    json_dump(args.out_dir / "eval_summary.json", base_summary)
    json_dump(args.out_dir / "structured_full_score_summary.json", score_summary)
    json_dump(args.out_dir / "error_breakdown_summary.json", breakdown)
    return base_summary


def pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{100.0 * float(value):.2f}%"


def num(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{float(value):.4f}"


def table_for(summary: dict[str, Any], key: str) -> list[str]:
    lines = [
        "| metric | @1 | @5 | @20 |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = summary[key]
    metric_names = [
        ("match", "match_at_k", pct),
        ("RMSE", "RMSE", num),
        ("readable", "readable", pct),
        ("formula_ok", "formula_ok", pct),
        ("atom_count_ok", "atom_count_ok", pct),
        ("SG_ok", "SG_ok", pct),
        ("valid", "valid", pct),
        ("strict_valid", "strict_valid", pct),
        ("eval_timeout", "eval_timeout", pct),
        ("render_success", "render_success", pct),
        ("WA_hit", "wa_hit_at_k", pct),
        ("skeleton_hit", "skeleton_hit_at_k", pct),
    ]
    for label, field, formatter in metric_names:
        vals = [formatter(rows[f"top{k}"].get(field)) for k in TOP_KS]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    return lines


def write_final_report(args: argparse.Namespace, audit: dict[str, Any], summary: dict[str, Any]) -> None:
    structured = summary["structured_only"]
    full = summary["full_test"]
    breakdown = summary["error_breakdown"]
    subset_rows = []
    for subset in [
        "overall",
        "n_sites>=6",
        "n_sites>=12",
        "n_sites>=20",
        "num_elements>=4",
        "rare_sg",
        "high_multiplicity_orbit",
        "extraction_hard",
    ]:
        sample_fail = breakdown["sample_primary_failure_counts"].get(subset, {})
        geom = breakdown["geometry_source_counts"].get(subset, {})
        subset_rows.append(
            f"| {subset} | {sum(sample_fail.values())} | "
            f"{sample_fail.get('matched', 0)} | {sample_fail.get('matcher_no_match', 0)} | "
            f"{sample_fail.get('formula_mismatch', 0)} | {sample_fail.get('SG_mismatch', 0)} | "
            f"{sample_fail.get('timeout', 0)} | {geom.get('same_wa', 0)} | {geom.get('same_skeleton', 0)} | "
            f"{geom.get('same_sg', 0)} | {geom.get('model_fallback', 0)} |"
        )
    s20 = structured["top20"]
    f20 = full["top20"]
    overall_diag = breakdown.get("sample_diagnostics", {}).get("overall", {})
    rmse_gets_worse = (
        not math.isnan(float(structured["top20"]["RMSE"]))
        and not math.isnan(float(structured["top1"]["RMSE"]))
        and structured["top20"]["RMSE"] > structured["top1"]["RMSE"]
    )
    if f20["match_at_k"] >= PUBLISHED_CRYSTALLM_A_MP20["match20"]:
        next_step = "full-test match@20 已达到 published CrystaLLM a 的 75.14%，这是强 positive signal；下一步只需要在 MP-20 上跑另一个确认配置 WA20x1 或 WA10x2。"
    elif s20["match_at_k"] >= PUBLISHED_CRYSTALLM_A_MP20["match20"]:
        next_step = "structured-only match@20 已达到 published CrystaLLM a 的 75.14%，但 full-test 仍要看结构化失败折算后的分数和 RMSE。"
    elif f20["match_at_k"] >= 0.72:
        next_step = "match@20 处在 72%-75% 区间，路线接近成功；下一步只跑 MP-20 WA10x2 或 WA15x2，不碰 MPTS-52。"
    elif f20["match_at_k"] < 0.70:
        next_step = "full-test match@20 低于 70%，不要跑新 config；先做 failure audit，重点查 geometry_source、strict_valid、formula_ok、SG_ok 和复杂子集。"
    else:
        next_step = "full-test match@20 超过 70% 但未到 72%，优先完成 failure audit，再决定是否跑 MP-20 WA10x2。"
    if f20["RMSE"] > PUBLISHED_CRYSTALLM_A_MP20["rmse20"]:
        next_step += " RMSE@20 明显高于 CrystaLLM a，说明即使 match 接近，结构质量仍有差距；下一步优先 geometry scoring/rerank，而不是继续加 WA candidates。"

    lines = [
        "# MP-20 wa5_geom4_stablekey_hybrid Final Summary",
        "",
        "## Scope",
        "",
        "本轮只跑 MP-20 / `wa5_geom4_stablekey_hybrid`，因为旧 Table 3 是 lower-bound，stablekey/free-param/render 截断修复后最缺的是这一个配置的 full pipeline match/RMSE。按任务约束，本轮没有跑 MPTS-52、没有跑 CrystaLLM+GT-SG baseline、没有训练新模型，也没有重 render。",
        "",
        "## Artifact Audit",
        "",
        f"- generated_cifs dirs: {audit['generated_cifs_dirs']}",
        f"- baseline.jsonl lines: {audit['baseline_jsonl_lines']}",
        f"- top20_predictions lines: {audit['top20_predictions_lines']}",
        f"- samples with 20 selected candidates: {audit['samples_with_20_selected_candidates']}/{audit['expected_structured_samples']}",
        f"- samples with 20 rendered CIF candidates: {audit['samples_with_20_rendered_candidates']}/{audit['expected_structured_samples']}",
        f"- missing_candidate placeholders: {audit['missing_candidate_placeholders_count']}; render failures without CIF: {audit['render_failure_no_cif_count']}",
        f"- missing CIF artifacts: {audit['missing_cifs_count']}; empty CIFs: {audit['empty_cifs_count']}; duplicate CIFs: {audit['duplicate_cifs_count']}",
        f"- reused existing render: yes; direct resume possible: {audit['can_resume_eval']}",
        "",
        "## Structured-Only Scores",
        "",
        *table_for(summary, "structured_only"),
        "",
        "Denominator = 8,893 structured MP-20 test samples.",
        "",
        "## Full-Test Scores",
        "",
        *table_for(summary, "full_test"),
        "",
        "Denominator = 9,046 original MP-20 test samples. The 153 non-structured samples are counted as failures for rate metrics; RMSE is still averaged over matched samples with an available RMS.",
        "",
        "## Comparisons",
        "",
        "| reference | match@1 | match@20 | RMSE@1 | RMSE@20 | note |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| old SymCIF WA20xgeom1 lower-bound | - | {pct(OLD_WA20_GEOM1['match20'])} | - | {num(OLD_WA20_GEOM1['rmse20'])} | old Table 3 provisional |",
        f"| this run structured-only | {pct(structured['top1']['match_at_k'])} | {pct(s20['match_at_k'])} | {num(structured['top1']['RMSE'])} | {num(s20['RMSE'])} | MP-20 structured subset |",
        f"| this run full-test | {pct(full['top1']['match_at_k'])} | {pct(f20['match_at_k'])} | {num(full['top1']['RMSE'])} | {num(f20['RMSE'])} | includes 153 structured failures |",
        f"| published CrystaLLM a MP-20 | {pct(PUBLISHED_CRYSTALLM_A_MP20['match1'])} | {pct(PUBLISHED_CRYSTALLM_A_MP20['match20'])} | {num(PUBLISHED_CRYSTALLM_A_MP20['rmse1'])} | {num(PUBLISHED_CRYSTALLM_A_MP20['rmse20'])} | composition-only |",
        "",
        "Important: SymCIF here uses composition + oracle GT space group. Published CrystaLLM a is composition-only, so this is not a strictly fair same-input comparison.",
        "",
        "## Error Breakdown",
        "",
        "| subset | samples | matched | matcher_no_match | formula_mismatch | SG_mismatch | timeout | same_wa attempts | same_skeleton attempts | same_sg attempts | model_fallback attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *subset_rows,
        "",
        "Top-20 candidate failure counts overall:",
        "",
        "```json",
        json.dumps(breakdown["candidate_failure_counts"].get("overall", {}), indent=2, sort_keys=True),
        "```",
        "",
        "Top-20 sample-level WA diagnostics overall:",
        "",
        "```json",
        json.dumps(overall_diag, indent=2, sort_keys=True),
        "```",
        "",
        "## Diagnosis",
        "",
        f"- Full-test match@20 is {pct(f20['match_at_k'])}; structured-only match@20 is {pct(s20['match_at_k'])}. The full-test number is lower by construction because 153 original-test samples failed structured extraction.",
        f"- This `wa5_geom4` budget has only 5 distinct WA candidates with 4 geometry variants each. Structured WA_hit@20 is {pct(s20['wa_hit_at_k'])}, so the selected WA coverage is already close to the observed match@20. Among {overall_diag.get('failed', 0)} failed structured samples, {overall_diag.get('failed_without_wa_hit', 0)} have no selected GT-WA hit and {overall_diag.get('failed_with_wa_hit', 0)} have a GT-WA hit but still fail matching. That makes selected candidate/rerank coverage the larger match@20 ceiling, while geometry/validity remains the main quality and RMSE gap.",
        f"- RMSE changes from {num(structured['top1']['RMSE'])} at @1 to {num(structured['top5']['RMSE'])} at @5 and {num(s20['RMSE'])} at @20. TopK RMSE {'gets worse' if rmse_gets_worse else 'does not get worse'} under best-match RMSE aggregation.",
        f"- readable@20 changed from old {pct(OLD_WA20_GEOM1['readable20'])} to {pct(s20['readable'])}; strict_valid@20 changed from old {pct(OLD_WA20_GEOM1['strict_valid20'])} to {pct(s20['strict_valid'])}.",
        "",
        "## Next Step",
        "",
        next_step,
        "",
        "## Artifacts",
        "",
        f"- audit: `{rel(args.out_dir / 'artifact_audit.md')}` / `{rel(args.out_dir / 'artifact_audit.json')}`",
        f"- metrics: `{rel(args.out_dir / 'metrics.jsonl')}` and `{rel(args.out_dir / 'metrics' / 'baseline_per_generation_metrics.jsonl')}`",
        f"- failed cases: `{rel(args.out_dir / 'failed_eval_cases.jsonl')}`",
        f"- summary: `{rel(args.out_dir / 'full_eval_summary.json')}`",
    ]
    report_path = args.out_dir / "mp20_wa5_geom4_final_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume MP-20 wa5_geom4 stablekey full eval and write final report.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "symcif_v4_table3_fixed_full_rerun_stablekey_hybrid"
        / "mp20"
        / "wa5_geom4_stablekey_hybrid",
    )
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "runs" / "symcif_v4_geometry_model_no_oversampling" / "ckpt_best.pt",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--eval-workers", type=int, default=32)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-match-sites", type=int, default=300)
    parser.add_argument("--max-eval-sites", type=int, default=300)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.data_root / "test.jsonl")
    if len(records) != STRUCTURED_TEST_DENOMINATOR:
        raise SystemExit(f"expected {STRUCTURED_TEST_DENOMINATOR} MP-20 structured samples, got {len(records)}")
    audit = audit_artifacts(args, records)
    if args.audit_only:
        print(json.dumps({"audit": str(args.out_dir / "artifact_audit.json"), "can_resume_eval": audit["can_resume_eval"]}, sort_keys=True))
        return 0
    if not audit["can_resume_eval"]:
        raise SystemExit("artifact audit failed; refusing to run full evaluator")
    meta_by_sample = load_meta_by_sample(args.out_dir / "top20_predictions.jsonl")
    grouped = load_generation_groups(args.out_dir / "generations" / "baseline.jsonl")
    run_resume_eval(args=args, records=records, grouped=grouped)
    metrics = write_metric_outputs(args, records, meta_by_sample)
    if len(metrics) != len(records) * args.top_k:
        raise SystemExit(f"expected {len(records) * args.top_k} metrics, got {len(metrics)}")
    summary = write_structured_outputs(args, records, metrics)
    write_final_report(args, audit, summary)
    print(
        json.dumps(
            {
                "structured_match20": summary["structured_only"]["top20"]["match_at_k"],
                "full_test_match20": summary["full_test"]["top20"]["match_at_k"],
                "structured_rmse20": summary["structured_only"]["top20"]["RMSE"],
                "summary": str(args.out_dir / "mp20_wa5_geom4_final_summary.md"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
