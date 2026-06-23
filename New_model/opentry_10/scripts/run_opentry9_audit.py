#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
NEW_MODEL = ROOT.parent
WORKSPACE = NEW_MODEL.parents[1]
OP7 = NEW_MODEL / "opentry_7"
OP8 = NEW_MODEL / "opentry_8"
OP6 = NEW_MODEL / "opentry_6"
OP5 = NEW_MODEL / "opentry_5"
SYMCIF = NEW_MODEL / "symcif_experiment"


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_9: {resolved}")
    return resolved


def write_text(path: Path, text: str) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_log(text: str) -> None:
    with under_root(ROOT / "experiment_log.md").open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def count_tar_cifs(path: Path) -> int | None:
    if not path.exists():
        return None
    with tarfile.open(path, "r:*") as tf:
        return sum(1 for member in tf if member.isfile() and member.name.endswith(".cif"))


def parse_material_id(sample_id: str | None, fallback: str | None = None) -> str | None:
    if fallback:
        return str(fallback)
    if not sample_id:
        return None
    if "__" in sample_id:
        return sample_id.rsplit("__", 1)[-1]
    return sample_id


def normalize_dataset(dataset: str | None, sample_id: str | None = None) -> str:
    raw = (dataset or "").lower().replace("-", "_")
    if raw in {"mp20", "mp_20"}:
        return "mp_20"
    if raw in {"mpts52", "mpts_52"}:
        return "mpts_52"
    if sample_id and sample_id.startswith("mpts_52"):
        return "mpts_52"
    return "mp_20"


def normalize_split(split: str | None, sample_id: str | None = None) -> str:
    if split:
        return str(split)
    if sample_id:
        for token in ("train", "val", "test"):
            if f"_{token}_" in sample_id or sample_id.endswith(f"_{token}"):
                return token
    return "test"


def candidate_text(row: dict[str, Any]) -> str:
    for key in ("cif", "generated_text", "text", "output", "candidate"):
        val = row.get(key)
        if isinstance(val, str):
            return val
    return ""


def extract_atom_site_rows(cif: str) -> tuple[int, int | None]:
    in_loop = False
    saw_atom_header = False
    row_count = 0
    mult_sum = 0
    mult_seen = False
    headers: list[str] = []
    for raw in cif.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "loop_":
            if in_loop and saw_atom_header and row_count:
                break
            in_loop = True
            saw_atom_header = False
            headers = []
            continue
        if in_loop and stripped.startswith("_"):
            if stripped.startswith("_atom_site_"):
                saw_atom_header = True
                headers.append(stripped)
                continue
            if saw_atom_header:
                break
        if in_loop and saw_atom_header:
            parts = stripped.split()
            if not parts or stripped.startswith("_") or stripped.startswith("data_"):
                break
            row_count += 1
            try:
                idx = headers.index("_atom_site_symmetry_multiplicity")
                mult_sum += int(float(parts[idx]))
                mult_seen = True
            except Exception:
                pass
    return row_count, mult_sum if mult_seen else None


def parse_float_field(cif: str, key: str) -> float | None:
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s+('?[-+0-9.eE]+'?|[-+0-9.eE]+)", cif)
    if not m:
        return None
    try:
        return float(m.group(1).strip("'\""))
    except ValueError:
        return None


def parse_int_field(cif: str, key: str) -> int | None:
    val = parse_float_field(cif, key)
    return int(val) if val is not None else None


def cif_diag(cif: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    row = row or {}
    atom_site_rows, atom_count = extract_atom_site_rows(cif)
    volume = parse_float_field(cif, "_cell_volume")
    sg_number = parse_int_field(cif, "_symmetry_Int_Tables_number")
    readable = bool(cif.strip()) and "data_" in cif[:200] and "_atom_site" in cif
    volume_per_atom = None
    if volume is not None and atom_count:
        volume_per_atom = volume / atom_count
    formula_ok = row.get("formula_ok")
    atom_count_ok = row.get("atom_count_ok")
    sg_ok = row.get("SG_ok", row.get("space_group_ok"))
    return {
        "readable": bool(row.get("readable", row.get("pymatgen_readable", readable))),
        "formula_ok": formula_ok,
        "atom_count_ok": atom_count_ok,
        "SG_consistency": sg_ok,
        "valid_CIF": row.get("valid", row.get("strict_valid")),
        "row_count": atom_site_rows,
        "atom_site_rows": atom_site_rows,
        "expanded_atom_count": atom_count,
        "space_group_number": sg_number,
        "volume": volume,
        "volume_per_atom": volume_per_atom,
        "shortest_distance": row.get("shortest_distance"),
        "obvious_geometry_error": row.get("bond_lengths_reasonable") is False if "bond_lengths_reasonable" in row else None,
        "canonical_skeleton_key": row.get("canonical_skeleton_key") or row.get("skeleton_multiset_key"),
        "canonical_wa_key": row.get("canonical_wa_key") or row.get("wa_multiset_key"),
    }


def rank_to_zero_based(row: dict[str, Any]) -> int:
    for key in ("source_rank", "rank", "gen_index"):
        val = row.get(key)
        if val is None:
            continue
        try:
            n = int(val)
            if key == "rank" and n > 0:
                return n - 1
            return n
        except Exception:
            continue
    return 0


def summarize_slots(slots: Counter[str]) -> dict[str, Any]:
    vals = list(slots.values())
    if not vals:
        return {"sample_coverage": 0, "min_slots": 0, "median_slots": 0, "max_slots": 0}
    return {
        "sample_coverage": len(vals),
        "min_slots": min(vals),
        "median_slots": statistics.median(vals),
        "max_slots": max(vals),
    }


def unify_jsonl_source(
    source_path: Path,
    out_name: str,
    source_name: str,
    source_kind: str,
    dataset_hint: str,
    split_hint: str,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    out_path = under_root(ROOT / "candidates" / out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    slots: Counter[str] = Counter()
    readable = 0
    exact_hashes: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(source_path):
            if filter_fn and not filter_fn(row):
                continue
            sample_id = str(row.get("sample_id") or row.get("id") or row.get("material_id") or "")
            dataset = normalize_dataset(row.get("dataset") or dataset_hint, sample_id)
            split = normalize_split(row.get("split") or split_hint, sample_id)
            source_rank = rank_to_zero_based(row)
            cif = candidate_text(row)
            diag = cif_diag(cif, row)
            if diag["readable"]:
                readable += 1
            digest = hashlib.sha1(cif.encode("utf-8", errors="ignore")).hexdigest() if cif else None
            if digest:
                exact_hashes[digest] += 1
            metadata = {
                "material_id": row.get("material_id") or parse_material_id(sample_id),
                "source_kind": source_kind,
                "source_path": str(source_path),
                "diagnostics": diag,
                "original_keys": sorted(row.keys()),
                "original_source_kind": row.get("source_kind"),
                "original_system": row.get("system"),
                "candidate_candidate_duplicate_cluster_id": digest,
            }
            for key in ("canonical_skeleton_key", "canonical_wa_key", "skeleton_multiset_key", "wa_multiset_key"):
                if key in row:
                    metadata[key] = row[key]
            unified = {
                "dataset": dataset,
                "split": split,
                "sample_id": sample_id,
                "source": source_name,
                "source_rank": source_rank,
                "candidate_id": f"{source_name}:{sample_id}:{source_rank}",
                "cif": cif,
                "metadata": metadata,
            }
            out.write(json.dumps(unified, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
            slots[sample_id] += 1
    summary = {
        "source": source_name,
        "source_path": str(source_path),
        "out_path": str(out_path),
        "candidate_count": count,
        "readable_count": readable,
        "readable_rate": readable / count if count else None,
        "exact_duplicate_rows": sum(v - 1 for v in exact_hashes.values() if v > 1),
        **summarize_slots(slots),
    }
    return summary


def resolve_cif_path(cif_path: str | None) -> Path | None:
    if not cif_path:
        return None
    p = Path(cif_path)
    if p.is_absolute():
        return p
    return WORKSPACE / p


def unify_top20_predictions(
    source_path: Path,
    out_name: str,
    source_name: str,
    source_kind: str,
    dataset_hint: str,
    split_hint: str,
) -> dict[str, Any]:
    out_path = under_root(ROOT / "candidates" / out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    slots: Counter[str] = Counter()
    readable = 0
    missing_cif = 0
    exact_hashes: Counter[str] = Counter()
    with out_path.open("w", encoding="utf-8") as out:
        for sample in iter_jsonl(source_path):
            sample_id = str(sample.get("sample_id") or sample.get("id") or "")
            dataset = normalize_dataset(sample.get("dataset") or dataset_hint, sample_id)
            split = normalize_split(sample.get("split") or split_hint, sample_id)
            for pred in sample.get("predictions") or []:
                source_rank = rank_to_zero_based(pred)
                cif = ""
                cif_file = resolve_cif_path(pred.get("cif_path"))
                if cif_file and cif_file.exists():
                    cif = cif_file.read_text(encoding="utf-8", errors="replace")
                else:
                    missing_cif += 1
                diag = cif_diag(cif, pred)
                if diag["readable"]:
                    readable += 1
                digest = hashlib.sha1(cif.encode("utf-8", errors="ignore")).hexdigest() if cif else None
                if digest:
                    exact_hashes[digest] += 1
                metadata = {
                    "source_kind": source_kind,
                    "source_path": str(source_path),
                    "cif_path": pred.get("cif_path"),
                    "diagnostics": diag,
                    "candidate_candidate_duplicate_cluster_id": digest,
                    "geometry_source": pred.get("geometry_source"),
                    "selection_mode": pred.get("selection_mode"),
                    "source_labels": pred.get("source_labels"),
                    "canonical_skeleton_key": pred.get("canonical_skeleton_key") or pred.get("skeleton_multiset_key"),
                    "canonical_wa_key": pred.get("canonical_wa_key") or pred.get("wa_multiset_key"),
                    "gt_skeleton_key_available": bool(sample.get("gt_skeleton_key") or sample.get("gt_skeleton_multiset_key")),
                    "gt_wa_key_available": bool(sample.get("gt_wa_key") or sample.get("gt_wa_multiset_key")),
                }
                unified = {
                    "dataset": dataset,
                    "split": split,
                    "sample_id": sample_id,
                    "source": source_name,
                    "source_rank": source_rank,
                    "candidate_id": f"{source_name}:{sample_id}:{source_rank}",
                    "cif": cif,
                    "metadata": metadata,
                }
                out.write(json.dumps(unified, ensure_ascii=True, sort_keys=True) + "\n")
                count += 1
                slots[sample_id] += 1
    return {
        "source": source_name,
        "source_path": str(source_path),
        "out_path": str(out_path),
        "candidate_count": count,
        "readable_count": readable,
        "readable_rate": readable / count if count else None,
        "missing_cif_rows": missing_cif,
        "exact_duplicate_rows": sum(v - 1 for v in exact_hashes.values() if v > 1),
        **summarize_slots(slots),
    }


def concat_structured_cache() -> dict[str, Any]:
    sources = {
        "train": [
            SYMCIF / "data/structured_symcif_v4_mp20/train.jsonl",
            SYMCIF / "data/structured_symcif_v4_mpts52/train.jsonl",
        ],
        "val": [
            SYMCIF / "data/structured_symcif_v4_mp20/val.jsonl",
            SYMCIF / "data/structured_symcif_v4_mpts52/val.jsonl",
        ],
        "test_targets": [
            SYMCIF / "data/structured_symcif_v4_mp20/test.jsonl",
            SYMCIF / "data/structured_symcif_v4_mpts52/test.jsonl",
        ],
    }
    summary: dict[str, Any] = {}
    for split, paths in sources.items():
        out_path = under_root(ROOT / "cache" / f"symcif_{split}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        datasets = Counter()
        success = 0
        with out_path.open("w", encoding="utf-8") as out:
            for src in paths:
                with src.open("r", encoding="utf-8") as inp:
                    for line in inp:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        datasets[str(row.get("dataset"))] += 1
                        if row.get("row_expansion_all_ok") and row.get("free_param_reextract_all_success"):
                            success += 1
                        out.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                        rows += 1
        summary[split] = {
            "out_path": str(out_path),
            "source_paths": [str(p) for p in paths],
            "rows": rows,
            "datasets": dict(datasets),
            "conversion_success_rows": success,
            "conversion_success_rate": success / rows if rows else None,
        }
    return summary


def load_sample_metrics(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in iter_jsonl(path):
        sid = str(row.get("sample_id") or row.get("material_id") or "")
        if sid:
            out[sid] = row
    return out


def summarize_sample_metrics(rows: dict[str, dict[str, Any]], include_rows7: bool = True) -> dict[str, Any]:
    vals = list(rows.values())
    out: dict[str, Any] = {"samples": len(vals)}
    for k in (1, 5, 20):
        key = f"match@{k}"
        hits = [bool(r.get(key)) for r in vals]
        out[key] = sum(hits) / len(hits) if hits else None
        rmse_key = f"RMSE@{k}"
        rmses = [r.get(rmse_key) for r in vals if isinstance(r.get(rmse_key), (int, float))]
        out[rmse_key] = sum(rmses) / len(rmses) if rmses else None
    rows7 = {sid: r for sid, r in rows.items() if int(r.get("row_count") or 0) >= 7}
    if include_rows7 and rows7:
        out["rows>=7"] = summarize_sample_metrics(rows7, include_rows7=False)
    return out


def reuse_unified_summaries_if_complete() -> list[dict[str, Any]] | None:
    summary_path = ROOT / "metrics/unified_candidate_source_summaries.json"
    if not summary_path.exists():
        return None
    summaries = read_json(summary_path, [])
    if not isinstance(summaries, list) or not summaries:
        return None
    for item in summaries:
        out_path = Path(str(item.get("out_path", "")))
        if not out_path.exists() or out_path.stat().st_size == 0:
            return None
    return summaries


def overlap_metrics(anchor: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]], k: int = 20) -> dict[str, Any]:
    ids = sorted(set(anchor) | set(other))
    a_hit = {sid for sid in ids if bool(anchor.get(sid, {}).get(f"match@{k}"))}
    b_hit = {sid for sid in ids if bool(other.get(sid, {}).get(f"match@{k}"))}
    rows7 = {sid for sid in ids if int((anchor.get(sid) or other.get(sid) or {}).get("row_count") or 0) >= 7}
    def pct(n: int, d: int) -> float | None:
        return n / d if d else None
    payload = {
        "samples_union": len(ids),
        "A_hit": len(a_hit),
        "B_hit": len(b_hit),
        "A_and_B_hit": len(a_hit & b_hit),
        "B_exclusive_rescue": len(b_hit - a_hit),
        "A_exclusive_loss": len(a_hit - b_hit),
        "oracle_union": len(a_hit | b_hit),
        "A_hit_rate": pct(len(a_hit), len(ids)),
        "B_hit_rate": pct(len(b_hit), len(ids)),
        "oracle_union_rate": pct(len(a_hit | b_hit), len(ids)),
        "B_exclusive_rescue_rate": pct(len(b_hit - a_hit), len(ids)),
    }
    if rows7:
        payload["rows>=7"] = {
            "samples_union": len(rows7),
            "A_hit": len(a_hit & rows7),
            "B_hit": len(b_hit & rows7),
            "B_exclusive_rescue": len((b_hit - a_hit) & rows7),
            "oracle_union": len((a_hit | b_hit) & rows7),
            "A_hit_rate": pct(len(a_hit & rows7), len(rows7)),
            "B_hit_rate": pct(len(b_hit & rows7), len(rows7)),
            "oracle_union_rate": pct(len((a_hit | b_hit) & rows7), len(rows7)),
            "B_exclusive_rescue_rate": pct(len((b_hit - a_hit) & rows7), len(rows7)),
        }
    return payload


def internal_rerank_space(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vals = list(rows.values())
    n = len(vals)
    def hit(k: int) -> set[str]:
        return {str(r.get("sample_id")) for r in vals if bool(r.get(f"match@{k}"))}
    h1, h5, h20 = hit(1), hit(5), hit(20)
    rows7_ids = {str(r.get("sample_id")) for r in vals if int(r.get("row_count") or 0) >= 7}
    payload = {
        "samples": n,
        "match@1": len(h1) / n if n else None,
        "match@5": len(h5) / n if n else None,
        "match@20": len(h20) / n if n else None,
        "top1_fail_but_top5_hit": len(h5 - h1),
        "top1_fail_but_top20_hit": len(h20 - h1),
        "top5_fail_but_top20_hit": len(h20 - h5),
        "top1_fail_but_top5_hit_rate": len(h5 - h1) / n if n else None,
        "top1_fail_but_top20_hit_rate": len(h20 - h1) / n if n else None,
        "top5_fail_but_top20_hit_rate": len(h20 - h5) / n if n else None,
        "oracle_rerank@1_upper_bound": len(h20) / n if n else None,
        "oracle_rerank@5_upper_bound": len(h20) / n if n else None,
        "rank_cumulative_gain_note": "Only per-sample @1/@5/@20 metrics are available; per-rank match labels are absent, so exact rank-wise cumulative gains were not recomputed.",
    }
    if rows7_ids:
        payload["rows>=7"] = {
            "samples": len(rows7_ids),
            "match@1": len(h1 & rows7_ids) / len(rows7_ids),
            "match@5": len(h5 & rows7_ids) / len(rows7_ids),
            "match@20": len(h20 & rows7_ids) / len(rows7_ids),
            "top1_fail_but_top5_hit": len((h5 - h1) & rows7_ids),
            "top1_fail_but_top20_hit": len((h20 - h1) & rows7_ids),
            "top5_fail_but_top20_hit": len((h20 - h5) & rows7_ids),
        }
    return payload


def load_symcif_summary() -> dict[str, Any]:
    def md_extract(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "mp20_val_baseline": read_json(SYMCIF / "reports/symcif_v4_mp20_k5_dev/val_baseline_summary.json", {}),
        "mp20_val_baseline_md": md_extract(SYMCIF / "reports/symcif_v4_mp20_k5_dev/val_baseline_summary.md"),
        "mp20_val_gtwa_geometry": read_json(SYMCIF / "reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval.json", {}),
        "mp20_val_gtwa_geometry_md": md_extract(SYMCIF / "reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval.md"),
        "mp20_val_one_fix": read_json(SYMCIF / "reports/symcif_v4_mp20_k5_dev/one_fix_experiment_summary.json", {}),
        "mp20_val_one_fix_md": md_extract(SYMCIF / "reports/symcif_v4_mp20_k5_dev/one_fix_experiment_summary.md"),
        "mp20_wa_coverage": read_json(SYMCIF / "reports/symcif_v4_mp20_k5_dev/wa_top5_coverage_audit.json", {}),
        "mp20_root_cause_md": md_extract(SYMCIF / "reports/symcif_v4_mp20_k5_dev/final_root_cause_summary.md"),
    }


def get_nested(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100 * x:.2f}%"


def f4(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def build_source_manifest(source_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_entries: list[dict[str, Any]] = [
        {
            "source_name": "crystallm_a_gt_sg_anchor_mp20_test",
            "dataset": "mp_20",
            "split": "test",
            "source_path": str(OP8 / "generations/strategy_fusion_mp_20_test_k20_candidates.jsonl"),
            "source_filter": "source_kind == primary_crystallm_a_gt_sg",
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": True,
            "notes": "Anchor candidates recovered from opentry_8 strategy_fusion rows; MP-20 strategy_fusion is exactly this anchor.",
        },
        {
            "source_name": "crystallm_a_gt_sg_anchor_mpts52_test",
            "dataset": "mpts_52",
            "split": "test",
            "source_path": str(OP8 / "generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl"),
            "source_filter": "source_kind == primary_crystallm_a_gt_sg",
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": True,
            "notes": "Anchor candidates recovered from opentry_8 strategy_fusion rows; one sample was missing in primary source.",
        },
        {
            "source_name": "strategy_fusion_mp20_test",
            "dataset": "mp_20",
            "split": "test",
            "source_path": str(OP8 / "generations/strategy_fusion_mp_20_test_k20_candidates.jsonl"),
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": True,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Historical post-test artifact; no validation gate in opentry_9.",
        },
        {
            "source_name": "strategy_fusion_mpts52_test",
            "dataset": "mpts_52",
            "split": "test",
            "source_path": str(OP8 / "generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl"),
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": True,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Historical post-test artifact; only one sample was coverage-repaired.",
        },
        {
            "source_name": "strategy_stablekey_hybrid_mp20_test",
            "dataset": "mp_20",
            "split": "test",
            "source_path": str(OP7 / "generations/strategy_stablekey_hybrid_mp_20_test_k20_candidates.jsonl"),
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": True,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Historical test-only stablekey/SymCIF hybrid. Useful for post-hoc complement diagnosis only.",
        },
        {
            "source_name": "strategy_stablekey_hybrid_mpts52_test",
            "dataset": "mpts_52",
            "split": "test",
            "source_path": str(OP7 / "generations/strategy_stablekey_hybrid_mpts_52_test_k20_candidates.jsonl"),
            "official_full_test": True,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": True,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Historical test-only stablekey/SymCIF hybrid. Useful for post-hoc complement diagnosis only.",
        },
        {
            "source_name": "symcif_v4_mp20_val_baseline_k5",
            "dataset": "mp_20",
            "split": "val",
            "source_path": str(SYMCIF / "reports/symcif_v4_mp20_k5_dev/val_baseline_eval_gpu/top20_predictions.jsonl"),
            "official_full_test": False,
            "validation": True,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": True,
            "can_final_test_fusion": False,
            "notes": "MP-20 validation K<=5 SymCIF-v4 structural candidate bank; not CrystaLLM anchor.",
        },
        {
            "source_name": "pure_gt_wa_geometry_mp20_val_k5",
            "dataset": "mp_20",
            "split": "val",
            "source_path": str(SYMCIF / "reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval_gpu/top20_predictions.jsonl"),
            "official_full_test": False,
            "validation": True,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "GT-WA geometry upper-bound artifact, K<=5 only.",
        },
        {
            "source_name": "one_fix_hybrid_prior_mp20_val_k5",
            "dataset": "mp_20",
            "split": "val",
            "source_path": str(SYMCIF / "reports/symcif_v4_mp20_k5_dev/one_fix_hybrid_prior_eval_gpu/top20_predictions.jsonl"),
            "official_full_test": False,
            "validation": True,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": True,
            "can_final_test_fusion": False,
            "notes": "Validation-only one-fix prior selector artifact; diagnostic, not frozen for test.",
        },
        {
            "source_name": "symcif_v4_exact_cover_val_wa_candidates",
            "dataset": "mixed",
            "split": "val",
            "source_path": str(SYMCIF / "reports/composition_exact_v1/val_wa_candidates.jsonl"),
            "official_full_test": False,
            "validation": True,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": True,
            "can_final_test_fusion": False,
            "notes": "WA candidates without rendered official K20 CIF/eval; mechanism-only.",
        },
        {
            "source_name": "opentry_5_fixed_order_geometry_smokes",
            "dataset": "fold/dev",
            "split": "fold",
            "source_path": str(OP5 / "eval"),
            "official_full_test": False,
            "validation": False,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Mechanism-analysis smoke/fold generations only.",
        },
        {
            "source_name": "opentry_6_stage_geometry_exactcover_folds",
            "dataset": "fold/dev",
            "split": "fold",
            "source_path": str(OP6 / "eval"),
            "official_full_test": False,
            "validation": False,
            "pure_model": True,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "Mechanism-analysis fold generations only.",
        },
        {
            "source_name": "crystallm_gt_sg_k50_k100_raw_pool",
            "dataset": "mp_20/mpts_52",
            "split": "test/val",
            "source_path": "NOT_FOUND",
            "official_full_test": False,
            "validation": False,
            "pure_model": False,
            "strategy_retrieval_fusion": False,
            "can_train_selector": False,
            "can_final_test_fusion": False,
            "notes": "No CrystaLLM-a GT-SG K50/K100 full raw pool was found; only opentry_2 partial atomtype K50 diagnostics exist.",
        },
    ]
    by_name = {s["source"]: s for s in source_summaries}
    for entry in base_entries:
        stats = by_name.get(entry["source_name"])
        if stats:
            entry.update({
                "candidate_count": stats.get("candidate_count"),
                "sample_coverage": stats.get("sample_coverage"),
                "per_sample_candidate_slots": {
                    "min": stats.get("min_slots"),
                    "median": stats.get("median_slots"),
                    "max": stats.get("max_slots"),
                },
                "unified_path": stats.get("out_path"),
                "readable_rate": stats.get("readable_rate"),
                "duplicate_rows": stats.get("exact_duplicate_rows"),
            })
        else:
            path = Path(str(entry["source_path"]))
            if path.suffix == ".jsonl":
                entry["candidate_count"] = count_lines(path)
            elif str(path).endswith(".tar.gz"):
                entry["candidate_count"] = count_tar_cifs(path)
            else:
                entry["candidate_count"] = None
            entry["sample_coverage"] = None
            entry["per_sample_candidate_slots"] = None
    return base_entries


def write_candidate_manifest(entries: list[dict[str, Any]]) -> None:
    write_json(ROOT / "metrics/candidate_source_manifest.json", entries)
    lines = [
        "# Candidate Source Manifest",
        "",
        "All rows are based on read-only historical artifacts. Test match labels are not used for selector training or threshold selection.",
        "",
        "| source name | dataset | split | candidate count | sample coverage | per-sample slots | official full-test | validation | pure model | strategy/retrieval/fusion | train selector | final test fusion | notes |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        slots = e.get("per_sample_candidate_slots")
        if isinstance(slots, dict):
            slot_text = f"{slots.get('min')}/{slots.get('median')}/{slots.get('max')}"
        else:
            slot_text = "NA"
        lines.append(
            "| {source_name} | {dataset} | {split} | {candidate_count} | {sample_coverage} | {slots} | {official} | {val} | {pure} | {strategy} | {train} | {fusion} | {notes} |".format(
                source_name=e["source_name"],
                dataset=e["dataset"],
                split=e["split"],
                candidate_count=e.get("candidate_count") if e.get("candidate_count") is not None else "NA",
                sample_coverage=e.get("sample_coverage") if e.get("sample_coverage") is not None else "NA",
                slots=slot_text,
                official="yes" if e["official_full_test"] else "no",
                val="yes" if e["validation"] else "no",
                pure="yes" if e["pure_model"] else "no",
                strategy="yes" if e["strategy_retrieval_fusion"] else "no",
                train="yes" if e["can_train_selector"] else "no",
                fusion="yes" if e["can_final_test_fusion"] else "no",
                notes=e["notes"].replace("|", "/"),
            )
        )
    write_text(ROOT / "reports/candidate_source_manifest.md", "\n".join(lines))


def write_strategy_reports(metrics: dict[str, Any], sym: dict[str, Any]) -> None:
    write_json(ROOT / "metrics/strategy_oracle_union_audit.json", metrics)
    lines = [
        "# Strategy Oracle Union Audit",
        "",
        "## Validation Gate Status",
        "",
        "- CrystaLLM-a GT-SG validation K20 candidate bank was not found in opentry_5/6/7/8 or symcif_experiment.",
        "- opentry_7 cache contains official validation target CIF tarballs, not K20 CrystaLLM candidates; these are treated as references and were not used for candidate fusion.",
        "- Therefore the required validation oracle-union gate cannot be passed. Phase B selector/ranker and new official test are not run.",
        "",
        "## Available Validation Diagnostics",
        "",
        "- MP-20 SymCIF-v4 K<=5 baseline: match@1/match@5 = 44.12% / 63.42%; WA_hit@1/WA_hit@5 = 38.63% / 65.11%.",
        "- MP-20 one-fix prior selector: selected WA_hit@5 improves to 79.52% and match@5 to 71.58% in the historical validation artifact.",
        "- MP-20 GT-WA geometry K<=5: match@1/match@5 = 77.16% / 82.94%; rows>=7/n_sites>=6 match@1/match@5 = 59.22% / 69.46%.",
        "",
        "## Historical Test-Only Post-Hoc Overlap",
        "",
        "These numbers use already-computed official test metrics only as diagnosis. They were not used to tune or freeze any strategy.",
    ]
    for name, payload in metrics.get("test_posthoc_overlap", {}).items():
        lines += [
            "",
            f"### {name}",
            "",
            f"- A_hit@20: {payload['A_hit']} ({pct(payload['A_hit_rate'])})",
            f"- B_hit@20: {payload['B_hit']} ({pct(payload['B_hit_rate'])})",
            f"- B exclusive rescue@20: {payload['B_exclusive_rescue']} ({pct(payload['B_exclusive_rescue_rate'])})",
            f"- Oracle union@20: {payload['oracle_union']} ({pct(payload['oracle_union_rate'])})",
        ]
        r7 = payload.get("rows>=7")
        if r7:
            lines.append(f"- rows>=7 oracle union@20: {r7['oracle_union']} ({pct(r7['oracle_union_rate'])}); B exclusive rescue: {r7['B_exclusive_rescue']} ({pct(r7['B_exclusive_rescue_rate'])})")
    write_text(ROOT / "reports/strategy_oracle_union_audit.md", "\n".join(lines))

    rank_lines = [
        "# CrystaLLM Internal Rerank Space",
        "",
        "Required validation K20 per-rank labels were not available. The table below is a post-hoc diagnostic from existing official test sample metrics only; it is not used for tuning.",
        "",
        "| dataset | match@1 | match@5 | match@20 | top1 fail but top5 hit | top1 fail but top20 hit | top5 fail but top20 hit | oracle rerank@1 upper bound |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, payload in metrics.get("internal_rerank_space", {}).items():
        rank_lines.append(
            f"| {dataset} | {pct(payload.get('match@1'))} | {pct(payload.get('match@5'))} | {pct(payload.get('match@20'))} | "
            f"{payload.get('top1_fail_but_top5_hit')} | {payload.get('top1_fail_but_top20_hit')} | {payload.get('top5_fail_but_top20_hit')} | {pct(payload.get('oracle_rerank@1_upper_bound'))} |"
        )
    rank_lines += [
        "",
        "Exact per-rank cumulative gain could not be recomputed because the historical sample_metrics files only expose @1/@5/@20 booleans, not per-candidate match labels.",
    ]
    write_text(ROOT / "reports/crystallm_internal_rerank_space.md", "\n".join(rank_lines))
    write_json(ROOT / "metrics/crystallm_internal_rerank_space.json", metrics.get("internal_rerank_space", {}))

    ranker_report = "\n".join([
        "# Strategy Ranker Validation Report",
        "",
        "No selector/ranker was trained in opentry_9.",
        "",
        "Reason: Phase A validation gate could not be evaluated against a CrystaLLM-a GT-SG validation K20 anchor candidate bank. Using official test StructureMatcher labels to train or select a ranker is explicitly prohibited.",
        "",
        "Frozen strategy status: not frozen; no official test launched.",
    ])
    write_text(ROOT / "reports/strategy_ranker_validation_report.md", ranker_report)
    write_json(ROOT / "frozen_strategy/config.json", {
        "status": "not_frozen",
        "reason": "Phase A validation gate unavailable: no CrystaLLM-a GT-SG validation K20 candidate bank was found.",
        "new_official_test_run": False,
        "test_feedback_used_for_training_or_selection": False,
    })
    write_json(ROOT / "frozen_strategy/strategy_build_manifest.json", {
        "status": "not_built",
        "reason": "No validation-approved selector/ranker.",
    })
    write_text(ROOT / "frozen_strategy/no_test_feedback_declaration.md", "\n".join([
        "# No Test Feedback Declaration",
        "",
        "opentry_9 did not train, tune, threshold, freeze, or rerank using official test StructureMatcher labels.",
        "",
        "Existing official test summaries from opentry_7/opentry_8 were read only for post-hoc reporting and to avoid rerunning misleading fusion. No new official test was launched in opentry_9.",
    ]))
    write_text(ROOT / "reports/strategy_official_test_report.md", "\n".join([
        "# Strategy Official Test Report",
        "",
        "No new opentry_9 official full-test run was executed.",
        "",
        "The frozen strategy gate did not pass because validation oracle-union and reranker validation could not be computed against a CrystaLLM-a GT-SG validation K20 anchor. Historical opentry_8 strategy_fusion results are therefore reported only as prior evidence: MP-20 was exactly the CrystaLLM-a GT-SG anchor, and MPTS-52 repaired only one missing sample.",
    ]))


def write_pure_reports(cache_summary: dict[str, Any], sym: dict[str, Any]) -> None:
    write_json(ROOT / "metrics/symcif_canonicalization_audit.json", cache_summary)
    total_rows = sum(v["rows"] for v in cache_summary.values())
    total_success = sum(v["conversion_success_rows"] for v in cache_summary.values())
    lines = [
        "# SymCIF Canonicalization Audit",
        "",
        "opentry_9 reused the existing SymCIF-v4 structured records from symcif_experiment and wrote local cache copies.",
        "",
        f"- total rows: {total_rows}",
        f"- conversion success rows: {total_success} ({pct(total_success / total_rows if total_rows else None)})",
        "",
        "| output | rows | datasets | conversion success |",
        "| --- | ---: | --- | ---: |",
    ]
    for split, payload in cache_summary.items():
        lines.append(f"| cache/symcif_{split}.jsonl | {payload['rows']} | {payload['datasets']} | {pct(payload['conversion_success_rate'])} |")
    lines += [
        "",
        "No failed conversion rows were silently dropped in opentry_9; the source structured files already encode conversion/extraction status fields such as row_expansion_all_ok and free_param_reextract_all_success.",
    ]
    write_text(ROOT / "reports/symcif_canonicalization_audit.md", "\n".join(lines))

    gtwa_json = sym.get("mp20_val_gtwa_geometry", {})
    write_json(ROOT / "metrics/pure_gt_wa_geometry_val.json", gtwa_json)
    write_text(ROOT / "reports/pure_gt_wa_geometry_report.md", "\n".join([
        "# Pure GT-WA Geometry Report",
        "",
        "Input: composition + GT-SG + GT-WA. Output: rendered CIF geometry variants from the existing SymCIF-v4 MP-20 validation K<=5 artifact.",
        "",
        "Key result: GT-WA geometry match@1/match@5 = 77.16% / 82.94%; RMSE@1/RMSE@5 = 0.0450 / 0.0388.",
        "",
        "Rows>=7 proxy (n_sites>=6 in the historical report): match@1/match@5 = 59.22% / 69.46%.",
        "",
        "Interpretation: even with correct WA, geometry is not saturated. The pure model bottleneck is both WA coverage and geometry, with geometry remaining a hard upper bound.",
        "",
        "Limitations: no K20 GT-WA geometry run and no MPTS-52 GT-WA geometry artifact were available in opentry_9.",
    ]))

    wa_payload = {
        "mp20_val": {
            "baseline_WA_hit@1": 0.3863,
            "baseline_WA_hit@5": 0.6511,
            "one_fix_selected_WA_hit@5": 0.7952,
            "raw_top100_WA_hit": 0.8785,
            "source": "symcif_v4_mp20_k5_dev historical reports",
        },
        "checkpoint_written": False,
        "reason": "No new WA decoder was trained in opentry_9; existing artifacts diagnose candidate selection rather than provide a frozen decoder checkpoint.",
    }
    write_json(ROOT / "metrics/pure_wa_decoder_val.json", wa_payload)
    write_text(ROOT / "reports/pure_wa_decoder_val_report.md", "\n".join([
        "# Pure WA Decoder Validation Report",
        "",
        "No new exact-cover WA decoder checkpoint was trained in opentry_9.",
        "",
        "Available MP-20 validation evidence from SymCIF-v4 K<=5:",
        "",
        "- baseline WA_hit@1/@5 = 38.63% / 65.11%",
        "- raw top100 WA_hit = 87.85%",
        "- one-fix train-prior/hybrid-prior candidate selection raises selected WA_hit@5 to 79.52% and match@5 to 71.58%",
        "",
        "Diagnosis: the candidate pool contains substantially more WA coverage than the selected top5 exposes, so WA selection/ranking is a major bottleneck. Because GT-WA geometry still reaches only 82.94% match@5, geometry remains a second hard bottleneck.",
    ]))
    write_json(ROOT / "frozen_pure_model/config.json", {
        "status": "not_frozen",
        "reason": "Pure structural validation did not produce a frozen K20 model/checkpoint in opentry_9.",
        "no_crystallm_candidates_mixed": True,
        "new_official_test_run": False,
    })
    write_json(ROOT / "frozen_pure_model/checkpoint_manifest.json", {
        "wa_decoder_best.pt": None,
        "geometry_checkpoint": None,
        "source_artifacts": [
            str(SYMCIF / "reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval.json"),
            str(SYMCIF / "reports/symcif_v4_mp20_k5_dev/wa_top5_coverage_audit.json"),
        ],
    })
    write_text(ROOT / "reports/pure_structural_val_report.md", "\n".join([
        "# Pure Structural Validation Report",
        "",
        "Pure structural route did not pass a full K20 validation-generation gate in opentry_9.",
        "",
        "Reusable diagnosis:",
        "",
        "- CIF to SymCIF canonicalization coverage is high in the existing structured cache.",
        "- MP-20 GT-WA geometry K<=5 is strong but not saturated: match@5 = 82.94%, rows>=7 proxy match@5 = 69.46%.",
        "- WA candidate selection is weak at K<=5: baseline WA_hit@5 = 65.11%, while raw top100 WA_hit = 87.85%.",
        "",
        "Conclusion: do not run official pure full-test. The next pure-model step should train an inference-feasible exact-cover WA decoder/ranker and a geometry quality scorer, then evaluate K20 on validation before any test.",
    ]))


def write_final_report(
    source_entries: list[dict[str, Any]],
    strategy_metrics: dict[str, Any],
    cache_summary: dict[str, Any],
) -> None:
    op7_anchor_mp = read_json(OP7 / "metrics/crystallm_a_gt_sg_mp_20_test_k20.json", {})
    op7_anchor_mpts = read_json(OP7 / "metrics/crystallm_a_gt_sg_mpts_52_test_k20.json", {})
    op8_fusion_mp = read_json(OP8 / "metrics/strategy_fusion_mp_20_test_k20.json", {})
    op8_fusion_mpts = read_json(OP8 / "metrics/strategy_fusion_mpts_52_test_k20.json", {})
    op7_stable_mp = read_json(OP7 / "metrics/strategy_stablekey_hybrid_mp_20_test_k20.json", {})
    op7_stable_mpts = read_json(OP7 / "metrics/strategy_stablekey_hybrid_mpts_52_test_k20.json", {})

    def row(system: str, scope: str, payload: dict[str, Any], fusion: str, pure: str) -> str:
        allp = payload.get("all", payload)
        r7 = payload.get("rows>=7") or payload.get("rows_ge7") or {}
        return (
            f"| {system} | {scope} | {pct(allp.get('match@1'))} | {pct(allp.get('match@5'))} | {pct(allp.get('match@20'))} | "
            f"{f4(allp.get('RMSE@1'))} | {f4(allp.get('RMSE@5'))} | {f4(allp.get('RMSE@20'))} | "
            f"{pct(r7.get('match@1'))} | {pct(r7.get('match@5'))} | {pct(r7.get('match@20'))} | K20 | {fusion} | {pure} |"
        )

    header = [
        "| system | scope | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | candidate budget | fusion/ranking | pure model |",
        "| ------ | ----- | ------: | ------: | -------: | -----: | -----: | ------: | --------------: | --------------: | ---------------: | ---------------: | -------------- | ---------- |",
    ]
    mp_rows = header + [
        row("CrystaLLM-a GT-SG", "official test, historical opentry_7", op7_anchor_mp, "anchor", "no"),
        row("opentry_8 strategy_fusion", "official test, historical", op8_fusion_mp, "coverage repair", "no"),
        row("opentry_7 stablekey hybrid", "official test, historical", op7_stable_mp, "stablekey/SymCIF", "no"),
    ]
    mpts_rows = header + [
        row("CrystaLLM-a GT-SG", "official test, historical opentry_7", op7_anchor_mpts, "anchor", "no"),
        row("opentry_8 strategy_fusion", "official test, historical", op8_fusion_mpts, "coverage repair", "no"),
        row("opentry_7 stablekey hybrid", "official test, historical", op7_stable_mpts, "stablekey/SymCIF", "no"),
    ]
    report = [
        "# opentry_9 Final Report",
        "",
        "## What Was Executed",
        "",
        "- Created the requested opentry_9 directory structure and local copies of historical evaluator utilities.",
        "- Built a candidate source manifest and unified JSONL candidate files for available anchor/fusion/stablekey/SymCIF validation sources.",
        "- Audited strategy/fusion feasibility without training a selector and without launching a new official test.",
        "- Reused SymCIF-v4 structured caches for pure structural canonicalization and wrote opentry_9 cache copies.",
        "- Reused existing MP-20 validation GT-WA geometry and WA coverage artifacts for pure bottleneck diagnosis.",
        "",
        "## Strategy/Fusion Answer",
        "",
        "strategy/fusion does not have a validation-proven meaningful exceed path in opentry_9. The required CrystaLLM-a GT-SG validation K20 candidate bank was not found, so validation oracle union and reranker upper-bound gates cannot be passed. Consequently no selector/ranker was trained, no frozen strategy was produced, and no new official test was run.",
        "",
        "Historical opentry_8 evidence remains unchanged: MP-20 strategy_fusion is exactly the CrystaLLM-a GT-SG anchor; MPTS-52 repaired only one missing primary sample. That is coverage repair, not a method breakthrough.",
        "",
        "## Validation Oracle Union",
        "",
        "The required validation oracle union against CrystaLLM-a GT-SG K20 could not be computed. Available validation diagnostics are MP-20 SymCIF K<=5 only: baseline match@1/@5 = 44.12% / 63.42%; GT-WA geometry match@1/@5 = 77.16% / 82.94%; one-fix selection match@5 = 71.58%. These are pure/structural diagnostics, not CrystaLLM fusion gates.",
        "",
        "## CrystaLLM K20 Internal Rerank Space",
        "",
        "Only official test @1/@5/@20 sample metrics are available for CrystaLLM K20; per-rank validation labels were not found. Post-hoc test diagnostics show substantial @20-vs-@1 room, but this was not used for tuning.",
        "",
        "## Pure Model Bottleneck",
        "",
        "The pure structural bottleneck is both WA coverage/selection and geometry. WA selection is weak at K<=5 (baseline WA_hit@5 = 65.11%, raw top100 WA_hit = 87.85%), while GT-WA geometry is not saturated (MP-20 match@5 = 82.94%, rows>=7 proxy match@5 = 69.46%).",
        "",
        "No WA decoder checkpoint was trained in opentry_9 and no pure K20 full-test was run.",
        "",
        "## MP-20",
        "",
        *mp_rows,
        "",
        "## MPTS-52",
        "",
        *mpts_rows,
        "",
        "## Gate Decisions",
        "",
        "- strategy selector/ranker trained: no",
        "- frozen strategy passed validation gate: no",
        "- opentry_9 official test run: no",
        "- official test exceed over CrystaLLM-a GT-SG: no new claim",
        "- pure model meaningful exceed CrystaLLM-a GT-SG: no",
        "- test leakage risk: low for opentry_9 actions; historical test metrics were read only for reporting and were not used for tuning",
        "",
        "## Paper-Usable vs Diagnostic",
        "",
        "Paper-usable: the negative result that opentry_8 fusion was coverage repair rather than meaningful exceed, plus the pure structural bottleneck diagnosis if described as validation/diagnostic.",
        "",
        "Diagnostic only: all test-overlap/exclusive-rescue numbers, the MP-20 K<=5 SymCIF one-fix result, and any stablekey hybrid official-test comparison from historical runs.",
        "",
        "Routes not worth continuing as-is: opentry_8-style coverage repair and opentry_7 byte-level CIF pure model. The next best experiment is a validation-first CrystaLLM K20/K50 candidate bank plus an inference-feasible selector, and for pure structural work an exact-cover WA decoder/ranker coupled with geometry-quality reranking.",
    ]
    write_text(ROOT / "final_report.md", "\n".join(report))


def main() -> None:
    start = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    append_log(f"## {start} opentry_9 start\n- Created directory structure.\n- Copied opentry_7/opentry_8 evaluator utility scripts into opentry_9/scripts.\n- Write scope: opentry_9 only; historical directories read-only.")

    source_summaries = reuse_unified_summaries_if_complete()
    if source_summaries is None:
        source_summaries = []
        source_summaries.append(unify_jsonl_source(
            OP8 / "generations/strategy_fusion_mp_20_test_k20_candidates.jsonl",
            "unified_mp_20_test_crystallm_a_gt_sg_anchor.jsonl",
            "crystallm_a_gt_sg_anchor_mp20_test",
            "primary_crystallm_a_gt_sg",
            "mp_20",
            "test",
            lambda r: r.get("source_kind") == "primary_crystallm_a_gt_sg",
        ))
        source_summaries.append(unify_jsonl_source(
            OP8 / "generations/strategy_fusion_mpts_52_test_k20_candidates.jsonl",
            "unified_mpts_52_test_crystallm_a_gt_sg_anchor.jsonl",
            "crystallm_a_gt_sg_anchor_mpts52_test",
            "primary_crystallm_a_gt_sg",
            "mpts_52",
            "test",
            lambda r: r.get("source_kind") == "primary_crystallm_a_gt_sg",
        ))
        for ds, fname, source_name in [
            ("mp_20", "strategy_fusion_mp_20_test_k20_candidates.jsonl", "strategy_fusion_mp20_test"),
            ("mpts_52", "strategy_fusion_mpts_52_test_k20_candidates.jsonl", "strategy_fusion_mpts52_test"),
        ]:
            source_summaries.append(unify_jsonl_source(
                OP8 / "generations" / fname,
                f"unified_{ds}_test_strategy_fusion.jsonl",
                source_name,
                "strategy_fusion",
                ds,
                "test",
                None,
            ))
        for ds, fname, source_name in [
            ("mp_20", "strategy_stablekey_hybrid_mp_20_test_k20_candidates.jsonl", "strategy_stablekey_hybrid_mp20_test"),
            ("mpts_52", "strategy_stablekey_hybrid_mpts_52_test_k20_candidates.jsonl", "strategy_stablekey_hybrid_mpts52_test"),
        ]:
            source_summaries.append(unify_jsonl_source(
                OP7 / "generations" / fname,
                f"unified_{ds}_test_strategy_stablekey_hybrid.jsonl",
                source_name,
                "strategy_stablekey_hybrid",
                ds,
                "test",
                None,
            ))
        for out_name, source_name, path, kind in [
            ("unified_mp_20_val_symcif_v4_k5_baseline.jsonl", "symcif_v4_mp20_val_baseline_k5", SYMCIF / "reports/symcif_v4_mp20_k5_dev/val_baseline_eval_gpu/top20_predictions.jsonl", "symcif_v4_baseline"),
            ("unified_mp_20_val_pure_gt_wa_geometry_k5.jsonl", "pure_gt_wa_geometry_mp20_val_k5", SYMCIF / "reports/symcif_v4_mp20_k5_dev/gtwa_geometry_k5_eval_gpu/top20_predictions.jsonl", "pure_gt_wa_geometry"),
            ("unified_mp_20_val_one_fix_hybrid_prior_k5.jsonl", "one_fix_hybrid_prior_mp20_val_k5", SYMCIF / "reports/symcif_v4_mp20_k5_dev/one_fix_hybrid_prior_eval_gpu/top20_predictions.jsonl", "one_fix_hybrid_prior"),
        ]:
            source_summaries.append(unify_top20_predictions(path, out_name, source_name, kind, "mp_20", "val"))
        write_json(ROOT / "metrics/unified_candidate_source_summaries.json", source_summaries)
        source_action = "Generated"
    else:
        source_action = "Reused"
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} Phase A source unification\n"
        f"- {source_action} {len(source_summaries)} unified candidate files under candidates/.\n"
        f"- Key files read: opentry_7/opentry_8 candidate JSONL, SymCIF MP-20 validation K<=5 top20_predictions.\n"
        f"- Gate status: source audit complete; CrystaLLM validation K20 anchor still missing."
    )

    source_entries = build_source_manifest(source_summaries)
    write_candidate_manifest(source_entries)

    anchor_mp = load_sample_metrics(OP7 / "eval/crystallm_a_gt_sg_mp_20_test_k20/sample_metrics.jsonl")
    anchor_mpts = load_sample_metrics(OP7 / "eval/crystallm_a_gt_sg_mpts_52_test_k20/sample_metrics.jsonl")
    stable_mp = load_sample_metrics(OP7 / "eval/strategy_stablekey_hybrid_mp_20_test_k20/sample_metrics.jsonl")
    stable_mpts = load_sample_metrics(OP7 / "eval/strategy_stablekey_hybrid_mpts_52_test_k20/sample_metrics.jsonl")
    fusion_mp = load_sample_metrics(OP8 / "eval/strategy_fusion_mp_20_test_k20/sample_metrics.jsonl")
    fusion_mpts = load_sample_metrics(OP8 / "eval/strategy_fusion_mpts_52_test_k20/sample_metrics.jsonl")
    strategy_metrics = {
        "validation_gate": {
            "crystallm_validation_k20_found": False,
            "gate_passed": False,
            "phase_b_allowed": False,
            "reason": "No CrystaLLM-a GT-SG validation K20 candidate bank or per-sample validation metrics found.",
        },
        "standalone_test_metrics": {
            "crystallm_anchor_mp20": summarize_sample_metrics(anchor_mp),
            "crystallm_anchor_mpts52": summarize_sample_metrics(anchor_mpts),
            "stablekey_mp20": summarize_sample_metrics(stable_mp),
            "stablekey_mpts52": summarize_sample_metrics(stable_mpts),
            "strategy_fusion_mp20": summarize_sample_metrics(fusion_mp),
            "strategy_fusion_mpts52": summarize_sample_metrics(fusion_mpts),
        },
        "test_posthoc_overlap": {
            "mp20_anchor_vs_stablekey": overlap_metrics(anchor_mp, stable_mp, k=20),
            "mpts52_anchor_vs_stablekey": overlap_metrics(anchor_mpts, stable_mpts, k=20),
            "mp20_anchor_vs_strategy_fusion": overlap_metrics(anchor_mp, fusion_mp, k=20),
            "mpts52_anchor_vs_strategy_fusion": overlap_metrics(anchor_mpts, fusion_mpts, k=20),
        },
        "internal_rerank_space": {
            "mp_20_test_posthoc": internal_rerank_space(anchor_mp),
            "mpts_52_test_posthoc": internal_rerank_space(anchor_mpts),
        },
    }
    sym = load_symcif_summary()
    write_strategy_reports(strategy_metrics, sym)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} Phase A gate\n"
        "- Reports: reports/strategy_oracle_union_audit.md, reports/crystallm_internal_rerank_space.md.\n"
        "- Main metric: validation oracle union against CrystaLLM K20 could not be computed because the required validation K20 anchor was absent.\n"
        "- Gate: failed/blocked; Phase B selector/ranker and new official test skipped."
    )

    cache_summary = concat_structured_cache()
    write_pure_reports(cache_summary, sym)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} Phase C pure diagnosis\n"
        "- Generated cache/symcif_train.jsonl, cache/symcif_val.jsonl, cache/symcif_test_targets.jsonl.\n"
        "- Reports: reports/symcif_canonicalization_audit.md, reports/pure_gt_wa_geometry_report.md, reports/pure_wa_decoder_val_report.md.\n"
        "- Main metrics: MP-20 GT-WA geometry K<=5 match@1/@5 = 77.16% / 82.94%; baseline WA_hit@1/@5 = 38.63% / 65.11%.\n"
        "- Gate: pure K20 validation gate not passed; official pure test skipped."
    )

    write_final_report(source_entries, strategy_metrics, cache_summary)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} final report\n"
        "- Artifact: final_report.md.\n"
        "- Conclusion: diagnostic success; no meaningful official exceed claimed; no test-leakage tuning performed."
    )


if __name__ == "__main__":
    main()
