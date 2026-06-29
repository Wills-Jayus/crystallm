#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import io
import json
import math
import os
import re
import signal
import sys
import tarfile
import time
import warnings
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from multiprocessing import Pool, Process, Queue
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


warnings.filterwarnings("ignore")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
CACHE = OUT_DIR / "cache"
CANDIDATES = OUT_DIR / "candidates"
OFFICIAL_EVAL = OUT_DIR / "official_eval"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"

CRYSTALLM_DIR = ROOT / "model/CrystaLLM"
if str(CRYSTALLM_DIR) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_DIR))
from crystallm import is_sensible  # noqa: E402


MPTS52_TEST_CSV = CRYSTALLM_DIR / "resources/benchmarks/mpts_52/test.csv"
CRYSTALLM_TAR = ROOT / "model/New_model/opentry_7/generations/pure_crystallm_gt_sg_mpts_52_test.tar.gz"
CRYSTALLM_BASELINE = ROOT / "model/New_model/opentry_7/metrics/pure_crystallm_gt_sg_mpts_52_test_k20.json"
VALIDATION_FROZEN = ROOT / "model/New_model/opentry_11/results/iteration_04b_fixed_hybrid_full_fallback.json"
SYMCIF_GEN = ROOT / "runs/symcif_v5_multidataset_test_full/mpts52/test/generations/v5_fullgen_eval_pool.jsonl"
SYMCIF_MET = ROOT / "runs/symcif_v5_multidataset_test_full/mpts52/test/metrics/v5_fullgen_eval_pool_metrics.jsonl"
SYMCIF_MANIFEST = ROOT / "runs/symcif_v5_multidataset_test_full/mpts52/test/generations/v5_fullgen_eval_pool_manifest.json"

BUDGETS = (1, 5, 20)
FROZEN_PATTERN = "C2S3C15"
FROZEN_SEQUENCE = ["C", "C", "S", "S", "S"] + ["C"] * 15
MAX_CPU_WORKERS = int(os.environ.get("OPENTRY12_MAX_CPU_WORKERS", "4"))


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_opentry12(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_12: {resolved}")
    return resolved


def ensure_report_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = REPORT.resolve()
    if resolved != allowed:
        raise RuntimeError(f"refusing to write report outside requested bundle: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = ensure_opentry12(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_report_once(marker: Path, title: str, body: str) -> None:
    marker = ensure_opentry12(marker)
    if marker.exists():
        return
    report_path = ensure_report_path(REPORT)
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_12 实验：{title}\n\n" + body.strip() + "\n")
    write_json(marker, {"time": now_iso(), "title": title, "report": str(report_path)})


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def clamp_cpu_workers(requested: int) -> int:
    requested = max(1, int(requested))
    cap = max(1, int(MAX_CPU_WORKERS))
    return min(requested, cap)


def raw_atom_site_row_count(cif_text: str) -> int | None:
    lines = cif_text.splitlines()
    in_loop = False
    atom_headers = False
    in_atom_data = False
    count = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower == "loop_":
            if in_atom_data:
                break
            in_loop = True
            atom_headers = False
            continue
        if in_loop and line.startswith("_"):
            if lower.startswith("_atom_site_"):
                atom_headers = True
            elif atom_headers:
                break
            continue
        if in_loop and atom_headers:
            if lower.startswith("data_"):
                break
            in_atom_data = True
            count += 1
            continue
        if in_atom_data:
            break
    return count if count > 0 else None


def declared_sg_number(cif_text: str) -> int | None:
    patterns = [
        r"_symmetry_Int_Tables_number\s+([0-9]+)",
        r"_space_group_IT_number\s+([0-9]+)",
        r"_space_group\.IT_number\s+([0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, cif_text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def load_targets(refresh: bool = False) -> list[dict[str, Any]]:
    cache_path = CACHE / "mpts52_test_targets.jsonl"
    if cache_path.exists() and not refresh:
        with cache_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rows: list[dict[str, Any]] = []
    with MPTS52_TEST_CSV.open("r", encoding="utf-8", newline="") as f:
        for record in csv.DictReader(f):
            material_id = str(record["material_id"]).strip()
            sample_id = f"mpts_52_test_orig__{material_id}"
            cif_text = record["cif"]
            struct = Structure.from_str(cif_text, fmt="cif")
            row_count = raw_atom_site_row_count(cif_text) or len(struct)
            try:
                analyzer = SpacegroupAnalyzer(struct, symprec=0.1)
                sg_number = int(analyzer.get_space_group_number())
                sg_symbol = str(analyzer.get_space_group_symbol())
            except Exception:
                sg_number = declared_sg_number(cif_text)
                sg_symbol = None
            rows.append(
                {
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "cif": cif_text,
                    "row_count": int(row_count),
                    "n_sites": int(len(struct)),
                    "formula": str(struct.composition.reduced_formula),
                    "sg_number": sg_number,
                    "sg_symbol": sg_symbol,
                }
            )
    out = ensure_opentry12(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return rows


def tar_member_mid(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".cif"):
        base = base[:-4]
    return base.rsplit("__", 1)[0]


def tar_member_rank(name: str) -> int:
    base = os.path.basename(name)
    if base.endswith(".cif"):
        base = base[:-4]
    try:
        return int(base.rsplit("__", 1)[1])
    except Exception:
        return 1


def load_crystallm_candidates() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with tarfile.open(CRYSTALLM_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            mid = tar_member_mid(member.name)
            rank = tar_member_rank(member.name)
            groups[mid].append(
                {
                    "source": "C",
                    "source_rank": rank,
                    "generated_text": fh.read().decode("utf-8", errors="replace"),
                    "source_member": member.name,
                    "generated_sha1": None,
                    "generation_score": None,
                    "exact_cover_feasible": None,
                }
            )
    for mid in groups:
        groups[mid].sort(key=lambda r: int(r["source_rank"]))
    return dict(groups)


def load_symcif_candidates(include_diagnostics: bool = True) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SYMCIF_GEN.open("r", encoding="utf-8") as gf, SYMCIF_MET.open("r", encoding="utf-8") as mf:
        for line_no, (gen_line, met_line) in enumerate(zip(gf, mf), start=1):
            if not gen_line.strip():
                continue
            gen = json.loads(gen_line)
            met = json.loads(met_line) if met_line.strip() else {}
            sample_id = str(gen["sample_id"])
            mid = sample_id.split("__")[-1]
            score_raw = gen.get("generation_score")
            score = float(score_raw) if score_raw is not None else -1.0e30
            gen_index = int(gen.get("gen_index", line_no))
            row = {
                "source": "S",
                "source_rank": None,
                "source_sample_id": sample_id,
                "generated_text": str(gen.get("generated_text") or ""),
                "generated_sha1": gen.get("generated_sha1"),
                "generation_score": score,
                "gen_index": gen_index,
                "geometry_rank": gen.get("geometry_rank"),
                "mode": gen.get("mode"),
                "exact_cover_feasible": bool(met.get("multiplicity_ok")) if met.get("multiplicity_ok") is not None else None,
                "formula_closure_success": gen.get("formula_closure_success"),
            }
            if include_diagnostics:
                row.update(
                    {
                        "diagnostic_skeleton_hit": gen.get("skeleton_hit"),
                        "diagnostic_wa_hit": gen.get("wa_hit"),
                        "diagnostic_row_count_hit": gen.get("row_count_hit"),
                        "precomputed_match_ok": met.get("match_ok"),
                        "precomputed_rms": met.get("rms"),
                        "precomputed_valid": met.get("valid"),
                        "precomputed_formula_ok": met.get("formula_ok"),
                        "precomputed_space_group_ok": met.get("space_group_ok"),
                    }
                )
            groups[mid].append(row)
    for mid, arr in groups.items():
        arr.sort(key=lambda r: (float(r.get("generation_score") or -1.0e30), -int(r.get("gen_index") or 0)), reverse=True)
        for i, row in enumerate(arr, start=1):
            row["source_rank"] = i
    return dict(groups)


def build_route_candidate_records(
    targets: list[dict[str, Any]],
    crystallm: dict[str, list[dict[str, Any]]],
    symcif: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_symcif = 0
    missing_crystallm_samples = 0
    partial_symcif_samples = 0
    source_counts: Counter[str] = Counter()
    for target in targets:
        mid = target["material_id"]
        sample_id = target["sample_id"]
        cands_c = crystallm.get(mid, [])
        cands_s = symcif.get(mid, [])
        if not cands_c:
            missing_crystallm_samples += 1
        use_full_crystallm_fallback = not bool(cands_s)
        if use_full_crystallm_fallback:
            missing_symcif += 1
        elif len(cands_s) < 3:
            partial_symcif_samples += 1
        c_idx = 0
        s_idx = 0
        sequence = ["C"] * 20 if use_full_crystallm_fallback else FROZEN_SEQUENCE
        for route_rank, requested_source in enumerate(sequence, start=1):
            source = requested_source
            cand: dict[str, Any] | None = None
            if requested_source == "S" and s_idx < len(cands_s):
                cand = cands_s[s_idx]
                s_idx += 1
            else:
                source = "C"
                if c_idx < len(cands_c):
                    cand = cands_c[c_idx]
                c_idx += 1
            if cand is None:
                cand = {
                    "source": source,
                    "source_rank": None,
                    "generated_text": "",
                    "generated_sha1": None,
                    "generation_score": None,
                    "exact_cover_feasible": None,
                }
            source_counts[source] += 1
            record = {
                "dataset": "mpts_52",
                "split": "test",
                "sample_id": sample_id,
                "material_id": mid,
                "route": FROZEN_PATTERN,
                "route_rank": route_rank,
                "requested_source": requested_source,
                "source": source,
                "source_rank": cand.get("source_rank"),
                "source_member": cand.get("source_member"),
                "source_sample_id": cand.get("source_sample_id"),
                "generated_sha1": cand.get("generated_sha1"),
                "generation_score": cand.get("generation_score"),
                "gen_index": cand.get("gen_index"),
                "geometry_rank": cand.get("geometry_rank"),
                "mode": cand.get("mode"),
                "exact_cover_feasible": cand.get("exact_cover_feasible"),
                "formula_closure_success": cand.get("formula_closure_success"),
                "generated_text": cand.get("generated_text") or "",
            }
            records.append(record)
    manifest = {
        "time": now_iso(),
        "route": FROZEN_PATTERN,
        "sequence": FROZEN_SEQUENCE,
        "samples": len(targets),
        "route_records": len(records),
        "missing_symcif_samples_full_crystallm_fallback": missing_symcif,
        "partial_symcif_samples_slot_fallback": partial_symcif_samples,
        "missing_crystallm_samples": missing_crystallm_samples,
        "source_counts": dict(source_counts),
        "crystallm_source": str(CRYSTALLM_TAR),
        "symcif_generation_source": str(SYMCIF_GEN),
        "symcif_metric_source_for_nonrouting_diagnostics": str(SYMCIF_MET),
        "routing_inputs": [
            "fixed source sequence C2S3C15",
            "CrystaLLM original candidate order",
            "SymCIF fullgen pool generation_score order with gen_index tie-break",
            "sample-level fallback to CrystaLLM when SymCIF has no official candidate",
        ],
        "not_used_for_routing": [
            "match",
            "rms",
            "StructureMatcher result",
            "test feedback",
            "GT-WA",
            "GT-skeleton",
            "diagnostic skeleton_hit/wa_hit labels",
        ],
    }
    return records, manifest


def write_route_candidates(records: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    out_path = ensure_opentry12(CANDIDATES / "c2s3c15_mpts52_test_k20.jsonl.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = dict(manifest)
    manifest["candidate_file"] = str(out_path)
    write_json(CANDIDATES / "c2s3c15_mpts52_test_k20.manifest.json", manifest)
    return out_path


def create_frozen_audit(route_manifest: dict[str, Any], eval_name: str) -> dict[str, Any]:
    validation = read_json(VALIDATION_FROZEN)
    baseline = read_json(CRYSTALLM_BASELINE)
    source_checks = {
        "mpts52_test_csv_exists": MPTS52_TEST_CSV.exists(),
        "crystallm_tar_exists": CRYSTALLM_TAR.exists(),
        "crystallm_baseline_exists": CRYSTALLM_BASELINE.exists(),
        "validation_frozen_result_exists": VALIDATION_FROZEN.exists(),
        "symcif_generation_exists": SYMCIF_GEN.exists(),
        "symcif_metrics_exists": SYMCIF_MET.exists(),
        "symcif_manifest_exists": SYMCIF_MANIFEST.exists(),
    }
    audit_pass = all(source_checks.values()) and route_manifest["route"] == FROZEN_PATTERN
    audit = {
        "time": now_iso(),
        "experiment": "experiment_1_c2s3c15_frozen_official_preaudit",
        "audit_pass": bool(audit_pass),
        "frozen_route": FROZEN_PATTERN,
        "sequence": FROZEN_SEQUENCE,
        "source_checks": source_checks,
        "route_manifest": route_manifest,
        "validation_frozen_source": str(VALIDATION_FROZEN),
        "validation_frozen_best": validation.get("best"),
        "official_baseline_source": str(CRYSTALLM_BASELINE),
        "official_baseline_all": baseline.get("all"),
        "official_baseline_rows_ge7": baseline.get("rows_ge7"),
        "leakage_audit": {
            "uses_fixed_candidate_order_only": True,
            "uses_match_or_rmsd_for_routing": False,
            "uses_structurematcher_label_for_routing": False,
            "uses_test_feedback_for_routing": False,
            "uses_gt_wa_or_gt_skeleton_for_routing": False,
            "ratio_search_after_freeze": False,
            "official_result_can_change_route": False,
        },
        "official_once_policy": {
            "allowed_route": FROZEN_PATTERN,
            "default_refuse_if_summary_exists": True,
            "summary_path": str(OFFICIAL_EVAL / eval_name / "summary.json"),
        },
    }
    write_json(RESULTS / "experiment_1_c2s3c15_frozen_official_audit.json", audit)
    return audit


class CandidateTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise CandidateTimeout("candidate StructureMatcher timeout")


def safe_rms(matcher: StructureMatcher, pred: Structure, gt: Structure, timeout_s: float) -> tuple[float | None, bool, str | None]:
    old_handler = None
    if timeout_s > 0:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        rms = matcher.get_rms_dist(pred, gt)
        if rms is None:
            return None, False, None
        return float(rms[0]), False, None
    except CandidateTimeout:
        return None, True, "rms_timeout"
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"
    finally:
        if timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler or signal.SIG_DFL)


def eval_sample(task: tuple[dict[str, Any], list[dict[str, Any]], float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target, candidates, timeout_s = task
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    sample_id = target["sample_id"]
    material_id = target["material_id"]
    gt = Structure.from_str(target["cif"], fmt="cif")
    target_formula = str(target["formula"])
    target_sg = target.get("sg_number")
    cand_rows: list[dict[str, Any]] = []
    rms_by_rank: list[float | None] = []
    match_by_rank: list[bool] = []
    for cand in candidates[:20]:
        cif = cand.get("generated_text") or ""
        parse_success = False
        valid = False
        formula_ok = False
        sg_ok = False
        rms = None
        matcher_timeout = False
        error = None
        try:
            pred = Structure.from_str(cif, fmt="cif")
            parse_success = True
            try:
                valid = bool(is_sensible(cif, 0.5, 1000.0, 10.0, 170.0))
            except Exception:
                valid = False
            try:
                formula_ok = str(pred.composition.reduced_formula) == target_formula
            except Exception:
                formula_ok = False
            pred_sg = declared_sg_number(cif)
            if pred_sg is None:
                try:
                    pred_sg = int(SpacegroupAnalyzer(pred, symprec=0.1).get_space_group_number())
                except Exception:
                    pred_sg = None
            sg_ok = bool(target_sg is not None and pred_sg == int(target_sg))
            rms, matcher_timeout, error = safe_rms(matcher, pred, gt, timeout_s)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        match_ok = rms is not None
        rms_by_rank.append(rms)
        match_by_rank.append(match_ok)
        cand_rows.append(
            {
                "sample_id": sample_id,
                "material_id": material_id,
                "route": FROZEN_PATTERN,
                "route_rank": cand.get("route_rank"),
                "source": cand.get("source"),
                "requested_source": cand.get("requested_source"),
                "source_rank": cand.get("source_rank"),
                "generated_sha1": cand.get("generated_sha1"),
                "generation_score": cand.get("generation_score"),
                "exact_cover_feasible": cand.get("exact_cover_feasible"),
                "parse_success": parse_success,
                "valid": valid,
                "formula_ok": formula_ok,
                "space_group_ok": sg_ok,
                "match_ok": match_ok,
                "rms": rms,
                "matcher_timeout": matcher_timeout,
                "error": error,
            }
        )
    while len(rms_by_rank) < 20:
        rms_by_rank.append(None)
        match_by_rank.append(False)
    top_hits = {}
    for k in BUDGETS:
        vals = [r for r in rms_by_rank[:k] if r is not None]
        top_hits[f"match@{k}"] = bool(vals)
        top_hits[f"RMSE@{k}"] = float(min(vals)) if vals else None
    best_rms = None
    best_idx = None
    for idx, rms in enumerate(rms_by_rank):
        if rms is None:
            continue
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_idx = idx
    sample_row = {
        "sample_id": sample_id,
        "material_id": material_id,
        "row_count": int(target["row_count"]),
        "n_sites": int(target["n_sites"]),
        "num_candidates_present": min(len(candidates), 20),
        "best_match_index": best_idx,
        "best_rms": best_rms,
        **top_hits,
    }
    return sample_row, cand_rows


def sample_timeout_output(
    task: tuple[dict[str, Any], list[dict[str, Any]], float],
    reason: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target, candidates, _timeout_s = task
    sample_row = {
        "sample_id": target["sample_id"],
        "material_id": target["material_id"],
        "row_count": int(target["row_count"]),
        "n_sites": int(target["n_sites"]),
        "num_candidates_present": min(len(candidates), 20),
        "best_match_index": None,
        "best_rms": None,
        "sample_timeout": True,
        "timeout_reason": reason,
    }
    for k in BUDGETS:
        sample_row[f"match@{k}"] = False
        sample_row[f"RMSE@{k}"] = None
    cand_rows: list[dict[str, Any]] = []
    for cand in candidates[:20]:
        cand_rows.append(
            {
                "sample_id": target["sample_id"],
                "material_id": target["material_id"],
                "route": FROZEN_PATTERN,
                "route_rank": cand.get("route_rank"),
                "source": cand.get("source"),
                "requested_source": cand.get("requested_source"),
                "source_rank": cand.get("source_rank"),
                "generated_sha1": cand.get("generated_sha1"),
                "generation_score": cand.get("generation_score"),
                "exact_cover_feasible": cand.get("exact_cover_feasible"),
                "parse_success": False,
                "valid": False,
                "formula_ok": False,
                "space_group_ok": False,
                "match_ok": False,
                "rms": None,
                "matcher_timeout": True,
                "error": reason,
            }
        )
    return sample_row, cand_rows


def _eval_sample_child(task: tuple[dict[str, Any], list[dict[str, Any]], float], queue: Queue) -> None:
    try:
        queue.put(eval_sample(task))
    except Exception as exc:  # noqa: BLE001
        queue.put(sample_timeout_output(task, f"child_exception: {type(exc).__name__}: {exc}"))


def finish_sample_process(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proc: Process = item["proc"]
    queue: Queue = item["queue"]
    task = item["task"]
    try:
        if proc.exitcode not in (0, None):
            return sample_timeout_output(task, f"sample process exitcode {proc.exitcode}")
        return queue.get_nowait()
    except Exception:
        return sample_timeout_output(task, "sample process returned no row")
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass


def run_isolated_task_stream(
    tasks: list[tuple[dict[str, Any], list[dict[str, Any]], float]],
    workers: int,
    sample_timeout_s: float,
):
    pending = deque(tasks)
    active: list[dict[str, Any]] = []
    workers = max(1, int(workers))
    sample_timeout_s = float(sample_timeout_s)
    while pending or active:
        while pending and len(active) < workers:
            task = pending.popleft()
            queue: Queue = Queue(maxsize=1)
            proc = Process(target=_eval_sample_child, args=(task, queue))
            proc.start()
            active.append({"task": task, "queue": queue, "proc": proc, "started": time.monotonic()})

        made_progress = False
        for item in list(active):
            proc: Process = item["proc"]
            expired = sample_timeout_s > 0 and (time.monotonic() - float(item["started"])) >= sample_timeout_s
            if proc.is_alive() and not expired:
                continue
            if expired and proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
                try:
                    item["queue"].close()
                    item["queue"].join_thread()
                except Exception:
                    pass
                result = sample_timeout_output(item["task"], "sample process timeout")
            else:
                proc.join(0)
                result = finish_sample_process(item)
            active.remove(item)
            made_progress = True
            yield result
        if not made_progress and active:
            time.sleep(0.1)


def load_route_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[str(row["sample_id"])].append(row)
    for sid in groups:
        groups[sid].sort(key=lambda r: int(r["route_rank"]))
    return dict(groups)


def load_symcif_diagnostic_map() -> dict[tuple[str, str | None], dict[str, Any]]:
    diag: dict[tuple[str, str | None], dict[str, Any]] = {}
    with SYMCIF_GEN.open("r", encoding="utf-8") as gf:
        for line in gf:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("sample_id")), row.get("generated_sha1"))
            diag[key] = {
                "diagnostic_skeleton_hit": row.get("skeleton_hit"),
                "diagnostic_wa_hit": row.get("wa_hit"),
                "diagnostic_row_count_hit": row.get("row_count_hit"),
            }
    return diag


def merge_symcif_diagnostics(cand_row: dict[str, Any], diag_map: dict[tuple[str, str | None], dict[str, Any]]) -> dict[str, Any]:
    if cand_row.get("source") != "S":
        return cand_row
    sample_id = f"mpts_52_test_orig__{cand_row['material_id']}"
    extra = diag_map.get((sample_id, cand_row.get("generated_sha1")))
    if extra:
        cand_row.update(extra)
    return cand_row


def completed_sample_ids(sample_path: Path) -> set[str]:
    if not sample_path.exists():
        return set()
    out: set[str] = set()
    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = str(row.get("sample_id") or "")
            if sid:
                out.add(sid)
    return out


def summarize_sample_rows(sample_rows: list[dict[str, Any]], rows_ge7_only: bool = False) -> dict[str, Any]:
    rows = [r for r in sample_rows if int(r.get("row_count") or 0) >= 7] if rows_ge7_only else list(sample_rows)
    denom = len(rows)
    out: dict[str, Any] = {"samples": denom}
    for k in BUDGETS:
        hits = [r for r in rows if bool(r.get(f"match@{k}"))]
        rms = [float(r[f"RMSE@{k}"]) for r in hits if r.get(f"RMSE@{k}") is not None]
        out[f"match@{k}"] = float(len(hits) / denom) if denom else None
        out[f"RMSE@{k}"] = float(sum(rms) / len(rms)) if rms else None
        out[f"matched_samples_for_RMSE@{k}"] = len(rms)
    out["positive_any"] = int(sum(1 for r in rows if r.get("best_match_index") is not None))
    out["positive_any_rate"] = float(out["positive_any"] / denom) if denom else None
    return out


def iter_gzip_jsonl_lenient(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                break
            if not line:
                break
            if line.strip():
                yield json.loads(line)


def next_candidate_eval_segment(out_dir: Path, done_count: int, resume: bool) -> Path:
    primary = out_dir / "candidate_eval.jsonl.gz"
    if not resume or not primary.exists():
        return primary
    idx = 1
    while True:
        path = out_dir / f"candidate_eval_resume_from_{done_count}_part{idx}.jsonl.gz"
        if not path.exists():
            return path
        idx += 1


def candidate_eval_segments(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("candidate_eval*.jsonl.gz"))


def summarize_candidate_rows(candidate_eval_paths: list[Path]) -> dict[str, Any]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    sym_skeleton_hits = 0
    sym_skeleton_hit_matches = 0
    sym_candidates = 0
    for candidate_eval_path in candidate_eval_paths:
        for row in iter_gzip_jsonl_lenient(candidate_eval_path):
            sid = str(row["sample_id"])
            by_sample[sid].append(row)
            source = str(row.get("source"))
            source_counts[source] += 1
            if source == "S":
                sym_candidates += 1
                if row.get("diagnostic_skeleton_hit") is True:
                    sym_skeleton_hits += 1
                    if row.get("match_ok") is True:
                        sym_skeleton_hit_matches += 1
    for sid in by_sample:
        by_sample[sid].sort(key=lambda r: int(r["route_rank"]))
    out: dict[str, Any] = {
        "evaluated_candidate_records": int(sum(len(v) for v in by_sample.values())),
        "source_counts": dict(source_counts),
        "symcif_candidates": sym_candidates,
        "symcif_skeleton_hit_candidates": sym_skeleton_hits,
        "symcif_skeleton_hit_to_match_conversion": (
            float(sym_skeleton_hit_matches / sym_skeleton_hits) if sym_skeleton_hits else None
        ),
    }
    for k in BUDGETS:
        samples = list(by_sample.values())
        denom = len(samples)
        out[f"valid_any@{k}"] = float(sum(any(bool(r.get("valid")) for r in rows[:k]) for rows in samples) / denom) if denom else None
        out[f"formula_ok_any@{k}"] = float(sum(any(bool(r.get("formula_ok")) for r in rows[:k]) for rows in samples) / denom) if denom else None
        out[f"space_group_ok_any@{k}"] = float(sum(any(bool(r.get("space_group_ok")) for r in rows[:k]) for rows in samples) / denom) if denom else None
        known_exact = [
            any(r.get("exact_cover_feasible") is True for r in rows[:k])
            for rows in samples
        ]
        out[f"known_exact_cover_any@{k}"] = float(sum(known_exact) / denom) if denom else None
    slot_total = 0
    slot_valid = 0
    slot_formula = 0
    slot_sg = 0
    slot_exact_known = 0
    slot_exact_true = 0
    for candidate_eval_path in candidate_eval_paths:
        for row in iter_gzip_jsonl_lenient(candidate_eval_path):
            slot_total += 1
            slot_valid += int(bool(row.get("valid")))
            slot_formula += int(bool(row.get("formula_ok")))
            slot_sg += int(bool(row.get("space_group_ok")))
            if row.get("exact_cover_feasible") is not None:
                slot_exact_known += 1
                slot_exact_true += int(bool(row.get("exact_cover_feasible")))
    out["slot_valid_rate"] = float(slot_valid / slot_total) if slot_total else None
    out["slot_formula_consistency_rate"] = float(slot_formula / slot_total) if slot_total else None
    out["slot_sg_consistency_rate"] = float(slot_sg / slot_total) if slot_total else None
    out["symcif_slot_exact_cover_feasible_rate"] = float(slot_exact_true / slot_exact_known) if slot_exact_known else None
    out["candidate_eval_segments"] = [str(p) for p in candidate_eval_paths]
    return out


def evaluate_official_once(
    route_path: Path,
    targets: list[dict[str, Any]],
    eval_name: str,
    workers: int,
    timeout_s: float,
    sample_timeout_s: float,
    resume: bool,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", eval_name):
        raise ValueError(f"unsafe eval_name: {eval_name!r}")
    out_dir = ensure_opentry12(OFFICIAL_EVAL / eval_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise RuntimeError(f"official summary already exists; refusing second official evaluation: {summary_path}")

    sample_path = out_dir / "sample_metrics.jsonl"
    done = completed_sample_ids(sample_path) if resume else set()
    candidate_eval_path = next_candidate_eval_segment(out_dir, len(done), resume=resume)
    route_groups = load_route_candidates(route_path)
    target_by_id = {row["sample_id"]: row for row in targets}
    tasks = [
        (target_by_id[sid], route_groups.get(sid, []), timeout_s)
        for sid in sorted(target_by_id)
        if sid not in done
    ]
    diag_map = load_symcif_diagnostic_map()
    sample_mode = "a" if resume and sample_path.exists() else "w"
    cand_mode = "wt"
    processed = len(done)
    started = time.time()
    with sample_path.open(sample_mode, encoding="utf-8") as sf, gzip.open(candidate_eval_path, cand_mode, encoding="utf-8") as cf:
        if sample_timeout_s > 0:
            for sample_row, cand_rows in run_isolated_task_stream(tasks, workers=max(1, int(workers)), sample_timeout_s=sample_timeout_s):
                sf.write(json.dumps(sample_row, sort_keys=True, ensure_ascii=False) + "\n")
                for cand_row in cand_rows:
                    cand_row = merge_symcif_diagnostics(cand_row, diag_map)
                    cf.write(json.dumps(cand_row, sort_keys=True, ensure_ascii=False) + "\n")
                sf.flush()
                cf.flush()
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - started
                    print(f"[official-eval] processed={processed}/{len(targets)} elapsed_s={elapsed:.1f}", flush=True)
                    gc.collect()
        elif workers > 1:
            with Pool(processes=workers, maxtasksperchild=100) as pool:
                for sample_row, cand_rows in pool.imap_unordered(eval_sample, tasks, chunksize=1):
                    sf.write(json.dumps(sample_row, sort_keys=True, ensure_ascii=False) + "\n")
                    for cand_row in cand_rows:
                        cand_row = merge_symcif_diagnostics(cand_row, diag_map)
                        cf.write(json.dumps(cand_row, sort_keys=True, ensure_ascii=False) + "\n")
                    sf.flush()
                    cf.flush()
                    processed += 1
                    if processed % 100 == 0:
                        elapsed = time.time() - started
                        print(f"[official-eval] processed={processed}/{len(targets)} elapsed_s={elapsed:.1f}", flush=True)
                        gc.collect()
        else:
            for task in tasks:
                sample_row, cand_rows = eval_sample(task)
                sf.write(json.dumps(sample_row, sort_keys=True, ensure_ascii=False) + "\n")
                for cand_row in cand_rows:
                    cand_row = merge_symcif_diagnostics(cand_row, diag_map)
                    cf.write(json.dumps(cand_row, sort_keys=True, ensure_ascii=False) + "\n")
                processed += 1
                if processed % 100 == 0:
                    print(f"[official-eval] processed={processed}/{len(targets)}", flush=True)

    sample_rows: list[dict[str, Any]] = []
    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_rows.append(json.loads(line))
    sample_rows.sort(key=lambda r: str(r["sample_id"]))
    if len(sample_rows) != len(targets):
        raise RuntimeError(f"incomplete official eval: {len(sample_rows)} of {len(targets)} samples")

    baseline = read_json(CRYSTALLM_BASELINE)
    all_summary = summarize_sample_rows(sample_rows, rows_ge7_only=False)
    rows7_summary = summarize_sample_rows(sample_rows, rows_ge7_only=True)
    candidate_eval_paths = candidate_eval_segments(out_dir)
    cand_summary = summarize_candidate_rows(candidate_eval_paths)
    delta_all = {f"match@{k}": all_summary[f"match@{k}"] - baseline["all"][f"match@{k}"] for k in BUDGETS}
    delta_rows7 = {f"match@{k}": rows7_summary[f"match@{k}"] - baseline["rows_ge7"][f"match@{k}"] for k in BUDGETS}
    official = {
        "time": now_iso(),
        "experiment": "experiment_1_c2s3c15_once_only_official",
        "route": FROZEN_PATTERN,
        "sequence": FROZEN_SEQUENCE,
        "dataset": "mpts_52",
        "split": "test",
        "candidate_source": str(route_path),
        "eval_name": eval_name,
        "sample_metrics": str(sample_path),
        "candidate_eval": [str(p) for p in candidate_eval_paths],
        "samples": len(sample_rows),
        "all": all_summary,
        "rows_ge7": rows7_summary,
        "candidate_quality": cand_summary,
        "baseline_source": str(CRYSTALLM_BASELINE),
        "baseline_all": baseline["all"],
        "baseline_rows_ge7": baseline["rows_ge7"],
        "delta_vs_crystallm_baseline_all": delta_all,
        "delta_vs_crystallm_baseline_rows_ge7": delta_rows7,
        "structure_matcher": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10},
        "candidate_timeout_seconds": timeout_s,
        "sample_timeout_seconds": sample_timeout_s,
        "official_once_policy": "This summary is the one-shot frozen official evaluation for C2S3C15. Do not tune route from this result.",
        "contribution_boundary": "auxiliary hybrid route only; not paper main method",
    }
    write_json(summary_path, official)
    write_json(RESULTS / "experiment_1_c2s3c15_official.json", official)
    return official


def fmt_match(summary: dict[str, Any]) -> str:
    return " / ".join(pct(summary.get(f"match@{k}")) for k in BUDGETS)


def fmt_rmse(summary: dict[str, Any]) -> str:
    return " / ".join("NA" if summary.get(f"RMSE@{k}") is None else f"{float(summary[f'RMSE@{k}']):.6f}" for k in BUDGETS)


def fmt_delta(delta: dict[str, Any]) -> str:
    return " / ".join(pp(delta.get(f"match@{k}")) for k in BUDGETS)


def write_experiment1_report(audit: dict[str, Any], official: dict[str, Any]) -> None:
    cq = official["candidate_quality"]
    body = f"""
时间：{official['time']}

实验逻辑：把 validation full 上已经达标的 `C2S3C15` 完全冻结后，只做 official 前审计与一次性 MPTS-52 official full-test。这个实验不是继续调比例，也不是训练新模型；它只回答 validation 上 match@5/match@20 的 +5pp 是否能泛化到 official。

为什么做：`C2S3C15` 在 MPTS-52 validation full fallback 上达到 match@5 +5.340pp、match@20 +5.020pp，rows>=7 match@5/match@20 也有明显提升。但它是 auxiliary hybrid route，必须用 frozen protocol 验证，不能根据 official 结果反向调 C/S 比例。

核心假设：CrystaLLM top20 与 SymCIF fullgen pool 的 coverage 互补能迁移到 official；固定顺序 `C1,C2,S1,S2,S3,C3...C17` 在不使用 match/rmsd/StructureMatcher label/test feedback/GT-WA/GT-skeleton 的情况下仍能提高 K5/K20。

数据规模：MPTS-52 official test {official['samples']} samples；rows>=7 samples={official['rows_ge7']['samples']}。route candidate records={cq['evaluated_candidate_records']}；source counts={cq['source_counts']}。SymCIF 缺失样本按审计统一回退 CrystaLLM，缺失数={audit['route_manifest']['missing_symcif_samples_full_crystallm_fallback']}。

baseline：CrystaLLM GT-SG official K20 = {fmt_match(official['baseline_all'])}；rows>=7 = {fmt_match(official['baseline_rows_ge7'])}。baseline RMSE@1/5/20 = {fmt_rmse(official['baseline_all'])}；rows>=7 RMSE = {fmt_rmse(official['baseline_rows_ge7'])}。

方法变化：只构造 frozen `C2S3C15` 候选序列；CrystaLLM 保持原始顺序，SymCIF 使用 fullgen pool 的 generation_score 固定排序，缺失 SymCIF artifact 的样本回退 CrystaLLM C20。未做 ratio search、threshold tuning、scorer、RF/HGB 或任何 official feedback 调整。

审计结果：audit_pass={audit['audit_pass']}；routing 只依赖固定候选顺序和 generation_score；不使用 match/rmsd/StructureMatcher result/test label/GT-WA/GT-skeleton。

official 结果：`C2S3C15` overall = {fmt_match(official['all'])}；delta = {fmt_delta(official['delta_vs_crystallm_baseline_all'])}；RMSE@1/5/20 = {fmt_rmse(official['all'])}。

rows>=7 结果：`C2S3C15` rows>=7 = {fmt_match(official['rows_ge7'])}；delta = {fmt_delta(official['delta_vs_crystallm_baseline_rows_ge7'])}；rows>=7 RMSE@1/5/20 = {fmt_rmse(official['rows_ge7'])}。

valid/formula/SG/exact-cover/skeleton-to-match：slot_valid_rate={pct(cq['slot_valid_rate'])}；slot_formula_consistency_rate={pct(cq['slot_formula_consistency_rate'])}；slot_sg_consistency_rate={pct(cq['slot_sg_consistency_rate'])}；SymCIF slot exact-cover feasible={pct(cq['symcif_slot_exact_cover_feasible_rate'])}；SymCIF skeleton-hit-to-match conversion={pct(cq['symcif_skeleton_hit_to_match_conversion'])}。

可信度：这是一次性 official full-test，候选路由先冻结并有审计记录；可信度高于 validation fallback。限制是该路线仍为 candidate fusion / auxiliary hybrid，不生成新 skeleton，也不修 geometry，不能作为论文主方法。

和历史实验关系：直接承接 opentry_11 迭代 04B 的 validation full result；本实验只验证 frozen route 是否泛化，不能反向修改后续路线。若 official 未达标，说明 validation hybrid 不泛化；若 official 达标，也只能写成 auxiliary hybrid result。

最终判决：`C2S3C15` official 结果只能作为 auxiliary hybrid route 判决，不能包装成 SymCIF/Wyckoff/geometry repair 主方法。后续实验必须回到 exact-cover constrained skeleton generation 与 symmetry-preserving learned geometry repair。

下一步：做实验 2 的收益归因，明确 SymCIF top3 救回哪些 CrystaLLM top20 失败样本、rows>=7 是否是主要来源、K1 是否牺牲，以及 invalid/formula/SG/RMSE 是否恶化。
"""
    append_report_once(RESULTS / "experiment_1_report_appended.json", "实验 1 C2S3C15 frozen official 前审计与一次性 official 验证", body)


def run_experiment1(args: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    targets = load_targets(refresh=args.refresh_targets)
    crystallm = load_crystallm_candidates()
    symcif = load_symcif_candidates(include_diagnostics=False)
    route_records, route_manifest = build_route_candidate_records(targets, crystallm, symcif)
    route_path = write_route_candidates(route_records, route_manifest)
    audit = create_frozen_audit(route_manifest, eval_name=args.eval_name)
    print(json.dumps({"audit_pass": audit["audit_pass"], "route_path": str(route_path), "route_manifest": route_manifest}, indent=2, ensure_ascii=False))
    if args.audit_only:
        return
    if not audit["audit_pass"]:
        raise RuntimeError("frozen audit failed; refusing official evaluation")
    official = evaluate_official_once(
        route_path,
        targets,
        eval_name=args.eval_name,
        workers=clamp_cpu_workers(args.workers),
        timeout_s=args.candidate_timeout_seconds,
        sample_timeout_s=args.sample_timeout_seconds,
        resume=args.resume,
    )
    write_experiment1_report(audit, official)
    print(json.dumps({"official_all": official["all"], "official_rows_ge7": official["rows_ge7"], "delta_all": official["delta_vs_crystallm_baseline_all"]}, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="opentry_12 CrystaLLM/SymCIF experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("experiment1")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--candidate-timeout-seconds", type=float, default=8.0)
    p.add_argument("--sample-timeout-seconds", type=float, default=180.0)
    p.add_argument("--refresh-targets", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--eval-name", default="c2s3c15_mpts52_test")
    p.set_defaults(func=run_experiment1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
