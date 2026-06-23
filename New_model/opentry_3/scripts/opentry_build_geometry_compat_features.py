#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import multiprocessing as mp
import signal
import sys
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
CRYSTALLM_BIN = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/bin").resolve()
if str(CRYSTALLM_BIN) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_BIN))
if str(OPENTRY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(OPENTRY_ROOT / "scripts"))

import benchmark_metrics as bm  # noqa: E402
from pymatgen.analysis.structure_matcher import StructureMatcher  # noqa: E402
from pymatgen.core import Structure  # noqa: E402

import opentry_selfscore_rendered_cifs as selfscore  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_records is not None and len(rows) >= int(max_records):
                break
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class RmsdTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float | None):
    if seconds is None or float(seconds) <= 0:
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise RmsdTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        signal.signal(signal.SIGALRM, old)


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["sample_id"]), []).append(row)
    for values in by_id.values():
        values.sort(key=lambda row: int(row.get("rank", 10**9)))
    return by_id


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if sg <= 2:
        return "triclinic"
    if sg <= 15:
        return "monoclinic"
    if sg <= 74:
        return "orthorhombic"
    if sg <= 142:
        return "tetragonal"
    if sg <= 167:
        return "trigonal"
    if sg <= 194:
        return "hexagonal"
    return "cubic"


def parse_wa_metadata(wa_key: str) -> dict[str, Any]:
    parts = [p for p in str(wa_key or "").split("|") if p]
    orbit_ids: list[str] = []
    elements: list[str] = []
    multiplicities: list[int] = []
    for part in parts:
        if ":" in part:
            orbit, element = part.rsplit(":", 1)
        else:
            orbit, element = part, ""
        orbit_ids.append(orbit)
        if element:
            elements.append(element)
        fields = orbit.split("|") if "|" in orbit else [orbit]
        text = fields[-1] if fields else orbit
        mult = ""
        for ch in text:
            if ch.isdigit():
                mult += ch
            elif mult:
                break
        if mult:
            try:
                multiplicities.append(int(mult))
            except ValueError:
                pass
    duplicate_orbits = len(orbit_ids) - len(set(orbit_ids))
    return {
        "candidate_row_count": len(parts),
        "candidate_unique_orbit_count": len(set(orbit_ids)),
        "candidate_duplicate_orbit_count": duplicate_orbits,
        "candidate_unique_element_count": len(set(elements)),
        "candidate_max_multiplicity": max(multiplicities) if multiplicities else 0,
        "candidate_mean_multiplicity": float(sum(multiplicities) / len(multiplicities)) if multiplicities else 0.0,
    }


def load_sample_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                row = json.loads(text)
                sample_id = row.get("sample_id")
                if sample_id is None and isinstance(row.get("keys"), dict):
                    sample_id = row["keys"].get("sample_id")
                if sample_id is None:
                    sample_id = row.get("material_id")
                if sample_id is not None:
                    allowed.add(str(sample_id))
            else:
                allowed.add(text.split()[0])
    return allowed


def load_repr_metadata(
    repr_jsonl: Path,
    max_records: int | None,
    sample_id_filter: set[str] | None = None,
    min_row_count: int = 0,
) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(repr_jsonl)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["keys"]["sample_id"])
        if sample_id_filter is not None and sample_id not in sample_id_filter:
            continue
        if int(min_row_count) > 0 and int(row["row_count"]) < int(min_row_count):
            continue
        out[sample_id] = {
            "material_id": row["keys"].get("material_id"),
            "sg": int(row["sg"]),
            "atom_count": int(row["atom_count"]),
            "target_row_count": int(row["row_count"]),
            "target_complex_flag": bool(row.get("complex_flag")),
            "target_wa_key": str(row["canonical_wa_key"]),
            "target_skeleton_key": str(row["canonical_skeleton_key"]),
            "formula_element_count": len(row.get("formula_counts") or {}),
        }
        if max_records is not None and len(out) >= int(max_records):
            break
    return out


def normalize_cif(cif: str) -> str:
    normalizer = getattr(bm, "_normalize_cif_symmops_to_declared_sg", None)
    if normalizer is None:
        return cif
    return normalizer(cif)


def match_candidate(
    *,
    matcher: StructureMatcher,
    pred_cif: str,
    true_struct: Structure | None,
    max_sites: int,
    timeout_s: float,
) -> tuple[bool, float | None, str | None]:
    if true_struct is None:
        return False, None, "true_parse_failed"
    try:
        pred = Structure.from_str(normalize_cif(pred_cif), fmt="cif")
    except Exception as exc:  # noqa: BLE001
        return False, None, f"pred_parse_failed:{type(exc).__name__}"
    try:
        if int(pred.num_sites) > int(max_sites) or int(true_struct.num_sites) > int(max_sites):
            return False, None, "skipped_large"
    except Exception:
        pass
    try:
        with time_limit(timeout_s):
            rms = matcher.get_rms_dist(pred, true_struct)
        if rms is None:
            return False, None, None
        return True, float(rms[0]), None
    except RmsdTimeout:
        return False, None, "rmsd_timeout"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"rmsd_error:{type(exc).__name__}"


def build_sample_rows(
    *,
    split: str,
    sample_id: str,
    meta: dict[str, Any],
    candidates: list[dict[str, Any]],
    true_cif: str | None,
    max_sites: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    try:
        true_struct = None if true_cif is None else Structure.from_str(normalize_cif(true_cif), fmt="cif")
    except Exception:
        true_struct = None
    out_rows: list[dict[str, Any]] = []
    for cand in candidates:
        row = dict(cand)
        if row.get("self_min_distance") is None or row.get("self_volume_per_atom") is None:
            row.update(selfscore.cif_self_features(str(row.get("cif") or "")))
        label_match, label_rmsd, label_error = match_candidate(
            matcher=matcher,
            pred_cif=str(row.get("cif") or ""),
            true_struct=true_struct,
            max_sites=int(max_sites),
            timeout_s=float(timeout_s),
        )
        wa_key = str(row.get("canonical_wa_key") or "")
        skel_key = str(row.get("canonical_skeleton_key") or "")
        out_rows.append(
            {
                **row,
                **parse_wa_metadata(wa_key),
                "split": str(split),
                "candidate_uid": f"{sample_id}::rank{int(row.get('rank', 0))}",
                "material_id": meta["material_id"],
                "sg": int(meta["sg"]),
                "crystal_system": crystal_system(int(meta["sg"])),
                "atom_count": int(meta["atom_count"]),
                "formula_element_count": int(meta["formula_element_count"]),
                "target_row_count": int(meta["target_row_count"]),
                "target_complex_flag": bool(meta["target_complex_flag"]),
                "target_wa_key": meta["target_wa_key"],
                "target_skeleton_key": meta["target_skeleton_key"],
                "candidate_wa_hit": wa_key == meta["target_wa_key"],
                "candidate_skeleton_hit": skel_key == meta["target_skeleton_key"],
                "label_match": bool(label_match),
                "label_rmsd": label_rmsd,
                "label_error": label_error,
            }
        )
    return out_rows


def sample_worker(queue: mp.Queue, payload: dict[str, Any]) -> None:
    try:
        rows = build_sample_rows(**payload)
        queue.put({"ok": True, "rows": rows})
    except Exception as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}", "rows": []})


def run_sample_with_timeout(payload: dict[str, Any], sample_timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=sample_worker, args=(queue, payload))
    proc.start()
    proc.join(float(sample_timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"ok": False, "timeout": True, "error": "sample_timeout", "rows": []}
    if not queue.empty():
        return dict(queue.get())
    return {"ok": False, "error": f"empty_worker_result_exitcode_{proc.exitcode}", "rows": []}


def build_sample_rows_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id"))
    try:
        rows = build_sample_rows(**payload)
        return {"sample_id": sample_id, "ok": True, "timeout": False, "error": None, "rows": rows}
    except Exception as exc:  # noqa: BLE001
        return {"sample_id": sample_id, "ok": False, "timeout": False, "error": f"{type(exc).__name__}: {exc}", "rows": []}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = grouped(rows)
    out: dict[str, Any] = {
        "samples": len(by_id),
        "rows": len(rows),
        "positive_rows": sum(int(row["label_match"]) for row in rows),
        "positive_rate": sum(int(row["label_match"]) for row in rows) / max(1, len(rows)),
    }
    for k in (1, 5, 20, 50):
        hits = []
        rms_values = []
        for items in by_id.values():
            top = items[:k]
            matched = [row for row in top if bool(row["label_match"])]
            hits.append(bool(matched))
            if matched:
                rms_values.append(min(float(row["label_rmsd"]) for row in matched if row.get("label_rmsd") is not None))
        out[f"match@{k}"] = sum(int(x) for x in hits) / max(1, len(hits))
        out[f"RMSE@{k}"] = None if not rms_values else float(sum(rms_values) / len(rms_values))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/val rendered-candidate features and StructureMatcher labels for geometry compatibility models.")
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--true-cif-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-candidates-per-sample", type=int, default=50)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    parser.add_argument("--restrict-to-rendered-samples", action="store_true")
    parser.add_argument("--min-row-count", type=int, default=0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_records = None if int(args.max_records) <= 0 else int(args.max_records)
    rendered_by_id = grouped(read_jsonl(args.rendered_jsonl))
    sample_id_filter = load_sample_id_filter(args.sample_ids_file)
    if bool(args.restrict_to_rendered_samples):
        rendered_ids = set(rendered_by_id)
        sample_id_filter = rendered_ids if sample_id_filter is None else sample_id_filter & rendered_ids
    repr_meta = load_repr_metadata(
        args.repr_jsonl,
        max_records,
        sample_id_filter=sample_id_filter,
        min_row_count=int(args.min_row_count),
    )
    sample_ids = list(repr_meta)
    true_cifs: dict[str, str | None] = {}
    for sample_id in sample_ids:
        path = args.true_cif_dir / f"{sample_id}.cif"
        try:
            true_cifs[sample_id] = path.read_text(encoding="utf-8")
        except Exception:
            true_cifs[sample_id] = None

    payloads: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        meta = repr_meta[sample_id]
        candidates = rendered_by_id.get(sample_id, [])[: int(args.max_candidates_per_sample)]
        payloads.append(
            {
                "split": str(args.split),
                "sample_id": sample_id,
                "meta": meta,
                "candidates": candidates,
                "true_cif": true_cifs.get(sample_id),
                "max_sites": int(args.max_sites),
                "timeout_s": float(args.rmsd_timeout_seconds),
            }
        )
    out_rows: list[dict[str, Any]] = []
    sample_status: list[dict[str, Any]] = []
    if int(args.workers) > 1:
        with mp.get_context("fork").Pool(processes=int(args.workers)) as pool:
            for idx, result in enumerate(pool.imap_unordered(build_sample_rows_payload, payloads, chunksize=1), start=1):
                rows = list(result.get("rows") or [])
                sample_status.append(
                    {
                        "sample_id": result.get("sample_id"),
                        "ok": bool(result.get("ok")),
                        "timeout": bool(result.get("timeout", False)),
                        "error": result.get("error"),
                        "rows": len(rows),
                    }
                )
                out_rows.extend(rows)
                if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                    print(f"[features] {args.split} {idx}/{len(sample_ids)} samples, rows={len(out_rows)}", flush=True)
    else:
        for idx, payload in enumerate(payloads, start=1):
            result = run_sample_with_timeout(payload, float(args.sample_timeout_seconds))
            rows = list(result.get("rows") or [])
            sample_status.append(
                {
                    "sample_id": payload["sample_id"],
                    "ok": bool(result.get("ok")),
                    "timeout": bool(result.get("timeout", False)),
                    "error": result.get("error"),
                    "rows": len(rows),
                }
            )
            out_rows.extend(rows)
            if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
                print(f"[features] {args.split} {idx}/{len(sample_ids)} samples, rows={len(out_rows)}", flush=True)

    write_jsonl(out_dir / f"{args.split}_candidate_features.jsonl", out_rows)
    write_jsonl(out_dir / f"{args.split}_sample_label_status.jsonl", sample_status)
    rows_ge_7 = [row for row in out_rows if int(row.get("target_row_count", 0)) >= 7]
    atoms_ge_12 = [row for row in out_rows if int(row.get("atom_count", 0)) >= 12]
    wa_hit = [row for row in out_rows if bool(row.get("candidate_wa_hit"))]
    summary = {
        "split": str(args.split),
        "rendered_jsonl": str(args.rendered_jsonl),
        "repr_jsonl": str(args.repr_jsonl),
        "true_cif_dir": str(args.true_cif_dir),
        "max_records": int(args.max_records),
        "max_candidates_per_sample": int(args.max_candidates_per_sample),
        "sample_ids_file": None if args.sample_ids_file is None else str(args.sample_ids_file),
        "restrict_to_rendered_samples": bool(args.restrict_to_rendered_samples),
        "min_row_count": int(args.min_row_count),
        "selected_samples": len(sample_ids),
        "rendered_samples": len(rendered_by_id),
        "sample_timeout_seconds": float(args.sample_timeout_seconds),
        "workers": int(args.workers),
        "sample_status": {
            "ok": sum(int(row["ok"]) for row in sample_status),
            "timeout": sum(int(row["timeout"]) for row in sample_status),
            "failed": sum(int(not row["ok"]) for row in sample_status),
        },
        "full": summarize(out_rows),
        "rows_ge_7": summarize(rows_ge_7),
        "atoms_ge_12": summarize(atoms_ge_12),
        "candidate_wa_hit_rows": len(wa_hit),
        "candidate_wa_hit_match_fail_rate": None
        if not wa_hit
        else sum(int(not row["label_match"]) for row in wa_hit) / len(wa_hit),
    }
    write_json(out_dir / f"{args.split}_candidate_features_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
