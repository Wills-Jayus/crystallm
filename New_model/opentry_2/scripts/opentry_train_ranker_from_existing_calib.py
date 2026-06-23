#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import opentry_validation_ranker_eval_gt_sg as ranker  # noqa: E402


def _ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def _write_json(path: Path, obj: Any) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_true_tar(tar_path: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir = _ensure_under_opentry(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".cif"):
                continue
            name = Path(member.name).name
            mid = name[:-4]
            dest = out_dir / name
            if not dest.is_file():
                src = tar.extractfile(member)
                if src is None:
                    continue
                dest.write_bytes(src.read())
            out[mid] = dest
    return out


def _candidate_index(path: Path) -> Optional[Tuple[str, int]]:
    m = ranker._GEN_RE.match(path.name)
    if m is None:
        return None
    return m.group("mid"), int(m.group("idx"))


def _group_candidate_dir(candidate_dir: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(candidate_dir.glob("*.cif")):
        parsed = _candidate_index(path)
        if parsed is None:
            continue
        mid, _idx = parsed
        grouped.setdefault(mid, []).append(path)
    return grouped


def _label_one(path: Path, true_cif: str, matcher: StructureMatcher, max_sites: int) -> Tuple[int, Optional[float]]:
    pred = ranker._parse_struct(ranker._read_text(path))
    gt = ranker._parse_struct(true_cif)
    if pred is None or gt is None:
        return 0, None
    try:
        if int(max_sites) > 0 and (int(pred.num_sites) > int(max_sites) or int(gt.num_sites) > int(max_sites)):
            return 0, None
    except Exception:
        pass
    try:
        rms = matcher.get_rms_dist(pred, gt)
    except Exception:
        return 0, None
    if rms is None:
        return 0, None
    return 1, float(rms[0])


def cmd_label_sample(args: argparse.Namespace) -> int:
    mid = str(args.mid)
    true_path = Path(args.true_dir).expanduser().resolve() / f"{mid}.cif"
    if not true_path.is_file():
        raise SystemExit(f"missing true CIF for {mid}: {true_path}")
    true_cif = ranker._read_text(true_path)
    row = ranker._row_from_prepared_cif(mid, true_cif)
    if row is None:
        raise SystemExit(f"unusable true CIF for {mid}: {true_path}")
    paths = sorted(
        [p for p in Path(args.candidate_dir).expanduser().resolve().glob(f"{mid}__*.cif") if _candidate_index(p) is not None],
        key=lambda p: _candidate_index(p)[1],
    )
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    records: List[Dict[str, Any]] = []
    for path in paths:
        cf = ranker._candidate_features(path, row)
        label, rms = _label_one(path, true_cif, matcher, max_sites=int(args.max_sites))
        rec = {
            "id": mid,
            "idx": int(cf.idx),
            "path": str(path),
            "prompt_path": str((Path(args.prompts_dir).expanduser().resolve() / f"{mid}.txt")),
            "label": int(label),
            "rms": rms,
            "features": {name: float(val) for name, val in zip(cf.names, cf.values)},
            "meta": cf.meta,
        }
        records.append(rec)
    _write_jsonl(Path(args.out), records)
    return 0


def _run_sample_label(
    *,
    mid: str,
    args: argparse.Namespace,
    out_file: Path,
) -> Dict[str, Any]:
    if out_file.is_file() and bool(args.resume):
        return {"mid": mid, "status": "cached"}
    cmd = [
        str(sys.executable),
        str(Path(__file__).resolve()),
        "label-sample",
        "--mid",
        mid,
        "--candidate-dir",
        str(Path(args.candidate_dir).expanduser().resolve()),
        "--true-dir",
        str(Path(args.true_dir).expanduser().resolve()),
        "--prompts-dir",
        str(Path(args.prompts_dir).expanduser().resolve()),
        "--out",
        str(out_file),
        "--max-sites",
        str(int(args.max_sites)),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(args.sample_timeout_seconds),
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {"mid": mid, "status": "failed", "returncode": int(proc.returncode), "stdout_tail": (proc.stdout or "")[-2000:]}
    return {"mid": mid, "status": "ok"}


def _load_records(sample_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(sample_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _train_rf(records: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    names = ranker._feature_names()
    X = np.array([[float((rec.get("features") or {}).get(name, 0.0)) for name in names] for rec in records], dtype=float)
    y = np.array([int(rec.get("label", 0)) for rec in records], dtype=int)
    if X.size == 0:
        raise SystemExit("no candidate records loaded")
    clf = RandomForestClassifier(
        n_estimators=int(args.rf_estimators),
        max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
        min_samples_leaf=int(args.rf_min_samples_leaf),
        class_weight="balanced_subsample",
        random_state=int(args.seed),
        n_jobs=int(args.rf_jobs),
    )
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    pred = (scores >= 0.5).astype(int)
    importances = getattr(clf, "feature_importances_", np.zeros(X.shape[1], dtype=float))
    summary = {
        "ranker_type": "random_forest",
        "feature_names": list(names),
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(float(y.sum()) / max(1.0, float(len(y)))),
        "train_acc": float(accuracy_score(y, pred)),
        "rf_estimators": int(args.rf_estimators),
        "rf_max_depth": None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
        "rf_min_samples_leaf": int(args.rf_min_samples_leaf),
        "feature_importances": {name: float(val) for name, val in zip(names, importances.tolist())},
    }
    return clf, summary


def cmd_label_train(args: argparse.Namespace) -> int:
    out_dir = _ensure_under_opentry(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    true_dir = out_dir / "true_cifs"
    sample_dir = out_dir / "sample_labels"
    sample_dir.mkdir(parents=True, exist_ok=True)
    true_paths = _extract_true_tar(Path(args.true_tar).expanduser().resolve(), true_dir)
    args.true_dir = str(true_dir)

    grouped = _group_candidate_dir(Path(args.candidate_dir).expanduser().resolve())
    allowed_ids: Optional[set[str]] = None
    if str(args.id_file or ""):
        allowed_ids = {
            line.strip()
            for line in Path(args.id_file).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    mids = sorted(mid for mid in grouped if mid in true_paths and (allowed_ids is None or mid in allowed_ids))
    if int(args.limit_ids) > 0:
        mids = mids[: int(args.limit_ids)]
    print(f"[label] candidate ids={len(grouped)} true ids={len(true_paths)} selected ids={len(mids)}", flush=True)

    statuses: List[Dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futures = {
            ex.submit(_run_sample_label, mid=mid, args=args, out_file=sample_dir / f"{mid}.jsonl"): mid
            for mid in mids
        }
        done = 0
        for fut in cf.as_completed(futures):
            done += 1
            try:
                status = fut.result()
            except subprocess.TimeoutExpired as exc:
                status = {"mid": futures[fut], "status": "timeout", "seconds": float(args.sample_timeout_seconds), "stdout_tail": str(exc)[-1000:]}
            except Exception as exc:  # noqa: BLE001
                status = {"mid": futures[fut], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            statuses.append(status)
            if done % max(1, int(args.progress_interval)) == 0:
                ok = sum(1 for s in statuses if s.get("status") in {"ok", "cached"})
                print(f"[label] {done}/{len(mids)} done ok/cached={ok}", flush=True)

    records = _load_records(sample_dir)
    all_jsonl = out_dir / "ranker_candidates_all.jsonl"
    _write_jsonl(all_jsonl, records)
    clf, summary = _train_rf(records, args)
    (out_dir / "ranker_model.pkl").write_bytes(pickle.dumps(clf))
    _write_json(out_dir / "ranker_summary.json", summary)
    run_summary = {
        "selected_ids": len(mids),
        "status_counts": {k: sum(1 for s in statuses if str(s.get("status")) == k) for k in sorted({str(s.get("status")) for s in statuses})},
        "candidate_records": len(records),
        "n_positive": int(summary["n_positive"]),
        "positive_rate": float(summary["positive_rate"]),
        "ranker_model": str((out_dir / "ranker_model.pkl").resolve()),
        "ranker_candidates_all": str(all_jsonl.resolve()),
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "label_status.json", statuses)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a ranker from existing calibration CIFs with per-sample hard timeouts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("label-sample")
    p_sample.add_argument("--mid", required=True)
    p_sample.add_argument("--candidate-dir", required=True)
    p_sample.add_argument("--true-dir", required=True)
    p_sample.add_argument("--prompts-dir", required=True)
    p_sample.add_argument("--out", required=True)
    p_sample.add_argument("--max-sites", type=int, default=512)
    p_sample.set_defaults(func=cmd_label_sample)

    p_train = sub.add_parser("label-train")
    p_train.add_argument("--candidate-dir", required=True)
    p_train.add_argument("--true-tar", required=True)
    p_train.add_argument("--prompts-dir", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--workers", type=int, default=16)
    p_train.add_argument("--sample-timeout-seconds", type=float, default=60.0)
    p_train.add_argument("--max-sites", type=int, default=512)
    p_train.add_argument("--limit-ids", type=int, default=0)
    p_train.add_argument("--id-file", default="")
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--progress-interval", type=int, default=25)
    p_train.add_argument("--rf-estimators", type=int, default=500)
    p_train.add_argument("--rf-max-depth", type=int, default=0)
    p_train.add_argument("--rf-min-samples-leaf", type=int, default=2)
    p_train.add_argument("--rf-jobs", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.set_defaults(func=cmd_label_train)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
