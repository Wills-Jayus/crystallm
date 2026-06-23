#!/usr/bin/env python3
"""Train a validation-supervised CIF candidate ranker and evaluate reranked test candidates.

The ranker is non-oracle for test data: labels are computed only on the calibration/validation
split. At test time it uses GT-free candidate features such as parse success, sensible checks,
density, SG consistency, and composition agreement with the requested formula.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

_ROOT = Path(
    os.environ.get(
        "CRYSTALLM_ROOT",
        "/data/users/xsw/autodlmini/model/scp_task/CrystaLLM",
    )
).resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "bin"))

from crystallm import (  # noqa: E402
    extract_formula_nonreduced,
    extract_space_group_symbol,
    get_atomic_props_block_for_formula,
    is_sensible,
    replace_symmetry_operators,
)
from benchmark_metrics import is_valid  # noqa: E402


_GEN_RE = re.compile(r"^(?P<mid>.+)__(?P<idx>\d+)\.cif$")


class _CandidateLabelTimeout(RuntimeError):
    pass


def _candidate_label_timeout_handler(signum: int, frame: Any) -> None:
    raise _CandidateLabelTimeout("candidate label timed out")


@dataclass(frozen=True)
class RowData:
    material_id: str
    cif: str
    formula: str
    comp: Composition
    data_formula: str


@dataclass(frozen=True)
class CandidateFeatures:
    mid: str
    idx: int
    path: Path
    values: np.ndarray
    names: Tuple[str, ...]
    meta: Dict[str, Any]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_capture(cmd: Sequence[str]) -> str:
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{out}")
    return out


def _parse_metric_dict(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


def _composition_from_cif(cif: str, fallback: str = "") -> Optional[Tuple[str, Composition, str]]:
    raw = ""
    try:
        raw = extract_formula_nonreduced(cif).strip().strip("'\"")
    except Exception:
        raw = ""
    if not raw:
        m = re.search(r"(?m)^\s*data_([A-Za-z0-9().]+)\s*$", cif or "")
        raw = m.group(1) if m else fallback
    if not raw:
        return None
    try:
        comp = Composition(raw.replace(" ", ""))
    except Exception:
        return None
    return raw, comp, comp.formula.replace(" ", "")


def _normalize_cif_symprec(cif: str) -> str:
    try:
        struct = Structure.from_str(cif, fmt="cif")
        return CifWriter(struct=struct, symprec=0.1).__str__()
    except Exception:
        return cif


def _row_from_cif(material_id: str, cif: str) -> Optional[RowData]:
    norm_cif = _normalize_cif_symprec(cif)
    parsed = _composition_from_cif(norm_cif, fallback=material_id)
    if parsed is None:
        return None
    formula, comp, data_formula = parsed
    return RowData(material_id=material_id, cif=norm_cif, formula=formula, comp=comp, data_formula=data_formula)


def _row_from_prepared_cif(material_id: str, cif: str) -> Optional[RowData]:
    parsed = _composition_from_cif(cif, fallback=material_id)
    if parsed is None:
        return None
    formula, comp, data_formula = parsed
    return RowData(material_id=material_id, cif=cif, formula=formula, comp=comp, data_formula=data_formula)


def _load_rows(csv_path: Path, *, start: int, limit: Optional[int]) -> List[RowData]:
    rows: List[RowData] = []
    start_i = max(0, int(start))
    stop_i = None if limit is None else start_i + max(0, int(limit))
    seen = 0
    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            mid = str(row.get("material_id") or "").strip()
            cif = str(row.get("cif") or "")
            if not mid or not cif:
                continue
            if seen < start_i:
                seen += 1
                continue
            if stop_i is not None and seen >= stop_i:
                break
            seen += 1
            row_data = _row_from_cif(mid, cif)
            if row_data is None:
                continue
            rows.append(row_data)
    return rows


def _load_rows_multi(paths: Sequence[Path], *, per_csv_limit: Optional[int], total_limit: Optional[int]) -> List[RowData]:
    out: List[RowData] = []
    seen: set[str] = set()
    for path in paths:
        rows = _load_rows(path, start=0, limit=per_csv_limit)
        for row in rows:
            if row.material_id in seen:
                continue
            seen.add(row.material_id)
            out.append(row)
            if total_limit is not None and len(out) >= int(total_limit):
                return out
    return out


def _replace_rows_from_gt_dir(rows: Sequence[RowData], gt_dir: Path) -> List[RowData]:
    out: List[RowData] = []
    missing: List[str] = []
    for row in rows:
        path = gt_dir / f"{row.material_id}.cif"
        if not path.is_file():
            missing.append(row.material_id)
            continue
        row_data = _row_from_prepared_cif(row.material_id, _read_text(path))
        if row_data is None:
            missing.append(row.material_id)
            continue
        out.append(row_data)
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"missing/unusable GT CIFs for {len(missing)} rows: {preview}")
    return out


def _prompt_base(row: RowData) -> str:
    atom_block = get_atomic_props_block_for_formula(row.data_formula).strip()
    sg = extract_space_group_symbol(row.cif)
    if sg is None:
        raise ValueError(f"missing space group for {row.material_id}")
    sg = str(sg).strip().strip("'\"")
    return f"data_{row.data_formula}\n{atom_block}\n_symmetry_space_group_name_H-M {sg}\n"


def _write_prompts(rows: Sequence[RowData], prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        _write_text(prompt_dir / f"{row.material_id}.txt", _prompt_base(row).rstrip() + "\n")


def _generate_prompt_dir(
    *,
    py: str,
    gen_script: Path,
    model_dir: Path,
    prompts_dir: Path,
    out_dir: Path,
    num_samples: int,
    seed: int,
    sample_seed_stride: int,
    sample_index_offset: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    dtype: str,
    workers: int,
    overwrite: bool,
) -> str:
    cmd = [
        str(py),
        str(gen_script),
        "--model-dir",
        str(model_dir),
        "--prompts-dir",
        str(prompts_dir),
        "--out-dir",
        str(out_dir),
        "--num-samples-per-prompt",
        str(int(num_samples)),
        "--sample-seed-stride",
        str(int(sample_seed_stride)),
        "--sample-index-offset",
        str(int(sample_index_offset)),
        "--seed",
        str(int(seed)),
        "--temperature",
        str(float(temperature)),
        "--top-k",
        str(int(top_k)),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--device",
        str(device),
        "--dtype",
        str(dtype),
        "--workers",
        str(int(workers)),
        "--batch-samples",
        "--retry-missing-single-worker",
        "--allow-partial",
    ]
    if bool(overwrite):
        cmd.append("--overwrite")
    return _run_capture(cmd)


def _postprocess_dir(
    *,
    py: str,
    post_script: Path,
    in_dir: Path,
    out_dir: Path,
    workers: int,
) -> str:
    cmd = [
        str(py),
        str(post_script),
        str(in_dir),
        str(out_dir),
        "--workers",
        str(int(workers)),
        "--resume",
    ]
    return _run_capture(cmd)


def _norm_sg(sg: Optional[str]) -> str:
    return re.sub(r"\s+", "", str(sg or "").strip().strip("'\""))


def _normalize_cif_for_parse(cif: str) -> str:
    try:
        sg = extract_space_group_symbol(cif)
    except Exception:
        sg = None
    if sg is not None and sg != "P 1":
        try:
            return replace_symmetry_operators(cif, sg, safe=True)
        except Exception:
            return cif
    return cif


def _parse_struct(cif: str) -> Optional[Structure]:
    try:
        return Structure.from_str(_normalize_cif_for_parse(cif), fmt="cif")
    except Exception:
        return None


def _comp_l1(a: Composition, b: Composition) -> float:
    da = a.get_el_amt_dict()
    db = b.get_el_amt_dict()
    sa = float(sum(da.values())) or 1.0
    sb = float(sum(db.values())) or 1.0
    elems = sorted(set(da) | set(db))
    return float(sum(abs(float(da.get(e, 0.0)) / sa - float(db.get(e, 0.0)) / sb) for e in elems))


def _feature_names() -> Tuple[str, ...]:
    return (
        "bias",
        "sample_rank_inv",
        "sample_rank_log_inv",
        "cif_len_log",
        "n_lines_log",
        "sensible",
        "parsed",
        "valid",
        "density",
        "density_ok",
        "density_log",
        "num_sites_log",
        "volume_per_site_log",
        "length_ratio",
        "angle_min",
        "angle_max",
        "angle_range",
        "stated_sg_present",
        "detected_sg_present",
        "sg_consistent",
        "stated_p1",
        "detected_p1",
        "detected_nontrivial",
        "reduced_formula_match",
        "elements_match",
        "comp_l1",
        "target_natoms_log",
        "sites_per_target_atom_log",
        "n_elements_gen",
        "n_elements_target",
        "n_elements_absdiff",
        "min_pair_dist",
        "min_pair_dist_ok",
        "median_nn_dist",
        "short_pair_frac_0p7",
        "very_short_pair_frac_0p5",
        "max_pair_dist_log",
        "mean_pair_dist_log",
        "cell_orthogonal_score",
        "angle_90_dev_mean",
        "angle_120_dev_min",
    )


def _candidate_features(path: Path, row: RowData) -> CandidateFeatures:
    m = _GEN_RE.match(path.name)
    if m is None:
        raise ValueError(f"unexpected generated CIF filename: {path.name}")
    mid = m.group("mid")
    idx = int(m.group("idx"))
    cif = _read_text(path)

    sensible = 0.0
    try:
        sensible = 1.0 if bool(is_sensible(cif, 0.5, 1000.0, 10.0, 170.0)) else 0.0
    except Exception:
        sensible = 0.0

    struct = _parse_struct(cif)
    parsed = 1.0 if struct is not None else 0.0
    valid = 0.0
    density = 0.0
    density_ok = 0.0
    num_sites = 0.0
    volume_per_site = 0.0
    length_ratio = 0.0
    angle_min = 0.0
    angle_max = 0.0
    angle_range = 0.0
    detected = None
    reduced_formula_match = 0.0
    elements_match = 0.0
    comp_l1 = 2.0
    sites_per_target_atom = 0.0
    n_elements_gen = 0.0
    n_elements_target = float(len(row.comp.elements))
    n_elements_absdiff = n_elements_target
    min_pair_dist = 0.0
    min_pair_dist_ok = 0.0
    median_nn_dist = 0.0
    short_pair_frac_0p7 = 0.0
    very_short_pair_frac_0p5 = 0.0
    max_pair_dist = 0.0
    mean_pair_dist = 0.0
    cell_orthogonal_score = 0.0
    angle_90_dev_mean = 0.0
    angle_120_dev_min = 0.0

    if struct is not None:
        try:
            valid = 1.0 if bool(is_valid(struct)) else 0.0
        except Exception:
            valid = 0.0
        try:
            density = float(struct.density)
            density_ok = 1.0 if math.isfinite(density) and 0.2 <= density <= 25.0 else 0.0
        except Exception:
            density = 0.0
        try:
            num_sites = float(struct.num_sites)
            volume_per_site = float(struct.volume) / max(1.0, num_sites)
            sites_per_target_atom = num_sites / max(1.0, float(sum(row.comp.get_el_amt_dict().values())))
        except Exception:
            pass
        try:
            lengths = list(struct.lattice.abc)
            length_ratio = float(max(lengths) / max(1e-8, min(lengths)))
            angles = list(struct.lattice.angles)
            angle_min = float(min(angles))
            angle_max = float(max(angles))
            angle_range = float(angle_max - angle_min)
            dev90 = [abs(float(a) - 90.0) for a in angles]
            angle_90_dev_mean = float(sum(dev90) / max(1, len(dev90)))
            angle_120_dev_min = float(min(abs(float(a) - 120.0) for a in angles))
            cell_orthogonal_score = float(sum(1 for d in dev90 if d <= 3.0) / max(1, len(dev90)))
        except Exception:
            pass
        try:
            detected = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_symbol()
        except Exception:
            detected = None
        try:
            reduced_formula_match = 1.0 if struct.composition.reduced_formula == row.comp.reduced_formula else 0.0
        except Exception:
            reduced_formula_match = 0.0
        try:
            gen_elements = {str(x) for x in struct.composition.elements}
            target_elements = {str(x) for x in row.comp.elements}
            n_elements_gen = float(len(gen_elements))
            n_elements_absdiff = float(abs(len(gen_elements) - len(target_elements)))
            elements_match = 1.0 if gen_elements == target_elements else 0.0
        except Exception:
            elements_match = 0.0
        try:
            comp_l1 = _comp_l1(struct.composition, row.comp)
        except Exception:
            comp_l1 = 2.0
        try:
            dm = np.array(struct.distance_matrix, dtype=float)
            if dm.size > 1:
                triu = dm[np.triu_indices(dm.shape[0], k=1)]
                triu = triu[np.isfinite(triu)]
                if triu.size:
                    min_pair_dist = float(np.min(triu))
                    max_pair_dist = float(np.max(triu))
                    mean_pair_dist = float(np.mean(triu))
                    short_pair_frac_0p7 = float(np.mean(triu < 0.7))
                    very_short_pair_frac_0p5 = float(np.mean(triu < 0.5))
                    min_pair_dist_ok = 1.0 if min_pair_dist >= 0.5 else 0.0
                    nn = []
                    for row_dm in dm:
                        vals = row_dm[np.isfinite(row_dm) & (row_dm > 1e-8)]
                        if vals.size:
                            nn.append(float(np.min(vals)))
                    if nn:
                        median_nn_dist = float(np.median(np.array(nn, dtype=float)))
        except Exception:
            pass

    try:
        stated = extract_space_group_symbol(cif)
    except Exception:
        stated = None

    stated_n = _norm_sg(stated)
    detected_n = _norm_sg(detected)
    values = np.array(
        [
            1.0,
            1.0 / float(max(1, idx)),
            1.0 / math.log(float(max(2, idx + 1))),
            math.log1p(float(len(cif))),
            math.log1p(float(len(cif.splitlines()))),
            sensible,
            parsed,
            valid,
            float(density) if math.isfinite(float(density)) else 0.0,
            density_ok,
            math.log1p(max(0.0, float(density))) if math.isfinite(float(density)) else 0.0,
            math.log1p(max(0.0, float(num_sites))),
            math.log1p(max(0.0, float(volume_per_site))),
            float(length_ratio),
            float(angle_min),
            float(angle_max),
            float(angle_range),
            1.0 if stated else 0.0,
            1.0 if detected else 0.0,
            1.0 if stated and detected and stated_n == detected_n else 0.0,
            1.0 if stated_n in {"P1", "P-1"} else 0.0,
            1.0 if detected_n in {"P1", "P-1"} else 0.0,
            1.0 if detected and detected_n not in {"P1", "P-1"} else 0.0,
            reduced_formula_match,
            elements_match,
            float(comp_l1),
            math.log1p(float(sum(row.comp.get_el_amt_dict().values()))),
            math.log1p(max(0.0, float(sites_per_target_atom))),
            float(n_elements_gen),
            float(n_elements_target),
            float(n_elements_absdiff),
            float(min_pair_dist),
            float(min_pair_dist_ok),
            float(median_nn_dist),
            float(short_pair_frac_0p7),
            float(very_short_pair_frac_0p5),
            math.log1p(max(0.0, float(max_pair_dist))),
            math.log1p(max(0.0, float(mean_pair_dist))),
            float(cell_orthogonal_score),
            float(angle_90_dev_mean),
            float(angle_120_dev_min),
        ],
        dtype=float,
    )
    values[~np.isfinite(values)] = 0.0
    return CandidateFeatures(
        mid=mid,
        idx=idx,
        path=path,
        values=values,
        names=_feature_names(),
        meta={
            "stated_sg": stated,
            "detected_sg": detected,
            "parsed": bool(parsed),
            "sensible": bool(sensible),
            "valid": bool(valid),
            "density": density,
            "comp_l1": comp_l1,
            "min_pair_dist": min_pair_dist,
            "median_nn_dist": median_nn_dist,
            "short_pair_frac_0p7": short_pair_frac_0p7,
        },
    )


def _group_generated(gen_dir: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(gen_dir.glob("*.cif")):
        m = _GEN_RE.match(path.name)
        if m is None:
            continue
        grouped.setdefault(m.group("mid"), []).append(path)
    return grouped


def _candidate_match_label(path: Path, true_cif: str, matcher: StructureMatcher, *, max_sites: int) -> Tuple[int, Optional[float]]:
    timeout_s = float(os.environ.get("OPENTRY_LABEL_TIMEOUT_SECONDS", "30"))
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _candidate_label_timeout_handler)
        if timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, timeout_s)
        pred = _parse_struct(_read_text(path))
        gt = _parse_struct(true_cif)
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
    except _CandidateLabelTimeout:
        return 0, None
    finally:
        if timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _record_candidate(
    *,
    path: Path,
    row: RowData,
    prompts_dir: Path,
    matcher: StructureMatcher,
    max_sites: int,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    cf = _candidate_features(path, row)
    label, rms = _candidate_match_label(path, row.cif, matcher, max_sites=int(max_sites))
    rec = {
        "id": row.material_id,
        "idx": int(cf.idx),
        "path": str(path),
        "prompt_path": str((prompts_dir / f"{row.material_id}.txt").resolve()),
        "label": int(label),
        "rms": rms,
        "features": {name: float(val) for name, val in zip(cf.names, cf.values)},
        "meta": cf.meta,
    }
    return cf.values, int(label), rec


def _label_generated_candidates(
    *,
    rows: Sequence[RowData],
    gen_dir: Path,
    prompts_dir: Path,
    matcher: StructureMatcher,
    max_sites: int,
) -> Tuple[List[np.ndarray], List[int], List[Dict[str, Any]]]:
    rows_by_id = {r.material_id: r for r in rows}
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    records: List[Dict[str, Any]] = []
    grouped = _group_generated(gen_dir)
    for mid, paths in sorted(grouped.items()):
        row = rows_by_id.get(mid)
        if row is None:
            continue
        for path in sorted(paths):
            x, y, rec = _record_candidate(
                path=path,
                row=row,
                prompts_dir=prompts_dir,
                matcher=matcher,
                max_sites=int(max_sites),
            )
            Xs.append(x)
            ys.append(y)
            records.append(rec)
    return Xs, ys, records


def _expected_sample_paths(mid: str, out_dir: Path, *, offset: int, count: int) -> List[Path]:
    return [out_dir / f"{mid}__{i}.cif" for i in range(int(offset) + 1, int(offset) + int(count) + 1)]


def _generate_calib_adaptive(
    *,
    rows: Sequence[RowData],
    py: str,
    gen_script: Path,
    model_dir: Path,
    prompts_root: Path,
    out_dir: Path,
    matcher: StructureMatcher,
    args: argparse.Namespace,
) -> Tuple[str, List[np.ndarray], List[int], List[Dict[str, Any]], Dict[str, Any]]:
    """Generate train/val candidates in rounds, using GT labels only to decide train-side stopping.

    This is intentionally only used for calibration/training data. Test generation remains fixed-budget
    and non-oracle.
    """

    prompts_dir = prompts_root / "calib"
    rows_by_id = {r.material_id: r for r in rows}
    max_candidates = max(int(args.num_samples_per_prompt), int(args.train_max_candidates_per_id))
    round_size = int(args.train_candidate_round_size) if int(args.train_candidate_round_size) > 0 else int(args.num_samples_per_prompt)
    round_size = max(1, int(round_size))
    min_candidates = int(args.train_min_candidates_per_id) if int(args.train_min_candidates_per_id) > 0 else int(args.num_samples_per_prompt)
    min_candidates = min(max_candidates, max(1, int(min_candidates)))
    target_pos = max(0, int(args.train_target_positives_per_id))

    active = list(rows)
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    records: List[Dict[str, Any]] = []
    pos_by_id: Dict[str, int] = {r.material_id: 0 for r in rows}
    generated_by_id: Dict[str, int] = {r.material_id: 0 for r in rows}
    log_parts: List[str] = []
    round_summaries: List[Dict[str, Any]] = []

    offset = 0
    round_idx = 0
    while active and offset < max_candidates:
        count = min(round_size, max_candidates - offset)
        round_idx += 1
        round_prompt_dir = prompts_root / f"calib_round_{round_idx:03d}_{offset + 1}_{offset + count}"
        _write_prompts(active, round_prompt_dir)
        print(
            f"[ranker] adaptive calib round={round_idx} active={len(active)} samples={offset + 1}-{offset + count}",
            flush=True,
        )
        round_log = _generate_prompt_dir(
            py=str(py),
            gen_script=gen_script,
            model_dir=model_dir,
            prompts_dir=round_prompt_dir,
            out_dir=out_dir,
            num_samples=int(count),
            sample_index_offset=int(offset),
            seed=int(args.seed),
            sample_seed_stride=int(args.sample_seed_stride),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            max_new_tokens=int(args.max_new_tokens),
            device=str(args.device),
            dtype=str(args.dtype),
            workers=int(args.gen_workers),
            overwrite=bool(args.overwrite),
        )
        log_parts.append(f"\n===== round {round_idx} offset={offset} count={count} active={len(active)} =====\n")
        log_parts.append(round_log)

        new_records = 0
        new_positives = 0
        for row in active:
            generated_by_id[row.material_id] = offset + count
            for path in _expected_sample_paths(row.material_id, out_dir, offset=offset, count=count):
                if not path.exists():
                    continue
                x, y, rec = _record_candidate(
                    path=path,
                    row=row,
                    prompts_dir=prompts_dir,
                    matcher=matcher,
                    max_sites=int(args.max_sites),
                )
                Xs.append(x)
                ys.append(y)
                records.append(rec)
                new_records += 1
                if int(y) > 0:
                    pos_by_id[row.material_id] += 1
                    new_positives += 1

        next_active: List[RowData] = []
        for row in active:
            generated_n = int(generated_by_id.get(row.material_id, 0))
            positive_n = int(pos_by_id.get(row.material_id, 0))
            if generated_n < min_candidates:
                next_active.append(row)
            elif target_pos > 0 and positive_n < target_pos and generated_n < max_candidates:
                next_active.append(row)
        round_summaries.append(
            {
                "round": int(round_idx),
                "offset": int(offset),
                "count": int(count),
                "active_before": int(len(active)),
                "active_after": int(len(next_active)),
                "new_records": int(new_records),
                "new_positives": int(new_positives),
                "records_total": int(len(records)),
                "positives_total": int(sum(ys)),
            }
        )
        active = next_active
        offset += count

    stats = {
        "enabled": True,
        "round_size": int(round_size),
        "min_candidates_per_id": int(min_candidates),
        "max_candidates_per_id": int(max_candidates),
        "target_positives_per_id": int(target_pos),
        "n_rounds": int(round_idx),
        "n_ids": int(len(rows)),
        "n_records": int(len(records)),
        "n_positive": int(sum(ys)),
        "ids_with_positive": int(sum(1 for v in pos_by_id.values() if int(v) > 0)),
        "ids_reached_max": int(sum(1 for v in generated_by_id.values() if int(v) >= max_candidates)),
        "generated_candidates_by_id_min": int(min(generated_by_id.values()) if generated_by_id else 0),
        "generated_candidates_by_id_max": int(max(generated_by_id.values()) if generated_by_id else 0),
        "rounds": round_summaries,
    }
    # Re-label once at the end so feature arrays and records are in a deterministic order.
    Xs, ys, records = _label_generated_candidates(
        rows=rows,
        gen_dir=out_dir,
        prompts_dir=prompts_dir,
        matcher=matcher,
        max_sites=int(args.max_sites),
    )
    return "".join(log_parts), Xs, ys, records, stats


def _fit_logistic(X: np.ndarray, y: np.ndarray, *, iters: int, lr: float, l2: float) -> Dict[str, Any]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    Xs[:, 0] = 1.0
    w = np.zeros(Xs.shape[1], dtype=float)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    if pos <= 0:
        # Degenerate: no positives. Fall back to original order via sample rank features.
        w[1] = 1.0
        return {"weights": w.tolist(), "mean": mu.tolist(), "std": sigma.tolist(), "note": "no_positive_labels"}
    pos_w = neg / max(1.0, pos)
    sample_w = np.where(y > 0.5, pos_w, 1.0)
    for _ in range(int(iters)):
        z = np.clip(Xs @ w, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (Xs.T @ ((p - y) * sample_w)) / max(1.0, float(sample_w.sum()))
        grad += float(l2) * w
        grad[0] -= float(l2) * w[0]
        w -= float(lr) * grad
    z = np.clip(Xs @ w, -40.0, 40.0)
    p = 1.0 / (1.0 + np.exp(-z))
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean())
    return {
        "weights": w.tolist(),
        "mean": mu.tolist(),
        "std": sigma.tolist(),
        "train_acc": acc,
        "n_train": int(len(y)),
        "n_positive": int(pos),
        "positive_rate": float(pos / max(1.0, float(len(y)))),
    }


def _predict_logistic(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    w = np.array(model["weights"], dtype=float)
    mu = np.array(model["mean"], dtype=float)
    sigma = np.array(model["std"], dtype=float)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma
    Xs[:, 0] = 1.0
    z = np.clip(Xs @ w, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_ranker(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ranker_type: str,
    feature_names: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    if str(ranker_type) == "logistic":
        model = _fit_logistic(X, y, iters=int(args.train_iters), lr=float(args.train_lr), l2=float(args.train_l2))
        model["ranker_type"] = "logistic"
        model["feature_names"] = list(feature_names)
        scores = _predict_logistic(model, X)
        return model, model, scores

    if str(ranker_type) == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(args.rf_estimators),
            max_depth=None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            class_weight="balanced_subsample",
            random_state=int(args.seed),
            n_jobs=int(args.rf_jobs),
        )
        clf.fit(X, y.astype(int))
        scores = clf.predict_proba(X)[:, 1]
        pred = (scores >= 0.5).astype(int)
        importances = getattr(clf, "feature_importances_", np.zeros(X.shape[1], dtype=float))
        summary = {
            "ranker_type": "random_forest",
            "feature_names": list(feature_names),
            "n_train": int(len(y)),
            "n_positive": int(y.sum()),
            "positive_rate": float(float(y.sum()) / max(1.0, float(len(y)))),
            "train_acc": float(accuracy_score(y.astype(int), pred)),
            "rf_estimators": int(args.rf_estimators),
            "rf_max_depth": None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth),
            "rf_min_samples_leaf": int(args.rf_min_samples_leaf),
            "feature_importances": {
                name: float(val) for name, val in zip(feature_names, importances.tolist())
            },
        }
        return clf, summary, scores

    raise ValueError(f"unknown ranker_type: {ranker_type}")


def _predict_ranker(model_obj: Any, model_summary: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    if str(model_summary.get("ranker_type")) == "logistic":
        return _predict_logistic(model_summary, X)
    if str(model_summary.get("ranker_type")) == "random_forest":
        return model_obj.predict_proba(X)[:, 1]
    raise ValueError(f"unknown ranker_type: {model_summary.get('ranker_type')}")


def _rank_hit_metrics(records: Sequence[Dict[str, Any]], scores: Sequence[float], budgets: Sequence[int]) -> Dict[str, Any]:
    by_id: Dict[str, List[Tuple[float, int, int]]] = {}
    for rec, score in zip(records, scores):
        by_id.setdefault(str(rec["id"]), []).append((float(score), int(rec.get("idx", 0)), int(rec.get("label", 0))))
    out: Dict[str, Any] = {"n_ids": len(by_id)}
    for k in budgets:
        hits = 0
        for items in by_id.values():
            ranked = sorted(items, key=lambda x: (-x[0], x[1]))
            if any(label > 0 for _score, _idx, label in ranked[: int(k)]):
                hits += 1
        out[f"hit_rate_at_{int(k)}"] = None if not by_id else float(hits / len(by_id))
    return out


def _feature_proxy_score(rec: Dict[str, Any]) -> float:
    f = rec.get("features") or {}
    return float(
        3.0 * f.get("parsed", 0.0)
        + 2.0 * f.get("sensible", 0.0)
        + 2.0 * f.get("valid", 0.0)
        + 3.0 * f.get("reduced_formula_match", 0.0)
        + 2.0 * f.get("elements_match", 0.0)
        + 1.5 * f.get("sg_consistent", 0.0)
        + 1.0 * f.get("density_ok", 0.0)
        + 0.5 * f.get("detected_nontrivial", 0.0)
        - 2.0 * f.get("comp_l1", 2.0)
        - 1.0 * f.get("short_pair_frac_0p7", 0.0)
        - 2.0 * f.get("very_short_pair_frac_0p5", 0.0)
        - 0.1 * f.get("angle_range", 0.0)
        - 0.5 * f.get("detected_p1", 0.0)
    )


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _select_rejection_records(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        rec = dict(rec)
        rec["proxy_score"] = _feature_proxy_score(rec)
        by_id.setdefault(str(rec["id"]), []).append(rec)

    selected: List[Dict[str, Any]] = []
    stats = {
        "n_ids": len(by_id),
        "n_all": len(records),
        "n_positive_all": int(sum(1 for r in records if int(r.get("label", 0)) > 0)),
        "ids_with_positive": 0,
    }
    for _mid, items in sorted(by_id.items()):
        positives = [r for r in items if int(r.get("label", 0)) > 0]
        negatives = [r for r in items if int(r.get("label", 0)) <= 0]
        positives = sorted(positives, key=lambda r: (float("inf") if r.get("rms") is None else float(r["rms"]), int(r.get("idx", 0))))
        hard_negs = sorted(negatives, key=lambda r: (-float(r.get("proxy_score", 0.0)), int(r.get("idx", 0))))
        easy_negs = sorted(negatives, key=lambda r: (float(r.get("proxy_score", 0.0)), int(r.get("idx", 0))))

        if positives:
            stats["ids_with_positive"] += 1
        selected.extend(positives[: max(0, int(args.max_positives_per_id))])
        selected.extend(hard_negs[: max(0, int(args.hard_negatives_per_id))])
        selected.extend(easy_negs[: max(0, int(args.easy_negatives_per_id))])

    # De-duplicate records that can appear in both hard/easy negative sets.
    dedup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rec in selected:
        dedup[(str(rec["id"]), int(rec["idx"]))] = rec
    selected = list(dedup.values())
    stats.update(
        {
            "n_selected": len(selected),
            "n_positive_selected": int(sum(1 for r in selected if int(r.get("label", 0)) > 0)),
            "positive_rate_selected": float(
                sum(1 for r in selected if int(r.get("label", 0)) > 0) / max(1, len(selected))
            ),
        }
    )
    return selected, stats


def _write_dpo_pairs(records: Sequence[Dict[str, Any]], out_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        rec = dict(rec)
        rec["proxy_score"] = _feature_proxy_score(rec)
        by_id.setdefault(str(rec["id"]), []).append(rec)

    n_pairs = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for mid, items in sorted(by_id.items()):
            positives = [r for r in items if int(r.get("label", 0)) > 0]
            negatives = [r for r in items if int(r.get("label", 0)) <= 0]
            if not positives or not negatives:
                continue
            positives = sorted(positives, key=lambda r: (float("inf") if r.get("rms") is None else float(r["rms"]), int(r.get("idx", 0))))
            hard_negs = sorted(negatives, key=lambda r: (-float(r.get("proxy_score", 0.0)), int(r.get("idx", 0))))
            for pos in positives[: max(1, int(args.dpo_positives_per_id))]:
                for neg in hard_negs[: max(1, int(args.dpo_negatives_per_positive))]:
                    prompt_path = str(pos.get("prompt_path") or "")
                    chosen_path = str(pos.get("path") or "")
                    rejected_path = str(neg.get("path") or "")
                    pair = {
                        "id": mid,
                        "prompt_path": prompt_path,
                        "chosen_path": chosen_path,
                        "rejected_path": rejected_path,
                        "chosen_idx": int(pos.get("idx", 0)),
                        "rejected_idx": int(neg.get("idx", 0)),
                        "chosen_rms": pos.get("rms"),
                        "rejected_proxy_score": neg.get("proxy_score"),
                        "chosen_meta": pos.get("meta"),
                        "rejected_meta": neg.get("meta"),
                    }
                    if bool(args.dpo_include_text):
                        pair["prompt"] = _read_text(Path(prompt_path)) if prompt_path else ""
                        pair["chosen"] = _read_text(Path(chosen_path)) if chosen_path else ""
                        pair["rejected"] = _read_text(Path(rejected_path)) if rejected_path else ""
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    n_pairs += 1
    return {"n_pairs": int(n_pairs), "path": str(out_path)}


def _write_ranked(grouped_feats: Dict[str, List[CandidateFeatures]], scores: Dict[Path, float], out_dir: Path) -> Dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for mid, feats in sorted(grouped_feats.items()):
        ranked = sorted(feats, key=lambda cf: (-float(scores.get(cf.path, 0.0)), cf.idx))
        summary[mid] = [
            {
                "old_idx": int(cf.idx),
                "new_idx": int(i),
                "score": float(scores.get(cf.path, 0.0)),
                "meta": cf.meta,
            }
            for i, cf in enumerate(ranked, start=1)
        ]
        for i, cf in enumerate(ranked, start=1):
            shutil.copyfile(cf.path, out_dir / f"{mid}__{i}.cif")
    return summary


def _make_gen_tar(out_path: Path, gen_dir: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for path in sorted(gen_dir.glob("*.cif")):
            tar.add(str(path), arcname=path.name)


def _make_true_tar_from_rows(out_path: Path, rows: Sequence[RowData], *, gt_dir: Optional[Path] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for row in sorted(rows, key=lambda r: r.material_id):
            if gt_dir is not None:
                p = gt_dir / f"{row.material_id}.cif"
                if p.exists():
                    data = p.read_bytes()
                else:
                    data = row.cif.encode("utf-8")
            else:
                data = row.cif.encode("utf-8")
            ti = tarfile.TarInfo(name=f"{row.material_id}.cif")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))


def _benchmark(
    *,
    py: str,
    bench_script: Path,
    gen_tar: Path,
    true_tar: Path,
    num_gens: int,
    max_sites: int,
    rmsd_timeout_seconds: float,
    workers: int,
) -> Tuple[str, Dict[str, Any]]:
    cmd = [
        str(py),
        str(bench_script),
        str(gen_tar),
        str(true_tar),
        "--num-gens",
        str(int(num_gens)),
        "--max-sites",
        str(int(max_sites)),
        "--rmsd-timeout-seconds",
        str(float(rmsd_timeout_seconds)),
        "--workers",
        str(int(workers)),
    ]
    out = _run_capture(cmd)
    return out, _parse_metric_dict(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--calib-csv", required=True)
    parser.add_argument("--ranker-train-csvs", default="", help="Optional comma-separated CSVs for ranker training, e.g. train.csv,val.csv. Overrides --calib-csv rows when set.")
    parser.add_argument("--ranker-train-per-csv-limit", type=int, default=0)
    parser.add_argument("--ranker-train-limit", type=int, default=0)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--test-gt-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--calib-start-index", type=int, default=0)
    parser.add_argument("--calib-limit", type=int, default=50)
    parser.add_argument("--test-start-index", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--num-samples-per-prompt", type=int, default=10)
    parser.add_argument("--train-max-candidates-per-id", type=int, default=0, help="If > --num-samples-per-prompt, adaptively generate train/val candidates up to this maximum per material.")
    parser.add_argument("--train-min-candidates-per-id", type=int, default=0, help="Minimum train/val candidates per material before positive-count early stopping is allowed; <=0 uses --num-samples-per-prompt.")
    parser.add_argument("--train-candidate-round-size", type=int, default=0, help="Train/val adaptive generation batch size; <=0 uses --num-samples-per-prompt.")
    parser.add_argument("--train-target-positives-per-id", type=int, default=3, help="For train/val adaptive generation, continue a material until this many positives or max candidates.")
    parser.add_argument("--eval-budgets", default="1,5,10")
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gen-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-seed-stride", type=int, default=100000)
    parser.add_argument("--max-sites", type=int, default=512)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--metrics-workers", type=int, default=1)
    parser.add_argument("--ranker-type", choices=["logistic", "random_forest"], default="random_forest")
    parser.add_argument("--train-iters", type=int, default=600)
    parser.add_argument("--train-lr", type=float, default=0.1)
    parser.add_argument("--train-l2", type=float, default=0.01)
    parser.add_argument("--rf-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=0, help="<=0 means unlimited")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--use-rejection-sampled-train", action="store_true")
    parser.add_argument("--max-positives-per-id", type=int, default=8)
    parser.add_argument("--hard-negatives-per-id", type=int, default=10)
    parser.add_argument("--easy-negatives-per-id", type=int, default=2)
    parser.add_argument("--dpo-positives-per-id", type=int, default=3)
    parser.add_argument("--dpo-negatives-per-positive", type=int, default=3)
    parser.add_argument("--dpo-include-text", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.out_root).expanduser().resolve() / str(args.run_name)
    prompts_root = run_dir / "prompts"
    cifs_root = run_dir / "cifs"
    tars_root = run_dir / "tars"
    metrics_root = run_dir / "metrics"
    for d in (prompts_root, cifs_root, tars_root, metrics_root):
        d.mkdir(parents=True, exist_ok=True)

    if str(args.ranker_train_csvs).strip():
        train_paths = [Path(x).expanduser().resolve() for x in str(args.ranker_train_csvs).split(",") if x.strip()]
        calib_rows = _load_rows_multi(
            train_paths,
            per_csv_limit=None if int(args.ranker_train_per_csv_limit) <= 0 else int(args.ranker_train_per_csv_limit),
            total_limit=None if int(args.ranker_train_limit) <= 0 else int(args.ranker_train_limit),
        )
    else:
        calib_rows = _load_rows(Path(args.calib_csv).expanduser().resolve(), start=int(args.calib_start_index), limit=int(args.calib_limit))
    test_rows = _load_rows(Path(args.test_csv).expanduser().resolve(), start=int(args.test_start_index), limit=int(args.test_limit))
    if not calib_rows or not test_rows:
        raise SystemExit("empty calib/test rows")

    gen_script = _ROOT / "bin/generate_cifs_from_prompts_dir.py"
    post_script = _ROOT / "bin/postprocess.py"
    bench_script = _ROOT / "bin/benchmark_metrics.py"
    model_dir = Path(args.model_dir).expanduser().resolve()
    test_gt_dir = Path(args.test_gt_dir).expanduser().resolve()
    test_rows = _replace_rows_from_gt_dir(test_rows, test_gt_dir)

    print(f"[ranker] calib rows={len(calib_rows)} test rows={len(test_rows)}", flush=True)
    _write_prompts(calib_rows, prompts_root / "calib")
    _write_prompts(test_rows, prompts_root / "test")
    _make_true_tar_from_rows(tars_root / "calib_true.tar.gz", calib_rows)
    _make_true_tar_from_rows(tars_root / "test_true.tar.gz", test_rows, gt_dir=test_gt_dir)

    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    adaptive_train_stats: Dict[str, Any] = {"enabled": False}

    print("[ranker] generate calib candidates", flush=True)
    if int(args.train_max_candidates_per_id) > int(args.num_samples_per_prompt):
        log, Xs, ys, train_records, adaptive_train_stats = _generate_calib_adaptive(
            rows=calib_rows,
            py=str(args.py),
            gen_script=gen_script,
            model_dir=model_dir,
            prompts_root=prompts_root,
            out_dir=cifs_root / "calib_raw",
            matcher=matcher,
            args=args,
        )
    else:
        log = _generate_prompt_dir(
            py=str(args.py),
            gen_script=gen_script,
            model_dir=model_dir,
            prompts_dir=prompts_root / "calib",
            out_dir=cifs_root / "calib_raw",
            num_samples=int(args.num_samples_per_prompt),
            sample_index_offset=0,
            seed=int(args.seed),
            sample_seed_stride=int(args.sample_seed_stride),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            max_new_tokens=int(args.max_new_tokens),
            device=str(args.device),
            dtype=str(args.dtype),
            workers=int(args.gen_workers),
            overwrite=bool(args.overwrite),
        )
        post_log = _postprocess_dir(
            py=str(args.py),
            post_script=post_script,
            in_dir=cifs_root / "calib_raw",
            out_dir=cifs_root / "calib_baseline",
            workers=int(args.metrics_workers),
        )
        _write_text(run_dir / "postprocess_calib.log", post_log)
        print("[ranker] label calib candidates", flush=True)
        Xs, ys, train_records = _label_generated_candidates(
            rows=calib_rows,
            gen_dir=cifs_root / "calib_baseline",
            prompts_dir=prompts_root / "calib",
            matcher=matcher,
            max_sites=int(args.max_sites),
        )
    _write_text(run_dir / "generate_calib.log", log)

    print("[ranker] generate test candidates", flush=True)
    log = _generate_prompt_dir(
        py=str(args.py),
        gen_script=gen_script,
        model_dir=model_dir,
        prompts_dir=prompts_root / "test",
        out_dir=cifs_root / "test_raw",
        num_samples=int(args.num_samples_per_prompt),
        sample_index_offset=0,
        seed=int(args.seed),
        sample_seed_stride=int(args.sample_seed_stride),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        max_new_tokens=int(args.max_new_tokens),
        device=str(args.device),
        dtype=str(args.dtype),
        workers=int(args.gen_workers),
        overwrite=bool(args.overwrite),
    )
    _write_text(run_dir / "generate_test.log", log)
    post_log = _postprocess_dir(
        py=str(args.py),
        post_script=post_script,
        in_dir=cifs_root / "test_raw",
        out_dir=cifs_root / "test_baseline",
        workers=int(args.metrics_workers),
    )
    _write_text(run_dir / "postprocess_test.log", post_log)

    if not Xs:
        raise SystemExit("no calib candidate features")
    _write_jsonl(run_dir / "ranker_candidates_all.jsonl", train_records)
    selected_train_records, rejection_stats = _select_rejection_records(train_records, args)
    _write_jsonl(run_dir / "ranker_candidates_rejection_sampled.jsonl", selected_train_records)
    dpo_stats = _write_dpo_pairs(train_records, run_dir / "dpo_pairs.jsonl", args)

    if bool(args.use_rejection_sampled_train):
        index_by_key = {(str(r["id"]), int(r["idx"])): i for i, r in enumerate(train_records)}
        selected_idx = [index_by_key[(str(r["id"]), int(r["idx"]))] for r in selected_train_records]
        X = np.vstack([Xs[i] for i in selected_idx])
        y = np.array([ys[i] for i in selected_idx], dtype=float)
    else:
        X = np.vstack(Xs)
        y = np.array(ys, dtype=float)

    model_obj, model, train_scores = _fit_ranker(
        X,
        y,
        ranker_type=str(args.ranker_type),
        feature_names=_feature_names(),
        args=args,
    )
    X_all = np.vstack(Xs)
    all_train_scores = _predict_ranker(model_obj, model, X_all)
    budgets = [int(x) for x in str(args.eval_budgets).split(",") if str(x).strip()]
    calib_rank_metrics = _rank_hit_metrics(train_records, all_train_scores, budgets)
    print(
        f"[ranker] train positives={int(y.sum())}/{len(y)} all_positives={int(sum(ys))}/{len(ys)} acc={model.get('train_acc')}",
        flush=True,
    )
    _write_json(run_dir / "ranker_model.json", model)
    with (run_dir / "ranker_model.pkl").open("wb") as f:
        pickle.dump(model_obj, f)
    _write_json(run_dir / "calib_train_records.json", train_records)

    rows_by_id_test = {r.material_id: r for r in test_rows}
    grouped_test_paths = _group_generated(cifs_root / "test_baseline")
    grouped_feats: Dict[str, List[CandidateFeatures]] = {}
    score_by_path: Dict[Path, float] = {}
    for mid, paths in sorted(grouped_test_paths.items()):
        row = rows_by_id_test.get(mid)
        if row is None:
            continue
        feats = [_candidate_features(path, row) for path in sorted(paths)]
        if not feats:
            continue
        Xtest = np.vstack([cf.values for cf in feats])
        pred = _predict_ranker(model_obj, model, Xtest)
        grouped_feats[mid] = feats
        for cf, sc in zip(feats, pred):
            score_by_path[cf.path] = float(sc)

    ranking = _write_ranked(grouped_feats, score_by_path, cifs_root / "test_ranked")
    _write_json(run_dir / "test_ranking.json", ranking)
    _make_gen_tar(tars_root / "test_baseline.tar.gz", cifs_root / "test_baseline")
    _make_gen_tar(tars_root / "test_ranked.tar.gz", cifs_root / "test_ranked")

    summary_lines = ["group\tK\tmatch_rate\trmse\tparse_rate_candidate\tvalid_rate_candidate"]
    results: Dict[str, Dict[str, Any]] = {}
    for group, tar_name in [("baseline", "test_baseline.tar.gz"), ("ranked", "test_ranked.tar.gz")]:
        for k in budgets:
            print(f"[ranker] benchmark {group} k={k}", flush=True)
            out, metrics = _benchmark(
                py=str(args.py),
                bench_script=bench_script,
                gen_tar=tars_root / tar_name,
                true_tar=tars_root / "test_true.tar.gz",
                num_gens=int(k),
                max_sites=int(args.max_sites),
                rmsd_timeout_seconds=float(args.rmsd_timeout_seconds),
                workers=int(args.metrics_workers),
            )
            _write_text(metrics_root / f"{group}_k{k}.txt", out)
            _write_json(metrics_root / f"{group}_k{k}.json", metrics)
            results[f"{group}_k{k}"] = metrics
            summary_lines.append(
                "\t".join(
                    [
                        group,
                        str(int(k)),
                        str(metrics.get("match_rate")),
                        str(metrics.get("rms_dist")),
                        str(metrics.get("parse_rate_candidate")),
                        str(metrics.get("valid_rate_candidate")),
                    ]
                )
            )
    _write_text(run_dir / "summary_metrics.tsv", "\n".join(summary_lines) + "\n")
    _write_json(
        run_dir / "run_summary.json",
        {
            "calib_csv": str(Path(args.calib_csv).expanduser().resolve()),
            "test_csv": str(Path(args.test_csv).expanduser().resolve()),
            "test_gt_dir": str(test_gt_dir),
            "n_calib": len(calib_rows),
            "n_test": len(test_rows),
            "num_samples_per_prompt": int(args.num_samples_per_prompt),
            "model": model,
            "calib_rank_metrics": calib_rank_metrics,
            "adaptive_train_generation": adaptive_train_stats,
            "rejection_sampling": rejection_stats,
            "dpo_pairs": dpo_stats,
            "used_rejection_sampled_train": bool(args.use_rejection_sampled_train),
            "results": results,
        },
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
