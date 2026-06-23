#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import re
import signal
import sys
import tarfile
import time
from collections import Counter, defaultdict, deque
from functools import lru_cache
from multiprocessing import Pool, Process, Queue
from pathlib import Path
from typing import Any

import numpy as np
import smact
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from smact.screening import pauling_test

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
CRYSTALLM = WORKSPACE / "model/CrystaLLM"
if str(CRYSTALLM) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM))

from crystallm import is_sensible  # noqa: E402


class EvalTimeoutError(TimeoutError):
    pass


def _eval_timeout_handler(signum: int, frame: Any) -> None:
    raise EvalTimeoutError("candidate evaluation timed out")

DATASETS = {
    "mp20": {
        "bench": "mp_20",
        "prefix": "mp_20",
        "csv_dir": CRYSTALLM / "resources/benchmarks/mp_20",
    },
    "mpts52": {
        "bench": "mpts_52",
        "prefix": "mpts_52",
        "csv_dir": CRYSTALLM / "resources/benchmarks/mpts_52",
    },
}


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_7: {resolved}")
    return resolved


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


def append_log(text: str) -> None:
    path = under_root(ROOT / "opentry_7_experiment_log.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def dataset_info(dataset: str) -> dict[str, Any]:
    key = dataset.lower().replace("-", "").replace("_", "")
    if key == "mp20":
        return DATASETS["mp20"]
    if key in {"mpts52", "mpts"}:
        return DATASETS["mpts52"]
    raise ValueError(f"unknown dataset: {dataset}")


def csv_path(dataset: str, split: str) -> Path:
    info = dataset_info(dataset)
    path = info["csv_dir"] / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sample_id_for(dataset: str, split: str, material_id: str) -> str:
    info = dataset_info(dataset)
    return f"{info['prefix']}_{split}_orig__{material_id}"


def sample_aliases_for(dataset: str, split: str, material_id: str) -> list[str]:
    info = dataset_info(dataset)
    aliases = [
        sample_id_for(dataset, split, material_id),
        f"{info['prefix']}_{split}__{material_id}",
        str(material_id),
    ]
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            out.append(alias)
    return out


def target_aliases(target: dict[str, Any]) -> list[str]:
    aliases = list(target.get("sample_aliases") or [])
    sample_id = str(target.get("sample_id") or "")
    if sample_id and sample_id not in aliases:
        aliases.insert(0, sample_id)
    material_id = str(target.get("material_id") or "")
    dataset = str(target.get("dataset") or "")
    split = str(target.get("split") or "")
    if material_id and dataset and split:
        prefix = dataset_info(dataset)["prefix"] if dataset.replace("_", "") in {"mp20", "mpts52"} else dataset
        for alias in (f"{prefix}_{split}_orig__{material_id}", f"{prefix}_{split}__{material_id}", material_id):
            if alias not in aliases:
                aliases.append(alias)
    return aliases


def read_csv_records(dataset: str, split: str) -> list[dict[str, str]]:
    path = csv_path(dataset, split)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_data_name(cif: str) -> str:
    m = re.search(r"(?im)^\s*data_([^\s]+)", cif or "")
    return m.group(1).strip() if m else "UNKNOWN"


def extract_formula_text(cif: str, data_name: str | None = None) -> str:
    for key in ("_chemical_formula_structural", "_chemical_formula_sum"):
        for line in cif.splitlines():
            stripped = line.strip()
            if stripped.startswith(key):
                return stripped[len(key):].strip().strip("'\"")
    return data_name or ""


def count_atom_site_rows(cif: str) -> int:
    in_atom_loop = False
    saw_atom_header = False
    count = 0
    for raw in cif.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "loop_":
            if in_atom_loop and saw_atom_header and count:
                break
            in_atom_loop = False
            saw_atom_header = False
            continue
        if stripped.startswith("_atom_site_"):
            in_atom_loop = True
            saw_atom_header = True
            continue
        if stripped.startswith("_") or stripped.startswith("data_"):
            if in_atom_loop and saw_atom_header:
                break
            continue
        if in_atom_loop and saw_atom_header:
            count += 1
    return count


def quote_sg(symbol: str) -> str:
    symbol = str(symbol or "P 1").strip()
    if symbol.startswith("'") and symbol.endswith("'"):
        return symbol
    return f"'{symbol}'"


def get_sg_symbol_from_number(number: int | None) -> str | None:
    if not number:
        return None
    try:
        return SpaceGroup.from_int_number(int(number)).symbol
    except Exception:
        return None


def analyze_target_cif(
    cif: str,
    explicit_sg_number: int | None = None,
    symprec: float = 0.1,
    fast_row_count: bool = False,
) -> dict[str, Any]:
    if fast_row_count:
        data_name = extract_data_name(cif)
        row_count = count_atom_site_rows(cif)
        return {
            "data_name": data_name,
            "formula": extract_formula_text(cif, data_name=data_name),
            "n_sites": int(row_count),
            "row_count": int(row_count),
            "row_count_method": "raw_atom_site_rows_fast_no_structure_parse",
            "sg_number": int(explicit_sg_number) if explicit_sg_number is not None else None,
            "sg_symbol": get_sg_symbol_from_number(explicit_sg_number) or "P 1",
            "analyzer_error": "fast_row_count: Structure parser and SpacegroupAnalyzer skipped",
        }

    struct = Structure.from_str(cif, fmt="cif")
    sg_number = explicit_sg_number
    sg_symbol = get_sg_symbol_from_number(explicit_sg_number)
    row_count = None
    analyzer_error = None
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        if sg_number is None:
            sg_number = int(sga.get_space_group_number())
        if sg_symbol is None:
            sg_symbol = str(sga.get_space_group_symbol())
        symm = sga.get_symmetrized_structure()
        row_count = len(symm.equivalent_indices)
    except Exception as exc:
        analyzer_error = repr(exc)
    return {
        "data_name": extract_data_name(cif),
        "formula": str(struct.composition.reduced_formula),
        "n_sites": int(len(struct)),
        "row_count": int(row_count) if row_count is not None else int(len(struct)),
        "row_count_method": (
            "n_sites_fast_no_sga"
            if fast_row_count
            else ("SpacegroupAnalyzer.equivalent_indices" if row_count is not None else "fallback_n_sites")
        ),
        "sg_number": int(sg_number) if sg_number is not None else None,
        "sg_symbol": sg_symbol or "P 1",
        "analyzer_error": analyzer_error,
    }


def target_cache_path(dataset: str, split: str) -> Path:
    return ROOT / "cache" / f"{dataset_info(dataset)['prefix']}_{split}_targets.jsonl"


def build_target_cache(
    dataset: str,
    split: str,
    refresh: bool = False,
    symprec: float = 0.1,
    fast_row_count: bool = False,
) -> list[dict[str, Any]]:
    cache = target_cache_path(dataset, split)
    if cache.exists() and not refresh:
        return read_jsonl(cache)
    rows = []
    for record in read_csv_records(dataset, split):
        material_id = str(record.get("material_id") or record.get("id") or record.get("Unnamed: 0") or "").strip()
        if not material_id:
            raise ValueError(f"missing material_id in {dataset}/{split}")
        cif = record["cif"]
        explicit = record.get("spacegroup.number")
        explicit_sg = int(float(explicit)) if explicit not in (None, "", "nan") else None
        meta = analyze_target_cif(
            cif,
            explicit_sg_number=explicit_sg,
            symprec=symprec,
            fast_row_count=fast_row_count,
        )
        rows.append({
            "dataset": dataset_info(dataset)["prefix"],
            "split": split,
            "sample_id": sample_id_for(dataset, split, material_id),
            "sample_aliases": sample_aliases_for(dataset, split, material_id),
            "material_id": material_id,
            "csv_index": record.get("") or record.get("Unnamed: 0"),
            "cif": cif,
            **meta,
        })
    write_jsonl(cache, rows)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def make_prompt_tar(dataset: str, split: str, prompt_kind: str, out: Path, refresh_targets: bool = False) -> None:
    targets = build_target_cache(dataset, split, refresh=refresh_targets)
    out = under_root(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for row in targets:
            data_name = row.get("data_name") or row.get("formula") or row["material_id"]
            if prompt_kind == "composition":
                prompt = f"data_{data_name}\n"
            elif prompt_kind == "gt_sg":
                prompt = f"data_{data_name}\n_symmetry_space_group_name_H-M   {quote_sg(row.get('sg_symbol'))}\n"
            else:
                raise ValueError(f"unknown prompt kind: {prompt_kind}")
            info = tarfile.TarInfo(name=f"{row['sample_id']}.txt")
            data = prompt.encode("utf-8")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    write_json(
        out.with_suffix(out.suffix + ".manifest.json"),
        {
            "dataset": dataset_info(dataset)["prefix"],
            "split": split,
            "prompt_kind": prompt_kind,
            "num_prompts": len(targets),
            "out": str(out),
        },
    )


def generated_id_from_tar_member(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".cif"):
        base = base[:-4]
    return base.rsplit("__", 1)[0]


def generated_index_from_tar_member(name: str) -> int:
    base = os.path.basename(name)
    if base.endswith(".cif"):
        base = base[:-4]
    try:
        return int(base.rsplit("__", 1)[1]) - 1
    except Exception:
        return 0


def read_candidates_tar(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            sample_id = generated_id_from_tar_member(member.name)
            gen_index = generated_index_from_tar_member(member.name)
            groups[sample_id].append({
                "sample_id": sample_id,
                "gen_index": gen_index,
                "generated_text": f.read().decode("utf-8", errors="replace"),
                "source_path": str(path),
            })
    for key in groups:
        groups[key].sort(key=lambda r: int(r.get("gen_index", 0)))
    return groups


def candidate_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "cif", "generated_cif", "text"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def read_candidates_jsonl(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id:
                continue
            gen_index = row.get("gen_index", row.get("rank", row.get("seed", line_no)))
            try:
                gen_index = int(gen_index)
            except Exception:
                gen_index = line_no
            groups[sample_id].append({
                **row,
                "sample_id": sample_id,
                "gen_index": gen_index,
                "generated_text": candidate_text(row),
                "source_path": str(path),
                "_line_no": line_no,
            })
    for key in groups:
        groups[key].sort(key=lambda r: (int(r.get("gen_index", 0)), int(r.get("_line_no", 0))))
    return groups


def load_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    if path.suffix == ".jsonl":
        return read_candidates_jsonl(path)
    if str(path).endswith(".tar.gz") or path.suffix == ".tar":
        return read_candidates_tar(path)
    raise ValueError(f"unsupported candidate input: {path}")


def structure_validity(crystal: Structure, cutoff: float = 0.5) -> bool:
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    return bool(dist_mat.min() >= cutoff and crystal.volume >= 0.1)


@lru_cache(maxsize=20000)
def smact_validity_cached(reduced_formula: str) -> bool:
    try:
        comp_obj = Composition(reduced_formula)
        atom_types = []
        for elem, count in comp_obj.get_el_amt_dict().items():
            atom_types.extend([str(elem)] * int(round(count)))
        elem_counter = Counter(atom_types)
        elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        comp, elem_counts = list(zip(*elems))
        elem_counts = np.array(elem_counts)
        elem_counts = elem_counts / np.gcd.reduce(elem_counts)
        count = tuple(elem_counts.astype("int").tolist())
        elem_symbols = tuple(comp)
        if len(set(elem_symbols)) == 1:
            return True
        if all(elem_s in smact.metals for elem_s in elem_symbols):
            return True
        space = smact.element_dictionary(elem_symbols)
        smact_elems = [e[1] for e in space.items()]
        electronegs = [e.pauling_eneg for e in smact_elems]
        ox_combos = [e.oxidation_states for e in smact_elems]
        threshold = np.max(count)
        oxn = 1
        for oxc in ox_combos:
            oxn *= len(oxc)
        if oxn > 1e7:
            return False
        for ox_states in __import__("itertools").product(*ox_combos):
            stoichs = [(c,) for c in count]
            cn_e, _ = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
            if cn_e:
                try:
                    if pauling_test(ox_states, electronegs):
                        return True
                except TypeError:
                    return True
        return False
    except Exception:
        return False


def is_valid_official_like(struct: Structure) -> bool:
    try:
        return smact_validity_cached(struct.composition.reduced_formula) and structure_validity(struct)
    except Exception:
        return False


def safe_rms(matcher: StructureMatcher, pred: Structure, gt: Structure) -> float | None:
    try:
        rms = matcher.get_rms_dist(pred, gt)
        if rms is None:
            return None
        return float(rms[0])
    except EvalTimeoutError:
        raise
    except Exception:
        return None


def eval_one(args: tuple[Any, ...]) -> dict[str, Any]:
    if len(args) == 3:
        target, cands, budget = args
        diagnostics = True
        candidate_timeout_seconds = 0.0
    elif len(args) == 4:
        target, cands, budget, diagnostics = args
        candidate_timeout_seconds = 0.0
        sample_timeout_seconds = 0.0
    elif len(args) == 5:
        target, cands, budget, diagnostics, candidate_timeout_seconds = args
        sample_timeout_seconds = 0.0
    else:
        target, cands, budget, diagnostics, candidate_timeout_seconds, sample_timeout_seconds = args
    candidate_timeout_seconds = float(candidate_timeout_seconds or 0.0)
    sample_timeout_seconds = float(sample_timeout_seconds or 0.0)
    out: dict[str, Any] = {
        "sample_id": target["sample_id"],
        "material_id": target["material_id"],
        "row_count": int(target.get("row_count") or target.get("n_sites") or 0),
        "row_count_method": str(target.get("row_count_method") or "unknown"),
        "n_sites": int(target.get("n_sites") or 0),
        "num_candidates_present": min(len(cands), budget),
        "candidate_parse_ok": 0,
        "candidate_official_valid": 0,
        "candidate_sensible": 0,
        "candidate_timeout": 0,
        "diagnostics_enabled": bool(diagnostics),
        "best_match_index": None,
        "best_rms": None,
    }
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    gt = Structure.from_str(target["cif"], fmt="cif")
    rms_by_rank: list[float | None] = []
    sample_deadline = time.monotonic() + sample_timeout_seconds if sample_timeout_seconds > 0 else None
    cands_for_eval = cands[:budget]
    for cand_i, cand in enumerate(cands_for_eval):
        if sample_deadline is not None and time.monotonic() >= sample_deadline:
            out["candidate_timeout"] += len(cands_for_eval) - cand_i
            rms_by_rank.extend([None] * (len(cands_for_eval) - cand_i))
            break
        cif = candidate_text(cand)
        rms = None
        old_handler = None
        if candidate_timeout_seconds > 0:
            old_handler = signal.signal(signal.SIGALRM, _eval_timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, candidate_timeout_seconds)
        if diagnostics:
            try:
                sensible = is_sensible(cif, 0.5, 1000.0, 10.0, 170.0)
            except Exception:
                sensible = False
            if sensible:
                out["candidate_sensible"] += 1
        try:
            pred = Structure.from_str(cif, fmt="cif")
            out["candidate_parse_ok"] += 1
            if diagnostics and is_valid_official_like(pred):
                out["candidate_official_valid"] += 1
            # CrystaLLM benchmark_metrics.py computes CSP match/RMSE from
            # StructureMatcher directly; validity is diagnostic only.
            rms = safe_rms(matcher, pred, gt)
        except EvalTimeoutError:
            out["candidate_timeout"] += 1
        except Exception:
            rms = None
        finally:
            if candidate_timeout_seconds > 0:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler or signal.SIG_DFL)
        rms_by_rank.append(rms)
    while len(rms_by_rank) < budget:
        rms_by_rank.append(None)

    best = None
    best_i = None
    for i, rms in enumerate(rms_by_rank):
        if rms is None:
            continue
        if best is None or rms < best:
            best = rms
            best_i = i
    out["best_match_index"] = best_i
    out["best_rms"] = best
    for k in (1, 5, 20):
        if k > budget:
            continue
        top = [r for r in rms_by_rank[:k] if r is not None]
        out[f"match@{k}"] = bool(top)
        out[f"RMSE@{k}"] = float(min(top)) if top else None
    gc.collect()
    return out


def sample_timeout_row(
    target: dict[str, Any],
    cands: list[dict[str, Any]],
    budget: int,
    diagnostics: bool,
    reason: str,
) -> dict[str, Any]:
    present = min(len(cands), int(budget))
    out: dict[str, Any] = {
        "sample_id": target["sample_id"],
        "material_id": target["material_id"],
        "row_count": int(target.get("row_count") or target.get("n_sites") or 0),
        "row_count_method": str(target.get("row_count_method") or "unknown"),
        "n_sites": int(target.get("n_sites") or 0),
        "num_candidates_present": present,
        "candidate_parse_ok": 0,
        "candidate_official_valid": 0,
        "candidate_sensible": 0,
        "candidate_timeout": present,
        "diagnostics_enabled": bool(diagnostics),
        "sample_timeout": True,
        "timeout_reason": reason,
        "best_match_index": None,
        "best_rms": None,
    }
    for k in (1, 5, 20):
        if k <= int(budget):
            out[f"match@{k}"] = False
            out[f"RMSE@{k}"] = None
    return out


def _eval_one_child(task: tuple[Any, ...], queue: Queue) -> None:
    try:
        queue.put(eval_one(task))
    except Exception as exc:  # noqa: BLE001
        queue.put({"__eval_error__": f"{type(exc).__name__}: {exc}"})


def eval_one_isolated(task: tuple[Any, ...], timeout_seconds: float) -> dict[str, Any]:
    target = task[0]
    cands = task[1]
    budget = int(task[2])
    diagnostics = bool(task[3]) if len(task) > 3 else True
    timeout_seconds = float(timeout_seconds or 0.0)
    if timeout_seconds <= 0:
        return eval_one(task)

    queue: Queue = Queue(maxsize=1)
    proc = Process(target=_eval_one_child, args=(task, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        return sample_timeout_row(target, cands, budget, diagnostics, "sample process timeout")
    if proc.exitcode not in (0, None):
        return sample_timeout_row(target, cands, budget, diagnostics, f"sample process exitcode {proc.exitcode}")
    try:
        row = queue.get_nowait()
    except Exception:
        return sample_timeout_row(target, cands, budget, diagnostics, "sample process returned no row")
    if isinstance(row, dict) and row.get("__eval_error__"):
        out = sample_timeout_row(target, cands, budget, diagnostics, str(row["__eval_error__"]))
        out["sample_timeout"] = False
        return out
    return row


def finish_isolated_process(proc: Process, queue: Queue, task: tuple[Any, ...]) -> dict[str, Any]:
    target = task[0]
    cands = task[1]
    budget = int(task[2])
    diagnostics = bool(task[3]) if len(task) > 3 else True
    proc.join(0)
    try:
        if proc.exitcode not in (0, None):
            return sample_timeout_row(target, cands, budget, diagnostics, f"sample process exitcode {proc.exitcode}")
        row = queue.get_nowait()
    except Exception:
        return sample_timeout_row(target, cands, budget, diagnostics, "sample process returned no row")
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass
    if isinstance(row, dict) and row.get("__eval_error__"):
        out = sample_timeout_row(target, cands, budget, diagnostics, str(row["__eval_error__"]))
        out["sample_timeout"] = False
        return out
    return row


def run_isolated_task_stream(
    tasks: list[tuple[Any, ...]],
    workers: int,
    timeout_seconds: float,
):
    pending = deque(tasks)
    active: list[dict[str, Any]] = []
    workers = max(1, int(workers))
    timeout_seconds = float(timeout_seconds or 0.0)

    while pending or active:
        while pending and len(active) < workers:
            task = pending.popleft()
            queue: Queue = Queue(maxsize=1)
            proc = Process(target=_eval_one_child, args=(task, queue))
            proc.start()
            active.append({"task": task, "queue": queue, "proc": proc, "started": time.monotonic()})

        made_progress = False
        for item in list(active):
            proc = item["proc"]
            task = item["task"]
            queue = item["queue"]
            expired = timeout_seconds > 0 and (time.monotonic() - float(item["started"])) >= timeout_seconds
            if proc.is_alive() and not expired:
                continue
            if expired and proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
                row = sample_timeout_row(task[0], task[1], int(task[2]), bool(task[3]) if len(task) > 3 else True, "sample process timeout")
                try:
                    queue.close()
                    queue.join_thread()
                except Exception:
                    pass
            else:
                row = finish_isolated_process(proc, queue, task)
            active.remove(item)
            made_progress = True
            yield row
        if not made_progress and active:
            time.sleep(0.1)


def summarize_sample_metrics(rows: list[dict[str, Any]], rows_ge7_only: bool = False) -> dict[str, Any]:
    filt = [r for r in rows if (int(r.get("row_count") or 0) >= 7)] if rows_ge7_only else list(rows)
    denom = len(filt)
    out: dict[str, Any] = {"samples": denom}
    for k in (1, 5, 20):
        hits = [r for r in filt if bool(r.get(f"match@{k}"))]
        rms = [float(r[f"RMSE@{k}"]) for r in hits if r.get(f"RMSE@{k}") is not None]
        out[f"match@{k}"] = float(len(hits) / denom) if denom else None
        out[f"RMSE@{k}"] = float(sum(rms) / len(rms)) if rms else None
        out[f"matched_samples_for_RMSE@{k}"] = len(rms)
    out["positive_any"] = int(sum(1 for r in filt if r.get("best_match_index") is not None))
    out["positive_any_rate"] = float(out["positive_any"] / denom) if denom else None
    return out


def row_count_method_summary(targets: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    counts = Counter(str(r.get("row_count_method") or "unknown") for r in targets)
    text = "; ".join(f"{method}: {count}" for method, count in sorted(counts.items()))
    return text or "unknown", dict(counts)


def refresh_sample_row_counts(
    sample_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_by_alias: dict[str, dict[str, Any]] = {}
    for target in targets:
        for alias in target_aliases(target):
            target_by_alias[alias] = target
    refreshed: list[dict[str, Any]] = []
    for row in sample_rows:
        target = target_by_alias.get(str(row.get("sample_id") or ""))
        if target is None and row.get("material_id") is not None:
            target = target_by_alias.get(str(row.get("material_id")))
        if target is None:
            refreshed.append(row)
            continue
        updated = dict(row)
        updated["row_count"] = int(target.get("row_count") or target.get("n_sites") or 0)
        updated["row_count_method"] = str(target.get("row_count_method") or "unknown")
        updated["n_sites"] = int(target.get("n_sites") or 0)
        refreshed.append(updated)
    return refreshed


def normalize_generation_rows(
    targets: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    system: str,
    budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        group: list[dict[str, Any]] = []
        matched_alias = None
        for alias in target_aliases(target):
            if alias in candidates:
                group = candidates[alias]
                matched_alias = alias
                break
        for i in range(budget):
            if i < len(group):
                cand = group[i]
                rows.append({
                    "system": system,
                    "dataset": target["dataset"],
                    "split": target["split"],
                    "sample_id": target["sample_id"],
                    "material_id": target["material_id"],
                    "gen_index": i,
                    "missing": False,
                    "generated_text": candidate_text(cand),
                    "source_path": cand.get("source_path"),
                    "source_gen_index": cand.get("gen_index"),
                    "matched_candidate_alias": matched_alias,
                })
            else:
                rows.append({
                    "system": system,
                    "dataset": target["dataset"],
                    "split": target["split"],
                    "sample_id": target["sample_id"],
                    "material_id": target["material_id"],
                    "gen_index": i,
                    "missing": True,
                    "generated_text": "",
                })
    return rows


def evaluate_candidates(
    dataset: str,
    split: str,
    system: str,
    candidates_path: Path,
    out_dir: Path,
    budget: int = 20,
    workers: int = 1,
    refresh_targets: bool = False,
    fast_row_count: bool = False,
    write_normalized: bool = True,
    resume: bool = False,
    maxtasksperchild: int | None = None,
    diagnostics: bool = True,
    candidate_timeout_seconds: float = 0.0,
    sample_timeout_seconds: float = 0.0,
    per_sample_process: bool = False,
) -> None:
    targets = build_target_cache(dataset, split, refresh=refresh_targets, fast_row_count=fast_row_count)
    candidates = load_candidates(candidates_path)
    out_dir = under_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    present_counts = {sid: len(rows) for sid, rows in candidates.items()}
    resolved_candidates: dict[str, list[dict[str, Any]]] = {}
    resolved_alias: dict[str, str | None] = {}
    for target in targets:
        chosen: list[dict[str, Any]] = []
        chosen_alias: str | None = None
        for alias in target_aliases(target):
            if alias in candidates:
                chosen = candidates[alias]
                chosen_alias = alias
                break
        resolved_candidates[target["sample_id"]] = chosen
        resolved_alias[target["sample_id"]] = chosen_alias
    sample_path = out_dir / "sample_metrics.jsonl"
    sample_rows: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if resume and sample_path.exists():
        with sample_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(row.get("sample_id") or "")
                if not sample_id or sample_id in done_ids:
                    continue
                sample_rows.append(row)
                done_ids.add(sample_id)

    tasks = [
        (
            target,
            resolved_candidates.get(target["sample_id"], [])[:budget],
            budget,
            bool(diagnostics),
            float(candidate_timeout_seconds or 0.0),
            float(sample_timeout_seconds or 0.0),
        )
        for target in targets
        if str(target.get("sample_id") or "") not in done_ids
    ]
    write_mode = "a" if resume and sample_path.exists() else "w"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open(write_mode, encoding="utf-8") as sf:
        if per_sample_process:
            for row in run_isolated_task_stream(tasks, workers=max(1, int(workers)), timeout_seconds=sample_timeout_seconds):
                sf.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                sf.flush()
                sample_rows.append(row)
                done_ids.add(str(row.get("sample_id") or ""))
                if len(sample_rows) % 100 == 0:
                    gc.collect()
        elif workers > 1 or (workers == 1 and maxtasksperchild):
            mtpc = int(maxtasksperchild) if maxtasksperchild else None
            with Pool(processes=workers, maxtasksperchild=mtpc) as pool:
                for row in pool.imap(eval_one, tasks, chunksize=1 if mtpc else 16):
                    sf.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                    sf.flush()
                    sample_rows.append(row)
                    done_ids.add(str(row.get("sample_id") or ""))
                    if len(sample_rows) % 100 == 0:
                        gc.collect()
        else:
            for task in tasks:
                row = eval_one(task)
                sf.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                sf.flush()
                sample_rows.append(row)
                done_ids.add(str(row.get("sample_id") or ""))
                if len(sample_rows) % 100 == 0:
                    gc.collect()
    sample_rows.sort(key=lambda r: str(r.get("sample_id") or ""))

    row_method_text, row_method_counts = row_count_method_summary(targets)
    metrics = {
        "system": system,
        "dataset": dataset_info(dataset)["prefix"],
        "split": split,
        "candidate_source": str(candidates_path),
        "budget": budget,
        "official_split_samples": len(targets),
        "samples_with_any_candidate": int(sum(1 for target in targets if resolved_candidates.get(target["sample_id"]))),
        "total_candidate_records_seen": int(sum(min(len(resolved_candidates.get(target["sample_id"], [])), budget) for target in targets)),
        "candidate_id_aliasing": "targets are matched by sample_id, prefix_split__material_id, or material_id",
        "matched_candidate_alias_counts": dict(Counter(alias for alias in resolved_alias.values() if alias is not None)),
        "all": summarize_sample_metrics(sample_rows, rows_ge7_only=False),
        "rows_ge7": summarize_sample_metrics(sample_rows, rows_ge7_only=True),
        "row_count_method": f"target cache row_count >= 7; {row_method_text}",
        "row_count_method_counts": row_method_counts,
        "structure_matcher": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10},
        "official_validity": "diagnostic only; CSP match/RMSE follows benchmark_metrics.py and is independent of sensibility/validity",
    }
    write_json(out_dir / "summary.json", metrics)
    write_jsonl(sample_path, sample_rows)
    if write_normalized:
        norm = normalize_generation_rows(targets, candidates, system=system, budget=budget)
        write_jsonl(ROOT / "generations" / f"{system}_{dataset_info(dataset)['prefix']}_{split}_k{budget}.jsonl", norm)
    write_json(ROOT / "metrics" / f"{system}_{dataset_info(dataset)['prefix']}_{split}_k{budget}.json", metrics)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} evaluate {system} {dataset}/{split}\n"
        f"- candidate source: `{candidates_path}`\n"
        f"- official samples: {metrics['official_split_samples']}; with candidates: {metrics['samples_with_any_candidate']}\n"
        f"- all match@1/5/20: {metrics['all']['match@1']} / {metrics['all']['match@5']} / {metrics['all']['match@20']}\n"
        f"- all RMSE@1/5/20: {metrics['all']['RMSE@1']} / {metrics['all']['RMSE@5']} / {metrics['all']['RMSE@20']}\n"
        f"- rows>=7 samples: {metrics['rows_ge7']['samples']}; match@1/5/20: "
        f"{metrics['rows_ge7']['match@1']} / {metrics['rows_ge7']['match@5']} / {metrics['rows_ge7']['match@20']}; "
        f"positive-any: {metrics['rows_ge7']['positive_any']}"
    )


def recompute_eval_summary(
    dataset: str,
    split: str,
    system: str,
    out_dir: Path,
    budget: int = 20,
    refresh_targets: bool = False,
    fast_row_count: bool = False,
) -> None:
    targets = build_target_cache(dataset, split, refresh=refresh_targets, fast_row_count=fast_row_count)
    out_dir = under_root(out_dir)
    sample_path = out_dir / "sample_metrics.jsonl"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    sample_rows = refresh_sample_row_counts(read_jsonl(sample_path), targets)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        metrics = {
            "system": system,
            "dataset": dataset_info(dataset)["prefix"],
            "split": split,
            "budget": budget,
        }
    row_method_text, row_method_counts = row_count_method_summary(targets)
    metrics.update(
        {
            "system": system,
            "dataset": dataset_info(dataset)["prefix"],
            "split": split,
            "budget": budget,
            "official_split_samples": len(targets),
            "all": summarize_sample_metrics(sample_rows, rows_ge7_only=False),
            "rows_ge7": summarize_sample_metrics(sample_rows, rows_ge7_only=True),
            "row_count_method": f"target cache row_count >= 7; {row_method_text}",
            "row_count_method_counts": row_method_counts,
            "structure_matcher": {"ltol": 0.3, "stol": 0.5, "angle_tol": 10},
            "official_validity": "diagnostic only; CSP match/RMSE follows benchmark_metrics.py and is independent of sensibility/validity",
        }
    )
    write_json(summary_path, metrics)
    write_jsonl(sample_path, sample_rows)
    write_json(ROOT / "metrics" / f"{system}_{dataset_info(dataset)['prefix']}_{split}_k{budget}.json", metrics)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} recompute summary {system} {dataset}/{split}\n"
        f"- official samples: {metrics['official_split_samples']}\n"
        f"- row_count_method: {metrics['row_count_method']}\n"
        f"- all match@1/5/20: {metrics['all']['match@1']} / {metrics['all']['match@5']} / {metrics['all']['match@20']}\n"
        f"- rows>=7 samples: {metrics['rows_ge7']['samples']}; match@1/5/20: "
        f"{metrics['rows_ge7']['match@1']} / {metrics['rows_ge7']['match@5']} / {metrics['rows_ge7']['match@20']}; "
        f"positive-any: {metrics['rows_ge7']['positive_any']}"
    )


def resolve_read_path(path_text: str, base_dir: Path | None = None) -> Path:
    if not str(path_text or "").strip():
        return Path("__missing__")
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(WORKSPACE / path)
    candidates.append(ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def convert_stablekey_top20(input_path: Path, out_path: Path, system: str, max_rank: int = 20) -> None:
    input_path = input_path.resolve()
    out_path = under_root(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    samples = 0
    missing_cifs = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            sample = json.loads(line)
            samples += 1
            sample_id = str(sample.get("sample_id") or "")
            preds = list(sample.get("predictions") or [])
            preds.sort(key=lambda p: int(p.get("rank") or p.get("gen_index") or 0))
            for pred_i, pred in enumerate(preds[:max_rank]):
                rank = int(pred.get("rank") or (pred_i + 1))
                cif_path = resolve_read_path(str(pred.get("cif_path") or ""), input_path.parent)
                generated_text = ""
                if cif_path.exists():
                    generated_text = cif_path.read_text(encoding="utf-8", errors="replace")
                else:
                    missing_cifs += 1
                rows_out.append({
                    "system": system,
                    "sample_id": sample_id,
                    "material_id": sample_id.rsplit("__", 1)[-1] if "__" in sample_id else None,
                    "gen_index": rank - 1,
                    "rank": rank,
                    "generated_text": generated_text,
                    "missing": not bool(generated_text),
                    "source_path": str(input_path),
                    "source_line_no": line_no,
                    "source_cif_path": str(cif_path),
                    "source_labels": pred.get("source_labels"),
                    "geometry_source": pred.get("geometry_source"),
                    "selection_mode": pred.get("selection_mode"),
                    "render_success": pred.get("render_success"),
                })
    write_jsonl(out_path, rows_out)
    write_json(
        out_path.with_suffix(out_path.suffix + ".manifest.json"),
        {
            "system": system,
            "input_path": str(input_path),
            "out_path": str(out_path),
            "samples": samples,
            "candidate_rows": len(rows_out),
            "max_rank": max_rank,
            "missing_cifs": missing_cifs,
        },
    )
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} convert stablekey {system}\n"
        f"- input: `{input_path}`\n"
        f"- output: `{out_path}`\n"
        f"- samples: {samples}; candidate rows: {len(rows_out)}; missing CIFs: {missing_cifs}"
    )


def cmd_prepare_targets(args: argparse.Namespace) -> None:
    for dataset in args.dataset:
        for split in args.split:
            rows = build_target_cache(
                dataset,
                split,
                refresh=args.refresh,
                symprec=args.symprec,
                fast_row_count=args.fast_row_count,
            )
            append_log(
                f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} target cache {dataset}/{split}\n"
                f"- samples: {len(rows)}\n"
                f"- rows>=7: {sum(1 for r in rows if int(r.get('row_count') or 0) >= 7)}\n"
                f"- fast_row_count: {args.fast_row_count}\n"
                f"- cache: `{target_cache_path(dataset, split)}`"
            )


def cmd_make_prompts(args: argparse.Namespace) -> None:
    make_prompt_tar(args.dataset, args.split, args.prompt_kind, Path(args.out), refresh_targets=args.refresh_targets)
    append_log(
        f"## {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} prompts {args.dataset}/{args.split}/{args.prompt_kind}\n"
        f"- out: `{args.out}`"
    )


def cmd_eval(args: argparse.Namespace) -> None:
    evaluate_candidates(
        dataset=args.dataset,
        split=args.split,
        system=args.system,
        candidates_path=Path(args.candidates),
        out_dir=Path(args.out_dir),
        budget=args.budget,
        workers=args.workers,
        refresh_targets=args.refresh_targets,
        fast_row_count=args.fast_row_count,
        write_normalized=not args.no_normalized,
        resume=args.resume,
        maxtasksperchild=args.maxtasksperchild,
        diagnostics=not args.match_only,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
        sample_timeout_seconds=args.sample_timeout_seconds,
        per_sample_process=args.per_sample_process,
    )


def cmd_recompute_summary(args: argparse.Namespace) -> None:
    recompute_eval_summary(
        dataset=args.dataset,
        split=args.split,
        system=args.system,
        out_dir=Path(args.out_dir),
        budget=args.budget,
        refresh_targets=args.refresh_targets,
        fast_row_count=args.fast_row_count,
    )


def cmd_convert_stablekey(args: argparse.Namespace) -> None:
    convert_stablekey_top20(
        input_path=Path(args.input),
        out_path=Path(args.out),
        system=args.system,
        max_rank=args.max_rank,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="opentry_7 full benchmark utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-targets")
    p.add_argument("--dataset", nargs="+", choices=["mp20", "mpts52"], required=True)
    p.add_argument("--split", nargs="+", choices=["train", "val", "test"], required=True)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--symprec", type=float, default=0.1)
    p.add_argument("--fast-row-count", action="store_true")
    p.set_defaults(func=cmd_prepare_targets)

    p = sub.add_parser("make-prompts")
    p.add_argument("--dataset", choices=["mp20", "mpts52"], required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--prompt-kind", choices=["composition", "gt_sg"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--refresh-targets", action="store_true")
    p.set_defaults(func=cmd_make_prompts)

    p = sub.add_parser("eval")
    p.add_argument("--dataset", choices=["mp20", "mpts52"], required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--refresh-targets", action="store_true")
    p.add_argument("--fast-row-count", action="store_true")
    p.add_argument("--no-normalized", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--maxtasksperchild", type=int, default=None)
    p.add_argument("--match-only", action="store_true")
    p.add_argument("--candidate-timeout-seconds", type=float, default=0.0)
    p.add_argument("--sample-timeout-seconds", type=float, default=0.0)
    p.add_argument("--per-sample-process", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("recompute-summary")
    p.add_argument("--dataset", choices=["mp20", "mpts52"], required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--refresh-targets", action="store_true")
    p.add_argument("--fast-row-count", action="store_true")
    p.set_defaults(func=cmd_recompute_summary)

    p = sub.add_parser("convert-stablekey-top20")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--max-rank", type=int, default=20)
    p.set_defaults(func=cmd_convert_stablekey)
    return parser


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
