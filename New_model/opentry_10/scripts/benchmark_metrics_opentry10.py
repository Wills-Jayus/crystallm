import os
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from collections import Counter, deque
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import itertools
from tqdm import tqdm
import multiprocessing as mp

try:
    import signal  # type: ignore
except Exception:  # noqa: BLE001
    signal = None  # type: ignore[assignment]

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Optional imports: only required for unconditional metrics (COV/Wasserstein/fingerprints).
# Keep CSP match-rate metrics runnable even if these heavy deps (or their transitive deps)
# are unavailable or version-incompatible in the environment.
_UNCONDITIONAL_DEPS_ERROR: str | None = None
try:  # noqa: WPS501
    import smact  # type: ignore
    from smact.screening import pauling_test  # type: ignore
    from scipy.stats import wasserstein_distance  # type: ignore
    from scipy.spatial.distance import cdist  # type: ignore
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint  # type: ignore
    from matminer.featurizers.composition.composite import ElementProperty  # type: ignore
except Exception as exc:  # noqa: BLE001
    _UNCONDITIONAL_DEPS_ERROR = f"{type(exc).__name__}: {exc}"
    smact = None  # type: ignore[assignment]
    pauling_test = None  # type: ignore[assignment]
    wasserstein_distance = None  # type: ignore[assignment]
    cdist = None  # type: ignore[assignment]
    CrystalNNFingerprint = None  # type: ignore[assignment]
    ElementProperty = None  # type: ignore[assignment]

from crystallm import (
    extract_data_formula,
    extract_space_group_symbol,
    is_sensible,
    replace_symmetry_operators,
)

import warnings
warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_CRYSTAL_NN_FP = None
_COMP_FP = None

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.},
    "carbon": {"struc": 0.2, "comp": 4.},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class StandardScaler:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def transform(self, X):
        X = np.array(X).astype(float)
        return (X - self.means) / self.stds


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def smact_validity(atom_types, use_pauling_test=True, include_alloys=True):
    if smact is None:
        raise RuntimeError(f"unconditional deps unavailable: {_UNCONDITIONAL_DEPS_ERROR}")
    # atom_types e.g. ["Fe", "Fe", "O", "O", "O"]
    elem_counter = Counter(atom_types)
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))
    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())

    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def get_comp_fingerprint(struct):
    global _COMP_FP  # noqa: PLW0603
    if ElementProperty is None:
        raise RuntimeError(f"unconditional deps unavailable: {_UNCONDITIONAL_DEPS_ERROR}")
    if _COMP_FP is None:
        _COMP_FP = ElementProperty.from_preset("magpie")
    atom_types = [str(specie) for specie in struct.species]
    elem_counter = Counter(atom_types)
    comp = Composition(elem_counter)
    fp = _COMP_FP.featurize(comp)
    if np.isnan(fp).any():
        return None
    return fp


def get_struct_fingerprint(struct):
    global _CRYSTAL_NN_FP  # noqa: PLW0603
    if CrystalNNFingerprint is None:
        raise RuntimeError(f"unconditional deps unavailable: {_UNCONDITIONAL_DEPS_ERROR}")
    if _CRYSTAL_NN_FP is None:
        # NOTE: Some matminer/ruamel.yaml versions are incompatible (ruamel removed safe_load).
        # We keep this lazy so CSP match-rate metrics don't fail on import.
        _CRYSTAL_NN_FP = CrystalNNFingerprint.from_preset("ops")
    try:
        site_fps = [_CRYSTAL_NN_FP.featurize(struct, i) for i in range(len(struct))]
    except Exception:
        return None
    return np.array(site_fps).mean(axis=0)


# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def is_valid(struct):
    comp_valid = smact_validity(
        atom_types=[str(specie) for specie in struct.species]
    )
    struct_valid = structure_validity(struct)
    return comp_valid and struct_valid


def is_valid_unconditional(struct, fp):
    return is_valid(struct) and fp is not None


# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def compute_cov(gen_structs, true_structs, struc_cutoff, comp_cutoff, comp_scaler, num_gen_crystals=None):
    struc_fps = [struct_fp for _, struct_fp, _ in gen_structs]
    comp_fps = [comp_fp for _, _, comp_fp in gen_structs]
    gt_struc_fps = [struct_fp for _, struct_fp, _ in true_structs]
    gt_comp_fps = [comp_fp for _, _, comp_fp in true_structs]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
    # there may be odd cases when ground-truth CIFs may result in
    #  fingerprints with nan values; in those cases, we return None
    #  instead of the fingerprint, and consolidate those entries here
    gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

    comp_fps = comp_scaler.transform(comp_fps)
    gt_comp_fps = comp_scaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        "struc_recall_dist": struc_recall_dist.tolist(),
        "struc_precision_dist": struc_precision_dist.tolist(),
        "comp_recall_dist": comp_recall_dist.tolist(),
        "comp_precision_dist": comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py
def get_match_rate_and_rms(gen_structs, true_structs, matcher):
    """
    Compute match@K (any-match) and mean RMSD.

    IMPORTANT (experiment setting):
    - Matching is computed independently of is_sensible/is_valid.
    - If a generated CIF cannot be parsed to a Structure, it cannot be matched.
    - If StructureMatcher fails or returns None, that candidate is treated as unmatched.
    """

    class _RmsdTimeout(Exception):
        pass

    @contextlib.contextmanager
    def _time_limit(seconds: float | None):
        if seconds is None or seconds <= 0 or signal is None:
            yield
            return

        def _handler(signum, frame):  # noqa: ARG001
            raise _RmsdTimeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    max_sites = globals().get("_BENCH_MAX_SITES", None)
    timeout_s = globals().get("_BENCH_RMSD_TIMEOUT_S", None)
    n_timeout = 0
    n_skipped_large = 0
    n_error = 0

    def process_one(pred, gt):
        nonlocal n_timeout, n_skipped_large, n_error
        if max_sites is not None:
            try:
                if int(pred.num_sites) > int(max_sites) or int(gt.num_sites) > int(max_sites):
                    n_skipped_large += 1
                    return None
            except Exception:
                pass
        try:
            with _time_limit(timeout_s):
                rms_dist = matcher.get_rms_dist(pred, gt)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except _RmsdTimeout:
            n_timeout += 1
            return None
        except Exception:
            n_error += 1
            return None

    rms_dists = []
    for i in tqdm(range(len(gen_structs)), desc="comparing structures..."):
        tmp_rms_dists = []
        for j in range(len(gen_structs[i])):
            try:
                rmsd = process_one(gen_structs[i][j], true_structs[i])
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            except Exception:
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists, dtype=object)
    matched_mask = np.array([v is not None for v in rms_dists], dtype=bool)
    match_rate = float(matched_mask.sum() / max(1, len(gen_structs)))
    mean_rms_dist = None if matched_mask.sum() == 0 else float(np.array(rms_dists[matched_mask], dtype=float).mean())
    return {
        "match_rate": match_rate,
        "rms_dist": mean_rms_dist,
        "match_timeouts": int(n_timeout),
        "match_skipped_large": int(n_skipped_large),
        "match_errors": int(n_error),
        "bench_max_sites": None if max_sites is None else int(max_sites),
        "bench_rmsd_timeout_s": None if timeout_s is None else float(timeout_s),
    }


def _normalize_cif_symmops_to_declared_sg(cif: str) -> str:
    """
    Best-effort normalize CIF symmetry operators to the declared space group.

    This matches the behavior in get_structs(): for non-P1 CIFs, replace/expand the symmop loop so that
    CIFs containing only identity still expand correctly during parsing.
    """
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


_UNMATCHED_DIAGNOSTIC_FIELDS = [
    "parse_failed",
    "formula_mismatch",
    "element_set_mismatch",
    "composition_mismatch",
    "stated_sg_mismatch",
    "normalized_detected_sg_mismatch",
    "site_count_mismatch",
    "cell_length_mismatch",
    "cell_angle_mismatch",
    "volume_per_atom_mismatch",
    "density_mismatch",
    "wyckoff_multiset_mismatch",
    "sensible_false",
    "valid_false",
    "large_structure_skipped",
    "match_timeout_or_error",
]


def _rate(num, den):
    return None if den == 0 else float(num / den)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _safe_rel_error(a, b):
    try:
        a = float(a)
        b = float(b)
        denom = max(abs(b), 1e-12)
        return abs(a - b) / denom
    except Exception:
        return None


def _normalize_sg_symbol(sg):
    if sg is None:
        return None
    return str(sg).replace(" ", "").replace("_", "").strip().upper()


def _counter_to_jsonable(counter):
    return {"|".join(str(x) for x in key): int(value) for key, value in sorted(counter.items())}


def _composition_fraction_dict(comp):
    try:
        total = float(sum(comp.values()))
        if total <= 0:
            return None
        return {str(el): float(amount) / total for el, amount in comp.items()}
    except Exception:
        return None


def _composition_l1(comp_a, comp_b):
    fa = _composition_fraction_dict(comp_a)
    fb = _composition_fraction_dict(comp_b)
    if fa is None or fb is None:
        return None
    keys = set(fa) | set(fb)
    return float(sum(abs(float(fa.get(k, 0.0)) - float(fb.get(k, 0.0))) for k in keys))


def _composition_from_formula_text(cif):
    try:
        formula = extract_data_formula(cif)
        comp = Composition(formula)
        if len(comp) == 0:
            return None
        return comp
    except Exception:
        return None


def _reduced_formula_from_comp(comp):
    try:
        return Composition(comp).reduced_formula
    except Exception:
        return None


def _element_set_from_comp(comp):
    try:
        return sorted(str(el) for el in Composition(comp).elements)
    except Exception:
        return None


def _dataset_get(dataset, key):
    if hasattr(dataset, key):
        return getattr(dataset, key)
    try:
        return dataset[key]
    except Exception:
        return None


def normalize_for_benchmark_parse(cif):
    return _normalize_cif_symmops_to_declared_sg(cif)


def extract_declared_fields(cif):
    fields = {
        "data_formula": None,
        "data_formula_reduced": None,
        "stated_sg": None,
        "stated_sg_norm": None,
    }
    try:
        fields["data_formula"] = extract_data_formula(cif)
        comp = Composition(fields["data_formula"])
        fields["data_formula_reduced"] = comp.reduced_formula
    except Exception:
        pass
    try:
        fields["stated_sg"] = extract_space_group_symbol(cif)
        fields["stated_sg_norm"] = _normalize_sg_symbol(fields["stated_sg"])
    except Exception:
        pass
    return fields


def wyckoff_signature(structure):
    sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5)
    dataset = sga.get_symmetry_dataset()
    wyckoffs = _dataset_get(dataset, "wyckoffs")
    equivalent_atoms = _dataset_get(dataset, "equivalent_atoms")
    if wyckoffs is None:
        raise ValueError("missing wyckoffs in symmetry dataset")
    if equivalent_atoms is None:
        equivalent_atoms = list(range(len(structure)))

    orbits = {}
    for idx, orbit_id in enumerate(list(equivalent_atoms)):
        orbits.setdefault(int(orbit_id), []).append(int(idx))

    signature = Counter()
    for indices in orbits.values():
        if not indices:
            continue
        multiplicity = len(indices)
        representative = indices[0]
        letter = str(wyckoffs[representative])
        elems = Counter(str(structure[i].specie.symbol) for i in indices)
        for elem, count in elems.items():
            signature[(elem, letter, int(multiplicity))] += int(count)
    return signature


def structure_field_summary(cif, gt_cif=None):  # noqa: ARG001 - gt_cif kept for the public diagnostic API.
    declared = extract_declared_fields(cif)
    normalized_cif = normalize_for_benchmark_parse(cif)
    summary = {
        "declared": declared,
        "parse_error": None,
        "parsed": False,
        "formula": None,
        "elements": None,
        "composition": None,
        "num_sites": None,
        "lengths": None,
        "angles": None,
        "volume": None,
        "volume_per_atom": None,
        "density": None,
        "detected_sg": None,
        "detected_sg_norm": None,
        "wyckoff_signature": None,
        "_structure": None,
        "_composition": None,
    }
    try:
        struct = Structure.from_str(normalized_cif, fmt="cif")
        summary["_structure"] = struct
        summary["parsed"] = True
    except Exception as exc:  # noqa: BLE001
        summary["parse_error"] = f"{type(exc).__name__}: {exc}"
        comp = _composition_from_formula_text(cif)
        if comp is not None:
            summary["_composition"] = comp
            summary["formula"] = _reduced_formula_from_comp(comp)
            summary["elements"] = _element_set_from_comp(comp)
            summary["composition"] = _composition_fraction_dict(comp)
        return summary

    struct = summary["_structure"]
    comp = struct.composition
    summary["_composition"] = comp
    summary["formula"] = _reduced_formula_from_comp(comp)
    summary["elements"] = _element_set_from_comp(comp)
    summary["composition"] = _composition_fraction_dict(comp)
    summary["num_sites"] = int(struct.num_sites)
    summary["lengths"] = [float(x) for x in struct.lattice.abc]
    summary["angles"] = [float(x) for x in struct.lattice.angles]
    summary["volume"] = float(struct.volume)
    summary["volume_per_atom"] = float(struct.volume / max(1, int(struct.num_sites)))
    summary["density"] = float(struct.density)
    try:
        detected = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5).get_space_group_symbol()
        summary["detected_sg"] = detected
        summary["detected_sg_norm"] = _normalize_sg_symbol(detected)
    except Exception:
        pass
    try:
        summary["wyckoff_signature"] = _counter_to_jsonable(wyckoff_signature(struct))
    except Exception:
        pass
    return summary


def _set_field(record, field, status, detail=None):
    record["field_status"][field] = status
    record["field_mismatch"][field] = True if status == "mismatch" else False if status == "match" else None
    if detail is not None:
        record["field_detail"][field] = detail


def _same_or_unavailable(record, field, gen_value, gt_value):
    if gen_value is None or gt_value is None:
        _set_field(record, field, "unavailable", {"generated": gen_value, "gt": gt_value})
        return
    _set_field(
        record,
        field,
        "match" if gen_value == gt_value else "mismatch",
        {"generated": gen_value, "gt": gt_value},
    )


def _threshold_mismatch(record, field, value, threshold, detail):
    if value is None:
        _set_field(record, field, "unavailable", detail)
        return
    detail = dict(detail)
    detail["error"] = float(value)
    detail["threshold"] = float(threshold)
    _set_field(record, field, "mismatch" if float(value) > float(threshold) else "match", detail)


def _rms_match_status(pred, gt, matcher, timeout_s):
    if pred is None or gt is None:
        return None, False

    class _RmsdTimeout(Exception):
        pass

    @contextlib.contextmanager
    def _time_limit(seconds):
        if seconds is None or seconds <= 0 or signal is None:
            yield
            return

        def _handler(signum, frame):  # noqa: ARG001
            raise _RmsdTimeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    try:
        with _time_limit(timeout_s):
            rms_dist = matcher.get_rms_dist(pred, gt)
        return (None if rms_dist is None else float(rms_dist[0])), False
    except Exception:
        return None, True


def diagnose_candidate(
    candidate_cif,
    gt_cif,
    candidate_idx,
    *,
    matcher,
    length_lo,
    length_hi,
    angle_lo,
    angle_hi,
    max_sites,
    rmsd_timeout_s,
    compute_rms_status=False,
    gt_summary=None,
):
    record = {
        "candidate_idx": int(candidate_idx),
        "field_status": {},
        "field_mismatch": {},
        "field_detail": {},
        "metrics": {},
        "generated": {},
        "gt": {},
    }

    for field in _UNMATCHED_DIAGNOSTIC_FIELDS:
        _set_field(record, field, "unavailable")

    sensible = False
    try:
        sensible = bool(is_sensible(candidate_cif, length_lo, length_hi, angle_lo, angle_hi))
        _set_field(record, "sensible_false", "match" if sensible else "mismatch", {"sensible": sensible})
    except Exception as exc:  # noqa: BLE001
        _set_field(record, "sensible_false", "unavailable", {"error": f"{type(exc).__name__}: {exc}"})

    gen_summary = structure_field_summary(candidate_cif, gt_cif)
    if gt_summary is None:
        gt_summary = structure_field_summary(gt_cif)
    gen_struct = gen_summary.get("_structure")
    gt_struct = gt_summary.get("_structure")
    gen_comp = gen_summary.get("_composition")
    gt_comp = gt_summary.get("_composition")

    record["generated"] = {k: v for k, v in gen_summary.items() if not k.startswith("_")}
    record["gt"] = {k: v for k, v in gt_summary.items() if not k.startswith("_")}

    _set_field(
        record,
        "parse_failed",
        "mismatch" if not bool(gen_summary.get("parsed")) else "match",
        {"parse_error": gen_summary.get("parse_error")},
    )

    _same_or_unavailable(record, "formula_mismatch", gen_summary.get("formula"), gt_summary.get("formula"))
    _same_or_unavailable(record, "element_set_mismatch", gen_summary.get("elements"), gt_summary.get("elements"))

    comp_l1 = _composition_l1(gen_comp, gt_comp) if gen_comp is not None and gt_comp is not None else None
    record["metrics"]["composition_l1"] = comp_l1
    _threshold_mismatch(
        record,
        "composition_mismatch",
        comp_l1,
        1e-8,
        {"generated": gen_summary.get("composition"), "gt": gt_summary.get("composition")},
    )

    _same_or_unavailable(
        record,
        "stated_sg_mismatch",
        gen_summary.get("declared", {}).get("stated_sg_norm"),
        gt_summary.get("declared", {}).get("stated_sg_norm"),
    )
    _same_or_unavailable(
        record,
        "normalized_detected_sg_mismatch",
        gen_summary.get("detected_sg_norm"),
        gt_summary.get("detected_sg_norm"),
    )
    _same_or_unavailable(record, "site_count_mismatch", gen_summary.get("num_sites"), gt_summary.get("num_sites"))

    if gen_summary.get("lengths") is not None and gt_summary.get("lengths") is not None:
        length_rel_errors = [
            _safe_rel_error(a, b) for a, b in zip(gen_summary["lengths"], gt_summary["lengths"])
        ]
        length_max = None if any(v is None for v in length_rel_errors) else float(max(length_rel_errors))
    else:
        length_rel_errors = None
        length_max = None
    record["metrics"]["cell_length_rel_error_max"] = length_max
    _threshold_mismatch(
        record,
        "cell_length_mismatch",
        length_max,
        0.30,
        {"relative_errors": length_rel_errors, "generated": gen_summary.get("lengths"), "gt": gt_summary.get("lengths")},
    )

    if gen_summary.get("angles") is not None and gt_summary.get("angles") is not None:
        angle_abs_errors = [abs(float(a) - float(b)) for a, b in zip(gen_summary["angles"], gt_summary["angles"])]
        angle_max = float(max(angle_abs_errors))
    else:
        angle_abs_errors = None
        angle_max = None
    record["metrics"]["cell_angle_abs_error_max"] = angle_max
    _threshold_mismatch(
        record,
        "cell_angle_mismatch",
        angle_max,
        10.0,
        {"absolute_errors_deg": angle_abs_errors, "generated": gen_summary.get("angles"), "gt": gt_summary.get("angles")},
    )

    vpa_rel = _safe_rel_error(gen_summary.get("volume_per_atom"), gt_summary.get("volume_per_atom"))
    record["metrics"]["volume_per_atom_rel_error"] = vpa_rel
    _threshold_mismatch(
        record,
        "volume_per_atom_mismatch",
        vpa_rel,
        0.30,
        {"generated": gen_summary.get("volume_per_atom"), "gt": gt_summary.get("volume_per_atom")},
    )

    density_rel = _safe_rel_error(gen_summary.get("density"), gt_summary.get("density"))
    record["metrics"]["density_rel_error"] = density_rel
    _threshold_mismatch(
        record,
        "density_mismatch",
        density_rel,
        0.30,
        {"generated": gen_summary.get("density"), "gt": gt_summary.get("density")},
    )

    _same_or_unavailable(
        record,
        "wyckoff_multiset_mismatch",
        gen_summary.get("wyckoff_signature"),
        gt_summary.get("wyckoff_signature"),
    )

    if gen_struct is None:
        _set_field(record, "valid_false", "unavailable", {"reason": "generated structure parse failed"})
    else:
        try:
            valid = bool(is_valid(gen_struct))
            _set_field(record, "valid_false", "match" if valid else "mismatch", {"valid": valid})
        except Exception as exc:  # noqa: BLE001
            _set_field(record, "valid_false", "unavailable", {"error": f"{type(exc).__name__}: {exc}"})

    large_skipped = False
    if gen_struct is None or gt_struct is None:
        _set_field(record, "large_structure_skipped", "unavailable", {"reason": "parse failed"})
    else:
        try:
            large_skipped = max_sites is not None and (
                int(gen_struct.num_sites) > int(max_sites) or int(gt_struct.num_sites) > int(max_sites)
            )
            _set_field(
                record,
                "large_structure_skipped",
                "mismatch" if large_skipped else "match",
                {"max_sites": max_sites, "generated_sites": int(gen_struct.num_sites), "gt_sites": int(gt_struct.num_sites)},
            )
        except Exception as exc:  # noqa: BLE001
            _set_field(record, "large_structure_skipped", "unavailable", {"error": f"{type(exc).__name__}: {exc}"})

    if not compute_rms_status:
        _set_field(record, "match_timeout_or_error", "unavailable", {"reason": "per-candidate RMSD status not recomputed"})
        record["metrics"]["rmsd"] = None
    elif gen_struct is None or gt_struct is None or large_skipped:
        _set_field(record, "match_timeout_or_error", "unavailable", {"reason": "parse failed or large structure skipped"})
        record["metrics"]["rmsd"] = None
    else:
        rmsd, rmsd_error = _rms_match_status(gen_struct, gt_struct, matcher, rmsd_timeout_s)
        record["metrics"]["rmsd"] = rmsd
        _set_field(record, "match_timeout_or_error", "mismatch" if rmsd_error else "match", {"rmsd_error": rmsd_error})

    matched_fields = 0
    mismatch_fields = 0
    unavailable_fields = 0
    for field in _UNMATCHED_DIAGNOSTIC_FIELDS:
        status = record["field_status"].get(field, "unavailable")
        if status == "match":
            matched_fields += 1
        elif status == "mismatch":
            mismatch_fields += 1
        else:
            unavailable_fields += 1
    record["metrics"]["matched_field_count"] = int(matched_fields)
    record["metrics"]["mismatch_field_count"] = int(mismatch_fields)
    record["metrics"]["unavailable_field_count"] = int(unavailable_fields)
    return record


def choose_best_diagnostic_candidate(candidate_diagnostics):
    if not candidate_diagnostics:
        return None

    def _sort_key(record):
        parsed_rank = 0 if record["field_status"].get("parse_failed") == "match" else 1
        matched = -int(record.get("metrics", {}).get("matched_field_count", 0))
        comp_l1 = record.get("metrics", {}).get("composition_l1")
        if comp_l1 is None:
            comp_l1 = float("inf")
        vpa = record.get("metrics", {}).get("volume_per_atom_rel_error")
        if vpa is None:
            vpa = float("inf")
        return (parsed_rank, matched, float(comp_l1), float(vpa), int(record.get("candidate_idx", 10**9)))

    return sorted(candidate_diagnostics, key=_sort_key)[0]


def _candidate_list_for_k(cifs, n_gens):
    if n_gens is None:
        return list(cifs)
    return list(cifs[: int(n_gens)])


def _is_material_matched_for_diagnostics(gen_cifs, true_cif, n_gens, matcher, max_sites, rmsd_timeout_s):
    try:
        gt_struct = Structure.from_str(normalize_for_benchmark_parse(true_cif), fmt="cif")
    except Exception:
        return False
    for cif in _candidate_list_for_k(gen_cifs, n_gens):
        try:
            pred = Structure.from_str(normalize_for_benchmark_parse(cif), fmt="cif")
        except Exception:
            continue
        try:
            if max_sites is not None and (int(pred.num_sites) > int(max_sites) or int(gt_struct.num_sites) > int(max_sites)):
                continue
        except Exception:
            pass
        rmsd, rmsd_error = _rms_match_status(pred, gt_struct, matcher, rmsd_timeout_s)
        if not rmsd_error and rmsd is not None:
            return True
    return False


def _diagnose_unmatched_material(mid, gen_cifs, true_cif, n_gens, matcher, length_lo, length_hi, angle_lo, angle_hi, max_sites, rmsd_timeout_s):
    gen_cifs = _candidate_list_for_k(gen_cifs, n_gens)
    diagnostics = []
    gt_summary = structure_field_summary(true_cif)
    for idx, cif in enumerate(gen_cifs, start=1):
        diag = diagnose_candidate(
            cif,
            true_cif,
            idx,
            matcher=matcher,
            length_lo=length_lo,
            length_hi=length_hi,
            angle_lo=angle_lo,
            angle_hi=angle_hi,
            max_sites=max_sites,
            rmsd_timeout_s=rmsd_timeout_s,
            compute_rms_status=False,
            gt_summary=gt_summary,
        )
        diag["id"] = mid
        diagnostics.append(diag)

    best = choose_best_diagnostic_candidate(diagnostics)
    best_idx = None if best is None else int(best["candidate_idx"])
    best_status = {} if best is None else best.get("field_status", {})
    material_record = {
        "id": mid,
        "K": None if n_gens is None else int(n_gens),
        "n_candidates": int(len(gen_cifs)),
        "best_candidate_idx": best_idx,
        "best_mismatch_fields": [
            field for field in _UNMATCHED_DIAGNOSTIC_FIELDS if best_status.get(field) == "mismatch"
        ],
        "best_unavailable_fields": [
            field for field in _UNMATCHED_DIAGNOSTIC_FIELDS if best_status.get(field) == "unavailable"
        ],
        "best_metrics": {} if best is None else best.get("metrics", {}),
        "best_field_status": best_status,
    }
    return material_record, diagnostics


def _diagnose_unmatched_material_worker(task):
    (
        mid,
        gen_cifs,
        true_cif,
        n_gens,
        ltol,
        stol,
        angle_tol,
        length_lo,
        length_hi,
        angle_lo,
        angle_hi,
        max_sites,
        rmsd_timeout_s,
    ) = task
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    return _diagnose_unmatched_material(
        mid,
        gen_cifs,
        true_cif,
        n_gens,
        matcher,
        length_lo,
        length_hi,
        angle_lo,
        angle_hi,
        max_sites,
        rmsd_timeout_s,
    )


def _default_unmatched_diagnostics_dir(gen_cifs_path):
    parent = os.path.dirname(os.path.abspath(gen_cifs_path))
    if os.path.basename(parent) == "tars":
        return os.path.join(os.path.dirname(parent), "unmatched_diagnostics")
    return os.path.join(parent, "unmatched_diagnostics")


def _group_name_from_tar_path(gen_cifs_path):
    name = os.path.basename(gen_cifs_path)
    if name.endswith(".tar.gz"):
        name = name[: -len(".tar.gz")]
    elif name.endswith(".tgz"):
        name = name[: -len(".tgz")]
    if name.startswith("test_"):
        name = name[len("test_") :]
    return name


def write_unmatched_diagnostics(
    *,
    id_to_gen_cifs,
    id_to_true_cifs,
    gen_cifs_path,
    n_gens,
    matcher,
    length_lo,
    length_hi,
    angle_lo,
    angle_hi,
    max_sites,
    rmsd_timeout_s,
    diagnostics_mode,
    diagnostics_dir,
    overwrite,
    matched_rms_by_id=None,
    diagnostics_workers=1,
):
    if diagnostics_mode == "off":
        return None

    out_dir = diagnostics_dir or _default_unmatched_diagnostics_dir(gen_cifs_path)
    os.makedirs(out_dir, exist_ok=True)
    group = _group_name_from_tar_path(gen_cifs_path)
    k_label = "all" if n_gens is None else str(int(n_gens))
    prefix = f"{group}_k{k_label}"

    summary_path = os.path.join(out_dir, f"{prefix}_unmatched_summary.json")
    records_path = os.path.join(out_dir, f"{prefix}_unmatched_records.jsonl")
    candidates_path = os.path.join(out_dir, f"{prefix}_unmatched_candidates.jsonl")
    field_counts_path = os.path.join(out_dir, f"{prefix}_field_counts.tsv")

    paths = [summary_path, field_counts_path]
    if diagnostics_mode == "on":
        paths.extend([records_path, candidates_path])
    if not overwrite:
        for path in paths:
            if os.path.exists(path):
                raise FileExistsError(f"diagnostics output exists: {path}")

    ids = sorted(id_to_gen_cifs.keys())
    material_records = []
    candidate_records = []
    unmatched_tasks = []

    for mid in ids:
        if mid not in id_to_true_cifs:
            raise Exception(f"could not find ID `{mid}` in true CIFs")
        gen_cifs = _candidate_list_for_k(id_to_gen_cifs[mid], n_gens)
        true_cif = id_to_true_cifs[mid]
        if matched_rms_by_id is not None and mid in matched_rms_by_id:
            if matched_rms_by_id.get(mid) is not None:
                continue
        else:
            if _is_material_matched_for_diagnostics(gen_cifs, true_cif, n_gens, matcher, max_sites, rmsd_timeout_s):
                continue
        unmatched_tasks.append(
            (
                mid,
                gen_cifs,
                true_cif,
                n_gens,
                0.3,
                0.5,
                10.0,
                length_lo,
                length_hi,
                angle_lo,
                angle_hi,
                max_sites,
                rmsd_timeout_s,
            )
        )

    if unmatched_tasks:
        if diagnostics_workers is not None and int(diagnostics_workers) > 1 and len(unmatched_tasks) > 1:
            with mp.Pool(processes=int(diagnostics_workers)) as pool:
                iterator = pool.imap(_diagnose_unmatched_material_worker, unmatched_tasks)
                for material_record, diagnostics in tqdm(
                    iterator, total=len(unmatched_tasks), desc="diagnosing unmatched CIFs..."
                ):
                    material_record["group"] = group
                    for diag in diagnostics:
                        diag["group"] = group
                        diag["K"] = None if n_gens is None else int(n_gens)
                    material_records.append(material_record)
                    candidate_records.extend(diagnostics)
        else:
            for task in tqdm(unmatched_tasks, desc="diagnosing unmatched CIFs..."):
                material_record, diagnostics = _diagnose_unmatched_material_worker(task)
                material_record["group"] = group
                for diag in diagnostics:
                    diag["group"] = group
                    diag["K"] = None if n_gens is None else int(n_gens)
                material_records.append(material_record)
                candidate_records.extend(diagnostics)

    material_den = len(material_records)
    candidate_den = len(candidate_records)
    field_counts = {}
    for field in _UNMATCHED_DIAGNOSTIC_FIELDS:
        material_mismatch = sum(1 for rec in material_records if rec["best_field_status"].get(field) == "mismatch")
        material_unavailable = sum(1 for rec in material_records if rec["best_field_status"].get(field) == "unavailable")
        candidate_mismatch = sum(1 for rec in candidate_records if rec["field_status"].get(field) == "mismatch")
        candidate_unavailable = sum(1 for rec in candidate_records if rec["field_status"].get(field) == "unavailable")
        field_counts[field] = {
            "material_mismatch_n": int(material_mismatch),
            "material_mismatch_rate": _rate(material_mismatch, material_den),
            "material_unavailable_n": int(material_unavailable),
            "material_unavailable_rate": _rate(material_unavailable, material_den),
            "candidate_mismatch_n": int(candidate_mismatch),
            "candidate_mismatch_rate": _rate(candidate_mismatch, candidate_den),
            "candidate_unavailable_n": int(candidate_unavailable),
            "candidate_unavailable_rate": _rate(candidate_unavailable, candidate_den),
        }

    summary = {
        "group": group,
        "K": None if n_gens is None else int(n_gens),
        "n_ids": int(len(ids)),
        "unmatched_n": int(material_den),
        "unmatched_rate": _rate(material_den, len(ids)),
        "n_unmatched_candidates": int(candidate_den),
        "fields": list(_UNMATCHED_DIAGNOSTIC_FIELDS),
        "field_counts": field_counts,
        "paths": {
            "summary": summary_path,
            "records": records_path if diagnostics_mode == "on" else None,
            "candidates": candidates_path if diagnostics_mode == "on" else None,
            "field_counts": field_counts_path,
        },
    }

    with open(summary_path, "wt", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")

    with open(field_counts_path, "wt", encoding="utf-8") as f:
        f.write(
            "field\tmaterial_mismatch_n\tmaterial_mismatch_rate\tmaterial_unavailable_n\tmaterial_unavailable_rate\t"
            "candidate_mismatch_n\tcandidate_mismatch_rate\tcandidate_unavailable_n\tcandidate_unavailable_rate\n"
        )
        for field in _UNMATCHED_DIAGNOSTIC_FIELDS:
            counts = field_counts[field]
            f.write(
                "\t".join(
                    [
                        field,
                        str(counts["material_mismatch_n"]),
                        str(counts["material_mismatch_rate"]),
                        str(counts["material_unavailable_n"]),
                        str(counts["material_unavailable_rate"]),
                        str(counts["candidate_mismatch_n"]),
                        str(counts["candidate_mismatch_rate"]),
                        str(counts["candidate_unavailable_n"]),
                        str(counts["candidate_unavailable_rate"]),
                    ]
                )
                + "\n"
            )

    if diagnostics_mode == "on":
        with open(records_path, "wt", encoding="utf-8") as f:
            for rec in material_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        with open(candidates_path, "wt", encoding="utf-8") as f:
            for rec in candidate_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    print(f"[unmatched-diagnostics] wrote {summary_path}")
    return summary


# ---- Robust CSP matcher in worker processes (used when --workers>1 or --hard-timeout-seconds>0) ----
_CSP_WORKER_MATCHER = None
_CSP_WORKER_LENGTH_LO = None
_CSP_WORKER_LENGTH_HI = None
_CSP_WORKER_ANGLE_LO = None
_CSP_WORKER_ANGLE_HI = None
_CSP_WORKER_MAX_SITES = None
_CSP_WORKER_RMSD_TIMEOUT_S = None


def _init_csp_worker(
    ltol: float,
    stol: float,
    angle_tol: float,
    length_lo: float,
    length_hi: float,
    angle_lo: float,
    angle_hi: float,
    max_sites: int | None,
    rmsd_timeout_s: float | None,
):
    global _CSP_WORKER_MATCHER, _CSP_WORKER_LENGTH_LO, _CSP_WORKER_LENGTH_HI, _CSP_WORKER_ANGLE_LO, _CSP_WORKER_ANGLE_HI  # noqa: PLW0603
    global _CSP_WORKER_MAX_SITES, _CSP_WORKER_RMSD_TIMEOUT_S  # noqa: PLW0603
    _CSP_WORKER_MATCHER = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    _CSP_WORKER_LENGTH_LO = float(length_lo)
    _CSP_WORKER_LENGTH_HI = float(length_hi)
    _CSP_WORKER_ANGLE_LO = float(angle_lo)
    _CSP_WORKER_ANGLE_HI = float(angle_hi)
    _CSP_WORKER_MAX_SITES = None if max_sites is None else int(max_sites)
    _CSP_WORKER_RMSD_TIMEOUT_S = None if rmsd_timeout_s is None else float(rmsd_timeout_s)


def _csp_worker_set_limits(*, max_sites: int | None, rmsd_timeout_s: float | None):
    global _CSP_WORKER_MAX_SITES, _CSP_WORKER_RMSD_TIMEOUT_S  # noqa: PLW0603
    _CSP_WORKER_MAX_SITES = None if max_sites is None else int(max_sites)
    _CSP_WORKER_RMSD_TIMEOUT_S = None if rmsd_timeout_s is None else float(rmsd_timeout_s)


def _csp_worker_match_one(args):
    """
    Worker task for CSP matching.
    args: (id, gen_cifs: list[str], true_cif: str, n_gens: int|None)
    """
    mid, gen_cifs, true_cif, n_gens = args

    # Local timeout wrapper around get_rms_dist (best-effort; hard timeouts are enforced by the parent process).
    class _RmsdTimeout(Exception):
        pass

    @contextlib.contextmanager
    def _time_limit(seconds: float | None):
        if seconds is None or seconds <= 0 or signal is None:
            yield
            return

        def _handler(signum, frame):  # noqa: ARG001
            raise _RmsdTimeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    matcher = _CSP_WORKER_MATCHER
    if matcher is None:
        # Should not happen unless initializer failed.
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    # Parse/normalize true structure
    try:
        true_cif_n = _normalize_cif_symmops_to_declared_sg(true_cif)
        gt_struct = Structure.from_str(true_cif_n, fmt="cif")
    except Exception:
        return {
            "id": mid,
            "best_rmsd": None,
            "n_attempted": int(0 if n_gens == 0 else (len(gen_cifs) if n_gens is None else min(len(gen_cifs), int(n_gens)))),
            "n_parsed": 0,
            "n_sensible": 0,
            "n_valid": 0,
            "n_skipped_large": 0,
            "n_rmsd_timeouts": 0,
            "n_rmsd_errors": 1,
        }

    max_sites = _CSP_WORKER_MAX_SITES
    timeout_s = _CSP_WORKER_RMSD_TIMEOUT_S

    # Decide how many candidates we attempt (keep flags aligned with attempted candidates).
    cifs = gen_cifs if n_gens is None else gen_cifs[: int(n_gens)]

    n_attempted = 0
    n_parsed = 0
    n_sensible = 0
    n_valid = 0
    n_skipped_large = 0
    n_rmsd_timeouts = 0
    n_rmsd_errors = 0

    best = None
    for cif in cifs:
        n_attempted += 1
        sensible = False
        try:
            sensible = bool(is_sensible(cif, _CSP_WORKER_LENGTH_LO, _CSP_WORKER_LENGTH_HI, _CSP_WORKER_ANGLE_LO, _CSP_WORKER_ANGLE_HI))
        except Exception:
            sensible = False
        if sensible:
            n_sensible += 1

        # Normalize symmops to declared SG for parsing
        try:
            cif_n = _normalize_cif_symmops_to_declared_sg(cif)
            pred = Structure.from_str(cif_n, fmt="cif")
        except Exception:
            continue

        n_parsed += 1
        try:
            if bool(is_valid(pred)):
                n_valid += 1
        except Exception:
            # smact may be unavailable; treat as not-valid, but still allow match computation.
            pass

        try:
            if max_sites is not None and (int(pred.num_sites) > int(max_sites) or int(gt_struct.num_sites) > int(max_sites)):
                n_skipped_large += 1
                continue
        except Exception:
            pass

        try:
            with _time_limit(timeout_s):
                rms_dist = matcher.get_rms_dist(pred, gt_struct)
            rms = None if rms_dist is None else rms_dist[0]
        except _RmsdTimeout:
            n_rmsd_timeouts += 1
            rms = None
        except Exception:
            n_rmsd_errors += 1
            rms = None

        if rms is None:
            continue
        if best is None or float(rms) < float(best):
            best = float(rms)

    return {
        "id": mid,
        "best_rmsd": best,
        "n_attempted": int(n_attempted),
        "n_parsed": int(n_parsed),
        "n_sensible": int(n_sensible),
        "n_valid": int(n_valid),
        "any_sensible": bool(n_sensible > 0),
        "any_valid": bool(n_valid > 0),
        "n_skipped_large": int(n_skipped_large),
        "n_rmsd_timeouts": int(n_rmsd_timeouts),
        "n_rmsd_errors": int(n_rmsd_errors),
    }


def get_match_rate_and_rms_robust_mp(id_to_gen_cifs, id_to_true_cifs, *, n_gens, length_lo, length_hi, angle_lo, angle_hi, ltol, stol, angle_tol, max_sites, rmsd_timeout_s, workers, hard_timeout_s):
    """
    Robust CSP evaluation that avoids per-sample hangs by:
    - doing parsing+matching inside a multiprocessing Pool
    - enforcing a hard per-sample timeout in the parent process; on timeout, the pool is terminated and recreated

    This is intended for large runs where a single pathological StructureMatcher call can stall the whole job.
    """
    if workers is None or int(workers) <= 1 and (hard_timeout_s is None or float(hard_timeout_s) <= 0):
        raise ValueError("robust mp mode requires workers>1 or hard_timeout_s>0")

    # Pre-validate IDs against GT, keep deterministic order.
    ids = sorted(id_to_gen_cifs.keys())
    for mid in ids:
        if mid not in id_to_true_cifs:
            raise Exception(f"could not find ID `{mid}` in true CIFs")

    tasks = [(mid, id_to_gen_cifs[mid], id_to_true_cifs[mid], n_gens) for mid in ids]
    q = deque(tasks)

    total_ids = len(tasks)
    pbar = tqdm(total=total_ids, desc="comparing structures...")

    rms_by_id = {}
    n_attempted = n_parsed = n_sensible = n_valid = 0
    any_sensible_cnt = 0
    any_valid_cnt = 0
    n_skipped_large = 0
    n_rmsd_timeouts = 0
    n_rmsd_errors = 0
    n_hard_timeouts = 0

    # Keep the in-flight window small for K100 validation runs. The upstream
    # script uses workers*8; when one ordered result hits the hard timeout, all
    # later in-flight tasks are requeued, which causes heavy tail rework.
    window_env = os.environ.get("OPENTRY10_BENCH_WINDOW")
    if window_env:
        window = max(1, int(window_env))
    else:
        window_mult = float(os.environ.get("OPENTRY10_BENCH_WINDOW_MULT", "1"))
        window = max(1, int(int(workers) * window_mult))
    print(f"[opentry10] robust_mp workers={int(workers)} window={window}", flush=True)

    hard_timeout = None if hard_timeout_s is None or float(hard_timeout_s) <= 0 else float(hard_timeout_s)

    def _record_result(res):
        nonlocal n_attempted, n_parsed, n_sensible, n_valid
        nonlocal any_sensible_cnt, any_valid_cnt
        nonlocal n_skipped_large, n_rmsd_timeouts, n_rmsd_errors
        mid = res.get("id")
        rms_by_id[mid] = res.get("best_rmsd")
        n_attempted += int(res.get("n_attempted", 0))
        n_parsed += int(res.get("n_parsed", 0))
        n_sensible += int(res.get("n_sensible", 0))
        n_valid += int(res.get("n_valid", 0))
        any_sensible_cnt += int(bool(res.get("any_sensible", False)))
        any_valid_cnt += int(bool(res.get("any_valid", False)))
        n_skipped_large += int(res.get("n_skipped_large", 0))
        n_rmsd_timeouts += int(res.get("n_rmsd_timeouts", 0))
        n_rmsd_errors += int(res.get("n_rmsd_errors", 0))
        pbar.update(1)

    def _start_pool():
        pool = mp.Pool(
            processes=int(workers),
            initializer=_init_csp_worker,
            initargs=(
                float(ltol),
                float(stol),
                float(angle_tol),
                float(length_lo),
                float(length_hi),
                float(angle_lo),
                float(angle_hi),
                None if max_sites is None else int(max_sites),
                None if rmsd_timeout_s is None else float(rmsd_timeout_s),
            ),
        )
        return pool

    while q:
        pool = _start_pool()
        running = []

        def _submit_one(t):
            return {
                "task": t,
                "async": pool.apply_async(_csp_worker_match_one, (t,)),
                "started_at": time.monotonic(),
            }

        def _fill_window():
            while q and len(running) < window:
                running.append(_submit_one(q.popleft()))

        _fill_window()
        terminated = False
        while running:
            progressed = False
            still_running = []
            for item in running:
                t = item["task"]
                ar = item["async"]
                if ar.ready():
                    try:
                        _record_result(ar.get(timeout=0))
                    except Exception:
                        mid = t[0]
                        rms_by_id[mid] = None
                        n_rmsd_errors += 1
                        pbar.update(1)
                    progressed = True
                else:
                    still_running.append(item)
            running = still_running
            if progressed:
                _fill_window()
                continue

            timed_out_items = []
            if hard_timeout is not None:
                now = time.monotonic()
                for item in running:
                    if now - float(item["started_at"]) >= hard_timeout:
                        timed_out_items.append(item)
            if timed_out_items:
                # Harvesting above preserves completed work before killing the
                # pool. Every active task beyond the hard timeout is marked
                # unmatched; younger active tasks are retried after restart.
                timed_out_ids = {id(item) for item in timed_out_items}
                for item in timed_out_items:
                    mid = item["task"][0]
                    rms_by_id[mid] = None
                    n_hard_timeouts += 1
                    pbar.update(1)
                try:
                    pool.terminate()
                except Exception:
                    pass
                try:
                    pool.join()
                except Exception:
                    pass
                retry_tasks = [item["task"] for item in running if id(item) not in timed_out_ids]
                for t2 in reversed(retry_tasks):
                    q.appendleft(t2)
                terminated = True
                break

            time.sleep(0.2)

        if not terminated:
            try:
                pool.close()
            except Exception:
                pass
            try:
                pool.join()
            except Exception:
                pass

    pbar.close()
    matched = [mid for mid in ids if rms_by_id.get(mid) is not None]
    match_rate = float(len(matched) / max(1, len(ids)))
    mean_rms = None if len(matched) == 0 else float(np.mean([float(rms_by_id[mid]) for mid in matched]))
    globals()["_BENCH_LAST_RMS_BY_ID"] = dict(rms_by_id)
    return {
        "match_rate": match_rate,
        "rms_dist": mean_rms,
        "match_timeouts": int(n_rmsd_timeouts),
        "match_skipped_large": int(n_skipped_large),
        "match_errors": int(n_rmsd_errors),
        "match_hard_timeouts": int(n_hard_timeouts),
        "bench_max_sites": None if max_sites is None else int(max_sites),
        "bench_rmsd_timeout_s": None if rmsd_timeout_s is None else float(rmsd_timeout_s),
        "bench_workers": int(workers),
        "bench_hard_timeout_s": None if hard_timeout_s is None or float(hard_timeout_s) <= 0 else float(hard_timeout_s),
        "n_ids": int(len(ids)),
        "n_attempted_candidates": int(n_attempted),
        "parse_rate_candidate": None if n_attempted == 0 else float(n_parsed / n_attempted),
        "sensible_rate_candidate": None if n_attempted == 0 else float(n_sensible / n_attempted),
        "valid_rate_candidate": None if n_attempted == 0 else float(n_valid / n_attempted),
        "sensible_rate_any": None if len(ids) == 0 else float(any_sensible_cnt / len(ids)),
        "valid_rate_any": None if len(ids) == 0 else float(any_valid_cnt / len(ids)),
    }


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py
def get_unconditional_metrics(gen_structs, gen_comps, true_structs, n_gen, comp_scaler, cov_cutoffs, n_samples=1000):
    if _UNCONDITIONAL_DEPS_ERROR is not None:
        raise SystemExit(
            "Unconditional metrics require optional deps (smact/scipy/matminer) that failed to import.\n"
            f"Import error: {_UNCONDITIONAL_DEPS_ERROR}\n"
            "CSP match-rate metrics should still work without --unconditional."
        )
    valid_structs = []
    for struct, struct_fp, _ in tqdm(gen_structs, desc="getting valid structures..."):
        if is_valid_unconditional(struct, struct_fp):
            valid_structs.append(struct)
    if len(valid_structs) >= n_samples:
        sampled_indices = np.random.choice(len(valid_structs), n_samples, replace=False)
        valid_samples = [valid_structs[i] for i in sampled_indices]
    else:
        raise Exception(
            f"Insufficient valid crystals in the generated set: {len(valid_structs)}/{n_samples}")

    n_comp_valid = 0
    for comp in tqdm(gen_comps, desc="counting comp valid..."):
        # even if a structure is unreasonable or invalid,
        #  the generated composition might still be valid
        try:
            if smact_validity(
                atom_types=[str(elem) for elem, n in comp.items() for _ in range(int(n))]
            ):
                n_comp_valid += 1
        except Exception:
            pass
    comp_valid = n_comp_valid / n_gen
    n_struct_valid = 0
    for struct, _, _ in tqdm(gen_structs, desc="counting struct valid..."):
        if structure_validity(struct):
            n_struct_valid += 1
    struct_valid = n_struct_valid / n_gen
    valid = len(valid_structs) / n_gen
    valid_dict = {"comp_valid": comp_valid, "struct_valid": struct_valid, "valid": valid}

    print("computing wdist_density...")
    pred_densities = [struct.density for struct in valid_samples]
    gt_densities = [struct.density for struct, _, _ in true_structs]
    wdist_density = wasserstein_distance(pred_densities, gt_densities)
    wdist_density_dict = {"wdist_density": wdist_density}

    print("computing wdist_num_elems...")
    pred_nelems = [len(set(struct.species)) for struct in valid_samples]
    gt_nelems = [len(set(struct.species)) for struct, _, _ in true_structs]
    wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
    wdist_num_elems_dict = {"wdist_num_elems": wdist_num_elems}

    # TODO use property models to compute formation energy Wasserstein distances

    print("computing cov...")
    cutoff_dict = COV_Cutoffs[cov_cutoffs]
    cov_metrics_dict, _ = compute_cov(
        gen_structs,
        true_structs,
        struc_cutoff=cutoff_dict["struc"],
        comp_cutoff=cutoff_dict["comp"],
        comp_scaler=comp_scaler,
    )

    metrics = {}
    metrics.update(valid_dict)
    metrics.update({"n_sensible": len(gen_structs)})
    metrics.update(wdist_density_dict)
    # metrics.update(wdist_prop_dict)  # TODO
    metrics.update(wdist_num_elems_dict)
    metrics.update(cov_metrics_dict)

    return metrics


def get_comp_scaler_means_stds():
    with open(os.path.join(THIS_DIR, "../resources/comp_scaler_means.txt"), "rt") as f:
        comp_scaler_means = [float(num.strip()) for num in f.readlines()]
    with open(os.path.join(THIS_DIR, "../resources/comp_scaler_stds.txt"), "rt") as f:
        comp_scaler_stds = [float(num.strip()) for num in f.readlines()]
    return comp_scaler_means, comp_scaler_stds


def extract_cif_id(filepath):
    """
    Parses a filename assumed to be in the format "id__n.cif",
    returning the "id".

    :param filepath: a filename assumed to be in the format "id__n.cif"
    :return: the extracted values of `id`
    """
    filename = os.path.basename(filepath)
    # split from the right, once
    parts = filename.rsplit("__", 1)
    if len(parts) == 2:
        id_part, _ = parts
        return id_part
    else:
        raise ValueError(f"'{filename}' does not conform to expected format 'id__n.cif'")


def read_generated_cifs(input_path):
    generated_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                cif_id = extract_cif_id(member.name)
                if cif_id not in generated_cifs:
                    generated_cifs[cif_id] = []
                generated_cifs[cif_id].append(cif)
    return generated_cifs


def read_true_cifs(input_path):
    true_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting true CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                true_cifs[cif_id] = cif
    return true_cifs


def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi):
    gen_structs = []
    gen_sensible_flags = []
    gen_valid_flags = []
    true_structs = []
    for id, cifs in tqdm(id_to_gen_cifs.items(), desc="converting CIFs to Structures..."):
        if id not in id_to_true_cifs:
            raise Exception(f"could not find ID `{id}` in true CIFs")

        structs = []
        sensible_flags = []
        valid_flags = []
        for cif in cifs[:n_gens]:
            sensible = False
            try:
                try:
                    sensible = bool(is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi))
                except Exception:
                    sensible = False
                # Align with evaluate_cifs.py: normalize symmetry operators to the declared SG
                # so CIFs that contain only an identity symmop loop still expand correctly.
                try:
                    sg = extract_space_group_symbol(cif)
                except Exception:
                    sg = None
                if sg is not None and sg != "P 1":
                    try:
                        cif = replace_symmetry_operators(cif, sg, safe=True)
                    except Exception:
                        pass
                struct = Structure.from_str(cif, fmt="cif")
            except Exception:
                # Keep flags aligned with attempted candidates: a parse failure implies not valid.
                sensible_flags.append(bool(sensible))
                valid_flags.append(False)
                continue

            structs.append(struct)
            sensible_flags.append(bool(sensible))
            try:
                valid_flags.append(bool(is_valid(struct)))
            except Exception:
                # Allow match computation even if smact deps are unavailable.
                valid_flags.append(False)
        gen_structs.append(structs)
        gen_sensible_flags.append(sensible_flags)
        gen_valid_flags.append(valid_flags)

        # Best-effort normalize true CIF symmetry operators as well (should be a no-op for well-formed CIFs).
        true_cif = id_to_true_cifs[id]
        try:
            sg_true = extract_space_group_symbol(true_cif)
        except Exception:
            sg_true = None
        if sg_true is not None and sg_true != "P 1":
            try:
                true_cif = replace_symmetry_operators(true_cif, sg_true, safe=True)
            except Exception:
                pass
        true_structs.append(Structure.from_str(true_cif, fmt="cif"))
    return gen_structs, true_structs, gen_sensible_flags, gen_valid_flags


def get_gen_comps(id_to_gen_cifs):
    gen_comps = []
    for cifs in tqdm(id_to_gen_cifs.values(), desc="extracting generated compositions from CIFs..."):
        cif = cifs[0]
        try:
            data_formula = extract_data_formula(cif)
            comp = Composition(data_formula)
            if len(comp) == 0:
                continue
            gen_comps.append(comp)
        except Exception:
            pass
    return gen_comps


def get_gen_structs_unconditional(id_to_gen_cifs, length_lo, length_hi, angle_lo, angle_hi):
    gen_structs = []
    for cifs in tqdm(id_to_gen_cifs.values(), desc="converting CIFs to Structures and fingerprints..."):
        cif = cifs[0]
        try:
            if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                continue
            struct = Structure.from_str(cif, fmt="cif")
            # get the structure fingerprint only for a valid structure
            struct_fp = get_struct_fingerprint(struct) if structure_validity(struct) else None
            comp_fp = get_comp_fingerprint(struct)
            gen_structs.append((struct, struct_fp, comp_fp))
        except Exception:
            pass
    return gen_structs


def get_true_structs_unconditional(id_to_true_cifs):
    true_structs = []
    for cif in tqdm(id_to_true_cifs.values(), desc="converting true CIFs to Structures and fingerprints..."):
        struct = Structure.from_str(cif, fmt="cif")
        struct_fp = get_struct_fingerprint(struct)
        comp_fp = get_comp_fingerprint(struct)
        true_structs.append((struct, struct_fp, comp_fp))
    return true_structs


"""
This script performs the CDVAE and DiffCSP benchmark analysis, as described in:
https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform benchmark analysis.")
    parser.add_argument("gen_cifs",
                        help="Path to the .tar.gz file containing the generated CIF files.")
    parser.add_argument("true_cifs",
                        help="Path to the .tar.gz file containing the true CIF files.")
    parser.add_argument("--num-gens", required=False, default=0, type=int,
                        help="The maximum number of generations to use per structure. Default is 0, which means "
                             "use all of the available generations. (This argument is ignored for the unconditional "
                             "generation task metrics.)")
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    parser.add_argument("--unconditional", action="store_true",
                        help="If included, the unconditional generation task metrics will be computed "
                             "instead of the CSP task metrics")
    parser.add_argument("--cov-cutoffs", choices=["mp20", "carbon", "perovskite"],
                        required=False, default="perovskite",
                        help="The coverage cutoffs to use if the unconditional generation task metrics are "
                             "being computed. Default is 'perovskite'.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="The random seed to use for the unconditional generation task metrics.")
    parser.add_argument(
        "--max-sites",
        type=int,
        default=0,
        help=(
            "If >0, skip matching a candidate when pred.num_sites or gt.num_sites exceeds this threshold. "
            "This is a safety valve to prevent pathological StructureMatcher runtimes on malformed structures."
        ),
    )
    parser.add_argument(
        "--rmsd-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "If >0, apply a per-comparison wall-time timeout around StructureMatcher.get_rms_dist(). "
            "Timed-out comparisons are treated as unmatched."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "If >1, run CSP parsing+matching in a multiprocessing pool for speed and robustness. "
            "Recommended for large benchmarks (e.g., mp_20 test=9046)."
        ),
    )
    parser.add_argument(
        "--hard-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "If >0, enforce a hard per-ID wall-time timeout in the parent process when running with --workers>1. "
            "On timeout, the pool is terminated and recreated; the timed-out ID is treated as unmatched."
        ),
    )
    parser.add_argument(
        "--unmatched-diagnostics",
        choices=["on", "off", "summary"],
        default="on",
        help=(
            "For CSP benchmarks, write field-level diagnostics for materials that do not match within top-K. "
            "'summary' writes only summary/count files; 'off' disables diagnostics. Ignored for --unconditional."
        ),
    )
    parser.add_argument(
        "--unmatched-diagnostics-dir",
        default="",
        help=(
            "Directory for unmatched diagnostic files. Default: run/unmatched_diagnostics when gen tar is under run/tars, "
            "otherwise a sibling unmatched_diagnostics directory."
        ),
    )
    parser.add_argument(
        "--unmatched-diagnostics-overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing unmatched diagnostic files. This is the default behavior.",
    )
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    true_cifs_path = args.true_cifs
    n_gens = args.num_gens
    length_lo = args.length_lo
    length_hi = args.length_hi
    angle_lo = args.angle_lo
    angle_hi = args.angle_hi
    unconditional = args.unconditional
    cov_cutoffs = args.cov_cutoffs
    seed = args.seed
    # Expose as module globals so get_match_rate_and_rms can read without changing all call sites.
    globals()["_BENCH_MAX_SITES"] = int(args.max_sites) if int(args.max_sites) > 0 else None
    globals()["_BENCH_RMSD_TIMEOUT_S"] = float(args.rmsd_timeout_seconds) if float(args.rmsd_timeout_seconds) > 0 else None

    if n_gens == 0:
        n_gens = None
        print("using all available generations...")
    else:
        if n_gens < 0:
            raise Exception(f"invalid value for n_gens: {n_gens}")
        print(f"using a maximum of {n_gens} generation(s) per compound...")

    # defaults taken from DiffCSP
    ltol = 0.3
    stol = 0.5
    angle_tol = 10
    struct_matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    globals()["_BENCH_LAST_RMS_BY_ID"] = None

    id_to_gen_cifs = read_generated_cifs(gen_cifs_path)
    id_to_true_cifs = read_true_cifs(true_cifs_path)

    if unconditional:
        np.random.seed(seed)
        comp_scaler_means, comp_scaler_stds = get_comp_scaler_means_stds()
        comp_scaler = StandardScaler(
            means=np.array(comp_scaler_means),
            stds=np.array(comp_scaler_stds),
        )
        gen_structs = get_gen_structs_unconditional(
            id_to_gen_cifs, length_lo, length_hi, angle_lo, angle_hi
        )
        gen_comps = get_gen_comps(id_to_gen_cifs)
        true_structs = get_true_structs_unconditional(id_to_true_cifs)
        n_gens = len(id_to_gen_cifs)
        metrics = get_unconditional_metrics(gen_structs, gen_comps, true_structs, n_gens, comp_scaler, cov_cutoffs)
    else:
        max_sites = globals().get("_BENCH_MAX_SITES", None)
        rmsd_timeout_s = globals().get("_BENCH_RMSD_TIMEOUT_S", None)
        workers = int(args.workers)
        hard_timeout_s = float(args.hard_timeout_seconds) if float(args.hard_timeout_seconds) > 0 else None

        if workers > 1 or (hard_timeout_s is not None and hard_timeout_s > 0):
            metrics = get_match_rate_and_rms_robust_mp(
                id_to_gen_cifs,
                id_to_true_cifs,
                n_gens=n_gens,
                length_lo=length_lo,
                length_hi=length_hi,
                angle_lo=angle_lo,
                angle_hi=angle_hi,
                ltol=ltol,
                stol=stol,
                angle_tol=angle_tol,
                max_sites=max_sites,
                rmsd_timeout_s=rmsd_timeout_s,
                workers=workers,
                hard_timeout_s=hard_timeout_s,
            )
        else:
            gen_structs, true_structs, gen_sensible_flags, gen_valid_flags = get_structs(
                id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi
            )
            metrics = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher)
            # Report sensibility/validity independently (no gating relationship to match).
            total_attempted = sum(len(x) for x in gen_sensible_flags)
            total_parsed = sum(len(x) for x in gen_structs)
            total_sensible = sum(sum(bool(v) for v in x) for x in gen_sensible_flags)
            total_valid = sum(sum(bool(v) for v in x) for x in gen_valid_flags)
            n_ids = len(gen_sensible_flags)
            any_sensible = sum(1 for x in gen_sensible_flags if any(bool(v) for v in x))
            any_valid = sum(1 for x in gen_valid_flags if any(bool(v) for v in x))
            metrics.update(
                {
                    "n_ids": int(n_ids),
                    "n_attempted_candidates": int(total_attempted),
                    "parse_rate_candidate": None if total_attempted == 0 else float(total_parsed / total_attempted),
                    "sensible_rate_candidate": None if total_attempted == 0 else float(total_sensible / total_attempted),
                    "valid_rate_candidate": None if total_attempted == 0 else float(total_valid / total_attempted),
                    "sensible_rate_any": None if n_ids == 0 else float(any_sensible / n_ids),
                    "valid_rate_any": None if n_ids == 0 else float(any_valid / n_ids),
                }
            )
        write_unmatched_diagnostics(
            id_to_gen_cifs=id_to_gen_cifs,
            id_to_true_cifs=id_to_true_cifs,
            gen_cifs_path=gen_cifs_path,
            n_gens=n_gens,
            matcher=struct_matcher,
            length_lo=length_lo,
            length_hi=length_hi,
            angle_lo=angle_lo,
            angle_hi=angle_hi,
            max_sites=max_sites,
            rmsd_timeout_s=rmsd_timeout_s,
            diagnostics_mode=str(args.unmatched_diagnostics),
            diagnostics_dir=str(args.unmatched_diagnostics_dir or ""),
            overwrite=bool(args.unmatched_diagnostics_overwrite),
            matched_rms_by_id=globals().get("_BENCH_LAST_RMS_BY_ID", None),
            diagnostics_workers=workers,
        )

    print(metrics)
