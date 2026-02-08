from __future__ import annotations

import math
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from crystallm import (
    bond_length_reasonableness_score,
    extract_numeric_property,
    extract_space_group_symbol,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    is_space_group_consistent,
    remove_atom_props_block,
    replace_symmetry_operators,
)

from crystallm._metrics import space_group_consistency_details


def _drop_spurious_loop_markers(cif_str: str) -> str:
    """
    Some generated CIFs incorrectly insert a bare `loop_` before single-value tags
    like `_cell_length_*` or `_cell_angle_*`, which can break CIF parsing.

    We only drop `loop_` when it's immediately followed by scalar tags (most commonly
    broken generations like `loop_` then `_cell_length_c ...`). We must NOT drop
    legitimate loop headers for `_symmetry_equiv_pos_*` or `_atom_site_*`.
    """
    lines = cif_str.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "loop_":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].lstrip()
                # Keep legitimate loops
                if nxt.startswith(("_symmetry_equiv_pos_", "_atom_site_", "_space_group_symop_")):
                    out.append(ln)
                    i += 1
                    continue
                # Drop spurious loops before scalar tags
                if nxt.startswith(
                    (
                        "_cell_length_",
                        "_cell_angle_",
                        "_cell_volume",
                        "_cell_formula_units_Z",
                        "_chemical_",
                        "_symmetry_Int_Tables_number",
                        "_symmetry_space_group_name_H-M",
                    )
                ):
                    i += 1
                    continue
        out.append(ln)
        i += 1
    return "\n".join(out)


def _extract_space_group_symbol_best_effort(cif_str: str) -> Optional[str]:
    try:
        return extract_space_group_symbol(cif_str)
    except Exception:  # noqa: BLE001
        return None


def _infer_and_fill_missing_cell_lengths(cif_str: str) -> str:
    """
    Best-effort fix for incomplete lattice info.

    We see generated CIFs missing `_cell_length_a`/`_cell_length_b` while providing:
    - `_cell_length_c`
    - all angles (often 90)
    - `_cell_volume`
    For tetragonal groups (e.g. P4/mmm), we can infer a=b=sqrt(V/c) when angles are 90.
    """

    def _num(tag: str) -> Optional[float]:
        try:
            return float(extract_numeric_property(cif_str, tag))
        except Exception:  # noqa: BLE001
            return None

    a = _num("_cell_length_a")
    b = _num("_cell_length_b")
    if a is not None and b is not None:
        return cif_str

    c = _num("_cell_length_c")
    alpha = _num("_cell_angle_alpha")
    beta = _num("_cell_angle_beta")
    gamma = _num("_cell_angle_gamma")
    vol = _num("_cell_volume")

    sg = (_extract_space_group_symbol_best_effort(cif_str) or "").replace(" ", "")
    angles_orthogonal = all(v is not None and abs(float(v) - 90.0) < 1e-3 for v in [alpha, beta, gamma])
    tetragonal_like = bool(re.match(r"^[PI]4", sg))

    a_new = a
    b_new = b

    if tetragonal_like:
        if a_new is None and b_new is not None:
            a_new = b_new
        if b_new is None and a_new is not None:
            b_new = a_new
        if (a_new is None or b_new is None) and vol is not None and c is not None and angles_orthogonal:
            try:
                ab = math.sqrt(float(vol) / float(c))
                a_new = ab
                b_new = ab
            except Exception:  # noqa: BLE001
                pass
    else:
        if angles_orthogonal and vol is not None and c is not None:
            if a_new is None and b_new is not None:
                try:
                    a_new = float(vol) / (float(b_new) * float(c))
                except Exception:  # noqa: BLE001
                    pass
            if b_new is None and a_new is not None:
                try:
                    b_new = float(vol) / (float(a_new) * float(c))
                except Exception:  # noqa: BLE001
                    pass

    if a_new is None and b_new is None:
        return cif_str

    lines = cif_str.splitlines()
    has_a = any(ln.strip().startswith("_cell_length_a") for ln in lines)
    has_b = any(ln.strip().startswith("_cell_length_b") for ln in lines)

    inserts: List[str] = []
    if (not has_a) and a_new is not None:
        inserts.append(f"_cell_length_a {float(a_new):.4f}")
    if (not has_b) and b_new is not None:
        inserts.append(f"_cell_length_b {float(b_new):.4f}")
    if not inserts:
        return cif_str

    insert_at = 0
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("_symmetry_space_group_name_H-M"):
            insert_at = idx + 1
            break
        if ln.strip().startswith("data_"):
            insert_at = idx + 1

    new_lines = lines[:insert_at] + inserts + lines[insert_at:]
    return "\n".join(new_lines)


@dataclass
class CIFValidationResult:
    """Structured result describing whether a CIF string passes basic sanity checks."""

    valid: bool
    reasons: List[str]
    bond_length_score: Optional[float]
    bond_lengths_reasonable: Optional[bool]
    formula_ok: Optional[bool]
    formula_ok_relaxed: Optional[bool]
    space_group_ok: Optional[bool]
    atom_site_multiplicity_ok: Optional[bool]
    strict_valid: Optional[bool]
    validator_step_errors: Dict[str, str]
    validator_step_tracebacks: Dict[str, str]


def clean_cif(cif_str: str) -> str:
    """
    Apply lightweight cleaning steps to a CIF text blob.

    The cleaning logic mirrors MCTSEvaluator._postprocess: ensure the unit cell
    dimensions are sane, normalize symmetry operators according to the declared
    space group, and strip atom property blocks that tend to confuse downstream
    tooling.
    """
    # Best-effort repairs before deeper parsing/validation:
    cif_str = _drop_spurious_loop_markers(cif_str)
    cif_str = _infer_and_fill_missing_cell_lengths(cif_str)

    # Some generated CIFs omit parts of the unit-cell specification. The
    # original evaluator used a volume sanity check, but for robustness we
    # make this best-effort: missing values should not crash the entire clean
    # step (validation will surface the issue later).
    try:
        a = extract_numeric_property(cif_str, "_cell_length_a")
        b = extract_numeric_property(cif_str, "_cell_length_b")
        c = extract_numeric_property(cif_str, "_cell_length_c")
        alpha = extract_numeric_property(cif_str, "_cell_angle_alpha")
        beta = extract_numeric_property(cif_str, "_cell_angle_beta")
        gamma = extract_numeric_property(cif_str, "_cell_angle_gamma")
        get_unit_cell_volume(a, b, c, alpha, beta, gamma)
    except Exception:  # noqa: BLE001
        pass

    # Generated text may contain tokenizer UNK tokens; these are never part of a
    # valid CIF and frequently break pymatgen parsing.
    cif_str = cif_str.replace("<unk>", "")

    cif_str = remove_atom_props_block(cif_str)

    # Symmetry op replacement can help when the CIF declares a high-symmetry
    # space group but only provides the identity op. However, it can also make
    # downstream parsing fail (e.g. occupancy>1, or "no structures") for some
    # generated CIFs. Apply it only when it remains parseable.
    try:
        space_group_symbol = extract_space_group_symbol(cif_str)
    except Exception:  # noqa: BLE001
        space_group_symbol = None

    if space_group_symbol and space_group_symbol != "P 1":
        try:
            replaced = replace_symmetry_operators(cif_str, space_group_symbol)
        except Exception:  # noqa: BLE001
            replaced = cif_str

        if replaced != cif_str:
            try:
                # Validate that the replaced CIF still yields at least one
                # Structure. If not, keep the original operators.
                from pymatgen.io.cif import CifParser

                structures = CifParser.from_string(replaced).get_structures(primitive=False)
                if structures:
                    cif_str = replaced
            except Exception:  # noqa: BLE001
                pass

    return cif_str


def validate_cif(
    cif_str: str,
    bond_length_acceptability_cutoff: float = 1.0,
    check_composition: bool = True,
) -> CIFValidationResult:
    """
    Run the same structural sanity checks used by the legacy MCTS evaluator.

    Returns a CIFValidationResult containing both a boolean verdict and
    per-check flags, so callers can aggregate statistics or bubble the reasons
    up to the LLM evaluator summary.
    """
    reasons: List[str] = []
    step_errors: Dict[str, str] = {}
    step_tracebacks: Dict[str, str] = {}

    # Composition checks:
    # - Even when check_composition=False (you don't want to gate validity on composition),
    #   we still try to compute composition-related metrics for reporting.
    # - When check_composition=True, failures contribute to `valid` and `reasons`.
    formula_ok: Optional[bool]
    atom_site_multiplicity_ok: Optional[bool]

    try:
        formula_ok = bool(is_formula_consistent(cif_str))
    except Exception as exc:  # noqa: BLE001
        formula_ok = False
        step_errors["formula_check_error"] = f"{type(exc).__name__}: {exc}"
        step_tracebacks["formula_check_error"] = traceback.format_exc()

    try:
        atom_site_multiplicity_ok = bool(is_atom_site_multiplicity_consistent(cif_str))
    except Exception as exc:  # noqa: BLE001
        atom_site_multiplicity_ok = False
        step_errors["atom_site_multiplicity_check_error"] = f"{type(exc).__name__}: {exc}"
        step_tracebacks["atom_site_multiplicity_check_error"] = traceback.format_exc()

    # Relaxed formula check: only compare _chemical_formula_sum vs atom-site list/multiplicity.
    # This aligns with the legacy "relaxed" reporting used for debugging prompt drift.
    formula_ok_relaxed = atom_site_multiplicity_ok

    if check_composition:
        if formula_ok is False:
            reasons.append("composition inconsistent")
        if atom_site_multiplicity_ok is False:
            reasons.append("atom site multiplicity inconsistent")

    try:
        bond_length_score = bond_length_reasonableness_score(cif_str)
    except Exception as exc:  # noqa: BLE001
        bond_length_score = None
        step_errors["bond_length_score_error"] = f"{type(exc).__name__}: {exc}"
        step_tracebacks["bond_length_score_error"] = traceback.format_exc()
    bond_lengths_reasonable = (
        (bond_length_score is not None) and (float(bond_length_score) >= float(bond_length_acceptability_cutoff))
    )
    if not bond_lengths_reasonable:
        if bond_length_score is None:
            reasons.append("bond length score unavailable")
        else:
            pct_unreasonable = (1 - bond_length_score) * 100
            reasons.append(f"unreasonable bond lengths (~{pct_unreasonable:.0f}% flagged)")

    sg_details = space_group_consistency_details(cif_str)
    space_group_ok = sg_details.get("consistent")
    if space_group_ok is True:
        pass
    elif space_group_ok is False:
        reasons.append("space group inconsistent")
    else:
        reasons.append("space_group_check_error")
        err = sg_details.get("error")
        if err:
            step_errors["space_group_check_error"] = str(err)
        tb = sg_details.get("traceback")
        if tb:
            step_tracebacks["space_group_check_error"] = str(tb)

    valid = bool(bond_lengths_reasonable) and (space_group_ok is True)
    if check_composition:
        valid = bool(formula_ok) and bool(atom_site_multiplicity_ok) and valid

    # Legacy strict validity (for reporting): old `is_valid` verdict, including composition checks.
    # This is computed best-effort and does NOT affect `valid`.
    try:
        from crystallm._metrics import is_valid as _strict_is_valid  # noqa: WPS433

        strict_valid = bool(
            _strict_is_valid(cif_str, bond_length_acceptability_cutoff=bond_length_acceptability_cutoff, check_composition=True)
        )
    except Exception as exc:  # noqa: BLE001
        strict_valid = None
        step_errors["strict_valid_error"] = f"{type(exc).__name__}: {exc}"
        step_tracebacks["strict_valid_error"] = traceback.format_exc()

    return CIFValidationResult(
        valid=valid,
        reasons=reasons,
        bond_length_score=bond_length_score,
        bond_lengths_reasonable=bond_lengths_reasonable if bond_length_score is not None else None,
        formula_ok=formula_ok,
        formula_ok_relaxed=formula_ok_relaxed,
        space_group_ok=space_group_ok,
        atom_site_multiplicity_ok=atom_site_multiplicity_ok,
        strict_valid=strict_valid,
        validator_step_errors=step_errors,
        validator_step_tracebacks=step_tracebacks,
    )


def clean_and_validate_cif(
    cif_str: str,
    bond_length_acceptability_cutoff: float = 1.0,
    check_composition: bool = True,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Convenience helper that runs clean_cif + validate_cif and returns both the
    sanitized CIF (if valid) and a serializable metadata dict capturing the
    validation outcome.
    """
    metadata: Dict[str, Any] = {}

    try:
        cleaned = clean_cif(cif_str)
        metadata["cleaning_error"] = None
    except Exception as exc:  # noqa: BLE001
        formula_ok: Optional[bool]
        atom_site_multiplicity_ok: Optional[bool]
        if check_composition:
            formula_ok = False
            atom_site_multiplicity_ok = False
        else:
            formula_ok = None
            atom_site_multiplicity_ok = None
        metadata.update(
            {
                "valid": False,
                "reasons": [f"cleaning_failed: {exc}"],
                "bond_length_score": None,
                "bond_lengths_reasonable": None,
                "formula_ok": formula_ok,
                "formula_ok_relaxed": None,
                "space_group_ok": None,
                "atom_site_multiplicity_ok": atom_site_multiplicity_ok,
                "strict_valid": None,
                "validator_step_errors": {"cleaning_failed": f"{type(exc).__name__}: {exc}"},
                "validator_step_tracebacks": {"cleaning_failed": traceback.format_exc()},
            }
        )
        return None, metadata

    validation = validate_cif(cleaned, bond_length_acceptability_cutoff, check_composition=check_composition)
    metadata.update(
        {
            "valid": validation.valid,
            "reasons": validation.reasons,
            "bond_length_score": validation.bond_length_score,
            "bond_lengths_reasonable": validation.bond_lengths_reasonable,
            "formula_ok": validation.formula_ok,
            "formula_ok_relaxed": validation.formula_ok_relaxed,
            "space_group_ok": validation.space_group_ok,
            "atom_site_multiplicity_ok": validation.atom_site_multiplicity_ok,
            "strict_valid": validation.strict_valid,
            "validator_step_errors": validation.validator_step_errors,
            "validator_step_tracebacks": validation.validator_step_tracebacks,
        }
    )

    # Return the cleaned CIF even when it fails validation so downstream steps
    # (e.g. ALIGNN scoring / analysis) can still run on a parseable structure.
    return cleaned, metadata
