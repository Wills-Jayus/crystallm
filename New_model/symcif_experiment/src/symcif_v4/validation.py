from __future__ import annotations

import warnings
from io import StringIO
from typing import Any

from pymatgen.io.cif import CifParser  # type: ignore
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore

from .formula import normalize_formula_counts


warnings.filterwarnings("ignore")


def structure_counts(structure: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for element, amount in structure.composition.as_dict().items():
        out[str(element)] = int(round(float(amount)))
    return dict(sorted(out.items()))


def validate_cif(cif_text: str, target_counts: dict[str, int], target_sg: int) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "readable": False,
        "formula_ok": False,
        "sg_ok": False,
        "atom_count_ok": False,
        "composition_exact": False,
        "atom_count_after_expansion": None,
        "detected_sg": None,
        "error": None,
    }
    target = normalize_formula_counts(target_counts)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(StringIO(cif_text))
            if hasattr(parser, "parse_structures"):
                structures = parser.parse_structures(primitive=False)
            else:
                structures = parser.get_structures(primitive=False)
        if not structures:
            raise ValueError("no structures parsed")
        structure = structures[0]
        metric["readable"] = True
        got = structure_counts(structure)
        metric["formula_ok"] = got == target
        metric["composition_exact"] = metric["formula_ok"]
        metric["atom_count_after_expansion"] = int(len(structure))
        metric["atom_count_ok"] = int(len(structure)) == int(sum(target.values()))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detected = int(SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0).get_space_group_number())
            metric["detected_sg"] = detected
            metric["sg_ok"] = detected == int(target_sg)
        except Exception as exc:  # noqa: BLE001
            metric["error"] = f"sg_detect:{type(exc).__name__}:{exc}"
    except Exception as exc:  # noqa: BLE001
        metric["error"] = f"{type(exc).__name__}:{exc}"
    return metric

