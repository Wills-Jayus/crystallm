from __future__ import annotations

from collections import Counter

from pymatgen.core import Composition
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .models import SymCifRecord


def _formula_key(structure: Structure) -> str:
    return structure.composition.reduced_composition.alphabetical_formula


def _composition_key(formula: str) -> str:
    return Composition(formula).reduced_composition.alphabetical_formula


def _multiplicity_counter(record: SymCifRecord) -> Counter[tuple[str, int, str]]:
    return Counter((s.element, s.multiplicity, s.letter) for s in record.sites)


def validate_roundtrip(
    original: SymCifRecord,
    roundtrip_cif: str,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "pymatgen_readable": False,
        "formula_consistent": False,
        "space_group_consistent": False,
        "multiplicity_consistent": False,
    }
    try:
        rt = Structure.from_str(roundtrip_cif, fmt="cif")
        result["pymatgen_readable"] = True
        orig_formula = original.reduced_formula or original.cell_formula
        result["roundtrip_formula"] = rt.composition.reduced_formula
        result["formula_consistent"] = _formula_key(rt) == _composition_key(orig_formula)
        analyzer = SpacegroupAnalyzer(rt, symprec=symprec, angle_tolerance=angle_tolerance)
        result["roundtrip_sg_number"] = analyzer.get_space_group_number()
        result["roundtrip_sg_symbol"] = analyzer.get_space_group_symbol()
        result["space_group_consistent"] = int(result["roundtrip_sg_number"]) == int(original.sg_number)
        ds = analyzer.get_symmetry_dataset()
        equiv = list(ds.equivalent_atoms)
        wyckoffs = list(ds.wyckoffs)
        seen = sorted(set(int(i) for i in equiv))
        counts = Counter()
        for rep in seen:
            indices = [i for i, v in enumerate(equiv) if int(v) == rep]
            elem = rt[rep].specie.symbol if rt[rep].is_ordered else rt[rep].species_string
            counts[(elem, len(indices), str(wyckoffs[rep]))] += 1
        result["roundtrip_multiplicities"] = {str(k): v for k, v in counts.items()}
        result["multiplicity_consistent"] = counts == _multiplicity_counter(original)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
