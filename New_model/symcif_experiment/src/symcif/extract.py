from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .lookup import WyckoffLookup, close_fractional
from .models import LatticeParameters, SymCifRecord, WyckoffSite


class ExtractionError(ValueError):
    pass


def sample_id_from_path(path: str | Path) -> str:
    p = Path(path)
    parent = p.parent.name
    return f"{parent}__{p.stem}"


def _clean_formula_for_data_name(formula: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "", formula)


def read_structure(cif_path: str | Path) -> Structure:
    return Structure.from_file(str(cif_path))


def standardize_structure(structure: Structure, symprec: float, angle_tolerance: float) -> Structure:
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    return analyzer.get_conventional_standard_structure()


def _single_element(site) -> str:
    if not site.is_ordered:
        raise ExtractionError("disordered_site_unsupported")
    return site.specie.symbol


def _as_int_if_close(x: float, name: str) -> int:
    nearest = int(round(float(x)))
    if abs(float(x) - nearest) > 1e-5:
        raise ExtractionError(f"{name}_not_integer:{x}")
    return nearest


def extract_record_from_cif(
    cif_path: str | Path,
    lookup: WyckoffLookup,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
    standardize: bool = True,
) -> SymCifRecord:
    cif_path = Path(cif_path)
    original_cif = cif_path.read_text(encoding="utf-8", errors="replace")
    structure = read_structure(cif_path)
    working = standardize_structure(structure, symprec, angle_tolerance) if standardize else structure

    analyzer = SpacegroupAnalyzer(working, symprec=symprec, angle_tolerance=angle_tolerance)
    dataset = analyzer.get_symmetry_dataset()
    sg_number = int(dataset.number)
    sg_symbol = str(dataset.international)
    wyckoffs = list(dataset.wyckoffs)
    site_syms = list(dataset.site_symmetry_symbols)
    equiv = np.array(dataset.equivalent_atoms)

    if len(wyckoffs) != len(working):
        raise ExtractionError("symmetry_dataset_size_mismatch")

    comp = working.composition
    reduced_comp, factor = comp.get_reduced_composition_and_factor()
    z = _as_int_if_close(factor, "formula_units_z")
    lattice = working.lattice
    lattice_params = LatticeParameters(
        a=float(lattice.a),
        b=float(lattice.b),
        c=float(lattice.c),
        alpha=float(lattice.alpha),
        beta=float(lattice.beta),
        gamma=float(lattice.gamma),
        volume=float(lattice.volume),
    )

    sites: list[WyckoffSite] = []
    for site_index, rep_idx in enumerate(sorted(set(int(i) for i in equiv)), start=1):
        indices = tuple(int(i) for i in np.where(equiv == rep_idx)[0])
        elements = {_single_element(working[i]) for i in indices}
        if len(elements) != 1:
            raise ExtractionError(f"mixed_element_orbit:{sorted(elements)}")
        element = next(iter(elements))
        letter = str(wyckoffs[rep_idx])
        site_symmetry = str(site_syms[rep_idx])
        multiplicity = len(indices)
        template = lookup.get(sg_number, letter)
        if template.multiplicity != multiplicity:
            raise ExtractionError(
                f"multiplicity_mismatch:sg={sg_number}:letter={letter}:"
                f"dataset={multiplicity}:template={template.multiplicity}"
            )

        coord = tuple(float(v % 1.0) for v in working[rep_idx].frac_coords)
        fixed_values = []
        for axis, is_free in enumerate(template.free_mask):
            fixed = template.fixed_values[axis]
            if not is_free and not close_fractional(coord[axis], fixed, tol=max(symprec * 0.25, 5e-3)):
                # Same Wyckoff orbit can be represented by another fixed equivalent
                # coordinate (for example 1/4 vs 3/4). Keep the observed value for
                # reporting, but the template value remains the round-trip default.
                fixed = coord[axis]
            fixed_values.append(float(fixed))

        enum = template.enumeration
        if template.site_symmetry and template.site_symmetry != site_symmetry:
            # Prefer spglib's site symmetry for the current structure. This is also
            # reported so the conservative no-pyxtal path is auditable.
            enum = None
        if enum is None:
            sg_templates = [t for (sg, _), t in lookup.templates.items() if sg == sg_number]
            if len(sg_templates) == 1:
                enum = 0

        sites.append(
            WyckoffSite(
                index=site_index,
                element=element,
                multiplicity=multiplicity,
                letter=letter,
                representative_coord=coord,  # type: ignore[arg-type]
                free_mask=template.free_mask,
                fixed_values=tuple(fixed_values),  # type: ignore[arg-type]
                site_symmetry=site_symmetry,
                enumeration=enum,
                equivalent_indices=indices,
            )
        )

    sites.sort(key=lambda s: (s.letter, s.element, s.representative_coord))
    for idx, site in enumerate(sites, start=1):
        site.index = idx

    return SymCifRecord(
        sample_id=sample_id_from_path(cif_path),
        source_path=cif_path,
        cell_formula=comp.formula,
        reduced_formula=reduced_comp.reduced_formula,
        sg_number=sg_number,
        sg_symbol=sg_symbol,
        z=z,
        lattice=lattice_params,
        sites=sites,
        original_cif=original_cif,
        metadata={
            "standardized": standardize,
            "num_sites": len(working),
        },
    )
