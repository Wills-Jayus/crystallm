from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from symcif.lookup import WyckoffLookup, evaluate_xyz_expr
from symcif.models import SymCifRecord


def render_standard_cif_v2(
    record: SymCifRecord,
    symprec: float = 0.1,
    lookup: WyckoffLookup | None = None,
) -> str:
    lat = record.lattice
    lattice = Lattice.from_parameters(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    species: list[str] = []
    coords: list[tuple[float, float, float]] = []
    if lookup is not None:
        for site in record.sites:
            template = lookup.get(record.sg_number, site.letter)
            rep = []
            for axis in range(3):
                rep.append(site.representative_coord[axis] if site.free_mask[axis] else template.fixed_values[axis])
            rep_tuple = tuple(float(v % 1.0) for v in rep)
            operations = template.operations or (template.representative_expr,)
            for op in operations:
                species.append(site.element)
                coords.append(evaluate_xyz_expr(op, rep_tuple))
        struct = Structure(lattice, species, coords, coords_are_cartesian=False, to_unit_cell=True)
    else:
        for site in record.sites:
            coord = []
            for axis in range(3):
                coord.append(site.representative_coord[axis] if site.free_mask[axis] else site.fixed_values[axis])
            species.append(site.element)
            coords.append(tuple(float(v % 1.0) for v in coord))
        struct = Structure.from_spacegroup(record.sg_symbol, lattice, species, coords)
    try:
        return str(CifWriter(struct, symprec=symprec))
    except Exception:
        return str(CifWriter(struct, symprec=None))
