from __future__ import annotations

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from symcif.models import SymCifRecord


def render_standard_cif_v3(record: SymCifRecord, symprec: float = 0.1) -> str:
    lat = record.lattice
    lattice = Lattice.from_parameters(lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    species: list[str] = []
    coords: list[tuple[float, float, float]] = []
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
