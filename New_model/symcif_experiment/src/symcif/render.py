from __future__ import annotations

import re
import sys
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from .models import SymCifRecord


def fmt_num(x: float) -> str:
    return f"{float(x):.4f}"


def data_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "", text)
    return cleaned or "unknown"


def render_header(record: SymCifRecord) -> list[str]:
    return [
        f"data_{data_name(record.cell_formula)}",
        f"# sample_id: {record.sample_id}",
        f"_chemical_formula_sum '{record.cell_formula}'",
        f"_symmetry_Int_Tables_number {record.sg_number}",
        f"_symmetry_space_group_name_H-M '{record.sg_symbol}'",
        f"_cell_formula_units_Z {record.z}",
        "",
    ]


def _value_from_lines(lines: list[str], key: str) -> str | None:
    for line in lines:
        if line.startswith(key):
            parts = line.split(None, 1)
            return parts[1].strip() if len(parts) > 1 else ""
    return None


def _without_atom_type_block(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i] == "loop_":
            j = i + 1
            headers = []
            while j < len(lines) and lines[j].startswith("_"):
                headers.append(lines[j])
                j += 1
            if headers and all(h.startswith("_atom_type_") for h in headers):
                while j < len(lines) and lines[j] and not lines[j].startswith("_") and lines[j] != "loop_":
                    j += 1
                if j < len(lines) and not lines[j]:
                    j += 1
                i = j
                continue
        out.append(lines[i])
        i += 1
    return out


def atom_type_block_from_cif(cif_text: str) -> list[str]:
    baseline = render_baseline(cif_text)
    lines = baseline.splitlines()
    for i, line in enumerate(lines):
        if line != "loop_":
            continue
        j = i + 1
        headers = []
        while j < len(lines) and lines[j].startswith("_"):
            headers.append(lines[j])
            j += 1
        if not headers or not all(h.startswith("_atom_type_") for h in headers):
            continue
        while j < len(lines) and lines[j] and not lines[j].startswith("_") and lines[j] != "loop_":
            j += 1
        block = lines[i:j]
        return block + [""]
    return []


def render_lattice(record: SymCifRecord) -> list[str]:
    lat = record.lattice
    return [
        f"_cell_length_a {fmt_num(lat.a)}",
        f"_cell_length_b {fmt_num(lat.b)}",
        f"_cell_length_c {fmt_num(lat.c)}",
        f"_cell_angle_alpha {fmt_num(lat.alpha)}",
        f"_cell_angle_beta {fmt_num(lat.beta)}",
        f"_cell_angle_gamma {fmt_num(lat.gamma)}",
        f"_cell_volume {fmt_num(lat.volume)}",
        "",
    ]


def _site_coord_tokens(site) -> list[str]:
    out = []
    for axis in range(3):
        out.append(fmt_num(site.representative_coord[axis]) if site.free_mask[axis] else "FIXED")
    return out


def render_cf_like(record: SymCifRecord) -> str:
    lines = render_header(record)
    lines.extend(
        [
            "loop_",
            "_wyckoff_site_index",
            "_wyckoff_site_element",
            "_wyckoff_site_multiplicity",
            "_wyckoff_site_letter",
            "_wyckoff_free_x",
            "_wyckoff_free_y",
            "_wyckoff_free_z",
        ]
    )
    for site in record.sites:
        fields = [
            str(site.index),
            site.element,
            str(site.multiplicity),
            site.letter,
            *_site_coord_tokens(site),
        ]
        lines.append(" ".join(fields))
    lines.append("")
    lines.extend(render_lattice(record))
    return "\n".join(lines).rstrip() + "\n"


def render_symcif_v1(record: SymCifRecord) -> str:
    lines = render_header(record)
    lines.extend(
        [
            "loop_",
            "_wyckoff_site_index",
            "_wyckoff_site_element",
            "_wyckoff_site_multiplicity",
            "_wyckoff_site_letter",
            "_wyckoff_site_symmetry",
            "_wyckoff_site_enumeration",
            "_wyckoff_free_x",
            "_wyckoff_free_y",
            "_wyckoff_free_z",
        ]
    )
    for site in record.sites:
        fields = [
            str(site.index),
            site.element,
            str(site.multiplicity),
            site.letter,
            site.site_symmetry or "UNKNOWN",
            str(site.enumeration) if site.enumeration is not None else "UNKNOWN",
            *_site_coord_tokens(site),
        ]
        lines.append(" ".join(fields))
    lines.append("")
    lines.extend(render_lattice(record))
    return "\n".join(lines).rstrip() + "\n"


def render_symcif_v1_atomprops(record: SymCifRecord) -> str:
    lines = [
        f"data_{data_name(record.cell_formula)}",
        f"# sample_id: {record.sample_id}",
    ]
    if record.original_cif:
        lines.extend(atom_type_block_from_cif(record.original_cif))
    lines.extend(
        [
            f"_chemical_formula_sum '{record.cell_formula}'",
            f"_symmetry_Int_Tables_number {record.sg_number}",
            f"_symmetry_space_group_name_H-M '{record.sg_symbol}'",
            f"_cell_formula_units_Z {record.z}",
            "",
            "loop_",
            "_wyckoff_site_index",
            "_wyckoff_site_element",
            "_wyckoff_site_multiplicity",
            "_wyckoff_site_letter",
            "_wyckoff_site_symmetry",
            "_wyckoff_site_enumeration",
            "_wyckoff_free_x",
            "_wyckoff_free_y",
            "_wyckoff_free_z",
        ]
    )
    for site in record.sites:
        fields = [
            str(site.index),
            site.element,
            str(site.multiplicity),
            site.letter,
            site.site_symmetry or "UNKNOWN",
            str(site.enumeration) if site.enumeration is not None else "UNKNOWN",
            *_site_coord_tokens(site),
        ]
        lines.append(" ".join(fields))
    lines.append("")
    lines.extend(render_lattice(record))
    return "\n".join(lines).rstrip() + "\n"


def render_standard_cif(record: SymCifRecord, symprec: float = 0.1) -> str:
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
    return str(CifWriter(struct, symprec=symprec))


def render_baseline(cif_text: str, decimal_places: int = 4) -> str:
    """Apply the original CrystaLLM preprocessing when available."""
    try:
        root = Path(__file__).resolve().parents[2]
        crystallm_root = root / "external" / "CrystaLLM_code"
        if str(crystallm_root) not in sys.path:
            sys.path.insert(0, str(crystallm_root))
        from crystallm import (  # type: ignore
            add_atomic_props_block,
            extract_formula_units,
            replace_data_formula_with_nonreduced_formula,
            round_numbers,
            semisymmetrize_cif,
        )

        if extract_formula_units(cif_text) == 0:
            raise ValueError("formula_units_z_is_zero")
        out = replace_data_formula_with_nonreduced_formula(cif_text)
        out = semisymmetrize_cif(out)
        out = add_atomic_props_block(out, oxi=False)
        out = round_numbers(out, decimal_places=decimal_places)
    except Exception:
        out = cif_text

    lines = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "pymatgen" not in stripped:
            lines.append(stripped)
    lines.append("")
    return "\n".join(lines)


def render_baseline_minprompt(cif_text: str, decimal_places: int = 4) -> str:
    baseline = render_baseline(cif_text, decimal_places=decimal_places)
    lines = [line.rstrip() for line in baseline.splitlines()]
    data_line = next((line for line in lines if line.startswith("data_")), "data_unknown")
    formula = _value_from_lines(lines, "_chemical_formula_sum")
    sg_number = _value_from_lines(lines, "_symmetry_Int_Tables_number")
    sg_symbol = _value_from_lines(lines, "_symmetry_space_group_name_H-M")
    body = _without_atom_type_block(lines)
    body = [
        line
        for line in body
        if line
        and not line.startswith("data_")
        and not line.startswith("_chemical_formula_sum")
        and not line.startswith("_symmetry_Int_Tables_number")
        and not line.startswith("_symmetry_space_group_name_H-M")
    ]

    out = [data_line]
    if formula:
        out.append(f"_chemical_formula_sum {formula}")
    if sg_number:
        out.append(f"_symmetry_Int_Tables_number {sg_number}")
    if sg_symbol:
        out.append(f"_symmetry_space_group_name_H-M {sg_symbol}")
    out.append("")
    out.extend(body)
    out.append("")
    return "\n".join(out).rstrip() + "\n"
