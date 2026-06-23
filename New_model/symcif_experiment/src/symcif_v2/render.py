from __future__ import annotations

import re

from symcif.models import SymCifRecord


def fmt_num(x: float) -> str:
    return f"{float(x):.4f}"


def data_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "", text)
    return cleaned or "unknown"


def _site_coord_tokens(site) -> list[str]:
    out: list[str] = []
    for axis in range(3):
        out.append(fmt_num(site.representative_coord[axis]) if site.free_mask[axis] else "FIXED")
    return out


def render_symcif_v2(record: SymCifRecord) -> str:
    """Render the compact SymCIF-v2 text format.

    v2 leaves multiplicity, site symmetry, enumeration, and fixed/free masks out
    of the text. They are derived from SG + Wyckoff letter at parse/render time.
    """
    lines = [
        f"data_{data_name(record.cell_formula)}",
        f"# sample_id: {record.sample_id}",
        f"_chemical_formula_sum '{record.cell_formula}'",
        f"_symmetry_Int_Tables_number {record.sg_number}",
        f"_symmetry_space_group_name_H-M '{record.sg_symbol}'",
        "",
        "loop_",
        "_wyckoff_site_index",
        "_wyckoff_site_element",
        "_wyckoff_site_letter",
        "_wyckoff_free_x",
        "_wyckoff_free_y",
        "_wyckoff_free_z",
    ]
    for site in record.sites:
        fields = [
            str(site.index),
            site.element,
            site.letter,
            *_site_coord_tokens(site),
        ]
        lines.append(" ".join(fields))
    lat = record.lattice
    lines.extend(
        [
            "",
            f"_cell_length_a {fmt_num(lat.a)}",
            f"_cell_length_b {fmt_num(lat.b)}",
            f"_cell_length_c {fmt_num(lat.c)}",
            f"_cell_angle_alpha {fmt_num(lat.alpha)}",
            f"_cell_angle_beta {fmt_num(lat.beta)}",
            f"_cell_angle_gamma {fmt_num(lat.gamma)}",
            f"_cell_volume {fmt_num(lat.volume)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"

