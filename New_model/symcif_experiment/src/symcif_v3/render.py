from __future__ import annotations

import re

from symcif.models import SymCifRecord
from symcif.render import fmt_num


def data_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "", text)
    return cleaned or "unknown"


def _coord_tokens(site) -> list[str]:
    out: list[str] = []
    for axis in range(3):
        out.append(fmt_num(site.representative_coord[axis]) if site.free_mask[axis] else "FIXED")
    return out


def render_symcif_v3(record: SymCifRecord) -> str:
    """Render a CrystalFormer-like staged SymCIF text sequence.

    Ordering is intentionally factorized:
    condition/header -> Wyckoff skeleton W -> assignment A -> free coords X -> lattice L.
    The field names reuse the existing SymCIF tokenizer vocabulary.
    """

    lines = [
        f"data_{data_name(record.cell_formula)}",
        f"# sample_id: {record.sample_id}",
        f"_chemical_formula_sum '{record.cell_formula}'",
        f"_symmetry_Int_Tables_number {record.sg_number}",
        f"_symmetry_space_group_name_H-M '{record.sg_symbol}'",
        f"_cell_formula_units_Z {record.z}",
        "",
        "loop_",
        "_wyckoff_site_index",
        "_wyckoff_site_multiplicity",
        "_wyckoff_site_letter",
        "_wyckoff_site_symmetry",
        "_wyckoff_site_enumeration",
    ]
    for site in record.sites:
        lines.append(
            " ".join(
                [
                    str(site.index),
                    str(site.multiplicity),
                    site.letter,
                    site.site_symmetry or "UNKNOWN",
                    str(site.enumeration) if site.enumeration is not None else "UNKNOWN",
                ]
            )
        )

    lines.extend(
        [
            "",
            "loop_",
            "_wyckoff_site_index",
            "_wyckoff_site_element",
        ]
    )
    for site in record.sites:
        lines.append(f"{site.index} {site.element}")

    lines.extend(
        [
            "",
            "loop_",
            "_wyckoff_site_index",
            "_wyckoff_free_x",
            "_wyckoff_free_y",
            "_wyckoff_free_z",
        ]
    )
    for site in record.sites:
        lines.append(" ".join([str(site.index), *_coord_tokens(site)]))

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

