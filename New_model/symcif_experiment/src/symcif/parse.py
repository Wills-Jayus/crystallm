from __future__ import annotations

import re
from pathlib import Path

from .lookup import WyckoffLookup
from .models import LatticeParameters, SymCifRecord, WyckoffSite


class ParseError(ValueError):
    pass


def _value(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key):
            return line.split(None, 1)[1].strip().strip("'\"")
    raise ParseError(f"missing key {key}")


def parse_symcif_text(text: str, lookup: WyckoffLookup, source_path: str | Path | None = None) -> SymCifRecord:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    data = lines[0].removeprefix("data_") if lines and lines[0].startswith("data_") else "unknown"
    formula = _value(lines, "_chemical_formula_sum")
    sg_number = int(_value(lines, "_symmetry_Int_Tables_number"))
    sg_symbol = _value(lines, "_symmetry_space_group_name_H-M")
    z = int(float(_value(lines, "_cell_formula_units_Z")))
    lat = LatticeParameters(
        a=float(_value(lines, "_cell_length_a")),
        b=float(_value(lines, "_cell_length_b")),
        c=float(_value(lines, "_cell_length_c")),
        alpha=float(_value(lines, "_cell_angle_alpha")),
        beta=float(_value(lines, "_cell_angle_beta")),
        gamma=float(_value(lines, "_cell_angle_gamma")),
        volume=float(_value(lines, "_cell_volume")),
    )

    headers = []
    i = 0
    while i < len(lines):
        if lines[i] != "loop_":
            i += 1
            continue
        j = i + 1
        loop_headers = []
        while j < len(lines) and lines[j].startswith("_"):
            loop_headers.append(lines[j])
            j += 1
        if loop_headers and all(header.startswith("_wyckoff_") for header in loop_headers):
            headers = loop_headers
            i = j
            break
        i = j
    if not headers:
        raise ParseError("missing wyckoff loop headers")
    rows = []
    while i < len(lines) and not lines[i].startswith("_cell_") and not lines[i] == "loop_":
        rows.append(lines[i])
        i += 1

    sites: list[WyckoffSite] = []
    for row in rows:
        parts = row.split()
        if len(parts) != len(headers):
            raise ParseError(f"row/header length mismatch:{row}")
        rec = dict(zip(headers, parts))
        letter = rec["_wyckoff_site_letter"]
        template = lookup.get(sg_number, letter)
        coord = []
        for axis, key in enumerate(("_wyckoff_free_x", "_wyckoff_free_y", "_wyckoff_free_z")):
            token = rec[key]
            coord.append(template.fixed_values[axis] if token == "FIXED" else float(token))
        site_sym = rec.get("_wyckoff_site_symmetry", template.site_symmetry)
        enum_raw = rec.get("_wyckoff_site_enumeration")
        enum = None if enum_raw in {None, "UNKNOWN"} else int(enum_raw)
        sites.append(
            WyckoffSite(
                index=int(rec["_wyckoff_site_index"]),
                element=rec["_wyckoff_site_element"],
                multiplicity=int(rec["_wyckoff_site_multiplicity"]),
                letter=letter,
                representative_coord=tuple(float(v % 1.0) for v in coord),  # type: ignore[arg-type]
                free_mask=template.free_mask,
                fixed_values=template.fixed_values,
                site_symmetry=site_sym,
                enumeration=enum,
            )
        )
    sid = "parsed"
    match = re.search(r"#\s*sample_id:\s*(\S+)", text)
    if match:
        sid = match.group(1)
    return SymCifRecord(
        sample_id=sid,
        source_path=Path(source_path) if source_path else None,
        cell_formula=formula or data,
        reduced_formula="",
        sg_number=sg_number,
        sg_symbol=sg_symbol,
        z=z,
        lattice=lat,
        sites=sites,
    )
