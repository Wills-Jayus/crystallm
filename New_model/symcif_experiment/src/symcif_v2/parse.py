from __future__ import annotations

import re
from pathlib import Path

from symcif.lookup import WyckoffLookup
from symcif.models import LatticeParameters, SymCifRecord, WyckoffSite


class ParseError(ValueError):
    pass


SITE_HEADERS = [
    "_wyckoff_site_index",
    "_wyckoff_site_element",
    "_wyckoff_site_letter",
    "_wyckoff_free_x",
    "_wyckoff_free_y",
    "_wyckoff_free_z",
]


def _value(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key):
            return line.split(None, 1)[1].strip().strip("'\"")
    raise ParseError(f"missing key {key}")


def _parse_coord_token(token: str, *, free: bool, fixed_value: float, key: str) -> float:
    if free:
        if token.upper() == "FIXED":
            raise ParseError(f"free coordinate is FIXED for {key}")
        try:
            value = float(token) % 1.0
        except Exception as exc:
            raise ParseError(f"invalid free coordinate {key}: {token}") from exc
        return 0.0 if abs(value - 1.0) < 1e-8 or abs(value) < 1e-8 else float(value)
    if token.upper() != "FIXED":
        raise ParseError(f"fixed coordinate is not FIXED for {key}: {token}")
    return float(fixed_value)


def parse_symcif_v2_text(text: str, lookup: WyckoffLookup, source_path: str | Path | None = None) -> SymCifRecord:
    lines = [ln.strip() for ln in text.replace("<unk>", "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    data = lines[0].removeprefix("data_") if lines and lines[0].startswith("data_") else "unknown"
    formula = _value(lines, "_chemical_formula_sum")
    sg_number = int(float(_value(lines, "_symmetry_Int_Tables_number")))
    sg_symbol = _value(lines, "_symmetry_space_group_name_H-M")
    lat = LatticeParameters(
        a=float(_value(lines, "_cell_length_a")),
        b=float(_value(lines, "_cell_length_b")),
        c=float(_value(lines, "_cell_length_c")),
        alpha=float(_value(lines, "_cell_angle_alpha")),
        beta=float(_value(lines, "_cell_angle_beta")),
        gamma=float(_value(lines, "_cell_angle_gamma")),
        volume=float(_value(lines, "_cell_volume")),
    )

    headers: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i] != "loop_":
            i += 1
            continue
        j = i + 1
        loop_headers: list[str] = []
        while j < len(lines) and lines[j].startswith("_"):
            loop_headers.append(lines[j])
            j += 1
        if loop_headers and all(header in SITE_HEADERS for header in loop_headers):
            headers = loop_headers
            i = j
            break
        i = j
    if not headers:
        raise ParseError("missing v2 wyckoff loop headers")
    missing = [header for header in SITE_HEADERS if header not in headers]
    if missing:
        raise ParseError(f"missing v2 headers: {missing}")

    rows: list[str] = []
    while i < len(lines) and not lines[i].startswith("_cell_") and lines[i] != "loop_":
        rows.append(lines[i])
        i += 1
    if not rows:
        raise ParseError("empty v2 wyckoff loop")

    header_index = {header: headers.index(header) for header in SITE_HEADERS}
    sites: list[WyckoffSite] = []
    for row in rows:
        parts = row.split()
        if len(parts) != len(headers):
            raise ParseError(f"row/header length mismatch:{row}")
        rec = {key: parts[idx] for key, idx in header_index.items()}
        letter = rec["_wyckoff_site_letter"]
        template = lookup.get(sg_number, letter)
        coord = []
        for axis, key in enumerate(("_wyckoff_free_x", "_wyckoff_free_y", "_wyckoff_free_z")):
            coord.append(
                _parse_coord_token(
                    rec[key],
                    free=bool(template.free_mask[axis]),
                    fixed_value=float(template.fixed_values[axis]),
                    key=key,
                )
            )
        sites.append(
            WyckoffSite(
                index=int(float(rec["_wyckoff_site_index"])),
                element=rec["_wyckoff_site_element"],
                multiplicity=int(template.multiplicity),
                letter=letter,
                representative_coord=tuple(coord),  # type: ignore[arg-type]
                free_mask=template.free_mask,
                fixed_values=template.fixed_values,
                site_symmetry=template.site_symmetry,
                enumeration=template.enumeration,
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
        z=1,
        lattice=lat,
        sites=sites,
    )

