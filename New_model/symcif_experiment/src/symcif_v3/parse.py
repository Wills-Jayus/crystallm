from __future__ import annotations

import re
from pathlib import Path

from symcif.lookup import WyckoffLookup
from symcif.models import LatticeParameters, SymCifRecord, WyckoffSite


class ParseError(ValueError):
    pass


SKELETON_HEADERS = [
    "_wyckoff_site_index",
    "_wyckoff_site_multiplicity",
    "_wyckoff_site_letter",
    "_wyckoff_site_symmetry",
    "_wyckoff_site_enumeration",
]
ASSIGNMENT_HEADERS = [
    "_wyckoff_site_index",
    "_wyckoff_site_element",
]
COORD_HEADERS = [
    "_wyckoff_site_index",
    "_wyckoff_free_x",
    "_wyckoff_free_y",
    "_wyckoff_free_z",
]


def _value(lines: list[str], key: str) -> str:
    for line in lines:
        if line.startswith(key):
            return line.split(None, 1)[1].strip().strip("'\"")
    raise ParseError(f"missing key {key}")


def _loops(lines: list[str]) -> list[tuple[list[str], list[str]]]:
    loops: list[tuple[list[str], list[str]]] = []
    i = 0
    while i < len(lines):
        if lines[i] != "loop_":
            i += 1
            continue
        j = i + 1
        headers: list[str] = []
        while j < len(lines) and lines[j].startswith("_"):
            headers.append(lines[j])
            j += 1
        rows: list[str] = []
        while j < len(lines) and lines[j] != "loop_" and not lines[j].startswith("_cell_"):
            if lines[j] and not lines[j].startswith("_"):
                rows.append(lines[j])
            j += 1
        loops.append((headers, rows))
        i = j
    return loops


def _find_loop(loops: list[tuple[list[str], list[str]]], required: list[str]) -> tuple[list[str], list[str]]:
    required_set = set(required)
    for headers, rows in loops:
        if set(headers) == required_set:
            return headers, rows
    raise ParseError(f"missing loop with headers {required}")


def _row_dicts(headers: list[str], rows: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        parts = row.split()
        if len(parts) != len(headers):
            raise ParseError(f"row/header length mismatch:{row}")
        out.append(dict(zip(headers, parts)))
    return out


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


def parse_symcif_v3_text(text: str, lookup: WyckoffLookup, source_path: str | Path | None = None) -> SymCifRecord:
    lines = [ln.strip() for ln in text.replace("<unk>", "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    data = lines[0].removeprefix("data_") if lines and lines[0].startswith("data_") else "unknown"
    formula = _value(lines, "_chemical_formula_sum")
    sg_number = int(float(_value(lines, "_symmetry_Int_Tables_number")))
    sg_symbol = _value(lines, "_symmetry_space_group_name_H-M")
    try:
        z = int(float(_value(lines, "_cell_formula_units_Z")))
    except ParseError:
        z = 1
    lat = LatticeParameters(
        a=float(_value(lines, "_cell_length_a")),
        b=float(_value(lines, "_cell_length_b")),
        c=float(_value(lines, "_cell_length_c")),
        alpha=float(_value(lines, "_cell_angle_alpha")),
        beta=float(_value(lines, "_cell_angle_beta")),
        gamma=float(_value(lines, "_cell_angle_gamma")),
        volume=float(_value(lines, "_cell_volume")),
    )

    loops = _loops(lines)
    skeleton_headers, skeleton_rows = _find_loop(loops, SKELETON_HEADERS)
    assignment_headers, assignment_rows = _find_loop(loops, ASSIGNMENT_HEADERS)
    coord_headers, coord_rows = _find_loop(loops, COORD_HEADERS)
    skeleton = _row_dicts(skeleton_headers, skeleton_rows)
    assignments = {int(float(r["_wyckoff_site_index"])): r for r in _row_dicts(assignment_headers, assignment_rows)}
    coords = {int(float(r["_wyckoff_site_index"])): r for r in _row_dicts(coord_headers, coord_rows)}
    if not skeleton:
        raise ParseError("empty v3 skeleton loop")

    sites: list[WyckoffSite] = []
    for rec in skeleton:
        idx = int(float(rec["_wyckoff_site_index"]))
        if idx not in assignments:
            raise ParseError(f"missing assignment for site {idx}")
        if idx not in coords:
            raise ParseError(f"missing coordinates for site {idx}")
        letter = rec["_wyckoff_site_letter"]
        template = lookup.get(sg_number, letter)
        coord_rec = coords[idx]
        coord = []
        for axis, key in enumerate(("_wyckoff_free_x", "_wyckoff_free_y", "_wyckoff_free_z")):
            coord.append(
                _parse_coord_token(
                    coord_rec[key],
                    free=bool(template.free_mask[axis]),
                    fixed_value=float(template.fixed_values[axis]),
                    key=key,
                )
            )
        enum_raw = rec.get("_wyckoff_site_enumeration")
        enum = None if enum_raw in {None, "UNKNOWN"} else int(float(enum_raw))
        sites.append(
            WyckoffSite(
                index=idx,
                element=assignments[idx]["_wyckoff_site_element"],
                multiplicity=int(float(rec["_wyckoff_site_multiplicity"])),
                letter=letter,
                representative_coord=tuple(coord),  # type: ignore[arg-type]
                free_mask=template.free_mask,
                fixed_values=template.fixed_values,
                site_symmetry=rec.get("_wyckoff_site_symmetry", template.site_symmetry),
                enumeration=enum,
            )
        )
    sites.sort(key=lambda site: site.index)

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

