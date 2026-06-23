from __future__ import annotations

from dataclasses import replace
from typing import Any

from symcif.models import LatticeParameters, SymCifRecord, WyckoffSite
from symcif.lookup import evaluate_xyz_expr
from symcif_v3.to_cif import render_standard_cif_v3

from .wa_table import WATableCandidate


SAFE_VALUES = (0.125, 0.25, 0.375, 0.625, 0.75, 0.875)


def safe_coord(index: int, axis: int, candidate_index: int = 0) -> float:
    base = SAFE_VALUES[(index + axis + candidate_index) % len(SAFE_VALUES)]
    return float((base + 0.013 * (candidate_index % 7)) % 1.0)


def lattice_from_dict(raw: dict[str, Any]) -> LatticeParameters:
    return LatticeParameters(
        a=float(raw["a"]),
        b=float(raw["b"]),
        c=float(raw["c"]),
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        gamma=float(raw["gamma"]),
        volume=float(raw.get("volume", 0.0)),
    )


def median_lattice(rows: list[dict[str, Any]]) -> LatticeParameters:
    import statistics

    if not rows:
        return LatticeParameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0, 216.0)
    keys = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
    vals = {k: statistics.median(float(row["lattice"][k]) for row in rows) for k in keys}
    return lattice_from_dict(vals)


def coords_by_site(row: dict[str, Any]) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for coord in row["free_coords"]:
        key = f"{int(coord['multiplicity'])}{coord['letter']}"
        out.setdefault(key, []).append((float(coord["x"]), float(coord["y"]), float(coord["z"])))
    return out


def wa_to_record(
    wa: WATableCandidate,
    *,
    formula: str,
    sg_symbol: str,
    lattice: LatticeParameters,
    coord_source: dict[str, list[tuple[float, float, float]]] | None = None,
    candidate_index: int = 0,
    sample_id: str = "composition_exact",
) -> SymCifRecord:
    coord_source = coord_source or {}
    coord_used: dict[str, int] = {}
    sites: list[WyckoffSite] = []
    for idx, (element, token) in enumerate(wa.rows, start=1):
        key = token.short_key
        used = coord_used.get(key, 0)
        coord_used[key] = used + 1
        if key in coord_source and used < len(coord_source[key]):
            coord = coord_source[key][used]
        else:
            params = tuple(safe_coord(idx, axis, candidate_index) for axis in range(3))
            try:
                coord = evaluate_xyz_expr(token.representative_expr, params)
            except Exception:
                coord = tuple(
                    safe_coord(idx, axis, candidate_index) if token.free_mask[axis] else token.fixed_values[axis]
                    for axis in range(3)
                )
        sites.append(
            WyckoffSite(
                index=idx,
                element=element,
                multiplicity=token.multiplicity,
                letter=token.letter,
                representative_coord=tuple(float(v % 1.0) for v in coord),  # type: ignore[arg-type]
                free_mask=token.free_mask,
                fixed_values=token.fixed_values,
                site_symmetry=token.site_symmetry,
                enumeration=None if token.enumeration in {None, "None"} else int(token.enumeration),
            )
        )
    return SymCifRecord(
        sample_id=sample_id,
        source_path=None,
        cell_formula=formula,
        reduced_formula="",
        sg_number=int(wa.sg),
        sg_symbol=sg_symbol,
        z=1,
        lattice=lattice,
        sites=sites,
    )


def render_wa_cif(
    wa: WATableCandidate,
    *,
    formula: str,
    sg_symbol: str,
    lattice: LatticeParameters,
    coord_source: dict[str, list[tuple[float, float, float]]] | None = None,
    candidate_index: int = 0,
    sample_id: str = "composition_exact",
) -> tuple[str | None, dict[str, Any]]:
    try:
        record = wa_to_record(
            wa,
            formula=formula,
            sg_symbol=sg_symbol,
            lattice=lattice,
            coord_source=coord_source,
            candidate_index=candidate_index,
            sample_id=sample_id,
        )
        return render_standard_cif_v3(record), {"to_cif_success": True}
    except Exception as exc:
        return None, {"to_cif_success": False, "error": f"{type(exc).__name__}:{exc}"}
