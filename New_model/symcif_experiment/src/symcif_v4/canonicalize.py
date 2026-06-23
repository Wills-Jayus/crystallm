from __future__ import annotations

from typing import Any

from .orbit_engine import OrbitEngine


def params_from_coord(coord: dict[str, Any]) -> dict[str, float]:
    return {"x": float(coord["x"]), "y": float(coord["y"]), "z": float(coord["z"])}


def wa_table_from_structured(row: dict[str, Any], engine: OrbitEngine) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    coords_by_order = {int(coord["site_order"]): coord for coord in row["free_coords"]}
    wa_table: list[dict[str, Any]] = []
    free_params: dict[int, dict[str, float]] = {}
    for idx, assign in enumerate(sorted(row["assignment"], key=lambda x: int(x["site_order"]))):
        orbit = engine.get_orbit(int(row["sg"]), str(assign["letter"]))
        coord = coords_by_order.get(int(assign["site_order"]))
        if coord is not None:
            free_params[idx] = params_from_coord(coord)
        else:
            free_params[idx] = {"x": 0.123, "y": 0.234, "z": 0.345}
        wa_table.append(
            {
                "element": str(assign["element"]),
                "orbit_id": orbit.canonical_orbit_id,
                "sg": orbit.sg,
                "letter": orbit.letter,
                "multiplicity": orbit.multiplicity,
                "site_symmetry": orbit.site_symmetry,
                "enumeration": orbit.enumeration,
                "representative_expr": list(orbit.representative_expr),
                "free_symbols": list(orbit.free_symbols),
                "free_params": free_params[idx],
                "setting_id": orbit.setting_id,
                "hall_number": orbit.hall_number,
                "origin_shift": orbit.origin_shift,
                "basis_transform": orbit.basis_transform,
            }
        )
    return wa_table, free_params


def canonical_skeleton_key(wa_table: list[dict[str, Any]]) -> str:
    rows = sorted(wa_table, key=_canonical_row_sort_key)
    return "|".join(str(row["orbit_id"]) for row in rows)


def canonical_wa_key(wa_table: list[dict[str, Any]]) -> str:
    rows = sorted(wa_table, key=_canonical_row_sort_key)
    return "|".join(f"{row['orbit_id']}:{row['element']}" for row in rows)


def _canonical_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("multiplicity", 0) or 0),
        str(row.get("letter", "")),
        str(row.get("enumeration", "")),
        str(row.get("site_symmetry", "")),
        str(row.get("element", "")),
        str(row.get("orbit_id", "")),
    )
