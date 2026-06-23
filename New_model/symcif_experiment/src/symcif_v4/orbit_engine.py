from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from symcif.lookup import evaluate_xyz_expr

from .formula import normalize_formula_counts
from .orbit_token import (
    OrbitToken,
    canonical_orbit_id,
    expression_tuple,
    fixed_value_from_expr,
    free_symbols_from_expr,
)


def mod1(value: float) -> float:
    out = float(value) % 1.0
    if abs(out) < 1e-10 or abs(out - 1.0) < 1e-10:
        return 0.0
    return out


def coord_key(coord: tuple[float, float, float], symprec: float = 1e-5) -> tuple[int, int, int]:
    return tuple(int(round(mod1(v) / symprec)) for v in coord)  # type: ignore[return-value]


def unique_coords(coords: list[tuple[float, float, float]], symprec: float = 1e-5) -> list[tuple[float, float, float]]:
    seen: set[tuple[int, int, int]] = set()
    out: list[tuple[float, float, float]] = []
    for coord in coords:
        wrapped = tuple(mod1(v) for v in coord)  # type: ignore[assignment]
        key = coord_key(wrapped, symprec=symprec)
        if key in seen:
            continue
        seen.add(key)
        out.append(wrapped)
    return out


def _safe_data_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")
    return clean or "symcif_v4"


def formula_sum(counts: dict[str, Any]) -> str:
    norm = normalize_formula_counts(counts)
    return " ".join(f"{element}{count}" for element, count in sorted(norm.items()))


def formula_compact(counts: dict[str, Any]) -> str:
    norm = normalize_formula_counts(counts)
    return "".join(f"{element}{count}" for element, count in sorted(norm.items()))


def cell_volume(lattice: dict[str, Any]) -> float:
    a = float(lattice["a"])
    b = float(lattice["b"])
    c = float(lattice["c"])
    alpha = math.radians(float(lattice["alpha"]))
    beta = math.radians(float(lattice["beta"]))
    gamma = math.radians(float(lattice["gamma"]))
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    factor = max(0.0, 1.0 - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2.0 * cos_a * cos_b * cos_g)
    return a * b * c * math.sqrt(factor)


class OrbitEngine:
    def __init__(self, lookup_json: str | Path, sg_symbol_by_number: dict[int, str] | None = None):
        self.lookup_json = Path(lookup_json)
        self.sg_symbol_by_number = sg_symbol_by_number or {}
        raw = json.loads(self.lookup_json.read_text(encoding="utf-8"))
        self._orbits_by_sg: dict[int, list[OrbitToken]] = {}
        self._orbit_by_id: dict[str, OrbitToken] = {}
        self._orbit_by_sg_letter: dict[tuple[int, str], OrbitToken] = {}
        for sg_raw, entries in raw.items():
            sg = int(sg_raw)
            sg_symbol = self.sg_symbol_by_number.get(sg, f"SG{sg}")
            orbits: list[OrbitToken] = []
            for _, item in sorted(entries.items(), key=lambda kv: (int(kv[1]["multiplicity"]), str(kv[1]["letter"]))):
                rep = expression_tuple(str(item["template"]))
                rep_joined = ", ".join(rep)
                free_symbols = free_symbols_from_expr(rep_joined)
                free_mask = tuple(fixed_value_from_expr(part) is None for part in rep)
                fixed_values = tuple(fixed_value_from_expr(part) for part in rep)
                site_symmetry = str(item.get("site_symmetry") or "UNKNOWN")
                enumeration = item.get("enumeration")
                oid = canonical_orbit_id(
                    sg=sg,
                    letter=str(item["letter"]),
                    multiplicity=int(item["multiplicity"]),
                    site_symmetry=site_symmetry,
                    enumeration=enumeration,
                    setting_id="crystalformer",
                )
                token = OrbitToken(
                    sg=sg,
                    sg_symbol=sg_symbol,
                    hall_number=None,
                    setting_id="crystalformer",
                    letter=str(item["letter"]),
                    multiplicity=int(item["multiplicity"]),
                    site_symmetry=site_symmetry,
                    enumeration=enumeration,
                    representative_expr=rep,
                    free_symbols=free_symbols,
                    free_mask=free_mask,  # type: ignore[arg-type]
                    fixed_values=fixed_values,  # type: ignore[arg-type]
                    is_fully_fixed=not any(free_mask),
                    symmetry_ops=tuple(str(op) for op in item.get("operations") or [item["template"]]),
                    origin_shift=None,
                    basis_transform=None,
                    canonical_orbit_id=oid,
                )
                orbits.append(token)
                self._orbit_by_id[token.canonical_orbit_id] = token
                self._orbit_by_sg_letter[(sg, token.letter)] = token
            self._orbits_by_sg[sg] = sorted(orbits, key=lambda o: (o.multiplicity, o.letter, str(o.enumeration), o.site_symmetry))

    @classmethod
    def from_structured_root(cls, lookup_json: str | Path, structured_root: str | Path) -> "OrbitEngine":
        root = Path(structured_root)
        mapping: dict[int, str] = {}
        for split in ("train", "val", "test"):
            path = root / f"{split}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    mapping[int(row["sg"])] = str(row.get("sg_symbol") or f"SG{int(row['sg'])}")
        return cls(lookup_json=lookup_json, sg_symbol_by_number=mapping)

    def get_orbits(self, sg: int) -> list[OrbitToken]:
        return list(self._orbits_by_sg.get(int(sg), []))

    def get_orbit(self, sg: int, letter: str) -> OrbitToken:
        return self._orbit_by_sg_letter[(int(sg), str(letter))]

    def get_orbit_by_id(self, orbit_id: str) -> OrbitToken:
        return self._orbit_by_id[str(orbit_id)]

    def evaluate_representative(self, orbit: OrbitToken, free_params: dict[str, float]) -> tuple[float, float, float]:
        values = (
            float(free_params.get("x", 0.123)),
            float(free_params.get("y", 0.234)),
            float(free_params.get("z", 0.345)),
        )
        return evaluate_xyz_expr(", ".join(orbit.representative_expr), values)

    def expand_orbit(
        self,
        orbit: OrbitToken,
        free_params: dict[str, float],
        symprec: float = 1e-5,
    ) -> list[tuple[float, float, float]]:
        values = (
            float(free_params.get("x", 0.123)),
            float(free_params.get("y", 0.234)),
            float(free_params.get("z", 0.345)),
        )
        coords = [evaluate_xyz_expr(op, values) for op in orbit.symmetry_ops]
        return unique_coords(coords, symprec=symprec)

    def render_cif_from_wa_table(
        self,
        wa_table: list[dict[str, Any]],
        lattice: dict[str, Any],
        free_params_by_row: dict[int | str, dict[str, float]],
        formula_counts: dict[str, int],
        sg: int,
        sg_symbol: str | None = None,
        data_name: str = "symcif_v4",
        symprec: float = 1e-5,
    ) -> str:
        atom_rows: list[tuple[str, str, float, float, float]] = []
        for idx, row in enumerate(wa_table):
            orbit_id = str(row.get("orbit_id") or row.get("canonical_orbit_id"))
            orbit = self.get_orbit_by_id(orbit_id) if orbit_id and orbit_id != "None" else self.get_orbit(int(sg), str(row["letter"]))
            params = free_params_by_row.get(idx)
            if params is None:
                params = free_params_by_row.get(str(idx), {})
            expanded = self.expand_orbit(orbit, params or {}, symprec=symprec)
            for j, coord in enumerate(expanded):
                atom_rows.append((str(row["element"]), f"{row['element']}{idx}_{j}", coord[0], coord[1], coord[2]))
        formula = formula_sum(formula_counts)
        formula_id = formula_compact(formula_counts)
        symbol = sg_symbol or self.sg_symbol_by_number.get(int(sg), f"SG{int(sg)}")
        safe_name = _safe_data_name(data_name)
        block_name = f"{formula_id}_{safe_name}" if formula_id else safe_name
        lines = [
            "# generated by SymCIF-v4 OrbitEngine",
            f"data_{block_name}",
            f"_symmetry_space_group_name_H-M   '{symbol}'",
            f"_symmetry_Int_Tables_number   {int(sg)}",
            f"_cell_length_a   {float(lattice['a']):.8f}",
            f"_cell_length_b   {float(lattice['b']):.8f}",
            f"_cell_length_c   {float(lattice['c']):.8f}",
            f"_cell_angle_alpha   {float(lattice['alpha']):.8f}",
            f"_cell_angle_beta   {float(lattice['beta']):.8f}",
            f"_cell_angle_gamma   {float(lattice['gamma']):.8f}",
            f"_cell_volume   {cell_volume(lattice):.8f}",
            "_cell_formula_units_Z   1",
            f"_chemical_formula_structural   {formula_id}",
            f"_chemical_formula_sum   '{formula}'",
            "loop_",
            " _symmetry_equiv_pos_site_id",
            " _symmetry_equiv_pos_as_xyz",
            "  1  'x, y, z'",
            "loop_",
            " _atom_site_type_symbol",
            " _atom_site_label",
            " _atom_site_symmetry_multiplicity",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
            " _atom_site_occupancy",
        ]
        for element, label, x, y, z in atom_rows:
            lines.append(f"  {element}  {label}  1  {mod1(x):.8f}  {mod1(y):.8f}  {mod1(z):.8f}  1")
        return "\n".join(lines) + "\n"

    def expanded_atom_count(self, wa_table: list[dict[str, Any]], free_params_by_row: dict[int | str, dict[str, float]]) -> int:
        total = 0
        for idx, row in enumerate(wa_table):
            orbit = self.get_orbit_by_id(str(row["orbit_id"]))
            params = free_params_by_row.get(idx) or free_params_by_row.get(str(idx), {})
            total += len(self.expand_orbit(orbit, params))
        return int(total)
