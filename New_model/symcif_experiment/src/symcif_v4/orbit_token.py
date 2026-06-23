from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


_VAR_RE = re.compile(r"[xyz]")


@dataclass(frozen=True)
class OrbitToken:
    sg: int
    sg_symbol: str
    hall_number: int | None
    setting_id: str | None
    letter: str
    multiplicity: int
    site_symmetry: str
    enumeration: int | str | None
    representative_expr: tuple[str, str, str]
    free_symbols: tuple[str, ...]
    free_mask: tuple[bool, bool, bool]
    fixed_values: tuple[float | None, float | None, float | None]
    is_fully_fixed: bool
    symmetry_ops: tuple[str, ...]
    origin_shift: tuple[float, float, float] | None
    basis_transform: list[list[float]] | None
    canonical_orbit_id: str

    @property
    def short_key(self) -> str:
        return f"{self.multiplicity}{self.letter}"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "sg": self.sg,
            "sg_symbol": self.sg_symbol,
            "hall_number": self.hall_number,
            "setting_id": self.setting_id,
            "letter": self.letter,
            "multiplicity": self.multiplicity,
            "site_symmetry": self.site_symmetry,
            "enumeration": self.enumeration,
            "representative_expr": list(self.representative_expr),
            "free_symbols": list(self.free_symbols),
            "free_mask": list(self.free_mask),
            "fixed_values": [None if v is None else float(v) for v in self.fixed_values],
            "is_fully_fixed": self.is_fully_fixed,
            "symmetry_ops": list(self.symmetry_ops),
            "origin_shift": None if self.origin_shift is None else list(self.origin_shift),
            "basis_transform": self.basis_transform,
            "canonical_orbit_id": self.canonical_orbit_id,
        }


def expression_tuple(expr: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in str(expr).split(","))
    if len(parts) != 3:
        raise ValueError(f"expected three coordinate expressions, got {expr!r}")
    return parts  # type: ignore[return-value]


def free_symbols_from_expr(expr: str) -> tuple[str, ...]:
    return tuple(ch for ch in ("x", "y", "z") if re.search(ch, expr))


def fixed_value_from_expr(expr: str) -> float | None:
    stripped = expr.strip()
    if _VAR_RE.search(stripped):
        return None
    return float(Fraction(stripped))


def canonical_orbit_id(
    sg: int,
    letter: str,
    multiplicity: int,
    site_symmetry: str | None,
    enumeration: int | str | None,
    setting_id: str | None = None,
) -> str:
    enum = "None" if enumeration is None else str(enumeration)
    sym = "UNKNOWN" if not site_symmetry else str(site_symmetry)
    setting = "default" if not setting_id else str(setting_id)
    return f"setting={setting}|sg={int(sg)}|{int(multiplicity)}{letter}|enum={enum}|sym={sym}"
