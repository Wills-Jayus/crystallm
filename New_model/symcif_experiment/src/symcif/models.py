from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatticeParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float


@dataclass(frozen=True)
class WyckoffTemplate:
    sg_number: int
    letter: str
    multiplicity: int
    operations: tuple[str, ...]
    representative_expr: str
    free_mask: tuple[bool, bool, bool]
    fixed_values: tuple[float, float, float]
    site_symmetry: str | None = None
    enumeration: int | None = None
    site_symmetry_status: str = "unknown"
    source: str = "crystalformer_wyckoff_list"


@dataclass
class WyckoffSite:
    index: int
    element: str
    multiplicity: int
    letter: str
    representative_coord: tuple[float, float, float]
    free_mask: tuple[bool, bool, bool]
    fixed_values: tuple[float, float, float]
    site_symmetry: str | None
    enumeration: int | None
    equivalent_indices: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SymCifRecord:
    sample_id: str
    source_path: Path | None
    cell_formula: str
    reduced_formula: str
    sg_number: int
    sg_symbol: str
    z: int
    lattice: LatticeParameters
    sites: list[WyckoffSite]
    original_cif: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
