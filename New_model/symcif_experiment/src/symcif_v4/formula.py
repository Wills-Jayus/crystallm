from __future__ import annotations

import math
from functools import reduce
from typing import Any

from pymatgen.core import Composition


def parse_formula_counts(formula: str) -> dict[str, int]:
    comp = Composition(formula)
    counts: dict[str, int] = {}
    for element, amount in comp.as_dict().items():
        value = float(amount)
        rounded = int(round(value))
        if not math.isclose(value, rounded, abs_tol=1e-6):
            raise ValueError(f"non-integer formula amount: {formula} -> {element}={amount}")
        counts[str(element)] = rounded
    return dict(sorted(counts.items()))


def normalize_formula_counts(raw: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for element, amount in raw.items():
        value = float(amount)
        rounded = int(round(value))
        if not math.isclose(value, rounded, abs_tol=1e-6):
            raise ValueError(f"non-integer structured formula amount: {element}={amount}")
        if rounded < 0:
            raise ValueError(f"negative formula amount: {element}={amount}")
        if rounded:
            out[str(element)] = rounded
    return dict(sorted(out.items()))


def z_from_counts(counts: dict[str, int]) -> int:
    values = [int(v) for v in counts.values() if int(v) > 0]
    return reduce(math.gcd, values) if values else 1


def reduced_formula_from_counts(counts: dict[str, int]) -> str:
    z = z_from_counts(counts)
    parts: list[str] = []
    for element, count in sorted(counts.items()):
        value = int(count) // z
        parts.append(element if value == 1 else f"{element}{value}")
    return " ".join(parts)


def counts_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return normalize_formula_counts(a) == normalize_formula_counts(b)


def total_atoms(counts: dict[str, int]) -> int:
    return int(sum(int(v) for v in counts.values()))

