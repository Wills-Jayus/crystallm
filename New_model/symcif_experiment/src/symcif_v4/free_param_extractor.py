from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from symcif.lookup import affine_from_xyz_expr, evaluate_xyz_expr

from .orbit_engine import mod1, unique_coords
from .orbit_token import OrbitToken


_SYMBOLS = ("x", "y", "z")


@dataclass(frozen=True)
class FreeParamExtractionResult:
    free_params: dict[str, float]
    source_coord: tuple[float, float, float]
    mapped_coord: tuple[float, float, float]
    representative_coord: tuple[float, float, float]
    matched_operation: str
    extraction_residual: float
    extraction_method: str
    expansion_count_after_reextract: int
    expansion_ok: bool
    source_in_expanded: bool
    unsupported_reason: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "free_params": {k: float(v) for k, v in self.free_params.items()},
            "source_coord": list(self.source_coord),
            "mapped_coord": list(self.mapped_coord),
            "representative_coord": list(self.representative_coord),
            "matched_operation": self.matched_operation,
            "extraction_residual": float(self.extraction_residual),
            "extraction_method": self.extraction_method,
            "expansion_count_after_reextract": int(self.expansion_count_after_reextract),
            "expansion_ok": bool(self.expansion_ok),
            "source_in_expanded": bool(self.source_in_expanded),
            "unsupported_reason": self.unsupported_reason,
        }


def _periodic_delta(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 0.5) % 1.0) - 0.5)


def _coord_residual(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum(_periodic_delta(x, y) ** 2 for x, y in zip(a, b)))


def _coord_near(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    tolerance: float,
) -> bool:
    return max(_periodic_delta(x, y) for x, y in zip(a, b)) <= float(tolerance)


def _values_from_params(params: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(params.get("x", 0.0)),
        float(params.get("y", 0.0)),
        float(params.get("z", 0.0)),
    )


def _has_linear_combination(expr: str) -> bool:
    for part in str(expr).replace(" ", "").split(","):
        if sum(ch in part for ch in _SYMBOLS) > 1:
            return True
    return False


def _source_in_expanded(
    source_coord: tuple[float, float, float],
    expanded: list[tuple[float, float, float]],
    tolerance: float,
) -> bool:
    return any(_coord_near(source_coord, coord, tolerance) for coord in expanded)


def _solve_operation_candidates(
    source_coord: tuple[float, float, float],
    orbit_token: OrbitToken,
    operation_expr: str,
    tolerance: float,
) -> list[FreeParamExtractionResult]:
    variable_symbols = tuple(orbit_token.free_symbols)
    rot, trans = affine_from_xyz_expr(operation_expr)
    source = np.array(source_coord, dtype=float)
    out: list[FreeParamExtractionResult] = []

    if not variable_symbols:
        values = (0.0, 0.0, 0.0)
        predicted = evaluate_xyz_expr(operation_expr, values)
        residual = _coord_residual(source_coord, predicted)
        expanded = unique_coords([evaluate_xyz_expr(op, values) for op in orbit_token.symmetry_ops], symprec=tolerance)
        if _source_in_expanded(source_coord, expanded, tolerance):
            out.append(
                FreeParamExtractionResult(
                    free_params={},
                    source_coord=source_coord,
                    mapped_coord=evaluate_xyz_expr(", ".join(orbit_token.representative_expr), values),
                    representative_coord=evaluate_xyz_expr(", ".join(orbit_token.representative_expr), values),
                    matched_operation=operation_expr,
                    extraction_residual=residual,
                    extraction_method="fixed_orbit",
                    expansion_count_after_reextract=len(expanded),
                    expansion_ok=len(expanded) == int(orbit_token.multiplicity),
                    source_in_expanded=True,
                )
            )
        return out

    columns = [_SYMBOLS.index(symbol) for symbol in variable_symbols]
    matrix = rot[:, columns]
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    if rank == 0:
        return out

    method = "linear_lstsq"
    if _has_linear_combination(operation_expr):
        method = "linear_lstsq_combination_expr"

    def append_verified(params: dict[str, float], method_name: str) -> None:
        values = _values_from_params(params)
        predicted = evaluate_xyz_expr(operation_expr, values)
        residual = _coord_residual(source_coord, predicted)
        if residual > max(tolerance * math.sqrt(3.0), 1e-8):
            return
        representative = evaluate_xyz_expr(", ".join(orbit_token.representative_expr), values)
        expanded = unique_coords([evaluate_xyz_expr(op, values) for op in orbit_token.symmetry_ops], symprec=tolerance)
        source_ok = _source_in_expanded(source_coord, expanded, tolerance)
        if not source_ok:
            return
        out.append(
            FreeParamExtractionResult(
                free_params=params,
                source_coord=source_coord,
                mapped_coord=representative,
                representative_coord=representative,
                matched_operation=operation_expr,
                extraction_residual=residual,
                extraction_method=method_name,
                expansion_count_after_reextract=len(expanded),
                expansion_ok=len(expanded) == int(orbit_token.multiplicity),
                source_in_expanded=True,
            )
        )

    # Fast path for the common Wyckoff forms used by the problematic rows:
    # each equation either fixes a coordinate or contains one variable.
    per_var_values: dict[str, set[float]] = {symbol: set() for symbol in variable_symbols}
    direct_usable = True
    for row_index in range(3):
        nonzero_cols = [j for j, value in enumerate(matrix[row_index]) if abs(float(value)) > 1e-12]
        if len(nonzero_cols) > 1:
            direct_usable = False
            break
        if len(nonzero_cols) == 1:
            local_col = nonzero_cols[0]
            symbol = variable_symbols[local_col]
            coef = float(matrix[row_index, local_col])
            for lift in range(-2, 3):
                value = (float(source[row_index]) + lift - float(trans[row_index])) / coef
                per_var_values[symbol].add(round(mod1(value), 12))
        else:
            fixed_residual = _periodic_delta(float(source[row_index]), float(trans[row_index]))
            if fixed_residual > tolerance:
                direct_usable = False
                break
    if direct_usable and all(per_var_values[symbol] for symbol in variable_symbols):
        value_lists = [sorted(per_var_values[symbol]) for symbol in variable_symbols]
        combination_count = math.prod(len(values) for values in value_lists)
        if combination_count <= 4096:
            for combo in itertools.product(*value_lists):
                append_verified({symbol: float(value) for symbol, value in zip(variable_symbols, combo)}, "direct_coordinate_solve")
            if out:
                return out

    # Coordinates are fractional modulo one. Try nearby integer lifts of the
    # source coordinate, solve in R, then verify modulo one against the orbit.
    for shift in itertools.product(range(-1, 2), repeat=3):
        target = source + np.array(shift, dtype=float) - trans
        try:
            solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        params = {symbol: mod1(value) for symbol, value in zip(variable_symbols, solution)}
        values = _values_from_params(params)
        predicted = evaluate_xyz_expr(operation_expr, values)
        residual = _coord_residual(source_coord, predicted)
        if residual > max(0.5, 1000.0 * tolerance):
            continue
        representative = evaluate_xyz_expr(", ".join(orbit_token.representative_expr), values)
        expanded = unique_coords([evaluate_xyz_expr(op, values) for op in orbit_token.symmetry_ops], symprec=tolerance)
        source_ok = _source_in_expanded(source_coord, expanded, tolerance)
        if not source_ok:
            continue
        out.append(
            FreeParamExtractionResult(
                free_params=params,
                source_coord=source_coord,
                mapped_coord=representative,
                representative_coord=representative,
                matched_operation=operation_expr,
                extraction_residual=residual,
                extraction_method=method,
                expansion_count_after_reextract=len(expanded),
                expansion_ok=len(expanded) == int(orbit_token.multiplicity),
                source_in_expanded=True,
            )
        )
    return out


def extract_free_params_detailed(
    source_coord: tuple[float, float, float] | list[float],
    orbit_token: OrbitToken,
    tolerance: float = 1e-4,
) -> FreeParamExtractionResult | None:
    coord = tuple(mod1(float(v)) for v in source_coord)
    candidates: list[FreeParamExtractionResult] = []
    for operation_expr in orbit_token.symmetry_ops:
        candidates.extend(_solve_operation_candidates(coord, orbit_token, operation_expr, tolerance=tolerance))
    if not candidates:
        return None
    def param_key(result: FreeParamExtractionResult) -> tuple[float, ...]:
        values = [float(result.free_params.get(symbol, 0.0)) for symbol in orbit_token.free_symbols]
        return tuple(round(v, 12) for v in values)

    def folded_param_norm(result: FreeParamExtractionResult) -> float:
        return float(sum(min(v, 1.0 - v) for v in param_key(result)))

    candidates.sort(
        key=lambda r: (
            not r.expansion_ok,
            not r.source_in_expanded,
            float(r.extraction_residual),
            folded_param_norm(r),
            param_key(r),
            str(r.matched_operation),
        )
    )
    best = candidates[0]
    if best.extraction_residual > tolerance * math.sqrt(3.0):
        return None
    return best


def extract_free_params_from_source_coord(
    source_coord: tuple[float, float, float] | list[float],
    orbit_token: OrbitToken,
    tolerance: float = 1e-4,
) -> dict[str, float] | None:
    result = extract_free_params_detailed(source_coord, orbit_token, tolerance=tolerance)
    if result is None:
        return None
    if not result.expansion_ok:
        return None
    return result.free_params
