#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine, cell_volume  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402

import run_mp20_geometry_breakthrough as gb  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
import train_symcif_v4_geometry_model as geom  # noqa: E402

try:
    from pymatgen.core.periodic_table import Element  # type: ignore
    from pymatgen.core import Structure  # type: ignore
except Exception:  # pragma: no cover - optional fallback for environments without pymatgen metadata.
    Element = None  # type: ignore
    Structure = None  # type: ignore


_ENGINE: OrbitEngine | None = None
_GEOM_INDEX: gb.GeometryIndex | None = None
_TOP_K = 1
_GEOMETRY_MODE = "e08"
_GEOMETRY_RANKS_PER_WA = 5
_GEOMETRY_PLAN_MODE = "wa_diverse"
_GEOMETRY_SOURCE_STRATEGY = "row_conditioned_knn"
_GEOMETRY_LATTICE_MODE = "source"
_GEOMETRY_PARAM_VARIANT_MODE = "none"
_HYBRID_GEOMETRY_WA = 5
_HYBRID_GEOMETRY_RANKS = 3
_REQUIRE_COMPOSITION_EXACT = True
_EXCLUDE_SELF_SOURCE = False
_PHYSICAL_SOURCE_SELECT_TOP_N = 3
_SOURCE_SUCCESS_PRIOR: dict[str, Any] = {}
_SOURCE_PRIOR_WEIGHT = 0.0
_SOURCE_TEMPLATE_BANK: dict[str, Any] = {}
_SOURCE_PROPOSAL_CACHE: dict[tuple[str, str], list[str]] = {}
_GEOM_SELECTOR: "OpentryGeometrySelector | None" = None
_WRITE_GEOMETRY_METADATA = False
_LATTICE_MODEL: geom.GeometryNet | None = None
_LATTICE_VOCABS: dict[str, dict[str, int]] = {}
_LATTICE_MEAN: torch.Tensor | None = None
_LATTICE_STD: torch.Tensor | None = None
_LATTICE_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    return [v2.geometry_training_record(row) for row in read_jsonl(path)]


def normalize_rows(engine: OrbitEngine, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        orbit = engine.get_orbit_by_id(str(row.get("orbit_id") or row.get("canonical_orbit_id")))
        out.append(
            {
                "element": str(row["element"]),
                "orbit_id": orbit.canonical_orbit_id,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
            }
        )
    return v2.canonical_rows({"wa_table": out})


def pseudo_record(record: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(record)
    out["wa_table"] = rows
    skel, wa = v2.canonical_keys_from_rows(rows)
    out["canonical_skeleton_key"] = skel
    out["canonical_wa_key"] = wa
    out["atom_count"] = sum(int(v) for v in out["formula_counts"].values())
    return out


def record_keys(record: dict[str, Any]) -> tuple[str, str]:
    return v2.canonical_keys_from_rows(v2.canonical_rows(record))


@lru_cache(maxsize=256)
def element_feature_vector(symbol: str) -> tuple[float, float, float, float, float]:
    if Element is None:
        return (0.5, 0.5, 0.5, 0.5, 0.5)
    try:
        el = Element(str(symbol))
        z = float(getattr(el, "Z", 0) or 0) / 100.0
        row = float(getattr(el, "row", 0) or 0) / 7.0
        group = float(getattr(el, "group", 0) or 0) / 18.0
        electronegativity = float(getattr(el, "X", 0) or 0) / 4.0
        radius_raw = getattr(el, "atomic_radius", None) or getattr(el, "atomic_radius_calculated", None) or 1.5
        radius = float(radius_raw) / 3.0
        return (z, row, group, electronegativity, radius)
    except Exception:
        return (0.5, 0.5, 0.5, 0.5, 0.5)


def formula_chem_vector(record: dict[str, Any]) -> tuple[float, ...]:
    cached = record.get("_opentry_formula_chem_vector")
    if isinstance(cached, tuple):
        return cached
    counts = {str(k): float(v) for k, v in dict(record.get("formula_counts") or {}).items()}
    total = max(1.0, sum(counts.values()))
    means = [0.0] * 5
    second = [0.0] * 5
    for element, count in counts.items():
        weight = float(count) / total
        vec = element_feature_vector(element)
        for idx, value in enumerate(vec):
            means[idx] += weight * float(value)
            second[idx] += weight * float(value) * float(value)
    stds = [max(0.0, second[idx] - means[idx] * means[idx]) ** 0.5 for idx in range(5)]
    out = tuple(means + stds)
    record["_opentry_formula_chem_vector"] = out
    return out


def chem_distance(target: dict[str, Any], source: dict[str, Any]) -> float:
    tv = formula_chem_vector(target)
    sv = formula_chem_vector(source)
    return float(sum(abs(float(a) - float(b)) for a, b in zip(tv, sv)))


def element_distance(a: str, b: str) -> float:
    av = element_feature_vector(str(a))
    bv = element_feature_vector(str(b))
    return float(sum(abs(float(x) - float(y)) for x, y in zip(av, bv)) / max(1, len(av)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def record_volume_per_atom(record: dict[str, Any]) -> float | None:
    lattice = record.get("lattice")
    if not isinstance(lattice, dict):
        return None
    try:
        volume = float(cell_volume({k: float(lattice[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}))
        atoms = max(1, gb.atom_count(record))
        if volume > 0.0:
            return volume / float(atoms)
    except Exception:
        return None
    return None


def circular_distance(a: float, b: float) -> float:
    diff = abs((float(a) % 1.0) - (float(b) % 1.0))
    return float(min(diff, 1.0 - diff))


class OpentryGeometrySelector:
    """Train-only source selector that keeps exact W/A analogues before kNN pruning."""

    def __init__(self, train_records: list[dict[str, Any]], base_index: gb.GeometryIndex, source_template_bank: dict[str, Any] | None = None) -> None:
        self.train_records = train_records
        self.base_index = base_index
        self.record_by_id = {str(record.get("sample_id")): record for record in train_records}
        self.by_wa: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        self.by_skeleton: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        self.by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.by_signature: dict[tuple[int, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
        self.by_sg_row: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        self.by_sg_atom_bucket: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        self.by_crystal_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.vpa_by_sg_atom_bucket: dict[tuple[int, str], list[float]] = defaultdict(list)
        self.vpa_by_crystal_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.vpa_by_atom_bucket: dict[str, list[float]] = defaultdict(list)
        self.global_vpa_values: list[float] = []
        self.param_values_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self.positive_param_values_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self.positive_vpa_values_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self.row_feasibility_values_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        self.row_pair_feasibility_by_key: dict[tuple[Any, ...], float] = {}
        self.geometry_bundle_entries_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        self.row_by_exact: dict[tuple[int, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.row_by_sg_orbit_symbols: dict[tuple[int, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.row_by_orbit_element_symbols: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.row_by_orbit_symbols: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.success_by_wa: dict[tuple[int, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_by_skeleton: dict[tuple[int, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_by_sg_row: dict[tuple[int, int], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_by_sg_atom_bucket: dict[tuple[int, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_by_sg: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_rows7_by_sg: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        self.success_row_by_exact: dict[tuple[int, str, str, str], list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.success_row_by_sg_orbit_symbols: dict[tuple[int, str, str], list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.success_row_by_orbit_element_symbols: dict[tuple[str, str, str], list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        self.success_row_by_orbit_symbols: dict[tuple[str, str], list[tuple[float, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for record in train_records:
            sg = int(record["sg"])
            skel, wa = record_keys(record)
            self.by_sg[sg].append(record)
            self.by_wa[(sg, wa)].append(record)
            self.by_skeleton[(sg, skel)].append(record)
            self.by_signature[(sg, gb.free_signature(record))].append(record)
            self.by_sg_row[(sg, gb.row_count(record))].append(record)
            self.by_sg_atom_bucket[(sg, gb.atom_bucket(gb.atom_count(record)))].append(record)
            self.by_crystal_bucket[(gb.crystal_system(sg), gb.atom_bucket(gb.atom_count(record)))].append(record)
            vpa = record_volume_per_atom(record)
            if vpa is not None:
                atom_bucket = gb.atom_bucket(gb.atom_count(record))
                self.vpa_by_sg_atom_bucket[(sg, atom_bucket)].append(vpa)
                self.vpa_by_crystal_bucket[(gb.crystal_system(sg), atom_bucket)].append(vpa)
                self.vpa_by_atom_bucket[atom_bucket].append(vpa)
                self.global_vpa_values.append(vpa)
            for row in v2.canonical_rows(record):
                oid = str(row["orbit_id"])
                element = str(row["element"])
                symbols = self._symbols_key(row)
                self.row_by_exact[(sg, oid, element, symbols)].append((record, row))
                self.row_by_sg_orbit_symbols[(sg, oid, symbols)].append((record, row))
                self.row_by_orbit_element_symbols[(oid, element, symbols)].append((record, row))
                self.row_by_orbit_symbols[(oid, symbols)].append((record, row))
                for sym, value in dict(row.get("free_params") or {}).items():
                    self._add_param_value(sg, row, str(sym), float(value))
        for values in self.param_values_by_key.values():
            values.sort()
        self._load_source_template_bank(source_template_bank or {})

    @staticmethod
    def _symbols_key(row: dict[str, Any]) -> str:
        return ",".join(sorted(str(sym) for sym in row.get("free_symbols") or []))

    def _param_keys(self, sg: int, row: dict[str, Any], symbol: str) -> list[tuple[Any, ...]]:
        oid = str(row.get("orbit_id"))
        element = str(row.get("element"))
        symbols = self._symbols_key(row)
        sym = str(symbol)
        return [
            ("sg_orbit_element_symbols_sym", int(sg), oid, element, symbols, sym),
            ("sg_orbit_symbols_sym", int(sg), oid, symbols, sym),
            ("orbit_element_symbols_sym", oid, element, symbols, sym),
            ("orbit_symbols_sym", oid, symbols, sym),
            ("orbit_sym", oid, sym),
        ]

    def _add_param_value(self, sg: int, row: dict[str, Any], symbol: str, value: float) -> None:
        if not (value == value):
            return
        for key in self._param_keys(int(sg), row, str(symbol)):
            self.param_values_by_key[key].append(float(value) % 1.0)

    def param_manifold_penalty(self, sg: int, row: dict[str, Any], symbol: str, value: float) -> float:
        values: list[float] | None = None
        for key in self._param_keys(int(sg), row, str(symbol)):
            bucket = self.param_values_by_key.get(key)
            if bucket and len(bucket) >= 4:
                values = bucket
                break
        if not values:
            return 0.0
        x = float(value) % 1.0
        pos = bisect_left(values, x)
        candidates = []
        for idx in (pos - 1, pos, 0, len(values) - 1):
            if 0 <= idx < len(values):
                candidates.append(float(values[idx]))
        nearest = min(circular_distance(x, v) for v in candidates)
        return float(min(2.0, nearest / 0.035))

    def param_manifold_quantile(self, sg: int, row: dict[str, Any], symbol: str, q: float) -> float:
        values: list[float] | None = None
        for key in self._param_keys(int(sg), row, str(symbol)):
            bucket = self.param_values_by_key.get(key)
            if bucket and len(bucket) >= 4:
                values = bucket
                break
        if values:
            return float(quantile(list(values), float(q))) % 1.0
        return float(self.base_index.free_value(str(row["orbit_id"]), str(symbol), float(q))) % 1.0

    @staticmethod
    def _circular_blend(start: float, end: float, alpha: float) -> float:
        s = float(start) % 1.0
        e = float(end) % 1.0
        delta = ((e - s + 0.5) % 1.0) - 0.5
        return float((s + float(alpha) * delta) % 1.0)

    def manifold_param_variant(self, target: dict[str, Any], params: dict[int, dict[str, float]], rank: int) -> dict[int, dict[str, float]]:
        if int(rank) <= 0:
            return params
        profiles = [
            (0.50, 0.00),
            (0.25, 0.35),
            (0.75, 0.35),
            (0.10, 0.60),
            (0.90, 0.60),
            (0.35, 0.50),
            (0.65, 0.50),
            (0.05, 0.75),
            (0.95, 0.75),
            (0.50, 0.85),
        ]
        q, alpha = profiles[min(max(0, int(rank)), len(profiles) - 1)]
        sg = int(target["sg"])
        out: dict[int, dict[str, float]] = {}
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            base_row = dict(params.get(int(row_idx), {}))
            out_row: dict[str, float] = {}
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                current = float(base_row.get(sym, self.base_index.free_value(str(row["orbit_id"]), sym, 0.5))) % 1.0
                manifold_value = self.param_manifold_quantile(sg, row, sym, float(q))
                out_row[sym] = self._circular_blend(current, manifold_value, float(alpha))
            out[int(row_idx)] = out_row
        return out

    def expected_vpa_quantile(self, target: dict[str, Any], q: float, k: int = 96) -> float | None:
        sg = int(target["sg"])
        atom_bucket = gb.atom_bucket(gb.atom_count(target))
        candidate_pools = [
            self.by_sg_atom_bucket.get((sg, atom_bucket), []),
            self.by_sg.get(sg, []),
            self.by_crystal_bucket.get((gb.crystal_system(sg), atom_bucket), []),
        ]
        nearest_values: list[float] = []
        for pool in candidate_pools:
            if not pool:
                continue
            reduced = pool
            if len(reduced) > int(k):
                reduced = sorted(reduced, key=lambda rec: self.cheap_chem_source_distance(target, rec))[: int(k)]
            nearest_values = [v for rec in reduced if (v := record_volume_per_atom(rec)) is not None]
            if len(nearest_values) >= 8:
                return float(quantile(nearest_values, float(q)))
        fallback_values = (
            self.vpa_by_sg_atom_bucket.get((sg, atom_bucket), [])
            or self.vpa_by_crystal_bucket.get((gb.crystal_system(sg), atom_bucket), [])
            or self.vpa_by_atom_bucket.get(atom_bucket, [])
            or self.global_vpa_values
        )
        if fallback_values:
            return float(quantile(list(fallback_values), float(q)))
        return None

    def manifold_lattice_variant(self, target: dict[str, Any], lattice: dict[str, float], rank: int) -> dict[str, float]:
        if int(rank) <= 0:
            return lattice
        profiles = [
            (0.50, 0.00),
            (0.35, 0.45),
            (0.65, 0.45),
            (0.20, 0.60),
            (0.80, 0.60),
            (0.10, 0.70),
            (0.90, 0.70),
        ]
        q, strength = profiles[min(max(0, int(rank)), len(profiles) - 1)]
        target_vpa = self.expected_vpa_quantile(target, float(q))
        try:
            current_vpa = float(cell_volume({k: float(lattice[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")})) / max(1.0, float(gb.atom_count(target)))
        except Exception:
            current_vpa = 0.0
        if target_vpa is None or target_vpa <= 0.0 or current_vpa <= 0.0:
            return lattice
        raw_factor = (float(target_vpa) / float(current_vpa)) ** (1.0 / 3.0)
        factor = 1.0 + float(strength) * (raw_factor - 1.0)
        factor = max(0.88, min(1.14, factor))
        out = dict(lattice)
        for key in ("a", "b", "c"):
            out[key] = max(0.5, float(out[key]) * factor)
        return out

    def positive_param_quantile(self, sg: int, row: dict[str, Any], symbol: str, q: float) -> float | None:
        for key in self._param_keys(int(sg), row, str(symbol)):
            bucket = self.positive_param_values_by_key.get(key)
            if bucket and len(bucket) >= 2:
                return float(quantile(list(bucket), float(q))) % 1.0
        return None

    def positive_param_variant(self, target: dict[str, Any], params: dict[int, dict[str, float]], rank: int) -> dict[int, dict[str, float]]:
        if int(rank) <= 0:
            return params
        profiles = [
            (0.50, 0.00),
            (0.50, 0.30),
            (0.35, 0.40),
            (0.65, 0.40),
            (0.25, 0.55),
            (0.75, 0.55),
            (0.15, 0.65),
            (0.85, 0.65),
            (0.50, 0.75),
            (0.05, 0.70),
        ]
        q, alpha = profiles[min(max(0, int(rank)), len(profiles) - 1)]
        sg = int(target["sg"])
        out: dict[int, dict[str, float]] = {}
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            base_row = dict(params.get(int(row_idx), {}))
            out_row: dict[str, float] = {}
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                current = float(base_row.get(sym, self.base_index.free_value(str(row["orbit_id"]), sym, 0.5))) % 1.0
                positive_value = self.positive_param_quantile(sg, row, sym, float(q))
                if positive_value is None:
                    out_row[sym] = current
                else:
                    out_row[sym] = self._circular_blend(current, positive_value, float(alpha))
            out[int(row_idx)] = out_row
        return out

    def positive_nearest_complex_variant(self, target: dict[str, Any], params: dict[int, dict[str, float]], rank: int) -> dict[int, dict[str, float]]:
        if int(rank) <= 0 or gb.row_count(target) < 7:
            return params
        alphas = [0.0, 0.18, 0.28, 0.38, 0.50, 0.62, 0.74, 0.86]
        alpha = float(alphas[min(max(0, int(rank)), len(alphas) - 1)])
        sg = int(target["sg"])
        out: dict[int, dict[str, float]] = {}
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            base_row = dict(params.get(int(row_idx), {}))
            out_row: dict[str, float] = {}
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                current = float(base_row.get(sym, self.base_index.free_value(str(row["orbit_id"]), sym, 0.5))) % 1.0
                bucket: list[float] | None = None
                for key in self._param_keys(sg, row, sym):
                    values = self.positive_param_values_by_key.get(key)
                    if values and len(values) >= 2:
                        bucket = list(values)
                        break
                if not bucket:
                    out_row[sym] = current
                    continue
                nearest = sorted((float(v) % 1.0 for v in bucket), key=lambda v: (circular_distance(current, v), v))
                window = nearest[: min(len(nearest), 10)]
                selected = window[(int(rank) + int(row_idx)) % len(window)]
                out_row[sym] = self._circular_blend(current, float(selected), alpha)
            out[int(row_idx)] = out_row
        return out

    def _row_feasibility_keys(self, sg: int, row: dict[str, Any], symbol: str) -> list[tuple[Any, ...]]:
        oid = str(row.get("orbit_id"))
        element = str(row.get("element"))
        sym = str(symbol)
        return [
            ("sg_orbit_element_sym", int(sg), oid, element, sym),
            ("sg_orbit_sym", int(sg), oid, sym),
            ("orbit_element_sym", oid, element, sym),
            ("orbit_sym", oid, sym),
            ("sg_sym", int(sg), sym),
        ]

    def row_feasibility_complex_variant(self, target: dict[str, Any], params: dict[int, dict[str, float]], rank: int) -> dict[int, dict[str, float]]:
        if int(rank) <= 0 or gb.row_count(target) < 7:
            return params
        alphas = [0.0, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82, 0.94]
        alpha = float(alphas[min(max(0, int(rank)), len(alphas) - 1)])
        sg = int(target["sg"])
        out: dict[int, dict[str, float]] = {}
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            base_row = dict(params.get(int(row_idx), {}))
            out_row: dict[str, float] = {}
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                current = float(base_row.get(sym, self.base_index.free_value(str(row["orbit_id"]), sym, 0.5))) % 1.0
                choices: list[dict[str, Any]] | None = None
                for key in self._row_feasibility_keys(sg, row, sym):
                    bucket = self.row_feasibility_values_by_key.get(key)
                    if bucket:
                        choices = bucket
                        break
                if not choices:
                    out_row[sym] = current
                    continue
                ranked = sorted(
                    choices[:8],
                    key=lambda item: (
                        -float(item.get("score", 0.0)),
                        circular_distance(current, float(item.get("center", 0.5))),
                        -int(item.get("positive", 0)),
                    ),
                )
                pick = ranked[(int(rank) + int(row_idx)) % len(ranked)]
                target_value = float(pick.get("center", current)) % 1.0
                out_row[sym] = self._circular_blend(current, target_value, alpha)
            out[int(row_idx)] = out_row
        return out

    def positive_expected_vpa_quantile(self, target: dict[str, Any], q: float) -> float | None:
        sg = int(target["sg"])
        atom_bucket = gb.atom_bucket(gb.atom_count(target))
        keys = [
            ("sg_atom_bucket", sg, atom_bucket),
            ("crystal_atom_bucket", gb.crystal_system(sg), atom_bucket),
            ("atom_bucket", atom_bucket),
            ("global",),
        ]
        for key in keys:
            values = self.positive_vpa_values_by_key.get(tuple(key))
            if values and len(values) >= 4:
                return float(quantile(list(values), float(q)))
        return None

    def positive_lattice_variant(self, target: dict[str, Any], lattice: dict[str, float], rank: int) -> dict[str, float]:
        if int(rank) <= 0:
            return lattice
        profiles = [
            (0.50, 0.00),
            (0.50, 0.30),
            (0.35, 0.35),
            (0.65, 0.35),
            (0.25, 0.45),
            (0.75, 0.45),
            (0.15, 0.50),
            (0.85, 0.50),
            (0.50, 0.60),
            (0.05, 0.55),
        ]
        q, strength = profiles[min(max(0, int(rank)), len(profiles) - 1)]
        target_vpa = self.positive_expected_vpa_quantile(target, float(q))
        try:
            current_vpa = float(cell_volume({k: float(lattice[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")})) / max(1.0, float(gb.atom_count(target)))
        except Exception:
            current_vpa = 0.0
        if target_vpa is None or target_vpa <= 0.0 or current_vpa <= 0.0:
            return lattice
        raw_factor = (float(target_vpa) / float(current_vpa)) ** (1.0 / 3.0)
        factor = 1.0 + float(strength) * (raw_factor - 1.0)
        factor = max(0.90, min(1.12, factor))
        out = dict(lattice)
        for key in ("a", "b", "c"):
            out[key] = max(0.5, float(out[key]) * factor)
        return out

    def _geometry_bundle_keys(self, target: dict[str, Any], broad: bool = False) -> list[tuple[Any, ...]]:
        sg = int(target["sg"])
        skel, wa = record_keys(target)
        atom_bucket = gb.atom_bucket(gb.atom_count(target))
        keys: list[tuple[Any, ...]] = [
            ("wa", sg, wa),
            ("skeleton", sg, skel),
        ]
        if broad:
            keys.extend(
                [
                    ("sg_row_atom", sg, gb.row_count(target), atom_bucket),
                    ("sg_atom", sg, atom_bucket),
                    ("sg", sg),
                ]
            )
        return keys

    def _select_geometry_bundle_entry(self, target: dict[str, Any], rank: int, broad: bool = False) -> dict[str, Any] | None:
        seen: set[str] = set()
        entries: list[dict[str, Any]] = []
        for key in self._geometry_bundle_keys(target, broad=bool(broad)):
            for entry in self.geometry_bundle_entries_by_key.get(tuple(key), [])[:96]:
                uid = str(entry.get("entry_id") or entry.get("source_candidate_uid") or id(entry))
                if uid in seen:
                    continue
                seen.add(uid)
                entries.append(entry)
            if entries and not broad:
                break
            if len(entries) >= 96:
                break
        if not entries:
            return None
        rows_ge_7 = gb.row_count(target) >= 7
        if rows_ge_7:
            entries.sort(key=lambda item: (int(item.get("target_row_count", 0)) >= 7, float(item.get("score", 0.0))), reverse=True)
        idx = int(rank) % len(entries)
        return entries[idx]

    def _entry_row_param_maps(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for row in list(entry.get("row_params") or []):
            item = dict(row)
            item["params"] = {str(k): float(v) % 1.0 for k, v in dict(item.get("params") or {}).items()}
            rows.append(item)
        return rows

    @staticmethod
    def _bundle_row_key(row: dict[str, Any], include_element: bool = True) -> tuple[Any, ...]:
        return (
            str(row.get("orbit_id")),
            str(row.get("element")) if include_element else "*",
            ",".join(sorted(str(sym) for sym in row.get("free_symbols") or [])),
            int(row.get("multiplicity", 1)),
            str(row.get("site_symmetry")),
        )

    def geometry_bundle_variant(
        self,
        target: dict[str, Any],
        params: dict[int, dict[str, float]],
        lattice: dict[str, float],
        rank: int,
        broad: bool = False,
    ) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
        entry = self._select_geometry_bundle_entry(target, int(rank), broad=bool(broad))
        if entry is None:
            return params, lattice
        target_rows = v2.canonical_rows(target)
        entry_rows = self._entry_row_param_maps(entry)
        exact: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        loose: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in entry_rows:
            exact[self._bundle_row_key(row, include_element=True)].append(row)
            loose[self._bundle_row_key(row, include_element=False)].append(row)
        used: set[int] = set()
        out_params: dict[int, dict[str, float]] = {}
        for row_idx, target_row in enumerate(target_rows):
            selected: dict[str, Any] | None = None
            for key in (self._bundle_row_key(target_row, include_element=True), self._bundle_row_key(target_row, include_element=False)):
                bucket = exact.get(key) if key[1] != "*" else loose.get(key)
                if not bucket:
                    continue
                for candidate in bucket:
                    marker = id(candidate)
                    if marker not in used:
                        selected = candidate
                        used.add(marker)
                        break
                if selected is not None:
                    break
            if selected is None and row_idx < len(entry_rows):
                selected = entry_rows[row_idx]
            base_row = dict(params.get(int(row_idx), {}))
            selected_params = dict((selected or {}).get("params") or {})
            out_row: dict[str, float] = {}
            for sym in target_row.get("free_symbols") or []:
                key = str(sym)
                out_row[key] = float(selected_params.get(key, base_row.get(key, self.base_index.free_value(str(target_row["orbit_id"]), key, 0.5)))) % 1.0
            out_params[int(row_idx)] = out_row
        entry_lattice = dict(entry.get("lattice") or {})
        out_lattice = dict(lattice)
        if all(key in entry_lattice for key in ("a", "b", "c", "alpha", "beta", "gamma")):
            out_lattice = {key: float(entry_lattice[key]) for key in ("a", "b", "c", "alpha", "beta", "gamma")}
            entry_vpa = None
            try:
                entry_atoms = max(1.0, float(entry.get("atom_count") or gb.atom_count(target)))
                entry_vpa = float(cell_volume(out_lattice)) / entry_atoms
            except Exception:
                entry_vpa = None
            target_vpa = self.expected_vpa(target)
            if entry_vpa is not None and target_vpa is not None and entry_vpa > 0.0 and target_vpa > 0.0:
                raw_factor = (float(target_vpa) / float(entry_vpa)) ** (1.0 / 3.0)
                factor = max(0.82, min(1.22, raw_factor))
                for key in ("a", "b", "c"):
                    out_lattice[key] = max(0.5, float(out_lattice[key]) * factor)
        return out_params, out_lattice

    def _extend_pool(
        self,
        *,
        target: dict[str, Any],
        source_pool: list[dict[str, Any]],
        tier: int,
        out: list[tuple[int, float, dict[str, Any]]],
        seen: set[str],
        k: int,
    ) -> None:
        if not source_pool:
            return
        limit = max(k * 16, 64)
        reduced = sorted(source_pool, key=lambda rec: gb.row_condition_distance(target, rec))[:limit]
        for rec in reduced:
            sid = str(rec["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            out.append((tier, gb.row_condition_distance(target, rec), rec))

    def _load_source_template_bank(self, bank: dict[str, Any]) -> None:
        for entry in list(bank.get("positive_param_values") or []):
            key = tuple(entry.get("key") or [])
            values = [float(v) % 1.0 for v in (entry.get("values") or []) if v is not None]
            if key and values:
                self.positive_param_values_by_key[key].extend(values)
        for entry in list(bank.get("positive_vpa_values") or []):
            key = tuple(entry.get("key") or [])
            values = [float(v) for v in (entry.get("values") or []) if v is not None and float(v) > 0.0]
            if key and values:
                self.positive_vpa_values_by_key[key].extend(values)
        for values in self.positive_param_values_by_key.values():
            values.sort()
        for values in self.positive_vpa_values_by_key.values():
            values.sort()
        for entry in list(bank.get("row_feasibility_param_bins") or []):
            key = tuple(entry.get("key") or [])
            values = [dict(v) for v in (entry.get("values") or []) if v is not None]
            if key and values:
                values.sort(key=lambda item: (float(item.get("score", 0.0)), int(item.get("positive", 0)), int(item.get("total", 0))), reverse=True)
                self.row_feasibility_values_by_key[key].extend(values[:12])
        for entry in list(bank.get("row_pair_feasibility") or []):
            key = tuple(entry.get("key") or [])
            if key:
                self.row_pair_feasibility_by_key[key] = float(entry.get("score") or 0.0)
        for entry in list(bank.get("geometry_bundle_entries") or []):
            for raw_key in list(entry.get("keys") or []):
                key = tuple(raw_key or [])
                if key:
                    self.geometry_bundle_entries_by_key[key].append(dict(entry))
        for values in self.geometry_bundle_entries_by_key.values():
            values.sort(key=lambda item: (float(item.get("score", 0.0)), int(item.get("target_row_count", 0))), reverse=True)
        entries = list(bank.get("entries") or [])
        for entry in entries:
            sid = str(entry.get("source_sample_id") or "")
            source = self.record_by_id.get(sid)
            if source is None:
                continue
            sg = int(entry.get("sg") or source.get("sg") or 0)
            score = float(entry.get("score") or 0.0)
            row_count = int(entry.get("target_row_count") or 0)
            atom_bucket = str(entry.get("target_atom_bucket") or gb.atom_bucket(gb.atom_count(source)))
            skel = str(entry.get("canonical_skeleton_key") or "")
            wa = str(entry.get("canonical_wa_key") or "")
            if wa:
                self.success_by_wa[(sg, wa)].append((score, source))
            if skel:
                self.success_by_skeleton[(sg, skel)].append((score, source))
            self.success_by_sg_row[(sg, row_count)].append((score, source))
            self.success_by_sg_atom_bucket[(sg, atom_bucket)].append((score, source))
            self.success_by_sg[sg].append((score, source))
            if row_count >= 7 or int(entry.get("rows_ge_7_positive_count") or 0) > 0:
                self.success_rows7_by_sg[sg].append((score, source))
            self._add_success_rows(source, score)
        for bucket in (
            self.success_by_wa,
            self.success_by_skeleton,
            self.success_by_sg_row,
            self.success_by_sg_atom_bucket,
            self.success_by_sg,
            self.success_rows7_by_sg,
        ):
            for key, values in bucket.items():
                values.sort(key=lambda item: float(item[0]), reverse=True)
        for bucket in (
            self.success_row_by_exact,
            self.success_row_by_sg_orbit_symbols,
            self.success_row_by_orbit_element_symbols,
            self.success_row_by_orbit_symbols,
        ):
            for key, values in bucket.items():
                values.sort(key=lambda item: float(item[0]), reverse=True)

    def _add_success_rows(self, source: dict[str, Any], score: float) -> None:
        sg = int(source["sg"])
        for row in v2.canonical_rows(source):
            oid = str(row["orbit_id"])
            element = str(row["element"])
            symbols = self._symbols_key(row)
            item = (float(score), source, row)
            self.success_row_by_exact[(sg, oid, element, symbols)].append(item)
            self.success_row_by_sg_orbit_symbols[(sg, oid, symbols)].append(item)
            self.success_row_by_orbit_element_symbols[(oid, element, symbols)].append(item)
            self.success_row_by_orbit_symbols[(oid, symbols)].append(item)

    def source_template_bank_candidates(self, target: dict[str, Any], k: int, use_freepattern_penalty: bool = False) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        skel, wa = record_keys(target)
        rows = gb.row_count(target)
        atom_bucket = gb.atom_bucket(gb.atom_count(target))
        pools: list[tuple[int, list[tuple[float, dict[str, Any]]]]] = [
            (0, self.success_by_wa.get((sg, wa), [])),
            (1, self.success_by_skeleton.get((sg, skel), [])),
        ]
        if rows >= 7:
            pools.append((2, self.success_rows7_by_sg.get(sg, [])))
        pools.extend(
            [
                (3, self.success_by_sg_row.get((sg, rows), [])),
                (4, self.success_by_sg_atom_bucket.get((sg, atom_bucket), [])),
                (5, self.success_by_sg.get(sg, [])),
            ]
        )
        seen: set[str] = set()
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        for tier, pool in pools:
            for bank_rank, (bank_score, rec) in enumerate(pool[: max(k * 16, 96)]):
                sid = str(rec.get("sample_id") or "")
                if sid in seen:
                    continue
                seen.add(sid)
                chem_row_dist = self.row_condition_chem_distance(target, rec)
                align_cost = self.source_alignment_cost(target, rec, chemical=True)
                quality_penalty = self.source_quality_penalty(rec)
                score = 0.18 * float(tier) + chem_row_dist + 0.35 * align_cost + quality_penalty - 0.10 * float(bank_score) + 0.0005 * float(bank_rank)
                if use_freepattern_penalty:
                    score += 0.65 * self.source_freepattern_compatibility_cost(target, rec)
                scored.append((score, chem_row_dist, quality_penalty, rec))
            if len(scored) >= max(k * 10, 64):
                break
        if len(scored) < max(k, 1):
            for rec in self.row_aligned_chem_quality_candidates(target, max(k * 2, k), use_freepattern_penalty=use_freepattern_penalty):
                sid = str(rec.get("sample_id") or "")
                if sid in seen:
                    continue
                seen.add(sid)
                chem_row_dist = self.row_condition_chem_distance(target, rec)
                align_cost = self.source_alignment_cost(target, rec, chemical=True)
                quality_penalty = self.source_quality_penalty(rec)
                score = 2.0 + chem_row_dist + 0.35 * align_cost + quality_penalty
                scored.append((score, chem_row_dist, quality_penalty, rec))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [rec for _, _, _, rec in scored[: max(k, 1)]]

    def anchored_successbank_candidates(self, target: dict[str, Any], k: int, use_freepattern_penalty: bool = False, anchor_count: int = 3) -> list[dict[str, Any]]:
        base = self.row_aligned_chem_quality_candidates(target, max(k, int(anchor_count)), use_freepattern_penalty=False)
        bank = self.source_template_bank_candidates(target, max(k * 2, int(anchor_count) + k), use_freepattern_penalty=use_freepattern_penalty)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in base[: max(0, min(int(anchor_count), k))]:
            sid = str(rec.get("sample_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            out.append(rec)
        for rec in bank:
            if len(out) >= max(k, 1):
                break
            sid = str(rec.get("sample_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            out.append(rec)
        for rec in base:
            if len(out) >= max(k, 1):
                break
            sid = str(rec.get("sample_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            out.append(rec)
        return out[: max(k, 1)]

    def source_proposal_candidates(self, target: dict[str, Any], k: int, anchor_count: int = 0) -> list[dict[str, Any]]:
        target_id = str(target.get("sample_id") or "")
        _skel, wa = record_keys(target)
        proposed_ids = list(_SOURCE_PROPOSAL_CACHE.get((target_id, wa), []))
        base = self.row_aligned_chem_quality_candidates(target, max(k * 2, int(anchor_count) + k))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rec in base[: max(0, min(int(anchor_count), k))]:
            sid = str(rec.get("sample_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            out.append(rec)
        for sid in proposed_ids:
            if len(out) >= max(k, 1):
                break
            if sid in seen:
                continue
            rec = self.record_by_id.get(str(sid))
            if rec is None:
                continue
            seen.add(str(sid))
            out.append(rec)
        for rec in base:
            if len(out) >= max(k, 1):
                break
            sid = str(rec.get("sample_id") or "")
            if sid in seen:
                continue
            seen.add(sid)
            out.append(rec)
        return out[: max(k, 1)]

    def _success_row_template_pool(self, target: dict[str, Any], row: dict[str, Any]) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
        sg = int(target["sg"])
        oid = str(row["orbit_id"])
        element = str(row["element"])
        symbols = self._symbols_key(row)
        pools = [
            self.success_row_by_exact.get((sg, oid, element, symbols), []),
            self.success_row_by_sg_orbit_symbols.get((sg, oid, symbols), []),
            self.success_row_by_orbit_element_symbols.get((oid, element, symbols), []),
            self.success_row_by_orbit_symbols.get((oid, symbols), []),
        ]
        seen: set[tuple[str, int]] = set()
        out: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for pool in pools:
            if not pool:
                continue
            for score, rec, proto_row in pool[:128]:
                key = (str(rec.get("sample_id")), id(proto_row))
                if key in seen:
                    continue
                seen.add(key)
                out.append((float(score), rec, proto_row))
            if len(out) >= 48:
                break
        return out

    def row_success_template_params(self, target: dict[str, Any], rank: int) -> tuple[dict[int, dict[str, float]], str | None, float | None]:
        params: dict[int, dict[str, float]] = {}
        source_ids: list[str] = []
        distances: list[float] = []
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            row_params: dict[str, float] = {}
            pool = self._success_row_template_pool(target, row)
            selected: tuple[float, dict[str, Any], dict[str, Any]] | None = None
            if pool:
                selected = pool[(int(rank) + row_idx * 5) % len(pool)]
            proto_params = dict(selected[2].get("free_params") or {}) if selected is not None else {}
            if selected is not None:
                source_ids.append(str(selected[1].get("sample_id")))
                distances.append(float(gb.row_condition_distance(target, selected[1])))
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                if sym in proto_params:
                    row_params[sym] = float(proto_params[sym]) % 1.0
                else:
                    row_params[sym] = self.base_index.free_value(str(row["orbit_id"]), sym, q)
            params[row_idx] = row_params
        source_summary = None
        if source_ids:
            source_summary = ";".join(source_ids[:5])
            if len(source_ids) > 5:
                source_summary += f";+{len(source_ids) - 5}"
        mean_distance = None if not distances else float(sum(distances) / len(distances))
        return params, source_summary, mean_distance

    def candidates(self, target: dict[str, Any], strategy: str, k: int) -> list[dict[str, Any]]:
        def without_self(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not _EXCLUDE_SELF_SOURCE:
                return rows
            target_id = str(target.get("sample_id") or "")
            return [row for row in rows if str(row.get("sample_id") or "") != target_id]

        if strategy == "row_conditioned_knn":
            return without_self(self.base_index.candidates(target, "row_conditioned_knn", max(k + 1, k)))
        if strategy == "row_aligned_knn":
            return without_self(self.base_index.candidates(target, "row_conditioned_knn", max(k + 1, k)))
        if strategy == "row_aligned_quality":
            return without_self(self.row_aligned_quality_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_aligned_chem_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_aligned_chem_freepattern_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k), use_freepattern_penalty=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_sourceprior_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k), use_source_prior=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_sourceprior_freepattern_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k), use_freepattern_penalty=True, use_source_prior=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_successbank_quality":
            return without_self(self.source_template_bank_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_aligned_chem_successbank_freepattern_quality":
            return without_self(self.source_template_bank_candidates(target, max(k + 1, k), use_freepattern_penalty=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_successbank_tail_quality":
            return without_self(self.anchored_successbank_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_aligned_chem_successbank_tail_freepattern_quality":
            return without_self(self.anchored_successbank_candidates(target, max(k + 1, k), use_freepattern_penalty=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_sourceproposal_quality":
            return without_self(self.source_proposal_candidates(target, max(k + 1, k), anchor_count=0))[: max(k, 1)]
        if strategy == "row_aligned_chem_sourceproposal_tail_quality":
            return without_self(self.source_proposal_candidates(target, max(k + 1, k), anchor_count=3))[: max(k, 1)]
        if strategy == "row_aligned_chem_rowbank_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_aligned_chem_rowbank_freepattern_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k), use_freepattern_penalty=True))[: max(k, 1)]
        if strategy == "row_aligned_chem_param_quality":
            return without_self(self.row_aligned_chem_quality_candidates(target, max(k + 1, k), use_param_penalty=True))[: max(k, 1)]
        if strategy == "row_pair_source_feasible_quality":
            return without_self(self.row_pair_source_feasible_quality_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy == "row_pair_source_feasible_freepattern_quality":
            return without_self(self.row_pair_source_feasible_quality_candidates(target, max(k + 1, k), use_freepattern_penalty=True))[: max(k, 1)]
        if strategy == "row_aligned_priority":
            return without_self(self.row_aligned_candidates(target, max(k + 1, k)))[: max(k, 1)]
        if strategy != "wa_skeleton_priority":
            raise ValueError(f"Unsupported geometry source strategy: {strategy}")

        sg = int(target["sg"])
        skel, wa = record_keys(target)
        scored: list[tuple[int, float, dict[str, Any]]] = []
        seen: set[str] = set()
        pools: list[tuple[int, list[dict[str, Any]]]] = [
            (0, self.by_wa.get((sg, wa), [])),
            (1, self.by_skeleton.get((sg, skel), [])),
            (2, self.by_signature.get((sg, gb.free_signature(target)), [])),
            (3, self.by_sg_row.get((sg, gb.row_count(target)), [])),
        ]
        for tier, pool in pools:
            self._extend_pool(target=target, source_pool=pool, tier=tier, out=scored, seen=seen, k=k)
            if len(scored) >= max(k * 12, 32):
                break

        for rank, rec in enumerate(self.base_index.candidates(target, "row_conditioned_knn", max(k * 3, k))):
            sid = str(rec["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            scored.append((4, gb.row_condition_distance(target, rec) + 1e-6 * rank, rec))

        scored.sort(key=lambda item: (item[0], item[1]))
        return without_self([rec for _, _, rec in scored])[: max(k, 1)]

    def source_quality_penalty(self, source: dict[str, Any]) -> float:
        penalty = 0.0
        if not bool(source.get("row_expansion_all_ok", True)):
            penalty += 2.0
        rows = v2.canonical_rows(source)
        free_rows = [row for row in rows if row.get("free_symbols")]
        if free_rows and not bool(source.get("free_param_reextract_all_success", True)):
            penalty += 0.7
        if free_rows:
            missing = 0
            for row in free_rows:
                params = dict(row.get("free_params") or {})
                symbols = {str(sym) for sym in row.get("free_symbols") or []}
                if not symbols.issubset(set(params)):
                    missing += 1
            penalty += 0.25 * float(missing) / max(1.0, float(len(free_rows)))
        return float(penalty)

    def row_aligned_quality_candidates(self, target: dict[str, Any], k: int) -> list[dict[str, Any]]:
        pool_limit = max(k * 10, 64)
        pool = self.base_index.candidates(target, "row_conditioned_knn", pool_limit)
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        seen: set[str] = set()
        for rec in pool:
            sid = str(rec["sample_id"])
            if sid in seen:
                continue
            seen.add(sid)
            row_dist = gb.row_condition_distance(target, rec)
            align_cost = self.source_alignment_cost(target, rec)
            quality_penalty = self.source_quality_penalty(rec)
            score = row_dist + 0.35 * align_cost + quality_penalty
            scored.append((score, row_dist, quality_penalty, rec))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [rec for _, _, _, rec in scored[: max(k, 1)]]

    def expected_vpa(self, target: dict[str, Any], k: int = 96) -> float | None:
        sg = int(target["sg"])
        atom_bucket = gb.atom_bucket(gb.atom_count(target))
        candidate_pools = [
            self.by_sg_atom_bucket.get((sg, atom_bucket), []),
            self.by_sg.get(sg, []),
            self.by_crystal_bucket.get((gb.crystal_system(sg), atom_bucket), []),
        ]
        nearest_values: list[float] = []
        for pool in candidate_pools:
            if not pool:
                continue
            reduced = pool
            if len(reduced) > int(k):
                reduced = sorted(reduced, key=lambda rec: self.cheap_chem_source_distance(target, rec))[: int(k)]
            nearest_values = [v for rec in reduced if (v := record_volume_per_atom(rec)) is not None]
            if len(nearest_values) >= 8:
                return float(quantile(nearest_values, 0.5))
        fallback_values = (
            self.vpa_by_sg_atom_bucket.get((sg, atom_bucket), [])
            or self.vpa_by_crystal_bucket.get((gb.crystal_system(sg), atom_bucket), [])
            or self.vpa_by_atom_bucket.get(atom_bucket, [])
            or self.global_vpa_values
        )
        if fallback_values:
            return float(quantile(list(fallback_values), 0.5))
        return None

    def vpa_calibrated_lattice(self, source: dict[str, Any], target: dict[str, Any], strength: float = 1.0) -> dict[str, float]:
        lattice = gb.source_lattice(source, target, "volume_scaled_source")
        source_vpa = record_volume_per_atom(source)
        target_vpa = self.expected_vpa(target)
        if source_vpa is None or target_vpa is None or source_vpa <= 0.0 or target_vpa <= 0.0:
            return lattice
        raw_factor = (float(target_vpa) / float(source_vpa)) ** (1.0 / 3.0)
        factor = 1.0 + float(strength) * (raw_factor - 1.0)
        factor = max(0.85, min(1.18, factor))
        out = dict(lattice)
        for key in ("a", "b", "c"):
            out[key] = max(0.5, float(out[key]) * factor)
        return out

    def cheap_chem_source_distance(self, target: dict[str, Any], source: dict[str, Any]) -> float:
        return float(
            0.20 * chem_distance(target, source)
            + 0.025 * abs(gb.atom_count(target) - gb.atom_count(source))
            + 0.075 * abs(gb.row_count(target) - gb.row_count(source))
            + (0.0 if int(target["sg"]) == int(source["sg"]) else 4.0)
        )

    def row_condition_chem_distance(self, target: dict[str, Any], source: dict[str, Any]) -> float:
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        dist = 0.0
        dist += 0.03 * abs(gb.atom_count(target) - gb.atom_count(source))
        dist += 0.20 * abs(len(t_rows) - len(s_rows))
        dist += 0.60 * chem_distance(target, source)
        dist += 0.10 * gb.formula_l1(gb.formula_frac(target), gb.formula_frac(source))
        if int(target["sg"]) != int(source["sg"]):
            dist += 4.0
        for idx in range(max(len(t_rows), len(s_rows))):
            if idx >= len(t_rows) or idx >= len(s_rows):
                dist += 0.7
                continue
            tr = t_rows[idx]
            sr = s_rows[idx]
            if str(tr.get("orbit_id")) != str(sr.get("orbit_id")):
                dist += 0.35
            dist += 0.08 * element_distance(str(tr.get("element")), str(sr.get("element")))
            if int(tr.get("multiplicity", 1)) != int(sr.get("multiplicity", 1)):
                dist += 0.20
            if set(tr.get("free_symbols") or []) != set(sr.get("free_symbols") or []):
                dist += 0.20
        return float(dist)

    def source_param_prior_penalty(self, target: dict[str, Any], source: dict[str, Any], chemical: bool = False) -> float:
        sg = int(target["sg"])
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        penalties: list[float] = []
        missing = 0
        for t_row in t_rows:
            best_idx: int | None = None
            best_cost = float("inf")
            for source_idx in unused:
                if chemical:
                    cost = self._row_match_cost_chem(t_row, s_rows[source_idx])
                else:
                    cost = self._row_match_cost(t_row, s_rows[source_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = source_idx
            source_params: dict[str, Any] = {}
            if best_idx is not None:
                unused.remove(best_idx)
                source_params = dict(s_rows[best_idx].get("free_params") or {})
            for symbol in t_row.get("free_symbols") or []:
                sym = str(symbol)
                if sym not in source_params:
                    missing += 1
                    continue
                penalties.append(self.param_manifold_penalty(sg, t_row, sym, float(source_params[sym])))
        if not penalties and missing <= 0:
            return 0.0
        total = sum(penalties) + 0.75 * float(missing)
        return float(total / max(1, len(penalties) + missing))

    def source_freepattern_compatibility_cost(self, target: dict[str, Any], source: dict[str, Any]) -> float:
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        total = 0.05 * abs(len(t_rows) - len(s_rows))
        missing_params = 0
        required_params = 0
        for t_row in t_rows:
            best_idx: int | None = None
            best_cost = float("inf")
            t_symbols = {str(sym) for sym in t_row.get("free_symbols") or []}
            for source_idx in unused:
                s_row = s_rows[source_idx]
                s_symbols = {str(sym) for sym in s_row.get("free_symbols") or []}
                cost = 0.0
                if str(t_row.get("orbit_id")) != str(s_row.get("orbit_id")):
                    cost += 1.0
                if t_symbols != s_symbols:
                    cost += 1.0
                if int(t_row.get("multiplicity", 1)) != int(s_row.get("multiplicity", 1)):
                    cost += 0.35
                if str(t_row.get("site_symmetry")) != str(s_row.get("site_symmetry")):
                    cost += 0.15
                cost += 0.05 * element_distance(str(t_row.get("element")), str(s_row.get("element")))
                if cost < best_cost:
                    best_cost = cost
                    best_idx = source_idx
            source_params: dict[str, Any] = {}
            if best_idx is None:
                total += 2.0
            else:
                unused.remove(best_idx)
                total += float(best_cost)
                source_params = dict(s_rows[best_idx].get("free_params") or {})
            for sym in t_row.get("free_symbols") or []:
                required_params += 1
                if str(sym) not in source_params:
                    missing_params += 1
        mean_cost = float(total) / max(1, len(t_rows))
        missing_rate = float(missing_params) / max(1, required_params)
        return float(mean_cost + 0.8 * missing_rate)

    def source_success_prior_penalty(self, target: dict[str, Any], source: dict[str, Any]) -> float:
        if not _SOURCE_SUCCESS_PRIOR:
            return 0.0
        sid = str(source.get("sample_id") or "")
        default = float(_SOURCE_SUCCESS_PRIOR.get("default_penalty", 0.0))
        if gb.row_count(target) >= 7:
            rows7 = dict(_SOURCE_SUCCESS_PRIOR.get("rows_ge_7_source_penalty") or {})
            if sid in rows7:
                return float(rows7[sid])
        return float(dict(_SOURCE_SUCCESS_PRIOR.get("source_penalty") or {}).get(sid, default))

    def row_aligned_chem_quality_candidates(self, target: dict[str, Any], k: int, use_param_penalty: bool = False, use_freepattern_penalty: bool = False, use_source_prior: bool = False) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        pool_limit = max(k * 80, 512)
        pools: list[list[dict[str, Any]]] = [
            self.by_sg_row.get((sg, gb.row_count(target)), []),
            self.by_sg_atom_bucket.get((sg, gb.atom_bucket(gb.atom_count(target))), []),
            self.by_sg.get(sg, []),
            self.by_crystal_bucket.get((gb.crystal_system(sg), gb.atom_bucket(gb.atom_count(target))), []),
            self.train_records,
        ]
        pool: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_pool in pools:
            if not source_pool:
                continue
            reduced = source_pool
            if len(reduced) > pool_limit:
                reduced = sorted(reduced, key=lambda rec: self.cheap_chem_source_distance(target, rec))[:pool_limit]
            for rec in reduced:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= max(k * 24, 128):
                break
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        for rec in pool:
            chem_row_dist = self.row_condition_chem_distance(target, rec)
            align_cost = self.source_alignment_cost(target, rec, chemical=True)
            quality_penalty = self.source_quality_penalty(rec)
            score = chem_row_dist + 0.35 * align_cost + quality_penalty
            if use_freepattern_penalty:
                score += 0.80 * self.source_freepattern_compatibility_cost(target, rec)
            if use_source_prior:
                score += float(_SOURCE_PRIOR_WEIGHT) * self.source_success_prior_penalty(target, rec)
            scored.append((score, chem_row_dist, quality_penalty, rec))
        if use_param_penalty:
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            refine_limit = max(k * 10, 24)
            refined: list[tuple[float, float, float, dict[str, Any]]] = []
            for score, chem_row_dist, quality_penalty, rec in scored[:refine_limit]:
                param_penalty = self.source_param_prior_penalty(target, rec, chemical=True)
                refined.append((score + 0.80 * param_penalty, chem_row_dist, quality_penalty + param_penalty, rec))
            if len(scored) > refine_limit:
                refined.extend(scored[refine_limit:])
            scored = refined
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [rec for _, _, _, rec in scored[: max(k, 1)]]

    def source_row_pair_feasibility_mean(self, target: dict[str, Any], source: dict[str, Any], chemical: bool = True) -> tuple[float, float]:
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        if not t_rows or not s_rows:
            return 0.0, 0.0
        total = 0.0
        hits = 0
        for t_row in t_rows:
            best_idx: int | None = None
            best_adjusted = float("inf")
            best_feasible = 0.0
            for source_idx in unused:
                s_row = s_rows[source_idx]
                cost = self._row_match_cost_chem(t_row, s_row) if chemical else self._row_match_cost(t_row, s_row)
                feasible = max(-1.5, min(1.5, self.row_pair_feasibility_score(t_row, s_row)))
                adjusted = float(cost) - 0.35 * float(feasible)
                if adjusted < best_adjusted:
                    best_idx = int(source_idx)
                    best_adjusted = float(adjusted)
                    best_feasible = float(feasible)
            if best_idx is None:
                continue
            unused.remove(best_idx)
            total += best_feasible
            if abs(best_feasible) > 1.0e-9:
                hits += 1
        denom = max(1, len(t_rows))
        return float(total / float(denom)), float(hits / float(denom))

    def row_pair_source_feasible_quality_candidates(self, target: dict[str, Any], k: int, use_freepattern_penalty: bool = False) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        target_rows = gb.row_count(target)
        pool_limit = max(k * 160, 768)
        pools: list[list[dict[str, Any]]] = [
            self.by_sg_row.get((sg, target_rows), []),
            self.by_sg_atom_bucket.get((sg, gb.atom_bucket(gb.atom_count(target))), []),
            self.by_sg.get(sg, []),
            self.by_crystal_bucket.get((gb.crystal_system(sg), gb.atom_bucket(gb.atom_count(target))), []),
            self.train_records,
        ]
        pool: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_pool in pools:
            if not source_pool:
                continue
            reduced = source_pool
            if len(reduced) > pool_limit:
                reduced = sorted(reduced, key=lambda rec: self.cheap_chem_source_distance(target, rec))[:pool_limit]
            for rec in reduced:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                pool.append(rec)
            if len(pool) >= max(k * 96, 512):
                break
        feasibility_weight = 0.60 if int(target_rows) >= 7 else 0.32
        scored: list[tuple[float, float, float, float, dict[str, Any]]] = []
        for rec in pool:
            chem_row_dist = self.row_condition_chem_distance(target, rec)
            align_cost = self.source_alignment_cost(target, rec, chemical=True)
            quality_penalty = self.source_quality_penalty(rec)
            feasible_mean, feasible_hit_rate = self.source_row_pair_feasibility_mean(target, rec, chemical=True)
            score = chem_row_dist + 0.35 * align_cost + quality_penalty - feasibility_weight * feasible_mean - 0.05 * feasible_hit_rate
            if use_freepattern_penalty:
                score += 0.45 * self.source_freepattern_compatibility_cost(target, rec)
            scored.append((score, chem_row_dist, align_cost, -feasible_mean, rec))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [rec for _, _, _, _, rec in scored[: max(k, 1)]]

    def row_aligned_candidates(self, target: dict[str, Any], k: int) -> list[dict[str, Any]]:
        sg = int(target["sg"])
        skel, wa = record_keys(target)
        pool_limit = max(k * 8, 32)
        pools: list[list[dict[str, Any]]] = [
            self.by_wa.get((sg, wa), []),
            self.by_skeleton.get((sg, skel), []),
            self.by_signature.get((sg, gb.free_signature(target)), []),
            self.base_index.candidates(target, "row_conditioned_knn", pool_limit),
        ]
        seen: set[str] = set()
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for pool in pools:
            if not pool:
                continue
            for rec in pool[:pool_limit]:
                sid = str(rec["sample_id"])
                if sid in seen:
                    continue
                seen.add(sid)
                align_cost = self.source_alignment_cost(target, rec)
                row_dist = gb.row_condition_distance(target, rec)
                # Alignment matters most; row distance keeps source/lattice scale plausible.
                score = 1.0 * align_cost + 0.25 * row_dist
                scored.append((score, row_dist, rec))
            if len(scored) >= max(k * 6, 24):
                break
        scored.sort(key=lambda item: (item[0], item[1]))
        return [rec for _, _, rec in scored[: max(k, 1)]]

    def _row_prototype_pool(self, target: dict[str, Any], row: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        sg = int(target["sg"])
        oid = str(row["orbit_id"])
        element = str(row["element"])
        symbols = self._symbols_key(row)
        pools = [
            self.row_by_exact.get((sg, oid, element, symbols), []),
            self.row_by_sg_orbit_symbols.get((sg, oid, symbols), []),
            self.row_by_orbit_element_symbols.get((oid, element, symbols), []),
            self.row_by_orbit_symbols.get((oid, symbols), []),
        ]
        seen: set[tuple[str, int]] = set()
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for pool in pools:
            if not pool:
                continue
            for rec, proto_row in pool[:256]:
                key = (str(rec["sample_id"]), id(proto_row))
                if key in seen:
                    continue
                seen.add(key)
                out.append((rec, proto_row))
            if len(out) >= 64:
                break
        return out

    def row_prototype_params(self, target: dict[str, Any], rank: int) -> tuple[dict[int, dict[str, float]], str | None, float | None]:
        params: dict[int, dict[str, float]] = {}
        source_ids: list[str] = []
        distances: list[float] = []
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        for row_idx, row in enumerate(v2.canonical_rows(target)):
            row_params: dict[str, float] = {}
            pool = self._row_prototype_pool(target, row)
            selected: tuple[dict[str, Any], dict[str, Any]] | None = None
            if pool:
                # Offset by row index to avoid copying all row parameters from one source mode.
                selected = pool[(int(rank) + row_idx * 3) % len(pool)]
            proto_params = dict(selected[1].get("free_params") or {}) if selected is not None else {}
            if selected is not None:
                source_ids.append(str(selected[0].get("sample_id")))
                distances.append(float(gb.row_condition_distance(target, selected[0])))
            for symbol in row.get("free_symbols") or []:
                sym = str(symbol)
                if sym in proto_params:
                    row_params[sym] = float(proto_params[sym]) % 1.0
                else:
                    row_params[sym] = self.base_index.free_value(str(row["orbit_id"]), sym, q)
            params[row_idx] = row_params
        source_summary = None
        if source_ids:
            source_summary = ";".join(source_ids[:5])
            if len(source_ids) > 5:
                source_summary += f";+{len(source_ids) - 5}"
        mean_distance = None if not distances else float(sum(distances) / len(distances))
        return params, source_summary, mean_distance

    def _row_match_cost(self, target_row: dict[str, Any], source_row: dict[str, Any]) -> float:
        cost = 0.0
        if str(target_row.get("orbit_id")) != str(source_row.get("orbit_id")):
            cost += 1.50
        if str(target_row.get("element")) != str(source_row.get("element")):
            cost += 0.70
        if self._symbols_key(target_row) != self._symbols_key(source_row):
            cost += 0.50
        if int(target_row.get("multiplicity", 1)) != int(source_row.get("multiplicity", 1)):
            cost += 0.30
        if str(target_row.get("site_symmetry")) != str(source_row.get("site_symmetry")):
            cost += 0.10
        return float(cost)

    def _row_match_cost_chem(self, target_row: dict[str, Any], source_row: dict[str, Any]) -> float:
        cost = 0.0
        if str(target_row.get("orbit_id")) != str(source_row.get("orbit_id")):
            cost += 1.50
        cost += 0.20 * element_distance(str(target_row.get("element")), str(source_row.get("element")))
        if self._symbols_key(target_row) != self._symbols_key(source_row):
            cost += 0.50
        if int(target_row.get("multiplicity", 1)) != int(source_row.get("multiplicity", 1)):
            cost += 0.30
        if str(target_row.get("site_symmetry")) != str(source_row.get("site_symmetry")):
            cost += 0.10
        return float(cost)

    def _row_pair_feasibility_keys(self, target_row: dict[str, Any], source_row: dict[str, Any]) -> list[tuple[Any, ...]]:
        toid = str(target_row.get("orbit_id"))
        telem = str(target_row.get("element"))
        tsym = self._symbols_key(target_row)
        soid = str(source_row.get("orbit_id"))
        selem = str(source_row.get("element"))
        ssym = self._symbols_key(source_row)
        return [
            ("target_source_orbit_element_symbols", toid, telem, tsym, soid, selem, ssym),
            ("target_source_orbit_symbols", toid, tsym, soid, ssym),
            ("target_orbit_source_orbit", toid, soid),
            ("compat_flags", int(toid == soid), int(telem == selem), int(tsym == ssym)),
            ("target_orbit", toid),
        ]

    def row_pair_feasibility_score(self, target_row: dict[str, Any], source_row: dict[str, Any]) -> float:
        for key in self._row_pair_feasibility_keys(target_row, source_row):
            if key in self.row_pair_feasibility_by_key:
                return float(self.row_pair_feasibility_by_key[key])
        return 0.0

    def source_alignment_cost(self, target: dict[str, Any], source: dict[str, Any], chemical: bool = False) -> float:
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        total_cost = 0.05 * abs(len(t_rows) - len(s_rows))
        for t_row in t_rows:
            best_idx: int | None = None
            best_cost = float("inf")
            for source_idx in unused:
                if chemical:
                    cost = self._row_match_cost_chem(t_row, s_rows[source_idx])
                else:
                    cost = self._row_match_cost(t_row, s_rows[source_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = source_idx
            if best_idx is not None:
                unused.remove(best_idx)
                total_cost += float(best_cost)
            else:
                total_cost += 2.0
        return float(total_cost) / max(1, len(t_rows))

    def source_aligned_params(self, target: dict[str, Any], source: dict[str, Any], rank: int, chemical: bool = False) -> tuple[dict[int, dict[str, float]], float]:
        params: dict[int, dict[str, float]] = {}
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        total_cost = 0.0
        for row_idx, t_row in enumerate(t_rows):
            best_idx: int | None = None
            best_cost = float("inf")
            for source_idx in unused:
                if chemical:
                    cost = self._row_match_cost_chem(t_row, s_rows[source_idx])
                else:
                    cost = self._row_match_cost(t_row, s_rows[source_idx])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = source_idx
            source_params: dict[str, Any] = {}
            if best_idx is not None:
                unused.remove(best_idx)
                source_params = dict(s_rows[best_idx].get("free_params") or {})
                total_cost += float(best_cost)
            else:
                total_cost += 2.0
            row_params: dict[str, float] = {}
            for symbol in t_row.get("free_symbols") or []:
                sym = str(symbol)
                if sym in source_params:
                    row_params[sym] = float(source_params[sym]) % 1.0
                else:
                    row_params[sym] = self.base_index.free_value(str(t_row["orbit_id"]), sym, q)
            params[row_idx] = row_params
        mean_cost = total_cost / max(1, len(t_rows))
        return params, float(mean_cost)

    def source_aligned_params_row_shift(self, target: dict[str, Any], source: dict[str, Any], rank: int, chemical: bool = False) -> tuple[dict[int, dict[str, float]], float]:
        params: dict[int, dict[str, float]] = {}
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        total_cost = 0.0
        for row_idx, t_row in enumerate(t_rows):
            scored: list[tuple[float, int]] = []
            for source_idx in unused:
                if chemical:
                    cost = self._row_match_cost_chem(t_row, s_rows[source_idx])
                else:
                    cost = self._row_match_cost(t_row, s_rows[source_idx])
                scored.append((float(cost), int(source_idx)))
            scored.sort(key=lambda item: (item[0], item[1]))
            best_idx: int | None = None
            best_cost = float("inf")
            if scored:
                best_cost = float(scored[0][0])
                near = [item for item in scored if item[0] <= best_cost + 0.21][:5]
                if not near:
                    near = scored[:1]
                choice = (int(rank) + int(row_idx)) % len(near)
                best_cost, best_idx = near[choice]
            source_params: dict[str, Any] = {}
            if best_idx is not None:
                unused.remove(best_idx)
                source_params = dict(s_rows[best_idx].get("free_params") or {})
                total_cost += float(best_cost)
            else:
                total_cost += 2.0
            row_params: dict[str, float] = {}
            for symbol in t_row.get("free_symbols") or []:
                sym = str(symbol)
                if sym in source_params:
                    row_params[sym] = float(source_params[sym]) % 1.0
                else:
                    row_params[sym] = self.base_index.free_value(str(t_row["orbit_id"]), sym, q)
            params[row_idx] = row_params
        mean_cost = total_cost / max(1, len(t_rows))
        return params, float(mean_cost)

    def source_aligned_params_row_pair_feasible(self, target: dict[str, Any], source: dict[str, Any], rank: int, chemical: bool = False) -> tuple[dict[int, dict[str, float]], float]:
        params: dict[int, dict[str, float]] = {}
        t_rows = v2.canonical_rows(target)
        s_rows = v2.canonical_rows(source)
        unused = set(range(len(s_rows)))
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        weight_by_rank = [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.40, 1.60]
        weight = float(weight_by_rank[min(max(0, int(rank)), len(weight_by_rank) - 1)])
        total_cost = 0.0
        for row_idx, t_row in enumerate(t_rows):
            scored: list[tuple[float, float, float, int]] = []
            for source_idx in unused:
                s_row = s_rows[source_idx]
                if chemical:
                    cost = self._row_match_cost_chem(t_row, s_row)
                else:
                    cost = self._row_match_cost(t_row, s_row)
                feasible = max(-1.5, min(1.5, self.row_pair_feasibility_score(t_row, s_row)))
                adjusted = float(cost) - weight * float(feasible)
                scored.append((adjusted, float(cost), -float(feasible), int(source_idx)))
            scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            best_idx: int | None = None
            best_cost = float("inf")
            if scored:
                _adjusted, best_cost, _neg_feasible, best_idx = scored[0]
            source_params: dict[str, Any] = {}
            if best_idx is not None:
                unused.remove(best_idx)
                source_params = dict(s_rows[best_idx].get("free_params") or {})
                total_cost += float(best_cost)
            else:
                total_cost += 2.0
            row_params: dict[str, float] = {}
            for symbol in t_row.get("free_symbols") or []:
                sym = str(symbol)
                if sym in source_params:
                    row_params[sym] = float(source_params[sym]) % 1.0
                else:
                    row_params[sym] = self.base_index.free_value(str(t_row["orbit_id"]), sym, q)
            params[int(row_idx)] = row_params
        mean_cost = total_cost / max(1, len(t_rows))
        return params, float(mean_cost)

    def source_cluster_lattice(self, target: dict[str, Any], strategy: str, rank: int, k: int = 24) -> dict[str, float]:
        source_strategy = "row_conditioned_knn" if strategy == "row_prototype" else strategy
        sources = self.candidates(target, source_strategy, max(1, int(k)))
        if not sources:
            return self.base_index.lattice_quantile(target, 0.5)
        q_by_rank = [0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.65, 0.2, 0.8, 0.05]
        q = q_by_rank[min(max(0, int(rank)), len(q_by_rank) - 1)]
        scaled = [gb.source_lattice(source, target, "volume_scaled_source") for source in sources]
        values = [
            float(gb.quantile([float(lat["a"]) for lat in scaled], q)),
            float(gb.quantile([float(lat["b"]) for lat in scaled], q)),
            float(gb.quantile([float(lat["c"]) for lat in scaled], q)),
            float(gb.quantile([float(lat["alpha"]) for lat in scaled], q)),
            float(gb.quantile([float(lat["beta"]) for lat in scaled], q)),
            float(gb.quantile([float(lat["gamma"]) for lat in scaled], q)),
        ]
        transformed = [torch.log(torch.tensor(max(values[0], 0.5))).item(), torch.log(torch.tensor(max(values[1], 0.5))).item(), torch.log(torch.tensor(max(values[2], 0.5))).item(), values[3] / 180.0, values[4] / 180.0, values[5] / 180.0]
        return v2.lattice_from_target([float(x) for x in transformed], int(target["sg"]))


def load_geometry_net_lattice(ckpt_path: str | None) -> None:
    global _LATTICE_MODEL, _LATTICE_VOCABS, _LATTICE_MEAN, _LATTICE_STD, _LATTICE_CACHE
    _LATTICE_CACHE = {}
    if not ckpt_path:
        _LATTICE_MODEL = None
        _LATTICE_VOCABS = {}
        _LATTICE_MEAN = None
        _LATTICE_STD = None
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _LATTICE_VOCABS = ckpt["vocabs"]
    config = ckpt.get("config") or {}
    _LATTICE_MODEL = geom.GeometryNet(
        {name: len(vocab) for name, vocab in _LATTICE_VOCABS.items()},
        hidden_dim=int(config.get("hidden_dim", 256)),
        emb_dim=int(config.get("emb_dim", 64)),
    )
    _LATTICE_MODEL.load_state_dict(ckpt["model_state"])
    _LATTICE_MODEL.eval()
    _LATTICE_MEAN = torch.tensor(ckpt["lattice_mean"], dtype=torch.float32)
    _LATTICE_STD = torch.tensor(ckpt["lattice_std"], dtype=torch.float32)


def lattice_cache_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("sample_id") or ""), str(record.get("canonical_wa_key") or record_keys(record)[1])


@torch.no_grad()
def predict_geometry_net_lattice(record: dict[str, Any]) -> dict[str, float]:
    if _LATTICE_MODEL is None or _LATTICE_MEAN is None or _LATTICE_STD is None:
        raise RuntimeError("geometry_net lattice mode requested without loaded checkpoint")
    key = lattice_cache_key(record)
    cached = _LATTICE_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    inference_record = dict(record)
    inference_record["lattice"] = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    batch = geom.collate_geometry([inference_record], vocabs=_LATTICE_VOCABS, lattice_mean=_LATTICE_MEAN, lattice_std=_LATTICE_STD)
    lattice_raw, _coords = _LATTICE_MODEL(batch)
    values = (lattice_raw[0].detach().cpu() * _LATTICE_STD.cpu() + _LATTICE_MEAN.cpu()).tolist()
    lattice = v2.lattice_from_target([float(x) for x in values], int(record["sg"]))
    _LATTICE_CACHE[key] = dict(lattice)
    return lattice


@torch.no_grad()
def populate_geometry_net_lattice_cache(record: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    if _GEOMETRY_LATTICE_MODE != "geometry_net" or _LATTICE_MODEL is None or _LATTICE_MEAN is None or _LATTICE_STD is None:
        return
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    max_wa = min(len(candidates), max(_TOP_K, 1))
    inference_records: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates[:max_wa]:
        rows = normalize_rows(_ENGINE, list(candidate.get("rows") or []))
        pred_record = pseudo_record(record, rows)
        key = lattice_cache_key(pred_record)
        if key in seen or key in _LATTICE_CACHE:
            continue
        seen.add(key)
        inference_record = dict(pred_record)
        inference_record["lattice"] = {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
        inference_records.append(inference_record)
        keys.append(key)
    if not inference_records:
        return
    batch = geom.collate_geometry(inference_records, vocabs=_LATTICE_VOCABS, lattice_mean=_LATTICE_MEAN, lattice_std=_LATTICE_STD)
    lattice_raw, _coords = _LATTICE_MODEL(batch)
    decoded = lattice_raw.detach().cpu() * _LATTICE_STD.cpu().unsqueeze(0) + _LATTICE_MEAN.cpu().unsqueeze(0)
    for key, values, rec in zip(keys, decoded.tolist(), inference_records):
        _LATTICE_CACHE[key] = v2.lattice_from_target([float(x) for x in values], int(rec["sg"]))


def variant_offset_and_scale(rank: int) -> tuple[float, float]:
    offsets = [0.0, 0.0125, -0.0125, 0.025, -0.025, 0.05, -0.05, 0.075, -0.075, 0.10]
    scales = [1.0, 0.9925, 1.0075, 0.985, 1.015, 0.970, 1.030, 0.955, 1.045, 1.060]
    idx = min(max(0, int(rank)), len(offsets) - 1)
    return float(offsets[idx]), float(scales[idx])


def apply_param_lattice_variant(
    params: dict[int, dict[str, float]],
    lattice: dict[str, float],
    *,
    rank: int,
    mode: str,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    if mode in {"manifold_params", "manifold_params_lattice", "positive_params", "positive_params_lattice", "positive_nearest_complex", "row_feasibility_complex", "row_pair_feasible", "positive_bundle_exact", "positive_bundle_broad"}:
        return params, lattice
    if mode == "none" or int(rank) <= 0:
        return params, lattice
    offset, scale = variant_offset_and_scale(int(rank))
    out_params: dict[int, dict[str, float]] = {}
    for row_idx, row_params in params.items():
        out_params[int(row_idx)] = {}
        for sym, value in row_params.items():
            sign = -1.0 if ((int(row_idx) + ord(str(sym)[0]) + int(rank)) % 2) else 1.0
            out_params[int(row_idx)][str(sym)] = (float(value) + sign * offset) % 1.0
    out_lattice = dict(lattice)
    if mode == "wrapped_jitter_lattice":
        for key in ("a", "b", "c"):
            out_lattice[key] = max(0.5, float(out_lattice[key]) * scale)
    return out_params, out_lattice


def cif_min_distance_and_vpa(cif: str) -> tuple[float | None, float | None]:
    if Structure is None or not cif:
        return None, None
    try:
        structure = Structure.from_str(cif, fmt="cif", primitive=False, merge_tol=0.0)
        sites = int(len(structure))
        vpa = None if sites <= 0 else float(structure.volume) / float(sites)
        min_distance = None
        if sites > 1:
            matrix = structure.distance_matrix
            vals = []
            for i in range(sites):
                for j in range(i + 1, sites):
                    vals.append(float(matrix[i][j]))
            min_distance = min(vals) if vals else None
        return min_distance, vpa
    except Exception:
        return None, None


def generated_physical_source_score(metric: dict[str, Any], min_distance: float | None, vpa: float | None, geom_distance: float | None) -> float:
    score = 0.0
    for key, weight in (
        ("readable", 1000.0),
        ("formula_ok", 1000.0),
        ("atom_count_ok", 1000.0),
        ("composition_exact", 1000.0),
        ("sg_ok", 1000.0),
    ):
        score += weight if bool(metric.get(key)) else -weight
    if min_distance is None:
        score -= 30.0
    else:
        d = float(min_distance)
        if d < 0.75:
            score -= 180.0
        elif d < 1.05:
            score -= 60.0
        elif d < 1.35:
            score -= 10.0
        elif d <= 2.80:
            score += 18.0
        elif d <= 4.50:
            score += 4.0
        else:
            score -= 20.0
    if vpa is None:
        score -= 8.0
    else:
        x = float(vpa)
        if x < 8.0:
            score -= 35.0
        elif x < 12.0:
            score -= 8.0
        elif x <= 35.0:
            score += 8.0
        elif x <= 70.0:
            score -= 4.0
        else:
            score -= 25.0
    if geom_distance is not None:
        score -= 1.5 * float(geom_distance)
    return float(score)


def init_worker(
    lookup_json: str,
    data_root: str,
    top_k: int,
    geometry_mode: str,
    geometry_ranks_per_wa: int,
    geometry_plan_mode: str,
    geometry_source_strategy: str,
    geometry_lattice_mode: str,
    geometry_lattice_ckpt: str,
    geometry_param_variant_mode: str,
    hybrid_geometry_wa: int,
    hybrid_geometry_ranks: int,
    require_composition_exact: bool,
    exclude_self_source: bool,
    source_prior_json: str,
    source_prior_weight: float,
    source_bank_json: str,
    source_proposal_json: str,
    write_geometry_metadata: bool,
    train_records: list[dict[str, Any]],
    split_records: list[dict[str, Any]],
) -> None:
    global _ENGINE, _GEOM_INDEX, _TOP_K, _GEOMETRY_MODE, _GEOMETRY_RANKS_PER_WA, _GEOMETRY_PLAN_MODE, _GEOMETRY_SOURCE_STRATEGY, _GEOMETRY_LATTICE_MODE, _GEOMETRY_PARAM_VARIANT_MODE, _HYBRID_GEOMETRY_WA, _HYBRID_GEOMETRY_RANKS, _REQUIRE_COMPOSITION_EXACT, _EXCLUDE_SELF_SOURCE, _SOURCE_SUCCESS_PRIOR, _SOURCE_PRIOR_WEIGHT, _SOURCE_TEMPLATE_BANK, _SOURCE_PROPOSAL_CACHE, _GEOM_SELECTOR, _WRITE_GEOMETRY_METADATA
    sg_symbols = v2.sg_symbols_from_splits({"train": train_records, "eval": split_records})
    _ENGINE = OrbitEngine(lookup_json, sg_symbols)
    _GEOM_INDEX = gb.GeometryIndex(train_records, [])
    _TOP_K = int(top_k)
    _GEOMETRY_MODE = str(geometry_mode)
    _GEOMETRY_RANKS_PER_WA = max(1, int(geometry_ranks_per_wa))
    _GEOMETRY_PLAN_MODE = str(geometry_plan_mode)
    _GEOMETRY_SOURCE_STRATEGY = str(geometry_source_strategy)
    _GEOMETRY_LATTICE_MODE = str(geometry_lattice_mode)
    _GEOMETRY_PARAM_VARIANT_MODE = str(geometry_param_variant_mode)
    load_geometry_net_lattice(geometry_lattice_ckpt or None)
    _HYBRID_GEOMETRY_WA = max(0, int(hybrid_geometry_wa))
    _HYBRID_GEOMETRY_RANKS = max(1, int(hybrid_geometry_ranks))
    _REQUIRE_COMPOSITION_EXACT = bool(require_composition_exact)
    _EXCLUDE_SELF_SOURCE = bool(exclude_self_source)
    _WRITE_GEOMETRY_METADATA = bool(write_geometry_metadata)
    _SOURCE_PRIOR_WEIGHT = float(source_prior_weight)
    _SOURCE_SUCCESS_PRIOR = {}
    if source_prior_json:
        prior_path = Path(source_prior_json).resolve()
        if OPENTRY_ROOT not in (prior_path, *prior_path.parents):
            raise RuntimeError(f"Refusing source prior outside opentry_3: {prior_path}")
        _SOURCE_SUCCESS_PRIOR = json.loads(prior_path.read_text(encoding="utf-8"))
    _SOURCE_TEMPLATE_BANK = {}
    if source_bank_json:
        bank_path = Path(source_bank_json).resolve()
        if OPENTRY_ROOT not in (bank_path, *bank_path.parents):
            raise RuntimeError(f"Refusing source bank outside opentry_3: {bank_path}")
        _SOURCE_TEMPLATE_BANK = json.loads(bank_path.read_text(encoding="utf-8"))
    _SOURCE_PROPOSAL_CACHE = {}
    if source_proposal_json:
        proposal_path = Path(source_proposal_json).resolve()
        if OPENTRY_ROOT not in (proposal_path, *proposal_path.parents):
            raise RuntimeError(f"Refusing source proposal cache outside opentry_3: {proposal_path}")
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        for entry in payload.get("entries") or []:
            sid = str(entry.get("sample_id") or "")
            wa = str(entry.get("canonical_wa_key") or "")
            sources = [str(item) for item in (entry.get("source_sample_ids") or []) if str(item)]
            if sid and wa and sources:
                _SOURCE_PROPOSAL_CACHE[(sid, wa)] = sources
    _GEOM_SELECTOR = OpentryGeometrySelector(train_records, _GEOM_INDEX, _SOURCE_TEMPLATE_BANK)


def render_from_source(pred_record: dict[str, Any], source: dict[str, Any], geometry_rank: int, source_name: str) -> tuple[dict[str, Any], dict[str, float], str | None, float | None, dict[int, dict[str, float]] | None]:
    if _ENGINE is None or _GEOM_INDEX is None or _GEOM_SELECTOR is None:
        raise RuntimeError("worker not initialized")
    if _GEOMETRY_LATTICE_MODE == "geometry_net":
        lattice = predict_geometry_net_lattice(pred_record)
    elif _GEOMETRY_LATTICE_MODE == "source_cluster_quantile":
        lattice = _GEOM_SELECTOR.source_cluster_lattice(pred_record, _GEOMETRY_SOURCE_STRATEGY, geometry_rank)
    elif _GEOMETRY_LATTICE_MODE == "source_vpa_calibrated":
        lattice = _GEOM_SELECTOR.vpa_calibrated_lattice(source, pred_record, strength=1.0)
    elif _GEOMETRY_LATTICE_MODE == "source_vpa_calibrated_soft":
        lattice = _GEOM_SELECTOR.vpa_calibrated_lattice(source, pred_record, strength=0.5)
    else:
        lattice = gb.source_lattice(source, pred_record, "volume_scaled_source")
    if _GEOMETRY_SOURCE_STRATEGY == "row_prototype":
        params, row_source_summary, row_distance = _GEOM_SELECTOR.row_prototype_params(pred_record, geometry_rank)
        source_sample_id = row_source_summary or str(source.get("sample_id"))
        geom_distance = row_distance if row_distance is not None else gb.row_condition_distance(pred_record, source)
    elif _GEOMETRY_SOURCE_STRATEGY in {"row_aligned_knn", "row_aligned_priority", "row_aligned_quality", "row_aligned_chem_quality", "row_aligned_chem_freepattern_quality", "row_aligned_chem_param_quality", "row_aligned_chem_physical_select", "row_aligned_chem_sourceprior_quality", "row_aligned_chem_sourceprior_freepattern_quality", "row_aligned_chem_successbank_quality", "row_aligned_chem_successbank_freepattern_quality", "row_aligned_chem_successbank_tail_quality", "row_aligned_chem_successbank_tail_freepattern_quality", "row_aligned_chem_sourceproposal_quality", "row_aligned_chem_sourceproposal_tail_quality", "row_aligned_chem_rowbank_quality", "row_aligned_chem_rowbank_freepattern_quality", "row_pair_source_feasible_quality", "row_pair_source_feasible_freepattern_quality"}:
        use_chemical = _GEOMETRY_SOURCE_STRATEGY in {"row_aligned_chem_quality", "row_aligned_chem_freepattern_quality", "row_aligned_chem_param_quality", "row_aligned_chem_physical_select", "row_aligned_chem_sourceprior_quality", "row_aligned_chem_sourceprior_freepattern_quality", "row_aligned_chem_successbank_quality", "row_aligned_chem_successbank_freepattern_quality", "row_aligned_chem_successbank_tail_quality", "row_aligned_chem_successbank_tail_freepattern_quality", "row_aligned_chem_sourceproposal_quality", "row_aligned_chem_sourceproposal_tail_quality", "row_aligned_chem_rowbank_quality", "row_aligned_chem_rowbank_freepattern_quality", "row_pair_source_feasible_quality", "row_pair_source_feasible_freepattern_quality"}
        if _GEOMETRY_SOURCE_STRATEGY in {"row_aligned_chem_rowbank_quality", "row_aligned_chem_rowbank_freepattern_quality"}:
            params, row_source_summary, row_distance = _GEOM_SELECTOR.row_success_template_params(pred_record, geometry_rank)
            source_sample_id = row_source_summary or str(source.get("sample_id"))
            geom_distance = row_distance if row_distance is not None else gb.row_condition_distance(pred_record, source)
        else:
            if _GEOMETRY_PARAM_VARIANT_MODE == "source_row_shift":
                params, align_cost = _GEOM_SELECTOR.source_aligned_params_row_shift(pred_record, source, geometry_rank, chemical=use_chemical)
            elif _GEOMETRY_PARAM_VARIANT_MODE == "row_pair_feasible":
                params, align_cost = _GEOM_SELECTOR.source_aligned_params_row_pair_feasible(pred_record, source, geometry_rank, chemical=use_chemical)
            else:
                params, align_cost = _GEOM_SELECTOR.source_aligned_params(pred_record, source, geometry_rank, chemical=use_chemical)
            source_sample_id = str(source.get("sample_id"))
            geom_distance = gb.row_condition_distance(pred_record, source) + 0.1 * float(align_cost)
    else:
        params = gb.params_from_source(_GEOM_INDEX, pred_record, source, geometry_rank)
        source_sample_id = str(source.get("sample_id"))
        geom_distance = gb.row_condition_distance(pred_record, source)
    if _GEOMETRY_PARAM_VARIANT_MODE in {"manifold_params", "manifold_params_lattice"}:
        params = _GEOM_SELECTOR.manifold_param_variant(pred_record, params, geometry_rank)
        if _GEOMETRY_PARAM_VARIANT_MODE == "manifold_params_lattice":
            lattice = _GEOM_SELECTOR.manifold_lattice_variant(pred_record, lattice, geometry_rank)
    if _GEOMETRY_PARAM_VARIANT_MODE in {"positive_params", "positive_params_lattice"}:
        params = _GEOM_SELECTOR.positive_param_variant(pred_record, params, geometry_rank)
        if _GEOMETRY_PARAM_VARIANT_MODE == "positive_params_lattice":
            lattice = _GEOM_SELECTOR.positive_lattice_variant(pred_record, lattice, geometry_rank)
    if _GEOMETRY_PARAM_VARIANT_MODE == "positive_nearest_complex":
        params = _GEOM_SELECTOR.positive_nearest_complex_variant(pred_record, params, geometry_rank)
    if _GEOMETRY_PARAM_VARIANT_MODE == "row_feasibility_complex":
        params = _GEOM_SELECTOR.row_feasibility_complex_variant(pred_record, params, geometry_rank)
    if _GEOMETRY_PARAM_VARIANT_MODE == "positive_bundle_exact":
        params, lattice = _GEOM_SELECTOR.geometry_bundle_variant(pred_record, params, lattice, geometry_rank, broad=False)
    if _GEOMETRY_PARAM_VARIANT_MODE == "positive_bundle_broad":
        params, lattice = _GEOM_SELECTOR.geometry_bundle_variant(pred_record, params, lattice, geometry_rank, broad=True)
    params, lattice = apply_param_lattice_variant(params, lattice, rank=geometry_rank, mode=_GEOMETRY_PARAM_VARIANT_MODE)
    candidate = {
        "rows": v2.canonical_rows(pred_record),
        "params": params,
        "lattice": lattice,
    }
    rendered = v2.render_candidate(_ENGINE, pred_record, candidate, geometry_rank, source_name)
    return rendered, lattice, source_sample_id, geom_distance, params


def render_with_geometry(target_record: dict[str, Any], rows: list[dict[str, Any]], geometry_rank: int, source_name: str) -> tuple[dict[str, Any], dict[str, float] | None, str | None, float | None, dict[int, dict[str, float]] | None]:
    if _ENGINE is None or _GEOM_INDEX is None or _GEOM_SELECTOR is None:
        raise RuntimeError("worker not initialized")
    pred_record = pseudo_record(target_record, rows)
    if _GEOMETRY_SOURCE_STRATEGY == "row_prototype":
        source_strategy = "row_conditioned_knn"
    elif _GEOMETRY_SOURCE_STRATEGY == "row_aligned_chem_physical_select":
        source_strategy = "row_aligned_chem_quality"
    else:
        source_strategy = _GEOMETRY_SOURCE_STRATEGY
    source_count = max(geometry_rank + 1, 1)
    if _GEOMETRY_SOURCE_STRATEGY == "row_aligned_chem_physical_select" and int(geometry_rank) == 0:
        source_count = max(source_count, int(_PHYSICAL_SOURCE_SELECT_TOP_N))
    sources = _GEOM_SELECTOR.candidates(pred_record, source_strategy, source_count)
    if not sources:
        rendered, lattice = gb.render_quantile_candidate(engine=_ENGINE, index=_GEOM_INDEX, record=pred_record, rank=geometry_rank, source_name=f"{source_name}_quantile")
        return rendered, lattice, None, None, None

    if _GEOMETRY_SOURCE_STRATEGY == "row_aligned_chem_physical_select" and int(geometry_rank) == 0:
        scored: list[tuple[float, int, dict[str, Any], dict[str, float], str | None, float | None, dict[int, dict[str, float]] | None]] = []
        for source_idx, source in enumerate(sources[: int(_PHYSICAL_SOURCE_SELECT_TOP_N)]):
            rendered, lattice, source_sample_id, geom_distance, params = render_from_source(pred_record, source, geometry_rank, source_name)
            cif = str(rendered.get("cif") or "")
            metric = validate_cif(cif, pred_record["formula_counts"], int(pred_record["sg"])) if cif else {}
            min_distance, vpa = cif_min_distance_and_vpa(cif)
            score = generated_physical_source_score(metric, min_distance, vpa, geom_distance)
            scored.append((score, -source_idx, rendered, lattice, source_sample_id, geom_distance, params))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _score, _source_idx, rendered, lattice, source_sample_id, geom_distance, params = scored[0]
        return rendered, lattice, source_sample_id, geom_distance, params

    source = sources[min(geometry_rank, len(sources) - 1)]
    return render_from_source(pred_record, source, geometry_rank, source_name)


def render_candidate(record: dict[str, Any], candidate: dict[str, Any], rank: int, geometry_rank: int) -> dict[str, Any]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    rows = normalize_rows(_ENGINE, list(candidate.get("rows") or []))
    rendered, lattice, source_sample_id, geom_distance, params = render_with_geometry(record, rows, geometry_rank, _GEOMETRY_MODE)
    cif = str(rendered.get("cif") or "")
    metric = validate_cif(cif, record["formula_counts"], int(record["sg"])) if cif else {
        "readable": False,
        "formula_ok": False,
        "sg_ok": False,
        "atom_count_ok": False,
        "composition_exact": False,
        "atom_count_after_expansion": None,
        "detected_sg": None,
        "error": rendered.get("error") or "empty_cif",
    }
    skel, wa = v2.canonical_keys_from_rows(rows)
    out = {
        "sample_id": record["sample_id"],
        "rank": rank,
        "geometry_mode": _GEOMETRY_MODE,
        "geometry_rank": int(geometry_rank),
        "geometry_source": f"{_GEOMETRY_MODE}_{_GEOMETRY_SOURCE_STRATEGY}",
        "geometry_lattice_mode": _GEOMETRY_LATTICE_MODE,
        "geometry_param_variant_mode": _GEOMETRY_PARAM_VARIANT_MODE,
        "source_sample_id": source_sample_id,
        "geometry_distance": None if geom_distance is None else float(geom_distance),
        "canonical_wa_key": wa,
        "canonical_skeleton_key": skel,
        "candidate_score": candidate.get("score"),
        "cif": cif,
        **metric,
    }
    if _WRITE_GEOMETRY_METADATA:
        out["geometry_lattice"] = None if lattice is None else {str(key): float(value) for key, value in dict(lattice).items()}
        out["geometry_params"] = None if params is None else {
            str(int(row_idx)): {str(symbol): float(value) for symbol, value in dict(row_params).items()}
            for row_idx, row_params in dict(params).items()
        }
    return out


def geometry_plan(
    mode: str,
    top_k: int,
    geometry_ranks_per_wa: int,
    plan_mode: str,
    hybrid_geometry_wa: int,
    hybrid_geometry_ranks: int,
) -> list[tuple[int, int]]:
    if mode == "e07":
        return [(i, g) for g in range(geometry_ranks_per_wa) for i in range(top_k)]
    if mode == "e07_rank0_e08":
        return [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)]
    if plan_mode == "geometry_interleave":
        return [(i, g) for i in range(top_k) for g in range(geometry_ranks_per_wa)]
    if plan_mode == "hybrid_top_wa":
        plan: list[tuple[int, int]] = []
        for i in range(min(top_k, int(hybrid_geometry_wa))):
            for g in range(min(geometry_ranks_per_wa, int(hybrid_geometry_ranks))):
                plan.append((i, g))
        seen = set(plan)
        for i in range(top_k):
            item = (i, 0)
            if item not in seen:
                plan.append(item)
                seen.add(item)
        for g in range(1, geometry_ranks_per_wa):
            for i in range(top_k):
                item = (i, g)
                if item not in seen:
                    plan.append(item)
                    seen.add(item)
        return plan
    return [(i, g) for g in range(geometry_ranks_per_wa) for i in range(top_k)]


def process_payload(payload: tuple[dict[str, Any], dict[str, Any] | None]) -> list[dict[str, Any]]:
    record, prediction = payload
    if prediction is None:
        return []
    cands = list(prediction.get("ranked_wa_candidates") or [])
    populate_geometry_net_lattice_cache(record, cands)
    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for wa_idx, geom_rank in geometry_plan(
        _GEOMETRY_MODE,
        _TOP_K,
        _GEOMETRY_RANKS_PER_WA,
        _GEOMETRY_PLAN_MODE,
        _HYBRID_GEOMETRY_WA,
        _HYBRID_GEOMETRY_RANKS,
    ):
        if len(out) >= _TOP_K:
            break
        if wa_idx >= len(cands):
            continue
        row = render_candidate(record, cands[wa_idx], len(out) + 1, geom_rank)
        if _REQUIRE_COMPOSITION_EXACT and not bool(row.get("composition_exact")):
            continue
        cif_hash = str(row.get("cif") or "")
        if cif_hash in seen_hashes:
            continue
        seen_hashes.add(cif_hash)
        out.append(row)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(r["readable"]) for r in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(r["formula_ok"]) for r in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(r["atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(r["sg_ok"]) for r in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(r["composition_exact"]) for r in rows) / max(1, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render opentry_3 Wyckoff predictions with e07/e08 row-conditioned train geometry.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--geometry-mode", choices=["e07", "e08", "e07_rank0_e08"], default="e08")
    parser.add_argument("--geometry-ranks-per-wa", type=int, default=5)
    parser.add_argument("--geometry-plan-mode", choices=["wa_diverse", "geometry_interleave", "hybrid_top_wa"], default="wa_diverse")
    parser.add_argument("--geometry-source-strategy", choices=["row_conditioned_knn", "wa_skeleton_priority", "row_prototype", "row_aligned_knn", "row_aligned_priority", "row_aligned_quality", "row_aligned_chem_quality", "row_aligned_chem_freepattern_quality", "row_aligned_chem_param_quality", "row_aligned_chem_physical_select", "row_aligned_chem_sourceprior_quality", "row_aligned_chem_sourceprior_freepattern_quality", "row_aligned_chem_successbank_quality", "row_aligned_chem_successbank_freepattern_quality", "row_aligned_chem_successbank_tail_quality", "row_aligned_chem_successbank_tail_freepattern_quality", "row_aligned_chem_sourceproposal_quality", "row_aligned_chem_sourceproposal_tail_quality", "row_aligned_chem_rowbank_quality", "row_aligned_chem_rowbank_freepattern_quality", "row_pair_source_feasible_quality", "row_pair_source_feasible_freepattern_quality"], default="row_conditioned_knn")
    parser.add_argument("--geometry-lattice-mode", choices=["source", "geometry_net", "source_cluster_quantile", "source_vpa_calibrated", "source_vpa_calibrated_soft"], default="source")
    parser.add_argument("--geometry-lattice-ckpt", type=Path, default=None)
    parser.add_argument("--geometry-param-variant-mode", choices=["none", "wrapped_jitter", "wrapped_jitter_lattice", "manifold_params", "manifold_params_lattice", "positive_params", "positive_params_lattice", "positive_nearest_complex", "row_feasibility_complex", "row_pair_feasible", "source_row_shift", "positive_bundle_exact", "positive_bundle_broad"], default="none")
    parser.add_argument("--hybrid-geometry-wa", type=int, default=5)
    parser.add_argument("--hybrid-geometry-ranks", type=int, default=3)
    parser.add_argument("--allow-non-composition-exact", action="store_true")
    parser.add_argument("--exclude-self-source", action="store_true")
    parser.add_argument("--source-prior-json", type=Path, default=None)
    parser.add_argument("--source-prior-weight", type=float, default=0.0)
    parser.add_argument("--source-bank-json", type=Path, default=None)
    parser.add_argument("--source-proposal-json", type=Path, default=None)
    parser.add_argument("--write-geometry-metadata", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-cif-files", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = out_dir / "rendered_cifs"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_records(args.data_root / "train.jsonl")
    split_records = load_records(args.data_root / f"{args.split}.jsonl")
    pred_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    payloads = [(record, pred_by_id.get(str(record["sample_id"]))) for record in split_records if str(record["sample_id"]) in pred_by_id]
    total_payloads = len(payloads)
    if int(args.start_index) > 0:
        payloads = payloads[int(args.start_index) :]
    if int(args.max_records) > 0:
        payloads = payloads[: int(args.max_records)]
    start_time = time.time()

    def report_progress(done: int, rendered_count: int) -> None:
        if int(args.progress_every) <= 0:
            return
        if done % int(args.progress_every) != 0 and done != len(payloads):
            return
        progress = {
            "completed_records": int(done),
            "selected_records": len(payloads),
            "total_available_records": int(total_payloads),
            "rendered_rows": int(rendered_count),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        (out_dir / "render_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)

    if int(args.workers) <= 1:
        init_worker(
            str(args.lookup_json),
            str(args.data_root),
            int(args.top_k),
            str(args.geometry_mode),
            int(args.geometry_ranks_per_wa),
            str(args.geometry_plan_mode),
            str(args.geometry_source_strategy),
            str(args.geometry_lattice_mode),
            "" if args.geometry_lattice_ckpt is None else str(args.geometry_lattice_ckpt),
            str(args.geometry_param_variant_mode),
            int(args.hybrid_geometry_wa),
            int(args.hybrid_geometry_ranks),
            not bool(args.allow_non_composition_exact),
            bool(args.exclude_self_source),
            "" if args.source_prior_json is None else str(args.source_prior_json),
            float(args.source_prior_weight),
            "" if args.source_bank_json is None else str(args.source_bank_json),
            "" if args.source_proposal_json is None else str(args.source_proposal_json),
            bool(args.write_geometry_metadata),
            train_records,
            split_records,
        )
        nested = []
        rendered_count = 0
        for idx, payload in enumerate(payloads, start=1):
            sub = process_payload(payload)
            nested.append(sub)
            rendered_count += len(sub)
            report_progress(idx, rendered_count)
    else:
        with mp.Pool(
            processes=int(args.workers),
            initializer=init_worker,
            initargs=(
                str(args.lookup_json),
                str(args.data_root),
                int(args.top_k),
                str(args.geometry_mode),
                int(args.geometry_ranks_per_wa),
                str(args.geometry_plan_mode),
                str(args.geometry_source_strategy),
                str(args.geometry_lattice_mode),
                "" if args.geometry_lattice_ckpt is None else str(args.geometry_lattice_ckpt),
                str(args.geometry_param_variant_mode),
                int(args.hybrid_geometry_wa),
                int(args.hybrid_geometry_ranks),
                not bool(args.allow_non_composition_exact),
                bool(args.exclude_self_source),
                "" if args.source_prior_json is None else str(args.source_prior_json),
                float(args.source_prior_weight),
                "" if args.source_bank_json is None else str(args.source_bank_json),
                "" if args.source_proposal_json is None else str(args.source_proposal_json),
                bool(args.write_geometry_metadata),
                train_records,
                split_records,
            ),
        ) as pool:
            nested = []
            rendered_count = 0
            for idx, sub in enumerate(pool.imap_unordered(process_payload, payloads, chunksize=4), start=1):
                nested.append(sub)
                rendered_count += len(sub)
                report_progress(idx, rendered_count)
    rows = [item for sub in nested for item in sub]
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    with (out_dir / "rendered_topk.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if i < int(args.write_cif_files):
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sample_id"]))
                (rendered_dir / f"{safe_id}_rank{row['rank']}.cif").write_text(str(row["cif"]), encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "split": args.split,
        "geometry_mode": args.geometry_mode,
        "geometry_plan_mode": args.geometry_plan_mode,
        "geometry_source_strategy": args.geometry_source_strategy,
        "geometry_lattice_mode": args.geometry_lattice_mode,
        "geometry_lattice_ckpt": None if args.geometry_lattice_ckpt is None else str(args.geometry_lattice_ckpt),
        "geometry_param_variant_mode": args.geometry_param_variant_mode,
        "hybrid_geometry_wa": int(args.hybrid_geometry_wa),
        "hybrid_geometry_ranks": int(args.hybrid_geometry_ranks),
        "start_index": int(args.start_index),
        "max_records": int(args.max_records),
        "total_available_records": int(total_payloads),
        "selected_records": len(payloads),
        "top_k": int(args.top_k),
        "samples_with_prediction_rows": len(payloads),
        "samples_with_rendered_candidates": len({r["sample_id"] for r in rows}),
        "rendered_rows": len(rows),
        "geometry_ranks_per_wa": int(args.geometry_ranks_per_wa),
        "require_composition_exact": not bool(args.allow_non_composition_exact),
        "exclude_self_source": bool(args.exclude_self_source),
        "source_prior_json": None if args.source_prior_json is None else str(args.source_prior_json),
        "source_prior_weight": float(args.source_prior_weight),
        "source_bank_json": None if args.source_bank_json is None else str(args.source_bank_json),
        "source_proposal_json": None if args.source_proposal_json is None else str(args.source_proposal_json),
        "write_geometry_metadata": bool(args.write_geometry_metadata),
        "overall_rows": summarize(rows),
        "rank1": summarize([r for r in rows if int(r["rank"]) == 1]),
        "rank_le_5": summarize([r for r in rows if int(r["rank"]) <= 5]),
        "rank_le_20": summarize([r for r in rows if int(r["rank"]) <= 20]),
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
