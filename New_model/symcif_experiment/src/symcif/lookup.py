from __future__ import annotations

import ast
import csv
import json
import math
import re
import string
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
from pymatgen.core import Element
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from .models import WyckoffTemplate


LETTERS = string.ascii_lowercase + string.ascii_uppercase


class LookupError(ValueError):
    pass


def _mod1(x: float) -> float:
    y = x % 1.0
    return 0.0 if abs(y - 1.0) < 1e-8 or abs(y) < 1e-8 else y


def _parse_number(token: str) -> float:
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def affine_from_xyz_expr(xyz_expr: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Wyckoff xyz expression into a 3x3 rotation and 3-vector shift."""
    rot = np.zeros((3, 3), dtype=float)
    trans = np.zeros(3, dtype=float)
    parts = xyz_expr.strip().replace(" ", "").lower().split(",")
    if len(parts) != 3:
        raise LookupError(f"expected 3 comma-separated coordinates, got {xyz_expr!r}")

    re_var = re.compile(r"([+-]?)([\d.]*)/?([\d.]*)([xyz])")
    re_const = re.compile(r"([+-]?)(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)(?![xyz])")
    for row, part in enumerate(parts):
        for match in re_var.finditer(part):
            sign = -1.0 if match.group(1) == "-" else 1.0
            num = match.group(2)
            den = match.group(3)
            factor = sign
            if num:
                factor *= float(num) / float(den) if den else float(num)
            col = "xyz".index(match.group(4))
            rot[row, col] += factor

        scrubbed = re_var.sub("", part)
        for match in re_const.finditer(scrubbed):
            sign = -1.0 if match.group(1) == "-" else 1.0
            trans[row] += sign * _parse_number(match.group(2))

    return rot, trans


def evaluate_xyz_expr(xyz_expr: str, values: tuple[float, float, float]) -> tuple[float, float, float]:
    rot, trans = affine_from_xyz_expr(xyz_expr)
    coord = rot @ np.array(values, dtype=float) + trans
    return tuple(float(_mod1(v)) for v in coord)


def lattice_for_space_group(sg: SpaceGroup) -> Lattice:
    cs = sg.crystal_system
    if cs == "cubic":
        return Lattice.cubic(6.0)
    if cs == "tetragonal":
        return Lattice.tetragonal(5.0, 7.0)
    if cs == "orthorhombic":
        return Lattice.orthorhombic(5.0, 6.0, 7.0)
    if cs in {"hexagonal", "trigonal"}:
        return Lattice.hexagonal(5.0, 8.0)
    if cs == "monoclinic":
        return Lattice.monoclinic(5.0, 6.0, 7.0, 110.0)
    return Lattice.from_parameters(5.0, 6.0, 7.0, 80.0, 95.0, 105.0)


class WyckoffLookup:
    """Wyckoff templates from CrystalFormer, locally checked with pymatgen/spglib."""

    def __init__(self, templates: dict[tuple[int, str], WyckoffTemplate]):
        self.templates = templates

    @classmethod
    def from_crystalformer_csv(
        cls,
        csv_path: str | Path,
        wyformer_json: str | Path | None = None,
        infer_site_symmetry: bool = True,
        symprec: float = 0.1,
        angle_tolerance: float = 5.0,
    ) -> "WyckoffLookup":
        templates: dict[tuple[int, str], WyckoffTemplate] = {}
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sg_number = int(row["Space Group"])
                wyckoff_positions = ast.literal_eval(row["Wyckoff Positions"])
                conventional_order = list(reversed(wyckoff_positions))
                for idx, operations in enumerate(conventional_order):
                    letter = LETTERS[idx]
                    rep = operations[0]
                    rot, trans = affine_from_xyz_expr(rep)
                    free_mask = tuple(bool(np.abs(rot[i]).sum() > 1e-12) for i in range(3))
                    fixed_values = tuple(0.0 if free_mask[i] else float(_mod1(trans[i])) for i in range(3))
                    templates[(sg_number, letter)] = WyckoffTemplate(
                        sg_number=sg_number,
                        letter=letter,
                        multiplicity=len(operations),
                        operations=tuple(operations),
                        representative_expr=rep,
                        free_mask=free_mask,  # type: ignore[arg-type]
                        fixed_values=fixed_values,  # type: ignore[arg-type]
                    )

        lookup = cls(templates)
        if wyformer_json is not None and Path(wyformer_json).exists():
            lookup = lookup.with_wyformer_mappings(wyformer_json)
            infer_site_symmetry = False
        if infer_site_symmetry:
            lookup = lookup.with_inferred_site_symmetry(symprec=symprec, angle_tolerance=angle_tolerance)
        return lookup

    def get(self, sg_number: int, letter: str) -> WyckoffTemplate:
        key = (int(sg_number), str(letter))
        try:
            return self.templates[key]
        except KeyError as exc:
            raise LookupError(f"missing Wyckoff template for SG={sg_number} letter={letter}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "WyckoffLookup":
        templates: dict[tuple[int, str], WyckoffTemplate] = {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for sg_raw, entries in raw.items():
            sg_number = int(sg_raw)
            for _, item in entries.items():
                letter = str(item["letter"])
                template = str(item["template"])
                templates[(sg_number, letter)] = WyckoffTemplate(
                    sg_number=sg_number,
                    letter=letter,
                    multiplicity=int(item["multiplicity"]),
                    operations=tuple(item.get("operations") or (template,)),
                    representative_expr=template,
                    free_mask=tuple(bool(v) for v in item["free_mask"]),  # type: ignore[arg-type]
                    fixed_values=tuple(float(v) for v in item["fixed_values"]),  # type: ignore[arg-type]
                    site_symmetry=item.get("site_symmetry"),
                    enumeration=item.get("enumeration"),
                    site_symmetry_status=str(item.get("site_symmetry_status", "loaded")),
                    source=str(item.get("source", "artifact")),
                )
        return cls(templates)

    def with_inferred_site_symmetry(self, symprec: float = 0.1, angle_tolerance: float = 5.0) -> "WyckoffLookup":
        updated: dict[tuple[int, str], WyckoffTemplate] = {}
        by_sg: dict[int, list[WyckoffTemplate]] = defaultdict(list)
        for template in self.templates.values():
            by_sg[template.sg_number].append(template)

        element_pool = [el.symbol for el in Element if el.Z <= 86 and el.symbol not in {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}]
        for sg_number, entries in by_sg.items():
            sg = SpaceGroup.from_int_number(sg_number)
            lattice = lattice_for_space_group(sg)
            entries_sorted = sorted(entries, key=lambda t: LETTERS.index(t.letter))
            species = element_pool[: len(entries_sorted)]
            coords = [evaluate_xyz_expr(t.representative_expr, (0.173, 0.271, 0.389)) for t in entries_sorted]
            inferred: dict[str, tuple[str | None, str]] = {}
            try:
                struct = Structure.from_spacegroup(sg.symbol, lattice, species, coords)
                analyzer = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
                dataset = analyzer.get_symmetry_dataset()
                if int(dataset.number) != sg_number:
                    raise LookupError(f"sg_mismatch:{dataset.number}")
                wyckoffs = list(dataset.wyckoffs)
                site_syms = list(dataset.site_symmetry_symbols)
                for symbol, expected in zip(species, entries_sorted):
                    matches = [i for i, site in enumerate(struct) if site.specie.symbol == symbol]
                    if not matches:
                        inferred[expected.letter] = (None, "element_marker_missing")
                        continue
                    idx = matches[0]
                    got_letter = str(wyckoffs[idx])
                    if got_letter != expected.letter:
                        inferred[expected.letter] = (str(site_syms[idx]), f"letter_mismatch:{got_letter}")
                    else:
                        inferred[expected.letter] = (str(site_syms[idx]), "inferred")
            except Exception as exc:  # noqa: BLE001 - recorded in lookup artifact
                for template in entries_sorted:
                    inferred[template.letter] = (None, f"infer_error:{type(exc).__name__}:{exc}")

            for template in entries_sorted:
                site_symmetry, status = inferred.get(template.letter, (None, "not_checked"))
                updated[(sg_number, template.letter)] = replace(
                    template,
                    site_symmetry=site_symmetry,
                    site_symmetry_status=status,
                )

        enum_by_sg_ss: dict[tuple[int, str], int] = defaultdict(int)
        enumerated: dict[tuple[int, str], WyckoffTemplate] = {}
        for sg_number in sorted(by_sg):
            for letter in sorted((t.letter for t in by_sg[sg_number]), key=LETTERS.index):
                template = updated[(sg_number, letter)]
                enum = None
                if template.site_symmetry:
                    key = (sg_number, template.site_symmetry)
                    enum = enum_by_sg_ss[key]
                    enum_by_sg_ss[key] += 1
                enumerated[(sg_number, letter)] = replace(template, enumeration=enum)
        return WyckoffLookup(enumerated)

    def with_wyformer_mappings(self, path: str | Path) -> "WyckoffLookup":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ss_from_letter = {int(sg): v for sg, v in raw["ss_from_letter"].items()}
        enum_from_letter = {int(sg): v for sg, v in raw["enum_from_ss_letter"].items()}
        updated: dict[tuple[int, str], WyckoffTemplate] = {}
        for (sg_number, letter), template in self.templates.items():
            ss = ss_from_letter.get(sg_number, {}).get(letter)
            enum = enum_from_letter.get(sg_number, {}).get(letter)
            updated[(sg_number, letter)] = replace(
                template,
                site_symmetry=ss,
                enumeration=None if enum is None else int(enum),
                site_symmetry_status="wyformer_reference" if ss is not None else "wyformer_missing",
                source="crystalformer_wyckoff_list+wyformer_enumeration",
            )
        return WyckoffLookup(updated)

    def to_jsonable(self) -> dict[str, dict[str, dict[str, object]]]:
        out: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        for (sg_number, letter), t in sorted(self.templates.items()):
            key = f"{t.multiplicity}{letter}"
            out[str(sg_number)][key] = {
                "multiplicity": t.multiplicity,
                "letter": letter,
                "site_symmetry": t.site_symmetry,
                "enumeration": t.enumeration,
                "free_mask": list(t.free_mask),
                "fixed_values": [round(v, 8) for v in t.fixed_values],
                "template": t.representative_expr,
                "operations": list(t.operations),
                "site_symmetry_status": t.site_symmetry_status,
                "source": t.source,
            }
        return dict(out)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_jsonable(), f, indent=2, sort_keys=True)


def close_fractional(a: float, b: float, tol: float = 5e-3) -> bool:
    return math.isclose(((a - b + 0.5) % 1.0) - 0.5, 0.0, abs_tol=tol)
