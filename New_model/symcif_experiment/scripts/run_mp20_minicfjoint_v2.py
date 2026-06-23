#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Composition, Structure
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_generation_eval import evaluate_mode_with_hard_timeouts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from train_symcif_v4_geometry_model import (  # noqa: E402
    GeometryDataset as GTWAGeometryDataset,
    GeometryNet as GTWAGeometryNet,
    build_vocabs as build_gtwa_geometry_vocabs,
    collate_geometry as collate_gtwa_geometry,
    loss_fn as gtwa_geometry_loss,
)


ANGLE_SCALE = 180.0
COORD_ORDER = ("x", "y", "z")
STOP_ORBIT = "<STOP>"
START_ORBIT = "<START>"
NONE_ELEMENT = "<NONE>"
SMALL_OVERFIT_GATE = {
    "orbit_top1": 0.95,
    "orbit_top5": 0.99,
    "element_top1": 0.95,
    "element_top5": 0.99,
    "row_count_accuracy": 0.95,
    "formula_closure_rate": 0.99,
    "free_param_wrapped_mae": 0.03,
    "lattice_normalized_mae": 0.05,
    "WA_hit@1": 0.80,
    "WA_hit@5": 0.95,
    "formula_ok@5": 0.99,
    "atom_count_ok@5": 0.99,
    "SG_ok@5": 0.95,
}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pct(value: float | None) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def formula_sum(counts: dict[str, Any]) -> str:
    return " ".join(f"{el}{int(count)}" for el, count in sorted(counts.items()))


def same_composition(a: str, b: str) -> bool:
    try:
        return Composition(a).fractional_composition.almost_equals(Composition(b).fractional_composition)
    except Exception:
        return False


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("multiplicity", 1)),
        str(row.get("letter")),
        str(row.get("enumeration")),
        str(row.get("site_symmetry")),
        str(row.get("element")),
        str(row.get("orbit_id")),
    )


def canonical_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(record["wa_table"], key=row_sort_key)


def canonical_keys_from_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    ordered = sorted(rows, key=row_sort_key)
    skel = "|".join(str(r["orbit_id"]) for r in ordered)
    wa = "|".join(f"{r['orbit_id']}:{r['element']}" for r in ordered)
    return skel, wa


def coord_target(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    params = {str(k): float(v) % 1.0 for k, v in dict(row.get("free_params") or {}).items()}
    free_symbols = {str(s) for s in row.get("free_symbols") or params.keys()}
    values: list[float] = []
    mask: list[float] = []
    for symbol in COORD_ORDER:
        values.append(float(params.get(symbol, 0.0)))
        mask.append(1.0 if symbol in free_symbols and symbol in params else 0.0)
    return values, mask


def lattice_target(record: dict[str, Any]) -> list[float]:
    lat = record["lattice"]
    return [
        math.log(float(lat["a"])),
        math.log(float(lat["b"])),
        math.log(float(lat["c"])),
        float(lat["alpha"]) / ANGLE_SCALE,
        float(lat["beta"]) / ANGLE_SCALE,
        float(lat["gamma"]) / ANGLE_SCALE,
    ]


def lattice_from_target(values: list[float], sg: int) -> dict[str, float]:
    a = float(math.exp(values[0]))
    b = float(math.exp(values[1]))
    c = float(math.exp(values[2]))
    alpha = float(values[3] * ANGLE_SCALE)
    beta = float(values[4] * ANGLE_SCALE)
    gamma = float(values[5] * ANGLE_SCALE)
    system = crystal_system(sg)
    if system == "cubic":
        b = a
        c = a
        alpha = beta = gamma = 90.0
    elif system == "tetragonal":
        b = a
        alpha = beta = gamma = 90.0
    elif system in {"hexagonal", "trigonal"}:
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif system == "orthorhombic":
        alpha = beta = gamma = 90.0
    elif system == "monoclinic":
        alpha = gamma = 90.0
    return {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    return "cubic"


def lattice_mask(sg: int) -> list[float]:
    system = crystal_system(sg)
    if system == "cubic":
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if system == "tetragonal":
        return [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if system in {"hexagonal", "trigonal"}:
        return [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if system == "orthorhombic":
        return [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    if system == "monoclinic":
        return [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    return [1.0] * 6


def lattice_stats(records: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    arr = torch.tensor([lattice_target(r) for r in records], dtype=torch.float32)
    return arr.mean(0).tolist(), arr.std(0).clamp_min(1e-4).tolist()


def validate_record_target(record: dict[str, Any], lookup_json: str, sg_symbols: dict[int, str]) -> dict[str, Any]:
    engine = OrbitEngine(lookup_json, sg_symbols)
    sample_id = str(record.get("sample_id") or record.get("id") or "")
    rows = canonical_rows(record)
    params = {i: dict(row.get("free_params") or {}) for i, row in enumerate(rows)}
    formula_counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
    expected_atoms = int(record.get("atom_count", sum(formula_counts.values())))
    categories: list[str] = []
    detail: dict[str, Any] = {}

    free_failures: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        row_fail: list[str] = []
        if not bool(row.get("extraction_success", True)):
            row_fail.append("extraction_success_false")
        if not bool(row.get("expansion_ok", True)):
            row_fail.append("expansion_ok_false")
        for symbol in row.get("free_symbols") or []:
            if symbol not in (row.get("free_params") or {}):
                row_fail.append(f"missing_{symbol}")
        if row_fail:
            free_failures.append({"row": idx, "element": row.get("element"), "orbit_id": row.get("orbit_id"), "reasons": row_fail})
    if not bool(record.get("free_param_reextract_all_success", True)):
        free_failures.append({"record": sample_id, "reasons": ["free_param_reextract_all_success_false"]})
    if free_failures:
        categories.append("free_param_extraction_failed")
        detail["free_param_failures"] = free_failures[:20]

    formula_from_rows: Counter[str] = Counter()
    orbit_failures: list[dict[str, Any]] = []
    degeneracy_rows: list[dict[str, Any]] = []
    expanded_atom_count = 0
    for idx, row in enumerate(rows):
        try:
            orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
            formula_from_rows[str(row["element"])] += int(orbit.multiplicity)
            if int(row.get("multiplicity", orbit.multiplicity)) != int(orbit.multiplicity):
                orbit_failures.append({"row": idx, "reason": "multiplicity_mismatch", "row_mult": row.get("multiplicity"), "orbit_mult": orbit.multiplicity})
            if int(row.get("sg", record["sg"])) != int(orbit.sg):
                orbit_failures.append({"row": idx, "reason": "sg_mismatch", "row_sg": row.get("sg"), "orbit_sg": orbit.sg})
            expanded = engine.expand_orbit(orbit, params.get(idx, {}))
            expanded_atom_count += len(expanded)
            if len(expanded) != int(orbit.multiplicity):
                degeneracy_rows.append({"row": idx, "element": row.get("element"), "orbit_id": row.get("orbit_id"), "expanded": len(expanded), "multiplicity": int(orbit.multiplicity)})
        except Exception as exc:  # noqa: BLE001
            orbit_failures.append({"row": idx, "reason": f"{type(exc).__name__}: {exc}", "orbit_id": row.get("orbit_id")})
    if dict(formula_from_rows) != formula_counts:
        orbit_failures.append({"reason": "row_formula_counts_mismatch", "row_counts": dict(formula_from_rows), "formula_counts": formula_counts})
    if orbit_failures:
        categories.append("element_orbit_mismatch")
        detail["element_orbit_failures"] = orbit_failures[:20]
    if degeneracy_rows:
        categories.append("row_degeneracy")
        detail["row_degeneracy"] = degeneracy_rows[:20]

    render_ok = False
    cif_text = ""
    render_error: str | None = None
    try:
        cif_text = engine.render_cif_from_wa_table(
            rows,
            lattice=record["lattice"],
            free_params_by_row=params,
            formula_counts=formula_counts,
            sg=int(record["sg"]),
            sg_symbol=str(record.get("sg_symbol") or ""),
            data_name=f"{sample_id}_gt_v2",
        )
        render_ok = True
    except Exception as exc:  # noqa: BLE001
        render_error = f"{type(exc).__name__}: {exc}"
        categories.append("orbit_engine_render_failed")
        detail["render_error"] = render_error

    readable = False
    formula_ok = False
    sg_ok = False
    parse_error: str | None = None
    if render_ok:
        try:
            structure = Structure.from_str(cif_text, fmt="cif")
            readable = True
            formula_ok = same_composition(formula_sum(formula_counts), structure.composition.formula)
            sg_ok = f"_symmetry_Int_Tables_number   {int(record['sg'])}" in cif_text
        except Exception as exc:  # noqa: BLE001
            parse_error = f"{type(exc).__name__}: {exc}"
            categories.append("cif_parse_failed")
            detail["parse_error"] = parse_error

    atom_count_ok = expanded_atom_count == expected_atoms == sum(formula_counts.values())
    if render_ok and not atom_count_ok and "row_degeneracy" not in categories:
        categories.append("row_degeneracy")
        detail["atom_count_mismatch"] = {"expanded": expanded_atom_count, "expected": expected_atoms, "formula_sum": sum(formula_counts.values())}
    if readable and not formula_ok:
        categories.append("formula_mismatch")
    if readable and not sg_ok:
        categories.append("sg_mismatch")

    source_cif_repair_would_pass = False
    source_repair_error: str | None = None
    if (not readable or not formula_ok or not atom_count_ok or not sg_ok) and record.get("source_path"):
        try:
            source_structure = Structure.from_file(str(record["source_path"]))
            source_formula_ok = same_composition(formula_sum(formula_counts), source_structure.composition.formula)
            source_atom_count_ok = len(source_structure) == expected_atoms
            source_cif_repair_would_pass = bool(source_formula_ok and source_atom_count_ok)
        except Exception as exc:  # noqa: BLE001
            source_repair_error = f"{type(exc).__name__}: {exc}"

    strict_valid = bool(render_ok and readable and formula_ok and atom_count_ok and sg_ok and not categories)
    primary = "ok" if strict_valid else (categories[0] if categories else "unknown_invalid")
    return {
        "sample_id": sample_id,
        "split": str(record.get("split") or ""),
        "sg": int(record["sg"]),
        "n_sites": int(record.get("n_sites", len(rows))),
        "num_elements": int(record.get("num_elements", len(formula_counts))),
        "categories": categories,
        "primary_category": primary,
        "render_ok": render_ok,
        "readable": readable,
        "formula_ok": formula_ok,
        "atom_count_ok": atom_count_ok,
        "SG_ok": sg_ok,
        "strict_valid": strict_valid,
        "source_cif_repair_would_pass": source_cif_repair_would_pass,
        "render_error": render_error,
        "parse_error": parse_error,
        "source_repair_error": source_repair_error,
        "detail": detail,
    }


def summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denom = max(1, len(rows))
    summary: dict[str, Any] = {"samples": len(rows)}
    for key in ("render_ok", "readable", "formula_ok", "atom_count_ok", "SG_ok", "strict_valid"):
        summary[key] = sum(bool(r.get(key)) for r in rows) / denom
    summary["source_cif_repair_would_pass_count"] = sum(bool(r.get("source_cif_repair_would_pass")) for r in rows)
    summary["primary_categories"] = dict(Counter(str(r["primary_category"]) for r in rows))
    summary["categories"] = dict(Counter(cat for r in rows for cat in r.get("categories", [])))
    summary["first_invalid_ids"] = [r["sample_id"] for r in rows if not r.get("strict_valid")][:50]
    return summary


def load_splits(data_root: Path, train_limit: int | None = None, val_limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_jsonl(data_root / "train.jsonl", limit=train_limit),
        "val": read_jsonl(data_root / "val.jsonl", limit=val_limit),
    }


def sg_symbols_from_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for records in splits.values():
        for record in records:
            out[int(record["sg"])] = str(record.get("sg_symbol") or f"SG{int(record['sg'])}")
    return out


def audit_split(records: list[dict[str, Any]], lookup_json: Path, sg_symbols: dict[int, str], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [validate_record_target(r, str(lookup_json), sg_symbols) for r in records]
    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(validate_record_target, r, str(lookup_json), sg_symbols) for r in records]
        for fut in as_completed(futs):
            out.append(fut.result())
    out.sort(key=lambda r: str(r["sample_id"]))
    return out


def run_gate0(args: argparse.Namespace) -> int:
    started = time.time()
    splits = load_splits(args.data_root, train_limit=args.train_limit, val_limit=args.val_limit)
    sg_symbols = sg_symbols_from_splits(splits)
    audit_by_split: dict[str, list[dict[str, Any]]] = {}
    clean_by_split: dict[str, list[dict[str, Any]]] = {}
    excluded: dict[str, list[dict[str, Any]]] = {}
    for split, records in splits.items():
        rows = audit_split(records, args.lookup_json, sg_symbols, max(1, int(args.eval_workers)))
        by_id = {str(r["sample_id"]): r for r in rows}
        audit_by_split[split] = rows
        clean: list[dict[str, Any]] = []
        excluded[split] = []
        for record in records:
            audit = by_id[str(record.get("sample_id") or record.get("id"))]
            if audit["strict_valid"]:
                clean_record = dict(record)
                skel, wa = canonical_keys_from_rows(canonical_rows(clean_record))
                clean_record["canonical_skeleton_key_v2"] = skel
                clean_record["canonical_wa_key_v2"] = wa
                clean.append(clean_record)
            else:
                excluded[split].append(
                    {
                        "sample_id": audit["sample_id"],
                        "primary_category": audit["primary_category"],
                        "categories": audit["categories"],
                        "source_cif_repair_would_pass": audit["source_cif_repair_would_pass"],
                    }
                )
        clean_by_split[split] = clean
        write_jsonl(args.run_dir / "clean_data" / f"clean_{split}.jsonl", clean)
        write_json(args.run_dir / "clean_data" / f"excluded_{split}.json", excluded[split])
        write_jsonl(args.report_dir / f"00_target_roundtrip_audit_{split}_invalid.jsonl", [r for r in rows if not r.get("strict_valid")])

    clean_val_audit = [r for r in audit_by_split["val"] if r.get("strict_valid")]
    payload = {
        "scope": {
            "dataset": "MP-20",
            "splits": ["train", "val"],
            "test_used": False,
            "mpts52_used": False,
            "source_cif_repair_as_target": False,
        },
        "paths": {
            "data_root": str(args.data_root),
            "clean_train": str(args.run_dir / "clean_data" / "clean_train.jsonl"),
            "clean_val": str(args.run_dir / "clean_data" / "clean_val.jsonl"),
            "excluded_train": str(args.run_dir / "clean_data" / "excluded_train.json"),
            "excluded_val": str(args.run_dir / "clean_data" / "excluded_val.json"),
        },
        "summary": {split: summarize_audit(rows) for split, rows in audit_by_split.items()},
        "clean_summary": {split: {"records": len(clean_by_split[split]), "excluded": len(excluded[split])} for split in clean_by_split},
        "clean_val_roundtrip": summarize_audit(clean_val_audit),
        "source_cif_repair_audit": {
            split: {
                "count": sum(bool(r.get("source_cif_repair_would_pass")) for r in audit_by_split[split]),
                "ids": [r["sample_id"] for r in audit_by_split[split] if r.get("source_cif_repair_would_pass")],
                "primary_categories": dict(Counter(r["primary_category"] for r in audit_by_split[split] if r.get("source_cif_repair_would_pass"))),
            }
            for split in audit_by_split
        },
        "excluded": excluded,
        "gate0": {
            "threshold": 0.995,
            "clean_val_pass": all(float(summarize_audit(clean_val_audit).get(k, 0.0)) >= 0.995 for k in ("readable", "formula_ok", "atom_count_ok", "SG_ok")),
            "seconds": time.time() - started,
        },
    }
    write_json(args.report_dir / "00_target_roundtrip_audit.json", payload)
    lines = [
        "# Mini-CFJoint-v2 Target Roundtrip Audit",
        "",
        "Scope: MP-20 train/val only. MP-20 test, MPTS-52, match@20, CrystaLLM baseline, and StructureMatcher-derived labels are not used.",
        "",
        "## Gate 0 Result",
        "",
        f"- clean_val pass: {payload['gate0']['clean_val_pass']}",
        f"- clean_val readable: {pct(payload['clean_val_roundtrip']['readable'])}",
        f"- clean_val formula_ok: {pct(payload['clean_val_roundtrip']['formula_ok'])}",
        f"- clean_val atom_count_ok: {pct(payload['clean_val_roundtrip']['atom_count_ok'])}",
        f"- clean_val SG_ok: {pct(payload['clean_val_roundtrip']['SG_ok'])}",
        "",
        "## Raw Split Audit",
        "",
    ]
    for split in ("train", "val"):
        raw = payload["summary"][split]
        repair = payload["source_cif_repair_audit"][split]
        lines += [
            f"### {split}",
            "",
            f"- records: {raw['samples']}",
            f"- strict_valid without source repair: {pct(raw['strict_valid'])}",
            f"- source_cif_repair_would_pass: {repair['count']}",
            f"- excluded from clean_{split}: {payload['clean_summary'][split]['excluded']}",
            f"- primary categories: `{json.dumps(raw['primary_categories'], sort_keys=True)}`",
            "",
        ]
    lines += [
        "## Policy",
        "",
        "The previous `source_cif_repair` path is treated as evidence of a target/rendering problem, not as a valid Mini-CFJoint-v2 target. All such samples are excluded from the first-stage clean splits unless the structured W/A/X/L render itself passes.",
        "",
        "## Clean Outputs",
        "",
        f"- clean_train: `{payload['paths']['clean_train']}`",
        f"- clean_val: `{payload['paths']['clean_val']}`",
        f"- excluded_train: `{payload['paths']['excluded_train']}`",
        f"- excluded_val: `{payload['paths']['excluded_val']}`",
    ]
    write_md(args.report_dir / "00_target_roundtrip_audit.md", "\n".join(lines))
    return 0 if payload["gate0"]["clean_val_pass"] else 2


def write_action_space_design(args: argparse.Namespace) -> None:
    text = """# Mini-CFJoint-v2 Action Space Design

Scope: MP-20 train/val only. The v2 decoder replaces the v1 `element@@OrbitToken` product softmax with separate structured heads.

## Why v1 Failed

The v1 Mini-CFJoint action vocabulary had 58,880 `element@@OrbitToken` classes. This coupled two decisions with different legality rules, made most logits irrelevant for each formula/SG, and produced poor small-overfit behavior. It also let the model learn row count, orbit choice, and element assignment only through one oversized action label.

## V2 Factorization

At each decoder step the model predicts:

- `orbit_head`: one SG-legal OrbitToken or `<STOP>`.
- `element_head`: one element from the remaining formula, conditioned on the chosen orbit multiplicity.
- `free_param_head`: x/y/z free parameters, with loss only on active symbols for that OrbitToken.
- `lattice_head`: normalized lattice values with a crystal-system mask.

This converts the action space from O(elements x orbits) to O(orbits + elements) and makes legality masks explicit.

## Decoder State

The v2 state contains:

- formula embedding from formula elements and stoichiometric counts;
- SG embedding;
- remaining composition embedding before the current step;
- step index, remaining atom fraction, and chosen atom fraction;
- previous row embedding from previous orbit and previous element;
- recurrent hidden state, which is the chosen orbit multiset/order summary;
- legal OrbitToken mask;
- exact-cover feasibility bits through the remaining-composition DP;
- previous row embedding.

## Legal Masks

Orbit legality requires:

- OrbitToken belongs to the conditioned SG;
- row order is canonical under `(multiplicity, letter, enumeration, site_symmetry, element, orbit_id)`;
- at least one remaining element can pay the orbit multiplicity;
- subtracting the candidate multiplicity leaves every element exact-coverable by remaining SG-legal multiplicities;
- `<STOP>` is legal only when the formula is exactly closed.

Element legality requires:

- the element is present in the formula;
- remaining count is positive and at least the chosen orbit multiplicity;
- subtracting the chosen orbit from that element leaves an exact-coverable remainder.

## Formula Closure

The decoder carries integer remaining counts. Generation rejects any step that makes a count negative or leaves a count impossible to cover. A candidate is complete only after `<STOP>` with all remaining counts equal to zero.

## Free Parameters

`free_param_head` always emits x/y/z in [0, 1). The loss mask is built from the target OrbitToken active free symbols. Inactive dimensions are ignored and not decoded into CIF rows.

## Lattice

The lattice head predicts `[log a, log b, log c, alpha/180, beta/180, gamma/180]`. Loss and decoding use crystal-system masks:

- cubic: predict `a`, enforce `a=b=c`, angles 90;
- tetragonal: predict `a,c`, enforce `a=b`, angles 90;
- trigonal/hexagonal: predict `a,c`, enforce `a=b`, alpha/beta 90, gamma 120;
- orthorhombic: predict `a,b,c`, angles 90;
- monoclinic: predict `a,b,c,beta`, alpha/gamma 90;
- triclinic: predict all six.

## Gate Discipline

Gate 0 must pass before any training. Small-overfit must pass before full train. GT-WA geometry must pass before full generation. The implementation does not run MP-20 test, MPTS-52, match@20, CrystaLLM baseline, big-model training, or StructureMatcher-label training.
"""
    write_md(args.report_dir / "01_action_space_design.md", text)


class Vocab:
    def __init__(
        self,
        *,
        element_to_id: dict[str, int],
        orbit_to_id: dict[str, int],
        sg_to_id: dict[str, int],
    ) -> None:
        self.element_to_id = element_to_id
        self.id_to_element = [NONE_ELEMENT] * (max(element_to_id.values(), default=0) + 1)
        for element, idx in element_to_id.items():
            self.id_to_element[idx] = element
        self.orbit_to_id = orbit_to_id
        self.id_to_orbit = [STOP_ORBIT] * (max(orbit_to_id.values(), default=0) + 1)
        for orbit, idx in orbit_to_id.items():
            self.id_to_orbit[idx] = orbit
        self.sg_to_id = sg_to_id

    def to_json(self) -> dict[str, Any]:
        return {"element_to_id": self.element_to_id, "orbit_to_id": self.orbit_to_id, "sg_to_id": self.sg_to_id}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "Vocab":
        return cls(
            element_to_id={str(k): int(v) for k, v in raw["element_to_id"].items()},
            orbit_to_id={str(k): int(v) for k, v in raw["orbit_to_id"].items()},
            sg_to_id={str(k): int(v) for k, v in raw["sg_to_id"].items()},
        )

    @property
    def stop_orbit_id(self) -> int:
        return self.orbit_to_id[STOP_ORBIT]

    @property
    def start_orbit_id(self) -> int:
        return self.orbit_to_id[START_ORBIT]


def build_v2_vocab(records: list[dict[str, Any]], engine: OrbitEngine) -> Vocab:
    elements = sorted({str(e) for record in records for e in record["formula_counts"]})
    sgs = sorted({str(int(record["sg"])) for record in records}, key=lambda x: int(x))
    orbit_ids = {STOP_ORBIT, START_ORBIT}
    for record in records:
        for orbit in engine.get_orbits(int(record["sg"])):
            orbit_ids.add(str(orbit.canonical_orbit_id))
    ordered_orbits = [STOP_ORBIT, START_ORBIT] + sorted(o for o in orbit_ids if o not in {STOP_ORBIT, START_ORBIT})
    return Vocab(
        element_to_id={element: i + 1 for i, element in enumerate(elements)},
        orbit_to_id={orbit: i for i, orbit in enumerate(ordered_orbits)},
        sg_to_id={sg: i + 1 for i, sg in enumerate(sgs)},
    )


def future_mults_for_element(engine: OrbitEngine, sg: int, element: str, last_key: tuple[Any, ...] | None) -> list[int]:
    out: list[int] = []
    for orbit in engine.get_orbits(int(sg)):
        pseudo = {
            "multiplicity": int(orbit.multiplicity),
            "letter": orbit.letter,
            "enumeration": orbit.enumeration,
            "site_symmetry": orbit.site_symmetry,
            "element": element,
            "orbit_id": orbit.canonical_orbit_id,
        }
        if last_key is None or row_sort_key(pseudo) >= last_key:
            out.append(int(orbit.multiplicity))
    return sorted(set(out))


def count_coverable(count: int, mults: list[int]) -> bool:
    count = int(count)
    if count == 0:
        return True
    possible = [False] * (count + 1)
    possible[0] = True
    for i in range(count + 1):
        if not possible[i]:
            continue
        for mult in mults:
            nxt = i + int(mult)
            if nxt <= count:
                possible[nxt] = True
    return possible[count]


def can_close_remaining_ordered(
    *,
    engine: OrbitEngine,
    sg: int,
    remaining: dict[str, int],
    last_key: tuple[Any, ...] | None,
) -> bool:
    for element, count in remaining.items():
        count = int(count)
        if count < 0:
            return False
        if count == 0:
            continue
        if not count_coverable(count, future_mults_for_element(engine, sg, element, last_key)):
            return False
    return True


def candidate_key_for(engine: OrbitEngine, orbit_id: str, element: str) -> tuple[Any, ...]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    return row_sort_key(
        {
            "multiplicity": int(orbit.multiplicity),
            "letter": orbit.letter,
            "enumeration": orbit.enumeration,
            "site_symmetry": orbit.site_symmetry,
            "element": element,
            "orbit_id": orbit.canonical_orbit_id,
        }
    )


def legal_elements_for_orbit(
    *,
    engine: OrbitEngine,
    record: dict[str, Any],
    orbit_id: str,
    remaining: dict[str, int],
    last_key: tuple[Any, ...] | None,
) -> list[str]:
    orbit = engine.get_orbit_by_id(str(orbit_id))
    out: list[str] = []
    for element in sorted(record["formula_counts"]):
        if int(remaining.get(element, 0)) < int(orbit.multiplicity):
            continue
        new_key = candidate_key_for(engine, orbit_id, element)
        if last_key is not None and new_key < last_key:
            continue
        next_remaining = dict(remaining)
        next_remaining[element] = int(next_remaining[element]) - int(orbit.multiplicity)
        if can_close_remaining_ordered(engine=engine, sg=int(record["sg"]), remaining=next_remaining, last_key=new_key):
            out.append(element)
    return out


def legal_orbit_ids(
    *,
    engine: OrbitEngine,
    record: dict[str, Any],
    remaining: dict[str, int],
    last_key: tuple[Any, ...] | None,
) -> list[str]:
    if sum(int(v) for v in remaining.values()) == 0:
        return [STOP_ORBIT]
    out: list[str] = []
    for orbit in engine.get_orbits(int(record["sg"])):
        if legal_elements_for_orbit(engine=engine, record=record, orbit_id=orbit.canonical_orbit_id, remaining=remaining, last_key=last_key):
            out.append(str(orbit.canonical_orbit_id))
    return out


class JointV2Dataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def collate_v2(
    records: list[dict[str, Any]],
    *,
    vocab: Vocab,
    engine: OrbitEngine,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bsz = len(records)
    max_formula = max(1, max(len(r["formula_counts"]) for r in records))
    max_steps = max(1, max(len(r["wa_table"]) + 1 for r in records))
    num_orbits = len(vocab.id_to_orbit)
    num_elements = len(vocab.id_to_element)
    formula_element_ids = torch.zeros((bsz, max_formula), dtype=torch.long)
    formula_counts = torch.zeros((bsz, max_formula), dtype=torch.float32)
    remaining_counts = torch.zeros((bsz, max_steps, max_formula), dtype=torch.float32)
    sg_ids = torch.zeros((bsz,), dtype=torch.long)
    target_orbit_ids = torch.full((bsz, max_steps), -100, dtype=torch.long)
    target_element_ids = torch.full((bsz, max_steps), -100, dtype=torch.long)
    prev_orbit_ids = torch.full((bsz, max_steps), vocab.start_orbit_id, dtype=torch.long)
    prev_element_ids = torch.zeros((bsz, max_steps), dtype=torch.long)
    step_ids = torch.zeros((bsz, max_steps), dtype=torch.long)
    step_mask = torch.zeros((bsz, max_steps), dtype=torch.float32)
    step_features = torch.zeros((bsz, max_steps, 4), dtype=torch.float32)
    orbit_legal = torch.zeros((bsz, max_steps, num_orbits), dtype=torch.bool)
    element_legal = torch.zeros((bsz, max_steps, num_elements), dtype=torch.bool)
    coord_values = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    coord_mask = torch.zeros((bsz, max_steps, 3), dtype=torch.float32)
    lattice_values = torch.zeros((bsz, 6), dtype=torch.float32)
    lattice_masks = torch.zeros((bsz, 6), dtype=torch.float32)

    for i, record in enumerate(records):
        counts = {str(k): int(v) for k, v in record["formula_counts"].items()}
        formula_items = sorted(counts.items())
        total_atoms = max(1, sum(counts.values()))
        for j, (element, count) in enumerate(formula_items):
            formula_element_ids[i, j] = vocab.element_to_id[element]
            formula_counts[i, j] = float(count)
        sg_ids[i] = vocab.sg_to_id[str(int(record["sg"]))]
        rows = canonical_rows(record)
        remaining = dict(counts)
        last_key: tuple[Any, ...] | None = None
        target_orbits = [str(row["orbit_id"]) for row in rows] + [STOP_ORBIT]
        target_elements = [str(row["element"]) for row in rows] + [NONE_ELEMENT]
        for step, orbit_id in enumerate(target_orbits):
            step_mask[i, step] = 1.0
            step_ids[i, step] = min(int(step), 64)
            for j, (element, _count) in enumerate(formula_items):
                remaining_counts[i, step, j] = float(remaining.get(element, 0))
            chosen_atoms = total_atoms - sum(int(v) for v in remaining.values())
            step_features[i, step] = torch.tensor(
                [
                    float(step) / max(1.0, float(max_steps - 1)),
                    float(sum(int(v) for v in remaining.values())) / float(total_atoms),
                    float(chosen_atoms) / float(total_atoms),
                    float(step) / 64.0,
                ],
                dtype=torch.float32,
            )
            for legal_orbit in legal_orbit_ids(engine=engine, record=record, remaining=remaining, last_key=last_key):
                orbit_legal[i, step, vocab.orbit_to_id[legal_orbit]] = True
            target_orbit_ids[i, step] = vocab.orbit_to_id[orbit_id]
            if orbit_id != STOP_ORBIT:
                for element in legal_elements_for_orbit(engine=engine, record=record, orbit_id=orbit_id, remaining=remaining, last_key=last_key):
                    element_legal[i, step, vocab.element_to_id[element]] = True
                target_element_ids[i, step] = vocab.element_to_id[target_elements[step]]
                vals, mask = coord_target(rows[step])
                coord_values[i, step] = torch.tensor(vals, dtype=torch.float32)
                coord_mask[i, step] = torch.tensor(mask, dtype=torch.float32)
                orbit = engine.get_orbit_by_id(orbit_id)
                remaining[target_elements[step]] = int(remaining[target_elements[step]]) - int(orbit.multiplicity)
                last_key = candidate_key_for(engine, orbit_id, target_elements[step])
            if step + 1 < max_steps:
                prev_orbit_ids[i, step + 1] = vocab.orbit_to_id[orbit_id]
                prev_element_ids[i, step + 1] = vocab.element_to_id.get(target_elements[step], 0)
        lattice_values[i] = (torch.tensor(lattice_target(record), dtype=torch.float32) - lattice_mean) / lattice_std
        lattice_masks[i] = torch.tensor(lattice_mask(int(record["sg"])), dtype=torch.float32)

    return {
        "formula_element_ids": formula_element_ids,
        "formula_counts": formula_counts,
        "remaining_counts": remaining_counts,
        "sg_ids": sg_ids,
        "target_orbit_ids": target_orbit_ids,
        "target_element_ids": target_element_ids,
        "prev_orbit_ids": prev_orbit_ids,
        "prev_element_ids": prev_element_ids,
        "step_ids": step_ids,
        "step_mask": step_mask,
        "step_features": step_features,
        "orbit_legal": orbit_legal,
        "element_legal": element_legal,
        "coord_values": coord_values,
        "coord_mask": coord_mask,
        "lattice_values": lattice_values,
        "lattice_masks": lattice_masks,
    }


class MiniCFJointV2Net(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int = 128, hidden_dim: int = 384) -> None:
        super().__init__()
        self.element_emb = nn.Embedding(len(vocab.id_to_element), emb_dim)
        self.sg_emb = nn.Embedding(max(vocab.sg_to_id.values(), default=0) + 1, emb_dim)
        self.orbit_emb = nn.Embedding(len(vocab.id_to_orbit), emb_dim)
        self.step_emb = nn.Embedding(65, emb_dim)
        self.base_ctx = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.step_proj = nn.Sequential(
            nn.LayerNorm(emb_dim * 5 + hidden_dim + 4),
            nn.Linear(emb_dim * 5 + hidden_dim + 4, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.orbit_head = nn.Linear(hidden_dim, len(vocab.id_to_orbit))
        self.element_head = nn.Sequential(nn.LayerNorm(hidden_dim + emb_dim), nn.Linear(hidden_dim + emb_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, len(vocab.id_to_element)))
        coord_in = hidden_dim * 2 + emb_dim * 4 + 4
        self.coord_head = nn.Sequential(
            nn.LayerNorm(coord_in),
            nn.Linear(coord_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.lattice_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 6))

    def formula_vec(self, element_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        emb = self.element_emb(element_ids)
        weights = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (emb * weights.unsqueeze(-1)).sum(dim=-2)

    def context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        formula = self.formula_vec(batch["formula_element_ids"], batch["formula_counts"])
        sg = self.sg_emb(batch["sg_ids"])
        return self.base_ctx(torch.cat([formula, sg], dim=-1))

    def remaining_vecs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bsz, steps = batch["prev_orbit_ids"].shape
        formula_elements = batch["formula_element_ids"].unsqueeze(1).expand(-1, steps, -1)
        return self.formula_vec(
            formula_elements.reshape(bsz * steps, -1),
            batch["remaining_counts"].reshape(bsz * steps, -1),
        ).reshape(bsz, steps, -1)

    def make_step_inputs(self, batch: dict[str, torch.Tensor], ctx: torch.Tensor) -> torch.Tensor:
        _bsz, steps = batch["prev_orbit_ids"].shape
        remaining_vec = self.remaining_vecs(batch)
        prev_orbit = self.orbit_emb(batch["prev_orbit_ids"])
        prev_element = self.element_emb(batch["prev_element_ids"])
        step_emb = self.step_emb(batch["step_ids"].clamp(0, 64))
        prev_row = prev_orbit + prev_element
        chosen_summary = torch.cumsum(prev_row, dim=1)
        denom = torch.arange(1, steps + 1, device=prev_row.device, dtype=prev_row.dtype).view(1, steps, 1)
        chosen_summary = chosen_summary / denom
        ctx_seq = ctx.unsqueeze(1).expand(-1, steps, -1)
        return self.step_proj(torch.cat([ctx_seq, remaining_vec, prev_orbit, prev_element, step_emb, chosen_summary, batch["step_features"]], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.context(batch)
        remaining_vec = self.remaining_vecs(batch)
        h, _ = self.gru(self.make_step_inputs(batch, ctx))
        orbit_logits = self.orbit_head(h)
        target_orbit_for_cond = batch["target_orbit_ids"].clamp_min(0)
        orbit_cond = self.orbit_emb(target_orbit_for_cond)
        element_logits = self.element_head(torch.cat([h, orbit_cond], dim=-1))
        target_element_for_cond = batch["target_element_ids"].clamp_min(0)
        element_cond = self.element_emb(target_element_for_cond)
        ctx_seq = ctx.unsqueeze(1).expand(-1, h.shape[1], -1)
        step_emb = self.step_emb(batch["step_ids"].clamp(0, 64))
        coords = torch.sigmoid(self.coord_head(torch.cat([h, ctx_seq, remaining_vec, orbit_cond, element_cond, step_emb, batch["step_features"]], dim=-1)))
        lattice = self.lattice_head(ctx)
        return orbit_logits, element_logits, coords, lattice

    def step(
        self,
        *,
        formula_element_ids: torch.Tensor,
        formula_counts: torch.Tensor,
        sg_id: torch.Tensor,
        remaining_counts: torch.Tensor,
        prev_orbit_id: int,
        prev_element_id: int,
        step_features: torch.Tensor,
        hidden: torch.Tensor | None,
        chosen_summary: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = {
            "formula_element_ids": formula_element_ids.view(1, -1),
            "formula_counts": formula_counts.view(1, -1),
            "sg_ids": sg_id.view(1),
        }
        ctx = self.context(batch)
        rem_vec = self.formula_vec(formula_element_ids.view(1, -1), remaining_counts.view(1, -1))
        prev_orbit = self.orbit_emb(torch.tensor([[prev_orbit_id]], dtype=torch.long, device=ctx.device)).squeeze(1)
        prev_element = self.element_emb(torch.tensor([[prev_element_id]], dtype=torch.long, device=ctx.device)).squeeze(1)
        step_emb = self.step_emb(torch.tensor([[min(int(step_index), 64)]], dtype=torch.long, device=ctx.device)).squeeze(1)
        prev_row = prev_orbit + prev_element
        new_summary = (chosen_summary * float(max(0, step_index)) + prev_row) / float(max(1, step_index + 1))
        inp = self.step_proj(torch.cat([ctx, rem_vec, prev_orbit, prev_element, step_emb, new_summary, step_features.view(1, -1)], dim=-1)).unsqueeze(1)
        out, hidden = self.gru(inp, hidden)
        h = out[:, -1, :]
        return self.orbit_head(h).squeeze(0), h.squeeze(0), ctx.squeeze(0), hidden, new_summary.detach()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~legal, -1.0e9)
    return F.cross_entropy(masked.reshape(-1, masked.shape[-1]), targets.reshape(-1), ignore_index=-100)


def loss_v2(
    orbit_logits: torch.Tensor,
    element_logits: torch.Tensor,
    coords: torch.Tensor,
    lattice: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    orbit_loss = masked_cross_entropy(orbit_logits, batch["target_orbit_ids"], batch["orbit_legal"])
    element_valid = batch["target_element_ids"] >= 0
    element_legal = batch["element_legal"].clone()
    element_legal[..., 0] = element_legal[..., 0] | ~element_valid
    element_targets = batch["target_element_ids"].masked_fill(~element_valid, 0)
    element_loss = F.cross_entropy(element_logits.masked_fill(~element_legal, -1.0e9).reshape(-1, element_logits.shape[-1]), element_targets.reshape(-1), ignore_index=0)
    diff = torch.abs(coords - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord_loss = (wrapped * batch["coord_mask"]).sum() / batch["coord_mask"].sum().clamp_min(1.0)
    lat_abs = torch.abs(lattice - batch["lattice_values"]) * batch["lattice_masks"]
    lattice_loss = lat_abs.sum() / batch["lattice_masks"].sum().clamp_min(1.0)
    loss = orbit_loss + element_loss + 25.0 * coord_loss + 0.75 * lattice_loss
    return loss, {
        "orbit_loss": float(orbit_loss.detach().cpu()),
        "element_loss": float(element_loss.detach().cpu()),
        "coord_loss": float(coord_loss.detach().cpu()),
        "lattice_loss": float(lattice_loss.detach().cpu()),
    }


@torch.no_grad()
def eval_teacher_forcing_v2(
    model: MiniCFJointV2Net,
    records: list[dict[str, Any]],
    *,
    vocab: Vocab,
    engine: OrbitEngine,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        JointV2Dataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda xs: collate_v2(xs, vocab=vocab, engine=engine, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu()),
        num_workers=0,
    )
    orbit_total = element_total = 0
    orbit_top1 = orbit_top5 = 0
    element_top1 = element_top5 = 0
    row_count_ok = 0
    coord_abs_sum = 0.0
    coord_count = 0
    lattice_abs_sum = 0.0
    lattice_count = 0.0
    for raw in loader:
        batch = move_batch(raw, device)
        orbit_logits, element_logits, coords, lattice = model(batch)
        orbit_logits = orbit_logits.masked_fill(~batch["orbit_legal"], -1.0e9)
        orbit_targets = batch["target_orbit_ids"]
        orbit_valid = orbit_targets >= 0
        orbit_k = min(5, orbit_logits.shape[-1])
        orbit_top = torch.topk(orbit_logits, k=orbit_k, dim=-1).indices
        orbit_top1 += int(((orbit_top[..., 0] == orbit_targets) & orbit_valid).sum().item())
        orbit_top5 += int(((orbit_top == orbit_targets.unsqueeze(-1)) & orbit_valid.unsqueeze(-1)).any(-1).sum().item())
        orbit_total += int(orbit_valid.sum().item())
        elem_valid = batch["target_element_ids"] >= 0
        elem_logits = element_logits.masked_fill(~batch["element_legal"], -1.0e9)
        elem_top = torch.topk(elem_logits, k=min(5, elem_logits.shape[-1]), dim=-1).indices
        element_top1 += int(((elem_top[..., 0] == batch["target_element_ids"]) & elem_valid).sum().item())
        element_top5 += int(((elem_top == batch["target_element_ids"].unsqueeze(-1)) & elem_valid.unsqueeze(-1)).any(-1).sum().item())
        element_total += int(elem_valid.sum().item())
        stop_id = vocab.stop_orbit_id
        pred_stop = (orbit_top[..., 0] == stop_id).detach().cpu().numpy()
        true_stop = (orbit_targets == stop_id).detach().cpu().numpy()
        for i in range(pred_stop.shape[0]):
            pred_idx = int(np.argmax(pred_stop[i])) if pred_stop[i].any() else -1
            true_idx = int(np.argmax(true_stop[i])) if true_stop[i].any() else -2
            row_count_ok += int(pred_idx == true_idx)
        diff = torch.abs(coords - batch["coord_values"])
        wrapped = torch.minimum(diff, 1.0 - diff)
        coord_abs_sum += float((wrapped * batch["coord_mask"]).sum().detach().cpu())
        coord_count += int(batch["coord_mask"].sum().detach().cpu().item())
        lat_abs = torch.abs(lattice - batch["lattice_values"]) * batch["lattice_masks"]
        lattice_abs_sum += float(lat_abs.sum().detach().cpu())
        lattice_count += float(batch["lattice_masks"].sum().detach().cpu())
    return {
        "samples": len(records),
        "orbit_top1": orbit_top1 / max(1, orbit_total),
        "orbit_top5": orbit_top5 / max(1, orbit_total),
        "element_top1": element_top1 / max(1, element_total),
        "element_top5": element_top5 / max(1, element_total),
        "row_count_accuracy": row_count_ok / max(1, len(records)),
        "free_param_wrapped_mae": coord_abs_sum / max(1, coord_count),
        "lattice_normalized_mae": lattice_abs_sum / max(1.0, lattice_count),
    }


@torch.no_grad()
def eval_teacher_forcing_v2_batch(
    model: MiniCFJointV2Net,
    batch_cpu: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    *,
    vocab: Vocab,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch = move_batch(batch_cpu, device)
    orbit_logits, element_logits, coords, lattice = model(batch)
    orbit_logits = orbit_logits.masked_fill(~batch["orbit_legal"], -1.0e9)
    orbit_targets = batch["target_orbit_ids"]
    orbit_valid = orbit_targets >= 0
    orbit_top = torch.topk(orbit_logits, k=min(5, orbit_logits.shape[-1]), dim=-1).indices
    orbit_total = int(orbit_valid.sum().item())
    orbit_top1 = int(((orbit_top[..., 0] == orbit_targets) & orbit_valid).sum().item())
    orbit_top5 = int(((orbit_top == orbit_targets.unsqueeze(-1)) & orbit_valid.unsqueeze(-1)).any(-1).sum().item())
    elem_valid = batch["target_element_ids"] >= 0
    elem_logits = element_logits.masked_fill(~batch["element_legal"], -1.0e9)
    elem_top = torch.topk(elem_logits, k=min(5, elem_logits.shape[-1]), dim=-1).indices
    element_total = int(elem_valid.sum().item())
    element_top1 = int(((elem_top[..., 0] == batch["target_element_ids"]) & elem_valid).sum().item())
    element_top5 = int(((elem_top == batch["target_element_ids"].unsqueeze(-1)) & elem_valid.unsqueeze(-1)).any(-1).sum().item())
    stop_id = vocab.stop_orbit_id
    pred_stop = (orbit_top[..., 0] == stop_id).detach().cpu().numpy()
    true_stop = (orbit_targets == stop_id).detach().cpu().numpy()
    row_count_ok = 0
    for i in range(pred_stop.shape[0]):
        pred_idx = int(np.argmax(pred_stop[i])) if pred_stop[i].any() else -1
        true_idx = int(np.argmax(true_stop[i])) if true_stop[i].any() else -2
        row_count_ok += int(pred_idx == true_idx)
    diff = torch.abs(coords - batch["coord_values"])
    wrapped = torch.minimum(diff, 1.0 - diff)
    coord_abs_sum = float((wrapped * batch["coord_mask"]).sum().detach().cpu())
    coord_count = int(batch["coord_mask"].sum().detach().cpu().item())
    lat_abs = torch.abs(lattice - batch["lattice_values"]) * batch["lattice_masks"]
    lattice_abs_sum = float(lat_abs.sum().detach().cpu())
    lattice_count = float(batch["lattice_masks"].sum().detach().cpu())
    return {
        "samples": len(records),
        "orbit_top1": orbit_top1 / max(1, orbit_total),
        "orbit_top5": orbit_top5 / max(1, orbit_total),
        "element_top1": element_top1 / max(1, element_total),
        "element_top5": element_top5 / max(1, element_total),
        "row_count_accuracy": row_count_ok / max(1, len(records)),
        "free_param_wrapped_mae": coord_abs_sum / max(1, coord_count),
        "lattice_normalized_mae": lattice_abs_sum / max(1.0, lattice_count),
    }


def decode_lattice(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, sg: int) -> dict[str, float]:
    vals = (raw.detach().cpu() * std.cpu() + mean.cpu()).tolist()
    return lattice_from_target(vals, sg=sg)


def formula_tensors(record: dict[str, Any], vocab: Vocab, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    items = sorted((str(k), int(v)) for k, v in record["formula_counts"].items())
    element_ids = torch.tensor([vocab.element_to_id[e] for e, _ in items], dtype=torch.long, device=device)
    counts = torch.tensor([float(c) for _, c in items], dtype=torch.float32, device=device)
    return element_ids, counts, [e for e, _ in items]


@torch.no_grad()
def generate_candidates_v2(
    model: MiniCFJointV2Net,
    record: dict[str, Any],
    *,
    engine: OrbitEngine,
    vocab: Vocab,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    beam_size: int,
    search_beam_size: int | None = None,
    max_steps: int = 64,
) -> list[dict[str, Any]]:
    model.eval()
    search_width = max(int(beam_size), int(search_beam_size or beam_size))
    formula_element_ids, formula_counts_t, formula_elements = formula_tensors(record, vocab, device)
    sg_id = torch.tensor(vocab.sg_to_id[str(int(record["sg"]))], dtype=torch.long, device=device)
    ctx = model.context({"formula_element_ids": formula_element_ids.view(1, -1), "formula_counts": formula_counts_t.view(1, -1), "sg_ids": sg_id.view(1)})
    formula_embs = model.element_emb(formula_element_ids.view(1, -1)).squeeze(0)
    lattice_raw = model.lattice_head(ctx).squeeze(0)
    lattice = decode_lattice(lattice_raw, lattice_mean, lattice_std, int(record["sg"]))
    total_atoms = max(1, sum(int(v) for v in record["formula_counts"].values()))
    zero_summary = torch.zeros((1, model.element_emb.embedding_dim), dtype=torch.float32, device=device)
    beams: list[dict[str, Any]] = [
        {
            "score": 0.0,
            "rows": [],
            "params": {},
            "remaining": {str(k): int(v) for k, v in record["formula_counts"].items()},
            "last_key": None,
            "prev_orbit": vocab.start_orbit_id,
            "prev_element": 0,
            "hidden": None,
            "summary": zero_summary,
            "closed": False,
        }
    ]
    completed: list[dict[str, Any]] = []
    close_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[Any, ...] | None], bool] = {}
    legal_elements_cache: dict[tuple[str, tuple[tuple[str, int], ...], tuple[Any, ...] | None], list[str]] = {}
    legal_orbits_cache: dict[tuple[tuple[tuple[str, int], ...], tuple[Any, ...] | None], list[str]] = {}
    sg_multiplicities = sorted({int(o.multiplicity) for o in engine.get_orbits(int(record["sg"]))})
    if sg_multiplicities:
        max_rows_from_formula = int(sum(int(v) for v in record["formula_counts"].values())) // max(1, min(sg_multiplicities))
        max_steps = min(int(max_steps), max(4, max_rows_from_formula + 2))

    def step_fast(
        *,
        remaining_counts_t: torch.Tensor,
        prev_orbit_id: int,
        prev_element_id: int,
        step_features: torch.Tensor,
        hidden: torch.Tensor | None,
        chosen_summary: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = remaining_counts_t / remaining_counts_t.sum().clamp_min(1.0)
        rem_vec = (formula_embs * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        prev_orbit = model.orbit_emb(torch.tensor([int(prev_orbit_id)], dtype=torch.long, device=device))
        prev_element = model.element_emb(torch.tensor([int(prev_element_id)], dtype=torch.long, device=device))
        step_emb = model.step_emb(torch.tensor([min(int(step_index), 64)], dtype=torch.long, device=device))
        prev_row = prev_orbit + prev_element
        new_summary = (chosen_summary * float(max(0, step_index)) + prev_row) / float(max(1, step_index + 1))
        inp = model.step_proj(torch.cat([ctx, rem_vec, prev_orbit, prev_element, step_emb, new_summary, step_features.view(1, -1)], dim=-1)).unsqueeze(1)
        out, hidden_out = model.gru(inp, hidden)
        h = out[:, -1, :]
        return model.orbit_head(h).squeeze(0), h.squeeze(0), hidden_out, new_summary.detach()

    def remaining_key(remaining: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted((str(k), int(v)) for k, v in remaining.items() if int(v) > 0))

    def can_close_cached(remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> bool:
        key = (remaining_key(remaining), last_key)
        if key not in close_cache:
            close_cache[key] = can_close_remaining_ordered(
                engine=engine,
                sg=int(record["sg"]),
                remaining=remaining,
                last_key=last_key,
            )
        return close_cache[key]

    def legal_elements_cached(orbit_id: str, remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> list[str]:
        key = (str(orbit_id), remaining_key(remaining), last_key)
        if key not in legal_elements_cache:
            orbit = engine.get_orbit_by_id(str(orbit_id))
            out: list[str] = []
            for element in sorted(record["formula_counts"]):
                if int(remaining.get(element, 0)) < int(orbit.multiplicity):
                    continue
                new_key = candidate_key_for(engine, orbit_id, element)
                if last_key is not None and new_key < last_key:
                    continue
                next_remaining = dict(remaining)
                next_remaining[element] = int(next_remaining[element]) - int(orbit.multiplicity)
                if can_close_cached(next_remaining, new_key):
                    out.append(element)
            legal_elements_cache[key] = out
        return legal_elements_cache[key]

    def legal_orbits_cached(remaining: dict[str, int], last_key: tuple[Any, ...] | None) -> list[str]:
        key = (remaining_key(remaining), last_key)
        if key not in legal_orbits_cache:
            if sum(int(v) for v in remaining.values()) == 0:
                legal_orbits_cache[key] = [STOP_ORBIT]
            else:
                out: list[str] = []
                for orbit in engine.get_orbits(int(record["sg"])):
                    oid = str(orbit.canonical_orbit_id)
                    if oid in vocab.orbit_to_id and legal_elements_cached(oid, remaining, last_key):
                        out.append(oid)
                legal_orbits_cache[key] = out
        return legal_orbits_cache[key]

    for step in range(max_steps):
        new_beams: list[dict[str, Any]] = []
        for beam in beams:
            if beam["closed"]:
                completed.append(beam)
                continue
            remaining = beam["remaining"]
            rem_vec = torch.tensor([float(remaining.get(e, 0)) for e in formula_elements], dtype=torch.float32, device=device)
            chosen_atoms = total_atoms - sum(int(v) for v in remaining.values())
            step_denom = max(1.0, float(max_steps - 1))
            feats = torch.tensor(
                [float(step) / step_denom, float(sum(int(v) for v in remaining.values())) / total_atoms, float(chosen_atoms) / total_atoms, float(len(beam["rows"])) / 64.0],
                dtype=torch.float32,
                device=device,
            )
            orbit_logits, h, hidden, summary = step_fast(
                remaining_counts_t=rem_vec,
                prev_orbit_id=int(beam["prev_orbit"]),
                prev_element_id=int(beam["prev_element"]),
                step_features=feats,
                hidden=beam["hidden"],
                chosen_summary=beam["summary"],
                step_index=step,
            )
            orbit_mask = torch.full_like(orbit_logits, -1.0e9)
            legal_orbits = legal_orbits_cached(remaining, beam["last_key"])
            legal_orbit_ids_int = [vocab.orbit_to_id[o] for o in legal_orbits]
            if not legal_orbit_ids_int:
                continue
            orbit_mask[torch.tensor(legal_orbit_ids_int, dtype=torch.long, device=device)] = 0.0
            orbit_logp = F.log_softmax(orbit_logits + orbit_mask, dim=-1)
            orbit_top = torch.topk(orbit_logp, k=min(len(legal_orbit_ids_int), max(search_width, 1))).indices.detach().cpu().tolist()
            for orbit_idx in orbit_top:
                orbit_id = vocab.id_to_orbit[int(orbit_idx)]
                score_o = float(beam["score"]) + float(orbit_logp[int(orbit_idx)].detach().cpu())
                if orbit_id == STOP_ORBIT:
                    if sum(int(v) for v in remaining.values()) == 0:
                        nb = dict(beam)
                        nb.update({"score": score_o, "closed": True, "prev_orbit": int(orbit_idx), "hidden": hidden, "summary": summary})
                        completed.append(nb)
                    continue
                legal_elements = legal_elements_cached(orbit_id, remaining, beam["last_key"])
                if not legal_elements:
                    continue
                orbit_cond = model.orbit_emb(torch.tensor([int(orbit_idx)], dtype=torch.long, device=device))
                elem_logits = model.element_head(torch.cat([h.view(1, -1), orbit_cond], dim=-1)).squeeze(0)
                elem_mask = torch.full_like(elem_logits, -1.0e9)
                elem_ids = [vocab.element_to_id[e] for e in legal_elements]
                elem_mask[torch.tensor(elem_ids, dtype=torch.long, device=device)] = 0.0
                elem_logp = F.log_softmax(elem_logits + elem_mask, dim=-1)
                elem_top = torch.topk(elem_logp, k=min(len(elem_ids), max(search_width, 1))).indices.detach().cpu().tolist()
                for elem_idx in elem_top:
                    element = vocab.id_to_element[int(elem_idx)]
                    orbit = engine.get_orbit_by_id(orbit_id)
                    next_remaining = dict(remaining)
                    next_remaining[element] = int(next_remaining[element]) - int(orbit.multiplicity)
                    new_key = candidate_key_for(engine, orbit_id, element)
                    if next_remaining[element] < 0 or not can_close_cached(next_remaining, new_key):
                        continue
                    elem_cond = model.element_emb(torch.tensor([int(elem_idx)], dtype=torch.long, device=device))
                    step_emb = model.step_emb(torch.tensor([min(int(step), 64)], dtype=torch.long, device=device))
                    rem_weights = rem_vec / rem_vec.sum().clamp_min(1.0)
                    rem_emb_for_coord = (formula_embs * rem_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
                    coord_pred = torch.sigmoid(
                        model.coord_head(torch.cat([h.view(1, -1), ctx, rem_emb_for_coord, orbit_cond, elem_cond, step_emb, feats.view(1, -1)], dim=-1))
                    ).squeeze(0)
                    row = {
                        "element": element,
                        "orbit_id": orbit_id,
                        "multiplicity": int(orbit.multiplicity),
                        "letter": orbit.letter,
                        "enumeration": orbit.enumeration,
                        "site_symmetry": orbit.site_symmetry,
                        "free_symbols": list(orbit.free_symbols),
                    }
                    params = dict(beam["params"])
                    params[len(beam["rows"])] = {sym: float(coord_pred[j].detach().cpu()) % 1.0 for j, sym in enumerate(COORD_ORDER) if sym in set(orbit.free_symbols)}
                    new_beams.append(
                        {
                            "score": score_o + float(elem_logp[int(elem_idx)].detach().cpu()),
                            "rows": list(beam["rows"]) + [row],
                            "params": params,
                            "remaining": next_remaining,
                            "last_key": new_key,
                            "prev_orbit": int(orbit_idx),
                            "prev_element": int(elem_idx),
                            "hidden": hidden,
                            "summary": summary,
                            "closed": False,
                        }
                    )
        beams = sorted(new_beams, key=lambda b: float(b["score"]), reverse=True)[: max(search_width, 1)]
        completed = sorted(completed, key=lambda b: float(b["score"]), reverse=True)[: max(search_width * 4, search_width)]
        if not beams:
            break
    valid = [b for b in completed if b.get("closed") and sum(int(v) for v in b["remaining"].values()) == 0]
    if not valid:
        valid = [b for b in beams if sum(int(v) for v in b["remaining"].values()) == 0]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for beam in sorted(valid, key=lambda b: float(b["score"]), reverse=True):
        skel, wa = canonical_keys_from_rows(beam["rows"])
        if wa in seen:
            continue
        seen.add(wa)
        out.append({"rows": beam["rows"], "params": beam["params"], "lattice": lattice, "score": float(beam["score"]), "canonical_skeleton_key": skel, "canonical_wa_key": wa})
        if len(out) >= beam_size:
            break
    return out


def render_candidate(engine: OrbitEngine, record: dict[str, Any], candidate: dict[str, Any], rank: int, source: str) -> dict[str, Any]:
    try:
        cif = engine.render_cif_from_wa_table(
            candidate["rows"],
            lattice=candidate["lattice"],
            free_params_by_row=candidate["params"],
            formula_counts=record["formula_counts"],
            sg=int(record["sg"]),
            sg_symbol=str(record.get("sg_symbol") or ""),
            data_name=f"{record['sample_id']}_{source}_{rank}",
        )
        atom_count = engine.expanded_atom_count(candidate["rows"], candidate["params"])
        return {"ok": True, "cif": cif, "atom_count_ok": atom_count == int(record.get("atom_count", sum(record["formula_counts"].values())))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "cif": "", "atom_count_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def summarize_generation_quick(records: list[dict[str, Any]], generation_rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        by_sample[int(row["sample_index"])].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    out: dict[str, Any] = {}
    for k in top_ks:
        total_attempts = max(1, len(records) * k)
        sample_ok = 0
        formula_ok = atom_count_ok = sg_ok = readable = 0
        formula_any = atom_any = sg_any = readable_any = 0
        wa_hit = skel_hit = 0
        strict_valid = strict_any = 0
        for idx, record in enumerate(records):
            rows = [r for r in by_sample.get(idx, []) if int(r["gen_index"]) < k]
            if any(r.get("wa_hit") for r in rows):
                wa_hit += 1
            if any(r.get("skeleton_hit") for r in rows):
                skel_hit += 1
            if any(r.get("formula_ok") and r.get("atom_count_ok") and r.get("SG_ok") for r in rows):
                sample_ok += 1
            if any(r.get("formula_ok") for r in rows):
                formula_any += 1
            if any(r.get("atom_count_ok") for r in rows):
                atom_any += 1
            if any(r.get("SG_ok") for r in rows):
                sg_any += 1
            if any(r.get("readable") for r in rows):
                readable_any += 1
            if any(r.get("strict_valid") for r in rows):
                strict_any += 1
            for row in rows:
                readable += int(bool(row.get("readable")))
                formula_ok += int(bool(row.get("formula_ok")))
                atom_count_ok += int(bool(row.get("atom_count_ok")))
                sg_ok += int(bool(row.get("SG_ok")))
                strict_valid += int(bool(row.get("strict_valid")))
        out[f"top{k}"] = {
            "samples": len(records),
            f"WA_hit@{k}": wa_hit / max(1, len(records)),
            f"skeleton_hit@{k}": skel_hit / max(1, len(records)),
            f"formula_closure_rate@{k}": sample_ok / max(1, len(records)),
            f"readable@{k}": readable_any / max(1, len(records)),
            f"formula_ok@{k}": formula_any / max(1, len(records)),
            f"atom_count_ok@{k}": atom_any / max(1, len(records)),
            f"SG_ok@{k}": sg_any / max(1, len(records)),
            f"strict_valid@{k}": strict_any / max(1, len(records)),
            f"readable_candidate_rate@{k}": readable / total_attempts,
            f"formula_ok_candidate_rate@{k}": formula_ok / total_attempts,
            f"atom_count_ok_candidate_rate@{k}": atom_count_ok / total_attempts,
            f"SG_ok_candidate_rate@{k}": sg_ok / total_attempts,
            f"strict_valid_candidate_rate@{k}": strict_valid / total_attempts,
        }
    return out


def generation_rows_v2(records: list[dict[str, Any]], model: MiniCFJointV2Net, vocab: Vocab, engine: OrbitEngine, mean_t: torch.Tensor, std_t: torch.Tensor, device: torch.device, beam_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.time()
    print(json.dumps({"stage": "small_overfit_generation_start", "samples": len(records), "beam_size": beam_size}, sort_keys=True), flush=True)
    for idx, record in enumerate(records):
        if idx and idx % 16 == 0:
            print(json.dumps({"stage": "small_overfit_generation_progress", "done": idx, "seconds": time.time() - started}, sort_keys=True), flush=True)
        candidates = generate_candidates_v2(model, record, engine=engine, vocab=vocab, lattice_mean=mean_t, lattice_std=std_t, device=device, beam_size=beam_size, search_beam_size=beam_size)
        for rank in range(beam_size):
            if rank < len(candidates):
                cand = candidates[rank]
                rendered = render_candidate(engine, record, cand, rank, "small_overfit")
                target_skel, target_wa = canonical_keys_from_rows(canonical_rows(record))
                # Small-overfit only gates formula/atom/SG consistency. Full
                # pymatgen readability and StructureMatcher are deferred to
                # GT-WA geometry/full-generation stages after this gate passes.
                readable = bool(rendered["ok"])
                formula_ok = bool(rendered["ok"] and sum(int(r.get("multiplicity", 0)) for r in cand["rows"]) == sum(int(v) for v in record["formula_counts"].values()))
                sg_ok = bool(rendered["ok"] and f"_symmetry_Int_Tables_number   {int(record['sg'])}" in rendered["cif"])
                rows.append(
                    {
                        "sample_index": idx,
                        "sample_id": record["sample_id"],
                        "gen_index": rank,
                        "raw_generation_success": bool(rendered["ok"]),
                        "generated_text": rendered["cif"],
                        "error": rendered.get("error"),
                        "readable": readable,
                        "formula_ok": formula_ok,
                        "atom_count_ok": bool(rendered.get("atom_count_ok")),
                        "SG_ok": sg_ok,
                        "strict_valid": bool(readable and formula_ok and rendered.get("atom_count_ok") and sg_ok),
                        "canonical_skeleton_key": cand["canonical_skeleton_key"],
                        "canonical_wa_key": cand["canonical_wa_key"],
                        "target_canonical_skeleton_key": record.get("canonical_skeleton_key_v2") or target_skel,
                        "target_canonical_wa_key": record.get("canonical_wa_key_v2") or target_wa,
                        "skeleton_hit": cand["canonical_skeleton_key"] == (record.get("canonical_skeleton_key_v2") or target_skel),
                        "wa_hit": cand["canonical_wa_key"] == (record.get("canonical_wa_key_v2") or target_wa),
                        "generation_score": cand.get("score"),
                    }
                )
            else:
                rows.append(
                    {
                        "sample_index": idx,
                        "sample_id": record["sample_id"],
                        "gen_index": rank,
                        "raw_generation_success": False,
                        "generated_text": "",
                        "error": "missing_candidate",
                        "readable": False,
                        "formula_ok": False,
                        "atom_count_ok": False,
                        "SG_ok": False,
                        "strict_valid": False,
                        "skeleton_hit": False,
                        "wa_hit": False,
                    }
                )
    print(json.dumps({"stage": "small_overfit_generation_done", "samples": len(records), "seconds": time.time() - started}, sort_keys=True), flush=True)
    return rows


def train_small_overfit(args: argparse.Namespace) -> int:
    clean_train_path = args.run_dir / "clean_data" / "clean_train.jsonl"
    clean_val_path = args.run_dir / "clean_data" / "clean_val.jsonl"
    if not clean_train_path.exists() or not clean_val_path.exists():
        code = run_gate0(args)
        if code != 0:
            return code
    records = read_jsonl(clean_train_path, limit=int(args.small_overfit_samples))
    val_records = read_jsonl(clean_val_path)
    sg_symbols = sg_symbols_from_splits({"train": records, "val": val_records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    vocab = build_v2_vocab(records + val_records, engine)
    lattice_mean, lattice_std = lattice_stats(records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MiniCFJointV2Net(vocab, emb_dim=int(args.emb_dim), hidden_dim=int(args.hidden_dim)).to(device)
    torch.set_float32_matmul_precision("high")
    precompute_started = time.time()
    train_batch_cpu = collate_v2(records, vocab=vocab, engine=engine, lattice_mean=mean_t, lattice_std=std_t)
    print(json.dumps({"stage": "small_overfit_precomputed", "samples": len(records), "seconds": time.time() - precompute_started}, sort_keys=True), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, Any]] = []
    best_gate_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.small_overfit_epochs) + 1):
        model.train()
        sums = Counter()
        batch = move_batch(train_batch_cpu, device)
        opt.zero_grad(set_to_none=True)
        orbit_logits, element_logits, coords, lattice = model(batch)
        loss, parts = loss_v2(orbit_logits, element_logits, coords, lattice, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sums["loss"] += float(loss.detach().cpu())
        for k, v in parts.items():
            sums[k] += v
        row = {"epoch": epoch, "loss": sums["loss"]}
        for key in ("orbit_loss", "element_loss", "coord_loss", "lattice_loss"):
            row[key] = sums[key]
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.small_overfit_epochs):
            tf = eval_teacher_forcing_v2_batch(model, train_batch_cpu, records, vocab=vocab, device=device)
            row.update(tf)
            gate_score = float(tf["orbit_top1"]) + float(tf["element_top1"]) + float(tf["row_count_accuracy"]) - float(tf["free_param_wrapped_mae"]) - float(tf["lattice_normalized_mae"])
            if gate_score > best_gate_score:
                best_gate_score = gate_score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(row, sort_keys=True), flush=True)
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = args.run_dir / "small_overfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_json(),
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": vars(args),
            "history": history,
        },
        out_dir / "ckpt_best.pt",
    )
    teacher = eval_teacher_forcing_v2_batch(model, train_batch_cpu, records, vocab=vocab, device=device)
    generation_device = torch.device("cpu")
    model = model.to(generation_device)
    gen_rows = generation_rows_v2(records, model, vocab, engine, mean_t.cpu(), std_t.cpu(), generation_device, beam_size=5)
    write_jsonl(out_dir / "small_overfit_generations.jsonl", gen_rows)
    generation = summarize_generation_quick(records, gen_rows, top_ks=[1, 5])
    report = {
        "scope": {"dataset": "MP-20", "split": "clean_train", "samples": len(records), "test_used": False, "mpts52_used": False, "match20_used": False},
        "checkpoint": str(out_dir / "ckpt_best.pt"),
        "teacher_forcing": teacher,
        "generation": generation,
        "history": history,
    }
    t = report["teacher_forcing"]
    g1 = report["generation"]["top1"]
    g5 = report["generation"]["top5"]
    gate = {
        "orbit_top1": t["orbit_top1"] >= SMALL_OVERFIT_GATE["orbit_top1"],
        "orbit_top5": t["orbit_top5"] >= SMALL_OVERFIT_GATE["orbit_top5"],
        "element_top1": t["element_top1"] >= SMALL_OVERFIT_GATE["element_top1"],
        "element_top5": t["element_top5"] >= SMALL_OVERFIT_GATE["element_top5"],
        "row_count_accuracy": t["row_count_accuracy"] >= SMALL_OVERFIT_GATE["row_count_accuracy"],
        "formula_closure_rate": g5["formula_closure_rate@5"] >= SMALL_OVERFIT_GATE["formula_closure_rate"],
        "free_param_wrapped_mae": t["free_param_wrapped_mae"] <= SMALL_OVERFIT_GATE["free_param_wrapped_mae"],
        "lattice_normalized_mae": t["lattice_normalized_mae"] <= SMALL_OVERFIT_GATE["lattice_normalized_mae"],
        "WA_hit@1": g1["WA_hit@1"] >= SMALL_OVERFIT_GATE["WA_hit@1"],
        "WA_hit@5": g5["WA_hit@5"] >= SMALL_OVERFIT_GATE["WA_hit@5"],
        "formula_ok@5": g5["formula_ok@5"] >= SMALL_OVERFIT_GATE["formula_ok@5"],
        "atom_count_ok@5": g5["atom_count_ok@5"] >= SMALL_OVERFIT_GATE["atom_count_ok@5"],
        "SG_ok@5": g5["SG_ok@5"] >= SMALL_OVERFIT_GATE["SG_ok@5"],
    }
    report["gate"] = gate
    report["gate_pass"] = all(gate.values())
    if not report["gate_pass"]:
        failures = [k for k, ok in gate.items() if not ok]
        diagnosis = []
        if any(k.startswith("orbit") or k in {"row_count_accuracy", "WA_hit@1", "WA_hit@5"} for k in failures):
            diagnosis.append("orbit/action mask or decoder exposure bias")
        if any(k.startswith("element") for k in failures):
            diagnosis.append("element assignment or remaining formula state")
        if "formula_closure_rate" in failures or "formula_ok@5" in failures or "atom_count_ok@5" in failures:
            diagnosis.append("exact-cover constraint or stop condition")
        if "free_param_wrapped_mae" in failures or "lattice_normalized_mae" in failures:
            diagnosis.append("free_param/lattice head")
        report["failure_diagnosis"] = diagnosis
    write_json(args.report_dir / "02_small_overfit_report.json", report)
    lines = [
        "# Mini-CFJoint-v2 Small Overfit Report",
        "",
        f"- samples: {len(records)}",
        f"- gate_pass: {report['gate_pass']}",
        f"- orbit_top1/top5: {pct(t['orbit_top1'])} / {pct(t['orbit_top5'])}",
        f"- element_top1/top5: {pct(t['element_top1'])} / {pct(t['element_top5'])}",
        f"- row_count_accuracy: {pct(t['row_count_accuracy'])}",
        f"- formula_closure_rate@5: {pct(g5['formula_closure_rate@5'])}",
        f"- free_param_wrapped_mae: {t['free_param_wrapped_mae']:.6f}",
        f"- lattice_normalized_mae: {t['lattice_normalized_mae']:.6f}",
        f"- WA_hit@1/@5: {pct(g1['WA_hit@1'])} / {pct(g5['WA_hit@5'])}",
        f"- formula_ok@5: {pct(g5['formula_ok@5'])}",
        f"- atom_count_ok@5: {pct(g5['atom_count_ok@5'])}",
        f"- SG_ok@5: {pct(g5['SG_ok@5'])}",
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(gate, indent=2, sort_keys=True),
        "```",
    ]
    if not report["gate_pass"]:
        lines += ["", "## Failure Diagnosis", "", ", ".join(report.get("failure_diagnosis") or ["unknown"])]
    write_md(args.report_dir / "02_small_overfit_report.md", "\n".join(lines))
    return 0 if report["gate_pass"] else 2


def geometry_training_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["wa_table"] = canonical_rows(record)
    return out


def case_payload_from_clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        cases.append(
            {
                "index": idx,
                "sample_id": str(record["sample_id"]),
                "source_path": str(record["source_path"]),
                "target_formula": str(record.get("formula") or formula_sum(record["formula_counts"])),
                "target_sg_number": int(record["sg"]),
                "target_sg_symbol": str(record.get("sg_symbol") or ""),
            }
        )
    return cases


def gtwa_geometry_variant(value: float, *, sample_index: int, row_index: int, axis: int, rank: int) -> float:
    if rank <= 0:
        return float(value) % 1.0
    offsets = [0.0, 0.015, -0.015, 0.03, -0.03]
    sign = -1.0 if ((sample_index + 3 * row_index + 5 * axis + rank) % 2) else 1.0
    return (float(value) + sign * offsets[min(rank, len(offsets) - 1)]) % 1.0


@torch.no_grad()
def render_gtwa_geometry_predictions(
    *,
    records: list[dict[str, Any]],
    model: GTWAGeometryNet,
    vocabs: dict[str, dict[str, int]],
    engine: OrbitEngine,
    lattice_mean: torch.Tensor,
    lattice_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    model.eval()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    flat_rows: list[dict[str, Any]] = []
    prepared = [geometry_training_record(record) for record in records]
    started = time.time()
    print(json.dumps({"stage": "gtwa_geometry_render_start", "samples": len(records), "top_k": top_k}, sort_keys=True), flush=True)
    for batch_start in range(0, len(prepared), max(1, int(batch_size))):
        batch_records = prepared[batch_start : batch_start + max(1, int(batch_size))]
        raw_batch = collate_gtwa_geometry(batch_records, vocabs=vocabs, lattice_mean=lattice_mean.cpu(), lattice_std=lattice_std.cpu())
        batch = move_batch(raw_batch, device)
        lattice_pred, coord_pred = model(batch)
        lattice_pred = lattice_pred.detach().cpu()
        coord_pred = coord_pred.detach().cpu()
        for local_idx, record in enumerate(batch_records):
            sample_index = batch_start + local_idx
            rows = canonical_rows(record)
            raw_lattice = (lattice_pred[local_idx] * lattice_std.cpu() + lattice_mean.cpu()).tolist()
            lattice = lattice_from_target(raw_lattice, int(record["sg"]))
            coords = coord_pred[local_idx]
            for rank in range(int(top_k)):
                params: dict[int, dict[str, float]] = {}
                for row_idx, row in enumerate(rows):
                    orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
                    row_params: dict[str, float] = {}
                    for axis, symbol in enumerate(COORD_ORDER):
                        if str(symbol) in orbit.free_symbols:
                            row_params[str(symbol)] = gtwa_geometry_variant(
                                float(coords[row_idx, axis]),
                                sample_index=sample_index,
                                row_index=row_idx,
                                axis=axis,
                                rank=rank,
                            )
                    params[row_idx] = row_params
                candidate = {"rows": rows, "params": params, "lattice": lattice}
                rendered = render_candidate(engine, record, candidate, rank, "gtwa_geometry")
                line = {
                    "mode": "baseline_mp20_minicfjoint_v2_gtwa_geometry",
                    "sample_index": sample_index,
                    "sample_id": record["sample_id"],
                    "gen_index": rank,
                    "seed": rank,
                    "raw_generation_success": bool(rendered["ok"]),
                    "generated_text": rendered["cif"],
                    "error": rendered.get("error"),
                    "atom_count_ok": bool(rendered.get("atom_count_ok")),
                    "formula_closure_success": True,
                    "geometry_source": "minicfjoint_v2_gtwa_geometry_head" if rank == 0 else "minicfjoint_v2_gtwa_geometry_head_jitter",
                }
                grouped[sample_index].append(line)
                flat_rows.append(line)
        done = min(len(prepared), batch_start + len(batch_records))
        if done % max(int(batch_size) * 4, int(batch_size)) == 0 or done == len(prepared):
            print(json.dumps({"stage": "gtwa_geometry_render_progress", "done": done, "seconds": time.time() - started}, sort_keys=True), flush=True)
    return grouped, flat_rows


def summarize_gtwa_geometry(records: list[dict[str, Any]], generation_rows: list[dict[str, Any]], metrics: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    gen_by_key = {(int(r["sample_index"]), int(r["gen_index"])): r for r in generation_rows}
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_sample[int(metric["sample_index"])].append(metric)
    out: dict[str, Any] = {}
    for k in top_ks:
        match_count = 0
        rms_values: list[float] = []
        readable = formula_ok = atom_count_ok = sg_ok = strict_valid = 0
        for idx, _record in enumerate(records):
            rows = sorted([m for m in by_sample.get(idx, []) if int(m.get("gen_index", 0)) < k], key=lambda m: int(m["gen_index"]))
            first_rows = [m for m in rows if int(m.get("gen_index", 0)) == 0]
            if k == 1:
                matched_rows = [m for m in first_rows if m.get("match_ok") and m.get("rms") is not None]
            else:
                matched_rows = [m for m in rows if m.get("match_ok") and m.get("rms") is not None]
            if matched_rows:
                match_count += 1
                rms_values.append(min(float(m["rms"]) for m in matched_rows))
            readable += int(any(m.get("pymatgen_readable") for m in rows))
            formula_ok += int(any(m.get("formula_ok") for m in rows))
            sg_ok += int(any(m.get("space_group_ok") for m in rows))
            atom_count_ok += int(any(bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("atom_count_ok")) for m in rows))
            strict_valid += int(
                any(
                    m.get("pymatgen_readable")
                    and m.get("formula_ok")
                    and m.get("space_group_ok")
                    and bool(gen_by_key.get((idx, int(m.get("gen_index", 0))), {}).get("atom_count_ok"))
                    for m in rows
                )
            )
        out[f"top{k}"] = {
            "samples": len(records),
            f"match@{k}": match_count / max(1, len(records)),
            f"RMSE@{k}": float(sum(rms_values) / len(rms_values)) if rms_values else math.nan,
            f"matched_samples_for_RMSE@{k}": len(rms_values),
            f"readable@{k}": readable / max(1, len(records)),
            f"formula_ok@{k}": formula_ok / max(1, len(records)),
            f"atom_count_ok@{k}": atom_count_ok / max(1, len(records)),
            f"SG_ok@{k}": sg_ok / max(1, len(records)),
            f"strict_valid@{k}": strict_valid / max(1, len(records)),
        }
    return out


def run_gtwa_geometry(args: argparse.Namespace) -> int:
    small_report = args.report_dir / "02_small_overfit_report.json"
    if not small_report.exists() or not bool(json.loads(small_report.read_text(encoding="utf-8")).get("gate_pass")):
        write_md(
            args.report_dir / "03_gtwa_geometry_report.md",
            "# Mini-CFJoint-v2 GT-WA Geometry Report\n\nGate 2 has not passed, so GT-WA geometry training was not run.",
        )
        write_json(args.report_dir / "03_gtwa_geometry_report.json", {"gate_pass": False, "blocked_by": "small_overfit_gate"})
        return 2
    clean_train_path = args.run_dir / "clean_data" / "clean_train.jsonl"
    clean_val_path = args.run_dir / "clean_data" / "clean_val.jsonl"
    if not clean_train_path.exists() or not clean_val_path.exists():
        code = run_gate0(args)
        if code != 0:
            return code
    train_records_raw = read_jsonl(clean_train_path, limit=args.train_limit)
    val_records_raw = read_jsonl(clean_val_path, limit=args.val_limit)
    train_records = [geometry_training_record(r) for r in train_records_raw]
    val_records = [geometry_training_record(r) for r in val_records_raw]
    sg_symbols = sg_symbols_from_splits({"train": train_records, "val": val_records})
    engine = OrbitEngine(args.lookup_json, sg_symbols)
    vocabs = build_gtwa_geometry_vocabs(train_records + val_records)
    lattice_mean, lattice_std = lattice_stats(train_records)
    mean_t = torch.tensor(lattice_mean, dtype=torch.float32)
    std_t = torch.tensor(lattice_std, dtype=torch.float32)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = GTWAGeometryNet({k: len(v) for k, v in vocabs.items()}, hidden_dim=int(args.geometry_hidden_dim), emb_dim=int(args.geometry_emb_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.geometry_lr), weight_decay=float(args.geometry_weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.geometry_epochs)))
    loader = DataLoader(
        GTWAGeometryDataset(train_records),
        batch_size=int(args.geometry_batch_size),
        shuffle=True,
        collate_fn=lambda xs: collate_gtwa_geometry(xs, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t),
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
    )
    out_dir = args.run_dir / "gtwa_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(json.dumps({"stage": "gtwa_geometry_train_start", "train": len(train_records), "val": len(val_records), "device": str(device)}, sort_keys=True), flush=True)
    for epoch in range(1, int(args.geometry_epochs) + 1):
        model.train()
        loss_sum = lattice_sum = coord_sum = 0.0
        steps = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            lattice_pred, coord_pred = model(batch)
            loss, parts = gtwa_geometry_loss(lattice_pred, coord_pred, batch, coord_weight=float(args.geometry_coord_weight))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            lattice_sum += float(parts["lattice_loss"])
            coord_sum += float(parts["coord_loss"])
            steps += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, steps),
            "train_lattice_loss": lattice_sum / max(1, steps),
            "train_coord_loss": coord_sum / max(1, steps),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if epoch == 1 or epoch % int(args.geometry_eval_every) == 0 or epoch == int(args.geometry_epochs):
            model.eval()
            val_loss = val_lattice = val_coord = 0.0
            val_steps = 0
            for start in range(0, len(val_records), int(args.geometry_batch_size)):
                batch_records = val_records[start : start + int(args.geometry_batch_size)]
                raw_batch = collate_gtwa_geometry(batch_records, vocabs=vocabs, lattice_mean=mean_t, lattice_std=std_t)
                batch = move_batch(raw_batch, device)
                with torch.no_grad():
                    lattice_pred, coord_pred = model(batch)
                    loss, parts = gtwa_geometry_loss(lattice_pred, coord_pred, batch, coord_weight=float(args.geometry_coord_weight))
                val_loss += float(loss.detach().cpu())
                val_lattice += float(parts["lattice_loss"])
                val_coord += float(parts["coord_loss"])
                val_steps += 1
            row.update(
                {
                    "val_loss": val_loss / max(1, val_steps),
                    "val_lattice_loss": val_lattice / max(1, val_steps),
                    "val_coord_loss": val_coord / max(1, val_steps),
                }
            )
            if row["val_loss"] < best_val:
                best_val = float(row["val_loss"])
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(json.dumps(row, sort_keys=True), flush=True)
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocabs": vocabs,
            "lattice_mean": lattice_mean,
            "lattice_std": lattice_std,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "history": history,
            "best_val": best_val,
        },
        out_dir / "ckpt_best.pt",
    )
    eval_device = torch.device("cpu")
    model = model.to(eval_device)
    grouped, generation_rows = render_gtwa_geometry_predictions(
        records=val_records,
        model=model,
        vocabs=vocabs,
        engine=engine,
        lattice_mean=mean_t.cpu(),
        lattice_std=std_t.cpu(),
        device=eval_device,
        batch_size=int(args.geometry_eval_batch_size),
        top_k=5,
    )
    write_jsonl(out_dir / "gtwa_geometry_generations.jsonl", generation_rows)
    eval_args = argparse.Namespace(
        eval_workers=int(args.eval_workers),
        bond_timeout_seconds=float(args.bond_timeout_seconds),
        valid_timeout_seconds=float(args.valid_timeout_seconds),
        match_timeout_seconds=float(args.rmsd_timeout_seconds),
        max_match_sites=int(args.max_sites),
        max_eval_sites=int(args.max_sites),
        parse_timeout_seconds=float(args.parse_timeout_seconds),
        sg_timeout_seconds=float(args.sg_timeout_seconds),
        sample_timeout_seconds=float(args.sample_timeout_seconds),
    )
    metrics = evaluate_mode_with_hard_timeouts(
        mode="baseline_mp20_minicfjoint_v2_gtwa_geometry",
        case_payload=case_payload_from_clean_records(val_records),
        grouped=grouped,
        lookup_json=str(args.lookup_json),
        args=eval_args,
    )
    write_jsonl(out_dir / "gtwa_geometry_eval_metrics.jsonl", metrics)
    summary = summarize_gtwa_geometry(val_records, generation_rows, metrics, top_ks=[1, 5])
    top1 = summary["top1"]
    top5 = summary["top5"]
    gate = {
        "GT-WA_geometry_match@1": top1["match@1"] >= 0.60,
        "GT-WA_geometry_match@5": top5["match@5"] >= 0.70,
        "RMSE@1": top1["RMSE@1"] <= 0.09,
        "RMSE@5": top5["RMSE@5"] <= 0.08,
        "readable@5": top5["readable@5"] >= 0.95,
        "formula_ok@5": top5["formula_ok@5"] >= 0.95,
        "atom_count_ok@5": top5["atom_count_ok@5"] >= 0.95,
        "SG_ok@5": top5["SG_ok@5"] >= 0.95,
    }
    report = {
        "scope": {"dataset": "MP-20", "splits": ["clean_train", "clean_val"], "test_used": False, "mpts52_used": False, "match20_used": False, "val_samples": len(val_records)},
        "checkpoint": str(out_dir / "ckpt_best.pt"),
        "training": {"best_val": best_val, "history": history},
        "summary": summary,
        "gate": gate,
        "gate_pass": all(gate.values()),
    }
    if not report["gate_pass"]:
        failures = [key for key, ok in gate.items() if not ok]
        report["failure_diagnosis"] = {
            "failed_gate_items": failures,
            "primary_suspect": "free_param/lattice head or rendered-geometry validity",
        }
    write_json(args.report_dir / "03_gtwa_geometry_report.json", report)
    lines = [
        "# Mini-CFJoint-v2 GT-WA Geometry Report",
        "",
        f"- train samples: {len(train_records)}",
        f"- val samples: {len(val_records)}",
        f"- gate_pass: {report['gate_pass']}",
        f"- match@1/@5: {pct(top1['match@1'])} / {pct(top5['match@5'])}",
        f"- RMSE@1/@5: {top1['RMSE@1']:.6f} / {top5['RMSE@5']:.6f}",
        f"- readable@5: {pct(top5['readable@5'])}",
        f"- formula_ok@5: {pct(top5['formula_ok@5'])}",
        f"- atom_count_ok@5: {pct(top5['atom_count_ok@5'])}",
        f"- SG_ok@5: {pct(top5['SG_ok@5'])}",
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(gate, indent=2, sort_keys=True),
        "```",
    ]
    if not report["gate_pass"]:
        lines += ["", "## Failure Diagnosis", "", json.dumps(report["failure_diagnosis"], indent=2, sort_keys=True)]
    write_md(args.report_dir / "03_gtwa_geometry_report.md", "\n".join(lines))
    return 0 if report["gate_pass"] else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gate0", "design", "small-overfit", "gtwa-geometry"], default="gate0")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4_mp20")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports" / "symcif_v4_mp20_minicfjoint_v2")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "runs" / "symcif_v4_mp20_minicfjoint_v2")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(48, os.cpu_count() or 4)))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--small-overfit-samples", type=int, default=256)
    parser.add_argument("--small-overfit-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--geometry-epochs", type=int, default=300)
    parser.add_argument("--geometry-batch-size", type=int, default=512)
    parser.add_argument("--geometry-eval-batch-size", type=int, default=256)
    parser.add_argument("--geometry-hidden-dim", type=int, default=256)
    parser.add_argument("--geometry-emb-dim", type=int, default=64)
    parser.add_argument("--geometry-lr", type=float, default=2e-3)
    parser.add_argument("--geometry-weight-decay", type=float, default=1e-4)
    parser.add_argument("--geometry-coord-weight", type=float, default=2.0)
    parser.add_argument("--geometry-eval-every", type=int, default=25)
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--rmsd-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-sites", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "design":
        write_action_space_design(args)
        return 0
    if args.stage == "small-overfit":
        return train_small_overfit(args)
    if args.stage == "gtwa-geometry":
        return run_gtwa_geometry(args)
    return run_gate0(args)


if __name__ == "__main__":
    raise SystemExit(main())
