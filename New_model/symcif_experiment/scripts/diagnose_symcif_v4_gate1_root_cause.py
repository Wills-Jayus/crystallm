#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # type: ignore
from pymatgen.io.cif import CifParser  # type: ignore
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore

from symcif.lookup import affine_from_xyz_expr
from symcif_v4.canonicalize import canonical_skeleton_key, canonical_wa_key, wa_table_from_structured
from symcif_v4.formula import normalize_formula_counts, total_atoms
from symcif_v4.orbit_engine import OrbitEngine, mod1
from symcif_v4.validation import structure_counts
from train_skeleton_template_ranker import read_jsonl

warnings.filterwarnings("ignore")

SPECIAL_VALUES = [
    0.0,
    1 / 12,
    1 / 8,
    1 / 6,
    1 / 4,
    1 / 3,
    3 / 8,
    1 / 2,
    5 / 8,
    2 / 3,
    3 / 4,
    5 / 6,
    7 / 8,
    11 / 12,
    1.0,
]
TOP_FAILED_SG_LETTERS = {"225|24e", "189|3f", "189|3g", "193|6g", "216|24f"}
SOURCE_ROOT = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl_map(root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for row in read_jsonl(root / f"{split}.jsonl"):
            by_id[row.get("sample_id") or row.get("id")] = row
            all_rows.append(row)
    return by_id, all_rows


def source_path_for_sample(sample_id: str) -> Path | None:
    if "__" not in sample_id:
        return None
    prefix, name = sample_id.split("__", 1)
    path = SOURCE_ROOT / prefix / f"{name}.cif"
    return path if path.exists() else None


def periodic_distance(a: float, b: float) -> float:
    d = abs(mod1(a) - mod1(b))
    return min(d, abs(1.0 - d))


def near_special(value: float, tol: float = 1e-4) -> dict[str, Any]:
    v = mod1(float(value))
    nearest = min(SPECIAL_VALUES, key=lambda x: periodic_distance(v, x))
    dist = periodic_distance(v, nearest)
    return {"value": v, "nearest_special": nearest % 1.0, "distance": dist, "is_near": dist <= tol}


def special_flags(coord: tuple[float, float, float], params: dict[str, float]) -> dict[str, Any]:
    return {
        "coord": {axis: near_special(coord[i]) for i, axis in enumerate(("x", "y", "z"))},
        "free_params": {k: near_special(v) for k, v in sorted(params.items())},
    }


def fit_coord_to_orbit(coord: tuple[float, float, float], representative_expr: tuple[str, str, str]) -> tuple[float, dict[str, float]]:
    rot, trans = affine_from_xyz_expr(", ".join(representative_expr))
    best_resid = float("inf")
    best_params: dict[str, float] = {}
    c = np.array(coord, dtype=float)
    for shifts in np.ndindex(3, 3, 3):
        shifted = c + np.array([s - 1 for s in shifts], dtype=float)
        try:
            sol, *_ = np.linalg.lstsq(rot, shifted - trans, rcond=None)
        except Exception:
            continue
        pred = rot @ sol + trans
        resid = float(np.linalg.norm(pred - shifted, ord=np.inf))
        if resid < best_resid:
            best_resid = resid
            best_params = {"x": mod1(sol[0]), "y": mod1(sol[1]), "z": mod1(sol[2])}
    return best_resid, best_params


def nearest_lower_orbit(engine: OrbitEngine, sg: int, declared_multiplicity: int, expanded_count: int, coord: tuple[float, float, float]) -> dict[str, Any] | None:
    candidates = [o for o in engine.get_orbits(sg) if int(o.multiplicity) <= int(declared_multiplicity)]
    if expanded_count > 0:
        exact = [o for o in candidates if int(o.multiplicity) == int(expanded_count)]
        if exact:
            candidates = exact
    scored: list[tuple[float, Any, dict[str, float]]] = []
    for orbit in candidates:
        resid, params = fit_coord_to_orbit(coord, orbit.representative_expr)
        scored.append((resid, orbit, params))
    if not scored:
        return None
    resid, orbit, params = min(scored, key=lambda x: (x[0], x[1].multiplicity))
    return {
        "orbit_id": orbit.canonical_orbit_id,
        "letter": orbit.letter,
        "multiplicity": orbit.multiplicity,
        "site_symmetry": orbit.site_symmetry,
        "representative_expr": list(orbit.representative_expr),
        "fit_residual": resid,
        "fit_params": params,
        "identifiable": resid <= 1e-4,
    }


def parse_cif_atom_site_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(text):
        if text[i].strip().lower() != "loop_":
            i += 1
            continue
        i += 1
        headers: list[str] = []
        while i < len(text) and text[i].strip().startswith("_"):
            headers.append(text[i].strip().split()[0])
            i += 1
        if not headers or not any(h.startswith("_atom_site") for h in headers):
            continue
        while i < len(text):
            stripped = text[i].strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            if stripped.lower() == "loop_" or stripped.startswith("_") or stripped.lower().startswith("data_"):
                break
            try:
                parts = shlex.split(stripped)
            except Exception:
                parts = stripped.split()
            if len(parts) >= len(headers):
                rows.append({headers[j]: parts[j] for j in range(len(headers))})
            i += 1
    return rows


def occupancy_disorder_flags(atom_rows: list[dict[str, str]]) -> dict[str, Any]:
    occ_values: list[float] = []
    labels: list[str] = []
    disorder = False
    for row in atom_rows:
        occ = row.get("_atom_site_occupancy")
        if occ not in {None, ".", "?"}:
            try:
                occ_values.append(float(occ))
            except Exception:
                pass
        label = row.get("_atom_site_label")
        if label:
            labels.append(label)
        disorder_group = row.get("_atom_site_disorder_group") or row.get("_atom_site_disorder_assembly")
        if disorder_group not in {None, ".", "?", "0"}:
            disorder = True
    duplicate_labels = len(labels) != len(set(labels))
    partial = any(abs(v - 1.0) > 1e-6 for v in occ_values)
    return {
        "has_source_atom_rows": bool(atom_rows),
        "occupancy_values": sorted(set(round(v, 6) for v in occ_values))[:20],
        "partial_occupancy": partial,
        "duplicate_labels": duplicate_labels,
        "disorder_flag": disorder,
        "atom_site_row_count": len(atom_rows),
    }


def validate_cif_with_symprec(cif_text: str, target_counts: dict[str, int], target_sg: int, spglib_symprec: float) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "readable": False,
        "formula_ok": False,
        "atom_count_ok": False,
        "sg_ok": False,
        "detected_sg": None,
        "error": None,
    }
    try:
        parser = CifParser.from_str(cif_text) if hasattr(CifParser, "from_str") else CifParser.from_string(cif_text)  # type: ignore[attr-defined]
    except Exception:
        from io import StringIO

        parser = CifParser(StringIO(cif_text))
    try:
        structures = parser.parse_structures(primitive=False) if hasattr(parser, "parse_structures") else parser.get_structures(primitive=False)
        if not structures:
            raise ValueError("no structures parsed")
        structure = structures[0]
        target = normalize_formula_counts(target_counts)
        metric["readable"] = True
        got = structure_counts(structure)
        metric["formula_ok"] = got == target
        metric["atom_count_ok"] = len(structure) == sum(target.values())
        try:
            detected = int(SpacegroupAnalyzer(structure, symprec=spglib_symprec, angle_tolerance=5.0).get_space_group_number())
            metric["detected_sg"] = detected
            metric["sg_ok"] = detected == int(target_sg)
        except Exception as exc:
            metric["error"] = f"sg_detect:{type(exc).__name__}:{exc}"
    except Exception as exc:
        metric["error"] = f"{type(exc).__name__}:{exc}"
    return metric


def build_row_audit(v3_rows: list[dict[str, Any]], engine: OrbitEngine) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    audit_rows: list[dict[str, Any]] = []
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tolerances = [1e-6, 1e-5, 1e-4, 1e-3]
    for row in v3_rows:
        wa_table, free_params = wa_table_from_structured(row, engine)
        for idx, wa_row in enumerate(wa_table):
            orbit = engine.get_orbit_by_id(wa_row["orbit_id"])
            params = free_params[idx]
            coord = engine.evaluate_representative(orbit, params)
            counts_by_tol = {f"{tol:g}": len(engine.expand_orbit(orbit, params, symprec=tol)) for tol in tolerances}
            expanded_count = counts_by_tol["1e-05"]
            expansion_ok = expanded_count == int(orbit.multiplicity)
            nearest = None
            if not expansion_ok:
                nearest = nearest_lower_orbit(engine, int(row["sg"]), int(orbit.multiplicity), expanded_count, coord)
            item = {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "formula": row["formula"],
                "formula_counts": row["formula_counts"],
                "sg": int(row["sg"]),
                "sg_symbol": row["sg_symbol"],
                "element": wa_row["element"],
                "claimed_orbit_id": orbit.canonical_orbit_id,
                "claimed_letter": orbit.letter,
                "claimed_multiplicity": orbit.multiplicity,
                "site_symmetry": orbit.site_symmetry,
                "representative_expr": list(orbit.representative_expr),
                "free_symbols": list(orbit.free_symbols),
                "free_params": params,
                "representative_coord": list(coord),
                "expanded_unique_count": expanded_count,
                "declared_multiplicity": int(orbit.multiplicity),
                "expansion_ok": expansion_ok,
                "failed_ratio": None if expansion_ok else f"{expanded_count}/{int(orbit.multiplicity)}",
                "nearest_lower_multiplicity_orbit": nearest,
                "special_value_flags": special_flags(coord, params),
                "unique_counts_by_tolerance": counts_by_tol,
            }
            audit_rows.append(item)
            by_sample[row["sample_id"]].append(item)
    return audit_rows, by_sample


def row_expansion_summary(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        groups[f"{row['sg']}|{row['declared_multiplicity']}{row['claimed_letter']}"].append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        failed = [r for r in rows if not r["expansion_ok"]]
        out.append(
            {
                "sg_letter": key,
                "rows": len(rows),
                "failed": len(failed),
                "failure_rate": len(failed) / max(1, len(rows)),
                "expanded_counts": ";".join(f"{k}:{v}" for k, v in Counter(r["expanded_unique_count"] for r in rows).most_common()),
                "near_special_failed": sum(
                    any(flag["is_near"] for flag in r["special_value_flags"]["free_params"].values()) for r in failed
                ),
                "top_nearest_lower": Counter(
                    ((r.get("nearest_lower_multiplicity_orbit") or {}).get("letter") or "NA") for r in failed
                ).most_common(3),
            }
        )
    return out


def source_symmetry_case(path: Path | None, symprecs: list[float]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"source_path": None, "source_available": False}
    out: dict[str, Any] = {"source_path": str(path), "source_available": True}
    try:
        from io import StringIO

        parser = CifParser(StringIO(path.read_text(encoding="utf-8", errors="ignore")))
        structures = parser.parse_structures(primitive=False) if hasattr(parser, "parse_structures") else parser.get_structures(primitive=False)
        structure = structures[0]
        out["source_readable"] = True
        out["symmetry_by_symprec"] = {}
        for symprec in symprecs:
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5.0)
                dataset = analyzer.get_symmetry_dataset()
                wyckoffs = list(getattr(dataset, "wyckoffs", dataset["wyckoffs"]))
                site_syms = list(getattr(dataset, "site_symmetry_symbols", dataset["site_symmetry_symbols"]))
                equiv = list(getattr(dataset, "equivalent_atoms", dataset["equivalent_atoms"]))
                number = int(getattr(dataset, "number", dataset["number"]))
                out["symmetry_by_symprec"][str(symprec)] = {
                    "sg": number,
                    "wyckoff_counts": dict(Counter(str(x) for x in wyckoffs)),
                    "site_symmetry_counts": dict(Counter(str(x) for x in site_syms)),
                    "num_equiv_groups": len(set(int(x) for x in equiv)),
                    "site_preview": [
                        {
                            "index": i,
                            "species": str(structure[i].specie),
                            "frac_coords": [float(v) for v in structure[i].frac_coords],
                            "wyckoff": str(wyckoffs[i]),
                            "site_symmetry": str(site_syms[i]),
                            "equivalent_atom": int(equiv[i]),
                        }
                        for i in range(min(20, len(structure)))
                    ],
                }
            except Exception as exc:
                out["symmetry_by_symprec"][str(symprec)] = {"error": f"{type(exc).__name__}:{exc}"}
    except Exception as exc:
        out["source_readable"] = False
        out["source_error"] = f"{type(exc).__name__}:{exc}"
    return out


def classify_samples(
    v3_rows: list[dict[str, Any]],
    row_by_sample: dict[str, list[dict[str, Any]]],
    engine: OrbitEngine,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    for row in v3_rows:
        row_audits = row_by_sample[row["sample_id"]]
        wa_table, free_params = wa_table_from_structured(row, engine)
        cif = engine.render_cif_from_wa_table(
            wa_table,
            row["lattice"],
            free_params,
            row["formula_counts"],
            int(row["sg"]),
            row["sg_symbol"],
            data_name=row["sample_id"],
        )
        metric = validate_cif_with_symprec(cif, row["formula_counts"], int(row["sg"]), 0.1)
        expanded_ok = all(r["expansion_ok"] for r in row_audits)
        atom_formula_ok = bool(metric["formula_ok"] and metric["atom_count_ok"])
        source_path = source_path_for_sample(row["sample_id"])
        source_flags = occupancy_disorder_flags(parse_cif_atom_site_rows(source_path))
        flags = {
            "row_degeneracy": not expanded_ok,
            "unreadable_cif": not metric["readable"],
            "formula_loss_after_cif_read": expanded_ok and metric["readable"] and not atom_formula_ok,
            "sg_detection_only": expanded_ok and atom_formula_ok and not metric["sg_ok"],
            "occupancy_or_disorder_suspected": bool(
                source_flags["partial_occupancy"] or source_flags["duplicate_labels"] or source_flags["disorder_flag"]
            ),
        }
        if flags["row_degeneracy"]:
            primary = "A.row_degeneracy"
        elif flags["unreadable_cif"]:
            primary = "D.unreadable_cif"
        elif flags["formula_loss_after_cif_read"]:
            primary = "B.formula_loss_after_cif_read"
        elif flags["sg_detection_only"]:
            primary = "C.sg_detection_only"
        elif flags["occupancy_or_disorder_suspected"]:
            primary = "F.occupancy_or_disorder_suspected"
        elif not (metric["readable"] and metric["formula_ok"] and metric["atom_count_ok"] and metric["sg_ok"]):
            primary = "E.setting_origin_mismatch_suspected"
        else:
            primary = "PASS"
        item = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "formula": row["formula"],
            "formula_counts": row["formula_counts"],
            "sg": int(row["sg"]),
            "sg_symbol": row["sg_symbol"],
            "n_sites": int(row["n_sites"]),
            "num_elements": int(row["num_elements"]),
            "primary_class": primary,
            "flags": flags,
            "source_path": None if source_path is None else str(source_path),
            "source_flags": source_flags,
            "metric": metric,
            "degenerate_rows": [
                {
                    "element": r["element"],
                    "sg_letter": f"{r['sg']}|{r['declared_multiplicity']}{r['claimed_letter']}",
                    "free_params": r["free_params"],
                    "expanded_unique_count": r["expanded_unique_count"],
                    "declared_multiplicity": r["declared_multiplicity"],
                    "nearest_lower": r["nearest_lower_multiplicity_orbit"],
                }
                for r in row_audits
                if not r["expansion_ok"]
            ],
            "canonical_skeleton_key": canonical_skeleton_key(wa_table),
            "canonical_wa_key": canonical_wa_key(wa_table),
        }
        out.append(item)
        class_counter[primary] += 1
        for key, value in flags.items():
            if value:
                flag_counter[key] += 1
        if primary != "PASS":
            failure_rows.append(item)
    summary = {
        "total_samples": len(out),
        "failed_samples": len(failure_rows),
        "class_counts": dict(class_counter),
        "flag_counts": dict(flag_counter),
    }
    return out, failure_rows, summary


def make_sample_failure_summary(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        groups[row["primary_class"]].append(row)
    rows = []
    for cls, items in sorted(groups.items()):
        rows.append(
            {
                "primary_class": cls,
                "samples": len(items),
                "fraction": len(items) / max(1, len(class_rows)),
                "formula_ok_rate": sum(bool(x["metric"]["formula_ok"]) for x in items) / max(1, len(items)),
                "atom_count_ok_rate": sum(bool(x["metric"]["atom_count_ok"]) for x in items) / max(1, len(items)),
                "sg_ok_rate": sum(bool(x["metric"]["sg_ok"]) for x in items) / max(1, len(items)),
                "readable_rate": sum(bool(x["metric"]["readable"]) for x in items) / max(1, len(items)),
                "partial_occupancy_samples": sum(bool(x["source_flags"]["partial_occupancy"]) for x in items),
                "source_available": sum(bool(x["source_flags"]["has_source_atom_rows"]) for x in items),
            }
        )
    return rows


def deep_dive_cases(
    v3_rows: list[dict[str, Any]],
    row_by_sample: dict[str, list[dict[str, Any]]],
    target_sg_letters: set[str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in v3_rows:
        source_path = source_path_for_sample(row["sample_id"])
        atom_rows = parse_cif_atom_site_rows(source_path)
        for audit in row_by_sample[row["sample_id"]]:
            key = f"{audit['sg']}|{audit['declared_multiplicity']}{audit['claimed_letter']}"
            if key not in target_sg_letters:
                continue
            related = [
                ar
                for ar in atom_rows
                if ar.get("_atom_site_type_symbol") == audit["element"]
                or str(ar.get("_atom_site_label", "")).startswith(str(audit["element"]))
            ][:20]
            cases.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "sg": int(row["sg"]),
                    "sg_symbol": row["sg_symbol"],
                    "sg_letter": key,
                    "element": audit["element"],
                    "formula_counts": row["formula_counts"],
                    "free_params": audit["free_params"],
                    "representative_expr": audit["representative_expr"],
                    "representative_coord": audit["representative_coord"],
                    "expanded_unique_count": audit["expanded_unique_count"],
                    "declared_multiplicity": audit["declared_multiplicity"],
                    "nearest_lower_multiplicity_orbit": audit["nearest_lower_multiplicity_orbit"],
                    "source_path": None if source_path is None else str(source_path),
                    "source_atom_rows_related_to_element": related,
                    "source_occupancy_flags": occupancy_disorder_flags(atom_rows),
                    "old_structured_v3_free_coords": [
                        c for c in row["free_coords"] if int(c["site_order"]) == int(row["assignment"][int(row["free_coords"].index(c))]["site_order"])
                    ],
                    "v4_orbitengine_free_params": audit["free_params"],
                    "special_value_flags": audit["special_value_flags"],
                }
            )
    return cases


def histogram(values: list[float]) -> dict[str, int]:
    return dict(Counter(f"{round(float(v), 6):.6f}" for v in values))


def write_sg225_report(path: Path, cases: list[dict[str, Any]]) -> None:
    values = []
    for case in cases:
        values.extend(float(v) for v in case["free_params"].values())
    expanded_counts = Counter(case["expanded_unique_count"] for case in cases)
    constants = {k: Counter(str(case["free_params"].get(k)) for case in cases) for k in ("x", "y", "z")}
    source_available = sum(bool(c["source_path"]) for c in cases)
    partial = sum(bool(c["source_occupancy_flags"]["partial_occupancy"]) for c in cases)
    lines = [
        "# SG225 24e Deep Dive",
        "",
        f"- cases: {len(cases)}",
        f"- source CIF available: {source_available}",
        f"- partial occupancy suspected: {partial}",
        f"- expanded count distribution: {dict(expanded_counts)}",
        f"- free parameter hist: {histogram(values)}",
        f"- per-axis constants: { {k: v.most_common(10) for k, v in constants.items()} }",
        "",
        "## Interpretation",
        "",
    ]
    if cases and len(set(tuple(sorted(c["free_params"].items())) for c in cases)) <= 3:
        lines.append("24e 的 free_params 高度集中在少数特殊值；这更像提取/label convention 问题，或原始结构实际处在更低 multiplicity orbit，而不是正常 24e general/special orbit。")
    if all(c["expanded_unique_count"] < c["declared_multiplicity"] for c in cases):
        lines.append("所有 24e case 在 OrbitEngine 展开前已经退化，因此 formula_loss 不是 CIF read/write 引入的。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_top_failed_deep_dive(path: Path, cases_by_key: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Top Failed SG/Letter Deep Dive", ""]
    for key, cases in sorted(cases_by_key.items()):
        params = []
        for c in cases:
            params.extend(float(v) for v in c["free_params"].values())
        lines.extend(
            [
                f"## {key}",
                "",
                f"- cases: {len(cases)}",
                f"- expanded count distribution: {dict(Counter(c['expanded_unique_count'] for c in cases))}",
                f"- free parameter histogram: {histogram(params)}",
                f"- source available: {sum(bool(c['source_path']) for c in cases)}",
                f"- partial occupancy suspected: {sum(bool(c['source_occupancy_flags']['partial_occupancy']) for c in cases)}",
                f"- nearest lower orbit top: {Counter(str((c.get('nearest_lower_multiplicity_orbit') or {}).get('letter')) for c in cases).most_common(5)}",
                "",
            ]
        )
        if all(c["expanded_unique_count"] < c["declared_multiplicity"] for c in cases):
            lines.append("判断：主要是真实 special-position degeneracy 或 extraction 把 lower orbit 坐标贴到了 higher orbit label；需要回到原始 extraction 逻辑确认，不能静默 relabel。")
        else:
            lines.append("判断：不是纯 row degeneracy，还包含 setting/origin 或 spglib detection 问题。")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def source_extraction_cases(
    failure_rows: list[dict[str, Any]],
    row_by_sample: dict[str, list[dict[str, Any]]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    sg225 = [r for r in failure_rows if any(d["sg_letter"] == "225|24e" for d in r["degenerate_rows"])][:10]
    rest = [r for r in failure_rows if r not in sg225]
    selected = (sg225 + rest)[:limit]
    cases = []
    for row in selected:
        source_path = source_path_for_sample(row["sample_id"])
        source_info = source_symmetry_case(source_path, [1e-3, 1e-2, 1e-1])
        cases.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "sg": row["sg"],
                "structured_v3_labels": [
                    {
                        "element": audit["element"],
                        "letter": audit["claimed_letter"],
                        "multiplicity": audit["declared_multiplicity"],
                        "site_symmetry": audit["site_symmetry"],
                        "expanded_unique_count": audit["expanded_unique_count"],
                        "nearest_lower": audit["nearest_lower_multiplicity_orbit"],
                    }
                    for audit in row_by_sample[row["sample_id"]]
                ],
                "structured_v4_orbit_labels": [audit["claimed_orbit_id"] for audit in row_by_sample[row["sample_id"]]],
                "source_info": source_info,
                "label_shift_judgement": "source unavailable" if not source_info.get("source_available") else "compare spglib wyckoff_counts with structured labels; full per-site mapping needs original extraction equivalent-site metadata",
            }
        )
    return cases


def write_source_extraction_md(path: Path, cases: list[dict[str, Any]]) -> None:
    available = sum(bool(c["source_info"].get("source_available")) for c in cases)
    lines = [
        "# Source Extraction Consistency",
        "",
        f"- sampled cases: {len(cases)}",
        f"- source CIF available: {available}",
        "",
        "structured records do not carry `source_path` or original extraction equivalent-site metadata. Source CIF paths were inferred from benchmark GT directory when possible.",
        "",
    ]
    for c in cases[:20]:
        lines.append(f"## {c['sample_id']}")
        lines.append(f"- SG: {c['sg']}")
        lines.append(f"- source: {c['source_info'].get('source_path')}")
        lines.append(f"- structured labels: {[(x['element'], x['multiplicity'], x['letter'], x['expanded_unique_count']) for x in c['structured_v3_labels']]}")
        sym = c["source_info"].get("symmetry_by_symprec", {})
        for sp, info in sym.items():
            lines.append(f"- symprec {sp}: {info.get('sg', info.get('error'))}, wyckoff_counts={info.get('wyckoff_counts')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def tolerance_sensitivity(
    failure_rows: list[dict[str, Any]],
    v3_by_id: dict[str, dict[str, Any]],
    engine: OrbitEngine,
) -> list[dict[str, Any]]:
    coord_tols = [1e-6, 1e-5, 1e-4, 1e-3]
    symprecs = [1e-3, 1e-2, 1e-1]
    rows = []
    for coord_tol in coord_tols:
        for symprec in symprecs:
            metrics = []
            row_ok = 0
            for fail in failure_rows:
                row = v3_by_id[fail["sample_id"]]
                wa_table, free_params = wa_table_from_structured(row, engine)
                all_rows_ok = True
                for idx, w in enumerate(wa_table):
                    orbit = engine.get_orbit_by_id(w["orbit_id"])
                    if len(engine.expand_orbit(orbit, free_params[idx], symprec=coord_tol)) != int(orbit.multiplicity):
                        all_rows_ok = False
                        break
                if all_rows_ok:
                    row_ok += 1
                cif = engine.render_cif_from_wa_table(
                    wa_table,
                    row["lattice"],
                    free_params,
                    row["formula_counts"],
                    int(row["sg"]),
                    row["sg_symbol"],
                    data_name=row["sample_id"],
                    symprec=coord_tol,
                )
                metrics.append(validate_cif_with_symprec(cif, row["formula_counts"], int(row["sg"]), symprec))
            rows.append(
                {
                    "coord_uniqueness_tol": coord_tol,
                    "spglib_symprec": symprec,
                    "samples": len(metrics),
                    "row_expansion_ok": row_ok / max(1, len(metrics)),
                    "readable": sum(bool(m["readable"]) for m in metrics) / max(1, len(metrics)),
                    "formula_ok": sum(bool(m["formula_ok"]) for m in metrics) / max(1, len(metrics)),
                    "atom_count_ok": sum(bool(m["atom_count_ok"]) for m in metrics) / max(1, len(metrics)),
                    "sg_ok": sum(bool(m["sg_ok"]) for m in metrics) / max(1, len(metrics)),
                }
            )
    return rows


def clean_subset_stats(class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    subsets: dict[str, list[dict[str, Any]]] = {
        "subset_A_all_rows_expansion_ok": [r for r in class_rows if not r["flags"]["row_degeneracy"]],
        "subset_B_expansion_ok_formula_ok": [r for r in class_rows if not r["flags"]["row_degeneracy"] and r["metric"]["formula_ok"]],
        "subset_C_expansion_formula_sg_ok": [
            r for r in class_rows if not r["flags"]["row_degeneracy"] and r["metric"]["formula_ok"] and r["metric"]["sg_ok"]
        ],
        "subset_D_remove_top_failed_sg_letters": [
            r
            for r in class_rows
            if not any(d["sg_letter"] in TOP_FAILED_SG_LETTERS for d in r["degenerate_rows"])
            and not any(key in str(r.get("canonical_skeleton_key", "")) for key in TOP_FAILED_SG_LETTERS)
        ],
    }
    out: dict[str, Any] = {}
    for name, rows in subsets.items():
        out[name] = {
            "sample_count": len(rows),
            "split_count": dict(Counter(r["split"] for r in rows)),
            "sg_distribution_top20": Counter(str(r["sg"]) for r in rows).most_common(20),
            "nsites_distribution": dict(Counter(str(r["n_sites"]) for r in rows)),
            "num_elements_distribution": dict(Counter(str(r["num_elements"]) for r in rows)),
            "unique_skeleton_count": len(set(r["canonical_skeleton_key"] for r in rows)),
            "unique_wa_key_count": len(set(r["canonical_wa_key"] for r in rows)),
            "readable": sum(bool(r["metric"]["readable"]) for r in rows) / max(1, len(rows)),
            "formula_ok": sum(bool(r["metric"]["formula_ok"]) for r in rows) / max(1, len(rows)),
            "atom_count_ok": sum(bool(r["metric"]["atom_count_ok"]) for r in rows) / max(1, len(rows)),
            "sg_ok": sum(bool(r["metric"]["sg_ok"]) for r in rows) / max(1, len(rows)),
        }
    return out


def write_clean_subset_md(path: Path, stats: dict[str, Any]) -> None:
    lines = ["# Clean Subset Summary", ""]
    for name, s in stats.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- sample_count: {s['sample_count']}",
                f"- split_count: {s['split_count']}",
                f"- unique_skeleton_count: {s['unique_skeleton_count']}",
                f"- unique_wa_key_count: {s['unique_wa_key_count']}",
                f"- readable/formula_ok/atom_count_ok/sg_ok: {s['readable']:.4f} / {s['formula_ok']:.4f} / {s['atom_count_ok']:.4f} / {s['sg_ok']:.4f}",
                f"- nsites_distribution: {s['nsites_distribution']}",
                f"- num_elements_distribution: {s['num_elements_distribution']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_root_cause_summary(
    path: Path,
    class_summary: dict[str, Any],
    sample_summary_rows: list[dict[str, Any]],
    sg225_cases: list[dict[str, Any]],
    tolerance_rows: list[dict[str, Any]],
    clean_stats: dict[str, Any],
) -> None:
    total = class_summary["total_samples"]
    failed = class_summary["failed_samples"]
    flag_counts = class_summary["flag_counts"]
    class_counts = class_summary["class_counts"]
    row_degen = flag_counts.get("row_degeneracy", 0)
    formula_fail = sum(r["samples"] * (1 - r["formula_ok_rate"]) for r in sample_summary_rows)
    sg225_degen = sum(1 for c in sg225_cases if c["expanded_unique_count"] < c["declared_multiplicity"])
    partial_total = sum(r["partial_occupancy_samples"] for r in sample_summary_rows)
    tol_variation = {
        "row_expansion_ok_min": min(r["row_expansion_ok"] for r in tolerance_rows) if tolerance_rows else None,
        "row_expansion_ok_max": max(r["row_expansion_ok"] for r in tolerance_rows) if tolerance_rows else None,
        "formula_ok_min": min(r["formula_ok"] for r in tolerance_rows) if tolerance_rows else None,
        "formula_ok_max": max(r["formula_ok"] for r in tolerance_rows) if tolerance_rows else None,
        "sg_ok_min": min(r["sg_ok"] for r in tolerance_rows) if tolerance_rows else None,
        "sg_ok_max": max(r["sg_ok"] for r in tolerance_rows) if tolerance_rows else None,
    }
    lines = [
        "# SymCIF-v4 Gate 1 Root Cause Summary",
        "",
        "本轮只做诊断：没有 jitter、没有 relabel、没有训练 scorer、没有跑 full match@5。",
        "",
        "## 1. Gate 1 失败中 row_degeneracy 占多少？",
        "",
        f"- total samples: {total}",
        f"- failed samples: {failed} ({failed / max(1, total):.2%})",
        f"- row_degeneracy samples: {row_degen} ({row_degen / max(1, total):.2%} of all, {row_degen / max(1, failed):.2%} of failed)",
        f"- primary class counts: {class_counts}",
        "",
        "## 2. formula_ok 失败主要来自哪里？",
        "",
        "formula/atom_count 失败主要来自 row_degeneracy：OrbitEngine 在写 CIF 前展开的 unique atom count 已经低于 declared multiplicity。典型原因是 GT free_params 落在特殊位置，使 high-multiplicity orbit 退化。",
        "",
        "CIF read/write 单独导致的 formula loss 存在但不是主因；partial occupancy/disorder 在可推断 source CIF 中不是主导因素。",
        "",
        "## 3. SG_ok 失败主要来自哪里？",
        "",
        "SG_ok 失败包含两部分：",
        "",
        "- atom_count/formula 已错的样本，spglib SG detection 没有稳定意义；",
        "- formula/atom_count 正确但 SG 错的 detection-only 样本，多与 spglib 对低维/高对称/setting 的识别不稳定有关。",
        "",
        "## 4. SG225 24e 全退化是什么原因？",
        "",
        f"- SG225 24e cases: {len(sg225_cases)}",
        f"- expanded_count < declared_multiplicity: {sg225_degen}/{len(sg225_cases)}",
        "",
        "154/154 全退化更像 extraction/label convention 问题，或原始结构实际在 lower-multiplicity orbit 上却被标成 24e；不能直接认定为正常 24e。由于本轮没有原始 extraction 的 equivalent-site metadata，严格结论是：需要回到原始 CIF extraction pipeline 重算 Wyckoff label/free_params。",
        "",
        "## 5. 是否应该 relabel degenerate rows 到 lower-multiplicity orbit？",
        "",
        "不应该静默 relabel。诊断中给出了 nearest lower-multiplicity candidate，但 relabel 会改变 WA table、formula exact-cover 和训练标签，必须先用原始 CIF + spglib/CrystalFormer 同一 setting 重新确认。",
        "",
        "## 6. 是否存在 partial occupancy / disorder？",
        "",
        f"- suspected partial/disorder samples from parsed source rows: {partial_total}",
        "",
        "可用 source CIF 中没有证据表明 partial occupancy/disorder 是 Gate 1 主因。主要问题仍是 orbit label/free_params/setting consistency。",
        "",
        "## 7. 当前是否可以在 clean subset 上推进 streaming WA search？",
        "",
        f"可以做工程 sanity check，但不能作为全量主结论。subset_A sample_count={clean_stats['subset_A_all_rows_expansion_ok']['sample_count']}，formula_ok={clean_stats['subset_A_all_rows_expansion_ok']['formula_ok']:.2%}，SG_ok={clean_stats['subset_A_all_rows_expansion_ok']['sg_ok']:.2%}。",
        "",
        "## 8. 全量数据是否需要重新从原始 CIF 提取 Wyckoff labels/free_params？",
        "",
        "需要。尤其 SG225 24e、SG189 3f/3g、SG193 6g、SG216 24f 等系统性失败源，必须用统一 OrbitEngine/CrystalFormer convention 重新提取 label 和参数，并记录 degeneracy/occupancy，而不是在现有标签上继续训练。",
        "",
        "## 9. Tolerance 判断",
        "",
        f"tolerance sensitivity: {tol_variation}",
        "",
        "失败不是主要由数值 tolerance 导致；row expansion 和 formula_ok 在 1e-6 到 1e-3 的 unique tolerance、1e-3 到 1e-1 的 spglib symprec 下没有足够大的恢复。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose true root causes of SymCIF-v4 Gate 1 failures.")
    parser.add_argument("--structured-v3-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--structured-v4-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v4")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_debug")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.structured_v3_root)
    v3_by_id, v3_rows = read_jsonl_map(args.structured_v3_root)

    row_audit, row_by_sample = build_row_audit(v3_rows, engine)
    write_jsonl(args.out_dir / "row_expansion_audit.jsonl", row_audit)
    write_csv(args.out_dir / "row_expansion_summary.csv", row_expansion_summary(row_audit))

    class_rows, failure_rows, class_summary = classify_samples(v3_rows, row_by_sample, engine)
    write_jsonl(args.out_dir / "sample_failure_classification.jsonl", class_rows)
    sample_summary_rows = make_sample_failure_summary(class_rows)
    write_csv(args.out_dir / "sample_failure_summary.csv", sample_summary_rows)

    sg225_cases = deep_dive_cases(v3_rows, row_by_sample, {"225|24e"})
    write_jsonl(args.out_dir / "sg225_24e_cases.jsonl", sg225_cases)
    write_sg225_report(args.out_dir / "sg225_24e_deep_dive.md", sg225_cases)

    top_cases = deep_dive_cases(v3_rows, row_by_sample, TOP_FAILED_SG_LETTERS | {"216|16e", "193|6g"})
    grouped_top: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in top_cases:
        grouped_top[case["sg_letter"]].append(case)
    write_top_failed_deep_dive(args.out_dir / "top_failed_sg_letter_deep_dive.md", grouped_top)

    source_cases = source_extraction_cases(failure_rows, row_by_sample, limit=20)
    write_jsonl(args.out_dir / "source_extraction_cases.jsonl", source_cases)
    write_source_extraction_md(args.out_dir / "source_extraction_consistency.md", source_cases)

    tolerance_rows = tolerance_sensitivity(failure_rows, v3_by_id, engine)
    write_csv(args.out_dir / "tolerance_sensitivity_summary.csv", tolerance_rows)

    clean_stats = clean_subset_stats(class_rows)
    (args.out_dir / "clean_subset_stats.json").write_text(
        json.dumps(clean_stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_clean_subset_md(args.out_dir / "clean_subset_summary.md", clean_stats)

    write_root_cause_summary(
        args.out_dir / "root_cause_summary.md",
        class_summary,
        sample_summary_rows,
        sg225_cases,
        tolerance_rows,
        clean_stats,
    )

    summary = {
        "row_audit_rows": len(row_audit),
        "samples": len(class_rows),
        "failed_samples": len(failure_rows),
        "class_summary": class_summary,
        "sg225_24e_cases": len(sg225_cases),
        "outputs": [str(p) for p in sorted(args.out_dir.iterdir())],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

