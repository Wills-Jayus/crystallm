#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import signal
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.canonicalize import canonical_skeleton_key, canonical_wa_key
from symcif_v4.free_param_extractor import extract_free_params_detailed
from symcif_v4.orbit_engine import OrbitEngine


CRYSTALLM_ROOT = Path("/data/users/xsw/autodlmini/model/scp_task/CrystaLLM")
BENCHMARK_CSV_ROOT = CRYSTALLM_ROOT / "resources" / "benchmarks"
PREPARED_TEST_CIF_ROOT = (
    CRYSTALLM_ROOT / "reproduce" / "benchmarks_gt_from_prepare_csv_benchmark_symprec0p1"
)
DATASET_TO_CSV_NAME = {"mp20": "mp_20", "mpts52": "mpts_52"}
DATASET_TO_OUT_NAME = {"mp20": "structured_symcif_v4_mp20", "mpts52": "structured_symcif_v4_mpts52"}


_ENGINE: OrbitEngine | None = None
_ARGS: dict[str, Any] = {}


class RecordTimeout(Exception):
    pass


def _raise_record_timeout(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise RecordTimeout("record_timeout")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def formula_counts_from_structure(structure: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for el, amount in structure.composition.as_dict().items():
        value = float(amount)
        rounded = int(round(value))
        if not math.isclose(value, rounded, abs_tol=1e-5):
            raise ValueError(f"non_integer_composition:{el}={value}")
        counts[str(el)] = rounded
    return dict(sorted(counts.items()))


def formula_string(counts: dict[str, int]) -> str:
    return " ".join(f"{el}{int(count)}" for el, count in sorted(counts.items()))


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_") or "sample"


def leading_multiplicity(wyckoff: str) -> int:
    match = re.match(r"^\s*(\d+)", str(wyckoff))
    if not match:
        raise ValueError(f"bad_wyckoff_symbol:{wyckoff}")
    return int(match.group(1))


def wyckoff_letter(wyckoff: str) -> str:
    match = re.match(r"^\s*\d+\s*([A-Za-z]+)", str(wyckoff))
    if not match:
        raise ValueError(f"bad_wyckoff_symbol:{wyckoff}")
    return match.group(1)


def fallback_params_from_coord(coord: tuple[float, float, float], free_symbols: list[str] | tuple[str, ...]) -> dict[str, float]:
    by_symbol = {"x": coord[0], "y": coord[1], "z": coord[2]}
    return {str(symbol): float(by_symbol.get(str(symbol), 0.0)) % 1.0 for symbol in free_symbols}


def single_element(site: Any) -> str:
    if not getattr(site, "is_ordered", True):
        raise ValueError("occupancy/disorder issue")
    species = list(site.species.items())
    if len(species) != 1:
        raise ValueError("occupancy/disorder issue")
    specie, occu = species[0]
    if not math.isclose(float(occu), 1.0, abs_tol=1e-5):
        raise ValueError("occupancy/disorder issue")
    return str(specie.symbol)


def symmetrized_groups(sym: Any) -> list[tuple[list[Any], str]]:
    groups = list(sym.equivalent_sites)
    wyckoffs = list(sym.wyckoff_symbols)
    if len(wyckoffs) == len(groups):
        return [(list(sites), str(wyckoff)) for sites, wyckoff in zip(groups, wyckoffs)]
    indices = list(getattr(sym, "equivalent_indices", []))
    if indices and len(wyckoffs) == len(sym):
        out = []
        for sites, idx_group in zip(groups, indices):
            out.append((list(sites), str(wyckoffs[int(idx_group[0])])))
        return out
    raise ValueError(f"wyckoff_group_shape_mismatch:groups={len(groups)} wyckoffs={len(wyckoffs)}")


def parse_structure(cif_text: str) -> Any:
    parser = CifParser.from_string(cif_text)
    if hasattr(parser, "parse_structures"):
        structures = parser.parse_structures(primitive=False)
    else:
        structures = parser.get_structures(primitive=False)
    if not structures:
        raise ValueError("parse fail:no_structure")
    return structures[0]


def read_csv_rows(dataset_key: str, split: str) -> list[dict[str, Any]]:
    csv_name = DATASET_TO_CSV_NAME[dataset_key]
    path = BENCHMARK_CSV_ROOT / csv_name / f"{split}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepared_test_cif_path(dataset_key: str, material_id: str) -> Path | None:
    csv_name = DATASET_TO_CSV_NAME[dataset_key]
    path = PREPARED_TEST_CIF_ROOT / f"{csv_name}_test_orig" / f"{material_id}.cif"
    return path if path.exists() else None


def init_worker(lookup_json: str, args_json: str) -> None:
    global _ENGINE, _ARGS
    _ENGINE = OrbitEngine(lookup_json)
    _ARGS = json.loads(args_json)


def process_one(payload: tuple[str, str, int, dict[str, Any]]) -> dict[str, Any]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    engine = _ENGINE
    dataset_key, split, input_index, csv_row = payload
    started = time.monotonic()
    material_id = str(csv_row.get("material_id") or csv_row.get("") or input_index)
    sample_prefix = f"{DATASET_TO_CSV_NAME[dataset_key]}_{split}"
    if split == "test":
        sample_prefix = f"{DATASET_TO_CSV_NAME[dataset_key]}_test_orig"
    sample_id = f"{sample_prefix}__{material_id}"
    result: dict[str, Any] = {
        "dataset": dataset_key,
        "split": split,
        "input_index": input_index,
        "material_id": material_id,
        "sample_id": sample_id,
        "ok": False,
        "record": None,
        "failure": None,
        "conventional_cif": None,
        "elapsed_s": None,
    }
    timeout_s = float(_ARGS.get("record_timeout_seconds") or 0.0)
    old_handler = None
    if timeout_s > 0:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_record_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        cif_source_kind = "csv"
        cif_text = str(csv_row.get("cif") or "")
        if split == "test" and bool(_ARGS.get("prefer_prepared_test_cif", True)):
            prepared = prepared_test_cif_path(dataset_key, material_id)
            if prepared is not None:
                cif_text = prepared.read_text(encoding="utf-8", errors="replace")
                cif_source_kind = "prepared_test_orig"
        if not cif_text.strip():
            raise ValueError("parse fail:empty_cif")
        try:
            structure = parse_structure(cif_text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"parse fail:{type(exc).__name__}:{exc}") from exc

        try:
            sga = SpacegroupAnalyzer(
                structure,
                symprec=float(_ARGS.get("symprec", 0.1)),
                angle_tolerance=float(_ARGS.get("angle_tolerance", 5.0)),
            )
            sg = int(sga.get_space_group_number())
            sg_symbol = str(sga.get_space_group_symbol())
            conventional = sga.get_conventional_standard_structure()
            conv_sga = SpacegroupAnalyzer(
                conventional,
                symprec=float(_ARGS.get("symprec", 0.1)),
                angle_tolerance=float(_ARGS.get("angle_tolerance", 5.0)),
            )
            sg = int(conv_sga.get_space_group_number())
            sg_symbol = str(conv_sga.get_space_group_symbol())
            sym = conv_sga.get_symmetrized_structure()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"SG detect fail:{type(exc).__name__}:{exc}") from exc

        formula_counts = formula_counts_from_structure(conventional)
        groups = symmetrized_groups(sym)
        wa_table: list[dict[str, Any]] = []
        free_param_rows: list[dict[str, Any]] = []
        for site_order, (sites, wyckoff) in enumerate(groups):
            if not sites:
                raise ValueError("WA extraction fail:empty_equivalent_site_group")
            element = single_element(sites[0])
            for site in sites[1:]:
                if single_element(site) != element:
                    raise ValueError("WA extraction fail:mixed_element_equivalent_group")
            mult = leading_multiplicity(wyckoff)
            letter = wyckoff_letter(wyckoff)
            if mult != len(sites):
                raise ValueError(f"WA extraction fail:multiplicity_group_size_mismatch:{wyckoff}:{len(sites)}")
            try:
                orbit = engine.get_orbit(int(sg), letter)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"OrbitEngine fail:sg={sg} letter={letter}:{type(exc).__name__}:{exc}") from exc
            extraction_options: list[Any] = []
            for site in sites:
                coord = tuple(float(x) % 1.0 for x in site.frac_coords)
                option = extract_free_params_detailed(coord, orbit, tolerance=float(_ARGS.get("free_param_tolerance", 1e-4)))
                if option is not None:
                    extraction_options.append(option)
            extraction_options.sort(
                key=lambda item: (
                    not bool(item.expansion_ok),
                    float(item.extraction_residual),
                    sum(float(item.free_params.get(symbol, 0.0)) for symbol in ("x", "y", "z")),
                    str(item.matched_operation),
                )
            )
            extraction = extraction_options[0] if extraction_options else None
            fallback_used = False
            fallback_reason = None
            if extraction is None:
                fallback_used = True
                fallback_reason = f"free_params extraction fail:sg={sg} wyckoff={wyckoff} element={element}"
                source_coord = tuple(float(x) % 1.0 for x in sites[0].frac_coords)
                fallback_params = fallback_params_from_coord(source_coord, orbit.free_symbols)
                expanded = engine.expand_orbit(orbit, fallback_params, symprec=float(_ARGS.get("free_param_tolerance", 1e-4)))
                extraction = type(
                    "FallbackExtraction",
                    (),
                    {
                        "free_params": fallback_params,
                        "source_coord": source_coord,
                        "mapped_coord": source_coord,
                        "representative_coord": engine.evaluate_representative(orbit, fallback_params),
                        "matched_operation": "fallback_source_coord_by_symbol",
                        "extraction_residual": None,
                        "extraction_method": "fallback_source_coord_by_symbol",
                        "expansion_count_after_reextract": len(expanded),
                        "expansion_ok": len(expanded) == int(mult),
                    },
                )()
            if not extraction.expansion_ok:
                raise ValueError(
                    f"free_params extraction fail:expansion_not_ok sg={sg} wyckoff={wyckoff} "
                    f"expanded={extraction.expansion_count_after_reextract} mult={mult}"
                )
            source_coord = tuple(float(x) % 1.0 for x in extraction.source_coord)
            row = {
                "element": element,
                "orbit_id": orbit.canonical_orbit_id,
                "sg": int(sg),
                "letter": orbit.letter,
                "multiplicity": int(orbit.multiplicity),
                "site_symmetry": orbit.site_symmetry,
                "enumeration": orbit.enumeration,
                "representative_expr": list(orbit.representative_expr),
                "free_symbols": list(orbit.free_symbols),
                "free_params": {str(k): float(v) for k, v in extraction.free_params.items()},
                "setting_id": orbit.setting_id,
                "hall_number": orbit.hall_number,
                "origin_shift": orbit.origin_shift,
                "basis_transform": orbit.basis_transform,
                "source_coord": list(source_coord),
                "mapped_coord": list(extraction.mapped_coord),
                "extraction_residual": None if extraction.extraction_residual is None else float(extraction.extraction_residual),
                "extraction_method": extraction.extraction_method,
                "expansion_count_after_reextract": int(extraction.expansion_count_after_reextract),
                "declared_multiplicity": int(mult),
                "expansion_ok": bool(extraction.expansion_ok),
                "extraction_success": not fallback_used,
                "fallback_reason": fallback_reason,
            }
            wa_table.append(row)
            free_param_rows.append(
                {
                    "site_order": int(site_order),
                    "element": element,
                    "sg_letter": f"{int(sg)}|{wyckoff}",
                    "orbit_id": orbit.canonical_orbit_id,
                    "representative_expr": list(orbit.representative_expr),
                    "new_free_params": dict(row["free_params"]),
                    "source_coord": list(source_coord),
                    "mapped_coord": list(extraction.mapped_coord),
                    "representative_coord": list(extraction.representative_coord),
                    "matched_operation": extraction.matched_operation,
                    "extraction_residual": None if extraction.extraction_residual is None else float(extraction.extraction_residual),
                    "extraction_method": extraction.extraction_method,
                    "extraction_success": not fallback_used,
                    "fallback_reason": fallback_reason,
                    "expansion_count_after_reextract": int(extraction.expansion_count_after_reextract),
                    "declared_multiplicity": int(mult),
                    "expansion_ok": bool(extraction.expansion_ok),
                }
            )

        expanded_counts: Counter[str] = Counter()
        for row in wa_table:
            expanded_counts[str(row["element"])] += int(row["multiplicity"])
        if dict(sorted(expanded_counts.items())) != dict(sorted(formula_counts.items())):
            raise ValueError(f"formula mismatch:wa={dict(sorted(expanded_counts.items()))} formula={formula_counts}")

        lattice = conventional.lattice
        record = {
            "id": sample_id,
            "sample_id": sample_id,
            "dataset": dataset_key,
            "split": split,
            "input_index": int(input_index),
            "material_id": material_id,
            "formula": formula_string(formula_counts),
            "pretty_formula": csv_row.get("pretty_formula"),
            "formula_counts": formula_counts,
            "sg": int(sg),
            "sg_symbol": sg_symbol,
            "wa_table": wa_table,
            "lattice": {
                "a": float(lattice.a),
                "b": float(lattice.b),
                "c": float(lattice.c),
                "alpha": float(lattice.alpha),
                "beta": float(lattice.beta),
                "gamma": float(lattice.gamma),
                "volume": float(lattice.volume),
            },
            "canonical_wa_key": canonical_wa_key(wa_table),
            "canonical_skeleton_key": canonical_skeleton_key(wa_table),
            "legacy_skeleton_template_key": "|".join(f"{int(r['multiplicity'])}{r['letter']}" for r in wa_table),
            "n_sites": int(len(wa_table)),
            "num_elements": int(len(formula_counts)),
            "atom_count": int(sum(formula_counts.values())),
            "source_path": None,
            "cif_source_kind": cif_source_kind,
            "free_param_reextract_rows": free_param_rows,
            "free_param_reextract_all_success": all(bool(row["extraction_success"]) for row in wa_table),
            "row_expansion_all_ok": True,
        }
        result["ok"] = True
        result["record"] = record
        result["conventional_cif"] = str(CifWriter(conventional, symprec=float(_ARGS.get("symprec", 0.1))))
    except RecordTimeout:
        result["failure"] = {
            "dataset": dataset_key,
            "split": split,
            "input_index": int(input_index),
            "material_id": material_id,
            "sample_id": sample_id,
            "reason": f"record_timeout>{timeout_s:g}s",
            "error_type": "record_timeout",
        }
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {
            "dataset": dataset_key,
            "split": split,
            "input_index": int(input_index),
            "material_id": material_id,
            "sample_id": sample_id,
            "reason": str(exc),
            "error_type": str(exc).split(":", 1)[0],
        }
    finally:
        if timeout_s > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
    result["elapsed_s"] = time.monotonic() - started
    return result


def q(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "p90": None, "max": None}
    vals = sorted(float(v) for v in values)
    return {
        "min": vals[0],
        "median": float(statistics.median(vals)),
        "mean": float(statistics.mean(vals)),
        "p90": vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
        "max": vals[-1],
    }


def split_audit(records: list[dict[str, Any]], input_count: int, failures: list[dict[str, Any]]) -> dict[str, Any]:
    sg_counter = Counter(str(int(r["sg"])) for r in records)
    n_sites = [int(r["n_sites"]) for r in records]
    num_elements = [int(r["num_elements"]) for r in records]
    atom_counts = [int(r["atom_count"]) for r in records]
    orbit_counts = [len(r["wa_table"]) for r in records]
    wa_rows = [row for record in records for row in record["wa_table"]]
    row_extraction_success = sum(1 for row in wa_rows if row.get("extraction_success"))
    row_fallback = sum(1 for row in wa_rows if not row.get("extraction_success"))
    sample_all_extraction_success = sum(1 for row in records if row.get("free_param_reextract_all_success"))
    return {
        "input_count": int(input_count),
        "structured_success": int(len(records)),
        "structured_success_rate": len(records) / max(1, input_count),
        "failures": int(len(failures)),
        "failure_reasons": dict(Counter(str(f.get("error_type")) for f in failures).most_common()),
        "unique_sg": int(len(sg_counter)),
        "sg_distribution_top30": [{"sg": k, "count": v} for k, v in sg_counter.most_common(30)],
        "n_sites_distribution": q(n_sites),
        "num_elements_distribution": q(num_elements),
        "atom_count_distribution": q(atom_counts),
        "orbit_count_distribution": q(orbit_counts),
        "wa_rows": int(len(wa_rows)),
        "row_free_param_extraction_success": int(row_extraction_success),
        "row_free_param_extraction_success_rate": row_extraction_success / max(1, len(wa_rows)),
        "row_free_param_fallback": int(row_fallback),
        "sample_all_free_param_extraction_success": int(sample_all_extraction_success),
        "sample_all_free_param_extraction_success_rate": sample_all_extraction_success / max(1, len(records)),
        "n_sites_ge6_rate": sum(1 for v in n_sites if v >= 6) / max(1, len(n_sites)),
        "n_sites_ge12_rate": sum(1 for v in n_sites if v >= 12) / max(1, len(n_sites)),
        "n_sites_ge20_rate": sum(1 for v in n_sites if v >= 20) / max(1, len(n_sites)),
        "num_elements_ge4_rate": sum(1 for v in num_elements if v >= 4) / max(1, len(num_elements)),
    }


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def write_audit_markdown(path: Path, dataset_key: str, audit: dict[str, Any]) -> None:
    lines = [
        f"# SymCIF-v4 benchmark data audit: {dataset_key}",
        "",
        "## split summary",
        "",
        "| split | input | structured | success | failures | row free-param success | row fallback | n_sites>=6 | n_sites>=12 | n_sites>=20 | num_elements>=4 | unique SG |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        row = audit["splits"][split]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(row["input_count"]),
                    str(row["structured_success"]),
                    pct(row["structured_success_rate"]),
                    str(row["failures"]),
                    pct(row["row_free_param_extraction_success_rate"]),
                    str(row["row_free_param_fallback"]),
                    pct(row["n_sites_ge6_rate"]),
                    pct(row["n_sites_ge12_rate"]),
                    pct(row["n_sites_ge20_rate"]),
                    pct(row["num_elements_ge4_rate"]),
                    str(row["unique_sg"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## notes",
            "",
            "- Official benchmark `train.csv`, `val.csv`, `test.csv` are kept as disjoint splits.",
            "- Structured records use conventional cell composition; reduced formula is not used as the exact-cover target.",
            "- SG number and Wyckoff groups are extracted with `symprec=0.1`.",
            "- Reports using these records must mark SymCIF-v4 input as `cell composition + oracle GT SG`.",
            "",
            "## artifacts",
            "",
            f"- structured data: `data/{DATASET_TO_OUT_NAME[dataset_key]}/`",
            f"- failed cases: `reports/symcif_v4_benchmark_data_audit/{dataset_key}/failed_cases.jsonl`",
            f"- audit json: `reports/symcif_v4_benchmark_data_audit/{dataset_key}/audit_summary.json`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def process_dataset(args: argparse.Namespace, dataset_key: str) -> None:
    out_root = PROJECT_ROOT / "data" / DATASET_TO_OUT_NAME[dataset_key]
    report_dir = PROJECT_ROOT / "reports" / "symcif_v4_benchmark_data_audit" / dataset_key
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, str, int, dict[str, Any]]] = []
    input_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = read_csv_rows(dataset_key, split)
        if args.max_records_per_split is not None:
            rows = rows[: int(args.max_records_per_split)]
        input_counts[split] = len(rows)
        payloads.extend((dataset_key, split, i, row) for i, row in enumerate(rows))

    worker_args = {
        "symprec": args.symprec,
        "angle_tolerance": args.angle_tolerance,
        "free_param_tolerance": args.free_param_tolerance,
        "prefer_prepared_test_cif": args.prefer_prepared_test_cif,
        "record_timeout_seconds": args.record_timeout_seconds,
    }
    if args.workers <= 1:
        init_worker(str(args.lookup_json), json.dumps(worker_args))
        results = [process_one(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=int(args.workers),
            initializer=init_worker,
            initargs=(str(args.lookup_json), json.dumps(worker_args)),
        ) as pool:
            results = list(pool.imap_unordered(process_one, payloads, chunksize=max(1, int(args.chunksize))))

    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_failures: list[dict[str, Any]] = []
    for result in results:
        split = str(result["split"])
        if result["ok"]:
            record = dict(result["record"])
            cif_dir = out_root / "cifs" / split
            cif_dir.mkdir(parents=True, exist_ok=True)
            cif_path = cif_dir / f"{safe_id(record['sample_id'])}.cif"
            cif_path.write_text(str(result["conventional_cif"]), encoding="utf-8")
            record["source_path"] = str(cif_path)
            records_by_split[split].append(record)
        else:
            failure = dict(result["failure"])
            failure["elapsed_s"] = result.get("elapsed_s")
            failures_by_split[split].append(failure)
            all_failures.append(failure)

    all_records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        records = sorted(records_by_split.get(split, []), key=lambda row: int(row["input_index"]))
        all_records.extend(records)
        write_jsonl(out_root / f"{split}.jsonl", records)
        write_jsonl(report_dir / f"{split}_failed_cases.jsonl", sorted(failures_by_split.get(split, []), key=lambda row: int(row["input_index"])))
    write_jsonl(out_root / "all.jsonl", sorted(all_records, key=lambda row: (str(row["split"]), int(row["input_index"]))))
    write_jsonl(report_dir / "failed_cases.jsonl", sorted(all_failures, key=lambda row: (str(row["split"]), int(row["input_index"]))))

    audit = {
        "dataset": dataset_key,
        "csv_root": str(BENCHMARK_CSV_ROOT / DATASET_TO_CSV_NAME[dataset_key]),
        "prepared_test_cif_root": str(PREPARED_TEST_CIF_ROOT / f"{DATASET_TO_CSV_NAME[dataset_key]}_test_orig"),
        "out_root": str(out_root),
        "config": {
            "symprec": args.symprec,
            "angle_tolerance": args.angle_tolerance,
            "free_param_tolerance": args.free_param_tolerance,
            "prefer_prepared_test_cif": args.prefer_prepared_test_cif,
            "record_timeout_seconds": args.record_timeout_seconds,
            "workers": args.workers,
            "max_records_per_split": args.max_records_per_split,
        },
        "splits": {
            split: split_audit(
                sorted(records_by_split.get(split, []), key=lambda row: int(row["input_index"])),
                input_counts.get(split, 0),
                failures_by_split.get(split, []),
            )
            for split in ("train", "val", "test")
        },
    }
    write_json(report_dir / "audit_summary.json", audit)
    write_audit_markdown(report_dir / "summary.md", dataset_key, audit)
    print(
        json.dumps(
            {
                "dataset": dataset_key,
                "out_root": str(out_root),
                "report_dir": str(report_dir),
                "splits": {
                    split: {
                        "input": audit["splits"][split]["input_count"],
                        "structured": audit["splits"][split]["structured_success"],
                        "success_rate": audit["splits"][split]["structured_success_rate"],
                    }
                    for split in ("train", "val", "test")
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SymCIF-v4 structured benchmark data for MP-20/MPTS-52.")
    parser.add_argument("--datasets", nargs="+", default=["mp20", "mpts52"], choices=sorted(DATASET_TO_CSV_NAME))
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    parser.add_argument("--free-param-tolerance", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--max-records-per-split", type=int, default=None)
    parser.add_argument("--prefer-prepared-test-cif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for dataset_key in args.datasets:
        process_dataset(args, dataset_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
