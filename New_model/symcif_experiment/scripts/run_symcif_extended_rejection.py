#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_generation_eval import (  # noqa: E402
    evaluate_mode_with_hard_timeouts,
    extract_generated_record,
    fmt_num,
    fmt_pct,
    generate_batch_mode_aware,
    load_model,
    load_test_cases,
    time_limit,
)
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.models import LatticeParameters, SymCifRecord, WyckoffSite  # noqa: E402
from symcif.render import render_standard_cif  # noqa: E402


MODE = "symcif_v1"
GATE_NAMES = (
    "schema_invalid",
    "sg_invalid",
    "sg_not_equal_prompt",
    "wyckoff_invalid",
    "coord_mask_invalid",
    "formula_not_closed",
    "cell_invalid",
    "render_failed",
    "pymatgen_unreadable",
)

SITE_HEADERS = [
    "_wyckoff_site_index",
    "_wyckoff_site_element",
    "_wyckoff_site_multiplicity",
    "_wyckoff_site_letter",
    "_wyckoff_site_symmetry",
    "_wyckoff_site_enumeration",
    "_wyckoff_free_x",
    "_wyckoff_free_y",
    "_wyckoff_free_z",
]

SCALAR_KEYS = {
    "_chemical_formula_sum",
    "_symmetry_Int_Tables_number",
    "_symmetry_space_group_name_H-M",
    "_cell_formula_units_Z",
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_cell_volume",
}


class GateError(ValueError):
    def __init__(
        self,
        gate: str,
        reason: str,
        *,
        generated_formula: str | None = None,
        formula_diff: dict[str, int] | None = None,
    ) -> None:
        super().__init__(reason)
        self.gate = gate
        self.reason = reason
        self.generated_formula = generated_formula
        self.formula_diff = formula_diff


@dataclass(frozen=True)
class ParsedSymCifText:
    data_name: str
    values: dict[str, str]
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class AcceptedInfo:
    standard_cif: str
    generated_formula: str
    valid_precheck: bool | None = None


def clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("<unk>", "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(ord(ch) < 9 for ch in line):
            continue
        lines.append(line)
    return lines


def parse_generated_text(text: str) -> ParsedSymCifText:
    lines = clean_lines(text)
    data_idx = next((i for i, line in enumerate(lines) if line.startswith("data_")), None)
    if data_idx is None:
        raise GateError("schema_invalid", "missing_data_line")
    lines = lines[data_idx:]
    data_name = lines[0].removeprefix("data_") or "unknown"

    values: dict[str, str] = {}
    headers: list[str] = []
    rows: list[list[str]] = []
    i = 1
    while i < len(lines):
        line = lines[i]
        if line == "loop_":
            j = i + 1
            loop_headers: list[str] = []
            while j < len(lines) and lines[j].startswith("_"):
                loop_headers.append(lines[j])
                j += 1
            if any(h.startswith("_wyckoff_") for h in loop_headers):
                headers = loop_headers
                while j < len(lines) and not lines[j].startswith("_cell_") and lines[j] != "loop_":
                    rows.append(lines[j].split())
                    j += 1
                i = j
                continue
            i = j
            continue
        if line.startswith("_"):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] in SCALAR_KEYS:
                values[parts[0]] = parts[1].strip().strip("'\"")
        i += 1

    if not headers:
        raise GateError("schema_invalid", "missing_wyckoff_loop")
    missing = [h for h in SITE_HEADERS if h not in headers]
    if missing:
        raise GateError("schema_invalid", f"missing_headers:{','.join(missing)}")
    if not rows:
        raise GateError("schema_invalid", "empty_wyckoff_loop")
    for row in rows:
        if len(row) != len(headers):
            raise GateError("schema_invalid", f"row_header_length_mismatch:{len(row)}!={len(headers)}")
    return ParsedSymCifText(data_name=data_name, values=values, headers=headers, rows=rows)


def parse_int(value: str | None, key: str, gate: str) -> int:
    if value is None:
        raise GateError(gate, f"missing_{key}")
    try:
        return int(float(str(value).strip()))
    except Exception as exc:
        raise GateError(gate, f"invalid_int_{key}:{value}") from exc


def parse_float(value: str | None, key: str) -> float:
    if value is None:
        raise GateError("cell_invalid", f"missing_{key}")
    try:
        out = float(str(value).strip())
    except Exception as exc:
        raise GateError("cell_invalid", f"invalid_float_{key}:{value}") from exc
    if not math.isfinite(out):
        raise GateError("cell_invalid", f"non_finite_{key}:{value}")
    return out


def parse_target_counts(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        raw = Composition(formula).as_dict()
    except Exception as exc:
        raise GateError("schema_invalid", f"target_formula_unparseable:{formula}") from exc
    for element, value in raw.items():
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) > 1e-6:
            raise GateError("schema_invalid", f"target_formula_non_integer:{formula}")
        counts[str(element)] = rounded
    return counts


def formula_from_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{el}{counts[el]}" for el in sorted(counts))


def formula_diff(target: dict[str, int], generated: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(target) | set(generated))
    return {key: generated.get(key, 0) - target.get(key, 0) for key in keys if generated.get(key, 0) != target.get(key, 0)}


def formula_diff_key(diff: dict[str, int]) -> str:
    return ";".join(f"{key}:{value:+d}" for key, value in sorted(diff.items())) or "closed"


def parse_space_group(parsed: ParsedSymCifText, target_sg_number: int | None) -> tuple[int, str]:
    sg_raw = parsed.values.get("_symmetry_Int_Tables_number")
    sg_symbol_raw = parsed.values.get("_symmetry_space_group_name_H-M")
    sg_number: int | None = None
    if sg_raw is not None:
        try:
            sg_number = int(float(sg_raw))
        except Exception as exc:
            raise GateError("sg_invalid", f"invalid_sg_number:{sg_raw}") from exc
    elif sg_symbol_raw:
        normalized = sg_symbol_raw.replace(" ", "")
        for number in range(1, 231):
            try:
                if SpaceGroup.from_int_number(number).symbol.replace(" ", "") == normalized:
                    sg_number = number
                    break
            except Exception:
                continue

    if sg_number is None:
        raise GateError("sg_invalid", "missing_space_group")
    if not 1 <= sg_number <= 230:
        raise GateError("sg_invalid", f"sg_out_of_range:{sg_number}")
    try:
        sg_symbol = SpaceGroup.from_int_number(sg_number).symbol
    except Exception as exc:
        raise GateError("sg_invalid", f"sg_unparseable:{sg_number}") from exc
    if target_sg_number is not None and int(target_sg_number) != sg_number:
        raise GateError("sg_not_equal_prompt", f"generated_sg={sg_number}:target_sg={target_sg_number}")
    return sg_number, sg_symbol


def parse_lattice(values: dict[str, str]) -> LatticeParameters:
    a = parse_float(values.get("_cell_length_a"), "_cell_length_a")
    b = parse_float(values.get("_cell_length_b"), "_cell_length_b")
    c = parse_float(values.get("_cell_length_c"), "_cell_length_c")
    alpha = parse_float(values.get("_cell_angle_alpha"), "_cell_angle_alpha")
    beta = parse_float(values.get("_cell_angle_beta"), "_cell_angle_beta")
    gamma = parse_float(values.get("_cell_angle_gamma"), "_cell_angle_gamma")
    volume = parse_float(values.get("_cell_volume"), "_cell_volume")
    if a <= 0 or b <= 0 or c <= 0:
        raise GateError("cell_invalid", "non_positive_cell_length")
    if volume <= 0:
        raise GateError("cell_invalid", "non_positive_cell_volume")
    if not all(30.0 <= x <= 150.0 for x in (alpha, beta, gamma)):
        raise GateError("cell_invalid", "angle_out_of_range")
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    except Exception as exc:
        raise GateError("cell_invalid", f"lattice_construction_failed:{type(exc).__name__}:{exc}") from exc
    if not math.isfinite(float(lattice.volume)) or float(lattice.volume) <= 0:
        raise GateError("cell_invalid", "computed_volume_invalid")
    return LatticeParameters(
        a=float(a),
        b=float(b),
        c=float(c),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        volume=float(volume),
    )


def accepted_filter(row: dict[str, Any], case: dict[str, Any], lookup: WyckoffLookup) -> AcceptedInfo:
    if not row.get("raw_generation_success"):
        raise GateError("schema_invalid", str(row.get("error") or "raw_generation_failed"))
    parsed = parse_generated_text(row.get("generated_text") or "")
    sg_number, sg_symbol = parse_space_group(parsed, case.get("target_sg_number"))

    header_index = {key: parsed.headers.index(key) for key in SITE_HEADERS}
    sites: list[WyckoffSite] = []
    generated_counts: Counter[str] = Counter()
    for new_index, raw_row in enumerate(parsed.rows, start=1):
        rec = {key: raw_row[idx] for key, idx in header_index.items()}
        element = rec["_wyckoff_site_element"]
        try:
            Element(element)
        except Exception as exc:
            raise GateError("schema_invalid", f"invalid_element:{element}") from exc

        letter = rec["_wyckoff_site_letter"]
        try:
            template = lookup.get(sg_number, letter)
        except Exception as exc:
            raise GateError("wyckoff_invalid", f"invalid_letter:{letter}") from exc

        try:
            generated_mult = int(float(rec["_wyckoff_site_multiplicity"]))
        except Exception as exc:
            raise GateError("wyckoff_invalid", f"invalid_multiplicity:{rec['_wyckoff_site_multiplicity']}") from exc
        if generated_mult != int(template.multiplicity):
            raise GateError(
                "wyckoff_invalid",
                f"multiplicity_mismatch:{letter}:generated={generated_mult}:expected={template.multiplicity}",
            )

        coords: list[float] = []
        for axis, key in enumerate(("_wyckoff_free_x", "_wyckoff_free_y", "_wyckoff_free_z")):
            token = rec[key]
            if template.free_mask[axis]:
                if token.upper() == "FIXED":
                    raise GateError("coord_mask_invalid", f"free_axis_marked_fixed:{letter}:{key}")
                try:
                    value = float(token) % 1.0
                except Exception as exc:
                    raise GateError("coord_mask_invalid", f"invalid_free_coord:{letter}:{key}:{token}") from exc
                if not math.isfinite(value):
                    raise GateError("coord_mask_invalid", f"non_finite_free_coord:{letter}:{key}:{token}")
                coords.append(0.0 if math.isclose(value, 1.0, abs_tol=1e-8) else float(value))
            else:
                if token.upper() != "FIXED":
                    raise GateError("coord_mask_invalid", f"fixed_axis_not_fixed:{letter}:{key}:{token}")
                coords.append(float(template.fixed_values[axis]))

        generated_counts[element] += int(template.multiplicity)
        sites.append(
            WyckoffSite(
                index=new_index,
                element=element,
                multiplicity=int(template.multiplicity),
                letter=letter,
                representative_coord=tuple(coords),  # type: ignore[arg-type]
                free_mask=template.free_mask,
                fixed_values=template.fixed_values,
                site_symmetry=template.site_symmetry,
                enumeration=template.enumeration,
            )
        )

    target_counts = parse_target_counts(str(case["target_formula"]))
    generated = dict(generated_counts)
    diff = formula_diff(target_counts, generated)
    generated_formula = formula_from_counts(generated)
    if diff:
        raise GateError(
            "formula_not_closed",
            f"diff={diff}",
            generated_formula=generated_formula,
            formula_diff=diff,
        )

    lattice = parse_lattice(parsed.values)
    z = parse_int(parsed.values.get("_cell_formula_units_Z"), "_cell_formula_units_Z", "schema_invalid")
    record = SymCifRecord(
        sample_id=str(row.get("sample_id") or case["sample_id"]),
        source_path=None,
        cell_formula=str(case["target_formula"]),
        reduced_formula="",
        sg_number=int(sg_number),
        sg_symbol=sg_symbol,
        z=z,
        lattice=lattice,
        sites=sites,
    )
    try:
        standard_cif = render_standard_cif(record, symprec=0.1)
    except Exception as exc:
        raise GateError("render_failed", f"{type(exc).__name__}:{exc}", generated_formula=generated_formula) from exc
    try:
        Structure.from_str(standard_cif, fmt="cif")
    except Exception as exc:
        raise GateError("pymatgen_unreadable", f"{type(exc).__name__}:{exc}", generated_formula=generated_formula) from exc
    return AcceptedInfo(standard_cif=standard_cif, generated_formula=generated_formula)


def case_to_payload(case: Any) -> dict[str, Any]:
    return {
        "index": case.index,
        "sample_id": case.sample_id,
        "source_path": case.source_path,
        "target_formula": case.target_formula,
        "target_sg_number": case.target_sg_number,
        "target_sg_symbol": case.target_sg_symbol,
        "prompt": case.prompts[MODE],
    }


def extended_generation_worker(
    *,
    model_dir: str,
    cases_payload: list[dict[str, Any]],
    out_dir: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    compile_model: bool,
    worker_id: int,
    seed: int,
    max_raw_attempts: int,
    accepted_k: int,
    generation_batch_size: int,
    lookup_json: str,
) -> None:
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = __import__("crystallm").CIFTokenizer()
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    lookup = WyckoffLookup.from_json(lookup_json)

    worker_dir = Path(out_dir) / "worker_parts"
    worker_dir.mkdir(parents=True, exist_ok=True)
    raw_path = worker_dir / f"symcif_v1_raw.worker{worker_id}.jsonl"
    accepted_path = worker_dir / f"symcif_v1_accepted.worker{worker_id}.jsonl"
    stats_path = worker_dir / f"sample_stats.worker{worker_id}.jsonl"

    with raw_path.open("w", encoding="utf-8") as f_raw, accepted_path.open("w", encoding="utf-8") as f_acc, stats_path.open(
        "w", encoding="utf-8"
    ) as f_stats:
        for local_i, case in enumerate(cases_payload, start=1):
            gate_counts: Counter[str] = Counter()
            formula_diffs: Counter[str] = Counter()
            accepted_count = 0
            raw_attempt_count = 0
            accepted_ranks: list[int] = []
            raw_index = 0
            while raw_index < max_raw_attempts and accepted_count < accepted_k:
                batch_n = min(generation_batch_size, max_raw_attempts - raw_index)
                seeds = [seed + raw_index + offset for offset in range(batch_n)]
                try:
                    texts = generate_batch_mode_aware(
                        model,
                        tokenizer,
                        case["prompt"],
                        seeds,
                        mode=MODE,
                        device=device,
                        dtype=dtype,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    for offset, batch_seed in enumerate(seeds):
                        if accepted_count >= accepted_k:
                            break
                        attempt_index = raw_index + offset
                        raw_attempt_count += 1
                        gate_counts["schema_invalid"] += 1
                        raw_record = {
                            "mode": MODE,
                            "sample_index": case["index"],
                            "sample_id": case["sample_id"],
                            "gen_index": attempt_index,
                            "seed": int(batch_seed),
                            "raw_generation_success": False,
                            "generated_text": "",
                            "error": err,
                            "accepted": False,
                            "accepted_rank": None,
                            "reject_gate": "schema_invalid",
                            "reject_reason": err,
                        }
                        f_raw.write(json.dumps(raw_record, ensure_ascii=True) + "\n")
                    raw_index += batch_n
                    continue

                for offset, (batch_seed, text) in enumerate(zip(seeds, texts)):
                    if accepted_count >= accepted_k:
                        break
                    attempt_index = raw_index + offset
                    raw_attempt_count += 1
                    record_text = extract_generated_record(text, MODE)
                    raw_record = {
                        "mode": MODE,
                        "sample_index": case["index"],
                        "sample_id": case["sample_id"],
                        "gen_index": attempt_index,
                        "seed": int(batch_seed),
                        "raw_generation_success": bool(record_text.strip()),
                        "generated_text": record_text,
                        "error": None,
                        "accepted": False,
                        "accepted_rank": None,
                        "reject_gate": None,
                        "reject_reason": None,
                    }
                    try:
                        info = accepted_filter(raw_record, case, lookup)
                        accepted_rank = accepted_count
                        accepted_count += 1
                        accepted_ranks.append(attempt_index)
                        raw_record["accepted"] = True
                        raw_record["accepted_rank"] = accepted_rank
                        raw_record["generated_formula_from_sites"] = info.generated_formula
                        accepted_record = dict(raw_record)
                        accepted_record["gen_index"] = accepted_rank
                        accepted_record["raw_gen_index"] = attempt_index
                        accepted_record["accepted_rank"] = accepted_rank
                        accepted_record["standard_cif"] = info.standard_cif
                        f_acc.write(json.dumps(accepted_record, ensure_ascii=True) + "\n")
                    except GateError as err:
                        gate = err.gate if err.gate in GATE_NAMES else "schema_invalid"
                        gate_counts[gate] += 1
                        raw_record["reject_gate"] = gate
                        raw_record["reject_reason"] = err.reason
                        raw_record["generated_formula_from_sites"] = err.generated_formula
                        if err.formula_diff:
                            key = formula_diff_key(err.formula_diff)
                            raw_record["formula_diff"] = err.formula_diff
                            raw_record["formula_diff_key"] = key
                            formula_diffs[key] += 1
                    f_raw.write(json.dumps(raw_record, ensure_ascii=True) + "\n")
                raw_index += batch_n

            sample_stat = {
                "sample_index": case["index"],
                "sample_id": case["sample_id"],
                "target_formula": case["target_formula"],
                "target_sg_number": case["target_sg_number"],
                "raw_attempt_count": raw_attempt_count,
                "accepted_count": accepted_count,
                "accepted_raw_indices": accepted_ranks,
                "accepted_target_reached": accepted_count >= accepted_k,
                "gate_failures": {gate: int(gate_counts.get(gate, 0)) for gate in GATE_NAMES},
                "formula_not_closed_diff_counts": dict(formula_diffs),
                "formula_not_closed_diff_top20": dict(formula_diffs.most_common(20)),
            }
            f_stats.write(json.dumps(sample_stat, ensure_ascii=True, sort_keys=True) + "\n")
            if local_i % 10 == 0 or local_i == len(cases_payload):
                print(
                    f"[extended:{device}:worker{worker_id}] {local_i}/{len(cases_payload)} "
                    f"last={case['sample_id']} raw={raw_attempt_count} accepted={accepted_count}",
                    flush=True,
                )


def merge_worker_parts(out_dir: Path) -> tuple[Path, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    worker_dir = out_dir / "worker_parts"
    raw_out = out_dir / "raw_generations" / "symcif_v1_raw.jsonl"
    acc_out = out_dir / "generations" / "symcif_v1_accepted.jsonl"
    stats_out = out_dir / "sample_stats.jsonl"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    acc_out.parent.mkdir(parents=True, exist_ok=True)

    def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        return records

    raw_records = read_jsonl(sorted(worker_dir.glob("symcif_v1_raw.worker*.jsonl")))
    raw_records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    with raw_out.open("w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")

    accepted_records = read_jsonl(sorted(worker_dir.glob("symcif_v1_accepted.worker*.jsonl")))
    accepted_records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    with acc_out.open("w", encoding="utf-8") as f:
        for rec in accepted_records:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")

    sample_stats = read_jsonl(sorted(worker_dir.glob("sample_stats.worker*.jsonl")))
    sample_stats.sort(key=lambda r: int(r["sample_index"]))
    with stats_out.open("w", encoding="utf-8") as f:
        for rec in sample_stats:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
    return raw_out, acc_out, stats_out, accepted_records, sample_stats


def run_extended_generation(args: argparse.Namespace, cases: list[Any]) -> tuple[Path, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    raw_out = out_dir / "raw_generations" / "symcif_v1_raw.jsonl"
    acc_out = out_dir / "generations" / "symcif_v1_accepted.jsonl"
    stats_out = out_dir / "sample_stats.jsonl"
    expected_samples = len(cases)
    if raw_out.exists() and acc_out.exists() and stats_out.exists() and not args.overwrite_generation:
        sample_stats = [json.loads(line) for line in stats_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        accepted_records = [json.loads(line) for line in acc_out.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(sample_stats) == expected_samples:
            print(f"[extended] found existing generation/filter outputs in {out_dir}, skipping generation", flush=True)
            return raw_out, acc_out, stats_out, accepted_records, sample_stats

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] or ["cpu"]
    chunks: list[list[Any]] = [[] for _ in devices]
    for i, case in enumerate(cases):
        chunks[i % len(devices)].append(case)

    model_dir = PROJECT_ROOT / "runs" / "exp_symcif_v1"
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        payload = [case_to_payload(case) for case in chunk]
        proc = ctx.Process(
            target=extended_generation_worker,
            kwargs={
                "model_dir": str(model_dir),
                "cases_payload": payload,
                "out_dir": str(out_dir),
                "device": device,
                "dtype": args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "compile_model": args.compile,
                "worker_id": worker_id,
                "seed": args.seed,
                "max_raw_attempts": args.max_raw_attempts,
                "accepted_k": args.accepted_k,
                "generation_batch_size": args.generation_batch_size,
                "lookup_json": str(args.lookup_json),
            },
        )
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"extended generation worker failed with exit code {proc.exitcode}")

    return merge_worker_parts(out_dir)


def grouped_records(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[int(rec["sample_index"])].append(rec)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    return dict(grouped)


def run_accepted_evaluation(
    args: argparse.Namespace,
    cases: list[Any],
    accepted_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics_dir = Path(args.out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "symcif_v1_accepted_per_generation_metrics.jsonl"
    if metrics_path.exists() and not args.overwrite_evaluation:
        return [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    case_payload = [
        {
            "index": c.index,
            "sample_id": c.sample_id,
            "source_path": c.source_path,
            "target_formula": c.target_formula,
            "target_sg_number": c.target_sg_number,
            "target_sg_symbol": c.target_sg_symbol,
        }
        for c in cases
    ]
    all_metrics = evaluate_mode_with_hard_timeouts(
        mode=MODE,
        case_payload=case_payload,
        grouped=grouped_records(accepted_records),
        lookup_json=str(args.lookup_json),
        args=args,
    )
    all_metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
    with metrics_path.open("w", encoding="utf-8") as f:
        for rec in all_metrics:
            f.write(json.dumps(rec, ensure_ascii=True, sort_keys=True) + "\n")
    return all_metrics


def summarize(
    *,
    args: argparse.Namespace,
    cases: list[Any],
    sample_stats: list[dict[str, Any]],
    accepted_records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    total_cases = len(cases)
    raw_attempts = sum(int(row.get("raw_attempt_count", 0)) for row in sample_stats)
    accepted_total = sum(int(row.get("accepted_count", 0)) for row in sample_stats)
    accepted_counts = [int(row.get("accepted_count", 0)) for row in sample_stats]
    raw_counts = [int(row.get("raw_attempt_count", 0)) for row in sample_stats]

    gate_counts = Counter()
    formula_diffs = Counter()
    for row in sample_stats:
        gate_counts.update({key: int(value) for key, value in row.get("gate_failures", {}).items()})
        formula_diffs.update(
            {key: int(value) for key, value in row.get("formula_not_closed_diff_counts", row.get("formula_not_closed_diff_top20", {})).items()}
        )

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        by_sample[int(metric["sample_index"])].append(metric)

    matched_samples = 0
    best_rms: list[float] = []
    for case in cases:
        rows = by_sample.get(int(case.index), [])
        rms_values = [float(row["rms"]) for row in rows if row.get("match_ok") and row.get("rms") is not None]
        if rms_values:
            matched_samples += 1
            best_rms.append(min(rms_values))

    accepted_metric_count = len(metrics)
    valid_count = sum(1 for row in metrics if row.get("valid"))
    match_candidate_count = sum(1 for row in metrics if row.get("match_ok"))

    return {
        "settings": {
            "test_samples": total_cases,
            "checkpoint": "runs/exp_symcif_v1/ckpt_best.pt",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_raw_attempts": args.max_raw_attempts,
            "accepted_k": args.accepted_k,
            "seed_start": args.seed,
            "devices": args.devices,
            "generation_batch_size": args.generation_batch_size,
            "max_match_sites": args.max_match_sites,
            "max_eval_sites": args.max_eval_sites,
        },
        "raw_attempt_view": {
            "raw_attempts": raw_attempts,
            "accepted_total": accepted_total,
            "raw_to_accepted_rate": accepted_total / raw_attempts if raw_attempts else math.nan,
            "gate_failures": {gate: int(gate_counts.get(gate, 0)) for gate in GATE_NAMES},
            "formula_not_closed_diff_top20": dict(formula_diffs.most_common(20)),
        },
        "accepted_count_stats": {
            "mean": float(np.mean(accepted_counts)) if accepted_counts else math.nan,
            "median": float(np.median(accepted_counts)) if accepted_counts else math.nan,
            "min": int(min(accepted_counts)) if accepted_counts else 0,
            "max": int(max(accepted_counts)) if accepted_counts else 0,
            "ge_20_count": sum(1 for x in accepted_counts if x >= args.accepted_k),
            "ge_20_rate": sum(1 for x in accepted_counts if x >= args.accepted_k) / total_cases if total_cases else math.nan,
            "zero_count": sum(1 for x in accepted_counts if x == 0),
            "zero_rate": sum(1 for x in accepted_counts if x == 0) / total_cases if total_cases else math.nan,
            "raw_attempt_mean": float(np.mean(raw_counts)) if raw_counts else math.nan,
            "raw_attempt_median": float(np.median(raw_counts)) if raw_counts else math.nan,
        },
        "accepted_at20_view": {
            "accepted_metric_count": accepted_metric_count,
            "valid_rate_among_accepted": valid_count / accepted_metric_count if accepted_metric_count else math.nan,
            "accepted_candidate_match_rate": match_candidate_count / accepted_metric_count if accepted_metric_count else math.nan,
            "match_rate_accepted_at20": matched_samples / total_cases if total_cases else math.nan,
            "RMSE": float(np.mean(best_rms)) if best_rms else math.nan,
            "matched_samples_for_RMSE": len(best_rms),
        },
        "baselines": {
            "baseline_old_match_at20": 0.45,
            "baseline_minprompt_match_at20": 0.482,
            "symcif_v1_old_raw_match_at20": 0.352,
        },
    }


def write_summary_csv(summary: dict[str, Any], out_dir: Path) -> None:
    row = {
        "mode": "symcif_v1_extended_rejection",
        "test_samples": summary["settings"]["test_samples"],
        "raw_attempts": summary["raw_attempt_view"]["raw_attempts"],
        "accepted_total": summary["raw_attempt_view"]["accepted_total"],
        "raw_to_accepted_rate": summary["raw_attempt_view"]["raw_to_accepted_rate"],
        "accepted_count_mean": summary["accepted_count_stats"]["mean"],
        "accepted_count_median": summary["accepted_count_stats"]["median"],
        "accepted_count_min": summary["accepted_count_stats"]["min"],
        "accepted_count_max": summary["accepted_count_stats"]["max"],
        "accepted_ge_20_rate": summary["accepted_count_stats"]["ge_20_rate"],
        "accepted_zero_rate": summary["accepted_count_stats"]["zero_rate"],
        "valid_rate_among_accepted": summary["accepted_at20_view"]["valid_rate_among_accepted"],
        "accepted_candidate_match_rate": summary["accepted_at20_view"]["accepted_candidate_match_rate"],
        "match_rate_accepted_at20": summary["accepted_at20_view"]["match_rate_accepted_at20"],
        "RMSE": summary["accepted_at20_view"]["RMSE"],
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sanity_accept_worker(out_queue: Any, row: dict[str, Any], case: dict[str, Any], lookup_json: str) -> None:
    try:
        lookup = WyckoffLookup.from_json(lookup_json)
        accepted_filter(row, case, lookup)
        out_queue.put({"accepted": True})
    except GateError as err:
        out_queue.put({"accepted": False, "reason": f"{err.gate}:{err.reason}"})
    except Exception as exc:  # noqa: BLE001
        out_queue.put({"accepted": False, "reason": f"unexpected:{type(exc).__name__}:{exc}"})


def sanity_accept_with_hard_timeout(
    row: dict[str, Any],
    case: dict[str, Any],
    lookup_json: Path,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    out_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_sanity_accept_worker, args=(out_queue, row, case, str(lookup_json)))
    proc.start()
    proc.join(timeout=max(0.1, float(timeout_seconds)))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)
        out_queue.close()
        return False, f"sanity_timeout>{timeout_seconds:g}s"
    result = None
    try:
        result = out_queue.get_nowait()
    except Exception:
        pass
    out_queue.close()
    if result and result.get("accepted"):
        return True, None
    if result and result.get("reason"):
        return False, str(result["reason"])
    return False, f"sanity_worker_failed_exitcode={proc.exitcode}"


def run_sanity_check(args: argparse.Namespace, cases: list[Any], lookup: WyckoffLookup) -> dict[str, Any]:
    metrics = load_jsonl(args.sanity_metrics_jsonl)
    generations = load_jsonl(args.sanity_generations_jsonl)
    if not metrics or not generations:
        return {
            "skipped": True,
            "reason": "missing sanity metrics or generations",
            "metrics_path": str(args.sanity_metrics_jsonl),
            "generations_path": str(args.sanity_generations_jsonl),
        }

    case_map = {
        int(case.index): {
            "sample_id": case.sample_id,
            "target_formula": case.target_formula,
            "target_sg_number": case.target_sg_number,
            "target_sg_symbol": case.target_sg_symbol,
        }
        for case in cases
    }
    gen_map = {(int(row["sample_index"]), int(row["gen_index"])): row for row in generations}
    matched = [row for row in metrics if row.get("match_ok")]
    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    for metric in matched:
        key = (int(metric["sample_index"]), int(metric["gen_index"]))
        row = gen_map.get(key)
        case = case_map.get(int(metric["sample_index"]))
        if not row or not case:
            rejected += 1
            reasons["missing_generation_or_case"] += 1
            continue
        try:
            ok, reason = sanity_accept_with_hard_timeout(row, case, args.lookup_json, args.sanity_timeout_seconds)
            if not ok:
                rejected += 1
                reasons[str(reason)] += 1
                continue
            accepted += 1
        except TimeoutError:
            rejected += 1
            reasons[f"sanity_timeout>{args.sanity_timeout_seconds:g}s"] += 1
        except GateError as err:
            rejected += 1
            reasons[f"{err.gate}:{err.reason}"] += 1
    return {
        "skipped": False,
        "raw_matched_generations": len(matched),
        "accepted": accepted,
        "rejected": rejected,
        "rejected_top_reasons": dict(reasons.most_common(20)),
        "metrics_path": str(args.sanity_metrics_jsonl),
        "generations_path": str(args.sanity_generations_jsonl),
    }


def write_report(summary: dict[str, Any], sanity: dict[str, Any], out_dir: Path, tag: str) -> Path:
    report_dir = PROJECT_ROOT / "Log_GPT"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"extended_rejection_report_{tag}.md"
    raw = summary["raw_attempt_view"]
    counts = summary["accepted_count_stats"]
    acc = summary["accepted_at20_view"]
    settings = summary["settings"]

    lines = [
        "# SymCIF-v1 Extended Rejection Sampling 报告",
        "",
        f"日期/标签：`{tag}`",
        "",
        "## 1. 设置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| checkpoint | `{settings['checkpoint']}` |",
        f"| test samples | {settings['test_samples']} |",
        f"| temperature / top_k | {settings['temperature']} / {settings['top_k']} |",
        f"| max_new_tokens | {settings['max_new_tokens']} |",
        f"| max_raw_attempts / accepted_k | {settings['max_raw_attempts']} / {settings['accepted_k']} |",
        f"| devices | `{settings['devices']}` |",
        f"| generation_batch_size | {settings['generation_batch_size']} |",
        f"| max_match_sites / max_eval_sites | {settings['max_match_sites']} / {settings['max_eval_sites']} |",
        f"| output dir | `{out_dir}` |",
        "",
        "accepted filter 是纯筛选，不使用 GT CIF、GT Wyckoff、GT 坐标或 GT cell，也不为了闭合公式自动增删 site / 替换元素。",
        "",
        "## 2. Raw-Attempt 口径",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| raw attempts | {raw['raw_attempts']} |",
        f"| accepted total | {raw['accepted_total']} |",
        f"| raw -> accepted pass rate | {fmt_pct(raw['raw_to_accepted_rate'])} |",
        "",
        "gate failure counts:",
        "",
        "| gate | count |",
        "| --- | ---: |",
    ]
    for gate in GATE_NAMES:
        lines.append(f"| {gate} | {raw['gate_failures'].get(gate, 0)} |")
    lines.extend(
        [
            "",
            "formula_not_closed diff Top 20:",
            "",
            "| diff | count |",
            "| --- | ---: |",
        ]
    )
    for key, value in raw["formula_not_closed_diff_top20"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 3. Accepted@20 口径",
            "",
            "| 指标 | 值 |",
            "| --- | ---: |",
            f"| accepted_count mean | {fmt_num(counts['mean'], 2)} |",
            f"| accepted_count median | {fmt_num(counts['median'], 2)} |",
            f"| accepted_count min / max | {counts['min']} / {counts['max']} |",
            f"| accepted_count >= 20 | {counts['ge_20_count']} ({fmt_pct(counts['ge_20_rate'])}) |",
            f"| accepted_count = 0 | {counts['zero_count']} ({fmt_pct(counts['zero_rate'])}) |",
            f"| accepted candidates valid rate | {fmt_pct(acc['valid_rate_among_accepted'])} |",
            f"| accepted candidates match rate | {fmt_pct(acc['accepted_candidate_match_rate'])} |",
            f"| match@20 over samples | {fmt_pct(acc['match_rate_accepted_at20'])} |",
            f"| RMSE | {fmt_num(acc['RMSE'])} |",
            f"| matched samples for RMSE | {acc['matched_samples_for_RMSE']} |",
            "",
            "对照：baseline old match@20 = 45.00%，baseline_minprompt match@20 = 48.20%，symcif_v1 old raw match@20 = 35.20%。",
            "",
            "## 4. Sanity Check",
            "",
        ]
    )
    if sanity.get("skipped"):
        lines.append(f"sanity check skipped: {sanity.get('reason')}")
    else:
        lines.extend(
            [
                f"- raw matched generations 总数：{sanity['raw_matched_generations']}",
                f"- accepted：{sanity['accepted']}",
                f"- rejected：{sanity['rejected']}",
                "",
                "| rejected reason | count |",
                "| --- | ---: |",
            ]
        )
        for key, value in sanity.get("rejected_top_reasons", {}).items():
            lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## 5. 输出文件",
            "",
            f"- summary JSON：`{out_dir / 'summary.json'}`",
            f"- summary CSV：`{out_dir / 'summary.csv'}`",
            f"- per-sample stats：`{out_dir / 'sample_stats.jsonl'}`",
            f"- raw generations：`{out_dir / 'raw_generations' / 'symcif_v1_raw.jsonl'}`",
            f"- accepted generations：`{out_dir / 'generations' / 'symcif_v1_accepted.jsonl'}`",
            f"- accepted metrics：`{out_dir / 'metrics' / 'symcif_v1_accepted_per_generation_metrics.jsonl'}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SymCIF-v1 extended rejection sampling.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "extended_rejection_symcif_v1_pilot_20260520")
    parser.add_argument("--tag", default="20260520_pilot")
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--max-raw-attempts", type=int, default=200)
    parser.add_argument("--accepted-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--generation-batch-size", type=int, default=20)
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup.json")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(32, os.cpu_count() or 4)))
    parser.add_argument("--bond-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-match-sites", type=int, default=96)
    parser.add_argument("--max-eval-sites", type=int, default=96)
    parser.add_argument("--overwrite-generation", action="store_true")
    parser.add_argument("--overwrite-evaluation", action="store_true")
    parser.add_argument("--sanity-timeout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--sanity-generations-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "eval_runs"
        / "generation_eval_t1_topk10_n20_20260519"
        / "generations"
        / "symcif_v1.jsonl",
    )
    parser.add_argument(
        "--sanity-metrics-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "eval_runs"
        / "generation_eval_t1_topk10_n20_20260520_fixedgt"
        / "metrics"
        / "symcif_v1_per_generation_metrics.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_test_cases(args.test_limit, modes=(MODE,))
    metadata = {
        "mode": MODE,
        "test_samples": len(cases),
        "prompt_rule": "symcif_v1 prefix from data_ line through _symmetry_space_group_name_H-M",
        "checkpoint": "runs/exp_symcif_v1/ckpt_best.pt",
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "max_raw_attempts": args.max_raw_attempts,
        "accepted_k": args.accepted_k,
        "seed_start": args.seed,
        "devices": args.devices,
        "dtype": args.dtype,
        "generation_batch_size": args.generation_batch_size,
        "max_match_sites": args.max_match_sites,
        "max_eval_sites": args.max_eval_sites,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _, _, _, accepted_records, sample_stats = run_extended_generation(args, cases)
    metrics = run_accepted_evaluation(args, cases, accepted_records)
    summary = summarize(args=args, cases=cases, sample_stats=sample_stats, accepted_records=accepted_records, metrics=metrics)
    lookup = WyckoffLookup.from_json(args.lookup_json)
    sanity_cases = load_test_cases(None, modes=(MODE,))
    sanity = run_sanity_check(args, sanity_cases, lookup)
    summary["sanity_check"] = sanity

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_csv(summary, args.out_dir)
    report_path = write_report(summary, sanity, args.out_dir, args.tag)
    print(f"[extended] summary -> {args.out_dir / 'summary.csv'}", flush=True)
    print(f"[extended] report -> {report_path}", flush=True)
    if float(summary["accepted_at20_view"]["match_rate_accepted_at20"]) <= 0.352:
        print("[extended] pilot did not improve over old symcif_v1 raw match@20=35.20%; pause before full run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
