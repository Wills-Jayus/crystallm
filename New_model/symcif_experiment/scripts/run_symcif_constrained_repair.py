#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_generation_eval import load_test_cases  # noqa: E402
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.models import LatticeParameters, SymCifRecord, WyckoffSite  # noqa: E402
from symcif.render import render_standard_cif  # noqa: E402


ALLOWED_KEYS = {
    "_chemical_formula_sum",
    "_symmetry_Int_Tables_number",
    "_symmetry_space_group_name_H-M",
    "_cell_formula_units_Z",
    "_wyckoff_site_index",
    "_wyckoff_site_element",
    "_wyckoff_site_multiplicity",
    "_wyckoff_site_letter",
    "_wyckoff_site_symmetry",
    "_wyckoff_site_enumeration",
    "_wyckoff_free_x",
    "_wyckoff_free_y",
    "_wyckoff_free_z",
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
    "_cell_volume",
}
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


class RepairError(ValueError):
    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        target_formula: str | None = None,
        generated_formula: str | None = None,
        sg_number: int | None = None,
        trace: list[str] | None = None,
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason
        self.target_formula = target_formula
        self.generated_formula = generated_formula
        self.sg_number = sg_number
        self.trace = trace or []


@dataclass
class ParsedText:
    data_name: str
    values: dict[str, str]
    headers: list[str]
    rows: list[list[str]]


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.replace("<unk>", "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(ord(ch) < 9 for ch in line):
            continue
        lines.append(line)
    return lines


def parse_generated_text(text: str, trace: list[str]) -> ParsedText:
    lines = clean_lines(text)
    data_idx = next((i for i, line in enumerate(lines) if line.startswith("data_")), None)
    if data_idx is None:
        raise RepairError("schema_invalid", "missing_data_line", trace=trace)
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
            loop_headers = []
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
            key = parts[0]
            if key in ALLOWED_KEYS and len(parts) == 2:
                values[key] = parts[1].strip().strip("'\"")
        i += 1
    if not headers:
        raise RepairError("schema_invalid", "missing_wyckoff_loop", trace=trace)
    missing = [h for h in SITE_HEADERS if h not in headers]
    if missing:
        raise RepairError("schema_invalid", f"missing_headers:{','.join(missing)}", trace=trace)
    fixed_rows = []
    for raw_row in rows:
        if len(raw_row) != len(headers):
            raise RepairError("schema_invalid", f"row_header_length_mismatch:{len(raw_row)}!={len(headers)}", trace=trace)
        fixed_rows.append(raw_row)
    trace.append("schema_normalized")
    return ParsedText(data_name=data_name, values=values, headers=headers, rows=fixed_rows)


def parse_int(value: str | None, key: str, trace: list[str]) -> int:
    if value is None:
        raise RepairError("schema_invalid", f"missing_{key}", trace=trace)
    try:
        return int(float(str(value).strip()))
    except Exception as exc:
        raise RepairError("schema_invalid", f"invalid_int_{key}:{value}", trace=trace) from exc


def parse_float(value: str | None, key: str, trace: list[str]) -> float:
    if value is None:
        raise RepairError("cell_missing", f"missing_{key}", trace=trace)
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise RepairError("cell_invalid", f"invalid_float_{key}:{value}", trace=trace) from exc


def parse_coord(token: str) -> float | None:
    if token.upper() == "FIXED":
        return None
    value = float(token)
    value = value % 1.0
    if math.isclose(value, 1.0, abs_tol=1e-8) or math.isclose(value, 0.0, abs_tol=1e-8):
        return 0.0
    return float(value)


def comp_counts(formula: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in Composition(formula).as_dict().items():
        nearest = int(round(float(value)))
        if abs(float(value) - nearest) > 1e-6:
            raise ValueError(f"non_integer_formula:{formula}")
        out[str(key)] = nearest
    return out


def formula_from_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{el}{counts[el]}" for el in sorted(counts))


def formula_diff(target: dict[str, int], generated: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(target) | set(generated))
    return {key: generated.get(key, 0) - target.get(key, 0) for key in keys if generated.get(key, 0) != target.get(key, 0)}


def normalize_sg(parsed: ParsedText, case: dict[str, Any], trace: list[str]) -> tuple[int, str]:
    sg_number_raw = parsed.values.get("_symmetry_Int_Tables_number")
    sg_symbol_raw = parsed.values.get("_symmetry_space_group_name_H-M")
    sg_number = None
    if sg_number_raw is not None:
        try:
            sg_number = int(float(sg_number_raw))
        except Exception as exc:
            raise RepairError("sg_invalid", f"invalid_sg_number:{sg_number_raw}", trace=trace) from exc
    if sg_number is None and sg_symbol_raw:
        for number in range(1, 231):
            try:
                if SpaceGroup.from_int_number(number).symbol.replace(" ", "") == sg_symbol_raw.replace(" ", ""):
                    sg_number = number
                    break
            except Exception:
                continue
    if sg_number is None:
        raise RepairError("sg_missing", "sg_number_and_symbol_missing", trace=trace)
    if not 1 <= sg_number <= 230:
        raise RepairError("sg_invalid", f"sg_out_of_range:{sg_number}", sg_number=sg_number, trace=trace)
    target_sg = case.get("target_sg_number")
    if target_sg is not None and int(target_sg) != sg_number:
        raise RepairError("sg_not_target", f"generated_sg={sg_number}:target_sg={target_sg}", sg_number=sg_number, trace=trace)
    sg_symbol = SpaceGroup.from_int_number(sg_number).symbol
    trace.append(f"sg_normalized:{sg_number}:{sg_symbol}")
    return sg_number, sg_symbol


def project_lattice(sg_number: int, values: dict[str, float], trace: list[str]) -> LatticeParameters:
    a = values["_cell_length_a"]
    b = values["_cell_length_b"]
    c = values["_cell_length_c"]
    alpha = values["_cell_angle_alpha"]
    beta = values["_cell_angle_beta"]
    gamma = values["_cell_angle_gamma"]
    if a <= 0 or b <= 0 or c <= 0:
        raise RepairError("cell_invalid", "non_positive_cell_length", sg_number=sg_number, trace=trace)
    if not all(30.0 <= x <= 150.0 for x in (alpha, beta, gamma)):
        raise RepairError("cell_invalid", "angle_out_of_range", sg_number=sg_number, trace=trace)
    crystal_system = SpaceGroup.from_int_number(sg_number).crystal_system
    if crystal_system == "cubic":
        avg = (a + b + c) / 3.0
        a = b = c = avg
        alpha = beta = gamma = 90.0
        trace.append("cell_projected:cubic")
    elif crystal_system == "tetragonal":
        avg = (a + b) / 2.0
        a = b = avg
        alpha = beta = gamma = 90.0
        trace.append("cell_projected:tetragonal")
    elif crystal_system == "orthorhombic":
        alpha = beta = gamma = 90.0
        trace.append("cell_projected:orthorhombic")
    elif crystal_system in {"hexagonal", "trigonal"}:
        avg = (a + b) / 2.0
        a = b = avg
        alpha = beta = 90.0
        gamma = 120.0
        trace.append(f"cell_projected:{crystal_system}")
    elif crystal_system == "monoclinic":
        alpha = gamma = 90.0
        trace.append("cell_projected:monoclinic")
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return LatticeParameters(
        a=float(lattice.a),
        b=float(lattice.b),
        c=float(lattice.c),
        alpha=float(lattice.alpha),
        beta=float(lattice.beta),
        gamma=float(lattice.gamma),
        volume=float(lattice.volume),
    )


def repair_one(
    row: dict[str, Any],
    case: dict[str, Any],
    lookup: WyckoffLookup,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    trace: list[str] = []
    text = row.get("generated_text") or ""
    target_formula = str(case["target_formula"])
    parsed = parse_generated_text(text, trace)
    sg_number, sg_symbol = normalize_sg(parsed, case, trace)

    header_index = {key: parsed.headers.index(key) for key in SITE_HEADERS}
    sites: list[WyckoffSite] = []
    generated_counts: Counter[str] = Counter()
    for new_index, raw_row in enumerate(parsed.rows, start=1):
        rec = {key: raw_row[idx] for key, idx in header_index.items()}
        element = rec["_wyckoff_site_element"]
        try:
            Element(element)
        except Exception as exc:
            raise RepairError("schema_invalid", f"invalid_element:{element}", sg_number=sg_number, trace=trace) from exc
        letter = rec["_wyckoff_site_letter"]
        try:
            template = lookup.get(sg_number, letter)
        except Exception as exc:
            raise RepairError("wyckoff_invalid", f"invalid_letter:{letter}", sg_number=sg_number, trace=trace) from exc
        coords = []
        for axis, key in enumerate(("_wyckoff_free_x", "_wyckoff_free_y", "_wyckoff_free_z")):
            token = rec[key]
            if template.free_mask[axis]:
                try:
                    value = parse_coord(token)
                except Exception as exc:
                    raise RepairError("coord_invalid", f"invalid_free_coord:{token}", sg_number=sg_number, trace=trace) from exc
                if value is None:
                    raise RepairError("coord_invalid", f"missing_free_coord:{key}", sg_number=sg_number, trace=trace)
                coords.append(value)
            else:
                coords.append(template.fixed_values[axis])
                if token.upper() != "FIXED":
                    trace.append(f"fixed_coord_overwrite:site={new_index}:{key}")
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
    trace.append("wyckoff_and_coord_checked")

    target_counts = comp_counts(target_formula)
    generated = dict(generated_counts)
    generated_formula = formula_from_counts(generated)
    diff = formula_diff(target_counts, generated)
    if diff:
        raise RepairError(
            "formula_not_closed",
            f"diff={diff}",
            target_formula=target_formula,
            generated_formula=generated_formula,
            sg_number=sg_number,
            trace=trace,
        )
    trace.append("formula_closed")

    cell_values = {
        key: parse_float(parsed.values.get(key), key, trace)
        for key in (
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        )
    }
    lattice = project_lattice(sg_number, cell_values, trace)
    trace.append("cell_checked")

    z = parse_int(parsed.values.get("_cell_formula_units_Z"), "_cell_formula_units_Z", trace)
    record = SymCifRecord(
        sample_id=str(row.get("sample_id") or case["sample_id"]),
        source_path=None,
        cell_formula=target_formula,
        reduced_formula="",
        sg_number=sg_number,
        sg_symbol=sg_symbol,
        z=z,
        lattice=lattice,
        sites=sites,
    )
    try:
        standard_cif = render_standard_cif(record, symprec=0.1)
        Structure.from_str(standard_cif, fmt="cif")
    except Exception as exc:
        raise RepairError(
            "render_cif_failed",
            f"{type(exc).__name__}:{exc}",
            target_formula=target_formula,
            generated_formula=generated_formula,
            sg_number=sg_number,
            trace=trace,
        ) from exc
    trace.append("rendered_and_readable")
    repaired = dict(row)
    repaired["mode"] = "symcif_v1_constrained"
    repaired["raw_generation_success"] = True
    repaired["generated_text"] = standard_cif
    repaired["error"] = None
    repaired["repair_success"] = True
    repaired["repair_trace"] = trace
    info = {
        "target_formula": target_formula,
        "generated_formula": generated_formula,
        "sg_number": sg_number,
        "trace": trace,
    }
    return repaired, standard_cif, info


def failure_record(row: dict[str, Any], case: dict[str, Any], err: RepairError) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id") or case["sample_id"],
        "generation_id": f"{row.get('sample_index')}:{row.get('gen_index')}",
        "sample_index": row.get("sample_index"),
        "gen_index": row.get("gen_index"),
        "stage": err.stage,
        "reason": err.reason,
        "target_formula": err.target_formula or case.get("target_formula"),
        "generated_formula": err.generated_formula,
        "sg_number": err.sg_number,
        "raw_text_snippet": (row.get("generated_text") or "")[:800],
        "repair_trace": err.trace,
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)[:120]


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative constrained repair/rejection for symcif_v1 generations.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=PROJECT_ROOT / "eval_runs" / "generation_eval_t1_topk10_n20_20260519" / "generations" / "symcif_v1.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "generation_eval_t1_topk10_n20_20260520_constrained")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup.json")
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = args.out_dir / "generations_repaired"
    cif_dir = args.out_dir / "standard_cifs"
    gen_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = gen_dir / "symcif_v1_constrained.jsonl"
    failed_path = args.out_dir / "failed_repair_cases.jsonl"

    cases = {case.index: case for case in load_test_cases(args.test_limit, modes=("symcif_v1",))}
    lookup = WyckoffLookup.from_json(args.lookup_json)
    counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    successes = 0

    with args.input_jsonl.open(encoding="utf-8") as f_in, out_jsonl.open("w", encoding="utf-8") as f_out, failed_path.open(
        "w", encoding="utf-8"
    ) as f_failed:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_index = int(row["sample_index"])
            if args.test_limit is not None and sample_index >= args.test_limit:
                continue
            case_obj = cases[sample_index]
            case = {
                "sample_id": case_obj.sample_id,
                "target_formula": case_obj.target_formula,
                "target_sg_number": case_obj.target_sg_number,
            }
            counts["total_generations"] += 1
            try:
                repaired, standard_cif, info = repair_one(row, case, lookup)
                counts["schema_normalize_success"] += 1
                counts["sg_check_success"] += 1
                counts["wyckoff_legality_success"] += 1
                counts["coord_mask_success"] += 1
                counts["formula_closed"] += 1
                counts["cell_check_success"] += 1
                counts["render_cif_success"] += 1
                counts["pymatgen_readable"] += 1
                successes += 1
                sample_dir = cif_dir / f"{int(row['sample_index']):04d}_{safe_name(str(row.get('sample_id') or case_obj.sample_id))}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                (sample_dir / f"gen_{int(row['gen_index']):02d}.cif").write_text(standard_cif, encoding="utf-8")
                f_out.write(json.dumps(repaired, ensure_ascii=True) + "\n")
            except RepairError as err:
                counts[f"{err.stage}_failed"] += 1
                failure_reasons[f"{err.stage}:{err.reason}"] += 1
                failed = failure_record(row, case, err)
                f_failed.write(json.dumps(failed, ensure_ascii=True, sort_keys=True) + "\n")
                rejected = dict(row)
                rejected["mode"] = "symcif_v1_constrained"
                rejected["raw_generation_success"] = False
                rejected["generated_text"] = ""
                rejected["error"] = f"{err.stage}: {err.reason}"
                rejected["repair_success"] = False
                f_out.write(json.dumps(rejected, ensure_ascii=True) + "\n")

    report = {
        "total_generations": int(counts["total_generations"]),
        "schema_normalize_success": int(counts["schema_normalize_success"]),
        "sg_check_success": int(counts["sg_check_success"]),
        "wyckoff_legality_success": int(counts["wyckoff_legality_success"]),
        "coord_mask_success": int(counts["coord_mask_success"]),
        "formula_closed": int(counts["formula_closed"]),
        "cell_check_success": int(counts["cell_check_success"]),
        "render_cif_success": int(counts["render_cif_success"]),
        "pymatgen_readable": int(counts["pymatgen_readable"]),
        "final_valid": 0,
        "match_at_1": 0,
        "match_at_20": 0,
        "rmse": 0,
        "successes": successes,
        "failure_reasons": dict(failure_reasons.most_common(50)),
        "output_generation_jsonl": str(out_jsonl),
        "standard_cifs": str(cif_dir),
        "failed_repair_cases": str(failed_path),
    }
    (args.out_dir / "repair_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[repair] successes={successes}/{counts['total_generations']} -> {out_jsonl}", flush=True)
    print(f"[repair] report -> {args.out_dir / 'repair_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
