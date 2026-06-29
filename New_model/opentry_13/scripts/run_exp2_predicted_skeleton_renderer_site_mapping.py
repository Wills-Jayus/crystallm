#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # noqa: E402

from run_symcif_v4_geometry_model_eval import (  # noqa: E402
    deterministic_params,
    flexible_params_from_reference,
    postprocess_lattice,
)
from symcif_v4.formula import normalize_formula_counts  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2_predicted_skeleton_renderer_site_mapping"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
EXP3_PROPOSALS = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("keys", {}).get("sample_id"))


def formula_counts(record: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in normalize_formula_counts(record["formula_counts"]).items()}


def formula_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return dict(sorted(a.items())) == dict(sorted(b.items()))


def assign_elements_to_rows(
    *,
    target_counts: dict[str, int],
    source_rows: list[dict[str, Any]],
    source_counts: dict[str, int] | None,
    prefer_source_elements: bool,
) -> tuple[list[str] | None, str | None]:
    if prefer_source_elements and source_counts is not None and formula_equal(target_counts, source_counts):
        elements = [str(row.get("element")) for row in source_rows]
        per_element: Counter[str] = Counter()
        for row, element in zip(source_rows, elements):
            per_element[element] += int(row.get("multiplicity") or 0)
        if dict(per_element) == target_counts:
            return elements, "source_formula_exact_order"

    mults = [int(row.get("multiplicity") or 0) for row in source_rows]
    order = sorted(range(len(source_rows)), key=lambda i: (-mults[i], str(source_rows[i].get("orbit_id")), i))
    remaining = dict(target_counts)
    assigned: list[str | None] = [None] * len(source_rows)

    def rec(pos: int) -> bool:
        if pos >= len(order):
            return all(int(v) == 0 for v in remaining.values())
        idx = order[pos]
        mult = mults[idx]
        preferred = str(source_rows[idx].get("element"))
        if prefer_source_elements:
            elements = sorted(remaining, key=lambda e: (0 if e == preferred else 1, -remaining[e], e))
        else:
            elements = sorted(remaining, key=lambda e: (-remaining[e], e))
        for element in elements:
            if remaining[element] < mult:
                continue
            assigned[idx] = element
            remaining[element] -= mult
            if rec(pos + 1):
                return True
            remaining[element] += mult
            assigned[idx] = None
        return False

    if not rec(0):
        return None, "exact_cover_assignment_failed"
    return [str(x) for x in assigned], "deterministic_exact_cover" if not prefer_source_elements else "source_preferred_exact_cover"


def source_skeleton_rows(engine: OrbitEngine, source_repr: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source_repr["skeleton_sequence"]:
        orbit = engine.get_orbit_by_id(str(row["orbit_id"]))
        rows.append(
            {
                "element": str(row.get("element") or "X"),
                "orbit_id": str(row["orbit_id"]),
                "letter": str(row.get("letter") or orbit.letter),
                "multiplicity": int(row.get("multiplicity") or orbit.multiplicity),
                "site_symmetry": str(row.get("site_symmetry") or orbit.site_symmetry),
                "free_symbols": list(orbit.free_symbols),
                "enumeration": row.get("enumeration", orbit.enumeration),
            }
        )
    return rows


def candidate_rows_for_mapping(
    *,
    engine: OrbitEngine,
    target_record: dict[str, Any],
    source_repr: dict[str, Any],
    prefer_source_elements: bool,
) -> tuple[list[dict[str, Any]] | None, str, str | None]:
    rows = source_skeleton_rows(engine, source_repr)
    elements, mapping_rule = assign_elements_to_rows(
        target_counts=formula_counts(target_record),
        source_rows=rows,
        source_counts=formula_counts(source_repr),
        prefer_source_elements=prefer_source_elements,
    )
    if elements is None:
        return None, mapping_rule or "failed", "site_mapping_failed"
    out: list[dict[str, Any]] = []
    per_element: Counter[str] = Counter()
    for row, element in zip(rows, elements):
        item = dict(row)
        item["element"] = element
        out.append(item)
        per_element[element] += int(item["multiplicity"])
    if dict(per_element) != formula_counts(target_record):
        return None, mapping_rule or "failed", "formula_after_assignment_mismatch"
    return out, mapping_rule or "unknown", None


def median_lattice_by_sg(records: list[dict[str, Any]]) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_lattices: list[dict[str, Any]] = []
    for record in records:
        lattice = dict(record.get("lattice") or {})
        if not lattice:
            continue
        by_sg[int(record["sg"])].append(lattice)
        all_lattices.append(lattice)

    def med(rows: list[dict[str, Any]]) -> dict[str, float]:
        keys = ("a", "b", "c", "alpha", "beta", "gamma")
        values = [math.log(max(1.0e-6, float(r[k]))) if k in {"a", "b", "c"} else float(r[k]) / 180.0 for r in rows for k in ()]
        del values
        raw: list[float] = []
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r]
            if not vals:
                vals = [6.0 if k in {"a", "b", "c"} else 90.0]
            if k in {"a", "b", "c"}:
                raw.append(math.log(max(1.5, sorted(vals)[len(vals) // 2])))
            else:
                raw.append(sorted(vals)[len(vals) // 2] / 180.0)
        sg = int(rows[0].get("sg", 1)) if rows and "sg" in rows[0] else 1
        return postprocess_lattice(raw, sg)

    out: dict[int, dict[str, float]] = {}
    for sg, rows in by_sg.items():
        with_sg = [dict(r, sg=sg) for r in rows]
        out[sg] = med(with_sg)
    global_lattice = med([dict(r, sg=1) for r in all_lattices]) if all_lattices else {"a": 6.0, "b": 6.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    return out, global_lattice


def params_for_mode(
    *,
    mode: str,
    engine: OrbitEngine,
    rows: list[dict[str, Any]],
    source_structured: dict[str, Any] | None,
) -> tuple[dict[int, dict[str, float]], int, str]:
    if mode == "train_prototype":
        if source_structured is not None:
            params, fallback_count = flexible_params_from_reference(engine, rows, source_structured, neural_params=None)
            return params, int(fallback_count), "train_source_params_with_deterministic_fallback"
        return {idx: deterministic_params(engine, str(row["orbit_id"]), idx) for idx, row in enumerate(rows)}, len(rows), "missing_train_source_deterministic_params"
    return {idx: deterministic_params(engine, str(row["orbit_id"]), idx) for idx, row in enumerate(rows)}, len(rows), "deterministic_params"


def lattice_for_mode(
    *,
    mode: str,
    target_sg: int,
    source_structured: dict[str, Any] | None,
    median_by_sg: dict[int, dict[str, float]],
    global_median: dict[str, float],
) -> tuple[dict[str, float], str]:
    if mode == "train_prototype" and source_structured is not None and source_structured.get("lattice"):
        return {k: float(source_structured["lattice"][k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")}, "train_source_lattice"
    if int(target_sg) in median_by_sg:
        return dict(median_by_sg[int(target_sg)]), "train_sg_median_lattice"
    return dict(global_median), "train_global_median_lattice"


def min_pair_distance(structure: Structure) -> float | None:
    if len(structure) < 2:
        return None
    vals: list[float] = []
    matrix = structure.distance_matrix
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            vals.append(float(matrix[i, j]))
    return min(vals) if vals else None


def structure_counts(structure: Structure) -> dict[str, int]:
    return {str(k): int(round(v)) for k, v in structure.composition.get_el_amt_dict().items()}


def eval_rendered(payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in payload.items() if k != "cif"}
    cif = str(payload.get("cif") or "")
    if not cif:
        out.update(
            {
                "parse_success": False,
                "formula_ok": False,
                "space_group_ok": False,
                "site_count_ok": False,
                "valid": False,
                "legal_cif": False,
                "min_pair_distance": None,
                "too_short_distance": None,
                "collision_flag": None,
                "eval_error": payload.get("error") or "empty_cif",
            }
        )
        return out
    try:
        structure = Structure.from_str(cif, fmt="cif")
        expected_counts = {str(k): int(v) for k, v in payload["formula_counts"].items()}
        formula_ok = structure_counts(structure) == expected_counts
        site_count_ok = int(len(structure)) == int(payload["target_atom_count"])
        try:
            detected_sg = int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        except Exception:
            detected_sg = None
        sg_ok = detected_sg == int(payload["sg"])
        min_dist = min_pair_distance(structure)
        too_short = bool(min_dist is not None and min_dist < 0.5)
        valid = bool(formula_ok and sg_ok and site_count_ok and not too_short)
        out.update(
            {
                "parse_success": True,
                "legal_cif": True,
                "formula_ok": formula_ok,
                "space_group_ok": sg_ok,
                "detected_sg": detected_sg,
                "site_count_ok": site_count_ok,
                "rendered_site_count": int(len(structure)),
                "valid": valid,
                "min_pair_distance": min_dist,
                "too_short_distance": too_short,
                "collision_flag": too_short,
                "eval_error": None,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "parse_success": False,
                "legal_cif": False,
                "formula_ok": False,
                "space_group_ok": False,
                "site_count_ok": False,
                "valid": False,
                "min_pair_distance": None,
                "too_short_distance": None,
                "collision_flag": None,
                "eval_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sorted({str(r["sample_id"]) for r in rows})
    out: dict[str, Any] = {
        "samples": len(samples),
        "candidate_records": len(rows),
        "render_success_rate": ratio(sum(bool(r.get("render_success")) for r in rows), len(rows)),
        "legal_cif_rate": ratio(sum(bool(r.get("legal_cif")) for r in rows), len(rows)),
        "valid_rate": ratio(sum(bool(r.get("valid")) for r in rows), len(rows)),
        "formula_consistency": ratio(sum(bool(r.get("formula_ok")) for r in rows), len(rows)),
        "sg_consistency": ratio(sum(bool(r.get("space_group_ok")) for r in rows), len(rows)),
        "exact_cover_retained": ratio(sum(bool(r.get("exact_cover_retained")) for r in rows), len(rows)),
        "site_count_consistency": ratio(sum(bool(r.get("site_count_ok")) for r in rows), len(rows)),
        "skeleton_row_count_consistency": ratio(sum(bool(r.get("skeleton_row_count_ok")) for r in rows), len(rows)),
        "target_row_count_match": ratio(sum(bool(r.get("target_row_count_match")) for r in rows), len(rows)),
        "site_mapping_failure_rate": ratio(sum(bool(r.get("site_mapping_failed")) for r in rows), len(rows)),
        "collision_rate": ratio(sum(bool(r.get("collision_flag")) for r in rows), len(rows)),
        "too_short_distance_rate": ratio(sum(bool(r.get("too_short_distance")) for r in rows), len(rows)),
    }
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    for k in (1, 5, 20):
        valid_any = formula_any = sg_any = exact_any = legal_any = 0
        for sid in samples:
            top = sorted(by_sid[sid], key=lambda r: int(r.get("rank") or 10**9))[:k]
            valid_any += int(any(bool(r.get("valid")) for r in top))
            formula_any += int(any(bool(r.get("formula_ok")) for r in top))
            sg_any += int(any(bool(r.get("space_group_ok")) for r in top))
            exact_any += int(any(bool(r.get("exact_cover_retained")) for r in top))
            legal_any += int(any(bool(r.get("legal_cif")) for r in top))
        out[f"valid_any@{k}"] = ratio(valid_any, len(samples))
        out[f"formula_ok_any@{k}"] = ratio(formula_any, len(samples))
        out[f"sg_ok_any@{k}"] = ratio(sg_any, len(samples))
        out[f"exact_cover_any@{k}"] = ratio(exact_any, len(samples))
        out[f"legal_cif_any@{k}"] = ratio(legal_any, len(samples))
    return out


def select_by_safe_checks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_sid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sid[str(row["sample_id"])].append(row)
    selected: list[dict[str, Any]] = []
    fallback_reasons: Counter[str] = Counter()
    for sid, sample_rows in by_sid.items():
        ordered = sorted(sample_rows, key=lambda r: int(r.get("rank") or 10**9))
        valid = [
            r
            for r in ordered
            if bool(r.get("legal_cif"))
            and bool(r.get("formula_ok"))
            and bool(r.get("space_group_ok"))
            and bool(r.get("exact_cover_retained"))
            and bool(r.get("site_count_ok"))
            and not bool(r.get("collision_flag"))
        ]
        if valid:
            row = dict(valid[0])
            row["selection_reason"] = "first_valid_formula_sg_exact_cover_no_collision"
            selected.append(row)
            continue
        partial = [
            r
            for r in ordered
            if bool(r.get("legal_cif"))
            and bool(r.get("formula_ok"))
            and bool(r.get("exact_cover_retained"))
            and bool(r.get("site_count_ok"))
        ]
        if partial:
            row = dict(partial[0])
            row["selection_reason"] = "fallback_legal_formula_exact_cover"
            fallback_reasons["fallback_legal_formula_exact_cover"] += 1
            selected.append(row)
            continue
        row = dict(ordered[0])
        row["selection_reason"] = "fallback_rank1"
        fallback_reasons["fallback_rank1"] += 1
        selected.append(row)
    return selected, {
        "selected_samples": len(selected),
        "fallback_reasons": dict(fallback_reasons),
        "fallback_rate": ratio(sum(fallback_reasons.values()), len(selected)),
    }


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def append_or_replace_report(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker in text:
        start = text.index(marker)
        next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
        if next_marker == -1:
            REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
        else:
            REPORT_PATH.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(replacement)


def report_body(result: dict[str, Any]) -> str:
    det_all = result["modes"]["deterministic"]["overall"]
    det_r7 = result["modes"]["deterministic"]["rows_ge7"]
    proto_all = result["modes"]["train_prototype"]["overall"]
    proto_r7 = result["modes"]["train_prototype"]["rows_ge7"]
    selected_all = result["modes"]["train_prototype"]["selected_by_safe_checks"]["overall"]
    selected_r7 = result["modes"]["train_prototype"]["selected_by_safe_checks"]["rows_ge7"]
    selected_meta = result["modes"]["train_prototype"]["selected_by_safe_checks"]["selection"]
    gate = result["gate"]
    return f"""## opentry_13 实验 2：predicted skeleton renderer / site mapping 稳定性

结果文件：`model/New_model/opentry_13/results/experiment_2_predicted_skeleton_renderer_site_mapping.json`

- 实验逻辑：使用 opentry_13 实验 3 的 validation predicted exact-cover skeleton proposals，固定 composition + GT-SG + predicted skeleton，只检查 skeleton 到 CIF 的 renderer/site mapping 是否保结构。没有训练 scorer，没有看 StructureMatcher match@k，没有使用 test true CIF。
- 为什么做：predicted skeleton 接 repair 后崩掉的首要风险是渲染链路不保 formula/SG/exact-cover，而不是模型分数；本实验先做结构完整性 gate，决定是否允许进入 learned geometry repair。
- 数据规模：samples={result['data_scale']['samples']}，rows>=7 samples={result['data_scale']['rows_ge7_samples']}，candidate records={result['data_scale']['candidate_records']}，top_k={result['data_scale']['top_k']}。
- 方法变化：比较两种 site mapping：deterministic composition exact-cover mapping，以及 train-prototype-preferred mapping。几何只作为渲染初始化：deterministic mode 用 train SG/global median lattice + deterministic free params；train-prototype mode 用 train source lattice/free params，不使用 GT-WA 作为推理输入。
- deterministic overall：valid={pct(det_all['valid_rate'])}，formula={pct(det_all['formula_consistency'])}，SG={pct(det_all['sg_consistency'])}，exact-cover={pct(det_all['exact_cover_retained'])}，legal CIF={pct(det_all['legal_cif_rate'])}，site-count={pct(det_all['site_count_consistency'])}，row-count={pct(det_all['skeleton_row_count_consistency'])}，site-mapping failure={pct(det_all['site_mapping_failure_rate'])}，collision={pct(det_all['collision_rate'])}。
- deterministic rows>=7：valid={pct(det_r7['valid_rate'])}，formula={pct(det_r7['formula_consistency'])}，SG={pct(det_r7['sg_consistency'])}，exact-cover={pct(det_r7['exact_cover_retained'])}，legal CIF={pct(det_r7['legal_cif_rate'])}，site-count={pct(det_r7['site_count_consistency'])}，row-count={pct(det_r7['skeleton_row_count_consistency'])}。
- train-prototype overall：valid={pct(proto_all['valid_rate'])}，formula={pct(proto_all['formula_consistency'])}，SG={pct(proto_all['sg_consistency'])}，exact-cover={pct(proto_all['exact_cover_retained'])}，legal CIF={pct(proto_all['legal_cif_rate'])}，site-count={pct(proto_all['site_count_consistency'])}，row-count={pct(proto_all['skeleton_row_count_consistency'])}，site-mapping failure={pct(proto_all['site_mapping_failure_rate'])}，collision={pct(proto_all['collision_rate'])}。
- train-prototype rows>=7：valid={pct(proto_r7['valid_rate'])}，formula={pct(proto_r7['formula_consistency'])}，SG={pct(proto_r7['sg_consistency'])}，exact-cover={pct(proto_r7['exact_cover_retained'])}，legal CIF={pct(proto_r7['legal_cif_rate'])}，site-count={pct(proto_r7['site_count_consistency'])}，row-count={pct(proto_r7['skeleton_row_count_consistency'])}。
- renderer/site-mapping fixed selector：只用 inference-safe structural checks 在 top20 中选第一个 legal/formula/SG/exact-cover/site-count/no-collision CIF，不看 match。selected overall valid={pct(selected_all['valid_rate'])}，formula={pct(selected_all['formula_consistency'])}，SG={pct(selected_all['sg_consistency'])}，exact-cover={pct(selected_all['exact_cover_retained'])}；selected rows>=7 valid={pct(selected_r7['valid_rate'])}，formula={pct(selected_r7['formula_consistency'])}，SG={pct(selected_r7['sg_consistency'])}，exact-cover={pct(selected_r7['exact_cover_retained'])}；fallback_rate={pct(selected_meta['fallback_rate'])}。
- gate 判定：passed={gate['passed']}；candidate_level_train_prototype_passed={gate['train_prototype_passed']}；selected_train_prototype_passed={gate['selected_train_prototype_passed']}；deterministic_passed={gate['deterministic_passed']}；失败原因={gate['failure_reasons']}。
- 可信度：中高。该实验覆盖所有 exp3 proposal 记录并真实 render/parse/SG-detect，但 valid 仍依赖当前 `SpacegroupAnalyzer(symprec=0.1)` 和 0.5A collision 阈值；它是结构 gate，不是 match 指标。
- 和历史实验关系：承接 opentry_13 实验 3 的 predicted skeleton proposals，并解释 opentry_13 旧 repair 结果中 formula/SG/valid 崩掉是否来自 site mapping/renderer。
- 最终判决：如果 gate 不过，禁止进入 learned geometry repair 或 official；必须继续修 renderer/site mapping。
- 下一步：只有当至少一个 mapping mode 过结构 gate，才继续 predicted-skeleton-aware learned geometry repair；否则优先修 SG/rendering、row expansion 或 geometry initializer。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    structured_train = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    structured_val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    train_median_by_sg, train_global_median = median_lattice_by_sg(list(structured_train.values()))
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(structured_val.values()) + list(structured_train.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)

    selected_sids = [sid for sid in sorted(proposals) if sid in val_repr and sid in structured_val]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    mapping_failures: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for sid in selected_sids:
        target_repr = val_repr[sid]
        target_structured = structured_val[sid]
        target_counts = formula_counts(target_structured)
        target_atom_count = sum(target_counts.values())
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_k)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_structured = structured_train.get(source_id)
            for mode, prefer_source in (("deterministic", False), ("train_prototype", True)):
                base = {
                    "sample_id": sid,
                    "material_id": str(target_structured.get("material_id") or sid.split("__")[-1]),
                    "rank": int(proposal.get("rank") or 0),
                    "mode": mode,
                    "sg": int(target_structured["sg"]),
                    "sg_symbol": str(target_structured.get("sg_symbol") or ""),
                    "row_count": int(target_repr.get("row_count") or 0),
                    "target_atom_count": int(target_atom_count),
                    "formula_counts": target_counts,
                    "source_sample_id": source_id,
                    "proposal_source": str(proposal.get("source") or ""),
                    "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                    "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                    "proposal_multiplicities": list(proposal.get("multiplicities") or []),
                }
                source_counts[base["proposal_source"]] += 1
                if source_repr is None:
                    row = dict(base)
                    row.update(
                        {
                            "render_success": False,
                            "site_mapping_failed": True,
                            "site_mapping_error": "missing_source_repr",
                            "exact_cover_retained": False,
                            "skeleton_row_count_ok": False,
                            "target_row_count_match": False,
                            "cif": "",
                        }
                    )
                    mapping_failures[f"{mode}:missing_source_repr"] += 1
                    payloads.append(row)
                    meta_rows.append({k: v for k, v in row.items() if k != "cif"})
                    continue
                rows, mapping_rule, mapping_error = candidate_rows_for_mapping(
                    engine=engine,
                    target_record=target_structured,
                    source_repr=source_repr,
                    prefer_source_elements=prefer_source,
                )
                if rows is None:
                    row = dict(base)
                    row.update(
                        {
                            "render_success": False,
                            "site_mapping_failed": True,
                            "site_mapping_rule": mapping_rule,
                            "site_mapping_error": mapping_error,
                            "exact_cover_retained": False,
                            "skeleton_row_count_ok": False,
                            "target_row_count_match": False,
                            "cif": "",
                        }
                    )
                    mapping_failures[f"{mode}:{mapping_error}"] += 1
                    payloads.append(row)
                    meta_rows.append({k: v for k, v in row.items() if k != "cif"})
                    continue
                params, param_fallbacks, param_source = params_for_mode(
                    mode=mode,
                    engine=engine,
                    rows=rows,
                    source_structured=source_structured,
                )
                lattice, lattice_source = lattice_for_mode(
                    mode=mode,
                    target_sg=int(target_structured["sg"]),
                    source_structured=source_structured,
                    median_by_sg=train_median_by_sg,
                    global_median=train_global_median,
                )
                try:
                    expanded = int(engine.expanded_atom_count(rows, params))
                    exact_cover_retained = expanded == target_atom_count
                    skeleton_row_count_ok = len(rows) == len(base["proposal_multiplicities"])
                    target_row_count_match = len(rows) == int(target_repr.get("row_count") or 0)
                    cif = engine.render_cif_from_wa_table(
                        rows,
                        lattice=lattice,
                        free_params_by_row=params,
                        formula_counts=target_counts,
                        sg=int(target_structured["sg"]),
                        sg_symbol=str(target_structured.get("sg_symbol") or ""),
                        data_name=f"{sid}_{mode}_rank{base['rank']}",
                    )
                    row = dict(base)
                    row.update(
                        {
                            "render_success": True,
                            "site_mapping_failed": False,
                            "site_mapping_rule": mapping_rule,
                            "site_mapping_error": None,
                            "exact_cover_retained": exact_cover_retained,
                            "skeleton_row_count_ok": skeleton_row_count_ok,
                            "target_row_count_match": target_row_count_match,
                            "candidate_skeleton_rows": len(rows),
                            "atom_count_after_expansion": expanded,
                            "param_source": param_source,
                            "param_fallback_rows": int(param_fallbacks),
                            "lattice_source": lattice_source,
                            "cif": cif,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    row = dict(base)
                    row.update(
                        {
                            "render_success": False,
                            "site_mapping_failed": False,
                            "site_mapping_rule": mapping_rule,
                            "site_mapping_error": None,
                            "render_error": f"{type(exc).__name__}: {exc}",
                            "exact_cover_retained": False,
                            "skeleton_row_count_ok": False,
                            "target_row_count_match": False,
                            "cif": "",
                        }
                    )
                payloads.append(row)
                meta_rows.append({k: v for k, v in row.items() if k != "cif"})

    write_jsonl(ARTIFACT_DIR / "render_payload_meta.jsonl", meta_rows)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_rendered, payload) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.append(fut.result())
            if i % 5000 == 0:
                print(f"[exp2-renderer] evaluated {i}/{len(futures)}", flush=True)
    evaluated.sort(key=lambda r: (str(r["sample_id"]), int(r.get("rank") or 0), str(r.get("mode"))))
    write_jsonl(ARTIFACT_DIR / "evaluated_renderer_site_mapping.jsonl", evaluated)

    modes: dict[str, dict[str, Any]] = {}
    for mode in ("deterministic", "train_prototype"):
        rows = [r for r in evaluated if str(r.get("mode")) == mode]
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        modes[mode] = {
            "overall": summarize(rows),
            "rows_ge7": summarize(rows7),
            "rows_lt7": summarize(rowslt7),
        }

    def mode_pass(m: dict[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        all_m = m["overall"]
        r7 = m["rows_ge7"]
        checks = [
            ("overall valid", all_m.get("valid_rate"), 0.95),
            ("overall formula", all_m.get("formula_consistency"), 0.95),
            ("overall SG", all_m.get("sg_consistency"), 0.95),
            ("overall exact-cover", all_m.get("exact_cover_retained"), 0.95),
            ("rows>=7 valid", r7.get("valid_rate"), 0.90),
            ("rows>=7 formula", r7.get("formula_consistency"), 0.95),
            ("rows>=7 SG", r7.get("sg_consistency"), 0.95),
            ("rows>=7 exact-cover", r7.get("exact_cover_retained"), 0.95),
        ]
        for label, value, threshold in checks:
            if value is None or float(value) < threshold:
                reasons.append(f"{label} {pct(value)} < {pct(threshold)}")
        return not reasons, reasons

    det_pass, det_reasons = mode_pass(modes["deterministic"])
    proto_pass, proto_reasons = mode_pass(modes["train_prototype"])
    selected_rows, selected_meta = select_by_safe_checks([r for r in evaluated if str(r.get("mode")) == "train_prototype"])
    selected_rows7 = [r for r in selected_rows if int(r.get("row_count") or 0) >= 7]
    selected_rowslt7 = [r for r in selected_rows if int(r.get("row_count") or 0) < 7]
    selected_block = {
        "selection": selected_meta,
        "overall": summarize(selected_rows),
        "rows_ge7": summarize(selected_rows7),
        "rows_lt7": summarize(selected_rowslt7),
    }
    modes["train_prototype"]["selected_by_safe_checks"] = selected_block
    selected_pass, selected_reasons = mode_pass(selected_block)
    gate = {
        "deterministic_passed": det_pass,
        "train_prototype_passed": proto_pass,
        "selected_train_prototype_passed": selected_pass,
        "passed": det_pass or proto_pass or selected_pass,
        "failure_reasons": {
            "deterministic": det_reasons,
            "train_prototype": proto_reasons,
            "selected_train_prototype": selected_reasons,
        },
        "minimum_standard": {
            "overall_valid": 0.95,
            "overall_formula": 0.95,
            "overall_sg": 0.95,
            "overall_exact_cover": 0.95,
            "rows_ge7_valid": 0.90,
            "rows_ge7_formula": 0.95,
            "rows_ge7_sg": 0.95,
            "rows_ge7_exact_cover": 0.95,
        },
    }

    result = {
        "experiment": "opentry_13_exp2_predicted_skeleton_renderer_site_mapping_stability",
        "created_at_utc": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "inputs": ["composition/formula", "GT-SG", "predicted exact-cover skeleton proposal"],
            "not_used": ["StructureMatcher match", "RMSD", "match label", "test CIF", "GT-WA as inference input", "RF/HGB/scorer"],
            "modes": {
                "deterministic": "composition exact-cover site mapping, train SG/global median lattice, deterministic free params",
                "train_prototype": "source-preferred exact-cover site mapping, train source lattice/free params with deterministic fallback",
            },
        },
        "data_scale": {
            "samples": len(selected_sids),
            "rows_ge7_samples": sum(int(val_repr[sid].get("row_count") or 0) >= 7 for sid in selected_sids),
            "top_k": int(args.top_k),
            "candidate_records": len(evaluated),
            "proposal_records_per_mode": len(evaluated) // 2,
            "source_counts": dict(source_counts),
        },
        "mapping_failures": dict(mapping_failures),
        "modes": modes,
        "gate": gate,
        "runtime_seconds": time.time() - started,
        "artifacts": {
            "render_payload_meta": str(ARTIFACT_DIR / "render_payload_meta.jsonl"),
            "evaluated_renderer_site_mapping": str(ARTIFACT_DIR / "evaluated_renderer_site_mapping.jsonl"),
        },
    }
    write_json(RESULT_DIR / "experiment_2_predicted_skeleton_renderer_site_mapping.json", result)
    append_or_replace_report("<!-- OPENTRY13_EXP2_RENDERER_SITE_MAPPING -->", report_body(result))
    print(json.dumps({"result": str(RESULT_DIR / "experiment_2_predicted_skeleton_renderer_site_mapping.json"), "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
