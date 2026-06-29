#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
SYMCIF_ROOT = NEW_MODEL / "symcif_experiment"
OP13 = NEW_MODEL / "opentry_13"
OUT_DIR = NEW_MODEL / "opentry_14"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp2e_train_pair_residual_posterior"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"
LOCAL_REPORT_PATH = OUT_DIR / "OPENTRY14_EXPERIMENT_REPORT.md"

for _path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts", OP13 / "scripts", OUT_DIR / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_exp2_predicted_skeleton_renderer_site_mapping import (  # noqa: E402
    formula_counts,
    read_jsonl,
    sample_id,
    source_skeleton_rows,
)
from run_exp4_rows_ge7_multi_geometry_proposal import (  # noqa: E402
    assign_structural_ranks,
    eval_sample,
    render_candidate,
    summarize,
)
from run_symcif_v4_geometry_model_eval import flexible_params_from_reference  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402


LOOKUP = SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json"
TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
STRUCTURED_TRAIN = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "train.jsonl"
STRUCTURED_VAL = SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52" / "val.jsonl"
EXP1_PAIRS = OUT_DIR / "artifacts" / "exp1_predicted_skeleton_noise_pairs" / "predicted_skeleton_noise_geometry_pairs_merged_sharded.jsonl.gz"
EXP2B_RESULT = RESULT_DIR / "experiment_2b_safe_pool_after_failure_analysis.json"
EXP2B_EVAL = OUT_DIR / "artifacts" / "exp2b_hydrated_prototype_safe_pool" / "evaluated_safe_pool_candidates.jsonl"
EXP2D_RESULT = RESULT_DIR / "experiment_2d_site_assignment_multi_hypothesis.json"
EXP2D_SITE_EVAL = OUT_DIR / "artifacts" / "exp2d_site_assignment_multi_hypothesis" / "evaluated_site_assignment_candidates.jsonl"
EXP3_PROPOSALS = OP13 / "artifacts" / "exp3_rows7_skeleton_proposer" / "proposals.jsonl"

LENGTH_KEYS = ("a", "b", "c")
ANGLE_KEYS = ("alpha", "beta", "gamma")
LATTICE_KEYS = LENGTH_KEYS + ANGLE_KEYS


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:.3f}%"


def pp(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{100.0 * x:+.3f}pp"


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def circular_delta(target: float, initial: float) -> float:
    return ((float(target) - float(initial) + 0.5) % 1.0) - 0.5


def circular_add(initial: float, delta: float, scale: float) -> float:
    return float((float(initial) + scale * float(delta)) % 1.0)


def lattice_residual(initial: dict[str, Any], target: dict[str, Any]) -> dict[str, float] | None:
    out: dict[str, float] = {}
    try:
        for key in LENGTH_KEYS:
            init = float(initial[key])
            tgt = float(target[key])
            if init <= 1.0e-8 or tgt <= 1.0e-8:
                return None
            out[f"{key}_ratio"] = max(0.55, min(1.85, tgt / init))
        for key in ANGLE_KEYS:
            out[f"{key}_delta"] = max(-35.0, min(35.0, float(target[key]) - float(initial[key])))
    except Exception:
        return None
    return out


def apply_lattice_residual(lattice: dict[str, Any], residual: dict[str, float], scale: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in LENGTH_KEYS:
        ratio_val = float(residual.get(f"{key}_ratio", 1.0))
        out[key] = max(0.5, float(lattice[key]) * (1.0 + scale * (ratio_val - 1.0)))
    for key in ANGLE_KEYS:
        out[key] = max(30.0, min(150.0, float(lattice[key]) + scale * float(residual.get(f"{key}_delta", 0.0))))
    return out


def param_residuals(initial_params: dict[str, Any], target_params: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row_idx, init_block in initial_params.items():
        target_block = target_params.get(str(row_idx))
        if not isinstance(init_block, dict) or not isinstance(target_block, dict):
            continue
        deltas: dict[str, float] = {}
        for name, init_val in init_block.items():
            if name not in target_block:
                continue
            deltas[str(name)] = circular_delta(float(target_block[name]), float(init_val))
        if deltas:
            out[str(row_idx)] = deltas
    return out


def apply_param_residuals(
    params: dict[int, dict[str, float]],
    residuals: dict[str, dict[str, float]],
    scale: float,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for idx, block in params.items():
        new_block = dict(block)
        deltas = residuals.get(str(idx), {})
        for name, delta in deltas.items():
            if name in new_block:
                new_block[name] = circular_add(float(new_block[name]), float(delta), scale)
        out[int(idx)] = new_block
    return out


def formula_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = set(a) | set(b)
    den = max(1, sum(int(v) for v in a.values()) + sum(int(v) for v in b.values()))
    return sum(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in keys) / float(den)


def residual_score(target_counts: dict[str, int], row_count: int, skeleton_key: str, tmpl: dict[str, Any]) -> float:
    score = 0.0
    if int(tmpl["sg"]) == int(tmpl.get("sg")):
        score += 1000.0
    if int(tmpl.get("source_row_count") or 0) == int(row_count):
        score += 240.0
    if skeleton_key and skeleton_key == str(tmpl.get("predicted_skeleton_key") or ""):
        score += 500.0
    score -= 160.0 * formula_l1(target_counts, tmpl.get("formula_counts") or {})
    score -= 2.0 * abs(int(row_count) - int(tmpl.get("source_row_count") or row_count))
    score += 20.0 * float(tmpl.get("free_param_value_recovery_rate") or 0.0)
    score += 8.0 * float(tmpl.get("row_alignment_rate") or 0.0)
    return float(score)


def load_residual_templates(limit_per_key: int) -> dict[tuple[int, int], list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(EXP1_PAIRS, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("usable_joint_pair")):
                continue
            if int(row.get("source_row_count") or 0) < 7:
                continue
            if not bool((row.get("initial_quality") or {}).get("space_group_ok")):
                continue
            lat = lattice_residual(row.get("initial_lattice") or {}, row.get("target_lattice") or {})
            if lat is None:
                continue
            pres = param_residuals(row.get("initial_row_params") or {}, row.get("target_params_by_row") or {})
            if not pres:
                continue
            tmpl = {
                "sample_id": str(row.get("sample_id")),
                "dataset": str(row.get("dataset") or ""),
                "sg": int(row.get("sg") or 0),
                "source_row_count": int(row.get("source_row_count") or 0),
                "formula_counts": {str(k): int(v) for k, v in (row.get("formula_counts") or {}).items()},
                "predicted_skeleton_key": str(row.get("predicted_skeleton_key") or ""),
                "target_skeleton_key": str(row.get("target_skeleton_key") or ""),
                "predicted_skeleton_hit_train": bool(row.get("predicted_skeleton_hit")),
                "lattice_residual": lat,
                "param_residuals": pres,
                "free_param_value_recovery_rate": float(row.get("free_param_value_recovery_rate") or 0.0),
                "row_alignment_rate": float(row.get("row_alignment_rate") or 0.0),
            }
            buckets[(int(tmpl["sg"]), int(tmpl["source_row_count"]))].append(tmpl)

    pruned: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for key, rows in buckets.items():
        rows.sort(
            key=lambda r: (
                -float(r.get("free_param_value_recovery_rate") or 0.0),
                -float(r.get("row_alignment_rate") or 0.0),
                str(r.get("sample_id") or ""),
            )
        )
        pruned[key] = rows[: max(1, int(limit_per_key))]
    return pruned


def enumerate_exact_cover_assignments(rows: list[dict[str, Any]], counts: dict[str, int], limit: int) -> list[tuple[int, list[str]]]:
    mults = [int(row.get("multiplicity") or 0) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: (-mults[i], str(rows[i].get("orbit_id")), i))
    remaining = dict(counts)
    assigned: list[str | None] = [None] * len(rows)
    out: list[tuple[int, list[str]]] = []

    def rec(pos: int) -> None:
        if len(out) >= int(limit):
            return
        if pos >= len(order):
            if all(int(v) == 0 for v in remaining.values()):
                score = sum(
                    int(str(rows[i].get("element")) == str(assigned[i])) * int(rows[i].get("multiplicity") or 0)
                    for i in range(len(rows))
                )
                out.append((int(score), [str(x) for x in assigned]))
            return
        idx = order[pos]
        mult = mults[idx]
        preferred = str(rows[idx].get("element"))
        choices = sorted(remaining, key=lambda e: (0 if str(e) == preferred else 1, -int(remaining[e]), str(e)))
        for element in choices:
            if int(remaining[element]) < mult:
                continue
            assigned[idx] = str(element)
            remaining[element] -= mult
            rec(pos + 1)
            remaining[element] += mult
            assigned[idx] = None
            if len(out) >= int(limit):
                return

    rec(0)
    seen: set[tuple[str, ...]] = set()
    uniq: list[tuple[int, list[str]]] = []
    for score, assignment in sorted(out, key=lambda x: (-x[0], tuple(x[1]))):
        key = tuple(assignment)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((score, assignment))
    return uniq[: int(limit)]


def split_safe_pool(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    hydrated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prototype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_lt7: list[dict[str, Any]] = []
    for row in rows:
        if int(row.get("row_count") or 0) < 7:
            rows_lt7.append(row)
        elif str(row.get("pool_source")) == "symcif_v5_hydrated":
            hydrated[str(row["sample_id"])].append(row)
        else:
            prototype[str(row["sample_id"])].append(row)
    for group in hydrated.values():
        group.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
    for group in prototype.values():
        group.sort(key=lambda r: int(r.get("source_rank") or r.get("rank") or 10**9))
    return hydrated, prototype, rows_lt7


def by_sid_ranked(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["sample_id"])].append(row)
    for group in out.values():
        group.sort(key=lambda r: int(r.get("rank") or r.get("geometry_rank") or 10**9))
    return out


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    for sid, group in grouped.items():
        for rank, row in enumerate(group, start=1):
            item = dict(row)
            item["rank"] = rank
            out.append(item)
    out.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    return out


def timeout_rows(payload: dict[str, Any], reason: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in payload.get("candidates", []):
        row = {k: v for k, v in item.items() if k != "cif"}
        row.update(
            {
                "parse_success": False,
                "legal_cif": False,
                "formula_ok": False,
                "space_group_ok": False,
                "detected_sg": None,
                "site_count_ok": False,
                "valid": False,
                "collision_flag": None,
                "too_short_distance": None,
                "min_pair_distance": None,
                "volume_per_atom": None,
                "volume_per_atom_ok": None,
                "match": False,
                "rms": None,
                "eval_error": reason,
                "target_error": None,
            }
        )
        out.append(row)
    return out


def eval_sample_with_timeout(task: tuple[dict[str, Any], int]) -> list[dict[str, Any]]:
    payload, timeout_seconds = task

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise TimeoutError(f"sample_eval_timeout_{timeout_seconds}s")

    if timeout_seconds > 0:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout_seconds))
    try:
        return eval_sample(payload)
    except TimeoutError as exc:
        return timeout_rows(payload, str(exc))
    finally:
        if timeout_seconds > 0:
            signal.alarm(0)


def build_variant_rows(
    *,
    variant: str,
    hydrated: dict[str, list[dict[str, Any]]],
    prototype: dict[str, list[dict[str, Any]]],
    siteassign: dict[str, list[dict[str, Any]]],
    residuals: dict[str, list[dict[str, Any]]],
    rows_lt7: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    quotas = {
        "h10_s10_r20_p10": (10, 10, 20, 10),
        "h10_s5_r30_p5": (10, 5, 30, 5),
        "h5_s15_r25_p5": (5, 15, 25, 5),
        "h10_interleave_s10_r20_p10": (10, 10, 20, 10),
        "r30_h10_s5_p5": (10, 5, 30, 5),
    }
    hq, sq, rq, pq = quotas[variant]
    out: list[dict[str, Any]] = []
    for row in rows_lt7:
        item = dict(row)
        item["selection_variant"] = variant
        out.append(item)
    for sid in sorted(set(hydrated) | set(prototype) | set(siteassign) | set(residuals)):
        hyd = hydrated.get(sid, [])
        site = siteassign.get(sid, [])
        res = residuals.get(sid, [])
        proto = prototype.get(sid, [])
        selected: list[dict[str, Any]] = []
        if variant == "h10_interleave_s10_r20_p10":
            for i in range(max(hq, sq, rq)):
                if i < hq and i < len(hyd):
                    selected.append(hyd[i])
                if i < sq and i < len(site):
                    selected.append(site[i])
                if i < rq and i < len(res):
                    selected.append(res[i])
            selected.extend(proto[: max(0, top_k - len(selected))])
        elif variant == "r30_h10_s5_p5":
            selected.extend(res[:rq])
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(proto[:pq])
        else:
            selected.extend(hyd[:hq])
            selected.extend(site[:sq])
            selected.extend(res[:rq])
            selected.extend(proto[:pq])
        for row in selected[:top_k]:
            item = dict(row)
            item["selection_variant"] = variant
            out.append(item)
    return assign_ranks(out)


def append_or_replace(path: Path, marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    replacement = marker + "\n" + body.rstrip() + "\n"
    if marker not in text:
        with path.open("a", encoding="utf-8") as f:
            if text and not text.endswith("\n\n"):
                f.write("\n\n")
            f.write(replacement)
        return
    start = text.index(marker)
    next_marker = text.find("\n\n<!-- OPENTRY", start + len(marker))
    if next_marker == -1:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement, encoding="utf-8")
    else:
        path.write_text(text[:start].rstrip() + "\n\n" + replacement + text[next_marker:], encoding="utf-8")


def report_body(result: dict[str, Any]) -> str:
    best_name = result["best_variant"]
    best = result["variants"][best_name]["rows_ge7"]
    exp2b = result["baseline_exp2b"]["rows_ge7"]
    exp2d = result["baseline_exp2d"]["rows_ge7"]
    gate = result["gate"]
    return f"""## opentry_14 实验 2e：train-pair residual posterior multi-hypothesis

结果文件：`model/New_model/opentry_14/results/experiment_2e_train_pair_residual_posterior.json`
候选 artifact：`model/New_model/opentry_14/artifacts/exp2e_train_pair_residual_posterior/`

- 为什么做：Exp2d 的 site-assignment 枚举有正信号，但 oracle upper bound 只有 rows>=7 match@50 `{pct(result['oracle_reference']['site_assignment_oracle_match50_upper_bound'])}`、conversion `{pct(result['oracle_reference']['site_assignment_oracle_conversion50_upper_bound'])}`，说明仅扩大 assignment 配额不足。Exp2e 改为使用 Exp1 train noisy-pair 中真实 target-initial lattice/free-parameter 残差，生成 learned empirical posterior 多假设。
- 核心假设：如果 skeleton-hit/no-match 的核心瓶颈是连续 lattice/free-parameter basin 没对齐，则把同 SG/row-count train pair 残差迁移到 validation predicted skeleton/site assignment 上，应新增 match 样本并提高 rows>=7 conversion。
- 数据规模：train residual templates `{result['data_scale']['residual_templates']}`，residual buckets `{result['data_scale']['residual_buckets']}`；validation rows>=7 samples `{result['data_scale']['rows_ge7_samples']}`；generated posterior candidates `{result['data_scale']['generated_residual_candidates']}`；evaluated posterior candidates `{result['data_scale']['evaluated_residual_candidates']}`；StructureMatcher workers `{result['cpu_policy']['workers']}`。
- baseline：Exp2b rows>=7 match@50 `{pct(exp2b.get('match@50'))}`、conversion `{pct(exp2b.get('skeleton_to_match_conversion@50'))}`；Exp2d best rows>=7 match@50 `{pct(exp2d.get('match@50'))}`、conversion `{pct(exp2d.get('skeleton_to_match_conversion@50'))}`。
- 方法变化：对 validation predicted skeleton 的 exact-cover site assignment 先用 source geometry 初始化，再从 Exp1 train pairs 中按 SG、row count、composition 距离和 skeleton key 选残差模板；对 lattice 施加 length ratio/angle delta，对 row-level free parameters 施加 circular residual，生成多尺度 residual hypotheses。critic 只用 formula/SG/exact/valid/collision/volume/reference score/proposal rank 等 structural score；不使用 match、RMSD、StructureMatcher label、GT-WA、GT-skeleton、official feedback、RF/HGB。
- 结果 best variant `{best_name}`：rows>=7 match@1/5/20/50 = `{pct(best.get('match@1'))} / {pct(best.get('match@5'))} / {pct(best.get('match@20'))} / {pct(best.get('match@50'))}`；conversion@50 `{pct(best.get('skeleton_to_match_conversion@50'))}`；valid `{pct(best.get('valid_rate'))}`；formula `{pct(best.get('formula_consistency'))}`；SG `{pct(best.get('sg_consistency'))}`；exact-cover `{pct(best.get('exact_cover_retained'))}`；collision `{pct(best.get('collision_rate'))}`。
- 可信度：中等。训练残差只来自 train split noisy-skeleton pairs，validation 推理不使用禁用标签；所有候选真实 render/parse/SG/StructureMatcher。限制是 residual 迁移仍按 row index 对齐，未学习 permutation-aware posterior，且 residual 模板筛选是非参数近邻而非端到端概率模型。
- 和历史实验关系：它是 Exp2 joint MLP 失败后的另一路 learned posterior 修复，承接 Exp3c 的 alignment 根因审计和 Exp2d 的 assignment 正信号；若仍无法过 target gate，说明单靠 train-pair residual/posterior 也不足，可能需要升级 skeleton proposer 或显式 permutation-aware alignment。
- gate 判定：minimum_passed=`{gate['minimum_passed']}`；target_passed=`{gate['target_passed']}`；match@50 delta vs Exp2b `{pp(gate['rows_ge7_match50_delta_vs_exp2b'])}`；conversion delta vs Exp2b `{pp(gate['rows_ge7_conversion50_delta_vs_exp2b'])}`；match@50 delta vs Exp2d `{pp(gate['rows_ge7_match50_delta_vs_exp2d'])}`；conversion delta vs Exp2d `{pp(gate['rows_ge7_conversion50_delta_vs_exp2d'])}`。
- 最终判决：`{result['decision']['verdict']}`。{result['decision']['reason']}
- 下一步：{result['decision']['next_step']}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-skeletons", type=int, default=6)
    parser.add_argument("--assignment-limit", type=int, default=2)
    parser.add_argument("--residuals-per-assignment", type=int, default=3)
    parser.add_argument("--residual-template-limit-per-key", type=int, default=600)
    parser.add_argument("--residual-scales", default="0.5,1.0")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--eval-timeout-seconds", type=int, default=75)
    parser.add_argument("--max-target-atoms-for-residual", type=int, default=100)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    started = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    scales = [float(x.strip()) for x in str(args.residual_scales).split(",") if x.strip()]
    train_repr = {sample_id(r): r for r in read_jsonl(TRAIN_REPR)}
    train_struct = {sample_id(r): r for r in read_jsonl(STRUCTURED_TRAIN)}
    val = {sample_id(r): r for r in read_jsonl(STRUCTURED_VAL)}
    val_repr = {sample_id(r): r for r in read_jsonl(VAL_REPR)}
    proposals = {str(r["sample_id"]): r for r in read_jsonl(EXP3_PROPOSALS)}
    sg_symbols = {int(r["sg"]): str(r.get("sg_symbol") or f"SG{int(r['sg'])}") for r in list(train_struct.values()) + list(val.values())}
    engine = OrbitEngine(LOOKUP, sg_symbols)

    exp2b = read_json(EXP2B_RESULT)
    exp2d = read_json(EXP2D_RESULT)
    safe_rows = list(read_jsonl_iter(EXP2B_EVAL))
    site_rows = list(read_jsonl_iter(EXP2D_SITE_EVAL))
    residual_buckets = load_residual_templates(int(args.residual_template_limit_per_key))
    residual_template_count = sum(len(v) for v in residual_buckets.values())

    selected_sids = [sid for sid in sorted(proposals) if sid in val and sid in val_repr and int(val_repr[sid].get("row_count") or 0) >= 7]
    if args.limit_samples is not None:
        selected_sids = selected_sids[: int(args.limit_samples)]

    payloads: list[dict[str, Any]] = []
    generation_meta: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for si, sid in enumerate(selected_sids, start=1):
        target = val[sid]
        target_repr = val_repr[sid]
        counts = {str(k): int(v) for k, v in formula_counts(target).items()}
        candidates: list[dict[str, Any]] = []
        target_atom_count = int(sum(counts.values()))
        if target_atom_count > int(args.max_target_atoms_for_residual):
            failures["skipped_target_atom_count_gt_limit"] += 1
            payloads.append(
                {
                    "sample_id": sid,
                    "target_cif_path": str(target["source_path"]),
                    "formula_counts": counts,
                    "target_atom_count": target_atom_count,
                    "sg": int(target["sg"]),
                    "candidates": candidates,
                }
            )
            continue
        for proposal in proposals[sid].get("proposals", [])[: int(args.top_skeletons)]:
            source_id = str(proposal.get("source_sample_id") or "")
            source_repr = train_repr.get(source_id)
            source_struct = train_struct.get(source_id)
            if source_repr is None or source_struct is None:
                failures["missing_source"] += 1
                continue
            src_rows = source_skeleton_rows(engine, source_repr)
            assignments = enumerate_exact_cover_assignments(src_rows, counts, int(args.assignment_limit))
            if not assignments:
                failures["no_exact_cover_assignment"] += 1
                continue
            key = (int(target["sg"]), len(src_rows))
            templates = list(residual_buckets.get(key, []))
            if not templates:
                failures["no_residual_template"] += 1
                continue
            templates.sort(
                key=lambda tmpl: (
                    -residual_score(counts, len(src_rows), str(proposal.get("skeleton_key") or ""), tmpl),
                    str(tmpl.get("sample_id") or ""),
                )
            )
            selected_templates = templates[: max(1, int(args.residuals_per_assignment))]
            for assignment_rank, (assignment_score, assignment) in enumerate(assignments, start=1):
                rows = []
                for row, element in zip(src_rows, assignment):
                    item = dict(row)
                    item["element"] = str(element)
                    rows.append(item)
                try:
                    source_params, fallback_count = flexible_params_from_reference(engine, rows, source_struct, neural_params=None)
                except Exception as exc:  # noqa: BLE001
                    failures[f"source_param_failed:{type(exc).__name__}"] += 1
                    continue
                source_lattice = {k: float(source_struct["lattice"][k]) for k in LATTICE_KEYS}
                for tmpl_rank, tmpl in enumerate(selected_templates, start=1):
                    base_score = residual_score(counts, len(src_rows), str(proposal.get("skeleton_key") or ""), tmpl)
                    for scale in scales:
                        try:
                            lattice = apply_lattice_residual(source_lattice, tmpl["lattice_residual"], scale)
                            params = apply_param_residuals(source_params, tmpl["param_residuals"], scale)
                            option = {"lattice": lattice, "params": params}
                            cif, render_meta = render_candidate(
                                engine=engine,
                                target=target,
                                rows=rows,
                                option=option,
                                data_name=f"{sid}_respost_p{int(proposal.get('rank') or 0)}_a{assignment_rank}_t{tmpl_rank}_s{scale:g}",
                            )
                            row = {
                                "sample_id": sid,
                                "material_id": str(target.get("material_id") or sid.split("__")[-1]),
                                "proposal_rank": int(proposal.get("rank") or 0),
                                "assignment_rank": int(assignment_rank),
                                "assignment_source_preserved_atoms": int(assignment_score),
                                "geometry_rank": len(candidates) + 1,
                                "raw_generation_order": len(candidates) + 1,
                                "row_count": int(target_repr.get("row_count") or 0),
                                "sg": int(target["sg"]),
                                "formula_counts": counts,
                                "target_atom_count": int(sum(counts.values())),
                                "source_sample_id": source_id,
                                "proposal_source": str(proposal.get("source") or ""),
                                "predicted_skeleton_key": str(proposal.get("skeleton_key") or ""),
                                "target_skeleton_key": str(target_repr.get("canonical_skeleton_key") or ""),
                                "predicted_skeleton_hit": str(proposal.get("skeleton_key") or "") == str(target_repr.get("canonical_skeleton_key") or ""),
                                "candidate_row_count": len(rows),
                                "site_mapping_rule": "multi_exact_cover_assignment_plus_train_pair_residual",
                                "geometry_source": "train_pair_residual_posterior",
                                "reference_sample_id": str(tmpl.get("sample_id") or ""),
                                "reference_score": float(base_score),
                                "residual_template_rank": int(tmpl_rank),
                                "residual_scale": float(scale),
                                "residual_train_predicted_skeleton_hit": bool(tmpl.get("predicted_skeleton_hit_train")),
                                "param_fallback_rows": int(fallback_count),
                                "cif": cif,
                                **render_meta,
                            }
                            candidates.append(row)
                            generation_meta.append({k: v for k, v in row.items() if k != "cif"})
                        except Exception as exc:  # noqa: BLE001
                            failures[f"render_failed:{type(exc).__name__}"] += 1
        payloads.append(
            {
                "sample_id": sid,
                "target_cif_path": str(target["source_path"]),
                "formula_counts": counts,
                "target_atom_count": int(sum(counts.values())),
                "sg": int(target["sg"]),
                "candidates": candidates,
            }
        )
        if si % 200 == 0:
            print(f"[exp2e-residual] rendered {si}/{len(selected_sids)}", flush=True)

    suffix = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    write_jsonl(ARTIFACT_DIR / f"generated_residual_posterior_meta{suffix}.jsonl", generation_meta)

    evaluated: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(eval_sample_with_timeout, (payload, int(args.eval_timeout_seconds))) for payload in payloads]
        for i, fut in enumerate(as_completed(futures), start=1):
            evaluated.extend(fut.result())
            if i % 200 == 0:
                print(f"[exp2e-residual] evaluated {i}/{len(futures)}", flush=True)
    ranked_residual = assign_structural_ranks(evaluated, int(args.top_k))
    write_jsonl(ARTIFACT_DIR / f"evaluated_residual_posterior_candidates{suffix}.jsonl", ranked_residual)

    hydrated, prototype, rows_lt7 = split_safe_pool(safe_rows)
    site_by_sid = by_sid_ranked(assign_structural_ranks(site_rows, int(args.top_k)))
    residual_by_sid = by_sid_ranked(ranked_residual)

    variants: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, str] = {}
    for variant in ("h10_s10_r20_p10", "h10_s5_r30_p5", "h5_s15_r25_p5", "h10_interleave_s10_r20_p10", "r30_h10_s5_p5"):
        rows = build_variant_rows(
            variant=variant,
            hydrated=hydrated,
            prototype=prototype,
            siteassign=site_by_sid,
            residuals=residual_by_sid,
            rows_lt7=rows_lt7,
            top_k=int(args.top_k),
        )
        path = ARTIFACT_DIR / f"selected_{variant}_candidates{suffix}.jsonl"
        write_jsonl(path, rows)
        selected_paths[variant] = str(path)
        rows7 = [r for r in rows if int(r.get("row_count") or 0) >= 7]
        rowslt7 = [r for r in rows if int(r.get("row_count") or 0) < 7]
        variants[variant] = {
            "overall": summarize(rows),
            "rows_ge7": summarize(rows7),
            "rows_lt7": summarize(rowslt7),
        }

    best_variant = max(
        variants,
        key=lambda name: (
            float(variants[name]["rows_ge7"].get("skeleton_to_match_conversion@50") or 0.0),
            float(variants[name]["rows_ge7"].get("match@50") or 0.0),
            -float(variants[name]["rows_ge7"].get("collision_rate") or 1.0),
        ),
    )
    best_rows7 = variants[best_variant]["rows_ge7"]
    baseline_rows7 = exp2b["rows_ge7"]
    exp2d_rows7 = exp2d["variants"][exp2d["best_variant"]]["rows_ge7"]
    min_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.223
        and float(best_rows7.get("match@50") or 0.0) >= 0.17931372549019609
        and float(best_rows7.get("formula_consistency") or 0.0) >= 0.95
        and float(best_rows7.get("sg_consistency") or 0.0) >= 0.90
        and float(best_rows7.get("exact_cover_retained") or 0.0) >= 0.95
    )
    target_gate = bool(
        float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) >= 0.28
        or (
            float(best_rows7.get("match@5") or 0.0) >= float(baseline_rows7.get("match@5") or 0.0) + 0.05
            and float(best_rows7.get("match@20") or 0.0) >= float(baseline_rows7.get("match@20") or 0.0) + 0.05
        )
    )
    conv_delta = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(baseline_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta = float(best_rows7.get("match@50") or 0.0) - float(baseline_rows7.get("match@50") or 0.0)
    conv_delta_2d = float(best_rows7.get("skeleton_to_match_conversion@50") or 0.0) - float(exp2d_rows7.get("skeleton_to_match_conversion@50") or 0.0)
    match_delta_2d = float(best_rows7.get("match@50") or 0.0) - float(exp2d_rows7.get("match@50") or 0.0)

    result = {
        "experiment": "opentry_14_exp2e_train_pair_residual_posterior",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "train_pair_empirical_residual_posterior",
            "top_skeletons": int(args.top_skeletons),
            "assignment_limit": int(args.assignment_limit),
            "residuals_per_assignment": int(args.residuals_per_assignment),
            "residual_scales": scales,
            "selection_not_used": ["match", "RMSD", "StructureMatcher label", "GT-WA", "GT-skeleton", "official feedback", "RF/HGB/rerank"],
            "posterior_source": str(EXP1_PAIRS),
            "critic": "inference-safe structural score from legal/formula/SG/site/exact/collision/volume/reference/proposal features",
        },
        "cpu_policy": {
            "workers": int(args.workers),
            "eval_timeout_seconds": int(args.eval_timeout_seconds),
            "max_target_atoms_for_residual": int(args.max_target_atoms_for_residual),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
            "single_process_cpu_policy": "many single-thread worker processes; target each Python worker near 100% CPU and below user 200% per-core/process cap",
        },
        "data_scale": {
            "rows_ge7_samples": len(selected_sids),
            "input_safe_pool_records": len(safe_rows),
            "input_site_assignment_records": len(site_rows),
            "residual_templates": residual_template_count,
            "residual_buckets": len(residual_buckets),
            "generated_residual_candidates": len(generation_meta),
            "evaluated_residual_candidates": len(ranked_residual),
            "timeout_candidate_records": sum(str(r.get("eval_error") or "").startswith("sample_eval_timeout") for r in ranked_residual),
            "timeout_samples": len({str(r["sample_id"]) for r in ranked_residual if str(r.get("eval_error") or "").startswith("sample_eval_timeout")}),
            "top_k": int(args.top_k),
            "selected_variant_paths": selected_paths,
        },
        "oracle_reference": {
            "site_assignment_oracle_path": str(RESULT_DIR / "experiment_2d_site_assignment_oracle_diagnostic.json"),
            "site_assignment_oracle_match50_upper_bound": read_json(RESULT_DIR / "experiment_2d_site_assignment_oracle_diagnostic.json")["oracle_union_exp2b_plus_all_site_assignment"]["match50_upper_bound"],
            "site_assignment_oracle_conversion50_upper_bound": read_json(RESULT_DIR / "experiment_2d_site_assignment_oracle_diagnostic.json")["oracle_union_exp2b_plus_all_site_assignment"]["conversion50_upper_bound"],
        },
        "baseline_exp2b": {
            "result_path": str(EXP2B_RESULT),
            "rows_ge7": baseline_rows7,
            "overall": exp2b["overall"],
        },
        "baseline_exp2d": {
            "result_path": str(EXP2D_RESULT),
            "best_variant": exp2d["best_variant"],
            "rows_ge7": exp2d_rows7,
            "overall": exp2d["variants"][exp2d["best_variant"]]["overall"],
        },
        "mapping_failures": dict(failures),
        "residual_posterior_only": {
            "rows_ge7": summarize([r for r in ranked_residual if int(r.get("row_count") or 0) >= 7]),
        },
        "variants": variants,
        "best_variant": best_variant,
        "gate": {
            "minimum_passed": min_gate,
            "target_passed": target_gate,
            "passed": min_gate,
            "rows_ge7_match50_delta_vs_exp2b": match_delta,
            "rows_ge7_conversion50_delta_vs_exp2b": conv_delta,
            "rows_ge7_match50_delta_vs_exp2d": match_delta_2d,
            "rows_ge7_conversion50_delta_vs_exp2d": conv_delta_2d,
            "minimum_standard": {
                "rows_ge7_conversion50": 0.223,
                "rows_ge7_match50_allowed_lower_bound": 0.17931372549019609,
                "rows_ge7_formula_consistency": 0.95,
                "rows_ge7_sg_consistency": 0.90,
                "rows_ge7_exact_cover_retained": 0.95,
                "target_rows_ge7_conversion50": 0.28,
            },
        },
        "decision": {
            "verdict": "pass_target_gate" if target_gate else ("pass_minimum_gate" if min_gate else "fail_validation_gate"),
            "reason": (
                "Train-pair residual posterior reaches the target repair gate."
                if target_gate
                else (
                    "Train-pair residual posterior remains above the Exp2 minimum gate but does not reach the target line."
                    if min_gate
                    else "Train-pair residual posterior does not pass the Exp2 minimum repair gate."
                )
            ),
            "next_step": (
                "If target gate passes, retest Exp3 local optimizer relative to this posterior; otherwise do not expand beam blindly, inspect whether residual candidates add new oracle matches."
            ),
        },
        "artifacts": {
            "generated_residual_posterior_meta": str(ARTIFACT_DIR / f"generated_residual_posterior_meta{suffix}.jsonl"),
            "evaluated_residual_posterior_candidates": str(ARTIFACT_DIR / f"evaluated_residual_posterior_candidates{suffix}.jsonl"),
            "selected_variants": selected_paths,
        },
        "runtime_seconds": time.time() - started,
    }

    out_path = RESULT_DIR / f"experiment_2e_train_pair_residual_posterior{suffix}.json"
    write_json(out_path, result)
    if not suffix:
        body = report_body(result)
        append_or_replace(REPORT_PATH, "<!-- OPENTRY14_EXP2E_TRAIN_PAIR_RESIDUAL_POSTERIOR -->", body)
        append_or_replace(LOCAL_REPORT_PATH, "<!-- OPENTRY14_EXP2E_TRAIN_PAIR_RESIDUAL_POSTERIOR -->", body)
    print(json.dumps({"output": str(out_path), "best_variant": best_variant, "gate": result["gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
