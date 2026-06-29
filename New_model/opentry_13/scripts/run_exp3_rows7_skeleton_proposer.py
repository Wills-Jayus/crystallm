#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


ROOT = Path("/data/users/xsw/autodlmini")
NEW_MODEL = ROOT / "model" / "New_model"
OUT_DIR = NEW_MODEL / "opentry_13"
RESULT_DIR = OUT_DIR / "results"
ARTIFACT_DIR = OUT_DIR / "artifacts" / "exp3_rows7_skeleton_proposer"
REPORT_PATH = NEW_MODEL / "GPT_REVIEW_BUNDLE.md"

TRAIN_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "train.jsonl"
VAL_REPR = NEW_MODEL / "opentry_3" / "data" / "wyckoff_repr_mpts52" / "val.jsonl"
SYMCIF_GEN = ROOT / "runs" / "symcif_v5_multidataset_wa_decoder" / "mpts52" / "val" / "generations" / "v5_fullgen_eval_pool.jsonl"
SYMCIF_MET = ROOT / "runs" / "symcif_v5_multidataset_wa_decoder" / "mpts52" / "val" / "metrics" / "v5_fullgen_eval_pool_metrics.jsonl"
OP12_EXP6 = NEW_MODEL / "opentry_12" / "results" / "experiment_6_rows_ge7_specialized_audit.json"

BUDGETS = (1, 5, 20, 50)


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


def append_report_once(marker: str, body: str) -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")
    if marker in text:
        return
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(marker)
        f.write("\n")
        f.write(body.rstrip())
        f.write("\n")


def sample_id(record: dict[str, Any]) -> str:
    return str(record["keys"]["sample_id"])


def formula_counts(record: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in record["formula_counts"].items()}


def formula_frac(record: dict[str, Any]) -> dict[str, float]:
    counts = formula_counts(record)
    total = max(1, sum(counts.values()))
    return {k: float(v) / total for k, v in counts.items()}


def formula_l1(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def formula_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return dict(sorted(a.items())) == dict(sorted(b.items()))


def multiplicities_from_key_or_record(record: dict[str, Any]) -> tuple[int, ...]:
    if "multiplicities" in record:
        return tuple(int(x) for x in record["multiplicities"])
    return tuple(int(row["multiplicity"]) for row in record["skeleton_sequence"])


@lru_cache(maxsize=1_000_000)
def exact_cover_feasible_cached(multiplicities: tuple[int, ...], counts: tuple[tuple[str, int], ...]) -> bool:
    targets = tuple(sorted((int(v) for _, v in counts), reverse=True))
    if sum(multiplicities) != sum(targets):
        return False
    mults = tuple(sorted((int(x) for x in multiplicities if int(x) > 0), reverse=True))
    if not mults:
        return all(v == 0 for v in targets)

    @lru_cache(maxsize=None)
    def rec(i: int, remaining: tuple[int, ...]) -> bool:
        if i >= len(mults):
            return all(v == 0 for v in remaining)
        m = mults[i]
        tried: set[int] = set()
        for idx, value in enumerate(remaining):
            if value < m or value in tried:
                continue
            tried.add(value)
            new_remaining = list(remaining)
            new_remaining[idx] -= m
            new_tuple = tuple(sorted(new_remaining, reverse=True))
            if rec(i + 1, new_tuple):
                return True
        return False

    return rec(0, targets)


def exact_cover_feasible(multiplicities: tuple[int, ...], counts: dict[str, int]) -> bool:
    return exact_cover_feasible_cached(tuple(sorted(multiplicities)), tuple(sorted(counts.items())))


def build_train_index(train_rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    index: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for order, record in enumerate(train_rows):
        record["_train_order"] = order
        index[(int(record["sg"]), int(record["atom_count"]))].append(record)
    return dict(index)


def source_label(source: dict[str, Any], exact_formula: bool) -> str:
    if int(source.get("row_count") or 0) >= 7 and exact_formula:
        return "train_rows_ge7_exact_formula"
    if int(source.get("row_count") or 0) >= 7:
        return "train_rows_ge7_exact_cover_formula_nearest"
    if exact_formula:
        return "train_all_exact_formula_fallback"
    return "train_all_exact_cover_fallback"


def propose_for_target(
    target: dict[str, Any],
    train_index: dict[tuple[int, int], list[dict[str, Any]]],
    *,
    top_n: int = 50,
) -> list[dict[str, Any]]:
    counts = formula_counts(target)
    target_frac = formula_frac(target)
    pool = train_index.get((int(target["sg"]), int(target["atom_count"])), [])
    scored: list[tuple[float, int, dict[str, Any], bool]] = []
    for source in pool:
        mults = multiplicities_from_key_or_record(source)
        if not exact_cover_feasible(mults, counts):
            continue
        exact_formula = formula_equal(counts, formula_counts(source))
        source_rows7 = int(source.get("row_count") or 0) >= 7
        # Proposal score uses only train source metadata plus target composition/GT-SG.
        # It does not use validation skeleton, WA, StructureMatcher labels, RMSD, or match flags.
        score = 0.0
        score += 3.0 if source_rows7 else 0.0
        score += 2.0 if exact_formula else 0.0
        score += 0.25 if int(source.get("num_elements") or 0) == int(target.get("num_elements") or 0) else 0.0
        score += 0.15 if set(formula_counts(source)) == set(counts) else 0.0
        score -= formula_l1(target_frac, formula_frac(source))
        score -= 1e-6 * int(source.get("_train_order") or 0)
        scored.append((score, int(source.get("_train_order") or 0), source, exact_formula))
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, (score, _order, source, exact_formula) in enumerate(scored, start=1):
        key = str(source["canonical_skeleton_key"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "rank": len(out) + 1,
                "skeleton_key": key,
                "short_skeleton_key": str(source.get("short_skeleton_key") or ""),
                "multiplicities": list(multiplicities_from_key_or_record(source)),
                "score": float(score),
                "source": source_label(source, exact_formula),
                "source_sample_id": sample_id(source),
                "source_row_count": int(source.get("row_count") or 0),
                "source_atom_count": int(source.get("atom_count") or 0),
                "source_formula": str(source.get("formula") or ""),
                "exact_cover_feasible": True,
                "train_rank_before_dedup": rank,
            }
        )
        if len(out) >= int(top_n):
            break
    return out


def material_id_from_sample_id(sid: str) -> str:
    return str(sid).split("__")[-1]


def load_hydrated_candidates() -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with SYMCIF_GEN.open("r", encoding="utf-8") as gf, SYMCIF_MET.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            if not gen_line.strip() or not met_line.strip():
                continue
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            sid = str(gen["sample_id"])
            groups[sid].append(
                {
                    "sample_id": sid,
                    "material_id": material_id_from_sample_id(sid),
                    "gen_index": int(gen.get("gen_index") or 0),
                    "generation_score": float(gen.get("generation_score") if gen.get("generation_score") is not None else -1e30),
                    "skeleton_key": str(gen.get("canonical_skeleton_key") or ""),
                    "target_skeleton_key": str(gen.get("target_canonical_skeleton_key") or ""),
                    "skeleton_hit": bool(gen.get("skeleton_hit")),
                    "wa_hit": bool(gen.get("wa_hit")),
                    "row_count_hit": bool(gen.get("row_count_hit")),
                    "source_experiment": str(gen.get("source_experiment") or ""),
                    "match": bool(met.get("match_ok")),
                    "rms": met.get("rms"),
                    "valid": bool(met.get("valid")),
                    "formula_ok": bool(met.get("formula_ok")),
                    "sg_ok": bool(met.get("space_group_ok")),
                    "exact_cover_feasible": bool(met.get("multiplicity_ok")) if met.get("multiplicity_ok") is not None else None,
                    "bond_length_score": met.get("bond_length_score"),
                }
            )
    for rows in groups.values():
        rows.sort(key=lambda r: (float(r["generation_score"]), -int(r["gen_index"])), reverse=True)
    return dict(groups)


def hydrate_by_proposal(proposals: list[dict[str, Any]], hydrated_rows: list[dict[str, Any]], max_candidates: int = 50) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hydrated_rows:
        by_key[str(row["skeleton_key"])].append(row)
    out: list[dict[str, Any]] = []
    for proposal in proposals[:50]:
        key = str(proposal["skeleton_key"])
        for row in by_key.get(key, []):
            item = dict(row)
            item["proposal_rank"] = int(proposal["rank"])
            item["proposal_source"] = str(proposal["source"])
            out.append(item)
            if len(out) >= int(max_candidates):
                return out
    return out


def ratio(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_subset(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "samples": len(sample_rows),
        "rows_ge7_samples": sum(int(r["row_count"]) >= 7 for r in sample_rows),
        "candidate_nonempty_rate": ratio(sum(int(r["proposal_count"] > 0) for r in sample_rows), len(sample_rows)),
        "proposal_count_mean": mean([float(r["proposal_count"]) for r in sample_rows]),
        "hydrated_candidate_count_mean": mean([float(r["hydrated_candidate_count"]) for r in sample_rows]),
        "hydrated_samples": sum(int(r["hydrated_candidate_count"] > 0) for r in sample_rows),
    }
    for k in BUDGETS:
        skel_hits = sum(int(r[f"skeleton_hit@{k}"]) for r in sample_rows)
        exact_any = sum(int(r[f"exact_cover_any@{k}"]) for r in sample_rows)
        hydrated_hits = sum(int(r[f"hydrated_match@{k}"]) for r in sample_rows)
        hydrated_skel_hits = sum(int(r[f"hydrated_target_skeleton@{k}"]) for r in sample_rows)
        valid_any = sum(int(r[f"hydrated_valid_any@{k}"]) for r in sample_rows)
        formula_any = sum(int(r[f"hydrated_formula_ok_any@{k}"]) for r in sample_rows)
        sg_any = sum(int(r[f"hydrated_sg_ok_any@{k}"]) for r in sample_rows)
        exact_hydrated_any = sum(int(r[f"hydrated_exact_cover_any@{k}"]) for r in sample_rows)
        out[f"top{k}_skeleton_hit_coverage"] = ratio(skel_hits, len(sample_rows))
        out[f"top{k}_exact_cover_feasible_any"] = ratio(exact_any, len(sample_rows))
        out[f"top{k}_hydrated_match_coverage"] = ratio(hydrated_hits, len(sample_rows))
        out[f"top{k}_hydrated_target_skeleton_coverage"] = ratio(hydrated_skel_hits, len(sample_rows))
        out[f"top{k}_proposal_skeleton_to_hydrated_match_conversion"] = ratio(hydrated_hits, skel_hits)
        out[f"top{k}_hydrated_skeleton_to_match_conversion"] = ratio(hydrated_hits, hydrated_skel_hits)
        out[f"top{k}_hydrated_valid_any"] = ratio(valid_any, len(sample_rows))
        out[f"top{k}_hydrated_formula_ok_any"] = ratio(formula_any, len(sample_rows))
        out[f"top{k}_hydrated_sg_ok_any"] = ratio(sg_any, len(sample_rows))
        out[f"top{k}_hydrated_exact_cover_any"] = ratio(exact_hydrated_any, len(sample_rows))
        rms_vals = [float(r[f"hydrated_rms@{k}"]) for r in sample_rows if r.get(f"hydrated_rms@{k}") is not None]
        out[f"top{k}_hydrated_RMSE"] = mean(rms_vals)
    return out


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def triplet(d: dict[str, Any], template: str) -> str:
    return " / ".join(pct(d.get(template.format(k=k))) for k in (1, 5, 20))


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(TRAIN_REPR)
    val_rows = read_jsonl(VAL_REPR)
    train_index = build_train_index(train_rows)
    hydrated_groups = load_hydrated_candidates()

    proposal_records: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    proposal_candidate_total = 0
    exact_cover_candidate_total = 0

    for record in val_rows:
        sid = sample_id(record)
        proposals = propose_for_target(record, train_index, top_n=50)
        proposal_records.append({"sample_id": sid, "material_id": record["keys"].get("material_id"), "proposals": proposals})
        source_counter.update(str(p["source"]) for p in proposals)
        proposal_candidate_total += len(proposals)
        exact_cover_candidate_total += sum(int(bool(p.get("exact_cover_feasible"))) for p in proposals)

        target_key = str(record["canonical_skeleton_key"])
        proposal_keys = [str(p["skeleton_key"]) for p in proposals]
        hydrated = hydrate_by_proposal(proposals, hydrated_groups.get(sid, []), max_candidates=50)

        row: dict[str, Any] = {
            "sample_id": sid,
            "material_id": record["keys"].get("material_id"),
            "sg": int(record["sg"]),
            "atom_count": int(record["atom_count"]),
            "row_count": int(record["row_count"]),
            "num_elements": int(record["num_elements"]),
            "proposal_count": len(proposals),
            "hydrated_candidate_count": len(hydrated),
            "target_skeleton_key": target_key,
        }
        for k in BUDGETS:
            top_keys = proposal_keys[:k]
            top_hydrated = hydrated[:k]
            row[f"skeleton_hit@{k}"] = target_key in top_keys
            row[f"exact_cover_any@{k}"] = any(bool(p.get("exact_cover_feasible")) for p in proposals[:k])
            row[f"hydrated_match@{k}"] = any(bool(c.get("match")) for c in top_hydrated)
            row[f"hydrated_target_skeleton@{k}"] = any(str(c.get("skeleton_key")) == target_key for c in top_hydrated)
            row[f"hydrated_valid_any@{k}"] = any(bool(c.get("valid")) for c in top_hydrated)
            row[f"hydrated_formula_ok_any@{k}"] = any(bool(c.get("formula_ok")) for c in top_hydrated)
            row[f"hydrated_sg_ok_any@{k}"] = any(bool(c.get("sg_ok")) for c in top_hydrated)
            row[f"hydrated_exact_cover_any@{k}"] = any(c.get("exact_cover_feasible") is True for c in top_hydrated)
            matched_rms = [float(c["rms"]) for c in top_hydrated if bool(c.get("match")) and c.get("rms") is not None]
            row[f"hydrated_rms@{k}"] = min(matched_rms) if matched_rms else None
        per_sample.append(row)

    rows7 = [r for r in per_sample if int(r["row_count"]) >= 7]
    overall = summarize_subset(per_sample)
    rows_ge7 = summarize_subset(rows7)

    baseline_reference: dict[str, Any] = {}
    if OP12_EXP6.exists():
        exp6 = json.loads(OP12_EXP6.read_text(encoding="utf-8"))
        baseline_reference = {
            "crystallm_k50_rows_ge7": exp6.get("crystallm_k50_rows_ge7"),
            "symcif_v5_rows_ge7": exp6.get("experiment5_rows_ge7_reference", {}).get("full_missing_as_fail"),
            "note": "validation baseline reference from opentry_12 exp6; not used for proposal generation",
        }

    result = {
        "experiment": "opentry_13_exp3_rows_ge7_train_derived_skeleton_proposer_validation_gate",
        "time": now_iso(),
        "dataset": "mpts_52",
        "split": "val",
        "method": {
            "name": "rows_ge7_train_exact_cover_skeleton_proposer",
            "inference_inputs": ["composition/formula", "GT-SG", "train-derived skeleton prototypes"],
            "not_used_at_inference": [
                "validation/test match labels",
                "RMSD",
                "StructureMatcher result",
                "GT-WA",
                "GT-skeleton",
                "candidate fusion",
                "hard-negative scorer",
                "RF/HGB rerank",
            ],
            "proposal_logic": (
                "Index train records by GT-SG and atom_count; prioritize train rows>=7 exact-cover skeletons, "
                "then exact-formula and formula-nearest exact-cover prototypes; keep unique top50 skeletons."
            ),
            "hydrated_match_logic": (
                "Map proposed skeletons to existing SymCIF v5 validation candidates that already have StructureMatcher metrics; "
                "unhydrated proposed skeletons are not counted as matches."
            ),
        },
        "data_scale": {
            "train_repr_records": len(train_rows),
            "train_rows_ge7_records": sum(int(r.get("row_count") or 0) >= 7 for r in train_rows),
            "validation_repr_records": len(val_rows),
            "validation_rows_ge7_records": len(rows7),
            "hydrated_eval_samples": len(hydrated_groups),
            "hydrated_eval_candidates": sum(len(v) for v in hydrated_groups.values()),
            "proposal_candidates": proposal_candidate_total,
        },
        "candidate_level": {
            "exact_cover_feasible_rate": ratio(exact_cover_candidate_total, proposal_candidate_total),
            "source_counts": dict(source_counter),
        },
        "overall": overall,
        "rows_ge7": rows_ge7,
        "baseline_reference": baseline_reference,
        "decision": {
            "validation_gate_pass": False,
            "reason": (
                "Although the proposer supplies exact-cover skeletons, hydrated rows>=7 match@20/top50 remains below "
                "the required +5pp route and skeleton-to-match conversion is low; this is a proposer diagnostic, not an official claim."
            ),
            "next_step": "bind proposed exact-cover skeletons to learned geometry repair instead of ordinary scorer/rerank",
        },
    }

    write_jsonl(ARTIFACT_DIR / "proposals.jsonl", proposal_records)
    write_jsonl(ARTIFACT_DIR / "per_sample_metrics.jsonl", per_sample)
    write_json(RESULT_DIR / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json", result)

    section = f"""## opentry_13 实验 3：rows>=7-specialized skeleton proposer validation gate

结果文件：`model/New_model/opentry_13/results/experiment_3_rows_ge7_skeleton_proposer_validation_gate.json`  
候选文件：`model/New_model/opentry_13/artifacts/exp3_rows7_skeleton_proposer/proposals.jsonl`

- 为什么做：停止把普通 rerank/scorer 当主方法，直接检查 train-derived rows>=7 skeleton proposer 是否让候选池中出现更多可 match 的 skeleton/结构。
- 核心假设：若 rows>=7 的主瓶颈是 skeleton coverage，则只用 composition + GT-SG + train rows>=7 prototypes 的 exact-cover proposer 应提高 top50 skeleton-hit；若 skeleton-hit 不能转化为 match，则下一步必须接 learned geometry repair。
- 数据规模：train repr `{len(train_rows)}`，其中 rows>=7 `{sum(int(r.get('row_count') or 0) >= 7 for r in train_rows)}`；validation repr `{len(val_rows)}`，其中 rows>=7 `{len(rows7)}`；已有 hydrated eval candidates `{sum(len(v) for v in hydrated_groups.values())}`。未重新跑 StructureMatcher。
- baseline：opentry_12 exp6 validation rows>=7 CrystaLLM K50 top5/top20/top50 = `{pct(baseline_reference.get('crystallm_k50_rows_ge7', {}).get('top5_match'))}` / `{pct(baseline_reference.get('crystallm_k50_rows_ge7', {}).get('top20_match'))}` / `{pct(baseline_reference.get('crystallm_k50_rows_ge7', {}).get('top50_match'))}`；SymCIF v5 rows>=7 top5/top20/top50 = `{pct(baseline_reference.get('symcif_v5_rows_ge7', {}).get('top5_match_coverage'))}` / `{pct(baseline_reference.get('symcif_v5_rows_ge7', {}).get('top20_match_coverage'))}` / `{pct(baseline_reference.get('symcif_v5_rows_ge7', {}).get('top50_match_coverage'))}`。
- 方法变化：新增 rows>=7 train exact-cover skeleton proposer。推理期只使用 formula/composition、GT-SG 和 train skeleton prototypes；不使用 GT-WA、GT-skeleton、match/RMSD/StructureMatcher label、RF/HGB/rerank 或 threshold tuning。
- 结果 overall：top1/5/20 skeleton-hit = `{triplet(overall, 'top{k}_skeleton_hit_coverage')}`；hydrated match@1/5/20 = `{triplet(overall, 'top{k}_hydrated_match_coverage')}`；top50 hydrated match = `{pct(overall.get('top50_hydrated_match_coverage'))}`。
- 结果 rows>=7：top1/5/20 skeleton-hit = `{triplet(rows_ge7, 'top{k}_skeleton_hit_coverage')}`；top50 skeleton-hit = `{pct(rows_ge7.get('top50_skeleton_hit_coverage'))}`；top1/5/20 hydrated match = `{triplet(rows_ge7, 'top{k}_hydrated_match_coverage')}`；top50 hydrated match = `{pct(rows_ge7.get('top50_hydrated_match_coverage'))}`；top50 exact-cover feasible any = `{pct(rows_ge7.get('top50_exact_cover_feasible_any'))}`；top50 proposal-skeleton-to-hydrated-match conversion = `{pct(rows_ge7.get('top50_proposal_skeleton_to_hydrated_match_conversion'))}`；top50 hydrated-skeleton-to-match conversion = `{pct(rows_ge7.get('top50_hydrated_skeleton_to_match_conversion'))}`。
- 可信度：中等。proposal-only 指标覆盖 validation repr 全部 `{len(val_rows)}` 样本；match 指标只在已有 SymCIF v5 hydrated/evaluated candidate 上计算，未把未渲染 skeleton 自动算成 match，因此偏保守。它是 validation gate，不是 official 结果。
- 和历史实验关系：区别于 opentry_12 exp6 的 scorer/边界审计，本实验真正输出 train-derived top50 skeleton proposals；但 hydrated match 仍受既有 geometry 生成质量限制。
- 最终判决：validation gate 未通过。rows>=7 skeleton-hit 有信号，但 hydrated match@5/20 和 skeleton-to-match conversion 不足，不能进入 official，也不能作为主结果 claim。
- 下一步：将这些 predicted exact-cover skeleton 绑定 learned geometry repair，重点修 lattice/free parameters/site mapping/collision/local geometry，而不是再做普通 scorer。
"""
    append_report_once("<!-- OPENTRY13_EXP3_ROWS7_SKELETON_PROPOSER -->", section)
    print(RESULT_DIR / "experiment_3_rows_ge7_skeleton_proposer_validation_gate.json")


if __name__ == "__main__":
    main()
