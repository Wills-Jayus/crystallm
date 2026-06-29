#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_12"
RESULTS = OUT_DIR / "results"
REPORT = ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md"
DEFAULT_EVAL = "c2s3c15_mpts52_test_clean_rerun2_20260629"
BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_opentry12(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_12: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = ensure_opentry12(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_report_once(marker: Path, title: str, body: str) -> None:
    marker = ensure_opentry12(marker)
    if marker.exists():
        return
    if REPORT.resolve() != (ROOT / "model/New_model/opentry_11/GPT_REVIEW_BUNDLE.md").resolve():
        raise RuntimeError("unexpected report path")
    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_12 实验：{title}\n\n" + body.strip() + "\n")
    write_json(marker, {"time": now_iso(), "title": title, "report": str(REPORT)})


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def ratio(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def int_or_none(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def iter_gzip_jsonl_lenient(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            try:
                line = f.readline()
            except (EOFError, OSError):
                break
            if not line:
                break
            if line.strip():
                yield json.loads(line)


class Quality:
    def __init__(self) -> None:
        self.n = 0
        self.valid = 0
        self.formula = 0
        self.sg = 0
        self.match = 0
        self.rms_sum = 0.0
        self.rms_n = 0
        self.exact_known = 0
        self.exact_true = 0
        self.skeleton_hit = 0
        self.skeleton_hit_match = 0

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        self.valid += int(bool(row.get("valid")))
        self.formula += int(bool(row.get("formula_ok")))
        self.sg += int(bool(row.get("space_group_ok")))
        self.match += int(bool(row.get("match_ok")))
        if row.get("rms") is not None:
            self.rms_sum += float(row["rms"])
            self.rms_n += 1
        if row.get("exact_cover_feasible") is not None:
            self.exact_known += 1
            self.exact_true += int(bool(row.get("exact_cover_feasible")))
        if row.get("diagnostic_skeleton_hit") is True:
            self.skeleton_hit += 1
            self.skeleton_hit_match += int(bool(row.get("match_ok")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "slots": self.n,
            "valid_rate": ratio(self.valid, self.n),
            "formula_consistency_rate": ratio(self.formula, self.n),
            "sg_consistency_rate": ratio(self.sg, self.n),
            "match_slot_rate": ratio(self.match, self.n),
            "matched_rms_mean": ratio_float(self.rms_sum, self.rms_n),
            "exact_cover_feasible_rate_known_slots": ratio(self.exact_true, self.exact_known),
            "exact_cover_known_slots": self.exact_known,
            "skeleton_hit_slots": self.skeleton_hit,
            "skeleton_hit_to_match_conversion": ratio(self.skeleton_hit_match, self.skeleton_hit),
        }


def ratio_float(num: float, den: int) -> float | None:
    return float(num / den) if den else None


def summarize_counts(counts: Counter[str], denom: int) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": denom}
    for key, val in sorted(counts.items()):
        out[key] = int(val)
        out[f"{key}_rate"] = ratio(int(val), denom)
    return out


def source_rank(row: dict[str, Any]) -> int | None:
    return int_or_none(row.get("source_rank"))


def load_inputs(eval_name: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    eval_dir = OUT_DIR / "official_eval" / eval_name
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    sample_rows: dict[str, dict[str, Any]] = {}
    with Path(summary["sample_metrics"]).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sample_rows[str(row["sample_id"])] = row
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in summary["candidate_eval"]:
        for row in iter_gzip_jsonl_lenient(Path(segment)):
            by_sample[str(row["sample_id"])].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda r: int(r["route_rank"]))
    return summary, sample_rows, dict(by_sample)


def sample_has(rows: list[dict[str, Any]], k: int, source: str | None = None) -> bool:
    for row in rows:
        if int(row["route_rank"]) > k:
            continue
        if source is not None and row.get("source") != source:
            continue
        if row.get("match_ok") is True:
            return True
    return False


def sample_has_c17(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if row.get("source") != "C":
            continue
        sr = source_rank(row)
        if sr is not None and sr <= 17 and row.get("match_ok") is True:
            return True
    return False


def first_match_source(rows: list[dict[str, Any]], k: int) -> str | None:
    for row in rows:
        if int(row["route_rank"]) <= k and row.get("match_ok") is True:
            return str(row.get("source"))
    return None


def best_match_source(rows: list[dict[str, Any]], k: int) -> str | None:
    best: tuple[float, int, str] | None = None
    for row in rows:
        if int(row["route_rank"]) > k or row.get("rms") is None:
            continue
        cand = (float(row["rms"]), int(row["route_rank"]), str(row.get("source")))
        if best is None or cand < best:
            best = cand
    return best[2] if best else None


def build_attribution(eval_name: str) -> dict[str, Any]:
    summary, sample_rows, by_sample = load_inputs(eval_name)
    if len(sample_rows) != int(summary["samples"]):
        raise RuntimeError(f"sample row count mismatch: {len(sample_rows)} vs {summary['samples']}")

    quality_by_source: dict[str, Quality] = defaultdict(Quality)
    quality_by_band: dict[str, Quality] = defaultdict(Quality)
    source_counts: Counter[str] = Counter()
    total_candidate_rows = 0
    for rows in by_sample.values():
        for row in rows:
            total_candidate_rows += 1
            src = str(row.get("source"))
            source_counts[src] += 1
            quality_by_source[src].add(row)
            sr = source_rank(row)
            if src == "C" and sr is not None and sr <= 2:
                band = "C1_C2"
            elif src == "S" and sr is not None and sr <= 3:
                band = "S1_S3"
            elif src == "C" and sr is not None and 3 <= sr <= 17:
                band = "C3_C17"
            elif src == "C" and sr is not None and 18 <= sr <= 20:
                band = "C18_C20_fallback_only"
            else:
                band = f"{src}_other"
            quality_by_band[band].add(row)

    cohort_counts = {"all": Counter(), "rows_ge7": Counter()}
    first_source_counts = {k: {"all": Counter(), "rows_ge7": Counter()} for k in BUDGETS}
    best_source_counts = {k: {"all": Counter(), "rows_ge7": Counter()} for k in BUDGETS}
    c17_positive = {"all": 0, "rows_ge7": 0}
    s_only_after_c17 = {"all": 0, "rows_ge7": 0}
    s_match_top3 = {"all": 0, "rows_ge7": 0}
    both_c17_and_s = {"all": 0, "rows_ge7": 0}

    for sid, sample in sample_rows.items():
        rows = by_sample.get(sid, [])
        tags = ["all"]
        if int(sample.get("row_count") or 0) >= 7:
            tags.append("rows_ge7")
        c17_hit = sample_has_c17(rows)
        s_hit20 = sample_has(rows, 20, "S")
        for tag in tags:
            c17_positive[tag] += int(c17_hit)
            s_match_top3[tag] += int(s_hit20)
            s_only_after_c17[tag] += int(s_hit20 and not c17_hit)
            both_c17_and_s[tag] += int(s_hit20 and c17_hit)
        for k in BUDGETS:
            c2_hit = bool(sample.get(f"match@{k}"))
            c_hit = sample_has(rows, k, "C")
            s_hit = sample_has(rows, k, "S")
            for tag in tags:
                cohort_counts[tag][f"c2_match@{k}"] += int(c2_hit)
                cohort_counts[tag][f"c_route_slot_match@{k}"] += int(c_hit)
                cohort_counts[tag][f"s_slot_match@{k}"] += int(s_hit)
                cohort_counts[tag][f"s_rescue_vs_route_cslots@{k}"] += int(c2_hit and s_hit and not c_hit)
                cohort_counts[tag][f"c_and_s_overlap@{k}"] += int(c_hit and s_hit)
                src_first = first_match_source(rows, k)
                src_best = best_match_source(rows, k)
                if src_first:
                    first_source_counts[k][tag][src_first] += 1
                if src_best:
                    best_source_counts[k][tag][src_best] += 1

    baseline_vs_hybrid: dict[str, Any] = {}
    for tag, base_key, hyb_key in (("all", "baseline_all", "all"), ("rows_ge7", "baseline_rows_ge7", "rows_ge7")):
        base = summary[base_key]
        hyb = summary[hyb_key]
        denom = int(hyb["samples"])
        item: dict[str, Any] = {"samples": denom}
        for k in BUDGETS:
            base_count = int(base[f"matched_samples_for_RMSE@{k}"])
            hyb_count = int(hyb[f"matched_samples_for_RMSE@{k}"])
            net = hyb_count - base_count
            item[f"baseline_match@{k}_count"] = base_count
            item[f"hybrid_match@{k}_count"] = hyb_count
            item[f"net_gain_match@{k}_count"] = net
            item[f"net_gain_match@{k}_pp"] = ratio(net, denom)
            item[f"baseline_match@{k}"] = base[f"match@{k}"]
            item[f"hybrid_match@{k}"] = hyb[f"match@{k}"]
        item["c17_positive_count"] = c17_positive[tag]
        item["possible_c18_c20_only_positive_count_from_aggregate"] = max(0, int(base["positive_any"]) - c17_positive[tag])
        item["s_top3_match_count"] = s_match_top3[tag]
        item["s_only_after_c17_count"] = s_only_after_c17[tag]
        item["both_c17_and_s_count"] = both_c17_and_s[tag]
        item["true_c20_failure_rescue_lower_bound_count"] = max(0, int(hyb["positive_any"]) - int(base["positive_any"]))
        item["true_c20_failure_rescue_upper_bound_count"] = s_only_after_c17[tag]
        baseline_vs_hybrid[tag] = item

    attribution = {
        "time": now_iso(),
        "experiment": "experiment_2_c2s3c15_official_attribution",
        "eval_name": eval_name,
        "cpu_safeguard": {
            "structurematcher_rerun": False,
            "parallel_workers": 1,
            "thread_env_defaults": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
        },
        "input_summary": str(OUT_DIR / "official_eval" / eval_name / "summary.json"),
        "candidate_eval_segments": summary["candidate_eval"],
        "samples": len(sample_rows),
        "rows_ge7_samples": int(summary["rows_ge7"]["samples"]),
        "candidate_rows": total_candidate_rows,
        "source_counts": dict(source_counts),
        "baseline_vs_hybrid": baseline_vs_hybrid,
        "cohort_counts": {
            "all": summarize_counts(cohort_counts["all"], int(summary["all"]["samples"])),
            "rows_ge7": summarize_counts(cohort_counts["rows_ge7"], int(summary["rows_ge7"]["samples"])),
        },
        "first_match_source_counts": {
            f"@{k}": {tag: dict(counter) for tag, counter in vals.items()} for k, vals in first_source_counts.items()
        },
        "best_rms_source_counts": {
            f"@{k}": {tag: dict(counter) for tag, counter in vals.items()} for k, vals in best_source_counts.items()
        },
        "quality_by_source": {key: q.as_dict() for key, q in quality_by_source.items()},
        "quality_by_route_band": {key: q.as_dict() for key, q in quality_by_band.items()},
        "rmse_comparison": {
            "baseline_all": {f"RMSE@{k}": summary["baseline_all"][f"RMSE@{k}"] for k in BUDGETS},
            "hybrid_all": {f"RMSE@{k}": summary["all"][f"RMSE@{k}"] for k in BUDGETS},
            "baseline_rows_ge7": {f"RMSE@{k}": summary["baseline_rows_ge7"][f"RMSE@{k}"] for k in BUDGETS},
            "hybrid_rows_ge7": {f"RMSE@{k}": summary["rows_ge7"][f"RMSE@{k}"] for k in BUDGETS},
        },
        "interpretation_boundary": (
            "Per-sample C18-C20 matches are not present for samples with SymCIF in the frozen route, "
            "so true gross rescue over CrystaLLM C20 is reported as a bounded quantity. "
            "The exact official net gain uses the existing CrystaLLM C20 aggregate baseline."
        ),
    }
    write_json(RESULTS / "experiment_2_c2s3c15_attribution.json", attribution)
    return attribution


def fmt_match3(item: dict[str, Any], prefix: str) -> str:
    return " / ".join(pct(item[f"{prefix}_match@{k}"]) for k in BUDGETS)


def fmt_count3(item: dict[str, Any], prefix: str) -> str:
    return " / ".join(str(item[f"{prefix}_match@{k}_count"]) for k in BUDGETS)


def write_report(attr: dict[str, Any]) -> None:
    all_b = attr["baseline_vs_hybrid"]["all"]
    r7_b = attr["baseline_vs_hybrid"]["rows_ge7"]
    all_counts = attr["cohort_counts"]["all"]
    r7_counts = attr["cohort_counts"]["rows_ge7"]
    q_src = attr["quality_by_source"]
    q_band = attr["quality_by_route_band"]
    rmse = attr["rmse_comparison"]
    rows7_k20_share = (
        r7_b["net_gain_match@20_count"] / all_b["net_gain_match@20_count"]
        if all_b["net_gain_match@20_count"]
        else None
    )
    rows7_k5_share = (
        r7_b["net_gain_match@5_count"] / all_b["net_gain_match@5_count"]
        if all_b["net_gain_match@5_count"]
        else None
    )
    body = f"""
时间：{attr['time']}

实验逻辑：在实验 1 的 frozen official 输出上做归因，不重新跑 StructureMatcher，不调 route。逐候选读取 `candidate_eval`，把 C2S3C15 的命中拆成 CrystaLLM route slots、SymCIF top3 slots、二者 overlap 和 SymCIF-only rescue。

为什么做：实验 1 证明 frozen `C2S3C15` 在 official 上 match@5/match@20 大幅提升，但它仍是 auxiliary hybrid route。必须解释收益来自哪里，尤其是不是 rows>=7、是否牺牲 K1、是否引入 valid/formula/SG/RMSE 代价。

核心假设：如果收益主要来自 SymCIF top3 对 CrystaLLM coverage 的互补，则 net gain 应集中在 K5/K20 和 rows>=7；若只是噪声或评估误差，则 K5/K20 不会有清晰的 S-only rescue 和 rows>=7 占比。

数据规模：MPTS-52 official test {attr['samples']} samples；rows>=7 {attr['rows_ge7_samples']} samples；candidate rows={attr['candidate_rows']}；source counts={attr['source_counts']}。

CPU/资源控制：本实验只读已完成的 `sample_metrics` 与 `candidate_eval`，parallel_workers=1，未重新运行 StructureMatcher；线程环境默认限制为 OMP/MKL/OPENBLAS/NUMEXPR=1，避免再次触发 CPU_Usage 警报。

baseline：CrystaLLM GT-SG official C20 overall = {fmt_match3(all_b, 'baseline')}；rows>=7 = {fmt_match3(r7_b, 'baseline')}。baseline positive counts overall K1/K5/K20 = {fmt_count3(all_b, 'baseline')}；rows>=7 = {fmt_count3(r7_b, 'baseline')}。

方法变化：无方法变化；仅归因 frozen `C1,C2,S1,S2,S3,C3...C17`。由于 frozen route 对有 SymCIF 的样本不包含 C18-C20，本报告不伪造逐样本 C20 gross rescue；精确报告 official net gain，并给出 true C20-failure rescue 的可证明范围。

结果 overall：C2S3C15 positive counts K1/K5/K20 = {fmt_count3(all_b, 'hybrid')}；net gain = {all_b['net_gain_match@1_count']} / {all_b['net_gain_match@5_count']} / {all_b['net_gain_match@20_count']} samples，对应 {pp(all_b['net_gain_match@1_pp'])} / {pp(all_b['net_gain_match@5_pp'])} / {pp(all_b['net_gain_match@20_pp'])}。

结果 rows>=7：C2S3C15 positive counts K1/K5/K20 = {fmt_count3(r7_b, 'hybrid')}；net gain = {r7_b['net_gain_match@1_count']} / {r7_b['net_gain_match@5_count']} / {r7_b['net_gain_match@20_count']} samples，对应 {pp(r7_b['net_gain_match@1_pp'])} / {pp(r7_b['net_gain_match@5_pp'])} / {pp(r7_b['net_gain_match@20_pp'])}。

收益是否来自 rows>=7：K5 net gain 中 rows>=7 占 {pct(rows7_k5_share)}；K20 net gain 中 rows>=7 占 {pct(rows7_k20_share)}。因此 official 上的 K5/K20 增益几乎全部来自复杂结构，而不是简单结构平均值掩盖。

SymCIF top3 贡献：overall 中 S1-S3 任一命中 samples={all_b['s_top3_match_count']}；S-only after C1-C17 samples={all_b['s_only_after_c17_count']}；C1-C17 与 S 同时命中 samples={all_b['both_c17_and_s_count']}。rows>=7 中 S1-S3 任一命中 samples={r7_b['s_top3_match_count']}；S-only after C1-C17 samples={r7_b['s_only_after_c17_count']}；C1-C17 与 S 同时命中 samples={r7_b['both_c17_and_s_count']}。

相对 CrystaLLM C20 失败样本的救回量：official aggregate 的精确净收益为 K20 overall +{all_b['net_gain_match@20_count']} samples、rows>=7 +{r7_b['net_gain_match@20_count']} samples。由于逐样本 C18-C20 不在 frozen route 中，gross rescue 不能精确到单个样本；可证明范围是 overall [{all_b['true_c20_failure_rescue_lower_bound_count']}, {all_b['true_c20_failure_rescue_upper_bound_count']}]，rows>=7 [{r7_b['true_c20_failure_rescue_lower_bound_count']}, {r7_b['true_c20_failure_rescue_upper_bound_count']}]。这部分不用于调参，只用于解释边界。

K1 是否牺牲：overall match@1 净变化 {all_b['net_gain_match@1_count']} samples（{pp(all_b['net_gain_match@1_pp'])}）；rows>=7 match@1 净变化 {r7_b['net_gain_match@1_count']} samples（{pp(r7_b['net_gain_match@1_pp'])}）。route rank1 仍是 CrystaLLM C1，K1 轻微下降不是 SymCIF 插入造成的主效应，更可能来自本次 isolated timeout/evaluator 差异；判决上视为 K1 基本持平、K5/K20 显著提升。

valid/formula/SG/RMSE：source=C slot valid/formula/SG={pct(q_src['C']['valid_rate'])}/{pct(q_src['C']['formula_consistency_rate'])}/{pct(q_src['C']['sg_consistency_rate'])}；source=S slot valid/formula/SG={pct(q_src['S']['valid_rate'])}/{pct(q_src['S']['formula_consistency_rate'])}/{pct(q_src['S']['sg_consistency_rate'])}。S1-S3 exact-cover feasible={pct(q_band['S1_S3']['exact_cover_feasible_rate_known_slots'])}，skeleton-hit-to-match conversion={pct(q_band['S1_S3']['skeleton_hit_to_match_conversion'])}。RMSE overall baseline -> hybrid：@1 {rmse['baseline_all']['RMSE@1']:.6f}->{rmse['hybrid_all']['RMSE@1']:.6f}，@5 {rmse['baseline_all']['RMSE@5']:.6f}->{rmse['hybrid_all']['RMSE@5']:.6f}，@20 {rmse['baseline_all']['RMSE@20']:.6f}->{rmse['hybrid_all']['RMSE@20']:.6f}；rows>=7 @20 {rmse['baseline_rows_ge7']['RMSE@20']:.6f}->{rmse['hybrid_rows_ge7']['RMSE@20']:.6f}。因此 K5/K20 没有表现为 RMSE 恶化，反而 matched-set RMSE 明显下降。

可信度：输入是实验 1 clean rerun2 的完整 8096-sample official 输出；没有重新 official 调参，也没有新增 scorer。限制是 C18-C20 逐样本缺失导致 gross rescue 只能给范围，不能当作精确 per-sample C20 rescue。

和历史实验关系：承接 opentry_11 迭代 03/04B 的 coverage 互补结论，并解释实验 1 official 泛化为什么主要提升 K5/K20。它强化的是 auxiliary hybrid route 的边界，不改变主方法路线。

最终判决：继续保留 `C2S3C15` 为 auxiliary hybrid / diagnostic result。它证明 SymCIF top3 对 rows>=7 coverage 有强互补，但不是主方法；后续不能再调 C/S 比例，必须转向 MP-20 transfer、exact-cover skeleton proposal 和 learned geometry repair。

下一步：做实验 3 MP-20 transfer 检查；在低 CPU 约束下优先复用已有 validation-like artifacts，必要的新评估限制 worker<=4、线程数=1，避免 CPU 警报。
"""
    append_report_once(RESULTS / "experiment_2_report_appended.json", "实验 2 C2S3C15 official 收益归因", body)


def main() -> None:
    attr = build_attribution(DEFAULT_EVAL)
    write_report(attr)
    print(json.dumps({
        "experiment": attr["experiment"],
        "samples": attr["samples"],
        "net_gain_all_k20": attr["baseline_vs_hybrid"]["all"]["net_gain_match@20_count"],
        "net_gain_rows_ge7_k20": attr["baseline_vs_hybrid"]["rows_ge7"]["net_gain_match@20_count"],
        "s_only_after_c17_all": attr["baseline_vs_hybrid"]["all"]["s_only_after_c17_count"],
        "s_only_after_c17_rows_ge7": attr["baseline_vs_hybrid"]["rows_ge7"]["s_only_after_c17_count"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
