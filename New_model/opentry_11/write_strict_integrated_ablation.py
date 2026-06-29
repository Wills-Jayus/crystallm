#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUT_DIR = Path("/data/users/xsw/autodlmini/model/New_model/opentry_11")
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_11: {resolved}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = ensure_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_report(title: str, body: str) -> None:
    with ensure_out(REPORT).open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_11 追加实验：{title}\n\n" + body.strip() + "\n")


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def fmt_metrics(metrics: dict[str, Any]) -> str:
    return " / ".join(pct(metrics[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(metrics: dict[str, Any]) -> str:
    return " / ".join(pct(metrics[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(delta: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(delta[f"{prefix}match@{b}"]) for b in BUDGETS)


def line(name: str, metrics: dict[str, Any], diag: dict[str, Any] | None = None) -> str:
    diag = diag or {}
    return (
        f"| {name} | {fmt_metrics(metrics)} | {fmt_rows7(metrics)} | "
        f"{pct(diag.get('valid_rate'))} | {pct(diag.get('formula_consistency'))} | "
        f"{pct(diag.get('sg_consistency'))} | {pct(diag.get('exact_cover_feasible_rate'))} | "
        f"{pct(diag.get('skeleton_hit_to_match_conversion'))} |"
    )


def main() -> None:
    full = load_json(RESULTS / "experiment_8_integrated_ablation.json")
    gpu = load_json(RESULTS / "experiment_4_6_gpu_scorer_half.json")
    repair = load_json(RESULTS / "experiment_5_half_geometry_repair.json")
    proposal = load_json(RESULTS / "experiment_7_half_exact_cover_generation.json")

    rows_by_name = {row["name"]: row for row in full["ablation_rows"]}
    base_full = rows_by_name["原 GT-SG baseline"]
    track_a_full = rows_by_name["baseline + Track A scorer"]
    exact_full = rows_by_name["baseline + exact-cover filter/proposal"]
    hn_full = rows_by_name["baseline + hard-negative structural scorer v2"]
    combo_full = rows_by_name["skeleton proposal + geometry repair + structural scorer"]

    gpu_base = gpu["policies"]["baseline"]
    gpu_general = gpu["policies"]["gpu_general"]
    gpu_rows7 = gpu["policies"]["gpu_rows7_route"]
    repair_base = repair["baseline"]
    repair_after = repair["repaired"]
    a1 = proposal["exact_cover_a1"]
    pool = proposal["fullgen_pool"]

    result = {
        "time": now_iso(),
        "scope": {
            "full_validation_table": "MPTS-52 validation K50 full, from prior full diagnostic run",
            "gpu_scorer_table": "MPTS-52 validation deterministic 50% subset, actual GPU OOF",
            "repair_table": "MPTS-52 validation deterministic 50% subset, full repair pool",
            "proposal_table": "MPTS-52 validation SymCIF v5 generation artifact deterministic half subset",
            "note": "Rows from different scopes are not directly pooled into one single metric.",
        },
        "full_validation": {
            "baseline": base_full,
            "track_a": track_a_full,
            "exact_cover_filter_proxy": exact_full,
            "cpu_hard_negative_v2": hn_full,
            "proxy_combined": combo_full,
        },
        "gpu_half": {
            "baseline": gpu_base,
            "gpu_general": gpu_general,
            "gpu_rows7": gpu_rows7,
            "deltas": gpu["deltas_vs_baseline"],
            "gate": gpu["gate"],
        },
        "repair_half": {
            "baseline": repair_base,
            "repaired": repair_after,
            "delta": repair["delta_vs_baseline"],
            "gate": repair["gate"],
            "repair_eval": repair["repair_eval"],
        },
        "proposal_half": {
            "baseline_a1": proposal["baseline"],
            "baseline_pool": proposal.get("pool_baseline"),
            "exact_cover_a1": a1,
            "fullgen_pool": pool,
            "deltas": proposal["deltas_vs_baseline"],
            "gate": proposal["gate"],
        },
        "final_judgment": {
            "two_match_metrics_over_5pp": False,
            "coverage_improved": False,
            "sorting_only_modules": ["Track A", "CPU hard-negative v2", "GPU hard-negative scorer"],
            "failed_conversion_modules": ["deterministic geometry repair", "SymCIF v5 exact-cover generation without effective geometry conversion"],
            "rows_ge7_best_signal": "generation artifacts improve rows>=7 K1/K5 but not K20 enough and hurt overall K20",
        },
    }
    write_json(RESULTS / "experiment_8_strict_integrated_ablation.json", result)

    base_m = base_full["metrics"]
    track_m = track_a_full["metrics"]
    exact_m = exact_full["metrics"]
    hn_m = hn_full["metrics"]
    combo_m = combo_full["metrics"]
    gpu_base_m = gpu_base["metrics"]
    gpu_gen_m = gpu_general["metrics"]
    gpu_row_m = gpu_rows7["metrics"]
    repair_base_m = repair_base["metrics"]
    repair_m = repair_after["metrics"]
    a1_m = a1["metrics"]
    pool_m = pool["metrics"]

    append_report(
        "实验 8C 严格整合消融与最终判决",
        f"""
时间：{result['time']}

实验逻辑：按最新口径重写整合消融。前一版实验 8 把 full validation、CPU scorer、repair pilot、proposal proxy 放在同一张表里，容易让人误以为所有模块都严格完成。本节把不同数据范围分开：full validation 诊断、GPU validation scorer、half validation repair、SymCIF generation artifact 复算分别报告，不把样本集合不同的结果硬合并。

GPU 必要性判断：实验 8 本身是整合分析，不训练模型，不需要 GPU；它只汇总前面模块。真正需要 GPU 的训练模块已经在实验 4C/6C 补做了 validation GPU scorer；train-data 级别 neural W/A decoder 或 learned geometry repair 尚未启动，因为现有 validation 证据未过 +5pp gate。

核心假设：如果主线有效，至少两个 match 指标应达到 +5pp，并且 rows>=7 不只是 K1/K5 局部上涨，K20 和 skeleton-hit-to-match conversion 也应改善。

数据规模：
- full validation 诊断：MPTS-52 validation K50，5000 samples / 250000 candidates。
- GPU scorer：MPTS-52 validation deterministic half，2500 samples / 125000 candidates。
- geometry repair：MPTS-52 validation deterministic half，repair pool 23738 candidates。
- generation proposal：SymCIF v5 MPTS-52 validation generation artifact half，A1 2364 samples / 11820 candidates，fullgen pool 2296 samples / 12976 candidates。

full validation 诊断表：

| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{line('原 GT-SG baseline', base_m, base_full['diagnostics'])}
{line('Track A scorer', track_m, track_a_full['diagnostics'])}
{line('exact-cover filter/proxy', exact_m, exact_full['diagnostics'])}
{line('CPU hard-negative v2', hn_m, hn_full['diagnostics'])}
{line('proxy combined', combo_m, combo_full['diagnostics'])}

GPU scorer half-data 表：

| 版本 | overall K1/K5/K20 | rows>=7 K1/K5/K20 | valid | formula | SG | exact-cover | skeleton-to-match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{line('half baseline', gpu_base_m, gpu_base['diagnostics'])}
{line('GPU hard-negative scorer', gpu_gen_m, gpu_general['diagnostics'])}
{line('GPU rows>=7 route', gpu_row_m, gpu_rows7['diagnostics'])}

GPU scorer delta：general = {fmt_delta(gpu['deltas_vs_baseline']['gpu_general'])}；rows>=7 delta = {fmt_delta(gpu['deltas_vs_baseline']['gpu_general'], 'rows>=7_')}。rows>=7 route overall delta = {fmt_delta(gpu['deltas_vs_baseline']['gpu_rows7_route'])}；rows>=7 delta = {fmt_delta(gpu['deltas_vs_baseline']['gpu_rows7_route'], 'rows>=7_')}。

geometry repair half-data：baseline = {fmt_metrics(repair_base_m)}；repair = {fmt_metrics(repair_m)}；delta = {fmt_delta(repair['delta_vs_baseline'])}。rows>=7 baseline = {fmt_rows7(repair_base_m)}；repair rows>=7 = {fmt_rows7(repair_m)}；rows>=7 delta = {fmt_delta(repair['delta_vs_baseline'], 'rows>=7_')}。repair conversion = {repair['repair_eval']['converted']}/{repair['repair_eval']['evaluated_valid']}，rows>=7 conversion = {repair['repair_eval']['rows>=7_converted']}/{repair['repair_eval']['rows>=7_evaluated_valid']}。

generation proposal half-data：A1 exact-cover = {fmt_metrics(a1_m)}，delta = {fmt_delta(proposal['deltas_vs_baseline']['exact_cover_a1'])}；rows>=7 = {fmt_rows7(a1_m)}，rows>=7 delta = {fmt_delta(proposal['deltas_vs_baseline']['exact_cover_a1'], 'rows>=7_')}。fullgen pool = {fmt_metrics(pool_m)}，delta = {fmt_delta(proposal['deltas_vs_baseline']['fullgen_pool'])}；rows>=7 = {fmt_rows7(pool_m)}，rows>=7 delta = {fmt_delta(proposal['deltas_vs_baseline']['fullgen_pool'], 'rows>=7_')}。

可信度：full validation 诊断可信但包含 proxy；GPU scorer 是实际 GPU OOF 但只排序已有候选；repair 是 half validation 全 repair pool 复核但不是 learned repair；proposal 是真实生成侧 artifact 复算但不是本轮重训。

和历史实验关系：结论与 opentry_10 和 SymCIF v5 历史一致：Track A/scorer 能局部排序，exact-cover 能改善 skeleton feasibility 或 rows>=7 K1/K5，但没有把 skeleton-hit 稳定转成 StructureMatcher match。

最终判决：没有达到“至少两个 match 指标 +5pp”。真正 coverage 提升证据不足；Track A、CPU/GPU scorer 主要是排序；deterministic repair conversion 为 0；exact-cover generation rows>=7 K1/K5 有信号但 overall K20 明显下降，rows>=7 K20 增益不足。

下一步：不要继续普通 rerank 或 weak repair。若要继续，必须定义 train-data 级别的 learned geometry repair 或 neural skeleton proposer，并先明确 train 数据规模、GPU 训练必要性、half/full gate 和 stop condition。
""",
    )
    print(json.dumps({"wrote": str(RESULTS / "experiment_8_strict_integrated_ablation.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
