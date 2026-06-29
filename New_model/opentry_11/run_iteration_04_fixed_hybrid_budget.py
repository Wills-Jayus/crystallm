#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/users/xsw/autodlmini")
OUT_DIR = ROOT / "model/New_model/opentry_11"
RESULTS = OUT_DIR / "results"
REPORT = OUT_DIR / "GPT_REVIEW_BUNDLE.md"
STRUCT_FEATURES = ROOT / "model/std_way/track_a_mpts52/outputs/structural_features.jsonl.gz"
V5_RUN = ROOT / "runs/symcif_v5_multidataset_wa_decoder/mpts52/val"
BUDGETS = (1, 5, 20)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_11: {resolved}")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path = ensure_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_report(title: str, body: str) -> None:
    with ensure_out(REPORT).open("a", encoding="utf-8") as f:
        f.write("\n\n" + f"## opentry_11 自主迭代实验：{title}\n\n" + body.strip() + "\n")


def pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.3f}%"


def pp(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):+.3f} pp"


def load_crystallm() -> dict[str, dict[str, Any]]:
    df = pd.read_json(STRUCT_FEATURES, lines=True, compression="gzip")
    df["match"] = df["match"].fillna(False).astype(bool)
    df["rank"] = df["rank"].astype(int)
    df["target_rows_ge7"] = df["target_rows_ge7"].astype(bool)
    out: dict[str, dict[str, Any]] = {}
    for mid, group in df.groupby("material_id"):
        g = group.sort_values("rank")
        matches = [False] + [bool(x) for x in g.head(50)["match"].tolist()]
        out[str(mid)] = {"rows7": bool(g["target_rows_ge7"].iloc[0]), "c_match": matches}
    return out


def load_symcif(name: str) -> dict[str, dict[str, Any]]:
    gen_path = V5_RUN / "generations" / f"{name}.jsonl"
    metric_path = V5_RUN / "metrics" / f"{name}_metrics.jsonl"
    by_mid: dict[str, list[tuple[float, int, bool, bool]]] = defaultdict(list)
    with gen_path.open("r", encoding="utf-8") as gf, metric_path.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            mid = str(gen["sample_id"]).split("__")[-1]
            score = float(gen.get("generation_score") if gen.get("generation_score") is not None else -1e9)
            by_mid[mid].append((score, int(gen["gen_index"]), bool(met.get("match_ok")), int(gen.get("row_count_target") or 0) >= 7))
    out: dict[str, dict[str, Any]] = {}
    for mid, arr in by_mid.items():
        arr = sorted(arr, key=lambda x: (x[0], -x[1]), reverse=True)
        matches = [False] + [bool(x[2]) for x in arr[:20]]
        out[mid] = {"rows7": bool(arr[0][3]) if arr else False, "s_match": matches, "candidate_count": len(arr)}
    return out


def make_sequence(pattern: str) -> list[str]:
    # Pattern syntax examples: C20, S5C15, C1S4C15.
    seq: list[str] = []
    i = 0
    while i < len(pattern):
        source = pattern[i]
        i += 1
        j = i
        while j < len(pattern) and pattern[j].isdigit():
            j += 1
        n = int(pattern[i:j])
        seq.extend([source] * n)
        i = j
    return seq[:20]


def evaluate(crystallm: dict[str, dict[str, Any]], sym: dict[str, dict[str, Any]], pattern: str) -> dict[str, Any]:
    seq = make_sequence(pattern)
    rows: list[dict[str, Any]] = []
    for mid in sorted(set(crystallm) & set(sym)):
        c = crystallm[mid]
        s = sym[mid]
        c_idx = 0
        s_idx = 0
        top_hits: list[bool] = []
        for source in seq:
            if source == "C":
                c_idx += 1
                top_hits.append(bool(c["c_match"][c_idx]) if c_idx < len(c["c_match"]) else False)
            else:
                s_idx += 1
                top_hits.append(bool(s["s_match"][s_idx]) if s_idx < len(s["s_match"]) else False)
        row = {"material_id": mid, "rows7": bool(c["rows7"])}
        for b in BUDGETS:
            row[f"hit@{b}"] = any(top_hits[:b])
        rows.append(row)
    rows7 = [r for r in rows if r["rows7"]]
    metrics: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7), "pattern": pattern}
    for b in BUDGETS:
        metrics[f"match@{b}"] = float(sum(1 for r in rows if r[f"hit@{b}"]) / max(1, len(rows)))
        metrics[f"rows>=7_match@{b}"] = float(sum(1 for r in rows7 if r[f"hit@{b}"]) / max(1, len(rows7)))
    return metrics


def delta(metrics: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {k: float(metrics[k] - base[k]) for k in [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]}


def achieved(d: dict[str, float]) -> bool:
    return sum(1 for b in BUDGETS if d[f"match@{b}"] >= 0.05) >= 2 or (d["rows>=7_match@5"] >= 0.05 and d["rows>=7_match@20"] >= 0.05)


def fmt_metrics(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"match@{b}"]) for b in BUDGETS)


def fmt_rows7(m: dict[str, Any]) -> str:
    return " / ".join(pct(m[f"rows>=7_match@{b}"]) for b in BUDGETS)


def fmt_delta(d: dict[str, float], prefix: str = "") -> str:
    return " / ".join(pp(d[f"{prefix}match@{b}"]) for b in BUDGETS)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    crystallm = load_crystallm()
    sym = load_symcif("v5_fullgen_eval_pool")
    patterns = [
        "C20",
        "S5C15",
        "C1S4C15",
        "C2S3C15",
        "C3S2C15",
        "C5S5C10",
        "C10S10",
        "C15S5",
        "C1S9C10",
        "C1S4C5S5C5",
        "S1C4S4C11",
    ]
    rows = []
    base = evaluate(crystallm, sym, "C20")
    for pattern in patterns:
        metrics = evaluate(crystallm, sym, pattern)
        d = delta(metrics, base)
        rows.append({"pattern": pattern, "metrics": metrics, "delta_vs_c20": d, "achieved": achieved(d)})
    best_by_k5 = max(rows, key=lambda r: (r["delta_vs_c20"]["match@5"], r["delta_vs_c20"]["match@20"]))
    best_by_k20 = max(rows, key=lambda r: (r["delta_vs_c20"]["match@20"], r["delta_vs_c20"]["match@5"]))
    best_rows7 = max(rows, key=lambda r: (r["delta_vs_c20"]["rows>=7_match@5"] + r["delta_vs_c20"]["rows>=7_match@20"], r["delta_vs_c20"]["match@20"]))
    result = {
        "time": now_iso(),
        "gpu_necessary": False,
        "gpu_reason": "fixed validation hybrid budget diagnostic over existing artifacts; no MP-20/MPTS-52 train dataset training",
        "contribution_boundary": "candidate fusion / auxiliary diagnostic, not main method",
        "baseline": base,
        "patterns": rows,
        "best_by_k5": best_by_k5,
        "best_by_k20": best_by_k20,
        "best_rows7": best_rows7,
        "achieved_any": any(r["achieved"] for r in rows),
    }
    write_json(RESULTS / "iteration_04_fixed_hybrid_budget.json", result)

    b5 = best_by_k5
    b20 = best_by_k20
    br = best_rows7
    append_report(
        "迭代 04 固定预算 CrystaLLM-SymCIF hybrid route",
        f"""
时间：{result['time']}

当前失败原因：迭代 03 证明 CrystaLLM 与 SymCIF generation 有 coverage 互补，但 union 是 oracle/coverage 视角；它没有说明一个固定、非 oracle 的 top20 列表能否同时提升 match@5 和 match@20。

实验假设：如果 coverage 互补足够强，预注册的固定预算列表，例如 `C1S4C15` 或 `C15S5`，应该在不看 GT match 的情况下把 SymCIF exact-cover candidates 插入 top20，并提升 K5/K20。若只提升 K20 或损害 K1/K5，则 fusion 方向只能作为诊断/辅助，不能继续当主线。

为什么可能解决问题：SymCIF fullgen pool 在 rows>=7 K1/K5 上强于 CrystaLLM，但 overall K20 弱；固定预算 hybrid 可能保留 CrystaLLM 的强 overall，同时补 rows>=7 coverage。

预期提升指标：优先看 match@5 和 match@20；同时必须报告 rows>=7 K1/K5/K20。

GPU 必要性判断：本轮只是固定预算 validation hybrid 诊断，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：overlap samples={base['samples']}；rows>=7 samples={base['rows>=7_samples']}；候选来源为 CrystaLLM validation K50 与 SymCIF v5 fullgen pool validation artifacts。

baseline C20：{fmt_metrics(base)}；rows>=7 = {fmt_rows7(base)}。

best_by_K5 pattern={b5['pattern']}：{fmt_metrics(b5['metrics'])}；delta = {fmt_delta(b5['delta_vs_c20'])}。rows>=7 = {fmt_rows7(b5['metrics'])}；rows>=7 delta = {fmt_delta(b5['delta_vs_c20'], 'rows>=7_')}。

best_by_K20 pattern={b20['pattern']}：{fmt_metrics(b20['metrics'])}；delta = {fmt_delta(b20['delta_vs_c20'])}。rows>=7 = {fmt_rows7(b20['metrics'])}；rows>=7 delta = {fmt_delta(b20['delta_vs_c20'], 'rows>=7_')}。

best_rows7 pattern={br['pattern']}：{fmt_metrics(br['metrics'])}；delta = {fmt_delta(br['delta_vs_c20'])}。rows>=7 = {fmt_rows7(br['metrics'])}；rows>=7 delta = {fmt_delta(br['delta_vs_c20'], 'rows>=7_')}。

可信度：固定预算 route 不使用 GT match 做 per-sample 选择，比 union oracle 更真实；但它仍是 candidate fusion / route engineering，不是主方法贡献。

和历史实验关系：直接检验实验 7C/迭代 03 的 coverage 互补是否能变成可执行 top20 route。

最终判决：achieved_any={result['achieved_any']}。即使达标，也只能写作 auxiliary hybrid route，不能写成主线；若未达标或只提升单一 K20，则 fusion 方向连续失败，应转向 geometry repair/skeleton proposer。

下一步：若没有至少两个 overall match 指标 +5pp，则停止 fixed fusion 方向，进入 skeleton-to-match conversion 迭代。
""",
    )
    print(json.dumps({"achieved_any": result["achieved_any"], "best_by_k5": b5, "best_by_k20": b20, "best_rows7": br}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
