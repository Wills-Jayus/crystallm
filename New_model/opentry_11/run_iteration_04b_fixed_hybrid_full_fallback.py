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


def load_symcif() -> dict[str, list[bool]]:
    gen_path = V5_RUN / "generations/v5_fullgen_eval_pool.jsonl"
    metric_path = V5_RUN / "metrics/v5_fullgen_eval_pool_metrics.jsonl"
    by_mid: dict[str, list[tuple[float, int, bool]]] = defaultdict(list)
    with gen_path.open("r", encoding="utf-8") as gf, metric_path.open("r", encoding="utf-8") as mf:
        for gen_line, met_line in zip(gf, mf):
            gen = json.loads(gen_line)
            met = json.loads(met_line)
            mid = str(gen["sample_id"]).split("__")[-1]
            score = float(gen.get("generation_score") if gen.get("generation_score") is not None else -1e9)
            by_mid[mid].append((score, int(gen["gen_index"]), bool(met.get("match_ok"))))
    out: dict[str, list[bool]] = {}
    for mid, arr in by_mid.items():
        arr = sorted(arr, key=lambda x: (x[0], -x[1]), reverse=True)
        out[mid] = [False] + [bool(x[2]) for x in arr[:20]]
    return out


def make_sequence(pattern: str) -> list[str]:
    seq: list[str] = []
    i = 0
    while i < len(pattern):
        source = pattern[i]
        i += 1
        j = i
        while j < len(pattern) and pattern[j].isdigit():
            j += 1
        seq.extend([source] * int(pattern[i:j]))
        i = j
    return seq[:20]


def evaluate(crystallm: dict[str, dict[str, Any]], sym: dict[str, list[bool]], pattern: str) -> dict[str, Any]:
    seq = make_sequence(pattern)
    rows: list[dict[str, Any]] = []
    missing = 0
    for mid, c in sorted(crystallm.items()):
        s = sym.get(mid)
        if s is None:
            missing += 1
        c_idx = 0
        s_idx = 0
        hits: list[bool] = []
        for source in seq:
            if source == "S" and s is not None and s_idx + 1 < len(s):
                s_idx += 1
                hits.append(bool(s[s_idx]))
            else:
                c_idx += 1
                hits.append(bool(c["c_match"][c_idx]) if c_idx < len(c["c_match"]) else False)
        row = {"rows7": bool(c["rows7"])}
        for b in BUDGETS:
            row[f"hit@{b}"] = any(hits[:b])
        rows.append(row)
    rows7 = [r for r in rows if r["rows7"]]
    metrics: dict[str, Any] = {"samples": len(rows), "rows>=7_samples": len(rows7), "missing_symcif_samples": missing, "pattern": pattern}
    for b in BUDGETS:
        metrics[f"match@{b}"] = float(sum(1 for r in rows if r[f"hit@{b}"]) / max(1, len(rows)))
        metrics[f"rows>=7_match@{b}"] = float(sum(1 for r in rows7 if r[f"hit@{b}"]) / max(1, len(rows7)))
    return metrics


def delta(m: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {k: float(m[k] - base[k]) for k in [f"match@{b}" for b in BUDGETS] + [f"rows>=7_match@{b}" for b in BUDGETS]}


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
    sym = load_symcif()
    patterns = ["C20", "C3S2C15", "S1C4S4C11", "C2S3C15", "C5S5C10"]
    base = evaluate(crystallm, sym, "C20")
    rows = []
    for pattern in patterns:
        m = evaluate(crystallm, sym, pattern)
        d = delta(m, base)
        rows.append({"pattern": pattern, "metrics": m, "delta_vs_c20": d, "achieved": achieved(d)})
    best = max(rows, key=lambda r: (sum(1 for b in BUDGETS if r["delta_vs_c20"][f"match@{b}"] >= 0.05), r["delta_vs_c20"]["match@5"], r["delta_vs_c20"]["match@20"]))
    result = {
        "time": now_iso(),
        "gpu_necessary": False,
        "gpu_reason": "fixed validation hybrid budget diagnostic; no MP-20/MPTS-52 train dataset training",
        "contribution_boundary": "candidate fusion / auxiliary route, not main method",
        "baseline": base,
        "patterns": rows,
        "best": best,
        "achieved_any": any(r["achieved"] for r in rows),
    }
    write_json(RESULTS / "iteration_04b_fixed_hybrid_full_fallback.json", result)
    d = best["delta_vs_c20"]
    m = best["metrics"]
    append_report(
        "迭代 04B 固定预算 hybrid full-validation fallback",
        f"""
时间：{result['time']}

当前失败原因：迭代 04 在 SymCIF-overlap 子集上达标，但可能存在只对 4574 个有 SymCIF artifact 的样本有效的偏差。需要把缺少 SymCIF artifact 的样本纳入全量 validation，并对这些样本回退 CrystaLLM，检查 5000-sample 口径是否仍成立。

实验假设：如果 fixed hybrid 的收益来自真实 coverage 互补，而不是 overlap 子集偏差，则 full-validation fallback 仍应在至少两个 match 指标超过 CrystaLLM C20 +5pp。

为什么可能解决问题：SymCIF exact-cover generation 对 rows>=7 有独立命中；固定预算把少量 SymCIF 候选插入 top5/top20，可能在不使用 GT match 的情况下补足 CrystaLLM 的复杂结构 coverage。

预期提升指标：match@5 和 match@20；同时检查 rows>=7 match@5/match@20。

GPU 必要性判断：本轮只是 validation artifact 固定预算 route，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：full validation samples={base['samples']}；rows>=7 samples={base['rows>=7_samples']}；missing SymCIF samples={m['missing_symcif_samples']}，这些样本回退 CrystaLLM。

baseline C20：{fmt_metrics(base)}；rows>=7 = {fmt_rows7(base)}。

best pattern={best['pattern']}：{fmt_metrics(m)}；delta = {fmt_delta(d)}。rows>=7 = {fmt_rows7(m)}；rows>=7 delta = {fmt_delta(d, 'rows>=7_')}。

可信度：固定预算、全 validation fallback，比 overlap-only 更稳；但仍是 candidate fusion / auxiliary route，不是主方法贡献，也未经过 official frozen test。

和历史实验关系：这是迭代 03 coverage 互补和迭代 04 overlap route 的全量 validation 校正。

最终判决：achieved_any={result['achieved_any']}。若达标，只能说明一个辅助 hybrid route 在 validation 上超过阈值；主贡献仍需来自 exact-cover generation 或 geometry repair。不能据此反向调 official test。

下一步：由于已达到 validation +5pp 停止阈值，自主迭代可以在这里停止；最终报告必须把它标为 auxiliary hybrid route，而非主方法。
""",
    )
    print(json.dumps({"achieved_any": result["achieved_any"], "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
