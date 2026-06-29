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
        row: dict[str, Any] = {"material_id": str(mid), "rows7": bool(g["target_rows_ge7"].iloc[0])}
        for k in (1, 5, 10, 15, 20, 30, 50):
            row[f"crystallm@{k}"] = bool(g.head(k)["match"].any())
        out[str(mid)] = row
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
        row: dict[str, Any] = {"material_id": mid, "rows7": bool(arr[0][3]) if arr else False, "candidate_count": len(arr)}
        for k in (1, 5, 10, 20):
            row[f"{name}@{k}"] = any(x[2] for x in arr[:k])
        out[mid] = row
    return out


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows7 = [r for r in rows if r["rows7"]]
    return {
        "samples": len(rows),
        "rows>=7_samples": len(rows7),
        "match": float(sum(1 for r in rows if r.get(key, False)) / max(1, len(rows))),
        "rows>=7_match": float(sum(1 for r in rows7 if r.get(key, False)) / max(1, len(rows7))),
    }


def add_union(rows: list[dict[str, Any]], left: str, right: str, out_key: str) -> None:
    for row in rows:
        row[out_key] = bool(row.get(left, False) or row.get(right, False))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    crystallm = load_crystallm()
    sources = {
        "a1": load_symcif("v5_a1_exact_cover_sg_formula_e08"),
        "pool": load_symcif("v5_fullgen_eval_pool"),
    }
    result: dict[str, Any] = {
        "time": now_iso(),
        "gpu_necessary": False,
        "gpu_reason": "coverage diagnostic over existing validation candidates and generation artifacts; no MP-20/MPTS-52 train dataset training",
        "source_note": "This is diagnostic coverage/fusion upper-bound analysis, not a main method contribution.",
        "subsets": {},
    }
    for name, sym in sources.items():
        mids = sorted(set(crystallm) & set(sym))
        rows: list[dict[str, Any]] = []
        for mid in mids:
            row = dict(crystallm[mid])
            row.update(sym[mid])
            rows.append(row)
        for c_key, s_key in [
            ("crystallm@20", f"v5_{'a1_exact_cover_sg_formula_e08' if name == 'a1' else 'fullgen_eval_pool'}@5"),
        ]:
            pass
        prefix = "v5_a1_exact_cover_sg_formula_e08" if name == "a1" else "v5_fullgen_eval_pool"
        add_union(rows, "crystallm@20", f"{prefix}@5", "union_c20_s5")
        add_union(rows, "crystallm@15", f"{prefix}@5", "budget_c15_s5")
        add_union(rows, "crystallm@10", f"{prefix}@10", "budget_c10_s10")
        add_union(rows, "crystallm@20", f"{prefix}@20", "union_c20_s20")
        add_union(rows, "crystallm@50", f"{prefix}@20", "coverage_c50_s20")
        metrics: dict[str, Any] = {}
        for key in [
            "crystallm@1",
            "crystallm@5",
            "crystallm@20",
            "crystallm@50",
            f"{prefix}@1",
            f"{prefix}@5",
            f"{prefix}@20",
            "union_c20_s5",
            "budget_c15_s5",
            "budget_c10_s10",
            "union_c20_s20",
            "coverage_c50_s20",
        ]:
            metrics[key] = summarize(rows, key)
        baseline = metrics["crystallm@20"]
        deltas = {
            key: {
                "match": metrics[key]["match"] - baseline["match"],
                "rows>=7_match": metrics[key]["rows>=7_match"] - baseline["rows>=7_match"],
            }
            for key in metrics
        }
        result["subsets"][name] = {
            "samples": len(rows),
            "rows>=7_samples": sum(1 for r in rows if r["rows7"]),
            "symcif_prefix": prefix,
            "metrics": metrics,
            "deltas_vs_crystallm20": deltas,
        }
    write_json(RESULTS / "iteration_03_generation_coverage_union.json", result)

    pool = result["subsets"]["pool"]
    a1 = result["subsets"]["a1"]
    append_report(
        "迭代 03 CrystaLLM-SymCIF coverage union 诊断",
        f"""
时间：{result['time']}

当前失败原因：实验 5B 显示 deterministic repair conversion=0；实验 7C 显示 SymCIF exact-cover generation rows>=7 K1/K5 有信号但 overall K20 下降；实验 8C 判断真正 coverage 和 skeleton-to-match conversion 仍不足。迭代 02 的安全 scorer 也没有达到两个 match 指标 +5pp，继续普通 rerank 不应作为主线。

实验假设：如果 SymCIF exact-cover generation 能补 CrystaLLM top20 没覆盖的样本，则 CrystaLLM K20 与 SymCIF topK 的 union coverage 应明显超过 CrystaLLM K20；如果 union coverage 仍不足，说明不是简单 hybrid/fusion 可以解决，必须回到生成侧 coverage 或 geometry repair。

为什么可能解决问题：SymCIF generation 是 exact-cover constrained skeleton proposal，理论上应补足 CrystaLLM 候选池里 skeleton coverage 的空洞；本实验检查它是否真的补了 StructureMatcher match coverage。

预期提升指标：主要看 match@20 coverage 上限和 rows>=7 match@20 coverage，上限若超过 baseline +5pp，才值得后续设计非 oracle selector 或 train-data 级生成模型。

GPU 必要性判断：本轮只评估已有 validation candidates 和 SymCIF generation artifacts，不使用 MP-20/MPTS-52 train 数据集训练模型，因此不需要 GPU。

数据规模：A1 overlap samples={a1['samples']}，rows>=7={a1['rows>=7_samples']}；fullgen pool overlap samples={pool['samples']}，rows>=7={pool['rows>=7_samples']}。

A1 子集结果：CrystaLLM@20 = {pct(a1['metrics']['crystallm@20']['match'])}，rows>=7={pct(a1['metrics']['crystallm@20']['rows>=7_match'])}；SymCIF A1@20 = {pct(a1['metrics']['v5_a1_exact_cover_sg_formula_e08@20']['match'])}，rows>=7={pct(a1['metrics']['v5_a1_exact_cover_sg_formula_e08@20']['rows>=7_match'])}；union CrystaLLM@20 OR A1@20 = {pct(a1['metrics']['union_c20_s20']['match'])}，delta={pp(a1['deltas_vs_crystallm20']['union_c20_s20']['match'])}；rows>=7={pct(a1['metrics']['union_c20_s20']['rows>=7_match'])}，rows>=7 delta={pp(a1['deltas_vs_crystallm20']['union_c20_s20']['rows>=7_match'])}。

fullgen pool 子集结果：CrystaLLM@20 = {pct(pool['metrics']['crystallm@20']['match'])}，rows>=7={pct(pool['metrics']['crystallm@20']['rows>=7_match'])}；SymCIF pool@20 = {pct(pool['metrics']['v5_fullgen_eval_pool@20']['match'])}，rows>=7={pct(pool['metrics']['v5_fullgen_eval_pool@20']['rows>=7_match'])}；union CrystaLLM@20 OR pool@20 = {pct(pool['metrics']['union_c20_s20']['match'])}，delta={pp(pool['deltas_vs_crystallm20']['union_c20_s20']['match'])}；rows>=7={pct(pool['metrics']['union_c20_s20']['rows>=7_match'])}，rows>=7 delta={pp(pool['deltas_vs_crystallm20']['union_c20_s20']['rows>=7_match'])}。

预算型非 oracle 粗诊断：CrystaLLM@15 OR pool@5 = {pct(pool['metrics']['budget_c15_s5']['match'])}，delta={pp(pool['deltas_vs_crystallm20']['budget_c15_s5']['match'])}；CrystaLLM@10 OR pool@10 = {pct(pool['metrics']['budget_c10_s10']['match'])}，delta={pp(pool['deltas_vs_crystallm20']['budget_c10_s10']['match'])}。

可信度：这是 coverage/fusion 诊断，不是主方法；union 使用“是否任一来源命中”的上限视角，不能作为可部署 selector，也不能作为论文主贡献。

和历史实验关系：直接回应实验 7C 的问题：SymCIF exact-cover 是否为 CrystaLLM 补 coverage。若 union gain 很小，说明 exact-cover generation 与 CrystaLLM 命中高度重叠或 geometry 转化仍失败。

最终判决：本实验不作为停止依据。它只判断是否值得继续做 hybrid selector 或 train-data generation。若 union 相对 CrystaLLM@20 仍小于 +5pp，则停止 fusion 方向，转向真正 geometry repair/skeleton proposer。

下一步：根据 union coverage 判断。若 fullgen union 未提供足够 +5pp coverage，上一个普通 hybrid 方向也应停止；下一轮必须是 skeleton-to-match conversion 方案。
""",
    )
    print(json.dumps(result["subsets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
