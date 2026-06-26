#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = {
    "match_at_1": 0.2523,
    "match_at_5": 0.3646,
    "match_at_20": 0.4396,
}
ROWS_GE7_ANCHOR = {
    "match_at_1": 0.2249,
    "match_at_5": 0.3337,
    "match_at_20": 0.4104,
}


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def parse_metric(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return ast.literal_eval(stripped)
    raise RuntimeError(f"no metric dictionary found in {path}")


def metric_key(k: int) -> str:
    return f"match_at_{k}"


def pp_delta(value: float, anchor: float) -> float:
    return (float(value) - float(anchor)) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MPTS-52 official-test match metrics.")
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--k1-raw", required=True)
    parser.add_argument("--k5-raw", required=True)
    parser.add_argument("--k20-raw", required=True)
    parser.add_argument("--rows-ge7-k1-raw")
    parser.add_argument("--rows-ge7-k5-raw")
    parser.add_argument("--rows-ge7-k20-raw")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    raw = {1: parse_metric(Path(args.k1_raw)), 5: parse_metric(Path(args.k5_raw)), 20: parse_metric(Path(args.k20_raw))}
    official = {metric_key(k): float(raw[k]["match_rate"]) for k in (1, 5, 20)}
    rmse = {f"rmse_at_{k}": raw[k].get("rms_dist") for k in (1, 5, 20)}
    deltas = {key: pp_delta(value, ANCHOR[key]) for key, value in official.items()}

    rows_raw: dict[int, dict[str, Any]] = {}
    rows_paths = {1: args.rows_ge7_k1_raw, 5: args.rows_ge7_k5_raw, 20: args.rows_ge7_k20_raw}
    for k, path in rows_paths.items():
        if path:
            rows_raw[k] = parse_metric(Path(path))
    rows_official = {metric_key(k): float(rows_raw[k]["match_rate"]) for k in sorted(rows_raw)}
    rows_delta = {key: pp_delta(value, ROWS_GE7_ANCHOR[key]) for key, value in rows_official.items()}

    success_reasons: list[str] = []
    failure_reasons: list[str] = []
    if max(deltas.values()) >= 1.0:
        success_reasons.append("at least one overall match metric improves by >= 1.0 pp")
    else:
        failure_reasons.append("no overall match metric improves by >= 1.0 pp")
    if deltas["match_at_20"] < -0.2:
        failure_reasons.append("match@20 drops by more than 0.2 pp")
    else:
        success_reasons.append("match@20 drop constraint is satisfied")
    if rows_delta:
        corresponding = []
        for key, delta in deltas.items():
            if delta >= 1.0 and key in rows_delta:
                corresponding.append(rows_delta[key])
        if corresponding and min(corresponding) < -0.2:
            failure_reasons.append("rows>=7 corresponding metric drops by more than 0.2 pp")
        elif corresponding:
            success_reasons.append("rows>=7 corresponding metric does not clearly drop")
        else:
            success_reasons.append("rows>=7 metrics computed; no >=1 pp overall metric needs a corresponding gate")
    else:
        failure_reasons.append("rows>=7 metrics are missing")

    summary = {
        "dataset": "MPTS-52",
        "system": args.system_id,
        "protocol": "CrystaLLM Table 3 official full-test",
        "anchor": ANCHOR,
        "rows_ge7_anchor": ROWS_GE7_ANCHOR,
        "official_test": {**official, **rmse, "n_ids": raw[20].get("n_ids")},
        "delta_vs_anchor_pp": deltas,
        "rows_ge7_official_test": rows_official,
        "rows_ge7_delta_vs_anchor_pp": rows_delta,
        "success_standard_met": bool(not failure_reasons and max(deltas.values()) >= 1.0),
        "success_reasons": success_reasons,
        "failure_reasons": failure_reasons,
        "test_feedback_policy": "Do not use these official aggregates to tune thresholds or select a new route.",
    }

    out_json = under_root(Path(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")

    out_report = under_root(Path(args.out_report))
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# MPTS-52 {args.system_id} Official Test",
        "",
        "| metric | anchor | official | delta | rows>=7 anchor | rows>=7 official | rows>=7 delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in (1, 5, 20):
        key = metric_key(k)
        rows_value = rows_official.get(key)
        rows_anchor = ROWS_GE7_ANCHOR[key]
        rows_d = rows_delta.get(key)
        lines.append(
            f"| match@{k} | {ANCHOR[key] * 100:.2f}% | {official[key] * 100:.3f}% | {deltas[key]:.3f} pp | "
            f"{rows_anchor * 100:.2f}% | {'' if rows_value is None else f'{rows_value * 100:.3f}%'} | "
            f"{'' if rows_d is None else f'{rows_d:.3f} pp'} |"
        )
    lines.extend(["", f"Success standard met: `{str(summary['success_standard_met']).lower()}`."])
    if failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for reason in failure_reasons:
            lines.append(f"- {reason}")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
