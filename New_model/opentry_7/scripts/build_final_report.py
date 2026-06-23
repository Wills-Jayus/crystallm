#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["mp_20", "mpts_52"]
REQUIRED_SYSTEMS = [
    "reproduced_crystallm_a_composition",
    "crystallm_a_gt_sg",
    "strategy_stablekey_hybrid",
    "pure_crystallm_gt_sg",
]

ROW_DEFS = [
    {
        "label": "published CrystaLLM-a composition-only",
        "kind": "published",
        "system": "published_crystallm_a_composition_only",
        "budget": "20",
        "fusion": "no",
        "pure": "no",
    },
    {
        "label": "reproduced CrystaLLM-a composition-only",
        "kind": "metric",
        "system": "reproduced_crystallm_a_composition",
        "budget": "20",
        "fusion": "no",
        "pure": "no",
    },
    {
        "label": "CrystaLLM-a GT-SG",
        "kind": "metric",
        "system": "crystallm_a_gt_sg",
        "budget": "20",
        "fusion": "no",
        "pure": "no",
    },
    {
        "label": "best strategy/fusion line",
        "kind": "metric",
        "system": "strategy_stablekey_hybrid",
        "budget": "20 final",
        "fusion": "yes",
        "pure": "no",
    },
    {
        "label": "best pure model line",
        "kind": "metric",
        "system": "pure_crystallm_gt_sg",
        "budget": "20",
        "fusion": "no",
        "pure": "yes",
    },
]

MATCH_KEYS = ["match@1", "match@5", "match@20"]
RMSE_KEYS = ["RMSE@1", "RMSE@5", "RMSE@20"]
ROW_MATCH_KEYS = ["rows>=7 match@1", "rows>=7 match@5", "rows>=7 match@20"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(system: str, dataset: str) -> Path:
    return ROOT / "metrics" / f"{system}_{dataset}_test_k20.json"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}"


def fmt_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{float(value) * 100:.2f}"


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def fmt_count(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def all_metric(metrics: dict[str, Any], key: str) -> Any:
    return metrics.get("all", {}).get(key)


def row_metric(metrics: dict[str, Any], key: str) -> Any:
    if key == "rows>=7 positive-any":
        return metrics.get("rows_ge7", {}).get("positive_any")
    return metrics.get("rows_ge7", {}).get(key.replace("rows>=7 ", ""))


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    if key in MATCH_KEYS or key in RMSE_KEYS:
        return all_metric(metrics, key)
    return row_metric(metrics, key)


def published_row(reference: dict[str, Any], dataset: str) -> dict[str, Any]:
    src = reference["published_crystallm_a_composition_only"][dataset]
    return {
        "match@1": src.get("match@1"),
        "match@5": src.get("match@5"),
        "match@20": src.get("match@20"),
        "RMSE@1": src.get("RMSE@1"),
        "RMSE@5": src.get("RMSE@5"),
        "RMSE@20": src.get("RMSE@20"),
        "rows>=7 match@1": None,
        "rows>=7 match@5": None,
        "rows>=7 match@20": None,
        "rows>=7 positive-any": None,
        "official_samples": None,
        "samples_with_any_candidate": None,
    }


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {key: metric_value(metrics, key) for key in MATCH_KEYS + RMSE_KEYS + ROW_MATCH_KEYS}
    row["rows>=7 positive-any"] = row_metric(metrics, "rows>=7 positive-any")
    row["official_samples"] = metrics.get("official_split_samples")
    row["samples_with_any_candidate"] = metrics.get("samples_with_any_candidate")
    return row


def row_to_table(label: str, row: dict[str, Any], budget: str, fusion: str, pure: str) -> str:
    values = [
        label,
        fmt_pct(row.get("match@1")),
        fmt_pct(row.get("match@5")),
        fmt_pct(row.get("match@20")),
        fmt_float(row.get("RMSE@1")),
        fmt_float(row.get("RMSE@5")),
        fmt_float(row.get("RMSE@20")),
        fmt_pct(row.get("rows>=7 match@1")),
        fmt_pct(row.get("rows>=7 match@5")),
        fmt_pct(row.get("rows>=7 match@20")),
        fmt_count(row.get("rows>=7 positive-any")),
        budget,
        fusion,
        pure,
    ]
    return "| " + " | ".join(values) + " |"


def collect_metrics() -> dict[str, dict[str, dict[str, Any]]]:
    missing: list[str] = []
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        out[dataset] = {}
        for system in REQUIRED_SYSTEMS:
            path = metrics_path(system, dataset)
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
            else:
                out[dataset][system] = load_json(path)
    if missing:
        raise SystemExit("Missing required metrics:\n" + "\n".join(f"- {m}" for m in missing))
    return out


def delta_row(system: str, dataset: str, metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    gt = metrics[dataset]["crystallm_a_gt_sg"]
    cur = metrics[dataset][system]
    keys = MATCH_KEYS + ROW_MATCH_KEYS
    return {key: metric_value(cur, key) - metric_value(gt, key) for key in keys}


def count_pure_wins(metrics: dict[str, dict[str, dict[str, Any]]]) -> tuple[int, list[str]]:
    wins: list[str] = []
    for dataset in DATASETS:
        deltas = delta_row("pure_crystallm_gt_sg", dataset, metrics)
        for key, value in deltas.items():
            if value is not None and value >= 0.05:
                wins.append(f"{dataset} {key} ({fmt_delta(value)} pp)")
    return len(wins), wins


def coverage_notes(metrics: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    notes: list[str] = []
    for dataset in DATASETS:
        for system in REQUIRED_SYSTEMS:
            src = metrics[dataset][system]
            official = src.get("official_split_samples")
            with_candidates = src.get("samples_with_any_candidate")
            if official is None or with_candidates is None:
                continue
            missing = int(official) - int(with_candidates)
            if missing:
                notes.append(f"{dataset} `{system}` has {missing} official sample(s) with no candidate and keeps them as failures.")
    return notes


def write_report(metrics: dict[str, dict[str, dict[str, Any]]]) -> None:
    reference = load_json(ROOT / "configs" / "crystallm_a_baselines.json")
    pure_cfg = load_json(ROOT / "frozen_pure_model" / "pure_model_protocol.json")
    strategy_cfg = load_json(ROOT / "frozen_strategy" / "strategy_stablekey_hybrid_config.json")

    lines: list[str] = []
    lines.append("# opentry_7 Final Report")
    lines.append("")
    lines.append("Protocol: official CrystaLLM Table 3 MP-20 and MPTS-52 splits, full test lists, 20 candidates per sample, StructureMatcher ltol=0.3/stol=0.5/angle_tol=10, and normalized RMSE from get_rms_dist(...)[0]. Missing or failed candidates remain failures.")
    lines.append("")
    lines.append("Rows>=7 is computed from raw `_atom_site` rows in the official benchmark CSV target CIF strings. Match/RMSE is independent of validity filtering; validity counts are diagnostics only.")
    notes = coverage_notes(metrics)
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")

    header = "| system | match@1 | match@5 | match@20 | RMSE@1 | RMSE@5 | RMSE@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 | rows>=7 positive-any | generation budget | fusion/ranking | pure model |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for spec in ROW_DEFS:
            if spec["kind"] == "published":
                row = published_row(reference, dataset)
            else:
                row = metric_row(metrics[dataset][spec["system"]])
            lines.append(row_to_table(spec["label"], row, spec["budget"], spec["fusion"], spec["pure"]))
        lines.append("")

        lines.append(f"### {dataset} deltas vs CrystaLLM-a GT-SG")
        lines.append("")
        lines.append("| line | match@1 pp | match@5 pp | match@20 pp | rows>=7 match@1 pp | rows>=7 match@5 pp | rows>=7 match@20 pp |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for label, system in (
            ("strategy/fusion", "strategy_stablekey_hybrid"),
            ("pure model", "pure_crystallm_gt_sg"),
        ):
            deltas = delta_row(system, dataset, metrics)
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        fmt_delta(deltas["match@1"]),
                        fmt_delta(deltas["match@5"]),
                        fmt_delta(deltas["match@20"]),
                        fmt_delta(deltas["rows>=7 match@1"]),
                        fmt_delta(deltas["rows>=7 match@5"]),
                        fmt_delta(deltas["rows>=7 match@20"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    win_count, win_list = count_pure_wins(metrics)
    lines.append("## Conclusions")
    lines.append("")
    lines.append(f"- Strategy/fusion line: `{strategy_cfg['strategy_id']}` uses candidate fusion/ranking and is not an independent model. It is the validation-frozen historical stablekey hybrid selected for this final K20 evaluation.")
    lines.append(f"- Pure model line: official full train/validation checkpoints selected by best validation loss; MP-20 iter {pure_cfg['checkpoints']['mp_20']['iter_num']} and MPTS-52 iter {pure_cfg['checkpoints']['mpts_52']['iter_num']}. Test candidate order is fixed: candidate 0 greedy, candidates 1-19 fixed-seed sampling, no reordering.")
    lines.append(f"- Pure model exceeds CrystaLLM-a GT-SG by at least 5 percentage points on {win_count} reported match metrics.")
    if win_list:
        lines.append("- Metrics meeting the >=5 pp pure-model criterion: " + "; ".join(win_list) + ".")
    else:
        lines.append("- Metrics meeting the >=5 pp pure-model criterion: none.")
    lines.append("- Strategy reaches current project historical best: yes within the audited available historical candidates. The final line is the frozen stablekey hybrid top20 from the strongest available validation-selected historical strategy; no test-label tuning or post-test reranking was performed inside opentry_7.")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `configs/`: evaluator, baseline and pure-train configs.")
    lines.append("- `checkpoints/`: pure-model working checkpoints.")
    lines.append("- `frozen_pure_model/`: frozen pure checkpoints and inference protocol.")
    lines.append("- `frozen_strategy/`: frozen strategy config.")
    lines.append("- `generations/`: final ordered K20 candidates and normalized generation JSONL files.")
    lines.append("- `eval/`: per-sample metrics and summaries.")
    lines.append("- `metrics/`: machine-readable summary JSON files.")
    lines.append("")

    out = ROOT / "opentry_7_final_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_report(collect_metrics())


if __name__ == "__main__":
    main()
