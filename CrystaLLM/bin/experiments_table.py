#!/usr/bin/env python3
"""
Collect key metrics from multiple CrystaLLM experiment directories into one CSV.

This is meant to reduce the manual effort of opening each `experiments/<exp>/round_XX/*`
folder to inspect results. It scans for:
  - per-round: round_XX/summary.json, round_XX/evaluator_summary.json
  - per-experiment: final_report.json

Example:
  python bin/experiments_table.py \
    --experiments-root experiments \
    --out experiments/experiments_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _round_index(path: Path) -> Optional[int]:
    # round_01 -> 1
    name = path.name
    if not name.startswith("round_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:  # noqa: BLE001
        return None


def _find_round_dirs(exp_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in exp_dir.iterdir():
        if not p.is_dir():
            continue
        idx = _round_index(p)
        if idx is None:
            continue
        out.append((idx, p))
    out.sort(key=lambda t: t[0])
    return out


def _iso_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001
        return ""


def _get(dct: Any, *keys: str) -> Any:
    cur = dct
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


@dataclass
class ExperimentRow:
    exp: str
    exp_path: str
    mtime: str

    model_dir: str
    initial_prompt_file: str

    rounds: Optional[int]
    samples_per_round: Optional[int]
    temperature: Optional[float]
    top_k: Optional[int]
    max_new_tokens: Optional[int]
    seed: Optional[int]
    device: str
    dtype: str

    last_round: Optional[int]
    last_validation_pass_ratio: Optional[float]
    last_formula_ok_ratio: Optional[float]
    last_atom_site_multiplicity_ok_ratio: Optional[float]
    last_space_group_ok_ratio: Optional[float]
    last_bond_lengths_reasonable_ratio: Optional[float]
    last_strict_valid_ratio: Optional[float]
    last_alignn_mean_formation_energy: Optional[float]
    last_alignn_mean_bandgap: Optional[float]

    best_overall_round: Optional[int]
    best_overall_sample_id: str
    best_overall_validation_ok: Optional[bool]
    best_overall_score_value: Optional[float]

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "exp",
            "exp_path",
            "mtime",
            "model_dir",
            "initial_prompt_file",
            "rounds",
            "samples_per_round",
            "temperature",
            "top_k",
            "max_new_tokens",
            "seed",
            "device",
            "dtype",
            "last_round",
            "last_validation_pass_ratio",
            "last_formula_ok_ratio",
            "last_atom_site_multiplicity_ok_ratio",
            "last_space_group_ok_ratio",
            "last_bond_lengths_reasonable_ratio",
            "last_strict_valid_ratio",
            "last_alignn_mean_formation_energy",
            "last_alignn_mean_bandgap",
            "best_overall_round",
            "best_overall_sample_id",
            "best_overall_validation_ok",
            "best_overall_score_value",
        ]

    def as_csv_row(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.csv_fieldnames()}


def _try_load_last_round_metrics(round_dir: Path) -> Dict[str, Any]:
    evaluator_path = round_dir / "evaluator_summary.json"
    if evaluator_path.exists():
        return _read_json(evaluator_path).get("metrics") or {}
    summary_path = round_dir / "summary.json"
    if summary_path.exists():
        return _read_json(summary_path).get("metrics") or {}
    return {}


def _try_load_args(round_dir: Path) -> Dict[str, Any]:
    summary_path = round_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return _read_json(summary_path).get("args") or {}


def _try_load_final_best(exp_dir: Path) -> Dict[str, Any]:
    p = exp_dir / "final_report.json"
    if not p.exists():
        return {}
    j = _read_json(p)
    if isinstance(j, dict):
        return j
    return {}


def _build_row(exp_dir: Path) -> Optional[ExperimentRow]:
    rounds = _find_round_dirs(exp_dir)
    if not rounds:
        return None

    last_idx, last_dir = rounds[-1]
    args = _try_load_args(last_dir)
    metrics = _try_load_last_round_metrics(last_dir)
    flt = metrics.get("filter") if isinstance(metrics, dict) else None
    if not isinstance(flt, dict):
        flt = {}
    alignn = metrics.get("alignn") if isinstance(metrics, dict) else None
    if not isinstance(alignn, dict):
        alignn = {}
    per_property = alignn.get("per_property")
    if not isinstance(per_property, dict):
        per_property = {}

    best = _try_load_final_best(exp_dir)
    best_overall = best.get("best_overall")
    best0 = best_overall[0] if isinstance(best_overall, list) and best_overall else {}
    if not isinstance(best0, dict):
        best0 = {}

    return ExperimentRow(
        exp=exp_dir.name,
        exp_path=str(exp_dir),
        mtime=_iso_mtime(exp_dir),
        model_dir=str(args.get("model_dir") or ""),
        initial_prompt_file=str(args.get("initial_prompt_file") or ""),
        rounds=int(args["rounds"]) if isinstance(args.get("rounds"), int) else None,
        samples_per_round=int(args["samples_per_round"]) if isinstance(args.get("samples_per_round"), int) else None,
        temperature=float(args["temperature"]) if isinstance(args.get("temperature"), (int, float)) else None,
        top_k=int(args["top_k"]) if isinstance(args.get("top_k"), int) else None,
        max_new_tokens=int(args["max_new_tokens"]) if isinstance(args.get("max_new_tokens"), int) else None,
        seed=int(args["seed"]) if isinstance(args.get("seed"), int) else None,
        device=str(args.get("device") or ""),
        dtype=str(args.get("dtype") or ""),
        last_round=last_idx,
        last_validation_pass_ratio=_get(metrics, "validation_pass_ratio"),
        last_formula_ok_ratio=_get(flt, "formula_ok", "ratio"),
        last_atom_site_multiplicity_ok_ratio=_get(flt, "atom_site_multiplicity_ok", "ratio"),
        last_space_group_ok_ratio=_get(flt, "space_group_ok", "ratio"),
        last_bond_lengths_reasonable_ratio=_get(flt, "bond_lengths_reasonable", "ratio"),
        last_strict_valid_ratio=_get(flt, "strict_valid", "ratio"),
        last_alignn_mean_formation_energy=_get(per_property, "formation_energy", "mean"),
        last_alignn_mean_bandgap=_get(per_property, "bandgap", "mean"),
        best_overall_round=int(best0.get("round")) if isinstance(best0.get("round"), int) else None,
        best_overall_sample_id=str(best0.get("sample_id") or ""),
        best_overall_validation_ok=best0.get("validation_ok") if isinstance(best0.get("validation_ok"), bool) else None,
        best_overall_score_value=float(best0.get("score_value"))
        if isinstance(best0.get("score_value"), (int, float))
        else None,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect CrystaLLM experiment metrics into one CSV table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiments-root",
        default=str(_ROOT / "experiments"),
        help="Root directory containing experiment subfolders.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <experiments-root>/experiments_table.csv",
    )
    p.add_argument("--pattern", default="*", help="Glob pattern for experiment folder names.")
    args = p.parse_args()

    root = Path(args.experiments_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (root / "experiments_table.csv")

    exp_dirs = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    rows: List[ExperimentRow] = []
    for exp_dir in exp_dirs:
        try:
            row = _build_row(exp_dir)
        except Exception:  # noqa: BLE001
            row = None
        if row is not None:
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ExperimentRow.csv_fieldnames())
        w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_row())

    print(f"[experiments_table] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

