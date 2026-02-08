#!/usr/bin/env python3
"""
Collect per-round metrics/prompts from multiple CrystaLLM experiment directories into one CSV.

This is meant to reduce manual browsing of:
  experiments/<exp>/round_XX/{prompt*.txt,evaluator_summary.json,qwen_output.json}

Example:
  python3 bin/experiments_rounds_table.py \
    --experiments-root experiments \
    --out experiments/experiments_rounds_table.csv
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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _single_line(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.replace("\n", "\\n")


def _round_index(path: Path) -> Optional[int]:
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


def _try_read_prompt(round_dir: Path, name: str) -> str:
    p = round_dir / name
    if not p.exists():
        return ""
    return _single_line(_read_text(p))


def _try_load_metrics(round_dir: Path) -> Dict[str, Any]:
    evaluator_path = round_dir / "evaluator_summary.json"
    if evaluator_path.exists():
        j = _read_json(evaluator_path)
        m = j.get("metrics")
        return m if isinstance(m, dict) else {}
    summary_path = round_dir / "summary.json"
    if summary_path.exists():
        j = _read_json(summary_path)
        m = j.get("metrics")
        return m if isinstance(m, dict) else {}
    return {}


def _try_load_qwen(round_dir: Path) -> Dict[str, Any]:
    for name in ["qwen_output.json", "llama_output.json"]:
        p = round_dir / name
        if p.exists():
            try:
                j = _read_json(p)
                return j if isinstance(j, dict) else {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _fmt_json(v: Any) -> str:
    if v is None:
        return ""
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=False)
    except Exception:  # noqa: BLE001
        return str(v)


@dataclass
class RoundRow:
    exp: str
    round: int
    round_path: str
    mtime: str

    prompt: str
    prompt_for_sampling: str

    n_total: Optional[int]
    alignn_total: Optional[int]
    alignn_ok: Optional[int]

    validation_ok_true: Optional[int]
    validation_ok_ratio: Optional[float]
    space_group_ok_true: Optional[int]
    space_group_ok_ratio: Optional[float]
    bond_lengths_reasonable_true: Optional[int]
    bond_lengths_reasonable_ratio: Optional[float]
    formula_ok_true: Optional[int]
    formula_ok_ratio: Optional[float]
    formula_ok_relaxed_true: Optional[int]
    formula_ok_relaxed_ratio: Optional[float]
    atom_site_multiplicity_ok_true: Optional[int]
    atom_site_multiplicity_ok_ratio: Optional[float]
    strict_valid_true: Optional[int]
    strict_valid_ratio: Optional[float]

    formation_energy_mean: Optional[float]
    formation_energy_max: Optional[float]

    qwen_analysis: str
    qwen_proposed_next_prompt_lines: str
    qwen_applied_key_changes: str

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "exp",
            "round",
            "round_path",
            "mtime",
            "prompt",
            "prompt_for_sampling",
            "n_total",
            "alignn_total",
            "alignn_ok",
            "validation_ok_true",
            "validation_ok_ratio",
            "space_group_ok_true",
            "space_group_ok_ratio",
            "bond_lengths_reasonable_true",
            "bond_lengths_reasonable_ratio",
            "formula_ok_true",
            "formula_ok_ratio",
            "formula_ok_relaxed_true",
            "formula_ok_relaxed_ratio",
            "atom_site_multiplicity_ok_true",
            "atom_site_multiplicity_ok_ratio",
            "strict_valid_true",
            "strict_valid_ratio",
            "formation_energy_mean",
            "formation_energy_max",
            "qwen_analysis",
            "qwen_proposed_next_prompt_lines",
            "qwen_applied_key_changes",
        ]

    def as_csv_row(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.csv_fieldnames()}


def _true_ratio(flt: Dict[str, Any], key: str) -> Tuple[Optional[int], Optional[float]]:
    v = flt.get(key)
    if not isinstance(v, dict):
        return None, None
    t = v.get("true")
    r = v.get("ratio")
    t_out = int(t) if isinstance(t, int) else None
    r_out = float(r) if isinstance(r, (int, float)) else None
    return t_out, r_out


def _build_round_row(exp_dir: Path, idx: int, round_dir: Path) -> RoundRow:
    metrics = _try_load_metrics(round_dir)
    flt = metrics.get("filter") if isinstance(metrics, dict) else None
    if not isinstance(flt, dict):
        flt = {}
    alignn = metrics.get("alignn") if isinstance(metrics, dict) else None
    if not isinstance(alignn, dict):
        alignn = {}
    per_property = alignn.get("per_property")
    if not isinstance(per_property, dict):
        per_property = {}
    fe = per_property.get("formation_energy")
    if not isinstance(fe, dict):
        fe = {}

    qwen = _try_load_qwen(round_dir)
    parsed = qwen.get("parsed")
    if not isinstance(parsed, dict):
        parsed = {}
    psa = qwen.get("prompt_sanitizer_audit")
    if not isinstance(psa, dict):
        psa = {}
    audit = psa.get("sanitizer_audit")
    if not isinstance(audit, dict):
        audit = {}

    validation_ok_true, validation_ok_ratio = _true_ratio(flt, "validation_ok")
    space_group_ok_true, space_group_ok_ratio = _true_ratio(flt, "space_group_ok")
    bond_lengths_reasonable_true, bond_lengths_reasonable_ratio = _true_ratio(flt, "bond_lengths_reasonable")
    formula_ok_true, formula_ok_ratio = _true_ratio(flt, "formula_ok")
    formula_ok_relaxed_true, formula_ok_relaxed_ratio = _true_ratio(flt, "formula_ok_relaxed")
    atom_site_multiplicity_ok_true, atom_site_multiplicity_ok_ratio = _true_ratio(flt, "atom_site_multiplicity_ok")
    strict_valid_true, strict_valid_ratio = _true_ratio(flt, "strict_valid")

    return RoundRow(
        exp=exp_dir.name,
        round=idx,
        round_path=str(round_dir),
        mtime=_iso_mtime(round_dir),
        prompt=_try_read_prompt(round_dir, "prompt.txt"),
        prompt_for_sampling=_try_read_prompt(round_dir, "prompt_for_sampling.txt"),
        n_total=int(flt["n_total"]) if isinstance(flt.get("n_total"), int) else None,
        alignn_total=int(alignn["total"]) if isinstance(alignn.get("total"), int) else None,
        alignn_ok=int(alignn["ok"]) if isinstance(alignn.get("ok"), int) else None,
        validation_ok_true=validation_ok_true,
        validation_ok_ratio=validation_ok_ratio,
        space_group_ok_true=space_group_ok_true,
        space_group_ok_ratio=space_group_ok_ratio,
        bond_lengths_reasonable_true=bond_lengths_reasonable_true,
        bond_lengths_reasonable_ratio=bond_lengths_reasonable_ratio,
        formula_ok_true=formula_ok_true,
        formula_ok_ratio=formula_ok_ratio,
        formula_ok_relaxed_true=formula_ok_relaxed_true,
        formula_ok_relaxed_ratio=formula_ok_relaxed_ratio,
        atom_site_multiplicity_ok_true=atom_site_multiplicity_ok_true,
        atom_site_multiplicity_ok_ratio=atom_site_multiplicity_ok_ratio,
        strict_valid_true=strict_valid_true,
        strict_valid_ratio=strict_valid_ratio,
        formation_energy_mean=float(fe["mean"]) if isinstance(fe.get("mean"), (int, float)) else None,
        formation_energy_max=float(fe["max"]) if isinstance(fe.get("max"), (int, float)) else None,
        qwen_analysis=_single_line(str(parsed.get("analysis") or "")),
        qwen_proposed_next_prompt_lines=_fmt_json(parsed.get("next_prompt_lines")),
        qwen_applied_key_changes=_fmt_json(audit.get("key_changes")),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect per-round CrystaLLM metrics/prompts into one CSV table.",
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
        help="Output CSV path. Defaults to <experiments-root>/experiments_rounds_table.csv",
    )
    p.add_argument("--pattern", default="*", help="Glob pattern for experiment folder names.")
    args = p.parse_args()

    root = Path(args.experiments_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else (root / "experiments_rounds_table.csv")

    exp_dirs = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    rows: List[RoundRow] = []
    for exp_dir in exp_dirs:
        rounds = _find_round_dirs(exp_dir)
        for idx, round_dir in rounds:
            try:
                rows.append(_build_round_row(exp_dir, idx, round_dir))
            except Exception:  # noqa: BLE001
                continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RoundRow.csv_fieldnames())
        w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_row())

    print(f"[experiments_rounds_table] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

