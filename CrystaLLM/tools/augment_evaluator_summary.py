#!/usr/bin/env python3
"""
Augment round_XX/evaluator_summary.json with failure reason distribution and examples.

Why:
  In many runs, `top_structures` are selected from `validation_ok=True` candidates,
  so their `validation_reasons` are often empty. This script makes sure the optimizer
  LLM can still "see" where failures concentrate by adding:

    evaluator_summary["validation_failure_reasons_top"]
    evaluator_summary["validation_failure_examples"]
    evaluator_summary["validation_counts"]

Inputs (expected in each round dir):
  - cif_quality.json (preferred) or cif_quality.csv
  - scores.csv (optional; used for ranking failure examples by score_property)
  - summary.json (optional; used to infer score_property/goal if scores.csv missing)

Usage:
  python model/CrystaLLM/tools/augment_evaluator_summary.py \
    /root/autodl-tmp/model/CrystaLLM/experiments/na2cl2_qwen_v7_small/round_01

  # or whole experiment directory (auto-detect round_XX/)
  python model/CrystaLLM/tools/augment_evaluator_summary.py \
    /root/autodl-tmp/model/CrystaLLM/experiments/na2cl2_qwen_v7_small
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment evaluator_summary.json with validation failure reason distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("path", help="A round_XX directory or an experiment directory containing round_XX/")
    p.add_argument("--top-n", type=int, default=20, help="How many top failure reasons to include")
    p.add_argument("--examples", type=int, default=5, help="How many failure examples to include")
    p.add_argument("--inplace", action="store_true", help="Write back to evaluator_summary.json (default true)")
    p.add_argument("--stdout", action="store_true", help="Print the updated evaluator_summary.json to stdout")
    return p.parse_args()


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip()
    if s in {"True", "true", "1", "1.0"}:
        return True
    if s in {"False", "false", "0", "0.0"}:
        return False
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cif_quality(round_dir: Path) -> List[Dict[str, Any]]:
    json_path = round_dir / "cif_quality.json"
    if json_path.exists():
        data = _load_json(json_path)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        raise ValueError(f"Unexpected cif_quality.json format: {json_path}")

    csv_path = round_dir / "cif_quality.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "cif_path": r.get("cif_path"),
                    "validation_ok": _as_bool(r.get("validation_ok")),
                    "validation_reasons": _parse_reasons(r.get("validation_reasons")),
                }
            )
        return out

    raise FileNotFoundError(f"Neither cif_quality.json nor cif_quality.csv found under {round_dir}")


def _parse_reasons(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    # common csv string form: "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # naive split that works for our simple reason strings
        parts = [p.strip().strip('"').strip("'").strip() for p in inner.split(",")]
        return [p for p in parts if p]
    return [s]


def _normalize_reason(reason: str) -> str:
    r = reason.strip()
    # Collapse parameterized reasons to a stable bucket.
    r = re.sub(r"unreasonable bond lengths \\(~\\d+% flagged\\)", "unreasonable bond lengths", r)
    r = re.sub(r"min_neighbor_dist<[^:]+", "min_neighbor_dist<symprec", r)
    return r


def _load_scores(round_dir: Path) -> List[Dict[str, Any]]:
    path = round_dir / "scores.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _infer_score_property_and_goal(round_dir: Path) -> Tuple[str, str]:
    summary_path = round_dir / "summary.json"
    if summary_path.exists():
        s = _load_json(summary_path)
        prop = s.get("score_property")
        goal = s.get("score_goal")
        if isinstance(prop, str) and prop:
            return prop, str(goal or "max")
    # fallback to common default in this repo
    return "bandgap", "max"


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _build_failure_examples(
    quality_rows: List[Dict[str, Any]],
    score_rows: List[Dict[str, Any]],
    score_property: str,
    score_goal: str,
    limit: int,
) -> List[Dict[str, Any]]:
    quality_by_path: Dict[str, Dict[str, Any]] = {}
    for r in quality_rows:
        path = str(r.get("cif_path") or "")
        if path:
            quality_by_path[path] = r

    failed: List[Dict[str, Any]] = []
    for s in score_rows:
        path = str(s.get("cif_path") or "")
        q = quality_by_path.get(path, {})
        ok = _as_bool(q.get("validation_ok"))
        if ok is not False:
            continue
        score = _float_or_none(s.get(score_property))
        failed.append(
            {
                "cif_path": path,
                "relative_path": s.get("relative_path"),
                "score_property": score_property,
                "score_value": score,
                "properties": {
                    "formation_energy": _float_or_none(s.get("formation_energy")),
                    "bandgap": _float_or_none(s.get("bandgap")),
                },
                "validation_reasons": _parse_reasons(q.get("validation_reasons")),
                "validator_step_errors": q.get("validator_step_errors") or {},
            }
        )

    reverse = str(score_goal).lower() != "min"
    failed.sort(key=lambda x: (x["score_value"] is None, x["score_value"]), reverse=reverse)
    return failed[:limit]


def augment_round(round_dir: Path, top_n: int, examples: int) -> Dict[str, Any]:
    evaluator_path = round_dir / "evaluator_summary.json"
    if not evaluator_path.exists():
        raise FileNotFoundError(f"Missing evaluator_summary.json: {evaluator_path}")

    evaluator = _load_json(evaluator_path)
    if not isinstance(evaluator, dict):
        raise ValueError(f"Unexpected evaluator_summary.json format: {evaluator_path}")

    quality_rows = _load_cif_quality(round_dir)
    fail_counter = Counter()
    n_pass = 0
    n_fail = 0
    n_total = 0
    for r in quality_rows:
        ok = _as_bool(r.get("validation_ok"))
        if ok is True:
            n_pass += 1
        elif ok is False:
            n_fail += 1
            for reason in _parse_reasons(r.get("validation_reasons")):
                fail_counter[_normalize_reason(reason)] += 1
        n_total += 1

    score_rows = _load_scores(round_dir)
    score_property, score_goal = _infer_score_property_and_goal(round_dir)
    failure_examples = _build_failure_examples(
        quality_rows=quality_rows,
        score_rows=score_rows,
        score_property=score_property,
        score_goal=score_goal,
        limit=examples,
    )

    evaluator["validation_counts"] = {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_ratio": (n_pass / n_total) if n_total else None,
        "check_composition": None,  # can be filled by caller if desired
    }
    evaluator["validation_failure_reasons_top"] = [
        {"reason": reason, "count": count} for reason, count in fail_counter.most_common(top_n)
    ]
    evaluator["validation_failure_examples"] = failure_examples
    return evaluator


def _iter_round_dirs(path: Path) -> List[Path]:
    if (path / "evaluator_summary.json").exists():
        return [path]
    return sorted([p for p in path.glob("round_*") if p.is_dir() and (p / "evaluator_summary.json").exists()])


def main() -> None:
    args = _parse_args()
    root = Path(args.path).expanduser().resolve()
    round_dirs = _iter_round_dirs(root)
    if not round_dirs:
        raise SystemExit(f"No round dirs found under {root}")

    updated: Dict[str, Any] | None = None
    for rd in round_dirs:
        updated = augment_round(rd, top_n=args.top_n, examples=args.examples)
        if args.inplace or (not args.stdout):
            (rd / "evaluator_summary.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.stdout and updated is not None:
        print(json.dumps(updated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

