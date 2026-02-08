"""
Batch test: sample prompts from CIFs and run multi-round CrystaLLM loops to check reproduction quality.

Outputs:
- Creates a "big experiment" folder under experiments/, containing one subfolder per prompt (small experiment).
- Prints per-round metrics (space_group_ok, bond_lengths_reasonable, formula_ok, atom_site_multiplicity_ok),
  Qwen analysis/next_prompt_lines, ALIGNN per-round mean scores, and per-small-experiment averages.
- Prints a big-experiment aggregate summary (per-prompt scores, mean bandgap/formation_energy, validation stats).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import pickle
import random
import re
import textwrap
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _require_supported_python() -> None:
    # This repo pins numpy/pymatgen versions in requirements.txt that do not support Python >= 3.12.
    if sys.version_info >= (3, 12):
        raise SystemExit(
            "Python version not supported by this repo's pinned dependencies.\n"
            f"Detected: {sys.version.split()[0]}\n"
            "Please run with Python 3.10 or 3.11, e.g.:\n"
            "  conda create -n crystallm310 python=3.10 -y\n"
            "  conda activate crystallm310\n"
            "  python -m pip install -r /root/autodl-tmp/model/CrystaLLM/requirements.txt\n"
            "Then rerun this script using that environment's python."
        )


def _read_cif_tags(cif: str) -> Tuple[str, str]:
    """Extract formula (fallback: data_ name) and space group (fallback: P1)."""

    def _tag(tag: str) -> Optional[str]:
        m = re.search(rf"(?im)^\s*{re.escape(tag)}\s+(.+?)\s*$", cif)
        if not m:
            return None
        v = m.group(1).strip()
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1].strip()
        return v

    formula = _tag("_chemical_formula_sum") or _tag("_chemical_formula_structural")
    if not formula:
        m = re.search(r"(?im)^\s*data_([^\s]+)", cif)
        formula = m.group(1) if m else "UNKNOWN"
    formula = re.sub(r"\s+", "", formula)

    sg = _tag("_symmetry_space_group_name_H-M") or _tag("_space_group_name_H-M_alt") or "P1"
    return formula, sg


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def _load_cifs(data_path: Path) -> List[Tuple[str, str]]:
    with gzip.open(data_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {data_path}")
    return data


def _pick_prompts(data: List[Tuple[str, str]], num: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    picked = rng.sample(data, num)
    out: List[Dict[str, Any]] = []
    for cid, cif in picked:
        formula, sg = _read_cif_tags(cif)
        prompt = f"data_{formula}\n_symmetry_space_group_name_H-M   {sg}\n"
        out.append({"id": str(cid), "formula": formula, "space_group": sg, "prompt": prompt})
    return out


def _write_prompts(prompts: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, item in enumerate(prompts):
        p = out_dir / f"prompt_{i:02d}_{_safe_name(item['id'])}.txt"
        p.write_text(item["prompt"], encoding="utf-8")
        paths.append(p)
    return paths


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _safe_ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def _truncate(s: str, max_len: int) -> str:
    s = str(s)
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    if max_len == 1:
        return s[:1]
    return s[: max_len - 1] + "…"


def _fmt_float(v: Any, decimals: int) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.{decimals}f}"
    return "NA"


def _infer_round_from_path(path: Any) -> Optional[int]:
    if not isinstance(path, str):
        return None
    m = re.search(r"/round_(\d{2})/", path)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _print_aligned_table(headers: List[str], rows: List[List[str]], max_widths: Optional[List[int]] = None) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))
    if max_widths:
        widths = [min(w, max_widths[i]) for i, w in enumerate(widths)]

    def _pad(i: int, s: str) -> str:
        s2 = _truncate(s, widths[i])
        # left align first column, right align others
        if i == 0:
            return s2.ljust(widths[i])
        return s2.rjust(widths[i])

    print("  ".join(_pad(i, headers[i]) for i in range(cols)))
    print("  ".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        print("  ".join(_pad(i, str(r[i])) for i in range(cols)))


def _load_quality_rows(round_dir: Path) -> List[Dict[str, Any]]:
    return json.loads((round_dir / "cif_quality.json").read_text(encoding="utf-8"))


def _pass_ratio_for_bool_key(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Dict[str, int]]:
    t = 0
    f = 0
    u = 0
    for r in rows:
        v = r.get(key)
        if v is True:
            t += 1
        elif v is False:
            f += 1
        else:
            u += 1
    total = t + f + u
    known = t + f
    ratio_known = _safe_ratio(t, known) if known else None
    return ratio_known, {"true": t, "false": f, "unknown": u, "total": total, "known_total": known}


def _bond_lengths_reasonable_ratio(rows: List[Dict[str, Any]], cutoff: float) -> Tuple[Optional[float], Dict[str, int]]:
    ok = 0
    bad = 0
    u = 0
    for r in rows:
        v = r.get("bond_length_score")
        if isinstance(v, (int, float)):
            if float(v) >= float(cutoff):
                ok += 1
            else:
                bad += 1
        else:
            u += 1
    total = ok + bad + u
    known = ok + bad
    ratio_known = _safe_ratio(ok, known) if known else None
    return ratio_known, {"ok": ok, "bad": bad, "unknown": u, "total": total, "known_total": known}


def _round_quality_metrics(round_dir: Path, bond_cutoff: float) -> Dict[str, Any]:
    rows = _load_quality_rows(round_dir)
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r.get("validation_ok") is True)
    n_fail = sum(1 for r in rows if r.get("validation_ok") is False)

    sg_ratio, sg_counts = _pass_ratio_for_bool_key(rows, "space_group_ok")
    fo_ratio, fo_counts = _pass_ratio_for_bool_key(rows, "formula_ok")
    am_ratio, am_counts = _pass_ratio_for_bool_key(rows, "atom_site_multiplicity_ok")
    bl_ratio, bl_counts = _bond_lengths_reasonable_ratio(rows, cutoff=bond_cutoff)

    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "validation_pass_ratio": _safe_ratio(n_pass, n_total),
        "space_group_ok_pass_ratio": sg_ratio,
        "bond_lengths_reasonable_pass_ratio": bl_ratio,
        "formula_ok_pass_ratio": fo_ratio,
        "atom_site_multiplicity_ok_pass_ratio": am_ratio,
        "space_group_ok_counts": sg_counts,
        "bond_lengths_reasonable_counts": bl_counts,
        "formula_ok_counts": fo_counts,
        "atom_site_multiplicity_ok_counts": am_counts,
    }


def _alignn_means(scores_csv: Path) -> Dict[str, Any]:
    vals_fe: List[float] = []
    vals_bg: List[float] = []
    err_counts: Dict[str, int] = {}
    with scores_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            alignn_ok = row.get("alignn_ok")
            is_ok = alignn_ok in ("True", "true", "TRUE", "1", "yes", True)
            if is_ok:
                try:
                    fe = float(row.get("formation_energy"))
                    bg = float(row.get("bandgap"))
                except Exception:
                    continue
                vals_fe.append(fe)
                vals_bg.append(bg)
                continue

            err_s = row.get("errors")
            if not err_s:
                continue
            try:
                payload = json.loads(err_s)
            except Exception:
                continue
            alignn_errors = payload.get("alignn_errors")
            if not isinstance(alignn_errors, dict):
                continue
            for prop, msg in alignn_errors.items():
                if not msg:
                    continue
                key = f"{prop}: {msg}"
                err_counts[key] = err_counts.get(key, 0) + 1
    out = {}
    if vals_fe:
        out["formation_energy_mean"] = sum(vals_fe) / len(vals_fe)
    if vals_bg:
        out["bandgap_mean"] = sum(vals_bg) / len(vals_bg)
    if err_counts and not (vals_fe or vals_bg):
        top_err = max(err_counts.items(), key=lambda kv: kv[1])[0]
        out["error_top"] = top_err
        out["error_total"] = sum(err_counts.values())
    return out


def _load_qwen_output(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed = data.get("parsed") or {}
    return {
        "analysis": parsed.get("analysis"),
        "next_prompt_lines": parsed.get("next_prompt_lines") or [],
    }


def _load_prompt_sanitize_audit(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data.get("sanitizer_audit") or {}
    key_changes = s.get("key_changes") or []
    return {
        "fallback_used": s.get("fallback_used"),
        "fallback_reason": s.get("fallback_reason"),
        "key_changes": key_changes,
        "key_changes_count": len(key_changes) if isinstance(key_changes, list) else 0,
    }


def _load_final_report(out_dir: Path) -> Dict[str, Any]:
    fp = out_dir / "final_report.json"
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class RoundReport:
    idx: int
    metrics: Dict[str, Any]
    qwen_analysis: Optional[str]
    qwen_next: List[str]
    alignn_means: Dict[str, Any]
    sanitizer: Dict[str, Any]


@dataclass
class ExperimentReport:
    prompt_path: Path
    out_dir: Path
    rounds: List[RoundReport]
    final_report: Dict[str, Any]

    def render_text_report(self) -> str:
        lines: List[str] = []
        lines.append(f"== Experiment: {self.out_dir.name}")
        lines.append("")
        lines.append("Initial prompt:")
        lines.append(self.prompt_path.read_text(encoding="utf-8").rstrip())
        lines.append("")
        def _fmt_ratio(v: Any) -> str:
            return f"{float(v):.3f}" if isinstance(v, (int, float)) else "NA"

        def _fmt_counts(d: Any, ok_key: str) -> str:
            if not isinstance(d, dict):
                return ""
            ok = d.get(ok_key)
            known_total = d.get("known_total")
            total = d.get("total")
            unk = d.get("unknown")
            if not isinstance(ok, int):
                return ""
            den = known_total if isinstance(known_total, int) else total
            if not isinstance(den, int):
                return ""
            extra = []
            if isinstance(unk, int) and unk:
                extra.append(f"unknown={unk}")
            if isinstance(total, int) and total != den:
                extra.append(f"total={total}")
            extra_s = (", " + ", ".join(extra)) if extra else ""
            return f" ({ok}/{den}{extra_s})"

        def _append_kv(label: str, value: str, label_width: int = 28) -> None:
            lines.append(f"  {label:<{label_width}} {value}")

        term_cols = shutil.get_terminal_size((160, 20)).columns
        wrap_width = max(60, min(160, term_cols - 6))

        for rr in self.rounds:
            m = rr.metrics
            am = rr.alignn_means
            lines.append(f"[Round {rr.idx:02d}]")

            _append_kv("validation_pass_ratio", _fmt_ratio(m.get("validation_pass_ratio")))
            _append_kv(
                "space_group_ok",
                _fmt_ratio(m.get("space_group_ok_pass_ratio"))
                + _fmt_counts(m.get("space_group_ok_counts"), "true"),
            )
            _append_kv(
                "bond_lengths_reasonable",
                _fmt_ratio(m.get("bond_lengths_reasonable_pass_ratio"))
                + _fmt_counts(m.get("bond_lengths_reasonable_counts"), "ok"),
            )
            _append_kv(
                "formula_ok",
                _fmt_ratio(m.get("formula_ok_pass_ratio")) + _fmt_counts(m.get("formula_ok_counts"), "true"),
            )
            _append_kv(
                "atom_site_multiplicity_ok",
                _fmt_ratio(m.get("atom_site_multiplicity_ok_pass_ratio"))
                + _fmt_counts(m.get("atom_site_multiplicity_ok_counts"), "true"),
            )

            if am:
                _append_kv("ALIGNN mean formation_energy", _fmt_float(am.get("formation_energy_mean"), 4))
                _append_kv("ALIGNN mean bandgap", _fmt_float(am.get("bandgap_mean"), 4))
                if am.get("error_top") and not (am.get("formation_energy_mean") or am.get("bandgap_mean")):
                    err = str(am.get("error_top"))
                    err_wrapped = textwrap.fill(
                        err,
                        width=wrap_width,
                        subsequent_indent="    ",
                        initial_indent="    ",
                    )
                    lines.append("  ALIGNN error_top:")
                    lines.extend(err_wrapped.splitlines())

            lines.append("  Qwen next_prompt_lines:")
            if rr.qwen_next:
                for s in rr.qwen_next[:50]:
                    lines.append(f"    - {s}")
            else:
                lines.append("    - (none)")

            lines.append("  Qwen analysis:")
            analysis = rr.qwen_analysis or ""
            if analysis.strip():
                wrapped = textwrap.fill(analysis, width=wrap_width, subsequent_indent="    ", initial_indent="    ")
                lines.extend(wrapped.splitlines())
            else:
                lines.append("    (none)")

            if rr.sanitizer:
                n_changes = rr.sanitizer.get("key_changes_count")
                fallback_used = rr.sanitizer.get("fallback_used")
                fallback_reason = rr.sanitizer.get("fallback_reason")
                change_keys: List[str] = []
                key_changes = rr.sanitizer.get("key_changes")
                if isinstance(key_changes, list):
                    for item in key_changes:
                        if isinstance(item, dict) and item.get("key"):
                            change_keys.append(str(item["key"]))
                extras = []
                if isinstance(change_keys, list) and change_keys:
                    extras.append(", ".join(change_keys[:6]))
                    if len(change_keys) > 6:
                        extras.append("…")
                extra_s = f" ({' '.join(extras)})" if extras else ""
                n_changes_s = str(n_changes) if isinstance(n_changes, int) else "NA"
                _append_kv("Sanitizer edits applied", f"{n_changes_s}{extra_s}")
                if fallback_used:
                    _append_kv("Sanitizer fallback_used", "True")
                    if isinstance(fallback_reason, str) and fallback_reason.strip():
                        wrapped = textwrap.fill(
                            fallback_reason,
                            width=wrap_width,
                            subsequent_indent="    ",
                            initial_indent="    ",
                        )
                        lines.append("  Sanitizer fallback_reason:")
                        lines.extend(wrapped.splitlines())

            lines.append("")

        avg = self.averaged_summary()
        lines.append("Per-experiment averages (over 3 rounds):")
        key_order = [
            "validation_pass_ratio",
            "space_group_ok_pass_ratio",
            "bond_lengths_reasonable_pass_ratio",
            "formula_ok_pass_ratio",
            "atom_site_multiplicity_ok_pass_ratio",
            "formation_energy_mean_over_rounds",
            "bandgap_mean_over_rounds",
            "top_structures_formation_energy_mean",
            "top_structures_bandgap_mean",
        ]

        key_width = 34
        if avg:
            key_width = min(44, max(34, max(len(k) for k in avg.keys())))

        def _append_avg_line(k: str) -> None:
            v = avg.get(k)
            decimals = 4 if ("formation_energy" in k or "bandgap" in k) else 3
            lines.append(f"  {_truncate(k, key_width):<{key_width}} {_fmt_float(v, decimals)}")

        for k in key_order:
            if k in avg:
                _append_avg_line(k)
        for k in sorted(set(avg.keys()) - set(key_order)):
            _append_avg_line(k)

        return "\n".join(lines).rstrip() + "\n"

    def per_round_summary(self) -> None:
        print(self.render_text_report())

    def write_reports(self) -> None:
        (self.out_dir / "report.txt").write_text(self.render_text_report(), encoding="utf-8")
        (self.out_dir / "report.json").write_text(
            json.dumps(
                {
                    "experiment": self.out_dir.name,
                    "initial_prompt": self.prompt_path.read_text(encoding="utf-8"),
                    "rounds": [
                        {
                            "round": rr.idx,
                            "quality_metrics": rr.metrics,
                            "alignn_means": rr.alignn_means,
                            "qwen_analysis": rr.qwen_analysis,
                            "qwen_next_prompt_lines": rr.qwen_next,
                            "sanitizer": rr.sanitizer,
                        }
                        for rr in self.rounds
                    ],
                    "averages": self.averaged_summary(),
                    "final_report": self.final_report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_csv_reports()

    def _load_round_summary(self, round_idx: int) -> Dict[str, Any]:
        fp = self.out_dir / f"round_{round_idx:02d}" / "summary.json"
        return json.loads(fp.read_text(encoding="utf-8"))

    def _write_csv_reports(self) -> None:
        # Per-round table
        rounds_csv = self.out_dir / "rounds.csv"
        with rounds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "validation_pass_ratio",
                    "space_group_ok_pass_ratio",
                    "bond_lengths_reasonable_pass_ratio",
                    "formula_ok_pass_ratio",
                    "atom_site_multiplicity_ok_pass_ratio",
                    "alignn_mean_formation_energy",
                    "alignn_mean_bandgap",
                    "qwen_key_changes_count",
                    "qwen_analysis",
                    "qwen_next_prompt_lines",
                ],
            )
            w.writeheader()
            for rr in self.rounds:
                m = rr.metrics
                w.writerow(
                    {
                        "round": rr.idx,
                        "validation_pass_ratio": m.get("validation_pass_ratio"),
                        "space_group_ok_pass_ratio": m.get("space_group_ok_pass_ratio"),
                        "bond_lengths_reasonable_pass_ratio": m.get("bond_lengths_reasonable_pass_ratio"),
                        "formula_ok_pass_ratio": m.get("formula_ok_pass_ratio"),
                        "atom_site_multiplicity_ok_pass_ratio": m.get("atom_site_multiplicity_ok_pass_ratio"),
                        "alignn_mean_formation_energy": rr.alignn_means.get("formation_energy_mean"),
                        "alignn_mean_bandgap": rr.alignn_means.get("bandgap_mean"),
                        "qwen_key_changes_count": rr.sanitizer.get("key_changes_count"),
                        "qwen_analysis": rr.qwen_analysis,
                        "qwen_next_prompt_lines": " | ".join(str(x) for x in (rr.qwen_next or [])),
                    }
                )

        # Per-experiment averages
        summary_csv = self.out_dir / "summary.csv"
        avg = self.averaged_summary()
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "value"])
            w.writeheader()
            for k in sorted(avg.keys()):
                w.writerow({"metric": k, "value": avg[k]})

        # Top structures per round (from round_xx/summary.json)
        top_csv = self.out_dir / "top_structures.csv"
        with top_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "rank",
                    "sample_id",
                    "validation_ok",
                    "formation_energy",
                    "bandgap",
                    "score_value",
                    "validation_reasons",
                ],
            )
            w.writeheader()
            for rr in self.rounds:
                try:
                    summary = self._load_round_summary(rr.idx)
                except Exception:
                    continue
                top = summary.get("top_structures") or []
                if not isinstance(top, list):
                    continue
                for rank, item in enumerate(top, start=1):
                    if not isinstance(item, dict):
                        continue
                    props = item.get("properties") or {}
                    if not isinstance(props, dict):
                        props = {}
                    w.writerow(
                        {
                            "round": rr.idx,
                            "rank": rank,
                            "sample_id": item.get("sample_id"),
                            "validation_ok": item.get("validation_ok"),
                            "formation_energy": props.get("formation_energy"),
                            "bandgap": props.get("bandgap"),
                            "score_value": item.get("score_value"),
                            "validation_reasons": json.dumps(item.get("validation_reasons") or [], ensure_ascii=False),
                        }
                    )

        # Best overall across rounds (from final_report.json)
        best_csv = self.out_dir / "best_overall.csv"
        best = self.final_report.get("best_overall") if isinstance(self.final_report, dict) else None
        with best_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "round",
                    "sample_id",
                    "validation_ok",
                    "formation_energy",
                    "bandgap",
                    "score_value",
                    "cif_path",
                ],
            )
            w.writeheader()
            if isinstance(best, list):
                for rank, item in enumerate(best, start=1):
                    if not isinstance(item, dict):
                        continue
                    props = item.get("properties") or {}
                    if not isinstance(props, dict):
                        props = {}
                    cif_path = item.get("cif_path")
                    round_guess = item.get("round")
                    if round_guess is None:
                        round_guess = _infer_round_from_path(cif_path)
                    w.writerow(
                        {
                            "rank": rank,
                            "round": round_guess,
                            "sample_id": item.get("sample_id"),
                            "validation_ok": item.get("validation_ok"),
                            "formation_energy": props.get("formation_energy"),
                            "bandgap": props.get("bandgap"),
                            "score_value": item.get("score_value"),
                            "cif_path": cif_path,
                        }
                    )

        # Concise top-structures summary (user-facing): only mean FE/BG for best_overall.
        # (Detailed rows are still available in top_structures.csv and best_overall.csv.)
        n_best_total = len(best) if isinstance(best, list) else 0
        n_best_valid = 0
        if isinstance(best, list):
            for item in best:
                if not isinstance(item, dict):
                    continue
                if item.get("validation_ok") in (True, "True", "true"):
                    n_best_valid += 1
        top_lines: List[str] = []
        top_lines.append(f"Top structures summary: {self.out_dir.name}")
        top_lines.append("")
        top_lines.append(f"best_overall_count: {n_best_total} (valid={n_best_valid})")
        top_lines.append(f"mean formation_energy: {_fmt_float(avg.get('top_structures_formation_energy_mean'), 4)}")
        top_lines.append(f"mean bandgap: {_fmt_float(avg.get('top_structures_bandgap_mean'), 4)}")
        top_lines.append("")
        top_lines.append("Details: top_structures.csv, best_overall.csv")
        (self.out_dir / "top_structures_table.txt").write_text("\n".join(top_lines).rstrip() + "\n", encoding="utf-8")

    def averaged_summary(self) -> Dict[str, float]:
        keys = [
            "validation_pass_ratio",
            "space_group_ok_pass_ratio",
            "bond_lengths_reasonable_pass_ratio",
            "formula_ok_pass_ratio",
            "atom_site_multiplicity_ok_pass_ratio",
        ]
        acc = {k: 0.0 for k in keys}
        cnt = {k: 0 for k in keys}
        for rr in self.rounds:
            for k in keys:
                v = rr.metrics.get(k)
                if isinstance(v, (int, float)):
                    acc[k] += v
                    cnt[k] += 1
        out = {}
        for k in keys:
            if cnt[k]:
                out[k] = acc[k] / cnt[k]
        fe_means = [rr.alignn_means.get("formation_energy_mean") for rr in self.rounds if "formation_energy_mean" in rr.alignn_means]
        bg_means = [rr.alignn_means.get("bandgap_mean") for rr in self.rounds if "bandgap_mean" in rr.alignn_means]
        if fe_means:
            out["formation_energy_mean_over_rounds"] = sum(fe_means) / len(fe_means)
        if bg_means:
            out["bandgap_mean_over_rounds"] = sum(bg_means) / len(bg_means)

        # "Top structures" (best_overall) means: summarize only valid, numeric properties.
        best = self.final_report.get("best_overall") if isinstance(self.final_report, dict) else None
        vals_fe: List[float] = []
        vals_bg: List[float] = []
        if isinstance(best, list):
            for item in best:
                if not isinstance(item, dict):
                    continue
                if item.get("validation_ok") not in (True, "True", "true"):
                    continue
                props = item.get("properties") or {}
                if not isinstance(props, dict):
                    continue
                fe = props.get("formation_energy")
                bg = props.get("bandgap")
                if isinstance(fe, (int, float)):
                    vals_fe.append(float(fe))
                if isinstance(bg, (int, float)):
                    vals_bg.append(float(bg))
        if vals_fe:
            out["top_structures_formation_energy_mean"] = sum(vals_fe) / len(vals_fe)
        if vals_bg:
            out["top_structures_bandgap_mean"] = sum(vals_bg) / len(vals_bg)
        return out


def _run_small_experiment(prompt_path: Path, out_dir: Path, args: argparse.Namespace, idx: int) -> ExperimentReport:
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "prompt_optimization_loop.py"),
        "--model-dir",
        str(Path(args.model_dir).resolve()),
        "--initial-prompt-file",
        str(prompt_path.resolve()),
        "--out-dir",
        str(out_dir.resolve()),
        "--rounds",
        "3",
        "--samples-per-round",
        str(args.samples_per_round),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed + idx * 1000),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--alignn-host",
        args.alignn_host,
        "--alignn-port",
        str(args.alignn_port),
        "--alignn-properties",
        "formation_energy",
        "bandgap",
        "--score-property",
        "bandgap",
        "--score-goal",
        "max",
        "--top-structures",
        str(args.top_structures),
        "--final-top-k",
        str(args.final_top_k),
        "--qwen-api-base",
        args.qwen_api_base,
        "--qwen-model",
        args.qwen_model,
        "--qwen-temperature",
        str(args.qwen_temperature),
        "--qwen-max-tokens",
        str(args.qwen_max_tokens),
    ]
    if args.no_validation_check_composition:
        cmd.append("--no-validation-check-composition")
    if args.validation_bond_cutoff is not None:
        cmd.extend(["--validation-bond-cutoff", str(args.validation_bond_cutoff)])
    env = os.environ.copy()
    env["CRYSTALLM_PROMPT_EDIT_SCOPE"] = "restricted"
    env["CRYSTALLM_COMPOSITION_POLICY"] = "locked"
    _run_cmd(cmd, env)

    rounds: List[RoundReport] = []
    for r in range(1, 4):
        rdir = out_dir / f"round_{r:02d}"
        cq = _round_quality_metrics(rdir, bond_cutoff=float(args.validation_bond_cutoff))
        q = _load_qwen_output(rdir / "qwen_output.json")
        am = _alignn_means(rdir / "scores.csv")
        sanitizer = _load_prompt_sanitize_audit(rdir / "prompt_sanitize_audit_post_llm.json")
        rounds.append(
            RoundReport(
                idx=r,
                metrics=cq,
                qwen_analysis=q.get("analysis"),
                qwen_next=q.get("next_prompt_lines", []),
                alignn_means=am,
                sanitizer=sanitizer,
            )
        )
    final_report = _load_final_report(out_dir)
    return ExperimentReport(prompt_path=prompt_path, out_dir=out_dir, rounds=rounds, final_report=final_report)


def _summarize_big(exp_reports: List[ExperimentReport]) -> None:
    print("\n==== Big Experiment Summary ====")
    headers = ["exp", "bg", "fe", "tbg", "tfe", "pass", "sg_ok", "bond_ok", "formula_ok", "mult_ok", "qwenΔ"]
    rows: List[List[str]] = []
    for rep in exp_reports:
        avg = rep.averaged_summary()
        qwen_key_changes_total = sum(int(r.sanitizer.get("key_changes_count") or 0) for r in rep.rounds)
        rows.append(
            [
                rep.out_dir.name,
                _fmt_float(avg.get("bandgap_mean_over_rounds"), 4),
                _fmt_float(avg.get("formation_energy_mean_over_rounds"), 4),
                _fmt_float(avg.get("top_structures_bandgap_mean"), 4),
                _fmt_float(avg.get("top_structures_formation_energy_mean"), 4),
                _fmt_float(avg.get("validation_pass_ratio"), 3),
                _fmt_float(avg.get("space_group_ok_pass_ratio"), 3),
                _fmt_float(avg.get("bond_lengths_reasonable_pass_ratio"), 3),
                _fmt_float(avg.get("formula_ok_pass_ratio"), 3),
                _fmt_float(avg.get("atom_site_multiplicity_ok_pass_ratio"), 3),
                str(qwen_key_changes_total),
            ]
        )

    term_cols = shutil.get_terminal_size((160, 20)).columns
    # exp column can be long; cap it so the table won't wrap easily
    max_exp = max(24, min(60, term_cols - 10 * 10))  # rough heuristic
    max_widths = [max_exp, 8, 8, 8, 8, 6, 6, 7, 9, 7, 6]
    _print_aligned_table(headers, rows, max_widths=max_widths)

    all_bg = []
    all_fe = []
    for rep in exp_reports:
        avg = rep.averaged_summary()
        if "bandgap_mean_over_rounds" in avg:
            all_bg.append(avg["bandgap_mean_over_rounds"])
        if "formation_energy_mean_over_rounds" in avg:
            all_fe.append(avg["formation_energy_mean_over_rounds"])
    if all_bg:
        print(f"\nOverall mean bandgap (over prompts, averaged per prompt over rounds): {sum(all_bg)/len(all_bg):.4f}")
    if all_fe:
        print(f"Overall mean formation_energy (over prompts, averaged per prompt over rounds): {sum(all_fe)/len(all_fe):.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch reproduction test for CrystaLLM using CIF prompts.")
    p.add_argument("--data-file", default="/root/autodl-tmp/model/CrystaLLM/resources/cifs_v1_test.pkl.gz")
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=20230610)
    p.add_argument("--samples-per-round", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--validation-bond-cutoff", type=float, default=1.0)
    p.add_argument("--no-validation-check-composition", action="store_true", help="Disable composition-related checks.")
    p.add_argument("--alignn-host", default="127.0.0.1")
    p.add_argument("--alignn-port", type=int, default=5555)
    p.add_argument("--top-structures", type=int, default=3)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument("--qwen-api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--qwen-model", default="Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--qwen-temperature", type=float, default=0.2)
    p.add_argument("--qwen-max-tokens", type=int, default=256)
    p.add_argument("--model-dir", default="/root/autodl-tmp/model/CrystaLLM/crystallm_v1_small")
    p.add_argument("--experiments-root", default="/root/autodl-tmp/model/CrystaLLM/experiments")
    p.add_argument("--batch-name", default=None, help="Optional name for big experiment; default uses timestamp.")
    p.add_argument(
        "--analyze-dir",
        default=None,
        help="Analyze an existing big experiment directory (generate reports) without running sampling.",
    )
    return p.parse_args()


def main() -> None:
    _require_supported_python()
    args = parse_args()
    if args.analyze_dir:
        big_dir = Path(args.analyze_dir).expanduser().resolve()
        _analyze_existing_big_dir(big_dir, args)
        return
    data_path = Path(args.data_file).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Download with: python bin/download.py cifs_v1_test.pkl.gz --out {data_path.parent}")

    data = _load_cifs(data_path)
    prompts = _pick_prompts(data, args.num_prompts, args.seed)

    ts = args.batch_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    big_dir = Path(args.experiments_root).expanduser() / f"reproduction_batch_{ts}"
    prompt_dir = big_dir / "initial_prompts"
    prompt_paths = _write_prompts(prompts, prompt_dir)

    print(f"Big experiment dir: {big_dir}")
    print(f"Using {len(prompt_paths)} prompts from {data_path}")

    exp_reports: List[ExperimentReport] = []
    for idx, prompt_path in enumerate(prompt_paths):
        out_dir = big_dir / prompt_path.stem
        rep = _run_small_experiment(prompt_path, out_dir, args, idx)
        rep.per_round_summary()
        rep.write_reports()
        exp_reports.append(rep)

    _write_big_reports(big_dir, exp_reports)
    _summarize_big(exp_reports)


def _analyze_existing_big_dir(big_dir: Path, args: argparse.Namespace) -> None:
    prompt_dir = big_dir / "initial_prompts"
    if not prompt_dir.exists():
        raise SystemExit(f"--analyze-dir expected {prompt_dir} to exist")

    exp_reports: List[ExperimentReport] = []
    for prompt_path in sorted(prompt_dir.glob("prompt_*.txt")):
        out_dir = big_dir / prompt_path.stem
        if not out_dir.exists():
            continue
        rounds: List[RoundReport] = []
        for r in range(1, 4):
            rdir = out_dir / f"round_{r:02d}"
            if not rdir.exists():
                continue
            cq = _round_quality_metrics(rdir, bond_cutoff=float(args.validation_bond_cutoff))
            q = _load_qwen_output(rdir / "qwen_output.json")
            am = _alignn_means(rdir / "scores.csv")
            sanitizer = _load_prompt_sanitize_audit(rdir / "prompt_sanitize_audit_post_llm.json")
            rounds.append(
                RoundReport(
                    idx=r,
                    metrics=cq,
                    qwen_analysis=q.get("analysis"),
                    qwen_next=q.get("next_prompt_lines", []),
                    alignn_means=am,
                    sanitizer=sanitizer,
                )
            )
        rep = ExperimentReport(
            prompt_path=prompt_path,
            out_dir=out_dir,
            rounds=rounds,
            final_report=_load_final_report(out_dir),
        )
        rep.write_reports()
        exp_reports.append(rep)
    _write_big_reports(big_dir, exp_reports)
    _summarize_big(exp_reports)


def _write_big_reports(big_dir: Path, exp_reports: List[ExperimentReport]) -> None:
    big_dir.mkdir(parents=True, exist_ok=True)
    # Machine-readable
    payload = []
    for rep in exp_reports:
        payload.append(
            {
                "experiment": rep.out_dir.name,
                "averages": rep.averaged_summary(),
                "qwen_key_changes_total": sum(int(r.sanitizer.get("key_changes_count") or 0) for r in rep.rounds),
            }
        )
    (big_dir / "batch_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable (tab + csv + fixed-width)
    lines: List[str] = []
    lines.append(f"Big experiment: {big_dir.name}")
    lines.append("")
    lines.append("Per-small-experiment averages (over 3 rounds):")
    lines.append("")
    header = [
        "exp",
        "avg_bandgap",
        "avg_formation_energy",
        "avg_top_bandgap",
        "avg_top_formation_energy",
        "avg_pass_ratio",
        "avg_space_group_ok",
        "avg_bond_lengths_reasonable",
        "avg_formula_ok",
        "avg_atom_site_multiplicity_ok",
        "qwen_key_changes_total",
    ]
    rows_tsv = ["\t".join(header)]
    rows_csv: List[List[str]] = [header]
    lines.append("\t".join(header))
    for rep in exp_reports:
        avg = rep.averaged_summary()
        qwen_key_changes_total = sum(int(r.sanitizer.get("key_changes_count") or 0) for r in rep.rounds)
        row = [
            rep.out_dir.name,
            f"{avg.get('bandgap_mean_over_rounds'):.4f}" if "bandgap_mean_over_rounds" in avg else "NA",
            f"{avg.get('formation_energy_mean_over_rounds'):.4f}" if "formation_energy_mean_over_rounds" in avg else "NA",
            f"{avg.get('top_structures_bandgap_mean'):.4f}" if "top_structures_bandgap_mean" in avg else "NA",
            f"{avg.get('top_structures_formation_energy_mean'):.4f}" if "top_structures_formation_energy_mean" in avg else "NA",
            f"{avg.get('validation_pass_ratio'):.3f}" if "validation_pass_ratio" in avg else "NA",
            f"{avg.get('space_group_ok_pass_ratio'):.3f}" if "space_group_ok_pass_ratio" in avg else "NA",
            f"{avg.get('bond_lengths_reasonable_pass_ratio'):.3f}" if "bond_lengths_reasonable_pass_ratio" in avg else "NA",
            f"{avg.get('formula_ok_pass_ratio'):.3f}" if "formula_ok_pass_ratio" in avg else "NA",
            f"{avg.get('atom_site_multiplicity_ok_pass_ratio'):.3f}" if "atom_site_multiplicity_ok_pass_ratio" in avg else "NA",
            str(qwen_key_changes_total),
        ]
        lines.append("\t".join(row))
        rows_tsv.append("\t".join(row))
        rows_csv.append(row)

    (big_dir / "batch_report.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    (big_dir / "batch_report.tsv").write_text("\n".join(rows_tsv).rstrip() + "\n", encoding="utf-8")
    with (big_dir / "batch_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows_csv)

    # Extra: fixed-width table for easier reading in terminal/editors.
    table_headers = ["exp", "bg", "fe", "tbg", "tfe", "pass", "sg_ok", "bond_ok", "formula_ok", "mult_ok", "qwenΔ"]
    table_rows: List[List[str]] = []
    for rep in exp_reports:
        avg = rep.averaged_summary()
        qwen_key_changes_total = sum(int(r.sanitizer.get("key_changes_count") or 0) for r in rep.rounds)
        table_rows.append(
            [
                rep.out_dir.name,
                _fmt_float(avg.get("bandgap_mean_over_rounds"), 4),
                _fmt_float(avg.get("formation_energy_mean_over_rounds"), 4),
                _fmt_float(avg.get("top_structures_bandgap_mean"), 4),
                _fmt_float(avg.get("top_structures_formation_energy_mean"), 4),
                _fmt_float(avg.get("validation_pass_ratio"), 3),
                _fmt_float(avg.get("space_group_ok_pass_ratio"), 3),
                _fmt_float(avg.get("bond_lengths_reasonable_pass_ratio"), 3),
                _fmt_float(avg.get("formula_ok_pass_ratio"), 3),
                _fmt_float(avg.get("atom_site_multiplicity_ok_pass_ratio"), 3),
                str(qwen_key_changes_total),
            ]
        )
    max_widths = [60, 8, 8, 8, 8, 6, 6, 7, 9, 7, 6]
    # Write fixed-width table to file
    table_lines: List[str] = []
    widths = [len(h) for h in table_headers]
    for r in table_rows:
        for i in range(len(table_headers)):
            widths[i] = max(widths[i], len(str(r[i])))
    widths = [min(widths[i], max_widths[i]) for i in range(len(widths))]
    def _pad(i: int, s: str) -> str:
        s2 = _truncate(str(s), widths[i])
        if i == 0:
            return s2.ljust(widths[i])
        return s2.rjust(widths[i])
    table_lines.append("  ".join(_pad(i, table_headers[i]) for i in range(len(table_headers))))
    table_lines.append("  ".join("-" * widths[i] for i in range(len(table_headers))))
    for r in table_rows:
        table_lines.append("  ".join(_pad(i, r[i]) for i in range(len(table_headers))))
    (big_dir / "batch_report_table.txt").write_text("\n".join(table_lines).rstrip() + "\n", encoding="utf-8")

    # Aggregate top structures / best overall for quick inspection across experiments
    top_agg_csv = big_dir / "top_structures_all.csv"
    with top_agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "round",
                "rank",
                "sample_id",
                "validation_ok",
                "formation_energy",
                "bandgap",
                "score_value",
            ],
        )
        w.writeheader()
        for rep in exp_reports:
            for rr in rep.rounds:
                fp = rep.out_dir / f"round_{rr.idx:02d}" / "summary.json"
                if not fp.exists():
                    continue
                try:
                    summary = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                top = summary.get("top_structures") or []
                if not isinstance(top, list):
                    continue
                for rank, item in enumerate(top, start=1):
                    if not isinstance(item, dict):
                        continue
                    props = item.get("properties") or {}
                    if not isinstance(props, dict):
                        props = {}
                    w.writerow(
                        {
                            "experiment": rep.out_dir.name,
                            "round": rr.idx,
                            "rank": rank,
                            "sample_id": item.get("sample_id"),
                            "validation_ok": item.get("validation_ok"),
                            "formation_energy": props.get("formation_energy"),
                            "bandgap": props.get("bandgap"),
                            "score_value": item.get("score_value"),
                        }
                    )

    best_agg_csv = big_dir / "best_overall_all.csv"
    with best_agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "rank",
                "round",
                "sample_id",
                "validation_ok",
                "formation_energy",
                "bandgap",
                "score_value",
                "cif_path",
            ],
        )
        w.writeheader()
        for rep in exp_reports:
            best = rep.final_report.get("best_overall") if isinstance(rep.final_report, dict) else None
            if not isinstance(best, list):
                continue
            for rank, item in enumerate(best, start=1):
                if not isinstance(item, dict):
                    continue
                props = item.get("properties") or {}
                if not isinstance(props, dict):
                    props = {}
                cif_path = item.get("cif_path")
                round_guess = item.get("round")
                if round_guess is None:
                    round_guess = _infer_round_from_path(cif_path)
                w.writerow(
                    {
                        "experiment": rep.out_dir.name,
                        "rank": rank,
                        "round": round_guess,
                        "sample_id": item.get("sample_id"),
                        "validation_ok": item.get("validation_ok"),
                        "formation_energy": props.get("formation_energy"),
                        "bandgap": props.get("bandgap"),
                        "score_value": item.get("score_value"),
                        "cif_path": cif_path,
                    }
                )

    # Concise best_overall summary per experiment (mean FE/BG only).
    table_hdr = ["experiment", "count", "valid", "mean_fe", "mean_bg"]
    table_rows2: List[List[str]] = []
    for rep in exp_reports:
        best = rep.final_report.get("best_overall") if isinstance(rep.final_report, dict) else None
        if not isinstance(best, list) or not best:
            continue
        n_total = len(best)
        n_valid = 0
        for item in best:
            if isinstance(item, dict) and item.get("validation_ok") in (True, "True", "true"):
                n_valid += 1
        avg = rep.averaged_summary()
        table_rows2.append(
            [
                rep.out_dir.name,
                str(n_total),
                str(n_valid),
                _fmt_float(avg.get("top_structures_formation_energy_mean"), 4),
                _fmt_float(avg.get("top_structures_bandgap_mean"), 4),
            ]
        )
    if table_rows2:
        widths = [len(h) for h in table_hdr]
        for r in table_rows2:
            for i in range(len(table_hdr)):
                widths[i] = max(widths[i], len(str(r[i])))
        widths = [min(widths[i], m) for i, m in enumerate([60, 6, 6, 12, 12])]

        def _p(i: int, s: str) -> str:
            s2 = _truncate(str(s), widths[i])
            if i == 0:
                return s2.ljust(widths[i])
            return s2.rjust(widths[i])

        out_lines = []
        out_lines.append("  ".join(_p(i, table_hdr[i]) for i in range(len(table_hdr))))
        out_lines.append("  ".join("-" * widths[i] for i in range(len(table_hdr))))
        for r in table_rows2:
            out_lines.append("  ".join(_p(i, r[i]) for i in range(len(table_hdr))))
        (big_dir / "best_overall_table.txt").write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
