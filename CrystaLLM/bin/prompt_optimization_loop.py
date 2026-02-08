#!/usr/bin/env python3
"""
CrystaLLM closed-loop prompt optimization:

Per round:
  1) Generate N CIFs via `bin/sample.py` given current prompt.txt
  2) Clean + validate each CIF (crystallm.cif_cleaning.clean_and_validate_cif)
  3) Score via ALIGNN ZMQ server (resources/alignn_zmq_server_multi.py)
  4) Build evaluator_summary.json (round-level compressed stats + top structures)
  5) Call optimizer LLM (bin/qwen_client.py) to propose next prompt lines

Outputs (under out_dir/round_XX/):
  - prompt.txt
  - prompt_for_sampling.txt
  - cifs/*.cif
  - cif_quality.json / cif_quality.csv / cif_quality_summary.txt
  - scores.csv
  - summary.json
  - evaluator_summary.json
  - qwen_output.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qwen_client import QwenClient, QwenConfig

# Ensure `import crystallm` works when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crystallm.cif_cleaning import clean_and_validate_cif  # noqa: E402
from crystallm.prompt_sanitizer import sanitize_prompt_lines_with_audit  # noqa: E402
from alignn_client import score_cif_via_alignn  # noqa: E402
from postprocess import postprocess as postprocess_cif  # noqa: E402


@dataclass
class ScoredSample:
    sample_id: str
    cif_path: str
    validation_ok: bool
    validation_reasons: List[str]
    bond_length_score: Optional[float]
    formula_ok: Optional[bool]
    atom_site_multiplicity_ok: Optional[bool]
    space_group_ok: Optional[bool]
    alignn_ok: bool
    properties: Dict[str, Optional[float]]
    errors: Dict[str, Any]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_and_sanitize_prompt(
    candidate_lines: List[str],
    fallback_lines: List[str],
    *,
    round_dir: Path,
    stage: str,
    mode: str = "on",
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Sanitize prompt lines and write an audit record.

    This prevents LLM prompt injection (atom-site tables / cell params / loops) from
    leaking into `prompt.txt` and destabilizing sampling/cleaning.
    """
    cand = [str(x).rstrip() for x in (candidate_lines or []) if str(x).strip()]
    fb = [str(x).rstrip() for x in (fallback_lines or []) if str(x).strip()]

    if mode == "off":
        # No sanitizer, no audit.
        return cand, {}

    sanitized, sanitizer_audit = sanitize_prompt_lines_with_audit(cand, fb)

    # Extra guard: ensure no known-dangerous prefixes survived.
    banned_prefixes = ("loop_", "_atom_site_", "_cell_length_", "_cell_angle_", "_cell_volume")
    survivors = [ln for ln in sanitized if any(ln.strip().startswith(b) for b in banned_prefixes)]
    if survivors:
        sanitized, sanitizer_audit = sanitize_prompt_lines_with_audit(fb, fb)

    # Module provenance (helps debug version skew / path issues)
    try:
        sanitizer_mod = inspect.getmodule(sanitize_prompt_lines_with_audit)
        sanitizer_file = getattr(sanitizer_mod, "__file__", None) if sanitizer_mod else None
        sanitizer_file_sha256 = _sha256_file(Path(sanitizer_file)) if sanitizer_file else None
    except Exception:  # noqa: BLE001
        sanitizer_file = None
        sanitizer_file_sha256 = None

    removed = [ln for ln in cand if ln not in sanitized]
    audit: Dict[str, Any] = {
        "stage": stage,
        "candidate_n": len(cand),
        "fallback_n": len(fb),
        "sanitized_n": len(sanitized),
        "removed_n": len(removed),
        "removed_preview": removed[:20],
        "sanitizer_audit": sanitizer_audit,
        "sanitizer_file": sanitizer_file,
        "sanitizer_file_sha256": sanitizer_file_sha256,
        "candidate_sha256": _sha256_text("\n".join(cand)),
        "sanitized_sha256": _sha256_text("\n".join(sanitized)),
        "fallback_sha256": _sha256_text("\n".join(fb)),
        "banned_survivors": survivors[:20],
    }
    _write_json(round_dir / f"prompt_sanitize_audit_{stage}.json", audit)
    if mode == "audit_only":
        # Do not let sanitizer output affect actual prompt lines used for sampling.
        return cand, audit
    return sanitized, audit


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CrystaLLM closed-loop prompt optimization runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, help="Directory containing CrystaLLM ckpt.pt")
    p.add_argument("--initial-prompt-file", required=True, help="Initial prompt.txt for round_01")
    p.add_argument("--out-dir", required=True, help="Experiment output directory")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--samples-per-round", type=int, default=8)

    # sampling
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument(
        "--prompt-sanitizer-mode",
        default="on",
        choices=["on", "audit_only", "off"],
        help=(
            "Control prompt sanitizer behavior. "
            "'on': sanitize + audit (default). "
            "'audit_only': audit but do NOT change prompt lines used for sampling. "
            "'off': disable sanitizer and audit."
        ),
    )
    p.add_argument(
        "--sampling-prompt-mode",
        default="filtered",
        choices=["filtered", "prompt_txt"],
        help=(
            "Which prompt content is fed into CrystaLLM sampling. "
            "'filtered' (default): prompt_for_sampling.txt keeps only lines starting with data_ or _. "
            "'prompt_txt': sample.py reads prompt.txt as-is."
        ),
    )

    # validation
    p.add_argument("--validation-bond-cutoff", type=float, default=1.0)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--validation-check-composition", action="store_true", help="Enable composition checks")
    g.add_argument("--no-validation-check-composition", action="store_true", help="Disable composition checks")

    # alignn
    p.add_argument("--alignn-host", default="127.0.0.1")
    p.add_argument("--alignn-port", type=int, default=5555)
    p.add_argument("--alignn-timeout-ms", type=int, default=10000)
    p.add_argument("--alignn-properties", nargs="+", default=["formation_energy", "bandgap"])

    # selection
    p.add_argument("--score-property", default="bandgap")
    p.add_argument("--score-goal", default="max", choices=["max", "min"])
    p.add_argument("--top-structures", type=int, default=3)
    p.add_argument("--final-top-k", type=int, default=5)
    p.add_argument(
        "--prefer-valid-structures",
        action="store_true",
        help="Prefer validation_ok=True when selecting (default true unless --no-prefer-valid-structures)",
    )
    p.add_argument(
        "--no-prefer-valid-structures",
        action="store_true",
        help="Allow selecting from validation_ok=False candidates as well",
    )

    # optimizer LLM (Qwen)
    p.add_argument("--qwen-api-base", dest="qwen_api_base", default=None)
    p.add_argument("--qwen-api-key", dest="qwen_api_key", default=None)
    p.add_argument("--qwen-model", dest="qwen_model", default=None)
    p.add_argument("--qwen-temperature", dest="qwen_temperature", type=float, default=None)
    p.add_argument("--qwen-top-p", dest="qwen_top_p", type=float, default=None)
    p.add_argument("--qwen-max-tokens", dest="qwen_max_tokens", type=int, default=None)
    p.add_argument("--qwen-timeout", dest="qwen_timeout", type=float, default=None)

    # Backward-compatible aliases
    p.add_argument("--llama-api-base", dest="qwen_api_base", default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-api-key", dest="qwen_api_key", default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-model", dest="qwen_model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-temperature", dest="qwen_temperature", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-top-p", dest="qwen_top_p", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-max-tokens", dest="qwen_max_tokens", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--llama-timeout", dest="qwen_timeout", type=float, default=None, help=argparse.SUPPRESS)

    # misc
    p.add_argument("--resume", action="store_true", help="Resume if round dirs already exist")
    pg = p.add_mutually_exclusive_group()
    pg.add_argument(
        "--print-round-summary",
        action="store_true",
        help="Print per-round cif_quality summary (key validation fields) to stdout",
    )
    pg.add_argument(
        "--no-print-round-summary",
        action="store_true",
        help="Disable printing per-round cif_quality summary to stdout",
    )

    qg = p.add_mutually_exclusive_group()
    qg.add_argument(
        "--print-qwen",
        action="store_true",
        help="Print Qwen ANALYSIS + proposed lines + sanitizer decision per round",
    )
    qg.add_argument(
        "--no-print-qwen",
        action="store_true",
        help="Disable printing Qwen suggestion/audit to stdout",
    )

    wg = p.add_mutually_exclusive_group()
    wg.add_argument(
        "--show-cif-warnings",
        action="store_true",
        help="Show noisy pymatgen CIF parsing warnings on stderr (may include CIF snippets)",
    )
    wg.add_argument(
        "--quiet-cif-warnings",
        action="store_true",
        help="Silence pymatgen CIF parsing warnings on stderr (default)",
    )

    # reporting / registry
    p.add_argument(
        "--experiments-table-out",
        default=None,
        help=(
            "Optional: after the run completes, write/update a single CSV table across all "
            "experiments under <experiments-root>. If omitted, no table is written."
        ),
    )
    p.add_argument(
        "--experiments-table-root",
        default=None,
        help="Optional: override the experiments root used by bin/experiments_table.py (default: parent of --out-dir).",
    )
    p.add_argument(
        "--experiments-rounds-table-out",
        default=None,
        help=(
            "Optional: after the run completes, write/update a single CSV table with one row per round across all "
            "experiments under <experiments-root>. If omitted, no table is written."
        ),
    )
    p.add_argument(
        "--experiments-rounds-table-root",
        default=None,
        help=(
            "Optional: override the experiments root used by bin/experiments_rounds_table.py "
            "(default: parent of --out-dir)."
        ),
    )
    return p.parse_args()


def _check_composition(args: argparse.Namespace) -> bool:
    if args.no_validation_check_composition:
        return False
    if args.validation_check_composition:
        return True
    return True


def _run_sample(
    model_dir: str,
    prompt_path: Path,
    out_cifs_dir: Path,
    samples: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    device: str,
    dtype: str,
) -> None:
    out_cifs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str((_ROOT / "bin" / "sample.py").resolve()),
        f"out_dir={model_dir}",
        f"start=FILE:{str(prompt_path)}",
        f"num_samples={samples}",
        f"temperature={temperature}",
        f"top_k={top_k}",
        f"max_new_tokens={max_new_tokens}",
        f"seed={seed}",
        f"device={device}",
        f"dtype={dtype}",
        "compile=False",
        "target=file",
    ]
    subprocess.run(cmd, cwd=str(out_cifs_dir), check=True)


def _collect_generated_cifs(out_cifs_dir: Path) -> List[Path]:
    return sorted(out_cifs_dir.glob("sample_*.cif"))


def _summarize_reasons(rows: List[Dict[str, Any]], top_n: int = 20) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    n_fail = 0
    for r in rows:
        if r.get("validation_ok") is True:
            continue
        if r.get("validation_ok") is False:
            n_fail += 1
        for reason in (r.get("validation_reasons") or []):
            if isinstance(reason, str) and reason.strip():
                counter[reason.strip()] += 1
    out: List[Dict[str, Any]] = []
    for k, v in counter.most_common(top_n):
        out.append({"reason": k, "count": v, "ratio": (v / n_fail) if n_fail else None})
    return out


def _write_cif_quality(
    round_dir: Path,
    quality_rows: List[Dict[str, Any]],
    bond_cutoff: float,
    check_composition: bool,
) -> None:
    _write_json(round_dir / "cif_quality.json", quality_rows)
    csv_path = round_dir / "cif_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "cif_path",
            "validation_ok",
            "validation_reasons",
            "bond_length_score",
            "bond_lengths_reasonable",
            "formula_ok",
            "formula_ok_relaxed",
            "atom_site_multiplicity_ok",
            "space_group_ok",
            "strict_valid",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in quality_rows:
            row = dict(r)
            row["validation_reasons"] = json.dumps(row.get("validation_reasons") or [], ensure_ascii=False)
            w.writerow({k: row.get(k) for k in fieldnames})

    # summary txt
    n_total = len(quality_rows)
    n_pass = sum(1 for r in quality_rows if r.get("validation_ok") is True)
    n_fail = sum(1 for r in quality_rows if r.get("validation_ok") is False)

    n_space_group_true = sum(1 for r in quality_rows if r.get("space_group_ok") is True)
    n_space_group_false = sum(1 for r in quality_rows if r.get("space_group_ok") is False)
    n_space_group_unknown = sum(1 for r in quality_rows if r.get("space_group_ok") is None)

    def _bond_ok(r: Dict[str, Any]) -> bool:
        v = r.get("bond_lengths_reasonable")
        if isinstance(v, bool):
            return bool(v)
        score = r.get("bond_length_score")
        return isinstance(score, (int, float)) and float(score) >= bond_cutoff

    n_bond_ok = sum(1 for r in quality_rows if _bond_ok(r))

    def _true(r: Dict[str, Any], key: str) -> bool:
        return r.get(key) is True

    def _ratio(true_n: int, total_n: int) -> str:
        return f"{true_n}/{total_n}" if total_n else "0/0"

    def _ratio_line(name: str, true_n: int, total_n: int) -> str:
        if not total_n:
            return f"{name}: 0/0 (0.000)"
        return f"{name}: {true_n}/{total_n} ({true_n / total_n:.3f})"

    top = _summarize_reasons(quality_rows, top_n=30)
    lines = [
        f"n_total: {n_total}",
        f"n_pass: {n_pass}",
        f"n_fail: {n_fail}",
        f"pass_ratio: {n_pass / n_total:.4f}" if n_total else "pass_ratio: null",
        "",
        _ratio_line("validation_ok", n_pass, n_total),
        _ratio_line("space_group_ok", n_space_group_true, n_total),
        _ratio_line("bond_lengths_reasonable", n_bond_ok, n_total),
    ]
    n_formula_ok = sum(1 for r in quality_rows if _true(r, "formula_ok"))
    n_formula_relaxed_ok = sum(1 for r in quality_rows if _true(r, "formula_ok_relaxed"))
    n_mult_ok = sum(1 for r in quality_rows if _true(r, "atom_site_multiplicity_ok"))
    n_strict_valid = sum(1 for r in quality_rows if _true(r, "strict_valid"))
    lines.extend(
        [
            _ratio_line("formula_ok", n_formula_ok, n_total),
            _ratio_line("formula_ok_relaxed", n_formula_relaxed_ok, n_total),
            _ratio_line("atom_site_multiplicity_ok", n_mult_ok, n_total),
            _ratio_line("strict_valid", n_strict_valid, n_total),
            f"check_composition_gate: {bool(check_composition)}",
        ]
    )
    lines.extend(["", "top_failure_reasons:"])
    for item in top:
        ratio = item.get("ratio")
        if isinstance(ratio, (int, float)):
            lines.append(f"- {item['reason']}: {item['count']} (ratio={ratio:.3f})")
        else:
            lines.append(f"- {item['reason']}: {item['count']}")
    _write_text(round_dir / "cif_quality_summary.txt", "\n".join(lines) + "\n")


def _print_round_quality_summary(
    round_index: int,
    quality_rows: List[Dict[str, Any]],
    *,
    bond_cutoff: float,
    check_composition: bool,
) -> None:
    """Print a compact, human-readable summary of key validation fields to stdout."""
    n_total = len(quality_rows)
    n_pass = sum(1 for r in quality_rows if r.get("validation_ok") is True)
    n_fail = sum(1 for r in quality_rows if r.get("validation_ok") is False)

    n_space_group_true = sum(1 for r in quality_rows if r.get("space_group_ok") is True)

    def _bond_ok(r: Dict[str, Any]) -> bool:
        v = r.get("bond_lengths_reasonable")
        if isinstance(v, bool):
            return bool(v)
        score = r.get("bond_length_score")
        return isinstance(score, (int, float)) and float(score) >= bond_cutoff

    n_bond_ok = sum(1 for r in quality_rows if _bond_ok(r))

    def _true(r: Dict[str, Any], key: str) -> bool:
        return r.get(key) is True

    def _ratio_line(name: str, true_n: int, total_n: int) -> str:
        if not total_n:
            return f"{name}: 0/0 (0.000)"
        return f"{name}: {true_n}/{total_n} ({true_n / total_n:.3f})"

    top = _summarize_reasons(quality_rows, top_n=5)

    print("")
    print(f"[round_{round_index:02d}] cif_quality summary")
    print(f"  - n_total: {n_total}  n_pass: {n_pass}  n_fail: {n_fail}")
    print(f"  - {_ratio_line('validation_ok', n_pass, n_total)}")
    print(f"  - {_ratio_line('space_group_ok', n_space_group_true, n_total)}")
    print(f"  - {_ratio_line('bond_lengths_reasonable', n_bond_ok, n_total)}")
    n_formula_ok = sum(1 for r in quality_rows if _true(r, "formula_ok"))
    n_formula_relaxed_ok = sum(1 for r in quality_rows if _true(r, "formula_ok_relaxed"))
    n_mult_ok = sum(1 for r in quality_rows if _true(r, "atom_site_multiplicity_ok"))
    n_strict_valid = sum(1 for r in quality_rows if _true(r, "strict_valid"))
    print(f"  - {_ratio_line('formula_ok', n_formula_ok, n_total)}")
    print(f"  - {_ratio_line('formula_ok_relaxed', n_formula_relaxed_ok, n_total)}")
    print(f"  - {_ratio_line('atom_site_multiplicity_ok', n_mult_ok, n_total)}")
    print(f"  - {_ratio_line('strict_valid', n_strict_valid, n_total)}")
    print(f"  - check_composition_gate: {bool(check_composition)}")
    if top:
        print("  - top_failure_reasons:")
        for item in top:
            print(f"    - {item['reason']}: {item['count']}")
    print("")


def _print_round_alignn_summary(round_index: int, scored: List[ScoredSample], properties: List[str]) -> None:
    """Print per-round ALIGNN ok ratio and basic stats for requested properties."""
    total = len(scored)
    ok = sum(1 for s in scored if s.alignn_ok)
    print("")
    print(f"[round_{round_index:02d}] ALIGNN summary")
    print(f"  - total/ok: {total}/{ok}")

    for p in properties:
        vals = [s.properties.get(p) for s in scored if s.alignn_ok and s.properties.get(p) is not None]
        vals_f = [float(v) for v in vals if isinstance(v, (int, float))]
        if not vals_f:
            print(f"  - {p}: (no valid values)")
            continue
        print(f"  - {p}: mean={statistics.fmean(vals_f):.4f}, min={min(vals_f):.4f}, max={max(vals_f):.4f}")


def _print_top_structures(round_index: int, top_structures: List[Dict[str, Any]]) -> None:
    """Print the selected top structures in a doc-style compact format."""
    print("")
    print(f"[round_{round_index:02d}] Top structures")
    if not top_structures:
        print("  - (none)")
        return
    for idx, item in enumerate(top_structures, start=1):
        cif_path = item.get("cif_path")
        props = item.get("properties") or {}
        bandgap = props.get("bandgap")
        formation_energy = props.get("formation_energy")
        validation_ok = item.get("validation_ok")
        print(f"  - rank{idx}: {cif_path}")
        print(f"    bandgap={bandgap}, formation_energy={formation_energy}, validation_ok={validation_ok}")


def _score_samples(
    round_dir: Path,
    quality_rows: List[Dict[str, Any]],
    cifs_dir: Path,
    alignn_host: str,
    alignn_port: int,
    alignn_timeout_ms: int,
    alignn_properties: List[str],
    bond_cutoff: float,
    check_composition: bool,
) -> List[ScoredSample]:
    scored: List[ScoredSample] = []
    for r in quality_rows:
        cif_path = Path(r["cif_path"])
        sample_id = str(r.get("sample_id") or cif_path.stem)
        validation_ok = bool(r.get("validation_ok"))
        postprocessed_cif_path = r.get("postprocessed_cif_path")
        cleaned_cif_path = r.get("cleaned_cif_path")
        errors: Dict[str, Any] = {}
        props: Dict[str, Optional[float]] = {p: None for p in alignn_properties}
        alignn_ok = False

        # ALIGNN scoring should use the postprocessed CIF text (bin/postprocess.py),
        # not the more invasive cifs_cleaned output.
        cif_for_alignn = None
        if isinstance(postprocessed_cif_path, str) and postprocessed_cif_path.strip():
            cif_for_alignn = postprocessed_cif_path
        elif isinstance(cleaned_cif_path, str) and cleaned_cif_path.strip():
            # Backward-compatible fallback for older experiment dirs.
            cif_for_alignn = cleaned_cif_path

        if isinstance(cif_for_alignn, str) and cif_for_alignn.strip():
            try:
                cleaned_text = _read_text(Path(cif_for_alignn))
                out = score_cif_via_alignn(
                    cleaned_text,
                    host=alignn_host,
                    port=alignn_port,
                    properties=alignn_properties,
                    timeout_ms=alignn_timeout_ms,
                )
                alignn_ok = bool(out.get("ok"))
                for p in alignn_properties:
                    v = out.get(p)
                    props[p] = float(v) if isinstance(v, (int, float)) else None
                if out.get("errors"):
                    errors["alignn_errors"] = out["errors"]
            except Exception as exc:  # noqa: BLE001
                errors["alignn_exception"] = f"{type(exc).__name__}: {exc}"

        scored.append(
            ScoredSample(
                sample_id=sample_id,
                cif_path=str(cif_path),
                validation_ok=validation_ok,
                validation_reasons=r.get("validation_reasons") or [],
                bond_length_score=r.get("bond_length_score"),
                formula_ok=r.get("formula_ok"),
                atom_site_multiplicity_ok=r.get("atom_site_multiplicity_ok"),
                space_group_ok=r.get("space_group_ok"),
                alignn_ok=alignn_ok,
                properties=props,
                errors=errors,
            )
        )
    return scored


def _write_scores_csv(round_dir: Path, scored: List[ScoredSample], score_property: str) -> None:
    out = round_dir / "scores.csv"
    props = sorted({k for s in scored for k in s.properties.keys()})
    fieldnames = [
        "sample_id",
        "cif_path",
        "validation_ok",
        "alignn_ok",
        *props,
        "score_property",
        "score_value",
        "validation_reasons",
        "errors",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in scored:
            row: Dict[str, Any] = {
                "sample_id": s.sample_id,
                "cif_path": s.cif_path,
                "validation_ok": s.validation_ok,
                "alignn_ok": s.alignn_ok,
                "score_property": score_property,
                "score_value": s.properties.get(score_property),
                "validation_reasons": json.dumps(s.validation_reasons, ensure_ascii=False),
                "errors": json.dumps(s.errors, ensure_ascii=False) if s.errors else "",
            }
            for p in props:
                row[p] = s.properties.get(p)
            w.writerow(row)


def _select_top_structures(
    scored: List[ScoredSample],
    score_property: str,
    score_goal: str,
    top_k: int,
    prefer_valid: bool,
) -> List[Dict[str, Any]]:
    candidates = [s for s in scored if s.alignn_ok and s.properties.get(score_property) is not None]
    if prefer_valid:
        valid = [s for s in candidates if s.validation_ok]
        if valid:
            candidates = valid
    reverse = score_goal == "max"
    candidates.sort(key=lambda s: float(s.properties.get(score_property) or float("nan")), reverse=reverse)
    out: List[Dict[str, Any]] = []
    for s in candidates[:top_k]:
        out.append(
            {
                "sample_id": s.sample_id,
                "cif_path": s.cif_path,
                "validation_ok": s.validation_ok,
                "validation_reasons": s.validation_reasons,
                "properties": s.properties,
                "score_property": score_property,
                "score_goal": score_goal,
                "score_value": s.properties.get(score_property),
            }
        )
    return out


def _structure_dict(s: ScoredSample, score_property: str, score_goal: str, tag: str | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "sample_id": s.sample_id,
        "cif_path": s.cif_path,
        "validation_ok": s.validation_ok,
        "validation_reasons": s.validation_reasons,
        "properties": s.properties,
        "score_property": score_property,
        "score_goal": score_goal,
        "score_value": s.properties.get(score_property),
    }
    if tag:
        out["tag"] = tag
    return out


def _select_representative_structures(scored: List[ScoredSample]) -> List[Dict[str, Any]]:
    """Pick small, representative samples to guide the optimizer LLM.

    This is intentionally compact and stable:
    - prefer validation_ok=True when available
    - show both primary (formation_energy) and secondary (bandgap) objectives
    - include at least one "high bandgap but invalid" example when possible
    """

    candidates = [s for s in scored if s.alignn_ok]
    valid = [s for s in candidates if s.validation_ok]
    invalid = [s for s in candidates if not s.validation_ok]

    def _pick(pool: List[ScoredSample], prop: str, goal: str) -> ScoredSample | None:
        pool2 = [s for s in pool if isinstance(s.properties.get(prop), (int, float))]
        if not pool2:
            return None
        reverse = goal == "max"
        return sorted(pool2, key=lambda s: float(s.properties.get(prop) or float("nan")), reverse=reverse)[0]

    reps: List[Dict[str, Any]] = []
    best_valid_fe = _pick(valid, "formation_energy", "min")
    if best_valid_fe is not None:
        reps.append(_structure_dict(best_valid_fe, "formation_energy", "min", tag="best_valid_low_formation_energy"))

    best_valid_bg = _pick(valid, "bandgap", "max")
    if best_valid_bg is not None:
        reps.append(_structure_dict(best_valid_bg, "bandgap", "max", tag="best_valid_high_bandgap"))

    best_invalid_bg = _pick(invalid, "bandgap", "max")
    if best_invalid_bg is not None:
        reps.append(_structure_dict(best_invalid_bg, "bandgap", "max", tag="best_invalid_high_bandgap"))

    if not reps:
        # Worst-case fallback: pick something informative even if nothing is valid.
        best_any_fe = _pick(candidates, "formation_energy", "min")
        if best_any_fe is not None:
            reps.append(_structure_dict(best_any_fe, "formation_energy", "min", tag="best_any_low_formation_energy"))
        best_any_bg = _pick(candidates, "bandgap", "max")
        if best_any_bg is not None:
            reps.append(_structure_dict(best_any_bg, "bandgap", "max", tag="best_any_high_bandgap"))

    return reps[:6]


def _build_round_metrics(
    quality_rows: List[Dict[str, Any]],
    scored: List[ScoredSample],
    check_composition: bool,
) -> Dict[str, Any]:
    n_total = len(quality_rows)
    n_validation_pass = sum(1 for r in quality_rows if r.get("validation_ok") is True)
    n_validation_fail = sum(1 for r in quality_rows if r.get("validation_ok") is False)
    n_scored_ok = sum(1 for s in scored if s.alignn_ok)
    return {
        "n_structures": n_total,
        "n_validation_pass": n_validation_pass,
        "n_validation_fail": n_validation_fail,
        "validation_pass_ratio": (n_validation_pass / n_total) if n_total else None,
        "n_scored_ok": n_scored_ok,
        "check_composition": check_composition,
    }


def _alignn_stats(scored: List[ScoredSample], properties: List[str]) -> Dict[str, Any]:
    total = len(scored)
    ok = sum(1 for s in scored if s.alignn_ok)
    out: Dict[str, Any] = {"total": total, "ok": ok, "per_property": {}}
    for p in properties:
        vals = [s.properties.get(p) for s in scored if s.alignn_ok and s.properties.get(p) is not None]
        vals_f = [float(v) for v in vals if isinstance(v, (int, float))]
        if not vals_f:
            out["per_property"][p] = None
            continue
        out["per_property"][p] = {
            "mean": statistics.fmean(vals_f),
            "min": min(vals_f),
            "max": max(vals_f),
        }
    return out


def _filter_ratios(quality_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(quality_rows)

    def _cnt_true(key: str) -> int:
        return sum(1 for r in quality_rows if r.get(key) is True)

    def _ratio(true_n: int) -> Optional[float]:
        return (true_n / n_total) if n_total else None

    return {
        "n_total": n_total,
        "validation_ok": {"true": _cnt_true("validation_ok"), "ratio": _ratio(_cnt_true("validation_ok"))},
        "space_group_ok": {"true": _cnt_true("space_group_ok"), "ratio": _ratio(_cnt_true("space_group_ok"))},
        "bond_lengths_reasonable": {
            "true": _cnt_true("bond_lengths_reasonable"),
            "ratio": _ratio(_cnt_true("bond_lengths_reasonable")),
        },
        "atom_site_multiplicity_ok": {
            "true": _cnt_true("atom_site_multiplicity_ok"),
            "ratio": _ratio(_cnt_true("atom_site_multiplicity_ok")),
        },
        "formula_ok": {"true": _cnt_true("formula_ok"), "ratio": _ratio(_cnt_true("formula_ok"))},
        "formula_ok_relaxed": {
            "true": _cnt_true("formula_ok_relaxed"),
            "ratio": _ratio(_cnt_true("formula_ok_relaxed")),
        },
        "strict_valid": {"true": _cnt_true("strict_valid"), "ratio": _ratio(_cnt_true("strict_valid"))},
    }


def _write_round_report(
    round_dir: Path,
    *,
    prompt_lines: List[str],
    scored: List[ScoredSample],
    quality_rows: List[Dict[str, Any]],
    top_structures: List[Dict[str, Any]],
    qwen_payload: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    def _fmt_ratio(true_n: int, total: int) -> str:
        return f"{true_n}/{total} ({(true_n / total):.3f})" if total else "0/0 (0.000)"

    n_total = len(quality_rows)
    counts = {
        "validation_ok": sum(1 for r in quality_rows if r.get("validation_ok") is True),
        "space_group_ok": sum(1 for r in quality_rows if r.get("space_group_ok") is True),
        "atom_site_multiplicity_ok": sum(1 for r in quality_rows if r.get("atom_site_multiplicity_ok") is True),
        "bond_lengths_reasonable": sum(1 for r in quality_rows if r.get("bond_lengths_reasonable") is True),
        "formula_ok": sum(1 for r in quality_rows if r.get("formula_ok") is True),
    }

    alignn = _alignn_stats(scored, list(args.alignn_properties))
    parsed = qwen_payload.get("parsed") if isinstance(qwen_payload, dict) else {}
    analysis = (parsed or {}).get("analysis") if isinstance(parsed, dict) else None
    next_lines = (parsed or {}).get("next_prompt_lines") if isinstance(parsed, dict) else None

    lines: List[str] = []
    lines.append(f"Round {round_dir.name.split('_')[-1]}")
    lines.append(f"  路径：{round_dir}")
    lines.append("  prompt.txt：")
    for ln in prompt_lines:
        s = str(ln).strip()
        if s:
            lines.append(f"    - {s}")

    lines.append("  ALIGNN 打分（summary.json）：")
    lines.append(f"    - total/ok: {alignn.get('total')}/{alignn.get('ok')}")
    per_prop = (alignn.get("per_property") or {}) if isinstance(alignn, dict) else {}
    # Keep bandgap first, then formation_energy if present.
    for prop in ["bandgap", "formation_energy"]:
        if prop in per_prop:
            stats = per_prop.get(prop)
            if not stats:
                lines.append(f"    - {prop}: (no valid values)")
            else:
                lines.append(
                    f"    - {prop}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
                )
    for prop, stats in per_prop.items():
        if prop in {"bandgap", "formation_energy"}:
            continue
        if not stats:
            lines.append(f"    - {prop}: (no valid values)")
        else:
            lines.append(f"    - {prop}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

    lines.append("  过滤器（cif_quality_summary.txt）：")
    lines.append(f"    - validation_ok: {_fmt_ratio(counts['validation_ok'], n_total)}")
    lines.append(f"    - space_group_ok: {_fmt_ratio(counts['space_group_ok'], n_total)}")
    lines.append(f"    - atom_site_multiplicity_ok: {_fmt_ratio(counts['atom_site_multiplicity_ok'], n_total)}")
    lines.append(f"    - bond_lengths_reasonable: {_fmt_ratio(counts['bond_lengths_reasonable'], n_total)}")
    lines.append(f"    - formula_ok: {_fmt_ratio(counts['formula_ok'], n_total)}")

    lines.append("  Top 结构（summary.json 的 top_structures）：")
    if not top_structures:
        lines.append("    - (none)")
    else:
        # For a small helpful note: detect if there's exactly one validation_ok=True in this round.
        n_validation_true = sum(1 for r in quality_rows if r.get("validation_ok") is True)
        for idx, item in enumerate(top_structures, start=1):
            cif_path = item.get("cif_path")
            try:
                if isinstance(cif_path, str) and cif_path:
                    cif_path = str(Path(cif_path).resolve().relative_to(round_dir))
            except Exception:  # noqa: BLE001
                pass
            props = item.get("properties") or {}
            lines.append(f"    - rank{idx}: {cif_path}")
            lines.append(f"      bandgap={props.get('bandgap')}, formation_energy={props.get('formation_energy')}")
            lines.append(f"      validation_ok={item.get('validation_ok')}")
            if n_validation_true == 1 and item.get("validation_ok") is True:
                lines.append("      （该轮唯一通过过滤器的结构）")

    lines.append("  Qwen 输出（qwen_output.json，parsed 部分）：")
    if isinstance(analysis, str) and analysis.strip():
        lines.append(f"    - analysis：{analysis.strip()}")
    if isinstance(next_lines, list) and next_lines:
        lines.append("    - next_prompt_lines（解析后生效）：")
        for ln in next_lines:
            s = str(ln).strip()
            if s:
                lines.append(f"        {s}")

    _write_text(round_dir / "round_report.txt", "\n".join(lines).rstrip() + "\n")


def _build_evaluator_summary(
    round_index: int,
    score_property: str,
    score_goal: str,
    top_structures: List[Dict[str, Any]],
    quality_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    scored: List[ScoredSample],
) -> Dict[str, Any]:
    return {
        "round": round_index,
        "score_property": score_property,
        "score_goal": score_goal,
        "metrics": metrics,
        "top_structures": top_structures,
        "optimizer_objective": {
            "primary": {"property": "formation_energy", "goal": "min"},
            "secondary": {"property": "bandgap", "goal": "max"},
            "note": "Prefer validation_ok=True when possible; otherwise prioritize improving validation_ok first.",
        },
        "representative_structures": _select_representative_structures(scored),
        "validation_failure_reasons_top": _summarize_reasons(quality_rows, top_n=20),
        "validation_counts": {
            "n_total": metrics.get("n_structures"),
            "n_pass": metrics.get("n_validation_pass"),
            "n_fail": metrics.get("n_validation_fail"),
            "pass_ratio": metrics.get("validation_pass_ratio"),
            "check_composition": metrics.get("check_composition"),
        },
    }


def _call_optimizer_llm(
    last_prompt_lines: List[str],
    evaluator_summary: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, Any]]:
    config = QwenConfig()
    if args.qwen_api_base:
        config.api_base = args.qwen_api_base
    if args.qwen_api_key:
        config.api_key = args.qwen_api_key
    if args.qwen_model:
        config.model = args.qwen_model
    if args.qwen_temperature is not None:
        config.temperature = args.qwen_temperature
    if args.qwen_top_p is not None:
        config.top_p = args.qwen_top_p
    if args.qwen_max_tokens is not None:
        config.max_tokens = args.qwen_max_tokens
    if args.qwen_timeout is not None:
        config.timeout = args.qwen_timeout

    client = QwenClient(config)
    parsed, raw_response, raw_text = client.optimize_prompt(last_prompt_lines, evaluator_summary)

    payload = {
        "raw_text": raw_text,
        "response": raw_response,
        "previous_prompt_lines": last_prompt_lines,
        "evaluator_summary": evaluator_summary,
        "parsed": parsed,
    }
    next_prompt_lines = parsed.get("next_prompt_lines") or last_prompt_lines
    next_prompt_lines = [str(x).rstrip() for x in next_prompt_lines if str(x).strip()]
    if not next_prompt_lines:
        next_prompt_lines = [str(x).rstrip() for x in last_prompt_lines if str(x).strip()]
    return next_prompt_lines, payload


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    check_comp = _check_composition(args)
    print_round_summary = bool(args.print_round_summary) or (not bool(args.no_print_round_summary))
    print_qwen = bool(getattr(args, "print_qwen", False)) or (not bool(getattr(args, "no_print_qwen", False)))
    show_cif_warnings = bool(getattr(args, "show_cif_warnings", False))
    prompt_text = _read_text(Path(args.initial_prompt_file).expanduser().resolve())
    current_prompt_lines = [ln.rstrip() for ln in prompt_text.splitlines() if ln.strip()]
    current_prompt_lines, _ = _audit_and_sanitize_prompt(
        current_prompt_lines,
        current_prompt_lines,
        round_dir=out_dir,
        stage="initial",
        mode=args.prompt_sanitizer_mode,
    )

    all_best: List[Dict[str, Any]] = []

    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r:02d}"
        if round_dir.exists() and not args.resume:
            raise SystemExit(f"{round_dir} exists; pass --resume to continue")
        round_dir.mkdir(parents=True, exist_ok=True)

        current_prompt_lines, _ = _audit_and_sanitize_prompt(
            current_prompt_lines,
            current_prompt_lines,
            round_dir=round_dir,
            stage="pre_round_write",
            mode=args.prompt_sanitizer_mode,
        )

        prompt_path = round_dir / "prompt.txt"
        _write_text(prompt_path, "\n".join(current_prompt_lines).rstrip() + "\n")

        # IMPORTANT: By default we feed a filtered prompt into the sampler because CrystaLLM's
        # tokenizer vocabulary does not cover natural-language comments and free-form lines.
        sampling_lines = (
            [ln for ln in current_prompt_lines if ln.strip().startswith(("data_", "_"))]
            if args.sampling_prompt_mode == "filtered"
            else list(current_prompt_lines)
        )
        sampling_prompt_path = round_dir / "prompt_for_sampling.txt"
        _write_text(sampling_prompt_path, "\n".join(sampling_lines).rstrip() + "\n")

        cifs_dir = round_dir / "cifs"
        _run_sample(
            model_dir=args.model_dir,
            prompt_path=(prompt_path if args.sampling_prompt_mode == "prompt_txt" else sampling_prompt_path),
            out_cifs_dir=cifs_dir,
            samples=args.samples_per_round,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + (r - 1),
            device=args.device,
            dtype=args.dtype,
        )

        cif_paths = _collect_generated_cifs(cifs_dir)
        quality_rows: List[Dict[str, Any]] = []
        for p in cif_paths:
            cif_text = _read_text(p)
            # Always postprocess raw generations (symmetry ops replacement + atom-props removal),
            # and use the postprocessed file as the ALIGNN input.
            postprocessed_text = postprocess_cif(cif_text, p.name)
            postprocessed_path: Optional[str] = None
            if isinstance(postprocessed_text, str) and postprocessed_text.strip():
                post_dir = round_dir / "cifs_postprocessed"
                post_dir.mkdir(parents=True, exist_ok=True)
                out_path = post_dir / f"{p.stem}.cif"
                _write_text(out_path, postprocessed_text)
                postprocessed_path = str(out_path)

            # pymatgen prints extremely verbose warnings (sometimes including CIF snippets)
            # when parsing fails. We silence these by default and rely on cif_quality.* files
            # plus our per-round summary instead.
            with warnings.catch_warnings():
                if not show_cif_warnings:
                    warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\.io\.cif")
                    warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\.analysis\.local_env")
                cleaned, meta = clean_and_validate_cif(
                    postprocessed_text,
                    bond_length_acceptability_cutoff=args.validation_bond_cutoff,
                    check_composition=check_comp,
                )
            cleaned_path: Optional[str] = None
            if isinstance(cleaned, str) and cleaned.strip():
                cleaned_dir = round_dir / "cifs_cleaned"
                cleaned_dir.mkdir(parents=True, exist_ok=True)
                out_path = cleaned_dir / f"{p.stem}.cif"
                _write_text(out_path, cleaned)
                cleaned_path = str(out_path)
            quality_rows.append(
                {
                    "sample_id": p.stem,
                    "cif_path": str(p),
                    "validation_ok": bool(meta.get("valid")),
                    "validation_reasons": meta.get("reasons") or [],
                    "bond_length_score": meta.get("bond_length_score"),
                    "bond_lengths_reasonable": meta.get("bond_lengths_reasonable"),
                    "formula_ok": meta.get("formula_ok"),
                    "formula_ok_relaxed": meta.get("formula_ok_relaxed"),
                    "atom_site_multiplicity_ok": meta.get("atom_site_multiplicity_ok"),
                    "space_group_ok": meta.get("space_group_ok"),
                    "strict_valid": meta.get("strict_valid"),
                    "cleaned_cif_path": cleaned_path,
                    "postprocessed_cif_path": postprocessed_path,
                    "validator_step_errors": meta.get("validator_step_errors") or {},
                }
            )

        _write_cif_quality(round_dir, quality_rows, bond_cutoff=args.validation_bond_cutoff, check_composition=check_comp)

        scored = _score_samples(
            round_dir=round_dir,
            quality_rows=quality_rows,
            cifs_dir=cifs_dir,
            alignn_host=args.alignn_host,
            alignn_port=args.alignn_port,
            alignn_timeout_ms=args.alignn_timeout_ms,
            alignn_properties=args.alignn_properties,
            bond_cutoff=args.validation_bond_cutoff,
            check_composition=check_comp,
        )
        _write_scores_csv(round_dir, scored, score_property=args.score_property)

        top_structures = _select_top_structures(
            scored=scored,
            score_property=args.score_property,
            score_goal=args.score_goal,
            top_k=args.top_structures,
            prefer_valid=(not args.no_prefer_valid_structures),
        )

        if print_round_summary:
            _print_round_alignn_summary(r, scored, args.alignn_properties)
            _print_round_quality_summary(
                r,
                quality_rows,
                bond_cutoff=args.validation_bond_cutoff,
                check_composition=check_comp,
            )
            _print_top_structures(r, top_structures)

        metrics = _build_round_metrics(quality_rows, scored, check_comp)
        # Extra reporting-friendly summaries.
        metrics["alignn"] = _alignn_stats(scored, list(args.alignn_properties))
        metrics["filter"] = _filter_ratios(quality_rows)
        summary = {
            "round": r,
            "score_property": args.score_property,
            "score_goal": args.score_goal,
            "metrics": metrics,
            "top_structures": top_structures,
            "args": vars(args),
        }
        _write_json(round_dir / "summary.json", summary)

        evaluator_summary = _build_evaluator_summary(
            round_index=r,
            score_property=args.score_property,
            score_goal=args.score_goal,
            top_structures=top_structures,
            quality_rows=quality_rows,
            metrics=metrics,
            scored=scored,
        )
        _write_json(round_dir / "evaluator_summary.json", evaluator_summary)

        try:
            next_prompt_lines, qwen_payload = _call_optimizer_llm(current_prompt_lines, evaluator_summary, args)
        except Exception as exc:  # noqa: BLE001
            # Robust fallback: allow the closed-loop to continue even if the optimizer LLM
            # endpoint is unavailable (e.g. vLLM server not running).
            next_prompt_lines = list(current_prompt_lines)
            err_text = f"{type(exc).__name__}: {exc}"
            qwen_payload = {
                "raw_text": "",
                "response": None,
                "previous_prompt_lines": current_prompt_lines,
                "evaluator_summary": evaluator_summary,
                "parsed": {
                    "analysis": (
                        "Optimizer LLM unavailable; keeping previous prompt lines. "
                        "Start the Qwen/vLLM server and retry if you want prompt edits. "
                        f"Error: {err_text}"
                    ),
                    "next_prompt_lines": list(current_prompt_lines),
                    "_raw": "<ERROR>",
                },
                "error": {
                    "message": err_text,
                    "type": type(exc).__name__,
                },
            }
        sanitized_next, audit = _audit_and_sanitize_prompt(
            next_prompt_lines,
            current_prompt_lines,
            round_dir=round_dir,
            stage="post_llm",
            mode=args.prompt_sanitizer_mode,
        )
        qwen_payload["prompt_sanitizer_audit"] = audit
        _write_json(round_dir / "qwen_output.json", qwen_payload)
        # Backward-compatible alias (older docs refer to llama_output.json).
        try:
            _write_json(round_dir / "llama_output.json", qwen_payload)
        except Exception:  # noqa: BLE001
            pass

        # Human-readable per-round report.
        try:
            _write_round_report(
                round_dir,
                prompt_lines=current_prompt_lines,
                scored=scored,
                quality_rows=quality_rows,
                top_structures=top_structures,
                qwen_payload=qwen_payload,
                args=args,
            )
        except Exception:  # noqa: BLE001
            pass

        if print_qwen:
            parsed = qwen_payload.get("parsed") or {}
            analysis_text = parsed.get("analysis")
            proposed = parsed.get("next_prompt_lines") or []
            sanitizer_audit = (audit.get("sanitizer_audit") or {}) if isinstance(audit, dict) else {}
            fallback_used = sanitizer_audit.get("fallback_used")
            fallback_reason = sanitizer_audit.get("fallback_reason")
            key_changes = sanitizer_audit.get("key_changes") or []
            kept = sanitizer_audit.get("candidate_lines_kept") or sanitized_next

            print(f"\n=== Qwen suggestion (round_{r:02d}) ===")
            if isinstance(analysis_text, str) and analysis_text.strip():
                print(f"ANALYSIS: {analysis_text.strip()}")
            if proposed:
                print("PROPOSED_NEXT_PROMPT_LINES:")
                for ln in proposed:
                    s = str(ln).rstrip()
                    if s:
                        print(f"  {s}")
            print(f"SANITIZER: fallback_used={fallback_used} reason={fallback_reason}")
            if key_changes:
                print("APPLIED_KEY_CHANGES:")
                for item in key_changes:
                    try:
                        k = item.get("key")
                        old = item.get("old")
                        new = item.get("new")
                        print(f"  {k}: {old} -> {new}")
                    except Exception:  # noqa: BLE001
                        print(f"  {item}")
            print("FINAL_PROMPT_LINES:")
            for ln in kept:
                s = str(ln).rstrip()
                if s:
                    print(f"  {s}")

        current_prompt_lines = sanitized_next
        all_best.extend(top_structures)

    # final report (best across rounds)
    def _score_value(item: Dict[str, Any]) -> float:
        v = item.get("score_value")
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    reverse = args.score_goal == "max"
    all_best_sorted = sorted(all_best, key=_score_value, reverse=reverse)
    final = {
        "score_property": args.score_property,
        "score_goal": args.score_goal,
        "final_top_k": args.final_top_k,
        "best_overall": all_best_sorted[: args.final_top_k],
    }
    _write_json(out_dir / "final_report.json", final)

    # Optional: update a single cross-experiment table for quick browsing.
    if args.experiments_table_out:
        try:
            table_root = Path(args.experiments_table_root).expanduser().resolve() if args.experiments_table_root else out_dir.parent
            table_out = Path(args.experiments_table_out).expanduser().resolve()
            cmd = [
                sys.executable,
                str((_ROOT / "bin" / "experiments_table.py").resolve()),
                "--experiments-root",
                str(table_root),
                "--out",
                str(table_out),
            ]
            subprocess.run(cmd, check=False)
        except Exception:  # noqa: BLE001
            pass

    # Optional: update a single per-round cross-experiment table for quick browsing.
    if args.experiments_rounds_table_out:
        try:
            table_root = (
                Path(args.experiments_rounds_table_root).expanduser().resolve()
                if args.experiments_rounds_table_root
                else out_dir.parent
            )
            table_out = Path(args.experiments_rounds_table_out).expanduser().resolve()
            cmd = [
                sys.executable,
                str((_ROOT / "bin" / "experiments_rounds_table.py").resolve()),
                "--experiments-root",
                str(table_root),
                "--out",
                str(table_out),
            ]
            subprocess.run(cmd, check=False)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
