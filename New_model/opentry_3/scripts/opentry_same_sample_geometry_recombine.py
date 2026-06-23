#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_positive_geometry_residual_examples as pos_geom  # noqa: E402
import opentry_pair_delta_geometry_variants as pair_delta  # noqa: E402
import run_mp20_minicfjoint_v2 as v2  # noqa: E402
from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cif_digest(cif: str) -> str:
    return hashlib.sha1(cif.encode("utf-8", errors="ignore")).hexdigest()


def rank_value(row: dict[str, Any]) -> int:
    try:
        return int(row.get("rank", row.get("original_rank", 10**9)))
    except Exception:
        return 10**9


def load_repr(path: Path, max_records: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row["keys"]["sample_id"])
        out[sample_id] = row
        if int(max_records) > 0 and len(out) >= int(max_records):
            break
    return out


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("sample_id") or ""), []).append(row)
    for values in out.values():
        values.sort(key=rank_value)
    return out


def valid_geometry(row: dict[str, Any]) -> bool:
    return bool(row.get("geometry_lattice")) and bool(row.get("geometry_params"))


def donor_score(row: dict[str, Any]) -> tuple[int, float, float]:
    sg_bad = 0 if bool(row.get("sg_ok")) else 1
    distance = float(row.get("geometry_distance") if row.get("geometry_distance") is not None else 1.0e6)
    return (sg_bad, distance, float(rank_value(row)))


def render_variant(
    engine: OrbitEngine,
    record_base: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
    mode: str,
    rank: int,
) -> dict[str, Any] | None:
    rec_params = dict(recipient.get("geometry_params") or {})
    donor_params = dict(donor.get("geometry_params") or {})
    rec_lattice = dict(recipient.get("geometry_lattice") or {})
    donor_lattice = dict(donor.get("geometry_lattice") or {})
    if mode == "lattice_only":
        params = rec_params
        lattice = donor_lattice
    elif mode == "params_only":
        params = donor_params
        lattice = rec_lattice
    elif mode == "lattice_params":
        params = donor_params
        lattice = donor_lattice
    else:
        raise ValueError(mode)

    rows = pos_geom.rows_from_wa_key(engine, str(recipient.get("canonical_wa_key") or ""), params)
    render = v2.render_candidate(engine, record_base, {"rows": rows, "params": params, "lattice": lattice}, rank, f"same_sample_{mode}")
    if not render.get("ok"):
        return None
    cif = str(render.get("cif") or "")
    metric = validate_cif(cif, record_base["formula_counts"], int(record_base["sg"]))
    return {
        **{key: value for key, value in recipient.items() if key not in {"rank", "cif"}},
        "rank": int(rank),
        "cif": cif,
        "readable": bool(metric.get("readable")),
        "formula_ok": bool(metric.get("formula_ok")),
        "composition_exact": bool(metric.get("composition_exact")),
        "atom_count_ok": bool(render.get("atom_count_ok")),
        "detected_sg": metric.get("detected_sg"),
        "sg_ok": bool(metric.get("sg_ok")),
        "geometry_lattice": lattice,
        "geometry_params": params,
        "geometry_source": "same_sample_geometry_recombine",
        "geometry_param_variant_mode": f"same_sample_{mode}",
        "geometry_lattice_mode": f"same_sample_{mode}",
        "same_sample_mode": mode,
        "donor_rank": int(rank_value(donor)),
        "donor_candidate_uid": str(donor.get("candidate_uid") or ""),
        "donor_canonical_wa_key": str(donor.get("canonical_wa_key") or ""),
        "recipient_rank": int(rank_value(recipient)),
        "recipient_candidate_uid": str(recipient.get("candidate_uid") or ""),
        "original_rank": int(rank_value(recipient)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=50)
    parser.add_argument("--max-recipients-per-sample", type=int, default=16)
    parser.add_argument("--max-donors-per-sample", type=int, default=10)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--include-params-only", action="store_true")
    parser.add_argument("--require-donor-sg-ok", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repr_rows = load_repr(args.repr_jsonl, int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))
    engine = OrbitEngine(args.lookup_json)

    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "samples_with_input": 0,
        "samples_with_output": 0,
        "input_rows_seen": 0,
        "rows_ge_7_input_seen": 0,
        "recipient_rows": 0,
        "donor_rows": 0,
        "same_signature_pairs": 0,
        "render_attempts": 0,
        "render_success": 0,
    }
    modes = ["lattice_only", "lattice_params"]
    if args.include_params_only:
        modes.append("params_only")

    for sample_id, repr_row in repr_rows.items():
        candidates = [row for row in rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)] if valid_geometry(row)]
        if candidates:
            stats["samples_with_input"] += 1
        stats["input_rows_seen"] += len(candidates)
        if int(repr_row["row_count"]) >= 7:
            stats["rows_ge_7_input_seen"] += len(candidates)
        if not candidates:
            continue
        sg = int(repr_row["sg"])
        record_base = {
            "sample_id": sample_id,
            "formula_counts": dict(repr_row["formula_counts"]),
            "sg": sg,
            "sg_symbol": str(repr_row.get("sg_symbol") or ""),
            "atom_count": int(repr_row["atom_count"]),
        }
        donors = [row for row in candidates if (not args.require_donor_sg_ok or bool(row.get("sg_ok")))]
        donors = sorted(donors, key=donor_score)[: int(args.max_donors_per_sample)]
        recipients = candidates[: int(args.max_recipients_per_sample)]
        stats["recipient_rows"] += len(recipients)
        stats["donor_rows"] += len(donors)
        generated: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        for recipient in recipients:
            rec_sig = pair_delta.param_signature(recipient, sg=sg)
            for donor in donors:
                donor_uid = str(donor.get("candidate_uid") or "")
                recipient_uid = str(recipient.get("candidate_uid") or "")
                if donor is recipient or (donor_uid and recipient_uid and donor_uid == recipient_uid):
                    continue
                donor_sig = pair_delta.param_signature(donor, sg=sg)
                same_sig = donor_sig == rec_sig
                if same_sig:
                    stats["same_signature_pairs"] += 1
                for mode in modes:
                    if mode in {"params_only", "lattice_params"} and not same_sig:
                        continue
                    stats["render_attempts"] += 1
                    row = render_variant(engine, record_base, recipient, donor, mode, len(generated) + 1)
                    if row is None:
                        continue
                    digest = cif_digest(str(row.get("cif") or ""))
                    if digest in seen_cifs:
                        continue
                    seen_cifs.add(digest)
                    generated.append(row)
                    stats["render_success"] += 1
                    if len(generated) >= int(args.max_output_candidates_per_sample):
                        break
                if len(generated) >= int(args.max_output_candidates_per_sample):
                    break
            if len(generated) >= int(args.max_output_candidates_per_sample):
                break
        if generated:
            stats["samples_with_output"] += 1
        out_rows.extend(generated[: int(args.max_output_candidates_per_sample)])

    write_jsonl(out_dir / "rendered_topk.jsonl", out_rows)
    summary = {
        "config": {
            "input_rendered_jsonl": str(args.input_rendered_jsonl),
            "repr_jsonl": str(args.repr_jsonl),
            "lookup_json": str(args.lookup_json),
            "max_records": int(args.max_records),
            "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
            "max_recipients_per_sample": int(args.max_recipients_per_sample),
            "max_donors_per_sample": int(args.max_donors_per_sample),
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "include_params_only": bool(args.include_params_only),
            "require_donor_sg_ok": bool(args.require_donor_sg_ok),
        },
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Same-sample geometry recombination is GT-free at inference: it recombines rendered candidate lattice/free-params within each sample and uses no StructureMatcher labels.",
    }
    write_json(out_dir / "same_sample_recombine_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
