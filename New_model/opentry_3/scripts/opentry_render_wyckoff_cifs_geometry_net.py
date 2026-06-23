#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

import torch


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine  # noqa: E402
from symcif_v4.validation import validate_cif  # noqa: E402

import run_mp20_minicfjoint_v2 as v2  # noqa: E402
import train_symcif_v4_geometry_model as geom  # noqa: E402


_ENGINE: OrbitEngine | None = None
_MODEL: geom.GeometryNet | None = None
_VOCABS: dict[str, dict[str, int]] = {}
_LATTICE_MEAN: torch.Tensor | None = None
_LATTICE_STD: torch.Tensor | None = None
_TOP_K = 50
_VARIANTS_PER_WA = 1
_PLAN_MODE = "wa_diverse"
_REQUIRE_COMPOSITION_EXACT = True


def ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if OPENTRY_ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_3: {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_rows(engine: OrbitEngine, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        orbit = engine.get_orbit_by_id(str(row.get("orbit_id") or row.get("canonical_orbit_id")))
        out.append(
            {
                "element": str(row["element"]),
                "orbit_id": orbit.canonical_orbit_id,
                "multiplicity": int(orbit.multiplicity),
                "letter": orbit.letter,
                "enumeration": orbit.enumeration,
                "site_symmetry": orbit.site_symmetry,
                "free_symbols": list(orbit.free_symbols),
            }
        )
    return v2.canonical_rows({"wa_table": out})


def pseudo_record(target: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "sample_id": target["sample_id"],
        "formula_counts": {str(k): int(v) for k, v in target["formula_counts"].items()},
        "sg": int(target["sg"]),
        "sg_symbol": str(target.get("sg_symbol") or ""),
        "atom_count": int(target.get("atom_count") or sum(int(v) for v in target["formula_counts"].values())),
        "n_sites": len(rows),
        "num_elements": len(target["formula_counts"]),
        "wa_table": rows,
        "lattice": target.get("lattice") or {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
    }
    skel, wa = v2.canonical_keys_from_rows(rows)
    out["canonical_skeleton_key"] = skel
    out["canonical_wa_key"] = wa
    return out


def decode_lattice(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, sg: int) -> dict[str, float]:
    values = (raw.detach().cpu() * std.cpu() + mean.cpu()).tolist()
    return v2.lattice_from_target([float(x) for x in values], int(sg))


def variant_offsets(variant: int) -> tuple[float, float]:
    coord_offsets = [0.0, 0.015, -0.015, 0.03, -0.03, 0.06, -0.06, 0.10, -0.10]
    length_scales = [1.0, 1.0, 1.0, 0.985, 1.015, 0.97, 1.03, 0.95, 1.05]
    idx = min(max(0, int(variant)), len(coord_offsets) - 1)
    return coord_offsets[idx], length_scales[idx]


def apply_variant(
    params: dict[int, dict[str, float]],
    lattice: dict[str, float],
    *,
    variant: int,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    if int(variant) <= 0:
        return params, lattice
    offset, scale = variant_offsets(int(variant))
    out_params: dict[int, dict[str, float]] = {}
    for row_idx, row_params in params.items():
        out_params[row_idx] = {}
        for sym, value in row_params.items():
            sign = -1.0 if ((int(row_idx) + ord(str(sym)[0]) + int(variant)) % 2) else 1.0
            out_params[row_idx][str(sym)] = (float(value) + sign * float(offset)) % 1.0
    out_lattice = dict(lattice)
    for key in ("a", "b", "c"):
        out_lattice[key] = max(0.5, float(out_lattice[key]) * float(scale))
    return out_params, out_lattice


@torch.no_grad()
def predict_geometry(record: dict[str, Any]) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    if _MODEL is None or _LATTICE_MEAN is None or _LATTICE_STD is None:
        raise RuntimeError("worker not initialized")
    batch = geom.collate_geometry([record], vocabs=_VOCABS, lattice_mean=_LATTICE_MEAN, lattice_std=_LATTICE_STD)
    lattice_raw, coords = _MODEL(batch)
    lattice = decode_lattice(lattice_raw[0], _LATTICE_MEAN, _LATTICE_STD, int(record["sg"]))
    coord_values = coords[0].detach().cpu()
    params: dict[int, dict[str, float]] = {}
    coord_index = {"x": 0, "y": 1, "z": 2}
    for row_idx, row in enumerate(record["wa_table"]):
        row_params: dict[str, float] = {}
        for sym in row.get("free_symbols") or []:
            if str(sym) in coord_index:
                row_params[str(sym)] = float(coord_values[row_idx, coord_index[str(sym)]]) % 1.0
        params[row_idx] = row_params
    return params, lattice


def init_worker(lookup_json: str, data_root: str, ckpt_path: str, top_k: int, variants_per_wa: int, plan_mode: str, require_composition_exact: bool) -> None:
    global _ENGINE, _MODEL, _VOCABS, _LATTICE_MEAN, _LATTICE_STD, _TOP_K, _VARIANTS_PER_WA, _PLAN_MODE, _REQUIRE_COMPOSITION_EXACT
    _ENGINE = OrbitEngine.from_structured_root(lookup_json, data_root)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _VOCABS = ckpt["vocabs"]
    config = ckpt.get("config") or {}
    hidden_dim = int(config.get("hidden_dim", 256))
    emb_dim = int(config.get("emb_dim", 64))
    _MODEL = geom.GeometryNet({name: len(vocab) for name, vocab in _VOCABS.items()}, hidden_dim=hidden_dim, emb_dim=emb_dim)
    _MODEL.load_state_dict(ckpt["model_state"])
    _MODEL.eval()
    _LATTICE_MEAN = torch.tensor(ckpt["lattice_mean"], dtype=torch.float32)
    _LATTICE_STD = torch.tensor(ckpt["lattice_std"], dtype=torch.float32)
    _TOP_K = int(top_k)
    _VARIANTS_PER_WA = max(1, int(variants_per_wa))
    _PLAN_MODE = str(plan_mode)
    _REQUIRE_COMPOSITION_EXACT = bool(require_composition_exact)


def geometry_plan(num_candidates: int) -> list[tuple[int, int]]:
    if _PLAN_MODE == "geometry_interleave":
        return [(i, v) for i in range(num_candidates) for v in range(_VARIANTS_PER_WA)]
    return [(i, v) for v in range(_VARIANTS_PER_WA) for i in range(num_candidates)]


def render_candidate(target: dict[str, Any], candidate: dict[str, Any], rank: int, variant: int) -> dict[str, Any]:
    if _ENGINE is None:
        raise RuntimeError("worker not initialized")
    rows = normalize_rows(_ENGINE, list(candidate.get("rows") or []))
    record = pseudo_record(target, rows)
    params, lattice = predict_geometry(record)
    params, lattice = apply_variant(params, lattice, variant=int(variant))
    rendered = v2.render_candidate(
        _ENGINE,
        record,
        {"rows": rows, "params": params, "lattice": lattice},
        rank,
        f"geometry_net_v{int(variant)}",
    )
    cif = str(rendered.get("cif") or "")
    metric = validate_cif(cif, record["formula_counts"], int(record["sg"])) if cif else {
        "readable": False,
        "formula_ok": False,
        "sg_ok": False,
        "atom_count_ok": False,
        "composition_exact": False,
        "atom_count_after_expansion": None,
        "detected_sg": None,
        "error": rendered.get("error") or "empty_cif",
    }
    skel, wa = v2.canonical_keys_from_rows(rows)
    return {
        "sample_id": target["sample_id"],
        "rank": rank,
        "geometry_mode": "geometry_net",
        "geometry_variant": int(variant),
        "geometry_source": "train_only_geometry_net",
        "canonical_wa_key": wa,
        "canonical_skeleton_key": skel,
        "candidate_score": candidate.get("score"),
        "cif": cif,
        **metric,
    }


def process_payload(payload: tuple[dict[str, Any], dict[str, Any] | None]) -> list[dict[str, Any]]:
    target, prediction = payload
    if prediction is None:
        return []
    candidates = list(prediction.get("ranked_wa_candidates") or [])
    out: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for wa_idx, variant in geometry_plan(len(candidates)):
        if len(out) >= _TOP_K:
            break
        if wa_idx >= len(candidates):
            continue
        row = render_candidate(target, candidates[wa_idx], len(out) + 1, variant)
        if _REQUIRE_COMPOSITION_EXACT and not bool(row.get("composition_exact")):
            continue
        cif_hash = hashlib.sha1(str(row.get("cif") or "").encode("utf-8", errors="ignore")).hexdigest()
        if cif_hash in seen_hashes:
            continue
        seen_hashes.add(cif_hash)
        out.append(row)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "readable": sum(bool(r["readable"]) for r in rows) / max(1, len(rows)),
        "formula_ok": sum(bool(r["formula_ok"]) for r in rows) / max(1, len(rows)),
        "atom_count_ok": sum(bool(r["atom_count_ok"]) for r in rows) / max(1, len(rows)),
        "sg_ok": sum(bool(r["sg_ok"]) for r in rows) / max(1, len(rows)),
        "composition_exact": sum(bool(r["composition_exact"]) for r in rows) / max(1, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render opentry_3 W/A predictions with a train-only geometry net.")
    parser.add_argument("--data-root", type=Path, default=SYMCIF_ROOT / "data" / "structured_symcif_v4_mpts52")
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--geometry-ckpt", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--geometry-variants-per-wa", type=int, default=1)
    parser.add_argument("--geometry-plan-mode", choices=["wa_diverse", "geometry_interleave"], default="wa_diverse")
    parser.add_argument("--allow-non-composition-exact", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-cif-files", type=int, default=100)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = out_dir / "rendered_cifs"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    split_records = read_jsonl(args.data_root / f"{args.split}.jsonl")
    pred_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    payloads = [(record, pred_by_id.get(str(record["sample_id"]))) for record in split_records if str(record["sample_id"]) in pred_by_id]

    require_composition_exact = not bool(args.allow_non_composition_exact)
    if int(args.workers) <= 1:
        init_worker(
            str(args.lookup_json),
            str(args.data_root),
            str(args.geometry_ckpt),
            int(args.top_k),
            int(args.geometry_variants_per_wa),
            str(args.geometry_plan_mode),
            require_composition_exact,
        )
        nested = [process_payload(payload) for payload in payloads]
    else:
        with mp.Pool(
            processes=int(args.workers),
            initializer=init_worker,
            initargs=(
                str(args.lookup_json),
                str(args.data_root),
                str(args.geometry_ckpt),
                int(args.top_k),
                int(args.geometry_variants_per_wa),
                str(args.geometry_plan_mode),
                require_composition_exact,
            ),
        ) as pool:
            nested = list(pool.imap_unordered(process_payload, payloads, chunksize=4))

    rows = [item for sub in nested for item in sub]
    rows.sort(key=lambda r: (str(r["sample_id"]), int(r["rank"])))
    with (out_dir / "rendered_topk.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if i < int(args.write_cif_files):
                safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["sample_id"]))
                (rendered_dir / f"{safe_id}_rank{row['rank']}.cif").write_text(str(row["cif"]), encoding="utf-8")
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "split": args.split,
        "top_k": int(args.top_k),
        "geometry_ckpt": str(args.geometry_ckpt),
        "geometry_variants_per_wa": int(args.geometry_variants_per_wa),
        "geometry_plan_mode": args.geometry_plan_mode,
        "require_composition_exact": require_composition_exact,
        "samples_with_prediction_rows": len(payloads),
        "samples_with_rendered_candidates": len({r["sample_id"] for r in rows}),
        "rendered_rows": len(rows),
        "overall_rows": summarize(rows),
        "rank1": summarize([r for r in rows if int(r["rank"]) == 1]),
        "rank_le_5": summarize([r for r in rows if int(r["rank"]) <= 5]),
        "rank_le_20": summarize([r for r in rows if int(r["rank"]) <= 20]),
        "rank_le_50": summarize([r for r in rows if int(r["rank"]) <= 50]),
    }
    (out_dir / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
