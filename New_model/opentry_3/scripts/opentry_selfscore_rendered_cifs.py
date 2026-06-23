#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


try:
    from pymatgen.core import Structure
except Exception:  # pragma: no cover - pymatgen is expected in the experiment env
    Structure = None  # type: ignore[assignment]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["sample_id"])].append(row)
    for items in by_id.values():
        items.sort(key=lambda row: int(row.get("rank", 10**9)))
    return dict(by_id)


def crystal_system(sg: int) -> str:
    sg = int(sg)
    if 1 <= sg <= 2:
        return "triclinic"
    if 3 <= sg <= 15:
        return "monoclinic"
    if 16 <= sg <= 74:
        return "orthorhombic"
    if 75 <= sg <= 142:
        return "tetragonal"
    if 143 <= sg <= 167:
        return "trigonal"
    if 168 <= sg <= 194:
        return "hexagonal"
    return "cubic"


def atom_bucket(n: int) -> str:
    n = int(n)
    if n < 6:
        return "lt6"
    if n < 12:
        return "ge6_lt12"
    return "ge12"


def quantile(values: list[float], q: float) -> float:
    xs = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def build_volume_priors(train_jsonl: Path | None) -> dict[str, dict[str, tuple[float, float]]]:
    if train_jsonl is None:
        return {}
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sg = int(row.get("sg") or 0)
            atom_count = int(row.get("atom_count") or sum(int(v) for v in dict(row.get("formula_counts") or {}).values()) or 0)
            lattice = dict(row.get("lattice") or {})
            volume = lattice.get("volume")
            if volume is None:
                a, b, c = lattice.get("a"), lattice.get("b"), lattice.get("c")
                if a is not None and b is not None and c is not None:
                    volume = float(a) * float(b) * float(c)
            if sg <= 0 or atom_count <= 0 or volume is None:
                continue
            vpa = float(volume) / float(atom_count)
            if not math.isfinite(vpa) or vpa <= 0:
                continue
            ab = atom_bucket(atom_count)
            system = crystal_system(sg)
            for key in (f"sg:{sg}|atom:{ab}", f"system:{system}|atom:{ab}", f"atom:{ab}", "global"):
                buckets[key]["vpa"].append(vpa)
    priors: dict[str, dict[str, tuple[float, float]]] = {}
    for key, payload in buckets.items():
        values = payload["vpa"]
        if not values:
            continue
        q25 = quantile(values, 0.25)
        q50 = quantile(values, 0.50)
        q75 = quantile(values, 0.75)
        scale = max(1.0, float(q75) - float(q25), 0.20 * float(q50))
        priors[key] = {"vpa": (float(q50), float(scale))}
    return priors


def parse_float_field(cif: str, key: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(key)}\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)", re.MULTILINE)
    match = pattern.search(cif)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_int_field(cif: str, key: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(key)}\s+([0-9]+)", re.MULTILINE)
    match = pattern.search(cif)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def cif_self_features(cif: str) -> dict[str, Any]:
    volume = parse_float_field(cif, "_cell_volume")
    min_distance: float | None = None
    volume_per_atom: float | None = None
    parsed_sites: int | None = None
    parse_error: str | None = None

    if Structure is not None and cif:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_str(cif, fmt="cif", primitive=False, merge_tol=0.0)
            parsed_sites = int(len(structure))
            if parsed_sites > 0:
                volume_per_atom = float(structure.volume) / float(parsed_sites)
            if parsed_sites > 1:
                matrix = structure.distance_matrix
                vals = []
                for i in range(parsed_sites):
                    for j in range(i + 1, parsed_sites):
                        vals.append(float(matrix[i][j]))
                min_distance = min(vals) if vals else None
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            parse_error = str(exc)[:200]

    return {
        "self_min_distance": min_distance,
        "self_volume": volume,
        "self_volume_per_atom": volume_per_atom,
        "self_parsed_sites": parsed_sites,
        "self_parse_error": parse_error,
    }


def distance_score(min_distance: float | None) -> float:
    if min_distance is None or not math.isfinite(float(min_distance)):
        return -25.0
    d = float(min_distance)
    if d < 0.45:
        return -250.0
    if d < 0.70:
        return -120.0
    if d < 0.90:
        return -45.0
    if d < 1.05:
        return -10.0
    if d <= 3.20:
        return min(25.0, 6.0 * d)
    if d <= 5.00:
        return 8.0
    return -10.0


def strict_distance_score(min_distance: float | None) -> float:
    if min_distance is None or not math.isfinite(float(min_distance)):
        return -80.0
    d = float(min_distance)
    if d < 0.55:
        return -400.0
    if d < 0.85:
        return -180.0
    if d < 1.05:
        return -75.0
    if d < 1.20:
        return -35.0
    if d < 1.45:
        return -10.0 + 40.0 * (d - 1.20)
    if d <= 2.40:
        return 8.0 + 24.0 * (d - 1.45)
    if d <= 3.40:
        return 30.0 - 8.0 * (d - 2.40)
    if d <= 5.50:
        return 4.0
    return -30.0


def volume_score(volume_per_atom: float | None) -> float:
    if volume_per_atom is None or not math.isfinite(float(volume_per_atom)):
        return -8.0
    vpa = float(volume_per_atom)
    if 3.0 <= vpa <= 80.0:
        return 12.0
    if 1.5 <= vpa <= 120.0:
        return 0.0
    return -20.0


def strict_volume_score(volume_per_atom: float | None) -> float:
    if volume_per_atom is None or not math.isfinite(float(volume_per_atom)):
        return -15.0
    vpa = float(volume_per_atom)
    if vpa < 5.0:
        return -80.0
    if vpa < 10.0:
        return -35.0 + 5.0 * (vpa - 5.0)
    if vpa < 13.0:
        return -6.0 + 6.0 * (vpa - 10.0)
    if vpa <= 28.0:
        return 14.0
    if vpa <= 45.0:
        return 8.0
    if vpa <= 80.0:
        return -8.0
    return -40.0


def train_volume_prior_score(row: dict[str, Any], priors: dict[str, dict[str, tuple[float, float]]]) -> float:
    if not priors:
        return 0.0
    vpa = row.get("self_volume_per_atom")
    if vpa is None or not math.isfinite(float(vpa)):
        return -2.0
    atom_count = row.get("atom_count_after_expansion") or row.get("self_parsed_sites")
    if atom_count is None:
        return 0.0
    sg = row.get("detected_sg")
    if sg is None:
        sg = parse_int_field(str(row.get("cif") or ""), "_symmetry_Int_Tables_number")
    if sg is None:
        return 0.0
    ab = atom_bucket(int(atom_count))
    system = crystal_system(int(sg))
    keys = [f"sg:{int(sg)}|atom:{ab}", f"system:{system}|atom:{ab}", f"atom:{ab}", "global"]
    median: float | None = None
    scale: float | None = None
    for key in keys:
        payload = priors.get(key)
        if payload and "vpa" in payload:
            median, scale = payload["vpa"]
            break
    if median is None or scale is None or scale <= 0:
        return 0.0
    z = abs(math.log(max(float(vpa), 1e-6) / max(float(median), 1e-6))) / max(math.log1p(float(scale) / max(float(median), 1e-6)), 0.15)
    return -min(8.0, float(z))


def self_score(
    row: dict[str, Any],
    *,
    geometry_distance_weight: float,
    original_rank_weight: float,
    train_volume_priors: dict[str, dict[str, tuple[float, float]]],
    train_volume_prior_weight: float,
    score_profile: str,
) -> float:
    score = 0.0
    for key, weight in (
        ("readable", 1000.0),
        ("formula_ok", 1000.0),
        ("atom_count_ok", 1000.0),
        ("composition_exact", 1000.0),
        ("sg_ok", 1000.0),
    ):
        score += weight if bool(row.get(key)) else -weight
    if score_profile == "strict_physical":
        score += strict_distance_score(row.get("self_min_distance"))
        score += strict_volume_score(row.get("self_volume_per_atom"))
    else:
        score += distance_score(row.get("self_min_distance"))
        score += volume_score(row.get("self_volume_per_atom"))
    prior_score = train_volume_prior_score(row, train_volume_priors)
    row["self_train_volume_prior_score"] = prior_score
    score += float(train_volume_prior_weight) * prior_score

    geom_dist = row.get("geometry_distance")
    if geom_dist is not None:
        score -= float(geometry_distance_weight) * float(geom_dist)
    score -= 0.25 * float(row.get("geometry_rank") or 0)
    score -= float(original_rank_weight) * float(row.get("original_rank") or row.get("rank") or 0)
    return float(score)


def rerank_global(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda row: (-float(row["self_score"]), int(row["original_rank"])))[:top_k]


def rerank_diverse(items: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    by_wa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(items, key=lambda item: int(item["original_rank"])):
        by_wa[str(row.get("canonical_wa_key") or "")].append(row)
    for rows in by_wa.values():
        rows.sort(key=lambda row: (-float(row["self_score"]), int(row["original_rank"])))

    wa_order = sorted(
        by_wa,
        key=lambda key: (
            -float(by_wa[key][0]["self_score"]),
            min(int(row["original_rank"]) for row in by_wa[key]),
        ),
    )
    out: list[dict[str, Any]] = []
    round_idx = 0
    seen_cifs: set[str] = set()
    while len(out) < top_k:
        added = False
        for wa_key in wa_order:
            rows = by_wa[wa_key]
            if round_idx >= len(rows):
                continue
            row = rows[round_idx]
            cif = str(row.get("cif") or "")
            if cif in seen_cifs:
                continue
            seen_cifs.add(cif)
            out.append(row)
            added = True
            if len(out) >= top_k:
                break
        if not added:
            break
        round_idx += 1
    return out


def score_group(payload: tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, tuple[float, float]]]]) -> list[dict[str, Any]]:
    sample_id, rows, config, train_volume_priors = payload
    scored: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["original_rank"] = int(row.get("rank", 10**9))
        item.update(cif_self_features(str(row.get("cif") or "")))
        item["self_score"] = self_score(
            item,
            geometry_distance_weight=float(config["geometry_distance_weight"]),
            original_rank_weight=float(config["original_rank_weight"]),
            train_volume_priors=train_volume_priors,
            train_volume_prior_weight=float(config["train_volume_prior_weight"]),
            score_profile=str(config["score_profile"]),
        )
        scored.append(item)
    if str(config["mode"]) == "global":
        selected = rerank_global(scored, int(config["top_k"]))
    else:
        selected = rerank_diverse(scored, int(config["top_k"]))
    out_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        out = dict(row)
        out["sample_id"] = sample_id
        out["rank"] = rank
        out_rows.append(out)
    return out_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    return {
        "rows": len(rows),
        "readable": sum(bool(row.get("readable")) for row in rows) / total,
        "formula_ok": sum(bool(row.get("formula_ok")) for row in rows) / total,
        "atom_count_ok": sum(bool(row.get("atom_count_ok")) for row in rows) / total,
        "composition_exact": sum(bool(row.get("composition_exact")) for row in rows) / total,
        "sg_ok": sum(bool(row.get("sg_ok")) for row in rows) / total,
        "min_distance_mean": None
        if not [row.get("self_min_distance") for row in rows if row.get("self_min_distance") is not None]
        else sum(float(row["self_min_distance"]) for row in rows if row.get("self_min_distance") is not None)
        / len([row for row in rows if row.get("self_min_distance") is not None]),
        "volume_per_atom_mean": None
        if not [row.get("self_volume_per_atom") for row in rows if row.get("self_volume_per_atom") is not None]
        else sum(float(row["self_volume_per_atom"]) for row in rows if row.get("self_volume_per_atom") is not None)
        / len([row for row in rows if row.get("self_volume_per_atom") is not None]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GT-free self-score and reorder rendered opentry_3 CIF candidates.")
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--mode", choices=["global", "diverse"], default="diverse")
    parser.add_argument("--geometry-distance-weight", type=float, default=5.0)
    parser.add_argument("--original-rank-weight", type=float, default=0.02)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--train-volume-prior-weight", type=float, default=0.0)
    parser.add_argument("--score-profile", choices=["standard", "strict_physical"], default="standard")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    start_time = time.time()
    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.rendered_jsonl)
    train_volume_priors = build_volume_priors(args.train_jsonl)
    by_id = grouped(rows)
    reranked: list[dict[str, Any]] = []
    config = {
        "top_k": int(args.top_k),
        "mode": str(args.mode),
        "geometry_distance_weight": float(args.geometry_distance_weight),
        "original_rank_weight": float(args.original_rank_weight),
        "train_volume_prior_weight": float(args.train_volume_prior_weight),
        "score_profile": str(args.score_profile),
    }
    payloads = [(sample_id, by_id[sample_id], config, train_volume_priors) for sample_id in sorted(by_id)]
    progress_every = max(1, int(args.progress_every))
    if int(args.workers) <= 1:
        for done, payload in enumerate(payloads, start=1):
            reranked.extend(score_group(payload))
            if done % progress_every == 0 or done == len(payloads):
                print(
                    json.dumps(
                        {
                            "completed_samples": done,
                            "total_samples": len(payloads),
                            "elapsed_s": time.time() - start_time,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    else:
        with mp.Pool(processes=int(args.workers)) as pool:
            for done, selected_rows in enumerate(pool.imap_unordered(score_group, payloads, chunksize=4), start=1):
                reranked.extend(selected_rows)
                if done % progress_every == 0 or done == len(payloads):
                    print(
                        json.dumps(
                            {
                                "completed_samples": done,
                                "total_samples": len(payloads),
                                "elapsed_s": time.time() - start_time,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    reranked.sort(key=lambda row: (str(row["sample_id"]), int(row["rank"])))
    write_jsonl(out_dir / "rendered_topk_selfscore.jsonl", reranked)
    summary = {
        "input": str(args.rendered_jsonl),
        "top_k": int(args.top_k),
        "mode": str(args.mode),
        "geometry_distance_weight": float(args.geometry_distance_weight),
        "original_rank_weight": float(args.original_rank_weight),
        "score_profile": str(args.score_profile),
        "train_jsonl": None if args.train_jsonl is None else str(args.train_jsonl),
        "train_volume_prior_weight": float(args.train_volume_prior_weight),
        "train_volume_prior_buckets": len(train_volume_priors),
        "workers": int(args.workers),
        "samples": len(by_id),
        "input_rows": len(rows),
        "output_rows": len(reranked),
        "overall": summarize(reranked),
        "rank1": summarize([row for row in reranked if int(row["rank"]) == 1]),
        "rank_le_5": summarize([row for row in reranked if int(row["rank"]) <= 5]),
        "rank_le_20": summarize([row for row in reranked if int(row["rank"]) <= 20]),
    }
    write_json(out_dir / "selfscore_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
