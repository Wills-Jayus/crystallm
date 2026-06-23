#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()
SYMCIF_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/symcif_experiment").resolve()
for path in (OPENTRY_ROOT / "scripts", SYMCIF_ROOT / "src", SYMCIF_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import opentry_build_geometry_compat_features as compat_features  # noqa: E402
import opentry_build_positive_geometry_residual_examples as pos_geom  # noqa: E402
import opentry_pair_delta_geometry_variants as delta_utils  # noqa: E402
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


def grouped(rows: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    out: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        out.setdefault(str(row.get("sample_id") or ""), []).append(row)
    for values in out.values():
        values.sort(key=lambda item: int(item.get("rank", item.get("original_rank", 10**9))))
    return out


def load_repr(path: Path, max_records: int = 0) -> OrderedDict[str, dict[str, Any]]:
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in read_jsonl(path):
        sample_id = str(row["keys"]["sample_id"])
        out[sample_id] = row
        if int(max_records) > 0 and len(out) >= int(max_records):
            break
    return out


def stable_u01(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return (value + 0.5) / float(1 << 64)


def lattice_volume(lat: dict[str, Any]) -> float:
    a, b, c = float(lat["a"]), float(lat["b"]), float(lat["c"])
    alpha = math.radians(float(lat["alpha"]))
    beta = math.radians(float(lat["beta"]))
    gamma = math.radians(float(lat["gamma"]))
    term = 1.0 + 2.0 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
    term -= math.cos(alpha) ** 2 + math.cos(beta) ** 2 + math.cos(gamma) ** 2
    return max(1.0e-6, a * b * c * math.sqrt(max(1.0e-8, term)))


def row_bucket(row_count: int) -> str:
    n = int(row_count)
    if n <= 4:
        return "r1_4"
    if n <= 6:
        return "r5_6"
    if n <= 8:
        return "r7_8"
    if n <= 10:
        return "r9_10"
    return "r11_plus"


def median(values: list[float], default: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float(default)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def split_wa(wa_key: str) -> list[tuple[str, str]]:
    return pos_geom.split_wa_key(str(wa_key or ""))


def candidate_row_count(row: dict[str, Any]) -> int:
    return len(split_wa(str(row.get("canonical_wa_key") or "")))


class TrainPrior:
    def __init__(self) -> None:
        self.lattice_by_key: dict[tuple[Any, ...], list[list[float]]] = defaultdict(list)
        self.vpa_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self.param_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self.summary: dict[str, Any] = {}

    def lattice_keys(self, sg: int, row_count: int) -> list[tuple[Any, ...]]:
        system = compat_features.crystal_system(int(sg))
        bucket = row_bucket(row_count)
        return [
            ("sg_row", int(sg), row_count),
            ("sg_bucket", int(sg), bucket),
            ("system_row", system, row_count),
            ("system_bucket", system, bucket),
            ("system", system),
            ("global",),
        ]

    def param_keys(self, orbit_id: str, symbol: str, site_symmetry: str | None = None) -> list[tuple[Any, ...]]:
        return [
            ("orbit_symbol", str(orbit_id), str(symbol)),
            ("site_symbol", str(site_symmetry or ""), str(symbol)),
            ("symbol", str(symbol)),
            ("global_symbol", str(symbol)),
        ]

    def select_lattice_base(self, sg: int, row_count: int, token: str) -> tuple[list[float], float, tuple[Any, ...]]:
        for key in self.lattice_keys(sg, row_count):
            values = self.lattice_by_key.get(key) or []
            if values:
                idx = int(stable_u01(token, key, "lat") * len(values)) % len(values)
                vpa = median(self.vpa_by_key.get(key) or [], 24.0)
                return list(values[idx]), vpa, key
        return [math.log(4.0), math.log(4.0), math.log(4.0), 90.0 / 180.0, 90.0 / 180.0, 90.0 / 180.0], 24.0, ("fallback",)

    def sample_param(self, mode: str, orbit_id: str, symbol: str, site_symmetry: str, token: str, jitter: float) -> tuple[float, tuple[Any, ...]]:
        if mode == "uniform":
            return stable_u01(token, orbit_id, symbol), ("uniform",)
        for key in self.param_keys(orbit_id, symbol, site_symmetry):
            values = self.param_by_key.get(key) or []
            if values:
                idx = int(stable_u01(token, key, "param") * len(values)) % len(values)
                value = float(values[idx])
                if float(jitter) > 0:
                    value = (value + (stable_u01(token, key, "jit") - 0.5) * float(jitter)) % 1.0
                return value, key
        return stable_u01(token, orbit_id, symbol), ("uniform_fallback",)


def build_prior(train_features: list[dict[str, Any]], positive_only: bool) -> TrainPrior:
    prior = TrainPrior()
    rows_used = 0
    positives = 0
    for row in train_features:
        if positive_only and not bool(row.get("label_match")):
            continue
        if not row.get("geometry_lattice") or not row.get("geometry_params"):
            continue
        rows_used += 1
        positives += int(bool(row.get("label_match")))
        sg = int(row.get("sg") or 0)
        rc = int(row.get("target_row_count") or candidate_row_count(row))
        lat = dict(row["geometry_lattice"])
        lat_vec = delta_utils.lattice_vector(lat)
        vpa = lattice_volume(lat) / max(1, int(row.get("atom_count") or 1))
        for key in prior.lattice_keys(sg, rc):
            prior.lattice_by_key[key].append(lat_vec)
            prior.vpa_by_key[key].append(vpa)
        params = dict(row.get("geometry_params") or {})
        for idx, (orbit_id, _element) in enumerate(split_wa(str(row.get("canonical_wa_key") or ""))):
            row_params = dict(params.get(str(idx)) or params.get(idx) or {})
            site_symmetry = ""
            if "|sym=" in orbit_id:
                site_symmetry = orbit_id.rsplit("|sym=", 1)[-1]
            for symbol, value in row_params.items():
                for key in prior.param_keys(orbit_id, str(symbol), site_symmetry):
                    prior.param_by_key[key].append(float(value) % 1.0)
    prior.summary = {
        "rows_used": rows_used,
        "positive_rows_used": positives,
        "positive_only": bool(positive_only),
        "lattice_keys": len(prior.lattice_by_key),
        "param_keys": len(prior.param_by_key),
        "lattice_entries": sum(len(v) for v in prior.lattice_by_key.values()),
        "param_entries": sum(len(v) for v in prior.param_by_key.values()),
    }
    return prior


def scale_lattice(vec: list[float], sg: int, atom_count: int, target_vpa: float, token: str, jitter: float) -> dict[str, float]:
    vals = list(vec)
    if float(jitter) > 0:
        vals[0] += (stable_u01(token, "a") - 0.5) * float(jitter)
        vals[1] += (stable_u01(token, "b") - 0.5) * float(jitter)
        vals[2] += (stable_u01(token, "c") - 0.5) * float(jitter)
    lat = v2.lattice_from_target(vals, int(sg))
    current = lattice_volume(lat)
    target = max(1.0, float(target_vpa) * max(1, int(atom_count)))
    scale = (target / max(1.0e-6, current)) ** (1.0 / 3.0)
    scaled = dict(lat)
    for axis in ("a", "b", "c"):
        scaled[axis] = float(scaled[axis]) * float(scale)
    return scaled


def params_from_prior(engine: OrbitEngine, prior: TrainPrior, wa_key: str, mode: str, token: str, jitter: float) -> tuple[dict[str, dict[str, float]], list[str]]:
    params: dict[str, dict[str, float]] = {}
    param_sources: list[str] = []
    for idx, (orbit_id, _element) in enumerate(split_wa(wa_key)):
        orbit = engine.get_orbit_by_id(str(orbit_id))
        row_params: dict[str, float] = {}
        for symbol in list(orbit.free_symbols):
            value, key = prior.sample_param(mode, orbit_id, str(symbol), str(orbit.site_symmetry), f"{token}:{idx}:{symbol}", jitter)
            row_params[str(symbol)] = float(value) % 1.0
            param_sources.append(repr(key))
        params[str(idx)] = row_params
    return params, param_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Source-free train-prior geometry sampler for predicted Wyckoff W/A candidates.")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--input-rendered-jsonl", type=Path, required=True)
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--lookup-json", type=Path, default=SYMCIF_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-input-candidates-per-sample", type=int, default=20)
    parser.add_argument("--max-output-candidates-per-sample", type=int, default=50)
    parser.add_argument("--min-target-row-count", type=int, default=7)
    parser.add_argument("--mode", choices=["uniform", "orbit_prior"], default="uniform")
    parser.add_argument("--positive-only-prior", action="store_true")
    parser.add_argument("--samples-per-wa", type=int, default=5)
    parser.add_argument("--lattice-jitter", type=float, default=0.08)
    parser.add_argument("--param-jitter", type=float, default=0.04)
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine(args.lookup_json)
    prior = build_prior(read_jsonl(args.train_features), positive_only=bool(args.positive_only_prior))
    repr_rows = load_repr(args.repr_jsonl, max_records=int(args.max_records))
    rendered_by_id = grouped(read_jsonl(args.input_rendered_jsonl))

    out_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "selected_samples": len(repr_rows),
        "skipped_target_rows_lt_min": 0,
        "samples_with_input": 0,
        "samples_with_output": 0,
        "input_wa_candidates_seen": 0,
        "render_attempts": 0,
        "render_ok": 0,
        "deduped_cifs": 0,
        **{f"prior_{k}": v for k, v in prior.summary.items()},
    }
    for sample_id, repr_row in repr_rows.items():
        target_row_count = int(repr_row["row_count"])
        if target_row_count < int(args.min_target_row_count):
            stats["skipped_target_rows_lt_min"] += 1
            continue
        input_rows = rendered_by_id.get(sample_id, [])[: int(args.max_input_candidates_per_sample)]
        if input_rows:
            stats["samples_with_input"] += 1
        sg = int(repr_row["sg"])
        record_base = {
            "sample_id": sample_id,
            "formula_counts": dict(repr_row["formula_counts"]),
            "sg": sg,
            "sg_symbol": str(repr_row.get("sg_symbol") or ""),
            "atom_count": int(repr_row["atom_count"]),
        }
        generated: list[dict[str, Any]] = []
        seen_cifs: set[str] = set()
        seen_wa: set[str] = set()
        for source_row in input_rows:
            wa_key = str(source_row.get("canonical_wa_key") or "")
            if not wa_key or wa_key in seen_wa:
                continue
            seen_wa.add(wa_key)
            stats["input_wa_candidates_seen"] += 1
            row_count = candidate_row_count(source_row)
            for sample_idx in range(int(args.samples_per_wa)):
                token = f"{sample_id}:{wa_key}:{sample_idx}:{args.mode}"
                lat_vec, vpa, lat_key = prior.select_lattice_base(sg, row_count, token)
                lattice = scale_lattice(lat_vec, sg, int(repr_row["atom_count"]), vpa, token, float(args.lattice_jitter))
                params, param_sources = params_from_prior(
                    engine,
                    prior,
                    wa_key,
                    str(args.mode),
                    token,
                    float(args.param_jitter),
                )
                rows = pos_geom.rows_from_wa_key(engine, wa_key, params)
                stats["render_attempts"] += 1
                render = v2.render_candidate(
                    engine,
                    record_base,
                    {"rows": rows, "params": params, "lattice": lattice},
                    len(generated) + 1,
                    f"source_free_{args.mode}_{sample_idx}",
                )
                if not render.get("ok"):
                    continue
                cif = str(render.get("cif") or "")
                digest = delta_utils.cif_digest(cif)
                if digest in seen_cifs:
                    stats["deduped_cifs"] += 1
                    continue
                seen_cifs.add(digest)
                metric = validate_cif(cif, record_base["formula_counts"], sg)
                stats["render_ok"] += 1
                generated.append(
                    {
                        **{key: value for key, value in source_row.items() if key not in {"rank", "cif"}},
                        "rank": len(generated) + 1,
                        "cif": cif,
                        "readable": bool(metric.get("readable")),
                        "formula_ok": bool(metric.get("formula_ok")),
                        "composition_exact": bool(metric.get("composition_exact")),
                        "atom_count_ok": bool(render.get("atom_count_ok")),
                        "detected_sg": metric.get("detected_sg"),
                        "sg_ok": bool(metric.get("sg_ok")),
                        "geometry_lattice": lattice,
                        "geometry_params": params,
                        "geometry_source": "source_free_train_prior",
                        "geometry_param_variant_mode": f"source_free_{args.mode}",
                        "geometry_lattice_mode": "source_free_train_prior_scaled",
                        "source_free_mode": str(args.mode),
                        "source_free_sample_index": int(sample_idx),
                        "source_free_lattice_key": repr(lat_key),
                        "source_free_param_source_keys": param_sources[:20],
                        "source_rank": int(source_row.get("rank", 0)),
                        "original_rank": int(source_row.get("rank", 0)),
                    }
                )
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
            "train_features": str(args.train_features),
            "input_rendered_jsonl": str(args.input_rendered_jsonl),
            "repr_jsonl": str(args.repr_jsonl),
            "max_records": int(args.max_records),
            "max_input_candidates_per_sample": int(args.max_input_candidates_per_sample),
            "max_output_candidates_per_sample": int(args.max_output_candidates_per_sample),
            "min_target_row_count": int(args.min_target_row_count),
            "mode": str(args.mode),
            "positive_only_prior": bool(args.positive_only_prior),
            "samples_per_wa": int(args.samples_per_wa),
            "lattice_jitter": float(args.lattice_jitter),
            "param_jitter": float(args.param_jitter),
        },
        "stats": stats,
        "rendered_rows": len(out_rows),
        "note": "Train-only prior; inference uses predicted W/A keys, formula, and GT-SG only. No source lattice/free-param copying from val candidates.",
    }
    write_json(out_dir / "source_free_prior_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
