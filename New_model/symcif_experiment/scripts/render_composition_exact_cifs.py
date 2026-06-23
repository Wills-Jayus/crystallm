#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from io import StringIO
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pymatgen.core import Composition  # type: ignore
from pymatgen.io.cif import CifParser  # type: ignore
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore

from symcif_v4.formula import normalize_formula_counts
from symcif_v4.render_cif import coords_by_site, lattice_from_dict, median_lattice, render_wa_cif
from symcif_v4.wa_table import WATableCandidate, gt_wa_key
from symcif_v4.wyckoff_table import WyckoffSiteToken, canonical_site_id, load_lookup
from train_skeleton_template_ranker import read_jsonl


warnings.filterwarnings("ignore")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_from_json(raw: dict[str, Any], lookup: Any | None = None, sg: int | None = None, letter: str | None = None) -> WyckoffSiteToken:
    if lookup is not None and sg is not None and letter is not None:
        template = lookup.get(int(sg), str(letter))
        site_symmetry = template.site_symmetry or raw.get("site_symmetry") or "UNKNOWN"
        enum = template.enumeration if template.enumeration is not None else raw.get("enumeration")
        return WyckoffSiteToken(
            sg=int(sg),
            letter=str(letter),
            multiplicity=int(template.multiplicity),
            site_symmetry=str(site_symmetry),
            enumeration=enum,
            free_mask=tuple(bool(v) for v in template.free_mask),  # type: ignore[arg-type]
            fixed_values=tuple(float(v) for v in template.fixed_values),  # type: ignore[arg-type]
            representative_expr=str(template.representative_expr),
            canonical_key=canonical_site_id(int(sg), int(template.multiplicity), str(letter), enum, str(site_symmetry)),
            is_fully_fixed=not any(bool(v) for v in template.free_mask),
        )
    return WyckoffSiteToken(
        sg=int(raw["sg"]),
        letter=str(raw["letter"]),
        multiplicity=int(raw["multiplicity"]),
        site_symmetry=str(raw.get("site_symmetry") or "UNKNOWN"),
        enumeration=raw.get("enumeration"),
        free_mask=tuple(bool(v) for v in raw.get("free_mask", [False, False, False])),  # type: ignore[arg-type]
        fixed_values=tuple(float(v) for v in raw.get("fixed_values", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
        representative_expr=str(raw.get("representative_expr", "")),
        canonical_key=str(raw["canonical_key"]),
        is_fully_fixed=bool(raw.get("is_fully_fixed", False)),
        max_repeat=int(raw.get("max_repeat", 1)),
    )


def wa_from_candidate(raw: dict[str, Any], lookup: Any | None = None) -> WATableCandidate:
    rows = []
    for item in raw["rows"]:
        token_raw = item.get("token") or {
            "sg": raw["sg"],
            "letter": item["letter"],
            "multiplicity": item["multiplicity"],
            "site_symmetry": item.get("site_symmetry"),
            "enumeration": item.get("enumeration"),
            "free_mask": item.get("free_mask", [False, False, False]),
            "fixed_values": [0.0, 0.0, 0.0],
            "canonical_key": item["canonical_key"],
        }
        rows.append((str(item["element"]), token_from_json(token_raw, lookup=lookup, sg=int(raw["sg"]), letter=str(item["letter"]))))
    return WATableCandidate(
        sg=int(raw["sg"]),
        formula_counts={str(k): int(v) for k, v in raw["formula_counts"].items()},
        rows=rows,
        skeleton_key=str(raw["skeleton_key"]),
        assignment_key=str(raw["assignment_key"]),
        wa_key=str(raw["wa_key"]),
        source=str(raw.get("source", "ranked")),
        score=None if raw.get("score") is None else float(raw["score"]),
    )


def comp_counts_from_structure(structure: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for element, amount in structure.composition.as_dict().items():
        out[str(element)] = int(round(float(amount)))
    return dict(sorted(out.items()))


def same_counts(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return normalize_formula_counts(a) == normalize_formula_counts(b)


def evaluate_cif(cif_text: str, target_counts: dict[str, int], target_sg: int) -> dict[str, Any]:
    metric = {
        "readable": False,
        "formula_ok": False,
        "space_group_ok": False,
        "composition_exact": False,
        "atom_count_after_expansion": None,
        "detected_sg": None,
        "error": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = CifParser(StringIO(cif_text))
            if hasattr(parser, "parse_structures"):
                structures = parser.parse_structures(primitive=False)
            else:
                structures = parser.get_structures(primitive=False)
        if not structures:
            raise ValueError("no structures parsed")
        structure = structures[0]
        metric["readable"] = True
        got_counts = comp_counts_from_structure(structure)
        metric["formula_ok"] = same_counts(got_counts, target_counts)
        metric["composition_exact"] = metric["formula_ok"]
        metric["atom_count_after_expansion"] = int(len(structure))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detected = int(SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0).get_space_group_number())
            metric["detected_sg"] = detected
            metric["space_group_ok"] = detected == int(target_sg)
        except Exception as exc:
            metric["error"] = f"sg_detect:{type(exc).__name__}:{exc}"
    except Exception as exc:
        metric["error"] = f"{type(exc).__name__}:{exc}"
    return metric


def train_retrieval_indexes(train_rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    by_wa: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_skel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        by_wa[gt_wa_key(row)].append(row)
        by_skel[f"{int(row['sg'])}|{row['skeleton_template_key']}"].append(row)
        by_sg[int(row["sg"])].append(row)
    return by_wa, by_skel, by_sg


def choose_sources(
    *,
    row: dict[str, Any],
    wa: WATableCandidate,
    coord_mode: str,
    by_wa: dict[str, list[dict[str, Any]]],
    by_skel: dict[str, list[dict[str, Any]]],
    by_sg: dict[int, list[dict[str, Any]]],
) -> tuple[dict[str, list[tuple[float, float, float]]] | None, Any, str]:
    if coord_mode == "gt_oracle" and wa.wa_key == gt_wa_key(row):
        return coords_by_site(row), lattice_from_dict(row["lattice"]), "gt_oracle_exact_wa"
    if coord_mode in {"gt_oracle", "retrieval"}:
        if by_wa.get(wa.wa_key):
            src = by_wa[wa.wa_key][0]
            return coords_by_site(src), lattice_from_dict(src["lattice"]), "train_same_wa"
        skey = f"{wa.sg}|{wa.skeleton_key}"
        if by_skel.get(skey):
            src = by_skel[skey][0]
            return coords_by_site(src), lattice_from_dict(src["lattice"]), "train_same_skeleton"
    if by_sg.get(int(wa.sg)):
        return None, median_lattice(by_sg[int(wa.sg)]), "sg_median_lattice"
    return None, median_lattice([]), "default_safe"


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"generations": len(metrics)}
    for key in ("to_cif_success", "readable", "formula_ok", "space_group_ok", "composition_exact"):
        out[key] = sum(bool(m.get(key)) for m in metrics) / max(1, len(metrics))
    out["samples_with_any_rendered"] = len({m["sample_id"] for m in metrics if m.get("to_cif_success")})
    out["samples_with_formula_ok"] = len({m["sample_id"] for m in metrics if m.get("formula_ok")})
    out["samples_with_sg_ok"] = len({m["sample_id"] for m in metrics if m.get("space_group_ok")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Render composition-exact WA candidates to CIF and audit CIF-level closure.")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1" / "test_wa_predictions.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--coord-mode", default="retrieval", choices=["gt_oracle", "retrieval", "default_safe"])
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "composition_exact_v1")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out_dir / "render_cif_examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.structured_root / f"{args.split}.jsonl")
    row_by_id = {row["sample_id"]: row for row in rows}
    train_rows = read_jsonl(args.structured_root / "train.jsonl")
    lookup = load_lookup(args.lookup_json)
    by_wa, by_skel, by_sg = train_retrieval_indexes(train_rows)
    preds = read_jsonl(args.predictions)
    metrics: list[dict[str, Any]] = []
    rendered_lines: list[dict[str, Any]] = []
    example_count = 0
    for pred in preds:
        row = row_by_id[pred["sample_id"]]
        target_counts = normalize_formula_counts(row["formula_counts"])
        for rank, cand_raw in enumerate(pred["ranked_wa_candidates"][: args.top_k], start=1):
            wa = wa_from_candidate(cand_raw, lookup=lookup)
            coord_source, lattice, source = choose_sources(
                row=row,
                wa=wa,
                coord_mode=args.coord_mode,
                by_wa=by_wa,
                by_skel=by_skel,
                by_sg=by_sg,
            )
            cif_text, meta = render_wa_cif(
                wa,
                formula=row["formula"],
                sg_symbol=row["sg_symbol"],
                lattice=lattice,
                coord_source=coord_source,
                candidate_index=rank,
                sample_id=row["sample_id"],
            )
            metric = {
                "sample_id": row["sample_id"],
                "rank": rank,
                "wa_key": wa.wa_key,
                "gt_wa": gt_wa_key(row),
                "coord_mode": args.coord_mode,
                "coord_source": source,
                **meta,
            }
            if cif_text is not None:
                metric.update(evaluate_cif(cif_text, target_counts, int(row["sg"])))
                if example_count < args.examples:
                    path = examples_dir / f"{example_count:03d}_{row['sample_id']}_r{rank}.cif"
                    path.write_text(cif_text, encoding="utf-8")
                    example_count += 1
            else:
                metric.update({"readable": False, "formula_ok": False, "space_group_ok": False, "composition_exact": False})
            metrics.append(metric)
            rendered_lines.append({**metric, "cif": cif_text})
    with (args.out_dir / "rendered_test_topk.jsonl").open("w", encoding="utf-8") as f:
        for row in rendered_lines:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "split": args.split,
        "coord_mode": args.coord_mode,
        "top_k": args.top_k,
        "overall": summarize(metrics),
        "rank1": summarize([m for m in metrics if int(m["rank"]) == 1]),
        "top5": summarize([m for m in metrics if int(m["rank"]) <= 5]),
        "top20": summarize([m for m in metrics if int(m["rank"]) <= 20]),
        "source_counts": dict(sorted({m["coord_source"]: sum(1 for x in metrics if x["coord_source"] == m["coord_source"]) for m in metrics}.items())),
    }
    (args.out_dir / "render_cif_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
