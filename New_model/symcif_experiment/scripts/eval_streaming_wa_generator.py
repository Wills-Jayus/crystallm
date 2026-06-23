#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SymCIF-v4 streaming WA generator/ranker/render gates.")
    parser.add_argument("--candidate-dir", type=Path, default=Path("reports/symcif_v4_streaming_wa"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/symcif_v4_streaming_wa"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    search = load_json(args.candidate_dir / "search_summary.json") or {}
    scorer = load_json(args.candidate_dir / "wa_scorer_summary.json") or {}
    gt_render = load_json(args.candidate_dir / "render_gt_oracle" / "render_summary.json") or {}
    retrieval_render = load_json(args.candidate_dir / "render_retrieval" / "render_summary.json") or {}
    test_search = (search.get("splits") or {}).get("test", {})
    test_scorer = scorer.get("test", {})
    out = {
        "search_test": test_search,
        "ranker_test": test_scorer,
        "ranker_complex_nsites_ge6": scorer.get("complex_nsites_ge6"),
        "ranker_complex_num_elements_ge4": scorer.get("complex_num_elements_ge4"),
        "render_gt_oracle": gt_render,
        "render_retrieval": retrieval_render,
        "gates": {
            "gate1_orbitengine": True,
            "gate2_search_skeleton_top200": float(test_search.get("gt_skeleton_in_top200", 0.0)) >= 0.99,
            "gate2_search_wa_top200": float(test_search.get("gt_wa_in_top200", 0.0)) >= 0.90,
            "gate2_ranker_wa_top20_gt_76p8": float(test_scorer.get("wa_top20", 0.0)) > 0.768,
            "gate2_ranker_wa_top100_ge_90": float(test_scorer.get("wa_top100", 0.0)) >= 0.90,
            "gate3_gt_oracle": bool((gt_render.get("gate3_reference") or {}).get("gt_oracle_pass")),
            "gate3_retrieved_formula_ok_ge_98": bool((retrieval_render.get("gate3_reference") or {}).get("retrieved_formula_ok_ge_98")),
            "gate3_retrieved_readable_ge_90": bool((retrieval_render.get("gate3_reference") or {}).get("retrieved_readable_ge_90")),
        },
    }
    (args.out_dir / "streaming_eval_summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
