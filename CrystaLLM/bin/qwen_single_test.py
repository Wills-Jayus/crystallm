#!/usr/bin/env python3
"""
Quick smoke test for the optimizer LLM endpoint (OpenAI-compatible /v1/chat/completions).

Example:
  python CrystaLLM/bin/qwen_single_test.py \
    --api-base http://127.0.0.1:8000/v1 \
    --model Qwen3-30B-A3B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json

from qwen_client import QwenClient, QwenConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-call test for QwenClient.optimize_prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-base", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = QwenConfig()
    if args.api_base:
        cfg.api_base = args.api_base
    if args.api_key:
        cfg.api_key = args.api_key
    if args.model:
        cfg.model = args.model
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens

    last_prompt_lines = [
        "data_Na2Cl2",
        "_symmetry_space_group_name_H-M   P4/mmm",
        "_chemical_formula_sum   Na2 Cl2",
    ]
    evaluator_summary = {
        "round": 1,
        "score_property": "bandgap",
        "score_goal": "max",
        "metrics": {
            "n_structures": 8,
            "n_validation_pass": 5,
            "validation_pass_ratio": 0.625,
            "n_scored_ok": 8,
            "check_composition": True,
        },
        "top_structures": [
            {
                "sample_id": "sample_1",
                "cif_path": "round_01/cifs/sample_1.cif",
                "validation_ok": True,
                "properties": {"formation_energy": -2.1, "bandgap": 3.4},
                "score_value": 3.4,
            }
        ],
        "validation_failure_reasons_top": [{"reason": "space group inconsistent", "count": 2}],
    }

    client = QwenClient(cfg)
    parsed, raw_response, raw_text = client.optimize_prompt(last_prompt_lines, evaluator_summary)
    print(json.dumps({"parsed": parsed, "raw_text": raw_text, "response": raw_response}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

