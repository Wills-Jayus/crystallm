#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def under_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_10: {resolved}")
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def candidate_text(row: dict[str, Any]) -> str:
    for key in ("generated_text", "cif", "generated_cif", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def source_rank(row: dict[str, Any]) -> int | None:
    for key in ("source_rank", "rank", "gen_index"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_prompt(cif: str) -> str:
    data_line: str | None = None
    sg_line: str | None = None
    for raw in cif.splitlines():
        line = raw.strip()
        if data_line is None and line.startswith("data_"):
            data_line = line
        if sg_line is None and line.startswith("_symmetry_space_group_name_H-M"):
            sg_line = line
        if data_line is not None and sg_line is not None:
            break
    if data_line is None:
        raise RuntimeError("candidate CIF has no data_ line")
    if sg_line is None:
        raise RuntimeError("candidate CIF has no _symmetry_space_group_name_H-M line")
    sg_line = re.sub(r"\s+", "   ", sg_line, count=1)
    return f"{data_line}\n{sg_line}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GT-SG prompt files from existing CrystaLLM anchor candidate prefixes.")
    parser.add_argument("--anchor-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    anchor = Path(args.anchor_jsonl)
    out_dir = under_root(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts: dict[str, dict[str, str]] = {}
    total_rows = 0
    rank0_rows = 0
    with anchor.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            rank = source_rank(row)
            if rank not in (0, 1):
                continue
            material_id = str(row.get("material_id") or (row.get("metadata") or {}).get("material_id") or "")
            sample_id = str(row.get("sample_id") or "")
            if not material_id:
                raise RuntimeError(f"anchor row missing material_id: {row.get('candidate_id')}")
            if material_id in prompts:
                continue
            prompt = extract_prompt(candidate_text(row))
            prompts[material_id] = {"sample_id": sample_id, "prompt": prompt}
            rank0_rows += 1

    if not prompts:
        raise RuntimeError("no rank0/rank1 prompts extracted")
    for material_id, payload in sorted(prompts.items()):
        (out_dir / f"{material_id}.txt").write_text(payload["prompt"], encoding="utf-8")

    manifest = {
        "created_at": now_iso(),
        "dataset": str(args.dataset),
        "anchor_jsonl": str(anchor.resolve()),
        "out_dir": str(out_dir.resolve()),
        "total_anchor_rows": int(total_rows),
        "prompt_files": int(len(prompts)),
        "rank0_or_rank1_rows_seen": int(rank0_rows),
        "filename_policy": "material_id.txt to match generate_cifs_from_prompts_dir.py output stems",
        "prompt_source": "existing GT-SG anchor candidate prefix; no test target structures or labels read",
    }
    under_root(Path(args.manifest)).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
