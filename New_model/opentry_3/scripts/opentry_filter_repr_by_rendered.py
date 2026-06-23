#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3").resolve()


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


def sample_id_from_repr(row: dict[str, Any]) -> str:
    keys = dict(row.get("keys") or {})
    return str(keys.get("sample_id") or row.get("sample_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a Wyckoff repr JSONL to sample IDs present in rendered candidates.")
    parser.add_argument("--repr-jsonl", type=Path, required=True)
    parser.add_argument("--rendered-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-name", default="subset.jsonl")
    args = parser.parse_args()

    out_dir = ensure_under_opentry(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_rows = read_jsonl(args.rendered_jsonl)
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for row in rendered_rows:
        sid = str(row.get("sample_id") or "")
        if sid and sid not in seen:
            seen.add(sid)
            ordered_ids.append(sid)
    repr_by_id = {sample_id_from_repr(row): row for row in read_jsonl(args.repr_jsonl)}
    subset: list[dict[str, Any]] = []
    missing: list[str] = []
    for sid in ordered_ids:
        row = repr_by_id.get(sid)
        if row is None:
            missing.append(sid)
            continue
        subset.append(row)
    write_jsonl(out_dir / args.out_name, subset)
    summary = {
        "repr_jsonl": str(args.repr_jsonl),
        "rendered_jsonl": str(args.rendered_jsonl),
        "rendered_unique_sample_ids": len(ordered_ids),
        "written_records": len(subset),
        "missing_records": len(missing),
        "missing_sample_ids_preview": missing[:20],
    }
    write_json(out_dir / "filter_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
