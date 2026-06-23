#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

OPENTRY_ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_2").resolve()


def _ensure_under_opentry(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OPENTRY_ROOT and OPENTRY_ROOT not in resolved.parents:
        raise ValueError(f"refusing to write outside opentry_2: {resolved}")
    return resolved


def _load_pkl(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _dump_pkl(path: Path, obj: Any) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_positive_raw(args: argparse.Namespace) -> None:
    records = []
    by_id: dict[str, list[dict[str, Any]]] = {}
    with Path(args.records_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec.get("label", 0)) <= 0:
                continue
            by_id.setdefault(str(rec["id"]), []).append(rec)
    for mid, items in sorted(by_id.items()):
        items = sorted(items, key=lambda r: (float(r.get("rms") if r.get("rms") is not None else 999.0), int(r.get("idx", 999))))
        for rec in items[: max(1, int(args.max_per_id))]:
            path = Path(str(rec["path"]))
            if not path.is_file():
                continue
            records.append((f"{mid}__pos{int(rec.get('idx', 0))}", path.read_text(encoding="utf-8", errors="replace")))
    _dump_pkl(Path(args.out), records)
    print(json.dumps({"positive_ids": len(by_id), "positive_rows": len(records), "out": str(Path(args.out).resolve())}, indent=2))


def combine_preprocessed(args: argparse.Namespace) -> None:
    base = list(_load_pkl(Path(args.base_train_preprocessed)))
    positives = list(_load_pkl(Path(args.positive_preprocessed)))
    combined = base + positives * max(1, int(args.positive_repeat))
    _dump_pkl(Path(args.out), combined)
    print(
        json.dumps(
            {
                "base_rows": len(base),
                "positive_rows": len(positives),
                "positive_repeat": int(args.positive_repeat),
                "combined_rows": len(combined),
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build opentry positive self-training CIF datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pos = sub.add_parser("positive-raw")
    p_pos.add_argument("--records-jsonl", required=True)
    p_pos.add_argument("--out", required=True)
    p_pos.add_argument("--max-per-id", type=int, default=3)
    p_pos.set_defaults(func=build_positive_raw)

    p_combine = sub.add_parser("combine-preprocessed")
    p_combine.add_argument("--base-train-preprocessed", required=True)
    p_combine.add_argument("--positive-preprocessed", required=True)
    p_combine.add_argument("--positive-repeat", type=int, default=5)
    p_combine.add_argument("--out", required=True)
    p_combine.set_defaults(func=combine_preprocessed)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
