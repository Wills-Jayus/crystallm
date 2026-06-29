#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


OUT_DIR = Path("/data/users/xsw/autodlmini/model/New_model/opentry_11")
_GT_CACHE: dict[str, Any] = {}
_MATCHER: Any = None


class MatchTimeout(TimeoutError):
    pass


def timeout_handler(signum: int, frame: Any) -> None:
    raise MatchTimeout("StructureMatcher timeout")


def ensure_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_11: {resolved}")
    return resolved


def get_matcher() -> Any:
    global _MATCHER
    if _MATCHER is None:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        _MATCHER = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    return _MATCHER


def load_gt(path: str) -> Any:
    if path not in _GT_CACHE:
        from pymatgen.core import Structure

        _GT_CACHE[path] = Structure.from_file(path)
    return _GT_CACHE[path]


def eval_one(row: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_s)
    after_match: bool | None
    error: str | None = None
    try:
        from pymatgen.core import Structure

        pred = Structure.from_str(row["after_cif"], fmt="cif")
        gt = load_gt(row["gt_path"])
        after_match = bool(get_matcher().fit(pred, gt))
    except Exception as exc:
        after_match = None
        error = exc.__class__.__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return {
        "sample_id": row["sample_id"],
        "material_id": row["material_id"],
        "rank": int(row["rank"]),
        "target_rows_ge7": bool(row["target_rows_ge7"]),
        "after_match": after_match,
        "collision_proxy": row.get("collision_proxy"),
        "min_radius_ratio": row.get("min_radius_ratio"),
        "error": error,
    }


def worker(payload: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, timeout_s = payload
    return eval_one(row, timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = ensure_out(Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with output_path.open("w", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            for result in ex.map(worker, ((row, int(args.timeout)) for row in rows), chunksize=16):
                out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
