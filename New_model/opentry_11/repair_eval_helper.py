#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


OUT_DIR = Path("/data/users/xsw/autodlmini/model/New_model/opentry_11")


class EvalTimeout(TimeoutError):
    pass


def timeout_handler(signum: int, frame: Any) -> None:
    raise EvalTimeout("StructureMatcher timeout")


def ensure_out(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = OUT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_11: {resolved}")
    return resolved


def match_one(cif: str, gt_cif: str, timeout_s: int) -> bool | None:
    old = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_s)
    try:
        pred = Structure.from_str(cif, fmt="cif")
        gt = Structure.from_str(gt_cif, fmt="cif")
        matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        return bool(matcher.fit(pred, gt))
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = ensure_out(Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            before = match_one(row["before_cif"], row["gt_cif"], int(args.timeout))
            after = match_one(row["after_cif"], row["gt_cif"], int(args.timeout))
            dst.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "material_id": row["material_id"],
                        "rank": row["rank"],
                        "target_rows_ge7": row["target_rows_ge7"],
                        "before_match": before,
                        "after_match": after,
                        "collision_proxy": row.get("collision_proxy"),
                        "min_radius_ratio": row.get("min_radius_ratio"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
