#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from symcif_v4.orbit_engine import OrbitEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OrbitEngine expanded multiplicities for every orbit.")
    parser.add_argument("--lookup-json", type=Path, default=PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json")
    parser.add_argument("--structured-root", type=Path, default=PROJECT_ROOT / "data" / "structured_symcif_v3")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "symcif_v4_orbit_engine")
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrbitEngine.from_structured_root(args.lookup_json, args.structured_root)
    params = {"x": 0.123, "y": 0.234, "z": 0.345}
    rows: list[dict[str, Any]] = []
    total = passed = 0
    failed_examples: list[dict[str, Any]] = []
    for sg in range(1, 231):
        for orbit in engine.get_orbits(sg):
            total += 1
            try:
                expanded = engine.expand_orbit(orbit, params, symprec=1e-5)
                ok = len(expanded) == int(orbit.multiplicity)
                reason = None if ok else "expanded_count_mismatch"
            except Exception as exc:  # noqa: BLE001
                expanded = []
                ok = False
                reason = f"{type(exc).__name__}:{exc}"
            if ok:
                passed += 1
            else:
                item = {
                    "sg": orbit.sg,
                    "letter": orbit.letter,
                    "multiplicity": orbit.multiplicity,
                    "expanded_count": len(expanded),
                    "representative_expr": list(orbit.representative_expr),
                    "reason": reason,
                }
                if len(failed_examples) < args.max_examples:
                    failed_examples.append(item)
            rows.append(
                {
                    "sg": orbit.sg,
                    "letter": orbit.letter,
                    "multiplicity": orbit.multiplicity,
                    "expanded_count": len(expanded),
                    "passed": ok,
                    "reason": reason,
                }
            )
    summary = {
        "total_orbits": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / max(1, total),
        "failed_examples": failed_examples,
    }
    (args.out_dir / "orbit_multiplicity_test.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "orbit_multiplicity_per_orbit.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

