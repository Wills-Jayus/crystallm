#!/usr/bin/env python3
"""
ZMQ client for `resources/alignn_zmq_server_multi.py`.

Primary uses:
  - Provide a small reusable helper to score CIFs via ALIGNN
  - Offer a CLI for quick manual scoring / debugging
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import zmq


def score_cif_via_alignn(
    cif_text: str,
    host: str,
    port: int,
    properties: List[str],
    timeout_ms: int = 10000,
) -> Dict[str, Any]:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://{host}:{port}")

    req = {"cif": cif_text, "properties": properties}
    socket.send_string(json.dumps(req, ensure_ascii=False))
    raw = socket.recv()
    return json.loads(raw.decode("utf-8"))


def score_cif_files_via_alignn(
    cif_paths: Iterable[Path],
    host: str,
    port: int,
    properties: List[str],
    timeout_ms: int = 10000,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in cif_paths:
        cif_text = p.read_text(encoding="utf-8", errors="ignore")
        out = score_cif_via_alignn(cif_text, host=host, port=port, properties=properties, timeout_ms=timeout_ms)
        row: Dict[str, Any] = {"cif_path": str(p), "ok": bool(out.get("ok"))}
        for prop in properties:
            row[prop] = out.get(prop)
        row["errors"] = out.get("errors")
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score CIFs via ALIGNN ZMQ server (multi-property).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--properties", nargs="+", default=["formation_energy", "bandgap"])
    p.add_argument("--cif", help="Path to a single .cif file")
    p.add_argument("--cifs-dir", help="Directory containing .cif files")
    p.add_argument("--out", help="Write CSV output here")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cif_paths: List[Path] = []
    if args.cif:
        cif_paths.append(Path(args.cif).expanduser().resolve())
    if args.cifs_dir:
        cif_paths.extend(sorted(Path(args.cifs_dir).expanduser().resolve().glob("*.cif")))
    if not cif_paths:
        raise SystemExit("Provide --cif or --cifs-dir")

    rows = score_cif_files_via_alignn(
        cif_paths=cif_paths,
        host=args.host,
        port=args.port,
        properties=args.properties,
        timeout_ms=args.timeout_ms,
    )

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["cif_path", "ok", *args.properties, "errors"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[alignn_client] wrote {len(rows)} rows -> {out}")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

