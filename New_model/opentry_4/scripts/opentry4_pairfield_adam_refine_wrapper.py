#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_4").resolve()
OP3_SCRIPT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3/scripts/opentry_pairfield_adam_refine.py").resolve()
OP3_SCRIPTS = OP3_SCRIPT.parent

if str(OP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(OP3_SCRIPTS))

spec = importlib.util.spec_from_file_location("opentry3_pairfield", OP3_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {OP3_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.OPENTRY_ROOT = ROOT


def _ensure_under_opentry(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT not in (resolved, *resolved.parents):
        raise SystemExit(f"Refusing to write outside opentry_4: {resolved}")
    return resolved


def _write_json(path: Path, payload: dict) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path = _ensure_under_opentry(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


mod.ensure_under_opentry = _ensure_under_opentry
mod.write_json = _write_json
mod.write_jsonl = _write_jsonl

if __name__ == "__main__":
    raise SystemExit(mod.main())
