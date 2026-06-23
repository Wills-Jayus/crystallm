#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_4").resolve()
OP3_SCRIPT = Path("/data/users/xsw/autodlmini/model/New_model/opentry_3/scripts/opentry_build_geometry_compat_features.py").resolve()
OP3_SCRIPTS = OP3_SCRIPT.parent

if str(OP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(OP3_SCRIPTS))

spec = importlib.util.spec_from_file_location("opentry3_features", OP3_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {OP3_SCRIPT}")
mod = importlib.util.module_from_spec(spec)
sys.modules["opentry3_features"] = mod
spec.loader.exec_module(mod)

mod.OPENTRY_ROOT = ROOT

if __name__ == "__main__":
    raise SystemExit(mod.main())
