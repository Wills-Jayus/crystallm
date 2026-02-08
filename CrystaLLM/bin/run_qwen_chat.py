#!/usr/bin/env python3
"""
Wrapper to start the local OpenAI-compatible chat API defined in `model/qwen_local_api.py`.

This is optional; you can also use vLLM's OpenAI server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run local chat API (uvicorn) for qwen_local_api.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"uvicorn not available: {exc}")

    uvicorn.run(
        "qwen_local_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

