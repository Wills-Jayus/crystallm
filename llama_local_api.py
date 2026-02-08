"""
Deprecated compatibility shim.

Please use `model/qwen_local_api.py`. This module keeps the old import path working:
  uvicorn llama_local_api:app ...
"""

from __future__ import annotations

from qwen_local_api import app  # noqa: F401

