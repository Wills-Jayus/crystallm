"""
Deprecated compatibility shim.

项目已迁移到 Qwen 命名：请改用 `bin/qwen_client.py`。
本文件仅为兼容旧脚本/旧命令保留，所有实现转发到 qwen_client。
"""

from __future__ import annotations

from qwen_client import (  # noqa: F401
    DEFAULT_SYSTEM_PROMPT,
    LlamaClient,
    LlamaConfig,
    QwenClient,
    QwenConfig,
    main,
    parse_args,
)


if __name__ == "__main__":
    main()
