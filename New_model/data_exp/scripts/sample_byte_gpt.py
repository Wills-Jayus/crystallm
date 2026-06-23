#!/usr/bin/env python3
"""Sample a CrystaLLM GPT checkpoint trained on byte-level token data."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from crystallm import GPT, GPTConfig


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore"))


def decode_ids(ids: list[int]) -> str:
    data = bytes([int(i) % 256 for i in ids])
    return data.decode("utf-8", errors="replace")


def load_model(out_dir: Path, device: str) -> GPT:
    ckpt = torch.load(out_dir / "ckpt.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = ckpt["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def generate_byte_ids(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int | None) -> torch.Tensor:
    block_size = int(model.config.block_size)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    ids = encode_text(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, ...]
    model = load_model(args.out_dir, args.device)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            with ctx:
                for i in range(args.num_samples):
                    y = generate_byte_ids(model, x, args.max_new_tokens, args.temperature, args.top_k)
                    text = decode_ids(y[0].tolist())
                    f.write(json.dumps({"gen_index": i, "seed": args.seed, "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
