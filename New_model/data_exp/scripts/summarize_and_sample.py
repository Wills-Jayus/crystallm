#!/usr/bin/env python3
"""Summarize scratch training logs and run a small val-only generation sanity check."""

from __future__ import annotations

import argparse
import csv
import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from crystallm import GPT, GPTConfig

try:
    from pymatgen.core import Structure
except Exception:  # pragma: no cover
    Structure = None


ROOT = Path("/data/users/xsw/autodlmini/model/New_model/data_exp")


def parse_log(path: Path) -> list[dict[str, Any]]:
    pattern = re.compile(r"step\s+(\d+):\s+train loss\s+([0-9.]+),\s+val loss\s+([0-9.]+)")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            rows.append({"step": int(m.group(1)), "train_loss": float(m.group(2)), "val_loss": float(m.group(3))})
    return rows


def write_curve(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_csv = ROOT / "eval" / f"{name}_loss_curve.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    final = rows[-1]
    return {"curve_csv": str(out_csv), "best": best, "final": final, "num_eval_points": len(rows)}


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="ignore"))


def decode_ids(ids: list[int]) -> str:
    return bytes([int(i) % 256 for i in ids]).decode("utf-8", errors="replace")


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


def load_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
    return rows


def first_nonempty_lines(text: str, n: int) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:n]).rstrip() + "\n"


def original_prompt(text: str) -> str:
    # Use a short condition prefix that usually includes data name and early lattice/formula metadata.
    return first_nonempty_lines(text, 8)


def symcif_prompt(text: str) -> str:
    # Keep formula/SG/Z header only; generation must continue the Wyckoff and lattice sections.
    lines = []
    for line in text.splitlines():
        if line.startswith("loop_"):
            break
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n\n"


def has_all(text: str, tags: list[str]) -> bool:
    return all(tag in text for tag in tags)


def pymatgen_parse_ok(text: str) -> bool:
    if Structure is None:
        return False
    try:
        Structure.from_str(text, fmt="cif")
        return True
    except Exception:
        return False


def symcif_basic_parse_ok(text: str) -> bool:
    required = [
        "_chemical_formula_sum",
        "_symmetry_Int_Tables_number",
        "_wyckoff_site_index",
        "_wyckoff_site_element",
        "_wyckoff_free_x",
        "_cell_length_a",
        "_cell_angle_alpha",
    ]
    if not has_all(text, required):
        return False
    row_re = re.compile(r"^\s*\d+\s+[A-Z][a-z]?\s+\d+\s+[a-z]\s+\S+\s+\d+\s+\S+\s+\S+\s+\S+", re.MULTILINE)
    return bool(row_re.search(text))


def sample_model(
    name: str,
    out_dir: Path,
    rows: list[dict[str, Any]],
    prompt_fn,
    device: str,
    num_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    model = load_model(out_dir, device)
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    out_path = ROOT / "eval" / f"{name}_quick_generations.jsonl"
    summary_rows: list[dict[str, Any]] = []
    torch.manual_seed(20260619)
    if "cuda" in device:
        torch.cuda.manual_seed(20260619)
    with out_path.open("w", encoding="utf-8") as f:
        with torch.no_grad(), ctx:
            for sample_index, row in enumerate(rows):
                prompt = prompt_fn(str(row["text"]))
                x = torch.tensor(encode_text(prompt), dtype=torch.long, device=device)[None, :]
                for gen_index in range(num_per_prompt):
                    y = generate_byte_ids(model, x, max_new_tokens, temperature, top_k)
                    text = decode_ids(y[0].tolist())
                    item = {
                        "sample_index": sample_index,
                        "sample_id": row.get("sample_id"),
                        "gen_index": gen_index,
                        "prompt": prompt,
                        "text": text,
                    }
                    if name == "orig_cif":
                        item["required_tags_ok"] = has_all(
                            text,
                            [
                                "_cell_length_a",
                                "_cell_angle_alpha",
                                "_atom_site_type_symbol",
                                "_atom_site_fract_x",
                                "_atom_site_fract_y",
                                "_atom_site_fract_z",
                            ],
                        )
                        item["pymatgen_parse_ok"] = pymatgen_parse_ok(text)
                    else:
                        item["required_tags_ok"] = symcif_basic_parse_ok(text)
                        item["pymatgen_parse_ok"] = False
                    summary_rows.append(item)
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    total = len(summary_rows)
    required_ok = sum(bool(x["required_tags_ok"]) for x in summary_rows)
    pymatgen_ok = sum(bool(x["pymatgen_parse_ok"]) for x in summary_rows)
    return {
        "generations_jsonl": str(out_path),
        "total": total,
        "required_tags_ok": required_ok,
        "required_tags_rate": required_ok / max(1, total),
        "pymatgen_parse_ok": pymatgen_ok,
        "pymatgen_parse_rate": pymatgen_ok / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--num-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    curves = {
        "orig_cif": write_curve("orig_cif", parse_log(ROOT / "logs" / "scratch_orig_cif_small.train.log")),
        "symcif_v4": write_curve("symcif_v4", parse_log(ROOT / "logs" / "scratch_symcif_v4_small.train.log")),
    }
    orig_rows = load_jsonl(ROOT / "data" / "mpts52_orig_cif_byte" / "val.jsonl", args.num_prompts)
    sym_rows = load_jsonl(ROOT / "data" / "mpts52_symcif_v4_byte" / "val.jsonl", args.num_prompts)
    samples = {
        "orig_cif": sample_model(
            "orig_cif",
            ROOT / "runs" / "scratch_orig_cif_small",
            orig_rows,
            original_prompt,
            "cuda:0",
            args.num_per_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        ),
        "symcif_v4": sample_model(
            "symcif_v4",
            ROOT / "runs" / "scratch_symcif_v4_small",
            sym_rows,
            symcif_prompt,
            "cuda:0",
            args.num_per_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        ),
    }
    summary = {
        "no_test_read": True,
        "curves": curves,
        "quick_sample": samples,
        "sampling": {
            "num_prompts": args.num_prompts,
            "num_per_prompt": args.num_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        },
    }
    out = ROOT / "eval" / "training_and_quick_sample_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
