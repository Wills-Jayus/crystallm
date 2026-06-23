#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import sys
import tarfile
import zlib
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
CRYSTALLM_ROOT = WORKSPACE / "model/scp_task/CrystaLLM"
if not CRYSTALLM_ROOT.exists():
    CRYSTALLM_ROOT = WORKSPACE / "model/CrystaLLM"
if str(CRYSTALLM_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYSTALLM_ROOT))

from crystallm import CIFTokenizer, GPT, GPTConfig  # noqa: E402


def under_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing to write outside opentry_7: {resolved}")
    return resolved


def strip_prompt_comment_lines(prompt: str) -> str:
    out_lines = []
    for line in prompt.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith(";"):
            continue
        out_lines.append(line.rstrip())
    return ("\n".join(out_lines).rstrip() + "\n") if out_lines else "\n"


def load_prompts(prompts_file: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with tarfile.open(prompts_file, "r:gz") as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile()), key=lambda m: m.name)
        for member in members:
            f = tar.extractfile(member)
            if f is None:
                continue
            prompt = f.read().decode("utf-8")
            cif_id = Path(member.name).name.replace(".txt", "")
            prompts.append((cif_id, prompt))
    return prompts


def load_model(model_dir: Path, device: str, compile_model: bool) -> GPT:
    ckpt_path = model_dir / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        compiled = torch.compile(model)
        compiled.config = model.config
        model = compiled
    return model


def disallowed_atom_ids(tokenizer: CIFTokenizer, prompt_ids: torch.Tensor) -> list[int]:
    try:
        prompt_text = tokenizer.decode(prompt_ids[0].tolist())
        match = re.search(r"^data_(\S+)", prompt_text, flags=re.MULTILINE)
        if not match:
            return []
        data_id = match.group(1).strip().strip("'\"")
        count = r"(?:\d+(?:\.\d+)?|\.\d+)"
        prefix = re.match(rf"^((?:[A-Z][a-z]?(?:{count})?)+)", data_id)
        formula_token = prefix.group(1) if prefix else data_id
        allowed_atoms = set(re.findall(r"[A-Z][a-z]?", formula_token))
        if not allowed_atoms:
            return []
        out = []
        for atom in tokenizer.atoms():
            if atom not in allowed_atoms:
                atom_id = tokenizer.token_to_id.get(atom)
                if atom_id is not None:
                    out.append(atom_id)
        return out
    except Exception:
        return []


@torch.no_grad()
def generate_one(
    model: GPT,
    tokenizer: CIFTokenizer,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
    generators: list[torch.Generator] | None = None,
    greedy_rows: list[bool] | None = None,
) -> torch.Tensor:
    newline_id = tokenizer.token_to_id["\n"]
    unk_id = tokenizer.token_to_id.get("<unk>")
    disallowed_ids = disallowed_atom_ids(tokenizer, idx)

    batch_size = idx.size(0)
    prev_id = torch.full((batch_size,), -1, dtype=torch.long, device=idx.device)
    done = torch.zeros((batch_size,), dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if unk_id is not None:
            logits[:, unk_id] = -float("Inf")
        if disallowed_ids:
            logits[:, disallowed_ids] = -float("Inf")
        if greedy:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            sample_logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(sample_logits, min(top_k, sample_logits.size(-1)))
                sample_logits[sample_logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(sample_logits, dim=-1)
            if generators is None:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                if len(generators) != batch_size:
                    raise ValueError("number of sampling generators must match batch size")
                if greedy_rows is None:
                    greedy_rows = [False] * batch_size
                if len(greedy_rows) != batch_size:
                    raise ValueError("number of greedy row flags must match batch size")
                next_rows = []
                for row_i, generator in enumerate(generators):
                    if bool(done[row_i]):
                        next_rows.append(torch.tensor([newline_id], dtype=torch.long, device=idx.device))
                    elif greedy_rows[row_i]:
                        next_rows.append(torch.argmax(logits[row_i], dim=-1, keepdim=True))
                    else:
                        next_rows.append(torch.multinomial(probs[row_i], num_samples=1, generator=generator))
                idx_next = torch.stack(next_rows, dim=0)
        if done.any():
            idx_next[done, 0] = newline_id
        idx = torch.cat((idx, idx_next), dim=1)
        idx_next_flat = idx_next[:, 0]
        done |= (prev_id == newline_id) & (idx_next_flat == newline_id)
        if bool(done.all()):
            break
        prev_id = idx_next_flat
    return idx


def encode_prompt(tokenizer: CIFTokenizer, prompt: str, device: str) -> torch.Tensor:
    prompt = strip_prompt_comment_lines(prompt)
    tokenized = tokenizer.tokenize_cif(prompt)
    if "<unk>" in tokenized:
        tokenized = [token for token in tokenized if token != "<unk>"]
    start_ids = tokenizer.encode(tokenized)
    return torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path = under_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def member_prompt_id(name: str) -> str:
    base = Path(name).name
    if base.endswith(".cif"):
        base = base[:-4]
    return base.rsplit("__", 1)[0]


def member_candidate_index(name: str) -> int | None:
    base = Path(name).name
    if base.endswith(".cif"):
        base = base[:-4]
    try:
        value = int(base.rsplit("__", 1)[1])
    except Exception:
        return None
    return value if value >= 1 else None


def recover_complete_groups(
    source: Path | None,
    prompt_ids: set[str],
    num_gens: int,
) -> tuple[dict[str, dict[int, tuple[str, bytes]]], dict[str, int]]:
    if source is None or not source.exists():
        return {}, {"source_exists": 0, "members_read": 0, "complete_prompts": 0}

    groups: dict[str, dict[int, tuple[str, bytes]]] = {}
    members_read = 0
    skipped = 0
    try:
        with tarfile.open(source, "r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".cif"):
                    continue
                prompt_id = member_prompt_id(member.name)
                cand_idx = member_candidate_index(member.name)
                if prompt_id not in prompt_ids or cand_idx is None or cand_idx > num_gens:
                    skipped += 1
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    skipped += 1
                    continue
                try:
                    payload = fileobj.read()
                except (EOFError, OSError, tarfile.TarError, gzip.BadGzipFile, zlib.error):
                    skipped += 1
                    break
                groups.setdefault(prompt_id, {})[cand_idx] = (member.name, payload)
                members_read += 1
    except (EOFError, OSError, tarfile.TarError, gzip.BadGzipFile, zlib.error):
        pass

    complete = {
        prompt_id: cand_map
        for prompt_id, cand_map in groups.items()
        if all(i in cand_map for i in range(1, num_gens + 1))
    }
    return complete, {
        "source_exists": 1,
        "members_read": members_read,
        "skipped_or_partial_members": skipped,
        "complete_prompts": len(complete),
    }


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    tarinfo = tarfile.TarInfo(name=name)
    tarinfo.size = len(payload)
    tar.addfile(tarinfo, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pure-model CIFs in fixed candidate order.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume-source", type=Path)
    parser.add_argument("--num-gens", type=int, default=20)
    parser.add_argument("--sampled-gens", type=int, default=19)
    parser.add_argument("--candidate0", choices=["greedy", "sampled"], default="greedy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if args.candidate0 == "greedy" and args.num_gens != args.sampled_gens + 1:
        raise ValueError("--num-gens must equal --sampled-gens + 1 because candidate 0 is greedy")
    if args.candidate0 == "sampled":
        args.sampled_gens = args.num_gens
    args.out = under_root(args.out)
    if args.resume_source is not None:
        args.resume_source = args.resume_source.expanduser().resolve()
        if args.resume_source == args.out.resolve():
            raise ValueError("--resume-source must differ from --out")
    if args.manifest is None:
        args.manifest = args.out.with_suffix(args.out.suffix + ".manifest.json")
    args.manifest = under_root(args.manifest)

    torch.manual_seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = CIFTokenizer()
    model = load_model(args.model, args.device, compile_model=not args.no_compile)
    prompts = load_prompts(args.prompts)
    prompt_ids = {cif_id for cif_id, _ in prompts}
    recovered_groups, recovery_stats = recover_complete_groups(args.resume_source, prompt_ids, args.num_gens)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    regenerated_prompts = 0
    with tarfile.open(args.out, "w:gz") as tar, torch.no_grad(), ctx:
        for prompt_index, (cif_id, prompt) in enumerate(tqdm(prompts, desc="generating ordered pure CIFs")):
            recovered = recovered_groups.get(cif_id)
            if recovered is not None:
                for output_index in range(1, args.num_gens + 1):
                    member_name, payload = recovered[output_index]
                    add_bytes_to_tar(tar, member_name, payload)
                continue

            regenerated_prompts += 1
            x = encode_prompt(tokenizer, prompt, args.device)
            ordered_x = x.repeat(args.num_gens, 1)
            generators: list[torch.Generator] = []
            for sampled_index in range(args.num_gens):
                sample_seed = args.seed + prompt_index * args.seed_stride + sampled_index
                generator = torch.Generator(device=args.device)
                generator.manual_seed(sample_seed)
                generators.append(generator)
            greedy_rows = (
                [True] + [False] * (args.num_gens - 1)
                if args.candidate0 == "greedy"
                else [False] * args.num_gens
            )
            ordered_ids = generate_one(
                model,
                tokenizer,
                ordered_x,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                greedy=False,
                generators=generators,
                greedy_rows=greedy_rows,
            )
            outputs = [
                tokenizer.decode(ordered_ids[output_index].tolist()).replace("<unk>", "")
                for output_index in range(args.num_gens)
            ]
            for output_index, cif in enumerate(outputs, start=1):
                cif_bytes = cif.encode("utf-8")
                add_bytes_to_tar(tar, f"{cif_id}__{output_index}.cif", cif_bytes)

    write_manifest(
        args.manifest,
        {
            "model": str(args.model),
            "prompts": str(args.prompts),
            "out": str(args.out),
            "resume_source": str(args.resume_source) if args.resume_source is not None else None,
            "recovery": recovery_stats,
            "regenerated_prompts": regenerated_prompts,
            "crystallm_root": str(CRYSTALLM_ROOT),
            "num_prompts": len(prompts),
            "num_gens": args.num_gens,
            "candidate0": args.candidate0,
            "candidate_0": (
                "greedy argmax with CrystaLLM atom and unk masks; stored as tar member suffix __1"
                if args.candidate0 == "greedy"
                else "sampled with fixed per-prompt/per-candidate seed; stored as tar member suffix __1"
            ),
            "candidates_1_to_19": "sampled with fixed per-prompt/per-candidate seeds; stored as tar member suffixes __2 through __20",
            "seed": args.seed,
            "seed_stride": args.seed_stride,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "dtype": args.dtype,
        },
    )


if __name__ == "__main__":
    main()
