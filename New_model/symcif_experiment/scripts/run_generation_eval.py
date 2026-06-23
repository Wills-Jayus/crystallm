#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTODL_ROOT = PROJECT_ROOT.parents[2]
CRYSTALLM_ROOT = PROJECT_ROOT / "external" / "CrystaLLM_code"

for path in (PROJECT_ROOT / "src", CRYSTALLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crystallm import (  # noqa: E402
    CIFTokenizer,
    GPT,
    GPTConfig,
    bond_length_reasonableness_score,
    extract_space_group_symbol,
    is_atom_site_multiplicity_consistent,
    is_valid,
    replace_symmetry_operators,
)
from symcif.lookup import WyckoffLookup  # noqa: E402
from symcif.parse import parse_symcif_text  # noqa: E402
from symcif.render import render_standard_cif  # noqa: E402
from symcif_v2.parse import parse_symcif_v2_text  # noqa: E402
from symcif_v2.to_cif import render_standard_cif_v2  # noqa: E402
from symcif_v3.parse import parse_symcif_v3_text  # noqa: E402
from symcif_v3.to_cif import render_standard_cif_v3  # noqa: E402

warnings.filterwarnings("ignore")

BASE_DATA_MODES = ("baseline", "cf_like", "symcif_v1")
DEFAULT_MODES = ("baseline", "cf_like", "symcif_v1")
DIRECT_CIF_MODES = {"baseline", "baseline_minprompt", "symcif_v1_constrained"}
MODE_DATA_ALIAS = {
    "symcif_v2": "symcif_v2",
    "symcif_v2_raw": "symcif_v2",
    "symcif_v2_constrained": "symcif_v2",
    "symcif_v2_full3500_constrained": "symcif_v2",
    "symcif_v3": "symcif_v3",
    "symcif_v3_raw": "symcif_v3",
    "symcif_v3_constrained": "symcif_v3",
}


def is_direct_cif_mode(mode: str) -> bool:
    return mode in DIRECT_CIF_MODES or mode.startswith("baseline")


def is_symcif_v2_mode(mode: str) -> bool:
    return mode.startswith("symcif_v2")


def is_symcif_v3_mode(mode: str) -> bool:
    return mode.startswith("symcif_v3")


@dataclass(frozen=True)
class TestCase:
    index: int
    sample_id: str
    source_path: str
    target_formula: str
    target_sg_number: int | None
    target_sg_symbol: str | None
    prompts: dict[str, str]


def split_concat_records(path: Path) -> list[str]:
    records: list[str] = []
    cur: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("data_") and cur:
            records.append("\n".join(cur).rstrip() + "\n")
            cur = [line]
        else:
            cur.append(line)
    if cur:
        records.append("\n".join(cur).rstrip() + "\n")
    return records


def sample_id_from_text(text: str) -> str | None:
    match = re.search(r"#\s*sample_id:\s*(\S+)", text)
    if match:
        return match.group(1)
    return None


def data_name_from_text(text: str) -> str:
    match = re.search(r"^data_(\S+)", text, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def extract_value(text: str, key: str) -> str | None:
    pat = re.compile(rf"^{re.escape(key)}\s+(.+?)\s*$", flags=re.MULTILINE)
    match = pat.search(text)
    if not match:
        return None
    return match.group(1).strip().strip("'\"")


def extract_prompt_to_space_group(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        if line.lstrip().startswith("_symmetry_space_group_name_H-M"):
            return "\n".join(out).rstrip() + "\n"
    raise ValueError("missing _symmetry_space_group_name_H-M line")


def strip_prompt_comment_lines(prompt: str) -> str:
    lines = []
    for line in prompt.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def load_manifest(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                out[row["sample_id"]] = row["source_path"]
    return out


def resolve_source_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return AUTODL_ROOT / path


def load_test_cases(test_limit: int | None = None, modes: tuple[str, ...] | None = None) -> list[TestCase]:
    requested_modes = modes or DEFAULT_MODES
    record_modes = list(BASE_DATA_MODES)
    for mode in requested_modes:
        data_mode = MODE_DATA_ALIAS.get(mode, mode)
        if data_mode not in record_modes and (PROJECT_ROOT / "data" / data_mode / "test.txt").exists():
            record_modes.append(data_mode)
    records = {mode: split_concat_records(PROJECT_ROOT / "data" / mode / "test.txt") for mode in record_modes}
    n = len(records["baseline"])
    if any(len(records[mode]) != n for mode in record_modes):
        sizes = {mode: len(records[mode]) for mode in record_modes}
        raise ValueError(f"test split size mismatch: {sizes}")
    if test_limit is not None:
        n = min(n, test_limit)

    manifest = load_manifest(PROJECT_ROOT / "data" / "split_manifest.csv")
    cases: list[TestCase] = []
    for i in range(n):
        cf_id = sample_id_from_text(records["cf_like"][i])
        sym_id = sample_id_from_text(records["symcif_v1"][i])
        sample_id = cf_id or sym_id or data_name_from_text(records["baseline"][i])
        if cf_id and sym_id and cf_id != sym_id:
            raise ValueError(f"sample_id mismatch at index {i}: {cf_id} != {sym_id}")
        if sample_id not in manifest:
            raise ValueError(f"missing test manifest entry for {sample_id}")

        target_formula = extract_value(records["cf_like"][i], "_chemical_formula_sum")
        target_sg_raw = extract_value(records["cf_like"][i], "_symmetry_Int_Tables_number")
        target_sg_symbol = extract_value(records["cf_like"][i], "_symmetry_space_group_name_H-M")
        if not target_formula:
            raise ValueError(f"missing target formula for {sample_id}")
        target_sg_number = int(float(target_sg_raw)) if target_sg_raw else None
        prompts = {mode: extract_prompt_to_space_group(records[mode][i]) for mode in record_modes}
        for mode in requested_modes:
            data_mode = MODE_DATA_ALIAS.get(mode, mode)
            if data_mode in prompts and mode not in prompts:
                prompts[mode] = prompts[data_mode]
        cases.append(
            TestCase(
                index=i,
                sample_id=sample_id,
                source_path=str(resolve_source_path(manifest[sample_id])),
                target_formula=target_formula,
                target_sg_number=target_sg_number,
                target_sg_symbol=target_sg_symbol,
                prompts=prompts,
            )
        )
    return cases


def load_model(model_dir: Path, device: str, dtype: str, compile_model: bool) -> GPT:
    ckpt_path = model_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)  # type: ignore[assignment]
    return model


def prompt_disallowed_atom_ids(tokenizer: CIFTokenizer, prompt_ids: list[int]) -> list[int]:
    disallowed: list[int] = []
    try:
        prompt_text = tokenizer.decode(prompt_ids)
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
        token_to_id = tokenizer.token_to_id
        for atom in tokenizer.atoms():
            if atom not in allowed_atoms and atom in token_to_id:
                disallowed.append(token_to_id[atom])
    except Exception:
        return []
    return disallowed


@torch.no_grad()
def generate_batch_mode_aware(
    model: GPT,
    tokenizer: CIFTokenizer,
    prompt: str,
    seeds: list[int],
    *,
    mode: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    prompt = strip_prompt_comment_lines(prompt)
    tokens = tokenizer.tokenize_cif(prompt)
    token_to_id = tokenizer.token_to_id
    unk_id = token_to_id.get("<unk>")
    vocab_size = int(getattr(model.config, "vocab_size", len(token_to_id)))
    prompt_ids = [
        token_to_id[t]
        for t in tokens
        if (unk_id is None or token_to_id[t] != unk_id) and 0 <= int(token_to_id[t]) < vocab_size
    ]
    if not prompt_ids:
        raise ValueError("empty prompt after tokenization")

    bsz = len(seeds)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(bsz, 1)
    newline_id = token_to_id["\n"]
    cell_volume_id = token_to_id.get("_cell_volume")
    disallowed_atom_ids = prompt_disallowed_atom_ids(tokenizer, prompt_ids)
    gens = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    done = [False] * bsz
    stop_lengths = [None] * bsz
    prev_ids: list[int | None] = [None] * bsz
    saw_cell_volume = [False] * bsz

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else torch.no_grad()

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]
        with ctx:
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if unk_id is not None and 0 <= int(unk_id) < logits.size(-1):
            logits[:, unk_id] = -float("inf")
        if disallowed_atom_ids:
            valid_disallowed = [idx for idx in disallowed_atom_ids if 0 <= int(idx) < logits.size(-1)]
            if valid_disallowed:
                logits[:, valid_disallowed] = -float("inf")
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)

        next_ids: list[int] = []
        for row in range(bsz):
            if done[row]:
                next_ids.append(newline_id)
                continue
            sampled = torch.multinomial(probs[row], num_samples=1, generator=gens[row])
            next_ids.append(int(sampled.item()))
        next_tensor = torch.tensor(next_ids, dtype=torch.long, device=device).unsqueeze(1)
        idx = torch.cat((idx, next_tensor), dim=1)

        for row, next_id in enumerate(next_ids):
            if done[row]:
                continue
            if cell_volume_id is not None and next_id == cell_volume_id:
                saw_cell_volume[row] = True
            if prev_ids[row] == newline_id and next_id == newline_id:
                if is_direct_cif_mode(mode) or saw_cell_volume[row]:
                    done[row] = True
                    stop_lengths[row] = idx.size(1)
            prev_ids[row] = next_id
        if all(done):
            break

    texts = []
    idx_cpu = idx.detach().cpu()
    for row in range(bsz):
        stop = stop_lengths[row] or idx_cpu.size(1)
        texts.append(tokenizer.decode(idx_cpu[row, :stop].tolist()))
    return texts


def extract_generated_record(text: str, mode: str) -> str:
    text = text.replace("<unk>", "")
    match = re.search(r"^data_", text, flags=re.MULTILINE)
    if match:
        text = text[match.start() :]
    next_match = re.search(r"\ndata_", text[1:])
    if next_match:
        text = text[: next_match.start() + 1]
    if not is_direct_cif_mode(mode):
        lines = text.splitlines()
        out: list[str] = []
        seen_cell_volume = False
        for line in lines:
            out.append(line)
            if line.lstrip().startswith("_cell_volume"):
                seen_cell_volume = True
                break
        if seen_cell_volume:
            text = "\n".join(out).rstrip() + "\n"
    return text.rstrip() + "\n"


def generation_worker(
    *,
    mode: str,
    model_dir: str,
    cases_payload: list[dict[str, Any]],
    seeds: list[int],
    out_path: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    compile_model: bool,
    worker_id: int,
) -> None:
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = CIFTokenizer()
    model = load_model(Path(model_dir), device=device, dtype=dtype, compile_model=compile_model)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for local_i, case in enumerate(cases_payload, start=1):
            prompt = case["prompt"]
            try:
                texts = generate_batch_mode_aware(
                    model,
                    tokenizer,
                    prompt,
                    seeds,
                    mode=mode,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                for gen_index, (seed, text) in enumerate(zip(seeds, texts)):
                    record_text = extract_generated_record(text, mode)
                    f.write(
                        json.dumps(
                            {
                                "mode": mode,
                                "sample_index": case["index"],
                                "sample_id": case["sample_id"],
                                "gen_index": gen_index,
                                "seed": int(seed),
                                "raw_generation_success": bool(record_text.strip()),
                                "generated_text": record_text,
                                "error": None,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                for gen_index, seed in enumerate(seeds):
                    f.write(
                        json.dumps(
                            {
                                "mode": mode,
                                "sample_index": case["index"],
                                "sample_id": case["sample_id"],
                                "gen_index": gen_index,
                                "seed": int(seed),
                                "raw_generation_success": False,
                                "generated_text": "",
                                "error": err,
                                "traceback": tb,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            if local_i % 25 == 0 or local_i == len(cases_payload):
                print(f"[generate:{mode}:worker{worker_id}] {local_i}/{len(cases_payload)} prompts done", flush=True)


def run_generation(args: argparse.Namespace, cases: list[TestCase], seeds: list[int]) -> None:
    model_dirs = {
        "baseline": PROJECT_ROOT / "runs" / "exp_baseline_rerun750",
        "baseline_minprompt": PROJECT_ROOT / "runs" / "exp_baseline_minprompt_rerun750",
        "cf_like": PROJECT_ROOT / "runs" / "exp_cf_like_rerun750",
        "symcif_v1": PROJECT_ROOT / "runs" / "exp_symcif_v1",
        "symcif_v1_atomprops": PROJECT_ROOT / "runs" / "exp_symcif_v1_atomprops_rerun750",
        "symcif_v2": PROJECT_ROOT / "runs" / "exp_symcif_v2",
        "symcif_v2_raw": PROJECT_ROOT / "runs" / "exp_symcif_v2",
        "symcif_v2_constrained": PROJECT_ROOT / "runs" / "exp_symcif_v2",
        "symcif_v3": PROJECT_ROOT / "runs" / "exp_symcif_v3_cf_order",
        "symcif_v3_raw": PROJECT_ROOT / "runs" / "exp_symcif_v3_cf_order",
        "symcif_v3_constrained": PROJECT_ROOT / "runs" / "exp_symcif_v3_cf_order",
    }
    generation_dir = Path(args.out_dir) / "generations"
    generation_dir.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = ["cpu"]

    for mode in args.modes:
        if mode not in model_dirs:
            raise ValueError(f"no model directory configured for generation mode {mode}")
        merged_path = generation_dir / f"{mode}.jsonl"
        expected = len(cases) * len(seeds)
        if merged_path.exists() and not args.overwrite:
            existing = sum(1 for _ in merged_path.open(encoding="utf-8"))
            if existing == expected:
                print(f"[generate:{mode}] found complete {merged_path}, skipping")
                continue
        print(f"[generate:{mode}] start: prompts={len(cases)} gens={len(seeds)} devices={devices}", flush=True)
        chunks: list[list[TestCase]] = [[] for _ in devices]
        for i, case in enumerate(cases):
            chunks[i % len(devices)].append(case)

        ctx = mp.get_context("spawn")
        procs: list[mp.Process] = []
        worker_paths: list[Path] = []
        for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
            payload = [
                {
                    "index": c.index,
                    "sample_id": c.sample_id,
                    "prompt": c.prompts[mode],
                }
                for c in chunk
            ]
            worker_path = generation_dir / f"{mode}.worker{worker_id}.jsonl"
            worker_paths.append(worker_path)
            proc = ctx.Process(
                target=generation_worker,
                kwargs={
                    "mode": mode,
                    "model_dir": str(model_dirs[mode]),
                    "cases_payload": payload,
                    "seeds": seeds,
                    "out_path": str(worker_path),
                    "device": device,
                    "dtype": args.dtype,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "compile_model": args.compile,
                    "worker_id": worker_id,
                },
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"generation worker for {mode} failed with exit code {proc.exitcode}")

        records: list[dict[str, Any]] = []
        for worker_path in worker_paths:
            with worker_path.open(encoding="utf-8") as f:
                records.extend(json.loads(line) for line in f if line.strip())
        records.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        with merged_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        print(f"[generate:{mode}] wrote {len(records)} records -> {merged_path}", flush=True)


def comp_dict(formula: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in Composition(formula).as_dict().items()}


def same_composition(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    try:
        da = comp_dict(a)
        db = comp_dict(b)
        if set(da) != set(db):
            return False
        return all(abs(da[k] - db[k]) < 1e-6 for k in da)
    except Exception:
        return False


def cif_dict(cif_text: str) -> dict[str, Any] | None:
    try:
        data = CifParser.from_string(cif_text).as_dict()
        if not data:
            return None
        return data[list(data.keys())[0]]
    except Exception:
        return None


def cif_value(cif_text: str, key: str) -> str | None:
    block = cif_dict(cif_text)
    if not block or key not in block:
        return None
    value = block[key]
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value).strip().strip("'\"")


def block_value(block: dict[str, Any] | None, key: str) -> str | None:
    if not block or key not in block:
        return None
    value = block[key]
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value).strip().strip("'\"")


def estimate_sites_from_cif_block(block: dict[str, Any] | None) -> int | None:
    if not block:
        return None
    mults = block.get("_atom_site_symmetry_multiplicity")
    if mults is not None:
        if not isinstance(mults, list):
            mults = [mults]
        total = 0
        ok = False
        for item in mults:
            try:
                total += int(float(str(item).strip().strip("'\"")))
                ok = True
            except Exception:
                pass
        if ok:
            return total
    atoms = block.get("_atom_site_type_symbol")
    if atoms is not None:
        return len(atoms) if isinstance(atoms, list) else 1
    return None


def cif_sg_number(cif_text: str, structure: Structure | None = None) -> int | None:
    value = cif_value(cif_text, "_symmetry_Int_Tables_number")
    if value:
        try:
            return int(float(value))
        except Exception:
            pass
    if structure is not None:
        try:
            return int(SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0).get_space_group_number())
        except Exception:
            return None
    return None


def prepare_cif_for_eval(cif_text: str) -> str:
    try:
        sg = extract_space_group_symbol(cif_text)
        if sg is not None and sg != "P 1":
            return replace_symmetry_operators(cif_text, sg)
    except Exception:
        pass
    return cif_text


def load_generation_records(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            grouped.setdefault(int(rec["sample_index"]), []).append(rec)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["gen_index"]))
    return grouped


@contextmanager
def time_limit(seconds: float | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ANN001
        raise TimeoutError(f"operation exceeded {seconds:g}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        signal.signal(signal.SIGALRM, old_handler)


def evaluate_one_sample(
    mode: str,
    case_dict: dict[str, Any],
    rows: list[dict[str, Any]],
    lookup_json: str,
    bond_timeout_seconds: float,
    valid_timeout_seconds: float,
    match_timeout_seconds: float,
    max_match_sites: int,
    max_eval_sites: int,
    parse_timeout_seconds: float = 8.0,
    sg_timeout_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    lookup = WyckoffLookup.from_json(lookup_json)
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    try:
        target_structure = Structure.from_file(case_dict["source_path"])
    except Exception:
        target_structure = None

    out: list[dict[str, Any]] = []
    cached_by_hash: dict[str, dict[str, Any]] = {}
    for row in rows:
        metric: dict[str, Any] = {
            "mode": mode,
            "sample_index": int(case_dict["index"]),
            "sample_id": case_dict["sample_id"],
            "gen_index": int(row["gen_index"]),
            "seed": int(row["seed"]),
            "raw_generation_success": bool(row.get("raw_generation_success")),
            "parse_success": False,
            "symcif_to_cif_success": False,
            "pymatgen_readable": False,
            "formula_ok": False,
            "space_group_ok": False,
            "multiplicity_ok": False,
            "bond_length_score": None,
            "valid": False,
            "match_ok": False,
            "rms": None,
            "estimated_sites": None,
            "parse_time_seconds": None,
            "sg_time_seconds": None,
                "matcher_time_seconds": None,
                "eval_timeout": False,
                "parse_timeout": False,
                "sg_timeout": False,
                "matcher_timeout": False,
            "early_match_skip_reason": None,
            "generation_time_seconds": row.get("generation_time_seconds"),
            "formula_closure_success": row.get("formula_closure_success"),
            "mask_rejected_tokens": row.get("mask_rejected_tokens"),
            "resample_count": row.get("resample_count"),
            "error": row.get("error"),
        }
        generated_text = row.get("generated_text") or ""
        generated_hash = hashlib.sha1(generated_text.encode("utf-8", errors="ignore")).hexdigest()
        if generated_hash in cached_by_hash:
            cached = copy.deepcopy(cached_by_hash[generated_hash])
            cached["gen_index"] = int(row["gen_index"])
            cached["seed"] = int(row["seed"])
            cached["generation_time_seconds"] = row.get("generation_time_seconds")
            cached["dedup_reused"] = True
            out.append(cached)
            continue
        metric["generated_sha1"] = generated_hash
        metric["dedup_reused"] = False
        standard_cif: str | None = None
        try:
            if not metric["raw_generation_success"]:
                raise ValueError("raw_generation_failed")
            if is_direct_cif_mode(mode):
                standard_cif = generated_text
                metric["symcif_to_cif_success"] = True
                block = cif_dict(standard_cif)
                metric["parse_success"] = block is not None
                est_sites = estimate_sites_from_cif_block(block)
                metric["estimated_sites"] = est_sites
                if est_sites is not None and max_eval_sites > 0 and est_sites > max_eval_sites:
                    metric["pymatgen_skipped_reason"] = "too_many_estimated_sites"
                    gen_formula = block_value(block, "_chemical_formula_sum") or data_name_from_text(standard_cif)
                    metric["formula_ok"] = same_composition(gen_formula, case_dict["target_formula"])
                    gen_sg_raw = block_value(block, "_symmetry_Int_Tables_number")
                    try:
                        gen_sg_num = int(float(gen_sg_raw)) if gen_sg_raw else None
                    except Exception:
                        gen_sg_num = None
                    target_sg_num = case_dict.get("target_sg_number")
                    metric["space_group_ok"] = bool(
                        gen_sg_num is not None and target_sg_num is not None and gen_sg_num == target_sg_num
                    )
                    try:
                        metric["multiplicity_ok"] = bool(is_atom_site_multiplicity_consistent(standard_cif))
                    except Exception:
                        metric["multiplicity_ok"] = False
                    out.append(metric)
                    continue
            elif is_symcif_v2_mode(mode):
                parsed = parse_symcif_v2_text(generated_text, lookup)
                metric["parse_success"] = True
                est_sites = sum(int(site.multiplicity) for site in parsed.sites)
                metric["estimated_sites"] = est_sites
                if max_eval_sites > 0 and est_sites > max_eval_sites:
                    metric["symcif_to_cif_success"] = False
                    metric["conversion_skipped_reason"] = "too_many_estimated_sites"
                    metric["formula_ok"] = same_composition(parsed.cell_formula, case_dict["target_formula"])
                    target_sg_num = case_dict.get("target_sg_number")
                    metric["space_group_ok"] = bool(
                        parsed.sg_number is not None and target_sg_num is not None and parsed.sg_number == target_sg_num
                    )
                    out.append(metric)
                    continue
                standard_cif = render_standard_cif_v2(parsed, symprec=0.1, lookup=lookup)
                metric["symcif_to_cif_success"] = True
            elif is_symcif_v3_mode(mode):
                parsed = parse_symcif_v3_text(generated_text, lookup)
                metric["parse_success"] = True
                est_sites = sum(int(site.multiplicity) for site in parsed.sites)
                metric["estimated_sites"] = est_sites
                if max_eval_sites > 0 and est_sites > max_eval_sites:
                    metric["symcif_to_cif_success"] = False
                    metric["conversion_skipped_reason"] = "too_many_estimated_sites"
                    metric["formula_ok"] = same_composition(parsed.cell_formula, case_dict["target_formula"])
                    target_sg_num = case_dict.get("target_sg_number")
                    metric["space_group_ok"] = bool(
                        parsed.sg_number is not None and target_sg_num is not None and parsed.sg_number == target_sg_num
                    )
                    out.append(metric)
                    continue
                standard_cif = render_standard_cif_v3(parsed, symprec=0.1)
                metric["symcif_to_cif_success"] = True
            else:
                parsed = parse_symcif_text(generated_text, lookup)
                metric["parse_success"] = True
                est_sites = sum(int(site.multiplicity) for site in parsed.sites)
                metric["estimated_sites"] = est_sites
                if max_eval_sites > 0 and est_sites > max_eval_sites:
                    metric["symcif_to_cif_success"] = False
                    metric["conversion_skipped_reason"] = "too_many_estimated_sites"
                    metric["formula_ok"] = same_composition(parsed.cell_formula, case_dict["target_formula"])
                    target_sg_num = case_dict.get("target_sg_number")
                    metric["space_group_ok"] = bool(
                        parsed.sg_number is not None and target_sg_num is not None and parsed.sg_number == target_sg_num
                    )
                    out.append(metric)
                    continue
                standard_cif = render_standard_cif(parsed, symprec=0.1)
                metric["symcif_to_cif_success"] = True

            eval_cif = prepare_cif_for_eval(standard_cif)
            parse_started = time.monotonic()
            try:
                with time_limit(parse_timeout_seconds):
                    structure = Structure.from_str(eval_cif, fmt="cif")
                metric["parse_time_seconds"] = time.monotonic() - parse_started
            except TimeoutError:
                metric["parse_timeout"] = True
                metric["error"] = metric.get("error") or f"parse_timeout>{parse_timeout_seconds:g}s"
                out.append(metric)
                cached_by_hash[generated_hash] = copy.deepcopy(metric)
                continue
            metric["pymatgen_readable"] = True

            gen_formula = cif_value(eval_cif, "_chemical_formula_sum") or data_name_from_text(eval_cif)
            metric["formula_ok"] = same_composition(gen_formula, case_dict["target_formula"])

            sg_started = time.monotonic()
            try:
                with time_limit(sg_timeout_seconds):
                    gen_sg_num = cif_sg_number(eval_cif, structure)
                metric["sg_time_seconds"] = time.monotonic() - sg_started
            except TimeoutError:
                gen_sg_num = None
                metric["sg_timeout"] = True
            target_sg_num = case_dict.get("target_sg_number")
            metric["space_group_ok"] = bool(gen_sg_num is not None and target_sg_num is not None and gen_sg_num == target_sg_num)

            try:
                metric["multiplicity_ok"] = bool(is_atom_site_multiplicity_consistent(eval_cif))
            except Exception:
                metric["multiplicity_ok"] = False

            too_many_eval_sites = max_eval_sites > 0 and len(structure) > max_eval_sites
            if too_many_eval_sites:
                metric["bond_skipped_reason"] = "too_many_sites"
                metric["valid_skipped_reason"] = "too_many_sites"
            else:
                try:
                    with time_limit(bond_timeout_seconds):
                        metric["bond_length_score"] = float(bond_length_reasonableness_score(eval_cif))
                except Exception:
                    metric["bond_length_score"] = None

                try:
                    with time_limit(valid_timeout_seconds):
                        metric["valid"] = bool(is_valid(eval_cif, bond_length_acceptability_cutoff=1.0))
                except Exception:
                    metric["valid"] = False

            if target_structure is not None:
                try:
                    if max_match_sites > 0 and (
                        len(structure) > max_match_sites or len(target_structure) > max_match_sites
                    ):
                        metric["match_skipped_reason"] = "too_many_sites"
                    elif not metric.get("formula_ok"):
                        metric["early_match_skip_reason"] = "formula_mismatch"
                    elif case_dict.get("target_formula") and len(structure) != int(round(sum(comp_dict(case_dict["target_formula"]).values()))):
                        metric["early_match_skip_reason"] = "atom_count_mismatch"
                    else:
                        matcher_started = time.monotonic()
                        with time_limit(match_timeout_seconds):
                            rms = matcher.get_rms_dist(structure, target_structure)
                        metric["matcher_time_seconds"] = time.monotonic() - matcher_started
                        if rms is not None:
                            metric["match_ok"] = True
                            metric["rms"] = float(rms[0])
                except TimeoutError:
                    metric["matcher_timeout"] = True
                    metric["error"] = metric.get("error") or f"matcher_timeout>{match_timeout_seconds:g}s"
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            metric["error"] = metric.get("error") or f"{type(exc).__name__}: {exc}"
        cached_by_hash[generated_hash] = copy.deepcopy(metric)
        out.append(metric)
    return out


def timeout_metrics_for_sample(
    mode: str,
    case_dict: dict[str, Any],
    rows: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "mode": mode,
                "sample_index": int(case_dict["index"]),
                "sample_id": case_dict["sample_id"],
                "gen_index": int(row["gen_index"]),
                "seed": int(row["seed"]),
                "raw_generation_success": bool(row.get("raw_generation_success")),
                "parse_success": False,
                "symcif_to_cif_success": False,
                "pymatgen_readable": False,
                "formula_ok": False,
                "space_group_ok": False,
                "multiplicity_ok": False,
                "bond_length_score": None,
                "valid": False,
                "match_ok": False,
                "rms": None,
                "estimated_sites": None,
                "generation_time_seconds": row.get("generation_time_seconds"),
                "formula_closure_success": row.get("formula_closure_success"),
                "mask_rejected_tokens": row.get("mask_rejected_tokens"),
                "resample_count": row.get("resample_count"),
                "eval_timeout": True,
                "parse_timeout": False,
                "sg_timeout": False,
                "matcher_timeout": False,
                "parse_time_seconds": None,
                "sg_time_seconds": None,
                "matcher_time_seconds": None,
                "early_match_skip_reason": reason,
                "error": row.get("error") or reason,
            }
        )
    return out


def _sample_eval_process(
    out_queue: Any,
    mode: str,
    case_dict: dict[str, Any],
    rows: list[dict[str, Any]],
    lookup_json: str,
    bond_timeout_seconds: float,
    valid_timeout_seconds: float,
    match_timeout_seconds: float,
    max_match_sites: int,
    max_eval_sites: int,
    parse_timeout_seconds: float,
    sg_timeout_seconds: float,
) -> None:
    try:
        metrics = evaluate_one_sample(
            mode,
            case_dict,
            rows,
            lookup_json,
            bond_timeout_seconds,
            valid_timeout_seconds,
            match_timeout_seconds,
            max_match_sites,
            max_eval_sites,
            parse_timeout_seconds,
            sg_timeout_seconds,
        )
        out_queue.put({"ok": True, "metrics": metrics})
    except Exception as exc:  # noqa: BLE001
        out_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def evaluate_mode_with_hard_timeouts(
    *,
    mode: str,
    case_payload: list[dict[str, Any]],
    grouped: dict[int, list[dict[str, Any]]],
    lookup_json: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    pending = list(case_payload)
    active: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    sample_timeout = float(args.sample_timeout_seconds or 0)

    def start_case(case: dict[str, Any]) -> None:
        rows = grouped.get(int(case["index"]), [])
        out_queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_sample_eval_process,
            args=(
                out_queue,
                mode,
                case,
                rows,
                lookup_json,
                args.bond_timeout_seconds,
                args.valid_timeout_seconds,
                args.match_timeout_seconds,
                args.max_match_sites,
                args.max_eval_sites,
                getattr(args, "parse_timeout_seconds", args.valid_timeout_seconds),
                getattr(args, "sg_timeout_seconds", args.valid_timeout_seconds),
            ),
        )
        proc.start()
        active.append(
            {
                "case": case,
                "rows": rows,
                "queue": out_queue,
                "proc": proc,
                "started": time.monotonic(),
            }
        )

    with tqdm(total=len(case_payload), desc=f"evaluating {mode}") as pbar:
        while pending or active:
            while pending and len(active) < args.eval_workers:
                start_case(pending.pop(0))

            now = time.monotonic()
            still_active: list[dict[str, Any]] = []
            for task in active:
                proc = task["proc"]
                timed_out = bool(sample_timeout > 0 and (now - float(task["started"])) > sample_timeout)
                if timed_out and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                    reason = f"sample_eval_timeout>{sample_timeout:g}s"
                    all_metrics.extend(timeout_metrics_for_sample(mode, task["case"], task["rows"], reason))
                    task["queue"].close()
                    pbar.update(1)
                    continue

                if proc.is_alive():
                    still_active.append(task)
                    continue

                proc.join()
                result = None
                try:
                    result = task["queue"].get_nowait()
                except Exception:
                    pass
                task["queue"].close()
                if result and result.get("ok"):
                    all_metrics.extend(result["metrics"])
                else:
                    reason = result.get("error") if isinstance(result, dict) else f"sample_eval_failed_exitcode={proc.exitcode}"
                    all_metrics.extend(timeout_metrics_for_sample(mode, task["case"], task["rows"], str(reason)))
                pbar.update(1)
            active = still_active
            if active:
                time.sleep(0.05)
    return all_metrics


def aggregate_metrics(metrics: list[dict[str, Any]], n: int, total_cases: int) -> dict[str, Any]:
    subset = [m for m in metrics if int(m["gen_index"]) < n]
    denom = total_cases * n
    out: dict[str, Any] = {"n": n, "num_attempts": denom}
    for key in (
        "raw_generation_success",
        "parse_success",
        "symcif_to_cif_success",
        "pymatgen_readable",
        "formula_ok",
        "space_group_ok",
        "multiplicity_ok",
        "valid",
        "eval_timeout",
    ):
        out[key] = sum(1 for m in subset if m.get(key)) / denom if denom else math.nan
    out["too_many_estimated_sites"] = (
        sum(1 for m in subset if m.get("pymatgen_skipped_reason") == "too_many_estimated_sites") / denom
        if denom
        else math.nan
    )
    out["too_many_eval_sites"] = (
        sum(1 for m in subset if m.get("bond_skipped_reason") == "too_many_sites") / denom if denom else math.nan
    )
    out["match_skipped_too_many_sites"] = (
        sum(1 for m in subset if m.get("match_skipped_reason") == "too_many_sites") / denom if denom else math.nan
    )

    score_sum = sum(float(m["bond_length_score"]) for m in subset if m.get("bond_length_score") is not None)
    out["bond_length_score"] = score_sum / denom if denom else math.nan

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for m in subset:
        by_sample.setdefault(int(m["sample_index"]), []).append(m)
    n1_matches = 0
    n_any_matches = 0
    best_rms: list[float] = []
    for sample_index in range(total_cases):
        rows = sorted(by_sample.get(sample_index, []), key=lambda r: int(r["gen_index"]))
        first = rows[0] if rows else None
        if first and first.get("match_ok"):
            n1_matches += 1
        rms_values = [float(r["rms"]) for r in rows if r.get("match_ok") and r.get("rms") is not None]
        if rms_values:
            n_any_matches += 1
            best_rms.append(min(rms_values))
    out["match_rate_n1"] = n1_matches / total_cases if total_cases else math.nan
    out["match_rate_n20"] = n_any_matches / total_cases if total_cases else math.nan
    out["RMSE"] = float(np.mean(best_rms)) if best_rms else math.nan
    out["matched_samples_for_RMSE"] = len(best_rms)
    gen_times = [float(m["generation_time_seconds"]) for m in subset if m.get("generation_time_seconds") is not None]
    out["average_generation_time"] = float(np.mean(gen_times)) if gen_times else math.nan
    closure_flags = [bool(m.get("formula_closure_success")) for m in subset if m.get("formula_closure_success") is not None]
    out["formula_closure_success_rate"] = (
        sum(1 for value in closure_flags if value) / len(closure_flags) if closure_flags else math.nan
    )
    mask_counts = [float(m["mask_rejected_tokens"]) for m in subset if m.get("mask_rejected_tokens") is not None]
    out["mask_rejection_count_mean"] = float(np.mean(mask_counts)) if mask_counts else math.nan
    resample_counts = [float(m["resample_count"]) for m in subset if m.get("resample_count") is not None]
    out["resample_count_mean"] = float(np.mean(resample_counts)) if resample_counts else math.nan
    return out


def run_evaluation(args: argparse.Namespace, cases: list[TestCase]) -> list[dict[str, Any]]:
    out_dir = Path(args.out_dir)
    generation_dir = Path(args.generation_dir) if args.generation_dir else out_dir / "generations"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    full_lookup = PROJECT_ROOT / "artifacts" / "wyckoff_lookup_full.json"
    lookup_json = str(full_lookup if full_lookup.exists() else PROJECT_ROOT / "artifacts" / "wyckoff_lookup.json")
    case_payload = [
        {
            "index": c.index,
            "sample_id": c.sample_id,
            "source_path": c.source_path,
            "target_formula": c.target_formula,
            "target_sg_number": c.target_sg_number,
            "target_sg_symbol": c.target_sg_symbol,
        }
        for c in cases
    ]

    summary_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        generation_path = generation_dir / f"{mode}.jsonl"
        if not generation_path.exists():
            raise FileNotFoundError(f"missing generation file for {mode}: {generation_path}")
        grouped = load_generation_records(generation_path)
        all_metrics = evaluate_mode_with_hard_timeouts(
            mode=mode,
            case_payload=case_payload,
            grouped=grouped,
            lookup_json=lookup_json,
            args=args,
        )
        all_metrics.sort(key=lambda r: (int(r["sample_index"]), int(r["gen_index"])))
        metrics_path = metrics_dir / f"{mode}_per_generation_metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as f:
            for rec in all_metrics:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

        for n in (1, args.num_gens):
            row = {"mode": mode}
            row.update(aggregate_metrics(all_metrics, n=n, total_cases=len(cases)))
            summary_rows.append(row)
        print(f"[evaluate:{mode}] wrote metrics -> {metrics_path}", flush=True)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_report(args, cases, summary_rows)
    print(f"[evaluate] wrote summary -> {summary_csv}", flush=True)
    return summary_rows


def fmt_pct(x: Any) -> str:
    try:
        value = float(x)
    except Exception:
        return "N/A"
    if math.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_num(x: Any, digits: int = 4) -> str:
    try:
        value = float(x)
    except Exception:
        return "N/A"
    if math.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


def write_report(args: argparse.Namespace, cases: list[TestCase], summary_rows: list[dict[str, Any]]) -> None:
    report_dir = PROJECT_ROOT / "Log_GPT"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    report_path = report_dir / f"generation_eval_report_{args.date_tag}.md"
    seed_list = [args.seed + i for i in range(args.num_gens)]
    lines = [
        "# SymCIF 三模型生成评估报告",
        "",
        f"日期：{args.date_tag}",
        "",
        "## 1. 实验设置",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| test split | {len(cases)} samples |",
        f"| prompt rule | 各格式从 `data_...` 截取到 `_symmetry_space_group_name_H-M`，保证公式与空间群条件一致 |",
        f"| temperature | {args.temperature} |",
        f"| top_k | {args.top_k} |",
        f"| n | {args.num_gens} |",
        f"| seed list | `{seed_list}` |",
        f"| max_new_tokens | {args.max_new_tokens} |",
        f"| evaluator timeouts | bond={args.bond_timeout_seconds}s, valid={args.valid_timeout_seconds}s, match={args.match_timeout_seconds}s |",
        f"| sample hard timeout | {args.sample_timeout_seconds}s |",
        f"| max match sites | {args.max_match_sites} |",
        f"| max eval sites | {args.max_eval_sites} |",
        f"| output dir | `{out_dir}` |",
        f"| generation dir | `{Path(args.generation_dir) if args.generation_dir else out_dir / 'generations'}` |",
        "",
        "使用 checkpoint：",
        "",
        "| mode | checkpoint dir |",
        "| --- | --- |",
        "| baseline | `runs/exp_baseline_rerun750/ckpt_best.pt` |",
        "| cf_like | `runs/exp_cf_like_rerun750/ckpt_best.pt` |",
        "| symcif_v1 | `runs/exp_symcif_v1/ckpt_best.pt` |",
        "",
        "baseline 直接按 CIF 进入 evaluator；`cf_like` 与 `symcif_v1` 先解析为 SymCIF record，再通过 `symcif_to_cif` 转回标准 CIF。baseline 的 `symcif_to_cif_success` 记为 direct-CIF passthrough success。",
        "",
        "## 2. 汇总结果",
        "",
        "| mode | n | raw generation success | parse success | symcif_to_cif success | pymatgen readable | formula_ok | space_group_ok | multiplicity_ok | bond_length_score | valid | eval_timeout | match_rate_n1 | match_rate_n20 | RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {mode} | {n} | {raw} | {parse} | {conv} | {readable} | {formula} | {sg} | {mult} | {bond} | {valid} | {timeout} | {mn1} | {mn20} | {rmse} |".format(
                mode=row["mode"],
                n=row["n"],
                raw=fmt_pct(row["raw_generation_success"]),
                parse=fmt_pct(row["parse_success"]),
                conv=fmt_pct(row["symcif_to_cif_success"]),
                readable=fmt_pct(row["pymatgen_readable"]),
                formula=fmt_pct(row["formula_ok"]),
                sg=fmt_pct(row["space_group_ok"]),
                mult=fmt_pct(row["multiplicity_ok"]),
                bond=fmt_num(row["bond_length_score"]),
                valid=fmt_pct(row["valid"]),
                timeout=fmt_pct(row.get("eval_timeout", 0.0)),
                mn1=fmt_pct(row["match_rate_n1"]),
                mn20=fmt_pct(row["match_rate_n20"]),
                rmse=fmt_num(row["RMSE"]),
            )
        )
    lines.extend(
        [
            "",
        "## 3. 指标口径",
            "",
            "- `parse_success`：baseline 为 CIF parser 可读取；`cf_like/symcif_v1` 为 SymCIF 文本可解析。",
            "- `symcif_to_cif_success`：baseline 为 direct-CIF passthrough；另外两组为 SymCIF record 成功渲染为标准 CIF。",
            "- `formula_ok` 和 `space_group_ok`：与同一个 test split 的 GT 公式和空间群编号比较。",
            "- `multiplicity_ok`、`bond_length_score`、`valid`：在转回标准 CIF 后使用同一套 CrystaLLM evaluator 口径。",
            "- `bond_length_score`：失败样本按 0 计入总尝试数均值。",
            "- `match_rate_n1`：只看每个样本的第 1 个 seed 生成。",
            "- `match_rate_n20`：每个样本 20 个生成中任一结构与 GT 匹配即计为成功。",
            "- `RMSE`：对发生匹配的样本取该样本 20 次内最小 RMS，再对匹配样本求均值；n=1 行则只基于第 1 次生成。",
            "- evaluator 超时：少数大结构的 bond/valid/match 单次计算超过设定秒数时，该项按失败或无匹配处理，避免长尾样本阻塞整轮统计。",
            "- sample hard timeout：若单个 test sample 的 20 个生成整体评估超过设定秒数，子进程会被终止，该 sample 的 20 个生成记为 `eval_timeout`。",
            "- match 大结构保护：generated 或 GT 结构 site 数超过 `max_match_sites` 时跳过 StructureMatcher，按无匹配处理。",
            "- bond/valid 大结构保护：generated 结构 site 数超过 `max_eval_sites` 时跳过 CrystalNN bond score 和 `valid`，bond 记为 0 贡献、valid 记为失败。",
            "",
            "## 4. 输出文件",
            "",
            f"- summary CSV：`{out_dir / 'summary.csv'}`",
            f"- summary JSON：`{out_dir / 'summary.json'}`",
            f"- raw generations：`{Path(args.generation_dir) if args.generation_dir else out_dir / 'generations'}`",
            f"- per-generation metrics：`{out_dir / 'metrics'}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared-prompt generation and evaluation for SymCIF experiments.")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_runs" / "generation_eval_t1_topk10_n20_20260519")
    parser.add_argument("--generation-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-gens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(16, os.cpu_count() or 4)))
    parser.add_argument("--bond-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--parse-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sg-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--valid-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--match-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sample-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-match-sites", type=int, default=128)
    parser.add_argument("--max-eval-sites", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--date-tag", default="20260519")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.modes = tuple(args.modes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_test_cases(args.test_limit, modes=args.modes)
    seeds = [args.seed + i for i in range(args.num_gens)]
    metadata = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "num_gens": args.num_gens,
        "seeds": seeds,
        "max_new_tokens": args.max_new_tokens,
        "bond_timeout_seconds": args.bond_timeout_seconds,
        "valid_timeout_seconds": args.valid_timeout_seconds,
        "match_timeout_seconds": args.match_timeout_seconds,
        "sample_timeout_seconds": args.sample_timeout_seconds,
        "max_match_sites": args.max_match_sites,
        "max_eval_sites": args.max_eval_sites,
        "test_samples": len(cases),
        "prompt_rule": "format-specific prefix from data_ line through _symmetry_space_group_name_H-M",
        "generation_dir": str(args.generation_dir) if args.generation_dir else str(Path(args.out_dir) / "generations"),
    }
    (Path(args.out_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_generation:
        run_generation(args, cases, seeds)
    if not args.skip_evaluation:
        run_evaluation(args, cases)


if __name__ == "__main__":
    main()
