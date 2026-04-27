#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


_ROOT = Path(__file__).resolve().parents[1]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _progress(i: int, n: int, name: str) -> None:
    print(("\r" + f"{i}/{n} {name}")[:120].ljust(120), end="", flush=True)


def _list_prompts(prompts_dir: Path, glob_pat: str) -> List[Path]:
    return sorted([p for p in prompts_dir.glob(glob_pat) if p.is_file()])


def _pick_dest(out_dir: Path, prompt_stem: str, suffix_index: int) -> Path:
    if suffix_index <= 0:
        return out_dir / f"{prompt_stem}.cif"
    return out_dir / f"{prompt_stem}_{suffix_index}.cif"


@dataclass(frozen=True)
class RunConfig:
    model_dir: str
    prompts_dir: str
    glob: str
    start_index: int
    num_prompts: Optional[int]
    shuffle: bool
    select_seed: int
    num_samples_per_prompt: int
    temperature: float
    top_k: int
    max_new_tokens: int
    seed: int
    device: str
    dtype: str
    out_dir: str
    out_root: Optional[str]
    run_name: Optional[str]
    workers: int
    resume: bool
    resume_seed_stride: int
    overwrite: bool
    quiet: bool


def _index_existing_outputs(out_dir: Path, prompt_stems: set[str]) -> dict[str, int]:
    """
    Count existing CIF outputs per prompt stem in out_dir.

    We treat files as belonging to a prompt stem if they match:
      - <stem>.cif
      - <stem>_<number>.cif
    """
    counts: dict[str, int] = {}
    if not out_dir.exists():
        return counts

    for p in out_dir.glob("*.cif"):
        if not p.is_file():
            continue
        base = p.name[:-4]  # strip ".cif"
        if base in prompt_stems:
            counts[base] = counts.get(base, 0) + 1
            continue
        if "_" in base:
            maybe_stem, maybe_num = base.rsplit("_", 1)
            if maybe_num.isdigit() and maybe_stem in prompt_stems:
                counts[maybe_stem] = counts.get(maybe_stem, 0) + 1
    return counts


def _run_one_prompt(
    *,
    sample_py: Path,
    model_dir: str,
    prompt_path: Path,
    out_dir: Path,
    num_samples: int,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    device: str,
    dtype: str,
    overwrite: bool,
    quiet: bool,
) -> int:
    prompt_abs = prompt_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="crystallm_sample_") as td:
        td_path = Path(td)
        cmd = [
            sys.executable,
            str(sample_py),
            f"out_dir={model_dir}",
            f"start=FILE:{str(prompt_abs)}",
            f"num_samples={num_samples}",
            f"temperature={temperature}",
            f"top_k={top_k}",
            f"max_new_tokens={max_new_tokens}",
            f"seed={seed}",
            f"device={device}",
            f"dtype={dtype}",
            "compile=False",
            "target=file",
        ]
        stdout = subprocess.DEVNULL if quiet else None
        stderr = subprocess.DEVNULL if quiet else None
        subprocess.run(cmd, cwd=str(td_path), check=True, stdout=stdout, stderr=stderr)

        produced = sorted(td_path.glob("sample_*.cif"))
        moved = 0
        for k, cif in enumerate(produced, start=1):
            # First sample uses "<prompt_stem>.cif"; later samples use suffix "_1", "_2", ...
            suffix_index = k - 1
            dest = _pick_dest(out_dir, prompt_path.stem, suffix_index)

            if dest.exists() and not overwrite:
                while dest.exists():
                    suffix_index += 1
                    dest = _pick_dest(out_dir, prompt_path.stem, suffix_index)

            if dest.exists() and overwrite:
                dest.unlink()

            shutil.move(str(cif), str(dest))
            moved += 1

        return moved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch sample CIFs from a directory of prompts using CrystaLLM bin/sample.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, help="Model directory containing ckpt.pt (sample.py out_dir=...).")
    p.add_argument("--prompts-dir", required=True, help="Directory containing prompt .txt files.")
    p.add_argument("--glob", default="*.txt", help="Glob pattern for prompt files under --prompts-dir.")
    p.add_argument("--start-index", type=int, default=0, help="Start from this index in the sorted prompt list.")
    p.add_argument("--num-prompts", type=int, default=None, help="How many prompts to sample; omit to sample all.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle prompt order before slicing.")
    p.add_argument("--select-seed", type=int, default=0, help="Seed used for --shuffle.")
    p.add_argument("--num-samples-per-prompt", type=int, default=1, help="num_samples passed to sample.py per prompt.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1337, help="Base seed; per prompt uses seed + i.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of prompts to sample concurrently (spawns that many sample.py subprocesses).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Full output directory path (overrides --out-root/--run-name).",
    )
    p.add_argument("--out-root", default=str(_ROOT / "reproduce"), help="Root directory where run folders are created.")
    p.add_argument(
        "--run-name",
        default=None,
        help="Output folder name under --out-root (i.e., customize the output directory name).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip prompts that already have enough CIFs in the output directory.",
    )
    p.add_argument(
        "--resume-seed-stride",
        type=int,
        default=1_000_000,
        help="When resuming partially completed prompts, add existing_count*stride to the seed.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing CIF filenames in the run dir.")
    p.add_argument("--no-quiet", dest="quiet", action="store_false", help="Show sample.py output (default quiet).")
    p.set_defaults(quiet=True, resume=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    prompts_dir = Path(args.prompts_dir).expanduser().resolve()
    sample_py = (_ROOT / "bin" / "sample.py").resolve()
    if not prompts_dir.exists():
        raise SystemExit(f"--prompts-dir not found: {prompts_dir}")
    if not sample_py.exists():
        raise SystemExit(f"sample.py not found: {sample_py}")

    if args.out_dir:
        run_dir = Path(args.out_dir).expanduser().resolve()
        out_root: Optional[str] = None
        run_name: Optional[str] = None
    else:
        out_root = str(Path(args.out_root).expanduser().resolve())
        run_name = args.run_name or f"sample_diy_{_now_tag()}"
        run_dir = Path(out_root) / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _list_prompts(prompts_dir, args.glob)
    if not prompts:
        raise SystemExit(f"No prompts matched under {prompts_dir} (glob={args.glob!r}).")

    if args.shuffle:
        rng = random.Random(int(args.select_seed))
        rng.shuffle(prompts)

    start = max(0, int(args.start_index))
    prompts = prompts[start:]
    if args.num_prompts is not None:
        prompts = prompts[: max(0, int(args.num_prompts))]
    if not prompts:
        raise SystemExit("No prompts to process after slicing.")

    workers = int(args.workers)
    if workers <= 0:
        raise SystemExit("--workers must be >= 1")

    prompt_stems = {p.stem for p in prompts}
    existing_counts = _index_existing_outputs(run_dir, prompt_stems) if (args.resume and not args.overwrite) else {}
    todo: list[tuple[int, Path, int]] = []
    done_initial = 0
    desired = int(args.num_samples_per_prompt)
    for abs_index, prompt_path in enumerate(prompts):
        existing = int(existing_counts.get(prompt_path.stem, 0))
        if args.resume and not args.overwrite and existing >= desired:
            done_initial += 1
            continue
        todo.append((abs_index, prompt_path, existing))

    cfg = RunConfig(
        model_dir=str(args.model_dir),
        prompts_dir=str(prompts_dir),
        glob=str(args.glob),
        start_index=int(args.start_index),
        num_prompts=int(args.num_prompts) if args.num_prompts is not None else None,
        shuffle=bool(args.shuffle),
        select_seed=int(args.select_seed),
        num_samples_per_prompt=int(args.num_samples_per_prompt),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        out_dir=str(run_dir),
        out_root=out_root,
        run_name=run_name,
        workers=workers,
        resume=bool(args.resume),
        resume_seed_stride=int(args.resume_seed_stride),
        overwrite=bool(args.overwrite),
        quiet=bool(args.quiet),
    )
    (run_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    n_total = len(prompts)
    n_pending = len(todo)
    if cfg.resume and not cfg.overwrite:
        print(f"[sample_diy] resume on: done={done_initial} pending={n_pending} total={n_total} workers={workers} -> {run_dir}")
    else:
        print(f"[sample_diy] resume off: pending={n_pending} total={n_total} workers={workers} -> {run_dir}")

    if n_pending == 0:
        print(f"[sample_diy] nothing to do -> {run_dir}")
        return

    def _task(abs_index: int, prompt_path: Path, existing: int) -> tuple[str, int]:
        remaining = desired
        seed = cfg.seed + abs_index
        if cfg.resume and not cfg.overwrite:
            remaining = max(0, desired - existing)
            if existing > 0 and remaining > 0:
                seed = seed + existing * cfg.resume_seed_stride

        moved = _run_one_prompt(
            sample_py=sample_py,
            model_dir=cfg.model_dir,
            prompt_path=prompt_path,
            out_dir=run_dir,
            num_samples=remaining,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            max_new_tokens=cfg.max_new_tokens,
            seed=seed,
            device=cfg.device,
            dtype=cfg.dtype,
            overwrite=cfg.overwrite,
            quiet=cfg.quiet,
        )
        return (prompt_path.name, moved)

    total_cifs = 0
    completed = done_initial
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_task, abs_index, prompt_path, existing) for (abs_index, prompt_path, existing) in todo]
        for fut in as_completed(futures):
            name, moved = fut.result()
            total_cifs += moved
            completed += 1
            _progress(completed, n_total, name)

    print(f"\n[sample_diy] done prompts={n_total} cifs={total_cifs} -> {run_dir}")


if __name__ == "__main__":
    main()
