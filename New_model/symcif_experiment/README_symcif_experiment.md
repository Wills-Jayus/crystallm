# SymCIF Format Experiment

This directory contains the first-stage A/B/C data-format experiment for CrystaLLM.

## Scope

This stage only changes the training text format. It does not add dynamic masks, constrained decoding, model-structure changes, loss changes, or multi-head prediction.

## Data Source

Approved source:

```bash
/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1
```

The corpus builder recursively reads all `.cif` files under this root and samples a shared `5000/500/500` split with seed `1337`.

## Formats

`baseline` keeps the original CrystaLLM-style CIF preprocessing.

`cf_like` writes:

```text
formula -> space group -> Wyckoff site fields -> lattice
```

`symcif_v1` adds `site_symmetry` and `enumeration` to the `cf_like` fields.

## Lookup Policy

`artifacts/source_tables/crystalformer_wyckoff_list.csv` is copied from CrystalFormer and used for Wyckoff templates, multiplicities, and free-coordinate masks.

`artifacts/source_tables/wyformer_wyckoffs_enumerated_by_ss.json` is copied from the WyckoffTransformer wheel without installing `pyxtal`. It supplies the WyFormer-style `site_symmetry` and `enumeration` mapping. The converted samples are still checked by local `pymatgen/spglib`; samples with uncertain round-trip behavior are recorded in `reports/failed_cases.jsonl` and excluded from the shared split.

## Build Commands

From the experiment root:

```bash
cd /data/users/xsw/autodlmini/model/New_model/symcif_experiment
conda run -n crystallm_env python scripts/build_pretrain_corpus.py \
  /data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1 \
  --out-dir data \
  --reports-dir reports \
  --train-size 5000 \
  --val-size 500 \
  --test-size 500 \
  --seed 1337
```

Run direct conversion with round-trip output:

```bash
conda run -n crystallm_env python scripts/convert_cif_to_symcif.py \
  /path/to/cifs \
  --out-dir reports/manual_conversion \
  --format all \
  --roundtrip
```

## Training Configs

The three configs are:

```text
configs/exp_baseline.yaml
configs/exp_cf_like.yaml
configs/exp_symcif_v1.yaml
```

They keep model size, batch size, block size, learning rate, steps, seed, sampling temperature, `top_k`, and `max_new_tokens` aligned.

The tokenizer does not need BPE training. The copied CrystaLLM tokenizer in `external/CrystaLLM_code/crystallm/_tokenizer.py` has a fixed vocabulary extended with the new Wyckoff/SymCIF tokens. All A/B/C tokenized datasets should be produced with this same extended tokenizer.

## Current Status

The full `5000/500/500` corpus build is expected to populate:

```text
data/baseline/{train,val,test}.txt
data/cf_like/{train,val,test}.txt
data/symcif_v1/{train,val,test}.txt
reports/conversion_report_baseline.json
reports/conversion_report_cf_like.json
reports/conversion_report_symcif_v1.json
reports/failed_cases.jsonl
reports/corpus_build_summary.json
```

After the build, check the conversion reports before training. If `pymatgen_readable < 95%`, `formula_consistent < 95%`, `space_group_consistent < 90%`, or `multiplicity_consistent < 95%`, do not train yet.

Tokenize the three corpora:

```bash
conda run -n crystallm_env python scripts/tokenize_pretrain_corpus.py data/baseline --out-dir data/tokens_baseline
conda run -n crystallm_env python scripts/tokenize_pretrain_corpus.py data/cf_like --out-dir data/tokens_cf_like
conda run -n crystallm_env python scripts/tokenize_pretrain_corpus.py data/symcif_v1 --out-dir data/tokens_symcif_v1
```

## Current Results

The retained shared split is complete:

```text
train: 5000
val:   500
test:  500
```

Retained conversion reports:

```text
cf_like:    6000/6000 pymatgen_readable, formula_consistent, space_group_consistent, multiplicity_consistent
symcif_v1:  6000/6000 pymatgen_readable, formula_consistent, space_group_consistent, multiplicity_consistent
```

Raw candidate statistics before filtering are saved as `reports/conversion_report_cf_like_raw.json` and `reports/conversion_report_symcif_v1_raw.json`. The build processed 6607 candidates to retain 6000. Main raw failure reasons were formula/multiplicity mismatches after round-trip, space-group mismatches after round-trip, and a small number of `spglib`/`CifWriter` failures.

Tokenized corpus sizes:

```text
baseline:   train 2,055,575 tokens, val 202,345 tokens
cf_like:    train 1,253,082 tokens, val 123,176 tokens
symcif_v1:  train 1,421,459 tokens, val 139,673 tokens
```

The extended tokenizer reports zero `<unk>` tokens on all train corpora.

## Training Readiness

The retained B/C data pass the stage-1 thresholds, and the A/B/C tokenized datasets and configs are present. The next step can be training from scratch with the three configs, keeping the same tokenizer vocabulary and hyperparameters.
