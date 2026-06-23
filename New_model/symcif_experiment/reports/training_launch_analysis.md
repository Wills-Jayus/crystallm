# SymCIF Experiment Training Launch Analysis

Date: 2026-05-19

## Launch Status

- `exp_baseline` is running from `configs/exp_baseline.yaml`.
- `exp_cf_like` is running from `configs/exp_cf_like.yaml`.
- `exp_symcif_v1` is queued by `scripts/run_recovery_cf_then_symcif.sh` and will start after `exp_cf_like` finishes.
- Active tmux sessions:
  - `symcif_train`: original baseline queue.
  - `symcif_recovery_cf_sym`: recovery queue for `cf_like` followed by `symcif_v1`.

The first `cf_like` attempt failed because the process only saw one logical CUDA device but was passed `cuda:1`. The queue script was fixed to expose one physical GPU per process and use logical `cuda:0` inside each process.

## Dataset And Corpus

- Source dataset: `/data/users/xsw/autodlmini/model/scp_task/CrystaLLM/reproduce/benchmarks_gt_from_prepare_csv_benchmark_symprec0p1`
- Split: 5000 train / 500 val / 500 test
- Processed candidates before accepted split completion: 6607
- Symmetry tolerance: `symprec=0.1`, angle tolerance `5.0`

Tokenized corpus sizes:

| mode | vocab | train tokens | val tokens | train token reduction vs baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline | 405 | 2055575 | 202345 | 0.0% |
| cf_like | 405 | 1253082 | 123176 | 39.0% |
| symcif_v1 | 405 | 1421459 | 139673 | 30.8% |

Conversion quality:

| mode | parse success | roundtrip readable | formula consistent | space group consistent | multiplicity consistent |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 6000/6000 | not applicable | not applicable | not applicable | not applicable |
| cf_like | 6000/6000 | 6000/6000 | 6000/6000 | 6000/6000 | 6000/6000 |
| symcif_v1 | 6000/6000 | 6000/6000 | 6000/6000 | 6000/6000 | 6000/6000 |

## Initial Training Signals

Latest checked validation points:

| experiment | step 0 val loss | step 250 val loss | step 500 val loss | step 750 val loss |
| --- | ---: | ---: | ---: | ---: |
| exp_baseline | 5.9138 | 0.8628 | 0.4158 | 0.3845 |
| exp_cf_like | 5.9810 | 1.0019 | 0.7062 | pending |

Both active runs were using about 6.3 GiB GPU memory per A800 and about 99% GPU utilization at the last check.

## Preliminary Interpretation

- `cf_like` is much more token-efficient than full CIF baseline: about 39% fewer train tokens. This should reduce sequence burden and usually improves sample efficiency, but it removes explicit atom-site coordinate detail from the model target.
- `symcif_v1` is still much shorter than baseline while carrying extra symmetry fields compared with `cf_like`. Its 13.4% token overhead relative to `cf_like` is expected; the value of that overhead should be judged by downstream validity, space-group recovery, Wyckoff consistency, and RMSD/match metrics, not by loss alone.
- Early loss is not directly comparable across formats because the targets have different entropy and token structure. The more important comparison will be generated structure quality under identical sampling settings.
- The clean conversion results for `cf_like` and `symcif_v1` mean the retained dataset is suitable for a controlled A/B/C comparison; remaining risk is mainly model-side generation and reconstruction reliability.

## Next Evaluation Targets

After checkpoints finish:

- sample the same number of generations per prompt for all three models;
- evaluate parse success, formula consistency, space-group consistency, Wyckoff/multiplicity consistency, structural validity, match rate, and RMSD;
- compare convergence by validation loss at matched steps, but use generation metrics as the primary experimental conclusion.
